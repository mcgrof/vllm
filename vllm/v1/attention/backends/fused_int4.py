# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused INT4 KV cache attention backend.

Stores K and V as packed INT4 (2 values per uint8 byte) with per-group
FP16 scales.  The decode kernel dequantizes in-register and computes
attention without materializing a full FP16 intermediate — this is the
"fused" path that eliminates the intermediate memory traffic.

Design notes for future asymmetric K/V:
  - K and V have independent scale tensors (k_scales, v_scales)
  - The cache layout stores K and V in separate halves of a (2, ...) tensor
  - Group size is a module-level constant today (GROUP_SIZE=32) but is
    threaded through as a parameter so it can become per-K / per-V later
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.cache import CacheDType
    from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GROUP_SIZE: int = int(os.environ.get("VLLM_FUSED_INT4_GROUP_SIZE", "32"))
INT4_RANGE: int = 7  # max absolute value representable in signed INT4

# Asymmetric quantization: uses min/max range with zero-point offset
# instead of symmetric abs-max.  Better for distributions with non-zero mean
# (common in V cache).  Set VLLM_FUSED_INT4_ASYMMETRIC=1 to enable.
ASYMMETRIC: bool = os.environ.get("VLLM_FUSED_INT4_ASYMMETRIC", "0") == "1"

# Minimum sequence length to use fused INT4 decode kernel.  Below this
# threshold the decode path reads from the FP16 paged cache (written by
# the standard reshape_and_cache_flash path) via SDPA.  The current
# validated H100/Qwen2.5-7B-Instruct policy is 48: this keeps the first
# 48 decode positions on the safer FP16-shadow / paged path, then enters
# the fused INT4 path once the tested correctness envelope remains stable.
#
# Why 48 instead of 64: the Qwen H100 policy sweep showed 48 preserves the
# tested text-match envelope while recovering 16 decode positions per
# sequence versus 64, i.e. a 25% smaller FP16 protected window and earlier
# entry into the lower-memory-traffic fused path.  This is a model-specific
# default, not a universal rule. Override with VLLM_FUSED_INT4_MIN_SEQ_LEN
# when calibrating other models or regimes.
MIN_FUSED_SEQ_LEN: int = int(
    os.environ.get("VLLM_FUSED_INT4_MIN_SEQ_LEN", "48")
)

# K-cache precision override for K/V precision-split experiments.
# "int4" (default): both K and V use INT4 (existing behaviour).
# "int8": K is quantised to symmetric INT8, V stays INT4.
# "fp16": K is read from the FP16 paged cache, V uses INT4.
# This tests the paper claim that Qwen-class 7B models need much higher
# precision for K than V.
K_PRECISION: str = os.environ.get("VLLM_FUSED_INT4_K_PRECISION", "int4")


def _cpu_dequant_slot(
    cache: torch.Tensor,
    scales: torch.Tensor,
    block_idx: int,
    block_offset: int,
    head_idx: int,
    head_size: int,
    n_values: int = 16,
    zeros: torch.Tensor | None = None,
) -> list[float]:
    """CPU-side dequant of packed INT4 bytes for one slot/head.

    Returns a list of n_values reconstructed float values.
    """
    packed = cache[block_idx, block_offset, head_idx].detach().cpu()
    sc = scales[block_idx, block_offset, head_idx].detach().cpu().float()
    group_size = GROUP_SIZE

    # Unpack: low nibble = even index, high nibble = odd index
    low = (packed & 0x0F).to(torch.int8)
    high = ((packed >> 4) & 0x0F).to(torch.int8)

    if zeros is not None:
        # Asymmetric: unsigned values, dequant = q * scale + zero_point
        zp = zeros[block_idx, block_offset, head_idx].detach().cpu().float()
    else:
        # Symmetric: signed offset, dequant = (q - 8) * scale
        low = low - 8
        high = high - 8
        zp = None

    # Interleave back to full head_size
    hd = packed.shape[0] * 2
    full = torch.empty(hd, dtype=torch.float32)
    full[0::2] = low.float()
    full[1::2] = high.float()

    # Apply per-group scales (and zero_points for asymmetric)
    num_groups = hd // group_size
    full = full.reshape(num_groups, group_size)
    full = full * sc[:num_groups].unsqueeze(1)
    if zp is not None:
        full = full + zp[:num_groups].unsqueeze(1)
    full = full.reshape(hd)

    return full[:n_values].tolist()


# Portable rounding helper that works on both CUDA and ROCm Triton
@triton.jit
def _triton_round(x):
    """Round-half-away-from-zero, portable across Triton backends."""
    return tl.where(x >= 0, tl.floor(x + 0.5), tl.ceil(x - 0.5))


# ===================================================================
# Triton kernel: reshape_and_cache_int4
# Quantizes FP16/BF16 K/V to INT4 and writes packed bytes + per-group
# scales into the paged cache.
# ===================================================================
@triton.jit
def _reshape_and_cache_int4_kernel(
    # Source K/V from the model (FP16/BF16)
    key_ptr,  # [num_tokens, num_heads, head_size]
    value_ptr,  # [num_tokens, num_heads, head_size]
    # Destination packed cache (uint8)
    key_cache_ptr,  # [num_blocks, block_size, num_heads, packed_dim]
    value_cache_ptr,  # [num_blocks, block_size, num_heads, packed_dim]
    # Destination scale cache (float16)
    k_scales_ptr,  # [num_blocks, block_size, num_heads, num_groups]
    v_scales_ptr,  # [num_blocks, block_size, num_heads, num_groups]
    # Destination zero-point cache (float16) — only used in asymmetric mode
    k_zeros_ptr,  # [num_blocks, block_size, num_heads, num_groups] or dummy
    v_zeros_ptr,
    # Slot mapping
    slot_mapping_ptr,  # [num_tokens]
    # Strides — source
    key_stride_token: tl.int64,
    key_stride_head: tl.int64,
    value_stride_token: tl.int64,
    value_stride_head: tl.int64,
    # Strides — packed cache
    kc_stride_block: tl.int64,
    kc_stride_page: tl.int64,
    kc_stride_head: tl.int64,
    vc_stride_block: tl.int64,
    vc_stride_page: tl.int64,
    vc_stride_head: tl.int64,
    # Strides — scales
    ks_stride_block: tl.int64,
    ks_stride_page: tl.int64,
    ks_stride_head: tl.int64,
    vs_stride_block: tl.int64,
    vs_stride_page: tl.int64,
    vs_stride_head: tl.int64,
    # Strides — zeros (same shape as scales)
    kz_stride_block: tl.int64,
    kz_stride_page: tl.int64,
    kz_stride_head: tl.int64,
    vz_stride_block: tl.int64,
    vz_stride_page: tl.int64,
    vz_stride_head: tl.int64,
    # Dims (constexpr for compile-time specialisation)
    num_heads: tl.constexpr,
    head_size: tl.constexpr,
    block_size: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    HALF_HD: tl.constexpr,
    IS_ASYMMETRIC: tl.constexpr,
):
    """One program instance handles one (token, head) pair."""
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    slot_idx = tl.load(slot_mapping_ptr + token_idx).to(tl.int64)
    if slot_idx < 0:
        return  # padding token

    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size

    # ---- Source offsets (FP16/BF16 key/value) ----
    src_k_base = token_idx * key_stride_token + head_idx * key_stride_head
    src_v_base = token_idx * value_stride_token + head_idx * value_stride_head

    # ---- Destination offsets (packed uint8 cache, scales) ----
    dst_kc_base = (block_idx * kc_stride_block
                   + block_offset * kc_stride_page
                   + head_idx * kc_stride_head)
    dst_vc_base = (block_idx * vc_stride_block
                   + block_offset * vc_stride_page
                   + head_idx * vc_stride_head)
    dst_ks_base = (block_idx * ks_stride_block
                   + block_offset * ks_stride_page
                   + head_idx * ks_stride_head)
    dst_vs_base = (block_idx * vs_stride_block
                   + block_offset * vs_stride_page
                   + head_idx * vs_stride_head)
    dst_kz_base = (block_idx * kz_stride_block
                   + block_offset * kz_stride_page
                   + head_idx * kz_stride_head)
    dst_vz_base = (block_idx * vz_stride_block
                   + block_offset * vz_stride_page
                   + head_idx * vz_stride_head)

    # Process each group
    for g in tl.static_range(0, NUM_GROUPS):
        g_start = g * GROUP_SIZE
        half_offs = tl.arange(0, GROUP_SIZE // 2)

        # ---- Key group ----
        k_low = tl.load(key_ptr + src_k_base + g_start + half_offs * 2,
                         mask=(g_start + half_offs * 2) < head_size).to(tl.float32)
        k_high = tl.load(key_ptr + src_k_base + g_start + half_offs * 2 + 1,
                          mask=(g_start + half_offs * 2 + 1) < head_size).to(tl.float32)

        if IS_ASYMMETRIC:
            # Asymmetric: map [min, max] -> [0, 15]
            offs = tl.arange(0, GROUP_SIZE)
            k_vals = tl.load(key_ptr + src_k_base + g_start + offs,
                             mask=(g_start + offs) < head_size).to(tl.float32)
            k_min = tl.min(k_vals)
            k_max = tl.max(k_vals)
            k_range = tl.maximum(k_max - k_min, 1e-8)
            k_scale = k_range / 15.0
            k_zero = k_min

            k_low_q = _triton_round((k_low - k_zero) / k_scale)
            k_low_q = tl.maximum(tl.minimum(k_low_q, 15.0), 0.0)
            k_low_u = k_low_q.to(tl.uint8)

            k_high_q = _triton_round((k_high - k_zero) / k_scale)
            k_high_q = tl.maximum(tl.minimum(k_high_q, 15.0), 0.0)
            k_high_u = k_high_q.to(tl.uint8)

            tl.store(k_zeros_ptr + dst_kz_base + g, k_zero.to(tl.float16))
        else:
            # Symmetric: map [-amax, +amax] -> [-8, 7] stored as [0, 15]
            offs = tl.arange(0, GROUP_SIZE)
            k_vals = tl.load(key_ptr + src_k_base + g_start + offs,
                             mask=(g_start + offs) < head_size).to(tl.float32)
            k_amax = tl.max(tl.abs(k_vals))
            k_amax = tl.maximum(k_amax, 1e-8)
            k_scale = k_amax / 7.0
            k_zero = 0.0  # unused but keeps code uniform

            k_low_q = _triton_round(k_low / k_scale)
            k_low_q = tl.maximum(tl.minimum(k_low_q, 7.0), -8.0)
            k_low_u = (k_low_q + 8.0).to(tl.uint8)

            k_high_q = _triton_round(k_high / k_scale)
            k_high_q = tl.maximum(tl.minimum(k_high_q, 7.0), -8.0)
            k_high_u = (k_high_q + 8.0).to(tl.uint8)

        k_packed = k_low_u | (k_high_u << 4)
        packed_offs = g * (GROUP_SIZE // 2) + half_offs
        tl.store(key_cache_ptr + dst_kc_base + packed_offs,
                 k_packed, mask=packed_offs < HALF_HD)
        tl.store(k_scales_ptr + dst_ks_base + g, k_scale.to(tl.float16))

        # ---- Value group ----
        v_low = tl.load(value_ptr + src_v_base + g_start + half_offs * 2,
                         mask=(g_start + half_offs * 2) < head_size).to(tl.float32)
        v_high = tl.load(value_ptr + src_v_base + g_start + half_offs * 2 + 1,
                          mask=(g_start + half_offs * 2 + 1) < head_size).to(tl.float32)

        if IS_ASYMMETRIC:
            v_vals = tl.load(value_ptr + src_v_base + g_start + offs,
                             mask=(g_start + offs) < head_size).to(tl.float32)
            v_min = tl.min(v_vals)
            v_max = tl.max(v_vals)
            v_range = tl.maximum(v_max - v_min, 1e-8)
            v_scale = v_range / 15.0
            v_zero = v_min

            v_low_q = _triton_round((v_low - v_zero) / v_scale)
            v_low_q = tl.maximum(tl.minimum(v_low_q, 15.0), 0.0)
            v_low_u = v_low_q.to(tl.uint8)

            v_high_q = _triton_round((v_high - v_zero) / v_scale)
            v_high_q = tl.maximum(tl.minimum(v_high_q, 15.0), 0.0)
            v_high_u = v_high_q.to(tl.uint8)

            tl.store(v_zeros_ptr + dst_vz_base + g, v_zero.to(tl.float16))
        else:
            v_vals = tl.load(value_ptr + src_v_base + g_start + offs,
                             mask=(g_start + offs) < head_size).to(tl.float32)
            v_amax = tl.max(tl.abs(v_vals))
            v_amax = tl.maximum(v_amax, 1e-8)
            v_scale = v_amax / 7.0

            v_low_q = _triton_round(v_low / v_scale)
            v_low_q = tl.maximum(tl.minimum(v_low_q, 7.0), -8.0)
            v_low_u = (v_low_q + 8.0).to(tl.uint8)

            v_high_q = _triton_round(v_high / v_scale)
            v_high_q = tl.maximum(tl.minimum(v_high_q, 7.0), -8.0)
            v_high_u = (v_high_q + 8.0).to(tl.uint8)

        v_packed = v_low_u | (v_high_u << 4)
        tl.store(value_cache_ptr + dst_vc_base + packed_offs,
                 v_packed, mask=packed_offs < HALF_HD)
        tl.store(v_scales_ptr + dst_vs_base + g, v_scale.to(tl.float16))


def reshape_and_cache_int4(
    key: torch.Tensor,  # [num_tokens, num_heads, head_size]
    value: torch.Tensor,  # [num_tokens, num_heads, head_size]
    key_cache: torch.Tensor,  # [num_blocks, block_size, num_heads, head_size//2]
    value_cache: torch.Tensor,
    k_scales: torch.Tensor,  # [num_blocks, block_size, num_heads, num_groups]
    v_scales: torch.Tensor,
    slot_mapping: torch.Tensor,  # [num_tokens]
    k_zeros: torch.Tensor | None = None,
    v_zeros: torch.Tensor | None = None,
    asymmetric: bool = False,
) -> None:
    """Quantize FP16/BF16 K/V to INT4 and write into paged cache."""
    num_tokens = key.shape[0]
    num_heads = key.shape[1]
    head_size = key.shape[2]
    block_size = key_cache.shape[1]
    num_groups = head_size // GROUP_SIZE
    half_hd = head_size // 2

    # Dummy zero tensors when symmetric (Triton needs valid pointers)
    if k_zeros is None:
        k_zeros = k_scales  # won't be written in symmetric mode
    if v_zeros is None:
        v_zeros = v_scales

    grid = (num_tokens, num_heads)
    _reshape_and_cache_int4_kernel[grid](
        key, value,
        key_cache, value_cache,
        k_scales, v_scales,
        k_zeros, v_zeros,
        slot_mapping,
        # Source strides
        key.stride(0), key.stride(1),
        value.stride(0), value.stride(1),
        # Packed cache strides
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
        # Scale strides
        k_scales.stride(0), k_scales.stride(1), k_scales.stride(2),
        v_scales.stride(0), v_scales.stride(1), v_scales.stride(2),
        # Zero strides
        k_zeros.stride(0), k_zeros.stride(1), k_zeros.stride(2),
        v_zeros.stride(0), v_zeros.stride(1), v_zeros.stride(2),
        # Dims
        num_heads=num_heads,
        head_size=head_size,
        block_size=block_size,
        GROUP_SIZE=GROUP_SIZE,
        NUM_GROUPS=num_groups,
        HALF_HD=half_hd,
        IS_ASYMMETRIC=asymmetric,
    )


# ===================================================================
# Triton kernel: fused INT4 decode attention
# Reads packed INT4 K/V from paged cache, dequantizes in-register,
# computes scaled-dot-product attention with online softmax.
# No FP16 intermediate is materialised in global memory.
# ===================================================================
@triton.jit
def _fused_int4_decode_kernel(
    # Query — one token per request (decode)
    Q_ptr,  # [num_seqs, num_heads, head_size]
    # Packed KV cache (paged)
    K_cache_ptr,  # [num_blocks, block_size, num_kv_heads, half_hd]
    V_cache_ptr,
    # Scales
    K_scales_ptr,  # [num_blocks, block_size, num_kv_heads, num_groups]
    V_scales_ptr,
    # Zero-points (asymmetric mode only)
    K_zeros_ptr,
    V_zeros_ptr,
    # Output
    Out_ptr,  # [num_seqs, num_heads, head_size]
    # Block table for paged KV
    block_table_ptr,  # [num_seqs, max_num_blocks_per_seq]
    # Sequence lengths
    seq_lens_ptr,  # [num_seqs]
    # Strides — Q
    q_stride_seq: tl.int64,
    q_stride_head: tl.int64,
    # Strides — packed cache
    kc_stride_block: tl.int64,
    kc_stride_page: tl.int64,
    kc_stride_head: tl.int64,
    vc_stride_block: tl.int64,
    vc_stride_page: tl.int64,
    vc_stride_head: tl.int64,
    # Strides — scales
    ks_stride_block: tl.int64,
    ks_stride_page: tl.int64,
    ks_stride_head: tl.int64,
    vs_stride_block: tl.int64,
    vs_stride_page: tl.int64,
    vs_stride_head: tl.int64,
    # Strides — zeros
    kz_stride_block: tl.int64,
    kz_stride_page: tl.int64,
    kz_stride_head: tl.int64,
    vz_stride_block: tl.int64,
    vz_stride_page: tl.int64,
    vz_stride_head: tl.int64,
    # Strides — output
    o_stride_seq: tl.int64,
    o_stride_head: tl.int64,
    # Strides — block table
    bt_stride_seq: tl.int64,
    # Dims
    num_kv_heads: tl.int32,
    scale: tl.float32,  # 1/sqrt(head_dim)
    # Compile-time constants
    HEAD_DIM: tl.constexpr,
    HALF_HD: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BYTES_PER_GROUP: tl.constexpr,  # GROUP_SIZE // 2
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,  # context tile size
    N_REP: tl.constexpr,  # num_heads // num_kv_heads (GQA ratio)
    IS_ASYMMETRIC: tl.constexpr,
):
    pid_sh = tl.program_id(0)
    # Decompose program id into (seq, head)
    num_q_heads = num_kv_heads * N_REP
    seq_idx = pid_sh // num_q_heads
    head_idx = pid_sh % num_q_heads
    kv_head_idx = head_idx // N_REP

    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # Load Q for this head — split into even/odd for packed K dot product
    q_base = Q_ptr + seq_idx * q_stride_seq + head_idx * q_stride_head
    even_offs = tl.arange(0, HALF_HD) * 2
    odd_offs = tl.arange(0, HALF_HD) * 2 + 1
    q_even = tl.load(q_base + even_offs, mask=even_offs < HEAD_DIM).to(tl.float32)
    q_odd = tl.load(q_base + odd_offs, mask=odd_offs < HEAD_DIM).to(tl.float32)

    packed_d_offs = tl.arange(0, HALF_HD)
    group_idx = packed_d_offs // BYTES_PER_GROUP  # which group each packed byte belongs to

    # Online softmax accumulators
    m_i = float("-inf")
    l_i = 0.0
    acc_even = tl.zeros([HALF_HD], dtype=tl.float32)
    acc_odd = tl.zeros([HALF_HD], dtype=tl.float32)

    # Iterate over context in tiles of BLOCK_N
    for start_n in range(0, seq_len, BLOCK_N):
        n_offs = start_n + tl.arange(0, BLOCK_N)
        n_mask = n_offs < seq_len

        # Resolve paged addresses: each position maps to a physical block
        logical_block = n_offs // BLOCK_SIZE
        block_offset = n_offs % BLOCK_SIZE
        phys_block = tl.load(
            block_table_ptr + seq_idx * bt_stride_seq + logical_block,
            mask=n_mask, other=0,
        ).to(tl.int64)

        # ---- Load and dequantize K ----
        k_base = (phys_block[:, None] * kc_stride_block
                  + block_offset[:, None] * kc_stride_page
                  + kv_head_idx * kc_stride_head)
        k_packed = tl.load(
            K_cache_ptr + k_base + packed_d_offs[None, :],
            mask=n_mask[:, None], other=0,
        ).to(tl.uint8)

        if IS_ASYMMETRIC:
            k_low = (k_packed & 0x0F).to(tl.float32)
            k_high = ((k_packed >> 4) & 0x0F).to(tl.float32)
        else:
            k_low = ((k_packed & 0x0F).to(tl.int8) - 8).to(tl.float32)
            k_high = (((k_packed >> 4) & 0x0F).to(tl.int8) - 8).to(tl.float32)

        # Load K scales and broadcast to packed dimension
        ks_base = (phys_block[:, None] * ks_stride_block
                   + block_offset[:, None] * ks_stride_page
                   + kv_head_idx * ks_stride_head)
        scale_k_expanded = tl.zeros([BLOCK_N, HALF_HD], dtype=tl.float32)
        zero_k_expanded = tl.zeros([BLOCK_N, HALF_HD], dtype=tl.float32)
        if IS_ASYMMETRIC:
            kz_base = (phys_block[:, None] * kz_stride_block
                       + block_offset[:, None] * kz_stride_page
                       + kv_head_idx * kz_stride_head)
        for g in tl.static_range(0, NUM_GROUPS):
            g_mask = (group_idx == g).to(tl.float32)
            sk_g = tl.load(
                K_scales_ptr + ks_base + g,
                mask=n_mask[:, None], other=1.0,
            ).to(tl.float32)
            scale_k_expanded += sk_g * g_mask[None, :]
            if IS_ASYMMETRIC:
                zk_g = tl.load(
                    K_zeros_ptr + kz_base + g,
                    mask=n_mask[:, None], other=0.0,
                ).to(tl.float32)
                zero_k_expanded += zk_g * g_mask[None, :]

        k_low = k_low * scale_k_expanded
        k_high = k_high * scale_k_expanded
        if IS_ASYMMETRIC:
            k_low = k_low + zero_k_expanded
            k_high = k_high + zero_k_expanded

        # QK^T dot product using even/odd decomposition
        qk = (tl.sum(q_even[None, :] * k_low, axis=1)
              + tl.sum(q_odd[None, :] * k_high, axis=1))
        qk = qk * scale
        qk = tl.where(n_mask, qk, float("-inf"))

        # Online softmax update
        m_ij = tl.max(qk)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new)
        l_i = l_i * alpha + tl.sum(p)
        acc_even = acc_even * alpha
        acc_odd = acc_odd * alpha

        # ---- Load and dequantize V ----
        v_base = (phys_block[:, None] * vc_stride_block
                  + block_offset[:, None] * vc_stride_page
                  + kv_head_idx * vc_stride_head)
        v_packed = tl.load(
            V_cache_ptr + v_base + packed_d_offs[None, :],
            mask=n_mask[:, None], other=0,
        ).to(tl.uint8)

        if IS_ASYMMETRIC:
            v_low = (v_packed & 0x0F).to(tl.float32)
            v_high = ((v_packed >> 4) & 0x0F).to(tl.float32)
        else:
            v_low = ((v_packed & 0x0F).to(tl.int8) - 8).to(tl.float32)
            v_high = (((v_packed >> 4) & 0x0F).to(tl.int8) - 8).to(tl.float32)

        vs_base = (phys_block[:, None] * vs_stride_block
                   + block_offset[:, None] * vs_stride_page
                   + kv_head_idx * vs_stride_head)
        scale_v_expanded = tl.zeros([BLOCK_N, HALF_HD], dtype=tl.float32)
        zero_v_expanded = tl.zeros([BLOCK_N, HALF_HD], dtype=tl.float32)
        if IS_ASYMMETRIC:
            vz_base = (phys_block[:, None] * vz_stride_block
                       + block_offset[:, None] * vz_stride_page
                       + kv_head_idx * vz_stride_head)
        for g in tl.static_range(0, NUM_GROUPS):
            g_mask = (group_idx == g).to(tl.float32)
            sv_g = tl.load(
                V_scales_ptr + vs_base + g,
                mask=n_mask[:, None], other=1.0,
            ).to(tl.float32)
            scale_v_expanded += sv_g * g_mask[None, :]
            if IS_ASYMMETRIC:
                zv_g = tl.load(
                    V_zeros_ptr + vz_base + g,
                    mask=n_mask[:, None], other=0.0,
                ).to(tl.float32)
                zero_v_expanded += zv_g * g_mask[None, :]

        v_low = v_low * scale_v_expanded
        v_high = v_high * scale_v_expanded
        if IS_ASYMMETRIC:
            v_low = v_low + zero_v_expanded
            v_high = v_high + zero_v_expanded

        # Weighted V accumulation
        acc_even += tl.sum(p[:, None] * v_low, axis=0)
        acc_odd += tl.sum(p[:, None] * v_high, axis=0)
        m_i = m_new

    # Normalise by softmax denominator
    acc_even = acc_even / l_i
    acc_odd = acc_odd / l_i

    # Write output — interleave even/odd back to full head_dim
    out_base = Out_ptr + seq_idx * o_stride_seq + head_idx * o_stride_head
    tl.store(out_base + even_offs, acc_even.to(tl.float16),
             mask=even_offs < HEAD_DIM)
    tl.store(out_base + odd_offs, acc_odd.to(tl.float16),
             mask=odd_offs < HEAD_DIM)


# ===================================================================
# Python wrappers
# ===================================================================

def fused_int4_decode(
    query: torch.Tensor,  # [num_seqs, num_heads, head_size]
    key_cache: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, half_hd]
    value_cache: torch.Tensor,
    k_scales: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, num_groups]
    v_scales: torch.Tensor,
    block_table: torch.Tensor,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens: torch.Tensor,  # [num_seqs]
    num_kv_heads: int,
    head_size: int,
    block_n: int = 64,
    k_zeros: torch.Tensor | None = None,
    v_zeros: torch.Tensor | None = None,
    asymmetric: bool = False,
) -> torch.Tensor:
    """Launch the fused INT4 decode attention kernel."""
    num_seqs = query.shape[0]
    num_heads = query.shape[1]
    n_rep = num_heads // num_kv_heads
    half_hd = head_size // 2
    num_groups = head_size // GROUP_SIZE
    bytes_per_group = GROUP_SIZE // 2
    block_size = key_cache.shape[1]
    scale = 1.0 / (head_size ** 0.5)

    # Dummy tensors when symmetric (Triton needs valid pointers)
    if k_zeros is None:
        k_zeros = k_scales
    if v_zeros is None:
        v_zeros = v_scales

    output = torch.empty_like(query)

    grid = (num_seqs * num_heads,)
    _fused_int4_decode_kernel[grid](
        query,
        key_cache, value_cache,
        k_scales, v_scales,
        k_zeros, v_zeros,
        output,
        block_table, seq_lens,
        # Q strides
        query.stride(0), query.stride(1),
        # Packed cache strides
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
        # Scale strides
        k_scales.stride(0), k_scales.stride(1), k_scales.stride(2),
        v_scales.stride(0), v_scales.stride(1), v_scales.stride(2),
        # Zero strides
        k_zeros.stride(0), k_zeros.stride(1), k_zeros.stride(2),
        v_zeros.stride(0), v_zeros.stride(1), v_zeros.stride(2),
        # Output strides
        output.stride(0), output.stride(1),
        # Block table stride
        block_table.stride(0),
        # Dims
        num_kv_heads=num_kv_heads,
        scale=scale,
        HEAD_DIM=head_size,
        HALF_HD=half_hd,
        NUM_GROUPS=num_groups,
        BYTES_PER_GROUP=bytes_per_group,
        BLOCK_SIZE=block_size,
        BLOCK_N=min(block_n, 128),
        N_REP=n_rep,
        IS_ASYMMETRIC=asymmetric,
    )
    return output


# ===================================================================
# Triton kernel: K_FP16 / V_INT4 hybrid decode attention
# Reads K from the FP16 paged cache, V from packed INT4 cache.
# Tests the paper claim: Qwen-class models need high K precision.
# ===================================================================
@triton.jit
def _kfp16_vint4_decode_kernel(
    # Query
    Q_ptr,  # [num_seqs, num_heads, head_size]
    # FP16 K paged cache (written by reshape_and_cache_flash)
    K_fp16_ptr,  # [num_blocks, block_size, num_kv_heads, head_size]
    # Packed INT4 V cache
    V_cache_ptr,  # [num_blocks, block_size, num_kv_heads, half_hd]
    V_scales_ptr,  # [num_blocks, block_size, num_kv_heads, num_groups]
    V_zeros_ptr,
    # Output
    Out_ptr,  # [num_seqs, num_heads, head_size]
    # Block table and seq lens
    block_table_ptr,
    seq_lens_ptr,
    # Strides — Q
    q_stride_seq: tl.int64,
    q_stride_head: tl.int64,
    # Strides — FP16 K cache
    kf_stride_block: tl.int64,
    kf_stride_page: tl.int64,
    kf_stride_head: tl.int64,
    # Strides — packed V cache
    vc_stride_block: tl.int64,
    vc_stride_page: tl.int64,
    vc_stride_head: tl.int64,
    # Strides — V scales
    vs_stride_block: tl.int64,
    vs_stride_page: tl.int64,
    vs_stride_head: tl.int64,
    # Strides — V zeros
    vz_stride_block: tl.int64,
    vz_stride_page: tl.int64,
    vz_stride_head: tl.int64,
    # Strides — output
    o_stride_seq: tl.int64,
    o_stride_head: tl.int64,
    # Strides — block table
    bt_stride_seq: tl.int64,
    # Dims
    num_kv_heads: tl.int32,
    scale: tl.float32,
    HEAD_DIM: tl.constexpr,
    HALF_HD: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BYTES_PER_GROUP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    N_REP: tl.constexpr,
    IS_ASYMMETRIC: tl.constexpr,
):
    """Hybrid decode: FP16 K (full precision) + INT4 V (quantised)."""
    pid_sh = tl.program_id(0)
    num_q_heads = num_kv_heads * N_REP
    seq_idx = pid_sh // num_q_heads
    head_idx = pid_sh % num_q_heads
    kv_head_idx = head_idx // N_REP

    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # Load Q — full precision for K dot product
    q_base = Q_ptr + seq_idx * q_stride_seq + head_idx * q_stride_head
    d_offs = tl.arange(0, HEAD_DIM)
    q_vec = tl.load(q_base + d_offs, mask=d_offs < HEAD_DIM).to(tl.float32)

    # V uses packed even/odd decomposition
    packed_d_offs = tl.arange(0, HALF_HD)
    group_idx = packed_d_offs // BYTES_PER_GROUP

    # Online softmax accumulators (V is still even/odd packed)
    m_i = float("-inf")
    l_i = 0.0
    acc_even = tl.zeros([HALF_HD], dtype=tl.float32)
    acc_odd = tl.zeros([HALF_HD], dtype=tl.float32)

    for start_n in range(0, seq_len, BLOCK_N):
        n_offs = start_n + tl.arange(0, BLOCK_N)
        n_mask = n_offs < seq_len

        logical_block = n_offs // BLOCK_SIZE
        block_offset = n_offs % BLOCK_SIZE
        phys_block = tl.load(
            block_table_ptr + seq_idx * bt_stride_seq + logical_block,
            mask=n_mask, other=0,
        ).to(tl.int64)

        # ---- Load K from FP16 paged cache (full precision) ----
        k_base = (phys_block[:, None] * kf_stride_block
                  + block_offset[:, None] * kf_stride_page
                  + kv_head_idx * kf_stride_head)
        k_fp16 = tl.load(
            K_fp16_ptr + k_base + d_offs[None, :],
            mask=n_mask[:, None] & (d_offs[None, :] < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)

        # QK^T: full-precision dot product
        qk = tl.sum(q_vec[None, :] * k_fp16, axis=1) * scale
        qk = tl.where(n_mask, qk, float("-inf"))

        # Online softmax
        m_ij = tl.max(qk)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new)
        l_i = l_i * alpha + tl.sum(p)
        acc_even = acc_even * alpha
        acc_odd = acc_odd * alpha

        # ---- Load and dequantize V from INT4 cache ----
        v_base = (phys_block[:, None] * vc_stride_block
                  + block_offset[:, None] * vc_stride_page
                  + kv_head_idx * vc_stride_head)
        v_packed = tl.load(
            V_cache_ptr + v_base + packed_d_offs[None, :],
            mask=n_mask[:, None], other=0,
        ).to(tl.uint8)

        if IS_ASYMMETRIC:
            v_low = (v_packed & 0x0F).to(tl.float32)
            v_high = ((v_packed >> 4) & 0x0F).to(tl.float32)
        else:
            v_low = ((v_packed & 0x0F).to(tl.int8) - 8).to(tl.float32)
            v_high = (((v_packed >> 4) & 0x0F).to(tl.int8) - 8).to(tl.float32)

        vs_base = (phys_block[:, None] * vs_stride_block
                   + block_offset[:, None] * vs_stride_page
                   + kv_head_idx * vs_stride_head)
        scale_v_expanded = tl.zeros([BLOCK_N, HALF_HD], dtype=tl.float32)
        zero_v_expanded = tl.zeros([BLOCK_N, HALF_HD], dtype=tl.float32)
        if IS_ASYMMETRIC:
            vz_base = (phys_block[:, None] * vz_stride_block
                       + block_offset[:, None] * vz_stride_page
                       + kv_head_idx * vz_stride_head)
        for g in tl.static_range(0, NUM_GROUPS):
            g_mask = (group_idx == g).to(tl.float32)
            sv_g = tl.load(
                V_scales_ptr + vs_base + g,
                mask=n_mask[:, None], other=1.0,
            ).to(tl.float32)
            scale_v_expanded += sv_g * g_mask[None, :]
            if IS_ASYMMETRIC:
                zv_g = tl.load(
                    V_zeros_ptr + vz_base + g,
                    mask=n_mask[:, None], other=0.0,
                ).to(tl.float32)
                zero_v_expanded += zv_g * g_mask[None, :]

        v_low = v_low * scale_v_expanded
        v_high = v_high * scale_v_expanded
        if IS_ASYMMETRIC:
            v_low = v_low + zero_v_expanded
            v_high = v_high + zero_v_expanded

        acc_even += tl.sum(p[:, None] * v_low, axis=0)
        acc_odd += tl.sum(p[:, None] * v_high, axis=0)
        m_i = m_new

    acc_even = acc_even / l_i
    acc_odd = acc_odd / l_i

    # Write output — interleave even/odd
    out_base = Out_ptr + seq_idx * o_stride_seq + head_idx * o_stride_head
    even_offs = tl.arange(0, HALF_HD) * 2
    odd_offs = tl.arange(0, HALF_HD) * 2 + 1
    tl.store(out_base + even_offs, acc_even.to(tl.float16),
             mask=even_offs < HEAD_DIM)
    tl.store(out_base + odd_offs, acc_odd.to(tl.float16),
             mask=odd_offs < HEAD_DIM)


def kfp16_vint4_decode(
    query: torch.Tensor,          # [num_seqs, num_heads, head_size]
    key_cache_fp16: torch.Tensor,  # [num_blocks, block_size, kv_heads, head_size]
    value_cache_int4: torch.Tensor,  # [num_blocks, block_size, kv_heads, half_hd]
    v_scales: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    num_kv_heads: int,
    head_size: int,
    block_n: int = 64,
    v_zeros: torch.Tensor | None = None,
    asymmetric: bool = False,
) -> torch.Tensor:
    """K_FP16 / V_INT4 hybrid decode: K from FP16 cache, V from INT4."""
    num_seqs = query.shape[0]
    num_heads = query.shape[1]
    n_rep = num_heads // num_kv_heads
    half_hd = head_size // 2
    num_groups = head_size // GROUP_SIZE
    bytes_per_group = GROUP_SIZE // 2
    block_size = key_cache_fp16.shape[1]
    scale = 1.0 / (head_size ** 0.5)

    if v_zeros is None:
        v_zeros = v_scales

    output = torch.empty_like(query)

    grid = (num_seqs * num_heads,)
    _kfp16_vint4_decode_kernel[grid](
        query,
        key_cache_fp16, value_cache_int4,
        v_scales, v_zeros,
        output,
        block_table, seq_lens,
        query.stride(0), query.stride(1),
        key_cache_fp16.stride(0), key_cache_fp16.stride(1),
        key_cache_fp16.stride(2),
        value_cache_int4.stride(0), value_cache_int4.stride(1),
        value_cache_int4.stride(2),
        v_scales.stride(0), v_scales.stride(1), v_scales.stride(2),
        v_zeros.stride(0), v_zeros.stride(1), v_zeros.stride(2),
        output.stride(0), output.stride(1),
        block_table.stride(0),
        num_kv_heads=num_kv_heads,
        scale=scale,
        HEAD_DIM=head_size,
        HALF_HD=half_hd,
        NUM_GROUPS=num_groups,
        BYTES_PER_GROUP=bytes_per_group,
        BLOCK_SIZE=block_size,
        BLOCK_N=min(block_n, 128),
        N_REP=n_rep,
        IS_ASYMMETRIC=asymmetric,
    )
    return output


# ===================================================================
# Triton kernel: K_INT8 quantize + cache write
# Quantises K to symmetric INT8 (1 byte per element, per-group scale).
# ===================================================================
@triton.jit
def _reshape_and_cache_k_int8_kernel(
    key_ptr,  # [num_tokens, num_heads, head_size]
    key_cache_ptr,  # [num_blocks, block_size, num_heads, head_size] int8
    k_scales_ptr,  # [num_blocks, block_size, num_heads, num_groups] fp16
    slot_mapping_ptr,
    key_stride_token: tl.int64,
    key_stride_head: tl.int64,
    kc_stride_block: tl.int64,
    kc_stride_page: tl.int64,
    kc_stride_head: tl.int64,
    ks_stride_block: tl.int64,
    ks_stride_page: tl.int64,
    ks_stride_head: tl.int64,
    num_heads: tl.constexpr,
    head_size: tl.constexpr,
    block_size: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
):
    """Quantise K to symmetric INT8 with per-group scales."""
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    slot_idx = tl.load(slot_mapping_ptr + token_idx).to(tl.int64)
    if slot_idx < 0:
        return

    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size

    src_base = token_idx * key_stride_token + head_idx * key_stride_head
    dst_base = (block_idx * kc_stride_block
                + block_offset * kc_stride_page
                + head_idx * kc_stride_head)
    dst_s_base = (block_idx * ks_stride_block
                  + block_offset * ks_stride_page
                  + head_idx * ks_stride_head)

    for g in tl.static_range(0, NUM_GROUPS):
        g_start = g * GROUP_SIZE
        offs = tl.arange(0, GROUP_SIZE)
        vals = tl.load(key_ptr + src_base + g_start + offs,
                       mask=(g_start + offs) < head_size).to(tl.float32)
        amax = tl.max(tl.abs(vals))
        amax = tl.maximum(amax, 1e-8)
        k_scale = amax / 127.0

        q_vals = _triton_round(vals / k_scale)
        q_vals = tl.maximum(tl.minimum(q_vals, 127.0), -128.0)

        tl.store(key_cache_ptr + dst_base + g_start + offs,
                 q_vals.to(tl.int8), mask=(g_start + offs) < head_size)
        tl.store(k_scales_ptr + dst_s_base + g, k_scale.to(tl.float16))


def reshape_and_cache_k_int8(
    key: torch.Tensor,
    key_cache_int8: torch.Tensor,  # [num_blocks, block_size, num_heads, head_size] int8
    k_scales: torch.Tensor,       # [num_blocks, block_size, num_heads, num_groups] fp16
    slot_mapping: torch.Tensor,
) -> None:
    """Quantise K to symmetric INT8 and write into dedicated cache."""
    num_tokens = key.shape[0]
    num_heads = key.shape[1]
    head_size = key.shape[2]
    block_size = key_cache_int8.shape[1]
    num_groups = head_size // GROUP_SIZE

    grid = (num_tokens, num_heads)
    _reshape_and_cache_k_int8_kernel[grid](
        key, key_cache_int8, k_scales, slot_mapping,
        key.stride(0), key.stride(1),
        key_cache_int8.stride(0), key_cache_int8.stride(1),
        key_cache_int8.stride(2),
        k_scales.stride(0), k_scales.stride(1), k_scales.stride(2),
        num_heads=num_heads,
        head_size=head_size,
        block_size=block_size,
        GROUP_SIZE=GROUP_SIZE,
        NUM_GROUPS=num_groups,
    )


# ===================================================================
# Triton kernel: K_INT8 / V_INT4 hybrid decode attention
# ===================================================================
@triton.jit
def _kint8_vint4_decode_kernel(
    Q_ptr,
    K_cache_ptr,  # [num_blocks, block_size, num_kv_heads, head_size] int8
    V_cache_ptr,  # [num_blocks, block_size, num_kv_heads, half_hd] uint8
    K_scales_ptr,
    V_scales_ptr,
    V_zeros_ptr,
    Out_ptr,
    block_table_ptr,
    seq_lens_ptr,
    q_stride_seq: tl.int64,
    q_stride_head: tl.int64,
    kc_stride_block: tl.int64,
    kc_stride_page: tl.int64,
    kc_stride_head: tl.int64,
    vc_stride_block: tl.int64,
    vc_stride_page: tl.int64,
    vc_stride_head: tl.int64,
    ks_stride_block: tl.int64,
    ks_stride_page: tl.int64,
    ks_stride_head: tl.int64,
    vs_stride_block: tl.int64,
    vs_stride_page: tl.int64,
    vs_stride_head: tl.int64,
    vz_stride_block: tl.int64,
    vz_stride_page: tl.int64,
    vz_stride_head: tl.int64,
    o_stride_seq: tl.int64,
    o_stride_head: tl.int64,
    bt_stride_seq: tl.int64,
    num_kv_heads: tl.int32,
    scale: tl.float32,
    HEAD_DIM: tl.constexpr,
    HALF_HD: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BYTES_PER_GROUP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    N_REP: tl.constexpr,
    IS_ASYMMETRIC: tl.constexpr,
    K_GROUP_SIZE: tl.constexpr,
):
    """Hybrid decode: INT8 K + INT4 V."""
    pid_sh = tl.program_id(0)
    num_q_heads = num_kv_heads * N_REP
    seq_idx = pid_sh // num_q_heads
    head_idx = pid_sh % num_q_heads
    kv_head_idx = head_idx // N_REP

    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # Load Q (full precision)
    q_base = Q_ptr + seq_idx * q_stride_seq + head_idx * q_stride_head
    d_offs = tl.arange(0, HEAD_DIM)
    q_vec = tl.load(q_base + d_offs, mask=d_offs < HEAD_DIM).to(tl.float32)

    # V packed offsets and group mapping
    packed_d_offs = tl.arange(0, HALF_HD)
    group_idx_packed = packed_d_offs // BYTES_PER_GROUP
    # K group mapping over full head_dim
    group_idx_full = d_offs // K_GROUP_SIZE

    m_i = float("-inf")
    l_i = 0.0
    acc_even = tl.zeros([HALF_HD], dtype=tl.float32)
    acc_odd = tl.zeros([HALF_HD], dtype=tl.float32)

    for start_n in range(0, seq_len, BLOCK_N):
        n_offs = start_n + tl.arange(0, BLOCK_N)
        n_mask = n_offs < seq_len

        logical_block = n_offs // BLOCK_SIZE
        block_offset = n_offs % BLOCK_SIZE
        phys_block = tl.load(
            block_table_ptr + seq_idx * bt_stride_seq + logical_block,
            mask=n_mask, other=0,
        ).to(tl.int64)

        # ---- Load K from INT8 cache ----
        k_base = (phys_block[:, None] * kc_stride_block
                  + block_offset[:, None] * kc_stride_page
                  + kv_head_idx * kc_stride_head)
        k_int8 = tl.load(
            K_cache_ptr + k_base + d_offs[None, :],
            mask=n_mask[:, None] & (d_offs[None, :] < HEAD_DIM),
            other=0,
        ).to(tl.float32)

        # Load K scales and dequantise
        ks_base = (phys_block[:, None] * ks_stride_block
                   + block_offset[:, None] * ks_stride_page
                   + kv_head_idx * ks_stride_head)
        scale_k_expanded = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
        for g in tl.static_range(0, NUM_GROUPS):
            g_mask_k = (group_idx_full == g).to(tl.float32)
            sk_g = tl.load(
                K_scales_ptr + ks_base + g,
                mask=n_mask[:, None], other=1.0,
            ).to(tl.float32)
            scale_k_expanded += sk_g * g_mask_k[None, :]

        k_deq = k_int8 * scale_k_expanded

        # QK^T
        qk = tl.sum(q_vec[None, :] * k_deq, axis=1) * scale
        qk = tl.where(n_mask, qk, float("-inf"))

        # Online softmax
        m_ij = tl.max(qk)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new)
        l_i = l_i * alpha + tl.sum(p)
        acc_even = acc_even * alpha
        acc_odd = acc_odd * alpha

        # ---- Load and dequantize V from INT4 cache ----
        v_base = (phys_block[:, None] * vc_stride_block
                  + block_offset[:, None] * vc_stride_page
                  + kv_head_idx * vc_stride_head)
        v_packed = tl.load(
            V_cache_ptr + v_base + packed_d_offs[None, :],
            mask=n_mask[:, None], other=0,
        ).to(tl.uint8)

        if IS_ASYMMETRIC:
            v_low = (v_packed & 0x0F).to(tl.float32)
            v_high = ((v_packed >> 4) & 0x0F).to(tl.float32)
        else:
            v_low = ((v_packed & 0x0F).to(tl.int8) - 8).to(tl.float32)
            v_high = (((v_packed >> 4) & 0x0F).to(tl.int8) - 8).to(tl.float32)

        vs_base = (phys_block[:, None] * vs_stride_block
                   + block_offset[:, None] * vs_stride_page
                   + kv_head_idx * vs_stride_head)
        scale_v_expanded = tl.zeros([BLOCK_N, HALF_HD], dtype=tl.float32)
        zero_v_expanded = tl.zeros([BLOCK_N, HALF_HD], dtype=tl.float32)
        if IS_ASYMMETRIC:
            vz_base = (phys_block[:, None] * vz_stride_block
                       + block_offset[:, None] * vz_stride_page
                       + kv_head_idx * vz_stride_head)
        for g in tl.static_range(0, NUM_GROUPS):
            g_mask_v = (group_idx_packed == g).to(tl.float32)
            sv_g = tl.load(
                V_scales_ptr + vs_base + g,
                mask=n_mask[:, None], other=1.0,
            ).to(tl.float32)
            scale_v_expanded += sv_g * g_mask_v[None, :]
            if IS_ASYMMETRIC:
                zv_g = tl.load(
                    V_zeros_ptr + vz_base + g,
                    mask=n_mask[:, None], other=0.0,
                ).to(tl.float32)
                zero_v_expanded += zv_g * g_mask_v[None, :]

        v_low = v_low * scale_v_expanded
        v_high = v_high * scale_v_expanded
        if IS_ASYMMETRIC:
            v_low = v_low + zero_v_expanded
            v_high = v_high + zero_v_expanded

        acc_even += tl.sum(p[:, None] * v_low, axis=0)
        acc_odd += tl.sum(p[:, None] * v_high, axis=0)
        m_i = m_new

    acc_even = acc_even / l_i
    acc_odd = acc_odd / l_i

    out_base = Out_ptr + seq_idx * o_stride_seq + head_idx * o_stride_head
    even_offs = tl.arange(0, HALF_HD) * 2
    odd_offs = tl.arange(0, HALF_HD) * 2 + 1
    tl.store(out_base + even_offs, acc_even.to(tl.float16),
             mask=even_offs < HEAD_DIM)
    tl.store(out_base + odd_offs, acc_odd.to(tl.float16),
             mask=odd_offs < HEAD_DIM)


def kint8_vint4_decode(
    query: torch.Tensor,
    key_cache_int8: torch.Tensor,  # [num_blocks, block_size, kv_heads, head_size] int8
    value_cache_int4: torch.Tensor,  # [num_blocks, block_size, kv_heads, half_hd] uint8
    k_scales: torch.Tensor,
    v_scales: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    num_kv_heads: int,
    head_size: int,
    block_n: int = 64,
    v_zeros: torch.Tensor | None = None,
    asymmetric: bool = False,
) -> torch.Tensor:
    """K_INT8 / V_INT4 hybrid decode."""
    num_seqs = query.shape[0]
    num_heads = query.shape[1]
    n_rep = num_heads // num_kv_heads
    half_hd = head_size // 2
    num_groups = head_size // GROUP_SIZE
    bytes_per_group = GROUP_SIZE // 2
    block_size = key_cache_int8.shape[1]
    scale_val = 1.0 / (head_size ** 0.5)

    if v_zeros is None:
        v_zeros = v_scales

    output = torch.empty_like(query)

    grid = (num_seqs * num_heads,)
    _kint8_vint4_decode_kernel[grid](
        query,
        key_cache_int8, value_cache_int4,
        k_scales, v_scales, v_zeros,
        output,
        block_table, seq_lens,
        query.stride(0), query.stride(1),
        key_cache_int8.stride(0), key_cache_int8.stride(1),
        key_cache_int8.stride(2),
        value_cache_int4.stride(0), value_cache_int4.stride(1),
        value_cache_int4.stride(2),
        k_scales.stride(0), k_scales.stride(1), k_scales.stride(2),
        v_scales.stride(0), v_scales.stride(1), v_scales.stride(2),
        v_zeros.stride(0), v_zeros.stride(1), v_zeros.stride(2),
        output.stride(0), output.stride(1),
        block_table.stride(0),
        num_kv_heads=num_kv_heads,
        scale=scale_val,
        HEAD_DIM=head_size,
        HALF_HD=half_hd,
        NUM_GROUPS=num_groups,
        BYTES_PER_GROUP=bytes_per_group,
        BLOCK_SIZE=block_size,
        BLOCK_N=min(block_n, 128),
        N_REP=n_rep,
        IS_ASYMMETRIC=asymmetric,
        K_GROUP_SIZE=GROUP_SIZE,
    )
    return output


def decode_fp16_sdpa(
    query: torch.Tensor,         # [num_seqs, num_heads, head_size]
    fp16_k: torch.Tensor,        # [max_seqs, max_ctx, num_kv_heads, head_size]
    fp16_v: torch.Tensor,        # [max_seqs, max_ctx, num_kv_heads, head_size]
    seq_lens: torch.Tensor,      # [num_seqs]
    num_kv_heads: int,
    num_heads: int,
    head_size: int,
) -> torch.Tensor:
    """Decode attention using FP16 shadow K/V for short sequences.

    Uses the original FP16 K/V stored in the shadow buffer, avoiding
    INT4 quantization error entirely.
    """
    num_seqs = query.shape[0]
    n_rep = num_heads // num_kv_heads
    scale = 1.0 / (head_size ** 0.5)

    outputs = []
    for s in range(num_seqs):
        sl = seq_lens[s].item()
        k_fp16 = fp16_k[s, :sl]  # [sl, num_kv_heads, head_size]
        v_fp16 = fp16_v[s, :sl]  # [sl, num_kv_heads, head_size]

        # GQA expansion
        if n_rep > 1:
            k_fp16 = k_fp16.repeat_interleave(n_rep, dim=1)
            v_fp16 = v_fp16.repeat_interleave(n_rep, dim=1)

        # SDPA: [1, H, S, D]
        q = query[s:s+1].transpose(0, 1).unsqueeze(0)   # [1, H, 1, D]
        k = k_fp16.transpose(0, 1).unsqueeze(0)          # [1, H, S, D]
        v = v_fp16.transpose(0, 1).unsqueeze(0)          # [1, H, S, D]

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=False,
            scale=scale,
        )  # [1, H, 1, D]
        outputs.append(out.squeeze(0).transpose(0, 1))   # [1, H, D]

    return torch.cat(outputs, dim=0)  # [num_seqs, H, D]


# ===================================================================
# Attention metadata
# ===================================================================

@dataclass
class FusedInt4AttentionMetadata:
    """Metadata for fused INT4 decode attention."""
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor


# ===================================================================
# Backend class
# ===================================================================

class FusedInt4AttentionBackend(AttentionBackend):
    """Attention backend using fused INT4 KV cache with in-kernel dequant."""

    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = False

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16, torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "int4_fused",
    ]

    @staticmethod
    def get_name() -> str:
        return "FUSED_INT4"

    @staticmethod
    def get_impl_cls() -> type["FusedInt4AttentionImpl"]:
        return FusedInt4AttentionImpl

    @staticmethod
    def get_builder_cls() -> type["FusedInt4AttentionMetadataBuilder"]:
        return FusedInt4AttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "int4_fused",
    ) -> tuple[int, ...]:
        """Cache shape for INT4 fused (allocator-compatible).

        The vLLM allocator (gpu_model_runner._reshape_kv_cache_tensors)
        reinterprets the raw byte buffer as model dtype (FP16/BF16) before
        reshaping via .view(dtype).view(shape).  To stay compatible we
        must return a shape whose total element count matches the
        FP16-reinterpreted buffer size.

        The allocator gives us page_size_bytes * num_blocks raw bytes.
        page_size_bytes = 2 * block_size * num_kv_heads * head_size *
        dtype_size (FP16 = 2).  After .view(float16) the element count
        is halved.  Our shape must consume exactly those elements:

            total_fp16_elems = page_size_bytes * num_blocks / 2
                             = 2 * block_size * num_kv_heads * head_size
                               * num_blocks

        Shape returned: (2, num_blocks, block_size, num_kv_heads, head_size)
        Total = 2 * num_blocks * block_size * num_kv_heads * head_size
        which matches.

        The forward() method then reinterprets as uint8 to get the actual
        packed INT4 cache with last dim = head_size * 2 packed bytes.
        Each K/V plane: [num_blocks, block_size, num_kv_heads, head_size]
        in FP16 view = [num_blocks, block_size, num_kv_heads, head_size*2]
        in uint8 view.  head_size*2 bytes = head_size packed INT4 pairs.
        """
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @classmethod
    def supports_kv_cache_dtype(
        cls, kv_cache_dtype: CacheDType | None,
    ) -> bool:
        if kv_cache_dtype is None:
            return True
        return kv_cache_dtype == "int4_fused"

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        # Head size must be divisible by GROUP_SIZE for clean grouping
        return head_size % GROUP_SIZE == 0 and head_size > 0

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER


# ===================================================================
# Metadata builder
# ===================================================================

class FusedInt4AttentionMetadataBuilder(
    AttentionMetadataBuilder[FusedInt4AttentionMetadata],
):
    _cudagraph_support = AttentionCGSupport.NEVER

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(reorder_batch_threshold=1)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FusedInt4AttentionMetadata:
        return FusedInt4AttentionMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
        )


# ===================================================================
# Attention implementation
# ===================================================================

class FusedInt4AttentionImpl(
    AttentionImpl[FusedInt4AttentionMetadata],
):
    """Fused INT4 attention with dedicated INT4 cache buffers.

    Architecture (separate-cache design):
      - The allocator-provided kv_cache (FP16 paged) is written by the
        standard do_kv_cache_update path (reshape_and_cache_flash).
      - A *separate* dedicated INT4 packed cache (uint8) + per-group
        scales (FP16) is allocated lazily and maintained by this impl.
      - Prefill uses fresh FP16 K/V directly with SDPA (no cache read).
      - Decode writes INT4 to the dedicated buffer and reads from it
        via the fused Triton kernel.  Short sequences (< MIN_FUSED_SEQ_LEN)
        read from the FP16 paged cache via SDPA instead.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "int4_fused",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
    ) -> None:
        if alibi_slopes is not None:
            raise NotImplementedError(
                "FusedInt4 backend does not support ALiBi yet"
            )
        if sliding_window is not None:
            raise NotImplementedError(
                "FusedInt4 backend does not support sliding window yet"
            )
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads or num_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.half_hd = head_size // 2
        self.num_groups = head_size // GROUP_SIZE

        # Dedicated INT4 packed cache — separate from allocator buffer.
        # Lazily allocated on first forward with cache.
        # Shape: [num_blocks, block_size, num_kv_heads, half_hd] uint8
        self._int4_key_cache: torch.Tensor | None = None
        self._int4_value_cache: torch.Tensor | None = None
        self._k_scales: torch.Tensor | None = None
        self._v_scales: torch.Tensor | None = None
        self._k_zeros: torch.Tensor | None = None
        self._v_zeros: torch.Tensor | None = None
        self._asymmetric: bool = ASYMMETRIC
        self._k_precision: str = K_PRECISION  # "int4" | "int8" | "fp16"

        # INT8 K cache (only allocated when k_precision == "int8")
        self._int8_key_cache: torch.Tensor | None = None
        self._k_int8_scales: torch.Tensor | None = None

        self._logged_init: bool = False

    def _ensure_int4_cache(
        self, num_blocks: int, block_size: int, device: torch.device,
    ) -> None:
        """Lazily allocate dedicated cache buffers (INT4 V always, K per policy)."""
        if self._int4_value_cache is not None:
            return
        logger.info(
            "[FusedInt4] Allocating caches: k_precision=%s, "
            "num_blocks=%d, block_size=%d, num_kv_heads=%d, half_hd=%d, "
            "asymmetric=%s, group_size=%d",
            self._k_precision,
            num_blocks, block_size, self.num_kv_heads, self.half_hd,
            self._asymmetric, GROUP_SIZE,
        )
        # V cache is always INT4
        self._int4_value_cache = torch.zeros(
            num_blocks, block_size, self.num_kv_heads, self.half_hd,
            dtype=torch.uint8, device=device,
        )
        self._v_scales = torch.zeros(
            num_blocks, block_size, self.num_kv_heads, self.num_groups,
            dtype=torch.float16, device=device,
        )
        if self._asymmetric:
            self._v_zeros = torch.zeros(
                num_blocks, block_size, self.num_kv_heads, self.num_groups,
                dtype=torch.float16, device=device,
            )

        if self._k_precision == "int4":
            # K cache as INT4 packed (existing behaviour)
            self._int4_key_cache = torch.zeros(
                num_blocks, block_size, self.num_kv_heads, self.half_hd,
                dtype=torch.uint8, device=device,
            )
            self._k_scales = torch.zeros(
                num_blocks, block_size, self.num_kv_heads, self.num_groups,
                dtype=torch.float16, device=device,
            )
            if self._asymmetric:
                self._k_zeros = torch.zeros(
                    num_blocks, block_size, self.num_kv_heads, self.num_groups,
                    dtype=torch.float16, device=device,
                )
        elif self._k_precision == "int8":
            # K cache as INT8 (1 byte per element, no packing)
            self._int8_key_cache = torch.zeros(
                num_blocks, block_size, self.num_kv_heads, self.head_size,
                dtype=torch.int8, device=device,
            )
            self._k_int8_scales = torch.zeros(
                num_blocks, block_size, self.num_kv_heads, self.num_groups,
                dtype=torch.float16, device=device,
            )
        # else: k_precision == "fp16" — K read from FP16 paged cache, no extra alloc

        # Scratch K buffers for non-int4 K modes: the reshape_and_cache_int4
        # kernel always writes both K and V, so we need somewhere for K to go.
        if self._k_precision != "int4":
            self._scratch_k_cache = torch.zeros(
                num_blocks, block_size, self.num_kv_heads, self.half_hd,
                dtype=torch.uint8, device=device,
            )
            self._scratch_k_scales = torch.zeros(
                num_blocks, block_size, self.num_kv_heads, self.num_groups,
                dtype=torch.float16, device=device,
            )
        else:
            self._scratch_k_cache = None
            self._scratch_k_scales = None

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Write FP16 K/V to the standard paged cache.

        Called by the framework because forward_includes_kv_cache_update
        is False.  This keeps the allocator-provided FP16 paged cache
        correct for the short-sequence SDPA fallback.
        """
        from vllm.v1.attention.backends.fa_utils import (
            reshape_and_cache_flash,
        )
        key_cache, value_cache = kv_cache.unbind(0)
        reshape_and_cache_flash(
            key, value, key_cache, value_cache, slot_mapping,
            "auto", layer._k_scale, layer._v_scale,
        )

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FusedInt4AttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # During warmup/profiling attn_metadata may be None
        if attn_metadata is None:
            if output is not None:
                output.zero_()
                return output
            return torch.zeros_like(query)

        if kv_cache.numel() == 0:
            if output is not None:
                output.zero_()
                return output
            return torch.zeros_like(query)

        # The allocator-provided kv_cache is FP16 paged:
        #   (2, num_blocks, block_size, num_kv_heads, head_size) in FP16
        # It was already written by do_kv_cache_update (reshape_and_cache_flash).
        # We use it for short-sequence SDPA decode fallback.
        key_cache_fp16 = kv_cache[0]   # [num_blocks, block_size, kv_heads, hd]
        value_cache_fp16 = kv_cache[1]
        num_blocks = key_cache_fp16.shape[0]
        block_size = key_cache_fp16.shape[1]

        # Ensure dedicated INT4 cache exists
        self._ensure_int4_cache(num_blocks, block_size, kv_cache.device)

        # ---- Quantised cache write ----
        num_actual_tokens = attn_metadata.num_actual_tokens
        if num_actual_tokens > 0 and key.numel() > 0:
            if key.dim() == 2:
                key = key.view(-1, self.num_kv_heads, self.head_size)
                value = value.view(-1, self.num_kv_heads, self.head_size)

            if self._k_precision == "int4":
                # Both K and V quantised to INT4 (original path)
                reshape_and_cache_int4(
                    key, value,
                    self._int4_key_cache, self._int4_value_cache,
                    self._k_scales, self._v_scales,
                    attn_metadata.slot_mapping,
                    k_zeros=self._k_zeros,
                    v_zeros=self._v_zeros,
                    asymmetric=self._asymmetric,
                )
            else:
                # V always INT4.  The kernel writes K and V to separate
                # buffers, so pass a scratch buffer for K (never read).
                reshape_and_cache_int4(
                    key, value,
                    self._scratch_k_cache, self._int4_value_cache,
                    self._scratch_k_scales, self._v_scales,
                    attn_metadata.slot_mapping,
                    k_zeros=None,
                    v_zeros=self._v_zeros,
                    asymmetric=self._asymmetric,
                )

            if self._k_precision == "int8":
                reshape_and_cache_k_int8(
                    key,
                    self._int8_key_cache,
                    self._k_int8_scales,
                    attn_metadata.slot_mapping,
                )

        # ---- Decode path ----
        if attn_metadata.max_query_len == 1:
            if query.dim() == 2:
                query = query.view(-1, self.num_heads, self.head_size)

            num_seqs = attn_metadata.seq_lens.shape[0]
            decode_query = query[:num_seqs]
            use_fused = attn_metadata.max_seq_len >= MIN_FUSED_SEQ_LEN

            if not self._logged_init:
                self._logged_init = True
                logger.info(
                    "[FusedInt4] Backend: separate-cache mode, "
                    "k_precision=%s, decode_kernel=%s, "
                    "fallback=fp16_paged_sdpa, "
                    "group_size=%d, num_kv_heads=%d, head_size=%d, "
                    "min_fused_seq_len=%d, asymmetric=%s",
                    self._k_precision,
                    {
                        "int4": "fused_int4_triton",
                        "int8": "kint8_vint4_triton",
                        "fp16": "kfp16_vint4_triton",
                    }.get(self._k_precision, "unknown"),
                    GROUP_SIZE, self.num_kv_heads,
                    self.head_size, MIN_FUSED_SEQ_LEN,
                    self._asymmetric,
                )

            if use_fused:
                if self._k_precision == "fp16":
                    decode_output = kfp16_vint4_decode(
                        decode_query,
                        key_cache_fp16,
                        self._int4_value_cache,
                        self._v_scales,
                        attn_metadata.block_table,
                        attn_metadata.seq_lens,
                        num_kv_heads=self.num_kv_heads,
                        head_size=self.head_size,
                        block_n=64,
                        v_zeros=self._v_zeros,
                        asymmetric=self._asymmetric,
                    )
                elif self._k_precision == "int8":
                    decode_output = kint8_vint4_decode(
                        decode_query,
                        self._int8_key_cache,
                        self._int4_value_cache,
                        self._k_int8_scales,
                        self._v_scales,
                        attn_metadata.block_table,
                        attn_metadata.seq_lens,
                        num_kv_heads=self.num_kv_heads,
                        head_size=self.head_size,
                        block_n=64,
                        v_zeros=self._v_zeros,
                        asymmetric=self._asymmetric,
                    )
                else:
                    decode_output = fused_int4_decode(
                        decode_query,
                        self._int4_key_cache, self._int4_value_cache,
                        self._k_scales, self._v_scales,
                        attn_metadata.block_table,
                        attn_metadata.seq_lens,
                        num_kv_heads=self.num_kv_heads,
                        head_size=self.head_size,
                        block_n=64,
                        k_zeros=self._k_zeros,
                        v_zeros=self._v_zeros,
                        asymmetric=self._asymmetric,
                    )
            else:
                # Short sequence: read from FP16 paged cache via SDPA
                decode_output = self._decode_from_paged_fp16(
                    decode_query,
                    key_cache_fp16, value_cache_fp16,
                    attn_metadata.block_table,
                    attn_metadata.seq_lens,
                )

            if output is not None:
                output[:decode_output.shape[0]].copy_(
                    decode_output.view(
                        output[:decode_output.shape[0]].shape)
                )
                return output
            return decode_output.view(decode_output.shape[0], -1)
        else:
            # Prefill: use fresh FP16 K/V directly with SDPA
            return self._prefill_fallback(
                query, key, value, kv_cache, attn_metadata, output,
            )

    def _decode_from_paged_fp16(
        self,
        query: torch.Tensor,        # [num_seqs, num_heads, head_size]
        key_cache: torch.Tensor,     # [num_blocks, block_size, kv_heads, hd]
        value_cache: torch.Tensor,
        block_table: torch.Tensor,   # [num_seqs, max_blocks_per_seq]
        seq_lens: torch.Tensor,      # [num_seqs]
    ) -> torch.Tensor:
        """Decode attention reading from the standard FP16 paged cache."""
        num_seqs = query.shape[0]
        n_rep = self.num_heads // self.num_kv_heads
        scale = 1.0 / (self.head_size ** 0.5)
        block_size = key_cache.shape[1]

        outputs = []
        for s in range(num_seqs):
            sl = seq_lens[s].item()
            blocks_needed = (sl + block_size - 1) // block_size
            k_list, v_list = [], []
            for b in range(blocks_needed):
                block_idx = block_table[s, b].item()
                tokens_in_block = min(block_size, sl - b * block_size)
                k_list.append(key_cache[block_idx, :tokens_in_block])
                v_list.append(value_cache[block_idx, :tokens_in_block])

            k_fp16 = torch.cat(k_list, dim=0)  # [sl, kv_heads, hd]
            v_fp16 = torch.cat(v_list, dim=0)

            if n_rep > 1:
                k_fp16 = k_fp16.repeat_interleave(n_rep, dim=1)
                v_fp16 = v_fp16.repeat_interleave(n_rep, dim=1)

            q = query[s:s+1].transpose(0, 1).unsqueeze(0)  # [1, H, 1, D]
            k = k_fp16.transpose(0, 1).unsqueeze(0)         # [1, H, S, D]
            v = v_fp16.transpose(0, 1).unsqueeze(0)

            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=False, scale=scale,
            )
            outputs.append(out.squeeze(0).transpose(0, 1))  # [1, H, D]

        return torch.cat(outputs, dim=0)

    def _prefill_fallback(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FusedInt4AttentionMetadata,
        output: torch.Tensor | None,
    ) -> torch.Tensor:
        """Prefill using fresh FP16 K/V directly with SDPA."""
        if query.dim() == 2:
            query = query.view(-1, self.num_heads, self.head_size)
        if key.dim() == 2:
            key = key.view(-1, self.num_kv_heads, self.head_size)
            value = value.view(-1, self.num_kv_heads, self.head_size)

        n_rep = self.num_heads // self.num_kv_heads
        if n_rep > 1:
            key = key.repeat_interleave(n_rep, dim=1)
            value = value.repeat_interleave(n_rep, dim=1)

        query_start_loc = attn_metadata.query_start_loc
        num_seqs = attn_metadata.seq_lens.shape[0]

        results = []
        for i in range(num_seqs):
            start = query_start_loc[i].item()
            end = query_start_loc[i + 1].item()
            q_i = query[start:end].transpose(0, 1).unsqueeze(0)
            k_i = key[start:end].transpose(0, 1).unsqueeze(0)
            v_i = value[start:end].transpose(0, 1).unsqueeze(0)
            out_i = torch.nn.functional.scaled_dot_product_attention(
                q_i, k_i, v_i, is_causal=True,
            )
            results.append(out_i.squeeze(0).transpose(0, 1))

        combined = torch.cat(results, dim=0)
        flat = combined.reshape(combined.shape[0], -1)
        if output is not None:
            output[:flat.shape[0]].copy_(
                flat.view(output[:flat.shape[0]].shape)
            )
            return output
        return flat
