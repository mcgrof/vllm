# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-launch NHD writer for native FP16 K and E4M3FN V."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _mixed_writer(
    K,
    V,
    KC,
    VC,
    SLOTS,
    SCALE,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    KS2: tl.constexpr,
    VS0: tl.constexpr,
    VS1: tl.constexpr,
    VS2: tl.constexpr,
    KB: tl.constexpr,
    KP: tl.constexpr,
    KH: tl.constexpr,
    VB: tl.constexpr,
    VP: tl.constexpr,
    VH: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    slot = tl.load(SLOTS + token).to(tl.int64)
    if slot < 0:
        return
    x = tl.arange(0, BLOCK)
    h, d = x // DIM, x % DIM
    mask = x < HEADS * DIM
    key = tl.load(K + token * KS0 + h * KS1 + d * KS2, mask, 0)
    value = tl.load(V + token * VS0 + h * VS1 + d * VS2, mask, 0)
    scale = tl.load(SCALE)
    encoded = tl.div_rn(value.to(tl.float32), scale)
    # Saturate infinities but retain the native writer's canonical FP8 NaN.
    encoded = tl.where(
        encoded != encoded, encoded, tl.minimum(tl.maximum(encoded, -448.0), 448.0)
    )
    page, offset = slot // PAGE, slot % PAGE
    tl.store(KC + page * KB + offset * KP + h * KH + d, key, mask)
    tl.store(VC + page * VB + offset * VP + h * VH + d, encoded, mask)


def mixed_reshape_and_cache(key, value, key_cache, value_cache, slot_mapping, v_scale):
    """Copy K and quantize V once, with independent element strides.

    Slots must be unique when nonnegative and must lie within both caches.
    This is the same caller-owned slot contract as the existing writer.
    Only FP16 inputs and NHD FP16/E4M3FN destinations are supported.
    """
    if key.dtype != torch.float16 or value.dtype != torch.float16:
        raise ValueError("mixed writer requires FP16 inputs")
    if key_cache.dtype != torch.float16 or value_cache.dtype != torch.float8_e4m3fn:
        raise ValueError("mixed writer requires FP16 K and E4M3FN V storage")
    if key.ndim != 3 or value.shape != key.shape:
        raise ValueError("key and value must have matching [token, head, dim] shapes")
    if (
        key_cache.ndim != 4
        or value_cache.shape != key_cache.shape
        or key_cache.shape[2:] != key.shape[1:]
        or key_cache.stride(-1) != 1
        or value_cache.stride(-1) != 1
        or key_cache.stride(-2) != key.shape[-1]
        or value_cache.stride(-2) != key.shape[-1]
    ):
        raise ValueError("mixed writer requires NHD caches with contiguous heads")
    if (
        slot_mapping.ndim != 1
        or slot_mapping.dtype != torch.int64
        or slot_mapping.numel() > key.shape[0]
        or not slot_mapping.is_contiguous()
        or v_scale.numel() != 1
        or v_scale.dtype != torch.float32
    ):
        raise ValueError(
            "mixed writer requires contiguous int64 slots and one FP32 scale"
        )
    tensors = (key, value, key_cache, value_cache, slot_mapping, v_scale)
    if not key.is_cuda or any(t.device != key.device for t in tensors):
        raise ValueError("all mixed-writer tensors must share one CUDA device")
    if slot_mapping.numel() == 0:
        return
    heads, dim = key.shape[1:]
    _mixed_writer[(slot_mapping.numel(),)](
        *tensors,
        *key.stride(),
        *value.stride(),
        *key_cache.stride()[:3],
        *value_cache.stride()[:3],
        heads,
        dim,
        key_cache.shape[1],
        triton.next_power_of_2(heads * dim),
        num_warps=4,
        enable_fp_fusion=False,
    )
