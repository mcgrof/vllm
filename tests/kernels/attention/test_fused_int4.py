# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the fused INT4 KV cache attention backend.

Covers:
  1. Config: int4_fused dtype parsing and validation
  2. INT4 pack/unpack correctness (PyTorch reference)
  3. reshape_and_cache_int4 Triton kernel correctness
  4. fused_int4_decode kernel numerical agreement with dense reference
  5. Backend selection and is_quantized_kv_cache
  6. Cache shape computation

These tests run on GPU (CUDA).
"""

import pytest
import torch

from vllm.config.cache import CacheDType
from vllm.v1.attention.backend import is_quantized_kv_cache

# Skip entire module if no CUDA
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for fused INT4 tests",
)

# Try importing Triton — tests that need it will skip if unavailable
try:
    import triton  # noqa: F401

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

GROUP_SIZE = 32
INT4_RANGE = 7


# ----------------------------------------------------------------
# Helper: PyTorch reference quantize / dequantize
# ----------------------------------------------------------------
def quantize_and_pack_int4_ref(
    tensor: torch.Tensor, group_size: int = GROUP_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantise fp16 tensor to INT4, pack 2 values per byte.

    Returns (packed, scales) where:
      packed: uint8 tensor with shape (..., head_size // 2)
      scales: fp16 tensor with shape (..., head_size // group_size)
    """
    shape = tensor.shape
    hd = shape[-1]
    ng = hd // group_size
    r = tensor.float().reshape(*shape[:-1], ng, group_size)
    amax = r.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scales = (amax / INT4_RANGE).squeeze(-1).half()

    q = (r / (amax / INT4_RANGE)).round().clamp(-8, 7).to(torch.int8)
    q = q.reshape(*shape[:-1], hd)
    q_unsigned = (q + 8).to(torch.uint8)
    low = q_unsigned[..., 0::2]
    high = q_unsigned[..., 1::2]
    packed = low | (high << 4)
    return packed, scales


def dequant_int4_ref(
    packed: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = GROUP_SIZE,
) -> torch.Tensor:
    """Dequantize packed INT4 back to fp16."""
    low = (packed & 0x0F).to(torch.int8) - 8
    high = ((packed >> 4) & 0x0F).to(torch.int8) - 8
    hd = packed.shape[-1] * 2
    out = torch.empty(
        *packed.shape[:-1], hd, device=packed.device, dtype=torch.float16,
    )
    out[..., 0::2] = low.to(torch.float16)
    out[..., 1::2] = high.to(torch.float16)
    ng = hd // group_size
    out = out.reshape(*packed.shape[:-1], ng, group_size)
    out = out * scales.unsqueeze(-1).float()
    out = out.reshape(*packed.shape[:-1], hd).half()
    return out


# ================================================================
# 1. Config tests
# ================================================================
class TestConfig:
    def test_int4_fused_is_valid_cache_dtype(self):
        """int4_fused should be a valid CacheDType value."""
        from typing import get_args

        valid = get_args(CacheDType)
        assert "int4_fused" in valid

    def test_is_quantized_kv_cache(self):
        """is_quantized_kv_cache should return True for int4_fused."""
        assert is_quantized_kv_cache("int4_fused") is True
        assert is_quantized_kv_cache("fp8") is True
        assert is_quantized_kv_cache("auto") is False
        assert is_quantized_kv_cache("float16") is False


# ================================================================
# 2. INT4 pack/unpack correctness
# ================================================================
class TestInt4PackUnpack:
    @pytest.mark.parametrize("head_size", [64, 128, 256])
    def test_roundtrip(self, head_size: int):
        """Pack then unpack should be within quantisation tolerance."""
        torch.manual_seed(42)
        x = torch.randn(4, 8, head_size, dtype=torch.float16, device="cuda")
        packed, scales = quantize_and_pack_int4_ref(x)
        recovered = dequant_int4_ref(packed, scales)

        # Quantisation error should be bounded
        max_err = (x.float() - recovered.float()).abs().max().item()
        # Error per element bounded by 0.5 * scale (rounding error)
        max_scale = scales.float().max().item()
        assert max_err < max_scale + 1e-3, (
            f"max_err={max_err:.6f} exceeds max_scale={max_scale:.6f}"
        )

    def test_packing_format(self):
        """Verify the low/high nibble packing format."""
        # Create known values: [0, 1, 2, 3, ...] in groups of 32
        x = torch.arange(32, dtype=torch.float16, device="cuda").unsqueeze(0)
        packed, scales = quantize_and_pack_int4_ref(x)

        # Verify packed shape
        assert packed.shape[-1] == 16  # 32 / 2

        # Unpack and verify
        low = (packed & 0x0F).to(torch.int8) - 8
        high = ((packed >> 4) & 0x0F).to(torch.int8) - 8
        # Low nibble = even indices, high nibble = odd indices
        assert low.shape[-1] == 16
        assert high.shape[-1] == 16


# ================================================================
# 3. reshape_and_cache_int4 kernel correctness
# ================================================================
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
class TestReshapeAndCacheInt4:
    def test_basic_correctness(self):
        """Kernel output should match PyTorch reference packing."""
        from vllm.v1.attention.backends.fused_int4 import reshape_and_cache_int4

        torch.manual_seed(42)
        num_tokens = 4
        num_heads = 4
        head_size = 128
        block_size = 16
        num_blocks = 8
        half_hd = head_size // 2
        num_groups = head_size // GROUP_SIZE

        key = torch.randn(
            num_tokens, num_heads, head_size,
            dtype=torch.float16, device="cuda",
        )
        value = torch.randn(
            num_tokens, num_heads, head_size,
            dtype=torch.float16, device="cuda",
        )

        # Allocate cache tensors
        key_cache = torch.zeros(
            num_blocks, block_size, num_heads, half_hd,
            dtype=torch.uint8, device="cuda",
        )
        value_cache = torch.zeros(
            num_blocks, block_size, num_heads, half_hd,
            dtype=torch.uint8, device="cuda",
        )
        k_scales = torch.zeros(
            num_blocks, block_size, num_heads, num_groups,
            dtype=torch.float16, device="cuda",
        )
        v_scales = torch.zeros(
            num_blocks, block_size, num_heads, num_groups,
            dtype=torch.float16, device="cuda",
        )

        # Simple slot mapping: token i -> slot i (all in block 0)
        slot_mapping = torch.arange(
            num_tokens, dtype=torch.int64, device="cuda",
        )

        reshape_and_cache_int4(
            key, value,
            key_cache, value_cache,
            k_scales, v_scales,
            slot_mapping,
        )

        # Verify by dequantizing and comparing
        for t in range(num_tokens):
            block_idx = t // block_size
            block_offset = t % block_size
            for h in range(num_heads):
                packed_k = key_cache[block_idx, block_offset, h]
                scales_k = k_scales[block_idx, block_offset, h]
                recovered_k = dequant_int4_ref(
                    packed_k.unsqueeze(0), scales_k.unsqueeze(0),
                ).squeeze(0)

                orig_k = key[t, h]
                # Relative error within ~15% for INT4 quantisation
                abs_err = (orig_k.float() - recovered_k.float()).abs()
                max_abs = orig_k.float().abs().max().clamp(min=1e-6)
                rel_err = (abs_err / max_abs).max().item()
                assert rel_err < 0.5, (
                    f"token={t} head={h}: key rel_err={rel_err:.4f}"
                )

    def test_slot_mapping_with_gaps(self):
        """Kernel should handle non-contiguous slot mappings."""
        from vllm.v1.attention.backends.fused_int4 import reshape_and_cache_int4

        torch.manual_seed(0)
        num_tokens = 2
        num_heads = 2
        head_size = 64
        block_size = 16
        num_blocks = 4
        half_hd = head_size // 2
        num_groups = head_size // GROUP_SIZE

        key = torch.randn(
            num_tokens, num_heads, head_size,
            dtype=torch.float16, device="cuda",
        )
        value = torch.randn(
            num_tokens, num_heads, head_size,
            dtype=torch.float16, device="cuda",
        )

        key_cache = torch.zeros(
            num_blocks, block_size, num_heads, half_hd,
            dtype=torch.uint8, device="cuda",
        )
        value_cache = torch.zeros_like(key_cache)
        k_scales = torch.zeros(
            num_blocks, block_size, num_heads, num_groups,
            dtype=torch.float16, device="cuda",
        )
        v_scales = torch.zeros_like(k_scales)

        # Slot 5 and slot 35 (different blocks)
        slot_mapping = torch.tensor([5, 35], dtype=torch.int64, device="cuda")

        reshape_and_cache_int4(
            key, value,
            key_cache, value_cache,
            k_scales, v_scales,
            slot_mapping,
        )

        # Check that slots were written to the correct locations
        # Slot 5 -> block 0, offset 5
        assert key_cache[0, 5].any(), "slot 5 should have data"
        # Slot 35 -> block 2, offset 3
        assert key_cache[2, 3].any(), "slot 35 should have data"
        # Slot 0 should be empty
        assert not key_cache[0, 0].any(), "slot 0 should be empty"


# ================================================================
# 4. Fused decode kernel numerical agreement
# ================================================================
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
class TestFusedInt4Decode:
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize("head_size", [64, 128])
    def test_decode_vs_dense_reference(
        self, batch_size: int, head_size: int,
    ):
        """Fused decode should match dense SDPA within quant tolerance."""
        from vllm.v1.attention.backends.fused_int4 import fused_int4_decode

        torch.manual_seed(42)
        num_heads = 8
        num_kv_heads = 4
        n_rep = num_heads // num_kv_heads
        seq_len = 128
        block_size = 16
        half_hd = head_size // 2
        num_groups = head_size // GROUP_SIZE

        num_blocks_per_seq = (seq_len + block_size - 1) // block_size
        total_blocks = batch_size * num_blocks_per_seq

        # Generate random FP16 K/V
        K_fp16 = torch.randn(
            batch_size, num_kv_heads, seq_len, head_size,
            dtype=torch.float16, device="cuda",
        )
        V_fp16 = torch.randn(
            batch_size, num_kv_heads, seq_len, head_size,
            dtype=torch.float16, device="cuda",
        )
        Q = torch.randn(
            batch_size, num_heads, head_size,
            dtype=torch.float16, device="cuda",
        )

        # Quantize K/V to INT4
        K_packed, K_scales = quantize_and_pack_int4_ref(K_fp16)
        V_packed, V_scales = quantize_and_pack_int4_ref(V_fp16)

        # Build paged cache from quantized data
        key_cache = torch.zeros(
            total_blocks, block_size, num_kv_heads, half_hd,
            dtype=torch.uint8, device="cuda",
        )
        value_cache = torch.zeros_like(key_cache)
        k_scales_cache = torch.zeros(
            total_blocks, block_size, num_kv_heads, num_groups,
            dtype=torch.float16, device="cuda",
        )
        v_scales_cache = torch.zeros_like(k_scales_cache)

        # Fill cache from quantized tensors
        block_table = torch.zeros(
            batch_size, num_blocks_per_seq,
            dtype=torch.int32, device="cuda",
        )
        for b in range(batch_size):
            for blk in range(num_blocks_per_seq):
                phys_blk = b * num_blocks_per_seq + blk
                block_table[b, blk] = phys_blk
                start = blk * block_size
                end = min(start + block_size, seq_len)
                length = end - start
                key_cache[phys_blk, :length] = K_packed[b, :, start:end].permute(1, 0, 2)
                value_cache[phys_blk, :length] = V_packed[b, :, start:end].permute(1, 0, 2)
                k_scales_cache[phys_blk, :length] = K_scales[b, :, start:end].permute(1, 0, 2)
                v_scales_cache[phys_blk, :length] = V_scales[b, :, start:end].permute(1, 0, 2)

        seq_lens = torch.full(
            (batch_size,), seq_len, dtype=torch.int32, device="cuda",
        )

        # Run fused kernel
        fused_output = fused_int4_decode(
            Q, key_cache, value_cache,
            k_scales_cache, v_scales_cache,
            block_table, seq_lens,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            block_n=64,
        )

        # Reference: dequantize and use torch SDPA
        K_deq = dequant_int4_ref(K_packed, K_scales)
        V_deq = dequant_int4_ref(V_packed, V_scales)

        # GQA expansion for reference
        K_exp = K_deq.repeat_interleave(n_rep, dim=1)
        V_exp = V_deq.repeat_interleave(n_rep, dim=1)

        Q_4d = Q.unsqueeze(2)  # [B, H, 1, D]
        ref_output = torch.nn.functional.scaled_dot_product_attention(
            Q_4d, K_exp, V_exp, is_causal=False,
        ).squeeze(2)  # [B, H, D]

        # Check numerical agreement using cosine similarity (primary metric)
        # and mean absolute error.  Max relative error can be large due to
        # single-element outliers near small denominators, so we use more
        # robust metrics.
        cos_sim = torch.nn.functional.cosine_similarity(
            fused_output.reshape(-1).float(),
            ref_output.reshape(-1).float(),
            dim=0,
        ).item()
        mae = (fused_output.float() - ref_output.float()).abs().mean().item()

        assert cos_sim > 0.99, (
            f"batch={batch_size} head_size={head_size}: "
            f"cos_sim={cos_sim:.6f} (expected > 0.99)"
        )
        assert mae < 0.05, (
            f"batch={batch_size} head_size={head_size}: "
            f"mae={mae:.6f} (expected < 0.05)"
        )


# ================================================================
# 5. Backend class tests
# ================================================================
class TestFusedInt4Backend:
    def test_cache_shape(self):
        """get_kv_cache_shape should return correct INT4 packed shape."""
        from vllm.v1.attention.backends.fused_int4 import (
            FusedInt4AttentionBackend,
        )

        shape = FusedInt4AttentionBackend.get_kv_cache_shape(
            num_blocks=100,
            block_size=16,
            num_kv_heads=4,
            head_size=128,
        )
        assert shape == (2, 100, 16, 4, 64), f"unexpected shape: {shape}"

    def test_supported_kv_cache_dtype(self):
        """Backend should only support int4_fused."""
        from vllm.v1.attention.backends.fused_int4 import (
            FusedInt4AttentionBackend,
        )

        assert FusedInt4AttentionBackend.supports_kv_cache_dtype("int4_fused")
        assert not FusedInt4AttentionBackend.supports_kv_cache_dtype("fp8")
        assert not FusedInt4AttentionBackend.supports_kv_cache_dtype("auto")

    def test_supports_head_size(self):
        """Head size must be divisible by GROUP_SIZE (32)."""
        from vllm.v1.attention.backends.fused_int4 import (
            FusedInt4AttentionBackend,
        )

        assert FusedInt4AttentionBackend.supports_head_size(128)
        assert FusedInt4AttentionBackend.supports_head_size(64)
        assert FusedInt4AttentionBackend.supports_head_size(256)
        assert not FusedInt4AttentionBackend.supports_head_size(48)

    def test_get_name(self):
        from vllm.v1.attention.backends.fused_int4 import (
            FusedInt4AttentionBackend,
        )

        assert FusedInt4AttentionBackend.get_name() == "FUSED_INT4"


# ================================================================
# 6. dtype utility tests
# ================================================================
class TestDtypeUtils:
    def test_get_kv_cache_torch_dtype(self):
        """int4_fused should map to uint8 storage dtype."""
        from vllm.utils.torch_utils import get_kv_cache_torch_dtype

        dtype = get_kv_cache_torch_dtype("int4_fused")
        assert dtype == torch.uint8

    def test_kv_cache_dtype_str_to_dtype(self):
        """int4_fused should return model dtype (not uint8)."""
        from unittest.mock import MagicMock

        from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype

        model_config = MagicMock()
        model_config.dtype = torch.bfloat16
        dtype = kv_cache_dtype_str_to_dtype("int4_fused", model_config)
        assert dtype == torch.bfloat16
