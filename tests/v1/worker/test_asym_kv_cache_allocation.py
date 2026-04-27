# SPDX-License-Identifier: Apache-2.0
"""Milestone 2 verification: vLLM asymmetric KV cache allocation.

Exercises only the allocator/materialization boundary (no model, no
forward, no FlashInfer kernel).  Proves the contract:

- when AttentionSpec.v_dtype differs from .dtype, the per-layer
  kv_cache is returned as a tuple (k_cache, v_cache).
- k_cache.dtype is the model dtype (BF16) and v_cache.dtype is FP8.
- shapes are 4-D NHD (no leading "2" dim).
- raw byte accounting is exactly K_bytes + V_bytes — i.e. the cache is
  0.75x of symmetric BF16 K+V at the vLLM allocation boundary, not
  just at the LMCache codec boundary.
- K and V tensors do not alias the same bytes.
- legacy symmetric path is unchanged: returns one 5-D tensor with
  K and V on dim[1].

This test runs on CPU.  No CUDA needed.
"""

import pytest
import torch

from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from vllm.v1.worker.gpu.attn_utils import _reshape_kv_cache


NUM_BLOCKS = 4
BLOCK_SIZE = 16
NUM_KV_HEADS = 8
HEAD_SIZE = 128
LAYER = "model.layers.0.attn"


def _build_kv_cache_config(spec: FullAttentionSpec) -> tuple[
    KVCacheConfig, dict[str, torch.Tensor]
]:
    """Build a KVCacheConfig and the matching raw tensor dict for one
    layer with the given spec.  The raw tensor's size in bytes equals
    spec.page_size_bytes * NUM_BLOCKS, matching what the real allocator
    in `_allocate_kv_cache` would produce."""
    raw_size = spec.page_size_bytes * NUM_BLOCKS
    raw = torch.zeros(raw_size, dtype=torch.int8)
    cfg = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[KVCacheTensor(size=raw_size, shared_by=[LAYER])],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=[LAYER], kv_cache_spec=spec),
        ],
    )
    raw_tensors = {LAYER: raw}
    return cfg, raw_tensors


def test_asym_kv_cache_reshape_splits_raw_buffer_into_bf16_k_and_fp8_v():
    """Asym spec produces a tuple of 4-D tensors with separate dtypes.

    Verifies:
      - returned kv_caches[layer] is a tuple of length 2
      - k_cache.dtype = bf16, v_cache.dtype = fp8_e4m3fn
      - shapes are NHD (num_blocks, block_size, num_kv_heads, head_size)
      - exact byte accounting: K_bytes + V_bytes
      - 0.75x of symmetric BF16 K+V at the allocation boundary
      - K and V do not alias each other
    """
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=torch.bfloat16,
        v_dtype=torch.float8_e4m3fn,
        use_mla=False,
    )

    cfg, raw_tensors = _build_kv_cache_config(spec)
    raw = raw_tensors[LAYER]

    # The asym path doesn't read attn_backends or cache_dtype, so we
    # can pass empty/dummy values to exercise the asym branch in
    # isolation.
    kv_caches = _reshape_kv_cache(
        kv_cache_config=cfg,
        kv_cache_raw_tensors=raw_tensors,
        attn_backends={LAYER: None},  # unused on asym path
        cache_dtype="auto",  # unused on asym path
    )

    assert LAYER in kv_caches
    out = kv_caches[LAYER]
    assert isinstance(out, tuple), f"asym should return tuple, got {type(out)}"
    assert len(out) == 2

    k_cache, v_cache = out

    # Dtypes
    assert k_cache.dtype == torch.bfloat16
    assert v_cache.dtype == torch.float8_e4m3fn

    # Shapes (NHD, no leading "2" dim)
    expected_shape = (NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
    assert tuple(k_cache.shape) == expected_shape
    assert tuple(v_cache.shape) == expected_shape

    # Byte accounting
    elements = NUM_BLOCKS * BLOCK_SIZE * NUM_KV_HEADS * HEAD_SIZE
    expected_k_bytes = elements * 2  # bf16
    expected_v_bytes = elements * 1  # fp8_e4m3fn
    expected_total = expected_k_bytes + expected_v_bytes

    assert k_cache.numel() * k_cache.element_size() == expected_k_bytes
    assert v_cache.numel() * v_cache.element_size() == expected_v_bytes
    assert raw.numel() == expected_total

    # 0.75x of symmetric BF16 K+V — the storage ratio that motivates
    # the whole design, asserted at the allocator boundary, not just
    # in the LMCache codec.
    sym_bf16_kv_bytes = elements * 2 * 2  # 2 planes × bf16
    assert expected_total * 4 == sym_bf16_kv_bytes * 3

    # No aliasing: K and V must occupy distinct byte regions.
    assert k_cache.data_ptr() != v_cache.data_ptr()


def test_asym_kv_cache_reshape_preserves_block_manager_ownership():
    """K and V returned by the asym path must be views into the
    block-manager-owned raw allocation, not freshly copied storage.

    Background: the block manager hands `_reshape_kv_cache` a single
    `torch.zeros(size, dtype=torch.int8, device=...)` tensor sized
    to `page_size_bytes * num_blocks`.  The block manager tracks
    occupancy and eviction against THAT raw tensor's storage.  If
    `_reshape_kv_cache` returns K/V tensors whose `data_ptr()` falls
    outside the raw tensor's byte range, the bookkeeping silently
    drifts away from the GPU memory the kernel reads and writes.

    PyTorch's `.contiguous()` returns a copy when the source is
    non-contiguous.  A slice of a 2-D `view(num_blocks, page_bytes)`
    IS non-contiguous along `dim=1`, so `.contiguous()` copies.

    This test is expected to FAIL on the current implementation —
    that's the gate that motivates the typed strided-view fix.
    """
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=torch.bfloat16,
        v_dtype=torch.float8_e4m3fn,
        use_mla=False,
    )
    cfg, raw_tensors = _build_kv_cache_config(spec)
    raw = raw_tensors[LAYER]

    raw_start = raw.data_ptr()
    raw_end = raw_start + raw.untyped_storage().nbytes()

    kv_caches = _reshape_kv_cache(
        kv_cache_config=cfg,
        kv_cache_raw_tensors=raw_tensors,
        attn_backends={LAYER: None},
        cache_dtype="auto",
    )
    k_cache, v_cache = kv_caches[LAYER]

    for side, t in [("K", k_cache), ("V", v_cache)]:
        ptr = t.data_ptr()
        assert raw_start <= ptr < raw_end, (
            f"asym {side} cache at 0x{ptr:x} is NOT a view into the "
            f"block-manager-owned raw tensor "
            f"[0x{raw_start:x}, 0x{raw_end:x}). "
            f"`.contiguous()` allocated fresh storage; the block "
            f"manager's occupancy bookkeeping is now lying about "
            f"what the kernel reads."
        )


def test_asym_kv_cache_reshape_byte_split_is_k_first_then_v():
    """Within the raw buffer's per-page layout, K bytes come first then
    V bytes.  This contract matters because the asym writer (M3) and
    the FlashInfer backend's cache-write op rely on it."""
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=torch.bfloat16,
        v_dtype=torch.float8_e4m3fn,
        use_mla=False,
    )
    cfg, raw_tensors = _build_kv_cache_config(spec)
    raw = raw_tensors[LAYER]

    # Stamp the K region with one byte pattern and the V region with
    # another, then verify K and V views see the right halves.
    elements = NUM_BLOCKS * BLOCK_SIZE * NUM_KV_HEADS * HEAD_SIZE
    k_bytes = elements * 2
    v_bytes = elements * 1
    page_bytes = (k_bytes + v_bytes) // NUM_BLOCKS

    raw_pages = raw.view(NUM_BLOCKS, page_bytes)
    raw_pages[:, : page_bytes - (v_bytes // NUM_BLOCKS)] = 0x11
    raw_pages[:, page_bytes - (v_bytes // NUM_BLOCKS) :] = 0x22

    kv_caches = _reshape_kv_cache(
        kv_cache_config=cfg,
        kv_cache_raw_tensors=raw_tensors,
        attn_backends={LAYER: None},
        cache_dtype="auto",
    )
    k_cache, v_cache = kv_caches[LAYER]

    # K-side bytes should all be 0x11 (when reinterpreted as int8)
    k_bytes_view = k_cache.contiguous().view(torch.int8)
    assert (k_bytes_view == 0x11).all()
    # V-side bytes should all be 0x22
    v_bytes_view = v_cache.contiguous().view(torch.int8)
    assert (v_bytes_view == 0x22).all()


def test_symmetric_kv_cache_reshape_preserves_legacy_contract():
    """When v_dtype is unset (or equals dtype), the legacy single
    5-D tensor with K/V on dim[1] is returned.  Exercises a path
    through the symmetric branch — needs an attn_backend to provide
    the shape, so we use a tiny stub."""

    class _StubBackend:
        @staticmethod
        def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size,
                                cache_dtype):
            return (num_blocks, 2, block_size, num_kv_heads, head_size)

        @staticmethod
        def get_kv_cache_stride_order():
            return (0, 1, 2, 3, 4)

    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=torch.bfloat16,
        v_dtype=None,  # symmetric
        use_mla=False,
    )

    cfg, raw_tensors = _build_kv_cache_config(spec)

    kv_caches = _reshape_kv_cache(
        kv_cache_config=cfg,
        kv_cache_raw_tensors=raw_tensors,
        attn_backends={LAYER: _StubBackend()},
        cache_dtype="auto",
    )

    out = kv_caches[LAYER]
    assert isinstance(out, torch.Tensor), \
        f"symmetric should return Tensor, got {type(out)}"
    assert out.dtype == torch.bfloat16
    # K/V on dim[1]
    assert out.shape[1] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
