# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM asymmetric KV cache allocation invariants.

Exercises only the allocator/materialization boundary (no model, no
forward, no FlashInfer kernel).  Proves the contract:

- when AttentionSpec.v_dtype differs from .dtype, the per-layer
  kv_cache is returned as a tuple (k_cache, v_cache).
- k_cache.dtype is the model dtype (BF16) and v_cache.dtype is FP8.
- shapes are 4-D NHD (no leading "2" dim).
- raw byte accounting is exactly K_bytes + V_bytes — i.e. the cache
  is 0.75x of symmetric BF16 K+V at the vLLM allocation boundary.
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

# Pure CPU tensor manipulation; skip the global GPU cleanup fixture.
pytestmark = pytest.mark.skip_global_cleanup

NUM_BLOCKS = 4
BLOCK_SIZE = 16
NUM_KV_HEADS = 8
HEAD_SIZE = 128
LAYER = "model.layers.0.attn"


def _build_kv_cache_config(
    spec: FullAttentionSpec,
) -> tuple[KVCacheConfig, dict[str, torch.Tensor]]:
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


class _StubBackend:
    """Minimal AttentionBackend stub for _reshape_kv_cache.

    The asym path never calls into the backend; the symmetric path only
    needs get_kv_cache_shape / get_kv_cache_stride_order. get_kv_cache_shape
    must accept ``cache_dtype_str`` as a keyword (that is how the production
    caller passes it).
    """

    @staticmethod
    def get_kv_cache_shape(
        num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"
    ):
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order():
        return (0, 1, 2, 3, 4)


def _reshape(cfg, spec, raw_tensors, backend=_StubBackend):
    """Call _reshape_kv_cache with a one-layer AttentionGroup, mirroring the
    production caller (attn_utils.init_kv_cache)."""
    from vllm.v1.worker.utils import AttentionGroup

    group = AttentionGroup(
        backend=backend,
        layer_names=[LAYER],
        kv_cache_spec=spec,
        kv_cache_group_id=0,
    )
    return _reshape_kv_cache(
        attn_groups=[group],
        kv_cache_raw_tensors=raw_tensors,
        cache_dtype="auto",
        kernel_block_sizes=[BLOCK_SIZE],
        shared_kv_cache_layers={},
        kv_cache_config=cfg,
    )


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
    )

    cfg, raw_tensors = _build_kv_cache_config(spec)
    raw = raw_tensors[LAYER]

    # The asym path doesn't read the backend or cache_dtype; the helper
    # exercises the asym branch in isolation via a one-layer group.
    kv_caches = _reshape(cfg, spec, raw_tensors)

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
    # the whole design, asserted at the allocator boundary.
    sym_bf16_kv_bytes = elements * 2 * 2  # 2 planes × bf16
    assert expected_total * 4 == sym_bf16_kv_bytes * 3

    # Region-split zero-copy views: K and V are contiguous slices of the
    # SAME raw buffer (all-K region then all-V region), not fresh
    # .contiguous() copies.  This is what lifts the ~2x init peak that
    # otherwise forced gpu_memory_utilization <= 0.5.  K starts at the
    # buffer base; V starts exactly where the K region ends.
    assert k_cache.data_ptr() == raw.data_ptr()
    assert v_cache.data_ptr() - k_cache.data_ptr() == expected_k_bytes


def test_asym_kv_cache_reshape_byte_split_is_k_first_then_v():
    """The raw buffer is region-split: all K bytes first, then all V
    bytes.  The split point must match the K-bytes-first accounting in
    AttentionSpec.real_page_size_bytes; a direction bug would silently
    hand K bytes to the V view and vice versa."""
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=torch.bfloat16,
        v_dtype=torch.float8_e4m3fn,
    )
    cfg, raw_tensors = _build_kv_cache_config(spec)
    raw = raw_tensors[LAYER]

    # Stamp the all-K region with one byte pattern and the all-V region
    # with another, then verify each view sees the right region.
    elements = NUM_BLOCKS * BLOCK_SIZE * NUM_KV_HEADS * HEAD_SIZE
    k_bytes = elements * 2  # bf16 K
    raw[:k_bytes] = 0x11
    raw[k_bytes:] = 0x22

    kv_caches = _reshape(cfg, spec, raw_tensors)
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
    through the symmetric branch — needs a backend to provide the
    shape, so we use the module-level _StubBackend."""
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=torch.bfloat16,
        v_dtype=None,  # symmetric
    )

    cfg, raw_tensors = _build_kv_cache_config(spec)

    kv_caches = _reshape(cfg, spec, raw_tensors)

    out = kv_caches[LAYER]
    assert isinstance(out, torch.Tensor), (
        f"symmetric should return Tensor, got {type(out)}"
    )
    assert out.dtype == torch.bfloat16
    # K/V on dim[1]
    assert out.shape[1] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
