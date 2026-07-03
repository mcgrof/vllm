# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer forward() tuple-handling helpers for asymmetric K/V.

Tests the in-tree helpers added to
`vllm/v1/attention/backends/flashinfer.py`:
  - `_is_asym_paged_kv_cache(kv_cache)`
  - `_derive_4d_stride_order_from_5d(stride_order_5d)`
  - `_prepare_flashinfer_paged_kv_cache(kv_cache)`

The forward()-level fail-closed guards for asymmetric caches
(cascade, DCP, TRTLLM prefill/decode raise NotImplementedError)
are exercised at runtime, not by this suite.

Pure-helper tests run on CPU.  No model, no FlashInfer kernel, no
GPU required (importing the backend module does require the
flashinfer package; skip otherwise).
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("flashinfer")

import vllm.v1.attention.backends.flashinfer as fi
from vllm.v1.attention.backends.flashinfer import (
    _derive_4d_stride_order_from_5d,
    _is_asym_paged_kv_cache,
    _prepare_flashinfer_paged_kv_cache,
)

# Pure CPU tensor manipulation; skip the global GPU cleanup fixture.
pytestmark = pytest.mark.skip_global_cleanup


def test_is_asym_recognises_tuple_of_two_tensors():
    a = torch.zeros(2, 3, 4, 5, dtype=torch.bfloat16)
    b = torch.zeros(2, 3, 4, 5, dtype=torch.float8_e4m3fn)
    assert _is_asym_paged_kv_cache((a, b)) is True


def test_is_asym_rejects_single_tensor():
    t = torch.zeros(2, 2, 16, 8, 64, dtype=torch.bfloat16)
    assert _is_asym_paged_kv_cache(t) is False


def test_is_asym_rejects_three_tuple():
    a = torch.zeros(2, 3, 4, 5)
    assert _is_asym_paged_kv_cache((a, a, a)) is False


def test_is_asym_rejects_tuple_with_non_tensor():
    a = torch.zeros(2, 3, 4, 5)
    assert _is_asym_paged_kv_cache((a, "not a tensor")) is False
    assert _is_asym_paged_kv_cache((1, 2)) is False
    assert _is_asym_paged_kv_cache(None) is False


def test_derive_4d_stride_NHD_is_identity():
    # NHD canonical 5-D shape (block, kv_side, block_size, kv_head, head_dim);
    # stride order (0,1,2,3,4) is identity.  Drop dim 1, renumber dims>1 down
    # by one -> (0,1,2,3) is also 4-D identity.
    assert _derive_4d_stride_order_from_5d((0, 1, 2, 3, 4)) == (0, 1, 2, 3)


def test_derive_4d_stride_HND_swaps_block_size_and_kv_heads():
    # HND 5-D order (0,1,3,2,4) swaps block_size and kv_heads.
    # Drop dim 1, renumber: (0, 2, 1, 3).
    assert _derive_4d_stride_order_from_5d((0, 1, 3, 2, 4)) == (0, 2, 1, 3)


def test_derive_rejects_wrong_ndim():
    with pytest.raises(AssertionError):
        _derive_4d_stride_order_from_5d((0, 1, 2, 3))


def test_prepare_symmetric_5d_returns_tensor(default_vllm_config):
    t = torch.zeros(4, 2, 16, 8, 128, dtype=torch.bfloat16)
    out = _prepare_flashinfer_paged_kv_cache(t)
    assert isinstance(out, torch.Tensor)
    # Shape preserved across permute (the result depends on layout
    # selected by get_kv_cache_layout — both orders preserve total
    # element count).
    assert out.numel() == t.numel()


def test_prepare_asym_returns_tuple_of_4d_tensors(default_vllm_config):
    k = torch.zeros(4, 16, 8, 128, dtype=torch.bfloat16)
    v = torch.zeros(4, 16, 8, 128, dtype=torch.float8_e4m3fn)
    out = _prepare_flashinfer_paged_kv_cache((k, v))
    assert isinstance(out, tuple)
    assert len(out) == 2
    k_out, v_out = out
    assert k_out.dtype == torch.bfloat16
    assert v_out.dtype == torch.float8_e4m3fn
    assert k_out.numel() == k.numel()
    assert v_out.numel() == v.numel()


def test_prepare_asym_views_not_copies(default_vllm_config):
    """Permute returns a view, so the prepared tuple must still alias
    the same storage as the allocator's tuple — otherwise writes
    through the cache writer wouldn't be visible to the kernel."""
    k = torch.zeros(4, 16, 8, 128, dtype=torch.bfloat16)
    v = torch.zeros(4, 16, 8, 128, dtype=torch.float8_e4m3fn)
    out = _prepare_flashinfer_paged_kv_cache((k, v))
    k_out, v_out = out
    assert k_out.data_ptr() == k.data_ptr()
    assert v_out.data_ptr() == v.data_ptr()


def test_prepare_asym_rejects_mismatched_shapes(default_vllm_config):
    k = torch.zeros(4, 16, 8, 128)
    v = torch.zeros(4, 16, 4, 128)  # different num_kv_heads
    with pytest.raises(AssertionError):
        _prepare_flashinfer_paged_kv_cache((k, v))


def test_prepare_asym_rejects_wrong_ndim(default_vllm_config):
    k = torch.zeros(4, 16, 8, 128, 1)  # 5-D
    v = torch.zeros(4, 16, 8, 128, 1)
    with pytest.raises(AssertionError):
        _prepare_flashinfer_paged_kv_cache((k, v))


def test_prepare_symmetric_rejects_wrong_ndim(default_vllm_config):
    t = torch.zeros(4, 16, 8, 128)  # 4-D not allowed for symmetric
    with pytest.raises(AssertionError):
        _prepare_flashinfer_paged_kv_cache(t)


# ---------------------------------------------------------------------------
# Decode-wrapper backend/tensor-core selection (asym is CUDA-core only).
#
# The asymmetric (DTypeK != DTypeV) decode kernel lives on the CUDA-core
# path.  FlashInfer fail-closes an asym cache + use_tensor_cores with
# NotImplementedError in decode.py plan(), so _get_decode_wrapper must build
# the asym wrapper with use_tensor_cores=False (and backend="fa2").  Under
# FULL-cudagraph capture the decode wrapper is planned on a pure-decode batch
# that bypasses the prefill-population guard, so an unconditional True would
# raise at boot.  Drive the unbound method with a fake self so the check runs
# on CPU without a GPU workspace or a real FlashInfer wrapper.
# ---------------------------------------------------------------------------


def _capture_decode_wrapper(monkeypatch, *, is_asymmetric, is_nvfp4=False):
    captured = {}

    class _FakeDecodeWrapper:
        def __init__(self, *args, use_tensor_cores=None, backend=None, **kwargs):
            captured["use_tensor_cores"] = use_tensor_cores
            captured["backend"] = backend

    monkeypatch.setattr(fi, "BatchDecodeWithPagedKVCacheWrapper", _FakeDecodeWrapper)
    monkeypatch.setattr(fi, "get_kv_cache_layout", lambda: "NHD")

    fake_self = SimpleNamespace(
        _decode_wrapper=None,
        is_kvcache_nvfp4=is_nvfp4,
        _is_asymmetric=is_asymmetric,
        _get_workspace_buffer=lambda: torch.empty(0, dtype=torch.uint8),
    )
    fi.FlashInferMetadataBuilder._get_decode_wrapper(
        fake_self, batch_size=1, use_cudagraph=False
    )
    return captured


def test_decode_wrapper_asym_forces_cuda_cores(monkeypatch):
    captured = _capture_decode_wrapper(monkeypatch, is_asymmetric=True)
    assert captured["backend"] == "fa2"
    # CUDA-core only: True would fail-close in FlashInfer decode.py plan().
    assert captured["use_tensor_cores"] is False


def test_decode_wrapper_symmetric_keeps_tensor_cores(monkeypatch):
    captured = _capture_decode_wrapper(monkeypatch, is_asymmetric=False)
    assert captured["backend"] == "auto"
    assert captured["use_tensor_cores"] is True
