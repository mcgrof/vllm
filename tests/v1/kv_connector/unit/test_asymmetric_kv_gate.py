# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for the asymmetric-K/V KV-connector admission contract.

Asymmetric K/V (bf16 K + fp8 V) hands each layer a ``(k_cache, v_cache)`` tuple
as its ``register_kv_caches`` value. Connectors assume one tensor per layer, so
the combination is rejected unless a connector opts in via the class-level
``supports_asymmetric_kv`` flag. These tests exercise the flag resolution, the
config-verify gate, and the register-time boundary guard without a GPU, a
model, or LMCache.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.config.vllm import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.utils import (
    kv_caches_contain_asymmetric_kv,
    verify_connector_supports_kv_caches,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1

# --- class-level admission flag (config path) --------------------------------


def test_base_connector_inherits_false():
    assert KVConnectorBase_V1.supports_asymmetric_kv is False


def test_class_check_true_for_plain_flag():
    class _AsymOK(KVConnectorBase_V1):
        supports_asymmetric_kv = True

    assert KVConnectorFactory._class_supports_asymmetric_kv(_AsymOK) is True


def test_class_check_default_false():
    class _Default(KVConnectorBase_V1):
        pass

    assert KVConnectorFactory._class_supports_asymmetric_kv(_Default) is False


def test_class_check_property_override_fails_closed():
    # A @property override must NOT read as supported: a class-level getattr
    # returns the (truthy) property object, and bool(...) would fail *open*.
    class _AsymProp(KVConnectorBase_V1):
        @property
        def supports_asymmetric_kv(self):  # type: ignore[override]
            return True

    assert KVConnectorFactory._class_supports_asymmetric_kv(_AsymProp) is False


def test_supports_config_reads_class_flag_for_non_multi():
    class _AsymOK(KVConnectorBase_V1):
        supports_asymmetric_kv = True

    class _Default(KVConnectorBase_V1):
        pass

    ktc = SimpleNamespace(kv_connector="Whatever")
    with patch.object(KVConnectorFactory, "get_connector_class", return_value=_AsymOK):
        assert KVConnectorFactory.supports_asymmetric_kv_config(ktc) is True
    with patch.object(KVConnectorFactory, "get_connector_class", return_value=_Default):
        assert KVConnectorFactory.supports_asymmetric_kv_config(ktc) is False


def test_multiconnector_empty_children_is_false():
    # all([]) is True -> would fail open; guard must return False.
    from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
        MultiConnector,
    )

    ktc = SimpleNamespace(
        kv_connector="MultiConnector",
        engine_id="e0",
        kv_connector_extra_config={"connectors": []},
    )
    assert MultiConnector.all_children_support_asymmetric_kv(ktc) is False


# --- register-time boundary guard (both register paths) ----------------------


def test_kv_caches_contain_asymmetric_kv():
    tuple_val = {"l0": (torch.empty(1), torch.empty(1))}
    tensor_val = {"l0": torch.empty(1)}
    list_val = {"l0": [torch.empty(1), torch.empty(1)]}  # Mamba/hybrid, not asym
    mixed = {"attn": torch.empty(1), "asym": (torch.empty(1), torch.empty(1))}
    assert kv_caches_contain_asymmetric_kv(tuple_val) is True
    assert kv_caches_contain_asymmetric_kv(tensor_val) is False
    assert kv_caches_contain_asymmetric_kv(list_val) is False
    assert kv_caches_contain_asymmetric_kv(mixed) is True


def test_boundary_guard_rejects_tuple_when_unsupported():
    conn = SimpleNamespace(runtime_supports_asymmetric_kv=False)
    kv_caches = {"l0": (torch.empty(1), torch.empty(1))}
    with pytest.raises(RuntimeError, match="asymmetric"):
        verify_connector_supports_kv_caches(conn, kv_caches)


def test_boundary_guard_allows_tuple_when_supported():
    conn = SimpleNamespace(runtime_supports_asymmetric_kv=True)
    kv_caches = {"l0": (torch.empty(1), torch.empty(1))}
    verify_connector_supports_kv_caches(conn, kv_caches)  # no raise


def test_boundary_guard_allows_tensor_and_list():
    conn = SimpleNamespace(runtime_supports_asymmetric_kv=False)
    kv_caches = {"attn": torch.empty(1), "mamba": [torch.empty(1), torch.empty(1)]}
    verify_connector_supports_kv_caches(conn, kv_caches)  # no raise


def test_boundary_guard_missing_attr_fails_closed():
    # A connector with no runtime_supports_asymmetric_kv attribute at all must
    # still reject a tuple (getattr default False).
    conn = SimpleNamespace()
    kv_caches = {"l0": (torch.empty(1), torch.empty(1))}
    with pytest.raises(RuntimeError):
        verify_connector_supports_kv_caches(conn, kv_caches)


# --- config-verify gate ------------------------------------------------------


def _run_gate(cache_dtype, kv_connector, supports, v_cache_dtype=None):
    self = SimpleNamespace(
        kv_transfer_config=(
            None if kv_connector is None else SimpleNamespace(kv_connector=kv_connector)
        ),
        cache_config=SimpleNamespace(
            cache_dtype=cache_dtype, v_cache_dtype=v_cache_dtype
        ),
        model_config=None,
    )
    with patch.object(
        KVConnectorFactory, "supports_asymmetric_kv_config", return_value=supports
    ):
        VllmConfig._verify_kv_transfer_compat(self)


def test_config_gate_raises_on_asym_unsupported():
    with pytest.raises(ValueError, match="asymmetric K/V"):
        _run_gate(("bfloat16", "fp8_e4m3"), "LMCacheConnectorV1", supports=False)


def test_config_gate_raises_on_v_cache_dtype_desync():
    # Real tuple trigger is v_cache_dtype (attn_utils), which can be set even
    # when cache_dtype stays a plain string on hand-constructed configs.
    with pytest.raises(ValueError, match="asymmetric K/V"):
        _run_gate(
            "bfloat16", "LMCacheConnectorV1", supports=False, v_cache_dtype="fp8_e4m3"
        )


def test_config_gate_passes_when_connector_opts_in():
    _run_gate(("bfloat16", "fp8_e4m3"), "SomeAsymConnector", supports=True)  # no raise


def test_config_gate_passes_for_symmetric_dtype():
    # Symmetric spec is not a tuple: is_asymmetric_kv is False, gate is inert.
    _run_gate("auto", "LMCacheConnectorV1", supports=False)


def test_config_gate_passes_for_degenerate_equal_tuple():
    # An equal (k, v) pair is not asymmetric.
    _run_gate(("fp8_e4m3", "fp8_e4m3"), "LMCacheConnectorV1", supports=False)


def test_config_gate_noop_without_connector():
    _run_gate(("bfloat16", "fp8_e4m3"), None, supports=False)  # no raise
