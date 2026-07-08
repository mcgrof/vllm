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
    KVPlane,
    asym_plane_key,
    kv_caches_contain_asymmetric_kv,
    parse_asym_plane_key,
    split_asymmetric_kv_planes,
    verify_asymmetric_kv_unit_scale,
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


# --- EM1: asymmetric-KV plane split (the connector-facing wire contract) ------

_REAL_LAYER = "model.layers.0.self_attn.attn"


def test_plane_key_roundtrip():
    for plane in (KVPlane.K, KVPlane.V):
        key = asym_plane_key(_REAL_LAYER, plane)
        assert parse_asym_plane_key(key) == (_REAL_LAYER, plane)


def test_parse_non_plane_key_returns_none():
    assert parse_asym_plane_key(_REAL_LAYER) is None


def test_asym_plane_key_rejects_delimiter_in_layer_name():
    # A key that already carries the delimiter cannot be re-encoded.
    already = asym_plane_key(_REAL_LAYER, KVPlane.K)
    with pytest.raises(ValueError, match="delimiter"):
        asym_plane_key(already, KVPlane.V)


def test_parse_malformed_plane_suffix_raises():
    # Delimiter present but the suffix is not a valid plane.
    bad = asym_plane_key(_REAL_LAYER, KVPlane.V)[:-1] + "x"
    with pytest.raises(ValueError, match="not a valid plane"):
        parse_asym_plane_key(bad)


def test_split_asymmetric_kv_planes_splits_tuple_preserving_dtype():
    k = torch.zeros(2, 4, dtype=torch.bfloat16)
    v = torch.zeros(2, 4, dtype=torch.float8_e4m3fn)
    src = {_REAL_LAYER: (k, v)}
    out = split_asymmetric_kv_planes(src)
    k_key = asym_plane_key(_REAL_LAYER, KVPlane.K)
    v_key = asym_plane_key(_REAL_LAYER, KVPlane.V)
    assert set(out.keys()) == {k_key, v_key}
    assert out[k_key] is k
    assert out[v_key] is v
    assert out[v_key].dtype is torch.float8_e4m3fn
    assert kv_caches_contain_asymmetric_kv(out) is False


def test_split_passes_through_non_tuples_by_identity():
    tensor = torch.zeros(2, 4)
    mamba = [torch.zeros(2), torch.zeros(2)]  # Mamba/hybrid list: NOT asymmetric
    src = {"attn": tensor, "mamba": mamba}
    out = split_asymmetric_kv_planes(src)
    assert set(out.keys()) == {"attn", "mamba"}
    assert out["attn"] is tensor
    assert out["mamba"] is mamba


def test_split_returns_fresh_dict_and_does_not_mutate_input():
    k = torch.zeros(1, dtype=torch.bfloat16)
    v = torch.zeros(1, dtype=torch.float8_e4m3fn)
    src = {_REAL_LAYER: (k, v)}
    out = split_asymmetric_kv_planes(src)
    assert out is not src
    assert isinstance(src[_REAL_LAYER], tuple)  # source unchanged (attention path)


def test_split_output_has_no_tuples():
    # EM3: nothing downstream can re-detect asymmetry from the connector dict.
    k = torch.zeros(1, dtype=torch.bfloat16)
    v = torch.zeros(1, dtype=torch.float8_e4m3fn)
    out = split_asymmetric_kv_planes({_REAL_LAYER: (k, v)})
    assert kv_caches_contain_asymmetric_kv(out) is False


# --- EM2: the ActiveKVConnector gating contract (emulated, CPU) ----------------
#
# ActiveKVConnector.__init__ cannot be imported without a live KV-transfer group,
# so this reproduces the exact gating from vllm/v1/worker/gpu/kv_connector.py and
# asserts the contract. The real wiring is exercised end-to-end by the GPU gate.


class _RecordingConn:
    def __init__(self, supports: bool):
        self.runtime_supports_asymmetric_kv = supports
        self.registered: dict | None = None

    def register_kv_caches(self, kv_caches: dict) -> None:
        self.registered = kv_caches


def _emulate_active_register(conn: _RecordingConn, kv_caches_dict: dict) -> None:
    if getattr(conn, "runtime_supports_asymmetric_kv", False) is True:
        connector_kv_caches = split_asymmetric_kv_planes(kv_caches_dict)
    else:
        connector_kv_caches = kv_caches_dict
    verify_connector_supports_kv_caches(conn, connector_kv_caches)
    conn.register_kv_caches(connector_kv_caches)


def test_opted_in_connector_receives_split_planes():
    k = torch.zeros(1, dtype=torch.bfloat16)
    v = torch.zeros(1, dtype=torch.float8_e4m3fn)
    conn = _RecordingConn(supports=True)
    _emulate_active_register(conn, {_REAL_LAYER: (k, v)})
    assert set(conn.registered.keys()) == {
        asym_plane_key(_REAL_LAYER, KVPlane.K),
        asym_plane_key(_REAL_LAYER, KVPlane.V),
    }
    assert kv_caches_contain_asymmetric_kv(conn.registered) is False


def test_non_opted_in_connector_trips_verify_before_register():
    k = torch.zeros(1, dtype=torch.bfloat16)
    v = torch.zeros(1, dtype=torch.float8_e4m3fn)
    conn = _RecordingConn(supports=False)
    with pytest.raises(RuntimeError, match="asymmetric"):
        _emulate_active_register(conn, {_REAL_LAYER: (k, v)})
    assert conn.registered is None  # register was never reached


# --- unit-scale guard for the byte-through fp8 V offload path -----------------
#
# verify_asymmetric_kv_unit_scale refuses to register an asymmetric layer whose
# per-layer V scale is not exactly 1.0 (byte-through offload stores raw e4m3
# codes with no scale), and refuses --calculate-kv-scales on the asym path.
# get_layers_from_vllm_config is patched because these CPU tests have no model.

_GET_LAYERS = (
    "vllm.distributed.kv_transfer.kv_connector.utils.get_layers_from_vllm_config"
)


def _asym_caches() -> dict:
    k = torch.zeros(1, dtype=torch.bfloat16)
    v = torch.zeros(1, dtype=torch.float8_e4m3fn)
    return {_REAL_LAYER: (k, v)}


def _fake_config(calculate_kv_scales: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=SimpleNamespace(calculate_kv_scales=calculate_kv_scales)
    )


def test_unit_scale_asym_layer_passes():
    layers = {_REAL_LAYER: SimpleNamespace(_v_scale_float=1.0)}
    with patch(_GET_LAYERS, return_value=layers):
        verify_asymmetric_kv_unit_scale(_fake_config(), _asym_caches())  # no raise


def test_non_unit_scale_asym_layer_rejected():
    layers = {_REAL_LAYER: SimpleNamespace(_v_scale_float=0.5)}
    with (
        patch(_GET_LAYERS, return_value=layers),
        pytest.raises(RuntimeError, match="non-unit V scale"),
    ):
        verify_asymmetric_kv_unit_scale(_fake_config(), _asym_caches())


def test_calculate_kv_scales_rejected_for_asym():
    # Refused before any per-layer inspection, so no layers patch is needed.
    with pytest.raises(RuntimeError, match="calculate-kv-scales"):
        verify_asymmetric_kv_unit_scale(
            _fake_config(calculate_kv_scales=True), _asym_caches()
        )


def test_unit_scale_guard_is_noop_without_asym():
    # A single-tensor (symmetric) cache is not asym, so the guard does nothing
    # even with calculate_kv_scales set (that combination is fine symmetrically).
    sym = {_REAL_LAYER: torch.zeros(1, dtype=torch.float8_e4m3fn)}
    verify_asymmetric_kv_unit_scale(_fake_config(calculate_kv_scales=True), sym)


def test_missing_v_scale_attr_treated_as_unit():
    # A layer that never had a scale set has no _v_scale_float -> unit -> allowed.
    layers = {_REAL_LAYER: SimpleNamespace()}
    with patch(_GET_LAYERS, return_value=layers):
        verify_asymmetric_kv_unit_scale(_fake_config(), _asym_caches())  # no raise
