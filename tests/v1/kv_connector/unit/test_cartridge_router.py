# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for CartridgeRouter — request → cartridge_id dispatch.

Covers:
  - ExplicitCartridgeRouter: reads cartridge_id from extras (top-level
    and nested kv_transfer_params), returns None when absent.
  - StaticCartridgeRouter: always returns the bound id.
  - LabelCartridgeRouter: looks up via CartridgeRegistry, returns None
    on miss.
  - CompositeRouter: first non-None wins, fall-through semantics.
  - build_router_from_config: each router type builds correctly.
"""

from types import SimpleNamespace

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_router import (
    CompositeRouter,
    ExplicitCartridgeRouter,
    LabelCartridgeRouter,
    StaticCartridgeRouter,
    build_router_from_config,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


# ---------------------------------------------------------------------------
# Fake Request / Registry helpers
# ---------------------------------------------------------------------------


def _req(extras: dict | None = None, request_id: str = "r0"):
    """Build a fake Request with sampling_params.extra_args = extras."""
    sp = SimpleNamespace(extra_args=extras)
    return SimpleNamespace(request_id=request_id, sampling_params=sp)


def _req_no_sp(request_id: str = "r0"):
    """Request whose sampling_params is None (e.g. pooling models)."""
    return SimpleNamespace(request_id=request_id, sampling_params=None)


class _FakeManifest:
    def __init__(self, cartridge_id: str):
        self.cartridge_id = cartridge_id


class _FakeRegistry:
    def __init__(self, table: dict[tuple[str, str], list[str]]):
        self._table = table

    def lookup_by_label(self, key: str, value: str):
        ids = self._table.get((key, value), [])
        return [_FakeManifest(i) for i in ids]


# ---------------------------------------------------------------------------
# ExplicitCartridgeRouter
# ---------------------------------------------------------------------------


class TestExplicitCartridgeRouter:
    def test_reads_top_level_key(self):
        r = ExplicitCartridgeRouter()
        assert r.resolve(_req({"cartridge_id": "patient_00"})) == "patient_00"

    def test_reads_nested_in_kv_transfer_params(self):
        r = ExplicitCartridgeRouter()
        extras = {"kv_transfer_params": {"cartridge_id": "patient_01"}}
        assert r.resolve(_req(extras)) == "patient_01"

    def test_top_level_takes_precedence(self):
        r = ExplicitCartridgeRouter()
        extras = {
            "cartridge_id": "top",
            "kv_transfer_params": {"cartridge_id": "nested"},
        }
        assert r.resolve(_req(extras)) == "top"

    def test_returns_none_when_missing(self):
        r = ExplicitCartridgeRouter()
        assert r.resolve(_req({})) is None
        assert r.resolve(_req(None)) is None

    def test_returns_none_when_no_sampling_params(self):
        r = ExplicitCartridgeRouter()
        assert r.resolve(_req_no_sp()) is None

    def test_custom_key_name(self):
        r = ExplicitCartridgeRouter(key="cart")
        assert r.resolve(_req({"cart": "a"})) == "a"
        assert r.resolve(_req({"cartridge_id": "b"})) is None

    def test_coerces_non_string_to_string(self):
        r = ExplicitCartridgeRouter()
        assert r.resolve(_req({"cartridge_id": 42})) == "42"


# ---------------------------------------------------------------------------
# StaticCartridgeRouter
# ---------------------------------------------------------------------------


class TestStaticCartridgeRouter:
    def test_always_returns_same(self):
        r = StaticCartridgeRouter("default")
        assert r.resolve(_req({})) == "default"
        assert r.resolve(_req({"cartridge_id": "ignored"})) == "default"
        assert r.resolve(_req_no_sp()) == "default"


# ---------------------------------------------------------------------------
# LabelCartridgeRouter
# ---------------------------------------------------------------------------


class TestLabelCartridgeRouter:
    def test_resolves_via_registry(self):
        reg = _FakeRegistry(
            {
                ("patient_id", "P001"): ["cart_001", "cart_002"],
            }
        )
        r = LabelCartridgeRouter(reg, label_key="patient_id", extras_key="patient_id")
        extras = {"patient_id": "P001"}
        # Returns the first match
        assert r.resolve(_req(extras)) == "cart_001"

    def test_registry_miss_returns_none(self):
        reg = _FakeRegistry({})
        r = LabelCartridgeRouter(reg, label_key="patient_id", extras_key="patient_id")
        assert r.resolve(_req({"patient_id": "nobody"})) is None

    def test_missing_extras_key_returns_none(self):
        reg = _FakeRegistry({("patient_id", "P001"): ["c"]})
        r = LabelCartridgeRouter(reg, label_key="patient_id", extras_key="patient_id")
        assert r.resolve(_req({})) is None

    def test_reads_nested_in_kv_transfer_params(self):
        reg = _FakeRegistry({("doc_type", "medical"): ["cart_m"]})
        r = LabelCartridgeRouter(reg, label_key="doc_type", extras_key="doc_type")
        extras = {"kv_transfer_params": {"doc_type": "medical"}}
        assert r.resolve(_req(extras)) == "cart_m"


# ---------------------------------------------------------------------------
# CompositeRouter
# ---------------------------------------------------------------------------


class TestCompositeRouter:
    def test_first_non_none_wins(self):
        r = CompositeRouter(
            [
                ExplicitCartridgeRouter(),
                StaticCartridgeRouter("fallback"),
            ]
        )
        # Explicit wins when set
        assert r.resolve(_req({"cartridge_id": "picked"})) == "picked"
        # Falls through to static when explicit returns None
        assert r.resolve(_req({})) == "fallback"

    def test_all_none_returns_none(self):
        r = CompositeRouter(
            [
                ExplicitCartridgeRouter(),
                ExplicitCartridgeRouter(key="other"),
            ]
        )
        assert r.resolve(_req({})) is None

    def test_order_matters(self):
        r_a = CompositeRouter(
            [
                StaticCartridgeRouter("first"),
                StaticCartridgeRouter("second"),
            ]
        )
        r_b = CompositeRouter(
            [
                StaticCartridgeRouter("second"),
                StaticCartridgeRouter("first"),
            ]
        )
        assert r_a.resolve(_req({})) == "first"
        assert r_b.resolve(_req({})) == "second"


# ---------------------------------------------------------------------------
# build_router_from_config
# ---------------------------------------------------------------------------


class TestBuildRouterFromConfig:
    def test_empty_config_uses_default(self):
        r = build_router_from_config({}, default_cartridge_id="d")
        assert isinstance(r, StaticCartridgeRouter)
        assert r.resolve(_req({})) == "d"

    def test_empty_config_no_default_raises(self):
        with pytest.raises(ValueError, match="router config is empty"):
            build_router_from_config({}, default_cartridge_id=None)

    def test_explicit_type(self):
        r = build_router_from_config({"type": "explicit"})
        assert isinstance(r, ExplicitCartridgeRouter)
        assert r.resolve(_req({"cartridge_id": "x"})) == "x"

    def test_explicit_custom_key(self):
        r = build_router_from_config({"type": "explicit", "key": "cid"})
        assert r.resolve(_req({"cid": "y"})) == "y"
        assert r.resolve(_req({"cartridge_id": "z"})) is None

    def test_static_type(self):
        r = build_router_from_config({"type": "static", "cartridge_id": "s"})
        assert isinstance(r, StaticCartridgeRouter)
        assert r.resolve(_req({})) == "s"

    def test_static_without_id_falls_back_to_default(self):
        r = build_router_from_config(
            {"type": "static"}, default_cartridge_id="fallback"
        )
        assert r.resolve(_req({})) == "fallback"

    def test_static_without_anything_raises(self):
        with pytest.raises(ValueError, match="static router"):
            build_router_from_config({"type": "static"})

    def test_label_requires_registry(self):
        with pytest.raises(ValueError, match="requires a Cartridge"):
            build_router_from_config(
                {"type": "label", "label_key": "patient_id"},
                registry=None,
            )

    def test_label_requires_label_key(self):
        reg = _FakeRegistry({})
        with pytest.raises(ValueError, match="label_key"):
            build_router_from_config({"type": "label"}, registry=reg)

    def test_label_builds(self):
        reg = _FakeRegistry({("patient_id", "X"): ["c_x"]})
        r = build_router_from_config(
            {"type": "label", "label_key": "patient_id", "extras_key": "patient_id"},
            registry=reg,
        )
        assert r.resolve(_req({"patient_id": "X"})) == "c_x"

    def test_composite(self):
        r = build_router_from_config(
            {
                "type": "composite",
                "routers": [
                    {"type": "explicit"},
                    {"type": "static", "cartridge_id": "fb"},
                ],
            }
        )
        assert isinstance(r, CompositeRouter)
        assert r.resolve(_req({"cartridge_id": "exp"})) == "exp"
        assert r.resolve(_req({})) == "fb"

    def test_composite_requires_routers_list(self):
        with pytest.raises(ValueError, match="composite router"):
            build_router_from_config({"type": "composite"})

    def test_unknown_type(self):
        with pytest.raises(ValueError, match="unknown router type"):
            build_router_from_config({"type": "nonsense"})
