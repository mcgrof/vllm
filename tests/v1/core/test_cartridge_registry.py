# SPDX-License-Identifier: Apache-2.0
"""Tests for CartridgeRegistry.

Covers:
1. Register and lookup by cartridge_id.
2. Lookup by label (patient_id, doc_type, etc.).
3. Lookup by model_id.
4. Unregister.
5. Update (re-register overwrites).
6. Empty registry.
7. Multiple cartridges.
"""
import tempfile
from pathlib import Path

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_manifest import (
    CartridgeManifest,
)
from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_registry import (
    CartridgeRegistry,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_manifest(cartridge_id="cart_01", model_id="meta-llama/Llama-3.2-3B-Instruct",
                   labels=None, **overrides) -> CartridgeManifest:
    defaults = dict(
        cartridge_id=cartridge_id,
        model_id=model_id,
        num_layers=28, num_kv_heads=8, head_dim=128,
        dtype="bfloat16",
        num_tokens_raw=2048, num_tokens_aligned=2048,
        block_size=16, num_blocks=128,
        has_frozen_prefix=True, num_frozen_tokens=1,
        checksum="abc123", file_size_bytes=100_000,
        training_steps=2694,
        source_document="test document",
        labels=labels or {},
    )
    defaults.update(overrides)
    return CartridgeManifest(**defaults)


@pytest.fixture
def registry():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    reg = CartridgeRegistry(db_path)
    yield reg
    reg.close()
    Path(db_path).unlink()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRegisterAndLookup:
    def test_register_and_lookup(self, registry):
        m = _make_manifest("cart_01")
        registry.register(m)
        result = registry.lookup("cart_01")
        assert result is not None
        assert result.cartridge_id == "cart_01"
        assert result.model_id == "meta-llama/Llama-3.2-3B-Instruct"
        assert result.num_layers == 28

    def test_lookup_missing(self, registry):
        assert registry.lookup("nonexistent") is None

    def test_count(self, registry):
        assert registry.count() == 0
        registry.register(_make_manifest("cart_01"))
        assert registry.count() == 1
        registry.register(_make_manifest("cart_02"))
        assert registry.count() == 2


class TestLookupByLabel:
    def test_find_by_patient_id(self, registry):
        registry.register(_make_manifest(
            "cart_p01", labels={"patient_id": "patient_01"}))
        registry.register(_make_manifest(
            "cart_p02", labels={"patient_id": "patient_02"}))
        registry.register(_make_manifest(
            "cart_p03", labels={"patient_id": "patient_01", "dept": "oncology"}))

        results = registry.lookup_by_label("patient_id", "patient_01")
        assert len(results) == 2
        ids = {r.cartridge_id for r in results}
        assert ids == {"cart_p01", "cart_p03"}

    def test_no_match(self, registry):
        registry.register(_make_manifest(
            "cart_01", labels={"patient_id": "patient_01"}))
        results = registry.lookup_by_label("patient_id", "patient_99")
        assert results == []

    def test_label_key_not_present(self, registry):
        registry.register(_make_manifest(
            "cart_01", labels={"patient_id": "patient_01"}))
        results = registry.lookup_by_label("department", "oncology")
        assert results == []


class TestLookupByModel:
    def test_find_by_model(self, registry):
        registry.register(_make_manifest("cart_llama", model_id="meta-llama/Llama-3.2-3B-Instruct"))
        registry.register(_make_manifest("cart_qwen", model_id="Qwen/Qwen2.5-7B-Instruct"))

        results = registry.lookup_by_model("meta-llama/Llama-3.2-3B-Instruct")
        assert len(results) == 1
        assert results[0].cartridge_id == "cart_llama"


class TestUnregister:
    def test_unregister_existing(self, registry):
        registry.register(_make_manifest("cart_01"))
        assert registry.unregister("cart_01") is True
        assert registry.lookup("cart_01") is None
        assert registry.count() == 0

    def test_unregister_missing(self, registry):
        assert registry.unregister("nonexistent") is False


class TestUpdate:
    def test_reregister_overwrites(self, registry):
        m1 = _make_manifest("cart_01", training_steps=100)
        registry.register(m1)
        assert registry.lookup("cart_01").training_steps == 100

        m2 = _make_manifest("cart_01", training_steps=2694)
        registry.register(m2)
        assert registry.lookup("cart_01").training_steps == 2694
        assert registry.count() == 1  # still one entry


class TestListAll:
    def test_empty(self, registry):
        assert registry.list_all() == []

    def test_multiple(self, registry):
        for i in range(5):
            registry.register(_make_manifest(f"cart_{i:02d}"))
        all_carts = registry.list_all()
        assert len(all_carts) == 5
