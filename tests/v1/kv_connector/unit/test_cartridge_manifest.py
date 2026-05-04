# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for CartridgeManifest.

Covers:
1. Model compatibility validation (wrong layers, heads, head_dim).
2. Block size validation.
3. JSON serialization round-trip.
4. from_cartridge construction from a real .pt file.
5. Label storage for routing metadata.
"""

import tempfile
from pathlib import Path

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_manifest import (
    CartridgeManifest,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_manifest(**overrides) -> CartridgeManifest:
    defaults = dict(
        cartridge_id="test_cart_01",
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        num_layers=28,
        num_kv_heads=8,
        head_dim=128,
        dtype="bfloat16",
        num_tokens_raw=2048,
        num_tokens_aligned=2048,
        block_size=16,
        num_blocks=128,
        has_frozen_prefix=True,
        num_frozen_tokens=1,
        checksum="abc123",
        file_size_bytes=100_000_000,
        training_steps=2694,
        source_document="patient_04 medical record",
        labels={"patient_id": "patient_04", "doc_type": "medical_record"},
    )
    defaults.update(overrides)
    return CartridgeManifest(**defaults)


def _make_trainable_cache_pt(
    path, num_layers=4, num_kv_heads=2, num_tokens=32, head_dim=8
):
    """Create a minimal TrainableCache .pt for from_cartridge tests."""
    cache = {
        "trainable_keys": [],
        "trainable_values": [],
        "frozen_keys": [],
        "frozen_values": [],
    }
    for _ in range(num_layers):
        cache["trainable_keys"].append(
            torch.nn.Parameter(torch.randn(1, num_kv_heads, num_tokens - 1, head_dim))
        )
        cache["trainable_values"].append(
            torch.nn.Parameter(torch.randn(1, num_kv_heads, num_tokens - 1, head_dim))
        )
        cache["frozen_keys"].append(
            torch.nn.Parameter(torch.randn(1, num_kv_heads, 1, head_dim))
        )
        cache["frozen_values"].append(
            torch.nn.Parameter(torch.randn(1, num_kv_heads, 1, head_dim))
        )
    torch.save(cache, path)


# ---------------------------------------------------------------------------
# Tests: model compatibility
# ---------------------------------------------------------------------------


class TestModelCompatibility:
    def test_compatible(self):
        m = _make_manifest()
        errors = m.validate_against_model(
            model_id="meta-llama/Llama-3.2-3B-Instruct",
            num_layers=28,
            num_kv_heads=8,
            head_dim=128,
        )
        assert errors == []

    def test_wrong_model_id(self):
        m = _make_manifest()
        errors = m.validate_against_model(
            model_id="Qwen/Qwen2.5-7B-Instruct",
            num_layers=28,
            num_kv_heads=8,
            head_dim=128,
        )
        assert len(errors) == 1
        assert "model_id" in errors[0]

    def test_wrong_num_layers(self):
        m = _make_manifest()
        errors = m.validate_against_model(
            model_id="meta-llama/Llama-3.2-3B-Instruct",
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
        )
        assert len(errors) == 1
        assert "num_layers" in errors[0]

    def test_wrong_num_kv_heads(self):
        m = _make_manifest()
        errors = m.validate_against_model(
            model_id="meta-llama/Llama-3.2-3B-Instruct",
            num_layers=28,
            num_kv_heads=4,
            head_dim=128,
        )
        assert len(errors) == 1
        assert "num_kv_heads" in errors[0]

    def test_wrong_head_dim(self):
        m = _make_manifest()
        errors = m.validate_against_model(
            model_id="meta-llama/Llama-3.2-3B-Instruct",
            num_layers=28,
            num_kv_heads=8,
            head_dim=64,
        )
        assert len(errors) == 1
        assert "head_dim" in errors[0]

    def test_multiple_mismatches(self):
        m = _make_manifest()
        errors = m.validate_against_model(
            model_id="wrong/model",
            num_layers=99,
            num_kv_heads=99,
            head_dim=99,
        )
        assert len(errors) == 4


class TestBlockSizeCompatibility:
    def test_compatible(self):
        m = _make_manifest(block_size=16)
        assert m.validate_against_block_size(16) == []

    def test_mismatch(self):
        m = _make_manifest(block_size=16)
        errors = m.validate_against_block_size(32)
        assert len(errors) == 1
        assert "block_size" in errors[0]


# ---------------------------------------------------------------------------
# Tests: JSON round-trip
# ---------------------------------------------------------------------------


class TestJsonSerialization:
    def test_round_trip(self):
        m = _make_manifest()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            m.to_json(f.name)
            m2 = CartridgeManifest.from_json(f.name)
            Path(f.name).unlink()

        assert m.cartridge_id == m2.cartridge_id
        assert m.model_id == m2.model_id
        assert m.num_layers == m2.num_layers
        assert m.num_kv_heads == m2.num_kv_heads
        assert m.head_dim == m2.head_dim
        assert m.checksum == m2.checksum
        assert m.labels == m2.labels
        assert m.training_steps == m2.training_steps

    def test_to_dict(self):
        m = _make_manifest()
        d = m.to_dict()
        assert isinstance(d, dict)
        assert d["cartridge_id"] == "test_cart_01"
        assert d["labels"]["patient_id"] == "patient_04"

    def test_from_dict_ignores_unknown_keys(self):
        m = _make_manifest()
        d = m.to_dict()
        d["unknown_future_field"] = "should be ignored"
        m2 = CartridgeManifest.from_dict(d)
        assert m2.cartridge_id == m.cartridge_id


# ---------------------------------------------------------------------------
# Tests: from_cartridge
# ---------------------------------------------------------------------------


class TestFromCartridge:
    def test_builds_from_pt_file(self):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            _make_trainable_cache_pt(
                f.name, num_layers=4, num_kv_heads=2, num_tokens=32, head_dim=8
            )
            m = CartridgeManifest.from_cartridge(
                cartridge_path=f.name,
                cartridge_id="test_from_pt",
                model_id="test/model",
                block_size=16,
                labels={"doc_id": "doc_001"},
                training_steps=100,
            )
            Path(f.name).unlink()

        assert m.cartridge_id == "test_from_pt"
        assert m.model_id == "test/model"
        assert m.num_layers == 4
        assert m.num_kv_heads == 2
        assert m.head_dim == 8
        assert m.num_tokens_raw == 32  # 1 frozen + 31 trainable
        assert m.num_tokens_aligned == 32  # 32 is already aligned to 16
        assert m.num_blocks == 2
        assert m.has_frozen_prefix is True
        assert m.num_frozen_tokens == 1
        assert m.checksum != ""
        assert m.file_size_bytes > 0
        assert m.labels == {"doc_id": "doc_001"}
        assert m.training_steps == 100


# ---------------------------------------------------------------------------
# Tests: labels
# ---------------------------------------------------------------------------


class TestLabels:
    def test_empty_labels(self):
        m = _make_manifest(labels={})
        assert m.labels == {}

    def test_custom_labels(self):
        m = _make_manifest(
            labels={
                "patient_id": "patient_04",
                "department": "oncology",
                "urgency": "routine",
            }
        )
        assert m.labels["department"] == "oncology"
        assert len(m.labels) == 3
