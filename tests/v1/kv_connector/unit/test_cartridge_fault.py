# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fault tests for CartridgeConnector.

These test that the system fails cleanly (not silently) when
cartridge data is corrupt, missing, or incompatible. A cartridge
serving system that silently serves garbage from a corrupted
checkpoint is worse than one that crashes.

Covers:
1. Corrupt checkpoint: truncated file, mangled tensor data.
2. Missing file: cartridge_path doesn't exist.
3. Partial checkpoint: missing trainable_values, missing layers.
4. Shape corruption: wrong head_dim mid-layer, NaN/Inf values.
5. Manifest checksum mismatch: file changed after manifest creation.
6. Empty cartridge: zero tokens.
7. Registry fault: lookup after close / missing DB file.
"""

import os
import pickle
import sqlite3
import tempfile
import zipfile
from pathlib import Path

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_connector import (
    align_to_block_size,
    load_cartridge,
)
from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_manifest import (
    CartridgeManifest,
)
from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_registry import (
    CartridgeRegistry,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_valid_cache(
    num_layers=2, num_kv_heads=2, num_tokens=32, head_dim=8, include_frozen=False
):
    """Create a valid TrainableCache dict."""
    cache = {"trainable_keys": [], "trainable_values": []}
    if include_frozen:
        cache["frozen_keys"] = []
        cache["frozen_values"] = []
    for _ in range(num_layers):
        cache["trainable_keys"].append(
            torch.nn.Parameter(torch.randn(1, num_kv_heads, num_tokens, head_dim))
        )
        cache["trainable_values"].append(
            torch.nn.Parameter(torch.randn(1, num_kv_heads, num_tokens, head_dim))
        )
        if include_frozen:
            cache["frozen_keys"].append(
                torch.nn.Parameter(torch.randn(1, num_kv_heads, 1, head_dim))
            )
            cache["frozen_values"].append(
                torch.nn.Parameter(torch.randn(1, num_kv_heads, 1, head_dim))
            )
    return cache


# torch.load surfaces corruption through several exception types
# depending on where parsing fails (zip container, pickle stream,
# tensor storage). Keep the acceptable set explicit so an unexpected
# silent success still fails the test.
CORRUPT_LOAD_ERRORS = (
    OSError,
    RuntimeError,
    ValueError,
    EOFError,
    zipfile.BadZipFile,
    pickle.UnpicklingError,
)


def _save(cache, path=None):
    if path is None:
        fd, path = tempfile.mkstemp(suffix=".pt")
        os.close(fd)
    torch.save(cache, path)
    return path


# ---------------------------------------------------------------------------
# Tests: corrupt checkpoint files
# ---------------------------------------------------------------------------


class TestCorruptCheckpoint:
    def test_truncated_file(self):
        """A truncated .pt file should raise, not return garbage."""
        path = _save(_make_valid_cache())
        # Truncate the file to half its size
        size = Path(path).stat().st_size
        with open(path, "r+b") as f:
            f.truncate(size // 2)

        with pytest.raises(CORRUPT_LOAD_ERRORS):
            # torch.load should raise on truncated pickle
            load_cartridge(path)
        Path(path).unlink()

    def test_random_bytes(self):
        """Random bytes are not a valid checkpoint."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            f.write(b"\x00\xff\xfe\xfd" * 1000)
            path = f.name

        with pytest.raises(CORRUPT_LOAD_ERRORS):
            load_cartridge(path)
        Path(path).unlink()

    def test_empty_file(self):
        """An empty file should raise."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        with pytest.raises(CORRUPT_LOAD_ERRORS):
            load_cartridge(path)
        Path(path).unlink()


# ---------------------------------------------------------------------------
# Tests: missing file
# ---------------------------------------------------------------------------


class TestMissingFile:
    def test_nonexistent_path(self):
        with pytest.raises(FileNotFoundError):
            load_cartridge("/nonexistent/path/to/cartridge.pt")


# ---------------------------------------------------------------------------
# Tests: partial / malformed checkpoints
# ---------------------------------------------------------------------------


class TestPartialCheckpoint:
    def test_missing_trainable_values(self):
        """Checkpoint with keys but no values should raise."""
        cache = {"trainable_keys": [torch.nn.Parameter(torch.randn(1, 2, 16, 8))]}
        path = _save(cache)
        with pytest.raises(ValueError, match="Cannot find"):
            load_cartridge(path)
        Path(path).unlink()

    def test_missing_trainable_keys(self):
        """Checkpoint with values but no keys should raise."""
        cache = {"trainable_values": [torch.nn.Parameter(torch.randn(1, 2, 16, 8))]}
        path = _save(cache)
        with pytest.raises(ValueError, match="Cannot find"):
            load_cartridge(path)
        Path(path).unlink()

    def test_empty_key_list(self):
        """Empty trainable_keys list should produce empty kv_data."""
        cache = {"trainable_keys": [], "trainable_values": []}
        path = _save(cache)
        # This will either raise or return empty — either is acceptable
        # as long as it doesn't silently produce garbage
        try:
            result = load_cartridge(path)
            assert result["num_layers"] == 0
        except (ValueError, IndexError):
            pass  # also acceptable
        Path(path).unlink()

    def test_mismatched_key_value_layer_count(self):
        """Different number of key and value layers should raise."""
        cache = {
            "trainable_keys": [
                torch.nn.Parameter(torch.randn(1, 2, 16, 8)),
                torch.nn.Parameter(torch.randn(1, 2, 16, 8)),
            ],
            "trainable_values": [
                torch.nn.Parameter(torch.randn(1, 2, 16, 8)),
            ],
        }
        path = _save(cache)
        # Should fail during layer iteration when accessing values[1]
        with pytest.raises((IndexError, ValueError)):
            load_cartridge(path)
        Path(path).unlink()


# ---------------------------------------------------------------------------
# Tests: shape corruption
# ---------------------------------------------------------------------------


class TestShapeCorruption:
    def test_cross_layer_token_count_mismatch(self):
        """Layer 0 has 32 tokens, layer 1 has 16 — should raise."""
        cache = {
            "trainable_keys": [
                torch.nn.Parameter(torch.randn(1, 2, 32, 8)),
                torch.nn.Parameter(torch.randn(1, 2, 16, 8)),
            ],
            "trainable_values": [
                torch.nn.Parameter(torch.randn(1, 2, 32, 8)),
                torch.nn.Parameter(torch.randn(1, 2, 16, 8)),
            ],
        }
        path = _save(cache)
        with pytest.raises(ValueError, match="Layer 1 K shape"):
            load_cartridge(path)
        Path(path).unlink()

    def test_cross_layer_head_dim_mismatch(self):
        """Layer 0 has head_dim=8, layer 1 has head_dim=4."""
        cache = {
            "trainable_keys": [
                torch.nn.Parameter(torch.randn(1, 2, 16, 8)),
                torch.nn.Parameter(torch.randn(1, 2, 16, 4)),
            ],
            "trainable_values": [
                torch.nn.Parameter(torch.randn(1, 2, 16, 8)),
                torch.nn.Parameter(torch.randn(1, 2, 16, 4)),
            ],
        }
        path = _save(cache)
        with pytest.raises(ValueError, match="Layer 1 K shape"):
            load_cartridge(path)
        Path(path).unlink()

    def test_nan_values_load_without_crash(self):
        """NaN values in KV should load (detection is caller's job)."""
        cache = _make_valid_cache(num_layers=1, num_tokens=16)
        cache["trainable_keys"][0].data.fill_(float("nan"))
        path = _save(cache)
        result = load_cartridge(path)
        # Should load — NaN detection is a separate validation step
        assert result["num_tokens"] == 16
        assert torch.isnan(result["kv_data"][0][0]).all()
        Path(path).unlink()

    def test_inf_values_load_without_crash(self):
        """Inf values in KV should load (detection is caller's job)."""
        cache = _make_valid_cache(num_layers=1, num_tokens=16)
        cache["trainable_keys"][0].data.fill_(float("inf"))
        path = _save(cache)
        result = load_cartridge(path)
        assert result["num_tokens"] == 16
        Path(path).unlink()


# ---------------------------------------------------------------------------
# Tests: manifest checksum mismatch
# ---------------------------------------------------------------------------


class TestManifestChecksum:
    def test_checksum_changes_after_modification(self):
        """If the .pt file is modified after manifest creation,
        the checksum should no longer match."""
        cache = _make_valid_cache()
        path = _save(cache)

        manifest = CartridgeManifest.from_cartridge(
            cartridge_path=path,
            cartridge_id="test",
            model_id="test/model",
            block_size=16,
        )
        original_checksum = manifest.checksum

        # Modify the file (re-save with different data)
        cache2 = _make_valid_cache()  # new random data
        torch.save(cache2, path)

        manifest2 = CartridgeManifest.from_cartridge(
            cartridge_path=path,
            cartridge_id="test",
            model_id="test/model",
            block_size=16,
        )

        assert manifest2.checksum != original_checksum
        Path(path).unlink()


# ---------------------------------------------------------------------------
# Tests: empty cartridge
# ---------------------------------------------------------------------------


class TestEmptyCartridge:
    def test_zero_token_cartridge(self):
        """A cartridge with 0 tokens should align to 0 blocks."""
        cache = {
            "trainable_keys": [
                torch.nn.Parameter(torch.randn(1, 2, 0, 8)),
            ],
            "trainable_values": [
                torch.nn.Parameter(torch.randn(1, 2, 0, 8)),
            ],
        }
        path = _save(cache)
        try:
            result = load_cartridge(path)
            assert result["num_tokens"] == 0
            aligned = align_to_block_size(0, 16)
            assert aligned == 0
        except (IndexError, ValueError):
            pass  # also acceptable
        Path(path).unlink()


# ---------------------------------------------------------------------------
# Tests: registry faults
# ---------------------------------------------------------------------------


class TestRegistryFaults:
    def test_lookup_after_close(self):
        """Accessing a closed registry should raise."""
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        reg = CartridgeRegistry(db_path)
        reg.close()
        with pytest.raises(sqlite3.ProgrammingError):
            reg.lookup("anything")
        Path(db_path).unlink()

    def test_missing_db_file_creates_new(self):
        """Opening a non-existent DB path should create a new database."""
        db_path = str(Path(tempfile.mkdtemp()) / "registry.db")
        reg = CartridgeRegistry(db_path)
        assert reg.count() == 0
        reg.close()
        Path(db_path).unlink()

    def test_register_then_delete_db_and_reopen(self):
        """Deleting the DB file and reopening should give empty registry."""
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        reg = CartridgeRegistry(db_path)
        m = CartridgeManifest(
            cartridge_id="cart_01",
            model_id="test",
            num_layers=2,
            num_kv_heads=2,
            head_dim=8,
            dtype="bfloat16",
            num_tokens_raw=32,
            num_tokens_aligned=32,
            block_size=16,
            num_blocks=2,
            has_frozen_prefix=False,
        )
        reg.register(m)
        assert reg.count() == 1
        reg.close()

        # Delete and reopen
        Path(db_path).unlink()
        reg2 = CartridgeRegistry(db_path)
        assert reg2.count() == 0
        reg2.close()
        Path(db_path).unlink()
