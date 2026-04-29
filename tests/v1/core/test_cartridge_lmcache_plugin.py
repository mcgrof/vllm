# SPDX-License-Identifier: Apache-2.0
"""Tests for CartridgeLMCachePlugin.

Tests the LMCache StoragePluginInterface adapter without requiring
LMCache to be installed. Uses mock CacheEngineKey objects and
verifies that the plugin correctly delegates to CartridgeStore.

Covers:
1. contains / get_blocking via CacheEngineKey with tags
2. Read-only semantics (put is no-op)
3. Pin / unpin / remove delegation
4. Invalid key handling (missing tags, wrong format)
5. Store injection via set_store
6. Close cleanup
"""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_lmcache_plugin import (
    CartridgeLMCachePlugin,
    _key_to_chunk,
)
from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_manifest import (
    CartridgeManifest,
)
from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_store import (
    CartridgeStore,
    ChunkKey,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cartridge_pt(path, num_layers=2, num_kv_heads=2,
                       num_tokens=32, head_dim=4):
    cache = {"trainable_keys": [], "trainable_values": []}
    for _ in range(num_layers):
        cache["trainable_keys"].append(
            torch.nn.Parameter(torch.randn(1, num_kv_heads, num_tokens, head_dim)))
        cache["trainable_values"].append(
            torch.nn.Parameter(torch.randn(1, num_kv_heads, num_tokens, head_dim)))
    torch.save(cache, path)


def _make_manifest(cartridge_id="cart_01"):
    return CartridgeManifest(
        cartridge_id=cartridge_id,
        model_id="test/model",
        num_layers=2, num_kv_heads=2, head_dim=4,
        dtype="float32",
        num_tokens_raw=32, num_tokens_aligned=32,
        block_size=16, num_blocks=2,
        has_frozen_prefix=False,
    )


def _make_cache_key(cartridge_id, layer_idx):
    """Create a mock CacheEngineKey with tags."""
    key = MagicMock()
    key.tags = (cartridge_id, str(layer_idx))
    return key


@pytest.fixture
def loaded_plugin():
    """Plugin with a loaded cartridge."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        _make_cartridge_pt(f.name)
        store = CartridgeStore(block_size=16)
        store.load("cart_01", f.name, _make_manifest())

        plugin = CartridgeLMCachePlugin(dst_device="cpu")
        plugin.set_store(store)
        yield plugin

        store.close()
        Path(f.name).unlink()


# ---------------------------------------------------------------------------
# Tests: _key_to_chunk helper
# ---------------------------------------------------------------------------

class TestKeyToChunk:
    def test_valid_key(self):
        key = _make_cache_key("cart_01", 5)
        chunk = _key_to_chunk(key)
        assert chunk is not None
        assert chunk.cartridge_id == "cart_01"
        assert chunk.layer_idx == 5

    def test_no_tags(self):
        key = MagicMock()
        key.tags = None
        assert _key_to_chunk(key) is None

    def test_empty_tags(self):
        key = MagicMock()
        key.tags = ()
        assert _key_to_chunk(key) is None

    def test_one_tag(self):
        key = MagicMock()
        key.tags = ("cart_01",)
        assert _key_to_chunk(key) is None

    def test_non_integer_layer(self):
        key = MagicMock()
        key.tags = ("cart_01", "not_a_number")
        assert _key_to_chunk(key) is None


# ---------------------------------------------------------------------------
# Tests: contains / get_blocking
# ---------------------------------------------------------------------------

class TestContainsAndGet:
    def test_contains_loaded_chunk(self, loaded_plugin):
        key = _make_cache_key("cart_01", 0)
        assert loaded_plugin.contains(key) is True

    def test_contains_missing_cartridge(self, loaded_plugin):
        key = _make_cache_key("nonexistent", 0)
        assert loaded_plugin.contains(key) is False

    def test_contains_missing_layer(self, loaded_plugin):
        key = _make_cache_key("cart_01", 99)
        assert loaded_plugin.contains(key) is False

    def test_get_blocking(self, loaded_plugin):
        key = _make_cache_key("cart_01", 0)
        result = loaded_plugin.get_blocking(key)
        assert result is not None
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 32, 2, 4)

    def test_get_blocking_missing(self, loaded_plugin):
        key = _make_cache_key("nonexistent", 0)
        assert loaded_plugin.get_blocking(key) is None

    def test_get_blocking_invalid_key(self, loaded_plugin):
        key = MagicMock()
        key.tags = None
        assert loaded_plugin.get_blocking(key) is None


# ---------------------------------------------------------------------------
# Tests: read-only semantics
# ---------------------------------------------------------------------------

class TestReadOnly:
    def test_put_is_noop(self, loaded_plugin):
        key = _make_cache_key("cart_01", 0)
        result = loaded_plugin.batched_submit_put_task([key], [MagicMock()])
        assert result is None

    def test_exists_in_put_tasks_always_false(self, loaded_plugin):
        key = _make_cache_key("cart_01", 0)
        assert loaded_plugin.exists_in_put_tasks(key) is False


# ---------------------------------------------------------------------------
# Tests: pin / unpin / remove
# ---------------------------------------------------------------------------

class TestPinUnpinRemove:
    def test_pin(self, loaded_plugin):
        key = _make_cache_key("cart_01", 0)
        assert loaded_plugin.pin(key) is True

    def test_unpin(self, loaded_plugin):
        key = _make_cache_key("cart_01", 0)
        loaded_plugin.pin(key)
        assert loaded_plugin.unpin(key) is True

    def test_remove_evicts_cartridge(self, loaded_plugin):
        key = _make_cache_key("cart_01", 0)
        assert loaded_plugin.remove(key, force=True) is True
        assert loaded_plugin.contains(key) is False

    def test_pin_invalid_key(self, loaded_plugin):
        key = MagicMock()
        key.tags = None
        assert loaded_plugin.pin(key) is False


# ---------------------------------------------------------------------------
# Tests: no store
# ---------------------------------------------------------------------------

class TestNoStore:
    def test_contains_without_store(self):
        plugin = CartridgeLMCachePlugin(dst_device="cpu")
        key = _make_cache_key("cart_01", 0)
        assert plugin.contains(key) is False

    def test_get_without_store(self):
        plugin = CartridgeLMCachePlugin(dst_device="cpu")
        key = _make_cache_key("cart_01", 0)
        assert plugin.get_blocking(key) is None


# ---------------------------------------------------------------------------
# Tests: close
# ---------------------------------------------------------------------------

class TestClose:
    def test_close_clears_store(self, loaded_plugin):
        key = _make_cache_key("cart_01", 0)
        assert loaded_plugin.contains(key) is True

        loaded_plugin.close()
        assert loaded_plugin.contains(key) is False
