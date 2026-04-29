# SPDX-License-Identifier: Apache-2.0
"""Tests for CartridgeStore.

Covers:
1. Load/get/evict lifecycle.
2. Ref counting (acquire/release).
3. Pin/unpin eviction prevention.
4. Multi-cartridge concurrent loading.
5. Memory usage tracking.
6. Thread safety under concurrent access.
7. Eviction with in-flight requests.
8. Double-load rejection.
9. Get after evict returns None.
"""
import tempfile
import threading
from pathlib import Path

import pytest
import torch

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


def _make_manifest(cartridge_id="cart_01", num_layers=2, num_kv_heads=2,
                   head_dim=4, num_tokens=32) -> CartridgeManifest:
    return CartridgeManifest(
        cartridge_id=cartridge_id,
        model_id="test/model",
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype="float32",
        num_tokens_raw=num_tokens,
        num_tokens_aligned=num_tokens,
        block_size=16,
        num_blocks=num_tokens // 16,
        has_frozen_prefix=False,
    )


@pytest.fixture
def store():
    return CartridgeStore(block_size=16)


@pytest.fixture
def cart_file():
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        _make_cartridge_pt(f.name)
        yield f.name
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def cart_file_4layer():
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        _make_cartridge_pt(f.name, num_layers=4, num_tokens=64)
        yield f.name
    Path(f.name).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Tests: load / get / evict lifecycle
# ---------------------------------------------------------------------------

class TestLoadGetEvict:
    def test_load_and_get(self, store, cart_file):
        manifest = _make_manifest()
        chunks = store.load("cart_01", cart_file, manifest)
        assert chunks == 2  # 2 layers

        chunk = store.get(ChunkKey("cart_01", 0))
        assert chunk is not None
        assert chunk.shape == (2, 32, 2, 4)  # (KV, tokens, heads, dim)

        chunk1 = store.get(ChunkKey("cart_01", 1))
        assert chunk1 is not None
        assert chunk1.shape == (2, 32, 2, 4)

    def test_get_missing_returns_none(self, store):
        assert store.get(ChunkKey("nonexistent", 0)) is None

    def test_get_all_layers(self, store, cart_file):
        store.load("cart_01", cart_file, _make_manifest())
        layers = store.get_all_layers("cart_01")
        assert layers is not None
        assert len(layers) == 2
        assert all(l.shape == (2, 32, 2, 4) for l in layers)

    def test_get_all_layers_missing(self, store):
        assert store.get_all_layers("nonexistent") is None

    def test_evict(self, store, cart_file):
        store.load("cart_01", cart_file, _make_manifest())
        assert store.contains("cart_01")

        result = store.evict("cart_01")
        assert result is True
        assert not store.contains("cart_01")
        assert store.get(ChunkKey("cart_01", 0)) is None

    def test_evict_missing(self, store):
        assert store.evict("nonexistent") is False

    def test_double_load_raises(self, store, cart_file):
        store.load("cart_01", cart_file, _make_manifest())
        with pytest.raises(ValueError, match="already loaded"):
            store.load("cart_01", cart_file, _make_manifest())

    def test_reload_after_evict(self, store, cart_file):
        store.load("cart_01", cart_file, _make_manifest())
        store.evict("cart_01")
        # Should work fine after eviction
        chunks = store.load("cart_01", cart_file, _make_manifest())
        assert chunks == 2


# ---------------------------------------------------------------------------
# Tests: ref counting
# ---------------------------------------------------------------------------

class TestRefCounting:
    def test_acquire_release(self, store, cart_file):
        store.load("cart_01", cart_file, _make_manifest())

        assert store.acquire("cart_01") is True
        res = store.get_residency("cart_01")
        assert res.ref_count == 1

        assert store.release("cart_01") is True
        res = store.get_residency("cart_01")
        assert res.ref_count == 0

    def test_acquire_missing(self, store):
        assert store.acquire("nonexistent") is False

    def test_release_missing(self, store):
        assert store.release("nonexistent") is False

    def test_release_below_zero(self, store, cart_file):
        store.load("cart_01", cart_file, _make_manifest())
        store.release("cart_01")  # already at 0
        res = store.get_residency("cart_01")
        assert res.ref_count == 0  # clamped, not negative

    def test_evict_blocked_by_ref(self, store, cart_file):
        store.load("cart_01", cart_file, _make_manifest())
        store.acquire("cart_01")

        # Should fail (ref_count > 0)
        assert store.evict("cart_01") is False
        assert store.contains("cart_01")

        # Force evict should work
        assert store.evict("cart_01", force=True) is True
        assert not store.contains("cart_01")

    def test_multiple_acquires(self, store, cart_file):
        store.load("cart_01", cart_file, _make_manifest())
        store.acquire("cart_01")
        store.acquire("cart_01")
        store.acquire("cart_01")

        res = store.get_residency("cart_01")
        assert res.ref_count == 3

        store.release("cart_01")
        store.release("cart_01")
        res = store.get_residency("cart_01")
        assert res.ref_count == 1
        assert not res.evictable


# ---------------------------------------------------------------------------
# Tests: pin / unpin
# ---------------------------------------------------------------------------

class TestPinUnpin:
    def test_pin_prevents_eviction(self, store, cart_file):
        store.load("cart_01", cart_file, _make_manifest())
        store.pin("cart_01")

        assert store.evict("cart_01") is False
        assert store.contains("cart_01")

    def test_unpin_allows_eviction(self, store, cart_file):
        store.load("cart_01", cart_file, _make_manifest())
        store.pin("cart_01")
        store.unpin("cart_01")

        assert store.evict("cart_01") is True

    def test_pin_missing(self, store):
        assert store.pin("nonexistent") is False

    def test_unpin_missing(self, store):
        assert store.unpin("nonexistent") is False


# ---------------------------------------------------------------------------
# Tests: multi-cartridge
# ---------------------------------------------------------------------------

class TestMultiCartridge:
    def test_load_multiple(self, store, cart_file, cart_file_4layer):
        store.load("cart_01", cart_file, _make_manifest("cart_01"))
        store.load("cart_02", cart_file_4layer,
                   _make_manifest("cart_02", num_layers=4, num_tokens=64))

        assert store.contains("cart_01")
        assert store.contains("cart_02")
        assert len(store.list_loaded()) == 2

        # Different shapes
        c1 = store.get(ChunkKey("cart_01", 0))
        c2 = store.get(ChunkKey("cart_02", 0))
        assert c1.shape[1] == 32  # 32 tokens
        assert c2.shape[1] == 64  # 64 tokens

    def test_evict_one_keeps_other(self, store, cart_file):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f2:
            _make_cartridge_pt(f2.name)
            store.load("cart_01", cart_file, _make_manifest("cart_01"))
            store.load("cart_02", f2.name, _make_manifest("cart_02"))

            store.evict("cart_01")
            assert not store.contains("cart_01")
            assert store.contains("cart_02")
            assert store.get(ChunkKey("cart_02", 0)) is not None
            Path(f2.name).unlink()


# ---------------------------------------------------------------------------
# Tests: memory tracking
# ---------------------------------------------------------------------------

class TestMemoryTracking:
    def test_memory_usage(self, store, cart_file):
        assert store.memory_usage_bytes() == 0

        store.load("cart_01", cart_file, _make_manifest())
        usage = store.memory_usage_bytes()
        assert usage > 0

        # 2 layers * (2 * 32 * 2 * 4) * 4 bytes (float32) = 4096
        expected = 2 * (2 * 32 * 2 * 4) * 4
        assert usage == expected

    def test_memory_drops_after_evict(self, store, cart_file):
        store.load("cart_01", cart_file, _make_manifest())
        assert store.memory_usage_bytes() > 0

        store.evict("cart_01")
        assert store.memory_usage_bytes() == 0


# ---------------------------------------------------------------------------
# Tests: thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_reads(self, store, cart_file):
        """Multiple threads reading the same cartridge concurrently."""
        store.load("cart_01", cart_file, _make_manifest())
        errors = []

        def reader():
            try:
                for _ in range(100):
                    chunk = store.get(ChunkKey("cart_01", 0))
                    if chunk is None:
                        errors.append("got None")
                    elif chunk.shape != (2, 32, 2, 4):
                        errors.append(f"wrong shape: {chunk.shape}")
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=reader) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []

    def test_concurrent_acquire_release(self, store, cart_file):
        """Multiple threads acquiring and releasing concurrently."""
        store.load("cart_01", cart_file, _make_manifest())
        n_ops = 100

        def worker():
            for _ in range(n_ops):
                store.acquire("cart_01")
                store.release("cart_01")

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        res = store.get_residency("cart_01")
        assert res.ref_count == 0  # all released


# ---------------------------------------------------------------------------
# Tests: close
# ---------------------------------------------------------------------------

class TestClose:
    def test_close_clears_everything(self, store, cart_file):
        store.load("cart_01", cart_file, _make_manifest())
        store.close()

        assert store.memory_usage_bytes() == 0
        assert store.list_loaded() == []
        assert store.get(ChunkKey("cart_01", 0)) is None
