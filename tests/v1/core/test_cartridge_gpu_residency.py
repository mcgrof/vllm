# SPDX-License-Identifier: Apache-2.0
"""Tests for GPUResidencyManager — bounded GPU tier for cartridge
KV chunks with LRU eviction and refcounted pinning.

Covers the seven acceptance cases from the design:

  1. GPU residency basics — miss→promote, hit (no re-copy).
  2. Refcount safety — concurrent pins, eviction blocked while
     pinned, eviction allowed when refcount hits zero.
  3. LRU eviction — least-recently-used evicted first, evicted
     entries re-promote cleanly.
  4. Pinned entry protection — pinned entries survive pressure;
     all-pinned scenario raises GPUResidencyError.
  5. Multi-cartridge routing correctness — different IDs flow
     through lookup and injection without cross-contamination.
  6. No eager immortal residency — 100 cartridges registered,
     nothing resident until requested.
  7. Metrics coverage — hit / miss / promote / evict counters and
     byte accounting all increment correctly.

All tests run on CPU tensors (device="cpu") since the residency
tier is device-agnostic. GPU integration is validated by the
Phase 4 smoke test on prune W7900.
"""
from __future__ import annotations

import threading

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_gpu_residency import (
    GPUResidencyError,
    GPUResidencyManager,
)
from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_manifest import (
    CartridgeManifest,
)
from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_store import (
    CartridgeStore,
    ChunkKey,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic cartridges in a CartridgeStore
# ---------------------------------------------------------------------------

NUM_LAYERS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 8
TOKENS_PER_CART = 32
BLOCK_SIZE = 16


def _make_store_with_cartridges(
    cart_ids: list[str],
    fill_pattern: dict[str, float] | None = None,
) -> CartridgeStore:
    """Build a CartridgeStore pre-loaded with N synthetic cartridges.

    The store's ``load`` method expects a file path; we stub it by
    placing the manifest + pre-built tensors directly into the
    store's internal state. This isolates residency tests from .pt
    file I/O.
    """
    store = CartridgeStore(block_size=BLOCK_SIZE)
    for cart_id in cart_ids:
        fill = (fill_pattern or {}).get(cart_id, hash(cart_id) % 100)
        manifest = CartridgeManifest(
            cartridge_id=cart_id,
            model_id="test/model",
            num_layers=NUM_LAYERS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            dtype="float32",
            num_tokens_raw=TOKENS_PER_CART,
            num_tokens_aligned=TOKENS_PER_CART,
            block_size=BLOCK_SIZE,
            num_blocks=TOKENS_PER_CART // BLOCK_SIZE,
            has_frozen_prefix=False,
        )
        # Per-chunk tensor: (2, tokens, heads, dim) = (K, V stacked
        # at dim 0).
        chunks = {}
        for layer_idx in range(NUM_LAYERS):
            tensor = torch.full(
                (2, TOKENS_PER_CART, NUM_KV_HEADS, HEAD_DIM),
                float(fill),
                dtype=torch.float32,
            )
            chunks[ChunkKey(cart_id, layer_idx)] = tensor
        # Inject directly into the store's state (bypass load()
        # which expects a .pt path).
        from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_store import (
            CartridgeResidency,
        )
        with store._lock:  # noqa: SLF001 — test-only private access
            store._chunks.update(chunks)
            store._residency[cart_id] = CartridgeResidency(
                cartridge_id=cart_id,
                manifest=manifest,
                num_layers=NUM_LAYERS,
                num_tokens=TOKENS_PER_CART,
            )
    return store


def _chunk_bytes(
    num_layers: int = NUM_LAYERS,
    tokens: int = TOKENS_PER_CART,
    heads: int = NUM_KV_HEADS,
    dim: int = HEAD_DIM,
    dtype: torch.dtype = torch.float32,
) -> int:
    """Byte size of a full cartridge (all layers) at the given
    dtype — used by capacity tests."""
    per_chunk = 2 * tokens * heads * dim
    return num_layers * per_chunk * torch.empty(0, dtype=dtype).element_size()


# ---------------------------------------------------------------------------
# 1. GPU residency basics
# ---------------------------------------------------------------------------

class TestResidencyBasics:
    def test_first_acquire_promotes(self):
        store = _make_store_with_cartridges(["A"])
        mgr = GPUResidencyManager(
            store, capacity_bytes=10 * _chunk_bytes(),
            device="cpu",
        )
        assert not mgr.is_resident("A")
        assert mgr.metrics.gpu_miss == 0

        mgr.acquire("A")
        assert mgr.is_resident("A")
        assert mgr.metrics.gpu_miss == 1
        assert mgr.metrics.gpu_hit == 0
        assert mgr.metrics.promote_count == 1
        assert mgr.metrics.bytes_promoted == _chunk_bytes()
        assert mgr.resident_bytes() == _chunk_bytes()
        mgr.release("A")

    def test_second_acquire_hits(self):
        store = _make_store_with_cartridges(["A"])
        mgr = GPUResidencyManager(
            store, capacity_bytes=10 * _chunk_bytes(), device="cpu",
        )
        mgr.acquire("A")
        mgr.release("A")
        assert mgr.metrics.gpu_miss == 1

        mgr.acquire("A")
        # Second acquire should be a hit, not another promotion.
        assert mgr.metrics.gpu_miss == 1
        assert mgr.metrics.gpu_hit == 1
        assert mgr.metrics.promote_count == 1
        # Total bytes promoted unchanged (no re-copy).
        assert mgr.metrics.bytes_promoted == _chunk_bytes()
        mgr.release("A")

    def test_get_chunk_returns_device_tensor(self):
        store = _make_store_with_cartridges(["A"])
        mgr = GPUResidencyManager(
            store, capacity_bytes=10 * _chunk_bytes(), device="cpu",
        )
        mgr.acquire("A")
        t = mgr.get_chunk("A", 0)
        assert t.device == torch.device("cpu")
        # Identity check: subsequent get returns the same cached
        # tensor (not a re-converted copy).
        t2 = mgr.get_chunk("A", 0)
        assert t.data_ptr() == t2.data_ptr()
        mgr.release("A")

    def test_get_chunk_without_acquire_raises(self):
        store = _make_store_with_cartridges(["A"])
        mgr = GPUResidencyManager(
            store, capacity_bytes=10 * _chunk_bytes(), device="cpu",
        )
        with pytest.raises(GPUResidencyError, match="not resident"):
            mgr.get_chunk("A", 0)

    def test_acquire_unknown_cartridge_raises(self):
        store = _make_store_with_cartridges(["A"])
        mgr = GPUResidencyManager(
            store, capacity_bytes=10 * _chunk_bytes(), device="cpu",
        )
        with pytest.raises(GPUResidencyError,
                           match="not loaded in backing"):
            mgr.acquire("does_not_exist")


# ---------------------------------------------------------------------------
# 2. Refcount safety
# ---------------------------------------------------------------------------

class TestRefcountSafety:
    def test_two_acquires_two_releases(self):
        store = _make_store_with_cartridges(["A"])
        mgr = GPUResidencyManager(
            store, capacity_bytes=10 * _chunk_bytes(), device="cpu",
        )
        mgr.acquire("A")
        mgr.acquire("A")
        assert mgr.ref_count("A") == 2
        # Two misses? No — first was miss, second was hit.
        assert mgr.metrics.gpu_miss == 1
        assert mgr.metrics.gpu_hit == 1

        mgr.release("A")
        assert mgr.ref_count("A") == 1
        mgr.release("A")
        assert mgr.ref_count("A") == 0

    def test_eviction_blocked_while_pinned(self):
        # Two cartridges, capacity only fits one — but A is pinned.
        store = _make_store_with_cartridges(["A", "B"])
        mgr = GPUResidencyManager(
            store, capacity_bytes=int(1.5 * _chunk_bytes()),
            device="cpu",
        )
        mgr.acquire("A")
        # Pin A's refcount so eviction skips it.
        with pytest.raises(GPUResidencyError, match="cannot fit"):
            mgr.acquire("B")
        # A is still resident (never evicted).
        assert mgr.is_resident("A")
        assert not mgr.is_resident("B")
        mgr.release("A")

    def test_refcount_zero_allows_eviction(self):
        store = _make_store_with_cartridges(["A", "B"])
        mgr = GPUResidencyManager(
            store, capacity_bytes=int(1.5 * _chunk_bytes()),
            device="cpu",
        )
        mgr.acquire("A")
        mgr.release("A")
        # A is now refcount=0, eligible for eviction.
        mgr.acquire("B")
        # A should have been evicted to make room for B.
        assert not mgr.is_resident("A")
        assert mgr.is_resident("B")
        assert mgr.metrics.evict_count == 1
        mgr.release("B")

    def test_release_unknown_is_warning_not_error(self):
        store = _make_store_with_cartridges(["A"])
        mgr = GPUResidencyManager(
            store, capacity_bytes=10 * _chunk_bytes(), device="cpu",
        )
        # Does not raise.
        mgr.release("never_acquired")


# ---------------------------------------------------------------------------
# 3. LRU eviction
# ---------------------------------------------------------------------------

class TestLRUEviction:
    def test_lru_evicts_oldest_unpinned(self):
        # Capacity fits exactly 2 cartridges.
        store = _make_store_with_cartridges(["A", "B", "C"])
        mgr = GPUResidencyManager(
            store, capacity_bytes=2 * _chunk_bytes(), device="cpu",
        )
        mgr.acquire("A")
        mgr.release("A")
        mgr.acquire("B")
        mgr.release("B")
        # LRU order now: [A, B]
        assert mgr.lru_order() == ["A", "B"]

        mgr.acquire("C")
        mgr.release("C")
        # A was oldest unpinned, should be evicted to fit C.
        assert not mgr.is_resident("A")
        assert mgr.is_resident("B")
        assert mgr.is_resident("C")
        # LRU order now: [B, C]
        assert mgr.lru_order() == ["B", "C"]
        assert mgr.metrics.evict_count == 1

    def test_access_promotes_to_most_recently_used(self):
        store = _make_store_with_cartridges(["A", "B", "C"])
        mgr = GPUResidencyManager(
            store, capacity_bytes=2 * _chunk_bytes(), device="cpu",
        )
        mgr.acquire("A")
        mgr.release("A")
        mgr.acquire("B")
        mgr.release("B")
        # Touch A — it should become MRU.
        mgr.acquire("A")
        mgr.release("A")
        assert mgr.lru_order() == ["B", "A"]

        # Adding C should now evict B, not A.
        mgr.acquire("C")
        mgr.release("C")
        assert mgr.is_resident("A")
        assert not mgr.is_resident("B")
        assert mgr.is_resident("C")

    def test_re_promotion_after_eviction(self):
        store = _make_store_with_cartridges(["A", "B"])
        mgr = GPUResidencyManager(
            store, capacity_bytes=int(1.5 * _chunk_bytes()),
            device="cpu",
        )
        mgr.acquire("A")
        mgr.release("A")
        mgr.acquire("B")
        mgr.release("B")
        assert not mgr.is_resident("A")

        # Re-promote A — should work, fresh promotion metrics.
        before = mgr.metrics.promote_count
        mgr.acquire("A")
        assert mgr.is_resident("A")
        assert mgr.metrics.promote_count == before + 1
        mgr.release("A")


# ---------------------------------------------------------------------------
# 4. Pinned entry protection
# ---------------------------------------------------------------------------

class TestPinnedProtection:
    def test_evict_unpinned_skips_pinned(self):
        store = _make_store_with_cartridges(["A", "B", "C"])
        mgr = GPUResidencyManager(
            store, capacity_bytes=2 * _chunk_bytes(), device="cpu",
        )
        mgr.acquire("A")  # A pinned (ref=1)
        mgr.acquire("B")
        mgr.release("B")  # B unpinned

        # LRU order: [A (pinned), B (unpinned)]. Adding C should
        # evict B, not A.
        mgr.acquire("C")
        mgr.release("C")
        assert mgr.is_resident("A")
        assert not mgr.is_resident("B")
        assert mgr.is_resident("C")
        mgr.release("A")

    def test_all_pinned_raises(self):
        store = _make_store_with_cartridges(["A", "B", "C"])
        mgr = GPUResidencyManager(
            store, capacity_bytes=2 * _chunk_bytes(), device="cpu",
        )
        mgr.acquire("A")
        mgr.acquire("B")
        # Both pinned; no room for C.
        with pytest.raises(GPUResidencyError, match="cannot fit"):
            mgr.acquire("C")
        # Nothing evicted, no partial state.
        assert mgr.is_resident("A")
        assert mgr.is_resident("B")
        assert not mgr.is_resident("C")
        mgr.release("A")
        mgr.release("B")

    def test_cartridge_larger_than_capacity_raises(self):
        store = _make_store_with_cartridges(["A"])
        # Capacity is half a cartridge — A will never fit.
        mgr = GPUResidencyManager(
            store, capacity_bytes=_chunk_bytes() // 2, device="cpu",
        )
        with pytest.raises(GPUResidencyError,
                           match="will never fit"):
            mgr.acquire("A")


# ---------------------------------------------------------------------------
# 5. Multi-cartridge routing correctness
# ---------------------------------------------------------------------------

class TestMultiCartridgeRouting:
    def test_two_carts_two_requests_isolation(self):
        store = _make_store_with_cartridges(
            ["A", "B"],
            fill_pattern={"A": 1.0, "B": 2.0},
        )
        mgr = GPUResidencyManager(
            store, capacity_bytes=10 * _chunk_bytes(), device="cpu",
        )
        mgr.acquire("A")
        mgr.acquire("B")

        # Each cartridge's layer 0 chunk has its own fill.
        chunk_a = mgr.get_chunk("A", 0)
        chunk_b = mgr.get_chunk("B", 0)
        assert torch.all(chunk_a == 1.0)
        assert torch.all(chunk_b == 2.0)
        # Different tensor objects — no cross-contamination.
        assert chunk_a.data_ptr() != chunk_b.data_ptr()

        mgr.release("A")
        mgr.release("B")

    def test_dtype_conversion_at_promote_time(self):
        store = _make_store_with_cartridges(["A"])
        mgr = GPUResidencyManager(
            store, capacity_bytes=10 * _chunk_bytes(), device="cpu",
        )
        # Store has float32; request promotion to bfloat16.
        mgr.acquire("A", dtype=torch.bfloat16)
        chunk = mgr.get_chunk("A", 0)
        assert chunk.dtype == torch.bfloat16
        # Second acquire hits — same cached bf16 tensor.
        mgr.release("A")
        mgr.acquire("A", dtype=torch.bfloat16)
        chunk2 = mgr.get_chunk("A", 0)
        assert chunk2.data_ptr() == chunk.data_ptr()
        mgr.release("A")


# ---------------------------------------------------------------------------
# 6. No eager immortal residency
# ---------------------------------------------------------------------------

class TestNoEagerResidency:
    def test_100_registered_zero_resident_after_init(self):
        cart_ids = [f"cart_{i:03d}" for i in range(100)]
        store = _make_store_with_cartridges(cart_ids)
        mgr = GPUResidencyManager(
            store, capacity_bytes=5 * _chunk_bytes(), device="cpu",
        )
        # Idle: nothing should be resident yet.
        assert mgr.resident_cartridges() == set()
        assert mgr.resident_bytes() == 0

    def test_only_served_cartridges_resident(self):
        cart_ids = [f"cart_{i:03d}" for i in range(10)]
        store = _make_store_with_cartridges(cart_ids)
        mgr = GPUResidencyManager(
            store, capacity_bytes=5 * _chunk_bytes(), device="cpu",
        )
        for cid in ["cart_002", "cart_005", "cart_007"]:
            mgr.acquire(cid)
            mgr.release(cid)
        resident = mgr.resident_cartridges()
        assert resident.issubset({"cart_002", "cart_005", "cart_007"})
        # None of the unused cartridges were promoted.
        for cid in cart_ids:
            if cid not in ("cart_002", "cart_005", "cart_007"):
                assert not mgr.is_resident(cid)


# ---------------------------------------------------------------------------
# 7. Metrics coverage
# ---------------------------------------------------------------------------

class TestMetricsCoverage:
    def test_hit_miss_counters(self):
        store = _make_store_with_cartridges(["A"])
        mgr = GPUResidencyManager(
            store, capacity_bytes=10 * _chunk_bytes(), device="cpu",
        )
        mgr.acquire("A")
        mgr.release("A")
        mgr.acquire("A")
        mgr.release("A")
        mgr.acquire("A")
        mgr.release("A")
        m = mgr.snapshot_metrics()
        assert m["gpu_miss"] == 1
        assert m["gpu_hit"] == 2
        assert m["promote_count"] == 1

    def test_bytes_accounting(self):
        store = _make_store_with_cartridges(["A", "B"])
        mgr = GPUResidencyManager(
            store, capacity_bytes=10 * _chunk_bytes(), device="cpu",
        )
        mgr.acquire("A")
        m1 = mgr.snapshot_metrics()
        assert m1["current_gpu_resident_bytes"] == _chunk_bytes()
        assert m1["current_gpu_resident_cartridges"] == 1
        assert m1["pinned_entries"] == 1

        mgr.acquire("B")
        m2 = mgr.snapshot_metrics()
        assert m2["current_gpu_resident_bytes"] == 2 * _chunk_bytes()
        assert m2["pinned_entries"] == 2
        mgr.release("A")
        mgr.release("B")

    def test_evict_metrics(self):
        store = _make_store_with_cartridges(["A", "B"])
        mgr = GPUResidencyManager(
            store, capacity_bytes=int(1.5 * _chunk_bytes()),
            device="cpu",
        )
        mgr.acquire("A")
        mgr.release("A")
        mgr.acquire("B")
        mgr.release("B")
        m = mgr.snapshot_metrics()
        assert m["evict_count"] == 1
        assert m["bytes_evicted"] == _chunk_bytes()
        assert m["demote_count"] == 1

    def test_wait_on_promotion_counter(self):
        store = _make_store_with_cartridges(["A", "B", "C"])
        mgr = GPUResidencyManager(
            store, capacity_bytes=10 * _chunk_bytes(), device="cpu",
        )
        for cid in ["A", "B", "C"]:
            mgr.acquire(cid)
            mgr.release(cid)
        # Each first-time acquire counts as a wait.
        assert mgr.snapshot_metrics()[
            "request_wait_on_promotion_count"] == 3
        # Re-acquire A: GPU hit, no new wait.
        mgr.acquire("A")
        mgr.release("A")
        assert mgr.snapshot_metrics()[
            "request_wait_on_promotion_count"] == 3


# ---------------------------------------------------------------------------
# Prefetch
# ---------------------------------------------------------------------------

class TestPrefetch:
    def test_prefetch_promotes_without_pinning(self):
        store = _make_store_with_cartridges(["A"])
        mgr = GPUResidencyManager(
            store, capacity_bytes=10 * _chunk_bytes(), device="cpu",
        )
        assert mgr.prefetch("A")
        assert mgr.is_resident("A")
        assert mgr.ref_count("A") == 0  # not pinned

    def test_prefetch_respects_pinned(self):
        # Capacity fits one. A is pinned.
        store = _make_store_with_cartridges(["A", "B"])
        mgr = GPUResidencyManager(
            store, capacity_bytes=int(1.5 * _chunk_bytes()),
            device="cpu",
        )
        mgr.acquire("A")  # pinned
        # Prefetch B cannot evict A; should return False, no raise.
        assert not mgr.prefetch("B")
        assert not mgr.is_resident("B")
        mgr.release("A")

    def test_prefetch_is_hit_on_subsequent_acquire(self):
        store = _make_store_with_cartridges(["A"])
        mgr = GPUResidencyManager(
            store, capacity_bytes=10 * _chunk_bytes(), device="cpu",
        )
        assert mgr.prefetch("A")
        before_hit = mgr.metrics.gpu_hit
        mgr.acquire("A")
        assert mgr.metrics.gpu_hit == before_hit + 1
        mgr.release("A")


# ---------------------------------------------------------------------------
# Thread safety smoke
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_acquires_and_releases(self):
        store = _make_store_with_cartridges(["A", "B", "C"])
        mgr = GPUResidencyManager(
            store, capacity_bytes=10 * _chunk_bytes(), device="cpu",
        )

        errors = []

        def worker(cart_id: str, n: int):
            try:
                for _ in range(n):
                    mgr.acquire(cart_id)
                    _ = mgr.get_chunk(cart_id, 0)
                    mgr.release(cart_id)
            except Exception as e:  # pragma: no cover
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(cid, 50))
            for cid in ["A", "B", "C"]
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"thread errors: {errors}"
        # After all threads, every cartridge should be at ref 0.
        for cid in ["A", "B", "C"]:
            assert mgr.ref_count(cid) == 0
