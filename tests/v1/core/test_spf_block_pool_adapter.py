# SPDX-License-Identifier: Apache-2.0
"""Unit tests for :class:`LiveBlockPool`.

Tests the adapter against a hand-rolled fake BlockPool that has
the same method signatures vLLM v1's real one does, but none of
its complex state. We verify:

  * ``touch`` resolves → delegates to pool.touch(block) → returns
    True on success / False on cache miss.
  * ``contains`` resolves → checks block_hash presence.
  * ``evict_lru`` pops from the free queue and returns string
    ids.
  * Blocks without a block_hash (raw, not cache-indexed) are
    NOT touchable.

No real vLLM engine spun up. The live run itself is deployment
work outside this test file.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

from vllm.v1.core.spf.block_pool_adapter import LiveBlockPool


# ---------------------------------------------------------------------------
# Fake BlockPool — mirrors the v1 API surface we need.
# ---------------------------------------------------------------------------

@dataclass
class FakeKVCacheBlock:
    block_id: int
    ref_cnt: int = 0
    is_null: bool = False
    block_hash: Optional[object] = None
    # Linked-list pointers for the fake free queue.
    prev_free_block: Optional["FakeKVCacheBlock"] = None
    next_free_block: Optional["FakeKVCacheBlock"] = None


@dataclass
class FakeFreeQueue:
    """Minimal doubly-linked list matching the subset of
    ``FreeKVCacheBlockQueue`` the adapter uses."""

    _nodes: list = field(default_factory=list)

    def append(self, block: FakeKVCacheBlock) -> None:
        self._nodes.append(block)

    def remove(self, block: FakeKVCacheBlock) -> None:
        self._nodes.remove(block)

    def popleft(self) -> Optional[FakeKVCacheBlock]:
        if not self._nodes:
            return None
        return self._nodes.pop(0)

    def __len__(self) -> int:
        return len(self._nodes)


@dataclass
class FakeBlockPool:
    """Minimal surface of :class:`vllm.v1.core.block_pool.BlockPool`
    the SPF adapter touches."""

    free_block_queue: FakeFreeQueue = field(
        default_factory=FakeFreeQueue)
    # Direct map used by touch() for verification in tests.
    touched: list = field(default_factory=list)

    def touch(self, blocks) -> None:
        for b in blocks:
            if b.ref_cnt == 0 and not b.is_null:
                try:
                    self.free_block_queue.remove(b)
                except ValueError:
                    pass
            b.ref_cnt += 1
            self.touched.append(b.block_id)

    def evict_blocks(self, block_ids: set) -> None:
        pass  # not used in these tests


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTouch:
    def test_touch_resident_block_succeeds(self):
        pool = FakeBlockPool()
        blk = FakeKVCacheBlock(
            block_id=1, block_hash="h-abc", ref_cnt=0)
        pool.free_block_queue.append(blk)
        resolver_map = {"rid-abc": blk}
        live = LiveBlockPool(
            pool, kv_cache_group_ids=[0],
            resolver=resolver_map.get,
        )
        assert live.touch("rid-abc") is True
        # Pool.touch was invoked.
        assert 1 in pool.touched
        # Ref count bumped.
        assert blk.ref_cnt == 1
        # Block removed from free queue (no longer evictable).
        assert len(pool.free_block_queue) == 0

    def test_touch_unresolvable_returns_false(self):
        pool = FakeBlockPool()
        live = LiveBlockPool(
            pool, kv_cache_group_ids=[0],
            resolver=lambda rid: None,
        )
        assert live.touch("rid-nope") is False
        # Pool.touch never called.
        assert pool.touched == []

    def test_touch_block_without_hash_returns_false(self):
        """A :class:`KVCacheBlock` with ``block_hash=None`` is not
        in the prefix cache — adapter refuses to touch it."""
        pool = FakeBlockPool()
        raw_blk = FakeKVCacheBlock(
            block_id=2, block_hash=None, ref_cnt=0)
        resolver_map = {"rid-raw": raw_blk}
        live = LiveBlockPool(
            pool, kv_cache_group_ids=[0],
            resolver=resolver_map.get,
        )
        assert live.touch("rid-raw") is False
        assert pool.touched == []


class TestContains:
    def test_contains_resolved_with_hash(self):
        pool = FakeBlockPool()
        blk = FakeKVCacheBlock(block_id=3, block_hash="h-xyz")
        live = LiveBlockPool(
            pool, kv_cache_group_ids=[0],
            resolver={"rid-xyz": blk}.get,
        )
        assert live.contains("rid-xyz") is True

    def test_contains_unresolvable_is_false(self):
        pool = FakeBlockPool()
        live = LiveBlockPool(
            pool, kv_cache_group_ids=[0],
            resolver=lambda rid: None,
        )
        assert live.contains("rid-nope") is False

    def test_contains_without_hash_is_false(self):
        pool = FakeBlockPool()
        raw_blk = FakeKVCacheBlock(block_id=4, block_hash=None)
        live = LiveBlockPool(
            pool, kv_cache_group_ids=[0],
            resolver={"rid-raw": raw_blk}.get,
        )
        assert live.contains("rid-raw") is False


class TestEvictLRU:
    def test_evict_lru_returns_string_ids(self):
        pool = FakeBlockPool()
        # Populate free queue with three cached blocks.
        for i, h in enumerate(["h1", "h2", "h3"]):
            blk = FakeKVCacheBlock(
                block_id=i, block_hash=h)
            pool.free_block_queue.append(blk)
        live = LiveBlockPool(
            pool, kv_cache_group_ids=[0],
            resolver=lambda rid: None,
        )
        evicted = live.evict_lru(n=2)
        assert len(evicted) == 2
        # Head of the queue came first — "h1" then "h2".
        assert evicted == ["h1", "h2"]
        assert len(pool.free_block_queue) == 1

    def test_evict_lru_on_empty_queue(self):
        pool = FakeBlockPool()
        live = LiveBlockPool(
            pool, kv_cache_group_ids=[0],
            resolver=lambda rid: None,
        )
        assert live.evict_lru(n=5) == []


# ---------------------------------------------------------------------------
# Integration with RetentionIntegration
# ---------------------------------------------------------------------------

class TestWithRetentionIntegration:
    def test_live_pool_plugs_into_retention_integration(self):
        """The Protocol contract: LiveBlockPool is
        interchangeable with FakeBlockPool as far as
        :class:`RetentionIntegration` is concerned."""
        from vllm.v1.core.spf.config import (
            MODE_RETENTION, SPFConfig,
        )
        from vllm.v1.core.spf.controller import SPFController
        from vllm.v1.core.spf.integrations import (
            RetentionIntegration,
        )

        ctrl = SPFController(SPFConfig(
            enabled=True, mode=MODE_RETENTION,
            metrics_interval=0, cooldown_steps=0,
        ))
        pool = FakeBlockPool()
        blk = FakeKVCacheBlock(block_id=1, block_hash="h-A")
        pool.free_block_queue.append(blk)
        live = LiveBlockPool(
            pool, kv_cache_group_ids=[0],
            resolver={"rid-A": blk}.get,
        )
        integ = RetentionIntegration(ctrl, live)

        # Hint applied ⇒ touch called ⇒ block removed from free
        # queue ⇒ SPF counters updated.
        assert integ.apply_hint("rid-A")
        snap = ctrl.metrics_snapshot
        assert snap["hints_issued"] == 1
        assert blk.ref_cnt == 1

    def test_unresolvable_hint_marks_waste(self):
        from vllm.v1.core.spf.config import (
            MODE_RETENTION, SPFConfig,
        )
        from vllm.v1.core.spf.controller import SPFController
        from vllm.v1.core.spf.integrations import (
            RetentionIntegration,
        )
        ctrl = SPFController(SPFConfig(
            enabled=True, mode=MODE_RETENTION,
            metrics_interval=0, cooldown_steps=0,
        ))
        pool = FakeBlockPool()
        live = LiveBlockPool(
            pool, kv_cache_group_ids=[0],
            resolver=lambda rid: None,
        )
        integ = RetentionIntegration(ctrl, live)
        assert integ.apply_hint("rid-missing") is False
        snap = ctrl.metrics_snapshot
        assert snap["hints_issued"] == 1
        assert snap["hints_wasted"] == 1
