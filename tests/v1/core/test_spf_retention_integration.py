# SPDX-License-Identifier: Apache-2.0
"""Phase-4 integration tests: SPF retention mode ↔ block pool.

Covers the four scenarios from SPF instruction item 18's
retention family:

  1. Hint arrives → block_pool.touch succeeds → block survives
     eviction pressure → later request hits.
  2. Hint issued → never consumed → eviction fires → outcome
     counted as waste.
  3. Hint issued but block was already evicted at touch time →
     immediate waste (the "too late" path).
  4. Resource identity is resource_id throughout, not
     first_block_hash (locks in the phase-1 fix at the
     integration boundary).

Uses the in-tree :class:`FakeBlockPool`; no real vLLM block-pool
hookup here (that's phase 5). The fake's behaviour matches the
real pool on the three methods the SPF integration touches.
"""
from __future__ import annotations

from vllm.v1.core.spf.config import MODE_RETENTION, SPFConfig
from vllm.v1.core.spf.controller import SPFController
from vllm.v1.core.spf.integrations import (
    FakeBlockPool,
    RetentionIntegration,
)
from vllm.v1.core.spf.resource import ResourceId


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _controller(**overrides) -> SPFController:
    defaults = dict(
        enabled=True, mode=MODE_RETENTION, metrics_interval=0,
        cooldown_steps=0,  # off so tests can drive hints directly
    )
    defaults.update(overrides)
    return SPFController(SPFConfig(**defaults))


def _resource(token_tail: int, session: str = "s0") -> ResourceId:
    """Build a ResourceId deterministically. The first 32 tokens
    are shared prefix-padding; the last token distinguishes."""
    tokens = list(range(32)) + [token_tail]
    return ResourceId.for_prefix(tokens, session_id=session)


# ---------------------------------------------------------------------------
# 1. Happy path: hint → touch → survive pressure → used
# ---------------------------------------------------------------------------

class TestRetentionHappyPath:
    def test_hint_succeeds_when_resource_resident(self):
        c = _controller()
        pool = FakeBlockPool(capacity=4)
        pool.insert("rA")
        integ = RetentionIntegration(c, pool)

        ok = integ.apply_hint("rA")
        assert ok is True
        assert "rA" in pool.touches
        # hints_issued is tracked; outstanding set has rA.
        assert c.metrics_snapshot["hints_issued"] == 1
        assert c.metrics_snapshot["outstanding"] == 1

    def test_block_survives_pressure_after_touch(self):
        """With LRU behaviour, a touched block moves to MRU and
        later insertions evict OTHER entries first."""
        pool = FakeBlockPool(capacity=3)
        pool.insert("a")
        pool.insert("b")
        pool.insert("c")
        # Touch 'a' — it's now MRU.
        assert pool.touch("a")
        # Insert 'd' — capacity is 3, LRU is 'b' now. Evict b.
        evicted = pool.insert("d")
        assert evicted == ["b"]
        assert pool.contains("a")
        assert not pool.contains("b")
        assert pool.contains("c")
        assert pool.contains("d")

    def test_later_request_marks_hint_used(self):
        c = _controller()
        pool = FakeBlockPool(capacity=4)
        r = _resource(1)
        pool.insert(r.resource_id)
        integ = RetentionIntegration(c, pool)

        integ.apply_hint(r.resource_id)
        # Simulate the request for r arriving.
        integ.on_request_arrival(r, arrival_time=100.0)

        snap = c.metrics_snapshot
        assert snap["hints_used"] == 1
        assert snap["hints_wasted"] == 0
        assert snap["outstanding"] == 0


# ---------------------------------------------------------------------------
# 2. Waste path: hint issued, never used, evicted
# ---------------------------------------------------------------------------

class TestRetentionWastePath:
    def test_eviction_of_hinted_resource_marks_waste(self):
        c = _controller()
        pool = FakeBlockPool(capacity=2)
        pool.insert("rA")
        integ = RetentionIntegration(c, pool)

        integ.apply_hint("rA")
        # No request arrives. Pool fills up; rA gets evicted.
        pool.insert("rB")
        pool.insert("rC")
        # Verify rA is gone, then notify integration.
        assert not pool.contains("rA")
        integ.on_eviction(["rA"])

        snap = c.metrics_snapshot
        assert snap["hints_wasted"] == 1
        assert snap["hints_used"] == 0
        assert snap["outstanding"] == 0

    def test_eviction_of_unhinted_resource_is_noop(self):
        """Evicting a resource we never hinted must not bump any
        retention counter — it's just normal LRU churn."""
        c = _controller()
        pool = FakeBlockPool(capacity=2)
        integ = RetentionIntegration(c, pool)

        # Only rA is hinted; rB is a bystander.
        pool.insert("rA")
        pool.insert("rB")
        integ.apply_hint("rA")

        integ.on_eviction(["rB"])  # bystander evicted

        snap = c.metrics_snapshot
        assert snap["hints_issued"] == 1
        assert snap["hints_wasted"] == 0  # rB not hinted


# ---------------------------------------------------------------------------
# 3. Too-late path: hint fired but resource already evicted
# ---------------------------------------------------------------------------

class TestRetentionTooLate:
    def test_touch_fails_immediately_counts_as_waste(self):
        c = _controller()
        pool = FakeBlockPool(capacity=1)
        integ = RetentionIntegration(c, pool)

        # Never inserted — touch returns False.
        ok = integ.apply_hint("never_there")
        assert ok is False

        snap = c.metrics_snapshot
        assert snap["hints_issued"] == 1
        assert snap["hints_wasted"] == 1  # immediate waste
        assert snap["outstanding"] == 0


# ---------------------------------------------------------------------------
# 4. Identity is resource_id, not first_block_hash
# ---------------------------------------------------------------------------

class TestRetentionIdentity:
    def test_two_resources_sharing_first_block_are_distinct(self):
        """Two resources that would collide under the legacy
        first-block-only identity must be tracked separately by
        the retention integration. Otherwise a hint on one would
        mistakenly resolve when the OTHER lands."""
        c = _controller()
        pool = FakeBlockPool(capacity=4)

        shared_prefix = list(range(16))
        r_A = ResourceId.for_prefix(
            shared_prefix + [100], session_id="s0")
        r_B = ResourceId.for_prefix(
            shared_prefix + [200], session_id="s0")
        # Legacy collision, honest distinction.
        assert r_A.first_block_hash == r_B.first_block_hash
        assert r_A.resource_id != r_B.resource_id

        pool.insert(r_A.resource_id)
        pool.insert(r_B.resource_id)
        integ = RetentionIntegration(c, pool)

        # Hint ONLY on r_A.
        integ.apply_hint(r_A.resource_id)

        # Simulate request for r_B arriving — it must NOT be
        # counted as "hint used" (we never hinted r_B).
        integ.on_request_arrival(r_B, arrival_time=200.0)

        snap = c.metrics_snapshot
        assert snap["hints_issued"] == 1
        assert snap["hints_used"] == 0
        # Now request for r_A arrives → that IS the hint payoff.
        integ.on_request_arrival(r_A, arrival_time=201.0)
        snap = c.metrics_snapshot
        assert snap["hints_used"] == 1


# ---------------------------------------------------------------------------
# Extra: honest retention never pretends to prefetch
# ---------------------------------------------------------------------------

class TestRetentionNamingHonesty:
    def test_snapshot_mode_is_retention(self):
        c = _controller()
        pool = FakeBlockPool(capacity=2)
        integ = RetentionIntegration(c, pool)
        pool.insert("x")
        integ.apply_hint("x")
        snap = c.metrics_snapshot
        assert snap["mode"] == "retention"
        # And the prefetch-only counters do not appear.
        assert "bytes_promoted" not in snap
        assert "prefetch_used" not in snap
