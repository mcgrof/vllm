# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase-7 shadow-baseline counterfactual tests.

Covers the ``ShadowBaseline`` contract:

  * The shadow observes access + insert events the real cache
    sees, but never sees SPF touches.
  * On insert-with-eviction, the shadow records whether the
    evicted id is one SPF saved (the rescue-pending case).
  * A subsequent reuse on a rescued id bumps
    ``rescued_reuses_count``.
  * A reuse that the shadow still has resident, on an id that
    the real cache evicted, bumps ``harmful_evictions_spf``.
  * Baseline evictions that later get reused are counted as
    ``harmful_evictions_baseline``.
  * The harmful_eviction_delta = baseline - spf; positive means
    SPF helped, negative means it hurt.

Uses a fake ``time.perf_counter``-style clock so retention
lifetime assertions are deterministic.
"""
from __future__ import annotations

from vllm.v1.core.spf.shadow_baseline import ShadowBaseline


class _Clock:
    """Monotonic ms-precision clock test fixture."""

    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def tick(self, dt_ms: float = 1.0) -> None:
        self.t += dt_ms


# ---------------------------------------------------------------------------
# Basic shape
# ---------------------------------------------------------------------------


class TestShape:

    def test_fresh_shadow_has_zero_metrics(self):
        s = ShadowBaseline(capacity=4)
        snap = s.metrics.snapshot()
        assert snap["rescued_reuses_count"] == 0
        assert snap["harmful_evictions_baseline"] == 0
        assert snap["harmful_evictions_spf"] == 0

    def test_capacity_bounded(self):
        s = ShadowBaseline(capacity=3)
        for i in range(10):
            s.on_insert(f"r{i}", num_bytes=100)
        # Only 3 latest remain resident.
        assert len(s) == 3
        assert s.shadow_resident_ids() == {"r7", "r8", "r9"}


# ---------------------------------------------------------------------------
# 1. Rescued reuse — the mechanism signal
# ---------------------------------------------------------------------------


class TestRescueMechanism:

    def test_rescued_reuse_is_counted(self):
        """SPF hint saves R from LRU; a later reuse of R lands —
        rescued_reuses_count += 1 and retention_lifetime_ms is
        recorded."""
        clock = _Clock()
        s = ShadowBaseline(capacity=2, now_ms=clock)

        s.on_insert("A", num_bytes=100)
        clock.tick(5)
        s.on_insert("B", num_bytes=100)
        clock.tick(5)
        # A is now LRU. SPF hints A → real cache keeps it; shadow
        # will evict it on the next insert.
        s.mark_rescue("A", num_bytes=100)
        clock.tick(10)
        # New insertion forces eviction; shadow loses A.
        # would_save_ids tells shadow that SPF saved A.
        evicted = s.on_insert("C", num_bytes=100, would_save_ids={"A"})
        assert "A" in evicted
        assert "A" in s.pending_rescue_ids()

        clock.tick(20)
        # Later: A is reused in the real cache. Tell the shadow.
        s.on_reuse("A")
        snap = s.metrics.snapshot()
        assert snap["rescued_reuses_count"] == 1
        assert snap["rescued_reuses_bytes"] == 100
        assert snap["harmful_evictions_baseline"] == 1
        # Retention lifetime = clock.tick(10)+tick(20) = 30ms
        assert snap["retention_lifetime_ms_mean"] == 30.0

    def test_unused_rescue_is_not_counted(self):
        """SPF rescued but no reuse arrives — counter stays 0."""
        s = ShadowBaseline(capacity=2)
        s.on_insert("A", num_bytes=50)
        s.on_insert("B", num_bytes=50)
        s.mark_rescue("A", num_bytes=50)
        s.on_insert("C", num_bytes=50, would_save_ids={"A"})
        # Never reuse A.
        snap = s.metrics.snapshot()
        assert snap["rescued_reuses_count"] == 0

    def test_multiple_rescues_and_reuses(self):
        clock = _Clock()
        s = ShadowBaseline(capacity=2, now_ms=clock)
        for rid, size in [("A", 100), ("B", 200), ("C", 50), ("D", 75)]:
            s.on_insert(rid, num_bytes=size)
            clock.tick(1)
        # Pool now contains {C, D}. Re-hint A (not resident; goes
        # to pending-rescue map).
        s.mark_rescue("A", num_bytes=100)
        s.mark_rescue("B", num_bytes=200)
        clock.tick(10)
        s.on_reuse("A")
        s.on_reuse("B")
        # Reuse C without rescue ⇒ no bump.
        s.on_reuse("C")
        snap = s.metrics.snapshot()
        assert snap["rescued_reuses_count"] == 2
        assert snap["rescued_reuses_bytes"] == 300


# ---------------------------------------------------------------------------
# 2. Harmful evictions — who got it wrong?
# ---------------------------------------------------------------------------


class TestHarmfulEvictions:

    def test_baseline_evicts_reused_resource(self):
        """Baseline-LRU evicts R; SPF saves it in real cache; a
        reuse later arrives. Baseline would have missed →
        harmful_evictions_baseline bumps."""
        s = ShadowBaseline(capacity=2)
        s.on_insert("A", num_bytes=10)
        s.on_insert("B", num_bytes=10)
        s.mark_rescue("A", num_bytes=10)
        s.on_insert("C", num_bytes=10, would_save_ids={"A"})
        # A lost in shadow, kept in real. Reuse:
        s.on_reuse("A")
        snap = s.metrics.snapshot()
        assert snap["harmful_evictions_baseline"] == 1
        assert snap["harmful_evictions_spf"] == 0
        assert snap["harmful_eviction_delta"] == 1  # SPF helped

    def test_spf_evicts_reused_resource(self):
        """Real cache evicts R (SPF got it wrong); shadow still
        has R; reuse lands → harmful_evictions_spf bumps."""
        s = ShadowBaseline(capacity=4)
        s.on_insert("A", num_bytes=10)
        s.on_insert("B", num_bytes=10)
        # Real cache evicts A.
        s.on_spf_real_eviction("A")
        # Shadow still has A. Reuse arrives:
        s.on_reuse("A")
        snap = s.metrics.snapshot()
        assert snap["harmful_evictions_spf"] == 1
        assert snap["harmful_eviction_delta"] == -1  # SPF hurt

    def test_both_arms_wrong_is_no_delta(self):
        """Shadow evicts AND real cache evicts (SPF didn't save);
        reuse arrives → not attributable to either arm."""
        s = ShadowBaseline(capacity=2)
        s.on_insert("A", num_bytes=10)
        s.on_insert("B", num_bytes=10)
        s.on_insert("C", num_bytes=10)  # evicts A from shadow
        s.on_spf_real_eviction("A")  # SPF also evicts A
        s.on_reuse("A")  # miss in both
        snap = s.metrics.snapshot()
        assert snap["harmful_evictions_baseline"] == 0
        assert snap["harmful_evictions_spf"] == 0


# ---------------------------------------------------------------------------
# 3. Reuse distance accounting
# ---------------------------------------------------------------------------


class TestReuseDistance:

    def test_reuse_distance_tracked(self):
        clock = _Clock()
        s = ShadowBaseline(capacity=2, now_ms=clock)
        s.on_insert("A", num_bytes=10)
        s.on_insert("B", num_bytes=10)
        s.mark_rescue("A", num_bytes=10)
        s.advance_step()  # step 1
        s.advance_step()  # step 2
        s.advance_step()  # step 3
        s.on_insert("C", num_bytes=10, would_save_ids={"A"})
        s.advance_step()  # step 4
        s.advance_step()  # step 5
        s.on_reuse("A")  # rescue distance = 5 - rescue_step=0 = 5

        snap = s.metrics.snapshot()
        assert snap["rescued_reuses_count"] == 1
        assert snap["rescued_reuse_distance_mean"] == 5.0


# ---------------------------------------------------------------------------
# 4. Unrelated accesses don't pollute counters
# ---------------------------------------------------------------------------


class TestUnrelatedTraffic:

    def test_access_on_nonresident_nonrescued_is_noop(self):
        s = ShadowBaseline(capacity=2)
        s.on_access("never_seen")  # no entry
        s.on_reuse("never_seen")  # no entry, no pending
        snap = s.metrics.snapshot()
        assert snap["rescued_reuses_count"] == 0
        assert snap["harmful_evictions_baseline"] == 0
        assert snap["harmful_evictions_spf"] == 0


# ---------------------------------------------------------------------------
# 5. Integration smoke: RetentionIntegration drives the shadow
# ---------------------------------------------------------------------------


class TestRetentionIntegrationWiring:

    def test_retention_integration_drives_shadow(self):
        from vllm.v1.core.spf.config import MODE_RETENTION, SPFConfig
        from vllm.v1.core.spf.controller import SPFController
        from vllm.v1.core.spf.integrations import (FakeBlockPool,
                                                   RetentionIntegration)
        from vllm.v1.core.spf.resource import ResourceId

        clock = _Clock()
        ctrl = SPFController(
            SPFConfig(
                enabled=True,
                mode=MODE_RETENTION,
                metrics_interval=0,
                cooldown_steps=0,
            ))
        pool = FakeBlockPool(capacity=2)
        shadow = ShadowBaseline(capacity=2, now_ms=clock)
        integ = RetentionIntegration(ctrl, pool, shadow=shadow)

        # Prime cache and shadow simultaneously.
        for rid in ["A", "B"]:
            pool.insert(rid)
            integ.on_insert(rid, num_bytes=100)
            clock.tick(1)

        # SPF hints A.
        assert integ.apply_hint("A", num_bytes=100)

        # C arrives — evict LRU in shadow; real cache doesn't
        # evict A because of the hint. We simulate this by
        # passing A in would_save_ids to the shadow insert.
        pool.insert("C")
        integ.on_insert("C", num_bytes=100, would_save_ids={"A"})
        clock.tick(10)

        # Later: A is reused.
        r_A = ResourceId(
            scope=__import__("vllm.v1.core.spf.resource",
                             fromlist=["ResourceScope"]).ResourceScope.PREFIX,
            resource_id="A",
            num_blocks=1,
            num_bytes=100,
            session_id="s",
        )
        integ.on_request_arrival(r_A, arrival_time=clock() / 1000.0)

        m = shadow.metrics.snapshot()
        assert m["rescued_reuses_count"] == 1
        assert m["harmful_evictions_baseline"] == 1
