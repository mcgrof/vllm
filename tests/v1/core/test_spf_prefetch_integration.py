# SPDX-License-Identifier: Apache-2.0
"""Phase-4 integration tests: SPF prefetch mode ↔ fake LMCache backend.

Covers SPF instruction item 18's prefetch scenarios:

  1. Hint issued before request → promotion completes before
     arrival → request sees prefetched hit (``prefetch_used``).
  2. Hint issued but request arrives BEFORE promotion completes →
     classified as ``"late"`` with
     ``request_wait_on_promotion_count += 1`` (not counted as
     prefetch_used).
  3. Hint issued but request never arrives (e.g. session dies) →
     the caller marks the handle wasted → ``prefetch_wasted``.
  4. ``bytes_promoted`` and ``promote_latency_ms`` accumulate on
     the controller's prefetch-mode metrics.

Uses :class:`FakePrefetchBackend` which lets tests explicitly
sequence "hint issued → arrival → completion" in any order. That's
the only way to make the "arrived-too-early" and
"completed-before-arrival" cases fully deterministic.
"""
from __future__ import annotations

from vllm.v1.core.spf.config import MODE_PREFETCH, SPFConfig
from vllm.v1.core.spf.controller import SPFController
from vllm.v1.core.spf.integrations import (
    FakePrefetchBackend,
    PrefetchIntegration,
)
from vllm.v1.core.spf.resource import ResourceId


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _controller() -> SPFController:
    return SPFController(SPFConfig(
        enabled=True,
        mode=MODE_PREFETCH,
        metrics_interval=0,
        cooldown_steps=0,
    ))


def _resource(tail: int, session: str = "s0") -> ResourceId:
    tokens = list(range(16)) + [tail]
    return ResourceId.for_prefix(tokens, session_id=session)


def _wire() -> tuple[SPFController, FakePrefetchBackend, PrefetchIntegration]:
    """Construct controller + backend + integration all wired
    together the way a real deployment would be.
    """
    c = _controller()
    backend = FakePrefetchBackend(bytes_per_block=1_000_000)
    integ = PrefetchIntegration(c, backend)
    backend.attach(integ)
    return c, backend, integ


# ---------------------------------------------------------------------------
# 1. Happy path: promotion completes before arrival
# ---------------------------------------------------------------------------

class TestPrefetchHappyPath:
    def test_completed_before_arrival_is_used(self):
        c, backend, integ = _wire()
        r = _resource(1)

        # t=100: hint issued.
        integ.apply_hint(r.resource_id, issued_at=100.0)
        # t=101: backend reports completion (1ms later).
        backend.complete(r.resource_id, completed_at=101.0,
                         bytes_promoted=2_000_000)
        # t=102: request arrives — well after completion.
        outcome = integ.on_request_arrival(r, arrival_time=102.0)
        assert outcome == "used"

        snap = c.metrics_snapshot
        assert snap["mode"] == "prefetch"
        assert snap["hints_issued"] == 1
        assert snap["prefetch_used"] == 1
        assert snap["prefetch_wasted"] == 0
        assert snap["bytes_promoted"] == 2_000_000
        assert snap["promote_latency_ms"] == 1000.0  # 1 second
        assert snap["request_wait_on_promotion_count"] == 0


# ---------------------------------------------------------------------------
# 2. Late: request arrives before promotion completes
# ---------------------------------------------------------------------------

class TestPrefetchLate:
    def test_arrival_before_completion_is_late_not_success(self):
        c, backend, integ = _wire()
        r = _resource(2)

        # t=100: hint issued; promotion kicked off.
        integ.apply_hint(r.resource_id, issued_at=100.0)
        # t=100.5: request arrives — promotion still in flight.
        outcome = integ.on_request_arrival(r, arrival_time=100.5)
        assert outcome == "late"

        snap = c.metrics_snapshot
        assert snap["hints_issued"] == 1
        assert snap["prefetch_used"] == 0  # NOT counted as success
        assert snap["request_wait_on_promotion_count"] == 1
        # Handle remains in flight — test cleanup via expire.
        integ.expire_wasted(r.resource_id)

    def test_completion_after_arrival_does_not_retroactively_use(self):
        """Even if the promotion eventually finishes, a request
        that arrived while in flight is 'late'. A second request
        much later could be a hit, but the first one isn't."""
        c, backend, integ = _wire()
        r = _resource(3)
        integ.apply_hint(r.resource_id, issued_at=100.0)
        # Request arrives before completion.
        integ.on_request_arrival(r, arrival_time=100.2)
        # Promotion finishes.
        backend.complete(r.resource_id, completed_at=101.0,
                         bytes_promoted=500_000)
        # No retroactive flip of used; bytes_promoted still
        # accumulates though — the move did happen.
        snap = c.metrics_snapshot
        assert snap["request_wait_on_promotion_count"] == 1
        assert snap["prefetch_used"] == 0
        assert snap["bytes_promoted"] == 500_000


# ---------------------------------------------------------------------------
# 3. Never consumed: waste via expire
# ---------------------------------------------------------------------------

class TestPrefetchWaste:
    def test_expire_wasted_flips_outcome(self):
        c, backend, integ = _wire()
        r = _resource(4)
        integ.apply_hint(r.resource_id, issued_at=100.0)
        backend.complete(r.resource_id, completed_at=101.0,
                         bytes_promoted=750_000)

        # Session dies; no request ever arrives. Mark wasted.
        integ.expire_wasted(r.resource_id)

        snap = c.metrics_snapshot
        assert snap["hints_issued"] == 1
        assert snap["prefetch_wasted"] == 1
        assert snap["prefetch_used"] == 0
        assert snap["bytes_promoted"] == 750_000  # bytes DID move

    def test_expire_unknown_is_noop(self):
        c, _, integ = _wire()
        # No hint issued for this resource.
        integ.expire_wasted("phantom")
        snap = c.metrics_snapshot
        assert snap["hints_issued"] == 0
        assert snap["prefetch_wasted"] == 0


# ---------------------------------------------------------------------------
# 4. Miss-without-hint
# ---------------------------------------------------------------------------

class TestPrefetchMiss:
    def test_unhinted_request_is_miss(self):
        c, _, integ = _wire()
        r = _resource(5)
        outcome = integ.on_request_arrival(r, arrival_time=100.0)
        assert outcome == "miss"

        snap = c.metrics_snapshot
        assert snap["hints_issued"] == 0
        assert snap["prefetch_used"] == 0
        assert snap["request_wait_on_promotion_count"] == 0


# ---------------------------------------------------------------------------
# Bytes / latency accumulation
# ---------------------------------------------------------------------------

class TestBytesAndLatency:
    def test_multiple_promotions_accumulate(self):
        c, backend, integ = _wire()
        rs = [_resource(i) for i in range(3)]

        # Three hints, all promoted and arrived cleanly.
        for i, r in enumerate(rs):
            t0 = 100.0 + i
            integ.apply_hint(r.resource_id, issued_at=t0)
            backend.complete(r.resource_id, completed_at=t0 + 0.5,
                             bytes_promoted=1_000_000 * (i + 1))
            integ.on_request_arrival(r, arrival_time=t0 + 1.0)

        snap = c.metrics_snapshot
        assert snap["hints_issued"] == 3
        assert snap["prefetch_used"] == 3
        # 1M + 2M + 3M = 6M
        assert snap["bytes_promoted"] == 6_000_000
        # Each took 0.5s = 500ms, three of them ⇒ 1500ms.
        assert snap["promote_latency_ms"] == 1500.0


# ---------------------------------------------------------------------------
# Honesty invariants
# ---------------------------------------------------------------------------

class TestPrefetchHonesty:
    def test_mode_tagged_prefetch(self):
        c, _, _ = _wire()
        snap = c.metrics_snapshot
        assert snap["mode"] == "prefetch"

    def test_retention_counters_absent_in_prefetch_mode(self):
        c, _, _ = _wire()
        snap = c.metrics_snapshot
        assert "hints_used" not in snap
        assert "hints_wasted" not in snap
        assert "harmful_evictions_avoided" not in snap

    def test_resource_id_not_first_block_hash(self):
        """Two resources with colliding first-block hashes must be
        tracked separately by the prefetch integration. Otherwise
        a hint on one would mistakenly resolve when the OTHER
        lands — the classic first-block-only bug."""
        c, backend, integ = _wire()

        shared = list(range(16))
        r_A = ResourceId.for_prefix(shared + [1], session_id="s0")
        r_B = ResourceId.for_prefix(shared + [2], session_id="s0")
        assert r_A.first_block_hash == r_B.first_block_hash
        assert r_A.resource_id != r_B.resource_id

        integ.apply_hint(r_A.resource_id, issued_at=100.0)
        backend.complete(r_A.resource_id, completed_at=100.5,
                         bytes_promoted=500_000)

        # B arrives. No hint for B → miss. (If we were keyed by
        # first_block_hash, this would be misreported as 'used'.)
        outcome = integ.on_request_arrival(r_B, arrival_time=101.0)
        assert outcome == "miss"

        snap = c.metrics_snapshot
        assert snap["prefetch_used"] == 0

        # A arrives. That's the real hit.
        outcome = integ.on_request_arrival(r_A, arrival_time=102.0)
        assert outcome == "used"

        snap = c.metrics_snapshot
        assert snap["prefetch_used"] == 1
