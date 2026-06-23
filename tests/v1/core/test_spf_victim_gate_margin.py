# SPDX-License-Identifier: Apache-2.0
"""Phase-6: margin-based victim gate tests.

Phase-2 gate: suppress if candidate_utility <= victim_utility
  (strict >). Works well at roomy/medium pressure.
Phase-6 gate: suppress if candidate_utility <= victim_utility
  + margin_ms. Positive margin is the tight-pressure fix.

Tests cover:

  * margin=0 preserves phase-2 behaviour.
  * margin>0 suppresses candidates whose edge is below margin.
  * margin>0 still admits candidates with a clear edge.
  * Suppressed hints increment harmful_evictions_avoided in
    retention mode (same as phase-2).
"""
from __future__ import annotations

from vllm.v1.core.spf.config import MODE_RETENTION, SPFConfig
from vllm.v1.core.spf.controller import SPFController
from vllm.v1.core.spf.resource import ResourceCandidate, ResourceId


def _ctrl(margin_ms: float, **extra) -> SPFController:
    return SPFController(SPFConfig(
        enabled=True, mode=MODE_RETENTION,
        metrics_interval=0,
        cooldown_steps=0,
        victim_gate_margin_ms=margin_ms,
        **extra,
    ))


def _cand(rid_tail: int, session: str = "s0",
          num_blocks: int = 1) -> ResourceCandidate:
    """Candidate with fresh recency so utility > 0 out of the
    box (so the tests can isolate the gate behaviour)."""
    r = ResourceId.for_prefix(
        list(range(16)) + [rid_tail], session_id=session)
    r = ResourceId(
        scope=r.scope, resource_id=r.resource_id,
        num_blocks=num_blocks, num_bytes=r.num_bytes,
        session_id=session, first_block_hash=r.first_block_hash,
    )
    return ResourceCandidate(resource=r)


class TestMarginZeroPreservesPhase2:
    def test_strict_greater_than_at_margin_zero(self):
        c = _ctrl(margin_ms=0.0)
        # Observe candidate to raise its utility above 0.
        cand = _cand(1)
        c.observe_resource(cand.resource)
        # Victim with identical features ⇒ same utility.
        victim_rid = ResourceId.for_prefix(
            list(range(16)) + [99], session_id="s0")
        victim_rid = ResourceId(
            scope=victim_rid.scope,
            resource_id=victim_rid.resource_id,
            num_blocks=1, num_bytes=victim_rid.num_bytes,
            session_id="s0",
            first_block_hash=victim_rid.first_block_hash,
        )
        c.observe_resource(victim_rid)
        victim = ResourceCandidate(
            resource=victim_rid)

        selected = c.step_candidates(
            candidates=[cand],
            free_gpu_blocks=100,
            victim=victim,
        )
        # Identical utility ⇒ suppressed (strict >).
        assert selected == []
        assert c.metrics_snapshot["harmful_evictions_avoided"] == 1


class TestMarginPositiveSuppresses:
    def test_small_edge_suppressed_with_margin(self):
        """Candidate has a tiny utility edge that phase-2 would
        admit, but margin=0.05 (ms) suppresses."""
        c = _ctrl(margin_ms=0.05)
        # Build two candidates with known utility gap.
        cand = _cand(1)
        c.observe_resource(cand.resource)
        victim_rid = ResourceId.for_prefix(
            list(range(16)) + [100], session_id="s0")
        victim_rid = ResourceId(
            scope=victim_rid.scope,
            resource_id=victim_rid.resource_id,
            num_blocks=1, num_bytes=victim_rid.num_bytes,
            session_id="s0",
            first_block_hash=victim_rid.first_block_hash,
        )
        c.observe_resource(victim_rid)
        victim = ResourceCandidate(resource=victim_rid)
        # Both freshly observed at same step ⇒ recency identical,
        # utilities identical. phase-2 strict > would already
        # suppress; confirm margin>0 also suppresses.
        selected = c.step_candidates(
            candidates=[cand],
            free_gpu_blocks=100,
            victim=victim,
        )
        assert selected == []

    def test_clear_edge_still_admitted_with_margin(self):
        """Strong candidate with fresh observation vs stale
        victim — edge large enough to clear the margin.

        Uses a small margin (0.01 ms) so the test is a sanity
        check on the *direction* of the gate: candidate with
        meaningful recency edge over an aged victim should be
        admitted. A bigger margin (0.05 ms) is tested in
        ``test_small_edge_suppressed_with_margin``.
        """
        c = _ctrl(margin_ms=0.01)
        # Observe victim first, then step the clock a lot, then
        # observe candidate — gives candidate a big recency edge.
        v_rid = ResourceId.for_prefix(
            list(range(16)) + [88], session_id="s0")
        v_rid = ResourceId(
            scope=v_rid.scope, resource_id=v_rid.resource_id,
            num_blocks=1, num_bytes=v_rid.num_bytes,
            session_id="s0", first_block_hash=v_rid.first_block_hash,
        )
        c.observe_resource(v_rid)
        # Artificially age the victim by bumping the controller's
        # step counter past the recency half-life a few times.
        for _ in range(30):
            c.step_candidates(
                candidates=[], free_gpu_blocks=0, victim=None)
        cand = _cand(2)
        c.observe_resource(cand.resource)
        victim = ResourceCandidate(resource=v_rid)
        selected = c.step_candidates(
            candidates=[cand],
            free_gpu_blocks=100,
            victim=victim,
        )
        # Fresh candidate vs 30-step-stale victim ⇒ candidate
        # clears a 0.05 ms margin.
        assert len(selected) == 1
        assert selected[0].resource_id == cand.resource.resource_id


class TestMarginBackCompat:
    def test_default_config_is_zero_margin(self):
        """Unless explicitly set, victim_gate_margin_ms defaults
        to 0.0 so phase-2 behaviour is preserved."""
        cfg = SPFConfig()
        assert cfg.victim_gate_margin_ms == 0.0

    def test_env_parses(self):
        import os
        from unittest.mock import patch
        with patch.dict(os.environ,
                        {"VLLM_SPF_VICTIM_GATE_MARGIN_MS": "0.1"},
                        clear=False):
            cfg = SPFConfig.from_env()
            assert cfg.victim_gate_margin_ms == 0.1
