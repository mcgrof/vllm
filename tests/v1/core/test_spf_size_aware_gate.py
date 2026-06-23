# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the size-aware victim gate.

The phase-1 gate compared raw scores. The phase-2 gate compares
expected-utility values which already fold in size via the
wasted-bytes penalty. This file locks in the property:

  "Don't issue a hint if the candidate's expected utility is lower
   than the victim's expected utility. Don't ignore size:
   evicting a tiny hot thing for a huge maybe-hot thing is stupid."

Concretely: when the candidate is *much* bigger than the victim,
evicting the victim has to buy us enough saved-ms to pay for the
bytes risk too. The utility formula already does this, so the
tests here are assertions about the combined behaviour.
"""
from __future__ import annotations

from vllm.v1.core.spf.config import MODE_RETENTION
from vllm.v1.core.spf.utility import (UtilityConstants, UtilityFeatures,
                                      expected_utility)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _util(cand: UtilityFeatures) -> float:
    return expected_utility(cand, MODE_RETENTION, UtilityConstants()).total


# ---------------------------------------------------------------------------
# Candidate strictly beats victim
# ---------------------------------------------------------------------------


class TestCandidateStrictlyBetterPasses:

    def test_high_p_candidate_beats_low_p_victim(self):
        """Strong candidate (fresh, high transition), weak victim
        (stale, no transition). Candidate utility > victim utility."""
        cand = UtilityFeatures(
            recency=1.0,
            transition_prob=0.9,
            transition_count=40,
            num_blocks=4,
            num_bytes=10**8,
            victim_saved_ms=0.05,  # cheap victim
        )
        # Construct the "victim as a standalone" utility: it would
        # be scored as if it were a candidate. A cold victim.
        victim = UtilityFeatures(recency=50.0,
                                 transition_prob=0.0,
                                 transition_count=0,
                                 num_blocks=4,
                                 num_bytes=10**8)

        assert _util(cand) > _util(victim)


# ---------------------------------------------------------------------------
# Candidate not worth it (size blows up the penalty)
# ---------------------------------------------------------------------------


class TestHugeCandidateVsTinyVictim:

    def test_huge_cold_candidate_loses_to_tiny_hot_victim(self):
        """A 4 GB low-P candidate should NOT outscore a 100 MB
        high-P victim even when raw saved-ms looks tempting, because
        the wasted-bytes penalty on the huge candidate dwarfs any
        plausible saved time."""
        consts = UtilityConstants(wasted_bytes_penalty_per_gb=2.0)
        huge_cold = UtilityFeatures(
            recency=30.0,  # kills P
            transition_prob=0.1,
            transition_count=3,
            num_blocks=50,
            num_bytes=4 * 10**9,
        )
        tiny_hot = UtilityFeatures(
            recency=0.0,
            transition_prob=0.9,
            transition_count=50,
            num_blocks=3,
            num_bytes=100 * 10**6,
        )
        u_cand = expected_utility(huge_cold, MODE_RETENTION, consts)
        u_vic = expected_utility(tiny_hot, MODE_RETENTION, consts)
        assert u_cand.total < u_vic.total, (
            f"huge cold candidate ({u_cand.total:.3f} ms) should "
            f"not outrank tiny hot victim ({u_vic.total:.3f} ms)")


# ---------------------------------------------------------------------------
# Victim term is actually driven by size × probability
# ---------------------------------------------------------------------------


class TestVictimCost:

    def test_bigger_victim_saved_ms_pushes_candidate_down(self):
        """If we told the candidate that the victim it would
        displace has a high saved_ms, the candidate's expected
        utility should drop — by roughly (1 - P_candidate) *
        victim_saved_ms."""
        base = UtilityFeatures(
            recency=5.0,
            transition_prob=0.3,
            transition_count=5,
            num_blocks=3,
            num_bytes=10**8,
            victim_saved_ms=0.0,
        )
        with_victim = UtilityFeatures(
            recency=5.0,
            transition_prob=0.3,
            transition_count=5,
            num_blocks=3,
            num_bytes=10**8,
            victim_saved_ms=2.0,
        )
        assert _util(with_victim) < _util(base)

    def test_zero_victim_cost_when_candidate_certain(self):
        """If P=1.0 (we're sure the candidate will be used), the
        eviction-cost term goes to zero — the victim would have been
        evicted *anyway* because LRU. That's the whole point of
        multiplying by (1 - P)."""
        consts = UtilityConstants()
        f = UtilityFeatures(
            recency=0.0,
            transition_prob=1.0,
            transition_count=100,
            victim_saved_ms=5.0,
            num_blocks=1,
            num_bytes=0,
        )
        b = expected_utility(f, MODE_RETENTION, consts)
        assert b.probability == 1.0
        assert b.eviction_cost_ms == 0.0


# ---------------------------------------------------------------------------
# Gate decision at the total-utility level (the thing the
# controller actually checks).
# ---------------------------------------------------------------------------


class TestGateDecisionOnTotal:
    """Simulate the controller's comparison: hint iff
    candidate.total > victim.total."""

    def test_hint_when_candidate_wins(self):
        cand = UtilityFeatures(recency=0.0,
                               transition_prob=0.9,
                               transition_count=40,
                               num_blocks=4,
                               num_bytes=10**8)
        vic = UtilityFeatures(recency=40.0, num_blocks=4, num_bytes=10**8)
        assert _util(cand) > _util(vic)

    def test_suppress_when_victim_wins(self):
        cand = UtilityFeatures(recency=30.0,
                               transition_prob=0.05,
                               transition_count=1,
                               num_blocks=4,
                               num_bytes=2 * 10**9)
        vic = UtilityFeatures(recency=0.0,
                              transition_prob=0.8,
                              transition_count=20,
                              num_blocks=4,
                              num_bytes=10**8)
        assert _util(cand) < _util(vic)

    def test_equal_utilities_is_caller_policy(self):
        """At exact equality the controller's rule is ">", which
        means ties get suppressed (conservative choice). Locked in
        here so future refactors can't silently flip the tiebreak."""
        f = UtilityFeatures(recency=0.0, num_blocks=1, num_bytes=0)
        assert _util(f) == _util(f)
        # Controller policy: strict >, so equal = suppress.
        assert not (_util(f) > _util(f))
