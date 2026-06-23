# SPDX-License-Identifier: Apache-2.0
"""Tests for the SPF transition model.

Covers the contract from ``transitions.py``:

  - Causality: transitions are only recorded after the destination
    is observed — never predict the future.
  - Per-session isolation: two sessions with identical histories
    don't share counts.
  - Probability normalisation: outgoing probs sum to ~1.0 when the
    (session, from) pair has any observations.
  - Unseen transitions return exactly 0.0 (no smoothing).
  - record_access() populates last-prefix and implicit one-step
    transitions.
"""
from __future__ import annotations

import math

from vllm.v1.core.spf.transitions import TransitionTable


# ---------------------------------------------------------------------------
# Causality
# ---------------------------------------------------------------------------

class TestCausality:
    def test_single_access_records_no_transition(self):
        tt = TransitionTable()
        tt.record_access("s", "A")
        # No from_prefix yet → no counts recorded.
        assert tt.count("s", "A", "B") == 0
        assert tt.total_from("s", "A") == 0

    def test_two_accesses_record_one_transition(self):
        tt = TransitionTable()
        tt.record_access("s", "A")
        tt.record_access("s", "B")
        assert tt.count("s", "A", "B") == 1
        assert tt.total_from("s", "A") == 1

    def test_three_step_chain(self):
        tt = TransitionTable()
        for p in ["A", "B", "C"]:
            tt.record_access("s", p)
        assert tt.count("s", "A", "B") == 1
        assert tt.count("s", "B", "C") == 1
        assert tt.count("s", "A", "C") == 0  # not directly observed

    def test_current_prefix_tracks_latest(self):
        tt = TransitionTable()
        assert tt.current_prefix("s") is None
        tt.record_access("s", "X")
        assert tt.current_prefix("s") == "X"
        tt.record_access("s", "Y")
        assert tt.current_prefix("s") == "Y"


# ---------------------------------------------------------------------------
# Per-session isolation
# ---------------------------------------------------------------------------

class TestSessionIsolation:
    def test_two_sessions_do_not_share(self):
        tt = TransitionTable()
        tt.record_access("sA", "X")
        tt.record_access("sA", "Y")
        # session sB has NO observations of X → Y.
        assert tt.count("sB", "X", "Y") == 0
        assert tt.probability("sB", "X", "Y") == 0.0
        # session sA does.
        assert tt.count("sA", "X", "Y") == 1
        assert tt.probability("sA", "X", "Y") == 1.0

    def test_current_prefix_per_session(self):
        tt = TransitionTable()
        tt.record_access("a", "alpha")
        tt.record_access("b", "beta")
        assert tt.current_prefix("a") == "alpha"
        assert tt.current_prefix("b") == "beta"
        tt.record_access("a", "omega")
        assert tt.current_prefix("a") == "omega"
        assert tt.current_prefix("b") == "beta"  # unchanged


# ---------------------------------------------------------------------------
# Probability normalisation
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_single_destination_probability_one(self):
        tt = TransitionTable()
        tt.record_transition("s", "A", "B")
        assert tt.probability("s", "A", "B") == 1.0

    def test_two_equal_destinations(self):
        tt = TransitionTable()
        tt.record_transition("s", "A", "B")
        tt.record_transition("s", "A", "C")
        assert tt.probability("s", "A", "B") == 0.5
        assert tt.probability("s", "A", "C") == 0.5

    def test_weighted_by_count(self):
        tt = TransitionTable()
        # A→B three times, A→C once.
        for _ in range(3):
            tt.record_transition("s", "A", "B")
        tt.record_transition("s", "A", "C")
        assert tt.probability("s", "A", "B") == 0.75
        assert tt.probability("s", "A", "C") == 0.25

    def test_outgoing_distribution_sums_to_one(self):
        tt = TransitionTable()
        for dst in ["P", "Q", "R", "P", "Q", "P"]:
            tt.record_transition("s", "X", dst)
        dist = tt.outgoing("s", "X")
        assert math.isclose(sum(dist.values()), 1.0)
        assert dist["P"] == 3 / 6
        assert dist["Q"] == 2 / 6
        assert dist["R"] == 1 / 6

    def test_outgoing_empty_when_no_observations(self):
        tt = TransitionTable()
        assert tt.outgoing("s", "X") == {}


# ---------------------------------------------------------------------------
# Unseen / no-smoothing
# ---------------------------------------------------------------------------

class TestNoSmoothing:
    def test_unseen_transition_returns_zero(self):
        tt = TransitionTable()
        tt.record_transition("s", "A", "B")
        assert tt.probability("s", "A", "NEVER") == 0.0

    def test_unseen_from_prefix_returns_zero(self):
        tt = TransitionTable()
        assert tt.probability("s", "unknown_from", "unknown_to") == 0.0


# ---------------------------------------------------------------------------
# record_access self-loops and reset
# ---------------------------------------------------------------------------

class TestAccessSemantics:
    def test_self_loop_is_recorded(self):
        """Two consecutive accesses to the same prefix record a
        self-transition. Consistent with how session_history treats
        repeated prefixes."""
        tt = TransitionTable()
        tt.record_access("s", "A")
        tt.record_access("s", "A")
        assert tt.count("s", "A", "A") == 1

    def test_reset_clears_everything(self):
        tt = TransitionTable()
        tt.record_access("s", "A")
        tt.record_access("s", "B")
        tt.reset()
        assert tt.probability("s", "A", "B") == 0.0
        assert tt.current_prefix("s") is None
        assert tt.outgoing("s", "A") == {}


# ---------------------------------------------------------------------------
# Contract used by the scorer: given session's current_prefix,
# get the outgoing distribution and the counts for robustness.
# ---------------------------------------------------------------------------

class TestScorerContract:
    def test_scorer_can_read_probability_and_count(self):
        tt = TransitionTable()
        for _ in range(10):
            tt.record_transition("s", "CUR", "HOT")
        for _ in range(2):
            tt.record_transition("s", "CUR", "WARM")
        # Scorer asks: given session s is at CUR, how likely is HOT next?
        assert tt.probability("s", "CUR", "HOT") == 10 / 12
        assert tt.count("s", "CUR", "HOT") == 10
        # And for WARM:
        assert tt.probability("s", "CUR", "WARM") == 2 / 12
        assert tt.count("s", "CUR", "WARM") == 2
        assert tt.total_from("s", "CUR") == 12
