# SPDX-License-Identifier: Apache-2.0
"""Tests for the expected-utility scorer.

Covers the contract from ``utility.py`` and ``scorer.py``:

  - probability_of_use returns values in [0, 1].
  - Each single feature's contribution is capped (no feature
    saturates the probability alone).
  - Transition + recency together push probability high.
  - Reuse distance penalises.
  - expected_utility decomposes into the four documented terms.
  - Mode selects the per-block saved-ms constant.
  - Ordering property: stronger features ⇒ higher total utility.
  - Size-aware victim term: a bigger victim = larger eviction cost.
  - ExpectedUtilityScorer wraps the function consistently with
    score_with_breakdown().
"""
from __future__ import annotations

import math

import pytest

from vllm.v1.core.spf.config import MODE_PREFETCH, MODE_RETENTION
from vllm.v1.core.spf.scorer import ExpectedUtilityScorer
from vllm.v1.core.spf.utility import (
    UtilityConstants,
    UtilityFeatures,
    expected_utility,
    probability_of_use,
)


# ---------------------------------------------------------------------------
# probability_of_use
# ---------------------------------------------------------------------------

class TestProbabilityBounds:
    def test_default_features_in_unit_interval(self):
        f = UtilityFeatures()
        p = probability_of_use(f, UtilityConstants())
        assert 0.0 <= p <= 1.0

    def test_all_max_features_caps_at_one(self):
        f = UtilityFeatures(
            recency=0.0,
            log_frequency=50.0,
            session_frequency=50.0,
            prefix_depth=10.0,
            last_access_gap=0.0,
            reuse_distance=0.0,
            transition_prob=1.0,
            transition_count=100,
            consecutive_steps=10,
        )
        p = probability_of_use(f, UtilityConstants())
        assert p == 1.0 or (0.99 <= p <= 1.0)

    def test_all_bad_features_hits_zero(self):
        f = UtilityFeatures(
            recency=1000.0,
            log_frequency=0.0,
            transition_prob=0.0,
            transition_count=0,
            consecutive_steps=0,
            reuse_distance=1000.0,
        )
        p = probability_of_use(f, UtilityConstants())
        assert p == 0.0

    def test_transition_prob_alone_capped(self):
        """The transition term cannot exceed per_feature_cap on its
        own — no single feature should saturate probability."""
        consts = UtilityConstants(per_feature_cap=0.5)
        f = UtilityFeatures(
            recency=1000.0,  # kill recency contribution
            transition_prob=1.0,
            transition_count=1000,
        )
        p = probability_of_use(f, consts)
        # recency contributes ~0 at recency=1000/half_life=3 → ~0.003
        # Still bounded by cap from transition alone.
        assert p <= 0.51  # small slack for recency epsilon


class TestProbabilityMonotonicity:
    def test_lower_recency_is_higher_p(self):
        c = UtilityConstants()
        p_old = probability_of_use(
            UtilityFeatures(recency=10.0), c)
        p_new = probability_of_use(
            UtilityFeatures(recency=1.0), c)
        assert p_new > p_old

    def test_higher_transition_is_higher_p(self):
        c = UtilityConstants()
        low = probability_of_use(
            UtilityFeatures(transition_prob=0.1, transition_count=10),
            c)
        hi = probability_of_use(
            UtilityFeatures(transition_prob=0.9, transition_count=10),
            c)
        assert hi > low

    def test_more_transition_counts_is_more_credible(self):
        c = UtilityConstants()
        few = probability_of_use(
            UtilityFeatures(transition_prob=0.5, transition_count=2),
            c)
        many = probability_of_use(
            UtilityFeatures(transition_prob=0.5, transition_count=200),
            c)
        assert many > few

    def test_reuse_distance_reduces_p(self):
        c = UtilityConstants()
        base = probability_of_use(
            UtilityFeatures(recency=1.0, transition_prob=0.5,
                            transition_count=10), c)
        far = probability_of_use(
            UtilityFeatures(recency=1.0, transition_prob=0.5,
                            transition_count=10,
                            reuse_distance=15.0),
            c)
        assert far < base


# ---------------------------------------------------------------------------
# expected_utility breakdown
# ---------------------------------------------------------------------------

class TestUtilityBreakdown:
    def test_retention_saved_ms_scales_with_blocks(self):
        c = UtilityConstants()
        one_block = expected_utility(
            UtilityFeatures(
                recency=0.0, transition_prob=1.0,
                transition_count=100, num_blocks=1),
            MODE_RETENTION, c)
        ten_blocks = expected_utility(
            UtilityFeatures(
                recency=0.0, transition_prob=1.0,
                transition_count=100, num_blocks=10),
            MODE_RETENTION, c)
        assert ten_blocks.saved_ms > one_block.saved_ms * 5

    def test_prefetch_saved_ms_is_larger_than_retention(self):
        """Prefetch avoids a full disk/remote→GPU move; retention
        avoids only a CPU→GPU refetch. Prefetch should have a
        higher per-block saved-ms constant."""
        c = UtilityConstants()
        f = UtilityFeatures(
            recency=0.0, transition_prob=1.0,
            transition_count=100, num_blocks=4)
        r = expected_utility(f, MODE_RETENTION, c)
        p = expected_utility(f, MODE_PREFETCH, c)
        assert p.saved_ms > r.saved_ms

    def test_wasted_bytes_penalty_scales_with_size_and_uncertainty(self):
        c = UtilityConstants(wasted_bytes_penalty_per_gb=1.0)
        # Big, low-probability candidate → high wasted-bytes penalty.
        big_cold = expected_utility(
            UtilityFeatures(
                num_bytes=4 * 10 ** 9,  # 4 GB
                recency=100.0,  # low recency signal
            ),
            MODE_RETENTION, c)
        # Small, same uncertainty → smaller penalty.
        small_cold = expected_utility(
            UtilityFeatures(
                num_bytes=100 * 10 ** 6,  # 100 MB
                recency=100.0,
            ),
            MODE_RETENTION, c)
        assert (big_cold.wasted_bytes_penalty_ms
                > small_cold.wasted_bytes_penalty_ms)

    def test_eviction_cost_scales_with_victim_value(self):
        c = UtilityConstants()
        no_victim = expected_utility(
            UtilityFeatures(
                recency=10.0, victim_saved_ms=0.0),
            MODE_RETENTION, c)
        with_victim = expected_utility(
            UtilityFeatures(
                recency=10.0, victim_saved_ms=5.0),
            MODE_RETENTION, c)
        assert with_victim.eviction_cost_ms > 0.0
        assert no_victim.eviction_cost_ms == 0.0

    def test_scheduler_overhead_is_always_deducted(self):
        """Overhead is always deducted from total. With a very
        cold candidate (high recency, no other signals), saved_ms
        is ~0 and total ≈ -overhead. Uses a big reuse_distance to
        zero the recency contribution completely; otherwise the
        ``1/(1+r/half_life)`` term leaves a small residual."""
        c = UtilityConstants(scheduler_overhead_ms=0.05)
        cold = UtilityFeatures(
            recency=1000.0, reuse_distance=1000.0)
        b = expected_utility(cold, MODE_RETENTION, c)
        assert b.scheduler_overhead_ms == 0.05
        assert b.probability == 0.0
        assert b.saved_ms == 0.0
        # Total equals -(overhead) exactly when no other terms
        # contribute.
        assert abs(b.total - (-0.05)) < 1e-9

    def test_total_equals_breakdown(self):
        c = UtilityConstants()
        f = UtilityFeatures(
            recency=1.0, log_frequency=3.0,
            transition_prob=0.7, transition_count=20,
            num_blocks=4, num_bytes=500 * 10 ** 6,
            victim_saved_ms=0.1)
        b = expected_utility(f, MODE_RETENTION, c)
        assert math.isclose(
            b.total,
            b.saved_ms - b.eviction_cost_ms
            - b.wasted_bytes_penalty_ms - b.scheduler_overhead_ms,
        )


# ---------------------------------------------------------------------------
# Mode selection
# ---------------------------------------------------------------------------

class TestModeSelection:
    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="unknown mode"):
            expected_utility(
                UtilityFeatures(), "cartridge", UtilityConstants())


# ---------------------------------------------------------------------------
# ExpectedUtilityScorer wrapper
# ---------------------------------------------------------------------------

class TestExpectedUtilityScorer:
    def test_score_returns_total(self):
        s = ExpectedUtilityScorer(MODE_RETENTION)
        f = UtilityFeatures(
            recency=0.0, transition_prob=1.0,
            transition_count=100, num_blocks=5)
        total = s.score(f)
        breakdown = s.score_with_breakdown(f)
        assert total == breakdown.total

    def test_ranking_consistency(self):
        """Candidates with strictly stronger signals must rank
        ahead of weaker ones. The utility-aware scorer is the
        production ranker in phase-2; this test is the ordering
        guarantee the controller depends on.
        """
        s = ExpectedUtilityScorer(MODE_RETENTION)
        weak = UtilityFeatures(
            recency=20.0, transition_prob=0.1, transition_count=2,
            num_blocks=4, num_bytes=10 ** 8)
        strong = UtilityFeatures(
            recency=1.0, transition_prob=0.9, transition_count=50,
            num_blocks=4, num_bytes=10 ** 8)
        assert s.score(strong) > s.score(weak)

    def test_mode_accessor(self):
        s_ret = ExpectedUtilityScorer(MODE_RETENTION)
        s_pf = ExpectedUtilityScorer(MODE_PREFETCH)
        assert s_ret.mode == MODE_RETENTION
        assert s_pf.mode == MODE_PREFETCH
