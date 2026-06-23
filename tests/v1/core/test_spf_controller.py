# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for SPF controller behavior.

Covers:
  - Candidate filtering (only resident prefixes from active sessions)
  - Score ordering (higher-scored candidates prefetched first)
  - Budget gating (never exceeds block budget)
  - Causal behavior (controller only uses past observations)
  - Metrics tracking
  - Session-aware scorer correctness
  - Learned scorer loading and scoring
"""
from __future__ import annotations

import json
import math
import tempfile

import pytest

from vllm.v1.core.spf.config import SPFConfig
from vllm.v1.core.spf.controller import (
    PrefetchHint,
    SPFController,
)
from vllm.v1.core.spf.scorer import (
    CandidateFeatures,
    LearnedScorer,
    SessionAwareScorer,
)


def _make_config(**overrides) -> SPFConfig:
    """Create a test config with sensible defaults."""
    defaults = dict(
        enabled=True,
        scorer="session_aware",
        max_prefetch_blocks=64,
        prefetch_fraction=0.10,
        lookahead_steps=4,
        metrics_interval=50,
        learned_weights_path="",
    )
    defaults.update(overrides)
    return SPFConfig(**defaults)


def _make_controller(**overrides) -> SPFController:
    return SPFController(_make_config(**overrides))


# ── Candidate Filtering ──────────────────────────────────────────────


class TestCandidateFiltering:
    """Only prefixes that are resident in LMCache AND belong to
    recently active sessions should become candidates."""

    def test_only_resident_prefixes_become_candidates(self):
        ctrl = _make_controller(max_prefetch_blocks=100, prefetch_fraction=1.0)
        ctrl.observe_request("s1", "hash_A")
        ctrl.observe_request("s1", "hash_B")

        # Only hash_A is in LMCache.
        hints = ctrl.step(free_gpu_blocks=100, resident_prefixes={"hash_A"})
        prefetched_hashes = {h.prefix_hash for h in hints}
        assert "hash_A" in prefetched_hashes
        assert "hash_B" not in prefetched_hashes

    def test_no_hints_when_no_resident_prefixes(self):
        ctrl = _make_controller()
        ctrl.observe_request("s1", "hash_A")
        hints = ctrl.step(free_gpu_blocks=100, resident_prefixes=set())
        assert hints == []

    def test_no_hints_when_resident_is_none(self):
        ctrl = _make_controller()
        ctrl.observe_request("s1", "hash_A")
        hints = ctrl.step(free_gpu_blocks=100, resident_prefixes=None)
        assert hints == []

    def test_inactive_sessions_excluded(self):
        """Sessions not active within lookahead_steps are ignored."""
        ctrl = _make_controller(
            lookahead_steps=2,
            max_prefetch_blocks=100,
            prefetch_fraction=1.0,
        )
        ctrl.observe_request("old_session", "hash_old")

        # Advance step_count past the lookahead window.
        for _ in range(4):
            ctrl.step(free_gpu_blocks=100, resident_prefixes=set())

        # Now observe a new session and step again.
        ctrl.observe_request("new_session", "hash_new")
        hints = ctrl.step(
            free_gpu_blocks=100,
            resident_prefixes={"hash_old", "hash_new"},
        )
        prefetched_hashes = {h.prefix_hash for h in hints}
        assert "hash_new" in prefetched_hashes
        # old_session was active at step 0, now at step 5, lookahead=2 → cutoff=3
        assert "hash_old" not in prefetched_hashes

    def test_multiple_sessions_generate_candidates(self):
        ctrl = _make_controller(
            max_prefetch_blocks=100, prefetch_fraction=1.0
        )
        ctrl.observe_request("s1", "hash_A")
        ctrl.observe_request("s2", "hash_B")
        hints = ctrl.step(
            free_gpu_blocks=100,
            resident_prefixes={"hash_A", "hash_B"},
        )
        prefetched_hashes = {h.prefix_hash for h in hints}
        assert "hash_A" in prefetched_hashes
        assert "hash_B" in prefetched_hashes


# ── Score Ordering ────────────────────────────────────────────────────


class TestScoreOrdering:
    """Higher-scored candidates should be prefetched first."""

    def test_more_recent_prefix_scores_higher(self):
        ctrl = _make_controller(
            max_prefetch_blocks=1,  # Only 1 block budget
            prefetch_fraction=1.0,
        )
        # Observe hash_old first, then hash_new.
        ctrl.observe_request("s1", "hash_old")
        # Advance a few steps so hash_old is stale.
        for _ in range(3):
            ctrl.step(free_gpu_blocks=100, resident_prefixes=set())
        ctrl.observe_request("s1", "hash_new")

        hints = ctrl.step(
            free_gpu_blocks=1,
            resident_prefixes={"hash_old", "hash_new"},
        )
        assert len(hints) == 1
        # The more recently observed prefix should win.
        assert hints[0].prefix_hash == "hash_new"

    def test_higher_frequency_prefix_preferred(self):
        ctrl = _make_controller(
            max_prefetch_blocks=1,
            prefetch_fraction=1.0,
        )
        # Access hash_freq many times, hash_rare once.
        for _ in range(10):
            ctrl.observe_request("s1", "hash_freq")
        ctrl.observe_request("s1", "hash_rare")

        hints = ctrl.step(
            free_gpu_blocks=1,
            resident_prefixes={"hash_freq", "hash_rare"},
        )
        assert len(hints) == 1
        assert hints[0].prefix_hash == "hash_freq"


# ── Budget Gating ─────────────────────────────────────────────────────


class TestBudgetGating:
    """Prefetch must never exceed the block budget."""

    def test_respects_max_prefetch_blocks(self):
        ctrl = _make_controller(
            max_prefetch_blocks=2,
            prefetch_fraction=1.0,
        )
        for i in range(5):
            ctrl.observe_request("s1", f"hash_{i}")

        hints = ctrl.step(
            free_gpu_blocks=100,
            resident_prefixes={f"hash_{i}" for i in range(5)},
        )
        total_blocks = sum(h.num_blocks for h in hints)
        assert total_blocks <= 2

    def test_respects_prefetch_fraction(self):
        ctrl = _make_controller(
            max_prefetch_blocks=100,
            prefetch_fraction=0.10,
        )
        for i in range(20):
            ctrl.observe_request("s1", f"hash_{i}")

        # 10 free blocks * 0.10 = 1 block budget.
        hints = ctrl.step(
            free_gpu_blocks=10,
            resident_prefixes={f"hash_{i}" for i in range(20)},
        )
        total_blocks = sum(h.num_blocks for h in hints)
        assert total_blocks <= 1

    def test_zero_free_blocks_yields_no_hints(self):
        ctrl = _make_controller()
        ctrl.observe_request("s1", "hash_A")
        hints = ctrl.step(free_gpu_blocks=0, resident_prefixes={"hash_A"})
        assert hints == []

    def test_budget_is_min_of_max_and_fraction(self):
        # max=3, fraction=1.0, free=10 → budget=3
        ctrl = _make_controller(
            max_prefetch_blocks=3, prefetch_fraction=1.0
        )
        for i in range(10):
            ctrl.observe_request("s1", f"h{i}")
        hints = ctrl.step(
            free_gpu_blocks=10,
            resident_prefixes={f"h{i}" for i in range(10)},
        )
        assert sum(h.num_blocks for h in hints) <= 3


# ── Causal Behavior ──────────────────────────────────────────────────


class TestCausalBehavior:
    """Controller must only use information available before step()."""

    def test_observe_before_step(self):
        """Observations after step() don't affect current hints."""
        ctrl = _make_controller(
            max_prefetch_blocks=100, prefetch_fraction=1.0
        )
        ctrl.observe_request("s1", "hash_A")
        hints_before = ctrl.step(
            free_gpu_blocks=100, resident_prefixes={"hash_A", "hash_B"}
        )
        # hash_B was never observed, so it should not be in hints.
        assert all(h.prefix_hash != "hash_B" for h in hints_before)

        # Now observe hash_B and step again.
        ctrl.observe_request("s1", "hash_B")
        hints_after = ctrl.step(
            free_gpu_blocks=100, resident_prefixes={"hash_A", "hash_B"}
        )
        assert any(h.prefix_hash == "hash_B" for h in hints_after)

    def test_step_count_increments(self):
        ctrl = _make_controller()
        assert ctrl._step_count == 0
        ctrl.step(free_gpu_blocks=100, resident_prefixes=set())
        assert ctrl._step_count == 1
        ctrl.step(free_gpu_blocks=100, resident_prefixes=set())
        assert ctrl._step_count == 2

    def test_session_state_tracks_history(self):
        ctrl = _make_controller()
        ctrl.observe_request("s1", "a")
        ctrl.observe_request("s1", "b")
        ctrl.observe_request("s1", "a")  # duplicate — should not re-append
        state = ctrl._sessions["s1"]
        assert state.prefix_history == ["a", "b", "a"]
        assert state.request_count == 3

    def test_duplicate_consecutive_prefix_not_appended(self):
        ctrl = _make_controller()
        ctrl.observe_request("s1", "a")
        ctrl.observe_request("s1", "a")  # same as last → skip
        state = ctrl._sessions["s1"]
        assert state.prefix_history == ["a"]
        assert state.request_count == 2


# ── Metrics ───────────────────────────────────────────────────────────


class TestMetrics:
    def test_metrics_record_candidates_and_issued(self):
        ctrl = _make_controller(
            max_prefetch_blocks=100, prefetch_fraction=1.0
        )
        ctrl.observe_request("s1", "h1")
        ctrl.observe_request("s1", "h2")
        ctrl.step(
            free_gpu_blocks=100, resident_prefixes={"h1", "h2"}
        )
        assert ctrl._metrics._candidates_scored >= 2
        assert ctrl._metrics._prefetch_issued >= 2

    def test_report_outcome_hit(self):
        ctrl = _make_controller()
        ctrl._metrics.track_outstanding("h1")
        ctrl.report_outcome("h1", hit=True)
        assert ctrl._metrics._prefetch_hit >= 1
        assert "h1" not in ctrl._metrics._outstanding

    def test_report_outcome_waste(self):
        ctrl = _make_controller()
        ctrl._metrics.track_outstanding("h1")
        ctrl.report_outcome("h1", hit=False)
        assert ctrl._metrics._prefetch_waste >= 1


# ── Scorer Unit Tests ─────────────────────────────────────────────────


class TestSessionAwareScorer:
    def test_recent_prefix_scores_higher(self):
        scorer = SessionAwareScorer()
        recent = CandidateFeatures(
            recency=1.0, log_frequency=1.0,
            session_frequency=1.0, prefix_depth=1.0, last_access_gap=1.0,
        )
        stale = CandidateFeatures(
            recency=100.0, log_frequency=1.0,
            session_frequency=1.0, prefix_depth=1.0, last_access_gap=100.0,
        )
        assert scorer.score(recent) > scorer.score(stale)

    def test_frequent_prefix_scores_higher(self):
        scorer = SessionAwareScorer()
        freq = CandidateFeatures(
            recency=1.0, log_frequency=math.log1p(100),
            session_frequency=1.0, prefix_depth=1.0, last_access_gap=1.0,
        )
        rare = CandidateFeatures(
            recency=1.0, log_frequency=math.log1p(1),
            session_frequency=1.0, prefix_depth=1.0, last_access_gap=1.0,
        )
        assert scorer.score(freq) > scorer.score(rare)

    def test_score_is_non_negative(self):
        scorer = SessionAwareScorer()
        features = CandidateFeatures(
            recency=0.0, log_frequency=0.0,
            session_frequency=0.0, prefix_depth=0.0, last_access_gap=0.0,
        )
        assert scorer.score(features) >= 0.0


class TestLearnedScorer:
    def test_loads_and_scores(self):
        weights = {
            "recency": -0.5,
            "log_frequency": 1.0,
            "session_frequency": 0.3,
            "prefix_depth": 0.1,
            "last_access_gap": -0.2,
            "intercept": 0.0,
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(weights, f)
            path = f.name

        scorer = LearnedScorer(path)
        features = CandidateFeatures(
            recency=1.0, log_frequency=2.0,
            session_frequency=3.0, prefix_depth=1.0, last_access_gap=1.0,
        )
        score = scorer.score(features)
        assert 0.0 <= score <= 1.0  # sigmoid output

    def test_sigmoid_bounds(self):
        weights = {
            "recency": 0.0,
            "log_frequency": 0.0,
            "session_frequency": 0.0,
            "prefix_depth": 0.0,
            "last_access_gap": 0.0,
            "intercept": 0.0,
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(weights, f)
            path = f.name

        scorer = LearnedScorer(path)
        features = CandidateFeatures(
            recency=0.0, log_frequency=0.0,
            session_frequency=0.0, prefix_depth=0.0, last_access_gap=0.0,
        )
        # All zeros → sigmoid(0) = 0.5
        assert abs(scorer.score(features) - 0.5) < 1e-6


# ── Config ────────────────────────────────────────────────────────────


class TestConfig:
    def test_from_env_defaults(self, monkeypatch):
        # Clear all SPF env vars.
        for key in list(vars(SPFConfig()).keys()):
            env_key = f"VLLM_SPF_{key.upper()}"
            monkeypatch.delenv(env_key, raising=False)
        monkeypatch.delenv("VLLM_SPF_LEARNED_WEIGHTS", raising=False)

        config = SPFConfig.from_env()
        assert config.enabled is False
        assert config.scorer == "session_aware"
        assert config.max_prefetch_blocks == 64

    def test_from_env_enabled(self, monkeypatch):
        monkeypatch.setenv("VLLM_SPF_ENABLED", "1")
        monkeypatch.setenv("VLLM_SPF_SCORER", "learned")
        monkeypatch.setenv("VLLM_SPF_MAX_PREFETCH_BLOCKS", "32")
        config = SPFConfig.from_env()
        assert config.enabled is True
        assert config.scorer == "learned"
        assert config.max_prefetch_blocks == 32


# ── PrefetchHint Dataclass ────────────────────────────────────────────


class TestPrefetchHint:
    def test_hint_fields(self):
        hint = PrefetchHint(
            prefix_hash="abc", session_id="s1", num_blocks=4
        )
        assert hint.prefix_hash == "abc"
        assert hint.session_id == "s1"
        assert hint.num_blocks == 4


# ── Integration-style test ────────────────────────────────────────────


class TestControllerIntegration:
    """Multi-step scenario simulating a conversation_tree workload."""

    def test_conversation_tree_simulation(self):
        """Simulate a multi-turn conversation tree pattern:
        Session s1: prompt_A → prompt_A+B → prompt_A+C
        Session s2: prompt_D → prompt_D+E
        """
        ctrl = _make_controller(
            max_prefetch_blocks=10,
            prefetch_fraction=0.5,
            lookahead_steps=3,
        )

        # Turn 1: Both sessions start.
        ctrl.observe_request("s1", "prefix_A")
        ctrl.observe_request("s2", "prefix_D")
        hints = ctrl.step(
            free_gpu_blocks=20,
            resident_prefixes={"prefix_A", "prefix_D"},
        )
        assert len(hints) >= 1  # Should suggest prefetching

        # Turn 2: Session s1 extends, session s2 extends.
        ctrl.observe_request("s1", "prefix_AB")
        ctrl.observe_request("s2", "prefix_DE")
        hints = ctrl.step(
            free_gpu_blocks=20,
            resident_prefixes={
                "prefix_A", "prefix_AB",
                "prefix_D", "prefix_DE",
            },
        )
        # Should include candidates from both sessions.
        sessions_in_hints = {h.session_id for h in hints}
        assert len(sessions_in_hints) >= 1

        # Turn 3: Session s1 branches (prefix_AC instead of prefix_AB).
        ctrl.observe_request("s1", "prefix_AC")
        hints = ctrl.step(
            free_gpu_blocks=20,
            resident_prefixes={
                "prefix_A", "prefix_AB", "prefix_AC",
                "prefix_D", "prefix_DE",
            },
        )
        # Budget = min(10, 20*0.5) = 10 blocks.
        assert sum(h.num_blocks for h in hints) <= 10

        # Report outcomes for earlier prefetches.
        for h in hints:
            ctrl.report_outcome(h.prefix_hash, hit=True)
        assert ctrl._metrics._prefetch_hit >= 1
