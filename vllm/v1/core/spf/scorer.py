# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SPF scoring strategies for prefetch candidate ranking.

Three strategies:
  - SessionAwareScorer: zero-parameter linear-weighted-sum
    heuristic. Default in phase 1, retained for A/B baselines.
  - LearnedScorer: logistic reranker with weights loaded from
    a JSON file.
  - ExpectedUtilityScorer (phase 2): ranks by expected utility:
        P(use_before_eviction) * saved_ms
      - eviction_cost_ms
      - wasted_bytes_penalty
      - scheduler_overhead
    The utility-aware scorer consumes :class:`UtilityFeatures` and
    returns the ``total`` field of the breakdown. It is the
    recommended scorer for phase-2 vanilla SPF — simple, no
    neural nonsense, no per-workload tuning.

All scorers expose the same ``score(features)`` contract; only
the feature type differs between the two families.
"""
from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from vllm.v1.core.spf.utility import (UtilityBreakdown, UtilityConstants,
                                      UtilityFeatures, expected_utility)


@dataclass
class CandidateFeatures:
    """Feature vector for a single prefetch candidate block.

    All features must be causal — derived only from information
    available before the prefetch decision is made.
    """

    # Recency: scheduler steps since last access to this prefix.
    recency: float
    # Log of total access frequency for this prefix.
    log_frequency: float
    # Number of accesses from the same session.
    session_frequency: float
    # Depth in the prefix tree (0 = root / system prompt).
    prefix_depth: float
    # Steps since last access from any session.
    last_access_gap: float


class Scorer(ABC):
    """Base class for SPF scorers."""

    @abstractmethod
    def score(self, features: CandidateFeatures) -> float:
        """Return a prefetch priority score (higher = prefetch first)."""
        ...


class SessionAwareScorer(Scorer):
    """Heuristic scorer from P1 results.

    Nearly matches the learned controller on conversation_tree at
    large scale. Recommended as the default.
    """

    def __init__(
        self,
        recency_weight: float = 0.4,
        frequency_weight: float = 0.3,
        session_weight: float = 0.2,
        depth_weight: float = 0.1,
    ):
        self._rw = recency_weight
        self._fw = frequency_weight
        self._sw = session_weight
        self._dw = depth_weight

    def score(self, features: CandidateFeatures) -> float:
        # Recency: prefer recently accessed (invert gap).
        recency_score = 1.0 / (1.0 + features.recency)
        return (self._rw * recency_score + self._fw * features.log_frequency +
                self._sw * features.session_frequency +
                self._dw * features.prefix_depth)


class LearnedScorer(Scorer):
    """Logistic reranker with weights loaded from P1 training.

    Weights file is a JSON dict mapping feature names to floats,
    plus an "intercept" key.
    """

    _FEATURE_ORDER = [
        "recency",
        "log_frequency",
        "session_frequency",
        "prefix_depth",
        "last_access_gap",
    ]

    def __init__(self, weights_path: str):
        with open(weights_path) as f:
            raw = json.load(f)
        self._weights = [raw[k] for k in self._FEATURE_ORDER]
        self._intercept = raw.get("intercept", 0.0)

    def score(self, features: CandidateFeatures) -> float:
        values = [
            features.recency,
            features.log_frequency,
            features.session_frequency,
            features.prefix_depth,
            features.last_access_gap,
        ]
        logit = self._intercept + sum(w * v
                                      for w, v in zip(self._weights, values))
        return 1.0 / (1.0 + math.exp(-logit))


class FrequencyAdmissionScorer(Scorer):
    """Frequency-only admission heuristic (Q4 survival-gate arm).

    A degenerate baseline that ignores recency, session, and depth —
    a block is admitted (high score) iff it has been accessed often.
    Mirrors a classic LFU prefix-cache admission policy. Use as the
    "frequency-aware admission" arm in the Q4 survival gate per
    Codex's plan.
    """

    def score(self, features: CandidateFeatures) -> float:
        return features.log_frequency


class SessionTTLScorer(Scorer):
    """Session-TTL + pinning scorer (Q4 survival-gate baseline).

    Issues a retention hint iff the candidate block belongs to a
    session that was active within the configured TTL window.
    Independent of frequency / transitions / depth. Mirrors the
    classic "pin blocks of recently-active sessions; let everything
    else fall to LRU" baseline that any sophisticated scorer must
    beat by a real margin per the Q4 survival gate.

    The session activity check is encoded in CandidateFeatures via
    last_access_gap (steps since this session last touched any
    prefix). Anything inside the TTL window gets a score of 1.0
    biased by recency (so the freshest sessions outrank stale ones
    within the window); anything outside gets 0.0.
    """

    def __init__(self, ttl_steps: int = 8):
        self._ttl = max(1, int(ttl_steps))

    def score(self, features: CandidateFeatures) -> float:
        if features.last_access_gap >= self._ttl:
            return 0.0
        # Recency tiebreaker so the freshest in-window session wins.
        return 1.0 / (1.0 + features.last_access_gap)


class ReuseDistanceOracleScorer(Scorer):
    """Belady-style oracle scorer (Q4 survival-gate upper bound).

    Reads a pre-computed reuse-distance map (``prefix_hash -> next
    access step index``) loaded from a workload trace. At score()
    time it returns ``1 / (1 + future_distance)`` so the smallest
    future distance dominates. Prefixes that never reappear in the
    future trace score 0.

    The oracle is unimplementable at serve time — it requires
    omniscient future knowledge — but it is the upper bound any
    learned scorer can reach. Use as the ceiling in the Q4 survival
    gate so SPF can be compared against both "what trivial heuristics
    can do" (session-TTL) and "what's left on the table" (oracle).

    Trace format: JSON file
        { "<prefix_hash>": [step_idx_a, step_idx_b, ...], ... }
    where each list is sorted ascending. The scorer maintains the
    current scheduler step internally so it can find the next access
    via bisect.

    Loaded from ``VLLM_SPF_ORACLE_TRACE_PATH`` env var by the
    controller's _build_scorer when ``config.scorer == "oracle"``.
    """

    def __init__(self, trace_path: str):
        import bisect
        with open(trace_path) as f:
            raw = json.load(f)
        self._reuse: dict[str, list[int]] = {
            k: list(v) for k, v in raw.items()
        }
        self._step = 0
        self._bisect = bisect.bisect_right
        # Map prefix_hash -> CandidateFeatures.last_access_gap
        # captured at score time so we can convert "steps since
        # last access" into a current scheduler step.

    def advance_step(self, current_step: int) -> None:
        """Called by the controller at each step() entry."""
        self._step = current_step

    def score(self, features: CandidateFeatures) -> float:
        # The features object doesn't carry a prefix_hash field, so
        # we use a sidecar set_prefix_hash() pattern — the controller
        # is responsible for setting _last_prefix_hash before calling
        # score(). Falls back to 0 if not set.
        ph = getattr(features, "_oracle_prefix_hash", None)
        if ph is None:
            return 0.0
        accesses = self._reuse.get(ph)
        if not accesses:
            return 0.0
        idx = self._bisect(accesses, self._step)
        if idx >= len(accesses):
            return 0.0
        future_distance = accesses[idx] - self._step
        return 1.0 / (1.0 + future_distance)


class ExpectedUtilityScorer:
    """Rank candidates by expected utility (phase-2 default).

    The output of ``score(features)`` is the scalar ``total`` from
    :class:`UtilityBreakdown` — in ms of expected saved time after
    subtracting eviction cost, wasted-bytes penalty, and per-hint
    scheduler overhead. Higher = more utility.

    A ``score_with_breakdown(features)`` variant returns the full
    breakdown for audit / test purposes.

    Note on interface: this scorer consumes :class:`UtilityFeatures`,
    not :class:`CandidateFeatures`, because it needs size, transition,
    and victim fields. The controller populates both feature types
    from the same underlying state.
    """

    def __init__(
        self,
        mode: str,
        consts: UtilityConstants | None = None,
    ):
        self._mode = mode
        self._consts = consts or UtilityConstants()

    @property
    def mode(self) -> str:
        return self._mode

    def score(self, features: UtilityFeatures) -> float:
        return expected_utility(features, self._mode, self._consts).total

    def score_with_breakdown(
        self,
        features: UtilityFeatures,
    ) -> UtilityBreakdown:
        return expected_utility(features, self._mode, self._consts)


# ---------------------------------------------------------------------------
# Cooldown tracker
# ---------------------------------------------------------------------------


@dataclass
class CooldownTracker:
    """Suppresses re-issuing a hint for the same resource.

    Without this, a candidate that stays hot over several scheduler
    steps gets hinted every step, which pumps ``hints_issued`` but
    doesn't actually change what's resident. Cooldown fixes that:
    once we've hinted resource R, we won't re-hint it for the next
    ``period_steps`` scheduler steps even if it still ranks at the
    top of the candidate list.

    Period defaults to 3 steps — long enough that a legitimately
    hot prefix gets re-hinted when it matters, short enough that
    a workload shift isn't blocked by stale cooldowns.

    All state is keyed by ``resource_id`` from the honest identity.
    """

    period_steps: int = 3
    _last_issued: dict = None  # populated in __post_init__

    def __post_init__(self):
        if self._last_issued is None:
            self._last_issued = {}

    def should_suppress(
        self,
        resource_id: str,
        current_step: int,
    ) -> bool:
        """Return True if ``resource_id`` is still in cooldown."""
        last = self._last_issued.get(resource_id)
        if last is None:
            return False
        return (current_step - last) < self.period_steps

    def mark_issued(
        self,
        resource_id: str,
        current_step: int,
    ) -> None:
        """Record that a hint was issued for ``resource_id``."""
        self._last_issued[resource_id] = current_step

    def clear(self, resource_id: str) -> None:
        """Forget the cooldown for a specific resource.

        Used when a hint is confirmed wasted — the caller may want
        to retry immediately rather than wait out cooldown.
        """
        self._last_issued.pop(resource_id, None)

    def reset(self) -> None:
        """Forget all cooldowns. Test helper."""
        self._last_issued.clear()
