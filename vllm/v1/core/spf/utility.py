# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expected-utility ranking for SPF candidates.

Replaces the old score-only ranking (linear-weighted-sum via
:class:`SessionAwareScorer`) with:

  expected_utility =
        P(use_before_eviction) * estimated_saved_ms
      - estimated_eviction_cost_ms
      - wasted_bytes_penalty
      - scheduler_overhead_penalty

Phase-2 rule: **simple heuristic estimates first**. No neural
scorer, no learned probabilities, no per-workload tuning. The
heuristic is documented inline so future phases can replace it with
a principled estimator and the diff will be obvious.

P(use_before_eviction) is a bounded [0, 1] mix of:

  * Transition probability (strongest signal when the transition
    table has data for this (session, current_prefix → candidate)).
  * Recency (recent use predicts imminent reuse).
  * Frequency (pattern of reuse).
  * Consecutive-session activity (hot sessions keep being hot).

The weights are the kind of round numbers you'd pick on a
whiteboard — the point isn't to optimise here, it's to be
defensibly better than "just sort by a linear score and pray."

Saved-ms and eviction-cost-ms are per-block estimates multiplied
by ``num_blocks``. The retention-mode saved cost is the avoided
CPU → GPU refetch; the prefetch-mode saved cost is the avoided
disk/remote → GPU promotion on the critical path. Both are crude
defaults; operators who measure their system can override via
env vars (see :class:`UtilityConstants`).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

from vllm.v1.core.spf.config import MODE_PREFETCH, MODE_RETENTION


# ---------------------------------------------------------------------------
# Constants — defensible defaults, overridable via env for calibration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UtilityConstants:
    """Per-mode calibration constants for the utility formula.

    All times are in milliseconds. Byte penalties are in ms per
    gigabyte of wasted residency (i.e. "it costs this much expected
    latency elsewhere to squat on this many wasted GPU bytes").
    """

    # Saved latency per block when the hint pays off.
    saved_ms_per_block_retention: float = 0.08
    saved_ms_per_block_prefetch: float = 2.0

    # Wasted bytes penalty (per GB retained without being used).
    wasted_bytes_penalty_per_gb: float = 0.5

    # Fixed per-hint scheduler cost (issuing, tracking outstanding,
    # cleaning up). Small but non-zero so utility has a real hurdle.
    scheduler_overhead_ms: float = 0.02

    # Recency-to-probability curve: a recency of this many steps
    # lands at P=0.5 for the recency term alone (before mixing
    # with transitions / frequency). 3 steps ≈ "one scheduler
    # round's worth of warmup".
    recency_half_life_steps: float = 3.0

    # Upper cap on the probability contribution from each single
    # feature. Prevents any one feature from saturating the P
    # estimate. 0.5 means no single term can exceed half of P.
    per_feature_cap: float = 0.5

    @classmethod
    def from_env(cls) -> "UtilityConstants":
        def _fget(name: str, default: float) -> float:
            raw = os.environ.get(name)
            if raw is None:
                return default
            try:
                return float(raw)
            except ValueError:
                return default
        return cls(
            saved_ms_per_block_retention=_fget(
                "VLLM_SPF_SAVED_MS_RETENTION", 0.08),
            saved_ms_per_block_prefetch=_fget(
                "VLLM_SPF_SAVED_MS_PREFETCH", 2.0),
            wasted_bytes_penalty_per_gb=_fget(
                "VLLM_SPF_WASTED_BYTES_PENALTY_PER_GB", 0.5),
            scheduler_overhead_ms=_fget(
                "VLLM_SPF_SCHEDULER_OVERHEAD_MS", 0.02),
            recency_half_life_steps=_fget(
                "VLLM_SPF_RECENCY_HALF_LIFE_STEPS", 3.0),
            per_feature_cap=_fget(
                "VLLM_SPF_PER_FEATURE_CAP", 0.5),
        )


# ---------------------------------------------------------------------------
# Utility features
# ---------------------------------------------------------------------------

@dataclass
class UtilityFeatures:
    """All the causal features the utility formula needs.

    Kept separate from :class:`CandidateFeatures` so the old
    linear-weighted-sum scorer keeps working unchanged for the one
    A/B test that still uses it, and so adding a feature here
    doesn't require editing that scorer.
    """

    # From the old scorer, unchanged semantics.
    recency: float = 0.0
    log_frequency: float = 0.0
    session_frequency: float = 0.0
    prefix_depth: float = 0.0
    last_access_gap: float = 0.0

    # Phase-2 additions.
    #
    # Reuse distance: approximate number of distinct prefixes the
    # session touched between the last access of this candidate and
    # the present. High reuse_distance ⇒ low P(use); low ⇒ high P.
    reuse_distance: float = 0.0

    # Transition probability: P(next prefix == candidate |
    # session's current prefix), from TransitionTable. 0.0 means
    # we have no evidence either way.
    transition_prob: float = 0.0

    # Transition count: raw count for robustness weighting. A
    # transition probability of 0.5 from 2 observations is weaker
    # evidence than 0.5 from 200.
    transition_count: int = 0

    # Consecutive scheduler steps the session has been active.
    consecutive_steps: int = 0

    # Size of the candidate resource in bytes. Used in the wasted-
    # bytes penalty — bigger hints risk more.
    num_bytes: int = 0

    # Number of blocks the candidate covers. Multiplies saved-ms.
    num_blocks: int = 1

    # Cost multiplier for the eviction victim, if any. 0.0 means
    # no victim would be displaced (or the gate didn't identify
    # one). Higher means the victim is more valuable.
    victim_saved_ms: float = 0.0

    # Victim resource size in bytes (for size-aware gate).
    victim_num_bytes: int = 0


# ---------------------------------------------------------------------------
# Probability heuristic
# ---------------------------------------------------------------------------

def probability_of_use(
    f: UtilityFeatures,
    consts: UtilityConstants,
) -> float:
    """Estimate P(candidate used before eviction).

    Bounded in [0, 1]. The formula is deliberately simple; every
    term is clamped by ``per_feature_cap`` so no single feature can
    saturate the estimate.

    Contribution sketch (values picked to sum to ~1.0 in the
    "all strong signals" limit after clamping):

      * transition_prob:    up to ~0.5 (the strongest signal)
      * recency:            up to ~0.5 (via 1/(1+r/half_life))
      * log_frequency:      up to ~0.25 (sqrt-ish)
      * consecutive_steps:  up to ~0.1

    Reuse distance subtracts from all of that — if the session has
    touched many other prefixes since this candidate was last
    relevant, the candidate is more likely cold.
    """
    cap = consts.per_feature_cap

    # Transition signal (strongest when present). Weight by log of
    # count so 50/100 is more credible than 5/10.
    if f.transition_prob > 0.0 and f.transition_count > 0:
        robustness = math.log1p(f.transition_count) / math.log1p(50.0)
        robustness = min(1.0, robustness)
        t_contrib = min(cap, f.transition_prob * robustness)
    else:
        t_contrib = 0.0

    # Recency: e^(-recency/half_life)-ish but simpler.
    half_life = max(1.0, consts.recency_half_life_steps)
    r_contrib = min(cap, 1.0 / (1.0 + f.recency / half_life))

    # Frequency: log1p bounded to ~0.25 at 50 accesses.
    f_contrib = min(
        cap * 0.5,
        math.log1p(max(0.0, f.log_frequency)) / math.log1p(50.0),
    )

    # Consecutive activity: small capped bonus.
    c_contrib = min(cap * 0.2, f.consecutive_steps / 15.0)

    # Reuse-distance penalty: linear, capped.
    reuse_penalty = min(cap, f.reuse_distance / 20.0)

    p = t_contrib + r_contrib + f_contrib + c_contrib - reuse_penalty
    # Final clamp to [0, 1].
    return max(0.0, min(1.0, p))


# ---------------------------------------------------------------------------
# Expected utility
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UtilityBreakdown:
    """Audit-friendly breakdown of a single utility calculation.

    Returned alongside the final utility value so tests and logs
    can inspect each term without recomputing anything.
    """

    probability: float
    saved_ms: float
    eviction_cost_ms: float
    wasted_bytes_penalty_ms: float
    scheduler_overhead_ms: float
    total: float


def expected_utility(
    features: UtilityFeatures,
    mode: str,
    consts: UtilityConstants,
) -> UtilityBreakdown:
    """Compute expected utility for a candidate.

    Mode selects which per-block saved-ms constant to use. The
    result is returned as a :class:`UtilityBreakdown` so callers can
    see each term — that's what makes the ranking defensible: a
    "why did you not hint this candidate?" question has a concrete
    numerical answer.
    """
    if mode == MODE_RETENTION:
        saved_per_block = consts.saved_ms_per_block_retention
    elif mode == MODE_PREFETCH:
        saved_per_block = consts.saved_ms_per_block_prefetch
    else:
        # Frozen dataclass constants never reach this branch, but
        # be explicit so a future mode addition doesn't silently
        # inherit retention pricing.
        raise ValueError(f"unknown mode: {mode!r}")

    p = probability_of_use(features, consts)
    saved_ms = p * saved_per_block * max(1, features.num_blocks)

    # If this hint would displace a victim, the cost scales by the
    # probability the victim would otherwise have been used. We
    # don't know it exactly, so use (1 - P_candidate) as a proxy —
    # if we're highly confident about the candidate, the victim's
    # opportunity cost is what matters; if we're uncertain, we're
    # risking eviction without a good reason either way.
    eviction_cost_ms = (1.0 - p) * features.victim_saved_ms

    # Wasted-bytes: proportional to bytes we'd hold for a hint
    # that never pays off. 1 GB held for nothing costs the
    # configured number of ms elsewhere. Keeps the formula
    # insensitive to absolute byte counts as long as the ratio is
    # right.
    wasted_bytes_gb = features.num_bytes / 1e9
    wasted_bytes_penalty_ms = (
        (1.0 - p) * wasted_bytes_gb
        * consts.wasted_bytes_penalty_per_gb
    )

    overhead = consts.scheduler_overhead_ms

    total = (
        saved_ms
        - eviction_cost_ms
        - wasted_bytes_penalty_ms
        - overhead
    )
    return UtilityBreakdown(
        probability=p,
        saved_ms=saved_ms,
        eviction_cost_ms=eviction_cost_ms,
        wasted_bytes_penalty_ms=wasted_bytes_penalty_ms,
        scheduler_overhead_ms=overhead,
        total=total,
    )
