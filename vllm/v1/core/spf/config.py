# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SPF (Speculative Prefetch) configuration.

All SPF settings are driven by environment variables with the
``VLLM_SPF_`` prefix, keeping them fully orthogonal to other
subsystems (e.g. fused quantization uses VLLM_FUSED_INT4_*).

Mode separation (phase-1 honest rewrite):

  - ``retention``: anti-eviction on the GPU prefix cache. Uses the
    existing block_pool / touch path. No external KV movement.
  - ``prefetch``: real movement of external KV before request
    arrival (LMCache disk→CPU, CPU→GPU, or equivalent async
    promotion). A "prefetch" is only counted if bytes started moving
    before the request arrived.

``retention`` is the default because the current in-tree integration
path is GPU-resident retention; prefetch mode requires LMCache
wiring that lands in a later phase.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


# Canonical mode names. Any other value from VLLM_SPF_MODE is
# rejected at config load time rather than silently defaulting —
# silent fallbacks would make A/B experiments meaningless.
MODE_RETENTION = "retention"
MODE_PREFETCH = "prefetch"
VALID_MODES = (MODE_RETENTION, MODE_PREFETCH)


@dataclass(frozen=True)
class SPFConfig:
    """Immutable configuration for the SPF controller."""

    # Master switch: set VLLM_SPF_ENABLED=1 to activate.
    enabled: bool = False

    # Operating mode — "retention" or "prefetch". See module
    # docstring. Every log line and metric snapshot includes this so
    # experiment runs are never ambiguous about which path produced
    # the numbers.
    mode: str = MODE_RETENTION

    # Scoring policy. Phase-2 recommends "expected_utility" as the
    # default for honest-mode experiments; "session_aware" is kept
    # for A/B baselines, and "learned" for users with trained
    # weights. Invalid values raise at construction.
    scorer: str = "session_aware"

    # Cooldown window: once a resource has been hinted, it is
    # suppressed from re-hinting for this many scheduler steps.
    # 0 disables cooldown entirely.
    cooldown_steps: int = 3

    # Whether to update the transition table as requests are
    # observed. Off by default so existing A/B tests don't pick
    # up transition-aware ranking implicitly; turn on when using
    # the expected-utility scorer.
    enable_transitions: bool = False

    # Victim gate margin in ms. The phase-2 gate suppressed a
    # hint iff candidate_utility <= victim_utility. At tight
    # cache pressure candidate and victim utilities both collapse
    # to small, noisy values and a strict > test fires too often,
    # evicting blocks more valuable than the candidate. The
    # margin requires the candidate to exceed the victim by at
    # least this many ms before the hint is allowed. 0.0 keeps
    # phase-2 behaviour; 0.05 ms (~2.5× the default
    # scheduler_overhead_ms) is recommended for pressured regimes.
    victim_gate_margin_ms: float = 0.0

    # Phase-7b policy-arm ablation knobs. Setting these
    # disables individual features of the utility scorer so we
    # can isolate which parts of the policy (recency, frequency,
    # transitions, size penalty) actually drive any measured
    # effect. Used by the 4-arm matrix (plain_lru / recency_only
    # / locality_heuristic / full_spf).
    disable_frequency: bool = False
    disable_size_penalty: bool = False

    # Hard cap on hint blocks per scheduler step. Applies to both
    # modes — in retention it's the budget for how many blocks we
    # ask the block_pool to protect per step; in prefetch it caps
    # the bytes_promoted-per-step.
    max_hint_blocks: int = 64

    # Fraction of free GPU blocks available for hints (0.0–1.0).
    hint_fraction: float = 0.10

    # Lookahead horizon (in scheduler steps) for candidate generation.
    lookahead_steps: int = 4

    # How often (in scheduler steps) to emit per-step metric summaries.
    metrics_interval: int = 50

    # Path to learned scorer weights (JSON). Ignored for session_aware.
    learned_weights_path: str = ""

    # Default K to request from BlockManifestProviders. Phase-1
    # vanilla SPF does NOT consult priors during scoring; this is
    # only used by legacy / future code paths that still carry the
    # provider interface. See AGENTS discussion about keeping
    # interfaces compatible without letting priors influence.
    target_K: int = 8

    # ---- Legacy alias fields (deprecated, do not use in new code)
    #
    # Real dataclass fields rather than properties so that a caller
    # can construct ``SPFConfig(max_prefetch_blocks=64, ...)`` and
    # have it reconcile in __post_init__. A handful of older tests
    # still pass these names. New code should use ``max_hint_blocks``
    # and ``hint_fraction``.
    max_prefetch_blocks: Optional[int] = None
    prefetch_fraction: Optional[float] = None

    def __post_init__(self):
        # Reconcile legacy aliases: if a caller passed the old
        # name, mirror it onto the canonical field. If both are
        # set with different values, the new-name field wins (and a
        # warning is not raised — the frozen dataclass makes
        # precedence deterministic and that is sufficient).
        if (self.max_prefetch_blocks is not None
                and self.max_prefetch_blocks != self.max_hint_blocks):
            object.__setattr__(
                self, "max_hint_blocks", self.max_prefetch_blocks)
        # After reconciliation, the alias *is* the canonical value
        # so reads from either name return the same thing.
        object.__setattr__(
            self, "max_prefetch_blocks", self.max_hint_blocks)

        if (self.prefetch_fraction is not None
                and self.prefetch_fraction != self.hint_fraction):
            object.__setattr__(
                self, "hint_fraction", self.prefetch_fraction)
        object.__setattr__(
            self, "prefetch_fraction", self.hint_fraction)

        # Validate mode once at construction so misconfigured
        # callers fail fast — we do this in __post_init__ (not
        # from_env) so direct constructors are also covered.
        if self.mode not in VALID_MODES:
            raise ValueError(
                f"SPFConfig.mode={self.mode!r} is not one of "
                f"{VALID_MODES}"
            )

    @classmethod
    def from_env(cls) -> "SPFConfig":
        """Build config from VLLM_SPF_* environment variables."""
        mode = os.environ.get("VLLM_SPF_MODE", MODE_RETENTION).lower()
        if mode not in VALID_MODES:
            raise ValueError(
                f"VLLM_SPF_MODE={mode!r} is not one of "
                f"{VALID_MODES}. Refusing to silently fall back so "
                f"A/B experiments stay meaningful."
            )

        # max_hint_blocks accepts either the new or legacy env var
        # name; the new name wins when both are set.
        max_hint = int(
            os.environ.get(
                "VLLM_SPF_MAX_HINT_BLOCKS",
                os.environ.get("VLLM_SPF_MAX_PREFETCH_BLOCKS", "64"),
            )
        )
        hint_frac = float(
            os.environ.get(
                "VLLM_SPF_HINT_FRACTION",
                os.environ.get("VLLM_SPF_PREFETCH_FRACTION", "0.10"),
            )
        )

        return cls(
            enabled=os.environ.get("VLLM_SPF_ENABLED", "0") == "1",
            mode=mode,
            scorer=os.environ.get("VLLM_SPF_SCORER", "session_aware"),
            max_hint_blocks=max_hint,
            hint_fraction=hint_frac,
            lookahead_steps=int(
                os.environ.get("VLLM_SPF_LOOKAHEAD_STEPS", "4")
            ),
            metrics_interval=int(
                os.environ.get("VLLM_SPF_METRICS_INTERVAL", "50")
            ),
            learned_weights_path=os.environ.get(
                "VLLM_SPF_LEARNED_WEIGHTS", ""
            ),
            target_K=int(os.environ.get("VLLM_SPF_TARGET_K", "8")),
            cooldown_steps=int(
                os.environ.get("VLLM_SPF_COOLDOWN_STEPS", "3")
            ),
            enable_transitions=(
                os.environ.get("VLLM_SPF_ENABLE_TRANSITIONS", "0")
                == "1"
            ),
            victim_gate_margin_ms=float(
                os.environ.get("VLLM_SPF_VICTIM_GATE_MARGIN_MS",
                               "0.0")
            ),
            disable_frequency=(
                os.environ.get("VLLM_SPF_DISABLE_FREQUENCY", "0")
                == "1"
            ),
            disable_size_penalty=(
                os.environ.get("VLLM_SPF_DISABLE_SIZE_PENALTY", "0")
                == "1"
            ),
        )
