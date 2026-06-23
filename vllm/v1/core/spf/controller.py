# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SPF Controller: the scheduler-side prefetch decision engine.

Responsibilities:
  1. Observe request arrival patterns (session IDs, prefix hashes).
  2. Generate candidate blocks for prefetch (from LMCache residency).
  3. Score candidates using the configured scorer.
  4. Gate prefetch within the block budget.
  5. Emit prefetch hints that flow through the existing KVConnector.

The controller is instantiated once per scheduler and called once per
scheduling round. It does NOT touch KV cache tensors, attention
backends, or the forward path — keeping it fully orthogonal to fused
quantization and other kernel-level work.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from vllm.v1.core.spf.config import SPFConfig
from vllm.v1.core.spf.manifest import (
    BlockManifest,
    BlockManifestProvider,
    NullManifestProvider,
)
from vllm.v1.core.spf.metrics import SPFMetrics, SPFStepMetrics
from vllm.v1.core.spf.resource import ResourceCandidate, ResourceId
from vllm.v1.core.spf.scorer import (
    CandidateFeatures,
    CooldownTracker,
    ExpectedUtilityScorer,
    LearnedScorer,
    Scorer,
    SessionAwareScorer,
)
from vllm.v1.core.spf.transitions import TransitionTable
from vllm.v1.core.spf.utility import UtilityConstants, UtilityFeatures

logger = logging.getLogger("vllm.spf")


@dataclass
class PrefetchCandidate:
    """A single candidate block for speculative prefetch."""

    # Token-prefix hash (matches LMCache key format).
    prefix_hash: str
    # Session ID that last accessed this prefix.
    session_id: str
    # Computed score from the scorer.
    score: float = 0.0
    # Number of KV cache blocks this prefix occupies (the unbounded
    # "N" cost — what SPF would have prefetched without KRI).
    num_blocks: int = 1
    # Optional opaque query hash for the request that produced this
    # candidate.  Computed at the request-ingestion boundary, NOT in
    # the scoring loop.  KRI-G providers ignore it; KRI-Q providers
    # require it.  ``None`` means "no query region was extracted from
    # this request" — providers that need a query will decline to
    # serve such candidates (manifest miss).
    query_hash: str | None = None


@dataclass
class PrefetchHint:
    """Output of the controller: a block to prefetch.

    These flow into the KVConnector as speculative load_specs.

    When ``manifest`` is set, the prefetch is bounded to exactly the
    blocks named in the manifest (the KRI K-slice).  When ``manifest``
    is ``None``, the prefetch covers the full ``num_blocks`` of the
    underlying resource — legacy behavior preserved.
    """

    prefix_hash: str
    session_id: str
    num_blocks: int
    manifest: BlockManifest | None = None


@dataclass
class SessionState:
    """Lightweight per-session bookkeeping (causal only)."""

    session_id: str
    # Prefix hashes seen from this session, most recent last.
    prefix_history: list[str] = field(default_factory=list)
    # Most recent query_hash observed for each prefix in this session.
    # Used so KRI-Q candidate generation can carry the right query
    # forward to the provider.  KRI-G ignores this field.
    prefix_query_hash: dict[str, str | None] = field(default_factory=dict)
    # Last step this session was active.
    last_active_step: int = 0
    # Total requests from this session.
    request_count: int = 0
    # Number of consecutive steps this session has been active.
    consecutive_steps: int = 0


class SPFController:
    """Scheduler-side speculative prefetch controller.

    Lifecycle:
        controller = SPFController(config)
        # Each scheduler round:
        hints = controller.step(
            arriving_sessions=[...],
            free_gpu_blocks=N,
        )
        # Feed hints into KVConnector load_spec generation.
    """

    def __init__(
        self,
        config: SPFConfig,
        manifest_provider: BlockManifestProvider | None = None,
    ):
        self._config = config
        self._scorer = self._build_scorer(config)
        self._metrics = SPFMetrics(
            _mode=config.mode,
            _interval=config.metrics_interval,
        )
        self._step_count = 0

        # Resource-level lifecycle tracking (phase-1 honest mode).
        # These map resource_id -> wall-time seconds so outcome
        # tracking can compute arrival/first-token/finish latencies.
        # They are entirely independent of the legacy prefix_hash
        # bookkeeping below and can be populated without any prior
        # observe_request() call.
        self._request_arrival: dict[str, float] = {}
        self._request_first_token: dict[str, float] = {}
        # resource_ids for which we have issued a hint and are still
        # awaiting an outcome (used or wasted). Used when a
        # matching request lands or when the resource is evicted.
        self._hinted_resources: set[str] = set()

        # Phase-2 additions: transition model (off by default),
        # utility-aware scorer, cooldown tracker. Phase-4 wires
        # them together in step_candidates().
        self._transitions: TransitionTable = TransitionTable()
        self._utility_scorer: ExpectedUtilityScorer = ExpectedUtilityScorer(
            mode=config.mode,
            consts=UtilityConstants.from_env(),
        )
        self._cooldown: CooldownTracker = CooldownTracker(
            period_steps=config.cooldown_steps)
        # Manifest provider is the seam to KRI / future KRI-x variants.
        # Default NullManifestProvider keeps SPF callsites unconditional —
        # no ``provider is not None`` checks scattered through the
        # scoring loop.  Real providers can be installed via
        # :meth:`set_manifest_provider`.
        self._manifest_provider: BlockManifestProvider = (
            manifest_provider if manifest_provider is not None
            else NullManifestProvider()
        )

        # Session state tracking (causal — only past observations).
        self._sessions: dict[str, SessionState] = {}

        # Prefix access tracking for feature computation.
        # prefix_hash -> (last_access_step, total_accesses)
        self._prefix_access: dict[str, tuple[int, int]] = {}

        # Block-to-session ownership: prefix_hash -> session_id that
        # last accessed it.  Used to score eviction victims from their
        # real session context instead of treating them as ownerless.
        self._block_session: dict[str, str] = {}

        # Multi-session access tracking: prefix_hash -> set of session
        # IDs that have accessed this block.  Blocks accessed by many
        # sessions (e.g. shared system prompts) receive a victim-score
        # bonus that makes them harder to displace via prefetch.
        self._block_sessions: dict[str, set[str]] = {}

        logger.info(
            "SPF controller initialized: mode=%s scorer=%s "
            "max_hint_blocks=%d hint_fraction=%.2f lookahead=%d",
            config.mode,
            config.scorer,
            config.max_hint_blocks,
            config.hint_fraction,
            config.lookahead_steps,
        )

    @staticmethod
    def _build_scorer(config: SPFConfig) -> Scorer:
        if config.scorer == "learned" and config.learned_weights_path:
            return LearnedScorer(config.learned_weights_path)
        return SessionAwareScorer()

    def set_manifest_provider(
        self, provider: BlockManifestProvider
    ) -> None:
        """Install or replace the active manifest provider.

        Used by the scheduler bridge once a real provider (e.g. a KRI
        prior loader, a RAG-KRI registry) becomes available after
        controller construction.  Calling with
        :class:`NullManifestProvider` cleanly disables KRI integration
        without tearing down the controller.
        """
        self._manifest_provider = provider

    def observe_request(
        self,
        session_id: str,
        prefix_hash: str,
        query_hash: str | None = None,
    ) -> None:
        """Record an arriving request (called before step()).

        This updates the causal state that the scorer uses.

        ``query_hash`` is an opaque tag for the query region of the
        prompt, computed at the request-ingestion boundary.  SPF
        passes it through to manifest providers verbatim — KRI-G
        providers ignore it, KRI-Q providers require it.  ``None`` is
        valid: it means "no query region was extracted" and any
        provider that requires a query will treat such a request as a
        manifest miss.
        """
        state = self._sessions.get(session_id)
        if state is None:
            state = SessionState(session_id=session_id)
            self._sessions[session_id] = state

        # Track consecutive activity for burst detection.
        if state.last_active_step == self._step_count:
            # Multiple blocks observed in the same step — no change.
            pass
        elif state.last_active_step == self._step_count - 1:
            state.consecutive_steps += 1
        else:
            state.consecutive_steps = 1

        state.last_active_step = self._step_count
        state.request_count += 1
        if not state.prefix_history or state.prefix_history[-1] != prefix_hash:
            state.prefix_history.append(prefix_hash)
        # Always update the latest query_hash for this prefix.  KRI-Q
        # cares about the most recent query a session ran against this
        # prefix; older queries are not useful as candidate features.
        state.prefix_query_hash[prefix_hash] = query_hash

        prev = self._prefix_access.get(prefix_hash, (0, 0))
        self._prefix_access[prefix_hash] = (self._step_count, prev[1] + 1)
        self._block_session[prefix_hash] = session_id
        if prefix_hash not in self._block_sessions:
            self._block_sessions[prefix_hash] = set()
        self._block_sessions[prefix_hash].add(session_id)

    def step(
        self,
        free_gpu_blocks: int,
        resident_prefixes: set[str] | None = None,
        gpu_lru_victim: str | None = None,
    ) -> list[PrefetchHint]:
        """Run one prefetch decision round.

        Args:
            free_gpu_blocks: Number of free GPU KV cache blocks.
            resident_prefixes: Set of prefix hashes currently resident
                in LMCache (if available). When None, candidate
                generation is skipped (no-op step).
            gpu_lru_victim: Prefix hash of the GPU cache's LRU tail
                block (the block that would be evicted next). When
                provided, the controller gates prefetch on the
                candidate scoring higher than the victim — preventing
                harmful evictions that displace hotter blocks.

        Returns:
            List of PrefetchHint objects to be issued as speculative
            load_specs through the KVConnector.
        """
        self._step_count += 1
        step_metrics = SPFStepMetrics()

        # Compute budget.
        budget = min(
            self._config.max_prefetch_blocks,
            int(free_gpu_blocks * self._config.prefetch_fraction),
        )
        step_metrics.budget_blocks = budget

        if budget <= 0 or resident_prefixes is None:
            self._metrics.record_step(step_metrics)
            return []

        # Generate and score candidates.
        candidates = self._generate_candidates(resident_prefixes)
        step_metrics.candidates_scored = len(candidates)

        for candidate in candidates:
            features = self._extract_features(candidate)
            candidate.score = self._scorer.score(features)

        # Sort by score descending, take top-K within budget.
        candidates.sort(key=lambda c: c.score, reverse=True)

        # Eviction-penalty gate: when the GPU cache is full and we
        # know which block would be evicted, score the victim using
        # its real session ownership.  Only prefetch if the candidate
        # genuinely outscores the victim.  This prevents the
        # shared_prefix regression where a lower-value candidate
        # displaces a hotter block the next request needs.
        victim_score = 0.0
        if gpu_lru_victim is not None:
            victim_score = self._score_block(gpu_lru_victim)

        hints: list[PrefetchHint] = []
        blocks_used = 0
        for candidate in candidates:
            # Resolve manifest first, because manifest.K is the
            # effective block cost — without this lookup the budget
            # arithmetic still uses the unbounded N from the candidate
            # and KRI's win is invisible.
            manifest = self._lookup_manifest(candidate)
            if manifest is not None:
                effective_blocks = manifest.K
                step_metrics.kri_manifest_hits += 1
                step_metrics.kri_blocks_saved += manifest.savings_blocks
            else:
                effective_blocks = candidate.num_blocks
                step_metrics.kri_manifest_misses += 1

            if blocks_used + effective_blocks > budget:
                continue
            # Gate: skip if candidate does not outscore the victim.
            if gpu_lru_victim is not None and candidate.score <= victim_score:
                logger.debug(
                    "SPF eviction gate: candidate %s (%.4f) <= "
                    "victim %s (%.4f), skipping",
                    candidate.prefix_hash[:8],
                    candidate.score,
                    gpu_lru_victim[:8],
                    victim_score,
                )
                continue
            hints.append(
                PrefetchHint(
                    prefix_hash=candidate.prefix_hash,
                    session_id=candidate.session_id,
                    num_blocks=effective_blocks,
                    manifest=manifest,
                )
            )
            blocks_used += effective_blocks
            self._metrics.track_outstanding(candidate.prefix_hash)

        step_metrics.prefetch_issued = len(hints)
        self._metrics.record_step(step_metrics)
        return hints

    def _lookup_manifest(
        self, candidate: PrefetchCandidate
    ) -> BlockManifest | None:
        """Ask the active provider for a manifest for this candidate.

        Catches the loud-failure path: if a provider raises ValueError
        because it required ``query_hash`` and the candidate did not
        carry one, that is a real bug to surface — log it and treat
        as a manifest miss for this candidate.  Other exceptions are
        re-raised because they signal real provider failures we want
        to debug, not silently absorb.
        """
        try:
            return self._manifest_provider.get_manifest(
                prefix_hash=candidate.prefix_hash,
                query_hash=candidate.query_hash,
                K=self._config.target_K,
            )
        except ValueError as exc:
            # KRI-Q-style providers raise ValueError on missing
            # query_hash.  This is loud-by-design: surface it once per
            # offending candidate so test runs and production logs both
            # show the integration bug, then fall through to legacy
            # full-prefetch behavior.
            logger.warning(
                "SPF manifest provider declined candidate %s "
                "(query_hash=%s): %s",
                candidate.prefix_hash[:8],
                candidate.query_hash,
                exc,
            )
            return None

    def _generate_candidates(
        self, resident_prefixes: set[str]
    ) -> list[PrefetchCandidate]:
        """Generate prefetch candidates from recently active sessions.

        Only considers prefixes that:
          1. Are resident in LMCache (confirmed by lookup).
          2. Belong to sessions active within the lookahead window.
          3. Pass the stale-session and self-prefetch guards.

        Two guards prevent "cache pollution" where SPF evicts hot
        blocks from the current session's working set:

        * Self-prefetch guard: the most recently active session's own
          already-accessed prefixes are excluded.  If LRU evicted them
          while the session was active, they are genuinely cold — re-
          promoting them would undo the LRU's correct eviction.

        * Stale-session guard: non-current sessions must have been
          active within the last 2 steps.  This prevents a completed
          burst's blocks from flooding the GPU cache and evicting the
          new session's hot working set (the batched_burst regression).
          A 2-step window still captures interleaving patterns
          (conversation_tree, mixed_session) where sessions alternate.
        """
        candidates: list[PrefetchCandidate] = []
        lookahead_cutoff = self._step_count - self._config.lookahead_steps

        # Identify the most recently active session(s).
        # observe_request() records last_active_step = _step_count
        # at call time, and step() increments _step_count before
        # calling this method, so prev_step is the observation step.
        prev_step = self._step_count - 1
        current_sessions: set[str] = set()
        for sid, state in self._sessions.items():
            if state.last_active_step >= prev_step:
                current_sessions.add(sid)

        for session_id, state in self._sessions.items():
            if state.last_active_step < lookahead_cutoff:
                continue

            is_current = session_id in current_sessions

            # Self-prefetch guard for burst sessions: if the
            # current session has been active for 3+ consecutive
            # steps, it's in "burst mode" — its own evicted blocks
            # were evicted by LRU for good reason and should not
            # be re-promoted.  For interleaving patterns (1-2
            # consecutive steps) self-prefetch is allowed because
            # the session may return and need those blocks.
            if is_current:
                cur_state = self._sessions.get(session_id)
                if cur_state and cur_state.consecutive_steps >= 3:
                    continue

            # Stale-session guard (2-step window).
            if (
                current_sessions
                and not is_current
                and state.last_active_step < prev_step - 1
            ):
                continue

            for prefix_hash in state.prefix_history:
                if prefix_hash in resident_prefixes:
                    candidates.append(
                        PrefetchCandidate(
                            prefix_hash=prefix_hash,
                            session_id=session_id,
                            query_hash=state.prefix_query_hash.get(
                                prefix_hash
                            ),
                        )
                    )
        return candidates

    def _score_block(self, prefix_hash: str) -> float:
        """Score an arbitrary block using its real session ownership.

        Used to evaluate the likely eviction victim so the controller
        can compare candidate benefit against victim cost.

        Blocks accessed by multiple sessions (shared prefixes, common
        system prompts) receive a multiplicative bonus via sqrt(N) where
        N is the number of unique sessions.  This prevents high-sharing
        blocks from being displaced by single-session candidates.
        """
        last_step, total_accesses = self._prefix_access.get(
            prefix_hash, (0, 0)
        )
        recency = float(self._step_count - last_step)
        victim_sid = self._block_session.get(prefix_hash)
        victim_sess = self._sessions.get(victim_sid) if victim_sid else None
        sess_freq = float(victim_sess.request_count) if victim_sess else 0.0
        depth = float(len(victim_sess.prefix_history)) if victim_sess else 0.0
        features = CandidateFeatures(
            recency=recency,
            log_frequency=math.log1p(total_accesses),
            session_frequency=sess_freq,
            prefix_depth=depth,
            last_access_gap=recency,
        )
        base_score = self._scorer.score(features)

        # Multi-session sharing bonus: blocks accessed by N unique
        # sessions get a sqrt(N) multiplier, making shared blocks
        # progressively harder to displace.
        n_sessions = len(self._block_sessions.get(prefix_hash, set()))
        if n_sessions > 1:
            base_score *= math.sqrt(n_sessions)

        return base_score

    def _extract_features(
        self, candidate: PrefetchCandidate
    ) -> CandidateFeatures:
        """Build causal feature vector for a candidate."""
        last_step, total_accesses = self._prefix_access.get(
            candidate.prefix_hash, (0, 0)
        )
        session_state = self._sessions.get(candidate.session_id)
        session_freq = session_state.request_count if session_state else 0
        prefix_depth = float(
            len(session_state.prefix_history) if session_state else 0
        )

        return CandidateFeatures(
            recency=float(self._step_count - last_step),
            log_frequency=math.log1p(total_accesses),
            session_frequency=float(session_freq),
            prefix_depth=prefix_depth,
            last_access_gap=float(self._step_count - last_step),
        )

    def report_outcome(
        self, prefix_hash: str, *, hit: bool
    ) -> None:
        """Report whether a prefetched block was actually used."""
        self._metrics.record_prefetch_outcome(prefix_hash, hit=hit)

    # ------------------------------------------------------------------
    # Phase-1 honest-mode lifecycle hooks (ResourceId-based)
    # ------------------------------------------------------------------
    #
    # These exist alongside the legacy prefix_hash-keyed API so
    # integration code written for the old surface keeps working
    # while new code paths can use the honest one. They never
    # consult routing priors (KRI, cartridge, etc.); those can
    # decorate candidates later without changing this contract.

    @property
    def mode(self) -> str:
        """Operating mode in force for this controller."""
        return self._config.mode

    def observe_resource(
        self,
        resource: ResourceId,
        query_hash: str | None = None,
    ) -> None:
        """Record an arriving request, resource-based API.

        Thin adapter on top of :meth:`observe_request` that preserves
        the legacy causal-state updates (prefix history, per-session
        bookkeeping) while the controller's public surface shifts
        from "bare prefix_hash" to "ResourceId". The identity field
        passed to the legacy code is ``resource.resource_id``, NOT
        ``resource.first_block_hash``.
        """
        self.observe_request(
            session_id=resource.session_id,
            prefix_hash=resource.resource_id,
            query_hash=query_hash,
        )

    def record_request_arrival(
        self,
        resource: ResourceId,
        arrival_time: float,
    ) -> None:
        """Note the wall-time at which a request for ``resource`` arrived.

        In prefetch mode this is what promotion latency is measured
        against: a prefetch counts only if it started moving bytes
        before this timestamp. In retention mode it's used to
        compute whether a hinted resource survived long enough to
        serve the request.
        """
        self._request_arrival[resource.resource_id] = arrival_time

    def record_first_token(
        self,
        resource: ResourceId,
        first_token_time: float,
    ) -> None:
        """Note first-token wall-time for a request."""
        self._request_first_token[resource.resource_id] = first_token_time

    def record_request_finished(
        self,
        resource: ResourceId,
    ) -> None:
        """Record that a request using ``resource`` has completed.

        If we had issued a hint for this resource and are still
        awaiting an outcome, the hint is counted as *used*
        (retention: ``hints_used``; prefetch: ``prefetch_used``).
        Lifecycle timers are cleared.
        """
        rid = resource.resource_id
        if rid in self._hinted_resources:
            self._metrics.record_hint_used(rid)
            self._hinted_resources.discard(rid)
        self._request_arrival.pop(rid, None)
        self._request_first_token.pop(rid, None)

    def record_hint_wasted(self, resource_id: str) -> None:
        """Resource was evicted before a request used it."""
        if resource_id in self._hinted_resources:
            self._metrics.record_hint_wasted(resource_id)
            self._hinted_resources.discard(resource_id)

    def record_harmful_eviction_avoided(self) -> None:
        """Victim gate saved us from displacing something valuable.

        Controller-level convenience for scheduler-integration code
        that recognises the situation; the phase-2 expected-utility
        ranker will invoke this automatically from the gate path.
        """
        self._metrics.record_harmful_eviction_avoided()

    def mark_hint_issued(self, resource_id: str) -> None:
        """Track that we emitted a hint for ``resource_id``.

        Called by the scheduler-integration layer (or tests) after
        a hint actually flows into the block_pool (retention) or
        LMCache (prefetch). Separate from ``step()`` because a hint
        may be suppressed post-scoring.
        """
        self._hinted_resources.add(resource_id)
        self._metrics.record_hint_issued(resource_id)

    @property
    def metrics_snapshot(self) -> dict:
        """Mode-tagged metrics dict — same shape across retention
        and prefetch paths so result files are uniform.
        """
        return self._metrics.snapshot()

    # ------------------------------------------------------------------
    # Phase-4: utility-aware candidate selection
    # ------------------------------------------------------------------

    @property
    def transitions(self) -> TransitionTable:
        """Read-only access to the transition table.

        Enabled tests and integration glue use this to inspect
        causally-updated counts. If ``config.enable_transitions``
        is False, the table is still accessible but will have
        been populated only by explicit record_transition() calls,
        not by observe_resource().
        """
        return self._transitions

    @property
    def cooldown(self) -> CooldownTracker:
        """Read-only access to the cooldown tracker."""
        return self._cooldown

    @property
    def utility_scorer(self) -> ExpectedUtilityScorer:
        return self._utility_scorer

    def observe_resource_with_transitions(
        self,
        resource: ResourceId,
        query_hash: str | None = None,
    ) -> None:
        """Like :meth:`observe_resource`, but ALSO updates the
        transition table causally (session's previous prefix →
        this one).

        Separate method from observe_resource so existing call
        sites don't pick up transition-aware behaviour implicitly.
        Controlled by ``config.enable_transitions``: if False,
        this method falls back to plain observe_resource and the
        transition table stays stable.
        """
        self.observe_resource(resource, query_hash=query_hash)
        if self._config.enable_transitions:
            self._transitions.record_access(
                session_id=resource.session_id,
                prefix=resource.resource_id,
            )

    def step_candidates(
        self,
        candidates: list[ResourceCandidate],
        free_gpu_blocks: int,
        victim: ResourceCandidate | None = None,
    ) -> list[ResourceCandidate]:
        """Utility-aware candidate selection.

        Takes a list of ``ResourceCandidate`` objects (produced by
        whatever mechanism the caller prefers — typically derived
        from session histories + LMCache residency), scores each
        via the expected-utility formula, gates against the
        optional ``victim`` candidate, applies cooldown suppression,
        and returns the ordered list of candidates to hint this
        step. Callers then pass each returned candidate to the
        mode-appropriate integration (retention or prefetch).

        Unlike the legacy :meth:`step` this path does NOT consult
        any routing priors — phase-1 rule "vanilla SPF only". It
        does not emit ``PrefetchHint`` objects either; hints are
        ``ResourceCandidate`` values, one per resource to apply.

        The step counter is incremented by this call so cooldown
        is tick-correct even when the legacy ``step()`` is not
        being used.
        """
        self._step_count += 1
        step_metrics = SPFStepMetrics(mode=self._config.mode)
        step_metrics.candidates_scored = len(candidates)

        budget = min(
            self._config.max_hint_blocks,
            int(free_gpu_blocks * self._config.hint_fraction),
        )
        step_metrics.budget_blocks = budget
        if budget <= 0 or not candidates:
            self._metrics.record_step(step_metrics)
            return []

        # Build victim utility once; reused for every candidate.
        victim_util_ms = 0.0
        victim_bytes = 0
        if victim is not None:
            vf = self._utility_features(victim)
            victim_util_ms = self._utility_scorer.score(vf)
            victim_bytes = victim.num_bytes

        # Score every candidate; collect those that pass the
        # strict-greater-than-victim gate AND are not in cooldown.
        scored: list[tuple[float, ResourceCandidate]] = []
        for cand in candidates:
            if self._cooldown.should_suppress(
                cand.resource_id, self._step_count,
            ):
                step_metrics.hints_suppressed += 1
                continue

            uf = self._utility_features(
                cand, victim_saved_ms=victim_util_ms,
                victim_num_bytes=victim_bytes,
            )
            total = self._utility_scorer.score(uf)
            cand.score = total

            # Margin-based victim gate (phase 6): require the
            # candidate to exceed the victim by at least
            # ``victim_gate_margin_ms``. margin=0 reproduces the
            # phase-2 strict-> behaviour. margin > 0 makes the
            # gate more conservative at tight pressure where
            # candidate/victim utilities both collapse to small,
            # noisy values.
            if (victim is not None
                    and total <= victim_util_ms
                    + self._config.victim_gate_margin_ms):
                step_metrics.hints_suppressed += 1
                if self._config.mode == "retention":
                    self._metrics.record_harmful_eviction_avoided()
                continue
            scored.append((total, cand))

        # Rank by utility descending, take top-K within budget.
        scored.sort(key=lambda t: t[0], reverse=True)

        selected: list[ResourceCandidate] = []
        blocks_used = 0
        for total, cand in scored:
            if total <= 0:
                # Nothing with non-positive utility is worth the
                # per-hint overhead.
                break
            if blocks_used + cand.num_blocks > budget:
                continue
            selected.append(cand)
            blocks_used += cand.num_blocks
            self._cooldown.mark_issued(
                cand.resource_id, self._step_count)

        # Counter plumbing: the number we actually selected is the
        # "hints_issued" contribution for the step. Integration
        # layer calls mark_hint_issued per candidate when it
        # actually applies the hint; that's what flips the
        # outstanding tracker. The counter here gives A/B dashboards
        # a step-level view.
        if self._config.mode == "retention":
            step_metrics.retention.hints_issued = len(selected)
        else:
            step_metrics.prefetch.hints_issued = len(selected)

        self._metrics.record_step(step_metrics)
        return selected

    def _utility_features(
        self,
        cand: ResourceCandidate,
        victim_saved_ms: float = 0.0,
        victim_num_bytes: int = 0,
    ) -> UtilityFeatures:
        """Materialise utility features for a candidate.

        Draws from session state (for recency / frequency /
        consecutive), the access log (for log_frequency), and the
        transition table (for transition_prob / count). Reuse
        distance is a crude "how many distinct prefixes in the
        session's recent history since this one was touched"
        count.
        """
        session_state = self._sessions.get(cand.session_id)
        # Recency: scheduler steps since last access.
        last_step, total_accesses = self._prefix_access.get(
            cand.resource_id, (0, 0))
        recency = float(max(0, self._step_count - last_step))
        session_freq = (
            float(session_state.request_count) if session_state else 0.0)
        consecutive = (
            session_state.consecutive_steps if session_state else 0)

        # Transition features: look at the session's current
        # prefix and ask the table for (current → candidate).
        current_prefix = self._transitions.current_prefix(
            cand.session_id)
        if current_prefix is not None:
            tp = self._transitions.probability(
                cand.session_id, current_prefix, cand.resource_id)
            tc = self._transitions.count(
                cand.session_id, current_prefix, cand.resource_id)
        else:
            tp, tc = 0.0, 0

        # Reuse distance: count distinct prefixes in session
        # history after the last access of this candidate.
        reuse_distance = 0.0
        if session_state is not None:
            hist = session_state.prefix_history
            if cand.resource_id in hist:
                idx = (len(hist) - 1
                       - hist[::-1].index(cand.resource_id))
                tail = hist[idx + 1:]
                reuse_distance = float(len(set(tail)))

        # Phase-7b ablation knobs:
        #   disable_frequency zeros the log_frequency and
        #     session_frequency contributions (recency_only arm).
        #   disable_size_penalty zeros num_bytes so the wasted-
        #     bytes term drops out of the utility formula.
        lf = float(total_accesses)
        sf = session_freq
        nb = cand.num_bytes
        if self._config.disable_frequency:
            lf = 0.0
            sf = 0.0
        if self._config.disable_size_penalty:
            nb = 0

        return UtilityFeatures(
            recency=recency,
            log_frequency=lf,
            session_frequency=sf,
            prefix_depth=float(
                len(session_state.prefix_history)
                if session_state else 0),
            last_access_gap=recency,
            reuse_distance=reuse_distance,
            transition_prob=tp,
            transition_count=tc,
            consecutive_steps=consecutive,
            num_bytes=nb,
            num_blocks=cand.num_blocks,
            victim_saved_ms=victim_saved_ms,
            victim_num_bytes=victim_num_bytes,
        )
