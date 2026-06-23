# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SPF metrics collection and logging.

Mode-separated counters so log lines and results files match what
the code actually did:

  - ``RetentionMetrics`` (mode=retention) — GPU prefix-cache
    anti-eviction. Counts hints issued and whether each hinted
    resource survived to be used.
  - ``PrefetchMetrics`` (mode=prefetch) — real external KV movement.
    A hint only counts as "prefetch_used" if bytes started moving
    before the request arrived.

Legacy counters (``prefetch_issued/prefetch_hit/prefetch_waste`` on
``SPFStepMetrics``) are retained because a handful of integration
tests written against the pre-honest API still read them. New code
paths should go through the mode-specific counters.

No Prometheus dependency at this stage — counters are plain ints
and the periodic summary logs expose them.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger("vllm.spf")

# ---------------------------------------------------------------------------
# Mode-specific counters
# ---------------------------------------------------------------------------


@dataclass
class RetentionMetrics:
    """Counters for retention-mode hints.

    Retention hints ask the GPU prefix-cache (block_pool) to keep a
    resource resident. A *used* hint is one whose resource was later
    referenced before being evicted. A *wasted* hint is one that
    expired without being used.

    ``harmful_evictions_avoided`` is incremented when the
    size-aware victim gate (see controller) rejected a hint because
    issuing it would have evicted something more valuable — that's a
    positive outcome (we avoided doing damage), so it's tracked
    separately from ``hints_suppressed``.
    """

    hints_issued: int = 0
    hints_retained: int = 0
    hints_used: int = 0
    hints_wasted: int = 0
    harmful_evictions_avoided: int = 0


@dataclass
class PrefetchMetrics:
    """Counters for prefetch-mode hints.

    A prefetch hint is counted as ``prefetch_used`` only if bytes
    actually started moving **before the request arrived**. Hints
    whose promotion completes after the request is already waiting
    are tracked via ``request_wait_on_promotion_count`` and counted
    as late (useful for telling "we were too slow" apart from
    "we prefetched the wrong thing").
    """

    hints_issued: int = 0
    bytes_promoted: int = 0
    promote_latency_ms: float = 0.0
    prefetch_used: int = 0
    prefetch_wasted: int = 0
    request_wait_on_promotion_count: int = 0


# ---------------------------------------------------------------------------
# Per-step snapshot
# ---------------------------------------------------------------------------


@dataclass
class SPFStepMetrics:
    """Metrics snapshot for a single scheduler step.

    ``mode`` identifies which family of counters (retention or
    prefetch) is authoritative for this step. The other family's
    counters stay at zero.

    ``kri_*`` fields are legacy — kept on the struct so older KRI
    integration tests don't have to be rewritten, but they are not
    used by vanilla-SPF scoring in phase 1.
    """

    mode: str = "retention"
    candidates_scored: int = 0
    hints_suppressed: int = 0
    budget_blocks: int = 0

    retention: RetentionMetrics = field(default_factory=RetentionMetrics)
    prefetch: PrefetchMetrics = field(default_factory=PrefetchMetrics)

    # Legacy — deprecated. Not filled by vanilla SPF paths.
    kri_manifest_hits: int = 0
    kri_manifest_misses: int = 0
    kri_blocks_saved: int = 0

    # ---- Legacy alias fields (read/write) --------------------------
    #
    # A couple of existing tests spell the counters with the old
    # prefetch_* names at the step level. We route those through
    # into the right mode bucket so neither new code nor legacy
    # tests have to know about the other's spelling.

    @property
    def prefetch_issued(self) -> int:
        """Legacy alias — returns the *mode*'s hints_issued."""
        if self.mode == "retention":
            return self.retention.hints_issued
        return self.prefetch.hints_issued

    @prefetch_issued.setter
    def prefetch_issued(self, v: int) -> None:
        if self.mode == "retention":
            self.retention.hints_issued = v
        else:
            self.prefetch.hints_issued = v

    @property
    def prefetch_hit(self) -> int:
        if self.mode == "retention":
            return self.retention.hints_used
        return self.prefetch.prefetch_used

    @prefetch_hit.setter
    def prefetch_hit(self, v: int) -> None:
        if self.mode == "retention":
            self.retention.hints_used = v
        else:
            self.prefetch.prefetch_used = v

    @property
    def prefetch_waste(self) -> int:
        if self.mode == "retention":
            return self.retention.hints_wasted
        return self.prefetch.prefetch_wasted

    @prefetch_waste.setter
    def prefetch_waste(self, v: int) -> None:
        if self.mode == "retention":
            self.retention.hints_wasted = v
        else:
            self.prefetch.prefetch_wasted = v


# ---------------------------------------------------------------------------
# Accumulator
# ---------------------------------------------------------------------------


@dataclass
class SPFMetrics:
    """Accumulating metrics across scheduler steps, mode-aware."""

    _mode: str = "retention"
    _interval: int = 50
    _step: int = 0

    # Rolling counters since last flush.
    _candidates_scored: int = 0
    _hints_suppressed: int = 0
    _budget_utilization_sum: float = 0.0
    _budget_samples: int = 0

    _retention: RetentionMetrics = field(default_factory=RetentionMetrics)
    _prefetch: PrefetchMetrics = field(default_factory=PrefetchMetrics)

    # Outstanding resource_ids awaiting outcome confirmation
    # (indexed so we can mark them used or wasted when the request
    # eventually lands or the resource is evicted).
    _outstanding: set[str] = field(default_factory=set)

    # Legacy KRI counters (not filled by vanilla SPF paths; kept for
    # existing tests that still read them).
    _kri_manifest_hits: int = 0
    _kri_manifest_misses: int = 0
    _kri_blocks_saved: int = 0

    def set_mode(self, mode: str) -> None:
        """Set the mode at runtime (e.g. when config is parsed)."""
        self._mode = mode

    @property
    def mode(self) -> str:
        return self._mode

    # ---- Legacy private-attribute aliases --------------------------
    #
    # A handful of existing tests reach into SPFMetrics with the old
    # ``_prefetch_issued`` / ``_prefetch_hit`` / ``_prefetch_waste``
    # names. Route those to the mode-correct counters so neither
    # reads nor writes silently diverge from the honest counters.

    @property
    def _prefetch_issued(self) -> int:
        return (self._retention.hints_issued
                if self._mode == "retention" else self._prefetch.hints_issued)

    @_prefetch_issued.setter
    def _prefetch_issued(self, v: int) -> None:
        if self._mode == "retention":
            self._retention.hints_issued = v
        else:
            self._prefetch.hints_issued = v

    @property
    def _prefetch_hit(self) -> int:
        return (self._retention.hints_used
                if self._mode == "retention" else self._prefetch.prefetch_used)

    @_prefetch_hit.setter
    def _prefetch_hit(self, v: int) -> None:
        if self._mode == "retention":
            self._retention.hints_used = v
        else:
            self._prefetch.prefetch_used = v

    @property
    def _prefetch_waste(self) -> int:
        return (self._retention.hints_wasted if self._mode == "retention" else
                self._prefetch.prefetch_wasted)

    @_prefetch_waste.setter
    def _prefetch_waste(self, v: int) -> None:
        if self._mode == "retention":
            self._retention.hints_wasted = v
        else:
            self._prefetch.prefetch_wasted = v

    def snapshot(self) -> dict:
        """Return a flat dict suitable for JSON / log rendering.

        Always includes the mode so consumers never have to guess
        which family of counters is meaningful.
        """
        avg_util = (self._budget_utilization_sum /
                    self._budget_samples if self._budget_samples > 0 else 0.0)
        snap = {
            "mode": self._mode,
            "step": self._step,
            "candidates_scored": self._candidates_scored,
            "hints_suppressed": self._hints_suppressed,
            "avg_budget_util": avg_util,
            "outstanding": len(self._outstanding),
        }
        if self._mode == "retention":
            snap.update({
                "hints_issued":
                self._retention.hints_issued,
                "hints_retained":
                self._retention.hints_retained,
                "hints_used":
                self._retention.hints_used,
                "hints_wasted":
                self._retention.hints_wasted,
                "harmful_evictions_avoided":
                self._retention.harmful_evictions_avoided,
            })
        else:
            snap.update({
                "hints_issued":
                self._prefetch.hints_issued,
                "bytes_promoted":
                self._prefetch.bytes_promoted,
                "promote_latency_ms":
                self._prefetch.promote_latency_ms,
                "prefetch_used":
                self._prefetch.prefetch_used,
                "prefetch_wasted":
                self._prefetch.prefetch_wasted,
                "request_wait_on_promotion_count":
                self._prefetch.request_wait_on_promotion_count,
            })
        # Legacy KRI numbers are always surfaced when non-zero so
        # older experiments that care about them aren't blind.
        if (self._kri_manifest_hits or self._kri_manifest_misses
                or self._kri_blocks_saved):
            snap["_legacy_kri"] = {
                "kri_manifest_hits": self._kri_manifest_hits,
                "kri_manifest_misses": self._kri_manifest_misses,
                "kri_blocks_saved": self._kri_blocks_saved,
            }
        return snap

    def record_step(self, step_metrics: SPFStepMetrics) -> None:
        """Record metrics from one scheduler step."""
        self._step += 1
        # If the step snapshot declares a different mode, it wins —
        # keeps metrics honest under runtime mode changes (rare, but
        # lets tests exercise both modes on one instance without
        # spinning up a fresh accumulator).
        if step_metrics.mode != self._mode:
            self._mode = step_metrics.mode

        self._candidates_scored += step_metrics.candidates_scored
        self._hints_suppressed += step_metrics.hints_suppressed

        if self._mode == "retention":
            r = self._retention
            s = step_metrics.retention
            r.hints_issued += s.hints_issued
            r.hints_retained += s.hints_retained
            r.hints_used += s.hints_used
            r.hints_wasted += s.hints_wasted
            r.harmful_evictions_avoided += s.harmful_evictions_avoided
        else:
            p = self._prefetch
            s = step_metrics.prefetch
            p.hints_issued += s.hints_issued
            p.bytes_promoted += s.bytes_promoted
            p.promote_latency_ms += s.promote_latency_ms
            p.prefetch_used += s.prefetch_used
            p.prefetch_wasted += s.prefetch_wasted
            p.request_wait_on_promotion_count += \
                s.request_wait_on_promotion_count

        self._kri_manifest_hits += step_metrics.kri_manifest_hits
        self._kri_manifest_misses += step_metrics.kri_manifest_misses
        self._kri_blocks_saved += step_metrics.kri_blocks_saved

        if step_metrics.budget_blocks > 0:
            issued = (self._retention.hints_issued if self._mode == "retention"
                      else self._prefetch.hints_issued)
            self._budget_utilization_sum += (step_metrics.candidates_scored /
                                             step_metrics.budget_blocks
                                             if step_metrics.budget_blocks else
                                             0.0)
            self._budget_samples += 1
            del issued  # linter shush

        if self._interval > 0 and self._step % self._interval == 0:
            self._flush()

    # ---- Outcome tracking -----------------------------------------

    def record_hint_issued(self, resource_id: str) -> None:
        """Mark a resource as hinted and awaiting outcome.

        Bumps the mode-appropriate ``hints_issued`` counter AND
        adds the resource_id to the outstanding set. This is the
        honest-mode lifecycle entry point — callers that go through
        :meth:`record_step` must NOT also call this, to avoid
        double-counting.

        The legacy :meth:`track_outstanding` exists for pre-existing
        code paths that count issues through step() and only need
        the outstanding-set tracking.
        """
        self._outstanding.add(resource_id)
        if self._mode == "retention":
            self._retention.hints_issued += 1
        else:
            self._prefetch.hints_issued += 1

    def record_hint_used(self, resource_id: str) -> None:
        """A hinted resource was later consumed by a request.

        In retention mode this increments ``hints_used``. In prefetch
        mode it increments ``prefetch_used``. Callers don't need to
        know which.
        """
        self._outstanding.discard(resource_id)
        if self._mode == "retention":
            self._retention.hints_used += 1
        else:
            self._prefetch.prefetch_used += 1

    def record_hint_wasted(self, resource_id: str) -> None:
        """A hinted resource was evicted without being used."""
        self._outstanding.discard(resource_id)
        if self._mode == "retention":
            self._retention.hints_wasted += 1
        else:
            self._prefetch.prefetch_wasted += 1

    def record_harmful_eviction_avoided(self) -> None:
        """Victim gate prevented a bad eviction (retention mode)."""
        self._retention.harmful_evictions_avoided += 1

    # ---- Legacy name compatibility ---------------------------------
    #
    # A small layer of old tests exercised these. Preserve the names;
    # they route into the mode-correct bucket under the hood.

    def track_outstanding(self, block_hash: str) -> None:
        """Legacy: only add to the outstanding set.

        Intentionally does NOT bump any counter. The pre-existing
        step()-based path in the controller counts issues via the
        SPFStepMetrics accumulator; those callers must continue to
        use this entry point to avoid double-counting. New code
        should use :meth:`record_hint_issued` instead.
        """
        self._outstanding.add(block_hash)

    def record_prefetch_outcome(
        self,
        block_hash: str,
        *,
        hit: bool,
    ) -> None:
        """Legacy alias: route to used/wasted."""
        if hit:
            self.record_hint_used(block_hash)
        else:
            self.record_hint_wasted(block_hash)

    # ---- Flush / log ----------------------------------------------

    def _flush(self) -> None:
        logger.info("SPF %s", self.snapshot())
        # Reset rolling-interval counters.
        self._candidates_scored = 0
        self._hints_suppressed = 0
        self._budget_utilization_sum = 0.0
        self._budget_samples = 0
        self._retention = RetentionMetrics()
        self._prefetch = PrefetchMetrics()
        self._kri_manifest_hits = 0
        self._kri_manifest_misses = 0
        self._kri_blocks_saved = 0
