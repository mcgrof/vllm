# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shadow-baseline counterfactual tracker for SPF retention.

Motivation: live TTFT is noisy at the effect sizes we've measured
in phase 5/6 (sub-percent hit-rate deltas). Even with 5 seeds and
thousands of requests the 95 % CI on p95 TTFT tends to be wider
than the effect itself. We need a direct mechanism-level signal
that SPF actually rescued reuse that plain LRU would have lost.

The trick is simple: while the real GPU cache is running with SPF
retention enabled, we also maintain a **shadow LRU** in parallel.
The shadow sees every access and insertion but does NOT see SPF
touches. On every insertion that causes an eviction, we compare:

  * Did plain LRU evict a different resource than the real cache?
  * Was the real cache's survival due to an SPF hint?
  * If so, did a later request reuse that rescued resource?

That gives us :class:`ShadowMetrics` counters that are robust to
TTFT noise:

  * ``rescued_reuses_count`` — resources SPF kept alive that the
    shadow baseline would have evicted AND a later request
    actually reused.
  * ``rescued_reuses_bytes`` — byte-scaled version.
  * ``harmful_evictions_baseline`` — evictions in the shadow that
    were later reused (what plain LRU got wrong).
  * ``harmful_evictions_spf`` — evictions in the real cache that
    were later reused (what SPF got wrong; should be lower).
  * ``retention_lifetime_ms`` — mean age-at-reuse of rescued
    resources.
  * ``rescued_reuse_distance`` — mean reuse distance, in steps,
    of rescued resources.

This module is **accounting only**. It does not affect serving
behaviour or SPF decisions; it observes them and records
counterfactuals. Enable via :class:`ShadowBaseline` and drive
from :class:`RetentionIntegration` — see ``integrations.py``.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class ShadowMetrics:
    """Counterfactual accounting over the shadow baseline.

    All counters are cumulative since the baseline was created.
    """

    # Core "rescue" counters — the mechanism signal.
    rescued_reuses_count: int = 0
    rescued_reuses_bytes: int = 0

    # Who gets hurt by naive LRU vs SPF.
    harmful_evictions_baseline: int = 0
    harmful_evictions_spf: int = 0

    # How long rescued blocks stayed alive before actually being
    # reused. Stored as a running sum + count so caller can
    # compute mean without storing every sample.
    _retention_lifetime_ms_sum: float = 0.0
    _retention_lifetime_samples: int = 0

    # Steps between hint issue and reuse (reuse distance for the
    # subset of hints that paid off).
    _rescued_reuse_distance_sum: int = 0
    _rescued_reuse_distance_samples: int = 0

    # Bookkeeping.
    shadow_evictions_total: int = 0
    spf_evictions_total: int = 0

    @property
    def retention_lifetime_ms_mean(self) -> float:
        if self._retention_lifetime_samples == 0:
            return 0.0
        return (self._retention_lifetime_ms_sum
                / self._retention_lifetime_samples)

    @property
    def rescued_reuse_distance_mean(self) -> float:
        if self._rescued_reuse_distance_samples == 0:
            return 0.0
        return (self._rescued_reuse_distance_sum
                / self._rescued_reuse_distance_samples)

    def snapshot(self) -> dict:
        return {
            "rescued_reuses_count": self.rescued_reuses_count,
            "rescued_reuses_bytes": self.rescued_reuses_bytes,
            "harmful_evictions_baseline":
                self.harmful_evictions_baseline,
            "harmful_evictions_spf": self.harmful_evictions_spf,
            "retention_lifetime_ms_mean":
                round(self.retention_lifetime_ms_mean, 3),
            "rescued_reuse_distance_mean":
                round(self.rescued_reuse_distance_mean, 2),
            "shadow_evictions_total": self.shadow_evictions_total,
            "spf_evictions_total": self.spf_evictions_total,
            # Derived convenience field: absolute drop in harmful
            # evictions. A positive value means SPF helped; zero
            # or negative means it hurt or was neutral.
            "harmful_eviction_delta": (
                self.harmful_evictions_baseline
                - self.harmful_evictions_spf),
        }


# ---------------------------------------------------------------------------
# Shadow baseline
# ---------------------------------------------------------------------------

@dataclass
class _ShadowEntry:
    """What the shadow LRU knows about one resource."""
    # When this resource entered the shadow pool (wall-time ms).
    inserted_at_ms: float
    # Step at which the most recent access happened. Used for
    # reuse-distance accounting.
    last_access_step: int
    # Size in bytes for byte-weighted metrics.
    num_bytes: int
    # True if this resource would have been evicted by shadow LRU
    # but was kept alive in the real cache by an SPF hint.
    rescued: bool = False
    # Step at which rescue happened (so reuse-distance is
    # observable on reuse).
    rescued_at_step: Optional[int] = None
    # Hint-issue wall-time for retention_lifetime accounting.
    rescued_at_ms: Optional[float] = None


class ShadowBaseline:
    """Parallel plain-LRU view of the cache, for SPF counterfactuals.

    ``ShadowBaseline`` maintains an LRU with the same capacity as
    the real cache, but never sees SPF touches. Callers feed it
    the same access/insert stream the real cache sees; the shadow
    evicts on its own. When the real cache is about to insert an
    item that would force an eviction, SPF may have already
    touched some resource to save it — pass the ``would_save``
    set to :meth:`on_insert` so the shadow can record which ones
    it would have evicted (and therefore would have lost).
    """

    def __init__(
        self,
        capacity: int,
        *,
        now_ms: "callable[[], float] | None" = None,
    ):
        self._capacity = capacity
        self._order: "OrderedDict[str, _ShadowEntry]" = OrderedDict()
        self._now_ms = (
            now_ms if now_ms is not None
            else lambda: time.perf_counter() * 1000.0
        )
        self._step = 0
        self.metrics = ShadowMetrics()

    # ------------------------------------------------------------------
    # Stream events
    # ------------------------------------------------------------------

    def advance_step(self) -> None:
        """Bump the step counter. Call between scheduler rounds so
        reuse-distance accounting reflects real scheduler timing
        rather than per-event order.
        """
        self._step += 1

    def on_access(self, resource_id: str) -> None:
        """Baseline sees an access. If resource is resident, move
        to MRU and update last_access_step. If not, this call is
        a no-op — the shadow records the miss via on_insert.
        """
        entry = self._order.get(resource_id)
        if entry is None:
            return
        self._order.move_to_end(resource_id)
        entry.last_access_step = self._step
        # If the reuse hit a rescued entry, the rescue paid off.
        if entry.rescued:
            self._record_rescue_payoff(entry, resource_id)
            entry.rescued = False
            entry.rescued_at_step = None
            entry.rescued_at_ms = None

    def on_insert(
        self,
        resource_id: str,
        num_bytes: int,
        would_save_ids: "set[str] | None" = None,
    ) -> list[str]:
        """Baseline sees an insertion that may force eviction.

        ``would_save_ids`` is the set of resource_ids that SPF has
        touched in the real cache this round — i.e. resources the
        real cache will NOT evict, even if LRU would have picked
        them. When the shadow's own LRU eviction lands on any of
        these ids, we know SPF's hint saved that id, so we mark
        the entry rescued in the shadow bookkeeping.

        Returns the list of resource_ids the shadow evicted.
        """
        if would_save_ids is None:
            would_save_ids = set()

        if resource_id in self._order:
            self._order.move_to_end(resource_id)
            return []

        evicted: list[str] = []
        now = self._now_ms()
        while len(self._order) >= self._capacity:
            victim_id, victim = next(iter(self._order.items()))
            self._order.pop(victim_id)
            self.metrics.shadow_evictions_total += 1
            evicted.append(victim_id)
            if victim_id in would_save_ids:
                # SPF saved this in the real cache. In the shadow
                # we lose it, and we mark it so that if a reuse
                # later happens on this id in the SHADOW it counts
                # as a harmful baseline eviction. But in our
                # tracker the id is gone now; we need to remember
                # that it was lost-with-intent-to-save for reuse
                # accounting. That memory lives in
                # ``_pending_rescue``.
                self._pending_rescue(victim_id, victim, now)

        self._order[resource_id] = _ShadowEntry(
            inserted_at_ms=now,
            last_access_step=self._step,
            num_bytes=num_bytes,
        )
        return evicted

    def on_spf_real_eviction(self, resource_id: str) -> None:
        """The real cache evicted ``resource_id``.

        If a later access on this resource happens that the
        shadow can still see (because the shadow never evicted
        it), we count it as a harmful_evictions_spf — SPF made the
        wrong decision.
        """
        self.metrics.spf_evictions_total += 1
        entry = self._order.get(resource_id)
        if entry is None:
            return
        # Flag the shadow entry as "evicted in real cache"; a
        # subsequent on_access on this id (in the shadow) means
        # SPF lost a reuse.
        entry.rescued = False  # not rescued; hurt
        entry.rescued_at_step = None
        entry.rescued_at_ms = None
        # We piggy-back on the rescue flag space by setting a
        # dedicated marker.
        setattr(entry, "_spf_lost", True)

    def mark_rescue(
        self,
        resource_id: str,
        num_bytes: int,
    ) -> None:
        """Record that SPF hinted ``resource_id`` and the shadow
        would have evicted it.

        Called by :class:`RetentionIntegration` when a hint lands
        on a resource that SPF's strict-``>`` victim check proved
        was about to be the LRU tail. The shadow LRU itself is
        independent; this is a bookkeeping mark that later turns
        into ``rescued_reuses_count`` when the resource is reused.
        """
        entry = self._order.get(resource_id)
        now = self._now_ms()
        if entry is None:
            # Resource isn't in the shadow; still record it for
            # the rescue counter via the pending map.
            self._pending_rescue(
                resource_id,
                _ShadowEntry(
                    inserted_at_ms=now,
                    last_access_step=self._step,
                    num_bytes=num_bytes,
                ),
                now,
            )
            return
        entry.rescued = True
        entry.rescued_at_step = self._step
        entry.rescued_at_ms = now

    # ------------------------------------------------------------------
    # Pending-rescue map: rescues that happened after shadow
    # eviction (so the entry is gone from self._order but we still
    # want to count the reuse if it lands later).
    # ------------------------------------------------------------------

    def __post_init__(self):  # pragma: no cover — manual init below
        pass

    def _pending_rescue(
        self,
        resource_id: str,
        entry: _ShadowEntry,
        now: float,
    ) -> None:
        """Move ``entry`` to the pending-rescue map.

        Preserves an existing ``rescued_at_{ms,step}`` if one was
        set (via an earlier :meth:`mark_rescue` call). This lets
        retention_lifetime_ms measure from the original hint time
        through to the eventual reuse, not just from "the moment
        shadow evicted it".
        """
        if not hasattr(self, "_pending"):
            self._pending = {}
        entry.rescued = True
        if entry.rescued_at_step is None:
            entry.rescued_at_step = self._step
        if entry.rescued_at_ms is None:
            entry.rescued_at_ms = now
        self._pending[resource_id] = entry

    def on_reuse(
        self,
        resource_id: str,
    ) -> None:
        """Mark that a request is about to consume ``resource_id``.

        Looks in both the live shadow and the pending-rescue map.
        If found with ``rescued=True``, counts it as a rescue
        payoff. If found with ``_spf_lost=True``, counts it as a
        harmful_evictions_spf because the real cache evicted it
        yet the shadow kept it alive and would have served the
        reuse.
        """
        # Fast path: entry alive in shadow.
        entry = self._order.get(resource_id)
        if entry is not None:
            if getattr(entry, "_spf_lost", False):
                self.metrics.harmful_evictions_spf += 1
                # Clear flag so we don't double-count.
                setattr(entry, "_spf_lost", False)
            if entry.rescued:
                self._record_rescue_payoff(entry, resource_id)
                entry.rescued = False
                entry.rescued_at_step = None
                entry.rescued_at_ms = None
            return

        # Shadow already evicted it but SPF may have rescued it in
        # the real cache. Consult pending-rescue map.
        pending = getattr(self, "_pending", {})
        entry = pending.get(resource_id)
        if entry is None:
            # Shadow never saw it; this was an unrelated reuse.
            return
        if entry.rescued:
            self._record_rescue_payoff(entry, resource_id)
        # Either way, remove from pending — one reuse, one count.
        del pending[resource_id]
        # Also count as a harmful baseline eviction since the
        # shadow would have lost this reuse.
        self.metrics.harmful_evictions_baseline += 1

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _record_rescue_payoff(
        self,
        entry: _ShadowEntry,
        resource_id: str,
    ) -> None:
        """One rescue paid off. Update metrics."""
        self.metrics.rescued_reuses_count += 1
        self.metrics.rescued_reuses_bytes += entry.num_bytes
        if entry.rescued_at_step is not None:
            dist = self._step - entry.rescued_at_step
            self.metrics._rescued_reuse_distance_sum += max(0, dist)
            self.metrics._rescued_reuse_distance_samples += 1
        if entry.rescued_at_ms is not None:
            lifetime = self._now_ms() - entry.rescued_at_ms
            self.metrics._retention_lifetime_ms_sum += (
                max(0.0, lifetime))
            self.metrics._retention_lifetime_samples += 1

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def shadow_resident_ids(self) -> "set[str]":
        return set(self._order.keys())

    def pending_rescue_ids(self) -> "set[str]":
        return set(getattr(self, "_pending", {}).keys())

    def __len__(self) -> int:
        return len(self._order)
