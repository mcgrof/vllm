# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration layer that ties SPF hints to real runtime effects.

Phase-4 honest-mode integrations. Two flavours, matching the mode
split from phase 1:

  - :class:`RetentionIntegration` — SPF hint ⇒ ``block_pool.touch``
    on the GPU prefix cache so the resource survives LRU eviction
    for at least this round.
  - :class:`PrefetchIntegration` — SPF hint ⇒ async promotion on
    an LMCache-like backend. A hint counts as a successful
    prefetch only if the promotion started moving bytes before the
    request arrived; hints whose promotion finishes after arrival
    are tracked as ``request_wait_on_promotion_count`` (late, not
    a success).

Both integrations accept a **protocol** for the underlying backend
so tests can use fakes without reaching into real vLLM block-pool
state or standing up an LMCache server. The real wiring (actual
vLLM block_pool, actual LMCache) lives outside this module and
imports these classes — phase-5 work.

Design rules enforced:

  * All identity is by ``resource_id`` — never the legacy
    ``first_block_hash``. Outcome tracking uses the honest id.
  * Retention hints never claim to "prefetch" (log and metric
    names come from the retention family).
  * Prefetch hints are counted as used **only** if the promotion
    started before arrival.
  * No serve-time writes to the cartridge or LMCache tier from
    SPF (retention mode is reads/touch; prefetch mode is reads
    only — the hint only asks the backend to move existing
    resources between tiers).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol

from vllm.v1.core.spf.controller import SPFController
from vllm.v1.core.spf.resource import ResourceId
from vllm.v1.core.spf.shadow_baseline import ShadowBaseline


# ===========================================================================
# Retention mode
# ===========================================================================

class BlockPool(Protocol):
    """Minimal GPU block-pool interface used by SPF retention.

    Concrete vLLM implementations expose a richer API; this is
    just the slice SPF needs. Tests supply an in-memory fake.
    """

    def touch(self, resource_id: str) -> bool:
        """Mark ``resource_id`` as recently used.

        Returns ``True`` if the resource was resident (touch
        succeeded) and ``False`` if it had already been evicted
        (touch is a no-op). A False return is the classic "hint
        arrived too late" signal.
        """
        ...

    def contains(self, resource_id: str) -> bool:
        """Whether ``resource_id`` is currently resident."""
        ...

    def evict_lru(self, n: int = 1) -> list[str]:
        """Evict up to ``n`` LRU entries. Return the evicted ids.

        Used by the pool's own memory-pressure handling; SPF
        listens via :meth:`RetentionIntegration.on_eviction` for
        book-keeping when hinted entries are displaced.
        """
        ...


class RetentionIntegration:
    """Glue: SPF retention hint ⇒ block_pool touch.

    Outcome tracking:

      * Hint applied successfully (block was resident) and a later
        request arrives for the resource → ``hints_used``.
      * Hint applied successfully and the resource is evicted
        before any request consumes it → ``hints_wasted``.
      * Hint applied but block_pool returned False (already
        evicted) → the controller's cooldown still fires (we did
        issue a hint), but the outcome is recorded as wasted
        immediately. This is the "hint too late" path.

    The integration is stateless beyond the controller + pool
    reference; all persistent state lives on the controller.
    """

    def __init__(
        self,
        controller: SPFController,
        block_pool: BlockPool,
        shadow: Optional[ShadowBaseline] = None,
    ):
        self._controller = controller
        self._pool = block_pool
        self._shadow = shadow

    @property
    def shadow(self) -> Optional[ShadowBaseline]:
        """The attached :class:`ShadowBaseline`, or None if the
        integration was constructed without a counterfactual
        tracker."""
        return self._shadow

    def apply_hint(self, resource_id: str,
                   num_bytes: int = 0) -> bool:
        """Ask the block_pool to retain ``resource_id``.

        ``num_bytes`` is optional byte size used by the shadow
        baseline for byte-weighted metrics. Defaults to 0 when
        the caller doesn't track bytes; the count-based metrics
        still work.

        Returns True if the touch succeeded (resource was
        resident). False means the hint fired too late.
        """
        self._controller.mark_hint_issued(resource_id)
        ok = self._pool.touch(resource_id)
        if not ok:
            # Touch failed ⇒ resource is already gone. Record the
            # waste eagerly so the metric reflects reality.
            self._controller.record_hint_wasted(resource_id)
            return False
        if self._shadow is not None:
            self._shadow.mark_rescue(resource_id, num_bytes)
        return True

    def on_access(
        self,
        resource_id: str,
        num_bytes: int = 0,
    ) -> None:
        """A request accessed ``resource_id`` (block-level).

        Feeds the shadow baseline so it can track access recency
        and credit any rescue payoff. Independent from
        :meth:`on_request_arrival` which is the higher-level
        request-lifecycle signal.
        """
        if self._shadow is None:
            return
        self._shadow.on_access(resource_id)
        self._shadow.on_reuse(resource_id)

    def on_insert(
        self,
        resource_id: str,
        num_bytes: int = 0,
        would_save_ids: Optional[set] = None,
    ) -> list[str]:
        """The real cache inserted ``resource_id``.

        ``would_save_ids`` is the set of resource_ids SPF has
        touched this round (the real cache won't evict them).
        Shadow uses this to record which of its own evictions
        were rescued by SPF.

        Returns the list of resource_ids the shadow evicted.
        """
        if self._shadow is None:
            return []
        return self._shadow.on_insert(
            resource_id, num_bytes, would_save_ids=would_save_ids)

    def on_request_arrival(
        self,
        resource: ResourceId,
        arrival_time: float,
    ) -> None:
        """A request arrived for ``resource``.

        Feeds the controller's arrival timestamp hook AND, if we
        had an outstanding hint on this resource, marks it used
        (request consumed the retained block).
        """
        self._controller.record_request_arrival(resource, arrival_time)
        if self._pool.contains(resource.resource_id):
            self._controller.record_request_finished(resource)
        if self._shadow is not None:
            self._shadow.on_reuse(resource.resource_id)

    def on_eviction(self, evicted_resource_ids: list[str]) -> None:
        """Block pool evicted one or more resources.

        If any evicted id had an outstanding hint, that hint was
        wasted (the retained block went away before serving a
        matching request).
        """
        for rid in evicted_resource_ids:
            self._controller.record_hint_wasted(rid)
            if self._shadow is not None:
                self._shadow.on_spf_real_eviction(rid)

    def advance_step(self) -> None:
        """Bump the shadow's step counter between scheduler
        rounds. Optional; used for reuse-distance accounting."""
        if self._shadow is not None:
            self._shadow.advance_step()


# ---------------------------------------------------------------------------
# Reference in-memory BlockPool — used by tests; real integrations
# plug in vLLM's actual pool.
# ---------------------------------------------------------------------------

class FakeBlockPool:
    """In-memory LRU block pool for retention-mode tests.

    Explicitly not a :class:`BlockPool` import because the
    Protocol compares at call sites via structural typing; the
    fake implements the same three methods and is usable wherever
    ``BlockPool`` is expected.
    """

    def __init__(self, capacity: int):
        self._capacity = capacity
        # Preserves insertion / touch order so LRU eviction is
        # deterministic.
        from collections import OrderedDict
        self._order: "OrderedDict[str, None]" = OrderedDict()
        # Audit trail for tests.
        self.touches: list[str] = []
        self.evicted: list[str] = []

    def insert(self, resource_id: str) -> list[str]:
        """Add ``resource_id``. If full, evict LRU entries and
        return their ids."""
        evicted: list[str] = []
        if resource_id in self._order:
            self._order.move_to_end(resource_id)
            return []
        while len(self._order) >= self._capacity:
            victim, _ = self._order.popitem(last=False)
            evicted.append(victim)
            self.evicted.append(victim)
        self._order[resource_id] = None
        return evicted

    def touch(self, resource_id: str) -> bool:
        self.touches.append(resource_id)
        if resource_id not in self._order:
            return False
        self._order.move_to_end(resource_id)
        return True

    def contains(self, resource_id: str) -> bool:
        return resource_id in self._order

    def evict_lru(self, n: int = 1) -> list[str]:
        evicted: list[str] = []
        for _ in range(n):
            if not self._order:
                break
            victim, _ = self._order.popitem(last=False)
            evicted.append(victim)
            self.evicted.append(victim)
        return evicted

    def resident(self) -> list[str]:
        return list(self._order.keys())


# ===========================================================================
# Prefetch mode
# ===========================================================================

@dataclass
class PromotionHandle:
    """Per-promotion state that :class:`PrefetchIntegration` tracks.

    The backend populates ``completed_at`` and ``bytes_promoted``
    when the promotion finishes. The integration reads them on
    request arrival to classify the outcome.
    """

    resource_id: str
    issued_at: float
    completed_at: Optional[float] = None
    bytes_promoted: int = 0
    num_blocks: int = 0

    @property
    def in_flight(self) -> bool:
        return self.completed_at is None

    @property
    def latency_ms(self) -> float:
        if self.completed_at is None:
            return 0.0
        return (self.completed_at - self.issued_at) * 1000.0


class PrefetchBackend(Protocol):
    """Abstract LMCache-ish async prefetch interface."""

    def promote_async(
        self,
        resource_id: str,
        issued_at: float,
    ) -> PromotionHandle:
        """Kick off async promotion for ``resource_id``.

        Must return immediately with a :class:`PromotionHandle`;
        the backend updates the handle's ``completed_at`` and
        ``bytes_promoted`` asynchronously.
        """
        ...

    def is_resident(
        self,
        resource_id: str,
        tier: str = "gpu",
    ) -> bool:
        """Whether ``resource_id`` is resident at the given tier."""
        ...


class PrefetchIntegration:
    """Glue: SPF prefetch hint ⇒ async promotion.

    Outcome tracking:

      * Promotion completed **before** request arrived → count as
        ``prefetch_used``. ``bytes_promoted`` and
        ``promote_latency_ms`` accumulate.
      * Promotion **still in flight** when the request arrived →
        ``request_wait_on_promotion_count`` += 1. This is NOT a
        success (the request had to wait), but it's also not an
        outright waste — the bytes did get moved; they just moved
        late. Tracked separately so experiments can tell "we were
        too slow" apart from "we prefetched the wrong thing".
      * Resource never consumed before the handle ages out →
        ``prefetch_wasted``.
    """

    def __init__(
        self,
        controller: SPFController,
        backend: PrefetchBackend,
    ):
        self._controller = controller
        self._backend = backend
        self._in_flight: dict[str, PromotionHandle] = {}

    def apply_hint(
        self,
        resource_id: str,
        issued_at: float,
    ) -> PromotionHandle:
        """Issue a prefetch hint. Returns the :class:`PromotionHandle`.

        Callers may inspect the handle later (tests do); the
        integration also retains it internally so
        ``on_request_arrival`` can find it by resource_id.
        """
        handle = self._backend.promote_async(resource_id, issued_at)
        self._controller.mark_hint_issued(resource_id)
        self._in_flight[resource_id] = handle
        return handle

    def on_request_arrival(
        self,
        resource: ResourceId,
        arrival_time: float,
    ) -> str:
        """Classify the outcome for ``resource`` at ``arrival_time``.

        Returns one of:

          * ``"used"``   — prefetch paid off (completed pre-arrival).
          * ``"late"``   — promotion still in flight at arrival.
          * ``"miss"``   — no hint had been issued for this resource.
        """
        self._controller.record_request_arrival(resource, arrival_time)

        handle = self._in_flight.get(resource.resource_id)
        if handle is None:
            return "miss"

        if handle.completed_at is not None and handle.completed_at <= arrival_time:
            # Clean prefetch hit. Record latency + bytes + success.
            self._controller.metrics_snapshot  # touch, no-op
            # Metrics: bytes_promoted, promote_latency_ms have
            # already been accumulated by on_promotion_complete;
            # here we just flip outcome to used.
            self._controller.record_request_finished(resource)
            del self._in_flight[resource.resource_id]
            return "used"

        # Promotion still in flight when request arrived.
        # Increment the "had to wait" counter on the prefetch
        # bucket without touching hints_issued / used / wasted.
        self._bump_wait_counter()
        # Don't clean up the handle; let on_promotion_complete or
        # expire_wasted handle it.
        return "late"

    def on_promotion_complete(
        self,
        resource_id: str,
        completed_at: float,
        bytes_promoted: int,
    ) -> None:
        """Backend reports a promotion finished.

        Accumulates ``bytes_promoted`` and ``promote_latency_ms``
        on the controller's metrics; outcome (used/wasted) is
        decided by whether a request arrived for the resource.
        """
        handle = self._in_flight.get(resource_id)
        if handle is None:
            # Stale / unknown completion — ignore.
            return
        handle.completed_at = completed_at
        handle.bytes_promoted = bytes_promoted

        m = self._controller._metrics._prefetch  # noqa: SLF001
        m.bytes_promoted += bytes_promoted
        m.promote_latency_ms += handle.latency_ms

    def expire_wasted(self, resource_id: str) -> None:
        """Forcefully mark an outstanding prefetch as wasted.

        Used when the backend or caller knows the resource will
        not be consumed (e.g. the backend evicted it, or the
        hint's deadline passed). The handle is cleared.
        """
        if resource_id in self._in_flight:
            self._controller.record_hint_wasted(resource_id)
            del self._in_flight[resource_id]

    def _bump_wait_counter(self) -> None:
        """Increment request_wait_on_promotion_count on the
        controller's prefetch bucket. No outcome mutation."""
        m = self._controller._metrics._prefetch  # noqa: SLF001
        m.request_wait_on_promotion_count += 1


# ---------------------------------------------------------------------------
# Reference FakePrefetchBackend — for tests only.
# ---------------------------------------------------------------------------

@dataclass
class FakePrefetchBackend:
    """In-memory async prefetch backend stub.

    Does NOT run an async task; tests call :meth:`complete` to
    simulate the promotion finishing at a specific time. This
    makes every test deterministic about the relative order of
    "promotion started", "promotion completed", "request arrived".
    """

    promote_ms_default: float = 2.0
    bytes_per_block: int = 1_000_000
    _handles: dict = field(default_factory=dict)
    _integration: Optional[PrefetchIntegration] = None

    # Audit fields.
    promoted_ids: list[str] = field(default_factory=list)

    def attach(self, integration: PrefetchIntegration) -> None:
        """Give the backend a back-reference so ``complete`` can
        call ``on_promotion_complete``. Keeps the real LMCache
        integration contract clean (backend ⇒ integration
        callback)."""
        self._integration = integration

    def promote_async(
        self,
        resource_id: str,
        issued_at: float,
    ) -> PromotionHandle:
        handle = PromotionHandle(
            resource_id=resource_id,
            issued_at=issued_at,
            num_blocks=1,
        )
        self._handles[resource_id] = handle
        self.promoted_ids.append(resource_id)
        return handle

    def is_resident(
        self,
        resource_id: str,
        tier: str = "gpu",
    ) -> bool:
        h = self._handles.get(resource_id)
        return h is not None and h.completed_at is not None

    def complete(
        self,
        resource_id: str,
        completed_at: float,
        bytes_promoted: int = 0,
    ) -> None:
        """Test helper: mark a promotion as done at ``completed_at``.

        Triggers the integration callback so bytes / latency
        metrics accumulate identically to a real backend.
        """
        if bytes_promoted == 0:
            bytes_promoted = self.bytes_per_block
        if self._integration is not None:
            self._integration.on_promotion_complete(
                resource_id, completed_at, bytes_promoted)
        elif resource_id in self._handles:
            # Backend without integration attached still updates
            # the handle for direct inspection.
            h = self._handles[resource_id]
            h.completed_at = completed_at
            h.bytes_promoted = bytes_promoted
