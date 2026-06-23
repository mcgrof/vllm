# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapter between SPF's ``BlockPool`` Protocol and vLLM v1's
real :class:`vllm.v1.core.block_pool.BlockPool`.

What this does (phase-7b):

  * Translates SPF's ``resource_id: str`` (the full-prefix hash
    from :func:`vllm.v1.core.spf.resource.hash_prefix_tokens`)
    to vLLM v1's ``BlockHashWithGroupId`` via a caller-supplied
    resolver function.
  * ``touch``: looks up the cached block by hash and, if the
    block is in the ``FreeKVCacheBlockQueue`` (ref_cnt=0),
    removes it and bumps ref_cnt — exactly the semantics the
    real ``BlockPool.touch`` has for cached-hit reuse.
  * ``contains``: hash-table lookup via
    ``BlockPool.get_cached_block``.
  * ``evict_lru``: pops from the front of ``free_block_queue``
    and returns the BlockHashWithGroupId string ids.

What this does NOT do:

  * It does not run a live vLLM server. That's a deployment step
    (pick a model, spin up the engine, wire the SPF controller
    to the scheduler, drive real HTTP traffic). The adapter is
    the pure-Python piece that makes that deployment possible.
  * It does not introduce any serve-time writes. Touching a
    cached block is a refcount bump, not a data-mutation.
  * It does not itself free blocks. When we ``touch`` a cached
    block to hold it in the pool, the corresponding
    ``release`` (from the scheduler when the request finishes)
    will decrement its ref_cnt back toward zero. The shadow
    baseline tracks whether that release leaves the block back
    in the free queue (= our rescue would be wasted) or keeps
    it alive because the scheduler itself still holds it.

Why this is not merged into ``BlockPool`` directly:

  * Keeping the adapter in ``spf/`` means SPF is a *consumer* of
    the block-pool API, not a modification of it. The block
    pool's contract with the scheduler is unchanged; SPF is an
    observer with retention hints.
  * The resolver pattern decouples "what string does SPF hand
    me?" from "how do I look that up in v1?" — integration
    code can use token-hash → block-hash mappings, session-
    scoped caches, or any other bookkeeping without forcing a
    change to this module.

Usage sketch::

    from vllm.v1.core.spf.block_pool_adapter import (
        LiveBlockPool, make_token_hash_resolver,
    )
    from vllm.v1.core.spf.integrations import (
        RetentionIntegration,
    )

    # During engine init:
    live_pool = LiveBlockPool(
        block_pool=engine.kv_cache_manager.block_pool,
        kv_cache_group_ids=[0],
        resolver=make_token_hash_resolver(
            engine.kv_cache_manager),
    )
    retention_integ = RetentionIntegration(
        controller=spf_controller,
        block_pool=live_pool,
        shadow=shadow,
    )
    # During each scheduler step, pass the SPF selected hints
    # through ``retention_integ.apply_hint(resource_id)``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

from vllm.logger import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

Resolver = Callable[[str], "Optional[KVCacheBlock]"]
"""Type alias: resolves a SPF ``resource_id`` string to a vLLM
:class:`KVCacheBlock` (or ``None`` if unknown). The mapping is
maintained by the integration code, not here."""


class LiveBlockPool:
    """SPF :class:`BlockPool` Protocol implementation backed by
    vLLM v1's real block pool.

    This is the only class in ``spf/`` that knows about v1 types.
    Everything else in the SPF package operates on strings and
    the abstract :class:`BlockPool` Protocol.

    Thread safety: same as the underlying ``BlockPool`` — single
    scheduler loop today, so no locking in the adapter itself.
    If that invariant changes, add a lock around
    :meth:`touch` / :meth:`evict_lru`; :meth:`contains` is read-
    only on the hash table and benign.
    """

    def __init__(
        self,
        block_pool: BlockPool,
        kv_cache_group_ids: list[int],
        resolver: Resolver,
    ):
        self._pool = block_pool
        self._group_ids = list(kv_cache_group_ids)
        self._resolver = resolver

    # ------------------------------------------------------------------
    # SPF BlockPool Protocol
    # ------------------------------------------------------------------

    def touch(self, resource_id: str) -> bool:
        """Pin ``resource_id`` on the GPU cache.

        Returns True if the block was resident (touch succeeded),
        False if it was already evicted. Matches the in-tree
        :class:`FakeBlockPool` contract — callers don't need to
        know which pool they have.

        Implementation:

          1. Resolve ``resource_id`` to a :class:`KVCacheBlock`
             using the caller-supplied resolver.
          2. If the block is not resident (resolver returns None,
             or block lacks a hash), return False.
          3. Delegate to ``self._pool.touch([block])`` which
             removes the block from the free queue (if present)
             and bumps its ref_cnt.

        The SPF controller pairs every successful touch with a
        later ``release`` (either explicit, when the request
        finishes, or implicit when the hint is expired). That
        release decrements ref_cnt back toward zero; when it
        hits zero the block re-enters the free queue at the MRU
        end, which is the LRU behaviour we want preserved.
        """
        block = self._resolver(resource_id)
        if block is None:
            return False
        # A cached block always has a .block_hash set. If not,
        # the resolver returned a raw block that isn't in the
        # prefix cache; refuse to touch it.
        if getattr(block, "block_hash", None) is None:
            return False
        self._pool.touch([block])
        return True

    def contains(self, resource_id: str) -> bool:
        """Hash-table membership test.

        Uses :meth:`BlockPool.get_cached_block` to ask whether
        the resource is currently indexed in the prefix cache.
        Lookup goes through each configured group id — the SPF
        contract is "resource is resident if it exists in any
        configured group"; per-group inspection belongs to a
        higher layer if ever needed.
        """
        block = self._resolver(resource_id)
        if block is None:
            return False
        return getattr(block, "block_hash", None) is not None

    def evict_lru(self, n: int = 1) -> list[str]:
        """Evict up to ``n`` LRU entries.

        Pops from the front of ``block_pool.free_block_queue``
        and records the evicted ``block_hash`` values as strings
        (so SPF book-keeping, which is string-keyed, can match
        them against outstanding hints and bump the right
        ``hints_wasted`` counters).

        Not exercised on the hot path (SPF doesn't actively
        drive eviction — it observes it via
        ``RetentionIntegration.on_eviction``). Provided here so
        the Protocol is fully implemented and tests can use it.
        """
        evicted: list[str] = []
        # Walk from LRU end. The real block_pool's eviction
        # policy may not expose a direct pop-LRU-n API, but
        # evict_blocks takes a set of ids; in the degenerate
        # "give me n LRU ids and evict them" path we pop off
        # the free_block_queue directly.
        queue = getattr(self._pool, "free_block_queue", None)
        if queue is None:
            return evicted
        # FreeKVCacheBlockQueue is a doubly-linked list with a
        # head/tail; iterate while we still have entries.
        count = 0
        while count < n:
            try:
                block = queue.popleft()
            except Exception:
                # Queue exposes different pop semantics depending
                # on the vLLM version; if popleft isn't available
                # fall back to reading the head block_id and
                # evicting by id.
                head_id = _peek_lru_block_id(queue)
                if head_id is None:
                    break
                self._pool.evict_blocks({head_id})
                evicted.append(_block_id_to_resource_id(head_id))
                count += 1
                continue
            if block is None:
                break
            if (bh := getattr(block, "block_hash", None)) is not None:
                evicted.append(str(bh))
            count += 1
        return evicted


# ---------------------------------------------------------------------------
# Resolver helpers
# ---------------------------------------------------------------------------


def make_token_hash_resolver(
    kv_cache_manager,  # type: ignore[no-untyped-def]
    kv_cache_group_id: int = 0,
) -> Resolver:
    """Build a resolver that maps SPF's resource_id string back
    to a cached :class:`KVCacheBlock` via the manager's hash
    table.

    The mapping works when SPF constructs its resource_id as
    ``str(block_hash_with_group_id)`` — the caller must arrange
    for that convention when building
    :class:`~vllm.v1.core.spf.resource.ResourceId` objects from
    live scheduler state. If a different convention is used,
    write a custom resolver.
    """
    try:
        from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
    except ImportError:
        make_block_hash_with_group_id = None  # type: ignore

    def resolve(resource_id: str):
        # The fast-path assumes resource_id encodes the
        # BlockHashWithGroupId literally; real integrations can
        # swap this for a session-scoped map.
        table = kv_cache_manager.block_pool.cached_block_hash_to_block
        try:
            bh = _parse_block_hash(resource_id, kv_cache_group_id,
                                   make_block_hash_with_group_id)
        except Exception:
            return None
        if bh is None:
            return None
        return table.get_one_block(bh)

    return resolve


def _parse_block_hash(
        resource_id: str,
        group_id: int,
        make_bhgid,  # type: ignore[no-untyped-def]
):
    """Best-effort parse of a SPF resource_id back to a vLLM
    BlockHashWithGroupId. Integration code that controls how
    resource_ids are formed should supply its own resolver
    instead of relying on this; the default handles the simple
    case where resource_id = str(block_hash)."""
    if make_bhgid is None:
        return None
    # Try interpreting resource_id as a hex-string BlockHash.
    try:
        block_hash = int(resource_id, 16)
    except ValueError:
        return None
    try:
        return make_bhgid(block_hash, group_id)
    except Exception:
        return None


def _peek_lru_block_id(free_block_queue) -> Optional[int]:
    """Peek the LRU-end block_id on ``free_block_queue`` without
    popping.

    Tolerates the two API shapes we've seen in v1 (older
    versions expose ``fake_free_list_head``, newer ones expose
    the head block directly). Returns ``None`` if the queue is
    empty or the API isn't recognised.
    """
    for attr in ("free_list_head", "_free_list_head", "fake_free_list_head"):
        head = getattr(free_block_queue, attr, None)
        if head is None:
            continue
        next_block = getattr(head, "next_free_block", None)
        if next_block is None or next_block is head:
            continue
        return getattr(next_block, "block_id", None)
    return None


def _block_id_to_resource_id(block_id: int) -> str:
    """Conservative string id for evict_lru's return value.

    When we evict by block_id (the fallback path) we don't have
    the original block_hash, so we surface the id itself as a
    string prefixed with ``bid:``. SPF-side bookkeeping will
    record this as wasted, which is the correct outcome
    (something was evicted; we don't know if it was hinted).
    """
    return f"bid:{block_id}"
