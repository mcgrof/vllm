# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU residency manager for cartridge KV chunks.

Sits between the CPU/disk-resident ``CartridgeStore`` and the
connector's inject path. Bounded GPU capacity with LRU eviction,
refcounted pins for in-flight requests, and chunk-keyed lookup.

This is the fifth production-architecture component called out in
``docs/design/cartridge_connector.md``. It graduates cartridge
serving from "load everything on CPU at init, pay a CPU->GPU copy
per request" to a real cache tier:

  - GPU memory is bounded by a configured capacity (bytes).
  - A cartridge resident on GPU gets a cheap ``get_chunk`` lookup
    (no PCIe copy). First request pays the promotion; subsequent
    requests hit.
  - Idle cartridges do not squat on GPU forever; they are evicted
    LRU when capacity pressure arrives.
  - Cartridges actively serving a request are pinned (``acquire``)
    and cannot be evicted mid-injection.

Ownership boundaries:

  - ``CartridgeStore``: owns durable CPU/disk storage. Read-only
    from the serve path. Backed by ``.pt`` files; chunk lookups
    return CPU tensors.
  - ``GPUResidencyManager``: owns GPU placement. Borrows tensors
    from the store, caches GPU copies, enforces capacity, tracks
    in-flight refcounts, evicts on pressure.
  - ``CartridgeConnector``: request-time consumer only. Calls
    ``acquire``/``get_chunk``/``release``. Never touches raw
    ``.to(device=...)`` on the hot path.
  - ``CartridgeLMCachePlugin``: unchanged, still read-only.
    Orthogonal to this tier — the residency manager borrows from
    the same ``CartridgeStore`` the plugin exposes to LMCache.

Chunk vs cartridge granularity: chunks are keyed by
``(cartridge_id, layer_idx)`` (one chunk per layer). A request
using cartridge A needs every layer chunk of A — partial residency
of A is useless for serving. So pin/unpin/evict operate at
**cartridge** granularity (N chunks together). Metrics and lookup
still use chunk keys for measurement precision.

No serve-time writes to the cartridge tier: cartridge content is
produced offline by the Self-Study training pipeline. This manager
only reads the store and holds a GPU-device mirror.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_store import (
        CartridgeStore,
    )

logger = init_logger(__name__)


@dataclass
class ResidencyMetrics:
    """Snapshot counters for the GPU residency tier.

    Incremented in place; ``.snapshot()`` returns a plain dict
    suitable for JSON logging or comparison in tests.
    """

    gpu_hit: int = 0
    gpu_miss: int = 0
    cpu_hit: int = 0
    disk_hit: int = 0
    promote_count: int = 0
    demote_count: int = 0
    evict_count: int = 0
    bytes_promoted: int = 0
    bytes_evicted: int = 0
    request_wait_on_promotion_count: int = 0

    def snapshot(self) -> dict[str, int]:
        return {
            "gpu_hit": self.gpu_hit,
            "gpu_miss": self.gpu_miss,
            "cpu_hit": self.cpu_hit,
            "disk_hit": self.disk_hit,
            "promote_count": self.promote_count,
            "demote_count": self.demote_count,
            "evict_count": self.evict_count,
            "bytes_promoted": self.bytes_promoted,
            "bytes_evicted": self.bytes_evicted,
            "request_wait_on_promotion_count": self.request_wait_on_promotion_count,
        }


@dataclass
class _CartridgeGPUEntry:
    """Per-cartridge GPU residency bookkeeping.

    Chunks for the cartridge live in ``chunks`` keyed by layer
    index. ``bytes`` is the summed tensor size on GPU (used for
    capacity accounting). ``ref_count`` is the number of in-flight
    requests holding this cartridge via acquire(). ``last_used`` is
    a monotonic tick bumped on every acquire() — the LRU key.
    """

    cartridge_id: str
    chunks: dict[int, torch.Tensor] = field(default_factory=dict)
    bytes: int = 0
    ref_count: int = 0
    last_used: int = 0


class GPUResidencyError(RuntimeError):
    """Raised when a cartridge cannot be made resident on GPU.

    Typical cause: all GPU-resident cartridges are pinned and the
    requested cartridge plus pinned ones exceed configured
    capacity. Caller has the option to back off, block, or fall
    through to a non-cartridge code path; this class does not
    decide policy for them.
    """


class GPUResidencyManager:
    """Bounded-capacity GPU tier for cartridge KV chunks.

    Reads from a backing ``CartridgeStore`` (CPU source of truth),
    caches promoted chunks on a GPU device, evicts under pressure
    using LRU, and protects in-flight cartridges via refcounts.
    """

    def __init__(
        self,
        store: CartridgeStore,
        capacity_bytes: int,
        device: torch.device | str = "cuda",
        eviction_policy: str = "lru",
    ):
        if capacity_bytes <= 0:
            raise ValueError(f"capacity_bytes must be positive, got {capacity_bytes}")
        if eviction_policy != "lru":
            raise ValueError(
                f"only 'lru' eviction is implemented, got {eviction_policy!r}"
            )

        self._store = store
        self._capacity_bytes = int(capacity_bytes)
        self._device = torch.device(device) if isinstance(device, str) else device
        self._eviction_policy = eviction_policy

        # LRU order: OrderedDict keyed by cartridge_id. Key ordering
        # is oldest->newest. On acquire we .move_to_end() to mark a
        # cartridge as most-recently-used.
        self._entries: OrderedDict[str, _CartridgeGPUEntry] = OrderedDict()
        self._bytes_resident: int = 0
        self._tick: int = 0
        self._lock = threading.RLock()

        self.metrics = ResidencyMetrics()

        logger.info(
            "GPUResidencyManager: device=%s capacity=%d bytes (%.2f GB) policy=%s",
            self._device,
            self._capacity_bytes,
            self._capacity_bytes / 1e9,
            self._eviction_policy,
        )

    # ------------------------------------------------------------------
    # Lifecycle: acquire / release / get_chunk / prefetch
    # ------------------------------------------------------------------

    def acquire(
        self,
        cartridge_id: str,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Ensure all chunks of ``cartridge_id`` are resident on the
        GPU tier and pin the cartridge against eviction.

        Atomic: if any required promotion would exceed capacity,
        this method evicts LRU unpinned cartridges first. If after
        evicting all unpinned entries the requested cartridge still
        does not fit, ``GPUResidencyError`` is raised and no refcount
        or residency changes take effect.

        Increments the in-flight refcount on success. The caller
        MUST pair a successful ``acquire`` with exactly one
        ``release`` when its request finishes.

        ``dtype`` optionally requests promotion into a specific GPU
        dtype (typically the paged-cache dtype, e.g. ``bfloat16``).
        If omitted, the store's native dtype is used.
        """
        with self._lock:
            entry = self._entries.get(cartridge_id)
            if entry is not None:
                # GPU hit: already resident, just bump refcount and LRU.
                entry.ref_count += 1
                self._tick += 1
                entry.last_used = self._tick
                self._entries.move_to_end(cartridge_id)
                self.metrics.gpu_hit += 1
                logger.debug(
                    "GPU hit: cartridge=%s ref=%d",
                    cartridge_id,
                    entry.ref_count,
                )
                return

            # GPU miss: we need to promote. Compute required bytes by
            # reading chunk sizes from the store without yet copying
            # to GPU.
            chunks_cpu = self._fetch_cpu_chunks(cartridge_id)
            required_bytes = sum(
                t.nelement() * self._element_size(t, dtype) for t in chunks_cpu.values()
            )
            if required_bytes > self._capacity_bytes:
                raise GPUResidencyError(
                    f"cartridge {cartridge_id!r} needs "
                    f"{required_bytes} bytes, capacity is "
                    f"{self._capacity_bytes} — will never fit"
                )

            self._evict_until_fits(required_bytes, cartridge_id)

            # Promote to GPU.
            new_entry = _CartridgeGPUEntry(cartridge_id=cartridge_id)
            promoted_bytes = 0
            for layer_idx, cpu_tensor in sorted(chunks_cpu.items()):
                gpu_tensor = cpu_tensor.to(
                    device=self._device,
                    dtype=dtype or cpu_tensor.dtype,
                    non_blocking=False,
                )
                new_entry.chunks[layer_idx] = gpu_tensor
                tensor_bytes = gpu_tensor.nelement() * gpu_tensor.element_size()
                new_entry.bytes += tensor_bytes
                promoted_bytes += tensor_bytes

            new_entry.ref_count = 1
            self._tick += 1
            new_entry.last_used = self._tick
            self._entries[cartridge_id] = new_entry
            self._bytes_resident += new_entry.bytes

            self.metrics.gpu_miss += 1
            self.metrics.request_wait_on_promotion_count += 1
            self.metrics.promote_count += 1
            self.metrics.bytes_promoted += promoted_bytes
            self.metrics.cpu_hit += len(chunks_cpu)
            logger.info(
                "GPU promote: cartridge=%s chunks=%d bytes=%d resident=%d/%d",
                cartridge_id,
                len(new_entry.chunks),
                new_entry.bytes,
                self._bytes_resident,
                self._capacity_bytes,
            )

    def release(self, cartridge_id: str) -> None:
        """Decrement the in-flight refcount for ``cartridge_id``.

        No-op if the cartridge isn't tracked (logs a warning). A
        released cartridge is eligible for eviction once its
        refcount reaches zero; eviction does NOT happen eagerly,
        only under capacity pressure.
        """
        with self._lock:
            entry = self._entries.get(cartridge_id)
            if entry is None:
                logger.warning(
                    "release: cartridge %s not resident; nothing to release",
                    cartridge_id,
                )
                return
            if entry.ref_count <= 0:
                logger.warning(
                    "release: cartridge %s already at ref_count=0",
                    cartridge_id,
                )
                return
            entry.ref_count -= 1

    def get_chunk(
        self,
        cartridge_id: str,
        layer_idx: int,
    ) -> torch.Tensor:
        """Return the GPU-resident chunk tensor for a layer.

        Caller MUST have called ``acquire(cartridge_id)`` earlier in
        the same logical request. Raises ``GPUResidencyError`` if
        the cartridge isn't resident or the layer index is out of
        range — that indicates a bug in the calling sequence, not a
        runtime eviction.
        """
        with self._lock:
            entry = self._entries.get(cartridge_id)
            if entry is None:
                raise GPUResidencyError(
                    f"cartridge {cartridge_id!r} not resident; call acquire() first"
                )
            chunk = entry.chunks.get(layer_idx)
            if chunk is None:
                raise GPUResidencyError(
                    f"cartridge {cartridge_id!r} layer "
                    f"{layer_idx} not in residency entry "
                    f"(have {sorted(entry.chunks.keys())})"
                )
            return chunk

    def prefetch(
        self,
        cartridge_id: str,
        dtype: torch.dtype | None = None,
    ) -> bool:
        """Best-effort promote ``cartridge_id`` to GPU without
        pinning.

        Used by speculative routing: when the router predicts a
        likely next cartridge, callers can pay the promotion cost
        now so the next request is a GPU hit.

        Returns True if the cartridge ends up resident (either
        already was, or promotion succeeded). Returns False if
        promotion could not complete within current capacity
        (e.g. all other entries are pinned). Prefetch NEVER raises,
        and NEVER evicts pinned entries.
        """
        with self._lock:
            entry = self._entries.get(cartridge_id)
            if entry is not None:
                return True
            try:
                chunks_cpu = self._fetch_cpu_chunks(cartridge_id)
            except Exception as e:  # pragma: no cover
                logger.warning(
                    "prefetch: cannot read cartridge %s: %s",
                    cartridge_id,
                    e,
                )
                return False
            required_bytes = sum(
                t.nelement() * self._element_size(t, dtype) for t in chunks_cpu.values()
            )
            if required_bytes > self._capacity_bytes:
                return False
            # Try to evict; if not enough unpinned room, back off.
            if not self._try_evict_until_fits(required_bytes, cartridge_id):
                return False

            new_entry = _CartridgeGPUEntry(cartridge_id=cartridge_id)
            for layer_idx, cpu_tensor in sorted(chunks_cpu.items()):
                gpu_tensor = cpu_tensor.to(
                    device=self._device,
                    dtype=dtype or cpu_tensor.dtype,
                    non_blocking=False,
                )
                new_entry.chunks[layer_idx] = gpu_tensor
                new_entry.bytes += gpu_tensor.nelement() * gpu_tensor.element_size()
            new_entry.ref_count = 0  # prefetch does NOT pin
            self._tick += 1
            new_entry.last_used = self._tick
            self._entries[cartridge_id] = new_entry
            self._bytes_resident += new_entry.bytes
            self.metrics.promote_count += 1
            self.metrics.bytes_promoted += new_entry.bytes
            self.metrics.cpu_hit += len(chunks_cpu)
            logger.info(
                "GPU prefetch: cartridge=%s chunks=%d bytes=%d (not pinned)",
                cartridge_id,
                len(new_entry.chunks),
                new_entry.bytes,
            )
            return True

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def resident_cartridges(self) -> set[str]:
        with self._lock:
            return set(self._entries.keys())

    def resident_bytes(self) -> int:
        with self._lock:
            return self._bytes_resident

    def pinned_cartridges(self) -> set[str]:
        with self._lock:
            return {cid for cid, e in self._entries.items() if e.ref_count > 0}

    def is_resident(self, cartridge_id: str) -> bool:
        with self._lock:
            return cartridge_id in self._entries

    def ref_count(self, cartridge_id: str) -> int:
        with self._lock:
            e = self._entries.get(cartridge_id)
            return e.ref_count if e is not None else 0

    def lru_order(self) -> list[str]:
        """Return cartridge IDs oldest-first (debug/test helper)."""
        with self._lock:
            return list(self._entries.keys())

    def snapshot_metrics(self) -> dict[str, int]:
        with self._lock:
            snap = self.metrics.snapshot()
            snap["current_gpu_resident_bytes"] = self._bytes_resident
            snap["current_gpu_resident_cartridges"] = len(self._entries)
            snap["pinned_entries"] = sum(
                1 for e in self._entries.values() if e.ref_count > 0
            )
            return snap

    def reset_metrics(self) -> None:
        with self._lock:
            self.metrics = ResidencyMetrics()

    def close(self) -> None:
        """Free every GPU-resident chunk."""
        with self._lock:
            for entry in self._entries.values():
                for layer_idx in list(entry.chunks.keys()):
                    # Drop reference; torch will reclaim on next gc.
                    del entry.chunks[layer_idx]
            self._entries.clear()
            self._bytes_resident = 0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _fetch_cpu_chunks(
        self,
        cartridge_id: str,
    ) -> dict[int, torch.Tensor]:
        """Read all layer chunks from the backing store.

        Returns a dict keyed by layer_idx. Store acquire/release
        bracket the read so the backing entry can't be evicted
        mid-promotion.
        """
        # Lazy import to avoid a circular dependency between
        # cartridge_store (which imports load_cartridge from
        # cartridge_connector) and cartridge_connector (which
        # imports this residency manager).
        from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_store import (
            ChunkKey,
        )

        residency = self._store.get_residency(cartridge_id)
        if residency is None:
            raise GPUResidencyError(
                f"cartridge {cartridge_id!r} not loaded in backing CartridgeStore"
            )

        num_layers = residency.num_layers
        # Borrow a ref on the store so it can't evict under us.
        self._store.acquire(cartridge_id)
        try:
            chunks: dict[int, torch.Tensor] = {}
            for layer_idx in range(num_layers):
                t = self._store.get(ChunkKey(cartridge_id, layer_idx))
                if t is None:
                    raise GPUResidencyError(
                        f"cartridge {cartridge_id!r} layer "
                        f"{layer_idx} missing from store"
                    )
                chunks[layer_idx] = t
            return chunks
        finally:
            self._store.release(cartridge_id)

    def _element_size(
        self,
        tensor: torch.Tensor,
        dtype: torch.dtype | None,
    ) -> int:
        """Return element size after optional dtype coercion."""
        if dtype is None or dtype == tensor.dtype:
            return tensor.element_size()
        # torch has no element_size() on dtype objects; construct a
        # zero-element tensor of the target dtype and read from it.
        return torch.empty(0, dtype=dtype).element_size()

    def _evict_until_fits(
        self,
        needed_bytes: int,
        incoming_cartridge_id: str,
    ) -> None:
        """Evict LRU unpinned cartridges until ``needed_bytes`` of
        free space exists. Raises if insufficient even after
        evicting all unpinned entries.
        """
        if not self._try_evict_until_fits(
            needed_bytes,
            incoming_cartridge_id,
        ):
            free_now = self._capacity_bytes - self._bytes_resident
            pinned = self.pinned_cartridges()
            raise GPUResidencyError(
                f"cannot fit cartridge "
                f"{incoming_cartridge_id!r}: needs "
                f"{needed_bytes} bytes, free={free_now}, "
                f"capacity={self._capacity_bytes}, "
                f"pinned={sorted(pinned)}"
            )

    def _try_evict_until_fits(
        self,
        needed_bytes: int,
        incoming_cartridge_id: str,
    ) -> bool:
        """Variant that returns False instead of raising — used by
        prefetch which treats capacity pressure as "skip" rather
        than an error.
        """
        free = self._capacity_bytes - self._bytes_resident
        if free >= needed_bytes:
            return True

        # Walk LRU order (oldest first) and evict unpinned entries
        # until we have room. Don't touch the incoming cartridge.
        for victim_id in list(self._entries.keys()):
            if free >= needed_bytes:
                break
            if victim_id == incoming_cartridge_id:
                continue
            entry = self._entries[victim_id]
            if entry.ref_count > 0:
                continue
            # Evict.
            freed = entry.bytes
            for layer_idx in list(entry.chunks.keys()):
                del entry.chunks[layer_idx]
            del self._entries[victim_id]
            self._bytes_resident -= freed
            free += freed
            self.metrics.evict_count += 1
            self.metrics.bytes_evicted += freed
            self.metrics.demote_count += 1
            logger.info(
                "GPU evict: cartridge=%s freed=%d bytes (new resident=%d/%d)",
                victim_id,
                freed,
                self._bytes_resident,
                self._capacity_bytes,
            )

        return (self._capacity_bytes - self._bytes_resident) >= needed_bytes
