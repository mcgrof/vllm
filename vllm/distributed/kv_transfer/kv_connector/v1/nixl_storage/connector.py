# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The storage-mediated prefill/decode connector.

Producer (prefill) side: when a request that the router marked for remote
decode finishes, its KV blocks are copied to staging, written to
extents of the local namespace through a NIXL storage backend and published as a
manifest; the response carries the producer request id so the router can
hand it to the decode instance. Consumer (decode) side: the scheduler
looks the manifest up, the worker leases it, reads the extents into host
staging, restores the blocks into its GPU cache and acknowledges.

Every phase is timed separately into a JSONL trace when trace_path is
set, together with the namespace descriptors each transfer submitted so
the kernel's NVMe command trace can be joined against them. Load errors
fail the request by default; recomputation is an explicit
opt-in that is counted.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger

from .coordinator import CoordinatorClient, OutOfSpace
from .manifest import Extent, HandoffManifest
from .storage import EXPORTERS, NixlStorageAdapter, PendingTransfer, StagingArena

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request

logger = init_logger(__name__)

PARAM_KEY = "storage_handoff"


@dataclass
class HandoffConfig:
    device_name: str
    backend: str = "IO_URING"
    async_iface: str = "io_uring"
    direct: bool = True
    buffer_mode: str = "host"
    range_offset: int = 0
    range_len: int = 0
    coordinator_socket: str = "/tmp/nixl-storage-handoff.sock"
    expect_serial: str = ""
    expect_wwid: str = ""
    num_tasks: int = 128
    max_io_bytes: int = 0
    fail_on_load_error: bool = True
    lookup_timeout_s: float = 30.0
    trace_path: str = ""
    # Staging arena: which exporter backs it (host memory, dma-bufs from
    # the system heap or udmabuf, or a CUDA allocation exported as
    # dma-bufs so the payload never touches host memory), its size (0:
    # twice the largest object), and the size of each dma-buf (the kernel
    # registers at most 1 GiB).
    staging: str = "host"
    staging_bytes: int = 0
    dmabuf_chunk_bytes: int = 1 << 30
    # A reservation (of namespace space or arena space) waits this long for
    # space to come back before the transfer is given up.
    reserve_timeout_s: float = 30.0
    # A submitted storage command must become terminal within this interval.
    # Timing out is fatal to the worker: its buffers and namespace range cannot
    # be reused while the backend may still own the command.
    transfer_timeout_s: float = 30.0
    # Test hooks: report the n-th completed store/load as failed so the
    # failure paths run against a live device.
    inject_store_error_nth: int = 0
    inject_load_error_nth: int = 0

    @classmethod
    def from_extra(cls, extra: dict[str, Any]) -> HandoffConfig:
        if "device_name" not in extra or not extra.get("range_len"):
            raise ValueError("storage handoff needs device_name and range_len")
        direct = extra.get("direct", True)
        if not isinstance(direct, bool):
            raise ValueError("storage handoff direct must be a boolean")
        cfg = cls(
            device_name=str(extra["device_name"]),
            backend=str(extra.get("backend", "IO_URING")).upper(),
            async_iface=str(extra.get("async", "io_uring")),
            direct=direct,
            buffer_mode=str(extra.get("buffer_mode", "host")),
            range_offset=int(extra.get("range_offset", 0)),
            range_len=int(extra["range_len"]),
            coordinator_socket=str(
                extra.get("coordinator_socket", cls.coordinator_socket)
            ),
            expect_serial=str(extra.get("expect_serial", "")),
            expect_wwid=str(extra.get("expect_wwid", "")),
            num_tasks=int(extra.get("num_tasks", 128)),
            max_io_bytes=int(extra.get("max_io_bytes", 0)),
            fail_on_load_error=bool(extra.get("fail_on_load_error", True)),
            lookup_timeout_s=float(extra.get("lookup_timeout_s", 30.0)),
            trace_path=str(extra.get("trace_path", "")),
            staging=str(extra.get("staging", "host")),
            staging_bytes=int(extra.get("staging_bytes", 0)),
            dmabuf_chunk_bytes=int(extra.get("dmabuf_chunk_bytes", 1 << 30)),
            reserve_timeout_s=float(extra.get("reserve_timeout_s", 30.0)),
            transfer_timeout_s=float(extra.get("transfer_timeout_s", 30.0)),
            inject_store_error_nth=int(extra.get("inject_store_error_nth", 0)),
            inject_load_error_nth=int(extra.get("inject_load_error_nth", 0)),
        )
        if cfg.staging not in EXPORTERS:
            raise ValueError(f"staging must be one of {EXPORTERS}, got {cfg.staging!r}")
        if cfg.backend not in ("IO_URING", "XNVME_NVM"):
            raise ValueError(
                "storage handoff backend must be IO_URING or XNVME_NVM, "
                f"got {cfg.backend!r}"
            )
        if cfg.backend == "IO_URING" and cfg.async_iface != "io_uring":
            raise ValueError("the IO_URING backend does not take an async transport")
        if (cfg.staging != "host") != (cfg.buffer_mode == "dmabuf"):
            raise ValueError(
                "a dma-buf staging exporter and buffer_mode=dmabuf go together: "
                f"staging={cfg.staging!r} buffer_mode={cfg.buffer_mode!r}"
            )
        if cfg.buffer_mode in ("fixed", "dmabuf") and cfg.async_iface != "io_uring":
            raise ValueError(
                f"buffer_mode={cfg.buffer_mode!r} rides io_uring fixed reads and "
                "writes on the block node: set async=io_uring and name /dev/nvmeXnY"
            )
        if cfg.transfer_timeout_s <= 0:
            raise ValueError("storage handoff transfer_timeout_s must be positive")
        return cfg

    def backend_params(self) -> dict[str, str]:
        p = {
            "device_name": self.device_name,
            "buffer_mode": self.buffer_mode,
            "num_tasks": str(self.num_tasks),
        }
        if self.backend == "XNVME_NVM":
            p["async"] = self.async_iface
        else:
            p["direct"] = str(self.direct).lower()
        if self.max_io_bytes:
            p["max_io_bytes"] = str(self.max_io_bytes)
        if self.expect_serial:
            p["expect_serial"] = self.expect_serial
        if self.expect_wwid:
            p["expect_wwid"] = self.expect_wwid
        return p


@dataclass
class StoreOrder:
    request_id: str
    block_ids: list[int]
    num_tokens: int


@dataclass
class LoadOrder:
    request_id: str
    producer_request_id: str
    producer_engine_id: str
    session_epoch: str
    manifests: dict[int, str]
    block_ids: list[int]
    num_tokens: int
    # Index within the object of the first block to fetch: a local prefix
    # cache hit covers the blocks before it.
    first_block: int


@dataclass
class HandoffMetadata(KVConnectorMetadata):
    stores: dict[str, StoreOrder] = field(default_factory=dict)
    loads: dict[str, LoadOrder] = field(default_factory=dict)


class NixlStorageHandoffConnector(KVConnectorBase_V1):
    def __init__(
        self, vllm_config: VllmConfig, role: KVConnectorRole, kv_cache_config=None
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        ktc = vllm_config.kv_transfer_config
        assert ktc is not None
        self.config = HandoffConfig.from_extra(ktc.kv_connector_extra_config or {})
        kv_role = ktc.kv_role or "kv_both"
        self.is_producer = kv_role in ("kv_producer", "kv_both")
        self.is_consumer = kv_role in ("kv_consumer", "kv_both")
        self.engine_id = ktc.engine_id or uuid.uuid4().hex
        self.block_size = vllm_config.cache_config.block_size
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.model_id = (
            f"{vllm_config.model_config.model}#{vllm_config.model_config.dtype}"
        )
        self._scheduler: _Scheduler | None = None
        self._worker: _Worker | None = None
        if role == KVConnectorRole.SCHEDULER:
            self._scheduler = _Scheduler(self)
        else:
            self._worker = _Worker(self, vllm_config)

    # ---- scheduler side -----------------------------------------------------

    def get_num_new_matched_tokens(self, request: Request, num_computed_tokens: int):
        assert self._scheduler is not None
        return self._scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ):
        assert self._scheduler is not None
        self._scheduler.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        assert self._scheduler is not None
        return self._scheduler.build_connector_meta()

    def request_finished(
        self, request: Request, block_ids
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self._scheduler is not None
        return self._scheduler.request_finished(request, block_ids)

    # ---- worker side ----------------------------------------------------------

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self._worker is not None
        self._worker.register_kv_caches(kv_caches)

    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        assert self._worker is not None
        meta = self._get_connector_metadata()
        if isinstance(meta, HandoffMetadata):
            self._worker.begin(meta)

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        return

    def wait_for_save(self):
        return

    def get_finished(self, finished_req_ids: set[str]):
        assert self._worker is not None
        return self._worker.poll()

    def get_block_ids_with_load_errors(self) -> set[int]:
        assert self._worker is not None
        out, self._worker.failed_blocks = self._worker.failed_blocks, set()
        return out

    def shutdown(self):
        if self._scheduler is not None:
            logger.info(
                "storage handoff scheduler %s counters: %s",
                self.engine_id,
                self._scheduler.counters,
            )
            self._scheduler.close()
        if self._worker is not None:
            self._worker.close()


class _Scheduler:
    def __init__(self, conn: NixlStorageHandoffConnector):
        self.conn = conn
        self.coord = CoordinatorClient(conn.config.coordinator_socket)
        self.session_epoch = self.coord.call("snapshot")["session_epoch"]
        self.pending_stores: dict[str, StoreOrder] = {}
        self.pending_loads: dict[str, LoadOrder] = {}
        self.matched: dict[str, tuple[dict[int, HandoffManifest], int, int]] = {}
        self.lookup_started: dict[str, float] = {}
        self.counters = {
            "stores_ordered": 0,
            "loads_ordered": 0,
            "lookup_timeouts": 0,
            "recompute_fallbacks": 0,
            "lookup_waits": 0,
        }

    def close(self) -> None:
        self.coord.close()

    def get_num_new_matched_tokens(self, request: Request, num_computed_tokens: int):
        params = request.kv_transfer_params or {}
        handoff = params.get(PARAM_KEY)
        if not handoff or not self.conn.is_consumer:
            return 0, False
        rid = request.request_id
        producer_rid = handoff["producer_request_id"]
        producer_engine = handoff["producer_engine_id"]
        session_epoch = handoff["session_epoch"]
        manifests = {
            rank: self.coord.lookup(producer_rid, producer_engine, rank)
            for rank in range(self.conn.tp_size)
        }
        if session_epoch != self.session_epoch:
            raise RuntimeError(
                f"storage handoff for {rid}: coordinator epoch changed from "
                f"{session_epoch} to {self.session_epoch}"
            )
        if any(manifest is None for manifest in manifests.values()):
            started = self.lookup_started.setdefault(rid, time.monotonic())
            if time.monotonic() - started > self.conn.config.lookup_timeout_s:
                self.counters["lookup_timeouts"] += 1
                self.lookup_started.pop(rid, None)
                if self.conn.config.fail_on_load_error:
                    raise RuntimeError(
                        f"storage handoff for {rid}: no READY manifest for producer "
                        f"request {producer_rid} within "
                        f"{self.conn.config.lookup_timeout_s}s"
                    )
                self.counters["recompute_fallbacks"] += 1
                logger.warning(
                    "storage handoff for %s: no READY manifest for producer request %s "
                    "within %.0fs; the prompt is recomputed here (counters: %s)",
                    rid,
                    producer_rid,
                    self.conn.config.lookup_timeout_s,
                    self.counters,
                )
                return 0, False
            self.counters["lookup_waits"] += 1
            return None, False
        rank_manifests = {
            rank: manifest
            for rank, manifest in manifests.items()
            if manifest is not None
        }
        for rank, manifest in rank_manifests.items():
            if (
                manifest.session_epoch != session_epoch
                or manifest.request_id != producer_rid
                or manifest.producer != producer_engine
                or manifest.tp_rank != rank
                or manifest.tp_size != self.conn.tp_size
                or manifest.model_id != self.conn.model_id
                or manifest.block_size != self.conn.block_size
            ):
                raise RuntimeError(
                    f"storage handoff for {rid}: invalid route for producer "
                    f"{producer_engine} request {producer_rid} TP rank {rank}"
                )
        manifest = rank_manifests[0]
        if any(
            (other.num_tokens, other.num_blocks, other.block_size)
            != (manifest.num_tokens, manifest.num_blocks, manifest.block_size)
            for other in rank_manifests.values()
        ):
            raise RuntimeError(
                f"storage handoff for {rid}: producer TP manifests disagree"
            )
        self.lookup_started.pop(rid, None)
        # All but the last prompt token come from storage; the last one is
        # recomputed so the first decode step has its logits.
        prompt_token_ids = request.prompt_token_ids or []
        usable = min(manifest.num_tokens, len(prompt_token_ids) - 1)
        n = usable - num_computed_tokens
        if n <= 0:
            return 0, False
        self.matched[rid] = (rank_manifests, n, num_computed_tokens)
        return n, True

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ):
        if num_external_tokens == 0:
            self.matched.pop(request.request_id, None)
            return
        manifests, n, num_computed = self.matched.pop(request.request_id)
        manifest = manifests[0]
        block_ids = list(blocks.get_block_ids()[0])
        first = num_computed // self.conn.block_size
        needed = math.ceil((num_computed + num_external_tokens) / self.conn.block_size)
        target = block_ids[first:needed]
        if len(target) > manifest.num_blocks:
            raise RuntimeError("more blocks to load than the object holds")
        self.pending_loads[request.request_id] = LoadOrder(
            request_id=request.request_id,
            producer_request_id=manifest.request_id,
            producer_engine_id=manifest.producer,
            session_epoch=manifest.session_epoch,
            manifests={rank: item.to_json() for rank, item in manifests.items()},
            block_ids=target,
            num_tokens=num_external_tokens,
            first_block=first,
        )
        self.counters["loads_ordered"] += 1

    def build_connector_meta(self) -> HandoffMetadata:
        meta = HandoffMetadata(stores=self.pending_stores, loads=self.pending_loads)
        self.pending_stores = {}
        self.pending_loads = {}
        return meta

    def request_finished(self, request: Request, block_ids):
        params = request.kv_transfer_params or {}
        if not self.conn.is_producer or not params.get("do_remote_decode"):
            return False, None
        if isinstance(block_ids, tuple):
            block_ids = block_ids[0]
        num_tokens = request.num_computed_tokens
        nblocks = math.ceil(num_tokens / self.conn.block_size)
        order = StoreOrder(
            request_id=request.request_id,
            block_ids=list(block_ids)[:nblocks],
            num_tokens=num_tokens,
        )
        self.pending_stores[request.request_id] = order
        self.counters["stores_ordered"] += 1
        # The blocks stay allocated until the worker reports the store done.
        return True, {
            PARAM_KEY: {
                "producer_request_id": request.request_id,
                "producer_engine_id": self.conn.engine_id,
                "session_epoch": self.session_epoch,
            },
            "do_remote_prefill": True,
            "do_remote_decode": False,
        }


@dataclass
class _Store:
    order: StoreOrder
    transfer_id: str
    generation: int
    manifest: HandoffManifest
    pending: PendingTransfer
    region: Extent
    t: dict[str, int]


@dataclass
class _Load:
    order: LoadOrder
    manifest: HandoffManifest
    pending: PendingTransfer
    region: Extent
    t: dict[str, int]


class _Worker:
    def __init__(self, conn: NixlStorageHandoffConnector, vllm_config: VllmConfig):
        self.conn = conn
        self.cfg = conn.config
        self.rank = vllm_config.parallel_config.rank
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.consumer_id = f"{conn.engine_id}:{self.rank}"
        self.coord = CoordinatorClient(self.cfg.coordinator_socket)
        self.session_epoch = self.coord.call("snapshot")["session_epoch"]
        self.adapter: NixlStorageAdapter | None = None
        self.arena: StagingArena | None = None
        self.device_kv: dict[str, torch.Tensor] = {}
        self.max_model_len = vllm_config.model_config.max_model_len
        self.stores: dict[str, _Store] = {}
        self.loads: dict[str, _Load] = {}
        self.failed_blocks: set[int] = set()
        self.counters = {
            "stores_started": 0,
            "stores_done": 0,
            "stores_failed": 0,
            "loads_started": 0,
            "loads_done": 0,
            "loads_failed": 0,
            "bytes_stored": 0,
            "bytes_loaded": 0,
            "reserve_waits": 0,
            "reserve_wait_ns": 0,
            "arena_waits": 0,
            "arena_wait_ns": 0,
        }
        # Owned by the worker and closed in close().
        self._trace = (
            open(self.cfg.trace_path, "a")  # noqa: SIM115
            if self.cfg.trace_path
            else None
        )
        # Completions gathered by _poll_transfers until get_finished drains them.
        self._done_send: set[str] = set()
        self._done_recv: set[str] = set()
        self._store_seq = 0
        self._load_seq = 0

    # -- setup --------------------------------------------------------------------

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self.device_kv = kv_caches
        adapter = NixlStorageAdapter(
            f"handoff-{self.conn.engine_id}-{self.rank}",
            self.cfg.backend_params(),
            self.cfg.range_offset,
            self.cfg.range_len,
            backend_name=self.cfg.backend,
            transfer_timeout_s=self.cfg.transfer_timeout_s,
        )
        arena = None
        try:
            adapter.register_cache(kv_caches)
            # The arena holds objects in flight, not a mirror of the cache: by
            # default room for two objects of the longest possible request.
            max_blocks = math.ceil(self.max_model_len / self.conn.block_size)
            nbytes = self.cfg.staging_bytes or 2 * adapter.object_bytes(max_blocks)
            first = next(iter(kv_caches.values()))
            if self.cfg.staging == "cuda" and not first.is_cuda:
                raise RuntimeError(
                    "storage handoff: staging=cuda needs the KV cache on a CUDA device"
                )
            arena = StagingArena(
                nbytes,
                self.cfg.staging,
                self.cfg.dmabuf_chunk_bytes,
                name=f"kv-staging-{self.conn.engine_id}-{self.rank}",
                device=first.device.index or 0 if first.is_cuda else 0,
            )
            pinned = arena.cuda_register() if first.is_cuda else False
            adapter.register_arena(arena)
            warm_ms = adapter.warm()
        except Exception:
            try:
                adapter.close()
            finally:
                if arena is not None:
                    arena.close()
            raise
        assert arena is not None
        self.adapter = adapter
        self.arena = arena
        self.kv_dtype = str(first.dtype)
        self.layout = (
            "x".join(str(d) for d in first.shape[2:]) if first.ndim > 2 else "flat"
        )
        logger.info(
            "storage handoff worker %s: %d layers, staging %s %d MiB in "
            "%d chunk(s)%s%s, "
            "device map warm-up %.1f ms, device %s, ceiling %s",
            self.consumer_id,
            len(kv_caches),
            self.arena.exporter,
            self.arena.nbytes >> 20,
            len(self.arena.chunks),
            " (CUDA-pinned)" if pinned else "",
            f" (dma-buf mapping {self.arena.mapping_type})"
            if self.arena.mapping_type
            else "",
            warm_ms,
            adapter.device_info.get("serial"),
            adapter.device_info.get("max_io_bytes"),
        )
        if first.is_cuda and self.arena.exporter != "cuda" and not pinned:
            logger.warning(
                "storage handoff worker %s: the staging arena is not CUDA-pinned; "
                "device copies are pageable",
                self.consumer_id,
            )

    def _stage(self, request_id: str, nbytes: int, t: dict[str, int]) -> Extent:
        """Take an arena region for the object, waiting bounded for transfers
        in flight to give theirs back."""
        assert self.arena is not None
        deadline = time.monotonic() + self.cfg.reserve_timeout_s
        while True:
            try:
                region = self.arena.allocate(nbytes)
                break
            except OutOfSpace:
                if time.monotonic() > deadline:
                    raise
                if "arena_wait_start_ns" not in t:
                    t["arena_wait_start_ns"] = time.monotonic_ns()
                    self.counters["arena_waits"] += 1
                self._poll_transfers()
                time.sleep(0.002)
        if "arena_wait_start_ns" in t:
            self.counters["arena_wait_ns"] += (
                time.monotonic_ns() - t["arena_wait_start_ns"]
            )
        return region

    def _identity(
        self,
        request_id: str,
        transfer_id: str,
        generation: int,
        num_tokens: int,
        num_blocks: int,
        layers,
    ) -> HandoffManifest:
        assert self.adapter is not None
        return HandoffManifest(
            session_epoch=self.session_epoch,
            request_id=request_id,
            transfer_id=transfer_id,
            generation=generation,
            model_id=self.conn.model_id,
            kv_dtype=self.kv_dtype,
            kv_scales=None,
            tp_rank=self.rank,
            tp_size=self.tp_size,
            block_size=self.conn.block_size,
            num_tokens=num_tokens,
            num_blocks=num_blocks,
            layout=self.layout,
            namespace_serial=self.adapter.device_info.get("serial", ""),
            namespace_wwid=self.adapter.device_info.get("wwid", ""),
            layers=layers,
            producer=self.conn.engine_id,
        )

    # -- per step -----------------------------------------------------------------

    def begin(self, meta: HandoffMetadata) -> None:
        for store_order in meta.stores.values():
            self._begin_store(store_order)
        for load_order in meta.loads.values():
            self._begin_load(load_order)

    def _begin_store(self, order: StoreOrder) -> None:
        assert self.adapter is not None and self.arena is not None
        n = len(order.block_ids)
        t: dict[str, int] = {}
        region = None
        try:
            region = self._stage(order.request_id, self.adapter.object_bytes(n), t)
            t["copy_start_ns"] = time.monotonic_ns()
            self.adapter.pack(self.device_kv, order.block_ids, region)
            t["copy_done_ns"] = time.monotonic_ns()
            transfer_id, generation, extents = self._reserve(order.request_id, n, t)
        except OutOfSpace as e:
            if region is not None:
                self.arena.free(region)
            self.counters["stores_failed"] += 1
            self._trace_event("store", order.request_id, "", 0, None, t, "no space")
            if self.cfg.fail_on_load_error:
                raise RuntimeError(
                    f"storage handoff: store of {order.request_id}: {e}"
                ) from e
            self._done_send.add(order.request_id)
            return
        except Exception:
            if region is not None:
                self.arena.free(region)
            raise
        t["reserved_ns"] = time.monotonic_ns()
        try:
            layers = self.adapter.layer_slices(extents, n)
            manifest = self._identity(
                order.request_id,
                transfer_id,
                generation,
                order.num_tokens,
                n,
                layers,
            )
            self.coord.call("mark_writing", transfer_id=transfer_id)
            pending = self.adapter.start("WRITE", layers, n, region)
        except Exception as e:
            self.coord.call(
                "fail",
                transfer_id=transfer_id,
                generation=generation,
                reason=f"write submission failed: {e}",
            )
            self.coord.call("producer_drained", transfer_id=transfer_id)
            self.coord.call("reclaim")
            self.arena.free(region)
            self.counters["stores_failed"] += 1
            self._trace_event(
                "store", order.request_id, transfer_id, generation, None, t, "error"
            )
            if self.cfg.fail_on_load_error:
                raise RuntimeError(
                    f"storage handoff: store of {order.request_id} failed to submit"
                ) from e
            self._done_send.add(order.request_id)
            return
        t["submitted_ns"] = time.monotonic_ns()
        self.stores[order.request_id] = _Store(
            order, transfer_id, generation, manifest, pending, region, t
        )
        self.counters["stores_started"] += 1

    def _reserve(self, request_id: str, num_blocks: int, t: dict[str, int]):
        """Reserve the object's extents. While the range is full, keep
        completing and publishing earlier stores (consumers free space by
        acknowledging them) and retry until reserve_timeout_s passes."""
        assert self.adapter is not None
        lengths = self.adapter.object_lengths(num_blocks)
        deadline = time.monotonic() + self.cfg.reserve_timeout_s
        while True:
            try:
                transfer_id, generation, extents, _ = self.coord.reserve(
                    request_id, self.conn.engine_id, self.rank, lengths
                )
                break
            except OutOfSpace:
                if time.monotonic() > deadline:
                    raise
                if "reserve_wait_start_ns" not in t:
                    t["reserve_wait_start_ns"] = time.monotonic_ns()
                    self.counters["reserve_waits"] += 1
                self._poll_transfers()
                time.sleep(0.002)
        if "reserve_wait_start_ns" in t:
            self.counters["reserve_wait_ns"] += (
                time.monotonic_ns() - t["reserve_wait_start_ns"]
            )
        return transfer_id, generation, extents

    def _begin_load(self, order: LoadOrder) -> None:
        assert self.adapter is not None and self.arena is not None
        try:
            manifest_json = order.manifests[self.rank]
        except KeyError as e:
            raise RuntimeError(
                f"storage handoff for {order.request_id}: no manifest for "
                f"TP rank {self.rank}"
            ) from e
        manifest = HandoffManifest.from_json(manifest_json)
        if (
            order.session_epoch != self.session_epoch
            or manifest.session_epoch != order.session_epoch
            or manifest.request_id != order.producer_request_id
            or manifest.producer != order.producer_engine_id
            or manifest.tp_rank != self.rank
        ):
            raise RuntimeError(
                f"storage handoff for {order.request_id}: manifest route does not "
                "match the scheduled producer"
            )
        local_layers = self.adapter.layer_slices(manifest.extents, manifest.num_blocks)
        local = self._identity(
            "",
            "",
            0,
            manifest.num_tokens,
            manifest.num_blocks,
            local_layers,
        )
        if not manifest.same_identity(local):
            raise RuntimeError(
                f"storage handoff for {order.request_id}: object identity "
                f"{manifest.model_id}/{manifest.kv_dtype}/{manifest.layout} differs "
                f"from this engine's {local.model_id}/{local.kv_dtype}/{local.layout}"
            )
        n = len(order.block_ids)
        expected_lengths = self.adapter.object_lengths(manifest.num_blocks)
        bad_extents = any(
            layer.extent.length < expected or layer.extent.length - expected >= 4096
            for layer, expected in zip(manifest.layers, expected_lengths)
        )
        if order.first_block + n > manifest.num_blocks or bad_extents:
            raise RuntimeError(
                f"storage handoff for {order.request_id}: manifest object geometry "
                "does not match the local cache"
            )
        t: dict[str, int] = {}
        region = self._stage(order.request_id, self.adapter.object_bytes(n), t)
        t["lease_start_ns"] = time.monotonic_ns()
        leased = False
        try:
            _, manifest = self.coord.lease(
                manifest.transfer_id, manifest.generation, self.consumer_id
            )
            leased = True
            pending = self.adapter.start(
                "READ",
                manifest.layers,
                n,
                region,
                first_block=order.first_block,
                object_blocks=manifest.num_blocks,
            )
        except Exception as e:
            if leased:
                self.coord.call(
                    "force_release",
                    transfer_id=manifest.transfer_id,
                    consumer=self.consumer_id,
                    reason=f"read submission failed: {e}",
                )
                self.coord.call("reclaim")
            self.arena.free(region)
            raise
        t["leased_ns"] = time.monotonic_ns()
        t["submitted_ns"] = time.monotonic_ns()
        self.loads[order.request_id] = _Load(order, manifest, pending, region, t)
        self.counters["loads_started"] += 1

    def poll(self) -> tuple[set[str], set[str]]:
        self._poll_transfers()
        done_send, self._done_send = self._done_send, set()
        done_recv, self._done_recv = self._done_recv, set()
        return done_send, done_recv

    def _poll_transfers(self) -> None:
        assert self.adapter is not None and self.arena is not None
        done_send = self._done_send
        done_recv = self._done_recv
        for rid, st in list(self.stores.items()):
            status = self.adapter.poll(st.pending)
            if status == "PROC":
                continue
            self._store_seq += 1
            if status == "DONE" and self._store_seq == self.cfg.inject_store_error_nth:
                logger.warning(
                    "storage handoff: injecting a write failure into store %s", rid
                )
                status = "ERR"
            st.t["done_ns"] = time.monotonic_ns()
            self.adapter.release(st.pending)
            self.arena.free(st.region)
            del self.stores[rid]
            if status == "DONE":
                self.coord.publish(st.transfer_id, st.generation, st.manifest)
                st.t["published_ns"] = time.monotonic_ns()
                self.counters["stores_done"] += 1
                self.counters["bytes_stored"] += st.pending.plan.bytes
                self._trace_event(
                    "store", rid, st.transfer_id, st.generation, st.pending, st.t, "ok"
                )
            else:
                self.coord.call(
                    "fail",
                    transfer_id=st.transfer_id,
                    generation=st.generation,
                    reason="write failed",
                )
                self.coord.call("producer_drained", transfer_id=st.transfer_id)
                self.counters["stores_failed"] += 1
                self._trace_event(
                    "store",
                    rid,
                    st.transfer_id,
                    st.generation,
                    st.pending,
                    st.t,
                    "error",
                )
                if self.cfg.fail_on_load_error:
                    raise RuntimeError(f"storage handoff: store of {rid} failed")
            done_send.add(rid)
        for rid, ld in list(self.loads.items()):
            status = self.adapter.poll(ld.pending)
            if status == "PROC":
                continue
            self._load_seq += 1
            if status == "DONE" and self._load_seq == self.cfg.inject_load_error_nth:
                logger.warning(
                    "storage handoff: injecting a read failure into load %s", rid
                )
                status = "ERR"
            ld.t["done_ns"] = time.monotonic_ns()
            self.adapter.release(ld.pending)
            del self.loads[rid]
            m = ld.manifest
            if status == "DONE":
                try:
                    self.adapter.unpack(self.device_kv, ld.order.block_ids, ld.region)
                    ld.t["restored_ns"] = time.monotonic_ns()
                    self.coord.call(
                        "ack",
                        transfer_id=m.transfer_id,
                        generation=m.generation,
                        consumer=self.consumer_id,
                    )
                    ld.t["acked_ns"] = time.monotonic_ns()
                except Exception as e:
                    logger.error("storage handoff: load of %s failed: %s", rid, e)
                    status = "ERR"
            if status == "DONE":
                self.counters["loads_done"] += 1
                self.counters["bytes_loaded"] += ld.pending.plan.bytes
                self._trace_event(
                    "load", rid, m.transfer_id, m.generation, ld.pending, ld.t, "ok"
                )
            else:
                self.coord.call(
                    "force_release",
                    transfer_id=m.transfer_id,
                    consumer=self.consumer_id,
                    reason="read failed",
                )
                self.coord.call("reclaim")
                self.counters["loads_failed"] += 1
                self._trace_event(
                    "load", rid, m.transfer_id, m.generation, ld.pending, ld.t, "error"
                )
                if self.cfg.fail_on_load_error:
                    self.arena.free(ld.region)
                    raise RuntimeError(f"storage handoff: load of {rid} failed")
                self.failed_blocks.update(ld.order.block_ids)
            self.arena.free(ld.region)
            done_recv.add(rid)

    def _trace_event(self, op, rid, tid, gen, pending, t, result) -> None:
        if self._trace is None:
            return
        rec = {
            "event": op,
            "request_id": rid,
            "transfer_id": tid,
            "generation": gen,
            "engine": self.consumer_id,
            "result": result,
            **t,
        }
        if pending is not None:
            rec.update(
                bytes=pending.plan.bytes,
                descriptors=pending.plan.descriptors,
                remote=[[off, length] for off, length, _ in pending.plan.remote],
            )
        if self.arena is not None:
            rec["arena_free_bytes"] = self.arena.free_bytes
        self._trace.write(json.dumps(rec) + "\n")
        self._trace.flush()

    def close(self) -> None:
        logger.info(
            "storage handoff worker %s counters: %s adapter: %s",
            self.consumer_id,
            self.counters,
            self.adapter.counters if self.adapter else {},
        )
        if self._trace:
            self._trace.close()
        try:
            if self.adapter:
                self.adapter.close()
        finally:
            try:
                if self.arena:
                    self.arena.close()
            finally:
                self.coord.close()
