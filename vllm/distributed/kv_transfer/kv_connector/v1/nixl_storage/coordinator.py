# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Same-host coordinator: the single owner of extents and object state.

Runs in-process (``StorageHandoffCoordinator``) or behind a Unix-socket
JSON line protocol (``CoordinatorServer`` / ``CoordinatorClient``) so a
prefill and a decode process on one host share it. Extents come from a
first-fit free list over one byte range of one namespace. Reclamation
happens only after every reader acknowledged and every party reported its
I/O drained; a lease timeout alone never frees a range. Each object is one
producer TP shard consumed by the corresponding TP rank of one decode engine.
"""

from __future__ import annotations

import json
import os
import socket
import socketserver
import threading
import time
import uuid
from dataclasses import dataclass, field

from .manifest import Extent, HandoffManifest, ObjectState


class HandoffError(Exception):
    pass


class StaleGeneration(HandoffError):
    pass


class NotReady(HandoffError):
    pass


class OutOfSpace(HandoffError):
    pass


@dataclass
class Lease:
    consumer: str
    lease_id: str
    granted_at: float
    expires_at: float
    acked: bool = False
    drained: bool = False


@dataclass
class HandoffObject:
    transfer_id: str
    request_id: str
    producer: str
    tp_rank: int
    session_epoch: str
    generation: int
    extents: list[Extent]
    state: ObjectState = ObjectState.RESERVED
    manifest: HandoffManifest | None = None
    producer_drained: bool = False
    leases: dict[str, Lease] = field(default_factory=dict)
    created_at: float = field(default_factory=time.monotonic)
    error: str | None = None


class ExtentAllocator:
    """First-fit allocator over [base, base + size) with a fixed alignment.
    Frees coalesce with their neighbours."""

    def __init__(self, base: int, size: int, align: int):
        if align <= 0 or base < 0 or size <= 0 or base % align or size % align:
            raise ValueError(
                "nonnegative base and positive size must be multiples of align"
            )
        self.base = base
        self.size = size
        self.align = align
        self._free: list[Extent] = [Extent(base, size)]
        self.allocated_bytes = 0

    def allocate(self, length: int) -> Extent:
        if length <= 0:
            raise ValueError("allocation length must be positive")
        length = (length + self.align - 1) // self.align * self.align
        for i, hole in enumerate(self._free):
            if hole.length >= length:
                out = Extent(hole.offset, length)
                rest = hole.length - length
                if rest:
                    self._free[i] = Extent(hole.offset + length, rest)
                else:
                    del self._free[i]
                self.allocated_bytes += length
                return out
        raise OutOfSpace(f"no hole of {length} bytes")

    def free(self, extent: Extent) -> None:
        if (
            extent.length <= 0
            or extent.offset < self.base
            or extent.end > self.base + self.size
            or extent.offset % self.align
            or extent.length % self.align
        ):
            raise HandoffError("invalid free extent")
        merged: list[Extent] = []
        # Validate on a temporary list: a rejected double free must not leave
        # overlapping holes available to subsequent allocations.
        for e in sorted([*self._free, extent], key=lambda e: e.offset):
            if merged and merged[-1].end == e.offset:
                merged[-1] = Extent(merged[-1].offset, merged[-1].length + e.length)
            elif merged and e.offset < merged[-1].end:
                raise HandoffError(f"double free of {e}")
            else:
                merged.append(e)
        self._free = merged
        self.allocated_bytes -= extent.length

    @property
    def free_bytes(self) -> int:
        return sum(e.length for e in self._free)


class StorageHandoffCoordinator:
    """Extent owner and manifest registry for one namespace range."""

    def __init__(
        self,
        base: int,
        size: int,
        align: int = 4096,
        lease_timeout_s: float = 60.0,
        session_epoch: str | None = None,
    ):
        self.allocator = ExtentAllocator(base, size, align)
        self.lease_timeout_s = lease_timeout_s
        self.session_epoch = session_epoch or uuid.uuid4().hex
        self._objects: dict[str, HandoffObject] = {}
        self._by_request: dict[tuple[str, str, int], str] = {}
        self._lock = threading.RLock()
        self.counters = {
            "reserved": 0,
            "published": 0,
            "leases": 0,
            "acks": 0,
            "duplicate_acks": 0,
            "stale_rejected": 0,
            "cancelled": 0,
            "reclaimed": 0,
            "lease_expired": 0,
        }

    # ---- producer side -----------------------------------------------------

    def reserve(
        self, request_id: str, producer: str, tp_rank: int, lengths: list[int]
    ) -> tuple[str, int, list[Extent]]:
        """Allocate extents for one producer request and TP rank.

        Repeating the same producer/request/rank key creates a new generation;
        the older generation stays allocated until it is reclaimed.
        """
        with self._lock:
            if not lengths or any(length <= 0 for length in lengths):
                raise ValueError("reservation lengths must be nonempty and positive")
            key = (producer, request_id, tp_rank)
            generation = 0
            old = self._by_request.get(key)
            if old is not None:
                generation = self._objects[old].generation + 1
            extents: list[Extent] = []
            try:
                for length in lengths:
                    extents.append(self.allocator.allocate(length))
            except OutOfSpace:
                for e in extents:
                    self.allocator.free(e)
                raise
            transfer_id = uuid.uuid4().hex
            obj = HandoffObject(
                transfer_id=transfer_id,
                request_id=request_id,
                producer=producer,
                tp_rank=tp_rank,
                session_epoch=self.session_epoch,
                generation=generation,
                extents=extents,
            )
            self._objects[transfer_id] = obj
            self._by_request[key] = transfer_id
            self.counters["reserved"] += 1
            return transfer_id, generation, extents

    def mark_writing(self, transfer_id: str) -> None:
        with self._lock:
            obj = self._get(transfer_id)
            self._require(obj, ObjectState.RESERVED)
            obj.state = ObjectState.WRITING

    def publish(
        self, transfer_id: str, generation: int, manifest: HandoffManifest
    ) -> None:
        """READY: only after the producer reports every write succeeded.
        Publication is idempotent for the same generation."""
        with self._lock:
            obj = self._get(transfer_id)
            self._check_generation(obj, generation)
            if obj.state == ObjectState.READY and obj.manifest is not None:
                if obj.manifest != manifest:
                    raise HandoffError(
                        "a different manifest is already published for this generation"
                    )
                return
            self._require(obj, ObjectState.WRITING, ObjectState.RESERVED)
            if manifest.extents != obj.extents:
                raise HandoffError("manifest extents differ from the reservation")
            if manifest.transfer_id != transfer_id or manifest.generation != generation:
                raise HandoffError("manifest identity differs from the reservation")
            if (
                manifest.session_epoch != self.session_epoch
                or manifest.request_id != obj.request_id
                or manifest.producer != obj.producer
                or manifest.tp_rank != obj.tp_rank
            ):
                raise HandoffError("manifest route differs from the reservation")
            obj.manifest = manifest
            obj.state = ObjectState.READY
            obj.producer_drained = True  # writes completed by definition
            self.counters["published"] += 1

    def fail(self, transfer_id: str, generation: int, reason: str) -> None:
        """A producer or consumer error: never publish, drain, then reclaim."""
        with self._lock:
            obj = self._get(transfer_id)
            self._check_generation(obj, generation)
            obj.error = reason
            obj.state = ObjectState.DRAINING
            self.counters["cancelled"] += 1

    def producer_drained(self, transfer_id: str) -> None:
        with self._lock:
            obj = self._get(transfer_id)
            obj.producer_drained = True

    # ---- consumer side -----------------------------------------------------

    def lookup(
        self, request_id: str, producer: str, tp_rank: int
    ) -> HandoffManifest | None:
        """Return the READY manifest for one producer request and TP rank."""
        with self._lock:
            tid = self._by_request.get((producer, request_id, tp_rank))
            if tid is None:
                return None
            obj = self._objects[tid]
            if obj.state in (ObjectState.READY, ObjectState.READING) and obj.manifest:
                return obj.manifest
            return None

    def state_of(self, transfer_id: str) -> ObjectState:
        with self._lock:
            return self._get(transfer_id).state

    def lease(
        self, transfer_id: str, generation: int, consumer: str
    ) -> tuple[str, HandoffManifest]:
        """Grant a consumer the right to read; READY -> READING."""
        with self._lock:
            obj = self._get(transfer_id)
            self._check_generation(obj, generation)
            self._require(obj, ObjectState.READY, ObjectState.READING)
            assert obj.manifest is not None
            existing = obj.leases.get(consumer)
            if existing is not None and not existing.acked:
                return existing.lease_id, obj.manifest
            if obj.leases:
                raise HandoffError(
                    f"{transfer_id} is already assigned to another decode rank"
                )
            now = time.monotonic()
            lease = Lease(consumer, uuid.uuid4().hex, now, now + self.lease_timeout_s)
            obj.leases[consumer] = lease
            obj.state = ObjectState.READING
            self.counters["leases"] += 1
            return lease.lease_id, obj.manifest

    def ack(self, transfer_id: str, generation: int, consumer: str) -> None:
        """The consumer restored the bytes and its reads drained. Duplicates
        are idempotent; a stale generation cannot release newer extents."""
        with self._lock:
            obj = self._get(transfer_id)
            self._check_generation(obj, generation)
            lease = obj.leases.get(consumer)
            if lease is None:
                raise HandoffError(f"{consumer} holds no lease on {transfer_id}")
            if lease.acked:
                self.counters["duplicate_acks"] += 1
                return
            lease.acked = True
            lease.drained = True
            self.counters["acks"] += 1
            if len(obj.leases) == 1 and all(
                lease.acked for lease in obj.leases.values()
            ):
                obj.state = ObjectState.ACKED

    def consumer_drained(self, transfer_id: str, consumer: str) -> None:
        with self._lock:
            obj = self._get(transfer_id)
            lease = obj.leases.get(consumer)
            if lease is not None:
                lease.drained = True

    # ---- reclamation -------------------------------------------------------

    def expire_leases(self, now: float | None = None) -> int:
        """Expired leases are recorded, not freed: the reader may still have
        I/O in flight. The holder must report drain (or be declared gone
        through force_release) before the extents move."""
        now = time.monotonic() if now is None else now
        expired = 0
        with self._lock:
            for obj in self._objects.values():
                for lease in obj.leases.values():
                    if not lease.acked and now > lease.expires_at:
                        expired += 1
            self.counters["lease_expired"] += expired
        return expired

    def force_release(self, transfer_id: str, consumer: str, reason: str) -> None:
        """An operator or a liveness check declares a consumer gone. Its
        lease is treated as acknowledged and drained."""
        with self._lock:
            obj = self._get(transfer_id)
            lease = obj.leases.get(consumer)
            if lease is None:
                return
            lease.acked = True
            lease.drained = True
            obj.error = obj.error or f"force_release({consumer}): {reason}"
            if len(obj.leases) == 1 and all(
                lease.acked for lease in obj.leases.values()
            ):
                obj.state = ObjectState.ACKED

    def reclaim(self) -> int:
        """Free the extents of every object whose producer and consumers
        have all drained: ACKED objects, and DRAINING ones once quiescent."""
        freed = 0
        with self._lock:
            for tid, obj in list(self._objects.items()):
                if obj.state == ObjectState.ACKED or obj.state == ObjectState.DRAINING:
                    ready = obj.producer_drained and all(
                        lease.drained for lease in obj.leases.values()
                    )
                else:
                    continue
                if not ready:
                    continue
                for e in obj.extents:
                    self.allocator.free(e)
                obj.state = ObjectState.RECLAIMABLE
                obj.extents = []
                key = (obj.producer, obj.request_id, obj.tp_rank)
                if self._by_request.get(key) == tid:
                    del self._by_request[key]
                del self._objects[tid]
                freed += 1
                self.counters["reclaimed"] += 1
        return freed

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "session_epoch": self.session_epoch,
                "objects": {
                    tid: {
                        "request_id": o.request_id,
                        "producer": o.producer,
                        "tp_rank": o.tp_rank,
                        "generation": o.generation,
                        "state": o.state.value,
                        "bytes": sum(e.length for e in o.extents),
                        "leases": {
                            consumer: lease.acked
                            for consumer, lease in o.leases.items()
                        },
                        "error": o.error,
                    }
                    for tid, o in self._objects.items()
                },
                "free_bytes": self.allocator.free_bytes,
                "allocated_bytes": self.allocator.allocated_bytes,
                "counters": dict(self.counters),
            }

    # ---- internals ---------------------------------------------------------

    def _get(self, transfer_id: str) -> HandoffObject:
        obj = self._objects.get(transfer_id)
        if obj is None:
            raise HandoffError(f"unknown transfer {transfer_id}")
        return obj

    @staticmethod
    def _require(obj: HandoffObject, *states: ObjectState) -> None:
        if obj.state not in states:
            raise HandoffError(
                f"{obj.transfer_id} is {obj.state.value}, expected "
                f"{[state.value for state in states]}"
            )

    def _check_generation(self, obj: HandoffObject, generation: int) -> None:
        if generation != obj.generation:
            self.counters["stale_rejected"] += 1
            raise StaleGeneration(
                f"{obj.transfer_id}: generation {generation} != current "
                f"{obj.generation}"
            )


# ---- Unix-socket protocol ---------------------------------------------------
#
# One JSON object per line: {"op": ..., "args": {...}} -> {"ok": true, "result": ...}
# or {"ok": false, "error": "...", "kind": "StaleGeneration"|"NotReady"|"HandoffError"}.


def _encode_extents(extents: list[Extent]) -> list[dict]:
    return [{"offset": e.offset, "length": e.length} for e in extents]


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        coord: StorageHandoffCoordinator = self.server.coordinator  # type: ignore[attr-defined]
        for line in self.rfile:
            try:
                msg = json.loads(line)
                result = _dispatch(coord, msg["op"], msg.get("args", {}))
                reply = {"ok": True, "result": result}
            except HandoffError as e:
                reply = {"ok": False, "error": str(e), "kind": type(e).__name__}
            except Exception as e:  # noqa: BLE001
                reply = {
                    "ok": False,
                    "error": f"{type(e).__name__}: {e}",
                    "kind": "HandoffError",
                }
            self.wfile.write((json.dumps(reply) + "\n").encode())
            self.wfile.flush()


def _dispatch(coord: StorageHandoffCoordinator, op: str, args: dict):
    if op == "reserve":
        tid, gen, extents = coord.reserve(
            args["request_id"],
            args["producer"],
            args["tp_rank"],
            args["lengths"],
        )
        return {
            "transfer_id": tid,
            "generation": gen,
            "extents": _encode_extents(extents),
            "session_epoch": coord.session_epoch,
        }
    if op == "mark_writing":
        coord.mark_writing(args["transfer_id"])
        return None
    if op == "publish":
        coord.publish(
            args["transfer_id"],
            args["generation"],
            HandoffManifest.from_json(args["manifest"]),
        )
        return None
    if op == "fail":
        coord.fail(args["transfer_id"], args["generation"], args["reason"])
        return None
    if op == "producer_drained":
        coord.producer_drained(args["transfer_id"])
        return None
    if op == "lookup":
        m = coord.lookup(args["request_id"], args["producer"], args["tp_rank"])
        return None if m is None else m.to_json()
    if op == "state_of":
        return coord.state_of(args["transfer_id"]).value
    if op == "lease":
        lease_id, m = coord.lease(
            args["transfer_id"], args["generation"], args["consumer"]
        )
        return {"lease_id": lease_id, "manifest": m.to_json()}
    if op == "ack":
        coord.ack(args["transfer_id"], args["generation"], args["consumer"])
        return None
    if op == "consumer_drained":
        coord.consumer_drained(args["transfer_id"], args["consumer"])
        return None
    if op == "expire_leases":
        return coord.expire_leases()
    if op == "force_release":
        coord.force_release(args["transfer_id"], args["consumer"], args["reason"])
        return None
    if op == "reclaim":
        return coord.reclaim()
    if op == "snapshot":
        return coord.snapshot()
    raise HandoffError(f"unknown op {op}")


class CoordinatorServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True
    # Every worker rank of both engines connects at start-up; the default
    # backlog of 5 refuses some of them.
    request_queue_size = 256

    def __init__(self, path: str, coordinator: StorageHandoffCoordinator):
        if os.path.exists(path):
            os.unlink(path)
        super().__init__(path, _Handler)
        self.coordinator = coordinator
        self.path = path

    def serve_in_thread(self) -> threading.Thread:
        t = threading.Thread(
            target=self.serve_forever, name="handoff-coordinator", daemon=True
        )
        t.start()
        return t


class CoordinatorClient:
    """Blocking client; one connection, one request at a time."""

    def __init__(self, path: str, timeout_s: float = 30.0):
        self.path = path
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.settimeout(timeout_s)
        self._sock.connect(path)
        self._file = self._sock.makefile("rwb")
        self._lock = threading.Lock()

    def close(self) -> None:
        self._file.close()
        self._sock.close()

    def call(self, op: str, **args):
        with self._lock:
            self._file.write((json.dumps({"op": op, "args": args}) + "\n").encode())
            self._file.flush()
            line = self._file.readline()
        if not line:
            raise HandoffError("coordinator closed the connection")
        reply = json.loads(line)
        if reply["ok"]:
            return reply["result"]
        kind = reply.get("kind", "HandoffError")
        exc = {
            "StaleGeneration": StaleGeneration,
            "NotReady": NotReady,
            "OutOfSpace": OutOfSpace,
        }.get(kind, HandoffError)
        raise exc(reply["error"])

    # typed conveniences
    def reserve(
        self, request_id: str, producer: str, tp_rank: int, lengths: list[int]
    ) -> tuple[str, int, list[Extent], str]:
        r = self.call(
            "reserve",
            request_id=request_id,
            producer=producer,
            tp_rank=tp_rank,
            lengths=lengths,
        )
        return (
            r["transfer_id"],
            r["generation"],
            [Extent(**e) for e in r["extents"]],
            r["session_epoch"],
        )

    def publish(
        self, transfer_id: str, generation: int, manifest: HandoffManifest
    ) -> None:
        self.call(
            "publish",
            transfer_id=transfer_id,
            generation=generation,
            manifest=manifest.to_json(),
        )

    def lookup(
        self, request_id: str, producer: str, tp_rank: int
    ) -> HandoffManifest | None:
        r = self.call(
            "lookup", request_id=request_id, producer=producer, tp_rank=tp_rank
        )
        return None if r is None else HandoffManifest.from_json(r)

    def lease(
        self, transfer_id: str, generation: int, consumer: str
    ) -> tuple[str, HandoffManifest]:
        r = self.call(
            "lease", transfer_id=transfer_id, generation=generation, consumer=consumer
        )
        return r["lease_id"], HandoffManifest.from_json(r["manifest"])


def main(argv: list[str] | None = None) -> int:
    """Run the coordinator as a daemon on a Unix socket."""
    import argparse
    import signal

    parser = argparse.ArgumentParser(description="storage handoff coordinator")
    parser.add_argument("--socket", required=True)
    parser.add_argument(
        "--base", type=int, required=True, help="first byte of the range"
    )
    parser.add_argument("--size", type=int, required=True, help="bytes in the range")
    parser.add_argument("--align", type=int, default=4096)
    parser.add_argument("--lease-timeout", type=float, default=60.0)
    parser.add_argument(
        "--snapshot", default="", help="write a JSON snapshot here on exit"
    )
    args = parser.parse_args(argv)
    coord = StorageHandoffCoordinator(
        args.base, args.size, args.align, args.lease_timeout
    )
    server = CoordinatorServer(args.socket, coord)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    server.serve_in_thread()
    print(
        f"coordinator on {args.socket} epoch={coord.session_epoch} "
        f"range=[{args.base}, +{args.size})",
        flush=True,
    )
    while not stop.wait(1.0):
        coord.expire_leases()
        coord.reclaim()
    if args.snapshot:
        with open(args.snapshot, "w") as f:
            json.dump(coord.snapshot(), f, indent=2)
    server.shutdown()
    server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
