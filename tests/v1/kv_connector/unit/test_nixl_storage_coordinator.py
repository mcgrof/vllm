# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the storage handoff coordinator and manifest (no GPU,
no NIXL, no device)."""

import os
import tempfile
import threading

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.nixl_storage.coordinator import (
    CoordinatorClient,
    CoordinatorServer,
    ExtentAllocator,
    HandoffError,
    OutOfSpace,
    StaleGeneration,
    StorageHandoffCoordinator,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl_storage.manifest import (
    Extent,
    HandoffManifest,
    LayerSlice,
    ObjectState,
)

pytestmark = pytest.mark.skip_global_cleanup

MiB = 1 << 20


def make_manifest(
    coord,
    transfer_id,
    generation,
    extents,
    request_id="r1",
    producer="prefill",
    tp_rank=0,
):
    layers = [
        LayerSlice(
            layer_name=f"layers.{i}",
            extent=e,
            bytes_per_block=4096,
            regions_per_block=2,
        )
        for i, e in enumerate(extents)
    ]
    return HandoffManifest(
        session_epoch=coord.session_epoch,
        request_id=request_id,
        transfer_id=transfer_id,
        generation=generation,
        model_id="test-model#cfg",
        kv_dtype="bfloat16",
        kv_scales=None,
        tp_rank=tp_rank,
        tp_size=1,
        block_size=16,
        num_tokens=100,
        num_blocks=7,
        layout="LBNHC",
        namespace_serial="nixlpd0",
        namespace_wwid="nvme.wwid",
        layers=layers,
        producer=producer,
    )


# --- allocator -------------------------------------------------------------------


def test_allocator_first_fit_and_coalesce():
    a = ExtentAllocator(base=0, size=16 * MiB, align=4096)
    x = a.allocate(4 * MiB)
    y = a.allocate(4 * MiB)
    z = a.allocate(4 * MiB)
    assert (x.offset, y.offset, z.offset) == (0, 4 * MiB, 8 * MiB)
    a.free(y)
    # A 4 MiB hole reopens in the middle and is used before the tail.
    assert a.allocate(4 * MiB).offset == 4 * MiB
    a.free(x)
    a.free(z)
    # The remaining hole after freeing x and z does not coalesce across y.
    with pytest.raises(OutOfSpace):
        a.allocate(9 * MiB)
    a.free(Extent(4 * MiB, 4 * MiB))
    assert a.allocate(12 * MiB).offset == 0
    assert a.free_bytes == 4 * MiB


def test_allocator_rounds_up_and_rejects_double_free():
    a = ExtentAllocator(base=4096, size=1 * MiB, align=4096)
    e = a.allocate(100)
    assert e.length == 4096 and e.offset == 4096
    a.free(e)
    with pytest.raises(HandoffError):
        a.free(e)


# --- lifecycle -------------------------------------------------------------------


def test_reserve_publish_lease_ack_reclaim():
    c = StorageHandoffCoordinator(base=0, size=64 * MiB, align=4096)
    tid, gen, extents = c.reserve("r1", "prefill", 0, [1 * MiB, 2 * MiB])
    assert gen == 0 and len(extents) == 2
    assert c.state_of(tid) == ObjectState.RESERVED
    assert c.lookup("r1", "prefill", 0) is None  # not published

    c.mark_writing(tid)
    m = make_manifest(c, tid, gen, extents)
    c.publish(tid, gen, m)
    assert c.state_of(tid) == ObjectState.READY
    assert c.lookup("r1", "prefill", 0).transfer_id == tid
    c.publish(tid, gen, m)  # idempotent
    changed = make_manifest(c, tid, gen, extents)
    changed.num_tokens += 1
    with pytest.raises(HandoffError, match="different manifest"):
        c.publish(tid, gen, changed)

    lease_id, got = c.lease(tid, gen, consumer="decode-0")
    assert got.total_bytes == 3 * MiB and c.state_of(tid) == ObjectState.READING
    assert c.lease(tid, gen, "decode-0")[0] == lease_id  # same lease back
    assert c.reclaim() == 0  # a leased object is not reclaimable

    c.ack(tid, gen, "decode-0")
    assert c.state_of(tid) == ObjectState.ACKED
    c.ack(tid, gen, "decode-0")  # duplicate
    assert c.counters["duplicate_acks"] == 1
    assert c.reclaim() == 1
    assert c.allocator.free_bytes == 64 * MiB
    assert c.lookup("r1", "prefill", 0) is None


def test_publish_requires_the_reserved_extents_and_identity():
    c = StorageHandoffCoordinator(base=0, size=8 * MiB)
    tid, gen, extents = c.reserve("r1", "prefill", 0, [1 * MiB])
    wrong = make_manifest(c, tid, gen, [Extent(2 * MiB, 1 * MiB)])
    with pytest.raises(HandoffError):
        c.publish(tid, gen, wrong)
    other = make_manifest(c, "not-the-tid", gen, extents)
    with pytest.raises(HandoffError):
        c.publish(tid, gen, other)
    wrong_route = make_manifest(c, tid, gen, extents, producer="other-prefill")
    with pytest.raises(HandoffError):
        c.publish(tid, gen, wrong_route)
    assert c.state_of(tid) == ObjectState.RESERVED


def test_request_index_isolated_by_producer_and_tp_rank():
    c = StorageHandoffCoordinator(base=0, size=8 * MiB)
    t0, g0, e0 = c.reserve("r1", "prefill-a", 0, [1 * MiB])
    t1, g1, e1 = c.reserve("r1", "prefill-a", 1, [1 * MiB])
    tb, gb, eb = c.reserve("r1", "prefill-b", 0, [1 * MiB])
    assert g0 == g1 == gb == 0
    c.publish(t0, g0, make_manifest(c, t0, g0, e0, producer="prefill-a"))
    c.publish(
        t1,
        g1,
        make_manifest(c, t1, g1, e1, producer="prefill-a", tp_rank=1),
    )
    c.publish(tb, gb, make_manifest(c, tb, gb, eb, producer="prefill-b"))
    assert c.lookup("r1", "prefill-a", 0).transfer_id == t0
    assert c.lookup("r1", "prefill-a", 1).transfer_id == t1
    assert c.lookup("r1", "prefill-b", 0).transfer_id == tb


def test_rank_shard_has_exactly_one_decode_consumer():
    c = StorageHandoffCoordinator(base=0, size=8 * MiB)
    tid, gen, extents = c.reserve("r1", "prefill", 0, [1 * MiB])
    c.publish(tid, gen, make_manifest(c, tid, gen, extents))
    c.lease(tid, gen, "decode-a:0")
    with pytest.raises(HandoffError):
        c.lease(tid, gen, "decode-b:0")
    c.ack(tid, gen, "decode-a:0")
    assert c.reclaim() == 1


def test_stale_generation_cannot_release_newer_extents():
    c = StorageHandoffCoordinator(base=0, size=8 * MiB)
    tid0, gen0, ext0 = c.reserve("r1", "prefill", 0, [1 * MiB])
    c.publish(tid0, gen0, make_manifest(c, tid0, gen0, ext0))
    lease0, _ = c.lease(tid0, gen0, "d")
    # A rewrite of the same request is a new generation with new extents;
    # the old one stays leased and READY for its reader.
    tid1, gen1, ext1 = c.reserve("r1", "prefill", 0, [1 * MiB])
    assert gen1 == 1 and ext1[0].offset != ext0[0].offset
    assert c.lookup("r1", "prefill", 0) is None  # newest one not published yet
    with pytest.raises(StaleGeneration):
        c.ack(tid1, gen0, "d")
    with pytest.raises(StaleGeneration):
        c.publish(tid1, gen0, make_manifest(c, tid1, gen0, ext1))
    assert c.counters["stale_rejected"] == 2
    c.ack(tid0, gen0, "d")
    assert c.reclaim() == 1
    assert c.allocator.allocated_bytes == 1 * MiB  # generation 1 still holds its extent


def test_lease_expiry_records_but_does_not_free():
    c = StorageHandoffCoordinator(base=0, size=8 * MiB, lease_timeout_s=0.0)
    tid, gen, ext = c.reserve("r1", "prefill", 0, [1 * MiB])
    c.publish(tid, gen, make_manifest(c, tid, gen, ext))
    c.lease(tid, gen, "d")
    assert c.expire_leases(now=1e12) == 1
    assert c.reclaim() == 0
    assert c.allocator.allocated_bytes == 1 * MiB
    c.force_release(tid, "d", reason="consumer process gone")
    assert c.reclaim() == 1
    assert c.allocator.allocated_bytes == 0


def test_failure_drains_before_reclaim():
    c = StorageHandoffCoordinator(base=0, size=8 * MiB)
    tid, gen, ext = c.reserve("r1", "prefill", 0, [1 * MiB])
    c.mark_writing(tid)
    c.fail(tid, gen, "write refused")
    assert c.state_of(tid) == ObjectState.DRAINING
    assert c.lookup("r1", "prefill", 0) is None
    assert c.reclaim() == 0  # producer has not reported drain
    c.producer_drained(tid)
    assert c.reclaim() == 1


def test_out_of_space_rolls_back_partial_reservation():
    c = StorageHandoffCoordinator(base=0, size=4 * MiB)
    with pytest.raises(OutOfSpace):
        c.reserve("r1", "prefill", 0, [3 * MiB, 3 * MiB])
    assert c.allocator.allocated_bytes == 0
    assert c.lookup("r1", "prefill", 0) is None


def test_manifest_round_trip_and_identity():
    c = StorageHandoffCoordinator(base=0, size=8 * MiB)
    tid, gen, ext = c.reserve("r1", "prefill", 0, [1 * MiB, 1 * MiB])
    m = make_manifest(c, tid, gen, ext)
    back = HandoffManifest.from_json(m.to_json())
    assert back == m
    other = make_manifest(c, tid, gen, ext)
    other.kv_dtype = "float8_e4m3"
    assert not m.same_identity(other)
    other = make_manifest(c, tid, gen, ext)
    other.tp_rank = 1
    assert not m.same_identity(other)
    other = make_manifest(c, tid, gen, ext)
    other.layers[0].bytes_per_block *= 2
    assert not m.same_identity(other)


# --- socket protocol ---------------------------------------------------------------


def test_server_and_client_round_trip():
    coord = StorageHandoffCoordinator(base=0, size=16 * MiB)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "handoff.sock")
        server = CoordinatorServer(path, coord)
        server.serve_in_thread()
        try:
            producer = CoordinatorClient(path)
            consumer = CoordinatorClient(path)
            tid, gen, extents, epoch = producer.reserve("r7", "prefill", 0, [2 * MiB])
            assert epoch == coord.session_epoch
            assert consumer.lookup("r7", "prefill", 0) is None
            producer.call("mark_writing", transfer_id=tid)
            producer.publish(
                tid, gen, make_manifest(coord, tid, gen, extents, request_id="r7")
            )
            got = consumer.lookup("r7", "prefill", 0)
            assert got is not None and got.transfer_id == tid
            lease_id, m = consumer.lease(tid, gen, "decode-0")
            assert m.extents == extents
            with pytest.raises(StaleGeneration):
                consumer.call(
                    "ack", transfer_id=tid, generation=gen + 1, consumer="decode-0"
                )
            consumer.call("ack", transfer_id=tid, generation=gen, consumer="decode-0")
            assert producer.call("reclaim") == 1
            snap = consumer.call("snapshot")
            assert snap["counters"]["acks"] == 1 and snap["allocated_bytes"] == 0
            # Concurrent clients do not corrupt the allocator, and every
            # connection is accepted.
            errors: list[BaseException] = []

            def worker(i):
                try:
                    cl = CoordinatorClient(path)
                    t, g, e, _ = cl.reserve(f"req-{i}", "prefill", 0, [256 * 1024])
                    cl.publish(
                        t, g, make_manifest(coord, t, g, e, request_id=f"req-{i}")
                    )
                    cl.lease(t, g, "c")
                    cl.call("ack", transfer_id=t, generation=g, consumer="c")
                    cl.close()
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            assert not errors, errors
            assert producer.call("reclaim") == 16
            assert coord.allocator.allocated_bytes == 0
            producer.close()
            consumer.close()
        finally:
            server.shutdown()
            server.server_close()
