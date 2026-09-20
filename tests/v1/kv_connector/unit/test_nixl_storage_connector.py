# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side pieces of the storage handoff connector that need no device:
configuration parsing and the bounded wait for namespace space."""

from types import SimpleNamespace

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.nixl_storage.connector import (
    PARAM_KEY,
    HandoffConfig,
    _Scheduler,
    _Worker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl_storage.coordinator import (
    OutOfSpace,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl_storage.manifest import (
    Extent,
    HandoffManifest,
    LayerSlice,
)


def test_config_defaults_and_hooks():
    cfg = HandoffConfig.from_extra({"device_name": "/dev/ng0n1", "range_len": 4096})
    assert cfg.reserve_timeout_s == 30.0
    assert cfg.inject_store_error_nth == 0 and cfg.inject_load_error_nth == 0
    cfg = HandoffConfig.from_extra(
        {
            "device_name": "/dev/ng0n1",
            "range_len": 4096,
            "reserve_timeout_s": "0.5",
            "inject_store_error_nth": 2,
        }
    )
    assert cfg.reserve_timeout_s == 0.5 and cfg.inject_store_error_nth == 2
    with pytest.raises(ValueError):
        HandoffConfig.from_extra({"device_name": "/dev/ng0n1"})


def test_config_ties_staging_exporter_to_buffer_mode():
    base = {"device_name": "/dev/nvme0n1", "range_len": 4096, "async": "io_uring"}
    cfg = HandoffConfig.from_extra(
        {**base, "staging": "system_heap", "buffer_mode": "dmabuf"}
    )
    assert cfg.staging == "system_heap" and cfg.dmabuf_chunk_bytes == 1 << 30
    with pytest.raises(ValueError):  # dma-buf staging without the dma-buf transport
        HandoffConfig.from_extra({**base, "staging": "udmabuf"})
    with pytest.raises(ValueError):  # dma-buf transport with host staging
        HandoffConfig.from_extra({**base, "buffer_mode": "dmabuf"})
    with pytest.raises(ValueError):  # registered modes need io_uring on the block node
        HandoffConfig.from_extra(
            {
                "device_name": "/dev/ng0n1",
                "range_len": 4096,
                "staging": "system_heap",
                "buffer_mode": "dmabuf",
            }
        )
    with pytest.raises(ValueError):
        HandoffConfig.from_extra({**base, "staging": "bogus"})


def _manifest(rank: int, epoch: str = "epoch") -> HandoffManifest:
    return HandoffManifest(
        session_epoch=epoch,
        request_id="prefill-request",
        transfer_id=f"transfer-{rank}",
        generation=0,
        model_id="model#dtype",
        kv_dtype="torch.bfloat16",
        kv_scales=None,
        tp_rank=rank,
        tp_size=2,
        block_size=16,
        num_tokens=32,
        num_blocks=2,
        layout="16x2x64",
        namespace_serial="serial",
        namespace_wwid="wwid",
        layers=[LayerSlice("layer.0", Extent(rank * 8192, 8192), 2048, 2)],
        producer="prefill-engine",
    )


class _LookupCoord:
    def __init__(self, manifests):
        self.manifests = manifests
        self.calls = []

    def lookup(self, request_id, producer, tp_rank):
        self.calls.append((request_id, producer, tp_rank))
        return self.manifests.get(tp_rank)


def _scheduler(manifests) -> _Scheduler:
    scheduler = _Scheduler.__new__(_Scheduler)
    scheduler.conn = SimpleNamespace(
        is_consumer=True,
        tp_size=2,
        block_size=16,
        model_id="model#dtype",
        config=HandoffConfig(device_name="file", range_len=1 << 20),
    )
    scheduler.coord = _LookupCoord(manifests)
    scheduler.session_epoch = "epoch"
    scheduler.pending_stores = {}
    scheduler.pending_loads = {}
    scheduler.matched = {}
    scheduler.lookup_started = {}
    scheduler.counters = {
        "stores_ordered": 0,
        "loads_ordered": 0,
        "lookup_timeouts": 0,
        "recompute_fallbacks": 0,
        "lookup_waits": 0,
    }
    return scheduler


def _decode_request(epoch="epoch"):
    return SimpleNamespace(
        request_id="decode-request",
        prompt_token_ids=list(range(40)),
        kv_transfer_params={
            PARAM_KEY: {
                "producer_request_id": "prefill-request",
                "producer_engine_id": "prefill-engine",
                "session_epoch": epoch,
            }
        },
    )


def test_scheduler_waits_for_every_tp_manifest():
    scheduler = _scheduler({0: _manifest(0)})
    assert scheduler.get_num_new_matched_tokens(_decode_request(), 0) == (
        None,
        False,
    )
    assert scheduler.coord.calls == [
        ("prefill-request", "prefill-engine", 0),
        ("prefill-request", "prefill-engine", 1),
    ]


def test_scheduler_routes_all_tp_manifests_by_engine_and_epoch():
    scheduler = _scheduler({0: _manifest(0), 1: _manifest(1)})
    assert scheduler.get_num_new_matched_tokens(_decode_request(), 0) == (32, True)
    manifests, matched, computed = scheduler.matched["decode-request"]
    assert set(manifests) == {0, 1} and matched == 32 and computed == 0
    with pytest.raises(RuntimeError, match="coordinator epoch changed"):
        scheduler.get_num_new_matched_tokens(_decode_request("stale"), 0)


class _Coord:
    def __init__(self, refusals: int):
        self.refusals = refusals
        self.calls = 0

    def reserve(self, request_id, producer, tp_rank, lengths):
        self.calls += 1
        if self.calls <= self.refusals:
            raise OutOfSpace("no hole")
        return "tid", 1, [Extent(offset=0, length=sum(lengths))], "epoch"


class _Adapter:
    def object_lengths(self, n):
        return [4096 * n]


def _worker(refusals: int, timeout_s: float) -> _Worker:
    w = _Worker.__new__(_Worker)
    w.cfg = HandoffConfig(
        device_name="/dev/ng0n1", range_len=1 << 20, reserve_timeout_s=timeout_s
    )
    w.coord = _Coord(refusals)
    w.conn = SimpleNamespace(engine_id="prefill")
    w.rank = 0
    w.adapter = _Adapter()
    w.counters = {
        "reserve_waits": 0,
        "reserve_wait_ns": 0,
        "arena_waits": 0,
        "arena_wait_ns": 0,
    }
    w.arena = None
    w.polls = 0

    def poll():
        w.polls += 1

    w._poll_transfers = poll
    return w


def test_reserve_retries_while_full_and_keeps_polling():
    w = _worker(refusals=3, timeout_s=5.0)
    t: dict[str, int] = {}
    tid, gen, extents = w._reserve("r1", 2, t)
    assert tid == "tid" and gen == 1 and extents[0].length == 8192
    assert w.coord.calls == 4 and w.polls == 3
    assert w.counters["reserve_waits"] == 1 and w.counters["reserve_wait_ns"] > 0
    assert "reserve_wait_start_ns" in t


def test_reserve_gives_up_after_timeout():
    w = _worker(refusals=10**6, timeout_s=0.05)
    with pytest.raises(OutOfSpace):
        w._reserve("r1", 1, {})
    assert w.polls >= 1 and w.counters["reserve_waits"] == 1


def test_reserve_without_pressure_does_not_wait():
    w = _worker(refusals=0, timeout_s=1.0)
    t: dict[str, int] = {}
    w._reserve("r1", 1, t)
    assert (
        w.polls == 0
        and w.counters["reserve_waits"] == 0
        and "reserve_wait_start_ns" not in t
    )


def test_stage_waits_for_arena_space_then_gives_up():
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl_storage.storage import (
        HUGE_2M,
        StagingArena,
    )

    w = _worker(refusals=0, timeout_s=0.05)
    w.arena = StagingArena(HUGE_2M, "host", chunk_bytes=HUGE_2M)
    try:
        t: dict[str, int] = {}
        held = w._stage("r1", HUGE_2M, t)
        assert held.length == HUGE_2M and w.polls == 0
        with pytest.raises(OutOfSpace):
            w._stage("r2", 4096, {})
        assert w.polls >= 1 and w.counters["arena_waits"] == 1
        w.arena.free(held)
        assert w._stage("r3", 4096, {}).offset == 0
    finally:
        w.arena.close()
