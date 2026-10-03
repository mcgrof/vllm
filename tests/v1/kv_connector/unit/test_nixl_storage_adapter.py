# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The staging arena and descriptor planner without a device, plus a
device-gated round trip through a selected NIXL storage backend.

NIXL_STORAGE_DEVICE names a file or device the process may write to,
NIXL_STORAGE_BACKEND selects the backend, and NIXL_STORAGE_STAGING selects
the exporter. The old XNVME_NVM variables remain accepted where applicable.
Without a target the device test skips.
"""

import ctypes
import os
from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.nixl_storage import storage
from vllm.distributed.kv_transfer.kv_connector.v1.nixl_storage.coordinator import (
    OutOfSpace,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl_storage.manifest import (
    Extent,
    LayerSlice,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl_storage.storage import (
    HUGE_2M,
    LayerGeometry,
    NixlStorageAdapter,
    StagingArena,
    StorageQuiescenceError,
    aligned_empty,
    object_layout,
    pack_blocks,
    plan_transfer,
    unpack_blocks,
)

pytestmark = pytest.mark.skip_global_cleanup

DEVICE = os.environ.get("NIXL_STORAGE_DEVICE") or os.environ.get(
    "NIXL_XNVME_NVM_DEVICE"
)
BACKEND = os.environ.get("NIXL_STORAGE_BACKEND", "IO_URING")
ASYNC = os.environ.get("NIXL_XNVME_NVM_ASYNC", "io_uring")
STAGING = os.environ.get("NIXL_STORAGE_STAGING", "host")
MiB = 1 << 20


def make_cache(
    num_layers, num_blocks, block_size=16, heads=2, dim=64, split_first=True
):
    caches = {}
    for i in range(num_layers):
        shape = (
            (2, num_blocks, block_size, heads, dim)
            if split_first
            else (num_blocks, 2, block_size, heads, dim)
        )
        caches[f"layers.{i}"] = aligned_empty(shape, torch.bfloat16)
        caches[f"layers.{i}"].zero_()
    return caches


def fill(caches, block_ids, seed):
    g = torch.Generator().manual_seed(seed)
    for name, t in caches.items():
        for b in block_ids:
            if t.shape[0] == 2:
                t[:, b] = torch.randn(t[:, b].shape, generator=g).to(torch.bfloat16)
            else:
                t[b] = torch.randn(t[b].shape, generator=g).to(torch.bfloat16)


def layers_for(geoms, num_blocks, first=4 * MiB):
    out, off = [], first
    for g in geoms:
        length = (g.object_bytes(num_blocks) + 4095) // 4096 * 4096
        out.append(
            LayerSlice(
                layer_name=g.layer_name,
                extent=Extent(off, length),
                bytes_per_block=g.block_bytes,
                regions_per_block=g.regions,
            )
        )
        off += length
    return out


def exporter_or_skip(name):
    try:
        return StagingArena(HUGE_2M, name, chunk_bytes=HUGE_2M)
    except OSError as e:
        pytest.skip(f"{name} unavailable here: {e}")


def test_aligned_empty_keeps_shape_stride_and_alignment():
    t = aligned_empty((2, 5, 16, 2, 64), torch.bfloat16)
    assert t.data_ptr() % 4096 == 0 and t.shape == (2, 5, 16, 2, 64)
    s = aligned_empty((5, 2, 8), torch.float16, stride=(16, 8, 1))
    assert s.stride() == (16, 8, 1) and s.data_ptr() % 4096 == 0


def test_geometry_regions_runs_and_layout():
    caches = make_cache(2, 8)
    g = LayerGeometry.from_tensor("l", caches["layers.0"])
    assert g.regions == 2 and g.num_blocks == 8 and g.split == 0
    assert g.block_bytes == 16 * 2 * 64 * 2 and g.inner_shape == (16, 2, 64)
    assert g.runs([2, 3, 4, 7]) == [(0, 2, 3), (3, 7, 1)]
    g2 = LayerGeometry.from_tensor("l", make_cache(1, 8, split_first=False)["layers.0"])
    assert g2.regions == 2 and g2.split == 1
    assert [
        v.shape[0]
        for v in g2.region_views(make_cache(1, 8, split_first=False)["layers.0"])
    ] == [8, 8]
    geoms = [LayerGeometry.from_tensor(n, t) for n, t in caches.items()]
    assert object_layout(geoms, 3) == [0, 3 * 2 * g.block_bytes]


def test_arena_is_contiguous_across_chunks_and_allocates_first_fit():
    arena = StagingArena(3 * HUGE_2M, "host", chunk_bytes=HUGE_2M)
    try:
        assert arena.nbytes == 3 * HUGE_2M and len(arena.chunks) == 3
        assert [c.addr - arena.base for c in arena.chunks] == [0, HUGE_2M, 2 * HUGE_2M]
        assert all(m == "" for _, _, _, m in arena.registrations())
        # A write through the tensor straddling a chunk boundary reads back
        # through the raw mapping.
        arena.tensor[HUGE_2M - 2 : HUGE_2M + 2] = torch.tensor(
            [1, 2, 3, 4], dtype=torch.uint8
        )
        raw = (ctypes.c_uint8 * 4).from_address(arena.base + HUGE_2M - 2)
        assert list(raw) == [1, 2, 3, 4]
        a = arena.allocate(HUGE_2M + 4096)
        assert a.offset == 0 and a.length == HUGE_2M + 4096
        pieces = arena.pieces(a.offset, a.length)
        assert [(p[0] - arena.base, p[1], p[2]) for p in pieces] == [
            (0, HUGE_2M, 0),
            (HUGE_2M, 4096, HUGE_2M),
        ]
        b = arena.allocate(HUGE_2M)
        assert b.offset == HUGE_2M + 4096
        with pytest.raises(OutOfSpace):
            arena.allocate(HUGE_2M)
        arena.free(a)
        c = arena.allocate(4096)
        assert c.offset == 0 and arena.allocated_bytes == HUGE_2M + 4096
        with pytest.raises(ValueError):
            arena.pieces(0, arena.nbytes + 1)
    finally:
        arena.close()


def test_arena_rejects_bad_parameters():
    with pytest.raises(ValueError):
        StagingArena(HUGE_2M, "bogus")
    with pytest.raises(ValueError):
        StagingArena(HUGE_2M, "host", chunk_bytes=4096)
    with pytest.raises(ValueError):
        StagingArena(0, "host")


def test_cuda_register_rolls_back_partial_registration(monkeypatch):
    arena = StagingArena(2 * HUGE_2M, "host", chunk_bytes=HUGE_2M)

    class Runtime:
        def __init__(self):
            self.registered = []
            self.unregistered = []

        def cudaHostRegister(self, addr, length, flags):
            del length, flags
            self.registered.append(addr)
            return 0 if len(self.registered) == 1 else 1

        def cudaHostUnregister(self, addr):
            self.unregistered.append(addr)
            return 0

    rt = Runtime()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "cudart", lambda: rt)
    monkeypatch.setattr(storage, "_clear_cuda_last_error", lambda: None)
    try:
        assert not arena.cuda_register()
        assert rt.registered == [c.addr for c in arena.chunks]
        assert rt.unregistered == [arena.chunks[0].addr]
        assert not arena.cuda_registered
        assert arena._cuda_registered_chunks == []
    finally:
        arena.close()


def test_cuda_registered_chunks_are_unregistered_on_close(monkeypatch):
    arena = StagingArena(2 * HUGE_2M, "host", chunk_bytes=HUGE_2M)

    class Runtime:
        def __init__(self):
            self.registered = []
            self.unregistered = []

        def cudaHostRegister(self, addr, length, flags):
            del length, flags
            self.registered.append(addr)
            return 0

        def cudaHostUnregister(self, addr):
            self.unregistered.append(addr)
            return 0

    rt = Runtime()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "cudart", lambda: rt)
    assert arena.cuda_register()
    expected = [c.addr for c in arena.chunks]
    arena.close()
    assert rt.registered == expected
    assert rt.unregistered == list(reversed(expected))
    assert not arena.cuda_registered
    assert arena._cuda_registered_chunks == []


class _FakeAgent:
    def __init__(self, fail_registration=0, fail_transfer=False, status="DONE"):
        self.fail_registration = fail_registration
        self.fail_transfer = fail_transfer
        self.registrations = []
        self.deregistrations = []
        self.released = []
        self.created_backends = []
        self.status = status

    def create_backend(self, name, params):
        self.created_backends.append((name, params))

    def register_memory(self, descs, mem_type, backends):
        del descs, mem_type, backends
        call = len(self.registrations) + 1
        if call == self.fail_registration:
            raise RuntimeError("injected registration failure")
        reg = f"reg-{call}"
        self.registrations.append(reg)
        return reg

    def deregister_memory(self, reg, backends):
        del backends
        self.deregistrations.append(reg)

    def query_memory(self, descs, backend, mem_type):
        del descs, backend, mem_type
        return [{"serial": "fake", "wwid": "fake.wwid"}]

    def get_xfer_descs(self, descs, mem_type):
        del mem_type
        return descs

    def initialize_xfer(self, op, local, remote, agent, metadata, backends):
        del op, local, remote, agent, metadata, backends
        return "handle"

    def transfer(self, handle):
        del handle
        if self.fail_transfer:
            raise RuntimeError("injected transfer failure")
        return self.status

    def check_xfer_state(self, handle):
        del handle
        return self.status

    def release_xfer_handle(self, handle):
        self.released.append(handle)


def test_adapter_rolls_back_partial_arena_registration():
    agent = _FakeAgent(fail_registration=3)
    adapter = NixlStorageAdapter("test", {}, 0, 4 * MiB, nixl_agent_cls=lambda _: agent)
    assert agent.created_backends == [("IO_URING", {})]
    arena = StagingArena(2 * HUGE_2M, "host", chunk_bytes=HUGE_2M)
    try:
        with pytest.raises(RuntimeError, match="injected registration failure"):
            adapter.register_arena(arena)
        assert adapter.arena is None and adapter._arena_regs == []
        assert agent.deregistrations == ["reg-2"]
    finally:
        adapter.close()
        arena.close()


def test_adapter_retains_owners_when_submission_outcome_is_unknown():
    agent = _FakeAgent(fail_transfer=True)
    adapter = NixlStorageAdapter("test", {}, 0, 4 * MiB, nixl_agent_cls=lambda _: agent)
    caches = make_cache(1, 4)
    adapter.register_cache(caches)
    arena = StagingArena(HUGE_2M, "host", chunk_bytes=HUGE_2M)
    adapter.register_arena(arena)
    layers = adapter.layer_slices([Extent(0, adapter.object_lengths(1)[0])], 1)
    region = arena.allocate(adapter.object_bytes(1))
    with pytest.raises(StorageQuiescenceError, match="submission outcome unknown"):
        adapter.start("WRITE", layers, 1, region)
    assert agent.released == []
    _assert_retained(adapter, arena, region, agent)


def test_adapter_times_out_without_releasing_nonterminal_transfer(monkeypatch):
    agent = _FakeAgent(status="PROC")
    adapter = NixlStorageAdapter(
        "test",
        {},
        0,
        4 * MiB,
        nixl_agent_cls=lambda _: agent,
        transfer_timeout_s=1.0,
    )
    caches = make_cache(1, 4)
    adapter.register_cache(caches)
    arena = StagingArena(HUGE_2M, "host", chunk_bytes=HUGE_2M)
    adapter.register_arena(arena)
    layers = adapter.layer_slices([Extent(0, adapter.object_lengths(1)[0])], 1)
    region = arena.allocate(adapter.object_bytes(1))
    pending = adapter.start("WRITE", layers, 1, region)
    monkeypatch.setattr(storage.time, "monotonic_ns", lambda: pending.deadline_ns)
    with pytest.raises(StorageQuiescenceError, match="terminal state"):
        adapter.poll(pending)
    with pytest.raises(RuntimeError, match="non-terminal"):
        adapter.release(pending)
    assert agent.released == []
    _assert_retained(adapter, arena, region, agent)


def _assert_retained(adapter, arena, region, agent):
    base = arena.base
    for operation in (
        adapter.close,
        arena.close,
        lambda: arena.free(region),
        lambda: arena.allocate(4096),
    ):
        with pytest.raises(StorageQuiescenceError):
            operation()
    assert arena.base == base
    assert arena.allocated_bytes == region.length
    assert agent.deregistrations == []


@pytest.mark.parametrize("failure", ["close", "poll", "unknown_status"])
def test_pending_transfer_prevents_teardown(failure, monkeypatch):
    agent = _FakeAgent(status="PROC")
    adapter = NixlStorageAdapter("test", {}, 0, 4 * MiB, nixl_agent_cls=lambda _: agent)
    adapter.register_cache(make_cache(1, 4))
    arena = StagingArena(HUGE_2M)
    adapter.register_arena(arena)
    region = arena.allocate(adapter.object_bytes(1))
    layers = adapter.layer_slices([Extent(0, region.length)], 1)
    pending = adapter.start("READ", layers, 1, region)
    if failure == "poll":

        def fail_poll(handle):
            raise OSError("completion unavailable")

        monkeypatch.setattr(agent, "check_xfer_state", fail_poll)
    elif failure == "unknown_status":
        agent.status = "UNRECOGNIZED"
    with pytest.raises(StorageQuiescenceError):
        if failure == "close":
            adapter.close()
        else:
            adapter.poll(pending)
    _assert_retained(adapter, arena, region, agent)
    assert agent.released == []


@pytest.mark.parametrize("operation", ["pack", "unpack"])
@pytest.mark.parametrize("sync_fails", [False, True])
def test_partial_copy_drains_or_retains_before_region_reuse(
    operation, sync_fails, monkeypatch
):
    agent = _FakeAgent()
    adapter = NixlStorageAdapter("test", {}, 0, 4 * MiB, nixl_agent_cls=lambda _: agent)
    arena = StagingArena(HUGE_2M)
    adapter.register_arena(arena)
    region = arena.allocate(4096)
    events = []

    def partial_copy(*args, **kwargs):
        events.append("queued")
        raise ValueError("later layer failed")

    def sync():
        events.append("sync")
        if sync_fails:
            raise RuntimeError("GPU unavailable")

    monkeypatch.setattr(storage, operation + "_blocks", partial_copy)
    monkeypatch.setattr(torch.accelerator, "synchronize", sync)
    with pytest.raises(StorageQuiescenceError if sync_fails else ValueError):
        getattr(adapter, operation)(
            {"layer": SimpleNamespace(is_cuda=True)}, [0], region
        )
    assert events == ["queued", "sync"]
    if sync_fails:
        _assert_retained(adapter, arena, region, agent)
    else:
        arena.free(region)
        adapter.close()
        adapter.close()
        arena.close()


def test_warmup_timeout_retains_registration_and_backing(monkeypatch):
    agent = _FakeAgent(status="PROC")
    adapter = NixlStorageAdapter(
        "test",
        {},
        0,
        4 * MiB,
        nixl_agent_cls=lambda _: agent,
        transfer_timeout_s=1,
    )
    arena = StagingArena(HUGE_2M)
    adapter.register_arena(arena)
    # The fake agent never uses the fd; select the real warm-up control path.
    arena.chunks[0].dmabuf_fd = os.open(os.devnull, os.O_RDONLY)
    ticks = iter([0, 0, 2_000_000_000])
    monkeypatch.setattr(storage.time, "monotonic_ns", lambda: next(ticks))
    with pytest.raises(StorageQuiescenceError, match="terminal state"):
        adapter.warm()
    with pytest.raises(StorageQuiescenceError):
        adapter.close()
    with pytest.raises(StorageQuiescenceError):
        arena.close()
    assert agent.released == [] and agent.deregistrations == []


def test_failed_registration_rollback_retains_backing(monkeypatch):
    agent = _FakeAgent(fail_registration=3)
    adapter = NixlStorageAdapter("test", {}, 0, 4 * MiB, nixl_agent_cls=lambda _: agent)
    arena = StagingArena(2 * HUGE_2M, chunk_bytes=HUGE_2M)

    def fail_unregister(*args):
        raise OSError("unregister failed")

    monkeypatch.setattr(agent, "deregister_memory", fail_unregister)
    with pytest.raises(StorageQuiescenceError, match="unregister outcome unknown"):
        adapter.register_arena(arena)
    with pytest.raises(StorageQuiescenceError):
        adapter.close()
    with pytest.raises(StorageQuiescenceError):
        arena.close()
    assert arena.base is not None


@pytest.mark.parametrize("exporter", ["system_heap", "udmabuf", "udmabuf_hugetlb"])
def test_dmabuf_exporters_hand_out_fds(exporter):
    arena = exporter_or_skip(exporter)
    try:
        assert len(arena.chunks) == 1 and arena.chunks[0].dmabuf_fd >= 0
        assert arena.registrations()[0][3] == f"dmabuf:{arena.chunks[0].dmabuf_fd}"
        assert (arena.chunks[0].memfd >= 0) == exporter.startswith("udmabuf")
        assert arena.mem_type == "DRAM"
        arena.tensor[:8] = torch.arange(8, dtype=torch.uint8)
        assert arena.tensor[:8].tolist() == list(range(8))
    finally:
        arena.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda_exporter_is_device_memory_in_dmabuf_chunks():
    try:
        arena = StagingArena(2 * HUGE_2M, "cuda", chunk_bytes=HUGE_2M)
    except OSError as e:
        pytest.skip(f"cuda exporter unavailable: {e}")
    try:
        assert arena.mem_type == "VRAM" and len(arena.chunks) == 2
        assert all(c.dmabuf_fd >= 0 for c in arena.chunks)
        assert arena.registrations()[1] == (
            arena.base + HUGE_2M,
            HUGE_2M,
            arena.device,
            f"dmabuf:{arena.chunks[1].dmabuf_fd}",
        )
        assert arena.tensor.is_cuda and arena.tensor.data_ptr() == arena.base
        # A device write straddling the chunk boundary reads back through the tensor.
        arena.tensor[HUGE_2M - 2 : HUGE_2M + 2] = torch.tensor(
            [1, 2, 3, 4], dtype=torch.uint8, device=arena.tensor.device
        )
        assert arena.tensor[HUGE_2M - 2 : HUGE_2M + 2].tolist() == [1, 2, 3, 4]
        # The dma-buf sizes match the chunks (a dma-buf fd reports its size via fstat).
        assert os.fstat(arena.chunks[0].dmabuf_fd).st_size in (0, HUGE_2M)
        caches = {n: t.cuda() for n, t in make_cache(2, 8).items()}
        geoms = [LayerGeometry.from_tensor(n, t) for n, t in caches.items()]
        block_ids = [1, 2, 5]
        fill_dev = {n: t.cpu() for n, t in caches.items()}
        fill(fill_dev, block_ids, seed=5)
        for n in caches:
            caches[n].copy_(fill_dev[n])
        obj = sum(g.object_bytes(3) for g in geoms)
        region = arena.allocate(obj)
        pack_blocks(geoms, caches, block_ids, arena, region, non_blocking=True)
        arena.fence_before_io()
        out = {n: torch.zeros_like(t) for n, t in caches.items()}
        unpack_blocks(geoms, out, [7, 0, 3], arena, region, non_blocking=True)
        torch.accelerator.synchronize()
        for n in caches:
            assert torch.equal(
                caches[n][:, block_ids].cpu(), out[n][:, [7, 0, 3]].cpu()
            )
        arena.free(region)
    finally:
        arena.close()


def test_plan_pack_and_unpack_round_trip_through_the_arena():
    src = make_cache(2, 8)
    dst = make_cache(2, 8)
    geoms = [LayerGeometry.from_tensor(n, t) for n, t in src.items()]
    g = geoms[0]
    block_ids = [2, 3, 4, 7]
    arena = StagingArena(2 * HUGE_2M, "host", chunk_bytes=HUGE_2M)
    try:
        # Place the object so that layer 1's V region crosses the chunk boundary.
        obj = sum(gg.object_bytes(4) for gg in geoms)
        region = Extent(HUGE_2M - obj + 3 * g.block_bytes, obj)
        layers = layers_for(geoms, 4)
        plan = plan_transfer(geoms, layers, 4, arena, region)
        # 2 layers x 2 regions, plus one split at the boundary.
        assert plan.descriptors == 5 and plan.bytes == obj
        assert plan.remote[0] == (layers[0].extent.offset, 4 * g.block_bytes, 0)
        assert plan.remote[1] == (
            layers[0].extent.offset + 4 * g.block_bytes,
            4 * g.block_bytes,
            0,
        )
        split = [d for d in plan.local if d[0] + d[1] == arena.base + HUGE_2M]
        assert len(split) == 1
        assert sum(d[1] for d in plan.local) == sum(d[1] for d in plan.remote) == obj
        fill(src, block_ids, seed=3)
        pack_blocks(geoms, src, block_ids, arena, region)
        # The staged bytes are the object layout: layer 0 K blocks, then V.
        k0 = arena.view(
            region.offset, 4 * g.block_bytes, torch.bfloat16, (4, *g.inner_shape)
        )
        assert torch.equal(k0, src["layers.0"][0, block_ids])
        unpack_blocks(geoms, dst, [5, 6, 0, 1], arena, region)
        for name in src:
            assert torch.equal(src[name][:, block_ids], dst[name][:, [5, 6, 0, 1]])
        with pytest.raises(ValueError):
            plan_transfer(geoms, layers, 4, arena, Extent(0, obj - 1))
    finally:
        arena.close()


def test_plan_reads_the_object_blocks_a_partial_restore_names():
    # A restore that starts past a local prefix-cache hit stages fewer
    # blocks than the object holds; on the namespace they are the object's
    # blocks from first_block on, and every region keeps the object's
    # stride, not the transfer's.
    src = make_cache(2, 8)
    geoms = [LayerGeometry.from_tensor(n, t) for n, t in src.items()]
    g = geoms[0]
    bb = g.block_bytes
    arena = StagingArena(2 * HUGE_2M, "host", chunk_bytes=HUGE_2M)
    try:
        layers = layers_for(geoms, 8)
        region = Extent(0, sum(gg.object_bytes(3) for gg in geoms))
        plan = plan_transfer(
            geoms, layers, 3, arena, region, first_block=5, object_blocks=8
        )
        assert plan.bytes == region.length
        assert plan.remote[0] == (layers[0].extent.offset + 5 * bb, 3 * bb, 0)
        assert plan.remote[1] == (
            layers[0].extent.offset + 8 * bb + 5 * bb,
            3 * bb,
            0,
        )
        assert plan.remote[2] == (layers[1].extent.offset + 5 * bb, 3 * bb, 0)
        # The staged side is the transfer's own layout, three blocks long.
        assert plan.local[1][0] == arena.base + 3 * bb
        with pytest.raises(ValueError):
            plan_transfer(
                geoms, layers, 3, arena, region, first_block=6, object_blocks=8
            )
        with pytest.raises(ValueError):
            plan_transfer(
                geoms, layers, 3, arena, region, first_block=-1, object_blocks=8
            )
    finally:
        arena.close()


@pytest.mark.skipif(DEVICE is None, reason="NIXL_STORAGE_DEVICE not set")
def test_write_then_read_blocks_through_the_namespace():
    params = {"device_name": DEVICE, "num_tasks": "32"}
    if BACKEND == "XNVME_NVM":
        params["async"] = ASYNC
    if STAGING != "host":
        params["buffer_mode"] = "dmabuf"
    expected_wwid = os.environ.get("NIXL_STORAGE_WWID") or os.environ.get(
        "NIXL_XNVME_NVM_WWID"
    )
    if expected_wwid:
        params["expect_wwid"] = expected_wwid
    adapter = NixlStorageAdapter(
        "adapter-test",
        params,
        range_offset=0,
        range_len=64 * MiB,
        backend_name=BACKEND,
    )
    assert "serial" in adapter.device_info
    src = make_cache(4, 32)
    dst = make_cache(4, 32)
    if STAGING == "cuda":
        # The paged cache lives on the GPU and the arena is GPU memory
        # exported as dma-bufs: the payload never touches host memory.
        src = {n: t.cuda() for n, t in src.items()}
        dst = {n: t.cuda() for n, t in dst.items()}
    adapter.register_cache(src)
    arena = StagingArena(4 * HUGE_2M, STAGING, chunk_bytes=HUGE_2M)
    adapter.register_arena(arena)
    warm_ms = adapter.warm()
    assert warm_ms >= 0
    block_ids = [1, 2, 3, 9, 10, 30]
    if STAGING == "cuda":
        host = {n: t.cpu() for n, t in src.items()}
        fill(host, block_ids, seed=7)
        for n in src:
            src[n].copy_(host[n])
    else:
        fill(src, block_ids, seed=7)
    geoms = list(adapter.geometries.values())
    layers = adapter.layer_slices(
        [layer.extent for layer in layers_for(geoms, len(block_ids))], len(block_ids)
    )
    obj = adapter.object_bytes(len(block_ids))
    region = arena.allocate(obj)
    adapter.pack(src, block_ids, region)
    pending = adapter.start("WRITE", layers, len(block_ids), region)
    while adapter.poll(pending) == "PROC":
        pass
    assert pending.status == "DONE"
    adapter.release(pending)
    arena.free(region)
    # Read the object back through a different region into other blocks.
    region = arena.allocate(obj)
    arena.tensor[region.offset : region.offset + obj].fill_(0xA5)
    pending = adapter.start("READ", layers, len(block_ids), region)
    while adapter.poll(pending) == "PROC":
        pass
    assert pending.status == "DONE"
    adapter.release(pending)
    dst_blocks = [20, 21, 22, 5, 6, 0]
    adapter.unpack(dst, dst_blocks, region)
    arena.free(region)
    for name in src:
        for sb, db in zip(block_ids, dst_blocks):
            assert torch.equal(src[name][:, sb].cpu(), dst[name][:, db].cpu()), (
                f"{name} block {sb}->{db}"
            )
    assert adapter.counters["writes"] == 1 and adapter.counters["reads"] == 1
    assert (
        adapter.counters["bytes_written"] == obj
        and adapter.counters["bytes_read"] == obj
    )
    assert adapter.counters["descriptors"] >= 2 * 2 * 4
    adapter.close()
    arena.close()
