# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NIXL storage adapter: moves KV blocks between a host staging arena and
extents of a local NVMe namespace through the XNVME_NVM backend.

The arena is a bounded host area objects pass through. A store packs the
blocks of one request into it and writes them out; a load reads an object
into it and scatters the blocks back into the paged cache. It is one
contiguous virtual range backed by plain host memory or by dma-bufs from
the system heap or udmabuf, each piece registered once with the backend.
An object occupies one contiguous arena region laid out exactly as it is
on the namespace, per layer all K blocks in order then all V blocks, so a
transfer is one descriptor per layer region, split only where the region
crosses a dma-buf boundary; the backend splits at its command ceiling.
"""

from __future__ import annotations

import ctypes
import fcntl
import mmap
import os
import struct
import time
from dataclasses import dataclass

import torch

from .coordinator import ExtentAllocator
from .manifest import Extent, LayerSlice

EXPORTERS = ("host", "system_heap", "udmabuf", "udmabuf_hugetlb", "cuda")
HUGE_2M = 2 << 20

# CUDA driver API constants (cuda.h)
_CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 124
_CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD = 1
_CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE = 1

_PROT_NONE = 0
_MAP_FIXED = 0x10
_MADV_HUGEPAGE = 14
_DMA_HEAP_IOCTL_ALLOC = 0xC0184800  # _IOWR('H', 0, struct dma_heap_allocation_data)
_UDMABUF_CREATE = 0x40187542  # _IOW('u', 0x42, struct udmabuf_create)
_UDMABUF_FLAGS_CLOEXEC = 1

_libc = ctypes.CDLL(None, use_errno=True)
_libc.mmap.restype = ctypes.c_void_p
_libc.mmap.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_long,
]
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_MAP_FAILED = ctypes.c_void_p(-1).value


def _mmap(addr: int, length: int, prot: int, flags: int, fd: int) -> int:
    p = _libc.mmap(addr, length, prot, flags, fd, 0)
    if p == _MAP_FAILED or p is None:
        e = ctypes.get_errno()
        raise OSError(e, f"mmap({length} bytes): {os.strerror(e)}")
    return p


class _Cuda:
    """The few CUDA driver entry points the cuda exporter needs, through
    libcuda directly so no extra Python package is required."""

    def __init__(self):
        try:
            self.lib = ctypes.CDLL("libcuda.so.1")
        except OSError as e:
            raise OSError(
                0, f"libcuda.so.1: {e}; the cuda exporter needs the NVIDIA driver"
            ) from e
        self.lib.cuGetErrorString.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self.lib.cuMemAlloc_v2.argtypes = [
            ctypes.POINTER(ctypes.c_ulonglong),
            ctypes.c_size_t,
        ]
        self.lib.cuMemFree_v2.argtypes = [ctypes.c_ulonglong]
        self.lib.cuDeviceGetAttribute.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.cuMemGetHandleForAddressRange.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulonglong,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_ulonglong,
        ]
        self.lib.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.cuDevicePrimaryCtxRetain.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
        ]
        self.lib.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        self.lib.cuInit.argtypes = [ctypes.c_uint]

    def check(self, rc: int, what: str) -> None:
        if rc != 0:
            msg = ctypes.c_char_p()
            self.lib.cuGetErrorString(rc, ctypes.byref(msg))
            raise RuntimeError(
                f"{what}: CUDA error {rc} ({(msg.value or b'?').decode()})"
            )

    def ensure_context(self, device: int) -> None:
        ctx = ctypes.c_void_p()
        self.check(self.lib.cuCtxGetCurrent(ctypes.byref(ctx)), "cuCtxGetCurrent")
        if ctx.value:
            return
        self.check(self.lib.cuInit(0), "cuInit")
        self.check(
            self.lib.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), device),
            "cuDevicePrimaryCtxRetain",
        )
        self.check(self.lib.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")

    def dmabuf_supported(self, device: int) -> bool:
        v = ctypes.c_int(0)
        self.check(
            self.lib.cuDeviceGetAttribute(
                ctypes.byref(v), _CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, device
            ),
            "cuDeviceGetAttribute(DMA_BUF_SUPPORTED)",
        )
        return bool(v.value)

    def alloc(self, nbytes: int) -> int:
        p = ctypes.c_ulonglong(0)
        self.check(
            self.lib.cuMemAlloc_v2(ctypes.byref(p), nbytes), f"cuMemAlloc({nbytes})"
        )
        return p.value

    def free(self, ptr: int) -> None:
        self.lib.cuMemFree_v2(ptr)

    def export(self, ptr: int, nbytes: int) -> tuple[int, str]:
        """A dma-buf fd over [ptr, ptr + nbytes) and the mapping type it
        was requested with (PCIe when the driver knows the flag)."""
        fd = ctypes.c_int(-1)
        rc = self.lib.cuMemGetHandleForAddressRange(
            ctypes.byref(fd),
            ptr,
            nbytes,
            _CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
            _CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE,
        )
        if rc == 0:
            return fd.value, "pcie"
        rc = self.lib.cuMemGetHandleForAddressRange(
            ctypes.byref(fd), ptr, nbytes, _CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0
        )
        self.check(rc, f"cuMemGetHandleForAddressRange({nbytes} bytes)")
        return fd.value, "default"


def _clear_cuda_last_error() -> None:
    """cudaGetLastError through the runtime library torch loaded (torch's own
    cudart binding does not expose it).

    The error sits in the last-error slot of the runtime that produced it, so
    it has to be cleared through that same library.  Resolve the symbol in the
    process first: a wheel directory can hold a cudart of a different major
    version than the one torch actually loaded, and clearing that one leaves
    the real error in place to surface from an unrelated later launch."""
    import glob

    try:
        ctypes.CDLL(None).cudaGetLastError()
        return
    except (OSError, AttributeError):
        pass

    here = os.path.dirname(torch.__file__)
    for pattern in (
        os.path.join(here, "lib", "libcudart*.so*"),
        os.path.join(
            os.path.dirname(here), "nvidia", "cuda_runtime", "lib", "libcudart*.so*"
        ),
    ):
        for lib in glob.glob(pattern):
            try:
                ctypes.CDLL(lib).cudaGetLastError()
                return
            except (OSError, AttributeError):
                continue


class _DevicePointer:
    """A CUDA device range torch can wrap without copying."""

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }


def aligned_empty(
    shape, dtype: torch.dtype, stride=None, align: int = 4096
) -> torch.Tensor:
    """A host tensor whose data pointer is aligned to `align` bytes: direct
    I/O and passthrough need at least logical-block alignment, and torch's
    CPU allocator only promises 64 bytes. `stride` lets the tensor keep a
    device cache's element order."""
    elem = torch.empty(0, dtype=dtype).element_size()
    if stride is None:
        t = torch.empty(shape, dtype=dtype)
        stride = t.stride()
    nelem = 1 + sum((s - 1) * st for s, st in zip(shape, stride)) if len(shape) else 1
    nbytes = nelem * elem
    raw = torch.empty(nbytes + align, dtype=torch.uint8)
    off = (-raw.data_ptr()) % align
    view = raw[off : off + nbytes].view(dtype)
    return torch.as_strided(view, tuple(shape), tuple(stride))


@dataclass
class ArenaChunk:
    """One mapping inside the arena: an anonymous host mapping or the
    mapping of a dma-buf's pages."""

    offset: int  # within the arena
    length: int
    addr: int  # virtual address; arena.base + offset
    dmabuf_fd: int = -1
    memfd: int = -1  # udmabuf only: the sealed memfd the mapping comes from

    @property
    def meta(self) -> str:
        return f"dmabuf:{self.dmabuf_fd}" if self.dmabuf_fd >= 0 else ""

    @property
    def end(self) -> int:
        return self.offset + self.length


class StagingArena:
    """A contiguous virtual range of `nbytes`, mapped chunk by chunk from
    the chosen exporter, with a first-fit allocator over it.

    `host` maps anonymous memory advised for transparent huge pages;
    `system_heap` allocates each chunk from /dev/dma_heap/system;
    `udmabuf` and `udmabuf_hugetlb` seal a memfd (4 KiB or 2 MiB folios)
    and export it through /dev/udmabuf, mapping the memfd; `cuda` is one
    device allocation on `device`, exported chunk by chunk as dma-bufs
    with cuMemGetHandleForAddressRange, so the payload never touches
    host memory. Chunks are at most `chunk_bytes` because the io_uring
    registration takes at most 1 GiB per dma-buf. The exporter is
    invisible past registration."""

    def __init__(
        self,
        nbytes: int,
        exporter: str = "host",
        chunk_bytes: int = 1 << 30,
        align: int = 4096,
        name: str = "kv-staging",
        device: int = 0,
    ):
        if exporter not in EXPORTERS:
            raise ValueError(
                f"staging exporter must be one of {EXPORTERS}, got {exporter!r}"
            )
        if nbytes <= 0:
            raise ValueError("arena size must be positive")
        if chunk_bytes <= 0 or chunk_bytes % HUGE_2M:
            raise ValueError("chunk_bytes must be a positive multiple of 2 MiB")
        self.exporter = exporter
        self.align = align
        self.name = name
        self.device = device
        self.nbytes = (nbytes + HUGE_2M - 1) // HUGE_2M * HUGE_2M
        self.chunk_bytes = chunk_bytes
        self.chunks: list[ArenaChunk] = []
        self.cuda_registered = False
        self._cuda_registered_chunks: list[ArenaChunk] = []
        self.mapping_type = ""
        self.waits = 0
        self.wait_ns = 0
        self.base: int | None = None
        self._cuda: _Cuda | None = None
        if exporter == "cuda":
            self._init_cuda()
            return
        # Reserve the whole range first so the chunks land back to back.
        self.base = _mmap(
            0, self.nbytes, _PROT_NONE, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS, -1
        )
        try:
            off = 0
            while off < self.nbytes:
                length = min(chunk_bytes, self.nbytes - off)
                self.chunks.append(self._map_chunk(off, length))
                off += length
        except Exception:
            self.close()
            raise
        self._alloc = ExtentAllocator(0, self.nbytes, align)
        self._raw = (ctypes.c_uint8 * self.nbytes).from_address(self.base)
        self.tensor = torch.frombuffer(self._raw, dtype=torch.uint8)

    @property
    def mem_type(self) -> str:
        """NIXL memory type of the chunks: VRAM for the cuda exporter."""
        return "VRAM" if self.exporter == "cuda" else "DRAM"

    def _init_cuda(self) -> None:
        if not torch.cuda.is_available():
            raise OSError(0, "the cuda exporter needs a CUDA device")
        torch.accelerator.set_device_index(self.device)
        torch.accelerator.synchronize()
        cuda = _Cuda()
        self._cuda = cuda
        cuda.ensure_context(self.device)
        if not cuda.dmabuf_supported(self.device):
            raise OSError(
                0,
                f"CUDA device {self.device} does not export dma-bufs "
                "(CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED is 0; the open kernel modules "
                "and a PCIe peer-mappable GPU are required)",
            )
        base = cuda.alloc(self.nbytes)
        self.base = base
        if base % 4096:
            cuda.free(base)
            self.base = None
            raise RuntimeError("cuMemAlloc returned a range that is not page aligned")
        try:
            off = 0
            while off < self.nbytes:
                length = min(self.chunk_bytes, self.nbytes - off)
                fd, mapping = cuda.export(base + off, length)
                self.mapping_type = mapping
                self.chunks.append(ArenaChunk(off, length, base + off, dmabuf_fd=fd))
                off += length
        except Exception:
            self.close()
            raise
        self._alloc = ExtentAllocator(0, self.nbytes, self.align)
        self.tensor = torch.as_tensor(
            _DevicePointer(base, self.nbytes), device=f"cuda:{self.device}"
        )
        assert self.tensor.data_ptr() == base

    # -- mapping -------------------------------------------------------------

    def _map_chunk(self, off: int, length: int) -> ArenaChunk:
        assert self.base is not None
        addr = self.base + off
        rw = mmap.PROT_READ | mmap.PROT_WRITE
        if self.exporter == "host":
            _mmap(
                addr, length, rw, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | _MAP_FIXED, -1
            )
            _libc.madvise(addr, length, _MADV_HUGEPAGE)
            return ArenaChunk(off, length, addr)
        if self.exporter == "system_heap":
            fd = self._heap_alloc(length)
            try:
                _mmap(addr, length, rw, mmap.MAP_SHARED | _MAP_FIXED, fd)
            except Exception:
                os.close(fd)
                raise
            return ArenaChunk(off, length, addr, dmabuf_fd=fd)
        memfd, fd = self._udmabuf_alloc(length, self.exporter == "udmabuf_hugetlb")
        try:
            _mmap(addr, length, rw, mmap.MAP_SHARED | _MAP_FIXED, memfd)
        except Exception:
            os.close(fd)
            os.close(memfd)
            raise
        return ArenaChunk(off, length, addr, dmabuf_fd=fd, memfd=memfd)

    @staticmethod
    def _heap_alloc(length: int) -> int:
        try:
            heap = os.open("/dev/dma_heap/system", os.O_RDONLY | os.O_CLOEXEC)
        except OSError as e:
            raise OSError(
                e.errno,
                f"/dev/dma_heap/system: {e.strerror}; the system heap needs "
                "CONFIG_DMABUF_HEAPS_SYSTEM and access to the node",
            ) from e
        try:
            # struct dma_heap_allocation_data contains len, fd, fd_flags,
            # and heap_flags.
            req = bytearray(struct.pack("QIIQ", length, 0, os.O_RDWR | os.O_CLOEXEC, 0))
            fcntl.ioctl(heap, _DMA_HEAP_IOCTL_ALLOC, req)
            return struct.unpack("QIIQ", bytes(req))[1]
        finally:
            os.close(heap)

    def _udmabuf_alloc(self, length: int, hugetlb: bool) -> tuple[int, int]:
        flags = os.MFD_ALLOW_SEALING
        if hugetlb:
            flags |= os.MFD_HUGETLB | os.MFD_HUGE_2MB
        memfd = os.memfd_create(self.name, flags)
        try:
            os.ftruncate(memfd, length)
            fcntl.fcntl(memfd, fcntl.F_ADD_SEALS, fcntl.F_SEAL_SHRINK)
            try:
                dev = os.open("/dev/udmabuf", os.O_RDWR | os.O_CLOEXEC)
            except OSError as e:
                raise OSError(
                    e.errno,
                    f"/dev/udmabuf: {e.strerror}; udmabuf needs CONFIG_UDMABUF "
                    "and access to the node",
                ) from e
            try:
                # struct udmabuf_create { u32 memfd; u32 flags; u64 offset; u64 size; }
                req = bytearray(
                    struct.pack("IIQQ", memfd, _UDMABUF_FLAGS_CLOEXEC, 0, length)
                )
                fd = fcntl.ioctl(dev, _UDMABUF_CREATE, req)
            finally:
                os.close(dev)
            if not isinstance(fd, int):
                raise RuntimeError("UDMABUF_CREATE returned no file descriptor")
            if fd < 0:
                raise OSError(-fd, f"UDMABUF_CREATE: {os.strerror(-fd)}")
            return memfd, fd
        except Exception:
            os.close(memfd)
            raise

    def cuda_register(self) -> bool:
        """Pin the mappings with the CUDA runtime so device copies into the
        arena are DMA. Returns False (and leaves copies pageable) when the
        runtime refuses; the caller logs it."""
        if not torch.cuda.is_available() or self.exporter == "cuda":
            return False
        rt = torch.cuda.cudart()
        for c in self.chunks:
            if rt.cudaHostRegister(c.addr, c.length, 0) != 0:
                for registered in reversed(self._cuda_registered_chunks):
                    rt.cudaHostUnregister(registered.addr)
                self._cuda_registered_chunks = []
                # The failure sits in the runtime's last-error slot and torch
                # would report it from an unrelated later kernel launch.
                _clear_cuda_last_error()
                return False
            self._cuda_registered_chunks.append(c)
        self.cuda_registered = True
        return True

    # -- allocation -----------------------------------------------------------

    def allocate(self, nbytes: int) -> Extent:
        return self._alloc.allocate(nbytes)

    def free(self, region: Extent) -> None:
        self._alloc.free(region)

    @property
    def free_bytes(self) -> int:
        return self._alloc.free_bytes

    @property
    def allocated_bytes(self) -> int:
        return self._alloc.allocated_bytes

    def pieces(self, offset: int, length: int) -> list[tuple[int, int, int]]:
        """(address, length, delta) pieces of the arena range, split where
        it crosses a chunk; delta is the piece's offset within the range."""
        if offset < 0 or offset + length > self.nbytes:
            raise ValueError("range outside the arena")
        assert self.base is not None
        out = []
        pos, end = offset, offset + length
        for c in self.chunks:
            if c.end <= pos:
                continue
            if c.offset >= end:
                break
            stop = min(end, c.end)
            out.append((self.base + pos, stop - pos, pos - offset))
            pos = stop
        return out

    def registrations(self) -> list[tuple[int, int, int, str]]:
        dev = self.device if self.exporter == "cuda" else 0
        return [(c.addr, c.length, dev, c.meta) for c in self.chunks]

    def fence_before_io(self) -> None:
        """Every device write into the arena is complete and visible before
        a command reads it (the cuda exporter; a no-op elsewhere)."""
        if self.exporter == "cuda":
            torch.accelerator.synchronize()

    def view(self, offset: int, nbytes: int, dtype: torch.dtype, shape) -> torch.Tensor:
        return self.tensor[offset : offset + nbytes].view(dtype).view(*shape)

    def close(self) -> None:
        if self._cuda_registered_chunks:
            rt = torch.cuda.cudart()
            for c in reversed(self._cuda_registered_chunks):
                rt.cudaHostUnregister(c.addr)
            self._cuda_registered_chunks = []
            self.cuda_registered = False
        for c in self.chunks:
            if c.dmabuf_fd >= 0:
                os.close(c.dmabuf_fd)
            if c.memfd >= 0:
                os.close(c.memfd)
        self.chunks = []
        self.tensor = None
        base = self.base
        if base is not None:
            if self.exporter == "cuda":
                assert self._cuda is not None
                self._cuda.free(base)
            else:
                _libc.munmap(base, self.nbytes)
            self.base = None


@dataclass
class LayerGeometry:
    """How one layer's KV cache tensor is addressed."""

    layer_name: str
    regions: int  # 2 (K and V) or 1 (MLA / single region)
    split: int | None  # dim of size 2 that separates K and V, if any
    blkdim: int  # dim indexed by block id
    block_bytes: int  # bytes of one K (or V, or single) region of one block
    num_blocks: int
    dtype: torch.dtype
    inner_shape: tuple  # shape of one block's region

    @classmethod
    def from_tensor(cls, layer_name: str, t: torch.Tensor) -> LayerGeometry:
        shape = t.shape
        if t.ndim >= 2 and shape[0] == 2:
            regions, split, blkdim = 2, 0, 1
        elif t.ndim >= 2 and shape[1] == 2:
            regions, split, blkdim = 2, 1, 0
        else:
            regions, split, blkdim = 1, None, 0
        inner = tuple(shape[d] for d in range(t.ndim) if d != split and d != blkdim)
        block_elems = 1
        for d in inner:
            block_elems *= d
        return cls(
            layer_name=layer_name,
            regions=regions,
            split=split,
            blkdim=blkdim,
            block_bytes=block_elems * t.element_size(),
            num_blocks=shape[blkdim],
            dtype=t.dtype,
            inner_shape=inner,
        )

    def object_bytes(self, num_blocks: int) -> int:
        return num_blocks * self.regions * self.block_bytes

    def region_views(self, t: torch.Tensor) -> list[torch.Tensor]:
        """One (num_blocks, ...) view per region of the cache tensor."""
        if self.split is None:
            return [t]
        if self.split == 0:
            return [t[0], t[1]]
        return [t[:, 0], t[:, 1]]

    @staticmethod
    def runs(block_ids: list[int]) -> list[tuple[int, int, int]]:
        """(first_index, first_block, count) runs of consecutive block ids."""
        out: list[tuple[int, int, int]] = []
        i = 0
        while i < len(block_ids):
            j = i + 1
            while j < len(block_ids) and block_ids[j] == block_ids[j - 1] + 1:
                j += 1
            out.append((i, block_ids[i], j - i))
            i = j
        return out


def object_layout(geometries: list[LayerGeometry], num_blocks: int) -> list[int]:
    """Offset of each layer's part within an object (and its arena region)."""
    out, off = [], 0
    for g in geometries:
        out.append(off)
        off += g.object_bytes(num_blocks)
    return out


@dataclass
class TransferPlan:
    local: list[tuple[int, int, int]]  # (host addr, len, dev id)
    remote: list[tuple[int, int, int]]  # (namespace offset, len, dev id)
    bytes: int
    descriptors: int


def plan_transfer(
    geometries: list[LayerGeometry],
    layers: list[LayerSlice],
    num_blocks: int,
    arena: StagingArena,
    region: Extent,
) -> TransferPlan:
    """Pair every layer region of the object staged at `region` with its
    place on the namespace. Layer i of `layers` must describe geometries[i]."""
    if len(geometries) != len(layers):
        raise ValueError("geometry/extent count mismatch")
    bases = object_layout(geometries, num_blocks)
    if (
        region.length < bases[-1] + geometries[-1].object_bytes(num_blocks)
        if geometries
        else 0
    ):
        raise ValueError("arena region smaller than the object")
    local: list[tuple[int, int, int]] = []
    remote: list[tuple[int, int, int]] = []
    total = 0
    for g, layer, base in zip(geometries, layers, bases):
        if (
            layer.bytes_per_block != g.block_bytes
            or layer.regions_per_block != g.regions
        ):
            raise ValueError(
                f"{g.layer_name}: object geometry differs from the local cache"
            )
        if g.object_bytes(num_blocks) > layer.extent.length:
            raise ValueError(
                f"{g.layer_name}: {num_blocks} blocks do not fit the extent"
            )
        region_bytes = num_blocks * g.block_bytes
        for k in range(g.regions):
            here = region.offset + base + k * region_bytes
            there = layer.extent.offset + k * region_bytes
            for addr, length, delta in arena.pieces(here, region_bytes):
                local.append((addr, length, 0))
                remote.append((there + delta, length, 0))
                total += length
    return TransferPlan(local=local, remote=remote, bytes=total, descriptors=len(local))


def pack_blocks(
    geometries: list[LayerGeometry],
    caches: dict[str, torch.Tensor],
    block_ids: list[int],
    arena: StagingArena,
    region: Extent,
    non_blocking: bool = False,
) -> None:
    """Gather the blocks of every layer into the arena region, in object order."""
    n = len(block_ids)
    for g, base in zip(geometries, object_layout(geometries, n)):
        views = g.region_views(caches[g.layer_name])
        for k, src in enumerate(views):
            dst = arena.view(
                region.offset + base + k * n * g.block_bytes,
                n * g.block_bytes,
                g.dtype,
                (n, *g.inner_shape),
            )
            for idx, first, count in g.runs(block_ids):
                dst[idx : idx + count].copy_(
                    src[first : first + count], non_blocking=non_blocking
                )


def unpack_blocks(
    geometries: list[LayerGeometry],
    caches: dict[str, torch.Tensor],
    block_ids: list[int],
    arena: StagingArena,
    region: Extent,
    non_blocking: bool = False,
) -> None:
    """Scatter an object staged at `region` back into the given blocks."""
    n = len(block_ids)
    for g, base in zip(geometries, object_layout(geometries, n)):
        views = g.region_views(caches[g.layer_name])
        for k, dst in enumerate(views):
            src = arena.view(
                region.offset + base + k * n * g.block_bytes,
                n * g.block_bytes,
                g.dtype,
                (n, *g.inner_shape),
            )
            for idx, first, count in g.runs(block_ids):
                dst[first : first + count].copy_(
                    src[idx : idx + count], non_blocking=non_blocking
                )


@dataclass
class PendingTransfer:
    handle: object
    plan: TransferPlan
    started_ns: int
    op: str
    status: str = "PROC"
    done_ns: int = 0


class NixlStorageAdapter:
    """One NIXL agent with the XNVME_NVM backend, one registered namespace
    range, the paged cache's geometry, and a registered staging arena."""

    def __init__(
        self,
        agent_name: str,
        backend_params: dict[str, str],
        range_offset: int,
        range_len: int,
        nixl_agent_cls=None,
    ):
        if nixl_agent_cls is None:
            from nixl._api import nixl_agent, nixl_agent_config

            # The XNVME_NVM backend progresses from checkXfer; a NIXL
            # progress thread would only spin.
            def nixl_agent_cls(name):
                try:
                    cfg = nixl_agent_config(enable_prog_thread=False, backends=[])
                except TypeError:
                    cfg = nixl_agent_config(backends=[])
                return nixl_agent(name, cfg)

        self.agent_name = agent_name
        self.agent = nixl_agent_cls(agent_name)
        self.agent.create_backend("XNVME_NVM", backend_params)
        self.range_offset = range_offset
        self.range_len = range_len
        self._blk_reg = self.agent.register_memory(
            [(range_offset, range_len, 0, "")], "BLOCK", backends=["XNVME_NVM"]
        )
        self._arena_regs: list = []
        self.arena: StagingArena | None = None
        self.geometries: dict[str, LayerGeometry] = {}
        try:
            self.device_info = self._query_device()
        except Exception:
            self.agent.deregister_memory(self._blk_reg, ["XNVME_NVM"])
            raise
        self.counters = {
            "writes": 0,
            "reads": 0,
            "bytes_written": 0,
            "bytes_read": 0,
            "errors": 0,
            "descriptors": 0,
            "warm_ms": 0.0,
        }

    def _query_device(self) -> dict[str, str]:
        resp = self.agent.query_memory(
            [(self.range_offset, self.range_len, 0, "")], "XNVME_NVM", "BLOCK"
        )
        if not resp or resp[0] is None:
            raise RuntimeError(
                "XNVME_NVM does not report the registered range as accessible"
            )
        return dict(resp[0])

    def register_cache(self, caches: dict[str, torch.Tensor]) -> None:
        """Record the paged cache's geometry; the tensors themselves are
        never handed to the backend."""
        for name, t in caches.items():
            self.geometries[name] = LayerGeometry.from_tensor(name, t)

    def register_arena(self, arena: StagingArena) -> None:
        regs = []
        try:
            for addr, length, dev, meta in arena.registrations():
                regs.append(
                    self.agent.register_memory(
                        [(addr, length, dev, meta)],
                        arena.mem_type,
                        backends=["XNVME_NVM"],
                    )
                )
        except Exception:
            for reg in reversed(regs):
                self.agent.deregister_memory(reg, ["XNVME_NVM"])
            raise
        self._arena_regs.extend(regs)
        self.arena = arena

    def warm(self) -> float:
        """Read one logical block into the start of every dma-buf chunk so
        the kernel builds each device mapping now rather than under the
        first store. Returns the wall time in ms."""
        assert self.arena is not None
        t0 = time.monotonic_ns()
        for c in self.arena.chunks:
            if c.dmabuf_fd < 0:
                continue
            dev = self.arena.device if self.arena.exporter == "cuda" else 0
            local = self.agent.get_xfer_descs(
                [(c.addr, 4096, dev)], self.arena.mem_type
            )
            remote = self.agent.get_xfer_descs([(self.range_offset, 4096, 0)], "BLOCK")
            handle = self.agent.initialize_xfer(
                "READ", local, remote, self.agent_name, b"", ["XNVME_NVM"]
            )
            try:
                status = self.agent.transfer(handle)
                while status == "PROC":
                    status = self.agent.check_xfer_state(handle)
            finally:
                self.agent.release_xfer_handle(handle)
            if status != "DONE":
                raise RuntimeError("warm-up read of a staging chunk failed")
        ms = (time.monotonic_ns() - t0) / 1e6
        self.counters["warm_ms"] = ms
        return ms

    def layer_slices(self, extents: list[Extent], num_blocks: int) -> list[LayerSlice]:
        names = list(self.geometries)
        if len(extents) != len(names):
            raise ValueError("one extent per layer is required")
        return [
            LayerSlice(
                layer_name=n,
                extent=e,
                bytes_per_block=self.geometries[n].block_bytes,
                regions_per_block=self.geometries[n].regions,
            )
            for n, e in zip(names, extents)
        ]

    def object_lengths(self, num_blocks: int) -> list[int]:
        return [g.object_bytes(num_blocks) for g in self.geometries.values()]

    def object_bytes(self, num_blocks: int) -> int:
        return sum(self.object_lengths(num_blocks))

    def pack(
        self, caches: dict[str, torch.Tensor], block_ids: list[int], region: Extent
    ) -> None:
        assert self.arena is not None
        geoms = list(self.geometries.values())
        cuda = any(t.is_cuda for t in caches.values())
        pack_blocks(
            geoms,
            caches,
            block_ids,
            self.arena,
            region,
            non_blocking=cuda
            and (self.arena.cuda_registered or self.arena.exporter == "cuda"),
        )
        if cuda:
            # Device writes into the arena (host or GPU) are complete before
            # the command that reads them is submitted.
            torch.accelerator.synchronize()

    def unpack(
        self, caches: dict[str, torch.Tensor], block_ids: list[int], region: Extent
    ) -> None:
        assert self.arena is not None
        geoms = list(self.geometries.values())
        cuda = any(t.is_cuda for t in caches.values())
        # The command's completion was observed before this call, so the
        # bytes are in the arena; the copies are ordered behind it by the
        # stream and waited for before the blocks are handed back.
        unpack_blocks(
            geoms,
            caches,
            block_ids,
            self.arena,
            region,
            non_blocking=cuda
            and (self.arena.cuda_registered or self.arena.exporter == "cuda"),
        )
        if cuda:
            torch.accelerator.synchronize()

    def start(
        self, op: str, layers: list[LayerSlice], num_blocks: int, region: Extent
    ) -> PendingTransfer:
        assert self.arena is not None
        geoms = [self.geometries[layer.layer_name] for layer in layers]
        plan = plan_transfer(geoms, layers, num_blocks, self.arena, region)
        dev = self.arena.device if self.arena.exporter == "cuda" else 0
        local = self.agent.get_xfer_descs(
            [(a, n, dev) for a, n, _ in plan.local], self.arena.mem_type
        )
        remote = self.agent.get_xfer_descs(plan.remote, "BLOCK")
        handle = self.agent.initialize_xfer(
            op, local, remote, self.agent_name, b"", ["XNVME_NVM"]
        )
        started = time.monotonic_ns()
        try:
            status = self.agent.transfer(handle)
        except Exception:
            self.agent.release_xfer_handle(handle)
            raise
        pending = PendingTransfer(
            handle=handle, plan=plan, started_ns=started, op=op, status=status
        )
        self.counters["descriptors"] += plan.descriptors
        if status == "DONE":
            pending.done_ns = time.monotonic_ns()
            self._account(pending)
        elif status == "ERR":
            self.counters["errors"] += 1
        return pending

    def poll(self, pending: PendingTransfer) -> str:
        if pending.status == "PROC":
            pending.status = self.agent.check_xfer_state(pending.handle)
            if pending.status == "DONE":
                pending.done_ns = time.monotonic_ns()
                self._account(pending)
            elif pending.status == "ERR":
                self.counters["errors"] += 1
        return pending.status

    def release(self, pending: PendingTransfer) -> None:
        self.agent.release_xfer_handle(pending.handle)

    def _account(self, p: PendingTransfer) -> None:
        if p.op == "WRITE":
            self.counters["writes"] += 1
            self.counters["bytes_written"] += p.plan.bytes
        else:
            self.counters["reads"] += 1
            self.counters["bytes_read"] += p.plan.bytes

    def close(self) -> None:
        for reg in self._arena_regs:
            self.agent.deregister_memory(reg, ["XNVME_NVM"])
        self._arena_regs = []
        self.agent.deregister_memory(self._blk_reg, ["XNVME_NVM"])
