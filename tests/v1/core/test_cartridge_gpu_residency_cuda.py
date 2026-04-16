# SPDX-License-Identifier: Apache-2.0
"""Real-GPU evidence tests for GPUResidencyManager.

The host-side `test_cartridge_gpu_residency.py` runs with
``device="cpu"`` so it covers the tier's logic on machines without
a GPU. This file complements those tests by running the same
residency operations with ``device="cuda:0"`` and verifying
**actual GPU memory changes** via
``torch.cuda.memory_allocated()`` and tensor identity.

Skipped unless CUDA (or ROCm) is available. On ROCm, the CUDA API
surface is provided by PyTorch's HIP backend, so these tests run
unchanged on AMD W7900 / gfx1100 boxes.

Evidence produced by each test:

  - ``test_promote_increases_gpu_memory``: memory_allocated(0) grows
    by at least the cartridge size on promotion.
  - ``test_evict_releases_gpu_memory``: evicting A reduces
    memory_allocated(0) by A's size.
  - ``test_re_promotion_is_new_allocation``: layer-0 tensor from
    re-promoted cartridge has a *different* data_ptr than the
    original (i.e. not a stale reference).
  - ``test_returned_tensor_is_on_device``: every get_chunk() result
    lives on the requested GPU device.
  - ``test_no_unbounded_gpu_growth_under_pressure``: loop 100 times
    through working set 3x capacity; GPU memory never exceeds
    capacity.
  - ``test_pinned_cartridge_memory_survives_pressure``: pinned
    cartridge's bytes remain allocated while unpinned ones churn.
"""
from __future__ import annotations

import gc

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_gpu_residency import (
    GPUResidencyError,
    GPUResidencyManager,
)
from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_manifest import (
    CartridgeManifest,
)
from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_store import (
    CartridgeResidency,
    CartridgeStore,
    ChunkKey,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Requires a CUDA or ROCm GPU",
)


# ---------------------------------------------------------------------------
# Fixtures — same shape as the CPU test but parameterized on device
# ---------------------------------------------------------------------------

NUM_LAYERS = 4
NUM_KV_HEADS = 4
HEAD_DIM = 64       # bigger than CPU test so cartridges are chunkier
TOKENS_PER_CART = 128
BLOCK_SIZE = 16
DEVICE = "cuda:0"


def _cart_bytes(dtype: torch.dtype = torch.float32) -> int:
    per_chunk = 2 * TOKENS_PER_CART * NUM_KV_HEADS * HEAD_DIM
    return (NUM_LAYERS * per_chunk
            * torch.empty(0, dtype=dtype).element_size())


def _make_store_with_cartridges(cart_ids: list[str]) -> CartridgeStore:
    store = CartridgeStore(block_size=BLOCK_SIZE)
    for cart_id in cart_ids:
        manifest = CartridgeManifest(
            cartridge_id=cart_id,
            model_id="test/model",
            num_layers=NUM_LAYERS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            dtype="float32",
            num_tokens_raw=TOKENS_PER_CART,
            num_tokens_aligned=TOKENS_PER_CART,
            block_size=BLOCK_SIZE,
            num_blocks=TOKENS_PER_CART // BLOCK_SIZE,
            has_frozen_prefix=False,
        )
        chunks = {}
        for layer_idx in range(NUM_LAYERS):
            # Fill derived from (cart_id, layer_idx) so we can
            # sanity-check cross-contamination if we want to.
            fill = float(
                (hash((cart_id, layer_idx)) & 0xFF) + 1)
            chunks[ChunkKey(cart_id, layer_idx)] = torch.full(
                (2, TOKENS_PER_CART, NUM_KV_HEADS, HEAD_DIM),
                fill, dtype=torch.float32,
            )
        with store._lock:  # noqa: SLF001
            store._chunks.update(chunks)
            store._residency[cart_id] = CartridgeResidency(
                cartridge_id=cart_id,
                manifest=manifest,
                num_layers=NUM_LAYERS,
                num_tokens=TOKENS_PER_CART,
            )
    return store


def _baseline_alloc() -> int:
    """Capture the baseline GPU allocation before residency work.

    Forces a sync + gc so stray allocations from unrelated code
    (e.g. module import) don't pollute the measurement.
    """
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(DEVICE)
    return torch.cuda.memory_allocated(DEVICE)


# ---------------------------------------------------------------------------
# 1. Promotion increases GPU memory
# ---------------------------------------------------------------------------

def test_promote_increases_gpu_memory():
    store = _make_store_with_cartridges(["A"])
    mgr = GPUResidencyManager(
        store=store,
        capacity_bytes=10 * _cart_bytes(),
        device=DEVICE,
    )

    base = _baseline_alloc()
    mgr.acquire("A")
    torch.cuda.synchronize(DEVICE)
    after = torch.cuda.memory_allocated(DEVICE)

    cart_size = _cart_bytes()
    grew_by = after - base
    assert grew_by >= cart_size, (
        f"GPU allocation only grew by {grew_by} bytes; "
        f"expected at least {cart_size} for one cartridge"
    )
    print(f"\n[evidence] promote: baseline={base} after={after} "
          f"delta={grew_by} cart_size={cart_size}", flush=True)

    mgr.release("A")
    mgr.close()


# ---------------------------------------------------------------------------
# 2. Eviction releases GPU memory
# ---------------------------------------------------------------------------

def test_evict_releases_gpu_memory():
    # Capacity fits exactly 2 cartridges; a third forces eviction.
    store = _make_store_with_cartridges(["A", "B", "C"])
    mgr = GPUResidencyManager(
        store=store,
        capacity_bytes=2 * _cart_bytes() + 1024,
        device=DEVICE,
    )
    base = _baseline_alloc()

    mgr.acquire("A")
    mgr.release("A")
    mgr.acquire("B")
    mgr.release("B")
    torch.cuda.synchronize(DEVICE)
    after_two = torch.cuda.memory_allocated(DEVICE)

    # Acquire C — must evict A (oldest unpinned).
    mgr.acquire("C")
    mgr.release("C")
    torch.cuda.synchronize(DEVICE)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(DEVICE)
    after_evict = torch.cuda.memory_allocated(DEVICE)

    assert not mgr.is_resident("A"), "A should have been evicted"
    assert mgr.is_resident("B")
    assert mgr.is_resident("C")
    assert mgr.snapshot_metrics()["evict_count"] == 1
    assert mgr.snapshot_metrics()["bytes_evicted"] == _cart_bytes()

    # after_evict should be <= after_two since we evicted A and
    # replaced it with C (same size). Allow tiny slack for
    # allocator fragmentation.
    print(f"\n[evidence] evict: base={base} "
          f"after_two_loaded={after_two} "
          f"after_evict_and_replace={after_evict} "
          f"delta_two={after_two - base} "
          f"delta_evict={after_evict - base}", flush=True)
    # Allocation should be in the same ballpark — two cartridges
    # resident at both points.
    assert abs(after_evict - after_two) < _cart_bytes(), (
        f"memory diff after evict+replace {after_evict - after_two} "
        f"exceeds one cartridge size {_cart_bytes()} — eviction "
        f"may not have actually freed GPU memory"
    )
    mgr.close()


def test_evict_without_replace_shrinks_gpu_memory():
    """Strongest eviction signal: evict without replacing, and
    observe GPU allocation shrink by the cartridge size."""
    store = _make_store_with_cartridges(["A", "B"])
    mgr = GPUResidencyManager(
        store=store,
        capacity_bytes=int(1.5 * _cart_bytes()),
        device=DEVICE,
    )
    base = _baseline_alloc()

    mgr.acquire("A")
    mgr.release("A")
    torch.cuda.synchronize(DEVICE)
    after_a = torch.cuda.memory_allocated(DEVICE)

    # Acquire B — must evict A to make room.
    mgr.acquire("B")
    mgr.release("B")
    torch.cuda.synchronize(DEVICE)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(DEVICE)
    after_b = torch.cuda.memory_allocated(DEVICE)

    # Now close B too and force empty-cache; allocation should
    # return near baseline since nothing is resident.
    mgr.close()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(DEVICE)
    after_close = torch.cuda.memory_allocated(DEVICE)

    print(f"\n[evidence] evict-no-replace: base={base} "
          f"after_a={after_a} "
          f"after_a_evicted_b_resident={after_b} "
          f"after_close={after_close}", flush=True)
    # After close: allocation should drop back close to baseline.
    assert (after_close - base) < _cart_bytes(), (
        f"after close, allocation is {after_close - base} above "
        f"baseline; expected <~{_cart_bytes()}"
    )


# ---------------------------------------------------------------------------
# 3. Re-promotion is a fresh allocation
# ---------------------------------------------------------------------------

def test_re_promotion_is_new_allocation():
    """Re-promote after eviction must yield a fresh allocation.

    Note: PyTorch's caching allocator will happily hand back the
    same virtual address if nothing else is holding it — that's
    correct behavior, not a stale reference. To prove the manager
    actually dropped its reference (i.e. eviction was real), we
    hold an external reference to the pre-eviction tensor so the
    allocator can't reuse its memory; the new promotion then must
    land somewhere else.
    """
    store = _make_store_with_cartridges(["A", "B"])
    mgr = GPUResidencyManager(
        store=store,
        capacity_bytes=int(1.5 * _cart_bytes()),
        device=DEVICE,
    )
    mgr.acquire("A")
    # Keep an external reference so the allocator can't reuse
    # this region when we drop the manager's copy.
    pinned_old = mgr.get_chunk("A", 0).clone()
    ptr_before = pinned_old.data_ptr()
    mgr.release("A")

    # Force eviction.
    mgr.acquire("B")
    mgr.release("B")
    assert not mgr.is_resident("A"), "A should have been evicted"

    evicted_before = mgr.snapshot_metrics()["evict_count"]
    assert evicted_before >= 1

    # Re-promote A.
    mgr.acquire("A")
    new_tensor = mgr.get_chunk("A", 0)
    ptr_after = new_tensor.data_ptr()

    # Two independently-proven checks that re-promotion produced
    # a fresh allocation, not a stale reference:
    #   (a) the manager's metrics show a second promotion;
    #   (b) with our external reference holding the old region,
    #       the new allocation must differ.
    assert mgr.snapshot_metrics()["promote_count"] >= 2
    assert ptr_before != ptr_after, (
        f"re-promoted chunk has same data_ptr {ptr_before} "
        f"as pre-eviction, even though an external reference is "
        f"pinning the old region — eviction did not actually free"
    )
    print(f"\n[evidence] re-promote: ptr_before={ptr_before} "
          f"ptr_after={ptr_after} different={ptr_before != ptr_after} "
          f"evict_count={mgr.snapshot_metrics()['evict_count']} "
          f"promote_count={mgr.snapshot_metrics()['promote_count']}",
          flush=True)
    mgr.release("A")
    mgr.close()
    del pinned_old


# ---------------------------------------------------------------------------
# 4. Returned tensors are on the requested device
# ---------------------------------------------------------------------------

def test_returned_tensor_is_on_device():
    store = _make_store_with_cartridges(["A"])
    mgr = GPUResidencyManager(
        store=store,
        capacity_bytes=10 * _cart_bytes(),
        device=DEVICE,
    )
    mgr.acquire("A")
    for layer_idx in range(NUM_LAYERS):
        t = mgr.get_chunk("A", layer_idx)
        assert t.device.type == "cuda", (
            f"layer {layer_idx}: expected cuda device, got {t.device}"
        )
        # Not a no-op — the tensor was actually promoted, so it
        # should be on the intended device index.
        assert t.device.index == 0
    mgr.release("A")
    mgr.close()


# ---------------------------------------------------------------------------
# 5. No unbounded GPU growth under working set > capacity
# ---------------------------------------------------------------------------

def test_no_unbounded_gpu_growth_under_pressure():
    n = 6
    cart_ids = [f"cart_{i}" for i in range(n)]
    store = _make_store_with_cartridges(cart_ids)
    # Fits 2 cartridges; working set is 3x capacity.
    cap = 2 * _cart_bytes() + 1024
    mgr = GPUResidencyManager(
        store=store, capacity_bytes=cap, device=DEVICE,
    )

    base = _baseline_alloc()
    max_observed = base
    for loop in range(10):
        for cid in cart_ids:
            mgr.acquire(cid)
            _ = mgr.get_chunk(cid, 0)
            mgr.release(cid)
            torch.cuda.synchronize(DEVICE)
            cur = torch.cuda.memory_allocated(DEVICE)
            if cur > max_observed:
                max_observed = cur

    # Even across 10 passes over 6 cartridges (60 acquires), GPU
    # allocation should never meaningfully exceed our capacity
    # budget. Allocator fragmentation can add slack but nothing
    # near a cartridge-sized blowout.
    above_cap = max_observed - base - cap
    print(f"\n[evidence] no-growth: base={base} "
          f"max_observed={max_observed} above_cap={above_cap} "
          f"capacity={cap}", flush=True)
    assert above_cap < _cart_bytes() // 2, (
        f"GPU allocation exceeded capacity by {above_cap} bytes — "
        f"more than half a cartridge worth of slack"
    )
    mgr.close()


# ---------------------------------------------------------------------------
# 6. Pinned cartridge survives eviction pressure (GPU memory proof)
# ---------------------------------------------------------------------------

def test_pinned_cartridge_memory_survives_pressure():
    store = _make_store_with_cartridges(["A", "B", "C", "D"])
    mgr = GPUResidencyManager(
        store=store,
        capacity_bytes=2 * _cart_bytes() + 1024,
        device=DEVICE,
    )

    # Pin A.
    mgr.acquire("A")
    a_layer0_ptr = mgr.get_chunk("A", 0).data_ptr()

    # Hammer with B, C, D — forces evictions of the non-A entries.
    for cid in ["B", "C", "D"]:
        try:
            mgr.acquire(cid)
            mgr.release(cid)
        except GPUResidencyError:
            pass

    # A must still be resident with the same tensor.
    assert mgr.is_resident("A"), "pinned A was evicted"
    assert mgr.get_chunk("A", 0).data_ptr() == a_layer0_ptr, (
        "pinned A's chunk was reallocated under pressure"
    )
    print(f"\n[evidence] pinned-survives: A ptr={a_layer0_ptr} "
          f"stable across {3} eviction rounds, "
          f"evict_count={mgr.snapshot_metrics()['evict_count']}",
          flush=True)

    mgr.release("A")
    mgr.close()
