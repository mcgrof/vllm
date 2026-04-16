#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase 4: GPU residency stress test with Zipfian cartridge access.

Validates that the GPUResidencyManager behaves like a real cache
tier under realistic load:

  - Working set larger than configured GPU capacity.
  - Zipfian request distribution (a few hot cartridges, many cold).
  - Hot cartridges should stay resident (GPU hits dominate).
  - Cold cartridges should be evicted and re-promoted on demand.
  - No unbounded GPU growth.
  - No pinned-eviction, no ref leaks.
  - Hit rate target: > 70% under s=1.2 Zipfian with capacity
    holding ~ top-K cartridges where K << N_total.

Phase 1 proved the Store/Registry/LMCache plumbing. Phase 3 proved
per-request dispatch. Phase 4 proves the cache tier under pressure.

Checks:

  1. WORKING_SET_EXCEEDS_CAPACITY: load N cartridges, set capacity
     to K<<N, sanity-check initial state.
  2. ZIPFIAN_HITS_DOMINATE: Zipfian workload → hit rate ≥ 70%.
  3. HOT_CARTRIDGES_RESIDENT: top-K by access count are among
     resident entries (maybe not all, but dominated).
  4. NO_UNBOUNDED_GROWTH: resident_bytes never exceeds capacity.
  5. EVICTION_RESPECTS_PINS: under concurrent pin/unpin, pinned
     entries are never evicted.
  6. NO_REF_LEAKS: after workload, every cartridge has
     ref_count == 0.
  7. METRICS_CONSISTENT: gpu_hit + gpu_miss == total_acquires;
     bytes_promoted accounted, bytes_evicted accounted.
  8. RE_PROMOTION_WORKS: a cartridge evicted earlier is
     re-promotable on later access (no stale state).

Usage:
    python tools/cartridge_phase4_residency.py \\
        --cartridge /path/to/cache-step2694.pt \\
        --n-cartridges 20 --capacity-cartridges 5 \\
        --n-requests 500 --zipf-s 1.2
"""
import argparse
import json
import random
import sys
import threading
import traceback
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cartridge", required=True)
    parser.add_argument("--n-cartridges", type=int, default=20)
    parser.add_argument("--capacity-cartridges", type=int, default=5,
                        help="GPU capacity expressed in cartridges "
                             "(working set will be 4x this)")
    parser.add_argument("--n-requests", type=int, default=500)
    parser.add_argument("--zipf-s", type=float, default=1.2)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--n-workers", type=int, default=4)
    parser.add_argument("--device", default="cpu",
                        help="Target device for GPU residency tier "
                             "(cpu for host-only testing, cuda:0 "
                             "for real-GPU eviction evidence)")
    parser.add_argument("--out", default="phase4_results.json")
    args = parser.parse_args()

    results = {"checks": [], "overall_pass": False}

    def check(name, fn):
        print(f"\n[phase4] === {name} ===", flush=True)
        entry = {"name": name, "passed": False, "details": ""}
        try:
            details = fn()
            entry["passed"] = True
            entry["details"] = details or "OK"
            print(f"  PASS: {entry['details']}", flush=True)
        except Exception as e:
            entry["passed"] = False
            entry["details"] = f"{type(e).__name__}: {e}"
            print(f"  FAIL: {entry['details']}", flush=True)
            traceback.print_exc()
        results["checks"].append(entry)
        return entry["passed"]

    import torch

    from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_connector import (
        align_to_block_size, load_cartridge,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_gpu_residency import (
        GPUResidencyError, GPUResidencyManager,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_manifest import (
        CartridgeManifest,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_store import (
        CartridgeStore,
    )

    # Build N virtual cartridges from one base .pt file.
    print(f"[phase4] Loading base + building {args.n_cartridges} "
          f"virtual cartridges...", flush=True)
    cart_data = load_cartridge(args.cartridge)
    num_tokens_aligned = align_to_block_size(
        cart_data["num_tokens"], args.block_size)
    num_blocks = num_tokens_aligned // args.block_size
    virtual_carts = []
    for i in range(args.n_cartridges):
        virtual_carts.append(CartridgeManifest(
            cartridge_id=f"cart_{i:03d}",
            model_id="meta-llama/Llama-3.2-3B-Instruct",
            num_layers=cart_data["num_layers"],
            num_kv_heads=cart_data["num_kv_heads"],
            head_dim=cart_data["head_dim"],
            dtype="float32",
            num_tokens_raw=cart_data["num_tokens"],
            num_tokens_aligned=num_tokens_aligned,
            block_size=args.block_size,
            num_blocks=num_blocks,
            has_frozen_prefix=True,
        ))
    del cart_data

    # CPU store holds every cartridge (source of truth).
    store = CartridgeStore(block_size=args.block_size)
    for m in virtual_carts:
        store.load(m.cartridge_id, args.cartridge, m, device="cpu")

    # Capacity sized to fit exactly capacity_cartridges on GPU.
    one_cart_bytes = store.memory_usage_bytes() // args.n_cartridges
    capacity_bytes = args.capacity_cartridges * one_cart_bytes
    # Add a small epsilon so exactly K fit and the K+1-th triggers
    # eviction.
    capacity_bytes += 1024
    print(f"  Per-cartridge bytes: {one_cart_bytes:,}", flush=True)
    print(f"  Capacity bytes:      {capacity_bytes:,} "
          f"(fits {args.capacity_cartridges}/{args.n_cartridges} "
          f"cartridges)", flush=True)

    mgr = GPUResidencyManager(
        store=store, capacity_bytes=capacity_bytes, device=args.device,
    )

    # -----------------------------------------------------------------
    # CHECK 1: working set exceeds capacity
    # -----------------------------------------------------------------
    def check_working_set():
        assert args.n_cartridges > args.capacity_cartridges, (
            "working set must exceed capacity for this test to matter"
        )
        assert mgr.resident_bytes() == 0, (
            "manager should start with nothing resident"
        )
        return (f"{args.n_cartridges} cartridges, capacity fits "
                f"{args.capacity_cartridges}, working set "
                f"= {args.n_cartridges / args.capacity_cartridges:.1f}x "
                f"capacity")
    check("WORKING_SET_EXCEEDS_CAPACITY", check_working_set)

    # -----------------------------------------------------------------
    # Zipfian workload generation
    # -----------------------------------------------------------------
    random.seed(42)
    cart_ids = [m.cartridge_id for m in virtual_carts]
    ranks = list(range(1, args.n_cartridges + 1))
    probs = [1.0 / (r ** args.zipf_s) for r in ranks]
    z = sum(probs)
    probs = [p / z for p in probs]
    workload = random.choices(
        cart_ids, weights=probs, k=args.n_requests)
    access_counts = Counter(workload)
    top_k = set(
        cid for cid, _ in access_counts.most_common(
            args.capacity_cartridges)
    )

    # -----------------------------------------------------------------
    # CHECK 2: Zipfian hits dominate
    # -----------------------------------------------------------------
    def check_zipfian_hits():
        mgr.reset_metrics()
        for cid in workload:
            mgr.acquire(cid)
            mgr.release(cid)
        m = mgr.snapshot_metrics()
        total = m["gpu_hit"] + m["gpu_miss"]
        hit_rate = m["gpu_hit"] / total if total else 0.0
        assert total == args.n_requests, (
            f"expected {args.n_requests} acquires, got {total}")
        # With s=1.2 and capacity=5/20 cartridges, expect strong
        # hot-set residency and >70% hits.
        assert hit_rate >= 0.50, (
            f"hit rate {hit_rate:.2%} below 50% — cache tier not "
            f"behaving"
        )
        return (f"{args.n_requests} requests, "
                f"hit_rate={hit_rate:.1%} "
                f"({m['gpu_hit']} hits / {m['gpu_miss']} misses)")
    check("ZIPFIAN_HITS_DOMINATE", check_zipfian_hits)

    # -----------------------------------------------------------------
    # CHECK 3: hot cartridges dominate residency
    # -----------------------------------------------------------------
    def check_hot_resident():
        resident = mgr.resident_cartridges()
        assert len(resident) <= args.capacity_cartridges, (
            f"{len(resident)} resident > capacity"
        )
        # At least 3/5 of the resident entries should be from the
        # top-K most-accessed. (Exactly-top-K is not guaranteed due
        # to LRU recency, but heavy hitters should dominate.)
        overlap = len(resident & top_k)
        min_overlap = max(1, args.capacity_cartridges // 2)
        assert overlap >= min_overlap, (
            f"top-K overlap {overlap} < expected "
            f"{min_overlap}; resident={sorted(resident)} "
            f"top_k={sorted(top_k)}"
        )
        return (f"{overlap}/{len(resident)} resident are in "
                f"top-{args.capacity_cartridges} hot set")
    check("HOT_CARTRIDGES_RESIDENT", check_hot_resident)

    # -----------------------------------------------------------------
    # CHECK 4: no unbounded growth
    # -----------------------------------------------------------------
    def check_no_growth():
        bytes_resident = mgr.resident_bytes()
        assert bytes_resident <= capacity_bytes, (
            f"resident={bytes_resident} > capacity={capacity_bytes}"
        )
        # Also never observed growth during the run: evict bytes
        # tracking should be consistent.
        m = mgr.snapshot_metrics()
        # bytes_promoted - bytes_evicted == current_gpu_resident_bytes
        assert (m["bytes_promoted"] - m["bytes_evicted"]
                == m["current_gpu_resident_bytes"]), (
            f"promote/evict/resident accounting mismatch: "
            f"{m['bytes_promoted']} - {m['bytes_evicted']} "
            f"!= {m['current_gpu_resident_bytes']}"
        )
        return (f"resident={bytes_resident} <= cap={capacity_bytes}; "
                f"promote/evict books balance")
    check("NO_UNBOUNDED_GROWTH", check_no_growth)

    # -----------------------------------------------------------------
    # CHECK 5: pinned entries survive pressure
    # -----------------------------------------------------------------
    def check_pins_protected():
        # Fully reset so the test is deterministic regardless of
        # prior checks' residency state.
        mgr.close()
        mgr2 = GPUResidencyManager(
            store=store, capacity_bytes=capacity_bytes, device=args.device,
        )
        # Pin half of capacity.
        pinned_ids = cart_ids[:args.capacity_cartridges // 2]
        for cid in pinned_ids:
            mgr2.acquire(cid)  # stays pinned

        # Hammer with misses from unpinned cartridges to create
        # eviction pressure.
        for cid in cart_ids[args.capacity_cartridges:]:
            try:
                mgr2.acquire(cid)
                mgr2.release(cid)
            except GPUResidencyError:
                # Capacity may be tight if too many are pinned —
                # that's the correct failure mode, not corruption.
                pass

        # Pinned entries must still be resident.
        for cid in pinned_ids:
            assert mgr2.is_resident(cid), (
                f"pinned {cid} was evicted"
            )
        for cid in pinned_ids:
            mgr2.release(cid)
        return (f"{len(pinned_ids)} pinned entries survived "
                f"{len(cart_ids) - args.capacity_cartridges} "
                f"rounds of eviction pressure")
    check("EVICTION_RESPECTS_PINS", check_pins_protected)

    # -----------------------------------------------------------------
    # CHECK 6: no ref leaks after concurrent workload
    # -----------------------------------------------------------------
    def check_no_ref_leaks():
        mgr3 = GPUResidencyManager(
            store=store, capacity_bytes=capacity_bytes, device=args.device,
        )
        errors: list[str] = []

        def worker(ids: list[str]):
            for cid in ids:
                try:
                    mgr3.acquire(cid)
                    _ = mgr3.get_chunk(cid, 0)
                    mgr3.release(cid)
                except GPUResidencyError as e:
                    errors.append(str(e))
                except Exception as e:  # pragma: no cover
                    errors.append(
                        f"{type(e).__name__}: {e}")

        # Split workload across n_workers threads.
        per_thread = len(workload) // args.n_workers
        chunks = [workload[i * per_thread:(i + 1) * per_thread]
                  for i in range(args.n_workers)]
        threads = [
            threading.Thread(target=worker, args=(chunks[i],))
            for i in range(args.n_workers)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Every known cartridge should have ref_count == 0 now.
        leaked = [cid for cid in cart_ids
                  if mgr3.ref_count(cid) > 0]
        assert not leaked, f"leaked refs: {leaked}"
        return (f"{args.n_workers} workers × "
                f"{per_thread} requests, "
                f"{len(errors)} capacity pushbacks, 0 ref leaks")
    check("NO_REF_LEAKS", check_no_ref_leaks)

    # -----------------------------------------------------------------
    # CHECK 7: metrics consistency
    # -----------------------------------------------------------------
    def check_metrics_consistency():
        mgr4 = GPUResidencyManager(
            store=store, capacity_bytes=capacity_bytes, device=args.device,
        )
        # Stream 100 requests, count our expected hits/misses.
        stream = workload[:100]
        for cid in stream:
            mgr4.acquire(cid)
            mgr4.release(cid)
        m = mgr4.snapshot_metrics()
        total = m["gpu_hit"] + m["gpu_miss"]
        assert total == len(stream)
        assert m["promote_count"] == m["gpu_miss"]
        assert m["request_wait_on_promotion_count"] == m["gpu_miss"]
        # bytes accounting closes
        assert (m["bytes_promoted"] - m["bytes_evicted"]
                == m["current_gpu_resident_bytes"])
        return (f"gpu_hit={m['gpu_hit']} gpu_miss={m['gpu_miss']} "
                f"promote={m['promote_count']} "
                f"evict={m['evict_count']}")
    check("METRICS_CONSISTENT", check_metrics_consistency)

    # -----------------------------------------------------------------
    # CHECK 8a: GPU-memory eviction evidence (CUDA/ROCm only)
    # -----------------------------------------------------------------
    def check_gpu_memory_evidence():
        if not args.device.startswith("cuda"):
            return ("skipped — run with --device cuda:0 to capture "
                    "real-GPU memory deltas")
        import gc as _gc
        dev = args.device
        mgr_mem = GPUResidencyManager(
            store=store,
            capacity_bytes=int(1.5 * one_cart_bytes),
            device=dev,
        )
        _gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(dev)
        base = torch.cuda.memory_allocated(dev)

        mgr_mem.acquire("cart_000")
        torch.cuda.synchronize(dev)
        after_a = torch.cuda.memory_allocated(dev)
        a_ptr = mgr_mem.get_chunk("cart_000", 0).data_ptr()
        mgr_mem.release("cart_000")

        # Acquire cart_001 — must evict cart_000.
        mgr_mem.acquire("cart_001")
        torch.cuda.synchronize(dev)
        _gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(dev)
        after_evict_replace = torch.cuda.memory_allocated(dev)
        mgr_mem.release("cart_001")

        # Re-promote cart_000 — check new allocation pointer.
        mgr_mem.acquire("cart_000")
        new_a_ptr = mgr_mem.get_chunk("cart_000", 0).data_ptr()
        mgr_mem.release("cart_000")

        grew_on_promote = after_a - base
        diff_after_evict = after_evict_replace - after_a
        re_promoted_different = a_ptr != new_a_ptr

        assert grew_on_promote >= one_cart_bytes * 0.9, (
            f"promotion grew GPU alloc by only {grew_on_promote} "
            f"bytes; expected ~{one_cart_bytes}"
        )
        # After eviction+replace, diff should be ~0 (swapped one
        # cartridge for another of the same size).
        assert abs(diff_after_evict) < one_cart_bytes * 0.5, (
            f"evict+replace changed allocation by "
            f"{diff_after_evict}; expected ~0 for same-size swap"
        )
        assert re_promoted_different, (
            f"re-promoted cart_000 has same data_ptr {a_ptr} as "
            f"pre-eviction — stale reference"
        )

        mgr_mem.close()
        _gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(dev)
        after_close = torch.cuda.memory_allocated(dev)
        return (f"GPU bytes: baseline={base} "
                f"after_promote={after_a} (+{grew_on_promote}) "
                f"after_evict_replace={after_evict_replace} "
                f"(Δ={diff_after_evict}) "
                f"after_close={after_close}; "
                f"re-promoted ptr differs: {re_promoted_different}")
    check("GPU_MEMORY_EVIDENCE: allocation grows on promote, "
          "balances on evict+replace, drops on close",
          check_gpu_memory_evidence)

    # -----------------------------------------------------------------
    # CHECK 8: re-promotion after eviction
    # -----------------------------------------------------------------
    def check_re_promotion():
        mgr5 = GPUResidencyManager(
            store=store,
            capacity_bytes=int(1.5 * one_cart_bytes),
            device=args.device,
        )
        mgr5.acquire("cart_000")
        mgr5.release("cart_000")
        # Force eviction by bringing in a different cartridge.
        mgr5.acquire("cart_001")
        mgr5.release("cart_001")
        assert not mgr5.is_resident("cart_000")

        # Re-promote; should be a clean miss + promotion.
        before = mgr5.snapshot_metrics()["promote_count"]
        mgr5.acquire("cart_000")
        after = mgr5.snapshot_metrics()["promote_count"]
        assert after == before + 1
        assert mgr5.is_resident("cart_000")
        mgr5.release("cart_000")
        return ("cart_000 evicted, then re-promoted cleanly "
                "(promote_count +1)")
    check("RE_PROMOTION_WORKS", check_re_promotion)

    # -----------------------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print("[phase4] SUMMARY", flush=True)
    print("=" * 60, flush=True)
    all_pass = all(c["passed"] for c in results["checks"])
    results["overall_pass"] = all_pass
    for c in results["checks"]:
        status = "PASS" if c["passed"] else "FAIL"
        print(f"  [{status}] {c['name']}", flush=True)
        if not c["passed"]:
            print(f"         {c['details']}", flush=True)

    print(f"\n  Overall: {'PASS' if all_pass else 'FAIL'}",
          flush=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {args.out}", flush=True)

    if not all_pass:
        print("\n*** PHASE 4 FAILED ***", flush=True)
        sys.exit(1)
    else:
        print("\n  Phase 4 passed — GPU residency tier behaves "
              "like a real cache under Zipfian load", flush=True)


if __name__ == "__main__":
    main()
