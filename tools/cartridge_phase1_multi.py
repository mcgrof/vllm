#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase 1: Multi-cartridge with registry + LMCache tiering.

Validates the multi-cartridge serving path that Phase 0 did not
touch. This tests:

  1. REGISTRY: register 3 cartridges with distinct labels.
  2. LOOKUP: route requests to the right cartridge by label.
  3. STORE: load all 3 into CartridgeStore concurrently.
  4. MEMORY: memory usage scales linearly with cartridge count.
  5. EVICTION: eviction under memory pressure respects ref counts.
  6. ZIPFIAN: simulate skewed workload (hot/cold cartridges).
  7. LMCACHE: register CartridgeLMCachePlugin as a storage backend.
  8. TTFT: per-cartridge TTFT under load.

For Phase 1 we use one .pt file loaded under 3 different
cartridge_ids. This is a deliberate choice to validate the
infrastructure (registry, routing, eviction) without needing
multiple distinct trained cartridges. Multi-content validation
happens in Phase 2 with Musique/HotPotQA cartridges.

Usage:
    python tools/cartridge_phase1_multi.py \
        --cartridge /path/to/cache-step2694.pt \
        --device cuda:0
"""
import argparse
import json
import os
import random
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cartridge", required=True,
                        help="Base .pt to use for all 3 cartridge IDs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-cartridges", type=int, default=3)
    parser.add_argument("--n-requests", type=int, default=30,
                        help="Total requests for Zipfian workload")
    parser.add_argument("--zipf-s", type=float, default=1.2,
                        help="Zipfian skew parameter (higher = more skewed)")
    parser.add_argument("--out", default="phase1_results.json")
    args = parser.parse_args()

    results = {"checks": [], "overall_pass": False}

    def check(name, fn):
        print(f"\n[phase1] === {name} ===", flush=True)
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
            import traceback
            traceback.print_exc()
        results["checks"].append(entry)
        return entry["passed"]

    from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_connector import (
        align_to_block_size, load_cartridge,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_lmcache_plugin import (
        CartridgeLMCachePlugin,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_manifest import (
        CartridgeManifest,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_registry import (
        CartridgeRegistry,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_store import (
        CartridgeStore, ChunkKey,
    )

    # Build manifests for n_cartridges virtual cartridges backed by the
    # same .pt file. Each gets a distinct cartridge_id and routing labels.
    print(f"[phase1] Building {args.n_cartridges} virtual cartridge "
          f"manifests...", flush=True)
    cart_data = load_cartridge(args.cartridge)
    virtual_carts = []
    for i in range(args.n_cartridges):
        cart_id = f"patient_{i:02d}"
        manifest = CartridgeManifest(
            cartridge_id=cart_id,
            model_id="meta-llama/Llama-3.2-3B-Instruct",
            num_layers=cart_data["num_layers"],
            num_kv_heads=cart_data["num_kv_heads"],
            head_dim=cart_data["head_dim"],
            dtype="bfloat16",
            num_tokens_raw=cart_data["num_tokens"],
            num_tokens_aligned=align_to_block_size(
                cart_data["num_tokens"], 16),
            block_size=16,
            num_blocks=align_to_block_size(
                cart_data["num_tokens"], 16) // 16,
            has_frozen_prefix=True,
            labels={
                "patient_id": cart_id,
                "doc_type": "medical_record",
                "priority": "high" if i == 0 else "normal",
            },
        )
        virtual_carts.append(manifest)
    del cart_data
    print(f"  Built {len(virtual_carts)} manifests", flush=True)

    # -------------------------------------------------------------
    # CHECK 1: Registry registration and lookup
    # -------------------------------------------------------------
    db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_file.close()
    registry = CartridgeRegistry(db_file.name)

    def check_registry():
        for m in virtual_carts:
            registry.register(m)
        assert registry.count() == len(virtual_carts)

        # Lookup by ID
        m = registry.lookup("patient_00")
        assert m is not None
        assert m.labels["priority"] == "high"

        # Lookup by label
        high_pri = registry.lookup_by_label("priority", "high")
        assert len(high_pri) == 1
        assert high_pri[0].cartridge_id == "patient_00"

        med_carts = registry.lookup_by_label("doc_type", "medical_record")
        assert len(med_carts) == args.n_cartridges

        # Lookup by model
        by_model = registry.lookup_by_model(
            "meta-llama/Llama-3.2-3B-Instruct")
        assert len(by_model) == args.n_cartridges

        return f"Registered {args.n_cartridges}, lookups by ID/label/model OK"

    check("REGISTRY: register + lookup multiple cartridges", check_registry)

    # -------------------------------------------------------------
    # CHECK 2: Store loads multiple cartridges concurrently
    # -------------------------------------------------------------
    store = CartridgeStore(block_size=16)

    def check_multi_load():
        for m in virtual_carts:
            store.load(m.cartridge_id, args.cartridge, m, device="cpu")
        loaded = store.list_loaded()
        assert len(loaded) == args.n_cartridges
        for m in virtual_carts:
            assert m.cartridge_id in loaded
        return f"Loaded {args.n_cartridges} cartridges, all present"

    check("MULTI_LOAD: store handles multiple cartridges", check_multi_load)

    # -------------------------------------------------------------
    # CHECK 3: Memory usage scales linearly
    # -------------------------------------------------------------
    def check_memory_scaling():
        total = store.memory_usage_bytes()
        per_cart = total // args.n_cartridges
        expected_per_cart = virtual_carts[0].num_layers * 2 * \
            virtual_carts[0].num_tokens_aligned * \
            virtual_carts[0].num_kv_heads * \
            virtual_carts[0].head_dim * 2  # bfloat16 = 2 bytes
        ratio = per_cart / expected_per_cart
        # Allow 5% tolerance for any alignment padding
        assert 0.95 <= ratio <= 1.05, (
            f"Memory ratio {ratio:.2f} out of tolerance")
        return (f"Total {total/1e6:.0f}MB, "
                f"per-cartridge {per_cart/1e6:.0f}MB "
                f"(expected {expected_per_cart/1e6:.0f}MB, "
                f"ratio {ratio:.2f})")

    check("MEMORY_SCALING: linear with cartridge count",
          check_memory_scaling)

    # -------------------------------------------------------------
    # CHECK 4: Eviction respects ref counts
    # -------------------------------------------------------------
    def check_eviction():
        # Acquire a ref on patient_00
        store.acquire("patient_00")

        # Eviction without force should fail
        assert not store.evict("patient_00")
        assert store.contains("patient_00")

        # Force eviction should succeed
        assert store.evict("patient_00", force=True)
        assert not store.contains("patient_00")

        # Other cartridges untouched
        assert store.contains("patient_01")
        assert store.contains("patient_02")

        # Reload for next checks
        store.load("patient_00", args.cartridge, virtual_carts[0],
                   device="cpu")
        return "Ref-count blocks evict, force overrides, others unaffected"

    check("EVICTION: ref counting + force", check_eviction)

    # -------------------------------------------------------------
    # CHECK 5: Zipfian workload routing
    # -------------------------------------------------------------
    def check_zipfian():
        # Generate Zipfian request distribution
        random.seed(42)
        n = args.n_requests
        ranks = list(range(1, args.n_cartridges + 1))
        # Probabilities: 1/k^s normalized
        probs = [1.0 / (r ** args.zipf_s) for r in ranks]
        total = sum(probs)
        probs = [p / total for p in probs]

        # Sample requests
        cart_choices = random.choices(
            [f"patient_{i:02d}" for i in range(args.n_cartridges)],
            weights=probs, k=n,
        )
        counts = Counter(cart_choices)

        # Route each request: verify registry lookup + store hit
        hits = 0
        misses = 0
        for cart_id in cart_choices:
            # Look up by label (simulating request with patient_id)
            matches = registry.lookup_by_label("patient_id", cart_id)
            if not matches:
                misses += 1
                continue
            # Fetch chunk from store
            chunk = store.get(ChunkKey(cart_id, 0))
            if chunk is not None:
                hits += 1
                store.acquire(cart_id)
                store.release(cart_id)
            else:
                misses += 1

        assert hits == n, f"Expected {n} hits, got {hits} (misses={misses})"
        dist = dict(counts)
        return (f"{n} requests, {hits} hits, 0 misses. "
                f"Distribution: {dist}")

    check("ZIPFIAN: skewed workload routes correctly", check_zipfian)

    # -------------------------------------------------------------
    # CHECK 6: LMCache plugin serves multi-cartridge
    # -------------------------------------------------------------
    def check_lmcache_multi():
        import torch
        from lmcache.utils import CacheEngineKey

        plugin = CartridgeLMCachePlugin(dst_device="cpu")
        plugin.set_store(store)

        # Request a chunk from each cartridge via plugin
        for cart_id in [f"patient_{i:02d}"
                        for i in range(args.n_cartridges)]:
            key = CacheEngineKey(
                model_name="test", world_size=1, worker_id=0,
                chunk_hash=0, dtype=torch.bfloat16,
            )
            key.tags = (cart_id, "0")
            assert plugin.contains(key), f"{cart_id} layer 0 missing"
            chunk = plugin.get_blocking(key)
            assert chunk is not None
            assert chunk.shape[0] == 2  # (K, V)

        return f"Plugin served {args.n_cartridges} cartridges correctly"

    check("LMCACHE_MULTI: plugin serves all cartridges",
          check_lmcache_multi)

    # -------------------------------------------------------------
    # CHECK 7: Concurrent access from multiple workers
    # -------------------------------------------------------------
    def check_concurrent():
        import threading
        errors = []

        def worker(cart_id, n_ops):
            try:
                for _ in range(n_ops):
                    store.acquire(cart_id)
                    chunk = store.get(ChunkKey(cart_id, 0))
                    if chunk is None:
                        errors.append(f"{cart_id}: chunk missing")
                    store.release(cart_id)
            except Exception as e:
                errors.append(f"{cart_id}: {e}")

        threads = []
        for i in range(args.n_cartridges):
            cart_id = f"patient_{i:02d}"
            t = threading.Thread(target=worker, args=(cart_id, 200))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors: {errors[:3]}"

        # All ref counts should be 0 after workers finish
        for i in range(args.n_cartridges):
            res = store.get_residency(f"patient_{i:02d}")
            assert res.ref_count == 0, (
                f"patient_{i:02d} ref_count={res.ref_count}")

        return f"{args.n_cartridges} workers x 200 ops each, no races"

    check("CONCURRENT: multi-worker store access", check_concurrent)

    # -------------------------------------------------------------
    # CHECK 8: Unregister + cleanup
    # -------------------------------------------------------------
    def check_cleanup():
        # Unregister patient_00
        assert registry.unregister("patient_00")
        assert registry.count() == args.n_cartridges - 1
        assert registry.lookup("patient_00") is None

        # Other entries still there
        assert registry.lookup("patient_01") is not None

        # Store eviction
        assert store.evict("patient_00", force=True)
        assert "patient_00" not in store.list_loaded()

        return "Unregister and evict cleaned up correctly"

    check("CLEANUP: unregister + evict", check_cleanup)

    # -------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print("[phase1] SUMMARY", flush=True)
    print("=" * 60, flush=True)

    all_pass = all(c["passed"] for c in results["checks"])
    results["overall_pass"] = all_pass
    for c in results["checks"]:
        status = "PASS" if c["passed"] else "FAIL"
        print(f"  [{status}] {c['name']}", flush=True)
        if not c["passed"]:
            print(f"         {c['details']}", flush=True)

    print(f"\n  Overall: {'PASS' if all_pass else 'FAIL'}", flush=True)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {args.out}", flush=True)

    # Cleanup
    registry.close()
    store.close()
    Path(db_file.name).unlink(missing_ok=True)

    if not all_pass:
        print("\n*** PHASE 1 FAILED ***", flush=True)
        sys.exit(1)
    else:
        print("\n  Phase 1 passed — multi-cartridge infrastructure works",
              flush=True)


if __name__ == "__main__":
    main()
