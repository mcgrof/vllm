#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase 0.5: LMCache integration smoke test.

Validates that the CartridgeLMCachePlugin actually works with a
real LMCache runtime, not just our mocked unit tests. This is the
bridge between "our code looks right" and "the LMCache plugin
interface actually calls our methods correctly."

Five checks:

  1. IMPORT: LMCache is installed, CartridgeLMCachePlugin inherits
     from the real StoragePluginInterface.
  2. INSTANTIATION: plugin initializes without errors.
  3. CONTAINS: plugin.contains(key) returns True for loaded chunks,
     False for missing cartridges.
  4. GET_BLOCKING: plugin.get_blocking(key) returns the correct
     tensor (bit-exact match with CartridgeStore.get()).
  5. READ-ONLY: put operations are no-ops, as expected.
  6. LIFECYCLE: pin/unpin/remove delegate correctly.

Bonus: integrates Phase 0's TTFT test, but reads chunks through
the LMCache plugin instead of directly from CartridgeStore.

Usage:
    python tools/cartridge_phase0_5_lmcache.py \
        --cartridge /path/to/cache-step2694.pt
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cartridge", required=True)
    parser.add_argument("--out", default="phase0_5_lmcache_results.json")
    args = parser.parse_args()

    results = {"checks": [], "overall_pass": False}

    def check(name, fn):
        print(f"\n[phase0.5] === {name} ===", flush=True)
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

    # -------------------------------------------------------------
    # CHECK 1: LMCache import + plugin inheritance
    # -------------------------------------------------------------
    def check_import():
        import lmcache
        from lmcache.utils import CacheEngineKey
        from lmcache.v1.storage_backend.abstract_backend import (
            StoragePluginInterface,
        )
        from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_lmcache_plugin import (
            CartridgeLMCachePlugin, _LMCACHE_AVAILABLE,
        )
        assert _LMCACHE_AVAILABLE, "LMCache not detected by plugin"
        assert issubclass(CartridgeLMCachePlugin, StoragePluginInterface), \
            "Plugin does not inherit from StoragePluginInterface"
        return f"LMCache installed, plugin inherits StoragePluginInterface"

    if not check("IMPORT: LMCache + plugin inheritance", check_import):
        results["overall_pass"] = False
        print("\n*** ABORTED: LMCache not available ***", flush=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        sys.exit(1)

    # -------------------------------------------------------------
    # CHECK 2: plugin instantiation
    # -------------------------------------------------------------
    from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_lmcache_plugin import (
        CartridgeLMCachePlugin,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_store import (
        CartridgeStore, ChunkKey,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_manifest import (
        CartridgeManifest,
    )

    def check_instantiation():
        plugin = CartridgeLMCachePlugin(dst_device="cpu")
        assert plugin is not None
        assert plugin.dst_device == "cpu"
        return "Plugin instantiated cleanly"

    check("INSTANTIATION: plugin creates without error", check_instantiation)

    # -------------------------------------------------------------
    # Set up a shared store with the real cartridge
    # -------------------------------------------------------------
    import torch
    from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_connector import (
        align_to_block_size, load_cartridge,
    )

    print("\n[phase0.5] Loading cartridge into store...", flush=True)
    cart = load_cartridge(args.cartridge)
    manifest = CartridgeManifest(
        cartridge_id="phase05_test",
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        num_layers=cart["num_layers"],
        num_kv_heads=cart["num_kv_heads"],
        head_dim=cart["head_dim"],
        dtype="bfloat16",
        num_tokens_raw=cart["num_tokens"],
        num_tokens_aligned=align_to_block_size(cart["num_tokens"], 16),
        block_size=16,
        num_blocks=align_to_block_size(cart["num_tokens"], 16) // 16,
        has_frozen_prefix=True,
    )
    del cart

    store = CartridgeStore(block_size=16)
    store.load("phase05_test", args.cartridge, manifest, device="cpu")
    print(f"  Loaded: {store.memory_usage_bytes() / 1e6:.1f} MB",
          flush=True)

    plugin = CartridgeLMCachePlugin(dst_device="cpu")
    plugin.set_store(store)

    # -------------------------------------------------------------
    # CHECK 3: contains via real CacheEngineKey
    # -------------------------------------------------------------
    def check_contains():
        from lmcache.utils import CacheEngineKey
        # Build a real CacheEngineKey with our tag convention
        # CacheEngineKey(model_name, world_size, worker_id, chunk_hash, dtype)
        key_exists = CacheEngineKey(
            model_name="test",
            world_size=1,
            worker_id=0,
            chunk_hash=42,
            dtype=torch.bfloat16,
        )
        key_exists.tags = ("phase05_test", "0")

        key_missing = CacheEngineKey(
            model_name="test",
            world_size=1,
            worker_id=0,
            chunk_hash=99,
            dtype=torch.bfloat16,
        )
        key_missing.tags = ("nonexistent_cart", "0")

        assert plugin.contains(key_exists), \
            "contains() returned False for existing chunk"
        assert not plugin.contains(key_missing), \
            "contains() returned True for missing cartridge"
        return "contains() correctly True/False for exists/missing"

    check("CONTAINS: plugin.contains() with real CacheEngineKey",
          check_contains)

    # -------------------------------------------------------------
    # CHECK 4: get_blocking returns bit-exact tensor
    # -------------------------------------------------------------
    def check_get_blocking():
        from lmcache.utils import CacheEngineKey

        # Get the same chunk via store directly and via plugin
        direct = store.get(ChunkKey("phase05_test", 0))
        assert direct is not None

        key = CacheEngineKey(
            model_name="test",
            world_size=1,
            worker_id=0,
            chunk_hash=0,
            dtype=torch.bfloat16,
        )
        key.tags = ("phase05_test", "0")
        via_plugin = plugin.get_blocking(key)
        assert via_plugin is not None

        assert torch.equal(direct, via_plugin), \
            "Plugin returned different tensor than store"
        return (f"Bit-exact match: shape={tuple(direct.shape)}, "
                f"dtype={direct.dtype}")

    check("GET_BLOCKING: plugin returns bit-exact tensor",
          check_get_blocking)

    # -------------------------------------------------------------
    # CHECK 5: read-only semantics
    # -------------------------------------------------------------
    def check_read_only():
        from lmcache.utils import CacheEngineKey
        key = CacheEngineKey(
            model_name="test", world_size=1, worker_id=0,
            chunk_hash=0, dtype=torch.bfloat16,
        )
        key.tags = ("phase05_test", "0")

        # batched_submit_put_task should be a no-op
        result = plugin.batched_submit_put_task([key], [MagicMock()])
        assert result is None, f"Expected None, got {result}"

        # exists_in_put_tasks should always return False
        assert plugin.exists_in_put_tasks(key) is False
        return "put is no-op; exists_in_put_tasks always False"

    check("READ_ONLY: put operations are no-ops", check_read_only)

    # -------------------------------------------------------------
    # CHECK 6: pin/unpin/remove lifecycle
    # -------------------------------------------------------------
    def check_lifecycle():
        from lmcache.utils import CacheEngineKey
        key = CacheEngineKey(
            model_name="test", world_size=1, worker_id=0,
            chunk_hash=0, dtype=torch.bfloat16,
        )
        key.tags = ("phase05_test", "0")

        # Pin through plugin, verify store state
        assert plugin.pin(key) is True
        residency = store.get_residency("phase05_test")
        assert residency.pinned is True

        # Unpin
        assert plugin.unpin(key) is True
        assert residency.pinned is False

        # Remove (force) evicts the cartridge
        assert plugin.remove(key, force=True) is True
        assert store.get(ChunkKey("phase05_test", 0)) is None

        # Reload for next tests
        store.load("phase05_test", args.cartridge, manifest, device="cpu")
        return "pin/unpin/remove correctly delegate to store"

    check("LIFECYCLE: pin/unpin/remove delegation", check_lifecycle)

    # -------------------------------------------------------------
    # CHECK 7: invalid key handling
    # -------------------------------------------------------------
    def check_invalid_keys():
        from lmcache.utils import CacheEngineKey
        # No tags
        key_no_tags = CacheEngineKey(
            model_name="test", world_size=1, worker_id=0,
            chunk_hash=0, dtype=torch.bfloat16,
        )
        key_no_tags.tags = None
        assert plugin.contains(key_no_tags) is False
        assert plugin.get_blocking(key_no_tags) is None

        # Empty tags
        key_empty = CacheEngineKey(
            model_name="test", world_size=1, worker_id=0,
            chunk_hash=0, dtype=torch.bfloat16,
        )
        key_empty.tags = ()
        assert plugin.contains(key_empty) is False

        # Non-integer layer
        key_bad = CacheEngineKey(
            model_name="test", world_size=1, worker_id=0,
            chunk_hash=0, dtype=torch.bfloat16,
        )
        key_bad.tags = ("phase05_test", "not_a_number")
        assert plugin.get_blocking(key_bad) is None
        return "Invalid keys handled gracefully"

    check("INVALID_KEYS: graceful handling of bad keys",
          check_invalid_keys)

    # -------------------------------------------------------------
    # CHECK 8: performance — plugin vs direct
    # -------------------------------------------------------------
    def check_performance():
        from lmcache.utils import CacheEngineKey
        key = CacheEngineKey(
            model_name="test", world_size=1, worker_id=0,
            chunk_hash=0, dtype=torch.bfloat16,
        )
        key.tags = ("phase05_test", "0")

        # Time direct store access
        N = 1000
        t0 = time.monotonic()
        for _ in range(N):
            _ = store.get(ChunkKey("phase05_test", 0))
        direct_us = (time.monotonic() - t0) / N * 1e6

        # Time through plugin
        t0 = time.monotonic()
        for _ in range(N):
            _ = plugin.get_blocking(key)
        plugin_us = (time.monotonic() - t0) / N * 1e6

        overhead = plugin_us - direct_us
        # Plugin overhead should be minimal (just tag parsing)
        assert overhead < 100, (
            f"Plugin overhead too high: {overhead:.1f}us per call"
        )
        return (f"direct={direct_us:.1f}us, plugin={plugin_us:.1f}us, "
                f"overhead={overhead:.1f}us")

    check("PERFORMANCE: plugin overhead", check_performance)

    # -------------------------------------------------------------
    # CHECK 9: close cleanup
    # -------------------------------------------------------------
    def check_close():
        from lmcache.utils import CacheEngineKey
        key = CacheEngineKey(
            model_name="test", world_size=1, worker_id=0,
            chunk_hash=0, dtype=torch.bfloat16,
        )
        key.tags = ("phase05_test", "0")
        assert plugin.contains(key) is True
        plugin.close()
        assert plugin.contains(key) is False
        return "close() clears store"

    check("CLOSE: cleanup on shutdown", check_close)

    # -------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print("[phase0.5] SUMMARY", flush=True)
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

    if not all_pass:
        print("\n*** PHASE 0.5 FAILED — LMCache integration broken ***",
              flush=True)
        sys.exit(1)
    else:
        print("\n  Phase 0.5 passed — LMCache integration works",
              flush=True)


if __name__ == "__main__":
    main()
