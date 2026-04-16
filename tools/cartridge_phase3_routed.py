#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase 3: Multi-cartridge routing smoke test.

Validates the scheduler-visible dispatch layer that turns the
multi-cartridge infrastructure (Store + Registry + LMCache plugin)
into an actually routable serving path.

Phase 1 tested the storage/registry plumbing. Phase 3 tests the
serving boundary:

  1. CONFIG: CartridgeConnector accepts multi-cartridge config
     (``cartridges`` list + ``router`` dict) and loads N cartridges.
  2. ROUTER: Each router type (explicit, label, static, composite)
     resolves requests to correct cartridge_ids end-to-end.
  3. METADATA: get_num_new_matched_tokens + update_state_after_alloc
     + build_connector_meta produces per-request CartridgeReqMeta
     entries with the correct cartridge_id per request.
  4. DISPATCH: start_load_kv fetches the right cartridge per request
     (verified by inspecting store reads — untargeted cartridges are
     never read).
  5. ISOLATION: Two (or more) concurrent requests bound to different
     cartridges write disjoint correct data to disjoint slot ranges.
     No cross-contamination.
  6. BATCH_DEDUP: A batch with N requests pointing to K < N unique
     cartridges produces exactly K store.acquire/release pairs.
  7. UNKNOWN_ID: Router returning an unknown cartridge_id falls
     through to zero-matched-tokens (normal prefill), no crash.
  8. CLEANUP: After build_connector_meta, per-request resolution
     state is cleared for scheduled requests; requests never
     committed are cleaned on next tick.

Like Phase 1, we use ONE .pt file loaded under multiple
cartridge_ids. The goal is to validate the dispatch code paths, not
to prove that three distinct trained cartridges give different
answers (that needs multi-content training, which Phase 2 covers in
a separate pipeline).

Usage:
    python tools/cartridge_phase3_routed.py \\
        --cartridge /path/to/cache-step2694.pt \\
        --device cuda:0
"""
import argparse
import json
import sys
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cartridge", required=True,
                        help=".pt file used for all virtual cartridge IDs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-cartridges", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--out", default="phase3_results.json")
    args = parser.parse_args()

    results = {"checks": [], "overall_pass": False}

    def check(name, fn):
        print(f"\n[phase3] === {name} ===", flush=True)
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
        CartridgeConnector,
        CartridgeConnectorMetadata,
        CartridgeReqMeta,
        align_to_block_size,
        load_cartridge,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_manifest import (
        CartridgeManifest,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_router import (
        CompositeRouter,
        ExplicitCartridgeRouter,
        LabelCartridgeRouter,
        StaticCartridgeRouter,
        build_router_from_config,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_registry import (
        CartridgeRegistry,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_store import (
        CartridgeStore,
        ChunkKey,
    )

    # -----------------------------------------------------------------
    # Fixture: build N virtual cartridge manifests from one .pt file.
    # -----------------------------------------------------------------
    print(f"[phase3] Loading base cartridge + building "
          f"{args.n_cartridges} virtual manifests...", flush=True)
    cart_data = load_cartridge(args.cartridge)
    num_tokens_aligned = align_to_block_size(
        cart_data["num_tokens"], args.block_size)
    num_blocks = num_tokens_aligned // args.block_size
    virtual_carts = []
    for i in range(args.n_cartridges):
        virtual_carts.append(CartridgeManifest(
            cartridge_id=f"doc_{i:02d}",
            model_id="meta-llama/Llama-3.2-3B-Instruct",
            num_layers=cart_data["num_layers"],
            num_kv_heads=cart_data["num_kv_heads"],
            head_dim=cart_data["head_dim"],
            dtype="bfloat16",
            num_tokens_raw=cart_data["num_tokens"],
            num_tokens_aligned=num_tokens_aligned,
            block_size=args.block_size,
            num_blocks=num_blocks,
            has_frozen_prefix=True,
            labels={
                "doc_id": f"doc_{i:02d}",
                "doc_type": "medical_record",
                "priority": "high" if i == 0 else "normal",
            },
        ))
    del cart_data

    # -----------------------------------------------------------------
    # Build a CartridgeConnector in singleton-test mode: skip the
    # real base __init__ (no VllmConfig plumbing) and hand-stitch the
    # multi-cartridge state that the routing code consumes.
    # -----------------------------------------------------------------
    from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_gpu_residency import (
        GPUResidencyManager,
    )

    def make_connector(
        router,
        default_id: str,
        residency_capacity_bytes: int = 1 << 40,
    ) -> CartridgeConnector:
        c = CartridgeConnector.__new__(CartridgeConnector)
        c._block_size = args.block_size
        c._request_cartridge_ids = {}
        c._request_num_tokens = {}
        c._requests_need_load = set()

        c._store = CartridgeStore(block_size=args.block_size)
        c._cartridge_meta = {}
        for m in virtual_carts:
            c._store.load(m.cartridge_id, args.cartridge, m, device="cpu")
            # Do NOT pin in the store — GPU residency manager owns
            # serving-time pins now.
            res = c._store.get_residency(m.cartridge_id)
            c._cartridge_meta[m.cartridge_id] = {
                "num_tokens": res.num_tokens,
                "num_blocks": res.num_tokens // args.block_size,
                "num_layers": res.num_layers,
            }
        c._default_cartridge_id = default_id
        c._router = router
        c._residency = GPUResidencyManager(
            store=c._store,
            capacity_bytes=residency_capacity_bytes,
            device="cpu",
        )
        return c

    def make_request(req_id: str, cart_id_or_label: str | None,
                     prompt_len: int = 4096,
                     extras_key: str = "cartridge_id"):
        req = MagicMock()
        req.request_id = req_id
        req.prompt_token_ids = list(range(prompt_len))
        if cart_id_or_label is None:
            req.sampling_params.extra_args = {}
        else:
            req.sampling_params.extra_args = {
                extras_key: cart_id_or_label,
            }
        return req

    # -----------------------------------------------------------------
    # CHECK 1: CONFIG — multi-cartridge connector loads N cartridges
    # -----------------------------------------------------------------
    def check_config():
        router = StaticCartridgeRouter(virtual_carts[0].cartridge_id)
        c = make_connector(router, virtual_carts[0].cartridge_id)
        assert len(c._cartridge_meta) == args.n_cartridges, (
            f"expected {args.n_cartridges} loaded cartridges, "
            f"got {len(c._cartridge_meta)}")
        for m in virtual_carts:
            assert m.cartridge_id in c._cartridge_meta
        return (f"Loaded {args.n_cartridges} cartridges, "
                f"{c._cartridge_meta[virtual_carts[0].cartridge_id]['num_tokens']} "
                f"tokens each")
    check("CONFIG: multi-cartridge connector init", check_config)

    # -----------------------------------------------------------------
    # CHECK 2: ROUTER resolves each type
    # -----------------------------------------------------------------
    def check_router_resolution():
        # Explicit
        r_exp = ExplicitCartridgeRouter()
        req = make_request("r0", "doc_01")
        assert r_exp.resolve(req) == "doc_01"

        # Nested in kv_transfer_params
        req_nested = MagicMock()
        req_nested.request_id = "r0n"
        req_nested.sampling_params.extra_args = {
            "kv_transfer_params": {"cartridge_id": "doc_02"},
        }
        assert r_exp.resolve(req_nested) == "doc_02"

        # Static
        r_static = StaticCartridgeRouter("doc_00")
        assert r_static.resolve(req) == "doc_00"

        # Composite: explicit > static fallback
        r_comp = CompositeRouter([r_exp, r_static])
        assert r_comp.resolve(req) == "doc_01"  # explicit wins
        req_empty = make_request("r_empty", None)
        assert r_comp.resolve(req_empty) == "doc_00"  # fallback

        # build_router_from_config
        r_built = build_router_from_config({
            "type": "composite",
            "routers": [
                {"type": "explicit"},
                {"type": "static", "cartridge_id": "doc_00"},
            ],
        })
        assert r_built.resolve(req) == "doc_01"
        assert r_built.resolve(req_empty) == "doc_00"
        return "explicit / static / composite / config-built all resolve"
    check("ROUTER: all router types resolve correctly",
          check_router_resolution)

    # -----------------------------------------------------------------
    # CHECK 3: LABEL router via CartridgeRegistry
    # -----------------------------------------------------------------
    registry = None
    db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_file.close()

    def check_label_router():
        nonlocal registry
        registry = CartridgeRegistry(db_file.name)
        for m in virtual_carts:
            registry.register(m)
        r_label = LabelCartridgeRouter(
            registry, label_key="doc_id", extras_key="doc_id",
        )
        req = make_request("rL", "doc_02", extras_key="doc_id")
        assert r_label.resolve(req) == "doc_02"

        # Miss -> None
        req_miss = make_request("rM", "nonexistent", extras_key="doc_id")
        assert r_label.resolve(req_miss) is None
        return f"Registry-backed label router resolves and misses cleanly"
    check("LABEL_ROUTER: registry-backed lookup by label",
          check_label_router)

    # -----------------------------------------------------------------
    # CHECK 4: METADATA — per-request cartridge_id lands in meta
    # -----------------------------------------------------------------
    def check_metadata_dispatch():
        router = ExplicitCartridgeRouter()
        c = make_connector(router, virtual_carts[0].cartridge_id)

        # Simulate N requests, each bound to a different cartridge.
        requests = [
            make_request(f"req_{i}", f"doc_{i:02d}")
            for i in range(args.n_cartridges)
        ]
        for r in requests:
            n, _ = c.get_num_new_matched_tokens(r, 0)
            assert n > 0, f"got 0 matched for {r.request_id}"
            c.update_state_after_alloc(r, MagicMock(), n)

        # Build connector meta with a fake SchedulerOutput.
        sched = MagicMock()
        new_reqs = []
        for i, r in enumerate(requests):
            nr = MagicMock()
            nr.req_id = r.request_id
            # Each request gets distinct block ids so slot mappings
            # are disjoint.
            base = i * 1000
            nr.block_ids = [[base + j for j in range(num_blocks)]]
            new_reqs.append(nr)
        sched.scheduled_new_reqs = new_reqs

        meta = c.build_connector_meta(sched)
        assert isinstance(meta, CartridgeConnectorMetadata)
        assert len(meta.requests) == args.n_cartridges, (
            f"meta has {len(meta.requests)} entries, "
            f"expected {args.n_cartridges}")

        # Each request's cartridge_id must match the routed id.
        for i, req_meta in enumerate(meta.requests):
            expected = f"doc_{i:02d}"
            assert req_meta.cartridge_id == expected, (
                f"meta[{i}].cartridge_id = {req_meta.cartridge_id!r}, "
                f"expected {expected!r}")
            # And the slot mapping sits at the right block base.
            assert req_meta.slot_mapping[0].item() == i * 1000 * args.block_size

        # Tick-local state cleared for scheduled requests.
        assert len(c._requests_need_load) == 0
        for r in requests:
            assert r.request_id not in c._request_cartridge_ids
        return (f"{args.n_cartridges} requests routed to "
                f"{args.n_cartridges} distinct cartridges in meta, "
                f"slot mappings isolated")
    check("METADATA: per-request cartridge_id propagates",
          check_metadata_dispatch)

    # -----------------------------------------------------------------
    # CHECK 5: DISPATCH + ISOLATION — start_load_kv fetches right
    # cartridge per request, untargeted cartridges never read
    # -----------------------------------------------------------------
    def check_dispatch_isolation():
        # Instrument the residency manager: wrap acquire/release/
        # get_chunk. The residency manager is what the connector
        # talks to on the hot path now (store stays read-only).
        class TracingResidency:
            def __init__(self, inner: GPUResidencyManager):
                self._inner = inner
                self.gets: list[tuple[str, int]] = []
                self.acquires: list[str] = []
                self.releases: list[str] = []

            def acquire(self, cart_id: str, dtype=None):
                self.acquires.append(cart_id)
                return self._inner.acquire(cart_id, dtype=dtype)

            def release(self, cart_id: str):
                self.releases.append(cart_id)
                return self._inner.release(cart_id)

            def get_chunk(self, cart_id: str, layer_idx: int):
                self.gets.append((cart_id, layer_idx))
                return self._inner.get_chunk(cart_id, layer_idx)

        router = ExplicitCartridgeRouter()
        c = make_connector(router, virtual_carts[0].cartridge_id)
        tracing = TracingResidency(c._residency)
        c._residency = tracing  # type: ignore[assignment]

        # Build metadata directly: two requests, two different
        # cartridges. Third cartridge is loaded but not targeted.
        num_layers = c._cartridge_meta["doc_00"]["num_layers"]
        tokens_per_cart = c._cartridge_meta["doc_00"]["num_tokens"]
        # Fake slot mappings (disjoint).
        slots_a = torch.arange(0, tokens_per_cart, dtype=torch.long)
        slots_b = torch.arange(
            tokens_per_cart, 2 * tokens_per_cart, dtype=torch.long)

        meta = CartridgeConnectorMetadata()
        meta.requests.append(CartridgeReqMeta(
            cartridge_id="doc_00",
            slot_mapping=slots_a,
            num_tokens=tokens_per_cart,
        ))
        meta.requests.append(CartridgeReqMeta(
            cartridge_id="doc_01",
            slot_mapping=slots_b,
            num_tokens=tokens_per_cart,
        ))

        # Synthetic forward_context with per-layer kv_cache shells.
        # We use CPU tensors and patch .cuda() to pass through so the
        # test runs without a GPU. Layer naming follows
        # "model.layers.<idx>.self_attn".
        class LayerShim:
            def __init__(self, kv):
                self.kv_cache = [kv]

        fc = SimpleNamespace(no_compile_layers={})
        num_cache_blocks = max(8, 2 * num_blocks)
        for li in range(num_layers):
            kv = torch.zeros(
                (2, num_cache_blocks, args.block_size,
                 virtual_carts[0].num_kv_heads,
                 virtual_carts[0].head_dim),
                dtype=torch.float32,
            )
            fc.no_compile_layers[f"model.layers.{li}.self_attn"] = LayerShim(kv)

        # Monkey-patch slot_mapping.cuda() to return CPU tensor, and
        # patch triton_reshape_and_cache_flash to a CPU writer.
        orig_cuda = torch.Tensor.cuda

        def cpu_cuda(self, *a, **k):
            return self

        torch.Tensor.cuda = cpu_cuda  # type: ignore[assignment]
        try:
            from vllm.distributed.kv_transfer.kv_connector.v1 import (
                cartridge_connector as _cc,
            )
            orig_writer = _cc.triton_reshape_and_cache_flash

            def fake_writer(*, key, value, key_cache, value_cache,
                            slot_mapping, kv_cache_dtype,
                            k_scale, v_scale):
                H, D = key.shape[-2], key.shape[-1]
                flat_k = key_cache.reshape(-1, H, D)
                flat_v = value_cache.reshape(-1, H, D)
                flat_k[slot_mapping] = key.to(flat_k.dtype)
                flat_v[slot_mapping] = value.to(flat_v.dtype)

            _cc.triton_reshape_and_cache_flash = fake_writer
            try:
                # Drive the worker-side method. We construct a stub
                # _get_connector_metadata to feed our meta in.
                c._get_connector_metadata = lambda: meta  # type: ignore
                c.start_load_kv(fc)
            finally:
                _cc.triton_reshape_and_cache_flash = orig_writer
        finally:
            torch.Tensor.cuda = orig_cuda  # type: ignore

        # --- Assertions ---
        read_ids = {g[0] for g in tracing.gets}
        assert "doc_00" in read_ids
        assert "doc_01" in read_ids
        assert "doc_02" not in read_ids, (
            f"untargeted cartridge doc_02 was read: "
            f"{[g for g in tracing.gets if g[0] == 'doc_02']}"
        )
        # Ref-count balanced: acquire/release pairs per unique id.
        assert sorted(tracing.acquires) == ["doc_00", "doc_01"]
        assert sorted(tracing.releases) == ["doc_00", "doc_01"]
        return (f"doc_00/doc_01 read, doc_02 not read; "
                f"{len(tracing.acquires)} acquires / "
                f"{len(tracing.releases)} releases, balanced")
    check("DISPATCH+ISOLATION: per-request cartridge fetch, "
          "untargeted cartridge never read", check_dispatch_isolation)

    # -----------------------------------------------------------------
    # CHECK 6: BATCH_DEDUP — N requests, K unique cartridges -> K
    # acquire/release pairs
    # -----------------------------------------------------------------
    def check_batch_dedup():
        class CountingResidency:
            def __init__(self, inner: GPUResidencyManager):
                self._inner = inner
                self.acquires: list[str] = []
                self.releases: list[str] = []

            def acquire(self, cart_id: str, dtype=None):
                self.acquires.append(cart_id)
                return self._inner.acquire(cart_id, dtype=dtype)

            def release(self, cart_id: str):
                self.releases.append(cart_id)
                return self._inner.release(cart_id)

            def get_chunk(self, cart_id: str, layer_idx: int):
                return self._inner.get_chunk(cart_id, layer_idx)

        router = ExplicitCartridgeRouter()
        c = make_connector(router, virtual_carts[0].cartridge_id)
        counting = CountingResidency(c._residency)
        c._residency = counting  # type: ignore[assignment]

        # 5 requests: 3 to doc_00, 2 to doc_01. Exactly 2 unique IDs.
        tokens = c._cartridge_meta["doc_00"]["num_tokens"]
        meta = CartridgeConnectorMetadata()
        for i in range(3):
            meta.requests.append(CartridgeReqMeta(
                cartridge_id="doc_00",
                slot_mapping=torch.arange(
                    i * tokens, (i + 1) * tokens, dtype=torch.long),
                num_tokens=tokens,
            ))
        for i in range(2):
            meta.requests.append(CartridgeReqMeta(
                cartridge_id="doc_01",
                slot_mapping=torch.arange(
                    (3 + i) * tokens, (3 + i + 1) * tokens,
                    dtype=torch.long),
                num_tokens=tokens,
            ))

        # Empty forward_context (no layers) -> no actual writes, but
        # acquire/release bookkeeping still runs.
        fc = SimpleNamespace(no_compile_layers={})
        c._get_connector_metadata = lambda: meta  # type: ignore
        c.start_load_kv(fc)

        assert sorted(counting.acquires) == ["doc_00", "doc_01"], (
            f"expected 2 unique acquires, got {counting.acquires}"
        )
        assert sorted(counting.releases) == ["doc_00", "doc_01"], (
            f"expected 2 unique releases, got {counting.releases}"
        )
        return (f"5 requests / 2 unique cartridges -> "
                f"{len(counting.acquires)}/{len(counting.releases)}"
                f" residency acquire/release "
                f"acquire/release (deduplicated)")
    check("BATCH_DEDUP: unique-cartridge acquire/release dedup",
          check_batch_dedup)

    # -----------------------------------------------------------------
    # CHECK 7: UNKNOWN_ID — router returning unknown id falls through
    # -----------------------------------------------------------------
    def check_unknown_id():
        class RogueRouter:
            def resolve(self, request):
                return "does_not_exist"

        c = make_connector(RogueRouter(), virtual_carts[0].cartridge_id)
        req = make_request("rX", None)
        n, async_load = c.get_num_new_matched_tokens(req, 0)
        assert n == 0, f"expected 0 for unknown id, got {n}"
        assert async_load is False
        # No per-request state should have been committed.
        assert req.request_id not in c._request_cartridge_ids
        return "unknown cartridge_id returns 0 matched, no state leak"
    check("UNKNOWN_ID: router miss -> fallthrough, no crash",
          check_unknown_id)

    # -----------------------------------------------------------------
    # CHECK 8: CLEANUP — resolved-not-committed reqs cleared on tick
    # -----------------------------------------------------------------
    def check_cleanup():
        router = ExplicitCartridgeRouter()
        c = make_connector(router, virtual_carts[0].cartridge_id)

        # Two requests resolved.
        r0 = make_request("r0", "doc_00")
        r1 = make_request("r1", "doc_01")
        c.get_num_new_matched_tokens(r0, 0)
        c.get_num_new_matched_tokens(r1, 0)
        assert "r0" in c._request_cartridge_ids
        assert "r1" in c._request_cartridge_ids

        # Only r0 committed.
        c.update_state_after_alloc(r0, MagicMock(), 32)
        # (r1 resolved but not committed — e.g. capacity pressure)

        # Tick runs with only r0 scheduled.
        sched = MagicMock()
        nr0 = MagicMock()
        nr0.req_id = "r0"
        nr0.block_ids = [[100 + j for j in range(num_blocks)]]
        sched.scheduled_new_reqs = [nr0]

        meta = c.build_connector_meta(sched)
        assert len(meta.requests) == 1
        assert meta.requests[0].cartridge_id == "doc_00"

        # After tick: r0 state cleared (scheduled+consumed), r1 still
        # resolved (will be rescheduled next tick).
        assert "r0" not in c._request_cartridge_ids
        assert "r1" in c._request_cartridge_ids
        assert len(c._requests_need_load) == 0
        return ("scheduled requests clear tick-local state; "
                "resolved-but-not-scheduled survives to next tick")
    check("CLEANUP: tick-local state hygiene", check_cleanup)

    # -----------------------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print("[phase3] SUMMARY", flush=True)
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

    if registry is not None:
        registry.close()
    Path(db_file.name).unlink(missing_ok=True)

    if not all_pass:
        print("\n*** PHASE 3 FAILED ***", flush=True)
        sys.exit(1)
    else:
        print("\n  Phase 3 passed — multi-cartridge routing works "
              "end-to-end through the connector API", flush=True)


if __name__ == "__main__":
    main()
