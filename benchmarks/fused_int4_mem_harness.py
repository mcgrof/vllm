#!/usr/bin/env python3
"""B0 memory-accounting harness for fused INT4 KV cache.

Instruments _ensure_int4_cache and the vLLM memory planner to produce
a byte-level accounting of:
  - planner-budgeted cache size
  - actual allocated shadow cache size (the "ghost" ~1.75 GiB)
  - per-tensor allocation breakdown (shapes, dtypes, bytes)
  - what gpu_memory_utilization ceiling restores if fixed
  - whether CUDA graph capture can succeed

Output: JSON to stdout.

Usage:
    python benchmarks/fused_int4_mem_harness.py [--model Qwen/Qwen2.5-1.5B]
"""

import argparse
import gc
import json
import os
import sys
import time
import traceback

import torch


def bytes_to_gib(b):
    return b / (1024 ** 3)


def tensor_bytes(t):
    return t.nelement() * t.element_size()


def get_gpu_memory():
    """Get current GPU memory stats."""
    torch.cuda.synchronize()
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "max_allocated_bytes": torch.cuda.max_memory_allocated(),
        "total_bytes": torch.cuda.get_device_properties(0).total_mem,
        "free_bytes": torch.cuda.get_device_properties(0).total_mem - torch.cuda.memory_allocated(),
    }


def measure_fused_cache_allocation(model_name, gpu_mem_util=0.50):
    """Load model with fused INT4 and measure the cache allocation.

    Returns a detailed accounting of all shadow cache tensors.
    """
    from vllm import LLM, SamplingParams

    # Snapshot memory before loading
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mem_before_load = get_gpu_memory()

    print(f"[mem] Loading model with int4_fused, gpu_mem={gpu_mem_util}...",
          file=sys.stderr)
    t0 = time.time()

    # Monkey-patch _ensure_int4_cache to capture allocation details
    allocation_log = []
    original_ensure = None

    try:
        from vllm.v1.attention.backends.fused_int4 import FusedInt4AttentionImpl

        original_ensure = FusedInt4AttentionImpl._ensure_int4_cache

        def patched_ensure(self, num_blocks, block_size, device):
            mem_pre = torch.cuda.memory_allocated()
            original_ensure(self, num_blocks, block_size, device)
            mem_post = torch.cuda.memory_allocated()

            # Inventory all shadow cache tensors
            tensors = {}
            for attr_name in [
                "_int4_value_cache", "_v_scales", "_v_zeros",
                "_int4_key_cache", "_k_scales", "_k_zeros",
                "_int8_key_cache", "_k_int8_scales",
                "_scratch_k_cache", "_scratch_k_scales",
            ]:
                t = getattr(self, attr_name, None)
                if t is not None:
                    tensors[attr_name] = {
                        "shape": list(t.shape),
                        "dtype": str(t.dtype),
                        "bytes": tensor_bytes(t),
                        "gib": bytes_to_gib(tensor_bytes(t)),
                    }

            allocation_log.append({
                "num_blocks": num_blocks,
                "block_size": block_size,
                "num_kv_heads": self.num_kv_heads,
                "head_size": self.head_size,
                "half_hd": self.half_hd,
                "num_groups": self.num_groups,
                "k_precision": self._k_precision,
                "asymmetric": self._asymmetric,
                "mem_before_bytes": mem_pre,
                "mem_after_bytes": mem_post,
                "delta_bytes": mem_post - mem_pre,
                "delta_gib": bytes_to_gib(mem_post - mem_pre),
                "tensors": tensors,
                "total_shadow_bytes": sum(t["bytes"] for t in tensors.values()),
                "total_shadow_gib": bytes_to_gib(
                    sum(t["bytes"] for t in tensors.values())
                ),
            })

        FusedInt4AttentionImpl._ensure_int4_cache = patched_ensure
    except ImportError:
        print("[mem] WARNING: Could not monkey-patch _ensure_int4_cache",
              file=sys.stderr)

    llm = LLM(
        model=model_name,
        dtype="float16",
        kv_cache_dtype="int4_fused",
        max_model_len=2048,
        gpu_memory_utilization=gpu_mem_util,
        disable_log_stats=True,
        enforce_eager=True,
        max_num_seqs=1,
    )
    load_time = time.time() - t0

    mem_after_load = get_gpu_memory()

    # Run one generation to trigger _ensure_int4_cache
    print("[mem] Running one generation to trigger cache allocation...",
          file=sys.stderr)
    sp = SamplingParams(max_tokens=50, temperature=0.0)
    mem_before_gen = get_gpu_memory()
    _ = llm.generate(["Hello, world!"], sp)
    mem_after_gen = get_gpu_memory()

    # Restore original
    if original_ensure is not None:
        FusedInt4AttentionImpl._ensure_int4_cache = original_ensure

    # Compute what the planner thinks it budgeted
    total_gpu_bytes = torch.cuda.get_device_properties(0).total_mem
    planner_budget_bytes = int(total_gpu_bytes * gpu_mem_util)

    result = {
        "model": model_name,
        "gpu_memory_utilization": gpu_mem_util,
        "total_gpu_bytes": total_gpu_bytes,
        "total_gpu_gib": bytes_to_gib(total_gpu_bytes),
        "planner_budget_bytes": planner_budget_bytes,
        "planner_budget_gib": bytes_to_gib(planner_budget_bytes),
        "load_time_s": load_time,
        "mem_before_load": mem_before_load,
        "mem_after_load": mem_after_load,
        "mem_before_gen": mem_before_gen,
        "mem_after_gen": mem_after_gen,
        "model_weight_bytes": mem_after_load["allocated_bytes"] - mem_before_load["allocated_bytes"],
        "model_weight_gib": bytes_to_gib(
            mem_after_load["allocated_bytes"] - mem_before_load["allocated_bytes"]
        ),
        "gen_allocation_delta_bytes": mem_after_gen["allocated_bytes"] - mem_before_gen["allocated_bytes"],
        "gen_allocation_delta_gib": bytes_to_gib(
            mem_after_gen["allocated_bytes"] - mem_before_gen["allocated_bytes"]
        ),
        "allocation_log": allocation_log,
    }

    # Cleanup
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)

    return result


def test_gpu_mem_levels(model_name):
    """Test which gpu_memory_utilization levels work with and without fused."""
    from vllm import LLM, SamplingParams

    levels = [0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50]
    results = []

    for level in levels:
        gc.collect()
        torch.cuda.empty_cache()
        print(f"\n[mem] Testing gpu_mem={level} with int4_fused...",
              file=sys.stderr)

        try:
            llm = LLM(
                model=model_name,
                dtype="float16",
                kv_cache_dtype="int4_fused",
                max_model_len=2048,
                gpu_memory_utilization=level,
                disable_log_stats=True,
                enforce_eager=True,
                max_num_seqs=1,
            )
            sp = SamplingParams(max_tokens=20, temperature=0.0)
            out = llm.generate(["Hello"], sp)
            success = len(out) > 0 and len(out[0].outputs[0].token_ids) > 0
            mem = get_gpu_memory()
            results.append({
                "gpu_mem": level,
                "kv_dtype": "int4_fused",
                "success": success,
                "allocated_gib": bytes_to_gib(mem["allocated_bytes"]),
                "error": None,
            })
            del llm
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(2)
            print(f"[mem]   gpu_mem={level}: SUCCESS", file=sys.stderr)
        except Exception as e:
            results.append({
                "gpu_mem": level,
                "kv_dtype": "int4_fused",
                "success": False,
                "error": str(e)[:500],
            })
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(2)
            print(f"[mem]   gpu_mem={level}: FAILED ({e})", file=sys.stderr)

    return results


def test_cuda_graph_capture(model_name):
    """Test if CUDA graph capture works with fused INT4."""
    from vllm import LLM, SamplingParams

    results = {}
    for enforce_eager in [True, False]:
        gc.collect()
        torch.cuda.empty_cache()
        label = "eager" if enforce_eager else "graph"
        print(f"\n[mem] Testing CUDA graph capture (enforce_eager={enforce_eager})...",
              file=sys.stderr)
        try:
            llm = LLM(
                model=model_name,
                dtype="float16",
                kv_cache_dtype="int4_fused",
                max_model_len=2048,
                gpu_memory_utilization=0.50,
                disable_log_stats=True,
                enforce_eager=enforce_eager,
                max_num_seqs=1,
            )
            sp = SamplingParams(max_tokens=20, temperature=0.0)
            out = llm.generate(["Hello, world!"], sp)
            success = len(out) > 0 and len(out[0].outputs[0].token_ids) > 0
            results[label] = {
                "enforce_eager": enforce_eager,
                "success": success,
                "error": None,
            }
            print(f"[mem]   {label}: SUCCESS", file=sys.stderr)
            del llm
        except Exception as e:
            results[label] = {
                "enforce_eager": enforce_eager,
                "success": False,
                "error": str(e)[:500],
            }
            print(f"[mem]   {label}: FAILED ({e})", file=sys.stderr)
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(2)

    return results


def main():
    parser = argparse.ArgumentParser(description="B0 memory-accounting harness")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B",
                        help="Model to test (default: Qwen/Qwen2.5-1.5B)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: skip gpu_mem sweep, test only core accounting")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: stdout)")
    args = parser.parse_args()

    print(f"[mem] Model: {args.model}", file=sys.stderr)
    print(f"[mem] GPU: {torch.cuda.get_device_name(0)}", file=sys.stderr)
    print(f"[mem] Total GPU memory: {bytes_to_gib(torch.cuda.get_device_properties(0).total_mem):.2f} GiB",
          file=sys.stderr)

    # --- Core accounting ---
    print("\n=== CORE MEMORY ACCOUNTING ===", file=sys.stderr)
    accounting = measure_fused_cache_allocation(args.model, gpu_mem_util=0.50)

    # --- GPU mem level sweep ---
    gpu_mem_sweep = []
    if not args.quick:
        print("\n=== GPU MEMORY LEVEL SWEEP ===", file=sys.stderr)
        gpu_mem_sweep = test_gpu_mem_levels(args.model)

    # --- CUDA graph test ---
    print("\n=== CUDA GRAPH CAPTURE TEST ===", file=sys.stderr)
    cuda_graph_results = test_cuda_graph_capture(args.model)

    # --- Summary ---
    shadow_total = 0
    if accounting["allocation_log"]:
        shadow_total = accounting["allocation_log"][0].get("total_shadow_bytes", 0)

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"MEMORY HARNESS SUMMARY", file=sys.stderr)
    print(f"  Model weights:       {accounting['model_weight_gib']:.3f} GiB", file=sys.stderr)
    print(f"  Shadow cache total:  {bytes_to_gib(shadow_total):.3f} GiB", file=sys.stderr)
    print(f"  Gen alloc delta:     {accounting['gen_allocation_delta_gib']:.3f} GiB", file=sys.stderr)
    print(f"  Planner budget:      {accounting['planner_budget_gib']:.3f} GiB (at gpu_mem=0.50)", file=sys.stderr)

    if accounting["allocation_log"]:
        log = accounting["allocation_log"][0]
        print(f"  Num blocks:          {log['num_blocks']}", file=sys.stderr)
        print(f"  Block size:          {log['block_size']}", file=sys.stderr)
        print(f"  K precision:         {log['k_precision']}", file=sys.stderr)
        for name, info in log.get("tensors", {}).items():
            print(f"    {name}: shape={info['shape']} dtype={info['dtype']} "
                  f"= {info['gib']:.4f} GiB", file=sys.stderr)

    max_working_level = None
    for r in gpu_mem_sweep:
        if r["success"]:
            if max_working_level is None or r["gpu_mem"] > max_working_level:
                max_working_level = r["gpu_mem"]
    if max_working_level is not None:
        print(f"  Max working gpu_mem: {max_working_level}", file=sys.stderr)

    eager_ok = cuda_graph_results.get("eager", {}).get("success", False)
    graph_ok = cuda_graph_results.get("graph", {}).get("success", False)
    print(f"  Eager mode:          {'OK' if eager_ok else 'FAIL'}", file=sys.stderr)
    print(f"  CUDA graph capture:  {'OK' if graph_ok else 'FAIL'}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    # --- Build output ---
    manifest = {
        "harness": "fused_int4_mem_harness",
        "version": "1.0",
        "model": args.model,
        "env": {
            "gpu": torch.cuda.get_device_name(0),
            "total_gpu_gib": bytes_to_gib(torch.cuda.get_device_properties(0).total_mem),
            "k_precision": os.environ.get("VLLM_FUSED_INT4_K_PRECISION", "int4"),
        },
        "core_accounting": accounting,
        "gpu_mem_sweep": gpu_mem_sweep,
        "cuda_graph_results": cuda_graph_results,
        "summary": {
            "model_weight_gib": accounting["model_weight_gib"],
            "shadow_cache_total_gib": bytes_to_gib(shadow_total),
            "gen_alloc_delta_gib": accounting["gen_allocation_delta_gib"],
            "max_working_gpu_mem": max_working_level,
            "eager_mode_ok": eager_ok,
            "cuda_graph_ok": graph_ok,
            "planner_mismatch_gib": bytes_to_gib(shadow_total),
            "restored_gpu_mem_ceiling": max_working_level,
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    output = json.dumps(manifest, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"\nResults written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
