#!/usr/bin/env python3
"""
Probe: Separate-cache fix for fused INT4 backend.

Tests the structural fix where:
  - FP16 paged cache is written by standard reshape_and_cache_flash
  - Dedicated INT4 packed cache (contiguous uint8) is separate
  - Decode uses fused INT4 kernel reading from dedicated buffer
  - Short sequences fall back to FP16 paged cache via SDPA

Compares baseline (FlashAttention FP16) vs fused INT4 (int4_fused)
across a range of prompt lengths including the previously-failing ones.
"""

import argparse
import gc
import json
import os
import sys
import time

# Don't set VLLM_FUSED_INT4_MIN_SEQ_LEN — let the source default apply.
# The default in fused_int4.py is 999999 (FP16 paged SDPA for all decodes).
# Override via command line: VLLM_FUSED_INT4_MIN_SEQ_LEN=8 python ...


def _cleanup_gpu():
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def make_prompt(length: int) -> str:
    phrase = "The quick brown fox jumps over the lazy dog. "
    repeats = max(1, (length * 2) // len(phrase) + 1)
    return (phrase * repeats)[:length * 5]


def run_config(model, lengths, max_tokens, kv_cache_dtype, config_name):
    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=model,
        dtype="float16",
        max_model_len=512,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
    )
    if kv_cache_dtype:
        kwargs["kv_cache_dtype"] = kv_cache_dtype

    llm = LLM(**kwargs)
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    results = {}
    for length in lengths:
        prompt = make_prompt(length)
        result = llm.generate([prompt], params)
        ids = list(result[0].outputs[0].token_ids)
        text = result[0].outputs[0].text
        results[length] = {"ids": ids, "text": text}
        print(f"  [{config_name}] len={length}: {ids} -> {text!r}")

    del llm
    _cleanup_gpu()
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Separate-cache fix probe for fused INT4")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-tokens", type=int, default=5)
    parser.add_argument(
        "--lengths", type=str,
        default="17,18,20,26,28,32,47,48,50,64,80,96,128,160,256",
    )
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    lengths = [int(x) for x in args.lengths.split(",")]

    report = {
        "probe": "separate_cache_fix",
        "model": args.model,
        "max_tokens": args.max_tokens,
        "min_fused_seq_len": int(os.environ.get(
            "VLLM_FUSED_INT4_MIN_SEQ_LEN", "999999")),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "comparisons": [],
    }

    # Phase 1: Baseline (FlashAttention FP16)
    print("=" * 60)
    print("Phase 1: Baseline (FlashAttention FP16)")
    print("=" * 60)
    baseline = run_config(
        args.model, lengths, args.max_tokens,
        kv_cache_dtype=None, config_name="baseline",
    )

    # Phase 2: Fused INT4 (separate-cache fix)
    print("\n" + "=" * 60)
    print("Phase 2: Fused INT4 (separate-cache fix)")
    print("=" * 60)
    fused = run_config(
        args.model, lengths, args.max_tokens,
        kv_cache_dtype="int4_fused", config_name="fused_int4",
    )

    # Compare
    for length in lengths:
        bl = baseline.get(length, {})
        fu = fused.get(length, {})
        match = bl.get("ids", []) == fu.get("ids", [])

        comp = {
            "prompt_len": length,
            "baseline_ids": bl.get("ids", []),
            "fused_ids": fu.get("ids", []),
            "match": match,
            "baseline_text": bl.get("text", ""),
            "fused_text": fu.get("text", ""),
        }
        report["comparisons"].append(comp)

        tag = "OK" if match else "FAIL"
        print(f"\nlen={length}: {tag}")
        print(f"  baseline: {bl.get('ids', [])}")
        print(f"  fused:    {fu.get('ids', [])}")

    passes = sum(1 for c in report["comparisons"] if c["match"])
    total = len(report["comparisons"])

    report["summary"] = {
        "total": total,
        "passes": passes,
        "pass_rate": f"{passes}/{total}",
    }

    if passes == total:
        report["verdict"] = "SEPARATE_CACHE_FIX_WORKS"
    elif passes > total * 0.8:
        report["verdict"] = "MOSTLY_FIXED"
    else:
        report["verdict"] = "STILL_BROKEN"

    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {passes}/{total} match baseline")
    print(f"VERDICT: {report['verdict']}")
    print(f"{'=' * 60}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Results written to {args.output}")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
