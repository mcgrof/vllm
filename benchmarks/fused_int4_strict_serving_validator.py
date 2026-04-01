#!/usr/bin/env python3
"""Strict server-path validator for fused INT4 KV cache.

Compares FP16 baseline output to fused INT4 output via the LLM(...)
serving path.  Uses EXACT token-id agreement — prefix-only matches
are hard FAILs.

Usage:
    python benchmarks/fused_int4_strict_serving_validator.py [--json-out PATH]

Produces a JSON report with per-prompt PASS/FAIL and an overall verdict.
"""

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone

MODEL = "Qwen/Qwen2.5-7B-Instruct"
MAX_MODEL_LEN = 2048
GPU_MEM_UTIL = 0.80
MAX_TOKENS = 30  # Enough to catch garbage continuations
TEMPERATURE = 0.0  # Greedy for reproducibility

# Diverse prompt set: short, medium, and boundary-length prompts
PROMPTS = [
    "What is 2 + 2?",
    "The capital of France is",
    "Explain quantum computing in one sentence.",
    "Write a haiku about the ocean.",
    "List the first 5 prime numbers.",
    "The quick brown fox jumps over the lazy dog.",
    # Longer prompts that will exceed MSL=48 tokens
    ("The quick brown fox jumps over the lazy dog. "
     "The quick brown fox jumps over the lazy dog. "
     "The quick brown fox jumps over the lazy dog. "
     "Summarize the above."),
    ("In the year 2024, artificial intelligence continued to advance "
     "at a rapid pace. Large language models became more capable and "
     "efficient, enabling new applications across many industries. "
     "What is the main topic of this passage?"),
    # Very short
    "Hi",
    "1+1=",
    # Medium
    "Tell me a short joke about programmers.",
    "What are the three states of matter?",
]


def run_llm_path(prompts, kv_cache_dtype, label):
    """Run prompts through LLM(...) path and return token ids + text."""
    from vllm import LLM, SamplingParams

    print(f"\n[{label}] Loading model with kv_cache_dtype={kv_cache_dtype}...",
          file=sys.stderr)

    llm = LLM(
        model=MODEL,
        dtype="float16",
        kv_cache_dtype=kv_cache_dtype,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM_UTIL,
        disable_log_stats=True,
        enforce_eager=True,
    )

    sp = SamplingParams(max_tokens=MAX_TOKENS, temperature=TEMPERATURE)

    results = []
    for i, prompt in enumerate(prompts):
        outputs = llm.generate([prompt], sp)
        out = outputs[0].outputs[0]
        token_ids = list(out.token_ids)
        text = out.text
        results.append({
            "prompt_index": i,
            "prompt": prompt[:80] + ("..." if len(prompt) > 80 else ""),
            "token_ids": token_ids,
            "text": text,
            "num_tokens": len(token_ids),
        })
        print(f"  [{label}] prompt {i:2d}: {len(token_ids)} tokens, "
              f"ids={token_ids[:8]}{'...' if len(token_ids) > 8 else ''}",
              file=sys.stderr)

    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return results


def compare_results(baseline, fused):
    """Compare baseline vs fused results with strict token-id matching."""
    comparisons = []
    all_pass = True

    for b, f in zip(baseline, fused):
        b_ids = b["token_ids"]
        f_ids = f["token_ids"]

        exact_match = (b_ids == f_ids)

        # Check for prefix-only match (partial)
        min_len = min(len(b_ids), len(f_ids))
        prefix_match_len = 0
        for j in range(min_len):
            if b_ids[j] == f_ids[j]:
                prefix_match_len += 1
            else:
                break

        is_prefix_only = (prefix_match_len > 0 and not exact_match)

        # Detect garbage: repeated tokens, very different lengths, etc.
        garbage_indicators = []
        if len(f_ids) >= 3:
            # Check for token repetition (same token 3+ times in a row)
            for k in range(2, len(f_ids)):
                if f_ids[k] == f_ids[k-1] == f_ids[k-2]:
                    garbage_indicators.append(
                        f"triple_repeat_at_{k}(id={f_ids[k]})")
                    break

        # Overall verdict for this prompt
        verdict = "PASS" if exact_match else "FAIL"
        fail_reason = None
        if not exact_match:
            all_pass = False
            if is_prefix_only:
                fail_reason = (f"prefix_only_match({prefix_match_len}/"
                               f"{len(b_ids)})")
            else:
                fail_reason = "token_mismatch"
            if garbage_indicators:
                fail_reason += f" garbage={garbage_indicators}"

        comparisons.append({
            "prompt_index": b["prompt_index"],
            "prompt": b["prompt"],
            "verdict": verdict,
            "fail_reason": fail_reason,
            "exact_match": exact_match,
            "prefix_match_len": prefix_match_len,
            "baseline_ids": b_ids,
            "fused_ids": f_ids,
            "baseline_text": b["text"],
            "fused_text": f["text"],
            "garbage_indicators": garbage_indicators,
        })

    return comparisons, all_pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", default=None,
                        help="Path to write JSON report")
    parser.add_argument("--msl", type=int, default=None,
                        help="Override VLLM_FUSED_INT4_MIN_SEQ_LEN")
    args = parser.parse_args()

    if args.msl is not None:
        os.environ["VLLM_FUSED_INT4_MIN_SEQ_LEN"] = str(args.msl)
        print(f"[validator] MSL override: {args.msl}", file=sys.stderr)

    start = time.time()

    # Phase 1: FP16 baseline
    baseline = run_llm_path(PROMPTS, "auto", "baseline")

    # Phase 2: Fused INT4
    fused = run_llm_path(PROMPTS, "int4_fused", "fused")

    # Phase 3: Strict comparison
    comparisons, all_pass = compare_results(baseline, fused)

    elapsed = time.time() - start

    # Report
    num_pass = sum(1 for c in comparisons if c["verdict"] == "PASS")
    num_fail = sum(1 for c in comparisons if c["verdict"] == "FAIL")

    report = {
        "validator": "fused_int4_strict_serving_validator",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "msl": os.environ.get("VLLM_FUSED_INT4_MIN_SEQ_LEN", "48"),
        "overall_verdict": "PASS" if all_pass else "FAIL",
        "num_pass": num_pass,
        "num_fail": num_fail,
        "total_prompts": len(PROMPTS),
        "elapsed_seconds": round(elapsed, 1),
        "comparisons": comparisons,
    }

    # Print summary
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"STRICT VALIDATOR: {'PASS' if all_pass else 'FAIL'}", file=sys.stderr)
    print(f"  {num_pass}/{len(PROMPTS)} passed, {num_fail} failed", file=sys.stderr)
    print(f"  Elapsed: {elapsed:.1f}s", file=sys.stderr)
    if not all_pass:
        print(f"\nFailing prompts:", file=sys.stderr)
        for c in comparisons:
            if c["verdict"] == "FAIL":
                print(f"  [{c['prompt_index']}] {c['prompt']}", file=sys.stderr)
                print(f"      reason: {c['fail_reason']}", file=sys.stderr)
                print(f"      baseline: {c['baseline_text'][:60]!r}",
                      file=sys.stderr)
                print(f"      fused:    {c['fused_text'][:60]!r}",
                      file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    # Write JSON
    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report written to {args.json_out}", file=sys.stderr)

    # Also dump to stdout
    print(json.dumps(report, indent=2))

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
