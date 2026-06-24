#!/usr/bin/env python3
"""Phase 1: Decode-boundary correctness probe for fused INT4 vs baseline.

Runs baseline (FP16 FlashAttention) and fused (INT4) paths with:
  - max_tokens=2  (isolates the first decode step)
  - prompt lengths [1, 2, 15, 16, 17]  (block boundary probing)
  - batch_size=1  (one request at a time, no batching noise)

Uses the vLLM LLM API directly — no server needed.  Saves
machine-readable JSON with per-case token IDs, text, and match status.

Usage (on H100 GPU node with vLLM installed):
    python benchmarks/fused_int4_decode_boundary_probe.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --output /path/to/decode_boundary_probe.json
"""

import argparse
import json
import os
import sys
import time


DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MAX_TOKENS = 2
PROMPT_LENGTHS = [1, 2, 15, 16, 17]

# Deterministic filler for constructing prompts of known token length.
FILLER = ("The quick brown fox jumps over the lazy dog. " * 20)


def make_prompt_of_length(tokenizer, target_len: int) -> tuple[str, list[int]]:
    """Return (prompt_text, token_ids) where len(token_ids) == target_len."""
    if target_len <= 0:
        raise ValueError("target_len must be >= 1")

    tokens = tokenizer.encode(FILLER)
    if len(tokens) < target_len:
        tokens = tokenizer.encode(FILLER * 5)

    # Truncate to target length
    tokens = tokens[:target_len]
    prompt = tokenizer.decode(tokens, skip_special_tokens=True)
    # Verify roundtrip
    actual = tokenizer.encode(prompt)
    if len(actual) != target_len:
        # Fall back: just use raw token IDs
        prompt = tokenizer.decode(tokens[:target_len], skip_special_tokens=True)
        actual = tokenizer.encode(prompt)
    return prompt, actual


def run_mode(
    model_name: str,
    kv_cache_dtype: str,
    prompts: list[dict],
    max_tokens: int,
) -> dict:
    """Run vLLM with specified kv_cache_dtype, one prompt at a time."""
    from vllm import LLM, SamplingParams

    mode_label = "fused" if kv_cache_dtype == "int4_fused" else "baseline"
    print(f"[{mode_label}] Initializing LLM kv_cache_dtype={kv_cache_dtype}",
          file=sys.stderr)

    t0 = time.time()
    llm = LLM(
        model=model_name,
        dtype="float16",
        kv_cache_dtype=kv_cache_dtype,
        max_model_len=2048,
        gpu_memory_utilization=0.8,
        disable_log_stats=True,
        enforce_eager=True,
    )
    init_time = time.time() - t0
    print(f"[{mode_label}] Init took {init_time:.1f}s", file=sys.stderr)

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
    )

    results = []
    for p in prompts:
        t1 = time.time()
        outputs = llm.generate([p["prompt"]], sampling_params)
        gen_time = time.time() - t1
        out = outputs[0].outputs[0]
        token_ids = list(out.token_ids)
        results.append({
            "prompt_label": p["label"],
            "prompt_target_len": p["target_len"],
            "prompt_actual_len": p["actual_len"],
            "prompt": p["prompt"][:200],  # truncate for readability
            "generated_text": out.text,
            "generated_token_ids": token_ids,
            "generate_s": round(gen_time, 4),
        })
        print(f"  [{mode_label}] prompt_len={p['target_len']:2d} -> "
              f"ids={token_ids}  text={out.text!r}", file=sys.stderr)

    del llm
    return {
        "mode": mode_label,
        "kv_cache_dtype": kv_cache_dtype,
        "max_tokens": max_tokens,
        "init_s": round(init_time, 2),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fused INT4 decode-boundary correctness probe")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: stdout)")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--prompt-lengths", type=int, nargs="+",
                        default=PROMPT_LENGTHS)
    args = parser.parse_args()

    max_tokens = args.max_tokens
    prompt_lengths = args.prompt_lengths

    # Build prompts using the model's tokenizer
    from transformers import AutoTokenizer
    print(f"Loading tokenizer for {args.model}...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    prompts = []
    for tlen in prompt_lengths:
        text, tids = make_prompt_of_length(tokenizer, tlen)
        prompts.append({
            "label": f"len_{tlen}",
            "target_len": tlen,
            "actual_len": len(tids),
            "prompt": text,
            "prompt_token_ids": tids,
        })

    print(f"Prompt lengths (actual): {[p['actual_len'] for p in prompts]}",
          file=sys.stderr)
    print(f"max_tokens={max_tokens}, batch_size=1", file=sys.stderr)

    # Run baseline then fused
    baseline = run_mode(args.model, "auto", prompts, max_tokens)
    fused = run_mode(args.model, "int4_fused", prompts, max_tokens)

    # Token-level comparison
    comparisons = []
    for b_res, f_res in zip(baseline["results"], fused["results"]):
        b_ids = b_res["generated_token_ids"]
        f_ids = f_res["generated_token_ids"]
        match_token1 = (len(b_ids) > 0 and len(f_ids) > 0
                        and b_ids[0] == f_ids[0])
        match_token2 = (len(b_ids) > 1 and len(f_ids) > 1
                        and b_ids[1] == f_ids[1])
        match_all = b_ids == f_ids
        comparisons.append({
            "prompt_label": b_res["prompt_label"],
            "prompt_len": b_res["prompt_target_len"],
            "baseline_ids": b_ids,
            "fused_ids": f_ids,
            "baseline_text": b_res["generated_text"],
            "fused_text": f_res["generated_text"],
            "token1_match": match_token1,
            "token2_match": match_token2,
            "all_match": match_all,
        })

    summary = {
        "total_cases": len(comparisons),
        "token1_match_count": sum(
            1 for c in comparisons if c["token1_match"]),
        "token2_match_count": sum(
            1 for c in comparisons if c["token2_match"]),
        "all_match_count": sum(
            1 for c in comparisons if c["all_match"]),
    }

    manifest = {
        "probe": "decode_boundary_probe",
        "model": args.model,
        "max_tokens": max_tokens,
        "prompt_lengths": prompt_lengths,
        "batch_size": 1,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "baseline": baseline,
        "fused": fused,
        "comparisons": comparisons,
        "summary": summary,
    }

    json_str = json.dumps(manifest, indent=2, ensure_ascii=False)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            f.write(json_str + "\n")
        print(f"\nOutput written to {args.output}", file=sys.stderr)
    else:
        print(json_str)

    # Summary to stderr
    print(f"\n=== Decode Boundary Probe Summary ===", file=sys.stderr)
    print(f"Token 1 match: {summary['token1_match_count']}"
          f"/{summary['total_cases']}", file=sys.stderr)
    print(f"Token 2 match: {summary['token2_match_count']}"
          f"/{summary['total_cases']}", file=sys.stderr)
    print(f"All match:     {summary['all_match_count']}"
          f"/{summary['total_cases']}", file=sys.stderr)
    status = "PASS" if summary["all_match_count"] == summary["total_cases"] \
        else "FAIL"
    print(f"Status: {status}", file=sys.stderr)


if __name__ == "__main__":
    main()
