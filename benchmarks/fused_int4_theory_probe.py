#!/usr/bin/env python3
"""Theory-discriminating probe: find the exact failure cliff,
test min_seq_len policies, and isolate exposure-length vs token-position.

Theories under test:
  T1: Accumulated INT4 quality loss (inherent) — cliff-finder + token sweep
  T2/T3: Quantization regime (symmetric, group_size) — would need code change
  T5: Decode kernel numerical issue — cliff around BLOCK_N=64 boundary
  T6: Shadow window policy — min_seq_len sweep
  T8: BLOCK_N tile boundary — cliff at exactly 64

Probes:
  1. cliff_finder: prompt lengths 17-80 step 1, find exact pass/fail boundary
  2. min_seq_len_sweep: rerun failing lengths with VLLM_FUSED_INT4_MIN_SEQ_LEN
     set to 32, 48, 64, 96, 128
  3. token_depth: at failing lengths, generate 1-10 tokens to find divergence point
  4. extended_cliff: prompt lengths 80-256 step 8 to see if failures continue
"""

import json
import os
import sys
import time
import gc
import subprocess

MODEL = "Qwen/Qwen2.5-7B-Instruct"
FILLER = ("The quick brown fox jumps over the lazy dog. " * 40)


def make_prompt(tokenizer, target_len):
    tokens = tokenizer.encode(FILLER)[:target_len]
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    actual = tokenizer.encode(text)
    return {
        "target_len": target_len,
        "actual_len": len(actual),
        "text": text,
    }


def run_single(llm, text, max_tokens):
    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    out = llm.generate([text], sp)[0].outputs[0]
    return list(out.token_ids), out.text


def probe_cliff_finder(output_dir):
    """Probe 1: Fine-grained prompt length sweep to find failure cliff."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print("=== PROBE 1: Cliff Finder (len 17-80, step 1) ===", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    lengths = list(range(17, 81))
    prompts = [make_prompt(tokenizer, l) for l in lengths]

    max_tokens = 3
    results = []

    # Run baseline
    print("[baseline] Init...", file=sys.stderr)
    llm_base = LLM(
        model=MODEL, dtype="float16", kv_cache_dtype="auto",
        max_model_len=2048, gpu_memory_utilization=0.8,
        disable_log_stats=True, enforce_eager=True,
    )
    baseline = {}
    for p in prompts:
        ids, txt = run_single(llm_base, p["text"], max_tokens)
        baseline[p["actual_len"]] = {"ids": ids, "text": txt}
        print(f"  [base] len={p['actual_len']:3d} ids={ids} text={txt!r}",
              file=sys.stderr)
    del llm_base
    gc.collect()

    # Run fused
    print("[fused] Init...", file=sys.stderr)
    llm_fused = LLM(
        model=MODEL, dtype="float16", kv_cache_dtype="int4_fused",
        max_model_len=2048, gpu_memory_utilization=0.8,
        disable_log_stats=True, enforce_eager=True,
    )
    for p in prompts:
        ids, txt = run_single(llm_fused, p["text"], max_tokens)
        bl = baseline[p["actual_len"]]
        match = bl["ids"] == ids
        status = "OK" if match else "FAIL"
        print(f"  [fused] len={p['actual_len']:3d} ids={ids} text={txt!r} [{status}]",
              file=sys.stderr)
        results.append({
            "prompt_len": p["actual_len"],
            "baseline_ids": bl["ids"],
            "fused_ids": ids,
            "baseline_text": bl["text"],
            "fused_text": txt,
            "match": match,
        })
    del llm_fused
    gc.collect()

    manifest = {
        "probe": "cliff_finder",
        "model": MODEL,
        "max_tokens": max_tokens,
        "lengths": [r["prompt_len"] for r in results],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": results,
        "summary": {
            "total": len(results),
            "passes": sum(1 for r in results if r["match"]),
            "failures": [r["prompt_len"] for r in results if not r["match"]],
            "first_fail": next((r["prompt_len"] for r in results
                               if not r["match"]), None),
        },
    }
    path = os.path.join(output_dir, "cliff_finder.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Cliff finder saved to {path}", file=sys.stderr)
    return manifest


def probe_extended_cliff(output_dir):
    """Probe 2: Extended cliff (len 80-256, step 8)."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print("=== PROBE 2: Extended Cliff (len 80-256, step 8) ===", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    lengths = list(range(80, 257, 8))
    prompts = [make_prompt(tokenizer, l) for l in lengths]

    max_tokens = 3
    results = []

    print("[baseline] Init...", file=sys.stderr)
    llm_base = LLM(
        model=MODEL, dtype="float16", kv_cache_dtype="auto",
        max_model_len=2048, gpu_memory_utilization=0.8,
        disable_log_stats=True, enforce_eager=True,
    )
    baseline = {}
    for p in prompts:
        ids, txt = run_single(llm_base, p["text"], max_tokens)
        baseline[p["actual_len"]] = {"ids": ids, "text": txt}
    del llm_base
    gc.collect()

    print("[fused] Init...", file=sys.stderr)
    llm_fused = LLM(
        model=MODEL, dtype="float16", kv_cache_dtype="int4_fused",
        max_model_len=2048, gpu_memory_utilization=0.8,
        disable_log_stats=True, enforce_eager=True,
    )
    for p in prompts:
        ids, txt = run_single(llm_fused, p["text"], max_tokens)
        bl = baseline[p["actual_len"]]
        match = bl["ids"] == ids
        status = "OK" if match else "FAIL"
        print(f"  [ext] len={p['actual_len']:3d} ids={ids} [{status}]",
              file=sys.stderr)
        results.append({
            "prompt_len": p["actual_len"],
            "baseline_ids": bl["ids"],
            "fused_ids": ids,
            "baseline_text": bl["text"],
            "fused_text": txt,
            "match": match,
        })
    del llm_fused
    gc.collect()

    manifest = {
        "probe": "extended_cliff",
        "model": MODEL,
        "max_tokens": max_tokens,
        "lengths": [r["prompt_len"] for r in results],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": results,
        "summary": {
            "total": len(results),
            "passes": sum(1 for r in results if r["match"]),
            "failures": [r["prompt_len"] for r in results if not r["match"]],
        },
    }
    path = os.path.join(output_dir, "extended_cliff.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Extended cliff saved to {path}", file=sys.stderr)
    return manifest


def probe_token_depth(output_dir):
    """Probe 3: At failing lengths, how many tokens before divergence?"""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print("=== PROBE 3: Token Depth (1-10 tokens at len 32, 64) ===",
          file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    test_lengths = [32, 64]
    token_counts = list(range(1, 11))
    prompts = {l: make_prompt(tokenizer, l) for l in test_lengths}

    results = []

    print("[baseline] Init...", file=sys.stderr)
    llm_base = LLM(
        model=MODEL, dtype="float16", kv_cache_dtype="auto",
        max_model_len=2048, gpu_memory_utilization=0.8,
        disable_log_stats=True, enforce_eager=True,
    )
    baseline = {}
    for l in test_lengths:
        for tc in token_counts:
            ids, txt = run_single(llm_base, prompts[l]["text"], tc)
            baseline[(l, tc)] = {"ids": ids, "text": txt}
    del llm_base
    gc.collect()

    print("[fused] Init...", file=sys.stderr)
    llm_fused = LLM(
        model=MODEL, dtype="float16", kv_cache_dtype="int4_fused",
        max_model_len=2048, gpu_memory_utilization=0.8,
        disable_log_stats=True, enforce_eager=True,
    )
    for l in test_lengths:
        for tc in token_counts:
            ids, txt = run_single(llm_fused, prompts[l]["text"], tc)
            bl = baseline[(l, tc)]
            # Check token-by-token match
            matches = []
            for i in range(min(len(bl["ids"]), len(ids))):
                matches.append(bl["ids"][i] == ids[i])
            first_diverge = next((i for i, m in enumerate(matches) if not m),
                                 None)
            all_match = all(matches) and len(bl["ids"]) == len(ids)
            status = "OK" if all_match else f"DIVERGE@{first_diverge}"
            print(f"  [depth] len={l} tokens={tc} [{status}]",
                  file=sys.stderr)
            results.append({
                "prompt_len": l,
                "max_tokens": tc,
                "baseline_ids": bl["ids"],
                "fused_ids": ids,
                "per_token_match": matches,
                "first_diverge_pos": first_diverge,
                "all_match": all_match,
            })
    del llm_fused
    gc.collect()

    manifest = {
        "probe": "token_depth",
        "model": MODEL,
        "test_lengths": test_lengths,
        "token_counts": token_counts,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": results,
    }
    path = os.path.join(output_dir, "token_depth.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Token depth saved to {path}", file=sys.stderr)
    return manifest


def probe_min_seq_len_sweep(output_dir):
    """Probe 4: Sweep VLLM_FUSED_INT4_MIN_SEQ_LEN to see if shadow window fixes it.

    This must run as subprocesses to re-initialize the environment variable.
    """
    print("=== PROBE 4: min_seq_len sweep ===", file=sys.stderr)
    thresholds = [32, 48, 64, 96, 128]
    test_lengths = [32, 64, 96, 128]

    # Create a helper script inline
    helper = os.path.join(output_dir, "_msl_helper.py")
    with open(helper, "w") as f:
        f.write('''#!/usr/bin/env python3
import json, sys, os, gc
from transformers import AutoTokenizer
MODEL = "Qwen/Qwen2.5-7B-Instruct"
FILLER = ("The quick brown fox jumps over the lazy dog. " * 40)
def main():
    threshold = int(sys.argv[1])
    test_lengths = json.loads(sys.argv[2])
    output_path = sys.argv[3]
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    from vllm import LLM, SamplingParams
    prompts = {}
    for l in test_lengths:
        tokens = tokenizer.encode(FILLER)[:l]
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        actual = len(tokenizer.encode(text))
        prompts[l] = {"text": text, "actual_len": actual}
    sp = SamplingParams(max_tokens=3, temperature=0.0)
    # Baseline
    llm = LLM(model=MODEL, dtype="float16", kv_cache_dtype="auto",
              max_model_len=2048, gpu_memory_utilization=0.8,
              disable_log_stats=True, enforce_eager=True)
    baseline = {}
    for l in test_lengths:
        out = llm.generate([prompts[l]["text"]], sp)[0].outputs[0]
        baseline[l] = list(out.token_ids)
    del llm; gc.collect()
    # Fused with threshold
    os.environ["VLLM_FUSED_INT4_MIN_SEQ_LEN"] = str(threshold)
    llm = LLM(model=MODEL, dtype="float16", kv_cache_dtype="int4_fused",
              max_model_len=2048, gpu_memory_utilization=0.8,
              disable_log_stats=True, enforce_eager=True)
    results = []
    for l in test_lengths:
        out = llm.generate([prompts[l]["text"]], sp)[0].outputs[0]
        fused_ids = list(out.token_ids)
        match = baseline[l] == fused_ids
        results.append({"prompt_len": prompts[l]["actual_len"],
                        "baseline_ids": baseline[l],
                        "fused_ids": fused_ids, "match": match})
        status = "OK" if match else "FAIL"
        print(f"  msl={threshold} len={l} [{status}]", file=sys.stderr)
    del llm; gc.collect()
    with open(output_path, "w") as f:
        json.dump({"threshold": threshold, "results": results}, f, indent=2)
if __name__ == "__main__":
    main()
''')

    all_results = []
    for threshold in thresholds:
        out_path = os.path.join(output_dir, f"msl_{threshold}.json")
        env = os.environ.copy()
        env["VLLM_FUSED_INT4_MIN_SEQ_LEN"] = str(threshold)
        lengths_json = json.dumps(test_lengths)
        cmd = [sys.executable, helper, str(threshold), lengths_json, out_path]
        print(f"  Running min_seq_len={threshold}...", file=sys.stderr)
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                              timeout=600)
        if proc.returncode != 0:
            print(f"  FAILED: {proc.stderr[-500:]}", file=sys.stderr)
            all_results.append({"threshold": threshold, "error": proc.stderr[-500:]})
        else:
            with open(out_path) as f:
                data = json.load(f)
            all_results.append(data)
            for r in data["results"]:
                status = "OK" if r["match"] else "FAIL"
                print(f"    msl={threshold} len={r['prompt_len']} [{status}]",
                      file=sys.stderr)

    manifest = {
        "probe": "min_seq_len_sweep",
        "model": MODEL,
        "thresholds": thresholds,
        "test_lengths": test_lengths,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": all_results,
    }
    path = os.path.join(output_dir, "min_seq_len_sweep.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"min_seq_len sweep saved to {path}", file=sys.stderr)
    return manifest


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/tmp/fused_int4_theory_probe")
    parser.add_argument("--probe", choices=["cliff", "extended", "depth",
                                            "msl", "all"],
                        default="all")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    summaries = {}

    if args.probe in ("cliff", "all"):
        summaries["cliff"] = probe_cliff_finder(args.output_dir)

    if args.probe in ("extended", "all"):
        summaries["extended"] = probe_extended_cliff(args.output_dir)

    if args.probe in ("depth", "all"):
        summaries["depth"] = probe_token_depth(args.output_dir)

    if args.probe in ("msl", "all"):
        summaries["msl"] = probe_min_seq_len_sweep(args.output_dir)

    # Final summary
    summary_path = os.path.join(args.output_dir, "theory_probe_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "summaries": {k: v.get("summary", v) for k, v in summaries.items()},
        }, f, indent=2)
    print(f"\n=== ALL PROBES COMPLETE ===", file=sys.stderr)
    print(f"Results in: {args.output_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
