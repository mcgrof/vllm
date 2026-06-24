#!/usr/bin/env python3
"""Broad correctness probe: fused INT4 vs baseline across many prompt lengths.

Tests prompt lengths spanning below, at, and above the MIN_FUSED_SEQ_LEN
threshold, with 3 decode tokens to catch compounding error.
"""

import json
import sys
import time

PROMPT_LENGTHS = [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 32, 64]
MAX_TOKENS = 3
MODEL = "Qwen/Qwen2.5-7B-Instruct"
FILLER = ("The quick brown fox jumps over the lazy dog. " * 20)


def make_prompts():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompts = []
    for tl in PROMPT_LENGTHS:
        tokens = tokenizer.encode(FILLER)[:tl]
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        actual = tokenizer.encode(text)
        prompts.append({
            "label": "len_%d" % tl,
            "target_len": tl,
            "actual_len": len(actual),
            "text": text,
        })
    return prompts


def run_mode(prompts, kv_dtype, label):
    from vllm import LLM, SamplingParams

    print("[%s] Init..." % label, file=sys.stderr)
    llm = LLM(
        model=MODEL,
        dtype="float16",
        kv_cache_dtype=kv_dtype,
        max_model_len=2048,
        gpu_memory_utilization=0.8,
        disable_log_stats=True,
        enforce_eager=True,
    )
    sp = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0)
    results = []
    for p in prompts:
        out = llm.generate([p["text"]], sp)[0].outputs[0]
        ids = list(out.token_ids)
        alen = p["actual_len"]
        txt = out.text
        print("  [%s] prompt_len=%3d -> ids=%s  text=%r" % (
            label, alen, ids, txt), file=sys.stderr)
        results.append({
            "label": p["label"],
            "prompt_len": p["actual_len"],
            "ids": ids,
            "text": out.text,
        })
    del llm
    return results


def main():
    prompts = make_prompts()

    baseline = run_mode(prompts, "auto", "baseline")
    fused = run_mode(prompts, "int4_fused", "fused")

    comparisons = []
    for b, f in zip(baseline, fused):
        match = b["ids"] == f["ids"]
        comparisons.append({
            "label": b["label"],
            "prompt_len": b["prompt_len"],
            "baseline_ids": b["ids"],
            "fused_ids": f["ids"],
            "baseline_text": b["text"],
            "fused_text": f["text"],
            "match": match,
        })
        status = "OK" if match else "FAIL"
        print("  %s: baseline=%s fused=%s [%s]" % (
            b["label"], b["ids"], f["ids"], status), file=sys.stderr)

    total = len(comparisons)
    matches = sum(1 for c in comparisons if c["match"])
    print("\n=== SUMMARY: %d/%d match ===" % (matches, total), file=sys.stderr)

    manifest = {
        "probe": "broad_correctness",
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "prompt_lengths": PROMPT_LENGTHS,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "comparisons": comparisons,
        "summary": {
            "total": total,
            "matches": matches,
            "all_pass": matches == total,
        },
    }
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
