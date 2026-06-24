#!/usr/bin/env python3
"""Logit-level probe comparing baseline vs fused INT4 for the specific
prompt_len=2 case. Captures top-10 logits at each decode step to see
how close the fused output is to the baseline.

Usage (on H100 GPU node):
    python benchmarks/fused_int4_logit_probe.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --output /tmp/logit_probe.json
"""

import argparse
import json
import os
import sys
import time
import torch
import math

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def run_with_logprobs(model_name, kv_cache_dtype, prompts, max_tokens):
    """Run vLLM and collect top logprobs at each token position."""
    from vllm import LLM, SamplingParams

    mode = "fused" if kv_cache_dtype == "int4_fused" else "baseline"
    print(f"[{mode}] Init...", file=sys.stderr)

    llm = LLM(
        model=model_name,
        dtype="float16",
        kv_cache_dtype=kv_cache_dtype,
        max_model_len=2048,
        gpu_memory_utilization=0.8,
        disable_log_stats=True,
        enforce_eager=True,
    )

    sp = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
        logprobs=20,  # top-20 logprobs
    )

    results = []
    for p in prompts:
        outputs = llm.generate([p["prompt"]], sp)
        out = outputs[0].outputs[0]
        token_ids = list(out.token_ids)

        # Extract per-token logprob info
        per_token = []
        if out.logprobs:
            for step_lp in out.logprobs:
                top = []
                for tid, lp_obj in sorted(step_lp.items(),
                                           key=lambda x: -x[1].logprob):
                    top.append({
                        "token_id": tid,
                        "logprob": round(lp_obj.logprob, 6),
                        "decoded": lp_obj.decoded_token,
                    })
                per_token.append(top[:20])

        results.append({
            "label": p["label"],
            "prompt": p["prompt"][:200],
            "token_ids": token_ids,
            "text": out.text,
            "logprobs_per_step": per_token,
        })

        print(f"  [{mode}] {p['label']}: ids={token_ids} text={out.text!r}",
              file=sys.stderr)

    del llm
    return {"mode": mode, "kv_cache_dtype": kv_cache_dtype, "results": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-tokens", type=int, default=4)
    args = parser.parse_args()

    prompts = [
        {"label": "len_1", "prompt": "The"},
        {"label": "len_2", "prompt": "The quick"},
        {"label": "len_3", "prompt": "The quick brown"},
        {"label": "len_4", "prompt": "The quick brown fox"},
        {"label": "len_5", "prompt": "The quick brown fox jumps"},
    ]

    baseline = run_with_logprobs(args.model, "auto", prompts, args.max_tokens)
    fused = run_with_logprobs(args.model, "int4_fused", prompts, args.max_tokens)

    # Analyze logprob divergence
    comparisons = []
    for b_res, f_res in zip(baseline["results"], fused["results"]):
        steps = []
        for step_i, (b_lp, f_lp) in enumerate(zip(
            b_res.get("logprobs_per_step", []),
            f_res.get("logprobs_per_step", []),
        )):
            b_top_id = b_lp[0]["token_id"] if b_lp else None
            f_top_id = f_lp[0]["token_id"] if f_lp else None
            b_top_lp = b_lp[0]["logprob"] if b_lp else None
            f_top_lp = f_lp[0]["logprob"] if f_lp else None

            # Find baseline top-1 token in fused logprobs
            b_in_fused = None
            for f_entry in f_lp:
                if f_entry["token_id"] == b_top_id:
                    b_in_fused = f_entry["logprob"]
                    break

            # Find fused top-1 token in baseline logprobs
            f_in_baseline = None
            for b_entry in b_lp:
                if b_entry["token_id"] == f_top_id:
                    f_in_baseline = b_entry["logprob"]
                    break

            steps.append({
                "step": step_i,
                "match": b_top_id == f_top_id,
                "baseline_top1": {"id": b_top_id, "logprob": b_top_lp,
                                  "text": b_lp[0]["decoded"] if b_lp else ""},
                "fused_top1": {"id": f_top_id, "logprob": f_top_lp,
                               "text": f_lp[0]["decoded"] if f_lp else ""},
                "baseline_top1_in_fused_logprob": b_in_fused,
                "fused_top1_in_baseline_logprob": f_in_baseline,
                "baseline_top5": b_lp[:5] if b_lp else [],
                "fused_top5": f_lp[:5] if f_lp else [],
            })

        comparisons.append({
            "label": b_res["label"],
            "steps": steps,
        })

    manifest = {
        "probe": "logit_probe",
        "model": args.model,
        "max_tokens": args.max_tokens,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "baseline": baseline,
        "fused": fused,
        "comparisons": comparisons,
    }

    json_str = json.dumps(manifest, indent=2, ensure_ascii=False)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)),
                    exist_ok=True)
        with open(args.output, "w") as f:
            f.write(json_str + "\n")
        print(f"\nOutput: {args.output}", file=sys.stderr)
    else:
        print(json_str)

    # Quick summary
    print("\n=== Logit Probe Summary ===", file=sys.stderr)
    for comp in comparisons:
        print(f"  {comp['label']}:", file=sys.stderr)
        for s in comp["steps"]:
            match = "MATCH" if s["match"] else "DIFF"
            bl = s["baseline_top1"]
            fl = s["fused_top1"]
            print(f"    step {s['step']}: {match} "
                  f"baseline={bl['id']}({bl['text']!r} lp={bl['logprob']}) "
                  f"fused={fl['id']}({fl['text']!r} lp={fl['logprob']}) "
                  f"bl_in_fused={s['baseline_top1_in_fused_logprob']} "
                  f"fu_in_bl={s['fused_top1_in_baseline_logprob']}",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
