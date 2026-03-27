#!/usr/bin/env python3
"""Fused INT4 quantization policy probe.

Ratio-classifier proxy: measures per-layer INT4 quantization error for
different policies (symmetric vs asymmetric, GROUP_SIZE 16 vs 32) and
derives which layers are precision-sensitive.

Runs the target model through vLLM with different quantization configs
and compares INT4 decode output against FP16 baseline.

Usage:
    # Full matrix: sym/asym x g16/g32 x MSL sweep
    python benchmarks/fused_int4_quant_policy_probe.py

    # Single config
    VLLM_FUSED_INT4_ASYMMETRIC=1 VLLM_FUSED_INT4_GROUP_SIZE=16 \
    VLLM_FUSED_INT4_MIN_SEQ_LEN=8 \
    python benchmarks/fused_int4_quant_policy_probe.py --single

Environment:
    VLLM_FUSED_INT4_ASYMMETRIC: 0 or 1 (default 0)
    VLLM_FUSED_INT4_GROUP_SIZE: 16 or 32 (default 32)
    VLLM_FUSED_INT4_MIN_SEQ_LEN: threshold (default 999999)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


MODEL = "Qwen/Qwen2.5-7B-Instruct"
PROMPTS = [
    # Short (< 32 tokens)
    ("short_8", "What is 2+2?"),
    ("short_16", "Explain the concept of gravity in one sentence."),
    # Medium (32-64 tokens)
    ("med_32", "Write a haiku about the ocean and then explain what makes it a haiku."),
    ("med_48", "List the first 10 prime numbers and explain why 1 is not considered prime. Be concise."),
    # Long (64+ tokens)
    ("long_64", "Explain the difference between TCP and UDP protocols in networking. "
     "Include at least three key differences and provide a use case for each protocol. Be thorough."),
    ("long_96", "Write a short story about a robot that learns to paint. "
     "The story should have a beginning, middle, and end. "
     "Include dialogue between the robot and its creator. "
     "Make it exactly 3 paragraphs long."),
    ("long_128", "Explain the history of the Internet from ARPANET to modern day. "
     "Cover at least 5 major milestones. For each milestone, explain what it was, "
     "when it happened, and why it mattered. Include the roles of key organizations "
     "like DARPA, CERN, and major tech companies. Be detailed and thorough."),
]


def run_single_config(
    asymmetric: bool,
    group_size: int,
    min_seq_len: int,
    prompts: list[tuple[str, str]],
    max_tokens: int = 20,
) -> dict:
    """Run model with given config and capture outputs."""
    env = os.environ.copy()
    env["VLLM_FUSED_INT4_ASYMMETRIC"] = "1" if asymmetric else "0"
    env["VLLM_FUSED_INT4_GROUP_SIZE"] = str(group_size)
    env["VLLM_FUSED_INT4_MIN_SEQ_LEN"] = str(min_seq_len)

    script = f"""
import json, os, sys
os.environ["VLLM_FUSED_INT4_ASYMMETRIC"] = "{int(asymmetric)}"
os.environ["VLLM_FUSED_INT4_GROUP_SIZE"] = "{group_size}"
os.environ["VLLM_FUSED_INT4_MIN_SEQ_LEN"] = "{min_seq_len}"

from vllm import LLM, SamplingParams

llm = LLM(
    model="{MODEL}",
    kv_cache_dtype="int4_fused",
    max_model_len=512,
    gpu_memory_utilization=0.85,
    enforce_eager=True,
)

prompts = {json.dumps([(n, p) for n, p in prompts])}
params = SamplingParams(max_tokens={max_tokens}, temperature=0.0)

results = {{}}
for name, prompt in prompts:
    outputs = llm.generate([prompt], params)
    text = outputs[0].outputs[0].text.strip()
    token_ids = list(outputs[0].outputs[0].token_ids)
    results[name] = {{"text": text, "token_ids": token_ids}}

print("PROBE_RESULT:" + json.dumps(results))
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )
        stdout = result.stdout
        stderr = result.stderr

        # Extract result
        for line in stdout.split("\n"):
            if line.startswith("PROBE_RESULT:"):
                return {
                    "status": "ok",
                    "results": json.loads(line[len("PROBE_RESULT:"):]),
                    "config": {
                        "asymmetric": asymmetric,
                        "group_size": group_size,
                        "min_seq_len": min_seq_len,
                    },
                }

        return {
            "status": "error",
            "error": f"No PROBE_RESULT in output",
            "stdout_tail": stdout[-2000:] if stdout else "",
            "stderr_tail": stderr[-2000:] if stderr else "",
            "config": {
                "asymmetric": asymmetric,
                "group_size": group_size,
                "min_seq_len": min_seq_len,
            },
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "config": {
                "asymmetric": asymmetric,
                "group_size": group_size,
                "min_seq_len": min_seq_len,
            },
        }
    except Exception as e:
        return {
            "status": "exception",
            "error": str(e),
            "config": {
                "asymmetric": asymmetric,
                "group_size": group_size,
                "min_seq_len": min_seq_len,
            },
        }


def run_baseline(prompts: list[tuple[str, str]], max_tokens: int = 20) -> dict:
    """Run FP16 baseline (MSL=999999, always uses FP16 SDPA decode)."""
    return run_single_config(
        asymmetric=False, group_size=32,
        min_seq_len=999999, prompts=prompts, max_tokens=max_tokens,
    )


def compare_results(baseline: dict, test: dict) -> dict:
    """Compare test outputs against baseline."""
    if baseline["status"] != "ok" or test["status"] != "ok":
        return {
            "comparison": "cannot_compare",
            "baseline_status": baseline["status"],
            "test_status": test["status"],
        }

    comparison = {}
    b_results = baseline["results"]
    t_results = test["results"]

    total = 0
    matching = 0
    token_match_rates = []

    for name in b_results:
        if name not in t_results:
            comparison[name] = {"status": "missing_in_test"}
            continue

        b_text = b_results[name]["text"]
        t_text = t_results[name]["text"]
        b_ids = b_results[name]["token_ids"]
        t_ids = t_results[name]["token_ids"]

        text_match = b_text == t_text
        # Token-level comparison
        min_len = min(len(b_ids), len(t_ids))
        if min_len > 0:
            token_matches = sum(1 for i in range(min_len) if b_ids[i] == t_ids[i])
            token_rate = token_matches / min_len
        else:
            token_rate = 0.0

        comparison[name] = {
            "text_match": text_match,
            "token_match_rate": round(token_rate, 4),
            "first_diverge_pos": next(
                (i for i in range(min_len) if b_ids[i] != t_ids[i]), min_len
            ),
            "baseline_text": b_text[:100],
            "test_text": t_text[:100],
        }

        total += 1
        if text_match:
            matching += 1
        token_match_rates.append(token_rate)

    return {
        "comparison": comparison,
        "summary": {
            "total_prompts": total,
            "exact_text_matches": matching,
            "exact_match_rate": round(matching / max(total, 1), 4),
            "avg_token_match_rate": round(
                sum(token_match_rates) / max(len(token_match_rates), 1), 4
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", action="store_true",
                        help="Run single config from env vars")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path")
    parser.add_argument("--max-tokens", type=int, default=20)
    args = parser.parse_args()

    prompts = PROMPTS

    if args.single:
        asym = os.environ.get("VLLM_FUSED_INT4_ASYMMETRIC", "0") == "1"
        gs = int(os.environ.get("VLLM_FUSED_INT4_GROUP_SIZE", "32"))
        msl = int(os.environ.get("VLLM_FUSED_INT4_MIN_SEQ_LEN", "999999"))
        print(f"Running single config: asymmetric={asym}, "
              f"group_size={gs}, min_seq_len={msl}")
        result = run_single_config(asym, gs, msl, prompts, args.max_tokens)
        print(json.dumps(result, indent=2))
        return

    all_results = {
        "model": MODEL,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "configs": [],
    }

    # Step 1: Baseline (FP16 decode)
    print("=" * 60)
    print("Running FP16 baseline (MSL=999999)...")
    print("=" * 60)
    baseline = run_baseline(prompts, args.max_tokens)
    all_results["baseline"] = baseline
    if baseline["status"] == "ok":
        print(f"Baseline: {len(baseline['results'])} prompts completed")
    else:
        print(f"Baseline FAILED: {baseline.get('error', baseline['status'])}")
        if args.output:
            Path(args.output).write_text(json.dumps(all_results, indent=2))
        return

    # Step 2: Test matrix
    configs = [
        # (asymmetric, group_size, min_seq_len, label)
        (False, 32, 8, "sym_g32_msl8"),
        (True, 32, 8, "asym_g32_msl8"),
        (False, 16, 8, "sym_g16_msl8"),
        (True, 16, 8, "asym_g16_msl8"),
        # Also test with higher MSL to see if error is seq-len dependent
        (False, 32, 64, "sym_g32_msl64"),
        (True, 32, 64, "asym_g32_msl64"),
        (True, 16, 64, "asym_g16_msl64"),
    ]

    for asym, gs, msl, label in configs:
        print(f"\n{'='*60}")
        print(f"Config: {label} (asymmetric={asym}, group_size={gs}, msl={msl})")
        print(f"{'='*60}")

        result = run_single_config(asym, gs, msl, prompts, args.max_tokens)
        comp = compare_results(baseline, result)

        entry = {
            "label": label,
            "config": result.get("config", {}),
            "status": result["status"],
            "comparison": comp,
        }
        if result["status"] == "ok":
            entry["results"] = result["results"]
            s = comp.get("summary", {})
            print(f"  Text match: {s.get('exact_text_matches', '?')}/"
                  f"{s.get('total_prompts', '?')} "
                  f"({s.get('exact_match_rate', '?')})")
            print(f"  Avg token match: {s.get('avg_token_match_rate', '?')}")
        else:
            print(f"  FAILED: {result.get('error', result['status'])}")
            if "stderr_tail" in result:
                print(f"  stderr: ...{result['stderr_tail'][-500:]}")

        all_results["configs"].append(entry)

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Config':<20} {'Status':<8} {'Text Match':<12} {'Token Match':<12}")
    print("-" * 52)
    for entry in all_results["configs"]:
        s = entry.get("comparison", {}).get("summary", {})
        status = entry["status"]
        tm = f"{s.get('exact_text_matches', '?')}/{s.get('total_prompts', '?')}"
        tkm = f"{s.get('avg_token_match_rate', '?')}"
        print(f"{entry['label']:<20} {status:<8} {tm:<12} {tkm:<12}")

    # Save results
    output_path = args.output
    if output_path is None:
        output_path = f"quant_policy_probe_{time.strftime('%Y%m%d_%H%M%S')}.json"
    Path(output_path).write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
