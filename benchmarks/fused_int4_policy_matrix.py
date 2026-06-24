#!/usr/bin/env python3
"""Direct policy matrix test for fused INT4 quantization.

Tests different quantization policies by running each as a separate process
with environment variables to control the fused INT4 backend.

Each config runs in its own vLLM subprocess to ensure clean env propagation.
"""

import json
import os
import subprocess
import sys
import time


MODEL = "Qwen/Qwen2.5-7B-Instruct"
GPU_MEM = 0.70
MAX_MODEL_LEN = 512
MAX_TOKENS = 20

PROMPTS = [
    ("short_8", "What is 2+2?"),
    ("short_16", "Explain the concept of gravity in one sentence."),
    ("med_32", "Write a haiku about the ocean and then explain what makes it a haiku."),
    ("med_48", "List the first 10 prime numbers and explain why 1 is not considered prime. Be concise."),
    ("long_64", "Explain the difference between TCP and UDP protocols. Include three key differences and a use case for each."),
    ("long_96", "Write a short story about a robot that learns to paint. Include dialogue between the robot and its creator. Make it exactly 3 paragraphs."),
    ("long_128", "Explain the history of the Internet from ARPANET to modern day. Cover 5 major milestones with dates and significance. Be detailed."),
]

CONFIGS = [
    # (label, asymmetric, group_size, min_seq_len)
    ("baseline_fp16",    "0", "32", "999999"),  # FP16 decode (reference)
    ("sym_g32_msl8",     "0", "32", "8"),       # Current default fused INT4
    ("asym_g32_msl8",    "1", "32", "8"),       # Asymmetric INT4
    ("sym_g16_msl8",     "0", "16", "8"),       # Smaller groups
    ("asym_g16_msl8",    "1", "16", "8"),       # Asymmetric + smaller groups
    ("sym_g32_msl32",    "0", "32", "32"),      # Higher threshold
    ("asym_g32_msl32",   "1", "32", "32"),      # Asymmetric + higher threshold
    ("sym_g32_msl64",    "0", "32", "64"),      # Even higher threshold
    ("asym_g32_msl64",   "1", "32", "64"),      # Asymmetric + even higher
]


def run_config(label, asym, gs, msl):
    """Run a single config in a subprocess."""
    script = f'''
import json, os, sys
os.environ["VLLM_FUSED_INT4_ASYMMETRIC"] = "{asym}"
os.environ["VLLM_FUSED_INT4_GROUP_SIZE"] = "{gs}"
os.environ["VLLM_FUSED_INT4_MIN_SEQ_LEN"] = "{msl}"

from vllm import LLM, SamplingParams

llm = LLM(
    model="{MODEL}",
    kv_cache_dtype="int4_fused",
    max_model_len={MAX_MODEL_LEN},
    gpu_memory_utilization={GPU_MEM},
    enforce_eager=True,
)

prompts = {json.dumps(PROMPTS)}
params = SamplingParams(max_tokens={MAX_TOKENS}, temperature=0.0)

results = {{}}
for name, prompt in prompts:
    out = llm.generate([prompt], params)
    text = out[0].outputs[0].text.strip()
    ids = list(out[0].outputs[0].token_ids)
    results[name] = {{"text": text, "token_ids": ids}}

print("RESULT_JSON:" + json.dumps(results))
'''
    env = os.environ.copy()
    env["VLLM_FUSED_INT4_ASYMMETRIC"] = asym
    env["VLLM_FUSED_INT4_GROUP_SIZE"] = gs
    env["VLLM_FUSED_INT4_MIN_SEQ_LEN"] = msl

    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=300, env=env,
        )
        elapsed = time.time() - t0

        for line in result.stdout.split("\n"):
            if line.startswith("RESULT_JSON:"):
                return {
                    "status": "ok",
                    "results": json.loads(line[len("RESULT_JSON:"):]),
                    "elapsed_s": round(elapsed, 1),
                }

        return {
            "status": "no_output",
            "elapsed_s": round(elapsed, 1),
            "stderr_tail": result.stderr[-1500:] if result.stderr else "",
            "stdout_tail": result.stdout[-500:] if result.stdout else "",
        }
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def compare(baseline_results, test_results):
    """Compare test against baseline."""
    total = 0
    match = 0
    token_rates = []
    details = {}

    for name in baseline_results:
        if name not in test_results:
            details[name] = {"status": "missing"}
            continue
        b = baseline_results[name]
        t = test_results[name]
        text_match = b["text"] == t["text"]
        b_ids = b["token_ids"]
        t_ids = t["token_ids"]
        min_len = min(len(b_ids), len(t_ids))
        if min_len > 0:
            tok_match = sum(1 for i in range(min_len) if b_ids[i] == t_ids[i])
            tok_rate = tok_match / min_len
        else:
            tok_rate = 0.0
        first_div = next((i for i in range(min_len) if b_ids[i] != t_ids[i]), min_len)

        details[name] = {
            "text_match": text_match,
            "token_match_rate": round(tok_rate, 4),
            "first_diverge": first_div,
            "baseline": b["text"][:80],
            "test": t["text"][:80],
        }
        total += 1
        if text_match:
            match += 1
        token_rates.append(tok_rate)

    return {
        "total": total,
        "exact_matches": match,
        "exact_rate": round(match / max(total, 1), 4),
        "avg_token_rate": round(sum(token_rates) / max(len(token_rates), 1), 4),
        "details": details,
    }


def main():
    output_path = sys.argv[1] if len(sys.argv) > 1 else \
        f"policy_matrix_{time.strftime('%Y%m%d_%H%M%S')}.json"

    all_results = {
        "model": MODEL,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "configs": {},
    }

    # Run baseline first
    label, asym, gs, msl = CONFIGS[0]
    print(f"\n[{label}] Running baseline...")
    baseline = run_config(label, asym, gs, msl)
    all_results["configs"][label] = {
        "asymmetric": asym, "group_size": gs, "min_seq_len": msl,
        "run": baseline,
    }

    if baseline["status"] != "ok":
        print(f"  BASELINE FAILED: {baseline}")
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results: {output_path}")
        return

    print(f"  Baseline OK ({baseline['elapsed_s']}s)")
    baseline_results = baseline["results"]

    # Run all test configs
    for label, asym, gs, msl in CONFIGS[1:]:
        asym_str = "asym" if asym == "1" else "sym"
        print(f"\n[{label}] {asym_str} g{gs} msl{msl}...")
        result = run_config(label, asym, gs, msl)

        entry = {
            "asymmetric": asym, "group_size": gs, "min_seq_len": msl,
            "run": result,
        }

        if result["status"] == "ok":
            comp = compare(baseline_results, result["results"])
            entry["comparison"] = comp
            print(f"  OK ({result['elapsed_s']}s) - "
                  f"Text: {comp['exact_matches']}/{comp['total']} "
                  f"Token: {comp['avg_token_rate']}")
        else:
            print(f"  FAILED: {result['status']}")
            if "stderr_tail" in result:
                # Show last relevant error line
                for line in result["stderr_tail"].split("\n"):
                    if "Error" in line or "error" in line:
                        print(f"    {line.strip()[:120]}")
                        break

        all_results["configs"][label] = entry

    # Summary table
    print(f"\n{'='*70}")
    print(f"{'Config':<20} {'Asym':<6} {'GS':<4} {'MSL':<6} "
          f"{'Text':<8} {'TokenRate':<10}")
    print("-" * 70)
    for label, asym, gs, msl in CONFIGS:
        entry = all_results["configs"].get(label, {})
        comp = entry.get("comparison", {})
        run = entry.get("run", {})
        if label == CONFIGS[0][0]:
            print(f"{label:<20} {'n/a':<6} {gs:<4} {msl:<6} "
                  f"{'ref':<8} {'1.0000':<10}  [baseline]")
        elif run.get("status") == "ok":
            print(f"{label:<20} {asym:<6} {gs:<4} {msl:<6} "
                  f"{comp.get('exact_matches','?')}/{comp.get('total','?'):<4} "
                  f"{comp.get('avg_token_rate','?'):<10}")
        else:
            print(f"{label:<20} {asym:<6} {gs:<4} {msl:<6} "
                  f"{'FAIL':<8} {'---':<10}  [{run.get('status','')}]")

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {output_path}")


if __name__ == "__main__":
    main()
