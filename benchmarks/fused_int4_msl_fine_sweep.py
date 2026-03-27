#!/usr/bin/env python3
"""Fine MSL sweep: finds exact threshold where INT4 decode diverges from FP16."""

import json
import os
import subprocess
import sys
import time

MODEL = "Qwen/Qwen2.5-7B-Instruct"
PROMPTS = [
    ("short_8", "What is 2+2?"),
    ("short_16", "Explain the concept of gravity in one sentence."),
    ("med_32", "Write a haiku about the ocean and then explain what makes it a haiku."),
    ("med_48", "List the first 10 prime numbers and explain why 1 is not considered prime. Be concise."),
    ("long_64", "Explain the difference between TCP and UDP protocols. Include three key differences and a use case for each."),
    ("long_96", "Write a short story about a robot that learns to paint. Include dialogue between the robot and its creator. Make it exactly 3 paragraphs."),
    ("long_128", "Explain the history of the Internet from ARPANET to modern day. Cover 5 major milestones with dates and significance. Be detailed."),
]


INNER_SCRIPT_TEMPLATE = '''
import json, os
os.environ["VLLM_FUSED_INT4_MIN_SEQ_LEN"] = "{msl}"
from vllm import LLM, SamplingParams
llm = LLM(
    model="{model}",
    kv_cache_dtype="int4_fused",
    max_model_len=512,
    gpu_memory_utilization=0.70,
    enforce_eager=True,
)
prompts = json.loads('{prompts_json}')
params = SamplingParams(max_tokens=20, temperature=0.0)
results = {{}}
for name, prompt in prompts:
    out = llm.generate([prompt], params)
    results[name] = {{
        "text": out[0].outputs[0].text.strip(),
        "ids": list(out[0].outputs[0].token_ids),
    }}
print("RESULT:" + json.dumps(results))
'''


def run_msl(msl):
    prompts_json = json.dumps(PROMPTS).replace("'", "\\'")
    script = INNER_SCRIPT_TEMPLATE.format(
        msl=msl, model=MODEL, prompts_json=prompts_json,
    )
    env = os.environ.copy()
    env["VLLM_FUSED_INT4_MIN_SEQ_LEN"] = str(msl)

    r = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=300, env=env,
    )
    for line in r.stdout.split("\n"):
        if line.startswith("RESULT:"):
            return json.loads(line[7:])
    # Debug on failure
    print(f"  [stderr tail] {r.stderr[-300:]}")
    return None


def main():
    output_path = sys.argv[1] if len(sys.argv) > 1 else \
        f"msl_fine_sweep_{time.strftime('%Y%m%d_%H%M%S')}.json"

    print("Running FP16 baseline (MSL=999999)...")
    baseline = run_msl(999999)
    if not baseline:
        print("BASELINE FAILED")
        return

    print(f"Baseline: {len(baseline)} prompts OK")

    all_results = {
        "model": MODEL,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "baseline": baseline,
        "sweeps": {},
    }

    for msl in [16, 24, 32, 40, 48, 56, 64, 80, 96]:
        print(f"\nMSL={msl}:")
        result = run_msl(msl)
        if not result:
            print("  FAILED")
            all_results["sweeps"][str(msl)] = {"status": "failed"}
            continue

        matches = 0
        total = 0
        details = {}
        for name in baseline:
            if name not in result:
                continue
            total += 1
            b_text = baseline[name]["text"]
            t_text = result[name]["text"]
            match = b_text == t_text

            b_ids = baseline[name]["ids"]
            t_ids = result[name]["ids"]
            min_len = min(len(b_ids), len(t_ids))
            tok_match = sum(1 for i in range(min_len) if b_ids[i] == t_ids[i]) if min_len else 0
            tok_rate = tok_match / min_len if min_len else 0

            status = "PASS" if match else "FAIL"
            print(f"  {name}: {status} (tok_rate={tok_rate:.2f})")
            if not match:
                print(f"    base: {b_text[:60]}")
                print(f"    test: {t_text[:60]}")

            if match:
                matches += 1
            details[name] = {
                "match": match,
                "tok_rate": round(tok_rate, 4),
            }

        print(f"  Score: {matches}/{total}")
        all_results["sweeps"][str(msl)] = {
            "matches": matches,
            "total": total,
            "details": details,
        }

    # Summary
    print(f"\n{'='*50}")
    print(f"{'MSL':<6} {'Match':<8} {'Score':<8}")
    print("-" * 30)
    for msl_str, data in sorted(all_results["sweeps"].items(), key=lambda x: int(x[0])):
        if data.get("status") == "failed":
            print(f"{msl_str:<6} FAIL")
        else:
            print(f"{msl_str:<6} {data['matches']}/{data['total']:<4}")

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults: {output_path}")


if __name__ == "__main__":
    main()
