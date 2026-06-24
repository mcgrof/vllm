#!/usr/bin/env python3
"""Asymmetric + group_size sweep at critical MSL thresholds."""

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

INNER_SCRIPT = '''
import json, os
os.environ["VLLM_FUSED_INT4_MIN_SEQ_LEN"] = "{msl}"
os.environ["VLLM_FUSED_INT4_ASYMMETRIC"] = "{asym}"
os.environ["VLLM_FUSED_INT4_GROUP_SIZE"] = "{gs}"
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


def run_config(msl, asym, gs):
    prompts_json = json.dumps(PROMPTS).replace("'", "\\'")
    script = INNER_SCRIPT.format(
        msl=msl, asym=asym, gs=gs, model=MODEL, prompts_json=prompts_json,
    )
    env = os.environ.copy()
    env["VLLM_FUSED_INT4_MIN_SEQ_LEN"] = str(msl)
    env["VLLM_FUSED_INT4_ASYMMETRIC"] = str(asym)
    env["VLLM_FUSED_INT4_GROUP_SIZE"] = str(gs)

    r = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=300, env=env,
    )
    for line in r.stdout.split("\n"):
        if line.startswith("RESULT:"):
            return json.loads(line[7:])
    print(f"  [err] {r.stderr[-200:]}")
    return None


def compare(baseline, result):
    matches = 0
    total = 0
    for name in baseline:
        if name not in result:
            continue
        total += 1
        if baseline[name]["text"] == result[name]["text"]:
            matches += 1
    return matches, total


def main():
    output_path = sys.argv[1] if len(sys.argv) > 1 else \
        f"asym_msl_sweep_{time.strftime('%Y%m%d_%H%M%S')}.json"

    print("Baseline (MSL=999999)...")
    baseline = run_config(999999, 0, 32)
    if not baseline:
        print("BASELINE FAILED")
        return

    # Test matrix at critical thresholds
    configs = [
        # (msl, asym, gs, label)
        (32, 0, 32, "sym_g32_msl32"),
        (32, 1, 32, "asym_g32_msl32"),
        (32, 0, 16, "sym_g16_msl32"),
        (32, 1, 16, "asym_g16_msl32"),
        (40, 0, 32, "sym_g32_msl40"),
        (40, 1, 32, "asym_g32_msl40"),
        (40, 0, 16, "sym_g16_msl40"),
        (40, 1, 16, "asym_g16_msl40"),
        (48, 0, 32, "sym_g32_msl48"),
        (48, 1, 32, "asym_g32_msl48"),
        (48, 0, 16, "sym_g16_msl48"),
        (48, 1, 16, "asym_g16_msl48"),
    ]

    results = {"baseline": baseline, "configs": {}}

    for msl, asym, gs, label in configs:
        a_str = "asym" if asym else "sym"
        print(f"\n[{label}] {a_str} g{gs} msl{msl}...")
        result = run_config(msl, asym, gs)
        if result:
            m, t = compare(baseline, result)
            print(f"  Score: {m}/{t}")
            results["configs"][label] = {
                "matches": m, "total": t,
                "msl": msl, "asymmetric": bool(asym), "group_size": gs,
            }
        else:
            print("  FAILED")
            results["configs"][label] = {"status": "failed"}

    # Summary
    print(f"\n{'='*60}")
    print(f"{'Config':<22} {'MSL':<5} {'Asym':<6} {'GS':<4} {'Score':<8}")
    print("-" * 50)
    for label in [c[3] for c in configs]:
        d = results["configs"].get(label, {})
        if d.get("status") == "failed":
            print(f"{label:<22} {'FAIL':>30}")
        else:
            print(f"{label:<22} {d.get('msl',''):<5} "
                  f"{'Y' if d.get('asymmetric') else 'N':<6} "
                  f"{d.get('group_size',''):<4} "
                  f"{d.get('matches','?')}/{d.get('total','?')}")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {output_path}")


if __name__ == "__main__":
    main()
