#!/usr/bin/env python3
"""K_INT8/V_INT4 re-test after Triton constexpr fix.

Only re-runs the INT8 configs that failed in the main sweep.
"""

import json
import os
import subprocess
import sys
import time

MODELS = [
    ("Qwen/Qwen2-7B-Instruct", "qwen2"),
    ("Qwen/Qwen2.5-7B-Instruct", "qwen25"),
]
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

POLICIES = [
    ("k_int8_v_int4_msl48", 48, "int8", "0", "32"),
    ("k_int8_v_int4_msl8",   8, "int8", "0", "32"),
    ("k_int8_v_int4_msl24", 24, "int8", "0", "32"),
    ("k_int8_v_int4_msl1",   1, "int8", "0", "32"),
]

INNER_SCRIPT_TEMPLATE = '''
import json, os
os.environ["VLLM_FUSED_INT4_MIN_SEQ_LEN"] = "{msl}"
os.environ["VLLM_FUSED_INT4_ASYMMETRIC"] = "{asym}"
os.environ["VLLM_FUSED_INT4_GROUP_SIZE"] = "{gs}"
os.environ["VLLM_FUSED_INT4_K_PRECISION"] = "{k_precision}"
from vllm import LLM, SamplingParams
llm = LLM(
    model="{model}",
    kv_cache_dtype="int4_fused",
    max_model_len={max_model_len},
    gpu_memory_utilization={gpu_mem},
    enforce_eager=True,
)
prompts = json.loads('{prompts_json}')
params = SamplingParams(max_tokens={max_tokens}, temperature=0.0)
results = {{}}
for name, prompt in prompts:
    out = llm.generate([prompt], params)
    results[name] = {{
        "text": out[0].outputs[0].text.strip(),
        "ids": list(out[0].outputs[0].token_ids),
    }}
print("RESULT:" + json.dumps(results))
'''


def run_config(model, msl, k_precision="int4", asym="0", gs="32"):
    prompts_json = json.dumps(PROMPTS).replace("'", "\\'")
    script = INNER_SCRIPT_TEMPLATE.format(
        msl=msl, asym=asym, gs=gs, k_precision=k_precision,
        model=model, prompts_json=prompts_json,
        max_model_len=MAX_MODEL_LEN, gpu_mem=GPU_MEM,
        max_tokens=MAX_TOKENS,
    )
    env = os.environ.copy()
    env["VLLM_FUSED_INT4_MIN_SEQ_LEN"] = str(msl)
    env["VLLM_FUSED_INT4_ASYMMETRIC"] = str(asym)
    env["VLLM_FUSED_INT4_GROUP_SIZE"] = str(gs)
    env["VLLM_FUSED_INT4_K_PRECISION"] = str(k_precision)

    t0 = time.time()
    try:
        r = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=600, env=env,
        )
        elapsed = time.time() - t0
        for line in r.stdout.split("\n"):
            if line.startswith("RESULT:"):
                return json.loads(line[7:]), elapsed
        print(f"  [stderr tail] {r.stderr[-500:]}")
        return None, elapsed
    except subprocess.TimeoutExpired:
        print("  TIMEOUT")
        return None, time.time() - t0


def compare(baseline, result):
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
        tok_match = sum(1 for i in range(min_len)
                        if b_ids[i] == t_ids[i]) if min_len else 0
        tok_rate = tok_match / min_len if min_len else 0
        first_div = next(
            (i for i in range(min_len) if b_ids[i] != t_ids[i]), min_len)

        if match:
            matches += 1
        details[name] = {
            "match": match,
            "tok_rate": round(tok_rate, 4),
            "first_diverge": first_div,
        }

        status = "PASS" if match else "FAIL"
        print(f"    {name}: {status} (tok_rate={tok_rate:.2f} div@{first_div})")
        if not match:
            print(f"      base: {b_text[:60]}")
            print(f"      test: {t_text[:60]}")

    return matches, total, details


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    for model, tag in MODELS:
        print("=" * 70)
        print(f"MODEL: {model}")
        print("=" * 70)

        # Baseline
        print("\n  [baseline] FP16 ...")
        baseline, bl_time = run_config(model, 999999)
        if not baseline:
            print("  BASELINE FAILED")
            continue

        print(f"  Baseline OK ({bl_time:.1f}s)")

        results = {}
        for label, msl, k_prec, asym, gs in POLICIES:
            print(f"\n  [{label}] k={k_prec} msl={msl} ...")
            result, elapsed = run_config(model, msl, k_prec, asym, gs)
            if result:
                m, t, details = compare(baseline, result)
                print(f"    Score: {m}/{t}  ({elapsed:.1f}s)")
                results[label] = {
                    "matches": m, "total": t,
                    "k_precision": k_prec, "msl": msl,
                    "elapsed_s": round(elapsed, 1),
                    "details": details,
                    "raw_outputs": {
                        name: result[name]["text"][:120] for name in result
                    },
                }
            else:
                print("    FAILED")
                results[label] = {"status": "failed", "k_precision": k_prec,
                                   "msl": msl}

        # Summary
        print(f"\n{'='*70}")
        print(f"SUMMARY: {model}")
        print(f"{'Policy':<28} {'K-prec':<7} {'MSL':<5} {'Score':<8}")
        print("-" * 50)
        for label, msl, k_prec, _, _ in POLICIES:
            d = results.get(label, {})
            if d.get("status") == "failed":
                print(f"{label:<28} {k_prec:<7} {msl:<5} FAIL")
            else:
                print(f"{label:<28} {k_prec:<7} {msl:<5} "
                      f"{d.get('matches','?')}/{d.get('total','?')}")

        # Save
        path = os.path.join(outdir, f"{tag}_kint8_retest_{ts}.json")
        with open(path, "w") as f:
            json.dump({
                "model": model, "tag": tag,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "results": results,
            }, f, indent=2)
        print(f"  Results: {path}")


if __name__ == "__main__":
    main()
