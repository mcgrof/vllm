#!/usr/bin/env python3
"""Final focused retest: resolve Qwen2 vs Qwen2.5 K/V precision-split ambiguity.

Goals:
  1. Reproducibility: re-run K_FP16/V_INT4 at MSL=8 twice per model
     to verify the same prompts fail each time (determinism check).
  2. Extended prompts: add 5 more prompts to increase statistical surface
     from 7 to 12 prompts, reducing small-N artifacts.
  3. K_INT8 gap: re-run K_INT8/V_INT4 at MSL=8 to complete the comparison.

Usage:
    python benchmarks/fused_int4_kv_final_retest.py [OUTDIR]
"""

import json
import os
import subprocess
import sys
import time

# --- Configuration ---
MODELS = [
    ("Qwen/Qwen2-7B-Instruct", "qwen2"),
    ("Qwen/Qwen2.5-7B-Instruct", "qwen25"),
]
GPU_MEM = 0.70
MAX_MODEL_LEN = 512
MAX_TOKENS = 20

# Original 7 prompts (identical to prior sweep)
PROMPTS_ORIG = [
    ("short_8", "What is 2+2?"),
    ("short_16", "Explain the concept of gravity in one sentence."),
    ("med_32", "Write a haiku about the ocean and then explain what makes it a haiku."),
    ("med_48", "List the first 10 prime numbers and explain why 1 is not considered prime. Be concise."),
    ("long_64", "Explain the difference between TCP and UDP protocols. Include three key differences and a use case for each."),
    ("long_96", "Write a short story about a robot that learns to paint. Include dialogue between the robot and its creator. Make it exactly 3 paragraphs."),
    ("long_128", "Explain the history of the Internet from ARPANET to modern day. Cover 5 major milestones with dates and significance. Be detailed."),
]

# 5 additional prompts to expand test surface
PROMPTS_EXTRA = [
    ("short_12", "Name the four seasons."),
    ("med_40", "Explain the difference between a stack and a queue in computer science. Give one example of each."),
    ("med_56", "What is photosynthesis? Describe the process step by step including the role of chlorophyll and sunlight."),
    ("long_80", "Compare and contrast democracy and authoritarianism as forms of government. Discuss their strengths, weaknesses, and provide historical examples."),
    ("long_112", "Describe the water cycle in detail, including evaporation, condensation, precipitation, and collection. Explain how human activities like deforestation and urbanization affect each stage."),
]

ALL_PROMPTS = PROMPTS_ORIG + PROMPTS_EXTRA

# Configurations to test
# (label, msl, k_precision, repetitions)
CONFIGS = [
    # Reproducibility: K_FP16/V_INT4 at the critical MSL=8 (2 reps)
    ("k_fp16_v_int4_msl8", 8, "fp16", 2),
    # Key comparison: K_INT8/V_INT4 at MSL=8 (1 rep)
    ("k_int8_v_int4_msl8", 8, "int8", 1),
    # Check whether MSL=24 narrows the gap
    ("k_fp16_v_int4_msl24", 24, "fp16", 1),
    ("k_int8_v_int4_msl24", 24, "int8", 1),
    # MSL=48 sanity (should be 12/12)
    ("k_fp16_v_int4_msl48", 48, "fp16", 1),
]

INNER_SCRIPT_TEMPLATE = '''
import json, os
os.environ["VLLM_FUSED_INT4_MIN_SEQ_LEN"] = "{msl}"
os.environ["VLLM_FUSED_INT4_ASYMMETRIC"] = "0"
os.environ["VLLM_FUSED_INT4_GROUP_SIZE"] = "32"
os.environ["VLLM_FUSED_INT4_K_PRECISION"] = "{k_precision}"
os.environ["TRITON_ALLOW_NON_CONSTEXPR_GLOBALS"] = "1"
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


def run_config(model, msl, k_precision="int4"):
    """Run a single config in a subprocess."""
    prompts_json = json.dumps(ALL_PROMPTS).replace("'", "\\'")
    script = INNER_SCRIPT_TEMPLATE.format(
        msl=msl, k_precision=k_precision,
        model=model, prompts_json=prompts_json,
        max_model_len=MAX_MODEL_LEN, gpu_mem=GPU_MEM,
        max_tokens=MAX_TOKENS,
    )
    env = os.environ.copy()
    env["VLLM_FUSED_INT4_MIN_SEQ_LEN"] = str(msl)
    env["VLLM_FUSED_INT4_ASYMMETRIC"] = "0"
    env["VLLM_FUSED_INT4_GROUP_SIZE"] = "32"
    env["VLLM_FUSED_INT4_K_PRECISION"] = str(k_precision)
    env["TRITON_ALLOW_NON_CONSTEXPR_GLOBALS"] = "1"

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
    """Compare result against baseline, return per-prompt details."""
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
        print(f"    {name}: {status} (tok_rate={tok_rate:.2f}"
              f" div@{first_div})")
        if not match:
            print(f"      base: {b_text[:80]}")
            print(f"      test: {t_text[:80]}")

    return matches, total, details


def run_model(model, model_tag, outdir):
    """Run focused retest for one model."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    print("=" * 70)
    print(f"MODEL: {model} (12 prompts)")
    print("=" * 70)

    # --- FP16 baseline ---
    print("\n  [baseline] FP16 (MSL=999999) ...")
    baseline, bl_time = run_config(model, 999999, "int4")
    if not baseline:
        print("  BASELINE FAILED — aborting")
        return None
    print(f"  Baseline OK ({bl_time:.1f}s), {len(baseline)} prompts")
    for name in baseline:
        print(f"    {name}: {baseline[name]['text'][:70]}")

    # --- Run configs ---
    all_results = {}
    for label, msl, k_prec, reps in CONFIGS:
        for rep in range(reps):
            rep_label = f"{label}_r{rep+1}" if reps > 1 else label
            print(f"\n  [{rep_label}] k={k_prec} msl={msl} ...")
            result, elapsed = run_config(model, msl, k_prec)
            if result:
                m, t, details = compare(baseline, result)
                print(f"    Score: {m}/{t}  ({elapsed:.1f}s)")
                all_results[rep_label] = {
                    "matches": m, "total": t,
                    "k_precision": k_prec, "msl": msl,
                    "repetition": rep + 1,
                    "elapsed_s": round(elapsed, 1),
                    "details": details,
                    "raw_outputs": {
                        name: result[name]["text"][:120] for name in result
                    },
                }
            else:
                print("    FAILED")
                all_results[rep_label] = {
                    "status": "failed", "k_precision": k_prec, "msl": msl
                }

    # --- Summary ---
    print(f"\n{'='*70}")
    print(f"SUMMARY: {model}")
    print(f"{'Config':<30} {'K-prec':<7} {'MSL':<5} {'Orig':>5} {'Extra':>5} {'Total':>6}")
    print("-" * 65)
    for label in all_results:
        d = all_results[label]
        if d.get("status") == "failed":
            print(f"{label:<30} {d.get('k_precision','?'):<7} {d.get('msl','?'):<5} {'FAIL':>6}")
            continue
        det = d.get("details", {})
        orig_pass = sum(1 for k, v in det.items()
                        if v["match"] and k in dict(PROMPTS_ORIG))
        extra_pass = sum(1 for k, v in det.items()
                         if v["match"] and k in dict(PROMPTS_EXTRA))
        total_pass = d.get("matches", 0)
        total_n = d.get("total", 0)
        print(f"{label:<30} {d['k_precision']:<7} {d['msl']:<5} "
              f"{orig_pass}/7  {extra_pass}/5  {total_pass}/{total_n}")

    # --- Save ---
    result_path = os.path.join(
        outdir, f"{model_tag}_final_retest_{ts}.json")
    result_data = {
        "model": model,
        "model_tag": model_tag,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_prompts": len(ALL_PROMPTS),
        "prompt_names_orig": [p[0] for p in PROMPTS_ORIG],
        "prompt_names_extra": [p[0] for p in PROMPTS_EXTRA],
        "baseline_samples": {
            name: baseline[name]["text"][:120] for name in baseline
        },
        "configs": all_results,
    }
    with open(result_path, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"\nResults saved: {result_path}")
    return result_data


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    all_models = {}
    for model, tag in MODELS:
        data = run_model(model, tag, outdir)
        if data:
            all_models[tag] = data

    # --- Cross-model summary ---
    if len(all_models) > 1:
        print("\n" + "=" * 70)
        print("CROSS-MODEL COMPARISON (K_FP16/V_INT4 @ MSL=8)")
        print("=" * 70)
        for tag, data in all_models.items():
            for cfg_label, cfg_data in data.get("configs", {}).items():
                if "fp16" in cfg_label and "msl8" in cfg_label:
                    m = cfg_data.get("matches", "?")
                    t = cfg_data.get("total", "?")
                    print(f"  {tag} {cfg_label}: {m}/{t}")

    # Manifest
    manifest_path = os.path.join(outdir, f"final_retest_manifest_{ts}.json")
    manifest = {
        "experiment": "kv_precision_split_final_retest",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "description": (
            "Focused retest to resolve Qwen2 vs Qwen2.5 K/V precision-split "
            "ambiguity. Extended from 7 to 12 prompts. Reproducibility check "
            "with 2 repetitions at MSL=8."
        ),
        "models_tested": [m for m, _ in MODELS],
        "n_prompts": len(ALL_PROMPTS),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {manifest_path}")


if __name__ == "__main__":
    main()
