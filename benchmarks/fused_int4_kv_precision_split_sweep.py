#!/usr/bin/env python3
"""K/V precision-split sweep for Qwen-class 7B models on H100.

Tests the paper claim: Qwen-class models need much higher precision for K
than for V.  Three policies are compared against FP16 baseline:

  1. K_INT4 / V_INT4  (existing fused INT4 — current best with MSL=48)
  2. K_INT8 / V_INT4  (higher K precision via symmetric INT8)
  3. K_FP16 / V_INT4  (full K precision, only V quantised)

If the paper claim holds, K_FP16/V_INT4 should match baseline perfectly even
at MSL=8 (no warm-up needed), and K_INT8/V_INT4 should substantially narrow
the gap vs. K_INT4/V_INT4.

Usage:
    python benchmarks/fused_int4_kv_precision_split_sweep.py [OUTDIR]
    # Default OUTDIR = current directory
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

PROMPTS = [
    ("short_8", "What is 2+2?"),
    ("short_16", "Explain the concept of gravity in one sentence."),
    ("med_32", "Write a haiku about the ocean and then explain what makes it a haiku."),
    ("med_48", "List the first 10 prime numbers and explain why 1 is not considered prime. Be concise."),
    ("long_64", "Explain the difference between TCP and UDP protocols. Include three key differences and a use case for each."),
    ("long_96", "Write a short story about a robot that learns to paint. Include dialogue between the robot and its creator. Make it exactly 3 paragraphs."),
    ("long_128", "Explain the history of the Internet from ARPANET to modern day. Cover 5 major milestones with dates and significance. Be detailed."),
]

# Policies to test.  Each tuple: (label, msl, k_precision, asymmetric, group_size)
# MSL=8 is intentionally aggressive — if K_FP16 survives here, the claim is strong.
POLICIES = [
    # Reference: symmetric INT4 K+V at best-known MSL
    ("kv_int4_msl48",   48, "int4", "0", "32"),
    ("kv_int4_msl8",     8, "int4", "0", "32"),
    # K_INT8 / V_INT4 (the intermediate split)
    ("k_int8_v_int4_msl48", 48, "int8", "0", "32"),
    ("k_int8_v_int4_msl8",   8, "int8", "0", "32"),
    ("k_int8_v_int4_msl24", 24, "int8", "0", "32"),
    # K_FP16 / V_INT4 (maximum K precision, V quantised)
    ("k_fp16_v_int4_msl48", 48, "fp16", "0", "32"),
    ("k_fp16_v_int4_msl8",   8, "fp16", "0", "32"),
    ("k_fp16_v_int4_msl24", 24, "fp16", "0", "32"),
    ("k_fp16_v_int4_msl1",   1, "fp16", "0", "32"),
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
    """Run a single config in a subprocess (env read at import time)."""
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
    """Compare result against baseline."""
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
            print(f"      base: {b_text[:60]}")
            print(f"      test: {t_text[:60]}")

    return matches, total, details


def run_model_sweep(model, model_tag, outdir):
    """Run full precision-split sweep for one model."""
    ts = time.strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print(f"MODEL: {model}")
    print("=" * 70)

    # --- FP16 baseline (MSL=999999 → everything goes through FP16 SDPA) ---
    print("\n  [baseline] FP16 (MSL=999999) ...")
    baseline, bl_time = run_config(model, 999999)
    if not baseline:
        print("  BASELINE FAILED — aborting this model")
        return None

    print(f"  Baseline OK ({bl_time:.1f}s), {len(baseline)} prompts")
    for name in baseline:
        print(f"    {name}: {baseline[name]['text'][:60]}")

    # --- Policy sweep ---
    all_results = {}
    for label, msl, k_prec, asym, gs in POLICIES:
        print(f"\n  [{label}] k={k_prec} msl={msl} ...")
        result, elapsed = run_config(model, msl, k_prec, asym, gs)
        if result:
            m, t, details = compare(baseline, result)
            print(f"    Score: {m}/{t}  ({elapsed:.1f}s)")
            all_results[label] = {
                "matches": m, "total": t,
                "k_precision": k_prec, "msl": msl,
                "asymmetric": asym == "1", "group_size": int(gs),
                "elapsed_s": round(elapsed, 1),
                "details": details,
                "raw_outputs": {
                    name: result[name]["text"][:120] for name in result
                },
            }
        else:
            print("    FAILED")
            all_results[label] = {"status": "failed", "k_precision": k_prec,
                                   "msl": msl}

    # --- Summary table ---
    print(f"\n{'='*70}")
    print(f"SUMMARY: {model}")
    print(f"{'Policy':<28} {'K-prec':<7} {'MSL':<5} {'Score':<8} {'Verdict'}")
    print("-" * 65)
    for label, msl, k_prec, _, _ in POLICIES:
        d = all_results.get(label, {})
        if d.get("status") == "failed":
            print(f"{label:<28} {k_prec:<7} {msl:<5} {'FAIL':>8}")
        else:
            m = d.get("matches", "?")
            t = d.get("total", "?")
            perfect = m == t and t and t > 0
            verdict = "PERFECT" if perfect else "DEGRADED"
            print(f"{label:<28} {k_prec:<7} {msl:<5} {m}/{t:<5} {verdict}")

    # --- Classifier verdict ---
    print(f"\n{'='*70}")
    print("CLASSIFIER VERDICT")
    print("=" * 70)

    # Check: does K_FP16/V_INT4 at MSL=8 match baseline perfectly?
    kfp16_msl8 = all_results.get("k_fp16_v_int4_msl8", {})
    kfp16_perfect = (kfp16_msl8.get("matches") == kfp16_msl8.get("total")
                     and kfp16_msl8.get("total", 0) > 0)

    kint4_msl8 = all_results.get("kv_int4_msl8", {})
    kint4_degraded = (kint4_msl8.get("matches", 0) <
                      kint4_msl8.get("total", 0))

    kint8_msl8 = all_results.get("k_int8_v_int4_msl8", {})
    kint8_better_than_int4 = (kint8_msl8.get("matches", 0) >
                               kint4_msl8.get("matches", 0))

    if kfp16_perfect and kint4_degraded:
        print(">>> PAPER CLAIM CONFIRMED: K precision is the bottleneck.")
        print("    K_FP16/V_INT4 at MSL=8 is PERFECT while K_INT4/V_INT4 degrades.")
        claim_status = "CONFIRMED"
    elif kfp16_perfect:
        print(">>> PAPER CLAIM SUPPORTED: K_FP16/V_INT4 is perfect at MSL=8.")
        print("    (K_INT4/V_INT4 also passes — may need harder prompts.)")
        claim_status = "SUPPORTED_WEAK"
    elif kint4_degraded and not kfp16_perfect:
        print(">>> PAPER CLAIM PARTIALLY SUPPORTED: both degrade, "
              "but K_FP16 may be less degraded.")
        claim_status = "PARTIAL"
    else:
        print(">>> PAPER CLAIM NOT CONFIRMED in this test.")
        claim_status = "NOT_CONFIRMED"

    if kint8_better_than_int4:
        print(f"    K_INT8 improves over K_INT4 at MSL=8: "
              f"{kint8_msl8.get('matches')}/{kint8_msl8.get('total')} vs "
              f"{kint4_msl8.get('matches')}/{kint4_msl8.get('total')}")
    else:
        print(f"    K_INT8 does NOT improve over K_INT4 at MSL=8.")

    # --- Save results ---
    result_path = os.path.join(
        outdir, f"{model_tag}_kv_precision_split_{ts}.json")
    result_data = {
        "model": model,
        "model_tag": model_tag,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "classifier_verdict": claim_status,
        "baseline_samples": {
            name: baseline[name]["text"][:120] for name in baseline
        },
        "policies": all_results,
        "kfp16_msl8_perfect": kfp16_perfect,
        "kint4_msl8_degraded": kint4_degraded,
        "kint8_better_than_int4": kint8_better_than_int4,
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
        data = run_model_sweep(model, tag, outdir)
        if data:
            all_models[tag] = data

    # --- Cross-model comparison ---
    if len(all_models) > 1:
        print("\n" + "=" * 70)
        print("CROSS-MODEL COMPARISON")
        print("=" * 70)
        for tag, data in all_models.items():
            print(f"  {tag}: classifier_verdict={data['classifier_verdict']}")

    # --- Combined manifest ---
    manifest_path = os.path.join(
        outdir, f"kv_precision_split_manifest_{ts}.json")
    manifest = {
        "experiment": "kv_precision_split",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "description": (
            "Tests paper claim: Qwen-class 7B models need higher K "
            "precision than V. Compares K_INT4, K_INT8, K_FP16 "
            "with V always INT4."
        ),
        "models": {tag: data["classifier_verdict"]
                   for tag, data in all_models.items()},
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {manifest_path}")


if __name__ == "__main__":
    main()
