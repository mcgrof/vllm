#!/usr/bin/env python3
"""Qwen2-7B H100 fused INT4 policy sweep.

Combined MSL fine-sweep + asymmetric/group_size validation.
Mirrors the Mistral/Qwen2.5 H100 sweep methodology exactly.

Phase 1: MSL fine sweep (sym, GS=32) to find the correctness threshold.
Phase 2: Asym + GS sweep at critical MSL thresholds found in Phase 1.

Model: Qwen/Qwen2-7B-Instruct  (28 layers, 28 attn heads, 4 KV heads)
Goal: Verify whether the classifier-style policy result from Qwen2.5-7B
      transfers to Qwen2-7B on H100.
"""

import json
import os
import subprocess
import sys
import time

MODEL = "Qwen/Qwen2-7B-Instruct"
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


INNER_SCRIPT_TEMPLATE = '''
import json, os
os.environ["VLLM_FUSED_INT4_MIN_SEQ_LEN"] = "{msl}"
os.environ["VLLM_FUSED_INT4_ASYMMETRIC"] = "{asym}"
os.environ["VLLM_FUSED_INT4_GROUP_SIZE"] = "{gs}"
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


def run_config(msl, asym="0", gs="32"):
    """Run a single config in a subprocess."""
    prompts_json = json.dumps(PROMPTS).replace("'", "\\'")
    script = INNER_SCRIPT_TEMPLATE.format(
        msl=msl, asym=asym, gs=gs, model=MODEL,
        prompts_json=prompts_json,
        max_model_len=MAX_MODEL_LEN,
        gpu_mem=GPU_MEM,
        max_tokens=MAX_TOKENS,
    )
    env = os.environ.copy()
    env["VLLM_FUSED_INT4_MIN_SEQ_LEN"] = str(msl)
    env["VLLM_FUSED_INT4_ASYMMETRIC"] = str(asym)
    env["VLLM_FUSED_INT4_GROUP_SIZE"] = str(gs)

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
        print(f"  [stderr tail] {r.stderr[-400:]}")
        return None, elapsed
    except subprocess.TimeoutExpired:
        print("  TIMEOUT")
        return None, time.time() - t0


def compare(baseline, result):
    """Compare result against baseline. Returns (matches, total, details)."""
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
        print(f"  {name}: {status} (tok_rate={tok_rate:.2f}"
              f" div@{first_div})")
        if not match:
            print(f"    base: {b_text[:60]}")
            print(f"    test: {t_text[:60]}")

    return matches, total, details


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    # ── Phase 0: FP16 baseline ──
    print("=" * 60)
    print("PHASE 0: FP16 baseline (MSL=999999)")
    print("=" * 60)
    baseline, bl_time = run_config(999999)
    if not baseline:
        print("BASELINE FAILED — aborting")
        return

    print(f"Baseline OK ({bl_time:.1f}s), {len(baseline)} prompts")
    for name in baseline:
        print(f"  {name}: {baseline[name]['text'][:60]}")

    # ── Phase 1: MSL fine sweep (sym, GS=32) ──
    print("\n" + "=" * 60)
    print("PHASE 1: MSL fine sweep (symmetric, GROUP_SIZE=32)")
    print("=" * 60)

    msl_values = [8, 16, 24, 32, 40, 48, 56, 64, 80, 96]
    msl_results = {}

    for msl in msl_values:
        print(f"\nMSL={msl}:")
        result, elapsed = run_config(msl)
        if not result:
            print("  FAILED")
            msl_results[str(msl)] = {"status": "failed"}
            continue

        m, t, details = compare(baseline, result)
        print(f"  Score: {m}/{t}  ({elapsed:.1f}s)")
        msl_results[str(msl)] = {
            "matches": m, "total": t,
            "elapsed_s": round(elapsed, 1),
            "details": details,
        }

    # Identify critical thresholds
    perfect_msl = None
    for msl in msl_values:
        d = msl_results.get(str(msl), {})
        if d.get("matches") == d.get("total") and d.get("total", 0) > 0:
            perfect_msl = msl
            break

    print(f"\n{'='*50}")
    print("Phase 1 Summary: MSL Fine Sweep")
    print(f"{'MSL':<6} {'Score':<8}")
    print("-" * 20)
    for msl in msl_values:
        d = msl_results.get(str(msl), {})
        if d.get("status") == "failed":
            print(f"{msl:<6} FAIL")
        else:
            print(f"{msl:<6} {d.get('matches','?')}/{d.get('total','?')}")

    if perfect_msl:
        print(f"\nLowest perfect MSL: {perfect_msl}")
    else:
        print("\nNo perfect MSL found in sweep range!")

    # Save Phase 1 results
    phase1_path = os.path.join(outdir, f"qwen2_msl_fine_sweep_{ts}.json")
    phase1_data = {
        "model": MODEL,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "baseline_samples": {
            name: baseline[name]["text"][:80] for name in baseline
        },
        "sweeps": msl_results,
        "lowest_perfect_msl": perfect_msl,
    }
    with open(phase1_path, "w") as f:
        json.dump(phase1_data, f, indent=2)
    print(f"Phase 1 results: {phase1_path}")

    # ── Phase 2: Asym + GS at critical thresholds ──
    if perfect_msl is None:
        print("\nSkipping Phase 2 — no perfect MSL found")
        return

    critical_msls = sorted(set(
        m for m in [perfect_msl - 16, perfect_msl - 8, perfect_msl]
        if m >= 8
    ))

    print("\n" + "=" * 60)
    print(f"PHASE 2: Asym + GS sweep at critical MSLs: {critical_msls}")
    print("=" * 60)

    asym_configs = []
    for msl in critical_msls:
        for asym in ["0", "1"]:
            for gs in ["32", "16"]:
                a_str = "asym" if asym == "1" else "sym"
                label = f"{a_str}_g{gs}_msl{msl}"
                asym_configs.append((msl, asym, gs, label))

    asym_results = {}
    for msl, asym, gs, label in asym_configs:
        a_str = "asym" if asym == "1" else "sym"
        print(f"\n[{label}] {a_str} g{gs} msl{msl}...")
        result, elapsed = run_config(msl, asym, gs)
        if result:
            m, t, details = compare(baseline, result)
            print(f"  Score: {m}/{t}  ({elapsed:.1f}s)")
            asym_results[label] = {
                "matches": m, "total": t,
                "msl": msl, "asymmetric": asym == "1",
                "group_size": int(gs),
                "elapsed_s": round(elapsed, 1),
                "details": details,
            }
        else:
            print("  FAILED")
            asym_results[label] = {"status": "failed"}

    # Phase 2 summary
    print(f"\n{'='*60}")
    print("Phase 2 Summary: Asym + GS at critical thresholds")
    print(f"{'Config':<22} {'MSL':<5} {'Asym':<6} {'GS':<4} {'Score':<8}")
    print("-" * 50)
    for _, _, _, label in asym_configs:
        d = asym_results.get(label, {})
        if d.get("status") == "failed":
            print(f"{label:<22} {'FAIL':>30}")
        else:
            print(f"{label:<22} {d.get('msl',''):<5} "
                  f"{'Y' if d.get('asymmetric') else 'N':<6} "
                  f"{d.get('group_size',''):<4} "
                  f"{d.get('matches','?')}/{d.get('total','?')}")

    # Save Phase 2 results
    phase2_path = os.path.join(outdir, f"qwen2_asym_sweep_{ts}.json")
    phase2_data = {
        "model": MODEL,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "critical_msls": critical_msls,
        "configs": asym_results,
    }
    with open(phase2_path, "w") as f:
        json.dump(phase2_data, f, indent=2)
    print(f"Phase 2 results: {phase2_path}")

    # ── Final verdict ──
    print("\n" + "=" * 60)
    print("FINAL VERDICT")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"Lowest perfect MSL (sym, GS=32): {perfect_msl}")

    sub_threshold_improvement = False
    for _, _, _, label in asym_configs:
        d = asym_results.get(label, {})
        if d.get("status") == "failed":
            continue
        msl = d.get("msl", 0)
        if msl < perfect_msl and d.get("matches") == d.get("total"):
            sub_threshold_improvement = True
            print(f"  ** Sub-threshold perfect: {label} "
                  f"({d['matches']}/{d['total']})")

    if sub_threshold_improvement:
        print("Asym or smaller GS DOES help below the sym/g32 threshold!")
    else:
        print(f"Asym and smaller GS provide NO improvement below MSL={perfect_msl}")
        print(f"Recommended policy: VLLM_FUSED_INT4_MIN_SEQ_LEN={perfect_msl}")

    # ── Qwen2 vs Qwen2.5 comparison ──
    print("\n" + "=" * 60)
    print("COMPARISON: Qwen2-7B vs Qwen2.5-7B-Instruct")
    print("=" * 60)
    print(f"Qwen2-7B lowest perfect MSL (sym, GS=32): {perfect_msl}")
    print(f"Qwen2.5-7B lowest perfect MSL (sym, GS=32): 48  (prior result)")
    if perfect_msl == 48:
        print(">>> SAME THRESHOLD — classifier result transfers from Qwen2.5 to Qwen2")
    elif perfect_msl is not None:
        print(f">>> DIFFERENT THRESHOLD — Qwen2 needs MSL={perfect_msl} vs Qwen2.5 MSL=48")


if __name__ == "__main__":
    main()
