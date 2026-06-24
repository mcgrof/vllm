#!/usr/bin/env python3
"""min_seq_len sweep: test whether increasing the FP16 shadow window
fixes longer prompt lengths.

This must run as subprocesses because VLLM_FUSED_INT4_MIN_SEQ_LEN
is read at import time.
"""

import json
import os
import subprocess
import sys
import time

MODEL = "Qwen/Qwen2.5-7B-Instruct"
FILLER = ("The quick brown fox jumps over the lazy dog. " * 40)
MAX_TOKENS = 3

# Test matrix: for each threshold, test a few interesting lengths
# around and above the threshold
THRESHOLDS = [16, 32, 48, 64, 96, 128]
# Always test these lengths
TEST_LENGTHS = [17, 18, 20, 26, 28, 32, 47, 48, 50, 64, 80, 96, 128, 160]


def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/fused_int4_msl_sweep"
    os.makedirs(output_dir, exist_ok=True)

    # Write the single-run helper
    helper_path = os.path.join(output_dir, "_run_one.py")
    with open(helper_path, "w") as f:
        f.write(f'''#!/usr/bin/env python3
import json, sys, gc, os
from transformers import AutoTokenizer

MODEL = "{MODEL}"
FILLER = ("{FILLER[:200]}" * 4)
MAX_TOKENS = {MAX_TOKENS}

def main():
    threshold = int(sys.argv[1])
    test_lengths = json.loads(sys.argv[2])
    output_path = sys.argv[3]
    mode = sys.argv[4]  # "baseline" or "fused"

    os.environ["VLLM_FUSED_INT4_MIN_SEQ_LEN"] = str(threshold)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    from vllm import LLM, SamplingParams

    kv_dtype = "auto" if mode == "baseline" else "int4_fused"
    llm = LLM(model=MODEL, dtype="float16", kv_cache_dtype=kv_dtype,
              max_model_len=2048, gpu_memory_utilization=0.8,
              disable_log_stats=True, enforce_eager=True)
    sp = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0)

    results = []
    for tl in test_lengths:
        tokens = tokenizer.encode(FILLER)[:tl]
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        actual = len(tokenizer.encode(text))
        out = llm.generate([text], sp)[0].outputs[0]
        ids = list(out.token_ids)
        results.append({{"prompt_len": actual, "target_len": tl,
                        "ids": ids, "text": out.text}})
        print(f"  [{{mode}}] msl={{threshold}} len={{actual}} ids={{ids}}", file=sys.stderr)

    del llm; gc.collect()
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
''')

    all_results = []

    # First get baseline (doesn't depend on threshold)
    print("=== Running baseline ===", file=sys.stderr)
    base_path = os.path.join(output_dir, "baseline.json")
    cmd = [sys.executable, helper_path, "8",
           json.dumps(TEST_LENGTHS), base_path, "baseline"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        print(f"Baseline failed: {proc.stderr[-500:]}", file=sys.stderr)
        return
    with open(base_path) as f:
        baseline = json.load(f)
    baseline_map = {r["prompt_len"]: r for r in baseline}

    # Test each threshold
    for threshold in THRESHOLDS:
        print(f"=== Testing min_seq_len={threshold} ===", file=sys.stderr)
        fused_path = os.path.join(output_dir, f"fused_msl{threshold}.json")
        env = os.environ.copy()
        env["VLLM_FUSED_INT4_MIN_SEQ_LEN"] = str(threshold)
        cmd = [sys.executable, helper_path, str(threshold),
               json.dumps(TEST_LENGTHS), fused_path, "fused"]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                              timeout=600)
        if proc.returncode != 0:
            print(f"  FAILED: {proc.stderr[-500:]}", file=sys.stderr)
            all_results.append({
                "threshold": threshold,
                "error": proc.stderr[-500:],
            })
            continue

        with open(fused_path) as f:
            fused = json.load(f)

        comparisons = []
        for fr in fused:
            bl = baseline_map.get(fr["prompt_len"])
            if bl:
                match = bl["ids"] == fr["ids"]
                status = "OK" if match else "FAIL"
                print(f"  msl={threshold} len={fr['prompt_len']} [{status}]"
                      f" base={bl['ids']} fused={fr['ids']}",
                      file=sys.stderr)
                comparisons.append({
                    "prompt_len": fr["prompt_len"],
                    "baseline_ids": bl["ids"],
                    "fused_ids": fr["ids"],
                    "match": match,
                })

        passes = sum(1 for c in comparisons if c["match"])
        total = len(comparisons)
        all_results.append({
            "threshold": threshold,
            "comparisons": comparisons,
            "passes": passes,
            "total": total,
        })
        print(f"  min_seq_len={threshold}: {passes}/{total} pass",
              file=sys.stderr)

    # Summary
    manifest = {
        "probe": "min_seq_len_sweep",
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "thresholds": THRESHOLDS,
        "test_lengths": TEST_LENGTHS,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": all_results,
    }
    out_path = os.path.join(output_dir, "msl_sweep_results.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nResults saved to {out_path}", file=sys.stderr)

    # Print summary table
    print("\n=== MIN_SEQ_LEN SWEEP SUMMARY ===", file=sys.stderr)
    print(f"{'msl':>5} | {'passes':>6} | {'total':>5} | failing lengths",
          file=sys.stderr)
    print("-" * 60, file=sys.stderr)
    for r in all_results:
        if "error" in r:
            print(f"{r['threshold']:>5} | ERROR", file=sys.stderr)
            continue
        fails = [c["prompt_len"] for c in r["comparisons"] if not c["match"]]
        print(f"{r['threshold']:>5} | {r['passes']:>6} | {r['total']:>5} | {fails}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
