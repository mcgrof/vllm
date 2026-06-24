#!/usr/bin/env python3
"""Shadow buffer bug probe: test whether FP16 SDPA fallback itself is correct.

Key question: when min_seq_len is set high enough that ALL decode steps
use the FP16 shadow path (never the fused INT4 kernel), does output
match baseline?

If YES → the fused INT4 kernel has a quality/bug issue
If NO  → the shadow buffer or FP16 SDPA fallback has a bug

This runs as subprocesses to set VLLM_FUSED_INT4_MIN_SEQ_LEN.
"""

import json
import os
import subprocess
import sys
import time
import textwrap

MODEL = "Qwen/Qwen2.5-7B-Instruct"
# Use exact same filler as the broad probe
FILLER = ("The quick brown fox jumps over the lazy dog. " * 40)
MAX_TOKENS = 3
TEST_LENGTHS = [17, 18, 20, 26, 28, 32, 47, 48, 50, 64]


def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/fused_int4_shadow_bug_probe"
    os.makedirs(output_dir, exist_ok=True)

    # Write the helper script with proper FILLER
    helper_path = os.path.join(output_dir, "_helper.py")
    helper_code = textwrap.dedent('''\
        #!/usr/bin/env python3
        import json, sys, gc, os
        from transformers import AutoTokenizer

        MODEL = "Qwen/Qwen2.5-7B-Instruct"
        FILLER = ("The quick brown fox jumps over the lazy dog. " * 40)
        MAX_TOKENS = 3

        def main():
            kv_dtype = sys.argv[1]      # "auto" or "int4_fused"
            test_lengths = json.loads(sys.argv[2])
            output_path = sys.argv[3]
            msl = sys.argv[4] if len(sys.argv) > 4 else "8"

            os.environ["VLLM_FUSED_INT4_MIN_SEQ_LEN"] = msl

            tokenizer = AutoTokenizer.from_pretrained(MODEL)
            from vllm import LLM, SamplingParams

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
                results.append({"prompt_len": actual, "target_len": tl,
                                "ids": ids, "text": out.text})
                print(f"  [{kv_dtype} msl={msl}] len={actual} ids={ids} text={out.text!r}",
                      file=sys.stderr)

            del llm; gc.collect()
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

        if __name__ == "__main__":
            main()
    ''')
    with open(helper_path, "w") as f:
        f.write(helper_code)

    lengths_json = json.dumps(TEST_LENGTHS)

    # 1. Run baseline (FP16 KV cache)
    print("=== Running baseline (auto KV) ===", file=sys.stderr)
    base_path = os.path.join(output_dir, "baseline.json")
    proc = subprocess.run(
        [sys.executable, helper_path, "auto", lengths_json, base_path],
        capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        print(f"Baseline failed: {proc.stderr[-500:]}", file=sys.stderr)
        return
    print(proc.stderr, file=sys.stderr)
    with open(base_path) as f:
        baseline = json.load(f)

    # 2. Run fused with very high min_seq_len (force FP16 shadow for all decode)
    # Use min_seq_len=999 so ALL sequences use FP16 SDPA fallback
    print("=== Running fused with MSL=999 (always FP16 shadow) ===",
          file=sys.stderr)
    shadow_path = os.path.join(output_dir, "fused_msl999.json")
    env = os.environ.copy()
    env["VLLM_FUSED_INT4_MIN_SEQ_LEN"] = "999"
    proc = subprocess.run(
        [sys.executable, helper_path, "int4_fused", lengths_json,
         shadow_path, "999"],
        env=env, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        print(f"Fused MSL=999 failed: {proc.stderr[-500:]}", file=sys.stderr)
        return
    print(proc.stderr, file=sys.stderr)
    with open(shadow_path) as f:
        shadow = json.load(f)

    # 3. Run fused with default min_seq_len=8 (normal mode)
    print("=== Running fused with MSL=8 (default) ===", file=sys.stderr)
    fused_path = os.path.join(output_dir, "fused_msl8.json")
    proc = subprocess.run(
        [sys.executable, helper_path, "int4_fused", lengths_json,
         fused_path, "8"],
        capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        print(f"Fused MSL=8 failed: {proc.stderr[-500:]}", file=sys.stderr)
        return
    print(proc.stderr, file=sys.stderr)
    with open(fused_path) as f:
        fused = json.load(f)

    # Compare
    print("\n=== COMPARISON ===", file=sys.stderr)
    print(f"{'len':>4} | {'baseline':>20} | {'shadow(msl=999)':>20} | "
          f"{'fused(msl=8)':>20} | shadow_ok | fused_ok",
          file=sys.stderr)
    print("-" * 110, file=sys.stderr)

    comparisons = []
    for b, s, f in zip(baseline, shadow, fused):
        shadow_match = b["ids"] == s["ids"]
        fused_match = b["ids"] == f["ids"]
        shadow_status = "OK" if shadow_match else "FAIL"
        fused_status = "OK" if fused_match else "FAIL"
        print(f"{b['prompt_len']:>4} | {str(b['ids']):>20} | "
              f"{str(s['ids']):>20} | {str(f['ids']):>20} | "
              f"{shadow_status:>9} | {fused_status:>8}",
              file=sys.stderr)
        comparisons.append({
            "prompt_len": b["prompt_len"],
            "baseline_ids": b["ids"],
            "shadow_ids": s["ids"],
            "fused_ids": f["ids"],
            "shadow_match": shadow_match,
            "fused_match": fused_match,
        })

    shadow_passes = sum(1 for c in comparisons if c["shadow_match"])
    fused_passes = sum(1 for c in comparisons if c["fused_match"])
    total = len(comparisons)

    print(f"\nShadow (MSL=999): {shadow_passes}/{total} pass",
          file=sys.stderr)
    print(f"Fused  (MSL=8):   {fused_passes}/{total} pass",
          file=sys.stderr)

    if shadow_passes == total:
        print("\nCONCLUSION: Shadow FP16 path is CORRECT. "
              "Issue is in fused INT4 kernel quality.",
              file=sys.stderr)
    elif shadow_passes == fused_passes:
        print("\nCONCLUSION: Shadow path fails at same rate as fused. "
              "Shadow buffer has a BUG.",
              file=sys.stderr)
    else:
        print(f"\nCONCLUSION: Shadow helps ({shadow_passes} > {fused_passes}) "
              "but doesn't fully fix. Both issues present.",
              file=sys.stderr)

    manifest = {
        "probe": "shadow_bug",
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "test_lengths": TEST_LENGTHS,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "comparisons": comparisons,
        "summary": {
            "total": total,
            "shadow_passes": shadow_passes,
            "fused_passes": fused_passes,
        },
    }
    out_path = os.path.join(output_dir, "shadow_bug_probe.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nResults saved to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
