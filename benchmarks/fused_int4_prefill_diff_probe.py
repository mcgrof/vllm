#!/usr/bin/env python3
"""Prefill implementation difference probe.

Tests whether the fused INT4 failures are caused by using torch SDPA
for prefill instead of FlashAttention.

Runs:
1. Baseline with FlashAttention (default)
2. Baseline with TORCH_SDPA backend (same dtype, different prefill kernel)
3. Fused INT4 with MSL=999 (shadow FP16 decode, SDPA prefill)

If (1) != (2), the issue is prefill implementation difference.
If (2) == (3), confirms the fused path's only divergence is prefill.
"""

import json
import os
import subprocess
import sys
import time
import textwrap

MODEL = "Qwen/Qwen2.5-7B-Instruct"
FILLER = ("The quick brown fox jumps over the lazy dog. " * 40)
MAX_TOKENS = 3
TEST_LENGTHS = [1, 5, 10, 17, 18, 20, 26, 28, 32, 48, 64]


def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/fused_int4_prefill_diff"
    os.makedirs(output_dir, exist_ok=True)

    helper_path = os.path.join(output_dir, "_helper.py")
    helper_code = textwrap.dedent('''\
        #!/usr/bin/env python3
        import json, sys, gc, os
        from transformers import AutoTokenizer

        MODEL = "Qwen/Qwen2.5-7B-Instruct"
        FILLER = ("The quick brown fox jumps over the lazy dog. " * 40)
        MAX_TOKENS = 3

        def main():
            kv_dtype = sys.argv[1]
            test_lengths = json.loads(sys.argv[2])
            output_path = sys.argv[3]
            label = sys.argv[4]

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
                print(f"  [{label}] len={actual} ids={ids}",
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
    configs = [
        ("baseline_flash", "auto", {}, "baseline_flash"),
        ("baseline_sdpa", "auto",
         {"VLLM_ATTENTION_BACKEND": "TORCH_SDPA"}, "baseline_sdpa"),
        ("fused_msl999", "int4_fused",
         {"VLLM_FUSED_INT4_MIN_SEQ_LEN": "999"}, "fused_msl999"),
    ]

    all_data = {}
    for name, kv_dtype, extra_env, label in configs:
        print(f"\n=== Running {name} ===", file=sys.stderr)
        out_path = os.path.join(output_dir, f"{name}.json")
        env = os.environ.copy()
        env.update(extra_env)
        cmd = [sys.executable, helper_path, kv_dtype, lengths_json,
               out_path, label]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                              timeout=600)
        if proc.returncode != 0:
            print(f"  FAILED: {proc.stderr[-1000:]}", file=sys.stderr)
            all_data[name] = None
            continue
        print(proc.stderr, file=sys.stderr)
        with open(out_path) as f:
            all_data[name] = json.load(f)

    if any(v is None for v in all_data.values()):
        print("Some configurations failed, cannot compare", file=sys.stderr)
        # Try to compare what we have
        pass

    # Compare
    print("\n=== COMPARISON ===", file=sys.stderr)
    header = f"{'len':>4}"
    for name in all_data:
        header += f" | {name:>25}"
    header += " | flash_vs_sdpa | flash_vs_fused"
    print(header, file=sys.stderr)
    print("-" * len(header), file=sys.stderr)

    comparisons = []
    for i in range(len(TEST_LENGTHS)):
        row = {"prompt_len": TEST_LENGTHS[i]}
        ids = {}
        for name, data in all_data.items():
            if data:
                ids[name] = data[i]["ids"]
                row[name + "_ids"] = data[i]["ids"]

        flash_vs_sdpa = ids.get("baseline_flash") == ids.get("baseline_sdpa")
        flash_vs_fused = ids.get("baseline_flash") == ids.get("fused_msl999")
        sdpa_vs_fused = ids.get("baseline_sdpa") == ids.get("fused_msl999")
        row["flash_vs_sdpa"] = flash_vs_sdpa
        row["flash_vs_fused"] = flash_vs_fused
        row["sdpa_vs_fused"] = sdpa_vs_fused

        line = f"{TEST_LENGTHS[i]:>4}"
        for name in all_data:
            if all_data[name]:
                line += f" | {str(all_data[name][i]['ids']):>25}"
            else:
                line += f" | {'ERROR':>25}"
        line += f" | {'OK' if flash_vs_sdpa else 'DIFF':>13}"
        line += f" | {'OK' if flash_vs_fused else 'DIFF':>14}"
        print(line, file=sys.stderr)
        comparisons.append(row)

    manifest = {
        "probe": "prefill_diff",
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "comparisons": comparisons,
    }
    out_path = os.path.join(output_dir, "prefill_diff_results.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nResults saved to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
