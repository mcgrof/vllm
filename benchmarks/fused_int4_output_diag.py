#!/usr/bin/env python3
"""Run fused INT4 with decode output instrumentation.

VLLM_FUSED_INT4_DEBUG must be set BEFORE launching, so the EngineCore
subprocess inherits it.
"""
import os
import sys
import json

# These must be set before any vLLM import
os.environ["VLLM_FUSED_INT4_DEBUG"] = "1"
os.environ["VLLM_FUSED_INT4_DEBUG_PATH"] = "/tmp/fused_int4_output_debug.json"

from vllm import LLM, SamplingParams


def main():
    model = "Qwen/Qwen2.5-7B-Instruct"
    print(f"Loading {model} with int4_fused...", file=sys.stderr)

    llm = LLM(
        model=model,
        dtype="float16",
        kv_cache_dtype="int4_fused",
        max_model_len=2048,
        gpu_memory_utilization=0.8,
        disable_log_stats=True,
        enforce_eager=True,
    )

    sp = SamplingParams(max_tokens=2, temperature=0.0)

    for prompt in ["The quick", "The", "The quick brown fox jumps over the lazy dog. The quick brown fox jumps"]:
        print(f"\nPrompt: {prompt!r}", file=sys.stderr)
        outputs = llm.generate([prompt], sp)
        out = outputs[0].outputs[0]
        text = out.text
        ids = list(out.token_ids)
        print(f"  -> ids={ids} text={text!r}", file=sys.stderr)

    del llm

    # Read and print the debug file
    debug_path = "/tmp/fused_int4_output_debug.json"
    if os.path.exists(debug_path):
        with open(debug_path) as f:
            data = json.load(f)
        print(f"\nDebug snapshots: {len(data)}", file=sys.stderr)

        # Print decode_output snapshots
        for snap in data:
            if snap.get("phase") == "decode_output":
                print(f"  decode_output #{snap['fwd_index']}: "
                      f"seq_lens={snap['seq_lens']} "
                      f"out_abs_max={snap['out_abs_max']:.6f} "
                      f"out_abs_mean={snap['out_abs_mean']:.6f} "
                      f"out_zero_frac={snap['out_zero_frac']:.4f} "
                      f"nan={snap['out_nan']} inf={snap['out_inf']}",
                      file=sys.stderr)
                print(f"    out_first16={snap['out_first_head_16'][:8]}",
                      file=sys.stderr)
    else:
        print(f"No debug file at {debug_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
