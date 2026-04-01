#!/usr/bin/env python3
"""Fused-off control: verify INT4 cache with FP16-only decode produces
correct output (proves the quantize path is fine, only fused decode breaks)."""

import gc
import os
import sys
import time
import torch

# Force FP16-only decode by setting MSL impossibly high
os.environ["VLLM_FUSED_INT4_MIN_SEQ_LEN"] = "99999"

from vllm import LLM, SamplingParams

PROMPTS = [
    "What is 2 + 2?",
    ("The quick brown fox jumps over the lazy dog. " * 3 +
     "Summarize the above."),
    ("In the year 2024, artificial intelligence continued to advance "
     "at a rapid pace. Large language models became more capable and "
     "efficient. What is the main topic?"),
]

SP = SamplingParams(max_tokens=30, temperature=0.0)

print("=== FUSED-OFF CONTROL (MSL=99999, INT4 write + FP16 decode) ===")
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct", dtype="float16",
    kv_cache_dtype="int4_fused", max_model_len=2048,
    gpu_memory_utilization=0.8, disable_log_stats=True,
    enforce_eager=True, max_num_seqs=1,
)

fused_off = []
for p in PROMPTS:
    out = llm.generate([p], SP)[0].outputs[0]
    ids = list(out.token_ids)
    print(f"  prompt={p[:50]!r}")
    print(f"  text={out.text!r}")
    print(f"  ids={ids[:10]}...")
    print()
    fused_off.append({"prompt": p[:50], "ids": ids, "text": out.text})

del llm; gc.collect(); torch.cuda.empty_cache(); time.sleep(3)

print("=== FP16 BASELINE ===")
llm2 = LLM(
    model="Qwen/Qwen2.5-7B-Instruct", dtype="float16",
    kv_cache_dtype="auto", max_model_len=2048,
    gpu_memory_utilization=0.8, disable_log_stats=True,
    enforce_eager=True, max_num_seqs=1,
)

fp16 = []
for p in PROMPTS:
    out = llm2.generate([p], SP)[0].outputs[0]
    ids = list(out.token_ids)
    print(f"  prompt={p[:50]!r}")
    print(f"  text={out.text!r}")
    print(f"  ids={ids[:10]}...")
    print()
    fp16.append({"prompt": p[:50], "ids": ids, "text": out.text})

del llm2; gc.collect(); torch.cuda.empty_cache()

print("\n=== COMPARISON ===")
for fo, f16 in zip(fused_off, fp16):
    match = fo["ids"] == f16["ids"]
    print(f"  [{('PASS' if match else 'FAIL')}] {fo['prompt']!r}")
    if not match:
        print(f"    fused-off: {fo['text'][:80]!r}")
        print(f"    fp16:      {f16['text'][:80]!r}")
