#!/usr/bin/env python3
"""Minimal fused INT4 kernel diagnostic on A100.

Directly tests the quantize->dequantize->attend pipeline outside vLLM serving
to isolate whether the corruption is in:
  1. The quantization (reshape_and_cache_int4) kernel
  2. The dequantization in the decode kernel
  3. The attention computation in the decode kernel
  4. A shape/stride mismatch between write and read

Tests on synthetic data and on real model KV outputs.
"""

import os
import sys
import json
import time
import torch
import numpy as np

# Force fused backend constants
os.environ.setdefault("VLLM_FUSED_INT4_MIN_SEQ_LEN", "48")

from datetime import datetime, timezone


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def test_roundtrip_correctness():
    """Test: quantize FP16 -> INT4, then dequantize, check error."""
    from vllm.v1.attention.backends.fused_int4 import (
        reshape_and_cache_int4,
        GROUP_SIZE,
    )

    log("=== TEST 1: Quantize-dequantize roundtrip ===")

    num_tokens = 64
    num_heads = 4  # Qwen2.5-7B has 4 KV heads
    head_size = 128
    block_size = 16
    half_hd = head_size // 2
    num_groups = head_size // GROUP_SIZE
    num_blocks = (num_tokens + block_size - 1) // block_size

    # Create synthetic FP16 K/V with known distribution
    torch.manual_seed(42)
    key = torch.randn(num_tokens, num_heads, head_size,
                       dtype=torch.float16, device="cuda")
    value = torch.randn(num_tokens, num_heads, head_size,
                         dtype=torch.float16, device="cuda")

    # Allocate INT4 cache
    k_cache = torch.zeros(num_blocks, block_size, num_heads, half_hd,
                           dtype=torch.uint8, device="cuda")
    v_cache = torch.zeros(num_blocks, block_size, num_heads, half_hd,
                           dtype=torch.uint8, device="cuda")
    k_scales = torch.zeros(num_blocks, block_size, num_heads, num_groups,
                            dtype=torch.float16, device="cuda")
    v_scales = torch.zeros(num_blocks, block_size, num_heads, num_groups,
                            dtype=torch.float16, device="cuda")

    # Slot mapping: contiguous
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device="cuda")

    # Quantize
    reshape_and_cache_int4(
        key, value, k_cache, v_cache, k_scales, v_scales,
        slot_mapping, k_zeros=None, v_zeros=None, asymmetric=False,
    )

    # Manual dequantize and check error
    errors_k = []
    errors_v = []
    for t in range(num_tokens):
        block_idx = t // block_size
        block_offset = t % block_size
        for h in range(num_heads):
            packed = k_cache[block_idx, block_offset, h].cpu()
            scales = k_scales[block_idx, block_offset, h].cpu().float()

            low = (packed & 0x0F).to(torch.int8) - 8
            high = ((packed >> 4) & 0x0F).to(torch.int8) - 8

            full = torch.empty(head_size, dtype=torch.float32)
            full[0::2] = low.float()
            full[1::2] = high.float()

            for g in range(num_groups):
                start = g * GROUP_SIZE
                end = start + GROUP_SIZE
                full[start:end] *= scales[g].item()

            orig = key[t, h].cpu().float()
            err = (full - orig).abs().max().item()
            errors_k.append(err)

    max_err_k = max(errors_k)
    mean_err_k = sum(errors_k) / len(errors_k)
    log(f"  K roundtrip: max_err={max_err_k:.6f}, mean_err={mean_err_k:.6f}")

    # Check V similarly
    for t in range(num_tokens):
        block_idx = t // block_size
        block_offset = t % block_size
        for h in range(num_heads):
            packed = v_cache[block_idx, block_offset, h].cpu()
            scales = v_scales[block_idx, block_offset, h].cpu().float()

            low = (packed & 0x0F).to(torch.int8) - 8
            high = ((packed >> 4) & 0x0F).to(torch.int8) - 8

            full = torch.empty(head_size, dtype=torch.float32)
            full[0::2] = low.float()
            full[1::2] = high.float()

            for g in range(num_groups):
                start = g * GROUP_SIZE
                end = start + GROUP_SIZE
                full[start:end] *= scales[g].item()

            orig = value[t, h].cpu().float()
            err = (full - orig).abs().max().item()
            errors_v.append(err)

    max_err_v = max(errors_v)
    mean_err_v = sum(errors_v) / len(errors_v)
    log(f"  V roundtrip: max_err={max_err_v:.6f}, mean_err={mean_err_v:.6f}")

    return {
        "test": "roundtrip",
        "k_max_err": max_err_k,
        "k_mean_err": mean_err_k,
        "v_max_err": max_err_v,
        "v_mean_err": mean_err_v,
        "pass": max_err_k < 1.0 and max_err_v < 1.0,
    }


def test_decode_kernel_vs_fp16_sdpa():
    """Test: compare fused INT4 decode output to FP16 SDPA reference."""
    from vllm.v1.attention.backends.fused_int4 import (
        reshape_and_cache_int4,
        fused_int4_decode,
        GROUP_SIZE,
    )

    log("=== TEST 2: Fused decode kernel vs FP16 SDPA ===")

    num_seqs = 1
    num_heads = 28  # Qwen2.5-7B has 28 Q heads
    num_kv_heads = 4
    head_size = 128
    block_size = 16
    half_hd = head_size // 2
    num_groups = head_size // GROUP_SIZE
    n_rep = num_heads // num_kv_heads

    # Test at different sequence lengths around MSL=48
    results = []
    for seq_len in [16, 32, 47, 48, 49, 64, 96, 128]:
        num_tokens = seq_len
        num_blocks = (num_tokens + block_size - 1) // block_size

        torch.manual_seed(seq_len)
        # Simulate KV from the model
        key_full = torch.randn(num_tokens, num_kv_heads, head_size,
                                dtype=torch.float16, device="cuda")
        value_full = torch.randn(num_tokens, num_kv_heads, head_size,
                                  dtype=torch.float16, device="cuda")
        query = torch.randn(num_seqs, num_heads, head_size,
                             dtype=torch.float16, device="cuda")

        # --- FP16 reference ---
        # Expand KV for GQA
        k_expanded = key_full.unsqueeze(0).repeat_interleave(n_rep, dim=2)
        v_expanded = value_full.unsqueeze(0).repeat_interleave(n_rep, dim=2)
        # [1, num_heads, seq_len, head_size]
        k_t = k_expanded.permute(0, 2, 1, 3)
        v_t = v_expanded.permute(0, 2, 1, 3)
        q_t = query.unsqueeze(2)  # [1, num_heads, 1, head_size]

        ref_out = torch.nn.functional.scaled_dot_product_attention(
            q_t, k_t, v_t, is_causal=False,
            scale=1.0 / (head_size ** 0.5),
        )
        ref_out = ref_out.squeeze(2)  # [1, num_heads, head_size]

        # --- Fused INT4 path ---
        k_cache = torch.zeros(num_blocks, block_size, num_kv_heads, half_hd,
                               dtype=torch.uint8, device="cuda")
        v_cache = torch.zeros(num_blocks, block_size, num_kv_heads, half_hd,
                               dtype=torch.uint8, device="cuda")
        k_scales = torch.zeros(num_blocks, block_size, num_kv_heads, num_groups,
                                dtype=torch.float16, device="cuda")
        v_scales = torch.zeros(num_blocks, block_size, num_kv_heads, num_groups,
                                dtype=torch.float16, device="cuda")
        slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device="cuda")

        reshape_and_cache_int4(
            key_full, value_full, k_cache, v_cache, k_scales, v_scales,
            slot_mapping, k_zeros=None, v_zeros=None, asymmetric=False,
        )

        block_table = torch.arange(num_blocks, dtype=torch.int32,
                                    device="cuda").unsqueeze(0)
        seq_lens_t = torch.tensor([seq_len], dtype=torch.int32, device="cuda")

        fused_out = fused_int4_decode(
            query, k_cache, v_cache, k_scales, v_scales,
            block_table, seq_lens_t,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            block_n=64,
        )

        # Compare
        diff = (fused_out.float() - ref_out.float()).abs()
        max_err = diff.max().item()
        mean_err = diff.mean().item()
        # Check for NaN/Inf
        has_nan = torch.isnan(fused_out).any().item()
        has_inf = torch.isinf(fused_out).any().item()

        # Cosine similarity per head
        cos_sims = []
        for h in range(num_heads):
            r = ref_out[0, h].float()
            f = fused_out[0, h].float()
            cos = torch.nn.functional.cosine_similarity(
                r.unsqueeze(0), f.unsqueeze(0)).item()
            cos_sims.append(cos)
        min_cos = min(cos_sims)
        mean_cos = sum(cos_sims) / len(cos_sims)

        status = "PASS" if (max_err < 1.0 and min_cos > 0.9
                            and not has_nan and not has_inf) else "FAIL"

        log(f"  seq_len={seq_len:4d}: max_err={max_err:.4f}, "
            f"mean_err={mean_err:.6f}, "
            f"min_cos={min_cos:.4f}, mean_cos={mean_cos:.4f}, "
            f"nan={has_nan}, inf={has_inf} [{status}]")

        results.append({
            "seq_len": seq_len,
            "max_err": max_err,
            "mean_err": mean_err,
            "min_cos_sim": min_cos,
            "mean_cos_sim": mean_cos,
            "has_nan": has_nan,
            "has_inf": has_inf,
            "status": status,
        })

    return {"test": "decode_vs_sdpa", "results": results}


def test_real_model_kv():
    """Test: use real model to generate K/V, then compare decode paths."""
    log("=== TEST 3: Real model K/V capture and decode comparison ===")

    from vllm import LLM, SamplingParams

    prompts = [
        "What is 2 + 2?",
        ("The quick brown fox jumps over the lazy dog. " * 3 +
         "Summarize the above."),
        ("In the year 2024, artificial intelligence continued to advance "
         "at a rapid pace. Large language models became more capable and "
         "efficient. What is the main topic?"),
    ]

    log("  Loading FP16 model...")
    llm_fp16 = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        dtype="float16",
        kv_cache_dtype="auto",
        max_model_len=2048,
        gpu_memory_utilization=0.8,
        disable_log_stats=True,
        enforce_eager=True,
        max_num_seqs=1,
    )
    sp = SamplingParams(max_tokens=30, temperature=0.0)

    fp16_results = []
    for p in prompts:
        out = llm_fp16.generate([p], sp)[0].outputs[0]
        fp16_results.append({
            "prompt": p[:60],
            "ids": list(out.token_ids),
            "text": out.text,
        })
        log(f"    FP16: {p[:40]!r} -> {out.text[:60]!r}")

    del llm_fp16
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(3)

    log("  Loading INT4 fused model...")
    llm_int4 = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        dtype="float16",
        kv_cache_dtype="int4_fused",
        max_model_len=2048,
        gpu_memory_utilization=0.8,
        disable_log_stats=True,
        enforce_eager=True,
        max_num_seqs=1,
    )

    int4_results = []
    for p in prompts:
        out = llm_int4.generate([p], sp)[0].outputs[0]
        int4_results.append({
            "prompt": p[:60],
            "ids": list(out.token_ids),
            "text": out.text,
        })
        log(f"    INT4: {p[:40]!r} -> {out.text[:60]!r}")

    del llm_int4
    gc.collect()
    torch.cuda.empty_cache()

    # Compare
    comparisons = []
    for f, i in zip(fp16_results, int4_results):
        match = f["ids"] == i["ids"]
        prefix_len = 0
        for a, b in zip(f["ids"], i["ids"]):
            if a == b:
                prefix_len += 1
            else:
                break
        comparisons.append({
            "prompt": f["prompt"],
            "match": match,
            "prefix_len": prefix_len,
            "fp16_text": f["text"][:80],
            "int4_text": i["text"][:80],
        })

    return {"test": "real_model_kv", "comparisons": comparisons}


def main():
    artifact_dir = "/tmp/a100-kernel-diag"
    os.makedirs(artifact_dir, exist_ok=True)

    results = {}

    try:
        results["roundtrip"] = test_roundtrip_correctness()
    except Exception as e:
        log(f"ERROR in roundtrip test: {e}")
        import traceback
        results["roundtrip"] = {"error": str(e),
                                 "tb": traceback.format_exc()}

    try:
        results["decode_vs_sdpa"] = test_decode_kernel_vs_fp16_sdpa()
    except Exception as e:
        log(f"ERROR in decode test: {e}")
        import traceback
        results["decode_vs_sdpa"] = {"error": str(e),
                                      "tb": traceback.format_exc()}

    try:
        results["real_model"] = test_real_model_kv()
    except Exception as e:
        log(f"ERROR in real model test: {e}")
        import traceback
        results["real_model"] = {"error": str(e),
                                  "tb": traceback.format_exc()}

    # Save
    out_path = os.path.join(artifact_dir, "kernel_diag.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\nResults saved to {out_path}")
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
