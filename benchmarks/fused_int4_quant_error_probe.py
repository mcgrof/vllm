#!/usr/bin/env python3
"""Measure the attention output error from INT4 quantization vs FP16.

Tests: FP16 SDPA (baseline) vs fused INT4 decode (quantized K/V).
This measures the actual error users would see, not the kernel accuracy.
"""

import os
import sys
import json
import torch
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm.v1.attention.backends.fused_int4 import (
    reshape_and_cache_int4,
    fused_int4_decode,
    GROUP_SIZE,
)


def reference_fp16_sdpa(query, keys, values, scale, num_heads, num_kv_heads):
    """Reference FP16 SDPA. Same interface as fused_int4_decode output."""
    n_rep = num_heads // num_kv_heads
    if n_rep > 1:
        keys = keys.repeat_interleave(n_rep, dim=1)
        values = values.repeat_interleave(n_rep, dim=1)

    q = query.transpose(0, 1).float()   # [num_heads, 1, head_size]
    k = keys.transpose(0, 1).float()    # [num_heads, seq_len, head_size]
    v = values.transpose(0, 1).float()  # [num_heads, seq_len, head_size]

    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_weights = torch.softmax(attn_weights, dim=-1)
    out = torch.matmul(attn_weights, v)
    return out.transpose(0, 1).to(torch.float16)


def test_quant_error(
    seq_len, num_kv_heads=4, num_heads=28, head_size=128, block_size=16,
    device="cuda", seed=42, use_real_scale=True,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    half_hd = head_size // 2
    num_groups = head_size // GROUP_SIZE
    num_blocks = (seq_len + block_size - 1) // block_size + 2

    # Allocate with FP16 view (like real vLLM)
    kv_fp16 = torch.zeros(2, num_blocks, block_size, num_kv_heads, head_size,
                          dtype=torch.float16, device=device)
    kv_uint8 = kv_fp16.view(torch.uint8)
    key_cache = kv_uint8[0][..., :half_hd]
    value_cache = kv_uint8[1][..., :half_hd]

    k_scales = torch.zeros(num_blocks, block_size, num_kv_heads, num_groups,
                           dtype=torch.float16, device=device)
    v_scales = torch.zeros(num_blocks, block_size, num_kv_heads, num_groups,
                           dtype=torch.float16, device=device)

    # Generate K/V with realistic magnitude (like model outputs)
    if use_real_scale:
        # Some heads have large outliers (common in LLMs)
        all_keys = torch.randn(seq_len, num_kv_heads, head_size,
                               dtype=torch.float16, device=device)
        all_values = torch.randn(seq_len, num_kv_heads, head_size,
                                 dtype=torch.float16, device=device)
        # Add outliers to a few dimensions
        all_keys[:, :, 0] *= 5
        all_keys[:, :, 64] *= 3
        all_values[:, :, 0] *= 5
    else:
        all_keys = torch.randn(seq_len, num_kv_heads, head_size,
                               dtype=torch.float16, device=device) * 0.5
        all_values = torch.randn(seq_len, num_kv_heads, head_size,
                                 dtype=torch.float16, device=device) * 0.5

    # Write to INT4 cache
    slot_mapping = torch.arange(seq_len, dtype=torch.int64, device=device)
    reshape_and_cache_int4(
        all_keys, all_values,
        key_cache, value_cache,
        k_scales, v_scales,
        slot_mapping,
    )

    # Query
    query = torch.randn(1, num_heads, head_size,
                        dtype=torch.float16, device=device)

    block_table = torch.arange(num_blocks, dtype=torch.int32,
                               device=device).unsqueeze(0)
    seq_lens_t = torch.tensor([seq_len], dtype=torch.int32, device=device)

    # Fused INT4 decode
    fused_out = fused_int4_decode(
        query, key_cache, value_cache,
        k_scales, v_scales,
        block_table, seq_lens_t,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
    ).view(1, num_heads, head_size)

    # FP16 reference (what baseline FlashAttn would produce)
    scale = 1.0 / math.sqrt(head_size)
    ref_out = reference_fp16_sdpa(
        query, all_keys, all_values, scale, num_heads, num_kv_heads
    )

    torch.cuda.synchronize()

    diff = (fused_out.float() - ref_out.float()).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()

    # Cosine similarity per head
    cos_sim = torch.nn.functional.cosine_similarity(
        fused_out.view(num_heads, head_size).float(),
        ref_out.view(num_heads, head_size).float(),
        dim=1,
    )
    min_cos = cos_sim.min().item()
    mean_cos = cos_sim.mean().item()

    return {
        "seq_len": seq_len,
        "max_abs_err": round(max_err, 6),
        "mean_abs_err": round(mean_err, 6),
        "min_cosine_sim": round(min_cos, 6),
        "mean_cosine_sim": round(mean_cos, 6),
    }


def main():
    device = "cuda"
    results = []

    print("=== INT4 Quantization Error vs FP16 Baseline ===", file=sys.stderr)
    print("(with outlier features, like real LLM K/V)", file=sys.stderr)

    for sl in [1, 2, 3, 4, 8, 15, 16, 17, 32, 64, 128]:
        r = test_quant_error(sl, device=device, use_real_scale=True)
        results.append(r)
        status = "OK" if r["min_cosine_sim"] > 0.95 else "WARN"
        print(f"  seq_len={sl:3d}: max_err={r['max_abs_err']:.4f} "
              f"mean_err={r['mean_abs_err']:.4f} "
              f"min_cos={r['min_cosine_sim']:.4f} "
              f"mean_cos={r['mean_cosine_sim']:.4f} [{status}]",
              file=sys.stderr)

    print("\n=== Without outliers (uniform scale) ===", file=sys.stderr)
    for sl in [1, 2, 3, 4, 8, 16, 32, 64]:
        r = test_quant_error(sl, device=device, use_real_scale=False)
        r["variant"] = "uniform"
        results.append(r)
        status = "OK" if r["min_cosine_sim"] > 0.95 else "WARN"
        print(f"  seq_len={sl:3d}: max_err={r['max_abs_err']:.4f} "
              f"mean_err={r['mean_abs_err']:.4f} "
              f"min_cos={r['min_cosine_sim']:.4f} "
              f"mean_cos={r['mean_cosine_sim']:.4f} [{status}]",
              file=sys.stderr)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
