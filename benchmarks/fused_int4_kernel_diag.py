#!/usr/bin/env python3
"""Kernel-level diagnostic for fused INT4 decode correctness.

This script bypasses vLLM's serving infrastructure and directly tests
the quantize + fused-decode attention kernel pipeline against a torch
SDPA reference, isolating the exact failure mechanism.

We test: varying seq_lens with fixed block_size=16 to find where
the fused kernel diverges from reference.
"""

import os
import sys
import json
import torch
import math

# Ensure we can import the backend
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm.v1.attention.backends.fused_int4 import (
    reshape_and_cache_int4,
    fused_int4_decode,
    _cpu_dequant_slot,
    GROUP_SIZE,
)


def reference_sdpa(query, keys, values, scale):
    """Reference scaled dot-product attention using torch.

    query: [1, num_heads, head_size]
    keys:  [seq_len, num_kv_heads, head_size]
    values: [seq_len, num_kv_heads, head_size]

    Returns: [1, num_heads, head_size]
    """
    num_heads = query.shape[1]
    num_kv_heads = keys.shape[1]
    n_rep = num_heads // num_kv_heads

    # GQA expansion
    if n_rep > 1:
        keys = keys.repeat_interleave(n_rep, dim=1)
        values = values.repeat_interleave(n_rep, dim=1)

    # [1, num_heads, head_size] x [seq_len, num_heads, head_size]^T
    # -> [num_heads, 1, seq_len]
    q = query.transpose(0, 1)   # [num_heads, 1, head_size]
    k = keys.transpose(0, 1)    # [num_heads, seq_len, head_size]
    v = values.transpose(0, 1)  # [num_heads, seq_len, head_size]

    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_weights = torch.softmax(attn_weights, dim=-1)
    out = torch.matmul(attn_weights, v)  # [num_heads, 1, head_size]
    return out.transpose(0, 1)  # [1, num_heads, head_size]


def test_fused_decode_kernel(
    seq_len: int,
    num_kv_heads: int = 4,
    num_heads: int = 28,
    head_size: int = 128,
    block_size: int = 16,
    device: str = "cuda",
    seed: int = 42,
):
    """Test fused INT4 decode kernel against reference for given seq_len.

    Simulates: prefill wrote seq_len-1 tokens, then 1 decode token.
    The decode kernel reads all seq_len tokens from INT4 cache.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    half_hd = head_size // 2
    num_groups = head_size // GROUP_SIZE

    # Number of blocks needed
    num_blocks_needed = (seq_len + block_size - 1) // block_size
    total_blocks = num_blocks_needed + 2  # padding

    # Allocate INT4 cache (packed uint8) and scales
    # Use contiguous uint8 tensors directly (no FP16 view dance)
    key_cache = torch.zeros(total_blocks, block_size, num_kv_heads, half_hd,
                            dtype=torch.uint8, device=device)
    value_cache = torch.zeros(total_blocks, block_size, num_kv_heads, half_hd,
                              dtype=torch.uint8, device=device)
    k_scales = torch.zeros(total_blocks, block_size, num_kv_heads, num_groups,
                           dtype=torch.float16, device=device)
    v_scales = torch.zeros(total_blocks, block_size, num_kv_heads, num_groups,
                           dtype=torch.float16, device=device)

    # Generate random K/V for all seq_len tokens
    all_keys = torch.randn(seq_len, num_kv_heads, head_size,
                           dtype=torch.float16, device=device) * 0.5
    all_values = torch.randn(seq_len, num_kv_heads, head_size,
                             dtype=torch.float16, device=device) * 0.5

    # Write all tokens to cache using the quantization kernel
    # Map slots: token i -> slot i (all in block 0, then block 1, etc.)
    slot_mapping = torch.arange(seq_len, dtype=torch.int64, device=device)

    reshape_and_cache_int4(
        all_keys, all_values,
        key_cache, value_cache,
        k_scales, v_scales,
        slot_mapping,
    )

    # Now also test writing in two phases: prefill then decode
    key_cache_2 = torch.zeros_like(key_cache)
    value_cache_2 = torch.zeros_like(value_cache)
    k_scales_2 = torch.zeros_like(k_scales)
    v_scales_2 = torch.zeros_like(v_scales)

    # Phase 1: write prefill tokens (0..seq_len-2)
    if seq_len > 1:
        prefill_slots = torch.arange(seq_len - 1, dtype=torch.int64,
                                     device=device)
        reshape_and_cache_int4(
            all_keys[:seq_len-1], all_values[:seq_len-1],
            key_cache_2, value_cache_2,
            k_scales_2, v_scales_2,
            prefill_slots,
        )

    # Phase 2: write decode token (seq_len-1)
    decode_slot = torch.tensor([seq_len - 1], dtype=torch.int64,
                               device=device)
    reshape_and_cache_int4(
        all_keys[seq_len-1:seq_len], all_values[seq_len-1:seq_len],
        key_cache_2, value_cache_2,
        k_scales_2, v_scales_2,
        decode_slot,
    )

    # Verify cache contents match between 1-shot and 2-phase writes
    torch.cuda.synchronize()
    cache_match = torch.equal(key_cache, key_cache_2) and \
                  torch.equal(value_cache, value_cache_2) and \
                  torch.equal(k_scales, k_scales_2) and \
                  torch.equal(v_scales, v_scales_2)

    # Build block table: identity mapping (logical block i = physical block i)
    block_table = torch.arange(total_blocks, dtype=torch.int32,
                               device=device).unsqueeze(0)  # [1, total_blocks]

    # Query for the decode step
    query = torch.randn(1, num_heads, head_size,
                        dtype=torch.float16, device=device) * 0.5
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)

    # Run fused INT4 decode kernel
    fused_output = fused_int4_decode(
        query, key_cache, value_cache,
        k_scales, v_scales,
        block_table, seq_lens,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
    )

    # Also run with 2-phase cache
    fused_output_2 = fused_int4_decode(
        query, key_cache_2, value_cache_2,
        k_scales_2, v_scales_2,
        block_table, seq_lens,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
    )

    torch.cuda.synchronize()

    # Reference: dequantize from cache and compute SDPA
    # Dequantize all tokens from the INT4 cache
    dequant_keys = []
    dequant_values = []
    for t in range(seq_len):
        block_idx = t // block_size
        block_offset = t % block_size
        k_recon_full = []
        v_recon_full = []
        for h in range(num_kv_heads):
            k_vals = _cpu_dequant_slot(
                key_cache.cpu(), k_scales.cpu(),
                block_idx, block_offset, h, head_size, head_size)
            v_vals = _cpu_dequant_slot(
                value_cache.cpu(), v_scales.cpu(),
                block_idx, block_offset, h, head_size, head_size)
            k_recon_full.append(k_vals)
            v_recon_full.append(v_vals)
        dequant_keys.append(k_recon_full)
        dequant_values.append(v_recon_full)

    # Shape: [seq_len, num_kv_heads, head_size]
    dequant_k = torch.tensor(dequant_keys, dtype=torch.float32,
                             device="cpu").to(torch.float16).to(device)
    dequant_v = torch.tensor(dequant_values, dtype=torch.float32,
                             device="cpu").to(torch.float16).to(device)

    scale = 1.0 / math.sqrt(head_size)
    ref_output = reference_sdpa(query.float(), dequant_k.float(),
                                dequant_v.float(), scale)
    ref_output = ref_output.to(torch.float16)

    # Compare
    fused_flat = fused_output.view(1, num_heads, head_size)
    fused_flat_2 = fused_output_2.view(1, num_heads, head_size)

    max_abs_err = (fused_flat.float() - ref_output.float()).abs().max().item()
    max_abs_err_2 = (fused_flat_2.float() - ref_output.float()).abs().max().item()
    mean_abs_err = (fused_flat.float() - ref_output.float()).abs().mean().item()

    # Per-head max error
    per_head_err = (fused_flat.float() - ref_output.float()).abs()  # [1, H, D]
    per_head_max = per_head_err.squeeze(0).max(dim=-1).values  # [H]

    # Also compare fused vs fused_2 to see if two-phase writing matters
    fused_diff = (fused_flat.float() - fused_flat_2.float()).abs().max().item()

    return {
        "seq_len": seq_len,
        "cache_match": cache_match,
        "max_abs_err": round(max_abs_err, 6),
        "max_abs_err_2phase": round(max_abs_err_2, 6),
        "mean_abs_err": round(mean_abs_err, 6),
        "fused_1shot_vs_2phase_diff": round(fused_diff, 6),
        "per_head_max_err_first4": [round(x.item(), 6) for x in per_head_max[:4]],
        "fused_out_first8": fused_flat[0, 0, :8].float().tolist(),
        "ref_out_first8": ref_output[0, 0, :8].float().tolist(),
    }


def test_with_fp16_stride_view(
    seq_len: int,
    num_kv_heads: int = 4,
    num_heads: int = 28,
    head_size: int = 128,
    block_size: int = 16,
    device: str = "cuda",
    seed: int = 42,
):
    """Same test but using the FP16-view-then-uint8 cache layout
    (matching the real vLLM allocator path) to see if stride matters."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    half_hd = head_size // 2
    num_groups = head_size // GROUP_SIZE
    num_blocks_needed = (seq_len + block_size - 1) // block_size
    total_blocks = num_blocks_needed + 2

    # Allocate like vLLM: FP16 tensor then view as uint8
    kv_fp16 = torch.zeros(2, total_blocks, block_size, num_kv_heads, head_size,
                          dtype=torch.float16, device=device)
    kv_uint8 = kv_fp16.view(torch.uint8)
    key_cache = kv_uint8[0][..., :half_hd]
    value_cache = kv_uint8[1][..., :half_hd]

    # These have sparse strides!
    print(f"  [stride_view] key_cache shape={key_cache.shape} "
          f"strides={key_cache.stride()}", file=sys.stderr)

    k_scales = torch.zeros(total_blocks, block_size, num_kv_heads, num_groups,
                           dtype=torch.float16, device=device)
    v_scales = torch.zeros(total_blocks, block_size, num_kv_heads, num_groups,
                           dtype=torch.float16, device=device)

    all_keys = torch.randn(seq_len, num_kv_heads, head_size,
                           dtype=torch.float16, device=device) * 0.5
    all_values = torch.randn(seq_len, num_kv_heads, head_size,
                             dtype=torch.float16, device=device) * 0.5

    slot_mapping = torch.arange(seq_len, dtype=torch.int64, device=device)

    reshape_and_cache_int4(
        all_keys, all_values,
        key_cache, value_cache,
        k_scales, v_scales,
        slot_mapping,
    )

    block_table = torch.arange(total_blocks, dtype=torch.int32,
                               device=device).unsqueeze(0)
    query = torch.randn(1, num_heads, head_size,
                        dtype=torch.float16, device=device) * 0.5
    seq_lens_t = torch.tensor([seq_len], dtype=torch.int32, device=device)

    fused_output = fused_int4_decode(
        query, key_cache, value_cache,
        k_scales, v_scales,
        block_table, seq_lens_t,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
    )

    torch.cuda.synchronize()

    # Reference via CPU dequant
    dequant_keys = []
    dequant_values = []
    for t in range(seq_len):
        block_idx = t // block_size
        block_offset = t % block_size
        k_recon_full = []
        v_recon_full = []
        for h in range(num_kv_heads):
            # NOTE: _cpu_dequant_slot assumes contiguous cache
            # We need to handle sparse strides manually
            packed = key_cache[block_idx, block_offset, h].detach().cpu()
            sc = k_scales[block_idx, block_offset, h].detach().cpu().float()
            low = (packed & 0x0F).to(torch.int8) - 8
            high = ((packed >> 4) & 0x0F).to(torch.int8) - 8
            hd = packed.shape[0] * 2
            full = torch.empty(hd, dtype=torch.float32)
            full[0::2] = low.float()
            full[1::2] = high.float()
            num_g = hd // GROUP_SIZE
            full = full.reshape(num_g, GROUP_SIZE)
            full = full * sc[:num_g].unsqueeze(1)
            full = full.reshape(hd)
            k_vals = full[:head_size].tolist()

            packed_v = value_cache[block_idx, block_offset, h].detach().cpu()
            sc_v = v_scales[block_idx, block_offset, h].detach().cpu().float()
            low_v = (packed_v & 0x0F).to(torch.int8) - 8
            high_v = ((packed_v >> 4) & 0x0F).to(torch.int8) - 8
            hd_v = packed_v.shape[0] * 2
            full_v = torch.empty(hd_v, dtype=torch.float32)
            full_v[0::2] = low_v.float()
            full_v[1::2] = high_v.float()
            num_g_v = hd_v // GROUP_SIZE
            full_v = full_v.reshape(num_g_v, GROUP_SIZE)
            full_v = full_v * sc_v[:num_g_v].unsqueeze(1)
            full_v = full_v.reshape(hd_v)
            v_vals = full_v[:head_size].tolist()

            k_recon_full.append(k_vals)
            v_recon_full.append(v_vals)
        dequant_keys.append(k_recon_full)
        dequant_values.append(v_recon_full)

    dequant_k = torch.tensor(dequant_keys, dtype=torch.float32,
                             device="cpu").to(torch.float16).to(device)
    dequant_v = torch.tensor(dequant_values, dtype=torch.float32,
                             device="cpu").to(torch.float16).to(device)

    scale = 1.0 / math.sqrt(head_size)
    ref_output = reference_sdpa(query.float(), dequant_k.float(),
                                dequant_v.float(), scale)
    ref_output = ref_output.to(torch.float16)

    fused_flat = fused_output.view(1, num_heads, head_size)
    max_abs_err = (fused_flat.float() - ref_output.float()).abs().max().item()
    mean_abs_err = (fused_flat.float() - ref_output.float()).abs().mean().item()

    return {
        "seq_len": seq_len,
        "layout": "fp16_stride_view",
        "key_cache_strides": list(key_cache.stride()),
        "max_abs_err": round(max_abs_err, 6),
        "mean_abs_err": round(mean_abs_err, 6),
        "fused_out_first8": fused_flat[0, 0, :8].float().tolist(),
        "ref_out_first8": ref_output[0, 0, :8].float().tolist(),
    }


def main():
    device = "cuda"
    results = {"contiguous_cache": [], "fp16_stride_cache": []}

    test_seq_lens = [1, 2, 3, 4, 8, 15, 16, 17, 32, 33, 64, 65]

    print("=== Test 1: Contiguous uint8 cache (no stride gap) ===",
          file=sys.stderr)
    for sl in test_seq_lens:
        print(f"  seq_len={sl}...", file=sys.stderr, end="", flush=True)
        r = test_fused_decode_kernel(seq_len=sl, device=device)
        results["contiguous_cache"].append(r)
        status = "OK" if r["max_abs_err"] < 0.05 else "FAIL"
        print(f" max_err={r['max_abs_err']:.6f} "
              f"cache_match={r['cache_match']} [{status}]", file=sys.stderr)

    print("\n=== Test 2: FP16-stride-view cache (matching vLLM allocator) ===",
          file=sys.stderr)
    for sl in test_seq_lens:
        print(f"  seq_len={sl}...", file=sys.stderr, end="", flush=True)
        r = test_with_fp16_stride_view(seq_len=sl, device=device)
        results["fp16_stride_cache"].append(r)
        status = "OK" if r["max_abs_err"] < 0.05 else "FAIL"
        print(f" max_err={r['max_abs_err']:.6f} [{status}]",
              file=sys.stderr)

    json_str = json.dumps(results, indent=2)
    print(json_str)


if __name__ == "__main__":
    main()
