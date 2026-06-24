#!/usr/bin/env python3
"""Instrument the fused INT4 decode path to capture actual cache state.

Monkey-patches fused_int4_decode to capture its inputs and compare
the decode output to FP16 SDPA on the SAME actual cached data.
"""

import gc
import json
import os
import sys
import time
import torch

os.environ["VLLM_FUSED_INT4_MIN_SEQ_LEN"] = "1"  # Force fused for ALL seqs

import vllm.v1.attention.backends.fused_int4 as fused_mod

# Store captures
CAPTURES = []
CAPTURE_LIMIT = 5  # Only capture first N decode calls

_orig_fused_decode = fused_mod.fused_int4_decode


def instrumented_fused_decode(
    query, key_cache, value_cache, k_scales, v_scales,
    block_table, seq_lens, num_kv_heads, head_size,
    block_n=64, k_zeros=None, v_zeros=None, asymmetric=False,
):
    """Wrapper that captures inputs and compares to FP16 reference."""
    # Call original
    result = _orig_fused_decode(
        query, key_cache, value_cache, k_scales, v_scales,
        block_table, seq_lens, num_kv_heads, head_size,
        block_n, k_zeros, v_zeros, asymmetric,
    )

    if len(CAPTURES) < CAPTURE_LIMIT:
        seq_len_val = seq_lens[0].item()
        num_seqs = query.shape[0]
        num_heads = query.shape[1]
        n_rep = num_heads // num_kv_heads
        block_size = key_cache.shape[1]
        half_hd = head_size // 2
        num_groups = head_size // fused_mod.GROUP_SIZE
        scale = 1.0 / (head_size ** 0.5)

        # Dequantize the cached K/V to FP16 and compute SDPA reference
        # Read from the actual INT4 cache for this sequence
        blocks_needed = (seq_len_val + block_size - 1) // block_size
        k_dequant_list = []
        v_dequant_list = []

        for b in range(blocks_needed):
            phys_block = block_table[0, b].item()
            tokens_in_block = min(block_size, seq_len_val - b * block_size)

            for t in range(tokens_in_block):
                k_heads = []
                v_heads = []
                for h in range(num_kv_heads):
                    # Dequant K
                    k_packed = key_cache[phys_block, t, h].cpu()
                    k_sc = k_scales[phys_block, t, h].cpu().float()
                    k_low = ((k_packed & 0x0F).to(torch.int8) - 8).float()
                    k_high = (((k_packed >> 4) & 0x0F).to(torch.int8) - 8).float()
                    k_full = torch.empty(head_size, dtype=torch.float32)
                    k_full[0::2] = k_low
                    k_full[1::2] = k_high
                    for g in range(num_groups):
                        s = g * fused_mod.GROUP_SIZE
                        e = s + fused_mod.GROUP_SIZE
                        k_full[s:e] *= k_sc[g].item()
                    k_heads.append(k_full)

                    # Dequant V
                    v_packed = value_cache[phys_block, t, h].cpu()
                    v_sc = v_scales[phys_block, t, h].cpu().float()
                    v_low = ((v_packed & 0x0F).to(torch.int8) - 8).float()
                    v_high = (((v_packed >> 4) & 0x0F).to(torch.int8) - 8).float()
                    v_full = torch.empty(head_size, dtype=torch.float32)
                    v_full[0::2] = v_low
                    v_full[1::2] = v_high
                    for g in range(num_groups):
                        s = g * fused_mod.GROUP_SIZE
                        e = s + fused_mod.GROUP_SIZE
                        v_full[s:e] *= v_sc[g].item()
                    v_heads.append(v_full)

                k_dequant_list.append(torch.stack(k_heads))
                v_dequant_list.append(torch.stack(v_heads))

        if len(k_dequant_list) == 0:
            return result

        # Compute FP16 SDPA on dequantized data
        k_dequant = torch.stack(k_dequant_list).to(device="cuda", dtype=torch.float16)
        v_dequant = torch.stack(v_dequant_list).to(device="cuda", dtype=torch.float16)

        # k_dequant: [seq_len, kv_heads, head_size]
        # Expand for GQA
        if n_rep > 1:
            k_dequant = k_dequant.repeat_interleave(n_rep, dim=1)
            v_dequant = v_dequant.repeat_interleave(n_rep, dim=1)

        q = query[0:1].unsqueeze(2)  # [1, heads, 1, hd]
        k_t = k_dequant.unsqueeze(0).permute(0, 2, 1, 3)  # [1, heads, seq, hd]
        v_t = v_dequant.unsqueeze(0).permute(0, 2, 1, 3)

        ref = torch.nn.functional.scaled_dot_product_attention(
            q, k_t, v_t, is_causal=False, scale=scale,
        ).squeeze(2)  # [1, heads, hd]

        # Compare
        fused_out = result[0:1]
        diff = (fused_out.float() - ref.float()).abs()
        max_err = diff.max().item()
        mean_err = diff.mean().item()
        has_nan = torch.isnan(fused_out).any().item()

        # Per-head cosine sim
        min_cos = 1.0
        for hh in range(num_heads):
            cos = torch.nn.functional.cosine_similarity(
                ref[0, hh:hh+1].float(), fused_out[0, hh:hh+1].float()
            ).item()
            min_cos = min(min_cos, cos)

        # Check scale stats
        k_scale_stats = {
            "min": k_scales[:blocks_needed].min().item(),
            "max": k_scales[:blocks_needed].max().item(),
            "mean": k_scales[:blocks_needed].float().mean().item(),
            "has_zero": (k_scales[:blocks_needed] == 0).any().item(),
            "has_inf": torch.isinf(k_scales[:blocks_needed]).any().item(),
            "has_nan": torch.isnan(k_scales[:blocks_needed]).any().item(),
        }
        v_scale_stats = {
            "min": v_scales[:blocks_needed].min().item(),
            "max": v_scales[:blocks_needed].max().item(),
            "mean": v_scales[:blocks_needed].float().mean().item(),
        }

        capture = {
            "call_idx": len(CAPTURES),
            "seq_len": seq_len_val,
            "num_seqs": num_seqs,
            "blocks_needed": blocks_needed,
            "max_err_vs_dequant_sdpa": max_err,
            "mean_err_vs_dequant_sdpa": mean_err,
            "min_cos_vs_dequant_sdpa": min_cos,
            "has_nan": has_nan,
            "k_scale_stats": k_scale_stats,
            "v_scale_stats": v_scale_stats,
            "block_table_sample": block_table[0, :blocks_needed].tolist(),
            "query_shape": list(query.shape),
            "key_cache_shape": list(key_cache.shape),
            "key_cache_strides": list(key_cache.stride()),
        }

        print(f"[INSTRUMENT] call={len(CAPTURES)} seq_len={seq_len_val} "
              f"max_err={max_err:.4f} min_cos={min_cos:.4f} "
              f"k_scale_range=[{k_scale_stats['min']:.4f},{k_scale_stats['max']:.4f}]",
              file=sys.stderr, flush=True)
        CAPTURES.append(capture)

    return result

# Monkey-patch
fused_mod.fused_int4_decode = instrumented_fused_decode

# Now run the model
from vllm import LLM, SamplingParams

print("Loading INT4 fused model with MSL=1 (fused decode from first token)...",
      file=sys.stderr)

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct", dtype="float16",
    kv_cache_dtype="int4_fused", max_model_len=2048,
    gpu_memory_utilization=0.8, disable_log_stats=True,
    enforce_eager=True, max_num_seqs=1,
)

sp = SamplingParams(max_tokens=10, temperature=0.0)

# Use a single long prompt to trigger fused decode
prompt = ("The quick brown fox jumps over the lazy dog. " * 3 +
          "Summarize the above.")

out = llm.generate([prompt], sp)[0].outputs[0]
print(f"\nGenerated: {out.text!r}", file=sys.stderr)

del llm; gc.collect(); torch.cuda.empty_cache()

# Save captures
out_path = "/tmp/a100-decode-instrument.json"
with open(out_path, "w") as f:
    json.dump(CAPTURES, f, indent=2, default=str)

print(f"\nCaptures saved to {out_path}", file=sys.stderr)
print(json.dumps(CAPTURES, indent=2, default=str))
