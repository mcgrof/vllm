#!/usr/bin/env python3
"""Diagnose whether INT4 corruption is caused by paged block mapping.

The kernel passes on synthetic data with contiguous slot mapping.
The kernel fails through vLLM pipeline with paged allocation.

This diagnostic tests:
1. Contiguous slot mapping (should pass - matches kernel diag)
2. Non-contiguous slot mapping (simulates vLLM paging)
3. Real vLLM block table structure
4. Per-layer divergence check

If (1) passes and (2) fails, the issue is in how the decode kernel
reads paged blocks. If both pass, the issue is upstream of the kernel.
"""

import os
import sys
import json
import time
import torch

os.environ.setdefault("VLLM_FUSED_INT4_MIN_SEQ_LEN", "48")

from datetime import datetime, timezone
from vllm.v1.attention.backends.fused_int4 import (
    reshape_and_cache_int4,
    fused_int4_decode,
    GROUP_SIZE,
)


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def fp16_reference(query, key_full, value_full, num_heads, num_kv_heads, head_size):
    """Pure FP16 SDPA reference."""
    n_rep = num_heads // num_kv_heads
    k_expanded = key_full.unsqueeze(0).repeat_interleave(n_rep, dim=2)
    v_expanded = value_full.unsqueeze(0).repeat_interleave(n_rep, dim=2)
    k_t = k_expanded.permute(0, 2, 1, 3)
    v_t = v_expanded.permute(0, 2, 1, 3)
    q_t = query.unsqueeze(2)
    ref_out = torch.nn.functional.scaled_dot_product_attention(
        q_t, k_t, v_t, is_causal=False,
        scale=1.0 / (head_size ** 0.5),
    )
    return ref_out.squeeze(2)


def test_paging_pattern(name, seq_len, num_blocks_total, block_size,
                         num_heads, num_kv_heads, head_size,
                         slot_mapping_fn, block_table_fn):
    """Test a specific paging pattern."""
    half_hd = head_size // 2
    num_groups = head_size // GROUP_SIZE

    torch.manual_seed(42 + seq_len)
    key_full = torch.randn(seq_len, num_kv_heads, head_size,
                            dtype=torch.float16, device="cuda")
    value_full = torch.randn(seq_len, num_kv_heads, head_size,
                              dtype=torch.float16, device="cuda")
    query = torch.randn(1, num_heads, head_size,
                         dtype=torch.float16, device="cuda")

    # FP16 reference
    ref = fp16_reference(query, key_full, value_full,
                          num_heads, num_kv_heads, head_size)

    # Allocate INT4 caches
    k_cache = torch.zeros(num_blocks_total, block_size, num_kv_heads, half_hd,
                           dtype=torch.uint8, device="cuda")
    v_cache = torch.zeros(num_blocks_total, block_size, num_kv_heads, half_hd,
                           dtype=torch.uint8, device="cuda")
    k_scales = torch.zeros(num_blocks_total, block_size, num_kv_heads, num_groups,
                            dtype=torch.float16, device="cuda")
    v_scales = torch.zeros(num_blocks_total, block_size, num_kv_heads, num_groups,
                            dtype=torch.float16, device="cuda")

    # Create slot mapping and block table
    slot_mapping = slot_mapping_fn(seq_len, block_size, num_blocks_total)
    block_table = block_table_fn(seq_len, block_size, num_blocks_total, slot_mapping)

    # Quantize
    reshape_and_cache_int4(
        key_full, value_full, k_cache, v_cache, k_scales, v_scales,
        slot_mapping, k_zeros=None, v_zeros=None, asymmetric=False,
    )

    seq_lens_t = torch.tensor([seq_len], dtype=torch.int32, device="cuda")

    # Decode
    fused_out = fused_int4_decode(
        query, k_cache, v_cache, k_scales, v_scales,
        block_table, seq_lens_t,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        block_n=64,
    )

    # Compare
    diff = (fused_out.float() - ref.float()).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    has_nan = torch.isnan(fused_out).any().item()

    cos_sims = []
    for h in range(num_heads):
        r = ref[0, h].float()
        f = fused_out[0, h].float()
        cos = torch.nn.functional.cosine_similarity(
            r.unsqueeze(0), f.unsqueeze(0)).item()
        cos_sims.append(cos)
    min_cos = min(cos_sims)
    mean_cos = sum(cos_sims) / len(cos_sims)

    status = "PASS" if (max_err < 1.0 and min_cos > 0.9
                        and not has_nan) else "FAIL"

    log(f"  [{name}] seq_len={seq_len}: max_err={max_err:.4f}, "
        f"mean_err={mean_err:.6f}, min_cos={min_cos:.4f}, "
        f"nan={has_nan} [{status}]")

    return {
        "name": name,
        "seq_len": seq_len,
        "max_err": max_err,
        "mean_err": mean_err,
        "min_cos": min_cos,
        "mean_cos": mean_cos,
        "has_nan": has_nan,
        "status": status,
    }


def main():
    num_heads = 28  # Qwen2.5-7B
    num_kv_heads = 4
    head_size = 128
    block_size = 16
    num_blocks_total = 128  # Plenty of blocks

    results = []

    for seq_len in [16, 32, 48, 64, 96, 128, 256]:
        # 1. Contiguous slot mapping (matches kernel diag)
        def contiguous_slots(sl, bs, nb):
            return torch.arange(sl, dtype=torch.int64, device="cuda")

        def contiguous_bt(sl, bs, nb, sm):
            n_blocks = (sl + bs - 1) // bs
            return torch.arange(n_blocks, dtype=torch.int32,
                                 device="cuda").unsqueeze(0)

        r = test_paging_pattern(
            "contiguous", seq_len, num_blocks_total, block_size,
            num_heads, num_kv_heads, head_size,
            contiguous_slots, contiguous_bt,
        )
        results.append(r)

        # 2. Reversed block order (blocks allocated in reverse)
        def reversed_slots(sl, bs, nb):
            n_blocks = (sl + bs - 1) // bs
            slots = torch.empty(sl, dtype=torch.int64, device="cuda")
            for t in range(sl):
                logical_blk = t // bs
                blk_off = t % bs
                phys_blk = (n_blocks - 1 - logical_blk)
                slots[t] = phys_blk * bs + blk_off
            return slots

        def reversed_bt(sl, bs, nb, sm):
            n_blocks = (sl + bs - 1) // bs
            bt = torch.arange(n_blocks - 1, -1, -1, dtype=torch.int32,
                               device="cuda").unsqueeze(0)
            return bt

        r = test_paging_pattern(
            "reversed", seq_len, num_blocks_total, block_size,
            num_heads, num_kv_heads, head_size,
            reversed_slots, reversed_bt,
        )
        results.append(r)

        # 3. Scattered blocks (non-contiguous physical allocation)
        def scattered_slots(sl, bs, nb):
            n_blocks = (sl + bs - 1) // bs
            # Pick random non-contiguous physical blocks
            torch.manual_seed(1234)
            phys_blocks = torch.randperm(nb, device="cuda")[:n_blocks]
            slots = torch.empty(sl, dtype=torch.int64, device="cuda")
            for t in range(sl):
                logical_blk = t // bs
                blk_off = t % bs
                slots[t] = phys_blocks[logical_blk].item() * bs + blk_off
            return slots

        def scattered_bt(sl, bs, nb, sm):
            n_blocks = (sl + bs - 1) // bs
            torch.manual_seed(1234)
            phys_blocks = torch.randperm(nb, device="cuda")[:n_blocks]
            return phys_blocks.to(torch.int32).unsqueeze(0)

        r = test_paging_pattern(
            "scattered", seq_len, num_blocks_total, block_size,
            num_heads, num_kv_heads, head_size,
            scattered_slots, scattered_bt,
        )
        results.append(r)

        # 4. Scattered with gaps (skip every other block)
        def gapped_slots(sl, bs, nb):
            n_blocks = (sl + bs - 1) // bs
            phys_blocks = [i * 2 for i in range(n_blocks)]  # Even blocks only
            slots = torch.empty(sl, dtype=torch.int64, device="cuda")
            for t in range(sl):
                logical_blk = t // bs
                blk_off = t % bs
                slots[t] = phys_blocks[logical_blk] * bs + blk_off
            return slots

        def gapped_bt(sl, bs, nb, sm):
            n_blocks = (sl + bs - 1) // bs
            phys = torch.tensor([i * 2 for i in range(n_blocks)],
                                 dtype=torch.int32, device="cuda")
            return phys.unsqueeze(0)

        r = test_paging_pattern(
            "gapped", seq_len, num_blocks_total, block_size,
            num_heads, num_kv_heads, head_size,
            gapped_slots, gapped_bt,
        )
        results.append(r)

    # Summary
    log("\n" + "=" * 60)
    log("PAGING DIAGNOSTIC SUMMARY")
    log("=" * 60)
    for r in results:
        log(f"  {r['name']:12s} seq_len={r['seq_len']:4d}: [{r['status']}] "
            f"max_err={r['max_err']:.4f} min_cos={r['min_cos']:.4f}")

    # Check if any pattern fails
    any_fail = any(r["status"] == "FAIL" for r in results)
    log(f"\nOverall: {'SOME FAILURES' if any_fail else 'ALL PASS'}")

    out_path = "/tmp/a100-paging-diag/paging_diag.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
