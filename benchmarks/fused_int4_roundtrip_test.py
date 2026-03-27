#!/usr/bin/env python3
"""
Standalone test: INT4 quantize-dequantize roundtrip accuracy.

Tests reshape_and_cache_int4 write + CPU dequant read to verify
the quantization kernel is correct independently of the attention kernel.
"""

import torch
import sys
sys.path.insert(0, ".")

from vllm.v1.attention.backends.fused_int4 import (
    reshape_and_cache_int4,
    _cpu_dequant_slot,
    GROUP_SIZE,
)


def test_roundtrip(
    num_tokens: int = 4,
    num_kv_heads: int = 4,
    head_size: int = 128,
    block_size: int = 16,
    num_blocks: int = 4,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    half_hd = head_size // 2
    num_groups = head_size // GROUP_SIZE

    # Generate random FP16 key/value
    torch.manual_seed(42)
    key = torch.randn(num_tokens, num_kv_heads, head_size,
                       dtype=torch.float16, device=device)
    value = torch.randn(num_tokens, num_kv_heads, head_size,
                         dtype=torch.float16, device=device)

    # Allocate contiguous INT4 cache (same as separate-cache fix)
    key_cache = torch.zeros(
        num_blocks, block_size, num_kv_heads, half_hd,
        dtype=torch.uint8, device=device,
    )
    value_cache = torch.zeros(
        num_blocks, block_size, num_kv_heads, half_hd,
        dtype=torch.uint8, device=device,
    )
    k_scales = torch.zeros(
        num_blocks, block_size, num_kv_heads, num_groups,
        dtype=torch.float16, device=device,
    )
    v_scales = torch.zeros(
        num_blocks, block_size, num_kv_heads, num_groups,
        dtype=torch.float16, device=device,
    )

    # Slot mapping: tokens 0..num_tokens-1 go to slots 0..num_tokens-1
    slot_mapping = torch.arange(num_tokens, dtype=torch.long, device=device)

    # Write INT4
    reshape_and_cache_int4(
        key, value,
        key_cache, value_cache,
        k_scales, v_scales,
        slot_mapping,
    )

    if device != "cpu":
        torch.cuda.synchronize()

    # Read back and verify
    print(f"=== INT4 Roundtrip Test ===")
    print(f"num_tokens={num_tokens}, num_kv_heads={num_kv_heads}, "
          f"head_size={head_size}, block_size={block_size}")
    print(f"Cache shape: key_cache={key_cache.shape}, "
          f"k_scales={k_scales.shape}")
    print(f"Cache strides: key_cache.stride()={key_cache.stride()}, "
          f"k_scales.stride()={k_scales.stride()}")
    print()

    max_errors = []
    mean_errors = []

    for t in range(num_tokens):
        block_idx = t // block_size
        block_offset = t % block_size

        for h in range(num_kv_heads):
            k_orig = key[t, h].detach().cpu().float()
            v_orig = value[t, h].detach().cpu().float()

            k_recon = _cpu_dequant_slot(
                key_cache, k_scales,
                block_idx, block_offset, head_idx=h,
                head_size=head_size, n_values=head_size,
            )
            v_recon = _cpu_dequant_slot(
                value_cache, v_scales,
                block_idx, block_offset, head_idx=h,
                head_size=head_size, n_values=head_size,
            )

            k_recon_t = torch.tensor(k_recon)
            v_recon_t = torch.tensor(v_recon)

            k_err = (k_orig - k_recon_t).abs()
            v_err = (v_orig - v_recon_t).abs()

            max_errors.append(max(k_err.max().item(), v_err.max().item()))
            mean_errors.append(
                (k_err.mean().item() + v_err.mean().item()) / 2)

            if t < 2 and h == 0:
                print(f"Token {t}, Head {h}:")
                print(f"  K orig[:8]:  {k_orig[:8].tolist()}")
                print(f"  K recon[:8]: {k_recon[:8]}")
                print(f"  K max_err:   {k_err.max().item():.6f}")
                print(f"  K mean_err:  {k_err.mean().item():.6f}")
                print(f"  V max_err:   {v_err.max().item():.6f}")
                print(f"  V mean_err:  {v_err.mean().item():.6f}")
                # Check for all-zeros
                if k_recon_t.abs().max().item() == 0:
                    print(f"  WARNING: K reconstruction is all zeros!")
                if v_recon_t.abs().max().item() == 0:
                    print(f"  WARNING: V reconstruction is all zeros!")
                print()

    print(f"Summary across all {num_tokens} tokens x {num_kv_heads} heads:")
    print(f"  Max abs error: {max(max_errors):.6f}")
    print(f"  Mean abs error: {sum(mean_errors)/len(mean_errors):.6f}")
    print(f"  Errors > 0.5:  {sum(1 for e in max_errors if e > 0.5)}"
          f"/{len(max_errors)}")

    # Now test with the FUSED decode kernel directly
    print(f"\n=== Fused Decode Kernel Test ===")
    from vllm.v1.attention.backends.fused_int4 import fused_int4_decode

    # Create a query (1 sequence, full attention heads)
    num_heads = num_kv_heads * 7  # GQA ratio = 7 for Qwen
    query = torch.randn(1, num_heads, head_size,
                         dtype=torch.float16, device=device)

    # Block table: sequence uses blocks 0, 1, 2, ...
    blocks_needed = (num_tokens + block_size - 1) // block_size
    block_table = torch.arange(blocks_needed, dtype=torch.int32,
                                device=device).unsqueeze(0)
    seq_lens = torch.tensor([num_tokens], dtype=torch.int32, device=device)

    # Run fused decode
    fused_output = fused_int4_decode(
        query, key_cache, value_cache,
        k_scales, v_scales,
        block_table, seq_lens,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
    )

    if device != "cpu":
        torch.cuda.synchronize()

    fused_out_cpu = fused_output[0].detach().cpu().float()
    print(f"  Output shape: {fused_output.shape}")
    print(f"  Output abs max: {fused_out_cpu.abs().max().item():.6f}")
    print(f"  Output abs mean: {fused_out_cpu.abs().mean().item():.6f}")
    print(f"  Output all zeros: {(fused_out_cpu == 0).all().item()}")
    print(f"  Output has NaN: {torch.isnan(fused_out_cpu).any().item()}")
    print(f"  Output has Inf: {torch.isinf(fused_out_cpu).any().item()}")
    print(f"  Output head 0 first 8: {fused_out_cpu[0, :8].tolist()}")

    # Compare with CPU-side SDPA using dequantized cache
    print(f"\n=== CPU SDPA Reference ===")
    # Dequantize entire cache
    k_dequant_list = []
    v_dequant_list = []
    for t_idx in range(num_tokens):
        bi = t_idx // block_size
        bo = t_idx % block_size
        k_heads = []
        v_heads = []
        for h in range(num_kv_heads):
            kr = _cpu_dequant_slot(key_cache, k_scales, bi, bo,
                                    head_idx=h, head_size=head_size,
                                    n_values=head_size)
            vr = _cpu_dequant_slot(value_cache, v_scales, bi, bo,
                                    head_idx=h, head_size=head_size,
                                    n_values=head_size)
            k_heads.append(torch.tensor(kr))
            v_heads.append(torch.tensor(vr))
        k_dequant_list.append(torch.stack(k_heads))
        v_dequant_list.append(torch.stack(v_heads))

    k_full = torch.stack(k_dequant_list).to(torch.float16).to(device)
    v_full = torch.stack(v_dequant_list).to(torch.float16).to(device)
    # k_full: [num_tokens, num_kv_heads, head_size]

    n_rep = num_heads // num_kv_heads
    k_expanded = k_full.repeat_interleave(n_rep, dim=1)
    v_expanded = v_full.repeat_interleave(n_rep, dim=1)

    scale = 1.0 / (head_size ** 0.5)
    q = query[0:1].transpose(0, 1).unsqueeze(0)
    k = k_expanded.transpose(0, 1).unsqueeze(0)
    v = v_expanded.transpose(0, 1).unsqueeze(0)

    ref_output = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=False, scale=scale,
    )
    ref_out_cpu = ref_output.squeeze(0).transpose(0, 1)[0].detach().cpu().float()

    diff = (fused_out_cpu - ref_out_cpu).abs()
    print(f"  Ref output head 0 first 8: {ref_out_cpu[0, :8].tolist()}")
    print(f"  Fused vs Ref max diff: {diff.max().item():.6f}")
    print(f"  Fused vs Ref mean diff: {diff.mean().item():.6f}")

    if diff.max().item() < 0.01:
        print(f"\n  VERDICT: Fused kernel matches CPU reference (roundtrip OK)")
    else:
        print(f"\n  VERDICT: Fused kernel DIVERGES from CPU reference!")
        # Find the worst head
        per_head_max = diff.max(dim=1).values
        worst_head = per_head_max.argmax().item()
        print(f"  Worst head: {worst_head}, "
              f"max diff={per_head_max[worst_head].item():.6f}")


if __name__ == "__main__":
    test_roundtrip()
    print("\n" + "=" * 60)
    print("Testing with more tokens (like real prompts):")
    print("=" * 60)
    test_roundtrip(num_tokens=20)
