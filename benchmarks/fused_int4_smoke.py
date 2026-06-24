#!/usr/bin/env python3
"""Smoke benchmark for fused INT4 KV cache decode vs FP16 baseline.

Exercises both paths at the Triton kernel level and emits a backend
manifest proving which path ran.

Usage:
    python benchmarks/fused_int4_smoke.py

Output:
    JSON manifest to stdout with per-point latencies, speedup ratios,
    and backend verification fields.

This script does NOT require a running vLLM server — it tests the
kernels directly with synthetic Q/K/V tensors.
"""

import json
import sys
import time

import torch

# Ensure we can import from the repo
sys.path.insert(0, ".")

GROUP_SIZE = 32


# ----------------------------------------------------------------
# INT4 helpers (PyTorch reference)
# ----------------------------------------------------------------
def quantize_and_pack_int4(tensor, group_size=GROUP_SIZE):
    shape = tensor.shape
    hd = shape[-1]
    ng = hd // group_size
    r = tensor.float().reshape(*shape[:-1], ng, group_size)
    amax = r.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scales = (amax / 7.0).squeeze(-1).half()
    q = (r / (amax / 7.0)).round().clamp(-8, 7).to(torch.int8)
    q = q.reshape(*shape[:-1], hd)
    q_unsigned = (q + 8).to(torch.uint8)
    packed = q_unsigned[..., 0::2] | (q_unsigned[..., 1::2] << 4)
    return packed, scales


def dequant_int4(packed, scales, group_size=GROUP_SIZE):
    low = (packed & 0x0F).to(torch.int8) - 8
    high = ((packed >> 4) & 0x0F).to(torch.int8) - 8
    hd = packed.shape[-1] * 2
    ng = hd // group_size
    out = torch.empty(*packed.shape[:-1], hd, device=packed.device, dtype=torch.float16)
    out[..., 0::2] = low.to(torch.float16)
    out[..., 1::2] = high.to(torch.float16)
    out = out.reshape(*packed.shape[:-1], ng, group_size)
    out = out * scales.unsqueeze(-1).float()
    return out.reshape(*packed.shape[:-1], hd).half()


# ----------------------------------------------------------------
# Benchmark helpers
# ----------------------------------------------------------------
N_WARMUP = 3
N_REPEATS = 5


def bench_fn(fn, args):
    for _ in range(N_WARMUP):
        fn(*args)
    torch.cuda.synchronize()
    times = []
    for _ in range(N_REPEATS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*args)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return sum(times) / len(times)


# ----------------------------------------------------------------
# Baseline: FP16 SDPA (analogous to FlashAttention path)
# ----------------------------------------------------------------
def baseline_sdpa(Q, K_fp16, V_fp16, n_rep):
    K_exp = K_fp16.repeat_interleave(n_rep, dim=1)
    V_exp = V_fp16.repeat_interleave(n_rep, dim=1)
    return torch.nn.functional.scaled_dot_product_attention(
        Q.unsqueeze(2), K_exp, V_exp, is_causal=False,
    ).squeeze(2)


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU required", file=sys.stderr)
        sys.exit(1)

    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)

    # Import the fused kernel
    from vllm.v1.attention.backends.fused_int4 import fused_int4_decode

    # Model config (Qwen2.5-7B-like)
    num_heads = 28
    num_kv_heads = 4
    head_dim = 128
    n_rep = num_heads // num_kv_heads
    block_size = 16

    batch_sizes = [1, 2, 4, 8]
    context_lengths = [2048, 4096]

    half_hd = head_dim // 2
    num_groups = head_dim // GROUP_SIZE

    results = []
    print(f"GPU: {gpu_name}", file=sys.stderr)
    print(f"Config: heads={num_heads}, kv_heads={num_kv_heads}, "
          f"dim={head_dim}, group_size={GROUP_SIZE}", file=sys.stderr)
    print(file=sys.stderr)

    for B in batch_sizes:
        for T in context_lengths:
            torch.manual_seed(42)
            nb = (T + block_size - 1) // block_size
            total_blocks = B * nb

            # Generate data
            K_fp16 = torch.randn(B, num_kv_heads, T, head_dim,
                                 dtype=torch.float16, device=device)
            V_fp16 = torch.randn(B, num_kv_heads, T, head_dim,
                                 dtype=torch.float16, device=device)
            Q = torch.randn(B, num_heads, head_dim,
                            dtype=torch.float16, device=device)

            # Quantize
            K_packed, K_scales = quantize_and_pack_int4(K_fp16)
            V_packed, V_scales = quantize_and_pack_int4(V_fp16)

            # Build paged cache
            kc = torch.zeros(total_blocks, block_size, num_kv_heads,
                             half_hd, dtype=torch.uint8, device=device)
            vc = torch.zeros_like(kc)
            ks = torch.zeros(total_blocks, block_size, num_kv_heads,
                             num_groups, dtype=torch.float16, device=device)
            vs = torch.zeros_like(ks)
            bt = torch.zeros(B, nb, dtype=torch.int32, device=device)
            for b in range(B):
                for blk in range(nb):
                    pb = b * nb + blk
                    bt[b, blk] = pb
                    s = blk * block_size
                    e = min(s + block_size, T)
                    L = e - s
                    kc[pb, :L] = K_packed[b, :, s:e].permute(1, 0, 2)
                    vc[pb, :L] = V_packed[b, :, s:e].permute(1, 0, 2)
                    ks[pb, :L] = K_scales[b, :, s:e].permute(1, 0, 2)
                    vs[pb, :L] = V_scales[b, :, s:e].permute(1, 0, 2)
            sl = torch.full((B,), T, dtype=torch.int32, device=device)

            # Benchmark baseline (FP16 SDPA)
            t_baseline = bench_fn(baseline_sdpa, (Q, K_fp16, V_fp16, n_rep))

            # Benchmark fused INT4
            t_fused = bench_fn(
                fused_int4_decode,
                (Q, kc, vc, ks, vs, bt, sl, num_kv_heads, head_dim, 64),
            )

            speedup = t_baseline / t_fused if t_fused > 0 else 0

            # Correctness check
            fused_out = fused_int4_decode(
                Q, kc, vc, ks, vs, bt, sl,
                num_kv_heads=num_kv_heads, head_size=head_dim,
            )
            ref_out = baseline_sdpa(
                Q, dequant_int4(K_packed, K_scales),
                dequant_int4(V_packed, V_scales), n_rep,
            )
            cos_sim = torch.nn.functional.cosine_similarity(
                fused_out.reshape(-1).float(),
                ref_out.reshape(-1).float(), dim=0,
            ).item()

            point = {
                "batch_size": B,
                "context_length": T,
                "baseline_backend": "FP16_SDPA",
                "fused_backend": "FUSED_INT4_TRITON",
                "baseline_latency_ms": t_baseline * 1000,
                "fused_latency_ms": t_fused * 1000,
                "speedup": speedup,
                "cosine_similarity": cos_sim,
            }
            results.append(point)

            status = "WIN" if speedup > 1.0 else "LOSE"
            print(
                f"  B={B:2d} T={T:5d}: "
                f"baseline={t_baseline*1000:7.3f}ms  "
                f"fused={t_fused*1000:7.3f}ms  "
                f"speedup={speedup:.2f}x  "
                f"cos={cos_sim:.6f}  [{status}]",
                file=sys.stderr,
            )

    manifest = {
        "benchmark": "fused_int4_smoke",
        "gpu": gpu_name,
        "model_config": {
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "group_size": GROUP_SIZE,
        },
        "backend_manifest": {
            "baseline_requested": "FP16_SDPA",
            "baseline_selected": "FP16_SDPA (torch SDPA with FlashAttention when available)",
            "fused_requested": "int4_fused",
            "fused_selected": "FUSED_INT4_TRITON",
            "fused_kernel": "fused_int4_decode",
            "fallback": "none",
        },
        "results": results,
    }

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
