#!/usr/bin/env python3
"""Build synthetic priors for Test B comparison vs kri_d_kv_sum.

Generates 2 priors compatible with eval_niah_longctx.py format:
- uniform: highest affinity to N evenly-spaced blocks (one per K target)
- random: random affinities (control)

Usage:
    python build_synth_priors.py --num-blocks 521 --num-layers 28 \\
        --K 16 --out-dir /workspace/longctx
"""
import argparse
import torch
from pathlib import Path
import random

ap = argparse.ArgumentParser()
ap.add_argument("--num-blocks", type=int, required=True)
ap.add_argument("--num-layers", type=int, required=True)
ap.add_argument("--K", type=int, default=16)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)
n_blocks = args.num_blocks
n_layers = args.num_layers
K = args.K

# Uniform prior: high affinity at K evenly-spaced positions
uniform = torch.zeros(n_layers, 1, n_blocks)
stride = n_blocks / K
positions = sorted(set(int(i * stride) for i in range(K)))
print(f"[uniform] positions: {positions}")
for p in positions:
    uniform[:, :, p] = 1000.0

torch.save({
    "version": 1, "num_layers": n_layers, "num_kv_heads": 1,
    "num_blocks": n_blocks, "block_size": 16,
    "block_affinities": uniform, "prior_type": "uniform_spread",
}, out / f"prior_uniform_K{K}.pt")
print(f"wrote {out}/prior_uniform_K{K}.pt")

# Random prior
g = torch.Generator().manual_seed(args.seed)
randp = torch.rand(n_layers, 1, n_blocks, generator=g) * 1000
torch.save({
    "version": 1, "num_layers": n_layers, "num_kv_heads": 1,
    "num_blocks": n_blocks, "block_size": 16,
    "block_affinities": randp, "prior_type": "random",
}, out / f"prior_random_seed{args.seed}.pt")
print(f"wrote {out}/prior_random_seed{args.seed}.pt")
