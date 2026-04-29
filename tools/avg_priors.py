#!/usr/bin/env python3
"""Average N priors element-wise as block_affinities. Tests multi-probe mitigation."""
import sys
import torch
from pathlib import Path

if len(sys.argv) < 3:
    print(f"Usage: {sys.argv[0]} OUT_PATH prior1.pt [prior2.pt ...]")
    sys.exit(1)

out_path = sys.argv[1]
prior_paths = sys.argv[2:]

priors = []
for p in prior_paths:
    pr = torch.load(p, map_location="cpu", weights_only=False)
    print(f"Loaded {p}: {pr['block_affinities'].shape}")
    priors.append(pr)

# Average block_affinities (cast to float for stability)
affs = torch.stack([p["block_affinities"].float() for p in priors], dim=0)
mean_aff = affs.mean(dim=0)
print(f"Mean affinity range: {mean_aff.min().item():.3f} to {mean_aff.max().item():.3f}")
print(f"Top-16 from individual priors:")
for i, p in enumerate(priors):
    aff = p["block_affinities"].float().mean(dim=(0, 1))
    top16 = torch.topk(aff, 16).indices.sort().values.tolist()
    print(f"  prior{i}: {top16}")
mean_top16 = torch.topk(mean_aff.mean(dim=(0, 1)), 16).indices.sort().values.tolist()
print(f"  averaged: {mean_top16}")

# Save averaged prior
out = {**priors[0]}  # copy metadata
out["block_affinities"] = mean_aff
out["prior_type"] = "kri_d_kv_sum_multi_probe_avg"
torch.save(out, out_path)
print(f"Wrote {out_path}")
