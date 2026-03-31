#!/usr/bin/env python3
"""Generate a routing_prior.pt sidecar from cartridge KV and document.

This script:
1. Loads a cartridge checkpoint
2. Loads the model
3. Runs prefill to compute per-layer per-head block affinities
4. Saves the routing sidecar as routing_prior.pt

Usage:
    python generate_routing_prior.py \
        --model marin-8b-instruct \
        --cartridge-path /path/to/cartridge.pt \
        --prefix-token-ids /path/to/prefix_token_ids.json \
        --output /path/to/routing_prior.pt \
        --block-size 16 \
        --context-length 8192
"""
import argparse
import json
import time

import torch
import torch.nn.functional as F


def compute_block_affinities(
    model,
    input_ids: torch.Tensor,
    block_size: int,
    device: str = "cuda",
) -> torch.Tensor:
    """Compute per-layer per-KV-head block affinities from prefill attention.

    Runs forward pass with output_attentions=True, aggregates attention weights
    per block, returns (num_layers, num_kv_heads, num_blocks).
    """
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids.to(device),
            output_attentions=True,
            use_cache=False,
        )

    attentions = outputs.attentions  # list of (1, num_heads, seq_len, seq_len)
    num_layers = len(attentions)

    # Infer GQA structure
    num_heads = attentions[0].shape[1]
    # Try to get num_kv_heads from config
    config = model.config
    num_kv_heads = getattr(config, 'num_key_value_heads', num_heads)
    heads_per_group = num_heads // num_kv_heads

    seq_len = input_ids.shape[1]
    num_blocks = (seq_len + block_size - 1) // block_size

    # Aggregate: for each layer, head, compute mean attention mass per block
    # Use the last token's attention as the representative query
    all_affinities = []
    for layer_idx in range(num_layers):
        attn = attentions[layer_idx][0]  # (num_heads, seq_len, seq_len)
        # Use last-token attention pattern as representative
        last_token_attn = attn[:, -1, :]  # (num_heads, seq_len)

        # Group by KV head
        kv_affinities = []
        for kv_h in range(num_kv_heads):
            h_start = kv_h * heads_per_group
            h_end = h_start + heads_per_group
            group_attn = last_token_attn[h_start:h_end].mean(dim=0)  # (seq_len,)

            # Aggregate into blocks
            block_aff = torch.zeros(num_blocks, device=device)
            for b in range(num_blocks):
                s = b * block_size
                e = min(s + block_size, seq_len)
                block_aff[b] = group_attn[s:e].sum()
            kv_affinities.append(block_aff)

        all_affinities.append(torch.stack(kv_affinities))  # (num_kv_heads, num_blocks)

    return torch.stack(all_affinities)  # (num_layers, num_kv_heads, num_blocks)


def main():
    parser = argparse.ArgumentParser(description="Generate routing prior sidecar")
    parser.add_argument("--model", required=True, help="HuggingFace model name or path")
    parser.add_argument("--cartridge-path", required=True, help="Path to cartridge.pt")
    parser.add_argument("--prefix-token-ids", required=True, help="Path to prefix_token_ids.json")
    parser.add_argument("--output", required=True, help="Output path for routing_prior.pt")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",  # need output_attentions
        device_map=args.device,
    )

    with open(args.prefix_token_ids) as f:
        prefix_ids = json.load(f)

    # Truncate to context_length
    prefix_ids = prefix_ids[:args.context_length]
    input_ids = torch.tensor([prefix_ids], dtype=torch.long)

    print(f"Computing block affinities: {len(prefix_ids)} tokens, block_size={args.block_size}")
    t0 = time.time()
    block_affinities = compute_block_affinities(model, input_ids, args.block_size, args.device)
    elapsed = time.time() - t0
    print(f"Block affinities computed in {elapsed:.1f}s: shape={block_affinities.shape}")

    num_layers, num_kv_heads, num_blocks = block_affinities.shape

    prior = {
        "version": 1,
        "model_id": args.model,
        "block_size": args.block_size,
        "doc_len": len(prefix_ids),
        "num_blocks": num_blocks,
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "context_length": args.context_length,
        "block_affinities": block_affinities.cpu(),
        "query_adjust_weights": None,
        "experiment_root": "",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "cartridge_path": args.cartridge_path,
    }

    torch.save(prior, args.output)
    print(f"Saved routing prior to {args.output}")

    # Print summary
    for layer_idx in [0, num_layers // 2, num_layers - 1]:
        aff = block_affinities[layer_idx]
        print(f"  Layer {layer_idx}: max_aff={aff.max():.4f}, mean={aff.mean():.4f}, "
              f"sparsity={((aff < 0.01).sum() / aff.numel()):.1%}")


if __name__ == "__main__":
    main()
