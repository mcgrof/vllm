#!/usr/bin/env python3
"""Build NIAH-style long-context cartridges from a corpus.

Takes a long text corpus, inserts N "needle" facts at known positions,
prefills through a model to get a K/V cache, and saves in trainable
cartridge format (trainable_keys/values dict) for the existing eval
pipeline.

Outputs:
  - <out_dir>/cart_<size>.pt   — cartridge .pt file (trainable_keys/values)
  - <out_dir>/needles_<size>.json — needle positions + answers + questions
  - <out_dir>/corpus_<size>.txt   — the actual text used (with needles inserted)

Usage:
    python build_niah_longctx.py \\
        --model meta-llama/Llama-3.2-3B-Instruct \\
        --corpus /path/to/corpus.txt \\
        --out-dir /workspace/longctx \\
        --target-tokens 8192 \\
        --num-needles 50
"""
import argparse
import json
import random
import os
import sys
from pathlib import Path

import torch


def build_niah_text(corpus_text, target_tokens, num_needles, tokenizer, seed=42):
    """Insert needles into corpus, return text + needle metadata."""
    random.seed(seed)
    # Tokenize corpus and truncate/pad to target_tokens (leave room for needles)
    needle_overhead = num_needles * 12  # ~12 tokens per needle insertion
    base_tokens = target_tokens - needle_overhead

    corpus_ids = tokenizer.encode(corpus_text, add_special_tokens=False)
    if len(corpus_ids) < base_tokens:
        # Repeat to fill
        rep = (base_tokens // len(corpus_ids)) + 1
        corpus_ids = (corpus_ids * rep)[:base_tokens]
    else:
        corpus_ids = corpus_ids[:base_tokens]

    base_text = tokenizer.decode(corpus_ids, skip_special_tokens=True)

    # Generate N unique 5-digit codes
    codes = random.sample(range(10000, 100000), num_needles)
    # Pick N positions (in chars) roughly evenly spread
    positions = sorted([
        int(len(base_text) * (i + 0.5) / num_needles)
        for i in range(num_needles)
    ])

    # Insert needles at each position. Use distinct topic IDs.
    needles = []
    insertion_offsets = []  # cumulative char shifts
    new_text = base_text
    cum_offset = 0
    for i, (pos, code) in enumerate(zip(positions, codes)):
        topic_id = f"NIAH-{i:03d}"
        needle_text = f" The secret code for {topic_id} is {code}. "
        actual_pos = pos + cum_offset
        # Find nearest sentence boundary (period+space)
        adj_pos = new_text.find(". ", actual_pos)
        if adj_pos < 0 or adj_pos > actual_pos + 200:
            adj_pos = actual_pos
        else:
            adj_pos += 2  # after the ". "
        new_text = new_text[:adj_pos] + needle_text + new_text[adj_pos:]
        cum_offset += len(needle_text)
        needles.append({
            "topic_id": topic_id,
            "code": str(code),
            "char_pos": adj_pos,
            "fractional_pos": adj_pos / len(new_text),
            "question": f"What is the secret code for {topic_id}?",
            "answer": str(code),
        })

    # Verify token count
    final_ids = tokenizer.encode(new_text, add_special_tokens=False)
    return new_text, final_ids, needles


def prefill_and_save_cartridge(model, tokenizer, text_ids, out_path, device):
    """Prefill model on text_ids, save K/V cache as trainable cartridge format."""
    from transformers import DynamicCache

    input_ids = torch.tensor([text_ids], dtype=torch.long, device=device)
    cache = DynamicCache()

    print(f"[prefill] running model on {input_ids.shape[1]} tokens", flush=True)
    with torch.no_grad():
        out = model(input_ids=input_ids, past_key_values=cache, use_cache=True, return_dict=True)

    # Extract K/V per layer
    trainable_keys = []
    trainable_values = []
    pkv = out.past_key_values
    if hasattr(pkv, "layers"):
        for layer in pkv.layers:
            if hasattr(layer, "key_cache"):
                trainable_keys.append(layer.key_cache[0].cpu().clone())
                trainable_values.append(layer.value_cache[0].cpu().clone())
            else:
                trainable_keys.append(layer.keys.cpu().clone())
                trainable_values.append(layer.values.cpu().clone())
    else:
        for k, v in pkv:
            trainable_keys.append(k.cpu().clone())
            trainable_values.append(v.cpu().clone())

    cart = {
        "trainable_keys": trainable_keys,
        "trainable_values": trainable_values,
        "frozen_keys": [],
        "frozen_values": [],
        "format_version": 1,
        "source": "prefilled_niah_longctx",
    }
    torch.save(cart, out_path)
    print(f"[save] {out_path}: {input_ids.shape[1]} tokens, {len(trainable_keys)} layers, "
          f"{trainable_keys[0].shape[-2]} cache positions", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--corpus", required=True, help="Path to text corpus")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--target-tokens", type=int, required=True)
    ap.add_argument("--num-needles", type=int, default=50)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[load] model {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="eager",
    ).to(args.device).eval()

    print(f"[load] corpus {args.corpus}", flush=True)
    corpus_text = Path(args.corpus).read_text()

    print(f"[build] NIAH text targeting {args.target_tokens} tokens with {args.num_needles} needles", flush=True)
    text, text_ids, needles = build_niah_text(
        corpus_text, args.target_tokens, args.num_needles, tokenizer, args.seed
    )

    print(f"[built] {len(text_ids)} tokens, {len(needles)} needles", flush=True)

    # Save outputs
    size_label = f"{args.target_tokens // 1024}k"

    (out_dir / f"corpus_{size_label}.txt").write_text(text)
    (out_dir / f"needles_{size_label}.json").write_text(json.dumps(needles, indent=2))

    cart_path = out_dir / f"cart_{size_label}.pt"
    prefill_and_save_cartridge(model, tokenizer, text_ids, cart_path, args.device)

    print(f"[done] wrote {out_dir}/{{corpus,needles,cart}}_{size_label}.*", flush=True)


if __name__ == "__main__":
    sys.exit(main())
