#!/usr/bin/env python3
"""LongHealth MC eval with trained cartridge via HF DynamicCache + correct RoPE positions.

This bypasses the vLLM CartridgeConnector (which has a RoPE position bug
when routing at K < full) and uses the HF harness's position-preserving
cache_reindex path. The goal is to measure what trained-cartridge routing
SHOULD score with correct RoPE, to quantify the impact of the position fix.

Comparison targets:
  Plan 4C (vLLM, wrong positions): full=51.5%, K=64=35.5%, K=32=28.5%
  Plan 4E (vLLM, wrong positions): kri_t_content K=32=35.5%
  This script (HF, correct positions): ???

Usage:
    python eval_cartridge_hf_positions.py \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --cartridge /path/to/cache-step2694.pt \
        --prior /path/to/routing_prior_krit.pt \
        --K 32 \
        --out results_K32.json
"""
import argparse
import json
import os
import sys
import time

import torch


def load_trainable_cache_to_dynamic(cartridge_path, device="cpu"):
    """Load a TrainableCache .pt into HF DynamicCache."""
    from transformers import DynamicCache

    ckpt = torch.load(cartridge_path, map_location="cpu", weights_only=False)

    def _get(attr, obj):
        if hasattr(obj, attr):
            return getattr(obj, attr)
        if isinstance(obj, dict) and attr in obj:
            return obj[attr]
        return None

    trainable_keys = _get("trainable_keys", ckpt)
    trainable_values = _get("trainable_values", ckpt)
    frozen_keys = _get("frozen_keys", ckpt) or []
    frozen_values = _get("frozen_values", ckpt) or []

    num_layers = len(trainable_keys)
    cache = DynamicCache()

    for li in range(num_layers):
        k_trn = trainable_keys[li].data if hasattr(trainable_keys[li], "data") else trainable_keys[li]
        v_trn = trainable_values[li].data if hasattr(trainable_values[li], "data") else trainable_values[li]
        if frozen_keys:
            k_frz = frozen_keys[li].data if hasattr(frozen_keys[li], "data") else frozen_keys[li]
            v_frz = frozen_values[li].data if hasattr(frozen_values[li], "data") else frozen_values[li]
            k_full = torch.cat([k_frz, k_trn], dim=2)
            v_full = torch.cat([v_frz, v_trn], dim=2)
        else:
            k_full = k_trn
            v_full = v_trn

        # Shape: (1, num_kv_heads, num_tokens, head_dim) — already DynamicCache layout
        k_full = k_full.to(device).contiguous()
        v_full = v_full.to(device).contiguous()
        T = k_full.shape[2]
        cache_pos = torch.arange(T, dtype=torch.long, device=device)
        cache.update(k_full, v_full, layer_idx=li,
                     cache_kwargs={"cache_position": cache_pos})

    num_tokens = cache.get_seq_length()
    return cache, num_tokens, num_layers


def select_top_k_blocks(prior_path, K, num_blocks, block_size):
    """Load a routing prior .pt and return top-K block indices."""
    prior = torch.load(prior_path, map_location="cpu", weights_only=False)
    if isinstance(prior, dict) and "block_affinities" in prior:
        # Shape: (num_layers, 1_or_heads, num_blocks) — average across layers+heads
        aff = prior["block_affinities"].float()
        scores = aff.mean(dim=(0, 1))  # (num_blocks,)
    else:
        raise ValueError(f"Unknown prior format: {type(prior)}")

    scores = scores[:num_blocks]
    topk = torch.topk(scores, min(K, num_blocks)).indices.sort().values
    return topk.tolist()


def reindex_cache(cache, keep_token_indices, device):
    """Compact the cache to keep only specified token indices."""
    idx = keep_token_indices.to(device)
    for layer in cache.layers:
        if not hasattr(layer, "key_cache") and not hasattr(layer, "keys"):
            continue
        # DynamicCache stores as .key_cache / .value_cache (list) or
        # directly as tensors depending on transformers version
        if hasattr(layer, "key_cache"):
            for i in range(len(layer.key_cache)):
                layer.key_cache[i] = layer.key_cache[i].index_select(-2, idx).contiguous()
                layer.value_cache[i] = layer.value_cache[i].index_select(-2, idx).contiguous()
        else:
            layer.keys = layer.keys.index_select(-2, idx).contiguous()
            layer.values = layer.values.index_select(-2, idx).contiguous()


def decode_with_positions(model, tokenizer, cache, query_text,
                          total_prefix_tokens, kept_token_indices,
                          max_new_tokens=512, temperature=0.3, device="cuda"):
    """Query prefill + greedy decode with correct RoPE positions."""
    # Wrap in Llama chat template — the cartridge prefix ends with the
    # system message's <|eot_id|>, so the query must start with the user
    # header and end with the assistant header to trigger generation.
    wrapped = (
        "<|start_header_id|>user<|end_header_id|>\n\n"
        + query_text
        + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    query_ids = tokenizer(wrapped, return_tensors="pt",
                          add_special_tokens=False).input_ids.to(device)
    n_q = query_ids.shape[1]

    # Compact cache length
    L_compact = kept_token_indices.numel()

    # Query positions: original positions after the full prefix
    q_position_ids = torch.arange(
        total_prefix_tokens, total_prefix_tokens + n_q,
        dtype=torch.long, device=device
    ).unsqueeze(0)
    q_cache_position = torch.arange(
        L_compact, L_compact + n_q,
        dtype=torch.long, device=device
    )

    with torch.no_grad():
        out = model(
            input_ids=query_ids,
            past_key_values=cache,
            position_ids=q_position_ids,
            cache_position=q_cache_position,
            use_cache=True,
            return_dict=True,
        )

    tokens = []
    next_logits = out.logits[0, -1, :]
    base_pos = total_prefix_tokens + n_q
    base_slot = L_compact + n_q

    for step in range(max_new_tokens):
        if temperature > 0:
            probs = torch.softmax(next_logits / temperature, dim=-1)
            next_id = int(torch.multinomial(probs, 1).item())
        else:
            next_id = int(next_logits.argmax().item())
        tokens.append(next_id)
        if tokenizer.eos_token_id is not None and next_id == tokenizer.eos_token_id:
            break

        new_input = torch.tensor([[next_id]], dtype=torch.long, device=device)
        pos_ids = torch.tensor([[base_pos + step]], dtype=torch.long, device=device)
        cache_pos = torch.tensor([base_slot + step], dtype=torch.long, device=device)

        with torch.no_grad():
            out = model(
                input_ids=new_input,
                past_key_values=cache,
                position_ids=pos_ids,
                cache_position=cache_pos,
                use_cache=True,
                return_dict=True,
            )
        next_logits = out.logits[0, -1, :]

    return tokenizer.decode(tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--cartridge", required=True)
    parser.add_argument("--prior", required=True,
                        help="routing_prior .pt file, or 'full' for no routing")
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-impl", default="sdpa")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=200,
                        help="Max questions to eval")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    os.environ.setdefault("CARTRIDGES_DIR", "/data/cartridges")
    os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/tmp/cart_out")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from cartridges.data.longhealth.evals import LongHealthMultipleChoiceGenerateDataset

    print(f"[eval] Loading model {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_impl,
    ).to(args.device).eval()

    print(f"[eval] Loading dataset", flush=True)
    patient_ids = [f"patient_{i:02d}" for i in range(1, 11)]
    dataset = LongHealthMultipleChoiceGenerateDataset.Config(
        patient_ids=patient_ids
    ).instantiate(tokenizer=tokenizer, seed=args.seed)
    n_eval = min(args.limit, len(dataset))
    print(f"[eval] {n_eval} questions", flush=True)

    # Load cartridge once to get metadata
    print(f"[eval] Loading cartridge {args.cartridge}", flush=True)
    test_cache, num_cart_tokens, num_layers = load_trainable_cache_to_dynamic(
        args.cartridge, device="cpu"
    )
    del test_cache
    num_blocks = num_cart_tokens // args.block_size
    print(f"[eval] Cartridge: {num_cart_tokens} tokens, {num_blocks} blocks, "
          f"{num_layers} layers", flush=True)

    # Determine block selection
    is_full = (args.prior == "full")
    if is_full:
        selected_blocks = list(range(num_blocks))
        K_actual = num_blocks
    else:
        K_actual = min(args.K, num_blocks)
        selected_blocks = select_top_k_blocks(
            args.prior, K_actual, num_blocks, args.block_size
        )
    print(f"[eval] K={K_actual} blocks selected: {selected_blocks[:10]}{'...' if len(selected_blocks) > 10 else ''}",
          flush=True)

    # Build token indices for selected blocks
    token_indices = []
    for bi in selected_blocks:
        start = bi * args.block_size
        end = min(start + args.block_size, num_cart_tokens)
        token_indices.extend(range(start, end))
    kept_indices = torch.tensor(sorted(token_indices), dtype=torch.long)

    correct = 0.0
    per_question = []
    t_start = time.monotonic()

    for qi in range(n_eval):
        # Reload cartridge fresh each question (cache is mutated by decode)
        cache, _, _ = load_trainable_cache_to_dynamic(args.cartridge, device=args.device)

        # Compact to selected blocks
        if not is_full:
            reindex_cache(cache, kept_indices, args.device)

        elem = dataset[qi]
        pred = decode_with_positions(
            model, tokenizer, cache, elem.prompt,
            total_prefix_tokens=num_cart_tokens,
            kept_token_indices=kept_indices if not is_full else torch.arange(num_cart_tokens),
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device=args.device,
        )

        metrics, _ = dataset.score(pred=pred, answer=elem.answer, convo_id=elem.convo_id)
        if isinstance(metrics, dict):
            score = float(list(metrics.values())[0])
        else:
            score = float(bool(metrics))
        correct += score
        per_question.append(score)

        # Free cache memory
        del cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if (qi + 1) % 20 == 0 or qi == n_eval - 1:
            elapsed = time.monotonic() - t_start
            print(f"  [{qi+1}/{n_eval}] running_acc={correct/(qi+1):.1%} "
                  f"elapsed={elapsed:.0f}s", flush=True)

    acc = correct / max(n_eval, 1)
    result = {
        "tag": f"hf_positions_K{K_actual}" if not is_full else "hf_positions_full",
        "k_blocks": K_actual,
        "correct": correct,
        "total": n_eval,
        "accuracy": acc,
        "prior": args.prior,
        "model": args.model,
        "cartridge": args.cartridge,
        "elapsed_s": time.monotonic() - t_start,
        "per_question": per_question,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "per_question"}, indent=2))


if __name__ == "__main__":
    main()
