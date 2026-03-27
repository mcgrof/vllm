#!/usr/bin/env python3
"""
Probe T9b: Replace fused INT4 forward entirely with FlashAttention's forward.

Tests whether the FUSED_INT4 backend selection + metadata path causes
the decode divergence, independent of any fused code.

Strategy:
1. forward_includes_kv_cache_update = False (standard cache writes)
2. Add do_kv_cache_update that uses reshape_and_cache_flash
3. Replace forward() with FlashAttention's exact forward logic
   (flash_attn_varlen_func for both prefill and decode)

If this matches baseline: the fused metadata/layout is fine, the bug is in SDPA
If this diverges: the FUSED_INT4 backend selection causes structural issues
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("VLLM_FUSED_INT4_MIN_SEQ_LEN", "8")


def _cleanup_gpu():
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def make_prompt(length: int) -> str:
    phrase = "The quick brown fox jumps over the lazy dog. "
    repeats = max(1, (length * 2) // len(phrase) + 1)
    return (phrase * repeats)[:length * 5]


def _run_all_lengths(model, lengths, max_tokens, config_name, kv_cache_dtype):
    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=model,
        dtype="float16",
        max_model_len=512,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
    )
    if kv_cache_dtype:
        kwargs["kv_cache_dtype"] = kv_cache_dtype

    llm = LLM(**kwargs)
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    out = {}
    for length in lengths:
        prompt = make_prompt(length)
        result = llm.generate([prompt], params)
        ids = list(result[0].outputs[0].token_ids)
        text = result[0].outputs[0].text
        out[length] = {"ids": ids, "text": text}
        print(f"  [{config_name}] len={length}: {ids} -> {text!r}")

    del llm
    _cleanup_gpu()
    return out


def _run_all_lengths_flash_forward(model, lengths, max_tokens):
    """Run fused INT4 backend with FlashAttention's forward."""
    import torch
    from vllm.v1.attention.backends.fused_int4 import (
        FusedInt4AttentionBackend,
        FusedInt4AttentionImpl,
    )
    from vllm.v1.attention.backends.fa_utils import (
        flash_attn_varlen_func,
        reshape_and_cache_flash,
    )

    orig_fwd_includes = FusedInt4AttentionBackend.forward_includes_kv_cache_update
    orig_forward = FusedInt4AttentionImpl.forward
    had_do_kv = hasattr(FusedInt4AttentionImpl, 'do_kv_cache_update')

    def bypass_do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        key_cache, value_cache = kv_cache.unbind(0)
        reshape_and_cache_flash(
            key, value,
            key_cache, value_cache,
            slot_mapping,
            "auto",
            layer._k_scale,
            layer._v_scale,
        )

    def flash_forward(
        self, layer, query, key, value, kv_cache, attn_metadata,
        output=None, output_scale=None, output_block_scale=None,
    ):
        """Forward using FlashAttention for both prefill and decode."""
        if attn_metadata is None:
            if output is not None:
                output.zero_()
                return output
            return torch.zeros_like(query)

        if kv_cache.numel() == 0:
            if output is not None:
                output.zero_()
                return output
            return torch.zeros_like(query)

        num_actual_tokens = attn_metadata.num_actual_tokens
        key_cache, value_cache = kv_cache.unbind(0)

        if query.dim() == 2:
            query = query.view(-1, self.num_heads, self.head_size)

        # Use flash_attn_varlen_func for everything
        cu_seqlens_q = attn_metadata.query_start_loc
        max_seqlen_q = attn_metadata.max_query_len

        cu_seqlens_k = attn_metadata.query_start_loc  # For prefill, same as Q
        max_seqlen_k = attn_metadata.max_seq_len

        if attn_metadata.max_query_len == 1:
            # Decode path
            num_seqs = attn_metadata.seq_lens.shape[0]
            decode_query = query[:num_seqs]

            # Build cu_seqlens for decode
            cu_seqlens_q_decode = torch.arange(
                0, num_seqs + 1, dtype=torch.int32,
                device=query.device,
            )
            seq_lens = attn_metadata.seq_lens
            cu_seqlens_k_decode = torch.zeros(
                num_seqs + 1, dtype=torch.int32, device=query.device,
            )
            cu_seqlens_k_decode[1:] = torch.cumsum(seq_lens, dim=0)

            # Gather KV from paged cache
            block_table = attn_metadata.block_table
            block_size = key_cache.shape[1]

            k_list, v_list = [], []
            for s in range(num_seqs):
                sl = seq_lens[s].item()
                blocks_needed = (sl + block_size - 1) // block_size
                for b in range(blocks_needed):
                    block_idx = block_table[s, b].item()
                    tokens_in_block = min(block_size, sl - b * block_size)
                    k_list.append(key_cache[block_idx, :tokens_in_block])
                    v_list.append(value_cache[block_idx, :tokens_in_block])

            k_flat = torch.cat(k_list, dim=0)  # [total_kv_tokens, num_kv_heads, head_size]
            v_flat = torch.cat(v_list, dim=0)

            flash_out = flash_attn_varlen_func(
                q=decode_query,
                k=k_flat,
                v=v_flat,
                cu_seqlens_q=cu_seqlens_q_decode,
                cu_seqlens_k=cu_seqlens_k_decode.to(torch.int32),
                max_seqlen_q=1,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=1.0 / (self.head_size ** 0.5),
                causal=True,
                window_size=(-1, -1),
            )

            if isinstance(flash_out, tuple):
                flash_out = flash_out[0]

            flat_out = flash_out.view(num_seqs, -1)
            if output is not None:
                output[:num_seqs].copy_(flat_out.view(output[:num_seqs].shape))
                return output
            return flat_out

        else:
            # Prefill: use fresh K/V
            if key.dim() == 2:
                key = key.view(-1, self.num_kv_heads, self.head_size)
                value = value.view(-1, self.num_kv_heads, self.head_size)

            flash_out = flash_attn_varlen_func(
                q=query[:num_actual_tokens],
                k=key[:num_actual_tokens],
                v=value[:num_actual_tokens],
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_q,  # Same for prefill
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_q,
                softmax_scale=1.0 / (self.head_size ** 0.5),
                causal=True,
                window_size=(-1, -1),
            )

            if isinstance(flash_out, tuple):
                flash_out = flash_out[0]

            flat_out = flash_out.view(num_actual_tokens, -1)
            if output is not None:
                output[:num_actual_tokens].copy_(
                    flat_out.view(output[:num_actual_tokens].shape)
                )
                return output
            return flat_out

    # Apply patches
    FusedInt4AttentionBackend.forward_includes_kv_cache_update = False
    FusedInt4AttentionImpl.forward = flash_forward
    FusedInt4AttentionImpl.do_kv_cache_update = bypass_do_kv_cache_update

    from vllm import LLM, SamplingParams

    try:
        llm = LLM(
            model=model,
            dtype="float16",
            kv_cache_dtype="int4_fused",
            max_model_len=512,
            gpu_memory_utilization=0.5,
            enforce_eager=True,
        )
        params = SamplingParams(temperature=0.0, max_tokens=max_tokens)

        out = {}
        for length in lengths:
            prompt = make_prompt(length)
            result = llm.generate([prompt], params)
            ids = list(result[0].outputs[0].token_ids)
            text = result[0].outputs[0].text
            out[length] = {"ids": ids, "text": text}
            print(f"  [flash_fwd] len={length}: {ids} -> {text!r}")
    finally:
        FusedInt4AttentionBackend.forward_includes_kv_cache_update = orig_fwd_includes
        FusedInt4AttentionImpl.forward = orig_forward
        if had_do_kv:
            pass  # restore if needed
        elif hasattr(FusedInt4AttentionImpl, 'do_kv_cache_update'):
            delattr(FusedInt4AttentionImpl, 'do_kv_cache_update')

    del llm
    _cleanup_gpu()
    return out


def main():
    parser = argparse.ArgumentParser(description="T9b Flash forward probe")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-tokens", type=int, default=3)
    parser.add_argument(
        "--lengths", type=str,
        default="17,18,20,26,28,32,48,64,80",
    )
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    lengths = [int(x) for x in args.lengths.split(",")]

    results = {
        "probe": "flash_forward_bypass",
        "model": args.model,
        "max_tokens": args.max_tokens,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "comparisons": [],
    }

    print("=" * 60)
    print("Phase 1: Baseline (FlashAttention backend)")
    print("=" * 60)
    baselines = _run_all_lengths(
        args.model, lengths, args.max_tokens,
        config_name="baseline", kv_cache_dtype=None,
    )

    print("\n" + "=" * 60)
    print("Phase 2: Fused INT4 with FlashAttention forward")
    print("=" * 60)
    flash_fwd = _run_all_lengths_flash_forward(
        args.model, lengths, args.max_tokens,
    )

    for length in lengths:
        bl = baselines.get(length, {})
        ff = flash_fwd.get(length, {})

        comp = {
            "prompt_len": length,
            "baseline_ids": bl.get("ids", []),
            "flash_fwd_ids": ff.get("ids", []),
            "match": bl.get("ids", []) == ff.get("ids", []),
        }
        results["comparisons"].append(comp)

        tag = "OK" if comp["match"] else "FAIL"
        print(f"\nlen={length}: {tag}")
        print(f"  baseline:  {bl.get('ids', [])}")
        print(f"  flash_fwd: {ff.get('ids', [])}")

    passes = sum(1 for c in results["comparisons"] if c["match"])
    total = len(results["comparisons"])

    results["summary"] = {
        "total": total,
        "passes": passes,
    }

    if passes == total:
        results["verdict"] = "BUG IS IN SDPA PATH (not structural)"
    elif passes == 0:
        results["verdict"] = "BUG IS STRUCTURAL (backend selection level)"
    else:
        results["verdict"] = "PARTIAL - mixed results"

    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {passes}/{total} match")
    print(f"VERDICT: {results['verdict']}")
    print(f"{'=' * 60}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to {args.output}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
