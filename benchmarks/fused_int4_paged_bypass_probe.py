#!/usr/bin/env python3
"""
Probe T9: Paged-cache bypass for fused INT4.

Tests the hypothesis that the fused backend's own KV cache management
(reshape_and_cache_int4 + shadow buffer) is the root cause of decode
failures. Bypasses it by:

1. Setting forward_includes_kv_cache_update = False
2. Adding do_kv_cache_update() to write FP16 KV to the standard paged cache
3. Replacing forward() to read from the standard paged cache during decode

If bypass matches baseline: bug is in fused cache-write/shadow path.
If bypass still diverges: bug is upstream (metadata, scheduling, etc).
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


def _run_all_lengths_bypass(model, lengths, max_tokens):
    """Run fused INT4 with paged-cache bypass."""
    import torch
    from vllm.v1.attention.backends.fused_int4 import (
        FusedInt4AttentionBackend,
        FusedInt4AttentionImpl,
    )

    # We need reshape_and_cache_flash for standard paged writes
    from vllm.v1.attention.backends.fa_utils import (
        reshape_and_cache_flash,
    )

    # Save originals
    orig_fwd_includes = FusedInt4AttentionBackend.forward_includes_kv_cache_update
    orig_forward = FusedInt4AttentionImpl.forward
    had_do_kv = hasattr(FusedInt4AttentionImpl, 'do_kv_cache_update')
    orig_do_kv = getattr(FusedInt4AttentionImpl, 'do_kv_cache_update', None)

    # Monkey-patch: add do_kv_cache_update to write FP16 to paged cache
    def bypass_do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        """Write FP16 K/V to the standard paged cache."""
        key_cache, value_cache = kv_cache.unbind(0)
        reshape_and_cache_flash(
            key, value,
            key_cache, value_cache,
            slot_mapping,
            "auto",  # FP16 kv_cache_dtype
            layer._k_scale,
            layer._v_scale,
        )

    def bypass_forward(
        self, layer, query, key, value, kv_cache, attn_metadata,
        output=None, output_scale=None, output_block_scale=None,
    ):
        """Bypass forward: no fused cache writes, read from paged FP16 cache."""
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

        # KV cache is standard FP16 paged format, written by do_kv_cache_update
        key_cache_fp16 = kv_cache[0]
        value_cache_fp16 = kv_cache[1]

        if attn_metadata.max_query_len == 1:
            # ---- Decode: read FP16 from paged cache ----
            if query.dim() == 2:
                query = query.view(-1, self.num_heads, self.head_size)

            num_seqs = attn_metadata.seq_lens.shape[0]
            decode_query = query[:num_seqs]
            block_table = attn_metadata.block_table
            seq_lens = attn_metadata.seq_lens
            n_rep = self.num_heads // self.num_kv_heads
            scale = 1.0 / (self.head_size ** 0.5)
            block_size = key_cache_fp16.shape[1]

            outputs = []
            for s in range(num_seqs):
                sl = seq_lens[s].item()
                blocks_needed = (sl + block_size - 1) // block_size
                k_list, v_list = [], []
                for b in range(blocks_needed):
                    block_idx = block_table[s, b].item()
                    tokens_in_block = min(block_size, sl - b * block_size)
                    k_list.append(key_cache_fp16[block_idx, :tokens_in_block])
                    v_list.append(value_cache_fp16[block_idx, :tokens_in_block])

                k_fp16 = torch.cat(k_list, dim=0)
                v_fp16 = torch.cat(v_list, dim=0)

                if n_rep > 1:
                    k_fp16 = k_fp16.repeat_interleave(n_rep, dim=1)
                    v_fp16 = v_fp16.repeat_interleave(n_rep, dim=1)

                q = decode_query[s:s+1].transpose(0, 1).unsqueeze(0)
                k = k_fp16.transpose(0, 1).unsqueeze(0)
                v = v_fp16.transpose(0, 1).unsqueeze(0)

                out = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, is_causal=False, scale=scale,
                )
                outputs.append(out.squeeze(0).transpose(0, 1))

            decode_output = torch.cat(outputs, dim=0)
            if output is not None:
                output[:decode_output.shape[0]].copy_(
                    decode_output.view(output[:decode_output.shape[0]].shape)
                )
                return output
            return decode_output.view(decode_output.shape[0], -1)

        else:
            # ---- Prefill: use fresh K/V directly with SDPA ----
            if query.dim() == 2:
                query = query.view(-1, self.num_heads, self.head_size)
            if key.dim() == 2:
                key = key.view(-1, self.num_kv_heads, self.head_size)
                value = value.view(-1, self.num_kv_heads, self.head_size)

            n_rep = self.num_heads // self.num_kv_heads
            if n_rep > 1:
                key = key.repeat_interleave(n_rep, dim=1)
                value = value.repeat_interleave(n_rep, dim=1)

            query_start_loc = attn_metadata.query_start_loc
            num_seqs = attn_metadata.seq_lens.shape[0]

            results_list = []
            for i in range(num_seqs):
                start = query_start_loc[i].item()
                end = query_start_loc[i + 1].item()
                q_i = query[start:end].transpose(0, 1).unsqueeze(0)
                k_i = key[start:end].transpose(0, 1).unsqueeze(0)
                v_i = value[start:end].transpose(0, 1).unsqueeze(0)
                out_i = torch.nn.functional.scaled_dot_product_attention(
                    q_i, k_i, v_i, is_causal=True,
                )
                results_list.append(out_i.squeeze(0).transpose(0, 1))

            combined = torch.cat(results_list, dim=0)
            flat = combined.reshape(combined.shape[0], -1)
            if output is not None:
                output[:flat.shape[0]].copy_(
                    flat.view(output[:flat.shape[0]].shape)
                )
                return output
            return flat

    # Apply patches
    FusedInt4AttentionBackend.forward_includes_kv_cache_update = False
    FusedInt4AttentionImpl.forward = bypass_forward
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
            print(f"  [fused_bypass] len={length}: {ids} -> {text!r}")
    finally:
        FusedInt4AttentionBackend.forward_includes_kv_cache_update = orig_fwd_includes
        FusedInt4AttentionImpl.forward = orig_forward
        if had_do_kv:
            FusedInt4AttentionImpl.do_kv_cache_update = orig_do_kv
        elif hasattr(FusedInt4AttentionImpl, 'do_kv_cache_update'):
            delattr(FusedInt4AttentionImpl, 'do_kv_cache_update')

    del llm
    _cleanup_gpu()
    return out


def main():
    parser = argparse.ArgumentParser(description="T9 Paged-cache bypass probe")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-tokens", type=int, default=3)
    parser.add_argument(
        "--lengths", type=str,
        default="17,18,20,26,28,32,47,48,50,64,80",
        help="Comma-separated prompt lengths",
    )
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    lengths = [int(x) for x in args.lengths.split(",")]

    results = {
        "probe": "paged_cache_bypass",
        "model": args.model,
        "max_tokens": args.max_tokens,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "comparisons": [],
    }

    print("=" * 60)
    print("Phase 1: Baseline (FlashAttention FP16)")
    print("=" * 60)
    baselines = _run_all_lengths(
        args.model, lengths, args.max_tokens,
        config_name="baseline", kv_cache_dtype=None,
    )

    print("\n" + "=" * 60)
    print("Phase 2: Fused INT4 standard (current path)")
    print("=" * 60)
    fused_std = _run_all_lengths(
        args.model, lengths, args.max_tokens,
        config_name="fused_standard", kv_cache_dtype="int4_fused",
    )

    print("\n" + "=" * 60)
    print("Phase 3: Fused INT4 bypass (paged cache write + read)")
    print("=" * 60)
    fused_bypass = _run_all_lengths_bypass(
        args.model, lengths, args.max_tokens,
    )

    # Compare
    for length in lengths:
        bl = baselines.get(length, {})
        fs = fused_std.get(length, {})
        fb = fused_bypass.get(length, {})

        comp = {
            "prompt_len": length,
            "baseline_ids": bl.get("ids", []),
            "fused_std_ids": fs.get("ids", []),
            "fused_bypass_ids": fb.get("ids", []),
            "fused_std_match": bl.get("ids", []) == fs.get("ids", []),
            "fused_bypass_match": bl.get("ids", []) == fb.get("ids", []),
        }
        results["comparisons"].append(comp)

        tag_std = "OK" if comp["fused_std_match"] else "FAIL"
        tag_byp = "OK" if comp["fused_bypass_match"] else "FAIL"
        print(f"\nlen={length}: std={tag_std} bypass={tag_byp}")
        print(f"  baseline:  {bl.get('ids', [])}")
        print(f"  fused_std: {fs.get('ids', [])}")
        print(f"  bypass:    {fb.get('ids', [])}")

    std_passes = sum(1 for c in results["comparisons"] if c["fused_std_match"])
    bypass_passes = sum(1 for c in results["comparisons"] if c["fused_bypass_match"])
    total = len(results["comparisons"])

    results["summary"] = {
        "total": total,
        "fused_std_passes": std_passes,
        "fused_bypass_passes": bypass_passes,
    }

    verdict = "BUG IN CACHE-WRITE PATH" if bypass_passes > std_passes + 2 else \
              "BUG IS UPSTREAM" if bypass_passes <= std_passes else \
              "INCONCLUSIVE"

    results["verdict"] = verdict

    print(f"\n{'=' * 60}")
    print(f"SUMMARY: std={std_passes}/{total}, bypass={bypass_passes}/{total}")
    print(f"VERDICT: {verdict}")
    print(f"{'=' * 60}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to {args.output}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
