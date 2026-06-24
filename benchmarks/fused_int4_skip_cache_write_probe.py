#!/usr/bin/env python3
"""Skip cache write probe: tests whether skipping INT4 cache writes
(only using shadow buffer) fixes the quality issue.

If results match baseline → the INT4 cache write is somehow
corrupting the model state.
If results still differ → issue is elsewhere (shadow population or decode).
"""

import json
import os
import sys
import time
import gc

MODEL = "Qwen/Qwen2.5-7B-Instruct"
FILLER = ("The quick brown fox jumps over the lazy dog. " * 40)
MAX_TOKENS = 3
TEST_LENGTHS = [1, 5, 10, 17, 18, 20, 26, 28, 32, 48, 64]


def patch_skip_cache_write():
    """Skip the reshape_and_cache_int4 call to avoid INT4 writes."""
    import vllm.v1.attention.backends.fused_int4 as fused_mod
    original_forward = fused_mod.FusedInt4AttentionImpl.forward

    def _patched_forward(self, layer, query, key, value, kv_cache,
                          attn_metadata, output=None, output_scale=None,
                          output_block_scale=None):
        import torch
        global _debug_forward_count

        if attn_metadata is None:
            if output is not None:
                output.zero_()
                return output
            return torch.zeros_like(query)

        num_actual_tokens = attn_metadata.num_actual_tokens

        if kv_cache.numel() > 0:
            # Lazily allocate scale tensors (needed even if we skip cache write)
            if self._k_scales is None:
                kv_uint8 = kv_cache.view(torch.uint8)
                key_cache = kv_uint8[0][..., :self.half_hd]
                num_blocks = key_cache.shape[0]
                block_size = key_cache.shape[1]
                self._k_scales = torch.zeros(
                    num_blocks, block_size, self.num_kv_heads, self.num_groups,
                    dtype=torch.float16, device=key_cache.device,
                )
                self._v_scales = torch.zeros(
                    num_blocks, block_size, self.num_kv_heads, self.num_groups,
                    dtype=torch.float16, device=key_cache.device,
                )

            if num_actual_tokens > 0 and key.numel() > 0:
                if key.dim() == 2:
                    key = key.view(-1, self.num_kv_heads, self.head_size)
                    value = value.view(-1, self.num_kv_heads, self.head_size)

                # SKIP reshape_and_cache_int4! Just do shadow update.
                if fused_mod.MIN_FUSED_SEQ_LEN > 1:
                    self._update_fp16_shadow(key, value, attn_metadata)

            # Decode path
            if attn_metadata.max_query_len == 1:
                if query.dim() == 2:
                    query = query.view(-1, self.num_heads, self.head_size)
                num_seqs = attn_metadata.seq_lens.shape[0]
                decode_query = query[:num_seqs]

                # Always use FP16 shadow (MSL should be 999)
                decode_output = fused_mod.decode_fp16_sdpa(
                    decode_query,
                    self._fp16_k_shadow,
                    self._fp16_v_shadow,
                    attn_metadata.seq_lens,
                    num_kv_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    head_size=self.head_size,
                )

                if output is not None:
                    output[:decode_output.shape[0]].copy_(
                        decode_output.view(
                            output[:decode_output.shape[0]].shape))
                    return output
                return decode_output.view(decode_output.shape[0], -1)
            else:
                # Prefill: use fresh K/V directly
                return self._prefill_fallback(
                    query, key, value, kv_cache, attn_metadata, output)
        else:
            if output is not None:
                output.zero_()
                return output
            return torch.zeros_like(query)

    fused_mod.FusedInt4AttentionImpl.forward = _patched_forward
    print("Patched forward to skip INT4 cache writes", file=sys.stderr)


def run_mode(prompts, kv_dtype, label, skip_write=False, msl="8"):
    os.environ["VLLM_FUSED_INT4_MIN_SEQ_LEN"] = msl

    from vllm import LLM, SamplingParams

    if skip_write and kv_dtype == "int4_fused":
        patch_skip_cache_write()

    print(f"[{label}] Init...", file=sys.stderr)
    llm = LLM(
        model=MODEL, dtype="float16", kv_cache_dtype=kv_dtype,
        max_model_len=2048, gpu_memory_utilization=0.8,
        disable_log_stats=True, enforce_eager=True,
    )
    sp = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0)

    results = []
    for p in prompts:
        out = llm.generate([p["text"]], sp)[0].outputs[0]
        ids = list(out.token_ids)
        print(f"  [{label}] len={p['actual_len']:3d} ids={ids}",
              file=sys.stderr)
        results.append({"prompt_len": p["actual_len"], "ids": ids,
                        "text": out.text})
    del llm
    gc.collect()
    return results


def main():
    from transformers import AutoTokenizer

    output_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/fused_int4_skip_cache"
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompts = []
    for tl in TEST_LENGTHS:
        tokens = tokenizer.encode(FILLER)[:tl]
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        actual = len(tokenizer.encode(text))
        prompts.append({"target_len": tl, "actual_len": actual, "text": text})

    baseline = run_mode(prompts, "auto", "baseline")
    fused_msl999 = run_mode(prompts, "int4_fused", "fused_msl999",
                            skip_write=False, msl="999")
    skip_write = run_mode(prompts, "int4_fused", "skip_cache_write",
                          skip_write=True, msl="999")

    print("\n=== COMPARISON ===", file=sys.stderr)
    print(f"{'len':>4} | {'base':>20} | {'msl999':>20} | "
          f"{'skip_write':>20} | msl_ok | skip_ok",
          file=sys.stderr)
    print("-" * 110, file=sys.stderr)

    comparisons = []
    for b, m, s in zip(baseline, fused_msl999, skip_write):
        msl_match = b["ids"] == m["ids"]
        skip_match = b["ids"] == s["ids"]
        print(f"{b['prompt_len']:>4} | {str(b['ids']):>20} | "
              f"{str(m['ids']):>20} | {str(s['ids']):>20} | "
              f"{'OK' if msl_match else 'FAIL':>6} | "
              f"{'OK' if skip_match else 'FAIL':>7}",
              file=sys.stderr)
        comparisons.append({
            "prompt_len": b["prompt_len"],
            "baseline_ids": b["ids"],
            "msl999_ids": m["ids"],
            "skip_write_ids": s["ids"],
            "msl_match": msl_match,
            "skip_match": skip_match,
        })

    msl_p = sum(1 for c in comparisons if c["msl_match"])
    skip_p = sum(1 for c in comparisons if c["skip_match"])
    total = len(comparisons)

    print(f"\nMSL=999 (with write): {msl_p}/{total} pass", file=sys.stderr)
    print(f"Skip write:           {skip_p}/{total} pass", file=sys.stderr)

    manifest = {
        "probe": "skip_cache_write",
        "model": MODEL,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "comparisons": comparisons,
        "summary": {"total": total, "msl_passes": msl_p,
                     "skip_passes": skip_p},
    }
    out_path = os.path.join(output_dir, "skip_cache_write_results.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nResults saved to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
