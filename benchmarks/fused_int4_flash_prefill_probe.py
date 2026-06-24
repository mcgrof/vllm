#!/usr/bin/env python3
"""Flash Prefill Probe: tests whether replacing _prefill_fallback
with flash_attn_varlen_func fixes the quality issue.

Patches fused_int4.py at runtime to use flash attention for prefill
instead of the custom SDPA fallback.

If results match baseline → root cause is _prefill_fallback implementation
If results still differ → root cause is elsewhere
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


def patch_prefill_fallback():
    """Monkey-patch _prefill_fallback to use flash_attn_varlen_func."""
    import vllm.v1.attention.backends.fused_int4 as fused_mod

    try:
        from flash_attn import flash_attn_varlen_func
        has_flash = True
    except ImportError:
        has_flash = False

    if not has_flash:
        print("flash_attn not available, cannot patch", file=sys.stderr)
        return False

    original_prefill = fused_mod.FusedInt4Impl._prefill_fallback

    def _patched_prefill_fallback(self, query, key, value, kv_cache,
                                   attn_metadata, output):
        """Prefill using flash_attn_varlen_func instead of custom SDPA."""
        import torch

        if query.dim() == 2:
            query = query.view(-1, self.num_heads, self.head_size)
        if key.dim() == 2:
            key = key.view(-1, self.num_kv_heads, self.head_size)
            value = value.view(-1, self.num_kv_heads, self.head_size)

        # flash_attn_varlen_func expects [total_tokens, heads, head_dim]
        # with cumulative sequence lengths
        query_start_loc = attn_metadata.query_start_loc
        num_seqs = attn_metadata.seq_lens.shape[0]

        # Build cu_seqlens for flash attention
        # query_start_loc already has the right format [0, len1, len1+len2, ...]
        cu_seqlens_q = query_start_loc.to(torch.int32)
        cu_seqlens_k = query_start_loc.to(torch.int32)

        max_seqlen = attn_metadata.max_query_len

        # Call flash attention
        out = flash_attn_varlen_func(
            query,      # [total_tokens, num_heads, head_size]
            key,        # [total_tokens, num_kv_heads, head_size]
            value,      # [total_tokens, num_kv_heads, head_size]
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=True,
        )  # [total_tokens, num_heads, head_size]

        flat = out.reshape(out.shape[0], -1)  # [total_tokens, num_heads * head_size]
        if output is not None:
            output[:flat.shape[0]].copy_(
                flat.view(output[:flat.shape[0]].shape)
            )
            return output
        return flat

    fused_mod.FusedInt4Impl._prefill_fallback = _patched_prefill_fallback
    print("Patched _prefill_fallback to use flash_attn_varlen_func",
          file=sys.stderr)
    return True


def run_mode(prompts, kv_dtype, label, use_flash_prefill=False, msl="8"):
    """Run generation with the given configuration."""
    os.environ["VLLM_FUSED_INT4_MIN_SEQ_LEN"] = msl

    from vllm import LLM, SamplingParams

    if use_flash_prefill and kv_dtype == "int4_fused":
        patch_prefill_fallback()

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
        results.append({
            "prompt_len": p["actual_len"],
            "ids": ids,
            "text": out.text,
        })
    del llm
    gc.collect()
    return results


def main():
    from transformers import AutoTokenizer

    output_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/fused_int4_flash_prefill"
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompts = []
    for tl in TEST_LENGTHS:
        tokens = tokenizer.encode(FILLER)[:tl]
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        actual = len(tokenizer.encode(text))
        prompts.append({"target_len": tl, "actual_len": actual, "text": text})

    # 1. Baseline (FlashAttention, FP16 KV cache)
    baseline = run_mode(prompts, "auto", "baseline")

    # 2. Fused with custom SDPA prefill + shadow decode (MSL=999)
    fused_sdpa = run_mode(prompts, "int4_fused", "fused_sdpa_prefill",
                          use_flash_prefill=False, msl="999")

    # 3. Fused with FlashAttention prefill + shadow decode (MSL=999)
    fused_flash = run_mode(prompts, "int4_fused", "fused_flash_prefill",
                           use_flash_prefill=True, msl="999")

    # Compare
    print("\n=== COMPARISON ===", file=sys.stderr)
    print(f"{'len':>4} | {'base':>20} | {'sdpa_pfill':>20} | "
          f"{'flash_pfill':>20} | sdpa_ok | flash_ok",
          file=sys.stderr)
    print("-" * 110, file=sys.stderr)

    comparisons = []
    for b, s, f in zip(baseline, fused_sdpa, fused_flash):
        sdpa_match = b["ids"] == s["ids"]
        flash_match = b["ids"] == f["ids"]
        print(f"{b['prompt_len']:>4} | {str(b['ids']):>20} | "
              f"{str(s['ids']):>20} | {str(f['ids']):>20} | "
              f"{'OK' if sdpa_match else 'FAIL':>7} | "
              f"{'OK' if flash_match else 'FAIL':>8}",
              file=sys.stderr)
        comparisons.append({
            "prompt_len": b["prompt_len"],
            "baseline_ids": b["ids"],
            "sdpa_prefill_ids": s["ids"],
            "flash_prefill_ids": f["ids"],
            "sdpa_match": sdpa_match,
            "flash_match": flash_match,
        })

    sdpa_passes = sum(1 for c in comparisons if c["sdpa_match"])
    flash_passes = sum(1 for c in comparisons if c["flash_match"])
    total = len(comparisons)

    print(f"\nSDPA prefill:  {sdpa_passes}/{total} pass", file=sys.stderr)
    print(f"Flash prefill: {flash_passes}/{total} pass", file=sys.stderr)

    if flash_passes > sdpa_passes:
        print("\nCONCLUSION: Flash prefill fixes issues! "
              "Root cause is _prefill_fallback implementation.",
              file=sys.stderr)
    elif flash_passes == sdpa_passes:
        print("\nCONCLUSION: Flash prefill doesn't help. "
              "Root cause is NOT the prefill implementation.",
              file=sys.stderr)
    else:
        print("\nCONCLUSION: Mixed results — needs more investigation.",
              file=sys.stderr)

    manifest = {
        "probe": "flash_prefill",
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "comparisons": comparisons,
        "summary": {
            "total": total,
            "sdpa_passes": sdpa_passes,
            "flash_passes": flash_passes,
        },
    }
    out_path = os.path.join(output_dir, "flash_prefill_results.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nResults saved to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
