#!/usr/bin/env python3
"""Decode SDPA probe: tests whether replacing decode_fp16_sdpa
with flash_attn_with_kvcache fixes the quality issue.

Key insight: if replacing both prefill and decode with flash attention
makes fused_msl999 match baseline, the issue is in decode_fp16_sdpa.
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


def patch_both_flash():
    """Monkey-patch both prefill and decode to use flash attention."""
    import vllm.v1.attention.backends.fused_int4 as fused_mod
    import torch

    try:
        from flash_attn import flash_attn_varlen_func, flash_attn_func
        has_flash = True
    except ImportError:
        has_flash = False

    if not has_flash:
        print("flash_attn not available, cannot patch", file=sys.stderr)
        return False

    # Patch prefill
    def _patched_prefill(self, query, key, value, kv_cache, attn_metadata,
                          output):
        if query.dim() == 2:
            query = query.view(-1, self.num_heads, self.head_size)
        if key.dim() == 2:
            key = key.view(-1, self.num_kv_heads, self.head_size)
            value = value.view(-1, self.num_kv_heads, self.head_size)

        query_start_loc = attn_metadata.query_start_loc
        cu_seqlens = query_start_loc.to(torch.int32)
        max_seqlen = attn_metadata.max_query_len

        out = flash_attn_varlen_func(
            query, key, value,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=True,
        )
        flat = out.reshape(out.shape[0], -1)
        if output is not None:
            output[:flat.shape[0]].copy_(flat.view(output[:flat.shape[0]].shape))
            return output
        return flat

    # Patch decode_fp16_sdpa to use flash attention
    def _patched_decode_fp16_sdpa(
        query, fp16_k, fp16_v, seq_lens, num_kv_heads, num_heads, head_size
    ):
        num_seqs = query.shape[0]
        outputs = []
        for s in range(num_seqs):
            sl = seq_lens[s].item()
            k_fp16 = fp16_k[s, :sl]  # [sl, num_kv_heads, head_size]
            v_fp16 = fp16_v[s, :sl]

            q = query[s:s+1]  # [1, num_heads, head_size]

            # flash_attn_func: q [B, Sq, H, D], k [B, Sk, Hkv, D], v [B, Sk, Hkv, D]
            # Unsqueeze for batch dim
            q_flash = q.unsqueeze(0)    # [1, 1, num_heads, head_size]
            k_flash = k_fp16.unsqueeze(0)  # [1, sl, num_kv_heads, head_size]
            v_flash = v_fp16.unsqueeze(0)

            out = flash_attn_func(
                q_flash, k_flash, v_flash,
                causal=False,
            )  # [1, 1, num_heads, head_size]
            outputs.append(out.squeeze(0))  # [1, num_heads, head_size]
        return torch.cat(outputs, dim=0)  # [num_seqs, num_heads, head_size]

    fused_mod.FusedInt4Impl._prefill_fallback = _patched_prefill
    fused_mod.decode_fp16_sdpa = _patched_decode_fp16_sdpa
    print("Patched both prefill and decode to use flash attention",
          file=sys.stderr)
    return True


def run_mode(prompts, kv_dtype, label, patch=False, msl="8"):
    os.environ["VLLM_FUSED_INT4_MIN_SEQ_LEN"] = msl

    from vllm import LLM, SamplingParams

    if patch and kv_dtype == "int4_fused":
        patch_both_flash()

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

    output_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/fused_int4_decode_sdpa"
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompts = []
    for tl in TEST_LENGTHS:
        tokens = tokenizer.encode(FILLER)[:tl]
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        actual = len(tokenizer.encode(text))
        prompts.append({"target_len": tl, "actual_len": actual, "text": text})

    # 1. Baseline
    baseline = run_mode(prompts, "auto", "baseline")

    # 2. Fused MSL=999 with custom SDPA (original behavior)
    fused_sdpa = run_mode(prompts, "int4_fused", "fused_sdpa", patch=False,
                          msl="999")

    # 3. Fused MSL=999 with flash attention for both prefill and decode
    fused_flash = run_mode(prompts, "int4_fused", "fused_flash_both",
                           patch=True, msl="999")

    # Compare
    print("\n=== COMPARISON ===", file=sys.stderr)
    print(f"{'len':>4} | {'base':>20} | {'sdpa_both':>20} | "
          f"{'flash_both':>20} | sdpa_ok | flash_ok",
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
            "sdpa_ids": s["ids"],
            "flash_ids": f["ids"],
            "sdpa_match": sdpa_match,
            "flash_match": flash_match,
        })

    sdpa_p = sum(1 for c in comparisons if c["sdpa_match"])
    flash_p = sum(1 for c in comparisons if c["flash_match"])
    total = len(comparisons)

    print(f"\nSDPA both:  {sdpa_p}/{total} pass", file=sys.stderr)
    print(f"Flash both: {flash_p}/{total} pass", file=sys.stderr)

    if flash_p > sdpa_p:
        print("\nCONCLUSION: Flash for both fixes issues! "
              "Root cause is decode_fp16_sdpa or the shadow buffer.",
              file=sys.stderr)
    elif flash_p == total:
        print("\nCONCLUSION: ALL PASS with flash! Shadow data is correct, "
              "decode_fp16_sdpa has a numerical issue.", file=sys.stderr)
    elif flash_p == sdpa_p:
        print("\nCONCLUSION: Flash doesn't help. Shadow buffer data may be "
              "wrong, or there's a deeper issue.", file=sys.stderr)

    manifest = {
        "probe": "decode_sdpa",
        "model": MODEL,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "comparisons": comparisons,
        "summary": {"total": total, "sdpa_passes": sdpa_p,
                     "flash_passes": flash_p},
    }
    out_path = os.path.join(output_dir, "decode_sdpa_results.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nResults saved to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
