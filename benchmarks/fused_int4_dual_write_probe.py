#!/usr/bin/env python3
"""
Probe T10: Dual-write fix for fused INT4 backend.

The root cause: forward_includes_kv_cache_update=True makes the fused backend
manage its own KV cache writes, bypassing the standard paged cache path.
The shadow buffer for FP16 fallback has position tracking bugs for longer
sequences.

Fix: Use standard paged cache for both FP16 reads AND as the source for
INT4 quantization, by setting forward_includes_kv_cache_update=False and
adding do_kv_cache_update that writes to BOTH the FP16 paged cache and
the INT4 packed cache.

Tests three configurations:
1. Baseline (FlashAttention FP16)
2. Fused INT4 standard (original buggy path)
3. Fused INT4 with dual-write fix
"""

import argparse
import json
import os
import shutil
import sys
import time
import textwrap

os.environ.setdefault("VLLM_FUSED_INT4_MIN_SEQ_LEN", "8")

FUSED_INT4_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "vllm", "v1", "attention", "backends", "fused_int4.py",
)


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


# The dual-write fix replaces the forward method with one that:
# 1. Does NOT manage its own cache writes (done by do_kv_cache_update)
# 2. Still writes INT4 to packed cache from within forward (for fused decode)
# 3. Reads FP16 from paged cache for short-seq decode fallback (no shadow buffer)
# 4. Uses fused INT4 decode for long-seq decode

DUAL_WRITE_CODE = '''
    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        """Write FP16 K/V to standard paged cache (for correct FP16 reads)."""
        from vllm.v1.attention.backends.fa_utils import reshape_and_cache_flash
        key_cache, value_cache = kv_cache.unbind(0)
        reshape_and_cache_flash(
            key, value, key_cache, value_cache, slot_mapping,
            "auto", layer._k_scale, layer._v_scale,
        )

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FusedInt4AttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Dual-write forward: standard paged FP16 + INT4 packed cache.

        FP16 paged cache is written by do_kv_cache_update (standard path).
        INT4 packed cache is written here for the fused decode kernel.
        Short-seq decode reads from FP16 paged cache (not shadow buffer).
        Long-seq decode reads from INT4 packed cache via fused kernel.
        """
        if attn_metadata is None:
            if output is not None:
                output.zero_()
                return output
            return torch.zeros_like(query)

        num_actual_tokens = attn_metadata.num_actual_tokens

        if kv_cache.numel() == 0:
            if output is not None:
                output.zero_()
                return output
            return torch.zeros_like(query)

        # View as uint8 for INT4 packed cache
        kv_uint8 = kv_cache.view(torch.uint8)
        key_cache_int4 = kv_uint8[0][..., :self.half_hd]
        value_cache_int4 = kv_uint8[1][..., :self.half_hd]

        # FP16 paged cache (written by do_kv_cache_update via standard path)
        key_cache_fp16 = kv_cache[0]
        value_cache_fp16 = kv_cache[1]

        # Lazily allocate scale tensors
        if self._k_scales is None:
            num_blocks = key_cache_int4.shape[0]
            block_size = key_cache_int4.shape[1]
            self._k_scales = torch.zeros(
                num_blocks, block_size, self.num_kv_heads, self.num_groups,
                dtype=torch.float16, device=key_cache_int4.device,
            )
            self._v_scales = torch.zeros(
                num_blocks, block_size, self.num_kv_heads, self.num_groups,
                dtype=torch.float16, device=key_cache_int4.device,
            )

        # Write INT4 to packed cache (for fused decode kernel)
        if num_actual_tokens > 0 and key.numel() > 0:
            if key.dim() == 2:
                key_3d = key.view(-1, self.num_kv_heads, self.head_size)
                value_3d = value.view(-1, self.num_kv_heads, self.head_size)
            else:
                key_3d = key
                value_3d = value

            reshape_and_cache_int4(
                key_3d, value_3d,
                key_cache_int4, value_cache_int4,
                self._k_scales, self._v_scales,
                attn_metadata.slot_mapping,
            )

        # ---- Decode ----
        if attn_metadata.max_query_len == 1:
            if query.dim() == 2:
                query = query.view(-1, self.num_heads, self.head_size)

            num_seqs = attn_metadata.seq_lens.shape[0]
            decode_query = query[:num_seqs]
            use_fused = attn_metadata.max_seq_len >= MIN_FUSED_SEQ_LEN

            if not self._logged_init:
                self._logged_init = True
                logger.info(
                    "[FusedInt4] Backend verification: "
                    "selected_backend=FUSED_INT4, "
                    "kv_cache_dtype=int4_fused, "
                    "decode_kernel=fused_int4_triton, "
                    "group_size=%d, num_kv_heads=%d, head_size=%d, "
                    "min_fused_seq_len=%d, "
                    "fallback=paged_fp16_sdpa (dual-write fix)",
                    GROUP_SIZE, self.num_kv_heads,
                    self.head_size, MIN_FUSED_SEQ_LEN,
                )

            if use_fused:
                self._decode_fused_count += 1
                block_n = 64
                decode_output = fused_int4_decode(
                    decode_query,
                    key_cache_int4, value_cache_int4,
                    self._k_scales, self._v_scales,
                    attn_metadata.block_table,
                    attn_metadata.seq_lens,
                    num_kv_heads=self.num_kv_heads,
                    head_size=self.head_size,
                    block_n=block_n,
                )
            else:
                # Short-seq fallback: read from FP16 paged cache
                decode_output = self._decode_from_paged_fp16(
                    decode_query,
                    key_cache_fp16, value_cache_fp16,
                    attn_metadata.block_table,
                    attn_metadata.seq_lens,
                )

            if output is not None:
                output[:decode_output.shape[0]].copy_(
                    decode_output.view(output[:decode_output.shape[0]].shape)
                )
                return output
            return decode_output.view(decode_output.shape[0], -1)

        else:
            # Prefill: use fresh K/V directly with SDPA
            self._prefill_fallback_count += 1
            result = self._prefill_fallback(
                query, key, value, kv_cache, attn_metadata, output,
            )
            return result

    def _decode_from_paged_fp16(
        self,
        query: torch.Tensor,       # [num_seqs, num_heads, head_size]
        key_cache: torch.Tensor,    # [num_blocks, block_size, num_kv_heads, head_size]
        value_cache: torch.Tensor,
        block_table: torch.Tensor,  # [num_seqs, max_blocks_per_seq]
        seq_lens: torch.Tensor,     # [num_seqs]
    ) -> torch.Tensor:
        """Decode from FP16 paged cache using SDPA."""
        num_seqs = query.shape[0]
        n_rep = self.num_heads // self.num_kv_heads
        scale = 1.0 / (self.head_size ** 0.5)
        block_size = key_cache.shape[1]

        outputs = []
        for s in range(num_seqs):
            sl = seq_lens[s].item()
            blocks_needed = (sl + block_size - 1) // block_size
            k_list, v_list = [], []
            for b in range(blocks_needed):
                block_idx = block_table[s, b].item()
                tokens_in_block = min(block_size, sl - b * block_size)
                k_list.append(key_cache[block_idx, :tokens_in_block])
                v_list.append(value_cache[block_idx, :tokens_in_block])

            k_fp16 = torch.cat(k_list, dim=0)
            v_fp16 = torch.cat(v_list, dim=0)

            if n_rep > 1:
                k_fp16 = k_fp16.repeat_interleave(n_rep, dim=1)
                v_fp16 = v_fp16.repeat_interleave(n_rep, dim=1)

            q = query[s:s+1].transpose(0, 1).unsqueeze(0)
            k = k_fp16.transpose(0, 1).unsqueeze(0)
            v = v_fp16.transpose(0, 1).unsqueeze(0)

            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=False, scale=scale,
            )
            outputs.append(out.squeeze(0).transpose(0, 1))

        return torch.cat(outputs, dim=0)
'''


def patch_fused_int4(fused_path):
    """Patch fused_int4.py with dual-write fix."""
    backup_path = fused_path + ".bak"
    shutil.copy2(fused_path, backup_path)

    with open(fused_path, "r") as f:
        content = f.read()

    # 1. Change forward_includes_kv_cache_update to False
    content = content.replace(
        "forward_includes_kv_cache_update: bool = True",
        "forward_includes_kv_cache_update: bool = False",
    )

    # 2. Find the forward method and replace it + add do_kv_cache_update
    marker = "    def forward(\n        self,\n        layer: AttentionLayer,"
    idx = content.find(marker)

    if idx == -1:
        print(f"ERROR: Could not find forward() marker in {fused_path}")
        os.rename(backup_path, fused_path)
        return False

    # Replace from the forward method to end of file
    content = content[:idx] + DUAL_WRITE_CODE

    with open(fused_path, "w") as f:
        f.write(content)

    print(f"Patched {fused_path}")
    return True


def restore_fused_int4(fused_path):
    backup_path = fused_path + ".bak"
    if os.path.exists(backup_path):
        shutil.copy2(backup_path, fused_path)
        os.remove(backup_path)
        print(f"Restored {fused_path}")


def main():
    parser = argparse.ArgumentParser(description="T10 Dual-write probe")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-tokens", type=int, default=5)
    parser.add_argument(
        "--lengths", type=str,
        default="17,18,20,26,28,32,47,48,50,64,80,96,128",
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--fused-path", type=str, default=FUSED_INT4_PATH)
    args = parser.parse_args()

    lengths = [int(x) for x in args.lengths.split(",")]

    results = {
        "probe": "dual_write_fix",
        "model": args.model,
        "max_tokens": args.max_tokens,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "comparisons": [],
    }

    # Phase 1: Baseline
    print("=" * 60)
    print("Phase 1: Baseline (FlashAttention FP16)")
    print("=" * 60)
    baselines = _run_all_lengths(
        args.model, lengths, args.max_tokens,
        config_name="baseline", kv_cache_dtype=None,
    )

    # Phase 2: Fused standard
    print("\n" + "=" * 60)
    print("Phase 2: Fused INT4 standard (current path)")
    print("=" * 60)
    fused_std = _run_all_lengths(
        args.model, lengths, args.max_tokens,
        config_name="fused_std", kv_cache_dtype="int4_fused",
    )

    # Phase 3: Dual-write fix
    print("\n" + "=" * 60)
    print("Phase 3: Fused INT4 dual-write fix")
    print("=" * 60)

    fused_path = args.fused_path
    if not patch_fused_int4(fused_path):
        sys.exit(1)

    try:
        dualwrite = _run_all_lengths(
            args.model, lengths, args.max_tokens,
            config_name="dual_write", kv_cache_dtype="int4_fused",
        )
    finally:
        restore_fused_int4(fused_path)

    # Compare
    for length in lengths:
        bl = baselines.get(length, {})
        fs = fused_std.get(length, {})
        dw = dualwrite.get(length, {})

        comp = {
            "prompt_len": length,
            "baseline_ids": bl.get("ids", []),
            "fused_std_ids": fs.get("ids", []),
            "dual_write_ids": dw.get("ids", []),
            "fused_std_match": bl.get("ids", []) == fs.get("ids", []),
            "dual_write_match": bl.get("ids", []) == dw.get("ids", []),
        }
        results["comparisons"].append(comp)

        tag_std = "OK" if comp["fused_std_match"] else "FAIL"
        tag_dw = "OK" if comp["dual_write_match"] else "FAIL"
        print(f"\nlen={length}: std={tag_std} dual_write={tag_dw}")
        print(f"  baseline:   {bl.get('ids', [])}")
        print(f"  fused_std:  {fs.get('ids', [])}")
        print(f"  dual_write: {dw.get('ids', [])}")

    std_passes = sum(1 for c in results["comparisons"] if c["fused_std_match"])
    dw_passes = sum(1 for c in results["comparisons"] if c["dual_write_match"])
    total = len(results["comparisons"])

    results["summary"] = {
        "total": total,
        "fused_std_passes": std_passes,
        "dual_write_passes": dw_passes,
    }

    if dw_passes == total:
        results["verdict"] = "DUAL-WRITE FIX FULLY CORRECT"
    elif dw_passes > std_passes:
        results["verdict"] = f"DUAL-WRITE IMPROVES ({dw_passes} vs {std_passes})"
    else:
        results["verdict"] = "DUAL-WRITE NO IMPROVEMENT"

    print(f"\n{'=' * 60}")
    print(f"SUMMARY: std={std_passes}/{total}, dual_write={dw_passes}/{total}")
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
