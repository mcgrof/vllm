#!/usr/bin/env python3
"""
Probe T9c: On-disk bypass for fused INT4 backend.

Since vLLM V1 runs EngineCore in a subprocess, monkey-patches in the main
process don't propagate. This probe modifies fused_int4.py on disk before
creating the LLM instance, then restores it after.

The bypass:
1. Sets forward_includes_kv_cache_update = False
2. Adds do_kv_cache_update() for standard paged cache writes
3. Replaces forward() to read from standard paged FP16 cache

Tests whether selecting the FUSED_INT4 backend itself causes divergence
independent of the fused code paths.
"""

import argparse
import json
import os
import shutil
import sys
import time

os.environ.setdefault("VLLM_FUSED_INT4_MIN_SEQ_LEN", "8")

# We need to find fused_int4.py location
FUSED_INT4_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "vllm", "v1", "attention", "backends", "fused_int4.py",
)

BYPASS_FORWARD_CODE = '''
    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        """Write FP16 K/V to standard paged cache (bypass mode)."""
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
        """BYPASS forward: standard paged cache + SDPA attention."""
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

        # Standard paged FP16 cache (written by do_kv_cache_update)
        key_cache_fp16 = kv_cache[0]
        value_cache_fp16 = kv_cache[1]

        if attn_metadata.max_query_len == 1:
            # Decode: read from standard paged FP16 cache
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
            # Prefill: use fresh K/V directly with SDPA
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
'''


def patch_fused_int4(fused_path):
    """Patch fused_int4.py to use bypass forward."""
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
    # We'll replace from "def forward(" in FusedInt4AttentionImpl to end of file
    # Actually, let's be more surgical: find the class and replace the forward method

    # Find "    def forward(\n        self,\n        layer: AttentionLayer,"
    # and replace everything from there to end of file
    marker = "    def forward(\n        self,\n        layer: AttentionLayer,\n        query: torch.Tensor,\n        key: torch.Tensor,\n        value: torch.Tensor,\n        kv_cache: torch.Tensor,\n        attn_metadata: FusedInt4AttentionMetadata,\n        output: torch.Tensor | None = None,\n        output_scale: torch.Tensor | None = None,\n        output_block_scale: torch.Tensor | None = None,\n    ) -> torch.Tensor:"

    idx = content.find(marker)
    if idx == -1:
        # Try simpler marker
        marker = "    def forward(\n        self,\n        layer: AttentionLayer,"
        idx = content.find(marker)

    if idx == -1:
        print(f"ERROR: Could not find forward() marker in {fused_path}")
        os.rename(backup_path, fused_path)
        return False

    # Replace from the forward method to end of file
    content = content[:idx] + BYPASS_FORWARD_CODE

    with open(fused_path, "w") as f:
        f.write(content)

    print(f"Patched {fused_path}")
    print(f"Backup at {backup_path}")
    return True


def restore_fused_int4(fused_path):
    """Restore original fused_int4.py."""
    backup_path = fused_path + ".bak"
    if os.path.exists(backup_path):
        shutil.copy2(backup_path, fused_path)
        os.remove(backup_path)
        print(f"Restored {fused_path}")


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


def main():
    parser = argparse.ArgumentParser(description="T9c On-disk bypass probe")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-tokens", type=int, default=3)
    parser.add_argument(
        "--lengths", type=str,
        default="17,18,20,26,28,32,48,64,80",
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--fused-path", type=str, default=FUSED_INT4_PATH,
                       help="Path to fused_int4.py")
    args = parser.parse_args()

    lengths = [int(x) for x in args.lengths.split(",")]

    results = {
        "probe": "on_disk_bypass",
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
    print("Phase 2: Fused INT4 standard")
    print("=" * 60)
    fused_std = _run_all_lengths(
        args.model, lengths, args.max_tokens,
        config_name="fused_std", kv_cache_dtype="int4_fused",
    )

    # Phase 3: On-disk bypass
    print("\n" + "=" * 60)
    print("Phase 3: Fused INT4 bypass (on-disk patch)")
    print("=" * 60)

    fused_path = args.fused_path
    if not os.path.exists(fused_path):
        print(f"ERROR: {fused_path} not found")
        sys.exit(1)

    if not patch_fused_int4(fused_path):
        print("ERROR: Failed to patch")
        sys.exit(1)

    try:
        bypass = _run_all_lengths(
            args.model, lengths, args.max_tokens,
            config_name="bypass", kv_cache_dtype="int4_fused",
        )
    finally:
        restore_fused_int4(fused_path)

    # Compare
    for length in lengths:
        bl = baselines.get(length, {})
        fs = fused_std.get(length, {})
        bp = bypass.get(length, {})

        comp = {
            "prompt_len": length,
            "baseline_ids": bl.get("ids", []),
            "fused_std_ids": fs.get("ids", []),
            "bypass_ids": bp.get("ids", []),
            "fused_std_match": bl.get("ids", []) == fs.get("ids", []),
            "bypass_match": bl.get("ids", []) == bp.get("ids", []),
        }
        results["comparisons"].append(comp)

        tag_std = "OK" if comp["fused_std_match"] else "FAIL"
        tag_byp = "OK" if comp["bypass_match"] else "FAIL"
        print(f"\nlen={length}: std={tag_std} bypass={tag_byp}")
        print(f"  baseline: {bl.get('ids', [])}")
        print(f"  fused_std: {fs.get('ids', [])}")
        print(f"  bypass:    {bp.get('ids', [])}")

    std_passes = sum(1 for c in results["comparisons"] if c["fused_std_match"])
    bypass_passes = sum(1 for c in results["comparisons"] if c["bypass_match"])
    total = len(results["comparisons"])

    results["summary"] = {
        "total": total,
        "fused_std_passes": std_passes,
        "bypass_passes": bypass_passes,
    }

    if bypass_passes == total:
        results["verdict"] = "BUG IS IN FUSED CACHE/ATTENTION PATH"
    elif bypass_passes > std_passes + 2:
        results["verdict"] = "BYPASS SIGNIFICANTLY BETTER - cache path is a factor"
    elif bypass_passes <= std_passes:
        results["verdict"] = "BYPASS NO BETTER - bug is structural/upstream"
    else:
        results["verdict"] = "INCONCLUSIVE"

    print(f"\n{'=' * 60}")
    print(f"SUMMARY: std={std_passes}/{total}, bypass={bypass_passes}/{total}")
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
