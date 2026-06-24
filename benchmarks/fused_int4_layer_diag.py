#!/usr/bin/env python3
"""Layer-level diagnostic: Captures attention output at the first layer
during decode to compare fused vs expected behavior.

Monkey-patches FusedInt4AttentionImpl.forward to dump the decode output
and query/key/value shapes.
"""

import os
import sys
import json
import torch

os.environ["VLLM_FUSED_INT4_DEBUG"] = "1"
os.environ["VLLM_FUSED_INT4_DEBUG_PATH"] = "/tmp/layer_diag_debug.json"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_diag_captures = []
_diag_count = 0
_MAX_CAPTURES = 10


def patch_forward():
    from vllm.v1.attention.backends.fused_int4 import FusedInt4AttentionImpl

    original_forward = FusedInt4AttentionImpl.forward

    def patched_forward(self, layer, query, key, value, kv_cache,
                        attn_metadata, output=None, output_scale=None,
                        output_block_scale=None):
        global _diag_count

        result = original_forward(
            self, layer, query, key, value, kv_cache,
            attn_metadata, output, output_scale, output_block_scale,
        )

        if attn_metadata is not None and _diag_count < _MAX_CAPTURES:
            _diag_count += 1
            is_decode = (attn_metadata.max_query_len == 1)

            torch.cuda.synchronize()

            capture = {
                "call_index": _diag_count,
                "is_decode": is_decode,
                "num_actual_tokens": attn_metadata.num_actual_tokens,
                "max_query_len": attn_metadata.max_query_len,
                "max_seq_len": attn_metadata.max_seq_len,
                "query_shape": list(query.shape),
                "key_shape": list(key.shape) if key is not None else None,
                "value_shape": list(value.shape) if value is not None else None,
                "kv_cache_shape": list(kv_cache.shape) if kv_cache is not None else None,
                "output_shape": list(output.shape) if output is not None else None,
                "result_shape": list(result.shape),
                "seq_lens": attn_metadata.seq_lens.cpu().tolist()
                    if attn_metadata.seq_lens is not None else None,
                "slot_mapping_first8": attn_metadata.slot_mapping[:min(8, attn_metadata.slot_mapping.shape[0])]
                    .cpu().tolist() if attn_metadata.slot_mapping is not None else None,
            }

            # Capture actual output values for first sequence, first head
            if is_decode and output is not None:
                # output is [num_tokens, num_heads, head_size]
                out_vals = output[0, 0, :16].detach().cpu().float().tolist()
                capture["decode_output_0_0_16"] = out_vals

                # Check for all-zeros or NaN
                out_flat = output[0].detach().cpu().float()
                capture["output_has_nan"] = bool(torch.isnan(out_flat).any())
                capture["output_has_inf"] = bool(torch.isinf(out_flat).any())
                capture["output_abs_max"] = round(out_flat.abs().max().item(), 6)
                capture["output_abs_mean"] = round(out_flat.abs().mean().item(), 6)
                capture["output_zero_frac"] = round(
                    (out_flat == 0).float().mean().item(), 4)

                # Also capture query values
                q_vals = query[0, 0, :16].detach().cpu().float().tolist() \
                    if query.dim() == 3 else \
                    query[0, :16].detach().cpu().float().tolist()
                capture["decode_query_0_16"] = q_vals

                # Capture key_cache info
                if kv_cache is not None and kv_cache.numel() > 0:
                    kv_uint8 = kv_cache.view(torch.uint8)
                    half_hd = self.head_size // 2
                    key_cache = kv_uint8[0][..., :half_hd]
                    # Check if scales are populated
                    if self._k_scales is not None:
                        k_sc = self._k_scales[0, 0, 0].cpu().float().tolist()
                        capture["k_scales_block0_slot0_head0"] = k_sc
                    capture["key_cache_strides"] = list(key_cache.stride())

            _diag_captures.append(capture)
            print(f"[DIAG #{_diag_count}] decode={is_decode} "
                  f"max_q={attn_metadata.max_query_len} "
                  f"max_s={attn_metadata.max_seq_len} "
                  f"tokens={attn_metadata.num_actual_tokens} "
                  f"q_shape={list(query.shape)} "
                  f"slot_map={capture.get('slot_mapping_first8', '?')}",
                  file=sys.stderr, flush=True)

        return result

    FusedInt4AttentionImpl.forward = patched_forward


def main():
    patch_forward()

    from vllm import LLM, SamplingParams

    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"Loading {model_name} with int4_fused...", file=sys.stderr)

    llm = LLM(
        model=model_name,
        dtype="float16",
        kv_cache_dtype="int4_fused",
        max_model_len=2048,
        gpu_memory_utilization=0.8,
        disable_log_stats=True,
        enforce_eager=True,
    )

    sp = SamplingParams(max_tokens=3, temperature=0.0)

    print("Generating with prompt 'The quick'...", file=sys.stderr)
    outputs = llm.generate(["The quick"], sp)
    out = outputs[0].outputs[0]
    print(f"Output: ids={list(out.token_ids)} text={out.text!r}",
          file=sys.stderr)

    # Dump captures
    json_str = json.dumps(_diag_captures, indent=2)
    output_path = "/tmp/layer_diag.json"
    with open(output_path, "w") as f:
        f.write(json_str)
    print(f"\nDiag captures ({len(_diag_captures)}) written to {output_path}",
          file=sys.stderr)
    print(json_str)


if __name__ == "__main__":
    main()
