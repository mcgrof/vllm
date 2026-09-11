#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end CAS benchmark for cartridge-only and global asymmetric KV.

Run one mode per process so every arm starts with a fresh vLLM engine:

* ``bf16-cart``: regular BF16 paged KV; cartridge copied into each request.
* ``cart-k16-v8``: regular BF16 live KV plus one shared BF16-K/FP8-V cartridge,
  fused in one attention kernel.
* ``global-k16-v8``: BF16-K/FP8-V paged KV for both cartridge and live tokens.
* ``cart-global-k16-v8``: an FP8-V live cache plus the separately shared
  FP8-V cartridge, fused in one attention kernel.

The input JSON is the LongHealth ``{"patients": [...]}`` representation used by
the CAS evaluation scripts. Only the selected patient's question records are
read; the medical document is deliberately not sent to the model.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

# The asymmetric allocator is implemented by the V2 runner. Keeping every arm
# on it avoids measuring two different model runners. In-process execution lets
# us sample CUDA allocator state and ensures engine teardown at process exit.
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASHINFER")
os.environ.setdefault("VLLM_KV_CACHE_LAYOUT", "NHD")
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
# The forked attention kernels do not depend on FlashInfer's optional sampler;
# disabling it avoids a known vendored-CCCL version-skew failure on this stack.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import torch
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams, TokensPrompt
from vllm.platforms import current_platform

SYSTEM_PROMPT = (
    "Please reference the patient medical records to answer the user's "
    "questions. Choose the single best option and provide your answer exactly "
    "as it appears in the options.\n\n"
    "Wrap your answer in: <answer> The correct option text here </answer>"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=(
            "bf16-cart",
            "cart-k16-v8",
            "global-k16-v8",
            "cart-global-k16-v8",
        ),
    )
    parser.add_argument("--cartridge", type=Path, required=True)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patient", default="patient_02")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-questions", type=int, default=20)
    parser.add_argument("--quality-max-tokens", type=int, default=2048)
    parser.add_argument("--perf-output-tokens", type=int, default=32)
    parser.add_argument("--batch-sizes", default="1,4,16,64")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    return parser.parse_args()


def unwrap_tensor(value: Any) -> torch.Tensor:
    return torch.as_tensor(value.data if hasattr(value, "data") else value)


def cartridge_shape(path: Path) -> dict[str, int]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cache = (
        checkpoint.get("cache", checkpoint)
        if isinstance(checkpoint, dict)
        else checkpoint
    )

    def get(name: str, default: Any = None) -> Any:
        if isinstance(cache, dict):
            return cache.get(name, default)
        return getattr(cache, name, default)

    trainable_keys = get("trainable_keys")
    if not trainable_keys:
        raise ValueError(f"{path} has no trainable_keys")
    frozen_keys = get("frozen_keys", []) or []
    trainable = unwrap_tensor(trainable_keys[0])
    frozen_tokens = unwrap_tensor(frozen_keys[0]).shape[2] if frozen_keys else 0
    return {
        "num_layers": len(trainable_keys),
        "num_tokens": int(frozen_tokens + trainable.shape[2]),
        "num_kv_heads": int(trainable.shape[1]),
        "head_dim": int(trainable.shape[3]),
    }


def load_questions(path: Path, patient_id: str, limit: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    patients = payload["patients"] if isinstance(payload, dict) else payload
    patient = next((p for p in patients if p["patient_id"] == patient_id), None)
    if patient is None:
        raise ValueError(f"patient {patient_id!r} is absent from {path}")
    questions = patient["questions"][:limit]
    if not questions:
        raise ValueError(f"patient {patient_id!r} has no questions")
    return questions


def user_prompt(question: dict[str, Any]) -> str:
    return (
        f"{question['question']}\n\n"
        f"A. {question['answer_a']}  B. {question['answer_b']}  "
        f"C. {question['answer_c']}  D. {question['answer_d']}  "
        f"E. {question['answer_e']}"
    )


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def score_text(text: str, question: dict[str, Any]) -> bool:
    lower = text.lower()
    start = lower.rfind("<answer>")
    end = lower.find("</answer>", start + 8) if start >= 0 else -1
    answer = text[start + 8 : end] if start >= 0 and end >= 0 else text
    answer = normalize(answer)
    options = [question[f"answer_{letter}"] for letter in "abcde"]
    gold = normalize(str(question["correct"]))
    gold_idx = next(
        (i for i, option in enumerate(options) if normalize(option) == gold), None
    )
    if gold_idx is None:
        raise ValueError(
            f"gold answer not present in options: {question['question_id']}"
        )
    predicted = [i for i, option in enumerate(options) if normalize(option) in answer]
    if predicted:
        return gold_idx == predicted[0]
    # Also accept a bare option letter, useful if generation reaches the answer
    # but not the complete option text before its token cap.
    stripped = answer.strip(" .:()[]{}\n\t")
    return bool(stripped) and stripped[0:1].upper() == "ABCDE"[gold_idx]


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((p / 100.0) * (len(ordered) - 1)))
    return ordered[index]


def summarize_outputs(outputs: list[Any], wall_s: float) -> dict[str, Any]:
    output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    ttfts_ms = [
        output.metrics.first_token_latency * 1000.0
        for output in outputs
        if output.metrics is not None and output.metrics.first_token_latency > 0
    ]
    cached = [int(output.num_cached_tokens) for output in outputs]
    return {
        "requests": len(outputs),
        "wall_s": wall_s,
        "request_per_s": len(outputs) / wall_s,
        "output_tokens": output_tokens,
        "output_token_per_s": output_tokens / wall_s,
        "ttft_ms_p50": percentile(ttfts_ms, 50),
        "ttft_ms_p95": percentile(ttfts_ms, 95),
        "num_cached_tokens_min": min(cached),
        "num_cached_tokens_max": max(cached),
    }


def build_engine(args: argparse.Namespace, num_cartridge_tokens: int) -> LLM:
    extra: dict[str, Any] = {
        "cartridge_path": str(args.cartridge.resolve()),
        "preload": ["default"],
        "gpu_capacity_bytes": 2 << 30,
    }
    cache_dtype = "auto"
    if args.mode in ("cart-k16-v8", "cart-global-k16-v8"):
        extra.update(
            cartridge_attention_mode="separate_asymmetric",
            cartridge_num_tokens=num_cartridge_tokens,
        )
    if args.mode in ("global-k16-v8", "cart-global-k16-v8"):
        cache_dtype = "auto,fp8_e4m3"
        extra["asymmetric_kv"] = True

    kv_transfer_config = {
        "kv_connector": "CartridgeConnector",
        "kv_connector_module_path": (
            "vllm.distributed.kv_transfer.kv_connector.v1.cartridge_connector"
        ),
        "kv_connector_extra_config": extra,
        "kv_role": "kv_both",
    }
    return LLM(
        model=args.model,
        tokenizer=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        kv_cache_dtype=cache_dtype,
        kv_transfer_config=kv_transfer_config,
        attention_backend="FLASHINFER",
        # Hopper's default FlashInfer decode choice is XQA, whose public
        # interface rejects the 8-token pages required by the 632-token CAS
        # cartridge (79 exact pages).  Keep ordinary and cartridge attention
        # on FlashInfer-native so the comparison uses one valid page layout.
        attention_config={"use_trtllm_attention": False},
        # This benchmark fixes the kernel choice explicitly.  Running the
        # generic model-wide autotuner on every fresh-process arm only adds
        # startup work and cannot tune the split cartridge source.
        enable_flashinfer_autotune=False,
        block_size=args.block_size,
        max_model_len=args.max_model_len,
        max_num_seqs=max(
            max(int(x) for x in args.batch_sizes.split(",")), args.max_questions
        ),
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        enforce_eager=True,
        disable_log_stats=False,
        seed=0,
    )


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shape = cartridge_shape(args.cartridge)
    cart_tokens = shape["num_tokens"]
    if cart_tokens % args.block_size:
        raise ValueError(
            f"cartridge has {cart_tokens} tokens, not aligned to block size "
            f"{args.block_size}"
        )
    questions = load_questions(args.dataset_json, args.patient, args.max_questions)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dummy_id = tokenizer.bos_token_id
    if dummy_id is None:
        dummy_id = tokenizer.eos_token_id
    if dummy_id is None:
        raise ValueError("tokenizer has neither bos_token_id nor eos_token_id")

    question_token_ids: list[list[int]] = []
    for question in questions:
        ids = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt(question)},
            ],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        # Transformers 5 returns a BatchEncoding here even without
        # return_dict=True; older releases returned the input-id list.
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        elif isinstance(ids, dict):
            ids = ids["input_ids"]
        question_token_ids.append([dummy_id] * cart_tokens + list(ids))

    engine_start = time.perf_counter()
    llm = build_engine(args, cart_tokens)
    engine_load_s = time.perf_counter() - engine_start

    quality_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.quality_max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    perf_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.perf_output_tokens,
        ignore_eos=True,
    )
    base_prompts = [TokensPrompt(prompt_token_ids=ids) for ids in question_token_ids]

    for _ in range(args.warmups):
        llm.generate([base_prompts[0]], perf_params, use_tqdm=False)

    torch.accelerator.reset_peak_memory_stats()
    quality_start = time.perf_counter()
    quality_outputs = llm.generate(base_prompts, quality_params, use_tqdm=False)
    quality_wall_s = time.perf_counter() - quality_start
    quality_rows = []
    for question, output in zip(questions, quality_outputs, strict=True):
        text = output.outputs[0].text
        quality_rows.append(
            {
                "question_id": question["question_id"],
                "correct": score_text(text, question),
                "text": text,
                "token_ids": list(output.outputs[0].token_ids),
                "num_cached_tokens": int(output.num_cached_tokens),
            }
        )

    perf_rows = []
    batch_sizes = [int(value) for value in args.batch_sizes.split(",")]
    for batch_size in batch_sizes:
        prompts = [base_prompts[i % len(base_prompts)] for i in range(batch_size)]
        # Warm the exact batch shape before timing. FlashInfer plans and JIT
        # modules are shape-sensitive, especially for the second cartridge
        # source; including first-use compilation would swamp short runs.
        for _ in range(args.warmups):
            llm.generate(prompts, perf_params, use_tqdm=False)
        for repetition in range(args.repetitions):
            start = time.perf_counter()
            outputs = llm.generate(prompts, perf_params, use_tqdm=False)
            wall_s = time.perf_counter() - start
            row = summarize_outputs(outputs, wall_s)
            row.update(batch_size=batch_size, repetition=repetition)
            perf_rows.append(row)
            print(json.dumps({"mode": args.mode, "perf": row}), flush=True)

    accuracy = sum(row["correct"] for row in quality_rows) / len(quality_rows)
    result = {
        "schema": "cas-vllm-asymmetric-v1",
        "mode": args.mode,
        "model": args.model,
        "patient": args.patient,
        "cartridge": str(args.cartridge.resolve()),
        "cartridge_shape": shape,
        "block_size": args.block_size,
        "engine_load_s": engine_load_s,
        "quality": {
            "metric": "LongHealth exact-option accuracy; deterministic greedy",
            "accuracy": accuracy,
            "correct": sum(row["correct"] for row in quality_rows),
            "total": len(quality_rows),
            "wall_s": quality_wall_s,
            "rows": quality_rows,
        },
        "performance": perf_rows,
        "cuda_peak_allocated_bytes_after_warmup": (
            torch.accelerator.max_memory_allocated()
        ),
        "environment": {
            "gpu": current_platform.get_device_name(),
            "torch": torch.__version__,
            "vllm_attention_backend": os.environ["VLLM_ATTENTION_BACKEND"],
            "vllm_kv_cache_layout": os.environ["VLLM_KV_CACHE_LAYOUT"],
            "vllm_use_v2_model_runner": os.environ["VLLM_USE_V2_MODEL_RUNNER"],
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "mode": args.mode,
                "accuracy": accuracy,
                "output": str(args.output),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
