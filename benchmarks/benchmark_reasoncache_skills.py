# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Evaluate SkillsBench skill files with ReasonCACHE in vLLM.

Typical workflow:

1. Plan the ReasonCACHE arm and confirm placeholder sizing:

   python benchmarks/benchmark_reasoncache_skills.py plan \\
       --task-dir /workspace/skillsbench/tasks/exoplanet-detection-period \\
       --reasoncache-path /workspace/reasoncache/exoplanet.pt

2. Run one arm on a GPU node:

   python benchmarks/benchmark_reasoncache_skills.py run \\
       --model /workspace/models/Qwen2.5-7B-Instruct \\
       --task-dir /workspace/skillsbench/tasks/exoplanet-detection-period \\
       --arm reasoncache \\
       --reasoncache-path /workspace/reasoncache/exoplanet.pt \\
       --output-dir /workspace/results/exoplanet/reasoncache

3. Compare old cartridge/full/no-skill result JSON files with the new
   ReasonCACHE result:

   python benchmarks/benchmark_reasoncache_skills.py compare \\
       /workspace/results/exoplanet/full-skill/result.json \\
       /workspace/results/exoplanet/no-skill/result.json \\
       /workspace/results/exoplanet/reasoncache/result.json

4. Build a Pareto frontier report across tasks:

   python benchmarks/benchmark_reasoncache_skills.py pareto \\
       /workspace/results/*/*/result.json \\
       --output-json /workspace/results/pareto.json \\
       --output-csv /workspace/results/pareto.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm.benchmarks.reasoncache_skills import (  # noqa: E402
    ArmEvalResult,
    analyze_pareto_frontier,
    build_chat_messages,
    build_reasoncache_kv_transfer_config,
    build_reasoncache_request_plan,
    build_task_prompt,
    compare_arm_results,
    infer_reasoncache_placeholder_tokens,
    load_result_file,
    load_skillsbench_task,
)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def cmd_plan(args: argparse.Namespace) -> None:
    task = load_skillsbench_task(args.task_dir)
    placeholder_tokens = args.placeholder_tokens
    if placeholder_tokens is None:
        placeholder_tokens = infer_reasoncache_placeholder_tokens(
            args.reasoncache_path,
            block_size=args.block_size,
        )

    full_prompt = build_task_prompt(
        task, "full-skill", response_mode=args.response_mode
    )
    no_skill_prompt = build_task_prompt(
        task, "no-skill", response_mode=args.response_mode
    )
    reasoncache_cfg = build_reasoncache_kv_transfer_config(
        args.reasoncache_path,
        placeholder_tokens=placeholder_tokens,
        placeholder_token_id=args.placeholder_token_id,
    )
    plan = {
        "task": task.task_name,
        "skill_files": task.skill_files,
        "skill_sha256": task.skill_sha256,
        "full_skill_prompt_chars": len(full_prompt),
        "no_skill_prompt_chars": len(no_skill_prompt),
        "reasoncache_placeholder_tokens": placeholder_tokens,
        "reasoncache_placeholder_token_id": args.placeholder_token_id,
        "kv_transfer_config": reasoncache_cfg,
    }
    print(json.dumps(plan, indent=2))


def _llm_kwargs(
    args: argparse.Namespace, kv_transfer_config: dict[str, Any] | None = None
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": args.model,
    }
    if args.enforce_eager:
        kwargs["enforce_eager"] = True
    if args.dtype is not None:
        kwargs["dtype"] = args.dtype
    if args.max_model_len is not None:
        kwargs["max_model_len"] = args.max_model_len
    if args.gpu_memory_utilization is not None:
        kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
    if kv_transfer_config is not None:
        kwargs["kv_transfer_config"] = kv_transfer_config
    return kwargs


def cmd_run(args: argparse.Namespace) -> None:
    from vllm import LLM, SamplingParams

    arm = args.arm
    task = load_skillsbench_task(args.task_dir)
    prompt_text = build_task_prompt(task, arm, response_mode=args.response_mode)
    messages = build_chat_messages(prompt_text, system_prompt=args.system_prompt)

    placeholder_tokens = args.placeholder_tokens
    kv_transfer_config = None
    if arm == "reasoncache":
        if args.reasoncache_path is None:
            raise ValueError("--reasoncache-path is required for --arm reasoncache")
        if placeholder_tokens is None:
            placeholder_tokens = infer_reasoncache_placeholder_tokens(
                args.reasoncache_path,
                block_size=args.block_size,
            )
        kv_transfer_config = build_reasoncache_kv_transfer_config(
            args.reasoncache_path,
            placeholder_tokens=placeholder_tokens,
            placeholder_token_id=args.placeholder_token_id,
        )

    llm = LLM(**_llm_kwargs(args, kv_transfer_config))
    sampling_extra_args = None
    prompts: list[Any]
    if arm == "reasoncache":
        assert args.reasoncache_path is not None
        assert placeholder_tokens is not None
        token_prompt = llm.preprocess_chat(messages)[0]
        plan = build_reasoncache_request_plan(
            prompt_token_ids=token_prompt["prompt_token_ids"],
            reasoncache_path=args.reasoncache_path,
            placeholder_tokens=placeholder_tokens,
            placeholder_token_id=args.placeholder_token_id,
            prompt_text=prompt_text,
        )
        prompts = [plan.prompt]
        sampling_extra_args = plan.sampling_extra_args
        prompt_tokens = plan.online_prompt_tokens
        task_prompt_tokens = plan.task_prompt_tokens
    else:
        token_prompt = llm.preprocess_chat(messages)[0]
        prompts = [token_prompt]
        prompt_tokens = len(token_prompt["prompt_token_ids"])
        task_prompt_tokens = prompt_tokens

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        extra_args=sampling_extra_args,
    )

    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=not args.no_tqdm)
    elapsed = time.time() - t0
    completion = outputs[0].outputs[0]
    response = completion.text
    gen_token_ids = getattr(completion, "token_ids", None)
    gen_tokens = len(gen_token_ids) if gen_token_ids is not None else None

    out_dir = Path(args.output_dir)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs" / "model_response.txt").write_text(response)

    result = ArmEvalResult(
        task=task.task_name,
        arm=arm,
        prompt_tokens=prompt_tokens,
        gen_tokens=gen_tokens,
        gen_time_sec=round(elapsed, 3),
        metadata={
            "skill_sha256": task.skill_sha256,
            "skill_files": task.skill_files,
            "task_prompt_tokens": task_prompt_tokens,
            "placeholder_tokens": placeholder_tokens,
            "placeholder_token_id": args.placeholder_token_id
            if arm == "reasoncache"
            else None,
            "response_path": str(out_dir / "logs" / "model_response.txt"),
        },
    )
    _write_json(out_dir / "result.json", result.to_dict())
    print(json.dumps(result.to_dict(), indent=2))


def cmd_compare(args: argparse.Namespace) -> None:
    results = [load_result_file(path) for path in args.result_json]
    comparison = compare_arm_results(results, tolerance=args.tolerance)
    print(json.dumps(comparison, indent=2))


def _write_pareto_csv(path: Path, analysis: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scope",
        "task",
        "arm",
        "quality",
        "prompt_tokens",
        "gen_time_sec",
        "gen_tokens",
        "on_frontier",
        "dominated_by",
    ]
    rows: list[dict[str, Any]] = []

    for point in analysis["aggregate_by_arm"]["points"]:
        rows.append(
            {
                "scope": "aggregate_by_arm",
                "task": point["task"],
                "arm": point["arm"],
                "quality": point["quality"],
                "prompt_tokens": point["prompt_tokens"],
                "gen_time_sec": point["gen_time_sec"],
                "gen_tokens": point["gen_tokens"],
                "on_frontier": point["on_frontier"],
                "dominated_by": ";".join(point["dominated_by"]),
            }
        )

    for task, task_analysis in analysis["by_task"].items():
        for point in task_analysis["points"]:
            rows.append(
                {
                    "scope": f"task:{task}",
                    "task": point["task"],
                    "arm": point["arm"],
                    "quality": point["quality"],
                    "prompt_tokens": point["prompt_tokens"],
                    "gen_time_sec": point["gen_time_sec"],
                    "gen_tokens": point["gen_tokens"],
                    "on_frontier": point["on_frontier"],
                    "dominated_by": ";".join(point["dominated_by"]),
                }
            )

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def cmd_pareto(args: argparse.Namespace) -> None:
    results = [load_result_file(path) for path in args.result_json]
    analysis = analyze_pareto_frontier(
        results,
        tolerance=args.tolerance,
        include_latency=not args.no_latency,
        include_gen_tokens=args.include_gen_tokens,
    )
    if args.output_json is not None:
        _write_json(Path(args.output_json), analysis)
    if args.output_csv is not None:
        _write_pareto_csv(Path(args.output_csv), analysis)
    print(json.dumps(analysis, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="print ReasonCACHE eval plan")
    plan.add_argument("--task-dir", required=True)
    plan.add_argument("--reasoncache-path", required=True)
    plan.add_argument("--block-size", type=int, default=16)
    plan.add_argument("--placeholder-tokens", type=int, default=None)
    plan.add_argument("--placeholder-token-id", type=int, default=0)
    plan.add_argument("--response-mode", default="json")
    plan.set_defaults(func=cmd_plan)

    run = sub.add_parser("run", help="run one SkillsBench eval arm")
    run.add_argument("--model", required=True)
    run.add_argument("--task-dir", required=True)
    run.add_argument(
        "--arm", choices=["full-skill", "no-skill", "reasoncache"], required=True
    )
    run.add_argument("--reasoncache-path", default=None)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--block-size", type=int, default=16)
    run.add_argument("--placeholder-tokens", type=int, default=None)
    run.add_argument("--placeholder-token-id", type=int, default=0)
    run.add_argument("--response-mode", default="json")
    run.add_argument(
        "--system-prompt",
        default=("You are a precise assistant. Follow instructions exactly."),
    )
    run.add_argument("--max-tokens", type=int, default=2048)
    run.add_argument("--temperature", type=float, default=0.1)
    run.add_argument("--top-p", type=float, default=1.0)
    run.add_argument("--dtype", default=None)
    run.add_argument("--max-model-len", type=int, default=None)
    run.add_argument("--gpu-memory-utilization", type=float, default=None)
    run.add_argument("--enforce-eager", action="store_true")
    run.add_argument("--no-tqdm", action="store_true")
    run.set_defaults(func=cmd_run)

    compare = sub.add_parser("compare", help="compare arm result JSON files")
    compare.add_argument("result_json", nargs="+")
    compare.add_argument("--tolerance", type=float, default=0.0)
    compare.set_defaults(func=cmd_compare)

    pareto = sub.add_parser(
        "pareto",
        aliases=["frontier"],
        help="build Pareto frontier report from arm result JSON files",
    )
    pareto.add_argument("result_json", nargs="+")
    pareto.add_argument("--tolerance", type=float, default=0.0)
    pareto.add_argument("--output-json", default=None)
    pareto.add_argument("--output-csv", default=None)
    pareto.add_argument(
        "--no-latency",
        action="store_true",
        help="ignore generation latency even when every point has timing",
    )
    pareto.add_argument(
        "--include-gen-tokens",
        action="store_true",
        help="include generated token count as a minimized frontier axis",
    )
    pareto.set_defaults(func=cmd_pareto)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
