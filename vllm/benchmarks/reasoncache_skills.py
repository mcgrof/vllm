# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SkillsBench helpers for evaluating ReasonCACHE learned KV prefixes.

The historical KNLP cartridge experiments compared three arms:

* ``full-skill``: include ``environment/skills/**/SKILL.md`` in prompt.
* ``no-skill``: omit skill text.
* ``cartridge``: omit skill text and inject a learned KV prefix.

ReasonCACHE uses the same serving primitive as the cartridge arm on this
branch, except it must be represented as explicit placeholder prompt tokens
until vLLM grows true virtual-prefix scheduling. This module keeps the
SkillsBench file loading and result comparison logic in one place so real GPU
benchmarks and CPU unit tests exercise the same contract.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

SkillEvalArm = Literal["full-skill", "no-skill", "reasoncache"]

DEFAULT_SYSTEM_PROMPT = (
    "You are a precise assistant. Follow instructions exactly and output only "
    "what is requested."
)
DEFAULT_CONNECTOR_MODULE = (
    "vllm.distributed.kv_transfer.kv_connector.v1.cartridge_connector"
)
TEXT_DATA_SUFFIXES = frozenset(
    (".bib", ".csv", ".dat", ".json", ".jsonl", ".md", ".txt")
)
SKIPPED_DATA_DIRS = frozenset(("skills", "tests", "__pycache__"))


class TokenizerLike(Protocol):
    def encode(self, text: str, **kwargs: Any) -> list[int]: ...


@dataclass(frozen=True)
class SkillsBenchTask:
    """Resolved SkillsBench task inputs used by all eval arms."""

    task_dir: str
    task_name: str
    instruction: str
    skill_text: str
    skill_files: list[str]
    data_files: dict[str, str] = field(default_factory=dict)

    @property
    def skill_sha256(self) -> str:
        return hashlib.sha256(self.skill_text.encode()).hexdigest()


@dataclass(frozen=True)
class ReasonCacheRequestPlan:
    """Token-level request pieces needed by the ReasonCACHE arm."""

    prompt: dict[str, Any]
    sampling_extra_args: dict[str, Any]
    kv_transfer_config: dict[str, Any]
    placeholder_tokens: int
    placeholder_token_id: int
    task_prompt_tokens: int
    online_prompt_tokens: int


@dataclass(frozen=True)
class ArmEvalResult:
    """Comparable result summary for one SkillsBench arm."""

    task: str
    arm: str
    passed: int = 0
    total: int = 0
    pass_rate: float | None = None
    reward: float | None = None
    prompt_tokens: int | None = None
    gen_tokens: int | None = None
    gen_time_sec: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> ArmEvalResult:
        total = int(data.get("total", 0) or 0)
        passed = int(data.get("passed", 0) or 0)
        pass_rate = data.get("pass_rate")
        if pass_rate is None and total > 0:
            pass_rate = passed / total
        reward = data.get("reward")
        if reward is None and total > 0:
            reward = 1.0 if passed == total else 0.0
        prompt_tokens = data.get("prompt_tokens")
        if prompt_tokens is None:
            prompt_tokens = data.get("online_prompt_tokens")
        gen_time_sec = data.get("gen_time_sec")
        if gen_time_sec is None:
            gen_time_sec = data.get("total_gen_time_sec")
        metadata = {
            k: v
            for k, v in data.items()
            if k
            not in {
                "task",
                "arm",
                "passed",
                "total",
                "pass_rate",
                "reward",
                "prompt_tokens",
                "online_prompt_tokens",
                "gen_tokens",
                "gen_time_sec",
                "total_gen_time_sec",
            }
        }
        return cls(
            task=str(data.get("task", "")),
            arm=str(data.get("arm", "")),
            passed=passed,
            total=total,
            pass_rate=float(pass_rate) if pass_rate is not None else None,
            reward=float(reward) if reward is not None else None,
            prompt_tokens=int(prompt_tokens)
            if prompt_tokens is not None
            else None,
            gen_tokens=int(data["gen_tokens"])
            if data.get("gen_tokens") is not None
            else None,
            gen_time_sec=float(gen_time_sec)
            if gen_time_sec is not None
            else None,
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _is_under_skipped_dir(path: Path, root: Path) -> bool:
    rel_parts = path.relative_to(root).parts
    return any(part in SKIPPED_DATA_DIRS for part in rel_parts)


def load_skill_text(task_dir: str | Path) -> tuple[str, list[str]]:
    """Load the same ``SKILL.md`` files used by the cartridge harness."""
    task_path = Path(task_dir)
    skills_dir = task_path / "environment" / "skills"
    if not skills_dir.exists():
        return "", []

    parts: list[str] = []
    skill_files: list[str] = []
    for skill_file in sorted(skills_dir.rglob("SKILL.md")):
        rel = skill_file.relative_to(task_path)
        skill_files.append(str(rel))
        parts.append(
            f"### Skill: {skill_file.parent.name}\n\n"
            f"{_read_text(skill_file).strip()}"
        )
    return "\n\n".join(parts), skill_files


def load_task_data_files(
    task_dir: str | Path,
    *,
    max_chars_per_file: int = 4000,
    suffixes: frozenset[str] = TEXT_DATA_SUFFIXES,
) -> dict[str, str]:
    """Load small text-like task data files from ``environment``.

    Binary payloads such as XLSX workbooks are intentionally not read here.
    Real benchmark runners can still copy those task directories into place
    before running task-specific tests.
    """
    task_path = Path(task_dir)
    env_dir = task_path / "environment"
    if not env_dir.exists():
        return {}

    data_files: dict[str, str] = {}
    for path in sorted(env_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name == "Dockerfile" or path.suffix not in suffixes:
            continue
        if _is_under_skipped_dir(path, env_dir):
            continue
        content = _read_text(path)
        if max_chars_per_file > 0 and len(content) > max_chars_per_file:
            content = content[:max_chars_per_file] + "\n..."
        data_files[str(path.relative_to(env_dir))] = content
    return data_files


def load_skillsbench_task(task_dir: str | Path) -> SkillsBenchTask:
    task_path = Path(task_dir)
    instruction_path = task_path / "instruction.md"
    if not instruction_path.exists():
        raise FileNotFoundError(
            f"missing SkillsBench instruction: {instruction_path}"
        )
    skill_text, skill_files = load_skill_text(task_path)
    return SkillsBenchTask(
        task_dir=str(task_path),
        task_name=task_path.name,
        instruction=_read_text(instruction_path).strip(),
        skill_text=skill_text,
        skill_files=skill_files,
        data_files=load_task_data_files(task_path),
    )


def build_task_prompt(
    task: SkillsBenchTask,
    arm: SkillEvalArm,
    *,
    response_mode: str = "json",
    include_data: bool = True,
) -> str:
    """Build a prompt for one eval arm.

    The ReasonCACHE arm deliberately matches ``no-skill`` prompt content; the
    learned prefix is supplied through KV injection instead of prompt text.
    """
    if arm not in ("full-skill", "no-skill", "reasoncache"):
        raise ValueError(f"unknown SkillsBench arm: {arm}")

    parts: list[str] = []
    if arm == "full-skill" and task.skill_text:
        parts.append(f"<skills>\n{task.skill_text}\n</skills>\n")
    parts.append(f"<instruction>\n{task.instruction}\n</instruction>\n")

    if include_data and task.data_files:
        data_parts = [
            f"--- {name} ---\n{content}"
            for name, content in sorted(task.data_files.items())
        ]
        data_body = "\n\n".join(data_parts)
        parts.append(f"<data>\n{data_body}\n</data>\n")

    if response_mode == "json":
        parts.append(
            "Provide your answer as valid JSON. Output ONLY the "
            "JSON, no other text."
        )
    elif response_mode == "python":
        parts.append(
            "Write a complete Python script. Output ONLY the Python "
            "code in a single ```python code block."
        )
    elif response_mode:
        parts.append(response_mode)
    return "\n".join(parts)


def build_chat_messages(
    user_prompt: str,
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]


def build_reasoncache_kv_transfer_config(
    reasoncache_path: str | Path,
    *,
    placeholder_tokens: int,
    placeholder_token_id: int = 0,
    connector_module_path: str = DEFAULT_CONNECTOR_MODULE,
    extra_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if placeholder_tokens <= 0:
        raise ValueError("placeholder_tokens must be positive")
    cfg = {
        "reasoncache_path": str(reasoncache_path),
        "prefix_placeholder_tokens": int(placeholder_tokens),
        "prefix_placeholder_token_id": int(placeholder_token_id),
    }
    if extra_config:
        cfg.update(extra_config)
    return {
        "kv_connector": "ReasonCacheConnector",
        "kv_connector_module_path": connector_module_path,
        "kv_role": "kv_both",
        "kv_connector_extra_config": cfg,
    }


def build_reasoncache_sampling_extra_args(
    *,
    placeholder_tokens: int,
    placeholder_token_id: int = 0,
) -> dict[str, Any]:
    if placeholder_tokens <= 0:
        raise ValueError("placeholder_tokens must be positive")
    return {
        "kv_transfer_params": {
            "prefix_placeholder_tokens": int(placeholder_tokens),
            "prefix_placeholder_token_id": int(placeholder_token_id),
        }
    }


def build_reasoncache_token_prompt(
    prompt_token_ids: list[int],
    *,
    placeholder_tokens: int,
    placeholder_token_id: int = 0,
    prompt_text: str | None = None,
) -> dict[str, Any]:
    if placeholder_tokens <= 0:
        raise ValueError("placeholder_tokens must be positive")
    token_ids = [int(placeholder_token_id)] * int(placeholder_tokens)
    token_ids.extend(int(t) for t in prompt_token_ids)
    prompt: dict[str, Any] = {"prompt_token_ids": token_ids}
    if prompt_text is not None:
        prompt["prompt"] = prompt_text
    return prompt


def build_reasoncache_request_plan(
    *,
    prompt_token_ids: list[int],
    reasoncache_path: str | Path,
    placeholder_tokens: int,
    placeholder_token_id: int = 0,
    prompt_text: str | None = None,
) -> ReasonCacheRequestPlan:
    return ReasonCacheRequestPlan(
        prompt=build_reasoncache_token_prompt(
            prompt_token_ids,
            placeholder_tokens=placeholder_tokens,
            placeholder_token_id=placeholder_token_id,
            prompt_text=prompt_text,
        ),
        sampling_extra_args=build_reasoncache_sampling_extra_args(
            placeholder_tokens=placeholder_tokens,
            placeholder_token_id=placeholder_token_id,
        ),
        kv_transfer_config=build_reasoncache_kv_transfer_config(
            reasoncache_path,
            placeholder_tokens=placeholder_tokens,
            placeholder_token_id=placeholder_token_id,
        ),
        placeholder_tokens=placeholder_tokens,
        placeholder_token_id=placeholder_token_id,
        task_prompt_tokens=len(prompt_token_ids),
        online_prompt_tokens=placeholder_tokens + len(prompt_token_ids),
    )


def infer_reasoncache_placeholder_tokens(
    reasoncache_path: str | Path,
    *,
    block_size: int = 16,
    strict_resource_kind: bool = True,
) -> int:
    """Return the aligned learned-prefix length needed by this vLLM branch."""
    from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_connector import (  # noqa: E501
        align_to_block_size, load_cartridge)

    checkpoint = load_cartridge(str(reasoncache_path))
    resource_kind = checkpoint.get("resource_kind")
    if strict_resource_kind and resource_kind not in (
        "reasoncache",
        "kv_prefix",
    ):
        raise ValueError(
            f"expected ReasonCACHE/KV-prefix checkpoint, got {resource_kind!r}"
        )
    aligned = align_to_block_size(int(checkpoint["num_tokens"]), block_size)
    if aligned <= 0:
        raise ValueError(
            "ReasonCACHE prefix is shorter than one block after alignment"
        )
    return aligned


def load_result_file(path: str | Path) -> ArmEvalResult:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return ArmEvalResult.from_mapping(data)


def result_quality(result: ArmEvalResult) -> float:
    """Return the scalar quality score used by comparison/frontier logic."""
    if result.pass_rate is not None:
        return result.pass_rate
    if result.total > 0:
        return result.passed / result.total
    if result.reward is not None:
        return result.reward
    return 0.0


def compare_arm_results(
    results: list[ArmEvalResult],
    *,
    tolerance: float = 0.0,
) -> dict[str, Any]:
    """Compare ReasonCACHE with full-skill and no-skill baselines."""
    by_arm = {r.arm: r for r in results}
    if "reasoncache" not in by_arm:
        raise ValueError("comparison requires a reasoncache result")
    reasoncache = by_arm["reasoncache"]
    baselines = [
        by_arm[arm] for arm in ("full-skill", "no-skill") if arm in by_arm
    ]
    if not baselines:
        raise ValueError("comparison requires at least one baseline result")

    baseline_best = max(baselines, key=result_quality)
    reasoncache_quality = result_quality(reasoncache)
    best_baseline_quality = result_quality(baseline_best)
    deltas = {
        base.arm: reasoncache_quality - result_quality(base)
        for base in baselines
    }

    prompt_savings_vs_full = None
    full_skill = by_arm.get("full-skill")
    if (
        full_skill is not None
        and full_skill.prompt_tokens is not None
        and reasoncache.prompt_tokens is not None
    ):
        prompt_savings_vs_full = (
            full_skill.prompt_tokens - reasoncache.prompt_tokens
        )

    gets_better_results = all(delta > tolerance for delta in deltas.values())
    preserves_quality = reasoncache_quality + tolerance >= best_baseline_quality
    quality_preserving_prompt_win = (
        preserves_quality
        and prompt_savings_vs_full is not None
        and prompt_savings_vs_full > 0
    )

    return {
        "best_arm": max(results, key=result_quality).arm,
        "best_baseline_arm": baseline_best.arm,
        "reasoncache_quality": reasoncache_quality,
        "best_baseline_quality": best_baseline_quality,
        "quality_delta_vs_baselines": deltas,
        "prompt_token_savings_vs_full_skill": prompt_savings_vs_full,
        "reasoncache_gets_better_results": gets_better_results,
        "reasoncache_preserves_or_improves_quality": preserves_quality,
        "reasoncache_quality_preserving_prompt_win": (
            quality_preserving_prompt_win
        ),
        "reasoncache_better_suited": (
            gets_better_results or quality_preserving_prompt_win
        ),
        "arms": [result.to_dict() for result in results],
    }


def _result_pareto_point(
    result: ArmEvalResult,
    *,
    index: int,
) -> dict[str, Any]:
    return {
        "id": f"{index}:{result.task}:{result.arm}",
        "task": result.task,
        "arm": result.arm,
        "quality": result_quality(result),
        "prompt_tokens": result.prompt_tokens,
        "gen_time_sec": result.gen_time_sec,
        "gen_tokens": result.gen_tokens,
        "passed": result.passed,
        "total": result.total,
    }


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _aggregate_arm_pareto_points(
    results: list[ArmEvalResult],
) -> list[dict[str, Any]]:
    by_arm: dict[str, list[ArmEvalResult]] = defaultdict(list)
    for result in results:
        by_arm[result.arm].append(result)

    points: list[dict[str, Any]] = []
    for arm, arm_results in sorted(by_arm.items()):
        prompt_values = [
            float(r.prompt_tokens)
            for r in arm_results
            if r.prompt_tokens is not None
        ]
        time_values = [
            float(r.gen_time_sec)
            for r in arm_results
            if r.gen_time_sec is not None
        ]
        gen_values = [
            float(r.gen_tokens) for r in arm_results if r.gen_tokens is not None
        ]
        points.append(
            {
                "id": f"aggregate:{arm}",
                "task": "aggregate",
                "arm": arm,
                "quality": _mean_or_none(
                    [result_quality(r) for r in arm_results]
                ),
                "prompt_tokens": _mean_or_none(prompt_values),
                "gen_time_sec": _mean_or_none(time_values),
                "gen_tokens": _mean_or_none(gen_values),
                "result_count": len(arm_results),
                "tasks": sorted({r.task for r in arm_results}),
            }
        )
    return points


def _active_pareto_metrics(
    points: list[dict[str, Any]],
    *,
    include_latency: bool,
    include_gen_tokens: bool,
) -> list[dict[str, str]]:
    metrics = [{"name": "quality", "direction": "maximize"}]
    if points and all(
        point.get("prompt_tokens") is not None for point in points
    ):
        metrics.append({"name": "prompt_tokens", "direction": "minimize"})
    if (
        include_latency
        and points
        and all(point.get("gen_time_sec") is not None for point in points)
    ):
        metrics.append({"name": "gen_time_sec", "direction": "minimize"})
    if (
        include_gen_tokens
        and points
        and all(point.get("gen_tokens") is not None for point in points)
    ):
        metrics.append({"name": "gen_tokens", "direction": "minimize"})
    return metrics


def _dominates(
    challenger: dict[str, Any],
    candidate: dict[str, Any],
    metrics: list[dict[str, str]],
    *,
    tolerance: float,
) -> bool:
    strictly_better = False
    for metric in metrics:
        name = metric["name"]
        challenger_value = float(challenger[name])
        candidate_value = float(candidate[name])
        if metric["direction"] == "maximize":
            if challenger_value + tolerance < candidate_value:
                return False
            if challenger_value > candidate_value + tolerance:
                strictly_better = True
        else:
            if challenger_value > candidate_value + tolerance:
                return False
            if challenger_value < candidate_value - tolerance:
                strictly_better = True
    return strictly_better


def _pareto_scope_analysis(
    scope: str,
    points: list[dict[str, Any]],
    *,
    tolerance: float,
    include_latency: bool,
    include_gen_tokens: bool,
) -> dict[str, Any]:
    metrics = _active_pareto_metrics(
        points,
        include_latency=include_latency,
        include_gen_tokens=include_gen_tokens,
    )
    annotated: list[dict[str, Any]] = []
    for candidate in points:
        dominated_by = [
            challenger["id"]
            for challenger in points
            if challenger["id"] != candidate["id"]
            and _dominates(
                challenger,
                candidate,
                metrics,
                tolerance=tolerance,
            )
        ]
        annotated.append(
            {
                **candidate,
                "dominated_by": dominated_by,
                "on_frontier": not dominated_by,
            }
        )

    frontier = [point for point in annotated if point["on_frontier"]]
    frontier_arms = sorted({point["arm"] for point in frontier})
    return {
        "scope": scope,
        "metrics": metrics,
        "frontier_arms": frontier_arms,
        "reasoncache_on_frontier": "reasoncache" in frontier_arms,
        "full_skill_on_frontier": "full-skill" in frontier_arms,
        "no_skill_on_frontier": "no-skill" in frontier_arms,
        "frontier": frontier,
        "points": annotated,
    }


def _point_for_arm(
    scope_analysis: dict[str, Any],
    arm: str,
) -> dict[str, Any] | None:
    for point in scope_analysis["points"]:
        if point["arm"] == arm:
            return point
    return None


def _id_dominates_point(
    challenger: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
) -> bool:
    if challenger is None or candidate is None:
        return False
    return challenger["id"] in candidate["dominated_by"]


def _regular_vs_reasoncache_summary(
    by_task: dict[str, dict[str, Any]],
    aggregate: dict[str, Any],
) -> dict[str, Any]:
    tasks_where_reasoncache_dominates_full_skill: list[str] = []
    tasks_where_full_skill_dominates_reasoncache: list[str] = []
    tasks_where_both_on_frontier: list[str] = []
    task_summaries: list[dict[str, Any]] = []

    for task, task_analysis in sorted(by_task.items()):
        reasoncache = _point_for_arm(task_analysis, "reasoncache")
        full_skill = _point_for_arm(task_analysis, "full-skill")
        if _id_dominates_point(reasoncache, full_skill):
            tasks_where_reasoncache_dominates_full_skill.append(task)
        if _id_dominates_point(full_skill, reasoncache):
            tasks_where_full_skill_dominates_reasoncache.append(task)
        if (
            reasoncache is not None
            and full_skill is not None
            and reasoncache["on_frontier"]
            and full_skill["on_frontier"]
        ):
            tasks_where_both_on_frontier.append(task)

        prompt_savings = None
        if (
            reasoncache is not None
            and full_skill is not None
            and reasoncache.get("prompt_tokens") is not None
            and full_skill.get("prompt_tokens") is not None
        ):
            prompt_savings = (
                full_skill["prompt_tokens"] - reasoncache["prompt_tokens"]
            )

        latency_savings = None
        if (
            reasoncache is not None
            and full_skill is not None
            and reasoncache.get("gen_time_sec") is not None
            and full_skill.get("gen_time_sec") is not None
        ):
            latency_savings = (
                full_skill["gen_time_sec"] - reasoncache["gen_time_sec"]
            )

        quality_delta = None
        if reasoncache is not None and full_skill is not None:
            quality_delta = reasoncache["quality"] - full_skill["quality"]

        task_summaries.append(
            {
                "task": task,
                "reasoncache_on_frontier": (
                    reasoncache["on_frontier"]
                    if reasoncache is not None
                    else False
                ),
                "full_skill_on_frontier": (
                    full_skill["on_frontier"]
                    if full_skill is not None
                    else False
                ),
                "reasoncache_dominates_full_skill": _id_dominates_point(
                    reasoncache, full_skill
                ),
                "full_skill_dominates_reasoncache": _id_dominates_point(
                    full_skill, reasoncache
                ),
                "reasoncache_quality_delta_vs_full_skill": quality_delta,
                "reasoncache_prompt_token_savings_vs_full_skill": (
                    prompt_savings
                ),
                "reasoncache_latency_savings_sec_vs_full_skill": (
                    latency_savings
                ),
            }
        )

    aggregate_reasoncache = _point_for_arm(aggregate, "reasoncache")
    aggregate_full_skill = _point_for_arm(aggregate, "full-skill")
    return {
        "task_summaries": task_summaries,
        "reasoncache_frontier_task_count": sum(
            1
            for summary in task_summaries
            if summary["reasoncache_on_frontier"]
        ),
        "full_skill_frontier_task_count": sum(
            1 for summary in task_summaries if summary["full_skill_on_frontier"]
        ),
        "tasks_where_reasoncache_dominates_full_skill": (
            tasks_where_reasoncache_dominates_full_skill
        ),
        "tasks_where_full_skill_dominates_reasoncache": (
            tasks_where_full_skill_dominates_reasoncache
        ),
        "tasks_where_both_on_frontier": tasks_where_both_on_frontier,
        "aggregate_reasoncache_on_frontier": (
            aggregate_reasoncache["on_frontier"]
            if aggregate_reasoncache is not None
            else False
        ),
        "aggregate_full_skill_on_frontier": (
            aggregate_full_skill["on_frontier"]
            if aggregate_full_skill is not None
            else False
        ),
        "aggregate_reasoncache_dominates_full_skill": _id_dominates_point(
            aggregate_reasoncache, aggregate_full_skill
        ),
        "aggregate_full_skill_dominates_reasoncache": _id_dominates_point(
            aggregate_full_skill, aggregate_reasoncache
        ),
    }


def analyze_pareto_frontier(
    results: list[ArmEvalResult],
    *,
    tolerance: float = 0.0,
    include_latency: bool = True,
    include_gen_tokens: bool = False,
) -> dict[str, Any]:
    """Build Pareto frontier views for regular skill prompts vs ReasonCACHE.

    The primary frontier is per task: maximize quality while minimizing online
    prompt tokens, plus generation latency when every point in that task has
    timing. The aggregate frontier averages each arm across all provided tasks.
    """
    if not results:
        raise ValueError("Pareto analysis requires at least one result")

    points = [
        _result_pareto_point(result, index=index)
        for index, result in enumerate(results)
    ]
    all_results = _pareto_scope_analysis(
        "all_results",
        points,
        tolerance=tolerance,
        include_latency=include_latency,
        include_gen_tokens=include_gen_tokens,
    )

    points_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for point in points:
        points_by_task[point["task"]].append(point)
    by_task = {
        task: _pareto_scope_analysis(
            task,
            task_points,
            tolerance=tolerance,
            include_latency=include_latency,
            include_gen_tokens=include_gen_tokens,
        )
        for task, task_points in sorted(points_by_task.items())
    }

    aggregate_by_arm = _pareto_scope_analysis(
        "aggregate_by_arm",
        _aggregate_arm_pareto_points(results),
        tolerance=tolerance,
        include_latency=include_latency,
        include_gen_tokens=include_gen_tokens,
    )

    return {
        "tolerance": tolerance,
        "include_latency": include_latency,
        "include_gen_tokens": include_gen_tokens,
        "all_results": all_results,
        "by_task": by_task,
        "aggregate_by_arm": aggregate_by_arm,
        "regular_vs_reasoncache": _regular_vs_reasoncache_summary(
            by_task, aggregate_by_arm
        ),
    }
