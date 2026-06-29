# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import json
from pathlib import Path

import pytest

from vllm.benchmarks.reasoncache_skills import (
    ArmEvalResult, analyze_pareto_frontier,
    build_reasoncache_kv_transfer_config, build_reasoncache_request_plan,
    build_task_prompt, compare_arm_results,
    infer_reasoncache_placeholder_tokens, load_result_file,
    load_skillsbench_task)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _make_task(tmp_path: Path) -> Path:
    task_dir = tmp_path / "exoplanet-detection-period"
    _write(
        task_dir / "instruction.md",
        "Find the period of the transit and return JSON.",
    )
    _write(
        task_dir / "environment" / "skills" / "bls" / "SKILL.md",
        "Use BoxLeastSquares for transit-like dips.",
    )
    _write(
        task_dir / "environment" / "skills" / "validation" / "SKILL.md",
        "Validate odd/even transit depths and SNR.",
    )
    _write(
        task_dir / "environment" / "light_curve.txt",
        "time flux\n0.0 1.0\n1.0 0.99\n",
    )
    _write(task_dir / "environment" / "Dockerfile", "FROM python:3.12\n")
    _write(
        task_dir / "tests" / "test_outputs.py", "def test_placeholder(): pass\n"
    )
    return task_dir


def test_loads_same_skillsbench_skill_files(tmp_path: Path) -> None:
    task = load_skillsbench_task(_make_task(tmp_path))

    assert task.task_name == "exoplanet-detection-period"
    assert task.skill_files == [
        "environment/skills/bls/SKILL.md",
        "environment/skills/validation/SKILL.md",
    ]
    assert "BoxLeastSquares" in task.skill_text
    assert "odd/even" in task.skill_text
    assert task.data_files == {
        "light_curve.txt": "time flux\n0.0 1.0\n1.0 0.99\n"
    }


def test_reasoncache_prompt_reuses_no_skill_prompt(tmp_path: Path) -> None:
    task = load_skillsbench_task(_make_task(tmp_path))

    full_skill = build_task_prompt(task, "full-skill")
    no_skill = build_task_prompt(task, "no-skill")
    reasoncache = build_task_prompt(task, "reasoncache")

    assert "<skills>" in full_skill
    assert "BoxLeastSquares" in full_skill
    assert "<skills>" not in no_skill
    assert "<skills>" not in reasoncache
    assert reasoncache == no_skill


def test_reasoncache_config_and_request_plan() -> None:
    plan = build_reasoncache_request_plan(
        prompt_token_ids=[101, 102, 103],
        reasoncache_path="/tmp/reasoncache.pt",
        placeholder_tokens=32,
        placeholder_token_id=0,
        prompt_text="rendered no-skill prompt",
    )

    assert plan.placeholder_tokens == 32
    assert plan.task_prompt_tokens == 3
    assert plan.online_prompt_tokens == 35
    assert plan.prompt["prompt_token_ids"][:32] == [0] * 32
    assert plan.prompt["prompt_token_ids"][32:] == [101, 102, 103]
    assert plan.sampling_extra_args == {
        "kv_transfer_params": {
            "prefix_placeholder_tokens": 32,
            "prefix_placeholder_token_id": 0,
        }
    }
    assert plan.kv_transfer_config["kv_connector"] == "ReasonCacheConnector"
    assert plan.kv_transfer_config["kv_role"] == "kv_both"
    extra = plan.kv_transfer_config["kv_connector_extra_config"]
    assert extra["reasoncache_path"] == "/tmp/reasoncache.pt"
    assert extra["prefix_placeholder_tokens"] == 32


def test_reasoncache_config_rejects_zero_placeholder_tokens() -> None:
    with pytest.raises(ValueError, match="placeholder_tokens"):
        build_reasoncache_kv_transfer_config(
            "/tmp/reasoncache.pt",
            placeholder_tokens=0,
        )


def test_infers_aligned_reasoncache_placeholder_tokens(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    ckpt: dict[str, dict[str, list[object]]] = {
        "reasoncache": {
            "prefix_keys": [],
            "prefix_values": [],
        }
    }
    for _ in range(2):
        ckpt["reasoncache"]["prefix_keys"].append(torch.randn(1, 2, 34, 8))
        ckpt["reasoncache"]["prefix_values"].append(torch.randn(1, 2, 34, 8))
    path = tmp_path / "reasoncache.pt"
    torch.save(ckpt, path)

    assert infer_reasoncache_placeholder_tokens(path, block_size=16) == 32


def test_compare_marks_reasoncache_better_results() -> None:
    results = [
        ArmEvalResult(
            task="exoplanet",
            arm="full-skill",
            passed=3,
            total=4,
            pass_rate=0.75,
            prompt_tokens=2000,
        ),
        ArmEvalResult(
            task="exoplanet",
            arm="no-skill",
            passed=2,
            total=4,
            pass_rate=0.5,
            prompt_tokens=400,
        ),
        ArmEvalResult(
            task="exoplanet",
            arm="reasoncache",
            passed=4,
            total=4,
            pass_rate=1.0,
            prompt_tokens=450,
        ),
    ]

    comparison = compare_arm_results(results)

    assert comparison["best_arm"] == "reasoncache"
    assert comparison["reasoncache_gets_better_results"] is True
    assert comparison["reasoncache_better_suited"] is True
    assert comparison["quality_delta_vs_baselines"]["full-skill"] == 0.25


def test_compare_marks_quality_preserving_prompt_win() -> None:
    results = [
        ArmEvalResult(
            task="exoplanet",
            arm="full-skill",
            pass_rate=0.75,
            prompt_tokens=2000,
        ),
        ArmEvalResult(
            task="exoplanet",
            arm="reasoncache",
            pass_rate=0.75,
            prompt_tokens=450,
        ),
    ]

    comparison = compare_arm_results(results)

    assert comparison["reasoncache_gets_better_results"] is False
    assert comparison["reasoncache_quality_preserving_prompt_win"] is True
    assert comparison["reasoncache_better_suited"] is True
    assert comparison["prompt_token_savings_vs_full_skill"] == 1550


def test_pareto_marks_reasoncache_frontier_when_it_dominates_full_skill() -> (
    None
):
    results = [
        ArmEvalResult(
            task="exoplanet",
            arm="full-skill",
            pass_rate=0.75,
            prompt_tokens=2000,
            gen_time_sec=20.0,
        ),
        ArmEvalResult(
            task="exoplanet",
            arm="no-skill",
            pass_rate=0.5,
            prompt_tokens=400,
            gen_time_sec=8.0,
        ),
        ArmEvalResult(
            task="exoplanet",
            arm="reasoncache",
            pass_rate=0.75,
            prompt_tokens=450,
            gen_time_sec=10.0,
        ),
    ]

    analysis = analyze_pareto_frontier(results)
    task = analysis["by_task"]["exoplanet"]
    full_skill = next(
        point for point in task["points"] if point["arm"] == "full-skill"
    )
    reasoncache = next(
        point for point in task["points"] if point["arm"] == "reasoncache"
    )

    assert reasoncache["on_frontier"] is True
    assert full_skill["on_frontier"] is False
    assert reasoncache["id"] in full_skill["dominated_by"]
    summary = analysis["regular_vs_reasoncache"]
    assert summary["tasks_where_reasoncache_dominates_full_skill"] == [
        "exoplanet"
    ]
    assert summary["aggregate_reasoncache_dominates_full_skill"] is True


def test_pareto_keeps_full_skill_frontier_when_quality_is_higher() -> None:
    results = [
        ArmEvalResult(
            task="exoplanet",
            arm="full-skill",
            pass_rate=0.9,
            prompt_tokens=2000,
            gen_time_sec=20.0,
        ),
        ArmEvalResult(
            task="exoplanet",
            arm="reasoncache",
            pass_rate=0.75,
            prompt_tokens=450,
            gen_time_sec=10.0,
        ),
    ]

    analysis = analyze_pareto_frontier(results)
    task = analysis["by_task"]["exoplanet"]

    assert task["frontier_arms"] == ["full-skill", "reasoncache"]
    assert analysis["regular_vs_reasoncache"][
        "tasks_where_both_on_frontier"
    ] == ["exoplanet"]


def test_loads_historical_result_json_shape(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "task": "exoplanet",
                "arm": "reasoncache",
                "passed": 4,
                "total": 4,
                "prompt_tokens": 450,
                "total_gen_time_sec": 12.5,
                "skill_sha256": "abc",
            }
        )
    )

    result = load_result_file(result_path)

    assert result.arm == "reasoncache"
    assert result.pass_rate == 1.0
    assert result.reward == 1.0
    assert result.gen_time_sec == 12.5
    assert result.metadata["skill_sha256"] == "abc"
