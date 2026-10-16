# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
ANALYZER = REPO_ROOT / "benchmarks/kmesh/analyze_results.py"
pytestmark = pytest.mark.skip_global_cleanup


def _load_analyzer():
    spec = importlib.util.spec_from_file_location("kmesh_analyzer", ANALYZER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_bundle(
    root: Path,
    name: str,
    *,
    mode: str,
    replay_seconds: float,
    cached_fraction: float,
    outputs: list[list[int]] | None = None,
    admission_mode: str = "batch",
) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    config = {
        "schema_version": 1,
        "mode": mode,
        "admission_mode": admission_mode,
        "trace_sha256": "trace-sha",
        "trace_id": "trace-id",
        "trace_config": {"case": "same_order_different_offsets"},
        "model": "model",
        "model_revision": "model-revision",
        "tokenizer_revision": "tokenizer-revision",
        "dtype": "bfloat16",
        "max_model_len": 2048,
        "max_tokens": 1,
        "enforce_eager": True,
        "chunked_prefill": False,
        "async_scheduling": False,
        "disable_hybrid_kv_cache_manager": True,
    }
    environment = {
        "schema_version": 1,
        "source": {"sha": "source-sha"},
        "runtime": {"vllm": "version", "device": "test"},
    }
    summary = {
        "schema_version": 1,
        "status": "complete",
        "mode": mode,
        "replay_seconds": replay_seconds,
        "requests_per_second": 2 / replay_seconds,
        "cached_token_fraction": cached_fraction,
        "service_seconds_mean": replay_seconds / 2,
        "generated_token_ids": outputs or [[10], [11]],
    }
    if mode == "b2":
        summary["observer_metrics"] = {
            "source": "vllm_kv_connector_stats_prometheus_delta",
            "invariants": {"all_passed": True},
        }
    (run_dir / "config.json").write_text(json.dumps(config) + "\n")
    (run_dir / "environment.json").write_text(json.dumps(environment) + "\n")
    (run_dir / "summary.json").write_text(json.dumps(summary) + "\n")
    return run_dir


def test_five_equivalent_repetitions_open_timing_gate(tmp_path: Path) -> None:
    analyzer = _load_analyzer()
    for repetition in range(5):
        _write_bundle(
            tmp_path,
            f"b0-{repetition}",
            mode="b0",
            replay_seconds=2.0,
            cached_fraction=0.0,
        )
        _write_bundle(
            tmp_path,
            f"b1-{repetition}",
            mode="b1",
            replay_seconds=1.0,
            cached_fraction=0.8,
        )

    analysis = analyzer.analyze([tmp_path])

    assert analysis["bundle_count"] == 10
    assert len(analysis["comparisons"]) == 1
    comparison = analysis["comparisons"][0]["b0_b1"]
    assert comparison["output_equivalent"] is True
    assert comparison["apc_hit_evidence"] is True
    assert comparison["performance_claim_ready"] is True
    assert comparison["timing_metric"] == "replay_seconds"
    assert comparison["median_speedup_b1_over_b0"] == 2.0
    assert comparison["median_replay_speedup_b1_over_b0"] == 2.0


def test_sparse_or_mismatched_results_keep_timing_gate_closed(tmp_path: Path) -> None:
    analyzer = _load_analyzer()
    _write_bundle(
        tmp_path,
        "b0",
        mode="b0",
        replay_seconds=2.0,
        cached_fraction=0.0,
    )
    _write_bundle(
        tmp_path,
        "b1",
        mode="b1",
        replay_seconds=1.0,
        cached_fraction=0.8,
        outputs=[[99]],
    )

    comparison = analyzer.analyze([tmp_path])["comparisons"][0]["b0_b1"]

    assert comparison["output_equivalent"] is False
    assert comparison["performance_claim_ready"] is False
    assert comparison["median_replay_speedup_b1_over_b0"] is None


def test_paced_runs_compare_service_not_pacing_time(tmp_path: Path) -> None:
    analyzer = _load_analyzer()
    for repetition in range(5):
        _write_bundle(
            tmp_path,
            f"b0-{repetition}",
            mode="b0",
            replay_seconds=10.0,
            cached_fraction=0.0,
            admission_mode="serial_paced",
        )
        _write_bundle(
            tmp_path,
            f"b1-{repetition}",
            mode="b1",
            replay_seconds=5.0,
            cached_fraction=0.8,
            admission_mode="serial_paced",
        )

    comparison = analyzer.analyze([tmp_path])["comparisons"][0]["b0_b1"]

    assert comparison["timing_metric"] == "service_seconds_mean"
    assert comparison["median_speedup_b1_over_b0"] == 2.0
    assert comparison["median_replay_speedup_b1_over_b0"] is None


def test_b0_b2_behavior_gate_requires_equivalent_outputs_and_zero_hits(
    tmp_path: Path,
) -> None:
    analyzer = _load_analyzer()
    _write_bundle(
        tmp_path,
        "b0",
        mode="b0",
        replay_seconds=2.0,
        cached_fraction=0.0,
    )
    _write_bundle(
        tmp_path,
        "b2",
        mode="b2",
        replay_seconds=2.1,
        cached_fraction=0.0,
    )

    comparison = analyzer.analyze([tmp_path])["comparisons"][0]["b0_b2"]

    assert comparison["output_equivalent"] is True
    assert comparison["b2_zero_cached_tokens"] is True
    assert comparison["observer_evidence_all_passed"] is True
    assert comparison["behavior_claim_ready"] is True


def test_b0_b2_behavior_gate_rejects_changed_output_and_cache_hits(
    tmp_path: Path,
) -> None:
    analyzer = _load_analyzer()
    _write_bundle(
        tmp_path,
        "b0",
        mode="b0",
        replay_seconds=2.0,
        cached_fraction=0.0,
    )
    _write_bundle(
        tmp_path,
        "b2",
        mode="b2",
        replay_seconds=2.1,
        cached_fraction=0.1,
        outputs=[[99], [100]],
    )

    comparison = analyzer.analyze([tmp_path])["comparisons"][0]["b0_b2"]

    assert comparison["output_equivalent"] is False
    assert comparison["b2_zero_cached_tokens"] is False
    assert comparison["behavior_claim_ready"] is False


def test_failure_bundle_is_reported(tmp_path: Path) -> None:
    analyzer = _load_analyzer()
    run_dir = tmp_path / "failed"
    run_dir.mkdir()
    (run_dir / "config.json").write_text("{}\n")
    (run_dir / "environment.json").write_text("{}\n")
    (run_dir / "failure.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "exception_type": "RuntimeError",
                "message": "boom",
            }
        )
        + "\n"
    )

    analysis = analyzer.analyze([tmp_path])

    assert analysis["failure_count"] == 1
    assert analysis["failures"][0]["exception_type"] == "RuntimeError"
    assert not analysis["comparisons"]


def test_bundle_requires_exactly_one_terminal_record(tmp_path: Path) -> None:
    analyzer = _load_analyzer()
    run_dir = _write_bundle(
        tmp_path,
        "ambiguous",
        mode="b0",
        replay_seconds=1.0,
        cached_fraction=0.0,
    )
    (run_dir / "failure.json").write_text("{}\n")

    with pytest.raises(analyzer.AnalysisInputError, match="exactly one"):
        analyzer.analyze([run_dir])
