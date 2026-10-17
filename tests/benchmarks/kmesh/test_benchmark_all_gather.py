# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = REPO_ROOT / "benchmarks/kmesh/benchmark_all_gather.py"
pytestmark = pytest.mark.skip_global_cleanup


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location("kmesh_benchmark", BENCHMARK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_fixture(directory: Path) -> tuple[Path, Path]:
    trace_path = directory / "trace.jsonl"
    records = []
    for index in range(2):
        request_id = f"request-{index}"
        lineage = {
            "agent_id": f"agent-{index}",
            "execution_id": "execution-0",
            "group_kind": "agent_round",
            "group_id": "group-0",
            "round_id": 0,
            "branch_id": "root",
            "parent_branch_id": None,
            "branch_path": ["root"],
            "state_kind": "kv_cache",
            "segment_encoding": "sideband",
        }
        segments = [
            {
                "segment_id": f"private-{index}",
                "kind": "private_history",
                "token_start": 0,
                "token_end": 1,
                "logical_order": 0,
                "source_id": f"agent-{index}",
                "shareable": False,
                "lineage_id": f"private-lineage-{index}",
                "length": 1,
            },
            {
                "segment_id": "shared-0",
                "kind": "shared_update",
                "token_start": 1,
                "token_end": 3,
                "logical_order": 1,
                "source_id": "coordinator",
                "shareable": True,
                "lineage_id": "shared-lineage",
                "length": 2,
            },
            {
                "segment_id": f"task-{index}",
                "kind": "task",
                "token_start": 3,
                "token_end": 4,
                "logical_order": 2,
                "source_id": "coordinator",
                "shareable": False,
                "lineage_id": f"task-lineage-{index}",
                "length": 1,
            },
        ]
        context = {
            "schema_version": 1,
            "enabled": True,
            "execution_id": lineage["execution_id"],
            "group_kind": lineage["group_kind"],
            "group_id": lineage["group_id"],
            "round_id": lineage["round_id"],
            "branch_id": lineage["branch_id"],
            "parent_branch_id": lineage["parent_branch_id"],
            "state_kind": lineage["state_kind"],
            "cache_domain": "fixture",
            "segment_encoding": lineage["segment_encoding"],
            "segments": [
                {
                    key: segment[key]
                    for key in (
                        "segment_id",
                        "kind",
                        "token_start",
                        "token_end",
                        "logical_order",
                        "source_id",
                        "shareable",
                        "lineage_id",
                    )
                }
                for segment in segments
            ],
        }
        compatibility_key = {
            "execution_id": lineage["execution_id"],
            "group_id": lineage["group_id"],
            "round_id": lineage["round_id"],
            "model_id": "synthetic:model",
            "model_revision": "model-revision",
            "tokenizer_id_or_revision": "tokenizer-revision",
            "dtype": "bfloat16",
            "kv_cache_dtype": "bfloat16",
            "kv_layout": "paged-kv",
            "attention_kind": "full_attention",
            "rope_configuration": "default",
            "attention_backend": "reference",
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "lora_adapter_identity": None,
            "cache_salt": None,
            "cache_domain": "fixture",
            "active_prompt_length": 4,
            "cached_span": [0, 0],
            "prefill_chunk_span": [0, 4],
        }
        records.append(
            {
                "schema_version": 1,
                "trace_id": "fixture",
                "request_id": request_id,
                "sequence_index": index,
                "lineage": lineage,
                "compatibility_key": compatibility_key,
                "kmesh_context": context,
                "prompt_token_ids": [10 + index, 20, 21, 30 + index],
                "segments": segments,
            }
        )
    trace_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )
    digest = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "trace_file": trace_path.name,
                "trace_sha256": digest,
                "config": {
                    "block_size": 32,
                    "mode": "prefill_focus",
                    "qps": 1,
                },
                "counts": {"requests": len(records)},
                "oracle": {
                    "compatible_shared_tokens": 2,
                    "aggregate": {
                        "eligible_structural_shared_tokens_total": 2,
                        "compatible_pair_count": 1,
                    },
                },
                "observer_expectation": {
                    "contract": "kmesh_observer_supported_scope_v1",
                    "supported_request_count": len(records),
                    "required_fallback_counts": {},
                    "batch_group_presence_expected": True,
                    "serial_paced_group_presence_expected": False,
                },
            }
        )
        + "\n"
    )
    return trace_path, manifest_path


def test_load_trace_rejects_tampering(tmp_path: Path) -> None:
    module = _load_benchmark_module()
    trace_path, manifest_path = _write_fixture(tmp_path)
    trace_path.write_text(trace_path.read_text() + "\n")

    with pytest.raises(module.BenchmarkInputError, match="SHA-256"):
        module.load_trace(trace_path, manifest_path)


def test_load_trace_rejects_overlapping_segments(tmp_path: Path) -> None:
    module = _load_benchmark_module()
    trace_path, manifest_path = _write_fixture(tmp_path)
    lines = trace_path.read_text().splitlines()
    first = json.loads(lines[0])
    first["segments"][1]["token_start"] = 0
    first["segments"][1]["length"] = 3
    lines[0] = json.dumps(first, sort_keys=True)
    trace_path.write_text("\n".join(lines) + "\n")
    manifest = json.loads(manifest_path.read_text())
    manifest["trace_sha256"] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest) + "\n")

    with pytest.raises(module.BenchmarkInputError, match="invalid range"):
        module.load_trace(trace_path, manifest_path)


def test_load_trace_rejects_wire_analytic_drift(tmp_path: Path) -> None:
    module = _load_benchmark_module()
    trace_path, manifest_path = _write_fixture(tmp_path)
    lines = trace_path.read_text().splitlines()
    first = json.loads(lines[0])
    first["kmesh_context"]["segments"][1]["segment_id"] = "wrong-shared-id"
    lines[0] = json.dumps(first, sort_keys=True)
    trace_path.write_text("\n".join(lines) + "\n")
    manifest = json.loads(manifest_path.read_text())
    manifest["trace_sha256"] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest) + "\n")

    with pytest.raises(module.BenchmarkInputError, match="disagrees with analytics"):
        module.load_trace(trace_path, manifest_path)


def test_load_trace_rejects_compatibility_context_drift(tmp_path: Path) -> None:
    module = _load_benchmark_module()
    trace_path, manifest_path = _write_fixture(tmp_path)
    lines = trace_path.read_text().splitlines()
    first = json.loads(lines[0])
    first["compatibility_key"]["cache_domain"] = "other-domain"
    lines[0] = json.dumps(first, sort_keys=True)
    trace_path.write_text("\n".join(lines) + "\n")
    manifest = json.loads(manifest_path.read_text())
    manifest["trace_sha256"] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest) + "\n")

    with pytest.raises(module.BenchmarkInputError, match="disagrees with context"):
        module.load_trace(trace_path, manifest_path)


def test_load_trace_rejects_mixed_trace_identity(tmp_path: Path) -> None:
    module = _load_benchmark_module()
    trace_path, manifest_path = _write_fixture(tmp_path)
    lines = trace_path.read_text().splitlines()
    second = json.loads(lines[1])
    second["trace_id"] = "other-trace"
    lines[1] = json.dumps(second, sort_keys=True)
    trace_path.write_text("\n".join(lines) + "\n")
    manifest = json.loads(manifest_path.read_text())
    manifest["trace_sha256"] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest) + "\n")

    with pytest.raises(module.BenchmarkInputError, match="mixed trace IDs"):
        module.load_trace(trace_path, manifest_path)


def test_dry_run_writes_machine_readable_bundle(tmp_path: Path) -> None:
    trace_path, manifest_path = _write_fixture(tmp_path)
    output_dir = tmp_path / "result"

    completed = subprocess.run(
        [
            sys.executable,
            str(BENCHMARK),
            "--trace",
            str(trace_path),
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
            "--mode",
            "b1",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads((output_dir / "summary.json").read_text())
    config = json.loads((output_dir / "config.json").read_text())
    environment = json.loads((output_dir / "environment.json").read_text())
    assert summary["status"] == "validated"
    assert summary["request_count"] == 2
    assert summary["structural_oracle"]["compatible_shared_tokens"] == 2
    assert config["mode"] == "b1"
    assert environment["dry_run"] is True
    assert json.loads(completed.stdout)["status"] == "validated"


def test_b2_dry_run_records_strict_noop_observer_config(tmp_path: Path) -> None:
    trace_path, manifest_path = _write_fixture(tmp_path)
    output_dir = tmp_path / "b2-result"

    subprocess.run(
        [
            sys.executable,
            str(BENCHMARK),
            "--trace",
            str(trace_path),
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
            "--mode",
            "b2",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    config = json.loads((output_dir / "config.json").read_text())
    summary = json.loads((output_dir / "summary.json").read_text())
    connector = config["kv_transfer_config"]
    kmesh = connector["kv_connector_extra_config"]["kmesh"]
    assert connector["kv_connector"] == "KMeshConnector"
    assert connector["kv_role"] == "kv_both"
    assert kmesh == {
        "enabled": True,
        "fail_open": False,
        "max_group_size": 10,
        "max_group_wait_ms": 0,
        "mode": "observe",
        "pic_backend": "noop",
        "require_same_active_length": True,
        "segment_encoding": "sideband",
        "storage_mode": "none",
        "strict_validation": True,
    }
    assert config["engine_stats_enabled"] is True
    assert config["disable_hybrid_kv_cache_manager"] is True
    assert summary["observer_metrics"]["status"] == "not_executed_dry_run"
    assert summary["observer_metrics"]["executed_kv_reuse"] is False


def test_b2_engine_options_disable_apc_and_enable_public_metrics() -> None:
    module = _load_benchmark_module()
    args = module.build_parser().parse_args(
        [
            "--trace",
            "trace.jsonl",
            "--manifest",
            "manifest.json",
            "--output-dir",
            "result",
            "--mode",
            "b2",
        ]
    )

    options = module._engine_kwargs(args, 32)

    assert options["enable_prefix_caching"] is False
    assert options["disable_log_stats"] is False
    assert options["enable_chunked_prefill"] is False
    assert options["async_scheduling"] is False
    assert options["disable_hybrid_kv_cache_manager"] is True
    assert options["kv_transfer_config"] == module._b2_kv_transfer_config()


@pytest.mark.parametrize("control", ["chunked_prefill", "async_scheduling"])
def test_b2_rejects_scheduler_modes_that_break_same_step_scope(control: str) -> None:
    module = _load_benchmark_module()
    args = SimpleNamespace(
        mode="b2",
        chunked_prefill=control == "chunked_prefill",
        async_scheduling=control == "async_scheduling",
    )

    with pytest.raises(module.BenchmarkInputError, match="does not support"):
        module._validate_mode_controls(args)


def _public_metric(module, field: str, value: int, **labels: str) -> SimpleNamespace:
    return SimpleNamespace(
        name=f"{module._OBSERVER_PROM_PREFIX}{field}",
        labels=labels,
        value=value,
    )


def test_observer_metric_projection_and_delta_validate_accounting() -> None:
    module = _load_benchmark_module()
    before = [
        _public_metric(module, "planner_calls", 1),
        _public_metric(module, "observed_requests", 1),
        SimpleNamespace(
            name=module._OBSERVER_REASONS_METRIC,
            labels={"kind": "fallback", "reason": "no_metadata"},
            value=1,
        ),
    ]
    after = [
        _public_metric(module, "planner_calls", 3),
        _public_metric(module, "observed_requests", 5),
        _public_metric(module, "candidate_requests", 4),
        _public_metric(module, "formed_groups", 1),
        _public_metric(module, "shared_segments", 1),
        _public_metric(module, "shared_tokens", 32),
        _public_metric(module, "prospective_reused_segments", 1),
        _public_metric(module, "prospective_reused_tokens", 32),
        _public_metric(module, "structural_full_blocks", 1),
        _public_metric(module, "structural_full_block_tokens", 32),
        _public_metric(module, "prospective_reused_full_blocks", 1),
        _public_metric(module, "prospective_reused_full_block_tokens", 32),
        _public_metric(module, "planner_overhead_ns", 99),
        _public_metric(module, "block_size", 32),
        _public_metric(module, "max_group_size", 10),
        SimpleNamespace(
            name=module._OBSERVER_REASONS_METRIC,
            labels={"kind": "fallback", "reason": "no_metadata"},
            value=1,
        ),
        SimpleNamespace(
            name=module._OBSERVER_REASONS_METRIC,
            labels={"kind": "fallback", "reason": "singleton_group"},
            value=2,
        ),
        SimpleNamespace(
            name=module._OBSERVER_REASONS_METRIC,
            labels={"kind": "compatibility", "reason": "round_mismatch"},
            value=3,
        ),
        SimpleNamespace(
            name=module._OBSERVER_GROUP_SIZES_METRIC,
            labels={"group_size": "2"},
            value=1,
        ),
    ]

    observed = module._observer_metrics_delta(
        before,
        after,
        {
            "config": {"block_size": 32},
            "counts": {"requests": 4},
            "oracle": {
                "aggregate": {
                    "eligible_structural_shared_tokens_total": 32,
                    "compatible_pair_count": 1,
                }
            },
            "observer_expectation": {
                "contract": "kmesh_observer_supported_scope_v1",
                "supported_request_count": 4,
                "required_fallback_counts": {},
                "batch_group_presence_expected": True,
                "serial_paced_group_presence_expected": False,
            },
        },
        admission_mode="batch",
        cached_tokens=0,
    )

    assert observed["planner_calls"] == 2
    assert observed["observed_requests"] == 4
    assert observed["fallback_counts"] == {"singleton_group": 2}
    assert observed["compatibility_failure_counts"] == {"round_mismatch": 3}
    assert observed["group_size_counts"] == {"2": 1}
    assert observed["invariants"]["all_passed"] is True
    assert observed["executed_kv_reuse"] is False

    with pytest.raises(module.BenchmarkInputError, match="zero_cached_tokens"):
        module._observer_metrics_delta(
            before,
            after,
            {
                "config": {"block_size": 32},
                "counts": {"requests": 4},
                "oracle": {
                    "aggregate": {
                        "eligible_structural_shared_tokens_total": 32,
                        "compatible_pair_count": 1,
                    }
                },
                "observer_expectation": {
                    "contract": "kmesh_observer_supported_scope_v1",
                    "supported_request_count": 4,
                    "required_fallback_counts": {},
                    "batch_group_presence_expected": True,
                    "serial_paced_group_presence_expected": False,
                },
            },
            admission_mode="batch",
            cached_tokens=1,
        )

    with pytest.raises(
        module.BenchmarkInputError, match="observed_requests_match_manifest"
    ):
        module._observer_metrics_delta(
            before,
            after,
            {
                "config": {"block_size": 32},
                "counts": {"requests": 5},
                "oracle": {
                    "aggregate": {
                        "eligible_structural_shared_tokens_total": 32,
                        "compatible_pair_count": 1,
                    }
                },
                "observer_expectation": {
                    "contract": "kmesh_observer_supported_scope_v1",
                    "supported_request_count": 5,
                    "required_fallback_counts": {},
                    "batch_group_presence_expected": True,
                    "serial_paced_group_presence_expected": False,
                },
            },
            admission_mode="batch",
            cached_tokens=0,
        )


def test_observer_delta_rejects_missing_expected_batch_group() -> None:
    module = _load_benchmark_module()
    after = [
        _public_metric(module, "planner_calls", 1),
        _public_metric(module, "observed_requests", 5),
        _public_metric(module, "block_size", 32),
        _public_metric(module, "max_group_size", 10),
        SimpleNamespace(
            name=module._OBSERVER_REASONS_METRIC,
            labels={"kind": "fallback", "reason": "singleton_group"},
            value=5,
        ),
    ]
    manifest = {
        "config": {"block_size": 32},
        "counts": {"requests": 5},
        "oracle": {
            "aggregate": {
                "eligible_structural_shared_tokens_total": 4160,
                "compatible_pair_count": 10,
            }
        },
        "observer_expectation": {
            "contract": "kmesh_observer_supported_scope_v1",
            "supported_request_count": 5,
            "required_fallback_counts": {},
            "batch_group_presence_expected": True,
            "serial_paced_group_presence_expected": False,
        },
    }

    with pytest.raises(
        module.BenchmarkInputError,
        match="group_presence_matches_observer_expectation",
    ):
        module._observer_metrics_delta(
            [],
            after,
            manifest,
            admission_mode="batch",
            cached_tokens=0,
        )

    serial = module._observer_metrics_delta(
        [],
        after,
        manifest,
        admission_mode="serial_paced",
        cached_tokens=0,
    )
    assert serial["formed_groups"] == 0
    assert serial["oracle_expectation"]["trace_has_eligible_overlap"] is True
    assert serial["oracle_expectation"]["expected_formed_group_presence"] is False
    assert serial["invariants"]["all_passed"] is True


def test_observer_delta_accepts_unsupported_cot_with_typed_fallbacks() -> None:
    module = _load_benchmark_module()
    after = [
        _public_metric(module, "planner_calls", 1),
        _public_metric(module, "observed_requests", 3),
        _public_metric(module, "block_size", 32),
        _public_metric(module, "max_group_size", 10),
        SimpleNamespace(
            name=module._OBSERVER_REASONS_METRIC,
            labels={"kind": "fallback", "reason": "group_kind_unsupported"},
            value=3,
        ),
    ]
    manifest = {
        "config": {"block_size": 32},
        "counts": {"requests": 3},
        "oracle": {
            "aggregate": {
                "eligible_structural_shared_tokens_total": 128,
                "compatible_pair_count": 3,
            }
        },
        "observer_expectation": {
            "contract": "kmesh_observer_supported_scope_v1",
            "supported_request_count": 0,
            "required_fallback_counts": {"group_kind_unsupported": 3},
            "batch_group_presence_expected": False,
            "serial_paced_group_presence_expected": False,
        },
    }

    observed = module._observer_metrics_delta(
        [],
        after,
        manifest,
        admission_mode="batch",
        cached_tokens=0,
    )

    assert observed["formed_groups"] == 0
    assert observed["fallback_counts"] == {"group_kind_unsupported": 3}
    assert observed["oracle_expectation"]["trace_has_eligible_overlap"] is True
    assert observed["invariants"]["all_passed"] is True


def test_output_directory_must_be_empty(tmp_path: Path) -> None:
    module = _load_benchmark_module()
    output_dir = tmp_path / "result"
    output_dir.mkdir()
    (output_dir / "existing.json").write_text("{}\n")

    with pytest.raises(module.BenchmarkInputError, match="not empty"):
        module._prepare_output_dir(output_dir)
