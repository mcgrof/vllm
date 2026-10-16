# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay a deterministic KMesh All-Gather trace through offline vLLM.

The module deliberately imports vLLM only after trace validation and result
bundle creation. ``--dry-run`` therefore validates generated workloads on a
machine without a model runtime or GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RESULT_SCHEMA_VERSION = 1
TRACE_SCHEMA_VERSION = 1
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
B2_MAX_GROUP_SIZE = 10

_OBSERVER_METRICS_SOURCE = "vllm_kv_connector_stats_prometheus_delta"
_OBSERVER_METRICS_SEMANTICS = "sum_of_scheduler_step_prospective_observations"
_OBSERVER_OPPORTUNITY_SCOPE = "deterministic_selected_plan"
_OBSERVER_PROM_PREFIX = "vllm:kmesh_observer_"
_OBSERVER_COUNTER_FIELDS = (
    "planner_calls",
    "observed_requests",
    "candidate_requests",
    "formed_groups",
    "shared_segments",
    "shared_tokens",
    "prospective_reused_segments",
    "prospective_reused_tokens",
    "structural_full_blocks",
    "structural_full_block_tokens",
    "prospective_reused_full_blocks",
    "prospective_reused_full_block_tokens",
    "planner_overhead_ns",
)
_OBSERVER_GAUGE_FIELDS = ("block_size", "max_group_size")
_OBSERVER_REASONS_METRIC = f"{_OBSERVER_PROM_PREFIX}reasons"
_OBSERVER_GROUP_SIZES_METRIC = f"{_OBSERVER_PROM_PREFIX}formed_groups_by_size"

_WIRE_SEGMENT_FIELDS = frozenset(
    {
        "segment_id",
        "kind",
        "token_start",
        "token_end",
        "logical_order",
        "source_id",
        "shareable",
        "lineage_id",
    }
)
_CONTEXT_FIELDS = frozenset(
    {
        "schema_version",
        "enabled",
        "execution_id",
        "group_kind",
        "group_id",
        "round_id",
        "branch_id",
        "parent_branch_id",
        "state_kind",
        "cache_domain",
        "segment_encoding",
        "segments",
    }
)
_COMPATIBILITY_FIELDS = frozenset(
    {
        "execution_id",
        "group_id",
        "round_id",
        "model_id",
        "model_revision",
        "tokenizer_id_or_revision",
        "dtype",
        "kv_cache_dtype",
        "kv_layout",
        "attention_kind",
        "rope_configuration",
        "attention_backend",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "lora_adapter_identity",
        "cache_salt",
        "cache_domain",
        "active_prompt_length",
        "cached_span",
        "prefill_chunk_span",
    }
)


class BenchmarkInputError(ValueError):
    """Raised when a trace or manifest violates the replay contract."""


def _b2_kv_transfer_config() -> dict[str, Any]:
    """Return the strict, no-op observer configuration used by B2."""
    return {
        "kv_connector": "KMeshConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "kmesh": {
                "enabled": True,
                "mode": "observe",
                "segment_encoding": "sideband",
                "pic_backend": "noop",
                "max_group_size": B2_MAX_GROUP_SIZE,
                "max_group_wait_ms": 0,
                "require_same_active_length": True,
                "fail_open": False,
                "strict_validation": True,
                "storage_mode": "none",
            }
        },
    }


def _validate_mode_controls(args: argparse.Namespace) -> None:
    if args.mode == "b2" and args.chunked_prefill:
        raise BenchmarkInputError("B2 observer mode does not support chunked prefill")
    if args.mode == "b2" and args.async_scheduling:
        raise BenchmarkInputError("B2 observer mode does not support async scheduling")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkInputError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BenchmarkInputError(f"expected a JSON object in {path}")
    return value


def _require_int_list(value: Any, *, field: str) -> list[int]:
    if (
        not isinstance(value, list)
        or not value
        or any(type(token_id) is not int or token_id < 0 for token_id in value)
    ):
        raise BenchmarkInputError(
            f"{field} must be a non-empty list of non-negative integers"
        )
    return value


def _validate_segments(
    record: dict[str, Any], prompt_length: int
) -> list[dict[str, Any]]:
    segments = record.get("segments")
    if not isinstance(segments, list) or not segments:
        raise BenchmarkInputError(
            f"request {record.get('request_id')!r} has no segment list"
        )

    previous_end = 0
    seen_ids: set[str] = set()
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise BenchmarkInputError(f"segment {index} must be an object")
        segment_id = segment.get("segment_id")
        if not isinstance(segment_id, str) or not segment_id:
            raise BenchmarkInputError(f"segment {index} has no stable identifier")
        if segment_id in seen_ids:
            raise BenchmarkInputError(f"duplicate segment ID {segment_id!r}")
        seen_ids.add(segment_id)

        start = segment.get("token_start")
        end = segment.get("token_end")
        if type(start) is not int or type(end) is not int:
            raise BenchmarkInputError(
                f"segment {segment_id!r} must have integer token bounds"
            )
        if start < previous_end or end <= start or end > prompt_length:
            raise BenchmarkInputError(
                f"segment {segment_id!r} has invalid range [{start}, {end}) "
                f"for prompt length {prompt_length}"
            )
        if segment.get("length") != end - start:
            raise BenchmarkInputError(
                f"segment {segment_id!r} length disagrees with its range"
            )
        previous_end = end

        logical_order = segment.get("logical_order")
        if type(logical_order) is not int or logical_order != index:
            raise BenchmarkInputError(
                f"segment {segment_id!r} has invalid logical order"
            )
        if not isinstance(segment.get("kind"), str):
            raise BenchmarkInputError(f"segment {segment_id!r} has no kind")
        if type(segment.get("shareable")) is not bool:
            raise BenchmarkInputError(
                f"segment {segment_id!r} has no strict shareable flag"
            )
        for field in ("source_id", "lineage_id"):
            value = segment.get(field)
            if value is not None and (not isinstance(value, str) or not value):
                raise BenchmarkInputError(f"segment {segment_id!r} has invalid {field}")

    return segments


def _validate_span(value: object, *, field: str, prompt_length: int) -> None:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(type(item) is not int for item in value)
    ):
        raise BenchmarkInputError(f"{field} must be a two-integer list")
    start, end = value
    if start < 0 or end < start or end > prompt_length:
        raise BenchmarkInputError(f"{field} is outside the prompt")


def _validate_kmesh_contract(
    record: dict[str, Any],
    prompt_length: int,
    segments: list[dict[str, Any]],
) -> None:
    request_id = record.get("request_id")
    context = record.get("kmesh_context")
    if not isinstance(context, dict) or set(context) != _CONTEXT_FIELDS:
        raise BenchmarkInputError(
            f"request {request_id!r} has a non-canonical kmesh_context"
        )
    if context.get("schema_version") != TRACE_SCHEMA_VERSION:
        raise BenchmarkInputError(
            f"request {request_id!r} has unsupported KMesh schema"
        )
    if type(context.get("enabled")) is not bool:
        raise BenchmarkInputError(
            f"request {request_id!r} has a non-boolean KMesh enabled flag"
        )
    if context.get("segment_encoding") != "sideband":
        raise BenchmarkInputError(f"request {request_id!r} is not a sideband trace")

    wire_segments = context.get("segments")
    if not isinstance(wire_segments, list) or len(wire_segments) != len(segments):
        raise BenchmarkInputError(
            f"request {request_id!r} KMesh segment count disagrees with analytics"
        )
    for index, (wire, analytic) in enumerate(zip(wire_segments, segments, strict=True)):
        expected = {field: analytic.get(field) for field in _WIRE_SEGMENT_FIELDS}
        if not isinstance(wire, dict) or set(wire) != _WIRE_SEGMENT_FIELDS:
            raise BenchmarkInputError(
                f"request {request_id!r} KMesh segment {index} is non-canonical"
            )
        if wire != expected:
            raise BenchmarkInputError(
                f"request {request_id!r} KMesh segment {index} disagrees with analytics"
            )

    lineage = record.get("lineage")
    if not isinstance(lineage, dict):
        raise BenchmarkInputError(f"request {request_id!r} has no lineage object")
    for field in (
        "execution_id",
        "group_kind",
        "group_id",
        "round_id",
        "branch_id",
        "parent_branch_id",
        "state_kind",
        "segment_encoding",
    ):
        if context[field] != lineage.get(field):
            raise BenchmarkInputError(
                f"request {request_id!r} KMesh {field} disagrees with lineage"
            )

    compatibility = record.get("compatibility_key")
    if not isinstance(compatibility, dict) or set(compatibility) != (
        _COMPATIBILITY_FIELDS
    ):
        raise BenchmarkInputError(
            f"request {request_id!r} has a non-canonical compatibility key"
        )
    for field in ("execution_id", "group_id", "round_id", "cache_domain"):
        if compatibility[field] != context[field]:
            raise BenchmarkInputError(
                f"request {request_id!r} compatibility {field} disagrees with context"
            )
    if compatibility["active_prompt_length"] != prompt_length:
        raise BenchmarkInputError(
            f"request {request_id!r} active prompt length disagrees with tokens"
        )
    _validate_span(
        compatibility["cached_span"],
        field=f"request {request_id!r} cached_span",
        prompt_length=prompt_length,
    )
    _validate_span(
        compatibility["prefill_chunk_span"],
        field=f"request {request_id!r} prefill_chunk_span",
        prompt_length=prompt_length,
    )


def load_trace(
    trace_path: Path, manifest_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    """Load and strictly validate one generated trace and its manifest."""
    manifest = _read_json_object(manifest_path)
    if manifest.get("schema_version") != TRACE_SCHEMA_VERSION:
        raise BenchmarkInputError("unsupported manifest schema version")
    if manifest.get("trace_file") != trace_path.name:
        raise BenchmarkInputError(
            "manifest trace_file does not name the supplied trace"
        )
    manifest_config = manifest.get("config")
    if not isinstance(manifest_config, dict):
        raise BenchmarkInputError("manifest has no config object")
    block_size = manifest_config.get("block_size")
    if type(block_size) is not int or block_size <= 0:
        raise BenchmarkInputError("manifest config has no valid block_size")
    qps = manifest_config.get("qps")
    if not isinstance(qps, (int, float)) or isinstance(qps, bool) or not qps > 0:
        raise BenchmarkInputError("manifest config has no valid positive qps")
    if manifest_config.get("mode") not in ("prefill_focus", "round_e2e"):
        raise BenchmarkInputError("manifest config has an invalid workload mode")

    trace_hash = _sha256(trace_path)
    if manifest.get("trace_sha256") != trace_hash:
        raise BenchmarkInputError("trace SHA-256 does not match the manifest")

    records: list[dict[str, Any]] = []
    request_ids: set[str] = set()
    trace_id: str | None = None
    try:
        stream = trace_path.open()
    except OSError as exc:
        raise BenchmarkInputError(f"cannot open trace {trace_path}: {exc}") from exc

    with stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise BenchmarkInputError(f"blank trace line {line_number}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BenchmarkInputError(
                    f"invalid JSON on trace line {line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise BenchmarkInputError(
                    f"trace line {line_number} must contain an object"
                )
            if record.get("schema_version") != TRACE_SCHEMA_VERSION:
                raise BenchmarkInputError(
                    f"unsupported schema on trace line {line_number}"
                )
            record_trace_id = record.get("trace_id")
            if not isinstance(record_trace_id, str) or not record_trace_id:
                raise BenchmarkInputError(f"trace line {line_number} has no trace ID")
            if trace_id is None:
                trace_id = record_trace_id
            elif record_trace_id != trace_id:
                raise BenchmarkInputError("trace records contain mixed trace IDs")
            request_id = record.get("request_id")
            if not isinstance(request_id, str) or not request_id:
                raise BenchmarkInputError(f"trace line {line_number} has no request ID")
            if request_id in request_ids:
                raise BenchmarkInputError(f"duplicate request ID {request_id!r}")
            request_ids.add(request_id)
            sequence_index = record.get("sequence_index")
            if type(sequence_index) is not int or sequence_index != len(records):
                raise BenchmarkInputError(
                    "trace sequence_index values must be contiguous and ordered"
                )

            prompt_token_ids = _require_int_list(
                record.get("prompt_token_ids"),
                field=f"request {request_id!r} prompt_token_ids",
            )
            segments = _validate_segments(record, len(prompt_token_ids))
            _validate_kmesh_contract(record, len(prompt_token_ids), segments)
            records.append(record)

    if not records:
        raise BenchmarkInputError("trace contains no requests")
    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        raise BenchmarkInputError("manifest has no counts object")
    manifest_count = counts.get("requests")
    if manifest_count != len(records):
        raise BenchmarkInputError(
            f"manifest declares {manifest_count!r} requests, found {len(records)}"
        )
    return manifest, records, trace_hash


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _append_jsonl(path: Path, value: Any) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


def _git_identity() -> dict[str, Any]:
    source_root = Path(__file__).resolve().parents[2]

    def git(*arguments: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "-C", str(source_root), *arguments],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "tree": str(source_root),
        "sha": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "status_porcelain": git("status", "--porcelain"),
    }


def _prepare_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise BenchmarkInputError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _nearest_rank(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(percentile * len(ordered) + 0.999) - 1))
    return ordered[index]


def _metric_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise BenchmarkInputError(f"metric {field} is not numeric")
    integer = int(value)
    if integer < 0 or integer != value:
        raise BenchmarkInputError(f"metric {field} is not a nonnegative integer")
    return integer


def _project_observer_snapshot(metrics: Sequence[object]) -> dict[str, Any]:
    """Project a public ``LLM.get_metrics()`` snapshot into bounded KMesh data."""
    counters = {field: 0 for field in _OBSERVER_COUNTER_FIELDS}
    reason_counts: dict[str, dict[str, int]] = {
        "fallback": {},
        "compatibility": {},
    }
    group_size_counts: dict[str, int] = {}
    gauge_values: dict[str, set[int]] = {
        field: set() for field in _OBSERVER_GAUGE_FIELDS
    }
    counter_names = {
        f"{_OBSERVER_PROM_PREFIX}{field}": field for field in _OBSERVER_COUNTER_FIELDS
    }
    gauge_names = {
        f"{_OBSERVER_PROM_PREFIX}{field}": field for field in _OBSERVER_GAUGE_FIELDS
    }

    for metric in metrics:
        name = getattr(metric, "name", None)
        labels = getattr(metric, "labels", None)
        value = getattr(metric, "value", None)
        if not isinstance(name, str) or not isinstance(labels, dict):
            continue
        if name in counter_names:
            field = counter_names[name]
            counters[field] += _metric_int(value, field=name)
        elif name == _OBSERVER_REASONS_METRIC:
            kind = labels.get("kind")
            reason = labels.get("reason")
            if kind not in reason_counts or not isinstance(reason, str):
                raise BenchmarkInputError("KMesh reason metric has invalid labels")
            count = _metric_int(value, field=f"{name}[{kind},{reason}]")
            if count:
                reason_counts[kind][reason] = reason_counts[kind].get(reason, 0) + count
        elif name == _OBSERVER_GROUP_SIZES_METRIC:
            group_size = labels.get("group_size")
            if not isinstance(group_size, str):
                raise BenchmarkInputError("KMesh group metric has no group_size label")
            count = _metric_int(value, field=f"{name}[{group_size}]")
            if count:
                group_size_counts[group_size] = (
                    group_size_counts.get(group_size, 0) + count
                )
        elif name in gauge_names:
            field = gauge_names[name]
            gauge_values[field].add(_metric_int(value, field=name))

    gauges: dict[str, int | None] = {}
    for field, values in gauge_values.items():
        if len(values) > 1:
            raise BenchmarkInputError(
                f"KMesh observer gauge {field} disagrees across engines"
            )
        gauges[field] = next(iter(values)) if values else None
    return {
        **counters,
        "fallback_counts": reason_counts["fallback"],
        "compatibility_failure_counts": reason_counts["compatibility"],
        "group_size_counts": group_size_counts,
        **gauges,
    }


def _count_map_delta(
    before: dict[str, int], after: dict[str, int], *, field: str
) -> dict[str, int]:
    delta: dict[str, int] = {}
    for key in sorted(before.keys() | after.keys()):
        value = after.get(key, 0) - before.get(key, 0)
        if value < 0:
            raise BenchmarkInputError(f"KMesh counter {field}[{key!r}] decreased")
        if value:
            delta[key] = value
    return delta


def _observer_oracle_expectation(
    manifest: dict[str, Any], admission_mode: str
) -> dict[str, Any]:
    oracle = manifest.get("oracle")
    aggregate = oracle.get("aggregate") if isinstance(oracle, dict) else None
    if not isinstance(aggregate, dict):
        raise BenchmarkInputError("B2 requires the generated structural oracle")
    eligible_tokens = _metric_int(
        aggregate.get("eligible_structural_shared_tokens_total"),
        field="oracle.eligible_structural_shared_tokens_total",
    )
    compatible_pairs = _metric_int(
        aggregate.get("compatible_pair_count"),
        field="oracle.compatible_pair_count",
    )
    trace_has_eligible_overlap = eligible_tokens > 0 and compatible_pairs > 0
    observer_expectation = manifest.get("observer_expectation")
    if (
        not isinstance(observer_expectation, dict)
        or observer_expectation.get("contract") != "kmesh_observer_supported_scope_v1"
    ):
        raise BenchmarkInputError(
            "B2 requires a generated observer expectation; regenerate the trace"
        )
    expected_field = f"{admission_mode}_group_presence_expected"
    expected_presence = observer_expectation.get(expected_field)
    if type(expected_presence) is not bool:
        raise BenchmarkInputError(
            f"observer expectation has no strict {expected_field} value"
        )
    required_fallback_counts = observer_expectation.get("required_fallback_counts")
    if not isinstance(required_fallback_counts, dict) or any(
        not isinstance(reason, str) or type(count) is not int or count < 0
        for reason, count in required_fallback_counts.items()
    ):
        raise BenchmarkInputError(
            "observer expectation has invalid required fallback counts"
        )
    supported_request_count = _metric_int(
        observer_expectation.get("supported_request_count"),
        field="observer_expectation.supported_request_count",
    )
    manifest_request_count = _metric_int(
        manifest.get("counts", {}).get("requests"),
        field="manifest.counts.requests",
    )
    if supported_request_count + sum(required_fallback_counts.values()) != (
        manifest_request_count
    ):
        raise BenchmarkInputError(
            "observer expectation does not account for every manifest request"
        )
    return {
        "admission_mode": admission_mode,
        "eligible_structural_shared_tokens_total": eligible_tokens,
        "compatible_pair_count": compatible_pairs,
        "trace_has_eligible_overlap": trace_has_eligible_overlap,
        "supported_request_count": supported_request_count,
        "required_fallback_counts": dict(sorted(required_fallback_counts.items())),
        "expected_formed_group_presence": expected_presence,
    }


def _observer_metrics_delta(
    before_metrics: Sequence[object],
    after_metrics: Sequence[object],
    manifest: dict[str, Any],
    *,
    admission_mode: str,
    cached_tokens: int,
) -> dict[str, Any]:
    cached_tokens = _metric_int(cached_tokens, field="cached_tokens")
    before = _project_observer_snapshot(before_metrics)
    after = _project_observer_snapshot(after_metrics)
    observed: dict[str, Any] = {
        "source": _OBSERVER_METRICS_SOURCE,
        "semantics": _OBSERVER_METRICS_SEMANTICS,
        "opportunity_scope": _OBSERVER_OPPORTUNITY_SCOPE,
        "executed_kv_reuse": False,
    }
    for field in _OBSERVER_COUNTER_FIELDS:
        value = after[field] - before[field]
        if value < 0:
            raise BenchmarkInputError(f"KMesh counter {field} decreased")
        observed[field] = value
    observed["fallback_counts"] = _count_map_delta(
        before["fallback_counts"],
        after["fallback_counts"],
        field="fallback_counts",
    )
    observed["compatibility_failure_counts"] = _count_map_delta(
        before["compatibility_failure_counts"],
        after["compatibility_failure_counts"],
        field="compatibility_failure_counts",
    )
    observed["group_size_counts"] = _count_map_delta(
        before["group_size_counts"],
        after["group_size_counts"],
        field="group_size_counts",
    )
    for field in _OBSERVER_GAUGE_FIELDS:
        observed[field] = after[field]

    oracle_expectation = _observer_oracle_expectation(manifest, admission_mode)
    observed_group_presence = observed["formed_groups"] > 0
    observed_reused_token_presence = observed["prospective_reused_tokens"] > 0
    oracle_expectation["observed_formed_group_presence"] = observed_group_presence
    observed["oracle_expectation"] = oracle_expectation

    manifest_block_size = manifest.get("config", {}).get("block_size")
    manifest_request_count = _metric_int(
        manifest.get("counts", {}).get("requests"),
        field="manifest.counts.requests",
    )
    formed_groups_match = observed["formed_groups"] == sum(
        observed["group_size_counts"].values()
    )
    grouped_requests = 0
    for group_size, count in observed["group_size_counts"].items():
        if not group_size.isdecimal():
            raise BenchmarkInputError(
                "B2 cannot reconstruct an overflow group-size label"
            )
        grouped_requests += int(group_size) * count
    observed_requests_match = observed["observed_requests"] == (
        grouped_requests + sum(observed["fallback_counts"].values())
    )
    invariants = {
        "planner_calls_positive": observed["planner_calls"] > 0,
        "formed_groups_match_group_size_counts": formed_groups_match,
        "observed_requests_match_membership_outcomes": observed_requests_match,
        "observed_requests_match_manifest": (
            observed["observed_requests"] == manifest_request_count
        ),
        "runtime_block_size_matches_manifest": (
            observed["block_size"] == manifest_block_size
        ),
        "runtime_max_group_size_matches_b2": (
            observed["max_group_size"] == B2_MAX_GROUP_SIZE
        ),
        "zero_cached_tokens": cached_tokens == 0,
        "group_presence_has_prospective_tokens": (
            observed_group_presence == observed_reused_token_presence
        ),
        "group_presence_matches_observer_expectation": (
            observed_group_presence
            == oracle_expectation["expected_formed_group_presence"]
        ),
        "required_fallback_counts_match": all(
            observed["fallback_counts"].get(reason, 0) == count
            for reason, count in oracle_expectation["required_fallback_counts"].items()
        ),
    }
    failed = [name for name, passed in invariants.items() if not passed]
    if failed:
        raise BenchmarkInputError(
            "KMesh observer metric invariant failed: " + ", ".join(failed)
        )
    observed["invariants"] = {**invariants, "all_passed": True}
    return observed


def _summarize(
    *,
    mode: str,
    admission_mode: str,
    metrics: list[dict[str, Any]],
    replay_seconds: float,
    engine_startup_seconds: float,
) -> dict[str, Any]:
    service_times = [
        metric["service_seconds"]
        for metric in metrics
        if metric.get("service_seconds") is not None
    ]
    prompt_tokens = sum(metric["prompt_tokens"] for metric in metrics)
    cached_tokens = sum(metric["num_cached_tokens"] or 0 for metric in metrics)
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete",
        "mode": mode,
        "admission_mode": admission_mode,
        "request_count": len(metrics),
        "engine_startup_seconds": engine_startup_seconds,
        "replay_seconds": replay_seconds,
        "requests_per_second": len(metrics) / replay_seconds,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "cached_token_fraction": cached_tokens / prompt_tokens,
        "service_seconds_mean": (
            statistics.fmean(service_times) if service_times else None
        ),
        "service_seconds_p50": _nearest_rank(service_times, 0.50),
        "service_seconds_p95": _nearest_rank(service_times, 0.95),
        "generated_token_ids": [metric["output_token_ids"] for metric in metrics],
    }


def _base_environment(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "pyhashseed": os.environ.get("PYTHONHASHSEED"),
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
        "source": _git_identity(),
        "dry_run": args.dry_run,
    }


def _engine_kwargs(args: argparse.Namespace, block_size: int) -> dict[str, Any]:
    """Build comparable engine options, adding only B2's no-op connector."""
    options: dict[str, Any] = {
        "model": args.model,
        "revision": args.model_revision,
        "tokenizer_revision": args.tokenizer_revision,
        "dtype": args.dtype,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "max_model_len": args.max_model_len,
        "block_size": block_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enable_prefix_caching": args.mode == "b1",
        "enable_chunked_prefill": args.chunked_prefill,
        "async_scheduling": args.async_scheduling,
        # KMesh is an observer-only connector and does not implement hybrid
        # cache allocation. Keep this explicit and identical across B0/B1/B2.
        "disable_hybrid_kv_cache_manager": True,
        "enforce_eager": args.enforce_eager,
        "trust_remote_code": False,
        "seed": args.seed,
        # Offline LLM disables logging by default. Keep it enabled for every
        # baseline so B2's public get_metrics() path does not receive a free pass.
        "disable_log_stats": False,
    }
    if args.mode == "b2":
        options["kv_transfer_config"] = _b2_kv_transfer_config()
    return options


def _run_vllm(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    output_dir: Path,
    environment: dict[str, Any],
) -> dict[str, Any]:
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    import torch

    import vllm
    from vllm import LLM, SamplingParams, TokensPrompt
    from vllm.platforms import current_platform

    block_size = manifest.get("config", {}).get("block_size")
    if type(block_size) is not int or block_size <= 0:
        raise BenchmarkInputError("manifest config has no valid block_size")

    environment["runtime"] = {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "vllm": vllm.__version__,
        "accelerator_available": torch.accelerator.is_available(),
        "accelerator_count": torch.accelerator.device_count(),
        "accelerator_names": [
            current_platform.get_device_name(index)
            for index in range(torch.accelerator.device_count())
        ],
    }
    _write_json(output_dir / "environment.json", environment)

    engine_start = time.perf_counter()
    llm = LLM(**_engine_kwargs(args, block_size))
    engine_startup_seconds = time.perf_counter() - engine_start

    workload_mode = manifest.get("config", {}).get("mode")
    max_tokens = 1 if workload_mode == "prefill_focus" else args.max_tokens
    sampling_options = {
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "seed": args.seed,
    }
    if args.mode == "b2":
        sampling_params = [
            SamplingParams(
                **sampling_options,
                extra_args={"kmesh": record["kmesh_context"]},
            )
            for record in records
        ]
    else:
        sampling_params = SamplingParams(**sampling_options)

    if not args.no_warmup:
        llm.generate(
            [TokensPrompt(prompt_token_ids=[1, 2, 3, 4])],
            SamplingParams(temperature=0.0, max_tokens=1, seed=args.seed),
            use_tqdm=False,
        )

    observer_metrics_before = llm.get_metrics() if args.mode == "b2" else None

    prompts = [
        TokensPrompt(prompt_token_ids=record["prompt_token_ids"]) for record in records
    ]
    metrics_path = output_dir / "metrics.jsonl"
    metrics: list[dict[str, Any]] = []
    replay_start = time.perf_counter()

    if args.admission_mode == "batch":
        batch_start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        batch_seconds = time.perf_counter() - batch_start
        for record, output in zip(records, outputs, strict=True):
            completion = output.outputs[0]
            metric = {
                "schema_version": RESULT_SCHEMA_VERSION,
                "request_id": record["request_id"],
                "sequence_index": record["sequence_index"],
                "service_seconds": None,
                "batch_seconds": batch_seconds,
                "prompt_tokens": len(output.prompt_token_ids),
                "num_cached_tokens": output.num_cached_tokens,
                "output_token_ids": list(completion.token_ids),
                "finish_reason": completion.finish_reason,
            }
            metrics.append(metric)
            _append_jsonl(metrics_path, metric)
    else:
        qps = manifest.get("config", {}).get("qps")
        if not isinstance(qps, (int, float)) or isinstance(qps, bool) or qps <= 0:
            raise BenchmarkInputError("manifest config has no valid positive qps")
        for index, (record, prompt) in enumerate(zip(records, prompts, strict=True)):
            target_time = replay_start + index / qps
            delay = target_time - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            submit_time = time.perf_counter()
            request_sampling_params = (
                [sampling_params[index]]
                if isinstance(sampling_params, list)
                else sampling_params
            )
            output = llm.generate([prompt], request_sampling_params, use_tqdm=False)[0]
            completion_time = time.perf_counter()
            completion = output.outputs[0]
            metric = {
                "schema_version": RESULT_SCHEMA_VERSION,
                "request_id": record["request_id"],
                "sequence_index": record["sequence_index"],
                "scheduled_offset_seconds": index / qps,
                "submit_offset_seconds": submit_time - replay_start,
                "service_seconds": completion_time - submit_time,
                "batch_seconds": None,
                "prompt_tokens": len(output.prompt_token_ids),
                "num_cached_tokens": output.num_cached_tokens,
                "output_token_ids": list(completion.token_ids),
                "finish_reason": completion.finish_reason,
            }
            metrics.append(metric)
            _append_jsonl(metrics_path, metric)

    replay_seconds = time.perf_counter() - replay_start
    summary = _summarize(
        mode=args.mode,
        admission_mode=args.admission_mode,
        metrics=metrics,
        replay_seconds=replay_seconds,
        engine_startup_seconds=engine_startup_seconds,
    )
    summary["structural_oracle"] = manifest.get("oracle")
    if args.mode == "b2":
        assert observer_metrics_before is not None
        summary["observer_metrics"] = _observer_metrics_delta(
            observer_metrics_before,
            llm.get_metrics(),
            manifest,
            admission_mode=args.admission_mode,
            cached_tokens=summary["cached_tokens"],
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=("b0", "b1", "b2"), default="b0")
    parser.add_argument(
        "--admission-mode", choices=("serial_paced", "batch"), default="batch"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--tokenizer-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument(
        "--enforce-eager", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--chunked-prefill", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--async-scheduling", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _validate_mode_controls(args)
    if args.max_model_len <= 0 or args.max_tokens <= 0:
        raise BenchmarkInputError("token limits must be positive")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise BenchmarkInputError("GPU memory utilization must be in (0, 1]")

    manifest, records, trace_hash = load_trace(args.trace, args.manifest)
    if args.mode == "b2":
        _observer_oracle_expectation(manifest, args.admission_mode)
    generated_tokens = (
        1 if manifest["config"]["mode"] == "prefill_focus" else args.max_tokens
    )
    longest_prompt = max(len(record["prompt_token_ids"]) for record in records)
    if longest_prompt + generated_tokens > args.max_model_len:
        raise BenchmarkInputError(
            "trace plus requested output exceeds the configured model length"
        )
    _prepare_output_dir(args.output_dir)
    config = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "mode": args.mode,
        "admission_mode": args.admission_mode,
        "trace": str(args.trace.resolve()),
        "trace_sha256": trace_hash,
        "trace_id": records[0].get("trace_id"),
        "generator": manifest.get("generator"),
        "generator_version": manifest.get("generator_version"),
        "trace_config": manifest.get("config"),
        "model": args.model,
        "model_revision": args.model_revision,
        "tokenizer_revision": args.tokenizer_revision,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": args.seed,
        "enforce_eager": args.enforce_eager,
        "chunked_prefill": args.chunked_prefill,
        "async_scheduling": args.async_scheduling,
        "disable_hybrid_kv_cache_manager": True,
        "engine_stats_enabled": True,
        "kv_transfer_config": (_b2_kv_transfer_config() if args.mode == "b2" else None),
        "offline": args.offline,
        "dry_run": args.dry_run,
    }
    environment = _base_environment(args)
    _write_json(args.output_dir / "config.json", config)
    _write_json(args.output_dir / "environment.json", environment)

    try:
        if args.dry_run:
            summary = {
                "schema_version": RESULT_SCHEMA_VERSION,
                "status": "validated",
                "mode": args.mode,
                "admission_mode": args.admission_mode,
                "request_count": len(records),
                "trace_sha256": trace_hash,
                "structural_oracle": manifest.get("oracle"),
            }
            if args.mode == "b2":
                summary["observer_metrics"] = {
                    "status": "not_executed_dry_run",
                    "source": _OBSERVER_METRICS_SOURCE,
                    "semantics": _OBSERVER_METRICS_SEMANTICS,
                    "opportunity_scope": _OBSERVER_OPPORTUNITY_SCOPE,
                    "executed_kv_reuse": False,
                }
        else:
            summary = _run_vllm(args, manifest, records, args.output_dir, environment)
    except BaseException as exc:
        failure = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "failed",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(args.output_dir / "failure.json", failure)
        raise

    _write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
