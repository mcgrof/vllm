# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Aggregate KMesh benchmark bundles without overstating sparse evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

ANALYSIS_SCHEMA_VERSION = 1
MIN_STABLE_REPETITIONS = 5


class AnalysisInputError(ValueError):
    """Raised when a benchmark result bundle is incomplete or malformed."""


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisInputError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AnalysisInputError(f"expected a JSON object in {path}")
    return value


def discover_run_dirs(inputs: list[Path]) -> list[Path]:
    """Find unique result directories beneath explicit paths."""
    discovered: set[Path] = set()
    for input_path in inputs:
        if not input_path.exists():
            raise AnalysisInputError(f"result path does not exist: {input_path}")
        if input_path.is_file():
            if input_path.name not in ("summary.json", "failure.json"):
                raise AnalysisInputError(
                    f"result file must be summary.json or failure.json: {input_path}"
                )
            discovered.add(input_path.parent.resolve())
            continue
        if (input_path / "summary.json").is_file() or (
            input_path / "failure.json"
        ).is_file():
            discovered.add(input_path.resolve())
        for filename in ("summary.json", "failure.json"):
            discovered.update(
                path.parent.resolve() for path in input_path.rglob(filename)
            )
    return sorted(discovered)


def _load_bundle(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config.json"
    environment_path = run_dir / "environment.json"
    if not config_path.is_file() or not environment_path.is_file():
        raise AnalysisInputError(
            f"bundle lacks config.json or environment.json: {run_dir}"
        )
    summary_path = run_dir / "summary.json"
    failure_path = run_dir / "failure.json"
    if summary_path.is_file() == failure_path.is_file():
        raise AnalysisInputError(
            f"bundle must contain exactly one summary or failure: {run_dir}"
        )
    return {
        "run_dir": str(run_dir),
        "config": _read_object(config_path),
        "environment": _read_object(environment_path),
        "summary": _read_object(summary_path) if summary_path.is_file() else None,
        "failure": _read_object(failure_path) if failure_path.is_file() else None,
    }


def _comparison_identity(bundle: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    config = bundle["config"]
    environment = bundle["environment"]
    identity = {
        "trace_sha256": config.get("trace_sha256"),
        "trace_id": config.get("trace_id"),
        "trace_config": config.get("trace_config"),
        "admission_mode": config.get("admission_mode"),
        "model": config.get("model"),
        "model_revision": config.get("model_revision"),
        "tokenizer_revision": config.get("tokenizer_revision"),
        "dtype": config.get("dtype"),
        "max_model_len": config.get("max_model_len"),
        "max_tokens": config.get("max_tokens"),
        "enforce_eager": config.get("enforce_eager"),
        "chunked_prefill": config.get("chunked_prefill"),
        "async_scheduling": config.get("async_scheduling"),
        "disable_hybrid_kv_cache_manager": config.get(
            "disable_hybrid_kv_cache_manager"
        ),
        "source_sha": environment.get("source", {}).get("sha"),
        "runtime": environment.get("runtime"),
    }
    canonical = json.dumps(identity, separators=(",", ":"), sort_keys=True)
    group_id = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    return group_id, identity


def _metric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _output_signature(summary: dict[str, Any]) -> str | None:
    outputs = summary.get("generated_token_ids")
    if outputs is None:
        return None
    encoded = json.dumps(outputs, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _observer_evidence_passed(summary: dict[str, Any]) -> bool:
    observer = summary.get("observer_metrics")
    if not isinstance(observer, dict):
        return False
    invariants = observer.get("invariants")
    return (
        observer.get("source") == "vllm_kv_connector_stats_prometheus_delta"
        and isinstance(invariants, dict)
        and invariants.get("all_passed") is True
    )


def _aggregate_mode(bundles: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = [bundle["summary"] for bundle in bundles]
    signatures = [_output_signature(summary) for summary in summaries]
    unique_signatures = sorted(set(signatures), key=lambda value: str(value))
    replay_seconds = [
        float(summary["replay_seconds"])
        for summary in summaries
        if isinstance(summary.get("replay_seconds"), (int, float))
        and not isinstance(summary.get("replay_seconds"), bool)
    ]
    throughput = [
        float(summary["requests_per_second"])
        for summary in summaries
        if isinstance(summary.get("requests_per_second"), (int, float))
        and not isinstance(summary.get("requests_per_second"), bool)
    ]
    cached_fractions = [
        float(summary["cached_token_fraction"])
        for summary in summaries
        if isinstance(summary.get("cached_token_fraction"), (int, float))
        and not isinstance(summary.get("cached_token_fraction"), bool)
    ]
    service_means = [
        float(summary["service_seconds_mean"])
        for summary in summaries
        if isinstance(summary.get("service_seconds_mean"), (int, float))
        and not isinstance(summary.get("service_seconds_mean"), bool)
    ]
    observer_evidence = [_observer_evidence_passed(summary) for summary in summaries]
    return {
        "repetitions": len(bundles),
        "stable_timing_sample": len(bundles) >= MIN_STABLE_REPETITIONS,
        "output_consistent": len(unique_signatures) == 1,
        "output_signatures": unique_signatures,
        "replay_seconds": _metric_summary(replay_seconds),
        "requests_per_second": _metric_summary(throughput),
        "cached_token_fraction": _metric_summary(cached_fractions),
        "service_seconds_mean": _metric_summary(service_means),
        "zero_cached_tokens": (
            len(cached_fractions) == len(summaries)
            and all(value == 0 for value in cached_fractions)
        ),
        "observer_evidence_all_passed": bool(observer_evidence)
        and all(observer_evidence),
        "run_dirs": [bundle["run_dir"] for bundle in bundles],
    }


def _median(metric: dict[str, Any]) -> float | None:
    value = metric.get("median")
    return float(value) if isinstance(value, (int, float)) else None


def _compare_baselines(
    modes: dict[str, dict[str, Any]], admission_mode: object
) -> dict[str, Any]:
    b0 = modes.get("b0")
    b1 = modes.get("b1")
    if b0 is None or b1 is None:
        return {
            "available": False,
            "reason": "both b0 and b1 are required",
            "output_equivalent": None,
            "apc_hit_evidence": None,
            "performance_claim_ready": False,
            "timing_metric": None,
            "median_speedup_b1_over_b0": None,
            "median_replay_speedup_b1_over_b0": None,
        }

    output_equivalent = (
        b0["output_consistent"]
        and b1["output_consistent"]
        and b0["output_signatures"] == b1["output_signatures"]
        and b0["output_signatures"] != [None]
    )
    b0_cached = _median(b0["cached_token_fraction"])
    b1_cached = _median(b1["cached_token_fraction"])
    apc_hit_evidence = (
        b0_cached is not None and b1_cached is not None and b1_cached > b0_cached
    )
    timing_metric = (
        "replay_seconds" if admission_mode == "batch" else "service_seconds_mean"
    )
    b0_timing = _median(b0[timing_metric])
    b1_timing = _median(b1[timing_metric])
    performance_ready = bool(
        output_equivalent
        and b0["stable_timing_sample"]
        and b1["stable_timing_sample"]
        and b0_timing is not None
        and b1_timing is not None
    )
    speedup = (
        b0_timing / b1_timing
        if performance_ready
        and b0_timing is not None
        and b1_timing is not None
        and b1_timing > 0
        else None
    )
    return {
        "available": True,
        "reason": None,
        "output_equivalent": output_equivalent,
        "apc_hit_evidence": apc_hit_evidence,
        "performance_claim_ready": performance_ready,
        "timing_metric": timing_metric,
        "median_speedup_b1_over_b0": speedup,
        "median_replay_speedup_b1_over_b0": (
            speedup if timing_metric == "replay_seconds" else None
        ),
    }


def _compare_observer(modes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    b0 = modes.get("b0")
    b2 = modes.get("b2")
    if b0 is None or b2 is None:
        return {
            "available": False,
            "reason": "both b0 and b2 are required",
            "output_equivalent": None,
            "b2_zero_cached_tokens": None,
            "observer_evidence_all_passed": None,
            "behavior_claim_ready": False,
        }

    output_equivalent = (
        b0["output_consistent"]
        and b2["output_consistent"]
        and b0["output_signatures"] == b2["output_signatures"]
        and b0["output_signatures"] != [None]
    )
    zero_cached_tokens = b2["zero_cached_tokens"]
    observer_evidence = b2["observer_evidence_all_passed"]
    return {
        "available": True,
        "reason": None,
        "output_equivalent": output_equivalent,
        "b2_zero_cached_tokens": zero_cached_tokens,
        "observer_evidence_all_passed": observer_evidence,
        "behavior_claim_ready": bool(
            output_equivalent and zero_cached_tokens and observer_evidence
        ),
    }


def analyze(inputs: list[Path]) -> dict[str, Any]:
    """Load result bundles and build comparison groups with claim gates."""
    bundles = [_load_bundle(path) for path in discover_run_dirs(inputs)]
    if not bundles:
        raise AnalysisInputError("no result bundles found")

    failures = [
        {
            "run_dir": bundle["run_dir"],
            "exception_type": bundle["failure"].get("exception_type"),
            "message": bundle["failure"].get("message"),
        }
        for bundle in bundles
        if bundle["failure"] is not None
    ]
    validated = [
        bundle["run_dir"]
        for bundle in bundles
        if bundle["summary"] is not None
        and bundle["summary"].get("status") == "validated"
    ]
    completed = [
        bundle
        for bundle in bundles
        if bundle["summary"] is not None
        and bundle["summary"].get("status") == "complete"
    ]

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    identities: dict[str, dict[str, Any]] = {}
    for bundle in completed:
        group_id, identity = _comparison_identity(bundle)
        grouped[group_id].append(bundle)
        identities[group_id] = identity

    comparisons = []
    for group_id in sorted(grouped):
        by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for bundle in grouped[group_id]:
            mode = bundle["config"].get("mode")
            if not isinstance(mode, str):
                raise AnalysisInputError(
                    f"bundle has no string mode: {bundle['run_dir']}"
                )
            by_mode[mode].append(bundle)
        modes = {
            mode: _aggregate_mode(
                sorted(mode_bundles, key=lambda item: item["run_dir"])
            )
            for mode, mode_bundles in sorted(by_mode.items())
        }
        comparisons.append(
            {
                "comparison_id": group_id,
                "identity": identities[group_id],
                "modes": modes,
                "b0_b1": _compare_baselines(
                    modes, identities[group_id].get("admission_mode")
                ),
                "b0_b2": _compare_observer(modes),
            }
        )

    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "minimum_stable_repetitions": MIN_STABLE_REPETITIONS,
        "bundle_count": len(bundles),
        "completed_count": len(completed),
        "validated_count": len(validated),
        "failure_count": len(failures),
        "validated_run_dirs": validated,
        "failures": failures,
        "comparisons": comparisons,
    }


def _markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# KMesh benchmark analysis",
        "",
        (
            f"Bundles: {analysis['bundle_count']}; completed: "
            f"{analysis['completed_count']}; validation-only: "
            f"{analysis['validated_count']}; failed: {analysis['failure_count']}."
        ),
        "",
        "| Comparison | Mode | Repetitions | Output stable | Timing gate | "
        "Median replay (s) |",
        "| --- | --- | ---: | --- | --- | ---: |",
    ]
    for comparison in analysis["comparisons"]:
        for mode, aggregate in comparison["modes"].items():
            replay = aggregate["replay_seconds"]["median"]
            replay_text = f"{replay:.6f}" if replay is not None else "n/a"
            lines.append(
                f"| `{comparison['comparison_id']}` | {mode} | "
                f"{aggregate['repetitions']} | {aggregate['output_consistent']} | "
                f"{aggregate['stable_timing_sample']} | {replay_text} |"
            )
    lines.extend(
        [
            "",
            (
                f"A timing comparison requires at least "
                f"{analysis['minimum_stable_repetitions']} repetitions per mode "
                "and equivalent deterministic outputs."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()

    result = analyze(args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(_markdown(result))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
