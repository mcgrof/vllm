# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded observer telemetry for the KMesh connector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.v1.kmesh import KMeshFallbackReason, KMeshObserverMetrics
from vllm.v1.metrics.utils import create_metric_per_engine

KMESH_PROM_PREFIX = "vllm:kmesh_observer_"

_SCALAR_COUNTER_FIELDS = (
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
_GAUGE_FIELDS = ("block_size", "max_group_size")
_REASON_FIELDS = ("fallback_counts", "compatibility_failure_counts")
_ALL_DATA_FIELDS = frozenset(
    (*_SCALAR_COUNTER_FIELDS, *_GAUGE_FIELDS, *_REASON_FIELDS, "group_size_counts")
)
_REASON_VALUES = frozenset(reason.value for reason in KMeshFallbackReason)
_REASON_KINDS = {
    "fallback_counts": "fallback",
    "compatibility_failure_counts": "compatibility",
}
_MAX_EXACT_GROUP_SIZE_LABEL = 10
_OVERFLOW_GROUP_SIZE_LABEL = "11_plus"

_COUNTER_DOCUMENTATION = {
    "planner_calls": "Number of KMesh observer planner calls.",
    "observed_requests": (
        "Number of request observations considered by KMesh planner calls."
    ),
    "candidate_requests": (
        "Number of requests with complete evidence that were prospective KMesh "
        "grouping candidates."
    ),
    "formed_groups": "Number of prospective KMesh reuse groups selected.",
    "shared_segments": (
        "Number of logical shared segments in selected prospective KMesh plans."
    ),
    "shared_tokens": (
        "Number of logical shared tokens in selected prospective KMesh plans."
    ),
    "prospective_reused_segments": (
        "Number of segment copies a selected KMesh plan could prospectively reuse."
    ),
    "prospective_reused_tokens": (
        "Number of token positions a selected KMesh plan could prospectively reuse."
    ),
    "structural_full_blocks": (
        "Number of structurally reusable full blocks in selected KMesh plans."
    ),
    "structural_full_block_tokens": (
        "Number of tokens covered by structurally reusable full blocks in selected "
        "KMesh plans."
    ),
    "prospective_reused_full_blocks": (
        "Number of full-block copies a selected KMesh plan could prospectively reuse."
    ),
    "prospective_reused_full_block_tokens": (
        "Number of token positions in full-block copies a selected KMesh plan could "
        "prospectively reuse."
    ),
    "planner_overhead_ns": (
        "Total measured KMesh observer planner overhead in nanoseconds."
    ),
}


def _empty_data() -> dict[str, Any]:
    return {
        **{field: 0 for field in _SCALAR_COUNTER_FIELDS},
        **{field: 0 for field in _GAUGE_FIELDS},
        **{field: {} for field in _REASON_FIELDS},
        "group_size_counts": {},
    }


def _nonnegative_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _validate_count_map(
    value: object, field: str, *, valid_keys: frozenset[str] | None = None
) -> dict[str, int]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be a string-keyed mapping")
    counts: dict[str, int] = {}
    for key, count in value.items():
        if valid_keys is not None and key not in valid_keys:
            raise ValueError(f"{field} contains unknown key {key!r}")
        counts[key] = _nonnegative_int(count, f"{field}[{key!r}]")
    return counts


def _group_size_label(group_size: int) -> str:
    if 2 <= group_size <= _MAX_EXACT_GROUP_SIZE_LABEL:
        return str(group_size)
    return _OVERFLOW_GROUP_SIZE_LABEL


@dataclass
class KMeshConnectorStats(KVConnectorStats):
    """Serializable, interval-scoped KMesh observer counters."""

    def __post_init__(self) -> None:
        if not self.data:
            self.reset()
            return
        if set(self.data) != _ALL_DATA_FIELDS:
            raise ValueError("KMesh connector stats have an unexpected wire schema")
        for field in (*_SCALAR_COUNTER_FIELDS, *_GAUGE_FIELDS):
            self.data[field] = _nonnegative_int(self.data[field], field)
        for field in _REASON_FIELDS:
            self.data[field] = _validate_count_map(
                self.data[field], field, valid_keys=_REASON_VALUES
            )
        group_counts = _validate_count_map(
            self.data["group_size_counts"], "group_size_counts"
        )
        for group_size in group_counts:
            if not group_size.isdecimal() or int(group_size) < 2:
                raise ValueError(
                    "group_size_counts keys must be integers of at least two"
                )
        self.data["group_size_counts"] = group_counts

    def reset(self) -> None:
        self.data = _empty_data()

    def observe(self, metrics: KMeshObserverMetrics) -> None:
        """Accumulate one planner call without retaining per-request identities."""
        if not isinstance(metrics, KMeshObserverMetrics):
            raise TypeError("metrics must be KMeshObserverMetrics")
        if self.is_empty():
            self.data["block_size"] = metrics.block_size
            self.data["max_group_size"] = metrics.max_group_size
        elif (
            self.data["block_size"] != metrics.block_size
            or self.data["max_group_size"] != metrics.max_group_size
        ):
            raise ValueError("KMesh observer configuration changed within an interval")

        self.data["planner_calls"] += 1
        for field in _SCALAR_COUNTER_FIELDS[1:]:
            self.data[field] += getattr(metrics, field)
        for item in metrics.fallback_counts:
            counts = self.data["fallback_counts"]
            counts[item.reason.value] = counts.get(item.reason.value, 0) + item.count
        for item in metrics.compatibility_failures:
            counts = self.data["compatibility_failure_counts"]
            counts[item.reason.value] = counts.get(item.reason.value, 0) + item.count
        for group_size in metrics.group_sizes:
            key = str(group_size)
            counts = self.data["group_size_counts"]
            counts[key] = counts.get(key, 0) + 1

    def clone_and_reset(self) -> KMeshConnectorStats:
        """Swap out the current payload so subsequent calls see a fresh interval."""
        snapshot_data = self.data
        self.reset()
        return KMeshConnectorStats(data=snapshot_data)

    def is_empty(self) -> bool:
        return self.data["planner_calls"] == 0

    def aggregate(self, other: KVConnectorStats) -> KMeshConnectorStats:
        if not isinstance(other, KMeshConnectorStats):
            raise TypeError("KMesh stats can only aggregate KMesh stats")
        if other.is_empty():
            return self
        if self.is_empty():
            self.data["block_size"] = other.data["block_size"]
            self.data["max_group_size"] = other.data["max_group_size"]
        elif any(self.data[field] != other.data[field] for field in _GAUGE_FIELDS):
            raise ValueError(
                "cannot aggregate KMesh stats from different configurations"
            )
        for field in _SCALAR_COUNTER_FIELDS:
            self.data[field] += other.data[field]
        for field in (*_REASON_FIELDS, "group_size_counts"):
            destination = self.data[field]
            for key, count in other.data[field].items():
                destination[key] = destination.get(key, 0) + count
        return self

    def reduce(self) -> dict[str, int | float]:
        reduced = {
            field: self.data[field]
            for field in (*_SCALAR_COUNTER_FIELDS, *_GAUGE_FIELDS)
        }
        for field in _REASON_FIELDS:
            for reason, count in sorted(self.data[field].items()):
                reduced[f"{field}.{reason}"] = count
        for group_size, count in sorted(
            self.data["group_size_counts"].items(), key=lambda item: int(item[0])
        ):
            reduced[f"group_size_counts.{group_size}"] = count
        return reduced


class KMeshPromMetrics(KVConnectorPromMetrics):
    """Prometheus projection with only finite reason and group-size labels."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> None:
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)
        self.scalar_counters: dict[str, dict[int, PromMetricT]] = {}
        for field in _SCALAR_COUNTER_FIELDS:
            counter = self._counter_cls(
                name=f"{KMESH_PROM_PREFIX}{field}",
                documentation=_COUNTER_DOCUMENTATION[field],
                labelnames=labelnames,
            )
            self.scalar_counters[field] = create_metric_per_engine(
                counter, per_engine_labelvalues
            )

        reason_counter = self._counter_cls(
            name=f"{KMESH_PROM_PREFIX}reasons",
            documentation=(
                "KMesh prospective fallback outcomes and compatibility comparison "
                "failures by bounded reason."
            ),
            labelnames=labelnames + ["kind", "reason"],
        )
        self.reason_counters: dict[tuple[int, str, str], PromMetricT] = {}
        for engine_idx, labelvalues in per_engine_labelvalues.items():
            for kind in sorted(_REASON_KINDS.values()):
                for reason in sorted(_REASON_VALUES):
                    self.reason_counters[(engine_idx, kind, reason)] = (
                        reason_counter.labels(*labelvalues, kind, reason)
                    )

        group_size_counter = self._counter_cls(
            name=f"{KMESH_PROM_PREFIX}formed_groups_by_size",
            documentation=(
                "Prospective KMesh reuse groups by bounded request-count label."
            ),
            labelnames=labelnames + ["group_size"],
        )
        group_labels = [
            *(str(size) for size in range(2, _MAX_EXACT_GROUP_SIZE_LABEL + 1)),
            _OVERFLOW_GROUP_SIZE_LABEL,
        ]
        self.group_size_counters: dict[tuple[int, str], PromMetricT] = {}
        for engine_idx, labelvalues in per_engine_labelvalues.items():
            for group_label in group_labels:
                self.group_size_counters[(engine_idx, group_label)] = (
                    group_size_counter.labels(*labelvalues, group_label)
                )

        self.gauges: dict[str, dict[int, PromMetricT]] = {}
        for field in _GAUGE_FIELDS:
            gauge = self._gauge_cls(
                name=f"{KMESH_PROM_PREFIX}{field}",
                documentation=f"Configured KMesh observer {field.replace('_', ' ')}.",
                labelnames=labelnames,
            )
            self.gauges[field] = create_metric_per_engine(gauge, per_engine_labelvalues)

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0) -> None:
        stats = KMeshConnectorStats(data=transfer_stats_data)
        for field, counters in self.scalar_counters.items():
            counters[engine_idx].inc(stats.data[field])
        for field, kind in _REASON_KINDS.items():
            for reason, count in stats.data[field].items():
                self.reason_counters[(engine_idx, kind, reason)].inc(count)
        for group_size, count in stats.data["group_size_counts"].items():
            label = _group_size_label(int(group_size))
            self.group_size_counters[(engine_idx, label)].inc(count)
        for field, gauges in self.gauges.items():
            gauges[engine_idx].set(stats.data[field])
