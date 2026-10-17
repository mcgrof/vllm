# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import partial
from multiprocessing.reduction import ForkingPickler
from types import SimpleNamespace

import pytest
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from vllm.distributed.kv_transfer.kv_connector.v1.kmesh_connector import (
    KMeshConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.kmesh_metrics import (
    KMeshConnectorStats,
    KMeshPromMetrics,
)
from vllm.v1.kmesh import (
    KMeshFallbackReason,
    KMeshGroupObserverMetrics,
    KMeshObserverMetrics,
    KMeshProspectiveMembership,
    KMeshReasonCount,
)
from vllm.v1.metrics import reader as metrics_reader

pytestmark = pytest.mark.skip_global_cleanup


def _observer_metrics() -> KMeshObserverMetrics:
    group = KMeshGroupObserverMetrics(
        reuse_group_id="group-1",
        request_count=2,
        block_size=4,
        shared_segments=1,
        shared_tokens=4,
        prospective_reused_segments=1,
        prospective_reused_tokens=4,
        structural_full_blocks=1,
        structural_full_block_tokens=4,
        prospective_reused_full_blocks=1,
        prospective_reused_full_block_tokens=4,
    )
    membership = (
        KMeshProspectiveMembership("request-a", "group-1", "plan-1", None),
        KMeshProspectiveMembership("request-b", "group-1", "plan-1", None),
        KMeshProspectiveMembership(
            "request-c", None, None, KMeshFallbackReason.SINGLETON_GROUP
        ),
    )
    return KMeshObserverMetrics(
        observed_requests=3,
        candidate_requests=3,
        formed_groups=1,
        block_size=4,
        max_group_size=10,
        group_sizes=(2,),
        shared_segments=1,
        shared_tokens=4,
        prospective_reused_segments=1,
        prospective_reused_tokens=4,
        structural_full_blocks=1,
        structural_full_block_tokens=4,
        prospective_reused_full_blocks=1,
        prospective_reused_full_block_tokens=4,
        compatibility_failures=(
            KMeshReasonCount(KMeshFallbackReason.ROUND_MISMATCH, 2),
        ),
        fallback_counts=(KMeshReasonCount(KMeshFallbackReason.SINGLETON_GROUP, 1),),
        prospective_membership=membership,
        group_metrics=(group,),
        planner_overhead_ns=17,
    )


def _fallback_metrics() -> KMeshObserverMetrics:
    return KMeshObserverMetrics(
        observed_requests=1,
        candidate_requests=0,
        formed_groups=0,
        block_size=4,
        max_group_size=10,
        group_sizes=(),
        shared_segments=0,
        shared_tokens=0,
        prospective_reused_segments=0,
        prospective_reused_tokens=0,
        structural_full_blocks=0,
        structural_full_block_tokens=0,
        prospective_reused_full_blocks=0,
        prospective_reused_full_block_tokens=0,
        compatibility_failures=(),
        fallback_counts=(KMeshReasonCount(KMeshFallbackReason.NO_METADATA, 1),),
        prospective_membership=(
            KMeshProspectiveMembership(
                "warmup", None, None, KMeshFallbackReason.NO_METADATA
            ),
        ),
        group_metrics=(),
        planner_overhead_ns=3,
    )


def test_stats_aggregate_and_swap_reset() -> None:
    first = KMeshConnectorStats()
    second = KMeshConnectorStats()
    first.observe(_observer_metrics())
    second.observe(_observer_metrics())

    assert first.aggregate(second) is first
    assert first.data["planner_calls"] == 2
    assert first.data["observed_requests"] == 6
    assert first.data["fallback_counts"] == {"singleton_group": 2}
    assert first.data["compatibility_failure_counts"] == {"round_mismatch": 4}
    assert first.data["group_size_counts"] == {"2": 2}

    snapshot = first.clone_and_reset()
    assert first.is_empty()
    assert snapshot.data["planner_calls"] == 2
    first.observe(_observer_metrics())
    assert snapshot.data["planner_calls"] == 2
    assert first.data["planner_calls"] == 1


def test_connector_stats_reconstruct_after_transport() -> None:
    original = KMeshConnectorStats()
    original.observe(_observer_metrics())
    transported = ForkingPickler.loads(ForkingPickler.dumps(original.data))

    rebuilt = KMeshConnector.build_kv_connector_stats(transported)

    assert isinstance(rebuilt, KMeshConnectorStats)
    assert rebuilt.data == original.data
    assert rebuilt.reduce()["prospective_reused_tokens"] == 4


class _FakeMetricChild:
    def __init__(self) -> None:
        self.value = 0

    def inc(self, value: int) -> None:
        self.value += value

    def set(self, value: int) -> None:
        self.value = value


class _FakeMetric:
    def __init__(self, **kwargs: object) -> None:
        self.name = kwargs["name"]
        self.children: dict[tuple[object, ...], _FakeMetricChild] = {}

    def labels(self, *labelvalues: object) -> _FakeMetricChild:
        return self.children.setdefault(labelvalues, _FakeMetricChild())


def test_prometheus_projection_uses_bounded_reason_and_group_labels() -> None:
    stats = KMeshConnectorStats()
    stats.observe(_observer_metrics())
    prom = KMeshPromMetrics(
        SimpleNamespace(kv_transfer_config=SimpleNamespace()),
        metric_types={
            Counter: _FakeMetric,
            Gauge: _FakeMetric,
            Histogram: _FakeMetric,
        },
        labelnames=["model"],
        per_engine_labelvalues={0: ["fixture"]},
    )

    prom.observe(stats.data)

    assert prom.scalar_counters["planner_calls"][0].value == 1
    assert prom.scalar_counters["prospective_reused_tokens"][0].value == 4
    assert prom.reason_counters[(0, "fallback", "singleton_group")].value == 1
    assert prom.reason_counters[(0, "compatibility", "round_mismatch")].value == 2
    assert prom.group_size_counters[(0, "2")].value == 1
    assert prom.gauges["block_size"][0].value == 4
    assert prom.gauges["max_group_size"][0].value == 10


def test_registry_snapshot_names_labels_and_warmup_delta(monkeypatch) -> None:
    from benchmarks.kmesh.benchmark_all_gather import _observer_metrics_delta

    registry = CollectorRegistry()
    monkeypatch.setattr(metrics_reader, "REGISTRY", registry)
    prom = KMeshPromMetrics(
        SimpleNamespace(kv_transfer_config=SimpleNamespace()),
        metric_types={
            Counter: partial(Counter, registry=registry),
            Gauge: partial(Gauge, registry=registry),
            Histogram: partial(Histogram, registry=registry),
        },
        labelnames=["model"],
        per_engine_labelvalues={0: ["fixture"]},
    )
    warmup = KMeshConnectorStats()
    warmup.observe(_fallback_metrics())
    prom.observe(warmup.data)
    before = metrics_reader.get_metrics_snapshot()

    replay = KMeshConnectorStats()
    replay.observe(_observer_metrics())
    prom.observe(replay.data)
    after = metrics_reader.get_metrics_snapshot()

    planner = next(
        metric for metric in after if metric.name == "vllm:kmesh_observer_planner_calls"
    )
    singleton = next(
        metric
        for metric in after
        if metric.name == "vllm:kmesh_observer_reasons"
        and metric.labels.get("kind") == "fallback"
        and metric.labels.get("reason") == "singleton_group"
    )
    assert planner.labels == {"model": "fixture"}
    assert planner.value == 2
    assert singleton.labels == {
        "model": "fixture",
        "kind": "fallback",
        "reason": "singleton_group",
    }
    assert singleton.value == 1

    observed = _observer_metrics_delta(
        before,
        after,
        {
            "config": {"block_size": 4},
            "counts": {"requests": 3},
            "oracle": {
                "aggregate": {
                    "eligible_structural_shared_tokens_total": 4,
                    "compatible_pair_count": 1,
                }
            },
            "observer_expectation": {
                "contract": "kmesh_observer_supported_scope_v1",
                "supported_request_count": 3,
                "required_fallback_counts": {},
                "batch_group_presence_expected": True,
                "serial_paced_group_presence_expected": False,
            },
        },
        admission_mode="batch",
        cached_tokens=0,
    )
    assert observed["planner_calls"] == 1
    assert observed["observed_requests"] == 3
    assert observed["fallback_counts"] == {"singleton_group": 1}
    assert observed["group_size_counts"] == {"2": 1}
    assert observed["invariants"]["all_passed"] is True
