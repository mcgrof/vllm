# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import replace
from multiprocessing.reduction import ForkingPickler
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm import SamplingParams
from vllm.config import KVTransferConfig, OffloadConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorRequestSidecar,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.kmesh_connector import (
    KMeshConnector,
    KMeshConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.kmesh_metrics import (
    KMeshConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
    MultiKVConnectorMetadata,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.kmesh import (
    KMESH_COLLECTIVE_LOCKSTEP_V1_PROFILE,
    KMESH_REQUEST_SIDECAR_PROVIDER_ID,
    KMESH_SERIAL_PIC_V1_PROFILE,
    KMESH_TARGET_PROFILE_KEY,
    KMeshFallbackReason,
    KMeshRequestLookupDescriptor,
    KMeshRequestLookupIntent,
    KMeshValidationError,
    kmesh_token_slice_digest,
)
from vllm.v1.kmesh import config as kmesh_config_module
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)

pytestmark = pytest.mark.skip_global_cleanup


def _vllm_config(kmesh: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(
        kv_transfer_config=KVTransferConfig(
            kv_connector="KMeshConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"kmesh": kmesh},
        ),
        cache_config=SimpleNamespace(block_size=4, cache_dtype="auto"),
        model_config=SimpleNamespace(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            revision="model-sha",
            tokenizer="Qwen/Qwen2.5-0.5B-Instruct",
            tokenizer_revision="tokenizer-sha",
            dtype=torch.bfloat16,
            hf_text_config=SimpleNamespace(
                max_position_embeddings=32768,
                partial_rotary_factor=1.0,
                rms_norm_eps=1e-6,
                rope_parameters=None,
                rope_scaling=None,
                rope_theta=1_000_000.0,
                rotary_pct=None,
            ),
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, pipeline_parallel_size=1
        ),
        attention_config=SimpleNamespace(
            backend=None, compute_hash=lambda: "attention-config-sha"
        ),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=True),
        device_config=SimpleNamespace(device_type="cpu"),
        offload_config=OffloadConfig(),
    )


def _kv_cache_config() -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["model.layers.0.self_attn"],
                kv_cache_spec=FullAttentionSpec(
                    block_size=4,
                    num_kv_heads=2,
                    head_size=8,
                    dtype=torch.bfloat16,
                ),
            )
        ],
    )


def _context(request_id: str, *, domain: str = "tenant-a") -> dict[str, object]:
    return {
        "schema_version": 1,
        "enabled": True,
        "execution_id": "execution-1",
        "group_kind": "agent_round",
        "group_id": "gather-1",
        "round_id": 2,
        "branch_id": request_id,
        "parent_branch_id": None,
        "state_kind": "kv_cache",
        "cache_domain": domain,
        "segment_encoding": "sideband",
        "segments": [
            {
                "segment_id": f"private-{request_id}",
                "kind": "private_history",
                "token_start": 0,
                "token_end": 4,
                "logical_order": 0,
                "source_id": request_id,
                "shareable": False,
                "lineage_id": f"private-lineage-{request_id}",
            },
            {
                "segment_id": "shared-update-1",
                "kind": "shared_update",
                "token_start": 4,
                "token_end": 12,
                "logical_order": 1,
                "source_id": "coordinator",
                "shareable": True,
                "lineage_id": "round-1",
            },
            {
                "segment_id": f"task-{request_id}",
                "kind": "task",
                "token_start": 12,
                "token_end": 16,
                "logical_order": 2,
                "source_id": "coordinator",
                "shareable": False,
                "lineage_id": f"task-lineage-{request_id}",
            },
        ],
    }


def _without_sharing(context: dict[str, object]) -> dict[str, object]:
    segments = context["segments"]
    assert isinstance(segments, list)
    for segment in segments:
        assert isinstance(segment, dict)
        segment["shareable"] = False
    return context


def _request(request_id: str, context: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        prompt_token_ids=list(range(16)),
        prompt_embeds=None,
        prompt_is_token_ids=None,
        num_prompt_tokens=16,
        sampling_params=SamplingParams(max_tokens=1, extra_args={"kmesh": context}),
        pooling_params=None,
        lora_request=None,
        cache_salt=None,
        kv_transfer_params=None,
        mm_features=[],
        resumable=False,
        streaming_queue=None,
        spec_token_ids=[],
    )


def _new_request_data(
    request: SimpleNamespace, *, num_computed_tokens: int = 0
) -> NewRequestData:
    return NewRequestData(
        req_id=request.request_id,
        prompt_token_ids=request.prompt_token_ids,
        mm_features=[],
        sampling_params=request.sampling_params,
        pooling_params=None,
        block_ids=([0, 1, 2, 3],),
        num_computed_tokens=num_computed_tokens,
        lora_request=None,
        prefill_token_ids=request.prompt_token_ids,
    )


def _scheduler_output(
    *requests: SimpleNamespace,
    num_computed_tokens: tuple[int, ...] | None = None,
    num_scheduled_tokens: tuple[int, ...] | None = None,
) -> SchedulerOutput:
    computed = (
        num_computed_tokens if num_computed_tokens is not None else (0,) * len(requests)
    )
    if len(computed) != len(requests):
        raise ValueError("computed-token evidence must cover every request")
    scheduled_counts = (
        num_scheduled_tokens
        if num_scheduled_tokens is not None
        else tuple(
            request.num_prompt_tokens - cached
            for request, cached in zip(requests, computed, strict=True)
        )
    )
    if len(scheduled_counts) != len(requests):
        raise ValueError("scheduled-token evidence must cover every request")
    scheduled = {
        request.request_id: count
        for request, count in zip(requests, scheduled_counts, strict=True)
    }
    return SchedulerOutput(
        scheduled_new_reqs=[
            _new_request_data(request, num_computed_tokens=cached)
            for request, cached in zip(requests, computed, strict=True)
        ],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens=scheduled,
        total_num_scheduled_tokens=sum(scheduled.values()),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def _offset_collective_request(
    request_id: str,
    *,
    offset: int,
    domain: str = "tenant-a",
    prompt_length: int = 512,
) -> SimpleNamespace:
    shared_start = 24 + offset
    shared_end = 440 + offset
    context = _context(request_id, domain=domain)
    context["segments"] = [
        {
            "segment_id": f"private-prefix-{request_id}",
            "kind": "private_history",
            "token_start": 0,
            "token_end": shared_start,
            "logical_order": 0,
            "source_id": request_id,
            "shareable": False,
            "lineage_id": f"private-prefix-{request_id}",
        },
        {
            "segment_id": "shared-update-offset",
            "kind": "shared_update",
            "token_start": shared_start,
            "token_end": shared_end,
            "logical_order": 1,
            "source_id": "coordinator",
            "shareable": True,
            "lineage_id": "round-offset",
        },
        {
            "segment_id": f"private-tail-{request_id}",
            "kind": "private_history",
            "token_start": shared_end,
            "token_end": prompt_length,
            "logical_order": 2,
            "source_id": request_id,
            "shareable": False,
            "lineage_id": f"private-tail-{request_id}",
        },
    ]
    request = _request(request_id, context)
    prompt_token_ids = [offset] * prompt_length
    prompt_token_ids[shared_start:shared_end] = range(1000, 1416)
    request.prompt_token_ids = prompt_token_ids
    request.num_prompt_tokens = len(prompt_token_ids)
    return request


def _cached_scheduler_output(*requests: SimpleNamespace) -> SchedulerOutput:
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData(
            req_ids=[request.request_id for request in requests],
            resumed_req_ids=set(),
            new_token_ids=[[] for _ in requests],
            all_token_ids={},
            new_block_ids=[None for _ in requests],
            num_computed_tokens=[4 for _ in requests],
            num_output_tokens=[0 for _ in requests],
        ),
        num_scheduled_tokens={request.request_id: 12 for request in requests},
        total_num_scheduled_tokens=12 * len(requests),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def _serial_pic_cached_scheduler_output(
    request: SimpleNamespace,
    *,
    num_computed_tokens: int,
    num_scheduled_tokens: int,
    resumed: bool = False,
) -> SchedulerOutput:
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData(
            req_ids=[request.request_id],
            resumed_req_ids={request.request_id} if resumed else set(),
            new_token_ids=[[]],
            all_token_ids={},
            new_block_ids=[None],
            num_computed_tokens=[num_computed_tokens],
            num_output_tokens=[0],
        ),
        num_scheduled_tokens={request.request_id: num_scheduled_tokens},
        total_num_scheduled_tokens=num_scheduled_tokens,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def _observer_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {"enabled": True, "mode": "observe"}
    config.update(overrides)
    return config


def _serial_pic_v1_config(**overrides: object) -> SimpleNamespace:
    kmesh: dict[str, object] = {
        "enabled": True,
        "mode": "serial_pic",
        "segment_encoding": "sideband",
        "pic_backend": "lmcache",
        "check_layer": 1,
        "recompute_ratio": 0.15,
        "max_group_size": 10,
        "max_group_wait_ms": 0,
        "require_same_active_length": True,
        "fail_open": False,
        "strict_validation": True,
        "master_selection": "min_total_changed_blocks",
        "storage_mode": "none",
    }
    kmesh.update(overrides)
    config = _vllm_config(kmesh)
    config.device_config.device_type = "cuda"
    config.kv_transfer_config = KVTransferConfig(
        kv_connector="MultiConnector",
        kv_role="kv_both",
        engine_id="kmesh-serial-pic-v1-test",
        kv_buffer_device="cuda",
        kv_buffer_size=1e9,
        kv_rank=None,
        kv_parallel_size=1,
        kv_ip="127.0.0.1",
        kv_port=14579,
        kv_connector_module_path=None,
        enable_permute_local_kv=False,
        kv_load_failure_policy="recompute",
        kv_connector_extra_config={
            KMESH_TARGET_PROFILE_KEY: KMESH_SERIAL_PIC_V1_PROFILE,
            "connectors": [
                {
                    "kv_connector": "KMeshConnector",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {"kmesh": kmesh},
                },
                {
                    "kv_connector": "LMCacheConnectorV1",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {"use_native": False},
                },
            ],
        },
    )
    return config


def _collective_lockstep_v1_config(**overrides: object) -> SimpleNamespace:
    config = _serial_pic_v1_config(mode="collective_lockstep", **overrides)
    config.kv_transfer_config.engine_id = "kmesh-collective-lockstep-v1-test"
    config.kv_transfer_config.kv_connector_extra_config[KMESH_TARGET_PROFILE_KEY] = (
        KMESH_COLLECTIVE_LOCKSTEP_V1_PROFILE
    )
    return config


class _SerialPICAdapterProbe:
    """Public protocol-v2 peer used to observe composite handoffs."""

    KMESH_HANDOFF_PROTOCOL_VERSION = 2
    prepared_sidecars: dict[str, tuple[KVConnectorRequestSidecar, ...]] = {}
    bound_peer_metadata: tuple[KVConnectorMetadata, ...] | None = None

    def __init__(self, *args: object) -> None:
        pass

    @classmethod
    def reset(cls) -> None:
        cls.prepared_sidecars = {}
        cls.bound_peer_metadata = None

    def prepare_request_lookup(
        self,
        request: SimpleNamespace,
        peer_sidecars: tuple[KVConnectorRequestSidecar, ...],
    ) -> None:
        type(self).prepared_sidecars[request.request_id] = peer_sidecars

    def validate_peer_connector_metadata(self, *args: object) -> None:
        pass

    def bind_peer_connector_metadata(
        self, peer_connector_metadata: tuple[KVConnectorMetadata, ...]
    ) -> None:
        type(self).bound_peer_metadata = peer_connector_metadata

    def clear_peer_connector_metadata(self) -> None:
        pass

    def begin_connector_metadata_transaction(self) -> None:
        pass

    def commit_connector_metadata_transaction(self) -> None:
        pass

    def finalize_connector_metadata_transaction(self) -> None:
        pass

    def abort_connector_metadata_transaction(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def get_num_new_matched_tokens(self, *args: object) -> int:
        return 0

    def build_connector_meta(self, *args: object) -> KVConnectorMetadata:
        return KVConnectorMetadata()


@pytest.fixture
def serial_pic_connector(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., MultiConnector]:
    """Build serial-PIC only through a normally constructed MultiConnector."""

    module_name = "lmcache.integration.vllm.vllm_v1_adapter"
    adapter_module = ModuleType(module_name)
    adapter_module.LMCacheConnectorV1Impl = _SerialPICAdapterProbe  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module_name, adapter_module)
    monkeypatch.setattr(
        kmesh_config_module,
        "validate_kmesh_target_profile",
        lambda _config: KMESH_SERIAL_PIC_V1_PROFILE,
    )
    _SerialPICAdapterProbe.reset()

    def build(
        role: KVConnectorRole = KVConnectorRole.SCHEDULER,
        config: SimpleNamespace | None = None,
    ) -> MultiConnector:
        return MultiConnector(
            config or _serial_pic_v1_config(), role, _kv_cache_config()
        )

    return build


@pytest.fixture
def collective_lockstep_connector(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., MultiConnector]:
    """Build collective lockstep through the frozen sibling MultiConnector."""

    module_name = "lmcache.integration.vllm.vllm_v1_adapter"
    adapter_module = ModuleType(module_name)
    adapter_module.LMCacheConnectorV1Impl = _SerialPICAdapterProbe  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module_name, adapter_module)
    monkeypatch.setattr(
        kmesh_config_module,
        "validate_kmesh_target_profile",
        lambda _config: KMESH_COLLECTIVE_LOCKSTEP_V1_PROFILE,
    )
    _SerialPICAdapterProbe.reset()

    def build(
        role: KVConnectorRole = KVConnectorRole.SCHEDULER,
        config: SimpleNamespace | None = None,
    ) -> MultiConnector:
        return MultiConnector(
            config or _collective_lockstep_v1_config(), role, _kv_cache_config()
        )

    return build


def _captured_serial_pic_sidecars(
    connector: MultiConnector, request: SimpleNamespace
) -> tuple[KVConnectorRequestSidecar, ...]:
    connector.get_num_new_matched_tokens(request, 0)
    return _SerialPICAdapterProbe.prepared_sidecars[request.request_id]


def test_factory_registers_kmesh_connector() -> None:
    assert (
        KVConnectorFactory.get_connector_class_by_name("KMeshConnector")
        is KMeshConnector
    )


def test_factory_requires_hybrid_cache_manager_to_be_disabled() -> None:
    config = _vllm_config(_observer_config())
    config.scheduler_config.disable_hybrid_kv_cache_manager = False
    with pytest.raises(ValueError, match="does not support HMA"):
        KVConnectorFactory.create_connector(
            config, KVConnectorRole.SCHEDULER, _kv_cache_config()
        )

    config.scheduler_config.disable_hybrid_kv_cache_manager = True
    connector = KVConnectorFactory.create_connector(
        config, KVConnectorRole.SCHEDULER, _kv_cache_config()
    )
    assert isinstance(connector, KMeshConnector)


def test_observer_builds_and_transports_plan_without_kv_hits() -> None:
    config = _vllm_config(_observer_config())
    kv_cache_config = _kv_cache_config()
    scheduler = KMeshConnector(config, KVConnectorRole.SCHEDULER, kv_cache_config)
    first = _request("request-a", _context("request-a"))
    second = _request("request-b", _context("request-b"))

    scheduler.on_new_request(first)
    scheduler.on_new_request(second)
    assert scheduler.get_num_new_matched_tokens(first, 0) == (0, False)
    assert scheduler.get_num_new_matched_tokens(second, 0) == (0, False)
    assert scheduler.get_num_new_matched_tokens(first, 0) == (0, False)
    metadata = scheduler.build_connector_meta(_scheduler_output(first, second))

    assert isinstance(metadata, KMeshConnectorMetadata)
    assert metadata.plan_batch is not None
    assert metadata.plan_batch.metrics.observed_requests == 2
    assert metadata.plan_batch.metrics.formed_groups == 1
    assert metadata.plan_batch.plans[0].group.request_ids == (
        "request-a",
        "request-b",
    )
    assert metadata.step_generation == 1
    assert tuple(
        descriptor.request_id for descriptor in metadata.scheduled_descriptors
    ) == ("request-a", "request-b")
    assert all(
        descriptor.lookup_intent is KMeshRequestLookupIntent.OBSERVE_ONLY
        and not descriptor.permits_peer_lookup
        for descriptor in metadata.scheduled_descriptors
    )
    assert ForkingPickler.loads(ForkingPickler.dumps(metadata)) == metadata
    stats = scheduler.get_kv_connector_stats()
    assert isinstance(stats, KMeshConnectorStats)
    assert stats.data["planner_calls"] == 1
    assert stats.data["observed_requests"] == 2
    assert stats.data["formed_groups"] == 1
    assert stats.data["group_size_counts"] == {"2": 1}
    assert scheduler.get_kv_connector_stats() is None

    worker = KMeshConnector(config, KVConnectorRole.WORKER, kv_cache_config)
    worker.bind_connector_metadata(metadata)
    assert worker.get_kmesh_plan_batch() == metadata.plan_batch
    assert worker.get_kmesh_scheduled_descriptors() == (metadata.scheduled_descriptors)
    assert worker.get_kmesh_step_generation() == 1
    assert worker.has_connector_metadata()
    worker.clear_connector_metadata()
    assert worker.get_kmesh_plan_batch() is None
    assert worker.get_kmesh_scheduled_descriptors() == ()
    assert worker.get_kmesh_step_generation() == 0
    assert not worker.has_connector_metadata()

    with pytest.raises(ValueError, match="stale or was already consumed"):
        worker.bind_connector_metadata(metadata)

    next_metadata = scheduler.build_connector_meta(
        _cached_scheduler_output(first, second)
    )
    assert isinstance(next_metadata, KMeshConnectorMetadata)
    assert next_metadata.step_generation == 2
    assert tuple(
        descriptor.request_id for descriptor in next_metadata.scheduled_descriptors
    ) == ("request-a", "request-b")
    worker.bind_connector_metadata(next_metadata)
    assert worker.get_kmesh_step_generation() == 2


def test_metadata_rejects_plan_descriptor_drift_and_actionable_observer_intent():
    config = _vllm_config(_observer_config())
    kv_cache_config = _kv_cache_config()
    scheduler = KMeshConnector(config, KVConnectorRole.SCHEDULER, kv_cache_config)
    first = _request("request-a", _context("request-a"))
    second = _request("request-b", _context("request-b"))
    scheduler.on_new_request(first)
    scheduler.on_new_request(second)
    metadata = scheduler.build_connector_meta(_scheduler_output(first, second))
    assert isinstance(metadata, KMeshConnectorMetadata)

    with pytest.raises(TypeError, match="immutable tuple"):
        replace(metadata, scheduled_descriptors=[])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="nonnegative integer"):
        replace(metadata, step_generation=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="requires an admitted descriptor"):
        replace(metadata, scheduled_descriptors=())
    with pytest.raises(ValueError, match="compatibility drifted"):
        replace(
            metadata,
            scheduled_descriptors=(
                replace(metadata.scheduled_descriptors[0], cache_domain="tenant-b"),
                metadata.scheduled_descriptors[1],
            ),
        )
    actionable = replace(
        metadata.scheduled_descriptors[0],
        lookup_intent=KMeshRequestLookupIntent.SERIAL_PIC,
    )
    actionable_metadata = replace(
        metadata,
        scheduled_descriptors=(actionable, metadata.scheduled_descriptors[1]),
    )
    worker = KMeshConnector(config, KVConnectorRole.WORKER, kv_cache_config)
    with pytest.raises(ValueError, match="rejects actionable descriptors"):
        worker.bind_connector_metadata(actionable_metadata)


def test_scheduled_prompt_drift_drops_descriptor_and_plan_membership():
    config = _vllm_config(_observer_config())
    scheduler = KMeshConnector(config, KVConnectorRole.SCHEDULER, _kv_cache_config())
    request = _request("request-a", _context("request-a"))
    scheduler.on_new_request(request)
    scheduler_output = _scheduler_output(request)
    scheduler_output.scheduled_new_reqs[0].prompt_token_ids = [
        100,
        *request.prompt_token_ids[1:],
    ]

    metadata = scheduler.build_connector_meta(scheduler_output)

    assert isinstance(metadata, KMeshConnectorMetadata)
    assert metadata.scheduled_descriptors == ()
    assert metadata.plan_batch is not None
    assert metadata.plan_batch.outcome_for(request.request_id).fallback_reason is (
        KMeshFallbackReason.NO_METADATA
    )


def test_stale_same_id_scheduler_output_cannot_attach_a_newer_descriptor():
    config = _vllm_config(_observer_config())
    scheduler = KMeshConnector(config, KVConnectorRole.SCHEDULER, _kv_cache_config())
    old_request = _request("request-a", _context("request-a", domain="tenant-a"))
    scheduler.on_new_request(old_request)
    stale_output = _scheduler_output(old_request)

    new_request = _request("request-a", _context("request-a", domain="tenant-b"))
    scheduler.on_new_request(new_request)
    new_sidecar = scheduler.export_request_sidecar(new_request)
    assert new_sidecar is not None

    metadata = scheduler.build_connector_meta(stale_output)

    assert isinstance(metadata, KMeshConnectorMetadata)
    assert metadata.scheduled_descriptors == ()
    assert metadata.plan_batch is not None
    assert metadata.plan_batch.outcome_for(old_request.request_id).fallback_reason is (
        KMeshFallbackReason.NO_METADATA
    )


def test_stale_same_id_cached_output_cannot_attach_a_newer_descriptor():
    config = _vllm_config(_observer_config())
    scheduler = KMeshConnector(config, KVConnectorRole.SCHEDULER, _kv_cache_config())
    old_request = _request("request-a", _context("request-a", domain="tenant-a"))
    scheduler.on_new_request(old_request)
    first_step = scheduler.build_connector_meta(_scheduler_output(old_request))
    assert isinstance(first_step, KMeshConnectorMetadata)
    stale_output = _cached_scheduler_output(old_request)

    new_request = _request("request-a", _context("request-a", domain="tenant-b"))
    scheduler.on_new_request(new_request)
    new_sidecar = scheduler.export_request_sidecar(new_request)
    assert new_sidecar is not None

    metadata = scheduler.build_connector_meta(stale_output)

    assert isinstance(metadata, KMeshConnectorMetadata)
    assert metadata.scheduled_descriptors == ()
    assert metadata.plan_batch is not None
    assert metadata.plan_batch.outcome_for(old_request.request_id).fallback_reason is (
        KMeshFallbackReason.NO_METADATA
    )


def test_observer_exports_frozen_content_bound_lookup_candidate() -> None:
    config = _vllm_config(_observer_config())
    scheduler = KMeshConnector(config, KVConnectorRole.SCHEDULER, _kv_cache_config())
    request = _request("request-a", _context("request-a"))

    assert scheduler.export_request_sidecar(request) is None
    scheduler.on_new_request(request)
    sidecar = scheduler.export_request_sidecar(request)

    assert isinstance(sidecar, KVConnectorRequestSidecar)
    assert sidecar.provider_id == KMESH_REQUEST_SIDECAR_PROVIDER_ID
    descriptor = KMeshRequestLookupDescriptor.from_sidecar_payload(sidecar.payload)
    assert descriptor.lookup_intent is KMeshRequestLookupIntent.OBSERVE_ONLY
    assert descriptor.is_admitted
    assert not descriptor.permits_peer_lookup
    assert descriptor.request_id == "request-a"
    assert descriptor.admission_generation == 1
    assert descriptor.prompt_token_count == 16
    assert descriptor.cache_domain == "tenant-a"
    assert descriptor.admission_fallback is None
    assert tuple((span.token_start, span.token_end) for span in descriptor.spans) == (
        (4, 12),
    )
    assert descriptor.spans[0].shared_segment.content_digest == (
        kmesh_token_slice_digest(tuple(range(4, 12)))
    )
    assert scheduler.export_request_sidecar(request) == sidecar
    assert ForkingPickler.loads(ForkingPickler.dumps(sidecar)) == sidecar
    assert scheduler.request_finished(request, []) == (False, None)
    assert scheduler.export_request_sidecar(request) is None


def test_ttsep_admission_exports_the_same_normalized_span_contract() -> None:
    config = _vllm_config(_observer_config())
    scheduler = KMeshConnector(config, KVConnectorRole.SCHEDULER, _kv_cache_config())
    context = _context("request-a")
    context["segment_encoding"] = "ttsep"
    context["separator_token_ids"] = [99]
    segments = context["segments"]
    assert isinstance(segments, list)
    for segment in segments:
        assert isinstance(segment, dict)
        segment.pop("token_start")
        segment.pop("token_end")
    request = _request("request-a", context)
    request.prompt_token_ids = [*range(4), 99, *range(4, 12), 99, *range(12, 16)]
    request.num_prompt_tokens = len(request.prompt_token_ids)

    scheduler.on_new_request(request)
    sidecar = scheduler.export_request_sidecar(request)

    assert sidecar is not None
    descriptor = KMeshRequestLookupDescriptor.from_sidecar_payload(sidecar.payload)
    assert tuple((span.token_start, span.token_end) for span in descriptor.spans) == (
        (5, 13),
    )
    assert descriptor.spans[0].shared_segment.content_digest == (
        kmesh_token_slice_digest(tuple(range(4, 12)))
    )


def test_lookup_candidate_generation_advances_only_on_readmission() -> None:
    config = _vllm_config(_observer_config())
    scheduler = KMeshConnector(config, KVConnectorRole.SCHEDULER, _kv_cache_config())
    first_request = _request("request-a", _context("request-a"))

    scheduler.on_new_request(first_request)
    first = scheduler.export_request_sidecar(first_request)
    repeated = scheduler.export_request_sidecar(first_request)
    assert first is not None and repeated == first
    first_descriptor = KMeshRequestLookupDescriptor.from_sidecar_payload(first.payload)

    second_request = _request("request-a", _context("request-a"))
    scheduler.on_new_request(second_request)
    assert scheduler.export_request_sidecar(first_request) is None
    second = scheduler.export_request_sidecar(second_request)
    assert second is not None
    second_descriptor = KMeshRequestLookupDescriptor.from_sidecar_payload(
        second.payload
    )
    assert second_descriptor.admission_generation == (
        first_descriptor.admission_generation + 1
    )
    assert scheduler.request_finished(first_request, []) == (False, None)
    assert scheduler.export_request_sidecar(second_request) == second


def test_lookup_candidate_fails_closed_on_request_or_prompt_drift() -> None:
    config = _vllm_config(_observer_config())
    scheduler = KMeshConnector(config, KVConnectorRole.SCHEDULER, _kv_cache_config())
    request = _request("request-a", _context("request-a"))
    scheduler.on_new_request(request)
    assert scheduler.export_request_sidecar(request) is not None

    request.prompt_token_ids[4] = 100
    assert scheduler.export_request_sidecar(request) is None
    request.prompt_token_ids[4] = 4

    request.num_prompt_tokens = 15
    assert scheduler.export_request_sidecar(request) is None
    request.num_prompt_tokens = 16

    request.request_id = "request-b"
    assert scheduler.export_request_sidecar(request) is None


@pytest.mark.parametrize(
    "case", ["no_metadata", "disabled", "invalid", "domainless", "private"]
)
def test_lookup_candidate_fails_closed_without_eligible_admission(case: str) -> None:
    config = _vllm_config(_observer_config())
    scheduler = KMeshConnector(config, KVConnectorRole.SCHEDULER, _kv_cache_config())
    context = _context("request-a")
    if case == "no_metadata":
        request = _request("request-a", context)
        request.sampling_params = SamplingParams(max_tokens=1, extra_args={})
    elif case == "disabled":
        context["enabled"] = False
    elif case == "invalid":
        context["segments"][1]["token_end"] = 20  # type: ignore[index]
    elif case == "domainless":
        context["cache_domain"] = None
    else:
        context["segments"][1]["shareable"] = False  # type: ignore[index]
    if case != "no_metadata":
        request = _request("request-a", context)

    scheduler.on_new_request(request)

    assert scheduler.export_request_sidecar(request) is None


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("group_kind", "cot_branch", KMeshFallbackReason.GROUP_KIND_UNSUPPORTED),
        ("state_kind", "kv_diff", KMeshFallbackReason.STATE_KIND_UNSUPPORTED),
    ],
)
def test_lookup_candidate_rejects_unsupported_v0_semantics_before_peer_lookup(
    field: str, value: str, reason: KMeshFallbackReason
) -> None:
    config = _vllm_config(_observer_config())
    scheduler = KMeshConnector(config, KVConnectorRole.SCHEDULER, _kv_cache_config())
    context = _context("request-a")
    context[field] = value
    request = _request("request-a", context)

    scheduler.on_new_request(request)
    assert scheduler.export_request_sidecar(request) is None
    metadata = scheduler.build_connector_meta(_scheduler_output(request))

    assert isinstance(metadata, KMeshConnectorMetadata)
    assert metadata.plan_batch is not None
    assert metadata.plan_batch.outcome_for("request-a").fallback_reason is reason


def test_failed_strict_readmission_cannot_reuse_an_older_candidate() -> None:
    config = _vllm_config(_observer_config(strict_validation=True))
    scheduler = KMeshConnector(config, KVConnectorRole.SCHEDULER, _kv_cache_config())
    valid = _request("request-a", _context("request-a"))
    scheduler.on_new_request(valid)
    assert scheduler.export_request_sidecar(valid) is not None

    malformed = _context("request-a")
    malformed["segments"][1]["token_end"] = 20  # type: ignore[index]
    invalid = _request("request-a", malformed)
    with pytest.raises(KMeshValidationError):
        scheduler.on_new_request(invalid)

    assert scheduler.export_request_sidecar(valid) is None


def test_lookup_candidate_does_not_claim_a_selected_group_plan() -> None:
    config = _vllm_config(_observer_config())
    scheduler = KMeshConnector(config, KVConnectorRole.SCHEDULER, _kv_cache_config())
    request = _request("request-a", _context("request-a"))
    scheduler.on_new_request(request)

    assert scheduler.export_request_sidecar(request) is not None
    metadata = scheduler.build_connector_meta(_scheduler_output(request))

    assert isinstance(metadata, KMeshConnectorMetadata)
    assert metadata.plan_batch is not None
    assert metadata.plan_batch.plans == ()
    assert metadata.plan_batch.outcome_for("request-a").fallback_reason is (
        KMeshFallbackReason.SINGLETON_GROUP
    )


def test_observer_rejects_declared_shared_segment_with_different_tokens() -> None:
    config = _vllm_config(_observer_config())
    scheduler = KMeshConnector(config, KVConnectorRole.SCHEDULER, _kv_cache_config())
    first = _request("request-a", _context("request-a"))
    second = _request("request-b", _context("request-b"))
    second.prompt_token_ids[4:12] = range(100, 108)

    scheduler.on_new_request(first)
    scheduler.on_new_request(second)
    assert scheduler.get_num_new_matched_tokens(first, 0) == (0, False)
    assert scheduler.get_num_new_matched_tokens(second, 0) == (0, False)
    metadata = scheduler.build_connector_meta(_scheduler_output(first, second))

    assert isinstance(metadata, KMeshConnectorMetadata)
    assert metadata.plan_batch is not None
    assert metadata.plan_batch.plans == ()
    assert all(
        outcome.fallback_reason is KMeshFallbackReason.SEGMENT_CONTENT_MISMATCH
        for outcome in metadata.plan_batch.request_outcomes
    )


def test_observer_fails_open_with_structured_invalid_metadata() -> None:
    config = _vllm_config(_observer_config())
    scheduler = KMeshConnector(config, KVConnectorRole.SCHEDULER, _kv_cache_config())
    malformed = _context("request-a")
    malformed["segments"][1]["token_end"] = 20  # type: ignore[index]
    request = _request("request-a", malformed)

    scheduler.on_new_request(request)
    metadata = scheduler.build_connector_meta(_scheduler_output(request))

    assert isinstance(metadata, KMeshConnectorMetadata)
    assert metadata.plan_batch is not None
    assert metadata.plan_batch.outcome_for("request-a").fallback_reason is (
        KMeshFallbackReason.INVALID_METADATA
    )


@pytest.mark.parametrize(
    "strict_controls",
    [
        {"strict_validation": True},
        {"fail_open": False},
    ],
)
def test_observer_can_fail_closed_on_invalid_metadata(
    strict_controls: dict[str, object],
) -> None:
    config = _vllm_config(_observer_config(**strict_controls))
    scheduler = KMeshConnector(config, KVConnectorRole.SCHEDULER, _kv_cache_config())
    malformed = _context("request-a")
    malformed["segments"][1]["token_end"] = 20  # type: ignore[index]

    with pytest.raises(KMeshValidationError):
        scheduler.on_new_request(_request("request-a", malformed))


def test_missing_revision_fails_closed_as_incomplete_evidence() -> None:
    config = _vllm_config(_observer_config())
    config.model_config.revision = None
    scheduler = KMeshConnector(config, KVConnectorRole.SCHEDULER, _kv_cache_config())
    first = _request("request-a", _context("request-a"))
    second = _request("request-b", _context("request-b"))
    scheduler.on_new_request(first)
    scheduler.on_new_request(second)
    assert scheduler.export_request_sidecar(first) is None
    assert scheduler.export_request_sidecar(second) is None

    metadata = scheduler.build_connector_meta(_scheduler_output(first, second))

    assert isinstance(metadata, KMeshConnectorMetadata)
    assert metadata.plan_batch is not None
    assert metadata.plan_batch.plans == ()
    assert all(
        outcome.fallback_reason is KMeshFallbackReason.MISSING_COMPATIBILITY_EVIDENCE
        for outcome in metadata.plan_batch.request_outcomes
    )


def test_off_mode_does_not_parse_or_plan() -> None:
    config = _vllm_config({"enabled": False, "mode": "off"})
    connector = KMeshConnector(config, KVConnectorRole.SCHEDULER, _kv_cache_config())
    malformed = _context("request-a")
    malformed["segments"] = "not-a-list"
    request = _request("request-a", malformed)

    connector.on_new_request(request)
    assert connector.export_request_sidecar(request) is None
    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
    metadata = connector.build_connector_meta(_scheduler_output(request))

    assert isinstance(metadata, KMeshConnectorMetadata)
    assert metadata.plan_batch is None
    assert metadata.scheduled_descriptors == ()
    assert metadata.step_generation == 0
    with pytest.raises(TypeError, match="immutable tuple"):
        KMeshConnectorMetadata(None, [], 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="nonnegative integer"):
        KMeshConnectorMetadata(None, (), False)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "mode", ["serial_pic", "collective_lockstep", "collective_batched"]
)
def test_unqualified_recovery_modes_are_rejected(mode: str) -> None:
    config = _vllm_config({"enabled": True, "mode": mode})

    with pytest.raises(ValueError, match="no qualified runtime backend"):
        KMeshConnector(config, KVConnectorRole.SCHEDULER, _kv_cache_config())


def test_serial_pic_v1_exports_actionable_descriptors_only_under_named_profile(
    serial_pic_connector: Callable[..., MultiConnector],
) -> None:
    config = _serial_pic_v1_config()
    scheduler = serial_pic_connector(config=config)
    first = _request("request-a", _context("request-a"))
    second = _request("request-b", _context("request-b"))

    scheduler.on_new_request(first)
    scheduler.on_new_request(second)
    sidecars = _captured_serial_pic_sidecars(scheduler, first)
    metadata = scheduler.build_connector_meta(_scheduler_output(first, second))

    assert isinstance(metadata, MultiKVConnectorMetadata)
    kmesh_metadata = metadata.metadata[0]
    assert isinstance(kmesh_metadata, KMeshConnectorMetadata)
    sidecar = next(
        item
        for item in sidecars
        if item.provider_id == KMESH_REQUEST_SIDECAR_PROVIDER_ID
    )
    descriptor = KMeshRequestLookupDescriptor.from_sidecar_payload(sidecar.payload)
    assert descriptor.lookup_intent is KMeshRequestLookupIntent.SERIAL_PIC
    assert descriptor.permits_peer_lookup
    assert kmesh_metadata.scheduled_descriptors
    assert all(
        item.lookup_intent is KMeshRequestLookupIntent.SERIAL_PIC
        and item.permits_peer_lookup
        for item in kmesh_metadata.scheduled_descriptors
    )

    worker = serial_pic_connector(KVConnectorRole.WORKER, config)
    worker.bind_connector_metadata(metadata)
    mismatched = replace(
        kmesh_metadata,
        scheduled_descriptors=(
            replace(
                kmesh_metadata.scheduled_descriptors[0],
                lookup_intent=KMeshRequestLookupIntent.OBSERVE_ONLY,
            ),
            *kmesh_metadata.scheduled_descriptors[1:],
        ),
    )
    mismatched_multi = replace(
        metadata,
        metadata=(mismatched, *metadata.metadata[1:]),
    )
    mismatched_worker = serial_pic_connector(KVConnectorRole.WORKER, config)
    with pytest.raises(ValueError, match="requires serial_pic descriptors"):
        mismatched_worker.bind_connector_metadata(mismatched_multi)


def test_collective_lockstep_v1_transports_actionable_ordered_group_plan(
    collective_lockstep_connector: Callable[..., MultiConnector],
) -> None:
    config = _collective_lockstep_v1_config()
    scheduler = collective_lockstep_connector(config=config)
    first = _request("request-a", _context("request-a"))
    second = _request("request-b", _context("request-b"))

    scheduler.on_new_request(first)
    scheduler.on_new_request(second)
    sidecars = _captured_serial_pic_sidecars(scheduler, first)
    metadata = scheduler.build_connector_meta(_scheduler_output(first, second))

    kmesh_metadata = metadata.metadata[0]
    assert isinstance(kmesh_metadata, KMeshConnectorMetadata)
    assert kmesh_metadata.plan_batch is not None
    assert len(kmesh_metadata.plan_batch.plans) == 1
    plan = kmesh_metadata.plan_batch.plans[0]
    assert plan.group.request_ids == ("request-a", "request-b")
    assert (
        tuple(mapping.request_id for mapping in plan.request_segment_maps)
        == plan.group.request_ids
    )
    assert all(
        tuple(entry.shared_segment for entry in mapping.entries)
        == plan.group.shared_segments
        for mapping in plan.request_segment_maps
    )
    sidecar = next(
        item
        for item in sidecars
        if item.provider_id == KMESH_REQUEST_SIDECAR_PROVIDER_ID
    )
    descriptor = KMeshRequestLookupDescriptor.from_sidecar_payload(sidecar.payload)
    assert descriptor.lookup_intent is KMeshRequestLookupIntent.SERIAL_PIC
    assert descriptor.permits_peer_lookup
    assert all(
        item.lookup_intent is KMeshRequestLookupIntent.SERIAL_PIC
        and item.permits_peer_lookup
        for item in kmesh_metadata.scheduled_descriptors
    )

    mismatched = replace(
        kmesh_metadata,
        scheduled_descriptors=(
            replace(
                kmesh_metadata.scheduled_descriptors[0],
                lookup_intent=KMeshRequestLookupIntent.OBSERVE_ONLY,
            ),
            *kmesh_metadata.scheduled_descriptors[1:],
        ),
    )
    mismatched_worker = collective_lockstep_connector(KVConnectorRole.WORKER, config)
    with pytest.raises(
        ValueError, match="collective_lockstep mode requires serial_pic descriptors"
    ):
        mismatched_worker.bind_connector_metadata(
            replace(metadata, metadata=(mismatched, *metadata.metadata[1:]))
        )

    worker = collective_lockstep_connector(KVConnectorRole.WORKER, config)
    worker.bind_connector_metadata(metadata)
    bound_metadata = _SerialPICAdapterProbe.bound_peer_metadata
    assert bound_metadata is not None
    assert bound_metadata[0] is kmesh_metadata
    assert isinstance(bound_metadata[0], KMeshConnectorMetadata)
    assert bound_metadata[0].plan_batch is kmesh_metadata.plan_batch


def test_collective_lockstep_exports_explicit_no_shared_fallback(
    collective_lockstep_connector: Callable[..., MultiConnector],
) -> None:
    scheduler = collective_lockstep_connector()
    context = _without_sharing(_context("request-a"))
    request = _request("request-a", context)

    scheduler.on_new_request(request)
    sidecars = _captured_serial_pic_sidecars(scheduler, request)

    assert len(sidecars) == 1
    sidecar = sidecars[0]
    assert sidecar.provider_id == KMESH_REQUEST_SIDECAR_PROVIDER_ID
    descriptor = KMeshRequestLookupDescriptor.from_sidecar_payload(sidecar.payload)
    assert descriptor.lookup_intent is KMeshRequestLookupIntent.SERIAL_PIC
    assert descriptor.request_id == request.request_id
    assert descriptor.admission_generation == 1
    assert descriptor.prompt_token_count == request.num_prompt_tokens
    assert descriptor.cache_domain == "tenant-a"
    assert descriptor.admission_fallback is KMeshFallbackReason.NO_SHARED_SEGMENTS
    assert descriptor.spans == ()
    assert not descriptor.is_admitted
    assert not descriptor.permits_peer_lookup
    assert _captured_serial_pic_sidecars(scheduler, request) == sidecars

    metadata = scheduler.build_connector_meta(_scheduler_output(request))
    kmesh_metadata = metadata.metadata[0]
    assert isinstance(kmesh_metadata, KMeshConnectorMetadata)
    assert kmesh_metadata.plan_batch is not None
    assert kmesh_metadata.plan_batch.plans == ()
    assert kmesh_metadata.plan_batch.outcome_for(
        request.request_id
    ).fallback_reason is (KMeshFallbackReason.NO_SHARED_SEGMENTS)
    assert kmesh_metadata.scheduled_descriptors == (descriptor,)


def test_no_shared_fallback_export_is_collective_target_only(
    serial_pic_connector: Callable[..., MultiConnector],
) -> None:
    scheduler = serial_pic_connector()
    context = _without_sharing(_context("request-a"))
    request = _request("request-a", context)

    scheduler.on_new_request(request)

    assert _captured_serial_pic_sidecars(scheduler, request) == ()


def test_collective_no_shared_fallback_rejects_request_drift(
    collective_lockstep_connector: Callable[..., MultiConnector],
) -> None:
    scheduler = collective_lockstep_connector()
    context = _without_sharing(_context("request-a"))
    request = _request("request-a", context)
    scheduler.on_new_request(request)
    assert len(_captured_serial_pic_sidecars(scheduler, request)) == 1

    request.prompt_token_ids[0] = 100

    assert _captured_serial_pic_sidecars(scheduler, request) == ()


def test_collective_does_not_export_other_fallback_reasons(
    collective_lockstep_connector: Callable[..., MultiConnector],
) -> None:
    scheduler = collective_lockstep_connector()
    context = _context("request-a")
    context["cache_domain"] = None
    request = _request("request-a", context)

    scheduler.on_new_request(request)

    assert _captured_serial_pic_sidecars(scheduler, request) == ()


def test_collective_lockstep_groups_shifted_external_recovery_windows(
    collective_lockstep_connector: Callable[..., MultiConnector],
) -> None:
    scheduler = collective_lockstep_connector()
    requests = tuple(
        _offset_collective_request(f"request-{index}", offset=index - 1)
        for index in range(1, 5)
    )
    for request in requests:
        scheduler.on_new_request(request)

    metadata = scheduler.build_connector_meta(
        _scheduler_output(
            *requests,
            num_computed_tokens=(440, 441, 442, 443),
        )
    )

    kmesh_metadata = metadata.metadata[0]
    assert isinstance(kmesh_metadata, KMeshConnectorMetadata)
    assert kmesh_metadata.plan_batch is not None
    assert len(kmesh_metadata.plan_batch.plans) == 1
    plan = kmesh_metadata.plan_batch.plans[0]
    assert plan.group.request_ids == tuple(request.request_id for request in requests)
    assert plan.group.compatibility_key.cached_span == (0, 0)
    assert plan.group.compatibility_key.prefill_chunk_span == (0, 512)
    assert [
        [(span.token_start, span.token_end) for span in descriptor.spans]
        for descriptor in kmesh_metadata.scheduled_descriptors
    ] == [[(24, 440)], [(25, 441)], [(26, 442)], [(27, 443)]]
    assert [
        [(entry.token_start, entry.token_end) for entry in mapping.entries]
        for mapping in plan.request_segment_maps
    ] == [[(24, 440)], [(25, 441)], [(26, 442)], [(27, 443)]]


def test_serial_pic_keeps_external_recovery_windows_request_local(
    serial_pic_connector: Callable[..., MultiConnector],
) -> None:
    scheduler = serial_pic_connector()
    requests = tuple(
        _offset_collective_request(f"request-{index}", offset=index - 1)
        for index in range(1, 5)
    )
    for request in requests:
        scheduler.on_new_request(request)

    metadata = scheduler.build_connector_meta(
        _scheduler_output(
            *requests,
            num_computed_tokens=(440, 441, 442, 443),
        )
    )

    kmesh_metadata = metadata.metadata[0]
    assert isinstance(kmesh_metadata, KMeshConnectorMetadata)
    assert kmesh_metadata.plan_batch is not None
    assert kmesh_metadata.plan_batch.plans == ()
    assert all(
        outcome.fallback_reason is KMeshFallbackReason.CACHED_SPAN_MISMATCH
        for outcome in kmesh_metadata.plan_batch.request_outcomes
    )


def test_collective_lockstep_keeps_cache_domain_in_group_identity(
    collective_lockstep_connector: Callable[..., MultiConnector],
) -> None:
    scheduler = collective_lockstep_connector()
    first = _offset_collective_request("request-a", offset=0)
    second = _offset_collective_request("request-b", offset=1, domain="tenant-b")
    scheduler.on_new_request(first)
    scheduler.on_new_request(second)

    metadata = scheduler.build_connector_meta(
        _scheduler_output(
            first,
            second,
            num_computed_tokens=(440, 441),
        )
    )

    kmesh_metadata = metadata.metadata[0]
    assert isinstance(kmesh_metadata, KMeshConnectorMetadata)
    assert kmesh_metadata.plan_batch is not None
    assert kmesh_metadata.plan_batch.plans == ()
    assert all(
        outcome.fallback_reason is KMeshFallbackReason.CACHE_DOMAIN_MISMATCH
        for outcome in kmesh_metadata.plan_batch.request_outcomes
    )


def test_collective_lockstep_keeps_active_length_in_group_identity(
    collective_lockstep_connector: Callable[..., MultiConnector],
) -> None:
    scheduler = collective_lockstep_connector()
    first = _offset_collective_request("request-a", offset=0)
    second = _offset_collective_request("request-b", offset=1, prompt_length=513)
    scheduler.on_new_request(first)
    scheduler.on_new_request(second)

    metadata = scheduler.build_connector_meta(
        _scheduler_output(
            first,
            second,
            num_computed_tokens=(440, 441),
        )
    )

    kmesh_metadata = metadata.metadata[0]
    assert isinstance(kmesh_metadata, KMeshConnectorMetadata)
    assert kmesh_metadata.plan_batch is not None
    assert kmesh_metadata.plan_batch.plans == ()
    assert all(
        outcome.fallback_reason is KMeshFallbackReason.ACTIVE_LENGTH_MISMATCH
        for outcome in kmesh_metadata.plan_batch.request_outcomes
    )


def test_collective_lockstep_keeps_shared_content_in_group_identity(
    collective_lockstep_connector: Callable[..., MultiConnector],
) -> None:
    scheduler = collective_lockstep_connector()
    first = _offset_collective_request("request-a", offset=0)
    second = _offset_collective_request("request-b", offset=1)
    second.prompt_token_ids[25] = 9999
    scheduler.on_new_request(first)
    scheduler.on_new_request(second)

    metadata = scheduler.build_connector_meta(
        _scheduler_output(
            first,
            second,
            num_computed_tokens=(440, 441),
        )
    )

    kmesh_metadata = metadata.metadata[0]
    assert isinstance(kmesh_metadata, KMeshConnectorMetadata)
    assert kmesh_metadata.plan_batch is not None
    assert kmesh_metadata.plan_batch.plans == ()
    assert all(
        outcome.fallback_reason is KMeshFallbackReason.SEGMENT_CONTENT_MISMATCH
        for outcome in kmesh_metadata.plan_batch.request_outcomes
    )


def test_collective_lockstep_rejects_incomplete_full_prompt_schedule(
    collective_lockstep_connector: Callable[..., MultiConnector],
) -> None:
    scheduler = collective_lockstep_connector()
    first = _offset_collective_request("request-a", offset=0)
    second = _offset_collective_request("request-b", offset=1)
    scheduler.on_new_request(first)
    scheduler.on_new_request(second)

    metadata = scheduler.build_connector_meta(
        _scheduler_output(
            first,
            second,
            num_computed_tokens=(440, 441),
            num_scheduled_tokens=(71, 70),
        )
    )

    kmesh_metadata = metadata.metadata[0]
    assert isinstance(kmesh_metadata, KMeshConnectorMetadata)
    assert kmesh_metadata.plan_batch is not None
    assert kmesh_metadata.plan_batch.plans == ()
    assert all(
        outcome.fallback_reason is KMeshFallbackReason.MISSING_COMPATIBILITY_EVIDENCE
        for outcome in kmesh_metadata.plan_batch.request_outcomes
    )
    assert [
        [(span.token_start, span.token_end) for span in descriptor.spans]
        for descriptor in kmesh_metadata.scheduled_descriptors
    ] == [[(24, 440)], [(25, 441)]]


@pytest.mark.parametrize(
    ("resumed", "num_computed_tokens", "num_scheduled_tokens", "actionable"),
    [
        pytest.param(False, 0, 16, False, id="local-full-recompute"),
        pytest.param(True, 0, 16, True, id="resumed-full-prefill"),
        pytest.param(False, 0, 8, True, id="partial-from-zero"),
        pytest.param(False, 4, 8, True, id="partial-from-prefix"),
    ],
)
def test_serial_pic_v1_suppresses_only_local_full_recompute_descriptor(
    resumed: bool,
    num_computed_tokens: int,
    num_scheduled_tokens: int,
    actionable: bool,
    serial_pic_connector: Callable[..., MultiConnector],
) -> None:
    scheduler = serial_pic_connector()
    request = _request("request-a", _context("request-a"))
    scheduler.on_new_request(request)
    first = scheduler.build_connector_meta(_scheduler_output(request))
    first_kmesh = first.metadata[0]
    assert isinstance(first_kmesh, KMeshConnectorMetadata)
    assert len(first_kmesh.scheduled_descriptors) == 1

    retry = scheduler.build_connector_meta(
        _serial_pic_cached_scheduler_output(
            request,
            num_computed_tokens=num_computed_tokens,
            num_scheduled_tokens=num_scheduled_tokens,
            resumed=resumed,
        )
    )

    retry_kmesh = retry.metadata[0]
    assert isinstance(retry_kmesh, KMeshConnectorMetadata)
    assert bool(retry_kmesh.scheduled_descriptors) is actionable
    assert retry_kmesh.plan_batch is not None
    outcome = retry_kmesh.plan_batch.outcome_for(request.request_id)
    if actionable:
        assert outcome.fallback_reason is KMeshFallbackReason.SINGLETON_GROUP
    else:
        assert outcome.fallback_reason is KMeshFallbackReason.NO_METADATA

    worker = serial_pic_connector(KVConnectorRole.WORKER)
    worker.bind_connector_metadata(retry)
    worker.clear_connector_metadata()


@pytest.mark.parametrize(
    ("mode", "enabled"),
    [
        ("off", False),
        ("observe", True),
        ("collective_lockstep", True),
        ("collective_batched", True),
        ("master_mirror_reference", True),
    ],
)
def test_serial_pic_v1_profile_rejects_nonserial_modes(
    mode: str,
    enabled: bool,
    serial_pic_connector: Callable[..., MultiConnector],
) -> None:
    config = _serial_pic_v1_config(mode=mode, enabled=enabled)

    with pytest.raises(ValueError, match="requires enabled mode='serial_pic'"):
        serial_pic_connector(config=config)


def test_serial_pic_v1_profile_rejects_noop_backend(
    serial_pic_connector: Callable[..., MultiConnector],
) -> None:
    config = _serial_pic_v1_config(pic_backend="noop")

    with pytest.raises(ValueError, match="pic_backend='lmcache'"):
        serial_pic_connector(config=config)


def test_direct_child_profile_string_cannot_qualify_serial_pic(
    serial_pic_connector: Callable[..., MultiConnector],
) -> None:
    config = _serial_pic_v1_config()
    kmesh_cls, child_config = MultiConnector.inspect_connector_classes_and_configs(
        config
    )[0]

    with pytest.raises(ValueError, match="internally qualified outer MultiConnector"):
        kmesh_cls(child_config, KVConnectorRole.SCHEDULER, _kv_cache_config())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("use_structured_output", True, "structured output requests"),
        ("mm_features", [object()], "multimodal requests"),
        ("lora_request", object(), "LoRA requests"),
        ("cache_salt", "salt", "cache_salt"),
        (
            "kv_transfer_params",
            {"disagg_spec": {"req_id": "remote"}},
            "kv_transfer_params absent",
        ),
        ("prompt_embeds", torch.zeros(16, 8), "prompt embeddings"),
        ("prompt_is_token_ids", [True] * 16, "mixed token/embedding prompt masks"),
        ("pooling_params", object(), "pooling requests"),
        ("resumable", True, "non-resumable requests"),
        ("streaming_queue", object(), "streaming continuation state"),
        ("spec_token_ids", [1], "speculative tokens"),
    ],
)
def test_serial_pic_v1_rejects_request_local_global_feature_bypasses(
    field: str,
    value: object,
    message: str,
    serial_pic_connector: Callable[..., MultiConnector],
) -> None:
    connector = serial_pic_connector()
    request = _request("request-a", _context("request-a"))
    setattr(request, field, value)

    with pytest.raises(ValueError, match=message):
        connector.on_new_request(request)


def test_serial_pic_v1_rejects_non_token_prompt(
    serial_pic_connector: Callable[..., MultiConnector],
) -> None:
    connector = serial_pic_connector()
    request = _request("request-a", _context("request-a"))
    request.prompt_token_ids = None

    with pytest.raises(ValueError, match="token-ID prompts"):
        connector.on_new_request(request)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("prompt_logprobs", 1, "prompt_logprobs"),
        ("skip_reading_prefix_cache", True, "skip_reading_prefix_cache=false"),
        ("extra_args", {"kmesh": {}, "kv_transfer_params": {}}, "only the kmesh"),
    ],
)
def test_serial_pic_v1_rejects_sampling_bypasses(
    field: str,
    value: object,
    message: str,
    serial_pic_connector: Callable[..., MultiConnector],
) -> None:
    connector = serial_pic_connector()
    request = _request("request-a", _context("request-a"))
    setattr(request.sampling_params, field, value)

    with pytest.raises(ValueError, match=message):
        connector.on_new_request(request)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("use_structured_output", True),
        ("mm_features", [object()]),
        ("lora_request", object()),
        ("cache_salt", "salt"),
        ("kv_transfer_params", {"disagg_spec": {"req_id": "remote"}}),
        ("prompt_embeds", torch.zeros(16, 8)),
        ("prompt_is_token_ids", [True] * 16),
        ("pooling_params", object()),
        ("resumable", True),
        ("streaming_queue", object()),
        ("spec_token_ids", [1]),
    ],
)
def test_serial_pic_v1_request_surface_drift_revokes_actionable_sidecar(
    field: str,
    value: object,
    serial_pic_connector: Callable[..., MultiConnector],
) -> None:
    connector = serial_pic_connector()
    request = _request("request-a", _context("request-a"))
    connector.on_new_request(request)
    assert _captured_serial_pic_sidecars(connector, request)

    setattr(request, field, value)

    assert not _captured_serial_pic_sidecars(connector, request)


def test_serial_pic_v1_sampling_context_drift_revokes_actionable_sidecar(
    serial_pic_connector: Callable[..., MultiConnector],
) -> None:
    connector = serial_pic_connector()
    request = _request("request-a", _context("request-a"))
    connector.on_new_request(request)
    assert _captured_serial_pic_sidecars(connector, request)

    request.sampling_params.extra_args["kmesh"]["cache_domain"] = "tenant-b"

    assert not _captured_serial_pic_sidecars(connector, request)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prompt_logprobs", 1),
        ("skip_reading_prefix_cache", True),
        ("extra_args", {"kmesh": {}, "kv_transfer_params": {}}),
    ],
)
def test_serial_pic_v1_sampling_surface_drift_revokes_actionable_sidecar(
    field: str,
    value: object,
    serial_pic_connector: Callable[..., MultiConnector],
) -> None:
    connector = serial_pic_connector()
    request = _request("request-a", _context("request-a"))
    connector.on_new_request(request)
    assert _captured_serial_pic_sidecars(connector, request)

    setattr(request.sampling_params, field, value)

    assert not _captured_serial_pic_sidecars(connector, request)


def _mutate_scheduled_sampling(
    request: NewRequestData, field: str, value: object
) -> None:
    sampling_params = SamplingParams(
        max_tokens=1,
        extra_args={"kmesh": _context(request.req_id)},
    )
    setattr(sampling_params, field, value)
    request.sampling_params = sampling_params


@pytest.mark.parametrize(
    "mutation",
    [
        lambda request: setattr(request, "prompt_embeds", torch.zeros(16, 8)),
        lambda request: setattr(request, "prompt_is_token_ids", [True] * 16),
        lambda request: setattr(request, "mm_features", [object()]),
        lambda request: setattr(request, "pooling_params", object()),
        lambda request: setattr(request, "lora_request", object()),
        lambda request: setattr(
            request,
            "kv_transfer_params",
            {"disagg_spec": {"req_id": "remote"}},
        ),
        lambda request: _mutate_scheduled_sampling(request, "prompt_logprobs", 1),
        lambda request: _mutate_scheduled_sampling(
            request, "skip_reading_prefix_cache", True
        ),
        lambda request: setattr(request, "prefill_token_ids", None),
        lambda request: setattr(request, "prefill_token_ids", list(range(15))),
    ],
)
def test_serial_pic_v1_scheduled_new_request_drift_revokes_descriptor(
    mutation: Callable[[NewRequestData], None],
    serial_pic_connector: Callable[..., MultiConnector],
) -> None:
    connector = serial_pic_connector()
    request = _request("request-a", _context("request-a"))
    connector.on_new_request(request)
    output = _scheduler_output(request)
    scheduled = output.scheduled_new_reqs[0]

    mutation(scheduled)
    metadata = connector.build_connector_meta(output)

    kmesh_metadata = metadata.metadata[0]
    assert isinstance(kmesh_metadata, KMeshConnectorMetadata)
    assert kmesh_metadata.scheduled_descriptors == ()
    assert kmesh_metadata.plan_batch is not None
    assert kmesh_metadata.plan_batch.metrics.candidate_requests == 0


def test_serial_pic_v1_admitted_transfer_params_drift_revokes_scheduled_request(
    serial_pic_connector: Callable[..., MultiConnector],
) -> None:
    connector = serial_pic_connector()
    request = _request("request-a", _context("request-a"))
    connector.on_new_request(request)
    output = _scheduler_output(request)

    request.kv_transfer_params = {"disagg_spec": {"req_id": "remote"}}
    metadata = connector.build_connector_meta(output)

    kmesh_metadata = metadata.metadata[0]
    assert isinstance(kmesh_metadata, KMeshConnectorMetadata)
    assert kmesh_metadata.scheduled_descriptors == ()
    assert kmesh_metadata.plan_batch is not None
    assert kmesh_metadata.plan_batch.metrics.candidate_requests == 0


def test_serial_pic_v1_frozen_qualification_rejects_global_drift(
    serial_pic_connector: Callable[..., MultiConnector],
) -> None:
    config = _serial_pic_v1_config()
    connector = serial_pic_connector(config=config)
    request = _request("request-a", _context("request-a"))
    config.parallel_config.tensor_parallel_size = 7

    with pytest.raises(ValueError, match="configuration drifted"):
        connector.on_new_request(request)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda config: setattr(
            config.model_config.hf_text_config, "rms_norm_eps", 0.01
        ),
        lambda config: setattr(config.model_config.hf_text_config, "head_dim", 128),
        lambda config: setattr(config.offload_config.prefetch, "offload_group_size", 2),
        lambda config: setattr(
            config.attention_config,
            "backend",
            AttentionBackendEnum.TRITON_ATTN,
        ),
    ],
)
def test_serial_pic_v1_frozen_qualification_rejects_resolved_drift(
    mutation: Callable[[SimpleNamespace], None],
    serial_pic_connector: Callable[..., MultiConnector],
) -> None:
    config = _serial_pic_v1_config()
    connector = serial_pic_connector(config=config)
    request = _request("request-a", _context("request-a"))

    mutation(config)

    with pytest.raises(ValueError, match="configuration drifted"):
        connector.on_new_request(request)
