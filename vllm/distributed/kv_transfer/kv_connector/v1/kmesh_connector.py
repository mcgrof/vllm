# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRequestSidecar,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.kmesh_metrics import (
    KMeshConnectorStats,
    KMeshPromMetrics,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kmesh import (
    KMESH_COLLECTIVE_BATCHED_POSITION_RECOVERY_V1_PROFILE,
    KMESH_COLLECTIVE_BATCHED_QKV_V1_PROFILE,
    KMESH_COLLECTIVE_LOCKSTEP_V1_PROFILE,
    KMESH_COLLECTIVE_PROFILES,
    KMESH_REQUEST_SIDECAR_PROVIDER_ID,
    KMESH_SERIAL_PIC_V1_PROFILE,
    KMESH_TARGET_PROFILE_KEY,
    KMeshCompatibilityKey,
    KMeshEngineConfig,
    KMeshFallbackReason,
    KMeshMode,
    KMeshPlanBatch,
    KMeshPlanner,
    KMeshRequestContext,
    KMeshRequestContextParseResult,
    KMeshRequestLookupDescriptor,
    KMeshRequestLookupIntent,
    KMeshRequestObservation,
    parse_kmesh_extra_args,
)
from vllm.v1.kmesh.config import (
    _is_kmesh_target_profile_child_qualified,
    _target_profile_mode,
    _validate_kmesh_target_profile_child_runtime,
    _validate_target_profile_kmesh_config,
)

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)

_ACTIONABLE_PIC_MODES = frozenset(
    (
        KMeshMode.SERIAL_PIC,
        KMeshMode.COLLECTIVE_LOCKSTEP,
        KMeshMode.COLLECTIVE_BATCHED,
    )
)


def _json_value(value: object) -> object:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in fields(value)
        }
    return f"{type(value).__module__}.{type(value).__qualname__}:{value!s}"


def _fingerprint(label: str, value: object) -> str:
    encoded = json.dumps(
        _json_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return f"{label}-sha256:{hashlib.sha256(encoded).hexdigest()}"


def _bounded_identity(label: str, value: object) -> str:
    rendered = str(value)
    if rendered and rendered == rendered.strip() and len(rendered) <= 256:
        return rendered
    return _fingerprint(label, value)


def _optional_identity(label: str, value: object) -> str | None:
    return None if value is None else _bounded_identity(label, value)


def _lora_identity(lora_request: object) -> str | None:
    if lora_request is None:
        return None
    return _fingerprint(
        "lora",
        {
            "name": getattr(lora_request, "lora_name", None),
            "path": getattr(lora_request, "lora_path", None),
            "base_model": getattr(lora_request, "base_model_name", None),
            "load_inplace": getattr(lora_request, "load_inplace", None),
            "is_3d": getattr(lora_request, "is_3d_lora_weight", None),
        },
    )


def _serial_pic_v1_request_violation(request: Request) -> str | None:
    """Return the first request-local surface outside the frozen profile."""

    if getattr(request, "prompt_token_ids", None) is None:
        return "token-ID prompts"
    if getattr(request, "prompt_embeds", None) is not None:
        return "prompt embeddings and mixed token/embedding prompts"
    if getattr(request, "prompt_is_token_ids", None) is not None:
        return "mixed token/embedding prompt masks"
    if getattr(request, "use_structured_output", False):
        return "structured output requests"
    if getattr(request, "mm_features", None):
        return "multimodal requests"
    if getattr(request, "lora_request", None) is not None:
        return "LoRA requests"
    if getattr(request, "cache_salt", None) is not None:
        return "request cache_salt"
    if getattr(request, "kv_transfer_params", None) is not None:
        return "request kv_transfer_params absent"
    if getattr(request, "pooling_params", None) is not None:
        return "pooling requests"
    sampling_params = getattr(request, "sampling_params", None)
    if sampling_params is None:
        return "generation sampling parameters"
    if getattr(sampling_params, "prompt_logprobs", None) is not None:
        return "prompt_logprobs requests"
    if getattr(sampling_params, "skip_reading_prefix_cache", False) is not False:
        return "skip_reading_prefix_cache=false"
    extra_args = getattr(sampling_params, "extra_args", None)
    if extra_args is not None and (
        not isinstance(extra_args, Mapping) or set(extra_args) != {"kmesh"}
    ):
        return "only the kmesh sampling extra_args entry"
    if getattr(request, "resumable", False):
        return "non-resumable requests"
    if getattr(request, "streaming_queue", None) is not None:
        return "requests without streaming continuation state"
    if getattr(request, "spec_token_ids", None):
        return "requests without speculative tokens"
    return None


def _serial_pic_v1_scheduled_request_violation(
    request: object, admitted_prompt_token_ids: tuple[int, ...]
) -> str | None:
    """Reject NewRequestData drift before it can renew admission evidence."""

    if violation := _serial_pic_v1_request_violation(request):  # type: ignore[arg-type]
        return violation
    prefill_token_ids = getattr(request, "prefill_token_ids", None)
    if prefill_token_ids is None:
        return "Model Runner V2 prefill token IDs"
    if tuple(prefill_token_ids) != admitted_prompt_token_ids:
        return "Model Runner V2 prefill token IDs matching the admitted prompt"
    return None


@dataclass(frozen=True, slots=True)
class KMeshRuntimeIdentity:
    """Engine-wide evidence that is identical for every scheduler request."""

    model_id: str
    model_revision: str | None
    tokenizer_id_or_revision: str | None
    dtype: str
    kv_cache_dtype: str
    kv_layout: str
    attention_kind: str
    rope_configuration: str
    attention_backend: str
    tensor_parallel_size: int
    pipeline_parallel_size: int

    @classmethod
    def from_vllm_config(
        cls, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
    ) -> KMeshRuntimeIdentity:
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        parallel_config = vllm_config.parallel_config
        attention_config = vllm_config.attention_config

        revision = getattr(model_config, "revision", None)
        tokenizer_revision = getattr(model_config, "tokenizer_revision", None)
        tokenizer = getattr(model_config, "tokenizer", None)
        tokenizer_identity = (
            _fingerprint(
                "tokenizer",
                {"tokenizer": tokenizer, "revision": tokenizer_revision},
            )
            if tokenizer_revision is not None
            else None
        )
        model_dtype = str(model_config.dtype).removeprefix("torch.")
        configured_kv_dtype = str(cache_config.cache_dtype)
        effective_kv_dtype = (
            model_dtype if configured_kv_dtype == "auto" else configured_kv_dtype
        )

        group_specs = [group.kv_cache_spec for group in kv_cache_config.kv_cache_groups]
        attention_kind = "+".join(type(spec).__name__ for spec in group_specs)
        if not attention_kind:
            attention_kind = "no-kv-cache-groups"
        kv_layout = _fingerprint(
            "kv-layout",
            [
                {
                    "layers": group.layer_names,
                    "is_eagle_group": group.is_eagle_group,
                    "spec": group.kv_cache_spec,
                }
                for group in kv_cache_config.kv_cache_groups
            ],
        )

        hf_config = model_config.hf_text_config
        rope_configuration = _fingerprint(
            "rope",
            {
                field: getattr(hf_config, field, None)
                for field in (
                    "max_position_embeddings",
                    "partial_rotary_factor",
                    "rope_parameters",
                    "rope_scaling",
                    "rope_theta",
                    "rotary_pct",
                )
            },
        )
        backend = getattr(attention_config, "backend", None)
        attention_backend = _fingerprint(
            "attention",
            {
                "backend": getattr(backend, "name", backend) or "auto",
                "config_hash": attention_config.compute_hash(),
                "device": str(vllm_config.device_config.device_type),
            },
        )
        return cls(
            model_id=_bounded_identity("model", model_config.model),
            model_revision=(
                _bounded_identity("model-revision", revision)
                if revision is not None
                else None
            ),
            tokenizer_id_or_revision=tokenizer_identity,
            dtype=model_dtype,
            kv_cache_dtype=effective_kv_dtype,
            kv_layout=kv_layout,
            attention_kind=_bounded_identity("attention-kind", attention_kind),
            rope_configuration=rope_configuration,
            attention_backend=attention_backend,
            tensor_parallel_size=parallel_config.tensor_parallel_size,
            pipeline_parallel_size=parallel_config.pipeline_parallel_size,
        )

    @property
    def is_complete(self) -> bool:
        return (
            self.model_revision is not None
            and self.tokenizer_id_or_revision is not None
        )

    def make_compatibility_key(
        self,
        context: KMeshRequestContext,
        *,
        lora_adapter_identity: str | None,
        cache_salt: str | None,
        active_prompt_length: int,
        num_computed_tokens: int,
        num_scheduled_tokens: int,
    ) -> KMeshCompatibilityKey | None:
        if not self.is_complete:
            return None
        chunk_end = num_computed_tokens + num_scheduled_tokens
        if (
            active_prompt_length <= 0
            or num_computed_tokens < 0
            or num_scheduled_tokens <= 0
            or chunk_end > active_prompt_length
        ):
            return None
        assert self.model_revision is not None
        assert self.tokenizer_id_or_revision is not None
        return KMeshCompatibilityKey(
            execution_id=context.execution_id,
            group_id=context.group_id,
            round_id=context.round_id,
            model_id=self.model_id,
            model_revision=self.model_revision,
            tokenizer_id_or_revision=self.tokenizer_id_or_revision,
            dtype=self.dtype,
            kv_cache_dtype=self.kv_cache_dtype,
            kv_layout=self.kv_layout,
            attention_kind=self.attention_kind,
            rope_configuration=self.rope_configuration,
            attention_backend=self.attention_backend,
            tensor_parallel_size=self.tensor_parallel_size,
            pipeline_parallel_size=self.pipeline_parallel_size,
            lora_adapter_identity=lora_adapter_identity,
            cache_salt=cache_salt,
            cache_domain=context.cache_domain,
            active_prompt_length=active_prompt_length,
            cached_span=(0, num_computed_tokens),
            prefill_chunk_span=(num_computed_tokens, chunk_end),
        )


@dataclass(frozen=True, slots=True)
class _KMeshAdmission:
    request: Request
    result: KMeshRequestContextParseResult
    prompt_token_ids: tuple[int, ...]
    prompt_length: int
    lora_adapter_identity: str | None
    cache_salt: str | None
    generation: int
    lookup_descriptor: KMeshRequestLookupDescriptor


class _KMeshObserverController:
    def __init__(
        self,
        config: KMeshEngineConfig,
        runtime_identity: KMeshRuntimeIdentity,
        block_size: int,
        target_profile: str | None = None,
    ) -> None:
        self._config = config
        self._runtime_identity = runtime_identity
        self._planner = KMeshPlanner(
            block_size=block_size, max_group_size=config.max_group_size
        )
        self._admissions: dict[str, _KMeshAdmission] = {}
        self._scheduled_admission_generations: dict[str, int] = {}
        self._admission_generation = 0
        self._step_generation = 0
        self._target_profile = target_profile
        self._lookup_intent = (
            KMeshRequestLookupIntent.SERIAL_PIC
            if config.mode in _ACTIONABLE_PIC_MODES
            else KMeshRequestLookupIntent.OBSERVE_ONLY
        )

    def admit(self, request: Request) -> None:
        if self._config.mode in _ACTIONABLE_PIC_MODES and (
            violation := _serial_pic_v1_request_violation(request)
        ):
            raise ValueError(
                f"KMesh target profile {self._target_profile!r} requires " + violation
            )
        self._admission_generation += 1
        generation = self._admission_generation
        self._admissions.pop(request.request_id, None)
        self._scheduled_admission_generations.pop(request.request_id, None)
        prompt_token_ids = tuple(request.prompt_token_ids or ())
        sampling_params = request.sampling_params
        extra_args = sampling_params.extra_args if sampling_params is not None else None
        result = parse_kmesh_extra_args(extra_args, prompt_token_ids)
        if result.error is not None and (
            self._config.strict_validation or not self._config.fail_open
        ):
            result.raise_for_error()
        lookup_descriptor = KMeshRequestLookupDescriptor.from_admission(
            request_id=request.request_id,
            admission_generation=generation,
            prompt_token_ids=prompt_token_ids,
            prompt_token_count=request.num_prompt_tokens,
            context_result=result,
            lookup_intent=self._lookup_intent,
            has_complete_runtime_identity=self._runtime_identity.is_complete,
        )
        self._admissions[request.request_id] = _KMeshAdmission(
            request=request,
            result=result,
            prompt_token_ids=prompt_token_ids,
            prompt_length=request.num_prompt_tokens,
            lora_adapter_identity=_lora_identity(request.lora_request),
            cache_salt=_optional_identity("cache-salt", request.cache_salt),
            generation=generation,
            lookup_descriptor=lookup_descriptor,
        )

    def lookup_descriptor(
        self, request: Request
    ) -> KMeshRequestLookupDescriptor | None:
        """Return frozen admission evidence only while the request still matches."""
        descriptor = self._request_descriptor(request)
        if descriptor is None or not descriptor.is_admitted:
            return None
        return descriptor

    def no_shared_segments_descriptor(
        self, request: Request
    ) -> KMeshRequestLookupDescriptor | None:
        """Return only exact, current negative evidence for zero sharing."""
        descriptor = self._request_descriptor(request)
        if (
            descriptor is None
            or descriptor.admission_fallback
            is not KMeshFallbackReason.NO_SHARED_SEGMENTS
        ):
            return None
        return descriptor

    def _request_descriptor(
        self, request: Request
    ) -> KMeshRequestLookupDescriptor | None:
        """Return current evidence only for the identical admitted request."""
        descriptor = self._current_descriptor(request.request_id)
        admission = self._admissions.get(request.request_id)
        if descriptor is None or admission is None or admission.request is not request:
            return None
        return descriptor

    def _current_descriptor(
        self, request_id: str
    ) -> KMeshRequestLookupDescriptor | None:
        admission = self._admissions.get(request_id)
        if admission is None:
            return None
        request = admission.request
        descriptor = admission.lookup_descriptor
        if (
            request_id != descriptor.request_id
            or request.request_id != request_id
            or admission.generation != descriptor.admission_generation
        ):
            return None
        if (
            self._config.mode in _ACTIONABLE_PIC_MODES
            and _serial_pic_v1_request_violation(request) is not None
        ):
            return None
        if request.num_prompt_tokens != admission.prompt_length:
            return None
        if tuple(request.prompt_token_ids or ()) != admission.prompt_token_ids:
            return None
        sampling_params = request.sampling_params
        extra_args = sampling_params.extra_args if sampling_params is not None else None
        current_result = parse_kmesh_extra_args(extra_args, admission.prompt_token_ids)
        if current_result != admission.result:
            return None
        return descriptor

    def _observation(
        self,
        request_id: str,
        *,
        num_computed_tokens: int,
        num_scheduled_tokens: int,
    ) -> KMeshRequestObservation:
        admission = self._admissions.get(request_id)
        if admission is None:
            return KMeshRequestObservation(
                request_id,
                KMeshRequestContextParseResult.no_metadata(),
                None,
            )
        context = admission.result.context
        key = None
        if admission.result.is_enabled and context is not None:
            key = self._runtime_identity.make_compatibility_key(
                context,
                lora_adapter_identity=admission.lora_adapter_identity,
                cache_salt=admission.cache_salt,
                active_prompt_length=admission.prompt_length,
                num_computed_tokens=num_computed_tokens,
                num_scheduled_tokens=num_scheduled_tokens,
            )
            if key is not None and (self._target_profile in KMESH_COLLECTIVE_PROFILES):
                if (
                    num_computed_tokens + num_scheduled_tokens
                    != admission.prompt_length
                ):
                    key = None
                else:
                    # Collective execution owns the full prompt view. Request
                    # descriptors and peer metadata retain each sparse span
                    # and lookup end; neither belongs in the group identity.
                    key = replace(
                        key,
                        cached_span=(0, 0),
                        prefill_chunk_span=(0, admission.prompt_length),
                    )
        return KMeshRequestObservation(
            request_id,
            admission.result,
            key,
            prompt_token_ids=admission.prompt_token_ids,
        )

    def build_step(
        self, scheduler_output: SchedulerOutput
    ) -> tuple[KMeshPlanBatch, tuple[KMeshRequestLookupDescriptor, ...], int]:
        for request_id in scheduler_output.finished_req_ids:
            self._admissions.pop(request_id, None)
            self._scheduled_admission_generations.pop(request_id, None)

        scheduled: list[tuple[str, int, int, bool]] = []
        for request in scheduler_output.scheduled_new_reqs:
            admission = self._admissions.get(request.req_id)
            scheduled_tokens = (
                tuple(request.prompt_token_ids)
                if request.prompt_token_ids is not None
                else None
            )
            sampling_params = request.sampling_params
            extra_args = (
                sampling_params.extra_args if sampling_params is not None else None
            )
            scheduled_result = parse_kmesh_extra_args(
                extra_args, scheduled_tokens or ()
            )
            matches_admission = (
                admission is not None
                and scheduled_tokens == admission.prompt_token_ids
                and scheduled_result == admission.result
                and _lora_identity(request.lora_request)
                == admission.lora_adapter_identity
                and (
                    self._config.mode not in _ACTIONABLE_PIC_MODES
                    or (
                        _serial_pic_v1_scheduled_request_violation(
                            request, admission.prompt_token_ids
                        )
                        is None
                        and not scheduler_output.has_structured_output_requests
                        and request.req_id
                        not in scheduler_output.scheduled_spec_decode_tokens
                        and not scheduler_output.scheduled_encoder_inputs.get(
                            request.req_id
                        )
                    )
                )
            )
            scheduled.append(
                (
                    request.req_id,
                    request.num_computed_tokens,
                    scheduler_output.num_scheduled_tokens[request.req_id],
                    matches_admission,
                )
            )
        cached = scheduler_output.scheduled_cached_reqs
        if not isinstance(cached, list):
            for index, request_id in enumerate(cached.req_ids):
                if not cached.is_context_phase(request_id):
                    continue
                admission = self._admissions.get(request_id)
                num_computed = cached.num_computed_tokens[index]
                num_scheduled = scheduler_output.num_scheduled_tokens[request_id]
                matches_admission = (
                    admission is not None
                    and self._scheduled_admission_generations.get(request_id)
                    == admission.generation
                )
                if matches_admission and request_id in cached.all_token_ids:
                    assert admission is not None
                    matches_admission = (
                        tuple(cached.all_token_ids[request_id])
                        == admission.prompt_token_ids
                    )
                # A synchronous KV-load failure keeps the vLLM request live,
                # rolls its computed-token count back, and schedules the same
                # context locally without another connector lookup/allocation
                # round. The prior actionable descriptor belongs only to the
                # failed load attempt. Suppress it for this exact non-resumed
                # full-prompt recompute so peers agree that the retry is local.
                local_full_recompute = (
                    self._config.mode in _ACTIONABLE_PIC_MODES
                    and matches_admission
                    and request_id not in cached.resumed_req_ids
                    and cached.num_output_tokens[index] == 0
                    and admission is not None
                    and num_computed < admission.prompt_length
                    and num_computed + num_scheduled == admission.prompt_length
                )
                if local_full_recompute:
                    matches_admission = False
                scheduled.append(
                    (
                        request_id,
                        num_computed,
                        num_scheduled,
                        matches_admission,
                    )
                )

        observations: list[KMeshRequestObservation] = []
        descriptors: list[KMeshRequestLookupDescriptor] = []
        for request_id, num_computed, num_scheduled, matches_admission in scheduled:
            descriptor = self._current_descriptor(request_id)
            if not matches_admission:
                descriptor = None
            if descriptor is not None:
                descriptors.append(descriptor)
                self._scheduled_admission_generations[request_id] = (
                    descriptor.admission_generation
                )
                observations.append(
                    self._observation(
                        request_id,
                        num_computed_tokens=num_computed,
                        num_scheduled_tokens=num_scheduled,
                    )
                )
            else:
                observations.append(
                    KMeshRequestObservation(
                        request_id,
                        KMeshRequestContextParseResult.no_metadata(),
                        None,
                    )
                )

        plan_batch = self._planner.observe(observations)
        self._step_generation += 1
        return (
            plan_batch,
            tuple(sorted(descriptors, key=lambda item: item.request_id)),
            self._step_generation,
        )

    def discard(self, request_id: str, request: Request | None = None) -> None:
        admission = self._admissions.get(request_id)
        if admission is None or (
            request is not None and admission.request is not request
        ):
            return
        self._admissions.pop(request_id)
        self._scheduled_admission_generations.pop(request_id, None)


@dataclass(frozen=True, slots=True)
class KMeshConnectorMetadata(KVConnectorMetadata):
    """Self-contained KMesh step evidence transported to the worker."""

    plan_batch: KMeshPlanBatch | None = None
    scheduled_descriptors: tuple[KMeshRequestLookupDescriptor, ...] = ()
    step_generation: int = 0

    def __post_init__(self) -> None:
        if type(self.scheduled_descriptors) is not tuple:
            raise TypeError("scheduled_descriptors must be an immutable tuple")
        if type(self.step_generation) is not int or self.step_generation < 0:
            raise ValueError("step_generation must be a nonnegative integer")
        if self.plan_batch is None:
            if self.scheduled_descriptors or self.step_generation != 0:
                raise ValueError(
                    "metadata without a plan batch must have empty step evidence"
                )
            return
        if not isinstance(self.plan_batch, KMeshPlanBatch):
            raise TypeError("plan_batch must be KMeshPlanBatch or null")
        if self.step_generation == 0:
            raise ValueError("step_generation must be a positive integer")
        descriptor_ids: list[str] = []
        descriptors_by_id: dict[str, KMeshRequestLookupDescriptor] = {}
        for index, descriptor in enumerate(self.scheduled_descriptors):
            if not isinstance(descriptor, KMeshRequestLookupDescriptor):
                raise TypeError(
                    "scheduled_descriptors"
                    f"[{index}] must be KMeshRequestLookupDescriptor"
                )
            descriptor_ids.append(descriptor.request_id)
            descriptors_by_id[descriptor.request_id] = descriptor
        if tuple(descriptor_ids) != tuple(sorted(set(descriptor_ids))):
            raise ValueError("scheduled descriptor IDs must be unique and sorted")

        outcome_ids = {
            outcome.request_id for outcome in self.plan_batch.request_outcomes
        }
        if not set(descriptor_ids) <= outcome_ids:
            raise ValueError("scheduled descriptors must belong to this plan batch")
        for plan in self.plan_batch.plans:
            key = plan.group.compatibility_key
            for mapping in plan.request_segment_maps:
                planned_descriptor = descriptors_by_id.get(mapping.request_id)
                if planned_descriptor is None or not planned_descriptor.is_admitted:
                    raise ValueError(
                        "every planned request requires an admitted descriptor"
                    )
                if (
                    planned_descriptor.cache_domain != key.cache_domain
                    or planned_descriptor.prompt_token_count != key.active_prompt_length
                ):
                    raise ValueError(
                        "planned request compatibility drifted from its descriptor"
                    )
                descriptor_spans = {
                    span.shared_segment: (
                        span.token_start,
                        span.token_end,
                        span.logical_order,
                    )
                    for span in planned_descriptor.spans
                }
                for entry in mapping.entries:
                    if descriptor_spans.get(entry.shared_segment) != (
                        entry.token_start,
                        entry.token_end,
                        entry.logical_order,
                    ):
                        raise ValueError(
                            "planned segment map drifted from its descriptor"
                        )

    def to_primitive(self) -> dict[str, Any]:
        return {
            "plan_batch": (
                self.plan_batch.to_primitive() if self.plan_batch is not None else None
            ),
            "scheduled_descriptors": [
                descriptor.to_sidecar_payload()
                for descriptor in self.scheduled_descriptors
            ],
            "step_generation": self.step_generation,
        }


class KMeshConnector(KVConnectorBase_V1):
    """No-op KV connector that hosts KMesh's planning control plane.

    Observe mode parses, groups, records, and transports prospective plans but
    never loads, saves, aliases, or mutates KV. The named serial and collective
    profiles export actionable request evidence and execution-ready group plans
    to their LMCache peer; KMesh itself still performs no KV mutation. All other
    recovery modes remain rejected.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        self._kmesh_config = KMeshEngineConfig.from_connector_extra_config(
            self._kv_transfer_config.kv_connector_extra_config
        )
        target_profile = self._kv_transfer_config.kv_connector_extra_config.get(
            KMESH_TARGET_PROFILE_KEY
        )
        supported_profiles = (
            KMESH_SERIAL_PIC_V1_PROFILE,
            *KMESH_COLLECTIVE_PROFILES,
        )
        if target_profile is not None and target_profile not in supported_profiles:
            raise ValueError(f"unsupported KMesh target profile: {target_profile!r}")
        internally_qualified = isinstance(
            target_profile, str
        ) and _is_kmesh_target_profile_child_qualified(vllm_config, target_profile)
        expected_mode = (
            _target_profile_mode(target_profile)
            if isinstance(target_profile, str)
            else None
        )
        target_profile_enabled = (
            expected_mode is not None
            and internally_qualified
            and self._kmesh_config.mode is expected_mode
            and self._kmesh_config.pic_backend == "lmcache"
        )
        if target_profile is not None and not internally_qualified:
            raise ValueError(
                f"KMesh target profile {target_profile!r} requires an internally "
                "qualified outer MultiConnector"
            )
        if target_profile is not None and not target_profile_enabled:
            assert expected_mode is not None
            raise ValueError(
                f"KMesh target profile {target_profile!r} requires enabled "
                f"mode={expected_mode.value!r} with pic_backend='lmcache'"
            )
        if target_profile_enabled:
            assert isinstance(target_profile, str)
            _validate_target_profile_kmesh_config(self._kmesh_config, target_profile)
        if self._kmesh_config.mode not in (KMeshMode.OFF, KMeshMode.OBSERVE) and (
            not target_profile_enabled
        ):
            raise ValueError(
                f"KMesh mode {self._kmesh_config.mode.value!r} has no qualified "
                "runtime backend"
            )
        self._target_profile = target_profile if target_profile_enabled else None
        self._target_profile_armed = False
        self._observer: _KMeshObserverController | None = None
        self._observer_stats = None
        if (
            self._kmesh_config.mode is KMeshMode.OBSERVE
            and role is KVConnectorRole.SCHEDULER
        ):
            self._observer = _KMeshObserverController(
                self._kmesh_config,
                KMeshRuntimeIdentity.from_vllm_config(vllm_config, kv_cache_config),
                vllm_config.cache_config.block_size,
            )
            self._observer_stats = KMeshConnectorStats()
        self._worker_plan_batch: KMeshPlanBatch | None = None
        self._worker_scheduled_descriptors: tuple[
            KMeshRequestLookupDescriptor, ...
        ] = ()
        self._worker_step_generation = 0
        self._last_worker_step_generation = 0
        self._pending_worker_step_generation = 0
        self._metadata_transaction_active = False
        self._metadata_transaction_committed = False
        self._metadata_transaction_last_generation: int | None = None

    def _arm_target_profile(self, target_profile: str) -> None:
        """Enable actionable metadata only after composite construction."""

        if self._target_profile != target_profile:
            raise ValueError(f"cannot arm {target_profile} outside its target profile")
        validated_profile = _validate_kmesh_target_profile_child_runtime(
            self._vllm_config
        )
        if validated_profile != target_profile:
            raise ValueError("KMesh child target profile drifted before arming")
        if self._target_profile_armed:
            return
        observer: _KMeshObserverController | None = None
        observer_stats: KMeshConnectorStats | None = None
        if self._role is KVConnectorRole.SCHEDULER:
            observer = _KMeshObserverController(
                self._kmesh_config,
                KMeshRuntimeIdentity.from_vllm_config(
                    self._vllm_config, self._kv_cache_config
                ),
                self._vllm_config.cache_config.block_size,
                target_profile,
            )
            observer_stats = KMeshConnectorStats()
        self._observer = observer
        self._observer_stats = observer_stats
        self._target_profile_armed = True

    def arm_serial_pic_v1(self) -> None:
        """Arm the frozen request-local serial profile."""

        self._arm_target_profile(KMESH_SERIAL_PIC_V1_PROFILE)

    def arm_collective_lockstep_v1(self) -> None:
        """Arm the frozen collective-lockstep sibling profile."""

        self._arm_target_profile(KMESH_COLLECTIVE_LOCKSTEP_V1_PROFILE)

    def arm_collective_batched_position_recovery_v1(self) -> None:
        """Arm the batched-position-recovery collective sibling profile."""

        self._arm_target_profile(KMESH_COLLECTIVE_BATCHED_POSITION_RECOVERY_V1_PROFILE)

    def arm_collective_batched_qkv_v1(self) -> None:
        """Arm the batched-QKV collective sibling profile."""

        self._arm_target_profile(KMESH_COLLECTIVE_BATCHED_QKV_V1_PROFILE)

    def _arm_serial_pic_v1(self) -> None:
        """Compatibility alias for the public composite arming hook."""
        self.arm_serial_pic_v1()

    def _require_target_profile_armed(self) -> None:
        if self._target_profile is None:
            return
        if not self._target_profile_armed:
            raise ValueError(
                f"KMesh {self._target_profile} is not armed by a fully constructed "
                "outer MultiConnector"
            )
        profile = _validate_kmesh_target_profile_child_runtime(self._vllm_config)
        if profile != self._target_profile:
            raise ValueError("KMesh child target profile drifted after arming")

    def validate_runtime_profile(self, parent_config: VllmConfig) -> None:
        """Revalidate the frozen KMesh child without changing its state."""
        if self._target_profile is None:
            return
        profile = _validate_kmesh_target_profile_child_runtime(self._vllm_config)
        if profile != self._target_profile:
            raise ValueError("KMesh child target profile drifted at runtime")

    @property
    def requires_kv_delivery(self) -> bool:
        return False

    def validate_connector_metadata(
        self, connector_metadata: KVConnectorMetadata
    ) -> None:
        """Preflight KMesh step evidence without consuming its generation."""
        self._require_target_profile_armed()
        if not isinstance(connector_metadata, KMeshConnectorMetadata):
            raise TypeError("KMeshConnector requires KMeshConnectorMetadata")
        if self._kmesh_config.mode is KMeshMode.OFF:
            if connector_metadata.plan_batch is not None:
                raise ValueError("KMesh off mode rejects active step metadata")
        else:
            if connector_metadata.plan_batch is None:
                raise ValueError("enabled KMesh mode requires active step metadata")
            if connector_metadata.step_generation <= self._last_worker_step_generation:
                raise ValueError("KMesh step metadata is stale or was already consumed")
            expected_intent = (
                KMeshRequestLookupIntent.SERIAL_PIC
                if self._kmesh_config.mode in _ACTIONABLE_PIC_MODES
                else KMeshRequestLookupIntent.OBSERVE_ONLY
            )
            if any(
                descriptor.lookup_intent is not expected_intent
                for descriptor in connector_metadata.scheduled_descriptors
            ):
                if self._kmesh_config.mode is KMeshMode.OBSERVE:
                    raise ValueError(
                        "KMesh observe mode rejects actionable descriptors"
                    )
                raise ValueError(
                    f"KMesh {self._kmesh_config.mode.value} mode requires "
                    f"{expected_intent.value} descriptors"
                )

    def begin_connector_metadata_transaction(self) -> None:
        """Checkpoint replay state until the composite finalizes."""
        if self._metadata_transaction_active:
            raise RuntimeError("KMesh metadata transaction is already active")
        self._metadata_transaction_active = True
        self._metadata_transaction_committed = False
        self._metadata_transaction_last_generation = self._last_worker_step_generation
        self._pending_worker_step_generation = 0

    def commit_connector_metadata_transaction(self) -> None:
        """Tentatively consume the generation while retaining rollback."""
        if not self._metadata_transaction_active:
            return
        if self._metadata_transaction_committed:
            return
        self._last_worker_step_generation = max(
            self._last_worker_step_generation,
            self._pending_worker_step_generation,
        )
        self._metadata_transaction_committed = True

    def finalize_connector_metadata_transaction(self) -> None:
        """End the transaction while retaining cross-child rollback."""
        if not self._metadata_transaction_active:
            return
        self._pending_worker_step_generation = 0
        self._metadata_transaction_active = False
        self._metadata_transaction_committed = False

    def abort_connector_metadata_transaction(self) -> None:
        """Restore replay state even after commit and finalization."""
        if self._metadata_transaction_last_generation is not None:
            self._last_worker_step_generation = (
                self._metadata_transaction_last_generation
            )
        self._pending_worker_step_generation = 0
        self._metadata_transaction_active = False
        self._metadata_transaction_committed = False
        self._metadata_transaction_last_generation = None

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        self.validate_connector_metadata(connector_metadata)
        assert isinstance(connector_metadata, KMeshConnectorMetadata)
        super().bind_connector_metadata(connector_metadata)
        self._worker_plan_batch = connector_metadata.plan_batch
        self._worker_scheduled_descriptors = connector_metadata.scheduled_descriptors
        self._worker_step_generation = connector_metadata.step_generation
        if self._metadata_transaction_active:
            self._pending_worker_step_generation = connector_metadata.step_generation
        else:
            self._last_worker_step_generation = max(
                self._last_worker_step_generation, connector_metadata.step_generation
            )

    def clear_connector_metadata(self) -> None:
        self._worker_plan_batch = None
        self._worker_scheduled_descriptors = ()
        self._worker_step_generation = 0
        self._pending_worker_step_generation = 0
        super().clear_connector_metadata()

    def get_kmesh_plan_batch(self) -> KMeshPlanBatch | None:
        return self._worker_plan_batch

    def get_kmesh_scheduled_descriptors(
        self,
    ) -> tuple[KMeshRequestLookupDescriptor, ...]:
        return self._worker_scheduled_descriptors

    def get_kmesh_step_generation(self) -> int:
        return self._worker_step_generation

    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        return None

    def wait_for_layer_load(self, layer_name: str) -> None:
        return None

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        return None

    def wait_for_save(self) -> None:
        return None

    def on_new_request(self, request: Request) -> None:
        self._require_target_profile_armed()
        if self._observer is not None:
            self._observer.admit(request)

    def export_request_sidecar(
        self, request: Request
    ) -> KVConnectorRequestSidecar | None:
        """Export candidate or exact no-sharing negative evidence to a peer."""
        self._require_target_profile_armed()
        if self._observer is None:
            return None
        descriptor = self._observer.lookup_descriptor(request)
        if descriptor is None and self._target_profile in KMESH_COLLECTIVE_PROFILES:
            descriptor = self._observer.no_shared_segments_descriptor(request)
        if descriptor is None:
            return None
        return KVConnectorRequestSidecar(
            provider_id=KMESH_REQUEST_SIDECAR_PROVIDER_ID,
            payload=descriptor.to_sidecar_payload(),
        )

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        return 0, False

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ) -> None:
        return None

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        self._require_target_profile_armed()
        if self._observer is None:
            return KMeshConnectorMetadata()
        plan_batch, descriptors, step_generation = self._observer.build_step(
            scheduler_output
        )
        assert self._observer_stats is not None
        self._observer_stats.observe(plan_batch.metrics)
        if plan_batch.metrics.observed_requests:
            logger.info(
                "KMesh observer metrics: %s",
                json.dumps(plan_batch.metrics.to_primitive(), sort_keys=True),
            )
        return KMeshConnectorMetadata(plan_batch, descriptors, step_generation)

    def request_finished(
        self, request: Request, block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        if self._observer is not None:
            self._observer.discard(request.request_id, request)
        return False, None

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        if self._observer_stats is None or self._observer_stats.is_empty():
            return None
        return self._observer_stats.clone_and_reset()

    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> KVConnectorStats | None:
        return KMeshConnectorStats(data=data or {})

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> KVConnectorPromMetrics:
        return KMeshPromMetrics(
            vllm_config, metric_types, labelnames, per_engine_labelvalues
        )
