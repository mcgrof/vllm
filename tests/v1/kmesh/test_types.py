# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import FrozenInstanceError, fields, is_dataclass, replace
from multiprocessing.reduction import ForkingPickler

import msgspec
import pytest
import torch

from vllm.v1.kmesh import (
    KMESH_SCHEMA_VERSION,
    KMeshFallbackReason,
    KMeshGroupKind,
    KMeshMetadataError,
    KMeshMetadataErrorCode,
    KMeshPICExecutionSpec,
    KMeshPICOperationCounts,
    KMeshPICResult,
    KMeshPICSpan,
    KMeshPICTolerance,
    KMeshRequestContext,
    KMeshRequestContextParseResult,
    KMeshRequestLookupDescriptor,
    KMeshRequestLookupIntent,
    KMeshSegment,
    KMeshSegmentKind,
    KMeshStateKind,
    KMeshValidationError,
    PositionIndependentReuseBackend,
    kmesh_token_slice_digest,
)

pytestmark = pytest.mark.skip_global_cleanup


def make_context(*, enabled: bool = True) -> KMeshRequestContext:
    return KMeshRequestContext(
        schema_version=KMESH_SCHEMA_VERSION,
        enabled=enabled,
        execution_id="exec-1",
        group_kind=KMeshGroupKind.AGENT_ROUND,
        group_id="gather-1",
        round_id=3,
        branch_id="agent-0",
        parent_branch_id=None,
        state_kind=KMeshStateKind.KV_CACHE,
        cache_domain="tenant-a",
        segments=(
            KMeshSegment(
                segment_id="private-0",
                kind=KMeshSegmentKind.PRIVATE_HISTORY,
                token_start=0,
                token_end=2,
                logical_order=0,
                source_id="agent-0",
                shareable=False,
                lineage_id="lineage-0",
            ),
            KMeshSegment(
                segment_id="shared-0",
                kind=KMeshSegmentKind.SHARED_UPDATE,
                token_start=2,
                token_end=5,
                logical_order=1,
                source_id="update-0",
                shareable=True,
                lineage_id="lineage-0",
            ),
        ),
    )


def make_lookup_descriptor() -> KMeshRequestLookupDescriptor:
    return KMeshRequestLookupDescriptor.from_context(
        request_id="request-a",
        admission_generation=3,
        prompt_token_ids=(10, 11, 20, 21, 22),
        prompt_token_count=5,
        context=make_context(),
        lookup_intent=KMeshRequestLookupIntent.OBSERVE_ONLY,
    )


def make_pic_execution_spec() -> KMeshPICExecutionSpec:
    descriptor = make_lookup_descriptor()
    return KMeshPICExecutionSpec(
        plan_id="plan-a",
        request_id=descriptor.request_id,
        admission_generation=descriptor.admission_generation,
        step_generation=7,
        selected_spans=descriptor.spans,
        cached_span=KMeshPICSpan(0, 5),
        check_layer=1,
        recompute_ratio=0.25,
        block_size=2,
        declared_tolerance=KMeshPICTolerance(atol=1e-5, rtol=1e-4),
    )


def make_pic_result() -> KMeshPICResult:
    return KMeshPICResult(
        execution_spec=make_pic_execution_spec(),
        important_positions=(2, 4),
        changed_request_blocks=(1, 2),
        cache_hit_spans=(KMeshPICSpan(2, 5),),
        operation_counts=KMeshPICOperationCounts(
            serial_rope_calls=1,
            serial_diff_passes=1,
        ),
    )


def assert_sidecar_value(value: object) -> None:
    assert value is None or type(value) in (str, int, bool, tuple)
    if type(value) is tuple:
        for item in value:
            assert_sidecar_value(item)


def test_context_is_immutable_and_pickle_round_trips():
    context = make_context()

    assert ForkingPickler.loads(ForkingPickler.dumps(context)) == context
    with pytest.raises(FrozenInstanceError):
        context.group_id = "other"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        context.segments[0].token_end = 4  # type: ignore[misc]


def test_context_primitive_shape_is_msgpack_friendly():
    primitive = make_context().to_primitive()

    assert msgspec.msgpack.decode(msgspec.msgpack.encode(primitive)) == primitive
    assert primitive["group_kind"] == "agent_round"
    assert primitive["segments"][1]["kind"] == "shared_update"
    assert isinstance(primitive["segments"], list)


def test_direct_types_reject_empty_segments():
    with pytest.raises(KMeshValidationError) as exc_info:
        KMeshSegment(
            segment_id="empty",
            kind=KMeshSegmentKind.OTHER,
            token_start=2,
            token_end=2,
            logical_order=0,
            source_id=None,
            shareable=False,
            lineage_id=None,
        )

    assert exc_info.value.error.code is KMeshMetadataErrorCode.EMPTY_SEGMENT


def test_direct_context_rejects_overlapping_ranges():
    first, second = make_context().segments
    overlapping = KMeshSegment(
        segment_id=second.segment_id,
        kind=second.kind,
        token_start=1,
        token_end=second.token_end,
        logical_order=second.logical_order,
        source_id=second.source_id,
        shareable=second.shareable,
        lineage_id=second.lineage_id,
    )

    with pytest.raises(KMeshValidationError) as exc_info:
        KMeshRequestContext(
            **{
                **make_context().to_primitive(),
                "group_kind": KMeshGroupKind.AGENT_ROUND,
                "state_kind": KMeshStateKind.KV_CACHE,
                "segments": (first, overlapping),
            }
        )

    assert exc_info.value.error.code is KMeshMetadataErrorCode.OVERLAPPING_SEGMENTS


def test_identifiers_reject_unicode_lone_surrogates_before_transport():
    with pytest.raises(KMeshValidationError) as exc_info:
        replace(make_context(), execution_id=chr(0xD800))

    assert exc_info.value.error.code is KMeshMetadataErrorCode.INVALID_IDENTIFIER


def test_parse_result_has_stable_fallback_and_error_shape():
    error = KMeshMetadataError(
        code=KMeshMetadataErrorCode.UNKNOWN_ENUM_VALUE,
        path="group_kind",
        message="unknown KMeshGroupKind value",
    )
    result = KMeshRequestContextParseResult.invalid(error)

    assert not result.is_enabled
    assert result.fallback_reason is KMeshFallbackReason.INVALID_METADATA
    assert msgspec.msgpack.decode(msgspec.msgpack.encode(result.to_primitive())) == {
        "context": None,
        "fallback_reason": "invalid_metadata",
        "error": {
            "code": "unknown_enum_value",
            "path": "group_kind",
            "message": "unknown KMeshGroupKind value",
        },
    }
    with pytest.raises(KMeshValidationError) as exc_info:
        result.raise_for_error()
    assert exc_info.value.error == error


def test_disabled_context_is_valid_but_not_enabled():
    context = make_context(enabled=False)
    result = KMeshRequestContextParseResult.disabled(context)

    assert result.context is context
    assert result.fallback_reason is KMeshFallbackReason.DISABLED
    assert not result.is_enabled


def test_lookup_descriptor_is_immutable_tensor_free_and_forking_pickles():
    descriptor = make_lookup_descriptor()
    payload = descriptor.to_sidecar_payload()

    assert descriptor.lookup_intent is KMeshRequestLookupIntent.OBSERVE_ONLY
    assert descriptor.is_admitted
    assert not descriptor.permits_peer_lookup
    assert descriptor.spans[0].shared_segment.content_digest == (
        kmesh_token_slice_digest((20, 21, 22))
    )
    assert_sidecar_value(payload)
    assert KMeshRequestLookupDescriptor.from_sidecar_payload(payload) == descriptor
    assert ForkingPickler.loads(ForkingPickler.dumps(descriptor)) == descriptor
    assert ForkingPickler.loads(ForkingPickler.dumps(payload)) == payload

    with pytest.raises(FrozenInstanceError):
        descriptor.cache_domain = "tenant-b"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        descriptor.spans[0].token_end = 6  # type: ignore[misc]


def test_lookup_descriptor_intent_is_strict_and_serial_only_is_actionable():
    descriptor = make_lookup_descriptor()
    serial = replace(descriptor, lookup_intent=KMeshRequestLookupIntent.SERIAL_PIC)
    assert serial.permits_peer_lookup

    payload = descriptor.to_sidecar_payload()
    assert type(payload) is tuple
    unknown_intent = (payload[0], "load_anyway", *payload[2:])
    with pytest.raises(ValueError, match="lookup_intent has an unknown value"):
        KMeshRequestLookupDescriptor.from_sidecar_payload(unknown_intent)


def test_lookup_descriptor_wire_parser_rejects_nonprimitive_and_loose_shapes():
    payload = make_lookup_descriptor().to_sidecar_payload()
    assert type(payload) is tuple
    raw_spans = payload[7]
    assert type(raw_spans) is tuple
    raw_span = raw_spans[0]
    assert type(raw_span) is tuple

    tensor_span = (*raw_span[:7], torch.tensor(2), *raw_span[8:])
    tensor_payload = (*payload[:7], (tensor_span,))
    with pytest.raises(TypeError, match="token_start must be an integer"):
        KMeshRequestLookupDescriptor.from_sidecar_payload(tensor_payload)
    with pytest.raises(TypeError, match="descriptor must be a tuple"):
        KMeshRequestLookupDescriptor.from_sidecar_payload(list(payload))
    with pytest.raises(ValueError, match="exactly 8 items"):
        KMeshRequestLookupDescriptor.from_sidecar_payload(payload[:-1])


def test_lookup_descriptor_rejects_eligible_spans_with_fallback():
    descriptor = make_lookup_descriptor()
    payload = descriptor.to_sidecar_payload()
    assert type(payload) is tuple
    contradictory = (*payload[:6], KMeshFallbackReason.DISABLED.value, payload[7])

    with pytest.raises(ValueError, match="fallback descriptor cannot carry"):
        KMeshRequestLookupDescriptor.from_sidecar_payload(contradictory)

    fallback = KMeshRequestLookupDescriptor(
        lookup_intent=KMeshRequestLookupIntent.OBSERVE_ONLY,
        request_id="request-a",
        admission_generation=4,
        prompt_token_count=5,
        cache_domain=None,
        spans=(),
        admission_fallback=KMeshFallbackReason.INVALID_METADATA,
    )
    assert not fallback.is_admitted
    assert (
        KMeshRequestLookupDescriptor.from_sidecar_payload(fallback.to_sidecar_payload())
        == fallback
    )


def test_lookup_descriptor_requires_domain_spans_and_exact_prompt_evidence():
    descriptor = make_lookup_descriptor()
    with pytest.raises(ValueError, match="requires a cache domain"):
        replace(descriptor, cache_domain=None)
    with pytest.raises(ValueError, match="requires shareable spans"):
        replace(descriptor, spans=())
    with pytest.raises(ValueError, match="does not match the admitted count"):
        KMeshRequestLookupDescriptor.from_context(
            request_id="request-a",
            admission_generation=3,
            prompt_token_ids=(10, 11, 20, 21, 22),
            prompt_token_count=4,
            context=make_context(),
            lookup_intent=KMeshRequestLookupIntent.OBSERVE_ONLY,
        )


def test_lookup_admission_factory_freezes_stable_fallback_descriptors():
    error = KMeshMetadataError(
        code=KMeshMetadataErrorCode.INVALID_RANGE,
        path="segments[0].token_end",
        message="out of bounds",
    )
    private_context = replace(
        make_context(),
        segments=tuple(
            replace(segment, shareable=False) for segment in make_context().segments
        ),
    )
    cases = (
        (
            KMeshRequestContextParseResult.no_metadata(),
            True,
            5,
            KMeshFallbackReason.NO_METADATA,
        ),
        (
            KMeshRequestContextParseResult.invalid(error),
            True,
            5,
            KMeshFallbackReason.INVALID_METADATA,
        ),
        (
            KMeshRequestContextParseResult.disabled(make_context(enabled=False)),
            True,
            5,
            KMeshFallbackReason.DISABLED,
        ),
        (
            KMeshRequestContextParseResult.accepted(
                replace(make_context(), group_kind=KMeshGroupKind.COT_BRANCH)
            ),
            True,
            5,
            KMeshFallbackReason.GROUP_KIND_UNSUPPORTED,
        ),
        (
            KMeshRequestContextParseResult.accepted(
                replace(make_context(), state_kind=KMeshStateKind.KV_DIFF)
            ),
            True,
            5,
            KMeshFallbackReason.STATE_KIND_UNSUPPORTED,
        ),
        (
            KMeshRequestContextParseResult.accepted(make_context()),
            False,
            5,
            KMeshFallbackReason.MISSING_COMPATIBILITY_EVIDENCE,
        ),
        (
            KMeshRequestContextParseResult.accepted(
                replace(make_context(), cache_domain=None)
            ),
            True,
            5,
            KMeshFallbackReason.MISSING_COMPATIBILITY_EVIDENCE,
        ),
        (
            KMeshRequestContextParseResult.accepted(private_context),
            True,
            5,
            KMeshFallbackReason.NO_SHARED_SEGMENTS,
        ),
        (
            KMeshRequestContextParseResult.accepted(make_context()),
            True,
            4,
            KMeshFallbackReason.PROMPT_TOKEN_LENGTH_MISMATCH,
        ),
    )

    for generation, (result, complete_identity, prompt_count, reason) in enumerate(
        cases, start=1
    ):
        descriptor = KMeshRequestLookupDescriptor.from_admission(
            request_id="request-a",
            admission_generation=generation,
            prompt_token_ids=(10, 11, 20, 21, 22),
            prompt_token_count=prompt_count,
            context_result=result,
            lookup_intent=KMeshRequestLookupIntent.OBSERVE_ONLY,
            has_complete_runtime_identity=complete_identity,
        )
        assert descriptor.admission_fallback is reason
        assert descriptor.spans == ()
        assert not descriptor.is_admitted
        assert not descriptor.permits_peer_lookup
        assert (
            KMeshRequestLookupDescriptor.from_sidecar_payload(
                descriptor.to_sidecar_payload()
            )
            == descriptor
        )


def test_pic_execution_values_are_frozen_tensor_free_and_picklable():
    spec = make_pic_execution_spec()
    result = make_pic_result()

    assert result.execution_spec == spec
    assert result.fallback_reason is None
    assert result.changed_request_blocks == (1, 2)
    assert result.cache_hit_spans == (KMeshPICSpan(2, 5),)
    assert ForkingPickler.loads(ForkingPickler.dumps(spec)) == spec
    assert ForkingPickler.loads(ForkingPickler.dumps(result)) == result
    msgspec.msgpack.encode(spec.to_primitive())
    msgspec.msgpack.encode(result.to_primitive())

    def assert_tensor_free(value: object) -> None:
        assert not isinstance(value, torch.Tensor)
        if is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                assert_tensor_free(getattr(value, field.name))
        elif isinstance(value, dict):
            for key, item in value.items():
                assert_tensor_free(key)
                assert_tensor_free(item)
        elif isinstance(value, list | tuple):
            for item in value:
                assert_tensor_free(item)

    assert_tensor_free(spec)
    assert_tensor_free(result)
    assert_tensor_free(result.to_primitive())
    with pytest.raises(FrozenInstanceError):
        spec.step_generation = 8  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.important_positions = ()  # type: ignore[misc]


def test_pic_backend_protocol_exposes_only_request_owned_serial_recovery():
    class Backend:
        def recover_request_serial(
            self, request: object, execution_spec: KMeshPICExecutionSpec
        ) -> KMeshPICResult:
            assert request == "request-state"
            assert execution_spec == make_pic_execution_spec()
            return make_pic_result()

    backend = Backend()
    assert isinstance(backend, PositionIndependentReuseBackend)
    assert (
        backend.recover_request_serial("request-state", make_pic_execution_spec())
        == make_pic_result()
    )
    assert not hasattr(backend, "recover_group_reference")
    assert not hasattr(backend, "recover_group_batched")


def test_pic_result_fallback_cannot_claim_recovery_evidence():
    result = make_pic_result()
    fallback = KMeshPICResult(
        execution_spec=result.execution_spec,
        important_positions=(),
        changed_request_blocks=(),
        cache_hit_spans=(),
        operation_counts=result.operation_counts,
        fallback_reason=KMeshFallbackReason.PIC_CACHE_MISS,
    )
    assert fallback.fallback_reason is KMeshFallbackReason.PIC_CACHE_MISS

    with pytest.raises(ValueError, match="cannot claim recovery evidence"):
        replace(result, fallback_reason=KMeshFallbackReason.INTERNAL_ERROR)
    with pytest.raises(ValueError, match="requires cache-hit spans"):
        replace(
            result,
            important_positions=(),
            changed_request_blocks=(),
            cache_hit_spans=(),
        )


def test_pic_execution_values_reject_parameter_and_result_drift():
    spec = make_pic_execution_spec()
    result = make_pic_result()

    with pytest.raises(ValueError, match="step_generation"):
        replace(spec, step_generation=0)
    with pytest.raises(TypeError, match="recompute_ratio"):
        replace(spec, recompute_ratio=True)
    with pytest.raises(TypeError, match="immutable tuple"):
        replace(spec, selected_spans=list(spec.selected_spans))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="fit within cached_span"):
        replace(spec, cached_span=KMeshPICSpan(0, 4))
    with pytest.raises(ValueError, match="prompt-origin"):
        replace(spec, cached_span=KMeshPICSpan(1, 5))
    corrupted = make_pic_execution_spec()
    object.__setattr__(corrupted.selected_spans[0].shared_segment, "shareable", False)
    with pytest.raises(ValueError, match="shareable identities"):
        replace(corrupted)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        KMeshPICTolerance(atol=float("nan"), rtol=0.0)
    with pytest.raises(ValueError, match="nonnegative integer"):
        KMeshPICOperationCounts(serial_rope_calls=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unique and sorted"):
        replace(result, important_positions=(4, 2))
    with pytest.raises(ValueError, match="derived from important_positions"):
        replace(result, changed_request_blocks=(0,))
    with pytest.raises(ValueError, match="selected span boundaries"):
        replace(result, cache_hit_spans=(KMeshPICSpan(2, 4),))
