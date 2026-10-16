# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import deepcopy

import msgspec
import pytest

from vllm.v1.kmesh import (
    KMeshFallbackReason,
    KMeshMetadataErrorCode,
    KMeshSegmentKind,
    KMeshValidationError,
    kmesh_context_to_sideband,
    parse_kmesh_extra_args,
    parse_kmesh_request_context,
    parse_ttsep_segment_spans,
)

pytestmark = pytest.mark.skip_global_cleanup


def sideband_payload() -> dict:
    return {
        "schema_version": 1,
        "enabled": True,
        "execution_id": "exec-1",
        "group_kind": "agent_round",
        "group_id": "gather-1",
        "round_id": 3,
        "branch_id": "agent-0",
        "parent_branch_id": None,
        "state_kind": "kv_cache",
        "cache_domain": "tenant-a",
        "segment_encoding": "sideband",
        "segments": [
            {
                "segment_id": "private-0",
                "kind": "private_history",
                "token_start": 0,
                "token_end": 2,
                "logical_order": 0,
                "source_id": "agent-0",
                "shareable": False,
                "lineage_id": "lineage-0",
            },
            {
                "segment_id": "shared-0",
                "kind": "shared_update",
                "token_start": 2,
                "token_end": 5,
                "logical_order": 1,
                "source_id": "update-0",
                "shareable": True,
                "lineage_id": "lineage-0",
            },
        ],
    }


def ttsep_payload() -> dict:
    payload = sideband_payload()
    payload["segment_encoding"] = "ttsep"
    payload["separator_token_ids"] = [99, 98]
    payload["segments"] = [
        {
            "segment_id": "private-0",
            "kind": "private_history",
            "logical_order": 0,
            "source_id": "agent-0",
            "shareable": False,
            "lineage_id": "lineage-0",
        },
        {
            "segment_id": "shared-0",
            "kind": "shared_update",
            "logical_order": 1,
            "source_id": "update-0",
            "shareable": True,
            "lineage_id": "lineage-0",
        },
        {
            "segment_id": "task-0",
            "kind": "task",
            "logical_order": 2,
            "source_id": None,
            "shareable": True,
            "lineage_id": None,
        },
    ]
    return payload


def assert_invalid(payload: object, code: KMeshMetadataErrorCode) -> None:
    result = parse_kmesh_request_context(payload, [10, 11, 12, 13, 14])
    assert result.context is None
    assert result.fallback_reason is KMeshFallbackReason.INVALID_METADATA
    assert result.error is not None
    assert result.error.code is code


def test_sideband_parser_normalizes_to_immutable_segments():
    result = parse_kmesh_extra_args({"other": 1, "kmesh": sideband_payload()}, range(5))

    assert result.is_enabled
    assert result.fallback_reason is None
    assert result.error is None
    assert result.context is not None
    assert result.context.round_id == 3
    assert result.context.segments[1].kind is KMeshSegmentKind.SHARED_UPDATE
    assert result.context.segments[1].token_start == 2
    assert result.context.segments[1].token_end == 5


def test_absent_metadata_has_stable_no_metadata_fallback():
    for extra_args in (None, {}, {"other": "value"}):
        result = parse_kmesh_extra_args(extra_args, [1])
        assert result.context is None
        assert result.error is None
        assert result.fallback_reason is KMeshFallbackReason.NO_METADATA


def test_disabled_metadata_is_valid_and_measured():
    payload = sideband_payload()
    payload["enabled"] = False

    result = parse_kmesh_request_context(payload, range(5))

    assert result.context is not None
    assert not result.context.enabled
    assert result.error is None
    assert result.fallback_reason is KMeshFallbackReason.DISABLED


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (lambda p: p.update(schema_version=2), "unsupported_schema_version"),
        (lambda p: p.update(schema_version=True), "invalid_type"),
        (lambda p: p.update(enabled=1), "invalid_type"),
        (lambda p: p.update(group_kind="AGENT_ROUND"), "unknown_enum_value"),
        (lambda p: p.update(state_kind="weights"), "unknown_enum_value"),
        (lambda p: p.update(execution_id=""), "invalid_identifier"),
        (lambda p: p.update(group_id=" padded "), "invalid_identifier"),
        (lambda p: p.update(round_id=-1), "invalid_round_id"),
        (lambda p: p.update(round_id=True), "invalid_round_id"),
        (lambda p: p.update(cache_domain={"tenant": "a"}), "invalid_cache_domain"),
        (lambda p: p.update(segment_encoding="markers"), "unknown_enum_value"),
        (lambda p: p.update(unknown=True), "unknown_field"),
        (lambda p: p.pop("execution_id"), "missing_field"),
        (lambda p: p.update(segments=()), "expected_list"),
        (lambda p: p.update(segments=[]), "empty_segments"),
    ],
)
def test_context_schema_rejects_unknown_or_coerced_values(mutation, code):
    payload = sideband_payload()
    mutation(payload)

    assert_invalid(payload, KMeshMetadataErrorCode(code))


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (
            lambda p: p["segments"][0].update(token_start=-1),
            "invalid_range",
        ),
        (
            lambda p: p["segments"][0].update(token_end=0),
            "empty_segment",
        ),
        (
            lambda p: p["segments"][0].update(token_end=-1),
            "invalid_range",
        ),
        (
            lambda p: p["segments"][1].update(token_start=1),
            "overlapping_segments",
        ),
        (
            lambda p: p.update(segments=list(reversed(p["segments"]))),
            "unordered_segments",
        ),
        (
            lambda p: p["segments"][1].update(segment_id="private-0"),
            "duplicate_segment_id",
        ),
        (
            lambda p: p["segments"][1].update(logical_order=0),
            "invalid_logical_order",
        ),
        (
            lambda p: p["segments"][1].update(token_end=6),
            "out_of_bounds",
        ),
        (
            lambda p: p["segments"][0].update(shareable=1),
            "invalid_type",
        ),
        (
            lambda p: p["segments"][0].update(kind="SHARED_UPDATE"),
            "unknown_enum_value",
        ),
        (
            lambda p: p["segments"][0].update(extra=True),
            "unknown_field",
        ),
        (
            lambda p: p["segments"][0].pop("token_start"),
            "missing_field",
        ),
    ],
)
def test_sideband_segment_validation_is_fail_closed(mutation, code):
    payload = sideband_payload()
    mutation(payload)

    assert_invalid(payload, KMeshMetadataErrorCode(code))


def test_ttsep_front_end_produces_spans_and_does_not_leak_encoding():
    prompt = [10, 11, 99, 98, 20, 21, 99, 98, 30]

    result = parse_kmesh_request_context(ttsep_payload(), prompt)

    assert result.is_enabled
    assert result.context is not None
    assert [(s.token_start, s.token_end) for s in result.context.segments] == [
        (0, 2),
        (4, 6),
        (8, 9),
    ]
    assert "segment_encoding" not in result.context.to_primitive()
    assert "separator_token_ids" not in result.context.to_primitive()


def test_ttsep_without_a_separator_is_one_segment():
    payload = ttsep_payload()
    payload["segments"] = payload["segments"][:1]

    result = parse_kmesh_request_context(payload, [10, 11, 12])

    assert result.is_enabled
    assert result.context is not None
    assert result.context.segments[0].token_start == 0
    assert result.context.segments[0].token_end == 3


@pytest.mark.parametrize(
    "prompt",
    [
        [99, 98, 10],
        [10, 99, 98],
        [10, 99, 98, 99, 98, 20],
        [],
    ],
)
def test_ttsep_rejects_empty_segments(prompt):
    with pytest.raises(KMeshValidationError) as exc_info:
        parse_ttsep_segment_spans(prompt, [99, 98])

    assert exc_info.value.error.code is KMeshMetadataErrorCode.EMPTY_SEGMENT


def test_ttsep_requires_one_descriptor_per_generated_span():
    payload = ttsep_payload()
    payload["segments"].pop()

    result = parse_kmesh_request_context(payload, [10, 99, 98, 20, 99, 98, 30])

    assert result.error is not None
    assert result.error.code is KMeshMetadataErrorCode.SEGMENT_COUNT_MISMATCH


def test_ttsep_descriptors_cannot_supply_ranges():
    payload = ttsep_payload()
    payload["segments"][0]["token_start"] = 0

    result = parse_kmesh_request_context(payload, [10, 99, 98, 20, 99, 98, 30])

    assert result.error is not None
    assert result.error.code is KMeshMetadataErrorCode.UNKNOWN_FIELD


@pytest.mark.parametrize("separator", [[], [True], [-1], "99"])
def test_ttsep_rejects_invalid_separator_sequences(separator):
    payload = ttsep_payload()
    payload["separator_token_ids"] = separator

    result = parse_kmesh_request_context(payload, [10, 11])

    assert result.error is not None
    assert result.error.code in {
        KMeshMetadataErrorCode.EXPECTED_LIST,
        KMeshMetadataErrorCode.INVALID_SEPARATOR,
    }


def test_sideband_rejects_ttsep_only_fields():
    payload = sideband_payload()
    payload["separator_token_ids"] = [99]

    result = parse_kmesh_request_context(payload, range(5))

    assert result.error is not None
    assert result.error.code is KMeshMetadataErrorCode.ENCODING_FIELD_MISMATCH


def test_non_mapping_extra_args_and_payload_are_typed_failures():
    extra_result = parse_kmesh_extra_args([], [1])
    payload_result = parse_kmesh_extra_args({"kmesh": []}, [1])

    assert extra_result.error is not None
    assert extra_result.error.code is KMeshMetadataErrorCode.EXPECTED_MAPPING
    assert payload_result.error is not None
    assert payload_result.error.code is KMeshMetadataErrorCode.EXPECTED_MAPPING


def test_sideband_primitive_msgpack_round_trip():
    primitive = msgspec.msgpack.decode(msgspec.msgpack.encode(sideband_payload()))

    first = parse_kmesh_request_context(primitive, range(5))
    assert first.context is not None
    serialized = kmesh_context_to_sideband(first.context)
    second = parse_kmesh_request_context(
        msgspec.msgpack.decode(msgspec.msgpack.encode(serialized)), range(5)
    )

    assert second.context == first.context


def test_error_paths_identify_the_invalid_nested_field():
    payload = deepcopy(sideband_payload())
    payload["segments"][1]["token_start"] = True

    result = parse_kmesh_request_context(payload, range(5))

    assert result.error is not None
    assert result.error.code is KMeshMetadataErrorCode.INVALID_RANGE
    assert result.error.path == "segments[1].token_start"
