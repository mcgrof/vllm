# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import Any, TypeVar

from vllm.v1.kmesh.types import (
    KMESH_EXTRA_ARGS_KEY,
    KMESH_SCHEMA_VERSION,
    KMeshGroupKind,
    KMeshMetadataError,
    KMeshMetadataErrorCode,
    KMeshRequestContext,
    KMeshRequestContextParseResult,
    KMeshSegment,
    KMeshSegmentEncoding,
    KMeshSegmentKind,
    KMeshStateKind,
    KMeshValidationError,
    _fail,
    _is_strict_int,
)

_EnumT = TypeVar("_EnumT", bound=Enum)

_CONTEXT_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "enabled",
        "execution_id",
        "group_kind",
        "group_id",
        "round_id",
        "state_kind",
        "segment_encoding",
        "segments",
    }
)
_CONTEXT_OPTIONAL_FIELDS = frozenset(
    {"branch_id", "parent_branch_id", "cache_domain", "separator_token_ids"}
)
_SIDEBAND_SEGMENT_REQUIRED_FIELDS = frozenset(
    {
        "segment_id",
        "kind",
        "token_start",
        "token_end",
        "logical_order",
        "shareable",
    }
)
_TTSEP_SEGMENT_REQUIRED_FIELDS = frozenset(
    {"segment_id", "kind", "logical_order", "shareable"}
)
_SEGMENT_OPTIONAL_FIELDS = frozenset({"source_id", "lineage_id"})


def _expect_dict(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(
            KMeshMetadataErrorCode.EXPECTED_MAPPING,
            path,
            "expected a mapping with string keys",
        )
    if any(not isinstance(key, str) for key in value):
        _fail(
            KMeshMetadataErrorCode.INVALID_TYPE,
            path,
            "mapping keys must be strings",
        )
    return value


def _expect_list(value: object, path: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(KMeshMetadataErrorCode.EXPECTED_LIST, path, "expected a list")
    return value


def _check_fields(
    value: dict[str, Any],
    required: frozenset[str],
    optional: frozenset[str],
    path: str,
) -> None:
    if missing := sorted(required - value.keys()):
        field = missing[0]
        _fail(
            KMeshMetadataErrorCode.MISSING_FIELD,
            f"{path}.{field}" if path else field,
            "required field is missing",
        )
    if unknown := sorted(value.keys() - required - optional):
        field = unknown[0]
        _fail(
            KMeshMetadataErrorCode.UNKNOWN_FIELD,
            f"{path}.{field}" if path else field,
            "unknown field",
        )


def _parse_enum(value: object, enum_type: type[_EnumT], path: str) -> _EnumT:
    if not isinstance(value, str):
        _fail(
            KMeshMetadataErrorCode.INVALID_TYPE,
            path,
            "enum value must be a string",
        )
    try:
        return enum_type(value)
    except ValueError:
        _fail(
            KMeshMetadataErrorCode.UNKNOWN_ENUM_VALUE,
            path,
            f"unknown {enum_type.__name__} value",
        )


def _validate_schema_version(value: object) -> int:
    if not _is_strict_int(value):
        _fail(
            KMeshMetadataErrorCode.INVALID_TYPE,
            "schema_version",
            "schema_version must be an integer",
        )
    if value != KMESH_SCHEMA_VERSION:
        _fail(
            KMeshMetadataErrorCode.UNSUPPORTED_SCHEMA_VERSION,
            "schema_version",
            f"supported schema version is {KMESH_SCHEMA_VERSION}",
        )
    return value


def _validate_prompt_token_ids(value: object) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        _fail(
            KMeshMetadataErrorCode.INVALID_TYPE,
            "prompt_token_ids",
            "prompt_token_ids must be a sequence of token IDs",
        )
    token_ids: list[int] = []
    for index, token_id in enumerate(value):
        if not _is_strict_int(token_id) or token_id < 0:
            _fail(
                KMeshMetadataErrorCode.INVALID_TOKEN_ID,
                f"prompt_token_ids[{index}]",
                "token IDs must be nonnegative integers",
            )
        token_ids.append(token_id)
    return tuple(token_ids)


def _validate_separator_token_ids(value: object) -> tuple[int, ...]:
    raw_token_ids = _expect_list(value, "separator_token_ids")
    if not raw_token_ids:
        _fail(
            KMeshMetadataErrorCode.INVALID_SEPARATOR,
            "separator_token_ids",
            "separator token sequence must be nonempty",
        )
    token_ids: list[int] = []
    for index, token_id in enumerate(raw_token_ids):
        if not _is_strict_int(token_id) or token_id < 0:
            _fail(
                KMeshMetadataErrorCode.INVALID_SEPARATOR,
                f"separator_token_ids[{index}]",
                "separator token IDs must be nonnegative integers",
            )
        token_ids.append(token_id)
    return tuple(token_ids)


def parse_ttsep_segment_spans(
    prompt_token_ids: Sequence[int], separator_token_ids: Sequence[int]
) -> tuple[tuple[int, int], ...]:
    """Split a tokenized prompt into nonempty spans around TTSEP tokens."""
    prompt = _validate_prompt_token_ids(prompt_token_ids)
    separator = _validate_separator_token_ids(list(separator_token_ids))

    spans: list[tuple[int, int]] = []
    span_start = 0
    cursor = 0
    separator_length = len(separator)
    last_match_start = len(prompt) - separator_length
    while cursor <= last_match_start:
        if prompt[cursor : cursor + separator_length] != separator:
            cursor += 1
            continue
        if cursor == span_start:
            _fail(
                KMeshMetadataErrorCode.EMPTY_SEGMENT,
                "prompt_token_ids",
                "TTSEP cannot create an empty segment",
            )
        spans.append((span_start, cursor))
        cursor += separator_length
        span_start = cursor

    if span_start == len(prompt):
        _fail(
            KMeshMetadataErrorCode.EMPTY_SEGMENT,
            "prompt_token_ids",
            "TTSEP cannot create an empty trailing segment",
        )
    spans.append((span_start, len(prompt)))
    return tuple(spans)


def _rebase_error(error: KMeshMetadataError, prefix: str) -> KMeshValidationError:
    path = f"{prefix}.{error.path}" if error.path else prefix
    return KMeshValidationError(
        KMeshMetadataError(code=error.code, path=path, message=error.message)
    )


def _make_segment(
    raw_segment: object,
    index: int,
    encoding: KMeshSegmentEncoding,
    span: tuple[int, int] | None,
) -> KMeshSegment:
    path = f"segments[{index}]"
    segment = _expect_dict(raw_segment, path)
    required = (
        _SIDEBAND_SEGMENT_REQUIRED_FIELDS
        if encoding is KMeshSegmentEncoding.SIDEBAND
        else _TTSEP_SEGMENT_REQUIRED_FIELDS
    )
    _check_fields(segment, required, _SEGMENT_OPTIONAL_FIELDS, path)
    kind = _parse_enum(segment["kind"], KMeshSegmentKind, f"{path}.kind")

    if span is None:
        token_start = segment["token_start"]
        token_end = segment["token_end"]
    else:
        token_start, token_end = span

    try:
        return KMeshSegment(
            segment_id=segment["segment_id"],
            kind=kind,
            token_start=token_start,
            token_end=token_end,
            logical_order=segment["logical_order"],
            source_id=segment.get("source_id"),
            shareable=segment["shareable"],
            lineage_id=segment.get("lineage_id"),
        )
    except KMeshValidationError as exc:
        raise _rebase_error(exc.error, path) from exc


def _parse_context(
    raw_context: object, prompt_token_ids: Sequence[int]
) -> KMeshRequestContext:
    context = _expect_dict(raw_context, KMESH_EXTRA_ARGS_KEY)
    _check_fields(
        context,
        _CONTEXT_REQUIRED_FIELDS,
        _CONTEXT_OPTIONAL_FIELDS,
        "",
    )
    schema_version = _validate_schema_version(context["schema_version"])
    prompt = _validate_prompt_token_ids(prompt_token_ids)
    encoding = _parse_enum(
        context["segment_encoding"],
        KMeshSegmentEncoding,
        "segment_encoding",
    )
    raw_segments = _expect_list(context["segments"], "segments")

    spans: tuple[tuple[int, int], ...] | None = None
    if encoding is KMeshSegmentEncoding.TTSEP:
        if "separator_token_ids" not in context:
            _fail(
                KMeshMetadataErrorCode.MISSING_FIELD,
                "separator_token_ids",
                "TTSEP encoding requires separator_token_ids",
            )
        separator = _validate_separator_token_ids(context["separator_token_ids"])
        spans = parse_ttsep_segment_spans(prompt, separator)
        if len(spans) != len(raw_segments):
            _fail(
                KMeshMetadataErrorCode.SEGMENT_COUNT_MISMATCH,
                "segments",
                "segment descriptors must match TTSEP-produced spans",
            )
    elif "separator_token_ids" in context:
        _fail(
            KMeshMetadataErrorCode.ENCODING_FIELD_MISMATCH,
            "separator_token_ids",
            "separator_token_ids is only valid with TTSEP encoding",
        )

    segments = tuple(
        _make_segment(
            raw_segment,
            index,
            encoding,
            spans[index] if spans is not None and index < len(spans) else None,
        )
        for index, raw_segment in enumerate(raw_segments)
    )
    for index, segment in enumerate(segments):
        if segment.token_end > len(prompt):
            _fail(
                KMeshMetadataErrorCode.OUT_OF_BOUNDS,
                f"segments[{index}].token_end",
                "segment end exceeds prompt length",
            )

    return KMeshRequestContext(
        schema_version=schema_version,
        enabled=context["enabled"],
        execution_id=context["execution_id"],
        group_kind=_parse_enum(context["group_kind"], KMeshGroupKind, "group_kind"),
        group_id=context["group_id"],
        round_id=context["round_id"],
        branch_id=context.get("branch_id"),
        parent_branch_id=context.get("parent_branch_id"),
        state_kind=_parse_enum(context["state_kind"], KMeshStateKind, "state_kind"),
        cache_domain=context.get("cache_domain"),
        segments=segments,
    )


def parse_kmesh_request_context(
    raw_context: object, prompt_token_ids: Sequence[int]
) -> KMeshRequestContextParseResult:
    """Parse one primitive KMesh payload into an immutable request context."""
    try:
        context = _parse_context(raw_context, prompt_token_ids)
    except KMeshValidationError as exc:
        return KMeshRequestContextParseResult.invalid(exc.error)
    if not context.enabled:
        return KMeshRequestContextParseResult.disabled(context)
    return KMeshRequestContextParseResult.accepted(context)


def parse_kmesh_extra_args(
    extra_args: object, prompt_token_ids: Sequence[int]
) -> KMeshRequestContextParseResult:
    """Parse ``SamplingParams.extra_args['kmesh']`` without importing vLLM APIs."""
    if extra_args is None:
        return KMeshRequestContextParseResult.no_metadata()
    try:
        args = _expect_dict(extra_args, "extra_args")
    except KMeshValidationError as exc:
        return KMeshRequestContextParseResult.invalid(exc.error)
    if KMESH_EXTRA_ARGS_KEY not in args:
        return KMeshRequestContextParseResult.no_metadata()
    return parse_kmesh_request_context(args[KMESH_EXTRA_ARGS_KEY], prompt_token_ids)


def kmesh_context_to_sideband(context: KMeshRequestContext) -> dict[str, Any]:
    """Return a primitive sideband payload accepted by the boundary parser."""
    payload = context.to_primitive()
    payload["segment_encoding"] = KMeshSegmentEncoding.SIDEBAND.value
    return payload
