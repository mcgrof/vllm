# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, NoReturn, TypeGuard

KMESH_SCHEMA_VERSION = 1
KMESH_EXTRA_ARGS_KEY = "kmesh"

_MAX_IDENTIFIER_LENGTH = 256


class KMeshGroupKind(str, Enum):
    AGENT_ROUND = "agent_round"
    COT_BRANCH = "cot_branch"
    TEST_TIME_SAMPLE = "test_time_sample"
    SPECULATIVE_BRANCH = "speculative_branch"
    OTHER = "other"


class KMeshStateKind(str, Enum):
    KV_CACHE = "kv_cache"
    KV_DIFF = "kv_diff"
    PLAN = "plan"
    TOOL_RESULT = "tool_result"
    TRACE = "trace"
    OTHER = "other"


class KMeshSegmentKind(str, Enum):
    PRIVATE_HISTORY = "private_history"
    SHARED_UPDATE = "shared_update"
    TASK = "task"
    SYSTEM = "system"
    THOUGHT = "thought"
    TOOL_RESULT = "tool_result"
    OTHER = "other"


class KMeshSegmentEncoding(str, Enum):
    SIDEBAND = "sideband"
    TTSEP = "ttsep"


class KMeshFallbackReason(str, Enum):
    DISABLED = "disabled"
    NO_METADATA = "no_metadata"
    INVALID_METADATA = "invalid_metadata"
    UNSUPPORTED_MODEL = "unsupported_model"
    GROUP_KIND_UNSUPPORTED = "group_kind_unsupported"
    STATE_KIND_UNSUPPORTED = "state_kind_unsupported"
    MISSING_COMPATIBILITY_EVIDENCE = "missing_compatibility_evidence"
    MISSING_PROMPT_TOKEN_EVIDENCE = "missing_prompt_token_evidence"
    PROMPT_TOKEN_LENGTH_MISMATCH = "prompt_token_length_mismatch"
    SEGMENT_CONTENT_MISMATCH = "segment_content_mismatch"
    EXECUTION_ID_MISMATCH = "execution_id_mismatch"
    GROUP_ID_MISMATCH = "group_id_mismatch"
    ROUND_MISMATCH = "round_mismatch"
    MODEL_MISMATCH = "model_mismatch"
    TOKENIZER_MISMATCH = "tokenizer_mismatch"
    DTYPE_MISMATCH = "dtype_mismatch"
    KV_CACHE_DTYPE_MISMATCH = "kv_cache_dtype_mismatch"
    KV_LAYOUT_MISMATCH = "kv_layout_mismatch"
    ATTENTION_KIND_MISMATCH = "attention_kind_mismatch"
    ROPE_CONFIGURATION_MISMATCH = "rope_configuration_mismatch"
    ATTENTION_BACKEND_MISMATCH = "attention_backend_mismatch"
    TENSOR_PARALLEL_MISMATCH = "tensor_parallel_mismatch"
    PIPELINE_PARALLEL_MISMATCH = "pipeline_parallel_mismatch"
    LORA_MISMATCH = "lora_mismatch"
    CACHE_SALT_MISMATCH = "cache_salt_mismatch"
    CACHE_DOMAIN_MISMATCH = "cache_domain_mismatch"
    ACTIVE_LENGTH_MISMATCH = "active_length_mismatch"
    CACHED_SPAN_MISMATCH = "cached_span_mismatch"
    PREFILL_CHUNK_MISMATCH = "prefill_chunk_mismatch"
    NO_SHARED_SEGMENTS = "no_shared_segments"
    SINGLETON_GROUP = "singleton_group"
    SLOT_MAPPING_OVERLAP = "slot_mapping_overlap"
    CHUNKED_PREFILL_UNSUPPORTED = "chunked_prefill_unsupported"
    ASYNC_SCHEDULING_UNSUPPORTED = "async_scheduling_unsupported"
    PIC_BACKEND_UNAVAILABLE = "pic_backend_unavailable"
    PIC_CACHE_MISS = "pic_cache_miss"
    GROUP_TIMEOUT = "group_timeout"
    INTERNAL_ERROR = "internal_error"


class KMeshMetadataErrorCode(str, Enum):
    EXPECTED_MAPPING = "expected_mapping"
    EXPECTED_LIST = "expected_list"
    MISSING_FIELD = "missing_field"
    UNKNOWN_FIELD = "unknown_field"
    INVALID_TYPE = "invalid_type"
    UNSUPPORTED_SCHEMA_VERSION = "unsupported_schema_version"
    UNKNOWN_ENUM_VALUE = "unknown_enum_value"
    INVALID_IDENTIFIER = "invalid_identifier"
    INVALID_ROUND_ID = "invalid_round_id"
    INVALID_CACHE_DOMAIN = "invalid_cache_domain"
    INVALID_RANGE = "invalid_range"
    EMPTY_SEGMENT = "empty_segment"
    EMPTY_SEGMENTS = "empty_segments"
    UNORDERED_SEGMENTS = "unordered_segments"
    OVERLAPPING_SEGMENTS = "overlapping_segments"
    DUPLICATE_SEGMENT_ID = "duplicate_segment_id"
    INVALID_LOGICAL_ORDER = "invalid_logical_order"
    OUT_OF_BOUNDS = "out_of_bounds"
    INVALID_TOKEN_ID = "invalid_token_id"
    INVALID_SEPARATOR = "invalid_separator"
    SEGMENT_COUNT_MISMATCH = "segment_count_mismatch"
    ENCODING_FIELD_MISMATCH = "encoding_field_mismatch"


@dataclass(frozen=True, slots=True)
class KMeshMetadataError:
    code: KMeshMetadataErrorCode
    path: str
    message: str

    def to_primitive(self) -> dict[str, str]:
        return {
            "code": self.code.value,
            "path": self.path,
            "message": self.message,
        }


class KMeshValidationError(ValueError):
    def __init__(self, error: KMeshMetadataError):
        super().__init__(f"{error.path}: {error.message}")
        self.error = error


def _fail(code: KMeshMetadataErrorCode, path: str, message: str) -> NoReturn:
    raise KMeshValidationError(KMeshMetadataError(code, path, message))


def _is_strict_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_identifier(value: object, path: str) -> str:
    if not isinstance(value, str):
        _fail(
            KMeshMetadataErrorCode.INVALID_TYPE,
            path,
            "expected a string identifier",
        )
    if (
        not value
        or value != value.strip()
        or len(value) > _MAX_IDENTIFIER_LENGTH
        or any(
            ord(char) < 32 or ord(char) == 127 or 0xD800 <= ord(char) <= 0xDFFF
            for char in value
        )
    ):
        _fail(
            KMeshMetadataErrorCode.INVALID_IDENTIFIER,
            path,
            "identifier must be nonempty, trimmed, bounded, and contain no controls "
            "or surrogates",
        )
    return value


def _validate_optional_identifier(value: object, path: str) -> str | None:
    if value is None:
        return None
    return _validate_identifier(value, path)


@dataclass(frozen=True, slots=True)
class KMeshSegment:
    segment_id: str
    kind: KMeshSegmentKind
    token_start: int
    token_end: int
    logical_order: int
    source_id: str | None
    shareable: bool
    lineage_id: str | None

    def __post_init__(self) -> None:
        _validate_identifier(self.segment_id, "segment_id")
        if not isinstance(self.kind, KMeshSegmentKind):
            _fail(
                KMeshMetadataErrorCode.INVALID_TYPE,
                "kind",
                "expected KMeshSegmentKind",
            )
        if not _is_strict_int(self.token_start) or self.token_start < 0:
            _fail(
                KMeshMetadataErrorCode.INVALID_RANGE,
                "token_start",
                "token_start must be a nonnegative integer",
            )
        if not _is_strict_int(self.token_end) or self.token_end < 0:
            _fail(
                KMeshMetadataErrorCode.INVALID_RANGE,
                "token_end",
                "token_end must be a nonnegative integer",
            )
        if self.token_end == self.token_start:
            _fail(
                KMeshMetadataErrorCode.EMPTY_SEGMENT,
                "token_end",
                "empty segments are not supported",
            )
        if self.token_end < self.token_start:
            _fail(
                KMeshMetadataErrorCode.INVALID_RANGE,
                "token_end",
                "token_end must be greater than token_start",
            )
        if not _is_strict_int(self.logical_order) or self.logical_order < 0:
            _fail(
                KMeshMetadataErrorCode.INVALID_LOGICAL_ORDER,
                "logical_order",
                "logical_order must be a nonnegative integer",
            )
        _validate_optional_identifier(self.source_id, "source_id")
        if not isinstance(self.shareable, bool):
            _fail(
                KMeshMetadataErrorCode.INVALID_TYPE,
                "shareable",
                "shareable must be a boolean",
            )
        _validate_optional_identifier(self.lineage_id, "lineage_id")

    def to_primitive(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "kind": self.kind.value,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "logical_order": self.logical_order,
            "source_id": self.source_id,
            "shareable": self.shareable,
            "lineage_id": self.lineage_id,
        }


@dataclass(frozen=True, slots=True)
class KMeshRequestContext:
    schema_version: int
    enabled: bool
    execution_id: str
    group_kind: KMeshGroupKind
    group_id: str
    round_id: int | str
    branch_id: str | None
    parent_branch_id: str | None
    state_kind: KMeshStateKind
    cache_domain: str | None
    segments: tuple[KMeshSegment, ...]

    def __post_init__(self) -> None:
        if not _is_strict_int(self.schema_version):
            _fail(
                KMeshMetadataErrorCode.INVALID_TYPE,
                "schema_version",
                "schema_version must be an integer",
            )
        if self.schema_version != KMESH_SCHEMA_VERSION:
            _fail(
                KMeshMetadataErrorCode.UNSUPPORTED_SCHEMA_VERSION,
                "schema_version",
                f"supported schema version is {KMESH_SCHEMA_VERSION}",
            )
        if not isinstance(self.enabled, bool):
            _fail(
                KMeshMetadataErrorCode.INVALID_TYPE,
                "enabled",
                "enabled must be a boolean",
            )
        _validate_identifier(self.execution_id, "execution_id")
        if not isinstance(self.group_kind, KMeshGroupKind):
            _fail(
                KMeshMetadataErrorCode.INVALID_TYPE,
                "group_kind",
                "expected KMeshGroupKind",
            )
        _validate_identifier(self.group_id, "group_id")
        if _is_strict_int(self.round_id):
            if self.round_id < 0:
                _fail(
                    KMeshMetadataErrorCode.INVALID_ROUND_ID,
                    "round_id",
                    "integer round_id must be nonnegative",
                )
        elif isinstance(self.round_id, str):
            _validate_identifier(self.round_id, "round_id")
        else:
            _fail(
                KMeshMetadataErrorCode.INVALID_ROUND_ID,
                "round_id",
                "round_id must be a nonnegative integer or stable string",
            )
        _validate_optional_identifier(self.branch_id, "branch_id")
        _validate_optional_identifier(self.parent_branch_id, "parent_branch_id")
        if not isinstance(self.state_kind, KMeshStateKind):
            _fail(
                KMeshMetadataErrorCode.INVALID_TYPE,
                "state_kind",
                "expected KMeshStateKind",
            )
        if self.cache_domain is not None:
            if not isinstance(self.cache_domain, str):
                _fail(
                    KMeshMetadataErrorCode.INVALID_CACHE_DOMAIN,
                    "cache_domain",
                    "cache_domain must be a string or null",
                )
            try:
                _validate_identifier(self.cache_domain, "cache_domain")
            except KMeshValidationError as exc:
                _fail(
                    KMeshMetadataErrorCode.INVALID_CACHE_DOMAIN,
                    "cache_domain",
                    exc.error.message,
                )
        if not isinstance(self.segments, tuple):
            _fail(
                KMeshMetadataErrorCode.INVALID_TYPE,
                "segments",
                "segments must be a tuple",
            )
        if not self.segments:
            _fail(
                KMeshMetadataErrorCode.EMPTY_SEGMENTS,
                "segments",
                "at least one segment is required",
            )

        seen_ids: set[str] = set()
        previous: KMeshSegment | None = None
        for index, segment in enumerate(self.segments):
            path = f"segments[{index}]"
            if not isinstance(segment, KMeshSegment):
                _fail(
                    KMeshMetadataErrorCode.INVALID_TYPE,
                    path,
                    "expected KMeshSegment",
                )
            if segment.segment_id in seen_ids:
                _fail(
                    KMeshMetadataErrorCode.DUPLICATE_SEGMENT_ID,
                    f"{path}.segment_id",
                    "segment_id values must be unique within a request",
                )
            seen_ids.add(segment.segment_id)
            if previous is not None:
                if segment.token_start < previous.token_start:
                    _fail(
                        KMeshMetadataErrorCode.UNORDERED_SEGMENTS,
                        f"{path}.token_start",
                        "segments must be ordered by token range",
                    )
                if segment.token_start < previous.token_end:
                    _fail(
                        KMeshMetadataErrorCode.OVERLAPPING_SEGMENTS,
                        f"{path}.token_start",
                        "segment ranges must not overlap",
                    )
                if segment.logical_order <= previous.logical_order:
                    _fail(
                        KMeshMetadataErrorCode.INVALID_LOGICAL_ORDER,
                        f"{path}.logical_order",
                        "logical_order must increase with token order",
                    )
            previous = segment

    def to_primitive(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "enabled": self.enabled,
            "execution_id": self.execution_id,
            "group_kind": self.group_kind.value,
            "group_id": self.group_id,
            "round_id": self.round_id,
            "branch_id": self.branch_id,
            "parent_branch_id": self.parent_branch_id,
            "state_kind": self.state_kind.value,
            "cache_domain": self.cache_domain,
            "segments": [segment.to_primitive() for segment in self.segments],
        }


@dataclass(frozen=True, slots=True)
class KMeshRequestContextParseResult:
    context: KMeshRequestContext | None
    fallback_reason: KMeshFallbackReason | None
    error: KMeshMetadataError | None = None

    def __post_init__(self) -> None:
        if self.error is not None:
            if (
                self.context is not None
                or self.fallback_reason is not KMeshFallbackReason.INVALID_METADATA
            ):
                raise ValueError("invalid parse-result state")
        elif self.context is None:
            if self.fallback_reason is not KMeshFallbackReason.NO_METADATA:
                raise ValueError("missing context requires NO_METADATA")
        elif self.context.enabled:
            if self.fallback_reason is not None:
                raise ValueError("enabled context cannot have a fallback")
        elif self.fallback_reason is not KMeshFallbackReason.DISABLED:
            raise ValueError("disabled context requires DISABLED fallback")

    @property
    def is_enabled(self) -> bool:
        return self.context is not None and self.context.enabled and self.error is None

    @classmethod
    def accepted(cls, context: KMeshRequestContext) -> KMeshRequestContextParseResult:
        return cls(context=context, fallback_reason=None)

    @classmethod
    def disabled(cls, context: KMeshRequestContext) -> KMeshRequestContextParseResult:
        return cls(context=context, fallback_reason=KMeshFallbackReason.DISABLED)

    @classmethod
    def no_metadata(cls) -> KMeshRequestContextParseResult:
        return cls(context=None, fallback_reason=KMeshFallbackReason.NO_METADATA)

    @classmethod
    def invalid(cls, error: KMeshMetadataError) -> KMeshRequestContextParseResult:
        return cls(
            context=None,
            fallback_reason=KMeshFallbackReason.INVALID_METADATA,
            error=error,
        )

    def raise_for_error(self) -> None:
        if self.error is not None:
            raise KMeshValidationError(self.error)

    def to_primitive(self) -> dict[str, Any]:
        return {
            "context": self.context.to_primitive() if self.context else None,
            "fallback_reason": (
                self.fallback_reason.value if self.fallback_reason else None
            ),
            "error": self.error.to_primitive() if self.error else None,
        }
