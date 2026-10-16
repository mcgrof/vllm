# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from typing import Protocol, TypeAlias, runtime_checkable

from vllm.v1.kmesh.grouping import KMeshSharedSegmentIdentity
from vllm.v1.kmesh.types import (
    KMeshFallbackReason,
    KMeshGroupKind,
    KMeshRequestContext,
    KMeshRequestContextParseResult,
    KMeshSegment,
    KMeshSegmentKind,
    KMeshStateKind,
    _is_strict_int,
    _validate_identifier,
    _validate_optional_identifier,
)

KMESH_REQUEST_SIDECAR_PROVIDER_ID = "vllm.kmesh.request_lookup.v1"
_KMESH_REQUEST_LOOKUP_WIRE_TAG = "kmesh-request-lookup-v1"

KMeshRequestSidecarValue: TypeAlias = (
    str | int | bool | None | tuple["KMeshRequestSidecarValue", ...]
)


class KMeshRequestLookupIntent(str, Enum):
    """Whether a peer may act on an admitted lookup descriptor."""

    OBSERVE_ONLY = "observe_only"
    SERIAL_PIC = "serial_pic"


def _wire_tuple(
    value: object, field: str, *, length: int | None = None
) -> tuple[object, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{field} must be a tuple")
    if length is not None and len(value) != length:
        raise ValueError(f"{field} must contain exactly {length} items")
    return value


def _wire_optional_string(value: object, field: str) -> str | None:
    if value is not None and type(value) is not str:
        raise TypeError(f"{field} must be a string or null")
    return value


def _wire_int(value: object, field: str) -> int:
    if not _is_strict_int(value):
        raise TypeError(f"{field} must be an integer")
    return value


@dataclass(frozen=True, slots=True)
class KMeshRequestLookupSpan:
    """One admitted shareable span with exact prompt-content evidence."""

    shared_segment: KMeshSharedSegmentIdentity
    token_start: int
    token_end: int
    logical_order: int

    def __post_init__(self) -> None:
        if not isinstance(self.shared_segment, KMeshSharedSegmentIdentity):
            raise TypeError("shared_segment must be KMeshSharedSegmentIdentity")
        for field in ("token_start", "token_end", "logical_order"):
            value = getattr(self, field)
            if not _is_strict_int(value) or value < 0:
                raise ValueError(f"{field} must be a nonnegative integer")
        if self.token_end - self.token_start != self.shared_segment.token_length:
            raise ValueError("span range must match the shared segment length")

    @classmethod
    def from_segment(
        cls, segment: KMeshSegment, prompt_token_ids: Sequence[int]
    ) -> KMeshRequestLookupSpan:
        return cls(
            shared_segment=KMeshSharedSegmentIdentity.from_segment(
                segment, prompt_token_ids
            ),
            token_start=segment.token_start,
            token_end=segment.token_end,
            logical_order=segment.logical_order,
        )

    def to_sidecar_payload(self) -> KMeshRequestSidecarValue:
        identity = self.shared_segment
        return (
            identity.segment_id,
            identity.kind.value,
            identity.token_length,
            identity.source_id,
            identity.lineage_id,
            identity.shareable,
            identity.content_digest,
            self.token_start,
            self.token_end,
            self.logical_order,
        )

    @classmethod
    def from_sidecar_payload(cls, payload: object) -> KMeshRequestLookupSpan:
        fields = _wire_tuple(payload, "span", length=10)
        (
            segment_id,
            kind,
            token_length,
            source_id,
            lineage_id,
            shareable,
            content_digest,
            token_start,
            token_end,
            logical_order,
        ) = fields
        if type(segment_id) is not str:
            raise TypeError("span.segment_id must be a string")
        if type(kind) is not str:
            raise TypeError("span.kind must be a string")
        try:
            segment_kind = KMeshSegmentKind(kind)
        except ValueError as exc:
            raise ValueError(f"span.kind has unknown value {kind!r}") from exc
        token_length = _wire_int(token_length, "span.token_length")
        source_id = _wire_optional_string(source_id, "span.source_id")
        lineage_id = _wire_optional_string(lineage_id, "span.lineage_id")
        if type(shareable) is not bool:
            raise TypeError("span.shareable must be a boolean")
        if type(content_digest) is not str:
            raise TypeError("span.content_digest must be a string")
        token_start = _wire_int(token_start, "span.token_start")
        token_end = _wire_int(token_end, "span.token_end")
        logical_order = _wire_int(logical_order, "span.logical_order")
        return cls(
            shared_segment=KMeshSharedSegmentIdentity(
                segment_id=segment_id,
                kind=segment_kind,
                token_length=token_length,
                source_id=source_id,
                lineage_id=lineage_id,
                shareable=shareable,
                content_digest=content_digest,
            ),
            token_start=token_start,
            token_end=token_end,
            logical_order=logical_order,
        )


@dataclass(frozen=True, slots=True)
class KMeshRequestLookupDescriptor:
    """Immutable request admission evidence for a peer connector lookup.

    This value describes admission evidence, not a selected reuse plan. A peer
    must not alter lookup behavior unless ``permits_peer_lookup`` is true. Group
    selection is not known when connector lookups run; transported worker plan
    metadata remains authoritative before any later KMesh execution.
    """

    lookup_intent: KMeshRequestLookupIntent
    request_id: str
    admission_generation: int
    prompt_token_count: int
    cache_domain: str | None
    spans: tuple[KMeshRequestLookupSpan, ...]
    admission_fallback: KMeshFallbackReason | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.lookup_intent, KMeshRequestLookupIntent):
            raise TypeError("lookup_intent must be KMeshRequestLookupIntent")
        _validate_identifier(self.request_id, "request_id")
        if (
            not _is_strict_int(self.admission_generation)
            or self.admission_generation <= 0
        ):
            raise ValueError("admission_generation must be a positive integer")
        if not _is_strict_int(self.prompt_token_count) or self.prompt_token_count < 0:
            raise ValueError("prompt_token_count must be a nonnegative integer")
        _validate_optional_identifier(self.cache_domain, "cache_domain")
        if type(self.spans) is not tuple:
            raise TypeError("spans must be an immutable tuple")
        if self.admission_fallback is not None and not isinstance(
            self.admission_fallback, KMeshFallbackReason
        ):
            raise TypeError("admission_fallback must be KMeshFallbackReason or null")
        if self.admission_fallback is not None:
            if self.spans:
                raise ValueError("a fallback descriptor cannot carry eligible spans")
            return
        if self.cache_domain is None:
            raise ValueError("an eligible descriptor requires a cache domain")
        if not self.spans:
            raise ValueError("an eligible descriptor requires shareable spans")

        seen_segment_ids: set[str] = set()
        previous: KMeshRequestLookupSpan | None = None
        for index, span in enumerate(self.spans):
            if not isinstance(span, KMeshRequestLookupSpan):
                raise TypeError(f"spans[{index}] must be KMeshRequestLookupSpan")
            segment_id = span.shared_segment.segment_id
            if segment_id in seen_segment_ids:
                raise ValueError("shareable span IDs must be unique")
            seen_segment_ids.add(segment_id)
            if span.token_end > self.prompt_token_count:
                raise ValueError("shareable spans must fit within the admitted prompt")
            if previous is not None:
                if span.token_start < previous.token_end:
                    raise ValueError(
                        "shareable spans must be ordered and non-overlapping"
                    )
                if span.logical_order <= previous.logical_order:
                    raise ValueError("shareable span logical order must increase")
            previous = span

    @classmethod
    def from_context(
        cls,
        *,
        request_id: str,
        admission_generation: int,
        prompt_token_ids: Sequence[int],
        prompt_token_count: int,
        context: KMeshRequestContext,
        lookup_intent: KMeshRequestLookupIntent,
    ) -> KMeshRequestLookupDescriptor:
        if not context.enabled:
            raise ValueError("disabled KMesh contexts are not lookup eligible")
        if context.group_kind is not KMeshGroupKind.AGENT_ROUND:
            raise ValueError("only agent-round contexts are lookup eligible")
        if context.state_kind is not KMeshStateKind.KV_CACHE:
            raise ValueError("only KV-cache state is lookup eligible")
        if len(prompt_token_ids) != prompt_token_count:
            raise ValueError("prompt token evidence does not match the admitted count")
        spans = tuple(
            KMeshRequestLookupSpan.from_segment(segment, prompt_token_ids)
            for segment in context.segments
            if segment.shareable
        )
        return cls(
            lookup_intent=lookup_intent,
            request_id=request_id,
            admission_generation=admission_generation,
            prompt_token_count=prompt_token_count,
            cache_domain=context.cache_domain,
            spans=spans,
        )

    @classmethod
    def from_admission(
        cls,
        *,
        request_id: str,
        admission_generation: int,
        prompt_token_ids: Sequence[int],
        prompt_token_count: int,
        context_result: KMeshRequestContextParseResult,
        lookup_intent: KMeshRequestLookupIntent,
        has_complete_runtime_identity: bool,
    ) -> KMeshRequestLookupDescriptor:
        """Freeze either admitted spans or the exact admission fallback."""
        if not isinstance(has_complete_runtime_identity, bool):
            raise TypeError("has_complete_runtime_identity must be a boolean")
        context = context_result.context
        fallback = context_result.fallback_reason
        if context_result.is_enabled and context is not None:
            if context.group_kind is not KMeshGroupKind.AGENT_ROUND:
                fallback = KMeshFallbackReason.GROUP_KIND_UNSUPPORTED
            elif context.state_kind is not KMeshStateKind.KV_CACHE:
                fallback = KMeshFallbackReason.STATE_KIND_UNSUPPORTED
            elif not has_complete_runtime_identity or context.cache_domain is None:
                fallback = KMeshFallbackReason.MISSING_COMPATIBILITY_EVIDENCE
            elif not any(segment.shareable for segment in context.segments):
                fallback = KMeshFallbackReason.NO_SHARED_SEGMENTS
            elif len(prompt_token_ids) != prompt_token_count:
                fallback = KMeshFallbackReason.PROMPT_TOKEN_LENGTH_MISMATCH
            else:
                return cls.from_context(
                    request_id=request_id,
                    admission_generation=admission_generation,
                    prompt_token_ids=prompt_token_ids,
                    prompt_token_count=prompt_token_count,
                    context=context,
                    lookup_intent=lookup_intent,
                )
        if fallback is None:
            fallback = KMeshFallbackReason.INTERNAL_ERROR
        return cls(
            lookup_intent=lookup_intent,
            request_id=request_id,
            admission_generation=admission_generation,
            prompt_token_count=prompt_token_count,
            cache_domain=context.cache_domain if context is not None else None,
            spans=(),
            admission_fallback=fallback,
        )

    @property
    def is_admitted(self) -> bool:
        return self.admission_fallback is None

    @property
    def permits_peer_lookup(self) -> bool:
        return (
            self.is_admitted
            and self.lookup_intent is KMeshRequestLookupIntent.SERIAL_PIC
        )

    def to_sidecar_payload(self) -> KMeshRequestSidecarValue:
        return (
            _KMESH_REQUEST_LOOKUP_WIRE_TAG,
            self.lookup_intent.value,
            self.request_id,
            self.admission_generation,
            self.prompt_token_count,
            self.cache_domain,
            (
                self.admission_fallback.value
                if self.admission_fallback is not None
                else None
            ),
            tuple(span.to_sidecar_payload() for span in self.spans),
        )

    @classmethod
    def from_sidecar_payload(cls, payload: object) -> KMeshRequestLookupDescriptor:
        fields = _wire_tuple(payload, "descriptor", length=8)
        (
            wire_tag,
            lookup_intent,
            request_id,
            admission_generation,
            prompt_token_count,
            cache_domain,
            admission_fallback,
            raw_spans,
        ) = fields
        if type(wire_tag) is not str:
            raise TypeError("descriptor wire tag must be a string")
        if wire_tag != _KMESH_REQUEST_LOOKUP_WIRE_TAG:
            raise ValueError("descriptor has an unsupported wire tag")
        if type(lookup_intent) is not str:
            raise TypeError("descriptor.lookup_intent must be a string")
        try:
            intent = KMeshRequestLookupIntent(lookup_intent)
        except ValueError as exc:
            raise ValueError("descriptor.lookup_intent has an unknown value") from exc
        if type(request_id) is not str:
            raise TypeError("descriptor.request_id must be a string")
        if not _is_strict_int(admission_generation):
            raise TypeError("descriptor.admission_generation must be an integer")
        if not _is_strict_int(prompt_token_count):
            raise TypeError("descriptor.prompt_token_count must be an integer")
        cache_domain = _wire_optional_string(cache_domain, "descriptor.cache_domain")
        if admission_fallback is None:
            fallback = None
        else:
            if type(admission_fallback) is not str:
                raise TypeError(
                    "descriptor.admission_fallback must be a string or null"
                )
            try:
                fallback = KMeshFallbackReason(admission_fallback)
            except ValueError as exc:
                raise ValueError(
                    "descriptor.admission_fallback has an unknown value"
                ) from exc
        raw_spans = _wire_tuple(raw_spans, "descriptor.spans")
        spans = tuple(
            KMeshRequestLookupSpan.from_sidecar_payload(span) for span in raw_spans
        )
        return cls(
            lookup_intent=intent,
            request_id=request_id,
            admission_generation=admission_generation,
            prompt_token_count=prompt_token_count,
            cache_domain=cache_domain,
            spans=spans,
            admission_fallback=fallback,
        )


@dataclass(frozen=True, slots=True, order=True)
class KMeshPICSpan:
    """One nonempty request-relative token interval used by PIC execution."""

    token_start: int
    token_end: int

    def __post_init__(self) -> None:
        for field in ("token_start", "token_end"):
            value = getattr(self, field)
            if not _is_strict_int(value) or value < 0:
                raise ValueError(f"{field} must be a nonnegative integer")
        if self.token_end <= self.token_start:
            raise ValueError("PIC spans must be nonempty")

    def contains(self, token_position: int) -> bool:
        return self.token_start <= token_position < self.token_end

    def contains_span(self, span: KMeshPICSpan) -> bool:
        return self.token_start <= span.token_start and span.token_end <= self.token_end

    def to_primitive(self) -> tuple[int, int]:
        return (self.token_start, self.token_end)


def _normalized_nonnegative_real(value: object, field: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{field} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{field} must be finite and nonnegative")
    return normalized


@dataclass(frozen=True, slots=True)
class KMeshPICTolerance:
    """Declared absolute and relative tolerances for PIC parity checks."""

    atol: float
    rtol: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "atol", _normalized_nonnegative_real(self.atol, "atol")
        )
        object.__setattr__(
            self, "rtol", _normalized_nonnegative_real(self.rtol, "rtol")
        )

    def to_primitive(self) -> dict[str, float]:
        return {"atol": self.atol, "rtol": self.rtol}


@dataclass(frozen=True, slots=True)
class KMeshPICOperationCounts:
    """Named counters that distinguish serial from collective PIC work."""

    serial_rope_calls: int = 0
    collective_rope_calls: int = 0
    serial_diff_passes: int = 0
    collective_diff_passes: int = 0
    requests_per_collective_pass: int = 0
    tokens_per_collective_pass: int = 0

    def __post_init__(self) -> None:
        for field in (
            "serial_rope_calls",
            "collective_rope_calls",
            "serial_diff_passes",
            "collective_diff_passes",
            "requests_per_collective_pass",
            "tokens_per_collective_pass",
        ):
            value = getattr(self, field)
            if not _is_strict_int(value) or value < 0:
                raise ValueError(f"{field} must be a nonnegative integer")

    def to_primitive(self) -> dict[str, int]:
        return {
            "serial_rope_calls": self.serial_rope_calls,
            "collective_rope_calls": self.collective_rope_calls,
            "serial_diff_passes": self.serial_diff_passes,
            "collective_diff_passes": self.collective_diff_passes,
            "requests_per_collective_pass": self.requests_per_collective_pass,
            "tokens_per_collective_pass": self.tokens_per_collective_pass,
        }


@dataclass(frozen=True, slots=True)
class KMeshPICExecutionSpec:
    """Tensor-free, request-owned parameters for one serial PIC recovery."""

    plan_id: str
    request_id: str
    admission_generation: int
    step_generation: int
    selected_spans: tuple[KMeshRequestLookupSpan, ...]
    cached_span: KMeshPICSpan
    check_layer: int
    recompute_ratio: float
    block_size: int
    declared_tolerance: KMeshPICTolerance

    def __post_init__(self) -> None:
        _validate_identifier(self.plan_id, "plan_id")
        _validate_identifier(self.request_id, "request_id")
        for field in ("admission_generation", "step_generation"):
            value = getattr(self, field)
            if not _is_strict_int(value) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if type(self.selected_spans) is not tuple:
            raise TypeError("selected_spans must be an immutable tuple")
        if not self.selected_spans:
            raise ValueError("selected_spans must not be empty")
        if not isinstance(self.cached_span, KMeshPICSpan):
            raise TypeError("cached_span must be KMeshPICSpan")
        if self.cached_span.token_start != 0:
            raise ValueError("serial PIC v0 requires a prompt-origin cached_span")
        previous: KMeshRequestLookupSpan | None = None
        seen_segment_ids: set[str] = set()
        for index, span in enumerate(self.selected_spans):
            if not isinstance(span, KMeshRequestLookupSpan):
                raise TypeError(
                    f"selected_spans[{index}] must be KMeshRequestLookupSpan"
                )
            if not span.shared_segment.shareable:
                raise ValueError("selected spans must carry shareable identities")
            segment_id = span.shared_segment.segment_id
            if segment_id in seen_segment_ids:
                raise ValueError("selected span IDs must be unique")
            seen_segment_ids.add(segment_id)
            if not self.cached_span.contains_span(
                KMeshPICSpan(span.token_start, span.token_end)
            ):
                raise ValueError("selected spans must fit within cached_span")
            if previous is not None:
                if span.token_start < previous.token_end:
                    raise ValueError(
                        "selected spans must be ordered and non-overlapping"
                    )
                if span.logical_order <= previous.logical_order:
                    raise ValueError("selected span logical order must increase")
            previous = span
        if not _is_strict_int(self.check_layer) or self.check_layer < 0:
            raise ValueError("check_layer must be a nonnegative integer")
        if not isinstance(self.recompute_ratio, Real) or isinstance(
            self.recompute_ratio, bool
        ):
            raise TypeError("recompute_ratio must be a real number")
        ratio = float(self.recompute_ratio)
        if not math.isfinite(ratio) or not 0.0 < ratio <= 1.0:
            raise ValueError("recompute_ratio must be finite and in (0, 1]")
        object.__setattr__(self, "recompute_ratio", ratio)
        if not _is_strict_int(self.block_size) or self.block_size <= 0:
            raise ValueError("block_size must be a positive integer")
        if not isinstance(self.declared_tolerance, KMeshPICTolerance):
            raise TypeError("declared_tolerance must be KMeshPICTolerance")

    def to_primitive(self) -> dict[str, object]:
        return {
            "plan_id": self.plan_id,
            "request_id": self.request_id,
            "admission_generation": self.admission_generation,
            "step_generation": self.step_generation,
            "selected_spans": [
                span.to_sidecar_payload() for span in self.selected_spans
            ],
            "cached_span": self.cached_span.to_primitive(),
            "check_layer": self.check_layer,
            "recompute_ratio": self.recompute_ratio,
            "block_size": self.block_size,
            "declared_tolerance": self.declared_tolerance.to_primitive(),
        }


@dataclass(frozen=True, slots=True)
class KMeshPICResult:
    """Immutable serial PIC evidence bound to the exact execution parameters."""

    execution_spec: KMeshPICExecutionSpec
    important_positions: tuple[int, ...]
    changed_request_blocks: tuple[int, ...]
    cache_hit_spans: tuple[KMeshPICSpan, ...]
    operation_counts: KMeshPICOperationCounts
    fallback_reason: KMeshFallbackReason | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.execution_spec, KMeshPICExecutionSpec):
            raise TypeError("execution_spec must be KMeshPICExecutionSpec")
        if type(self.important_positions) is not tuple:
            raise TypeError("important_positions must be an immutable tuple")
        if type(self.changed_request_blocks) is not tuple:
            raise TypeError("changed_request_blocks must be an immutable tuple")
        if type(self.cache_hit_spans) is not tuple:
            raise TypeError("cache_hit_spans must be an immutable tuple")
        for field in ("important_positions", "changed_request_blocks"):
            values = getattr(self, field)
            for value in values:
                if not _is_strict_int(value) or value < 0:
                    raise ValueError(f"{field} must contain nonnegative integer values")
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{field} must be unique and sorted")
        cached_span = self.execution_spec.cached_span
        if any(
            not cached_span.contains(position) for position in self.important_positions
        ):
            raise ValueError("important_positions must fit within cached_span")
        expected_blocks = tuple(
            sorted(
                {
                    position // self.execution_spec.block_size
                    for position in self.important_positions
                }
            )
        )
        if self.changed_request_blocks != expected_blocks:
            raise ValueError(
                "changed_request_blocks must be derived from important_positions"
            )
        selected_ranges = {
            (span.token_start, span.token_end)
            for span in self.execution_spec.selected_spans
        }
        previous: KMeshPICSpan | None = None
        for index, span in enumerate(self.cache_hit_spans):
            if not isinstance(span, KMeshPICSpan):
                raise TypeError(f"cache_hit_spans[{index}] must be KMeshPICSpan")
            if not cached_span.contains_span(span):
                raise ValueError("cache_hit_spans must fit within cached_span")
            if (span.token_start, span.token_end) not in selected_ranges:
                raise ValueError("cache_hit_spans must match selected span boundaries")
            if previous is not None and span.token_start < previous.token_end:
                raise ValueError("cache_hit_spans must be ordered and non-overlapping")
            previous = span
        if not isinstance(self.operation_counts, KMeshPICOperationCounts):
            raise TypeError("operation_counts must be KMeshPICOperationCounts")
        if self.fallback_reason is not None and not isinstance(
            self.fallback_reason, KMeshFallbackReason
        ):
            raise TypeError("fallback_reason must be KMeshFallbackReason or null")
        if self.fallback_reason is None:
            if not self.cache_hit_spans:
                raise ValueError("a successful PIC result requires cache-hit spans")
        elif (
            self.important_positions
            or self.changed_request_blocks
            or self.cache_hit_spans
        ):
            raise ValueError("a fallback PIC result cannot claim recovery evidence")

    def to_primitive(self) -> dict[str, object]:
        return {
            "execution_spec": self.execution_spec.to_primitive(),
            "important_positions": list(self.important_positions),
            "changed_request_blocks": list(self.changed_request_blocks),
            "cache_hit_spans": [span.to_primitive() for span in self.cache_hit_spans],
            "operation_counts": self.operation_counts.to_primitive(),
            "fallback_reason": (
                self.fallback_reason.value if self.fallback_reason is not None else None
            ),
        }


@runtime_checkable
class PositionIndependentReuseBackend(Protocol):
    """Backend seam for one request-owned serial PIC recovery."""

    def recover_request_serial(
        self, request: object, execution_spec: KMeshPICExecutionSpec
    ) -> KMeshPICResult:
        """Recover one request without retaining cross-request state."""
        ...
