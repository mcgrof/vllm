# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from vllm.v1.kmesh.types import (
    KMeshFallbackReason,
    KMeshRequestContext,
    KMeshSegment,
    KMeshSegmentKind,
    _is_strict_int,
    _validate_identifier,
    _validate_optional_identifier,
)

_TOKEN_SLICE_DIGEST_PREFIX = "sha256-v1:"


def _validate_round_id(value: object) -> None:
    if _is_strict_int(value):
        if value < 0:
            raise ValueError("round_id must be nonnegative")
        return
    if isinstance(value, str):
        _validate_identifier(value, "round_id")
        return
    raise TypeError("round_id must be a nonnegative integer or stable string")


def _validate_positive_int(value: object, field: str) -> None:
    if not _is_strict_int(value) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")


def _validate_span(
    value: object,
    field: str,
    active_prompt_length: int,
    *,
    allow_empty: bool,
) -> None:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{field} must be a two-integer tuple")
    start, end = value
    if not _is_strict_int(start) or not _is_strict_int(end):
        raise TypeError(f"{field} must contain strict integers")
    if start < 0 or end < start or end > active_prompt_length:
        raise ValueError(f"{field} must be ordered and within the active prompt")
    if not allow_empty and start == end:
        raise ValueError(f"{field} must be nonempty")


def kmesh_token_slice_digest(token_ids: Sequence[int]) -> str:
    """Return a versioned deterministic digest of an exact token-ID sequence."""
    validated = []
    for index, token_id in enumerate(token_ids):
        if not _is_strict_int(token_id) or token_id < 0:
            raise ValueError(f"token_ids[{index}] must be a nonnegative integer")
        validated.append(token_id)
    payload = json.dumps(validated, separators=(",", ":"), ensure_ascii=True).encode()
    digest = hashlib.sha256(b"kmesh-token-slice-v1\0" + payload).hexdigest()
    return f"{_TOKEN_SLICE_DIGEST_PREFIX}{digest}"


@dataclass(frozen=True, slots=True)
class KMeshCompatibilityKey:
    """Complete runtime evidence required for safe same-step grouping.

    ``None`` means an observed absence for adapter and salt identity. An absent
    cache domain is incomplete evidence and is never admitted by the planner.
    RoPE configuration is a caller-supplied canonical identity or fingerprint.
    """

    execution_id: str
    group_id: str
    round_id: int | str
    model_id: str
    model_revision: str
    tokenizer_id_or_revision: str
    dtype: str
    kv_cache_dtype: str
    kv_layout: str
    attention_kind: str
    rope_configuration: str
    attention_backend: str
    tensor_parallel_size: int
    pipeline_parallel_size: int
    lora_adapter_identity: str | None
    cache_salt: str | None
    cache_domain: str | None
    active_prompt_length: int
    cached_span: tuple[int, int]
    prefill_chunk_span: tuple[int, int]

    def __post_init__(self) -> None:
        for field in (
            "execution_id",
            "group_id",
            "model_id",
            "model_revision",
            "tokenizer_id_or_revision",
            "dtype",
            "kv_cache_dtype",
            "kv_layout",
            "attention_kind",
            "rope_configuration",
            "attention_backend",
        ):
            _validate_identifier(getattr(self, field), field)
        _validate_round_id(self.round_id)
        _validate_positive_int(self.tensor_parallel_size, "tensor_parallel_size")
        _validate_positive_int(self.pipeline_parallel_size, "pipeline_parallel_size")
        _validate_optional_identifier(
            self.lora_adapter_identity, "lora_adapter_identity"
        )
        _validate_optional_identifier(self.cache_salt, "cache_salt")
        _validate_optional_identifier(self.cache_domain, "cache_domain")
        _validate_positive_int(self.active_prompt_length, "active_prompt_length")
        _validate_span(
            self.cached_span,
            "cached_span",
            self.active_prompt_length,
            allow_empty=True,
        )
        _validate_span(
            self.prefill_chunk_span,
            "prefill_chunk_span",
            self.active_prompt_length,
            allow_empty=False,
        )
        if self.cached_span[0] != 0:
            raise ValueError("cached_span must begin at the prompt origin")
        if self.prefill_chunk_span[0] != self.cached_span[1]:
            raise ValueError("prefill_chunk_span must follow cached_span contiguously")

    @property
    def has_complete_evidence(self) -> bool:
        """Whether the key contains an explicit isolation domain."""
        return self.cache_domain is not None

    def to_primitive(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "group_id": self.group_id,
            "round_id": self.round_id,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer_id_or_revision": self.tokenizer_id_or_revision,
            "dtype": self.dtype,
            "kv_cache_dtype": self.kv_cache_dtype,
            "kv_layout": self.kv_layout,
            "attention_kind": self.attention_kind,
            "rope_configuration": self.rope_configuration,
            "attention_backend": self.attention_backend,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "lora_adapter_identity": self.lora_adapter_identity,
            "cache_salt": self.cache_salt,
            "cache_domain": self.cache_domain,
            "active_prompt_length": self.active_prompt_length,
            "cached_span": list(self.cached_span),
            "prefill_chunk_span": list(self.prefill_chunk_span),
        }


_COMPATIBILITY_FIELDS: tuple[tuple[str, KMeshFallbackReason], ...] = (
    ("execution_id", KMeshFallbackReason.EXECUTION_ID_MISMATCH),
    ("group_id", KMeshFallbackReason.GROUP_ID_MISMATCH),
    ("round_id", KMeshFallbackReason.ROUND_MISMATCH),
    ("model_id", KMeshFallbackReason.MODEL_MISMATCH),
    ("model_revision", KMeshFallbackReason.MODEL_MISMATCH),
    ("tokenizer_id_or_revision", KMeshFallbackReason.TOKENIZER_MISMATCH),
    ("dtype", KMeshFallbackReason.DTYPE_MISMATCH),
    ("kv_cache_dtype", KMeshFallbackReason.KV_CACHE_DTYPE_MISMATCH),
    ("kv_layout", KMeshFallbackReason.KV_LAYOUT_MISMATCH),
    ("attention_kind", KMeshFallbackReason.ATTENTION_KIND_MISMATCH),
    ("rope_configuration", KMeshFallbackReason.ROPE_CONFIGURATION_MISMATCH),
    ("attention_backend", KMeshFallbackReason.ATTENTION_BACKEND_MISMATCH),
    ("tensor_parallel_size", KMeshFallbackReason.TENSOR_PARALLEL_MISMATCH),
    ("pipeline_parallel_size", KMeshFallbackReason.PIPELINE_PARALLEL_MISMATCH),
    ("lora_adapter_identity", KMeshFallbackReason.LORA_MISMATCH),
    ("cache_salt", KMeshFallbackReason.CACHE_SALT_MISMATCH),
    ("cache_domain", KMeshFallbackReason.CACHE_DOMAIN_MISMATCH),
    ("active_prompt_length", KMeshFallbackReason.ACTIVE_LENGTH_MISMATCH),
    ("cached_span", KMeshFallbackReason.CACHED_SPAN_MISMATCH),
    ("prefill_chunk_span", KMeshFallbackReason.PREFILL_CHUNK_MISMATCH),
)

_REASON_PRECEDENCE = {
    reason: index
    for index, (_, reason) in reversed(tuple(enumerate(_COMPATIBILITY_FIELDS)))
}


def compatibility_mismatch_reason(
    left: KMeshCompatibilityKey, right: KMeshCompatibilityKey
) -> KMeshFallbackReason | None:
    """Return the first mismatch according to the stable safety precedence."""
    for field, reason in _COMPATIBILITY_FIELDS:
        if getattr(left, field) != getattr(right, field):
            return reason
    return None


def compatibility_reason_precedence(reason: KMeshFallbackReason) -> int:
    """Return the stable ordering index for a compatibility failure."""
    return _REASON_PRECEDENCE.get(reason, len(_COMPATIBILITY_FIELDS))


def context_key_mismatch_reason(
    context: KMeshRequestContext, key: KMeshCompatibilityKey
) -> KMeshFallbackReason | None:
    """Verify request-declared grouping fields against runtime evidence."""
    comparisons = (
        (
            context.execution_id,
            key.execution_id,
            KMeshFallbackReason.EXECUTION_ID_MISMATCH,
        ),
        (context.group_id, key.group_id, KMeshFallbackReason.GROUP_ID_MISMATCH),
        (context.round_id, key.round_id, KMeshFallbackReason.ROUND_MISMATCH),
        (
            context.cache_domain,
            key.cache_domain,
            KMeshFallbackReason.CACHE_DOMAIN_MISMATCH,
        ),
    )
    for declared, observed, reason in comparisons:
        if declared != observed:
            return reason
    if any(
        segment.token_end > key.active_prompt_length for segment in context.segments
    ):
        return KMeshFallbackReason.ACTIVE_LENGTH_MISMATCH
    return None


@dataclass(frozen=True, slots=True)
class KMeshSharedSegmentIdentity:
    """Offset-independent, content-bound identity for one shareable segment.

    The digest is observer evidence only; it does not authorize off-position KV
    aliasing without the later PIC correctness gate.
    """

    segment_id: str
    kind: KMeshSegmentKind
    token_length: int
    source_id: str | None
    lineage_id: str | None
    shareable: bool
    content_digest: str

    def __post_init__(self) -> None:
        _validate_identifier(self.segment_id, "segment_id")
        if not isinstance(self.kind, KMeshSegmentKind):
            raise TypeError("kind must be KMeshSegmentKind")
        _validate_positive_int(self.token_length, "token_length")
        _validate_optional_identifier(self.source_id, "source_id")
        _validate_optional_identifier(self.lineage_id, "lineage_id")
        if self.shareable is not True:
            raise ValueError("shared segment identity requires shareable=true")
        if (
            not isinstance(self.content_digest, str)
            or not self.content_digest.startswith(_TOKEN_SLICE_DIGEST_PREFIX)
            or len(self.content_digest) != len(_TOKEN_SLICE_DIGEST_PREFIX) + 64
        ):
            raise ValueError("content_digest must be a versioned SHA-256 digest")
        digest_hex = self.content_digest[len(_TOKEN_SLICE_DIGEST_PREFIX) :]
        if any(char not in "0123456789abcdef" for char in digest_hex):
            raise ValueError("content_digest must use lowercase hexadecimal")

    @classmethod
    def from_segment(
        cls, segment: KMeshSegment, prompt_token_ids: Sequence[int]
    ) -> KMeshSharedSegmentIdentity:
        if not segment.shareable:
            raise ValueError("private segments have no shared identity")
        if segment.token_end > len(prompt_token_ids):
            raise ValueError("segment range exceeds prompt-token evidence")
        return cls(
            segment_id=segment.segment_id,
            kind=segment.kind,
            token_length=segment.token_end - segment.token_start,
            source_id=segment.source_id,
            lineage_id=segment.lineage_id,
            shareable=True,
            content_digest=kmesh_token_slice_digest(
                prompt_token_ids[segment.token_start : segment.token_end]
            ),
        )

    def declaration_key(self) -> tuple[str, str, int, str, str, bool]:
        """Return the caller-declared identity fields without content evidence."""
        return (
            self.segment_id,
            self.kind.value,
            self.token_length,
            self.source_id or "",
            self.lineage_id or "",
            self.shareable,
        )

    def sort_key(self) -> tuple[str, str, int, str, str, bool, str]:
        return (
            *self.declaration_key(),
            self.content_digest,
        )

    def to_primitive(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "kind": self.kind.value,
            "token_length": self.token_length,
            "source_id": self.source_id,
            "lineage_id": self.lineage_id,
            "shareable": self.shareable,
            "content_digest": self.content_digest,
        }


@dataclass(frozen=True, slots=True)
class KMeshReuseGroup:
    """One prospective same-step group with exact compatibility evidence."""

    reuse_group_id: str
    compatibility_key: KMeshCompatibilityKey
    request_ids: tuple[str, ...]
    shared_segments: tuple[KMeshSharedSegmentIdentity, ...]
    master_candidate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_identifier(self.reuse_group_id, "reuse_group_id")
        if not isinstance(self.compatibility_key, KMeshCompatibilityKey):
            raise TypeError("compatibility_key must be KMeshCompatibilityKey")
        if not self.compatibility_key.has_complete_evidence:
            raise ValueError("reuse groups require a complete compatibility key")
        if len(self.request_ids) < 2:
            raise ValueError("reuse groups must not be singletons")
        if self.request_ids != tuple(sorted(set(self.request_ids))):
            raise ValueError("request_ids must be unique and sorted")
        for request_id in self.request_ids:
            _validate_identifier(request_id, "request_ids")
        if not self.shared_segments:
            raise ValueError("reuse groups require at least one shared segment")
        expected_segments = tuple(
            sorted(set(self.shared_segments), key=lambda segment: segment.sort_key())
        )
        if self.shared_segments != expected_segments:
            raise ValueError("shared_segments must be unique and canonically sorted")
        if self.master_candidate_ids != tuple(sorted(set(self.master_candidate_ids))):
            raise ValueError("master_candidate_ids must be unique and sorted")
        if not self.master_candidate_ids or not set(self.master_candidate_ids) <= set(
            self.request_ids
        ):
            raise ValueError("master candidates must be nonempty group members")

    def to_primitive(self) -> dict[str, Any]:
        return {
            "reuse_group_id": self.reuse_group_id,
            "compatibility_key": self.compatibility_key.to_primitive(),
            "request_ids": list(self.request_ids),
            "shared_segments": [
                segment.to_primitive() for segment in self.shared_segments
            ],
            "master_candidate_ids": list(self.master_candidate_ids),
        }
