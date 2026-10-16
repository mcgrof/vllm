# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import hmac
import json
import math
import struct
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, cast

import torch

_MAGIC = b"KMSHMM01"
_SCHEMA = "vllm.kmesh.master_mirror"
_VERSION = 1
_ALIGNMENT = 8
_CHECKSUM_NBYTES = hashlib.sha256().digest_size
_FILE_HEADER = struct.Struct("<8sHHIQQ")
_MASTER_HEADER = struct.Struct("<4sH2xQQ")
_MIRROR_HEADER = struct.Struct("<4sHBBII")
_MASTER_TAG = b"MSTR"
_MIRROR_TAG = b"MIRR"
_INDEX_SHARED = 1
_INDEX_SEPARATE = 2

_DTYPE_TO_NAME = {torch.float16: "float16", torch.bfloat16: "bfloat16"}
_NAME_TO_DTYPE = {name: dtype for dtype, name in _DTYPE_TO_NAME.items()}


class MasterMirrorError(ValueError):
    """Base class for Master-Mirror codec errors."""


class MasterMirrorCorruptionError(MasterMirrorError):
    """Raised when an encoded family is malformed or corrupted."""


class MasterMirrorIdentityError(MasterMirrorError):
    """Raised when the encoded family does not match the requested family."""


class MasterMirrorRequestNotFoundError(MasterMirrorError):
    """Raised when an encoded family does not contain a requested member."""


@dataclass(frozen=True)
class MasterMirrorRequest:
    """One dense CPU-serializable K/V state in a related request family."""

    request_id: str
    key: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True)
class TensorInfo:
    """Dense tensor metadata recorded in the representation."""

    dtype: torch.dtype
    shape: tuple[int, ...]
    block_axis: int
    block_count: int
    block_nbytes: int
    nbytes: int


@dataclass(frozen=True)
class MirrorDiffInfo:
    """Block-diff metadata for one non-master request."""

    request_id: str
    index_encoding: Literal["shared", "separate"]
    key_changed_blocks: int
    value_changed_blocks: int


@dataclass(frozen=True)
class MasterMirrorCandidateMetrics:
    """Master-selection metrics, ordered from semantic cost to wire size."""

    master_request_id: str
    changed_tensor_nbytes: int
    changed_block_count: int
    encoded_nbytes: int

    @property
    def score(self) -> tuple[int, int, int, str]:
        return (
            self.changed_tensor_nbytes,
            self.changed_block_count,
            self.encoded_nbytes,
            self.master_request_id,
        )


@dataclass(frozen=True)
class MasterMirrorByteAccounting:
    """Exact serialized-byte components for one encoded family."""

    header_nbytes: int
    metadata_nbytes: int
    metadata_padding_nbytes: int
    body_nbytes: int
    record_header_nbytes: int
    identity_nbytes: int
    index_nbytes: int
    tensor_nbytes: int
    body_padding_nbytes: int
    checksum_nbytes: int

    @property
    def alignment_nbytes(self) -> int:
        return self.metadata_padding_nbytes + self.body_padding_nbytes

    @property
    def total_nbytes(self) -> int:
        return (
            self.header_nbytes
            + self.metadata_nbytes
            + self.metadata_padding_nbytes
            + self.body_nbytes
            + self.checksum_nbytes
        )


@dataclass(frozen=True)
class MasterMirrorInfo:
    """Validated identity, tensor metadata, and byte accounting."""

    family_id: str
    request_ids: tuple[str, ...]
    master_request_id: str
    mode: Literal["exact", "tolerance"]
    atol: float | None
    rtol: float | None
    key: TensorInfo
    value: TensorInfo
    mirrors: tuple[MirrorDiffInfo, ...]
    dense_family_nbytes: int
    accounting: MasterMirrorByteAccounting

    @property
    def encoded_nbytes(self) -> int:
        return self.accounting.total_nbytes


@dataclass(frozen=True)
class _NormalizedRequest:
    request_id: str
    key: torch.Tensor
    value: torch.Tensor
    key_bytes: bytes
    value_bytes: bytes


@dataclass(frozen=True)
class _ParsedMirror:
    request_id: str
    key_indices: tuple[int, ...]
    value_indices: tuple[int, ...]
    key_blocks: bytes
    value_blocks: bytes


@dataclass(frozen=True)
class _ParsedPayload:
    info: MasterMirrorInfo
    master_key: bytes
    master_value: bytes
    mirrors: tuple[_ParsedMirror, ...]


@dataclass(frozen=True)
class _EncodedCandidate:
    data: bytes
    metrics: MasterMirrorCandidateMetrics


@dataclass(frozen=True)
class _ValidatedMetadata:
    family_id: str
    request_ids: tuple[str, ...]
    master_request_id: str
    mode: Literal["exact", "tolerance"]
    atol: float | None
    rtol: float | None
    key: TensorInfo
    value: TensorInfo
    mirrors: tuple[MirrorDiffInfo, ...]


class _Writer:
    def __init__(self) -> None:
        self.data = bytearray()

    def write(self, data: bytes) -> None:
        self.data.extend(data)

    def align(self) -> None:
        self.data.extend(b"\x00" * _padding_nbytes(len(self.data)))


class _Reader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.position = 0

    def take(self, size: int, label: str) -> bytes:
        if size < 0 or self.position + size > len(self.data):
            raise MasterMirrorCorruptionError(f"missing or truncated {label}")
        start = self.position
        self.position += size
        return self.data[start : self.position]

    def unpack(self, layout: struct.Struct, label: str) -> tuple[object, ...]:
        return layout.unpack(self.take(layout.size, label))

    def align(self, label: str) -> None:
        padding = self.take(_padding_nbytes(self.position), f"{label} padding")
        if any(padding):
            raise MasterMirrorCorruptionError(f"nonzero {label} padding")

    def finish(self) -> None:
        if self.position != len(self.data):
            raise MasterMirrorCorruptionError(
                f"unexpected trailing body bytes: {len(self.data) - self.position}"
            )


class MasterMirrorCodec:
    """Deterministic offline Master-Mirror block replacement codec.

    The leading dimension is the block axis. Changed blocks contain complete
    replacement bytes rather than an arithmetic delta. Offline v0 supports only
    ``torch.float16`` and ``torch.bfloat16``. FP8 is not yet supported. Encoding
    is CPU-only and favors a transparent correctness oracle over throughput.
    """

    @classmethod
    def encode(
        cls,
        family_id: str,
        requests: Iterable[MasterMirrorRequest],
        *,
        mode: Literal["exact", "tolerance"] = "exact",
        atol: float | None = None,
        rtol: float | None = None,
    ) -> bytes:
        """Encode a family using semantic diff cost, then wire-size tie breakers.

        Candidates are ordered by changed tensor bytes, changed K/V block count,
        complete encoded bytes, and finally request identity.

        Args:
            family_id: Stable identity shared by all family members.
            requests: Dense K/V tensors, with blocks on dimension zero.
            mode: Bit-exact or tolerance-based changed-block detection.
            atol: Required absolute tolerance in tolerance mode.
            rtol: Required relative tolerance in tolerance mode.

        Returns:
            A deterministic, checksummed binary representation.

        Raises:
            ValueError: If identities, tensors, or tolerances are invalid.

        """
        prepared, key_info, value_info, mode, atol, rtol = _prepare(
            family_id, requests, mode, atol, rtol
        )
        candidates = (
            _encode_candidate(
                family_id,
                prepared,
                key_info,
                value_info,
                master_index,
                mode,
                atol,
                rtol,
            )
            for master_index in range(len(prepared))
        )
        selected = min(candidates, key=lambda candidate: candidate.metrics.score)
        _parse(selected.data)
        return selected.data

    @classmethod
    def candidate_metrics(
        cls,
        family_id: str,
        requests: Iterable[MasterMirrorRequest],
        *,
        mode: Literal["exact", "tolerance"] = "exact",
        atol: float | None = None,
        rtol: float | None = None,
    ) -> tuple[MasterMirrorCandidateMetrics, ...]:
        """Return independently testable metrics for every possible master."""
        prepared, key_info, value_info, mode, atol, rtol = _prepare(
            family_id, requests, mode, atol, rtol
        )
        return tuple(
            _encode_candidate(
                family_id,
                prepared,
                key_info,
                value_info,
                master_index,
                mode,
                atol,
                rtol,
            ).metrics
            for master_index in range(len(prepared))
        )

    @classmethod
    def candidate_encoded_nbytes(
        cls,
        family_id: str,
        requests: Iterable[MasterMirrorRequest],
        *,
        mode: Literal["exact", "tolerance"] = "exact",
        atol: float | None = None,
        rtol: float | None = None,
    ) -> dict[str, int]:
        """Measure complete serialized bytes for every possible master."""
        return {
            metrics.master_request_id: metrics.encoded_nbytes
            for metrics in cls.candidate_metrics(
                family_id,
                requests,
                mode=mode,
                atol=atol,
                rtol=rtol,
            )
        }

    @classmethod
    def inspect(cls, data: bytes, *, family_id: str | None = None) -> MasterMirrorInfo:
        """Validate an encoding and return its recorded metadata and accounting."""
        parsed = _parse(data)
        if family_id is not None and parsed.info.family_id != family_id:
            raise MasterMirrorIdentityError(
                "family identity mismatch: "
                f"expected {family_id!r}, encoded {parsed.info.family_id!r}"
            )
        return parsed.info

    @classmethod
    def decode(
        cls,
        data: bytes,
        *,
        family_id: str,
        request_id: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Restore one family member as dense contiguous CPU tensors."""
        parsed = _parse(data)
        info = parsed.info
        if info.family_id != family_id:
            raise MasterMirrorIdentityError(
                "family identity mismatch: "
                f"expected {family_id!r}, encoded {info.family_id!r}"
            )
        if request_id not in info.request_ids:
            raise MasterMirrorRequestNotFoundError(
                f"request {request_id!r} is missing from family {family_id!r}"
            )

        key = _tensor_from_bytes(parsed.master_key, info.key)
        value = _tensor_from_bytes(parsed.master_value, info.value)
        if request_id == info.master_request_id:
            return key, value

        mirror = next(
            (mirror for mirror in parsed.mirrors if mirror.request_id == request_id),
            None,
        )
        if mirror is None:
            raise MasterMirrorCorruptionError(
                f"missing diff record for request {request_id!r}"
            )
        _apply_blocks(key, mirror.key_indices, mirror.key_blocks, info.key)
        _apply_blocks(value, mirror.value_indices, mirror.value_blocks, info.value)
        return key, value


def _prepare(
    family_id: str,
    requests: Iterable[MasterMirrorRequest],
    mode: Literal["exact", "tolerance"],
    atol: float | None,
    rtol: float | None,
) -> tuple[
    tuple[_NormalizedRequest, ...],
    TensorInfo,
    TensorInfo,
    Literal["exact", "tolerance"],
    float | None,
    float | None,
]:
    _validate_identity(family_id, "family_id")
    if sys.byteorder != "little":
        raise RuntimeError(
            "Master-Mirror encoding currently requires little-endian CPU"
        )
    mode, atol, rtol = _validated_mode_and_tolerances(
        mode, atol, rtol, corruption=False
    )

    normalized = tuple(
        sorted((_normalize(r) for r in requests), key=lambda r: r.request_id)
    )
    if not normalized:
        raise ValueError("at least one request is required")
    request_ids = [request.request_id for request in normalized]
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("request_id values must be unique within a family")

    key_info = _tensor_info(normalized[0].key)
    value_info = _tensor_info(normalized[0].value)
    if key_info.block_count != value_info.block_count:
        raise ValueError("key and value tensors must have the same block count")
    for request in normalized[1:]:
        if _tensor_info(request.key) != key_info:
            raise ValueError("all key tensors must have identical shape and dtype")
        if _tensor_info(request.value) != value_info:
            raise ValueError("all value tensors must have identical shape and dtype")
    if mode == "tolerance" and (
        not key_info.dtype.is_floating_point or not value_info.dtype.is_floating_point
    ):
        raise ValueError("tolerance mode requires floating-point key and value tensors")
    return normalized, key_info, value_info, mode, atol, rtol


def _normalize(request: MasterMirrorRequest) -> _NormalizedRequest:
    if not isinstance(request, MasterMirrorRequest):
        raise TypeError("requests must contain MasterMirrorRequest values")
    _validate_identity(request.request_id, "request_id")
    key = _normalize_tensor(request.key, "key")
    value = _normalize_tensor(request.value, "value")
    return _NormalizedRequest(
        request_id=request.request_id,
        key=key,
        value=value,
        key_bytes=_tensor_bytes(key),
        value_bytes=_tensor_bytes(value),
    )


def _normalize_tensor(tensor: torch.Tensor, label: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{label} must be a torch.Tensor")
    if tensor.layout != torch.strided:
        raise ValueError(f"{label} must use strided dense layout")
    if tensor.dtype not in _DTYPE_TO_NAME:
        raise ValueError(
            f"unsupported {label} dtype {tensor.dtype}; offline v0 supports only "
            "torch.float16 and torch.bfloat16, and does not yet support FP8"
        )
    _validated_shape(tuple(tensor.shape), label, corruption=False)
    return tensor.detach().to(device="cpu").contiguous().clone()


def _tensor_info(tensor: torch.Tensor) -> TensorInfo:
    block_count = tensor.shape[0]
    nbytes = tensor.numel() * tensor.element_size()
    return TensorInfo(
        dtype=tensor.dtype,
        shape=tuple(tensor.shape),
        block_axis=0,
        block_count=block_count,
        block_nbytes=nbytes // block_count,
        nbytes=nbytes,
    )


def _encode_candidate(
    family_id: str,
    requests: tuple[_NormalizedRequest, ...],
    key_info: TensorInfo,
    value_info: TensorInfo,
    master_index: int,
    mode: Literal["exact", "tolerance"],
    atol: float | None,
    rtol: float | None,
) -> _EncodedCandidate:
    master = requests[master_index]
    mirrors: list[tuple[_NormalizedRequest, tuple[int, ...], tuple[int, ...]]] = []
    mirror_metadata: list[dict[str, object]] = []
    for index, request in enumerate(requests):
        if index == master_index:
            continue
        key_indices = _changed_blocks(
            request.key,
            master.key,
            request.key_bytes,
            master.key_bytes,
            key_info,
            mode,
            atol,
            rtol,
        )
        value_indices = _changed_blocks(
            request.value,
            master.value,
            request.value_bytes,
            master.value_bytes,
            value_info,
            mode,
            atol,
            rtol,
        )
        index_encoding = "shared" if key_indices == value_indices else "separate"
        mirrors.append((request, key_indices, value_indices))
        mirror_metadata.append(
            {
                "index_encoding": index_encoding,
                "key_changed_blocks": len(key_indices),
                "request_id": request.request_id,
                "value_changed_blocks": len(value_indices),
            }
        )

    body = _encode_body(master, mirrors, key_info, value_info)
    metadata = {
        "atol": atol,
        "byte_order": "little",
        "family_id": family_id,
        "format_version": _VERSION,
        "key": _tensor_info_dict(key_info),
        "master_request_id": master.request_id,
        "mirrors": mirror_metadata,
        "mode": mode,
        "request_ids": [request.request_id for request in requests],
        "rtol": rtol,
        "schema": _SCHEMA,
        "value": _tensor_info_dict(value_info),
    }
    metadata_bytes = json.dumps(
        metadata,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    header = _FILE_HEADER.pack(
        _MAGIC,
        _VERSION,
        _ALIGNMENT,
        0,
        len(metadata_bytes),
        len(body),
    )
    metadata_padding = b"\x00" * _padding_nbytes(len(header) + len(metadata_bytes))
    checksummed = header + metadata_bytes + metadata_padding + body
    data = checksummed + hashlib.sha256(checksummed).digest()
    changed_block_count = sum(
        len(key_indices) + len(value_indices)
        for _, key_indices, value_indices in mirrors
    )
    changed_tensor_nbytes = sum(
        len(key_indices) * key_info.block_nbytes
        + len(value_indices) * value_info.block_nbytes
        for _, key_indices, value_indices in mirrors
    )
    return _EncodedCandidate(
        data,
        MasterMirrorCandidateMetrics(
            master_request_id=master.request_id,
            changed_tensor_nbytes=changed_tensor_nbytes,
            changed_block_count=changed_block_count,
            encoded_nbytes=len(data),
        ),
    )


def _encode_body(
    master: _NormalizedRequest,
    mirrors: list[tuple[_NormalizedRequest, tuple[int, ...], tuple[int, ...]]],
    key_info: TensorInfo,
    value_info: TensorInfo,
) -> bytes:
    writer = _Writer()
    master_id = master.request_id.encode("utf-8")
    writer.write(
        _MASTER_HEADER.pack(
            _MASTER_TAG,
            len(master_id),
            len(master.key_bytes),
            len(master.value_bytes),
        )
    )
    writer.write(master_id)
    writer.align()
    writer.write(master.key_bytes)
    writer.align()
    writer.write(master.value_bytes)
    writer.align()

    for request, key_indices, value_indices in mirrors:
        request_id = request.request_id.encode("utf-8")
        shared = key_indices == value_indices
        writer.write(
            _MIRROR_HEADER.pack(
                _MIRROR_TAG,
                len(request_id),
                _INDEX_SHARED if shared else _INDEX_SEPARATE,
                0,
                len(key_indices),
                len(value_indices),
            )
        )
        writer.write(request_id)
        writer.align()
        writer.write(_encode_indices(key_indices))
        writer.align()
        if not shared:
            writer.write(_encode_indices(value_indices))
            writer.align()
        writer.write(_selected_blocks(request.key_bytes, key_indices, key_info))
        writer.align()
        writer.write(_selected_blocks(request.value_bytes, value_indices, value_info))
        writer.align()
    return bytes(writer.data)


def _changed_blocks(
    mirror: torch.Tensor,
    master: torch.Tensor,
    mirror_bytes: bytes,
    master_bytes: bytes,
    info: TensorInfo,
    mode: Literal["exact", "tolerance"],
    atol: float | None,
    rtol: float | None,
) -> tuple[int, ...]:
    changed: list[int] = []
    for block_index in range(info.block_count):
        if mode == "exact":
            start = block_index * info.block_nbytes
            end = start + info.block_nbytes
            same = mirror_bytes[start:end] == master_bytes[start:end]
        else:
            assert atol is not None and rtol is not None
            same = torch.allclose(
                mirror[block_index],
                master[block_index],
                atol=atol,
                rtol=rtol,
                equal_nan=True,
            )
        if not same:
            changed.append(block_index)
    return tuple(changed)


def _selected_blocks(data: bytes, indices: tuple[int, ...], info: TensorInfo) -> bytes:
    return b"".join(
        data[index * info.block_nbytes : (index + 1) * info.block_nbytes]
        for index in indices
    )


def _encode_indices(indices: tuple[int, ...]) -> bytes:
    if not indices:
        return b""
    return struct.pack(f"<{len(indices)}I", *indices)


def _parse(data: bytes) -> _ParsedPayload:
    if not isinstance(data, bytes):
        raise TypeError("encoded data must be bytes")
    if len(data) < _FILE_HEADER.size + _CHECKSUM_NBYTES:
        raise MasterMirrorCorruptionError("missing or truncated file header")
    magic, version, alignment, flags, metadata_nbytes, body_nbytes = (
        _FILE_HEADER.unpack_from(data)
    )
    if magic != _MAGIC:
        raise MasterMirrorCorruptionError("invalid Master-Mirror magic")
    if version != _VERSION:
        raise MasterMirrorCorruptionError(f"unsupported format version: {version}")
    if alignment != _ALIGNMENT or flags != 0:
        raise MasterMirrorCorruptionError("invalid alignment or header flags")

    metadata_start = _FILE_HEADER.size
    metadata_end = metadata_start + metadata_nbytes
    body_start = metadata_end + _padding_nbytes(metadata_end)
    body_end = body_start + body_nbytes
    expected_nbytes = body_end + _CHECKSUM_NBYTES
    if len(data) < expected_nbytes:
        raise MasterMirrorCorruptionError(
            f"payload is truncated: expected {expected_nbytes} bytes, got {len(data)}"
        )
    if len(data) > expected_nbytes:
        raise MasterMirrorCorruptionError(
            f"payload has {len(data) - expected_nbytes} trailing bytes"
        )
    expected_checksum = hashlib.sha256(data[:body_end]).digest()
    if not hmac.compare_digest(expected_checksum, data[body_end:]):
        raise MasterMirrorCorruptionError("payload checksum mismatch")
    if any(data[metadata_end:body_start]):
        raise MasterMirrorCorruptionError("nonzero metadata padding")

    try:
        metadata = json.loads(data[metadata_start:metadata_end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise MasterMirrorCorruptionError("invalid metadata JSON") from error
    try:
        canonical_metadata = json.dumps(
            metadata,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as error:
        raise MasterMirrorCorruptionError("invalid metadata values") from error
    if canonical_metadata != data[metadata_start:metadata_end]:
        raise MasterMirrorCorruptionError("metadata is not canonical")

    info_metadata = _validate_metadata(metadata)
    parsed_body = _parse_body(data[body_start:body_end], info_metadata)
    body_components = _body_component_nbytes(info_metadata)
    body_padding_nbytes = body_nbytes - sum(body_components)
    if body_padding_nbytes < 0:
        raise MasterMirrorCorruptionError("body byte accounting underflow")
    accounting = MasterMirrorByteAccounting(
        header_nbytes=_FILE_HEADER.size,
        metadata_nbytes=metadata_nbytes,
        metadata_padding_nbytes=body_start - metadata_end,
        body_nbytes=body_nbytes,
        record_header_nbytes=body_components[0],
        identity_nbytes=body_components[1],
        index_nbytes=body_components[2],
        tensor_nbytes=body_components[3],
        body_padding_nbytes=body_padding_nbytes,
        checksum_nbytes=_CHECKSUM_NBYTES,
    )
    if accounting.body_nbytes != (
        accounting.record_header_nbytes
        + accounting.identity_nbytes
        + accounting.index_nbytes
        + accounting.tensor_nbytes
        + accounting.body_padding_nbytes
    ):
        raise MasterMirrorCorruptionError("body byte accounting mismatch")
    info = MasterMirrorInfo(
        family_id=info_metadata.family_id,
        request_ids=info_metadata.request_ids,
        master_request_id=info_metadata.master_request_id,
        mode=info_metadata.mode,
        atol=info_metadata.atol,
        rtol=info_metadata.rtol,
        key=info_metadata.key,
        value=info_metadata.value,
        mirrors=info_metadata.mirrors,
        dense_family_nbytes=(
            len(info_metadata.request_ids)
            * (info_metadata.key.nbytes + info_metadata.value.nbytes)
        ),
        accounting=accounting,
    )
    if info.encoded_nbytes != len(data):
        raise MasterMirrorCorruptionError("serialized-byte accounting mismatch")
    return _ParsedPayload(info, *parsed_body)


def _validate_metadata(metadata: object) -> _ValidatedMetadata:
    if not isinstance(metadata, dict):
        raise MasterMirrorCorruptionError("metadata must be an object")
    required = {
        "atol",
        "byte_order",
        "family_id",
        "format_version",
        "key",
        "master_request_id",
        "mirrors",
        "mode",
        "request_ids",
        "rtol",
        "schema",
        "value",
    }
    if set(metadata) != required:
        raise MasterMirrorCorruptionError("metadata fields do not match the schema")
    if (
        metadata["schema"] != _SCHEMA
        or type(metadata["format_version"]) is not int
        or metadata["format_version"] != _VERSION
    ):
        raise MasterMirrorCorruptionError("metadata schema or version mismatch")
    if metadata["byte_order"] != "little":
        raise MasterMirrorCorruptionError("unsupported tensor byte order")
    family_id = _metadata_identity(metadata["family_id"], "family_id")
    master_request_id = _metadata_identity(
        metadata["master_request_id"], "master_request_id"
    )

    request_ids_raw = metadata["request_ids"]
    if not isinstance(request_ids_raw, list) or not request_ids_raw:
        raise MasterMirrorCorruptionError("request_ids must be a nonempty list")
    request_ids = tuple(
        _metadata_identity(request_id, "request_id") for request_id in request_ids_raw
    )
    if request_ids != tuple(sorted(set(request_ids))):
        raise MasterMirrorCorruptionError("request_ids must be unique and sorted")
    if master_request_id not in request_ids:
        raise MasterMirrorCorruptionError("master request is absent from request_ids")

    mode, atol, rtol = _validated_mode_and_tolerances(
        metadata["mode"],
        metadata["atol"],
        metadata["rtol"],
        corruption=True,
    )

    key_info = _parse_tensor_info(metadata["key"], "key")
    value_info = _parse_tensor_info(metadata["value"], "value")
    if key_info.block_count != value_info.block_count:
        raise MasterMirrorCorruptionError("key/value block counts differ")
    if mode == "tolerance" and (
        not key_info.dtype.is_floating_point or not value_info.dtype.is_floating_point
    ):
        raise MasterMirrorCorruptionError(
            "tolerance metadata contains a non-floating dtype"
        )

    mirror_ids = tuple(
        request_id for request_id in request_ids if request_id != master_request_id
    )
    mirrors_raw = metadata["mirrors"]
    if not isinstance(mirrors_raw, list) or len(mirrors_raw) != len(mirror_ids):
        raise MasterMirrorCorruptionError("mirror metadata count mismatch")
    mirrors: list[MirrorDiffInfo] = []
    for expected_id, raw in zip(mirror_ids, mirrors_raw):
        if not isinstance(raw, dict) or set(raw) != {
            "index_encoding",
            "key_changed_blocks",
            "request_id",
            "value_changed_blocks",
        }:
            raise MasterMirrorCorruptionError("invalid mirror metadata fields")
        request_id = _metadata_identity(raw["request_id"], "mirror request_id")
        if request_id != expected_id:
            raise MasterMirrorCorruptionError("mirror metadata identity mismatch")
        index_encoding = raw["index_encoding"]
        if index_encoding not in ("shared", "separate"):
            raise MasterMirrorCorruptionError("invalid mirror index encoding")
        key_count = _metadata_count(
            raw["key_changed_blocks"], key_info.block_count, "key"
        )
        value_count = _metadata_count(
            raw["value_changed_blocks"], value_info.block_count, "value"
        )
        if index_encoding == "shared" and key_count != value_count:
            raise MasterMirrorCorruptionError("shared index counts differ")
        mirrors.append(
            MirrorDiffInfo(request_id, index_encoding, key_count, value_count)
        )
    return _ValidatedMetadata(
        family_id=family_id,
        request_ids=request_ids,
        master_request_id=master_request_id,
        mode=mode,
        atol=atol,
        rtol=rtol,
        key=key_info,
        value=value_info,
        mirrors=tuple(mirrors),
    )


def _parse_tensor_info(raw: object, label: str) -> TensorInfo:
    if not isinstance(raw, dict) or set(raw) != {
        "block_axis",
        "block_count",
        "block_nbytes",
        "dtype",
        "layout",
        "nbytes",
        "shape",
    }:
        raise MasterMirrorCorruptionError(f"invalid {label} tensor metadata")
    numeric_fields = ("block_axis", "block_count", "block_nbytes", "nbytes")
    if any(type(raw[field]) is not int for field in numeric_fields):
        raise MasterMirrorCorruptionError(
            f"{label} tensor numeric metadata must contain exact integers"
        )
    dtype_name = raw["dtype"]
    if not isinstance(dtype_name, str) or dtype_name not in _NAME_TO_DTYPE:
        raise MasterMirrorCorruptionError(f"invalid {label} dtype")
    shape = _validated_shape(raw["shape"], label, corruption=True)
    dtype = _NAME_TO_DTYPE[dtype_name]
    nbytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    block_count = shape[0]
    block_nbytes = nbytes // block_count
    if (
        raw["layout"] != "dense_contiguous"
        or raw["block_axis"] != 0
        or raw["block_count"] != block_count
        or raw["block_nbytes"] != block_nbytes
        or raw["nbytes"] != nbytes
    ):
        raise MasterMirrorCorruptionError(f"inconsistent {label} tensor metadata")
    return TensorInfo(dtype, shape, 0, block_count, block_nbytes, nbytes)


def _body_component_nbytes(
    metadata: _ValidatedMetadata,
) -> tuple[int, int, int, int]:
    record_header_nbytes = _MASTER_HEADER.size + len(metadata.mirrors) * (
        _MIRROR_HEADER.size
    )
    identity_nbytes = len(metadata.master_request_id.encode("utf-8"))
    index_nbytes = 0
    tensor_nbytes = metadata.key.nbytes + metadata.value.nbytes
    for mirror in metadata.mirrors:
        identity_nbytes += len(mirror.request_id.encode("utf-8"))
        index_nbytes += mirror.key_changed_blocks * 4
        if mirror.index_encoding == "separate":
            index_nbytes += mirror.value_changed_blocks * 4
        tensor_nbytes += mirror.key_changed_blocks * metadata.key.block_nbytes
        tensor_nbytes += mirror.value_changed_blocks * metadata.value.block_nbytes
    return record_header_nbytes, identity_nbytes, index_nbytes, tensor_nbytes


def _parse_body(
    body: bytes, metadata: _ValidatedMetadata
) -> tuple[bytes, bytes, tuple[_ParsedMirror, ...]]:
    reader = _Reader(body)
    tag, request_id_nbytes, key_nbytes, value_nbytes = cast(
        tuple[bytes, int, int, int],
        reader.unpack(_MASTER_HEADER, "master header"),
    )
    if tag != _MASTER_TAG:
        raise MasterMirrorCorruptionError("invalid master record tag")
    master_request_id = _decode_body_identity(
        reader.take(request_id_nbytes, "master request identity")
    )
    if master_request_id != metadata.master_request_id:
        raise MasterMirrorCorruptionError("master record identity mismatch")
    key_info = metadata.key
    value_info = metadata.value
    if key_nbytes != key_info.nbytes or value_nbytes != value_info.nbytes:
        raise MasterMirrorCorruptionError("master tensor length mismatch")
    reader.align("master identity")
    master_key = reader.take(key_nbytes, "master key tensor")
    reader.align("master key")
    master_value = reader.take(value_nbytes, "master value tensor")
    reader.align("master value")

    parsed_mirrors: list[_ParsedMirror] = []
    for mirror_info in metadata.mirrors:
        tag, request_id_nbytes, index_mode, reserved, key_count, value_count = cast(
            tuple[bytes, int, int, int, int, int],
            reader.unpack(_MIRROR_HEADER, "mirror header"),
        )
        if tag != _MIRROR_TAG or reserved != 0:
            raise MasterMirrorCorruptionError("invalid mirror record header")
        request_id = _decode_body_identity(
            reader.take(request_id_nbytes, "mirror request identity")
        )
        if request_id != mirror_info.request_id:
            raise MasterMirrorCorruptionError("mirror record identity mismatch")
        expected_mode = (
            _INDEX_SHARED if mirror_info.index_encoding == "shared" else _INDEX_SEPARATE
        )
        if (
            index_mode != expected_mode
            or key_count != mirror_info.key_changed_blocks
            or value_count != mirror_info.value_changed_blocks
        ):
            raise MasterMirrorCorruptionError("mirror diff metadata mismatch")
        reader.align("mirror identity")
        key_indices = _decode_indices(reader, key_count, key_info.block_count, "key")
        reader.align("key indexes")
        if index_mode == _INDEX_SHARED:
            if key_count != value_count:
                raise MasterMirrorCorruptionError("shared index counts differ")
            value_indices = key_indices
        else:
            value_indices = _decode_indices(
                reader, value_count, value_info.block_count, "value"
            )
            reader.align("value indexes")
        key_blocks = reader.take(key_count * key_info.block_nbytes, "key diff blocks")
        reader.align("key diff")
        value_blocks = reader.take(
            value_count * value_info.block_nbytes, "value diff blocks"
        )
        reader.align("value diff")
        parsed_mirrors.append(
            _ParsedMirror(
                request_id,
                key_indices,
                value_indices,
                key_blocks,
                value_blocks,
            )
        )
    reader.finish()
    return master_key, master_value, tuple(parsed_mirrors)


def _decode_indices(
    reader: _Reader, count: int, block_count: int, label: str
) -> tuple[int, ...]:
    raw = reader.take(count * 4, f"{label} indexes")
    indices = struct.unpack(f"<{count}I", raw) if count else ()
    if any(index >= block_count for index in indices):
        raise MasterMirrorCorruptionError(f"{label} diff index is out of range")
    if tuple(sorted(set(indices))) != indices:
        raise MasterMirrorCorruptionError(
            f"{label} diff indexes must be unique and sorted"
        )
    return indices


def _tensor_from_bytes(data: bytes, info: TensorInfo) -> torch.Tensor:
    if len(data) != info.nbytes:
        raise MasterMirrorCorruptionError("dense tensor byte length mismatch")
    raw = torch.frombuffer(bytearray(data), dtype=torch.uint8).clone()
    return raw.view(info.dtype).reshape(info.shape).contiguous()


def _apply_blocks(
    tensor: torch.Tensor,
    indices: tuple[int, ...],
    blocks: bytes,
    info: TensorInfo,
) -> None:
    expected_nbytes = len(indices) * info.block_nbytes
    if len(blocks) != expected_nbytes:
        raise MasterMirrorCorruptionError("diff block byte length mismatch")
    if not indices:
        return
    target = tensor.view(torch.uint8).reshape(info.block_count, info.block_nbytes)
    source = torch.frombuffer(bytearray(blocks), dtype=torch.uint8).reshape(
        len(indices), info.block_nbytes
    )
    for source_index, block_index in enumerate(indices):
        target[block_index].copy_(source[source_index])


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    contiguous_bytes = tensor.view(torch.uint8).reshape(-1)
    return memoryview(contiguous_bytes.numpy()).tobytes()


def _tensor_info_dict(info: TensorInfo) -> dict[str, object]:
    return {
        "block_axis": info.block_axis,
        "block_count": info.block_count,
        "block_nbytes": info.block_nbytes,
        "dtype": _DTYPE_TO_NAME[info.dtype],
        "layout": "dense_contiguous",
        "nbytes": info.nbytes,
        "shape": list(info.shape),
    }


def _padding_nbytes(position: int) -> int:
    return (-position) % _ALIGNMENT


def _validate_identity(value: str, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    if len(value.encode("utf-8")) > 0xFFFF:
        raise ValueError(f"{label} exceeds the 65535-byte wire limit")


def _metadata_identity(value: object, label: str) -> str:
    try:
        _validate_identity(value, label)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise MasterMirrorCorruptionError(f"invalid {label}") from error
    assert isinstance(value, str)
    return value


def _decode_body_identity(data: bytes) -> str:
    try:
        value = data.decode("utf-8")
        _validate_identity(value, "request identity")
    except (UnicodeDecodeError, ValueError) as error:
        raise MasterMirrorCorruptionError("invalid body request identity") from error
    return value


def _metadata_count(value: object, limit: int, label: str) -> int:
    if type(value) is not int or not 0 <= value <= limit:
        raise MasterMirrorCorruptionError(f"invalid {label} changed-block count")
    return value


def _validated_mode_and_tolerances(
    mode: object,
    atol: object,
    rtol: object,
    *,
    corruption: bool,
) -> tuple[Literal["exact", "tolerance"], float | None, float | None]:
    error_type = MasterMirrorCorruptionError if corruption else ValueError
    if mode == "exact":
        if atol is not None or rtol is not None:
            raise error_type("exact mode does not accept atol or rtol")
        return "exact", None, None
    if mode != "tolerance":
        raise error_type(f"unsupported mode: {mode!r}")
    if atol is None or rtol is None:
        raise error_type("tolerance mode requires explicit atol and rtol")

    normalized: list[float] = []
    for label, value in (("atol", atol), ("rtol", rtol)):
        if type(value) not in (int, float):
            raise error_type(f"{label} must be an int or float, not bool")
        try:
            numeric = float(cast(int | float, value))
        except OverflowError as error:
            raise error_type(f"{label} must be finite and nonnegative") from error
        if not math.isfinite(numeric) or numeric < 0:
            raise error_type(f"{label} must be finite and nonnegative")
        normalized.append(numeric)
    return "tolerance", normalized[0], normalized[1]


def _validated_shape(
    shape: object,
    label: str,
    *,
    corruption: bool,
) -> tuple[int, ...]:
    error_type = MasterMirrorCorruptionError if corruption else ValueError
    if (
        not isinstance(shape, (list, tuple, torch.Size))
        or not shape
        or any(type(dim) is not int or dim <= 0 for dim in shape)
    ):
        raise error_type(f"{label} must have nonzero integer dimensions")
    return tuple(shape)
