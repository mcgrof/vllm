# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import hashlib
import json
import math
import struct
import tracemalloc
from collections.abc import Callable

import pytest
import torch

from vllm.v1.kmesh import (
    MasterMirrorCodec,
    MasterMirrorCorruptionError,
    MasterMirrorIdentityError,
    MasterMirrorRequest,
    MasterMirrorRequestNotFoundError,
)
from vllm.v1.kmesh.master_mirror import _tensor_bytes

pytestmark = pytest.mark.skip_global_cleanup


def _tensor(shape: tuple[int, ...], dtype: torch.dtype, offset: int = 0):
    values = torch.arange(math.prod(shape), dtype=torch.float64).reshape(shape)
    return (values + offset).to(dtype)


def _family(shape: tuple[int, ...], dtype: torch.dtype):
    base_key = _tensor(shape, dtype)
    base_value = _tensor(shape, dtype, offset=100)
    shared_key = base_key.clone()
    shared_value = base_value.clone()
    shared_key[1].add_(1)
    shared_value[1].add_(1)
    separate_key = base_key.clone()
    separate_value = base_value.clone()
    separate_key[2].add_(2)
    separate_value[3].add_(2)
    return [
        MasterMirrorRequest("request-a", base_key, base_value),
        MasterMirrorRequest("request-b", shared_key, shared_value),
        MasterMirrorRequest("request-c", separate_key, separate_value),
    ]


@pytest.mark.parametrize(
    ("shape", "dtype"),
    [
        ((5, 7), torch.float16),
        ((5, 2, 3), torch.float16),
        ((5, 2, 2, 4), torch.bfloat16),
        ((5, 3), torch.bfloat16),
    ],
)
def test_exact_round_trip_is_deterministic_and_dense(shape, dtype):
    requests = _family(shape, dtype)

    encoded = MasterMirrorCodec.encode("family-1", requests)
    reversed_encoded = MasterMirrorCodec.encode("family-1", reversed(requests))

    assert encoded == reversed_encoded
    info = MasterMirrorCodec.inspect(encoded, family_id="family-1")
    assert info.master_request_id == "request-a"
    assert info.encoded_nbytes == len(encoded)
    assert info.accounting.total_nbytes == len(encoded)
    assert info.accounting.body_nbytes == (
        info.accounting.record_header_nbytes
        + info.accounting.identity_nbytes
        + info.accounting.index_nbytes
        + info.accounting.tensor_nbytes
        + info.accounting.body_padding_nbytes
    )
    assert info.accounting.alignment_nbytes == (
        info.accounting.metadata_padding_nbytes + info.accounting.body_padding_nbytes
    )
    assert info.dense_family_nbytes == sum(
        request.key.nbytes + request.value.nbytes for request in requests
    )
    assert [mirror.index_encoding for mirror in info.mirrors] == [
        "shared",
        "separate",
    ]

    for request in requests:
        key, value = MasterMirrorCodec.decode(
            encoded,
            family_id="family-1",
            request_id=request.request_id,
        )
        assert key.device.type == "cpu" and key.is_contiguous()
        assert value.device.type == "cpu" and value.is_contiguous()
        assert torch.equal(key.view(torch.uint8), request.key.view(torch.uint8))
        assert torch.equal(value.view(torch.uint8), request.value.view(torch.uint8))


def test_master_uses_semantic_score_then_complete_serialized_size():
    requests = _family((5, 2, 4), torch.float16)

    metrics = MasterMirrorCodec.candidate_metrics("family-size", requests)
    encoded = MasterMirrorCodec.encode("family-size", requests)
    info = MasterMirrorCodec.inspect(encoded)
    expected = min(metrics, key=lambda candidate: candidate.score)

    assert info.master_request_id == expected.master_request_id
    assert info.encoded_nbytes == expected.encoded_nbytes
    assert {metric.master_request_id for metric in metrics} == {
        request.request_id for request in requests
    }


def test_semantic_changed_payload_score_selects_median_candidate():
    def request(request_id: str, bits: tuple[int, int]):
        tensor = torch.tensor(bits, dtype=torch.float16).reshape(2, 1)
        return MasterMirrorRequest(request_id, tensor, tensor.clone())

    requests = [
        request("a", (0, 0)),
        request("b", (0, 1)),
        request("c", (1, 1)),
    ]

    metrics = {
        metric.master_request_id: metric
        for metric in MasterMirrorCodec.candidate_metrics("family-median", requests)
    }
    encoded = MasterMirrorCodec.encode("family-median", requests)

    assert {
        request_id: (metric.changed_tensor_nbytes, metric.changed_block_count)
        for request_id, metric in metrics.items()
    } == {"a": (12, 6), "b": (8, 4), "c": (12, 6)}
    assert MasterMirrorCodec.inspect(encoded).master_request_id == "b"
    for original in requests:
        restored = MasterMirrorCodec.decode(
            encoded, family_id="family-median", request_id=original.request_id
        )
        assert torch.equal(restored[0], original.key)
        assert torch.equal(restored[1], original.value)


def test_matching_key_value_changes_share_one_index_encoding():
    key = torch.zeros((4, 2, 3), dtype=torch.float16)
    value = torch.zeros((4, 2, 3), dtype=torch.float16)
    shared_key, shared_value = key.clone(), value.clone()
    shared_key[1].fill_(1)
    shared_value[1].fill_(1)
    separate_key, separate_value = key.clone(), value.clone()
    separate_key[1].fill_(1)
    separate_value[2].fill_(1)

    shared = MasterMirrorCodec.encode(
        "family-index",
        [
            MasterMirrorRequest("base", key, value),
            MasterMirrorRequest("peer", shared_key, shared_value),
        ],
    )
    separate = MasterMirrorCodec.encode(
        "family-index",
        [
            MasterMirrorRequest("base", key, value),
            MasterMirrorRequest("peer", separate_key, separate_value),
        ],
    )

    shared_info = MasterMirrorCodec.inspect(shared)
    separate_info = MasterMirrorCodec.inspect(separate)
    assert shared_info.mirrors[0].index_encoding == "shared"
    assert separate_info.mirrors[0].index_encoding == "separate"
    assert shared_info.accounting.index_nbytes == 4
    assert separate_info.accounting.index_nbytes == 8
    assert len(shared) < len(separate)


def test_tolerance_mode_records_tolerances_and_bounds_restore_error():
    master_key = torch.zeros((3, 4), dtype=torch.float16)
    master_value = torch.ones((3, 4), dtype=torch.float16)
    mirror_key = master_key.clone()
    mirror_value = master_value.clone()
    mirror_key[0].add_(0.005)
    mirror_value[0].add_(0.005)
    mirror_key[1].add_(0.1)
    mirror_value[1].add_(0.1)
    requests = [
        MasterMirrorRequest("master", master_key, master_value),
        MasterMirrorRequest("mirror", mirror_key, mirror_value),
    ]

    encoded = MasterMirrorCodec.encode(
        "family-tolerance",
        requests,
        mode="tolerance",
        atol=0.01,
        rtol=0.0,
    )
    info = MasterMirrorCodec.inspect(encoded)
    restored_key, restored_value = MasterMirrorCodec.decode(
        encoded,
        family_id="family-tolerance",
        request_id="mirror",
    )

    assert info.mode == "tolerance"
    assert info.atol == 0.01 and info.rtol == 0.0
    assert info.mirrors[0].key_changed_blocks == 1
    assert info.mirrors[0].value_changed_blocks == 1
    assert not torch.equal(restored_key, mirror_key)
    assert torch.allclose(mirror_key, restored_key, atol=0.01, rtol=0.0)
    assert torch.allclose(mirror_value, restored_value, atol=0.01, rtol=0.0)


def test_exact_mode_preserves_nan_and_signed_zero_bytes():
    key = torch.tensor([[0.0, -0.0], [float("nan"), float("inf")]], dtype=torch.float16)
    value = key.flip(0).clone()
    peer_key, peer_value = key.clone(), value.clone()
    peer_key[1, 1] = float("-inf")
    requests = [
        MasterMirrorRequest("a", key, value),
        MasterMirrorRequest("b", peer_key, peer_value),
    ]

    encoded = MasterMirrorCodec.encode("family-special", requests)

    for request in requests:
        restored = MasterMirrorCodec.decode(
            encoded,
            family_id="family-special",
            request_id=request.request_id,
        )
        assert torch.equal(restored[0].view(torch.uint8), request.key.view(torch.uint8))
        assert torch.equal(
            restored[1].view(torch.uint8), request.value.view(torch.uint8)
        )


def test_key_and_value_metadata_can_have_different_shapes_and_dtypes():
    key = torch.arange(24, dtype=torch.float16).reshape(4, 2, 3)
    value = torch.arange(20, dtype=torch.bfloat16).reshape(4, 5)
    peer_key, peer_value = key.clone(), value.clone()
    peer_key[2].add_(1)
    peer_value[2].add_(1)
    requests = [
        MasterMirrorRequest("a", key, value),
        MasterMirrorRequest("b", peer_key, peer_value),
    ]

    encoded = MasterMirrorCodec.encode("family-asymmetric", requests)
    info = MasterMirrorCodec.inspect(encoded)

    assert info.key.shape == (4, 2, 3) and info.key.dtype == torch.float16
    assert info.value.shape == (4, 5) and info.value.dtype == torch.bfloat16
    for request in requests:
        restored = MasterMirrorCodec.decode(
            encoded,
            family_id="family-asymmetric",
            request_id=request.request_id,
        )
        assert torch.equal(restored[0], request.key)
        assert torch.equal(restored[1], request.value)


def test_decode_rejects_wrong_family_and_missing_request():
    request = MasterMirrorRequest(
        "request-a",
        torch.zeros((2, 3), dtype=torch.float16),
        torch.ones((2, 3), dtype=torch.float16),
    )
    encoded = MasterMirrorCodec.encode("family-identity", [request])

    with pytest.raises(MasterMirrorIdentityError, match="family identity mismatch"):
        MasterMirrorCodec.decode(
            encoded, family_id="wrong-family", request_id="request-a"
        )
    with pytest.raises(MasterMirrorRequestNotFoundError, match="is missing"):
        MasterMirrorCodec.decode(
            encoded, family_id="family-identity", request_id="missing"
        )


@pytest.mark.parametrize("mutation", ["checksum", "truncate", "append", "magic"])
def test_corruption_is_rejected(mutation):
    request = MasterMirrorRequest(
        "request-a",
        torch.zeros((2, 3), dtype=torch.float16),
        torch.ones((2, 3), dtype=torch.float16),
    )
    encoded = bytearray(MasterMirrorCodec.encode("family-corrupt", [request]))
    if mutation == "checksum":
        encoded[-1] ^= 1
    elif mutation == "truncate":
        encoded = encoded[:-5]
    elif mutation == "append":
        encoded.extend(b"extra")
    else:
        encoded[0] ^= 1

    with pytest.raises(MasterMirrorCorruptionError):
        MasterMirrorCodec.inspect(bytes(encoded))


def test_out_of_range_diff_index_is_rejected_even_with_valid_checksum():
    key = torch.zeros((3, 2), dtype=torch.float16)
    value = torch.zeros((3, 2), dtype=torch.float16)
    peer_key, peer_value = key.clone(), value.clone()
    peer_key[1].fill_(1)
    peer_value[1].fill_(1)
    encoded = bytearray(
        MasterMirrorCodec.encode(
            "family-wrong-diff",
            [
                MasterMirrorRequest("base", key, value),
                MasterMirrorRequest("peer", peer_key, peer_value),
            ],
        )
    )
    index_offset, checksum_offset = _first_diff_index_offset(encoded)
    struct.pack_into("<I", encoded, index_offset, 0xFFFFFFFF)
    encoded[checksum_offset:] = hashlib.sha256(encoded[:checksum_offset]).digest()

    with pytest.raises(MasterMirrorCorruptionError, match="index is out of range"):
        MasterMirrorCodec.decode(
            bytes(encoded), family_id="family-wrong-diff", request_id="peer"
        )


def test_invalid_input_contract_is_rejected():
    key = torch.zeros((2, 3), dtype=torch.float16)
    value = torch.zeros((2, 3), dtype=torch.float16)
    request = MasterMirrorRequest("request", key, value)

    with pytest.raises(ValueError, match="explicit atol and rtol"):
        MasterMirrorCodec.encode("family", [request], mode="tolerance")
    with pytest.raises(ValueError, match="does not accept"):
        MasterMirrorCodec.encode("family", [request], atol=0.0)
    with pytest.raises(ValueError, match="unique"):
        MasterMirrorCodec.encode("family", [request, request])
    with pytest.raises(ValueError, match="same block count"):
        MasterMirrorCodec.encode(
            "family",
            [
                MasterMirrorRequest(
                    "request", key, torch.zeros((3, 2), dtype=torch.float16)
                )
            ],
        )


@pytest.mark.parametrize(
    ("atol", "rtol"),
    [(True, 0.0), (0.0, False)],
)
def test_tolerance_rejects_booleans_and_normalizes_valid_numbers(atol, rtol):
    request = MasterMirrorRequest(
        "request",
        torch.zeros((2, 3), dtype=torch.float16),
        torch.ones((2, 3), dtype=torch.float16),
    )

    with pytest.raises(ValueError, match="not bool"):
        MasterMirrorCodec.encode(
            "family", [request], mode="tolerance", atol=atol, rtol=rtol
        )

    encoded = MasterMirrorCodec.encode(
        "family", [request], mode="tolerance", atol=1, rtol=0
    )
    info = MasterMirrorCodec.inspect(encoded)
    assert info.atol == 1.0 and type(info.atol) is float
    assert info.rtol == 0.0 and type(info.rtol) is float
    restored = MasterMirrorCodec.decode(
        encoded, family_id="family", request_id="request"
    )
    assert torch.equal(restored[0], request.key)
    assert torch.equal(restored[1], request.value)


def test_zero_sized_trailing_dimension_is_rejected():
    empty = torch.empty((2, 0, 3), dtype=torch.float16)

    with pytest.raises(ValueError, match="nonzero integer dimensions"):
        MasterMirrorCodec.encode(
            "family", [MasterMirrorRequest("request", empty, empty)]
        )

    tensor = torch.empty((2, 1, 3), dtype=torch.float16)
    encoded = MasterMirrorCodec.encode(
        "family", [MasterMirrorRequest("request", tensor, tensor)]
    )
    corrupted = _rewrite_metadata(
        encoded, lambda metadata: metadata["key"]["shape"].__setitem__(1, 0)
    )
    with pytest.raises(MasterMirrorCorruptionError, match="nonzero integer dimensions"):
        MasterMirrorCodec.inspect(corrupted)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        pytest.param(
            getattr(torch, "float8_e4m3fn", torch.float32),
            id="float8_e4m3fn",
        ),
    ],
)
def test_offline_v0_rejects_dtypes_outside_fp16_bf16(dtype):
    tensor = torch.empty((2, 3), dtype=dtype)

    with pytest.raises(ValueError, match="supports only.*does not yet support FP8"):
        MasterMirrorCodec.encode(
            "family", [MasterMirrorRequest("request", tensor, tensor)]
        )


def test_contiguous_uint8_copy_matches_legacy_bytes_with_lower_amplification():
    tensor = torch.arange(128 * 1024, dtype=torch.float32).to(torch.bfloat16)

    new_bytes, new_peak = _peak_traced_bytes(lambda: _tensor_bytes(tensor))
    legacy_bytes, legacy_peak = _peak_traced_bytes(
        lambda: bytes(tensor.view(torch.uint8).reshape(-1).tolist())
    )

    assert new_bytes == legacy_bytes
    assert new_peak * 4 < legacy_peak


def test_recursive_metadata_is_reported_as_corruption(monkeypatch):
    tensor = torch.zeros((2, 3), dtype=torch.float16)
    encoded = MasterMirrorCodec.encode(
        "family-recursive", [MasterMirrorRequest("request", tensor, tensor)]
    )

    def raise_recursion_error(*args, **kwargs):
        raise RecursionError

    monkeypatch.setattr("vllm.v1.kmesh.master_mirror.json.loads", raise_recursion_error)

    with pytest.raises(MasterMirrorCorruptionError, match="invalid metadata JSON"):
        MasterMirrorCodec.inspect(encoded)


def test_numeric_metadata_rejects_booleans_with_valid_checksum():
    tensor = torch.zeros((2, 3), dtype=torch.float16)
    encoded = MasterMirrorCodec.encode(
        "family-numeric", [MasterMirrorRequest("request", tensor, tensor)]
    )

    bad_version = _rewrite_metadata(
        encoded, lambda metadata: metadata.__setitem__("format_version", True)
    )
    with pytest.raises(MasterMirrorCorruptionError, match="schema or version"):
        MasterMirrorCodec.inspect(bad_version)

    bad_tensor_size = _rewrite_metadata(
        encoded, lambda metadata: metadata["key"].__setitem__("nbytes", True)
    )
    with pytest.raises(MasterMirrorCorruptionError, match="exact integers"):
        MasterMirrorCodec.inspect(bad_tensor_size)


def test_bool_tolerance_metadata_is_rejected_with_valid_checksum():
    tensor = torch.zeros((2, 3), dtype=torch.float16)
    encoded = MasterMirrorCodec.encode(
        "family-tolerance-metadata",
        [MasterMirrorRequest("request", tensor, tensor)],
        mode="tolerance",
        atol=0.0,
        rtol=0.0,
    )
    corrupted = _rewrite_metadata(
        encoded, lambda metadata: metadata.__setitem__("atol", True)
    )

    with pytest.raises(MasterMirrorCorruptionError, match="not bool"):
        MasterMirrorCodec.inspect(corrupted)


def test_checksummed_missing_diff_is_reported_clearly():
    key = torch.zeros((3, 2), dtype=torch.float16)
    value = torch.zeros((3, 2), dtype=torch.float16)
    peer_key, peer_value = key.clone(), value.clone()
    peer_key[1].fill_(1)
    peer_value[1].fill_(1)
    encoded = MasterMirrorCodec.encode(
        "family-missing-diff",
        [
            MasterMirrorRequest("base", key, value),
            MasterMirrorRequest("peer", peer_key, peer_value),
        ],
    )
    corrupted = _drop_first_mirror_record(encoded)

    with pytest.raises(
        MasterMirrorCorruptionError, match="missing or truncated mirror header"
    ):
        MasterMirrorCodec.inspect(corrupted)


def _peak_traced_bytes(operation: Callable[[], bytes]) -> tuple[bytes, int]:
    gc.collect()
    tracemalloc.start()
    try:
        result = operation()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, peak


def _build_payload(metadata: bytes, body: bytes) -> bytes:
    file_header = struct.Struct("<8sHHIQQ")
    alignment = 8
    header = file_header.pack(b"KMSHMM01", 1, alignment, 0, len(metadata), len(body))
    padding = b"\x00" * (
        _align(len(header) + len(metadata), alignment) - len(header) - len(metadata)
    )
    checksummed = header + metadata + padding + body
    return checksummed + hashlib.sha256(checksummed).digest()


def _rewrite_metadata(encoded: bytes, mutation: Callable[[dict], object]) -> bytes:
    file_header = struct.Struct("<8sHHIQQ")
    _, _, alignment, _, metadata_nbytes, body_nbytes = file_header.unpack_from(encoded)
    metadata_start = file_header.size
    metadata_end = metadata_start + metadata_nbytes
    body_start = _align(metadata_end, alignment)
    metadata = json.loads(encoded[metadata_start:metadata_end])
    mutation(metadata)
    rewritten = json.dumps(
        metadata,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _build_payload(rewritten, encoded[body_start : body_start + body_nbytes])


def _drop_first_mirror_record(encoded: bytes) -> bytes:
    file_header = struct.Struct("<8sHHIQQ")
    _, _, alignment, _, metadata_nbytes, _ = file_header.unpack_from(encoded)
    metadata_start = file_header.size
    metadata_end = metadata_start + metadata_nbytes
    body_start = _align(metadata_end, alignment)
    mirror_start = _first_mirror_offset(encoded, body_start, alignment)
    return _build_payload(
        encoded[metadata_start:metadata_end], encoded[body_start:mirror_start]
    )


def _first_mirror_offset(encoded: bytes | bytearray, body_start: int, alignment: int):
    master_header = struct.Struct("<4sH2xQQ")
    _, master_id_nbytes, key_nbytes, value_nbytes = master_header.unpack_from(
        encoded, body_start
    )
    position = _align(body_start + master_header.size + master_id_nbytes, alignment)
    position = _align(position + key_nbytes, alignment)
    return _align(position + value_nbytes, alignment)


def _first_diff_index_offset(encoded: bytearray) -> tuple[int, int]:
    file_header = struct.Struct("<8sHHIQQ")
    mirror_header = struct.Struct("<4sHBBII")
    _, _, alignment, _, metadata_nbytes, body_nbytes = file_header.unpack_from(encoded)
    body_start = _align(file_header.size + metadata_nbytes, alignment)
    position = _first_mirror_offset(encoded, body_start, alignment)
    _, mirror_id_nbytes, _, _, key_count, _ = mirror_header.unpack_from(
        encoded, position
    )
    assert key_count > 0
    position = _align(position + mirror_header.size + mirror_id_nbytes, alignment)
    return position, body_start + body_nbytes


def _align(position: int, alignment: int) -> int:
    return position + (-position % alignment)
