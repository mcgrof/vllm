# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import FrozenInstanceError, replace
from multiprocessing.reduction import ForkingPickler

import msgspec
import pytest

from vllm.v1.kmesh import (
    KMeshCompatibilityKey,
    KMeshFallbackReason,
    KMeshSegment,
    KMeshSegmentKind,
    KMeshSharedSegmentIdentity,
    compatibility_mismatch_reason,
    kmesh_token_slice_digest,
)

pytestmark = pytest.mark.skip_global_cleanup


def make_key(**overrides: object) -> KMeshCompatibilityKey:
    values = {
        "execution_id": "execution-1",
        "group_id": "gather-1",
        "round_id": 2,
        "model_id": "Qwen/Qwen2.5-7B",
        "model_revision": "model-sha",
        "tokenizer_id_or_revision": "tokenizer-sha",
        "dtype": "bfloat16",
        "kv_cache_dtype": "bfloat16",
        "kv_layout": "paged-nhd",
        "attention_kind": "full-decoder",
        "rope_configuration": '{"rope_theta":1000000}',
        "attention_backend": "rocm-attention",
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "lora_adapter_identity": None,
        "cache_salt": "salt-1",
        "cache_domain": "tenant-a",
        "active_prompt_length": 16,
        "cached_span": (0, 4),
        "prefill_chunk_span": (4, 16),
    }
    values.update(overrides)
    return KMeshCompatibilityKey(**values)  # type: ignore[arg-type]


def test_compatibility_key_is_complete_immutable_and_transport_safe():
    key = make_key()
    primitive = key.to_primitive()

    assert ForkingPickler.loads(ForkingPickler.dumps(key)) == key
    assert msgspec.msgpack.decode(msgspec.msgpack.encode(primitive)) == primitive
    assert tuple(primitive) == (
        "execution_id",
        "group_id",
        "round_id",
        "model_id",
        "model_revision",
        "tokenizer_id_or_revision",
        "dtype",
        "kv_cache_dtype",
        "kv_layout",
        "attention_kind",
        "rope_configuration",
        "attention_backend",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "lora_adapter_identity",
        "cache_salt",
        "cache_domain",
        "active_prompt_length",
        "cached_span",
        "prefill_chunk_span",
    )
    with pytest.raises(FrozenInstanceError):
        key.round_id = 4  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("execution_id", "execution-2", KMeshFallbackReason.EXECUTION_ID_MISMATCH),
        ("group_id", "gather-2", KMeshFallbackReason.GROUP_ID_MISMATCH),
        ("round_id", 3, KMeshFallbackReason.ROUND_MISMATCH),
        ("model_id", "other-model", KMeshFallbackReason.MODEL_MISMATCH),
        ("model_revision", "other-sha", KMeshFallbackReason.MODEL_MISMATCH),
        (
            "tokenizer_id_or_revision",
            "other-tokenizer",
            KMeshFallbackReason.TOKENIZER_MISMATCH,
        ),
        ("dtype", "float16", KMeshFallbackReason.DTYPE_MISMATCH),
        (
            "kv_cache_dtype",
            "fp8_e4m3",
            KMeshFallbackReason.KV_CACHE_DTYPE_MISMATCH,
        ),
        ("kv_layout", "paged-hnd", KMeshFallbackReason.KV_LAYOUT_MISMATCH),
        (
            "attention_kind",
            "sliding-window",
            KMeshFallbackReason.ATTENTION_KIND_MISMATCH,
        ),
        (
            "rope_configuration",
            '{"rope_theta":500000}',
            KMeshFallbackReason.ROPE_CONFIGURATION_MISMATCH,
        ),
        (
            "attention_backend",
            "flash-attention",
            KMeshFallbackReason.ATTENTION_BACKEND_MISMATCH,
        ),
        (
            "tensor_parallel_size",
            2,
            KMeshFallbackReason.TENSOR_PARALLEL_MISMATCH,
        ),
        (
            "pipeline_parallel_size",
            2,
            KMeshFallbackReason.PIPELINE_PARALLEL_MISMATCH,
        ),
        ("lora_adapter_identity", "adapter-a", KMeshFallbackReason.LORA_MISMATCH),
        ("cache_salt", "salt-2", KMeshFallbackReason.CACHE_SALT_MISMATCH),
        (
            "cache_domain",
            "tenant-b",
            KMeshFallbackReason.CACHE_DOMAIN_MISMATCH,
        ),
        (
            "active_prompt_length",
            17,
            KMeshFallbackReason.ACTIVE_LENGTH_MISMATCH,
        ),
        ("cached_span", (0, 5), KMeshFallbackReason.CACHED_SPAN_MISMATCH),
        (
            "prefill_chunk_span",
            (4, 15),
            KMeshFallbackReason.PREFILL_CHUNK_MISMATCH,
        ),
    ],
)
def test_each_safety_field_participates_in_exact_compatibility(
    field: str, value: object, reason: KMeshFallbackReason
):
    baseline = make_key()
    changes = {field: value}
    if field == "cached_span":
        changes["prefill_chunk_span"] = (5, 16)
    changed = replace(baseline, **changes)

    assert baseline != changed
    assert compatibility_mismatch_reason(baseline, changed) is reason


def test_compatibility_mismatch_uses_stable_safety_precedence():
    baseline = make_key()
    changed = replace(
        baseline,
        round_id=8,
        lora_adapter_identity="adapter-b",
        cache_domain="tenant-b",
        active_prompt_length=20,
        prefill_chunk_span=(4, 20),
    )

    assert (
        compatibility_mismatch_reason(baseline, changed)
        is KMeshFallbackReason.ROUND_MISMATCH
    )
    assert (
        compatibility_mismatch_reason(
            replace(baseline, round_id=8), replace(changed, round_id=8)
        )
        is KMeshFallbackReason.LORA_MISMATCH
    )


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"tensor_parallel_size": True}, ValueError),
        ({"pipeline_parallel_size": 0}, ValueError),
        ({"active_prompt_length": 0}, ValueError),
        ({"cached_span": [0, 4]}, TypeError),
        ({"cached_span": (0, 17)}, ValueError),
        ({"prefill_chunk_span": (4, 4)}, ValueError),
        ({"cached_span": (1, 4)}, ValueError),
        (
            {"cached_span": (0, 8), "prefill_chunk_span": (4, 16)},
            ValueError,
        ),
        (
            {"cached_span": (0, 4), "prefill_chunk_span": (8, 16)},
            ValueError,
        ),
        ({"cache_domain": ""}, ValueError),
        ({"model_revision": chr(0xD800)}, ValueError),
    ],
)
def test_compatibility_key_rejects_uncertain_or_invalid_shapes(
    overrides: dict[str, object], error: type[BaseException]
):
    with pytest.raises(error):
        make_key(**overrides)


def test_shared_identity_ignores_offsets_and_logical_order_only():
    first = KMeshSegment(
        segment_id="shared-update-1",
        kind=KMeshSegmentKind.SHARED_UPDATE,
        token_start=0,
        token_end=8,
        logical_order=0,
        source_id="agent-output-1",
        shareable=True,
        lineage_id="round-1",
    )
    moved = replace(first, token_start=12, token_end=20, logical_order=3)
    shared_tokens = tuple(range(100, 108))
    first_prompt = shared_tokens
    moved_prompt = (0,) * 12 + shared_tokens

    assert KMeshSharedSegmentIdentity.from_segment(
        first, first_prompt
    ) == KMeshSharedSegmentIdentity.from_segment(moved, moved_prompt)


@pytest.mark.parametrize(
    "changed",
    [
        {"segment_id": "other-id"},
        {"kind": KMeshSegmentKind.TOOL_RESULT},
        {"token_end": 9},
        {"source_id": "other-source"},
        {"lineage_id": "other-lineage"},
    ],
)
def test_shared_identity_requires_all_declared_identity_fields(
    changed: dict[str, object],
):
    segment = KMeshSegment(
        segment_id="shared-update-1",
        kind=KMeshSegmentKind.SHARED_UPDATE,
        token_start=0,
        token_end=8,
        logical_order=0,
        source_id="agent-output-1",
        shareable=True,
        lineage_id="round-1",
    )

    changed_segment = replace(segment, **changed)
    assert KMeshSharedSegmentIdentity.from_segment(
        segment, tuple(range(8))
    ) != KMeshSharedSegmentIdentity.from_segment(
        changed_segment, tuple(range(changed_segment.token_end))
    )


def test_shared_identity_is_bound_to_exact_token_content_and_transport_safe():
    segment = KMeshSegment(
        segment_id="shared-update-1",
        kind=KMeshSegmentKind.SHARED_UPDATE,
        token_start=0,
        token_end=4,
        logical_order=0,
        source_id="agent-output-1",
        shareable=True,
        lineage_id="round-1",
    )
    first = KMeshSharedSegmentIdentity.from_segment(segment, (10, 11, 12, 13))
    same = KMeshSharedSegmentIdentity.from_segment(segment, [10, 11, 12, 13])
    different = KMeshSharedSegmentIdentity.from_segment(segment, (10, 11, 12, 14))
    primitive = first.to_primitive()

    assert first == same
    assert first != different
    assert first.content_digest == kmesh_token_slice_digest((10, 11, 12, 13))
    assert first.content_digest.startswith("sha256-v1:")
    assert len(first.content_digest) == len("sha256-v1:") + 64
    assert ForkingPickler.loads(ForkingPickler.dumps(first)) == first
    assert msgspec.msgpack.decode(msgspec.msgpack.encode(primitive)) == primitive


def test_shared_identity_requires_token_evidence_covering_the_declared_slice():
    segment = KMeshSegment(
        segment_id="shared-update-1",
        kind=KMeshSegmentKind.SHARED_UPDATE,
        token_start=4,
        token_end=8,
        logical_order=0,
        source_id="agent-output-1",
        shareable=True,
        lineage_id="round-1",
    )

    with pytest.raises(ValueError, match="exceeds prompt-token evidence"):
        KMeshSharedSegmentIdentity.from_segment(segment, (0,) * 7)


def test_private_segment_cannot_be_promoted_to_shared_identity():
    segment = KMeshSegment(
        segment_id="private-1",
        kind=KMeshSegmentKind.PRIVATE_HISTORY,
        token_start=0,
        token_end=8,
        logical_order=0,
        source_id="agent-1",
        shareable=False,
        lineage_id="agent-1",
    )

    with pytest.raises(ValueError, match="private segments"):
        KMeshSharedSegmentIdentity.from_segment(segment, (0,) * 8)
