# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Sequence
from multiprocessing.reduction import ForkingPickler

import msgspec
import pytest

from vllm.v1.kmesh import (
    KMESH_SCHEMA_VERSION,
    KMESH_SELECTED_PLAN_OPPORTUNITY_SCOPE,
    KMeshCompatibilityKey,
    KMeshFallbackReason,
    KMeshGroupKind,
    KMeshMetadataError,
    KMeshMetadataErrorCode,
    KMeshPlanner,
    KMeshRequestContext,
    KMeshRequestContextParseResult,
    KMeshRequestObservation,
    KMeshSegment,
    KMeshSegmentKind,
    KMeshStateKind,
)

pytestmark = pytest.mark.skip_global_cleanup

_DEFAULT_KEY = object()
_DEFAULT_PROMPT_TOKEN_IDS = object()


def make_context(
    request_id: str,
    *,
    execution_id: str = "execution-1",
    group_id: str = "gather-1",
    round_id: int | str = 2,
    cache_domain: str | None = "tenant-a",
    active_length: int = 16,
    shared_start: int = 4,
    shared_end: int = 12,
    shared_id: str = "shared-update-1",
    shared_kind: KMeshSegmentKind = KMeshSegmentKind.SHARED_UPDATE,
    shared_source: str | None = "agent-output-1",
    shared_lineage: str | None = "round-1",
    shareable: bool = True,
    group_kind: KMeshGroupKind = KMeshGroupKind.AGENT_ROUND,
    state_kind: KMeshStateKind = KMeshStateKind.KV_CACHE,
    enabled: bool = True,
) -> KMeshRequestContext:
    segments = []
    logical_order = 0
    if shared_start:
        segments.append(
            KMeshSegment(
                segment_id=f"private-before-{request_id}",
                kind=KMeshSegmentKind.PRIVATE_HISTORY,
                token_start=0,
                token_end=shared_start,
                logical_order=logical_order,
                source_id=f"private-source-{request_id}",
                shareable=False,
                lineage_id=f"private-lineage-{request_id}",
            )
        )
        logical_order += 1
    segments.append(
        KMeshSegment(
            segment_id=shared_id,
            kind=shared_kind,
            token_start=shared_start,
            token_end=shared_end,
            logical_order=logical_order,
            source_id=shared_source,
            shareable=shareable,
            lineage_id=shared_lineage,
        )
    )
    logical_order += 1
    if shared_end < active_length:
        segments.append(
            KMeshSegment(
                segment_id=f"private-after-{request_id}",
                kind=KMeshSegmentKind.TASK,
                token_start=shared_end,
                token_end=active_length,
                logical_order=logical_order,
                source_id=f"task-source-{request_id}",
                shareable=False,
                lineage_id=f"task-lineage-{request_id}",
            )
        )
    return KMeshRequestContext(
        schema_version=KMESH_SCHEMA_VERSION,
        enabled=enabled,
        execution_id=execution_id,
        group_kind=group_kind,
        group_id=group_id,
        round_id=round_id,
        branch_id=request_id,
        parent_branch_id=None,
        state_kind=state_kind,
        cache_domain=cache_domain,
        segments=tuple(segments),
    )


def make_key(
    context: KMeshRequestContext, *, active_length: int = 16, **overrides: object
) -> KMeshCompatibilityKey:
    values = {
        "execution_id": context.execution_id,
        "group_id": context.group_id,
        "round_id": context.round_id,
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
        "cache_domain": context.cache_domain,
        "active_prompt_length": active_length,
        "cached_span": (0, 4),
        "prefill_chunk_span": (4, active_length),
    }
    values.update(overrides)
    return KMeshCompatibilityKey(**values)  # type: ignore[arg-type]


def make_prompt_token_ids(
    context: KMeshRequestContext, active_length: int
) -> tuple[int, ...]:
    token_ids = [1000 + index for index in range(active_length)]
    for segment in context.segments:
        if segment.shareable and segment.token_end <= active_length:
            token_ids[segment.token_start : segment.token_end] = range(
                100, 100 + segment.token_end - segment.token_start
            )
    return tuple(token_ids)


def observation(
    request_id: str,
    *,
    context: KMeshRequestContext | None = None,
    key: KMeshCompatibilityKey | None | object = _DEFAULT_KEY,
    prompt_token_ids: Sequence[int] | None | object = _DEFAULT_PROMPT_TOKEN_IDS,
) -> KMeshRequestObservation:
    context = context or make_context(request_id)
    compatibility_key = make_key(context) if key is _DEFAULT_KEY else key
    if prompt_token_ids is _DEFAULT_PROMPT_TOKEN_IDS:
        prompt_token_ids = (
            make_prompt_token_ids(context, compatibility_key.active_prompt_length)
            if isinstance(compatibility_key, KMeshCompatibilityKey)
            else None
        )
    return KMeshRequestObservation.from_context(
        request_id,
        context,
        compatibility_key,  # type: ignore[arg-type]
        prompt_token_ids,  # type: ignore[arg-type]
    )


def clock(values: list[int]) -> Callable[[], int]:
    iterator = iter(values)
    return iterator.__next__


def planner(
    *clock_values: int, block_size: int = 4, max_group_size: int = 10
) -> KMeshPlanner:
    clock_ns = clock(list(clock_values or (100, 125)))
    return KMeshPlanner(
        block_size=block_size,
        max_group_size=max_group_size,
        clock_ns=clock_ns,
    )


@pytest.mark.parametrize("request_count", [1, 3, 5, 10])
def test_same_step_planning_never_forms_singletons(request_count: int):
    observations = tuple(
        observation(f"request-{index}") for index in range(request_count)
    )

    batch = planner().observe(observations)

    assert batch.metrics.observed_requests == request_count
    assert batch.metrics.candidate_requests == request_count
    if request_count == 1:
        assert batch.plans == ()
        assert batch.metrics.group_sizes == ()
        assert (
            batch.request_outcomes[0].fallback_reason
            is KMeshFallbackReason.SINGLETON_GROUP
        )
    else:
        assert len(batch.plans) == 1
        assert batch.metrics.group_sizes == (request_count,)
        plan = batch.plans[0]
        assert len(plan.group.request_ids) == request_count
        assert plan.group.master_candidate_ids == plan.group.request_ids
        assert all(outcome.is_grouped for outcome in batch.request_outcomes)


def test_empty_scheduler_step_emits_an_empty_structured_observation():
    batch = planner().observe(())

    assert batch.plans == ()
    assert batch.request_outcomes == ()
    assert batch.metrics.observed_requests == 0
    assert batch.metrics.candidate_requests == 0
    assert batch.metrics.formed_groups == 0
    assert batch.metrics.block_size == 4
    assert batch.metrics.max_group_size == 10
    assert batch.metrics.opportunity_scope == KMESH_SELECTED_PLAN_OPPORTUNITY_SCOPE
    assert batch.metrics.to_primitive()["opportunity_scope"] == (
        "deterministic_selected_plan"
    )


def test_max_group_size_splits_deterministically_without_a_singleton_plan():
    observations = tuple(observation(f"request-{index:02}") for index in range(11))

    batch = planner(max_group_size=10).observe(tuple(reversed(observations)))

    assert batch.metrics.group_sizes == (2, 9)
    assert all(len(plan.group.request_ids) <= 10 for plan in batch.plans)
    assert all(outcome.is_grouped for outcome in batch.request_outcomes)
    assert {
        request_id for plan in batch.plans for request_id in plan.group.request_ids
    } == {observation.request_id for observation in observations}


def test_tight_group_cap_leaves_an_explicit_singleton_fallback():
    batch = planner(max_group_size=2).observe(
        tuple(observation(f"request-{index}") for index in range(3))
    )

    assert batch.metrics.group_sizes == (2,)
    assert sum(outcome.is_grouped for outcome in batch.request_outcomes) == 2
    assert (
        sum(
            outcome.fallback_reason is KMeshFallbackReason.SINGLETON_GROUP
            for outcome in batch.request_outcomes
        )
        == 1
    )


@pytest.mark.parametrize("max_group_size", [0, 1, True])
def test_invalid_max_group_size_is_rejected(max_group_size: int):
    with pytest.raises(ValueError, match="max_group_size"):
        KMeshPlanner(block_size=4, max_group_size=max_group_size)


def test_offsets_and_segment_order_are_preserved_in_explicit_request_maps():
    first_context = make_context(
        "request-a", active_length=16, shared_start=0, shared_end=8
    )
    second_context = make_context(
        "request-b", active_length=16, shared_start=8, shared_end=16
    )
    observations = (
        observation("request-a", context=first_context, key=make_key(first_context)),
        observation("request-b", context=second_context, key=make_key(second_context)),
    )

    assert observations[0].prompt_token_ids is not None
    assert observations[1].prompt_token_ids is not None
    assert (
        observations[0].prompt_token_ids[0:8]
        == (observations[1].prompt_token_ids[8:16])
    )

    plan = planner().observe(observations).plans[0]

    assert len(plan.group.shared_segments) == 1
    first_entry = plan.request_segment_maps[0].entries[0]
    second_entry = plan.request_segment_maps[1].entries[0]
    assert (
        first_entry.token_start,
        first_entry.token_end,
        first_entry.logical_order,
    ) == (
        0,
        8,
        0,
    )
    assert (
        second_entry.token_start,
        second_entry.token_end,
        second_entry.logical_order,
    ) == (8, 16, 1)
    assert first_entry.shared_segment == second_entry.shared_segment
    assert first_entry.shared_segment.content_digest.startswith("sha256-v1:")


def test_same_declared_segment_with_different_tokens_fails_closed():
    context_a = make_context("request-a")
    context_b = make_context("request-b")
    prompt_a = make_prompt_token_ids(context_a, 16)
    prompt_b = list(make_prompt_token_ids(context_b, 16))
    prompt_b[context_b.segments[1].token_start] += 1

    batch = planner().observe(
        (
            observation(
                "request-a",
                context=context_a,
                key=make_key(context_a),
                prompt_token_ids=prompt_a,
            ),
            observation(
                "request-b",
                context=context_b,
                key=make_key(context_b),
                prompt_token_ids=prompt_b,
            ),
        )
    )

    assert batch.plans == ()
    assert batch.metrics.candidate_requests == 2
    assert all(
        outcome.fallback_reason is KMeshFallbackReason.SEGMENT_CONTENT_MISMATCH
        for outcome in batch.request_outcomes
    )
    assert {
        failure.reason: failure.count
        for failure in batch.metrics.compatibility_failures
    } == {KMeshFallbackReason.SEGMENT_CONTENT_MISMATCH: 1}
    assert batch.metrics.shared_segments == 0
    assert batch.metrics.prospective_reused_tokens == 0


def test_content_mismatch_rejects_only_the_outlier_from_a_safe_subset():
    contexts = {
        request_id: make_context(request_id)
        for request_id in ("request-a", "request-b", "request-c")
    }
    outlier_prompt = list(make_prompt_token_ids(contexts["request-c"], 16))
    outlier_prompt[contexts["request-c"].segments[1].token_start] += 1

    batch = planner().observe(
        (
            observation("request-a", context=contexts["request-a"]),
            observation("request-b", context=contexts["request-b"]),
            observation(
                "request-c",
                context=contexts["request-c"],
                prompt_token_ids=outlier_prompt,
            ),
        )
    )

    assert len(batch.plans) == 1
    assert batch.plans[0].group.request_ids == ("request-a", "request-b")
    assert (
        batch.outcome_for("request-c").fallback_reason
        is KMeshFallbackReason.SEGMENT_CONTENT_MISMATCH
    )
    assert {
        failure.reason: failure.count
        for failure in batch.metrics.compatibility_failures
    } == {KMeshFallbackReason.SEGMENT_CONTENT_MISMATCH: 2}


def test_missing_shared_segment_forms_safe_subset_and_rejects_remainder():
    missing = make_context("request-c", shared_id="different-update")
    batch = planner().observe(
        (
            observation("request-a"),
            observation("request-b"),
            observation("request-c", context=missing, key=make_key(missing)),
        )
    )

    assert len(batch.plans) == 1
    assert batch.plans[0].group.request_ids == ("request-a", "request-b")
    assert (
        batch.outcome_for("request-c").fallback_reason
        is KMeshFallbackReason.NO_SHARED_SEGMENTS
    )


def test_requests_without_a_declared_shareable_segment_do_not_group():
    first = make_context("request-a", shareable=False)
    second = make_context("request-b", shareable=False)

    batch = planner().observe(
        (
            observation("request-a", context=first, key=make_key(first)),
            observation("request-b", context=second, key=make_key(second)),
        )
    )

    assert batch.plans == ()
    assert batch.metrics.candidate_requests == 0
    assert all(
        outcome.fallback_reason is KMeshFallbackReason.NO_SHARED_SEGMENTS
        for outcome in batch.request_outcomes
    )


def test_lone_request_without_a_shareable_segment_is_not_a_candidate():
    context = make_context("request-a", shareable=False)

    batch = planner().observe(
        (observation("request-a", context=context, key=make_key(context)),)
    )

    assert batch.plans == ()
    assert batch.metrics.candidate_requests == 0
    assert (
        batch.outcome_for("request-a").fallback_reason
        is KMeshFallbackReason.NO_SHARED_SEGMENTS
    )


def test_disjoint_shared_cohorts_form_independent_nonoverlapping_groups():
    observations = []
    for request_id, shared_id in (
        ("request-a", "shared-x"),
        ("request-b", "shared-x"),
        ("request-c", "shared-y"),
        ("request-d", "shared-y"),
    ):
        context = make_context(request_id, shared_id=shared_id)
        observations.append(
            observation(request_id, context=context, key=make_key(context))
        )

    batch = planner().observe(tuple(observations))

    assert batch.metrics.group_sizes == (2, 2)
    assert {plan.group.request_ids for plan in batch.plans} == {
        ("request-a", "request-b"),
        ("request-c", "request-d"),
    }
    assert (
        len(
            {
                request_id
                for plan in batch.plans
                for request_id in plan.group.request_ids
            }
        )
        == 4
    )


def test_private_history_identity_differences_do_not_block_declared_sharing():
    first = make_context("request-a")
    second = make_context("request-b")
    assert first.segments[0].segment_id != second.segments[0].segment_id
    assert first.segments[0].lineage_id != second.segments[0].lineage_id

    batch = planner().observe(
        (
            observation("request-a", context=first, key=make_key(first)),
            observation("request-b", context=second, key=make_key(second)),
        )
    )

    assert batch.plans[0].group.request_ids == ("request-a", "request-b")
    assert tuple(
        segment.segment_id for segment in batch.plans[0].group.shared_segments
    ) == ("shared-update-1",)


@pytest.mark.parametrize(
    ("variant", "reason"),
    [
        ("round", KMeshFallbackReason.ROUND_MISMATCH),
        ("domain", KMeshFallbackReason.CACHE_DOMAIN_MISMATCH),
        ("adapter", KMeshFallbackReason.LORA_MISMATCH),
        ("active_length", KMeshFallbackReason.ACTIVE_LENGTH_MISMATCH),
        ("cached_span", KMeshFallbackReason.CACHED_SPAN_MISMATCH),
        ("prefill_span", KMeshFallbackReason.PREFILL_CHUNK_MISMATCH),
    ],
)
def test_mixed_safety_evidence_never_enters_a_group(
    variant: str, reason: KMeshFallbackReason
):
    context = make_context("request-c")
    key_overrides: dict[str, object] = {}
    if variant == "round":
        context = make_context("request-c", round_id=3)
    elif variant == "domain":
        context = make_context("request-c", cache_domain="tenant-b")
    elif variant == "adapter":
        key_overrides["lora_adapter_identity"] = "adapter-b"
    elif variant == "active_length":
        key_overrides.update(
            active_prompt_length=20,
            prefill_chunk_span=(4, 20),
        )
    elif variant == "cached_span":
        key_overrides.update(
            cached_span=(0, 5),
            prefill_chunk_span=(5, 16),
        )
    elif variant == "prefill_span":
        key_overrides["prefill_chunk_span"] = (4, 15)
    outlier_key = make_key(context, **key_overrides)  # type: ignore[arg-type]

    batch = planner().observe(
        (
            observation("request-a"),
            observation("request-b"),
            observation("request-c", context=context, key=outlier_key),
        )
    )

    assert batch.plans[0].group.request_ids == ("request-a", "request-b")
    assert batch.outcome_for("request-c").fallback_reason is reason
    failures = {
        failure.reason: failure.count
        for failure in batch.metrics.compatibility_failures
    }
    assert failures[reason] == 2


def test_missing_or_domainless_evidence_fails_closed():
    domainless_context = make_context("request-b", cache_domain=None)
    batch = planner().observe(
        (
            observation("request-a", key=None),
            observation(
                "request-b",
                context=domainless_context,
                key=make_key(domainless_context),
            ),
        )
    )

    assert batch.plans == ()
    assert batch.metrics.candidate_requests == 0
    assert all(
        outcome.fallback_reason is KMeshFallbackReason.MISSING_COMPATIBILITY_EVIDENCE
        for outcome in batch.request_outcomes
    )


def test_missing_prompt_token_evidence_fails_closed_before_candidacy():
    batch = planner().observe((observation("request-a", prompt_token_ids=None),))

    assert batch.plans == ()
    assert batch.metrics.candidate_requests == 0
    assert (
        batch.outcome_for("request-a").fallback_reason
        is KMeshFallbackReason.MISSING_PROMPT_TOKEN_EVIDENCE
    )


def test_prompt_token_length_mismatch_fails_closed_and_is_counted():
    batch = planner().observe((observation("request-a", prompt_token_ids=(0,) * 15),))

    assert batch.plans == ()
    assert batch.metrics.candidate_requests == 0
    assert (
        batch.outcome_for("request-a").fallback_reason
        is KMeshFallbackReason.PROMPT_TOKEN_LENGTH_MISMATCH
    )
    assert batch.metrics.compatibility_failures[0].to_primitive() == {
        "reason": "prompt_token_length_mismatch",
        "count": 1,
    }


def test_observation_normalizes_and_validates_prompt_token_evidence():
    context = make_context("request-a")
    key = make_key(context)
    normalized = KMeshRequestObservation.from_context(
        "request-a", context, key, range(16)
    )

    assert normalized.prompt_token_ids == tuple(range(16))
    with pytest.raises(TypeError, match="immutable tuple"):
        KMeshRequestObservation(
            "request-a",
            normalized.context_result,
            key,
            list(range(16)),  # type: ignore[arg-type]
        )
    for invalid in ((0, True), (0, -1)):
        with pytest.raises(ValueError, match="nonnegative integer"):
            KMeshRequestObservation.from_context("request-a", context, key, invalid)


def test_runtime_evidence_must_match_request_declared_grouping_fields():
    context = make_context("request-a", cache_domain="tenant-a")
    conflicting_key = make_key(context, cache_domain="tenant-b")

    batch = planner().observe(
        (observation("request-a", context=context, key=conflicting_key),)
    )

    assert batch.plans == ()
    assert (
        batch.outcome_for("request-a").fallback_reason
        is KMeshFallbackReason.CACHE_DOMAIN_MISMATCH
    )
    assert batch.metrics.compatibility_failures[0].reason is (
        KMeshFallbackReason.CACHE_DOMAIN_MISMATCH
    )


def test_runtime_active_length_must_bound_every_normalized_segment():
    context = make_context("request-a", active_length=20, shared_start=8, shared_end=20)
    short_key = make_key(context, active_length=16)

    batch = planner().observe(
        (observation("request-a", context=context, key=short_key),)
    )

    assert batch.plans == ()
    assert (
        batch.outcome_for("request-a").fallback_reason
        is KMeshFallbackReason.ACTIVE_LENGTH_MISMATCH
    )


def test_fallback_precedence_is_parse_then_semantics_then_evidence():
    disabled = make_context(
        "disabled",
        enabled=False,
        group_kind=KMeshGroupKind.COT_BRANCH,
        state_kind=KMeshStateKind.PLAN,
    )
    unsupported_group = make_context(
        "group", group_kind=KMeshGroupKind.COT_BRANCH, state_kind=KMeshStateKind.PLAN
    )
    unsupported_state = make_context("state", state_kind=KMeshStateKind.PLAN)
    invalid = KMeshRequestContextParseResult.invalid(
        KMeshMetadataError(
            KMeshMetadataErrorCode.INVALID_TYPE,
            "kmesh",
            "malformed",
        )
    )
    observations = (
        observation("disabled", context=disabled, key=None),
        KMeshRequestObservation(
            "no-metadata", KMeshRequestContextParseResult.no_metadata(), None
        ),
        KMeshRequestObservation("invalid", invalid, None),
        observation("group", context=unsupported_group, key=None),
        observation("state", context=unsupported_state, key=None),
        observation("missing-evidence", key=None),
    )

    batch = planner().observe(observations)

    assert batch.outcome_for("disabled").fallback_reason is KMeshFallbackReason.DISABLED
    assert (
        batch.outcome_for("no-metadata").fallback_reason
        is KMeshFallbackReason.NO_METADATA
    )
    assert (
        batch.outcome_for("invalid").fallback_reason
        is KMeshFallbackReason.INVALID_METADATA
    )
    assert (
        batch.outcome_for("group").fallback_reason
        is KMeshFallbackReason.GROUP_KIND_UNSUPPORTED
    )
    assert (
        batch.outcome_for("state").fallback_reason
        is KMeshFallbackReason.STATE_KIND_UNSUPPORTED
    )
    assert (
        batch.outcome_for("missing-evidence").fallback_reason
        is KMeshFallbackReason.MISSING_COMPATIBILITY_EVIDENCE
    )


def test_multiple_mismatches_use_stable_round_first_precedence():
    first_context = make_context("request-a", round_id=2, cache_domain="tenant-a")
    second_context = make_context("request-b", round_id=3, cache_domain="tenant-b")
    first_key = make_key(first_context)
    second_key = make_key(second_context, lora_adapter_identity="adapter-b")

    batch = planner().observe(
        (
            observation("request-a", context=first_context, key=first_key),
            observation("request-b", context=second_context, key=second_key),
        )
    )

    assert batch.plans == ()
    assert all(
        outcome.fallback_reason is KMeshFallbackReason.ROUND_MISMATCH
        for outcome in batch.request_outcomes
    )


def test_structural_block_overlap_requires_matching_segment_relative_alignment():
    aligned = make_context("request-a", active_length=16, shared_start=0, shared_end=8)
    unaligned = make_context(
        "request-b", active_length=16, shared_start=1, shared_end=9
    )
    batch = planner(block_size=4).observe(
        (
            observation("request-a", context=aligned, key=make_key(aligned)),
            observation("request-b", context=unaligned, key=make_key(unaligned)),
        )
    )

    metrics = batch.metrics
    assert metrics.block_size == 4
    assert metrics.shared_segments == 1
    assert metrics.shared_tokens == 8
    assert metrics.prospective_reused_tokens == 8
    assert metrics.structural_full_blocks == 0
    assert metrics.structural_full_block_tokens == 0
    assert metrics.prospective_reused_full_blocks == 0
    assert metrics.prospective_reused_full_block_tokens == 0


def test_structural_block_overlap_counts_common_relative_full_blocks():
    first = make_context("request-a", active_length=16, shared_start=0, shared_end=8)
    moved_by_one_block = make_context(
        "request-b", active_length=16, shared_start=4, shared_end=12
    )

    metrics = (
        planner(block_size=4)
        .observe(
            (
                observation("request-a", context=first, key=make_key(first)),
                observation(
                    "request-b",
                    context=moved_by_one_block,
                    key=make_key(moved_by_one_block),
                ),
            )
        )
        .metrics
    )

    assert metrics.structural_full_blocks == 2
    assert metrics.structural_full_block_tokens == 8
    assert metrics.prospective_reused_full_blocks == 2
    assert metrics.prospective_reused_full_block_tokens == 8


def test_plan_ids_and_pickled_transport_are_deterministic_across_input_order():
    observations = (
        observation("request-c"),
        observation("request-a"),
        observation("request-b"),
    )
    deterministic_planner = planner(100, 140, 200, 240)

    first = deterministic_planner.observe(observations)
    second = deterministic_planner.observe(tuple(reversed(observations)))

    assert first == second
    assert bytes(ForkingPickler.dumps(first)) == bytes(ForkingPickler.dumps(second))
    assert (
        msgspec.msgpack.decode(msgspec.msgpack.encode(first.to_primitive()))
        == first.to_primitive()
    )
    primitive = first.to_primitive()
    assert primitive["metrics"]["opportunity_scope"] == ("deterministic_selected_plan")
    assert all(
        metric.opportunity_scope == KMESH_SELECTED_PLAN_OPPORTUNITY_SCOPE
        for metric in first.metrics.group_metrics
    )
    assert primitive["plans"][0]["group"]["shared_segments"][0][
        "content_digest"
    ].startswith("sha256-v1:")
    assert first.metrics.planner_overhead_ns == 40


def test_content_digest_changes_group_and_plan_ids_deterministically():
    contexts = {
        request_id: make_context(request_id)
        for request_id in ("request-a", "request-b")
    }

    def observations_with_shared_token(shared_token: int):
        observations = []
        for request_id, context in contexts.items():
            prompt = list(make_prompt_token_ids(context, 16))
            prompt[context.segments[1].token_start] = shared_token
            observations.append(
                observation(
                    request_id,
                    context=context,
                    prompt_token_ids=prompt,
                )
            )
        return tuple(observations)

    deterministic_planner = planner(100, 120, 200, 220)
    first = deterministic_planner.observe(observations_with_shared_token(700))
    second = deterministic_planner.observe(observations_with_shared_token(701))

    assert len(first.plans) == len(second.plans) == 1
    assert first.plans[0].group.reuse_group_id != (second.plans[0].group.reuse_group_id)
    assert first.plans[0].plan_id != second.plans[0].plan_id
    assert first.plans[0].group.shared_segments[0].content_digest != (
        second.plans[0].group.shared_segments[0].content_digest
    )


def test_unrelated_cancellation_does_not_perturb_existing_plan_identity():
    primary = (observation("request-a"), observation("request-b"))
    unrelated_context = make_context(
        "cancelled", execution_id="execution-2", group_id="gather-2"
    )
    cancelled = observation(
        "cancelled",
        context=unrelated_context,
        key=make_key(unrelated_context),
    )
    deterministic_planner = planner(100, 110, 200, 210)

    before_cancellation = deterministic_planner.observe((*primary, cancelled))
    after_cancellation = deterministic_planner.observe(primary)

    assert before_cancellation.plans == after_cancellation.plans
    assert before_cancellation.plans[0].plan_id == after_cancellation.plans[0].plan_id
    assert (
        before_cancellation.plans[0].group.reuse_group_id
        == after_cancellation.plans[0].group.reuse_group_id
    )
