# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from vllm.v1.kmesh.grouping import (
    KMeshCompatibilityKey,
    KMeshReuseGroup,
    KMeshSharedSegmentIdentity,
    compatibility_mismatch_reason,
    compatibility_reason_precedence,
    context_key_mismatch_reason,
)
from vllm.v1.kmesh.metrics import (
    KMeshGroupObserverMetrics,
    KMeshObserverMetrics,
    KMeshProspectiveMembership,
    KMeshReasonCount,
)
from vllm.v1.kmesh.types import (
    KMeshFallbackReason,
    KMeshGroupKind,
    KMeshRequestContext,
    KMeshRequestContextParseResult,
    KMeshSegment,
    KMeshStateKind,
    _is_strict_int,
    _validate_identifier,
)


@dataclass(frozen=True, slots=True)
class KMeshRequestObservation:
    """One same-step request plus metadata, runtime, and prompt-token evidence."""

    request_id: str
    context_result: KMeshRequestContextParseResult
    compatibility_key: KMeshCompatibilityKey | None
    prompt_token_ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        _validate_identifier(self.request_id, "request_id")
        if not isinstance(self.context_result, KMeshRequestContextParseResult):
            raise TypeError("context_result must be KMeshRequestContextParseResult")
        if self.compatibility_key is not None and not isinstance(
            self.compatibility_key, KMeshCompatibilityKey
        ):
            raise TypeError("compatibility_key must be KMeshCompatibilityKey or null")
        if self.prompt_token_ids is not None:
            if not isinstance(self.prompt_token_ids, tuple):
                raise TypeError("prompt_token_ids must be an immutable tuple or null")
            for index, token_id in enumerate(self.prompt_token_ids):
                if not _is_strict_int(token_id) or token_id < 0:
                    raise ValueError(
                        f"prompt_token_ids[{index}] must be a nonnegative integer"
                    )

    @classmethod
    def from_context(
        cls,
        request_id: str,
        context: KMeshRequestContext,
        compatibility_key: KMeshCompatibilityKey | None,
        prompt_token_ids: Sequence[int] | None = None,
    ) -> KMeshRequestObservation:
        result = (
            KMeshRequestContextParseResult.accepted(context)
            if context.enabled
            else KMeshRequestContextParseResult.disabled(context)
        )
        return cls(
            request_id,
            result,
            compatibility_key,
            tuple(prompt_token_ids) if prompt_token_ids is not None else None,
        )


@dataclass(frozen=True, slots=True)
class KMeshSegmentMapEntry:
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
            raise ValueError("mapped range must match the shared segment length")

    def to_primitive(self) -> dict[str, Any]:
        return {
            "shared_segment": self.shared_segment.to_primitive(),
            "token_start": self.token_start,
            "token_end": self.token_end,
            "logical_order": self.logical_order,
        }


@dataclass(frozen=True, slots=True)
class KMeshRequestSegmentMap:
    request_id: str
    entries: tuple[KMeshSegmentMapEntry, ...]

    def __post_init__(self) -> None:
        _validate_identifier(self.request_id, "request_id")
        if not self.entries:
            raise ValueError("request segment maps must not be empty")
        identities = tuple(entry.shared_segment for entry in self.entries)
        if len(identities) != len(set(identities)):
            raise ValueError("request segment maps must not repeat identities")

    def to_primitive(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "entries": [entry.to_primitive() for entry in self.entries],
        }


@dataclass(frozen=True, slots=True)
class KMeshReusePlan:
    """Immutable group execution plan with ordered request-relative mappings.

    The value is executable by a qualified backend but carries no tensors and
    performs no KV mutation by itself, so observe mode can transport it safely.
    """

    plan_id: str
    group: KMeshReuseGroup
    request_segment_maps: tuple[KMeshRequestSegmentMap, ...]
    block_size: int

    def __post_init__(self) -> None:
        _validate_identifier(self.plan_id, "plan_id")
        if not isinstance(self.group, KMeshReuseGroup):
            raise TypeError("group must be KMeshReuseGroup")
        if not _is_strict_int(self.block_size) or self.block_size <= 0:
            raise ValueError("block_size must be a positive integer")
        request_ids = tuple(mapping.request_id for mapping in self.request_segment_maps)
        if request_ids != self.group.request_ids:
            raise ValueError("segment maps must follow the group's request order")
        expected_identities = self.group.shared_segments
        for mapping in self.request_segment_maps:
            identities = tuple(entry.shared_segment for entry in mapping.entries)
            if identities != expected_identities:
                raise ValueError("every request map must cover all shared segments")

    def to_primitive(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "group": self.group.to_primitive(),
            "request_segment_maps": [
                mapping.to_primitive() for mapping in self.request_segment_maps
            ],
            "block_size": self.block_size,
        }


@dataclass(frozen=True, slots=True)
class KMeshPlanBatch:
    plans: tuple[KMeshReusePlan, ...]
    request_outcomes: tuple[KMeshProspectiveMembership, ...]
    metrics: KMeshObserverMetrics

    def __post_init__(self) -> None:
        if self.request_outcomes != self.metrics.prospective_membership:
            raise ValueError("batch outcomes must match observer membership metrics")
        plan_ids = tuple(plan.plan_id for plan in self.plans)
        if plan_ids != tuple(sorted(set(plan_ids))):
            raise ValueError("plans must have unique, sorted identities")
        expected_memberships: dict[str, tuple[str, str]] = {}
        for plan in self.plans:
            for request_id in plan.group.request_ids:
                if request_id in expected_memberships:
                    raise ValueError("a request must not enter multiple plans")
                expected_memberships[request_id] = (
                    plan.group.reuse_group_id,
                    plan.plan_id,
                )
        actual_memberships = {
            outcome.request_id: (outcome.reuse_group_id, outcome.plan_id)
            for outcome in self.request_outcomes
            if outcome.is_grouped
        }
        if actual_memberships != expected_memberships:
            raise ValueError("request outcomes must describe every plan member")

    def outcome_for(self, request_id: str) -> KMeshProspectiveMembership:
        for outcome in self.request_outcomes:
            if outcome.request_id == request_id:
                return outcome
        raise KeyError(request_id)

    def to_primitive(self) -> dict[str, Any]:
        return {
            "plans": [plan.to_primitive() for plan in self.plans],
            "request_outcomes": [
                outcome.to_primitive() for outcome in self.request_outcomes
            ],
            "metrics": self.metrics.to_primitive(),
        }


@dataclass(frozen=True, slots=True)
class _EligibleRequest:
    observation: KMeshRequestObservation
    context: KMeshRequestContext
    key: KMeshCompatibilityKey
    prompt_token_ids: tuple[int, ...]

    @property
    def request_id(self) -> str:
        return self.observation.request_id


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _content_id(prefix: str, value: object) -> str:
    digest = hashlib.sha256(_canonical_json(value).encode()).hexdigest()
    return f"{prefix}-{digest}"


def _shareable_segments(
    record: _EligibleRequest,
) -> dict[KMeshSharedSegmentIdentity, KMeshSegment]:
    return {
        KMeshSharedSegmentIdentity.from_segment(
            segment, record.prompt_token_ids
        ): segment
        for segment in record.context.segments
        if segment.shareable
    }


def _common_shared_segments(
    records: Sequence[_EligibleRequest],
    segment_maps: dict[str, dict[KMeshSharedSegmentIdentity, KMeshSegment]],
) -> tuple[KMeshSharedSegmentIdentity, ...]:
    common = set(segment_maps[records[0].request_id])
    for record in records[1:]:
        common.intersection_update(segment_maps[record.request_id])
    return tuple(sorted(common, key=lambda segment: segment.sort_key()))


def _select_disjoint_groups(
    records: Sequence[_EligibleRequest],
    max_group_size: int,
) -> list[tuple[tuple[_EligibleRequest, ...], tuple[KMeshSharedSegmentIdentity, ...]]]:
    segment_maps = {
        record.request_id: _shareable_segments(record) for record in records
    }
    remaining = {record.request_id: record for record in records}
    selected: list[
        tuple[tuple[_EligibleRequest, ...], tuple[KMeshSharedSegmentIdentity, ...]]
    ] = []

    while len(remaining) >= 2:
        membership_by_segment: dict[KMeshSharedSegmentIdentity, set[str]] = defaultdict(
            set
        )
        for request_id in remaining:
            for identity in segment_maps[request_id]:
                membership_by_segment[identity].add(request_id)

        memberships = {
            tuple(sorted(request_ids))
            for request_ids in membership_by_segment.values()
            if len(request_ids) >= 2
        }
        candidates = []
        for request_ids in memberships:
            candidate_records = tuple(
                remaining[request_id] for request_id in request_ids
            )
            shared_segments = _common_shared_segments(candidate_records, segment_maps)
            if shared_segments:
                candidates.append((candidate_records, shared_segments))
        if not candidates:
            break
        candidates.sort(
            key=lambda candidate: (
                -len(candidate[0]),
                -sum(segment.token_length for segment in candidate[1]),
                tuple(record.request_id for record in candidate[0]),
                tuple(segment.sort_key() for segment in candidate[1]),
            )
        )
        chosen_records, _ = candidates[0]
        grouped_records: list[_EligibleRequest] = []
        pending = list(chosen_records)
        while len(pending) > max_group_size:
            group_size = max_group_size
            if len(pending) - group_size == 1 and max_group_size > 2:
                group_size -= 1
            bounded_records = tuple(pending[:group_size])
            del pending[:group_size]
            selected.append(
                (
                    bounded_records,
                    _common_shared_segments(bounded_records, segment_maps),
                )
            )
            grouped_records.extend(bounded_records)
        if len(pending) >= 2:
            bounded_records = tuple(pending)
            selected.append(
                (
                    bounded_records,
                    _common_shared_segments(bounded_records, segment_maps),
                )
            )
            grouped_records.extend(bounded_records)
        for record in grouped_records:
            del remaining[record.request_id]

    return selected


def _make_segment_maps(
    records: Sequence[_EligibleRequest],
    shared_segments: tuple[KMeshSharedSegmentIdentity, ...],
) -> tuple[KMeshRequestSegmentMap, ...]:
    request_maps = []
    for record in records:
        by_identity = _shareable_segments(record)
        entries = tuple(
            KMeshSegmentMapEntry(
                shared_segment=identity,
                token_start=by_identity[identity].token_start,
                token_end=by_identity[identity].token_end,
                logical_order=by_identity[identity].logical_order,
            )
            for identity in shared_segments
        )
        request_maps.append(KMeshRequestSegmentMap(record.request_id, entries))
    return tuple(request_maps)


def _has_shared_peer(
    record: _EligibleRequest, peers: Sequence[_EligibleRequest]
) -> bool:
    identities = set(_shareable_segments(record))
    return any(
        identities.intersection(_shareable_segments(peer))
        for peer in peers
        if peer.request_id != record.request_id
    )


def _segment_content_mismatch(left: _EligibleRequest, right: _EligibleRequest) -> bool:
    left_identities = _shareable_segments(left)
    right_by_declaration = {
        identity.declaration_key(): identity for identity in _shareable_segments(right)
    }
    for identity in left_identities:
        right_identity = right_by_declaration.get(identity.declaration_key())
        if (
            right_identity is not None
            and identity.content_digest != right_identity.content_digest
        ):
            return True
    return False


def _has_content_mismatch(
    record: _EligibleRequest, peers: Sequence[_EligibleRequest]
) -> bool:
    return any(
        _segment_content_mismatch(record, peer)
        for peer in peers
        if peer.request_id != record.request_id
    )


def _full_block_offsets(entry: KMeshSegmentMapEntry, block_size: int) -> frozenset[int]:
    first_full_token = (entry.token_start + block_size - 1) // block_size * block_size
    return frozenset(
        token_start - entry.token_start
        for token_start in range(first_full_token, entry.token_end, block_size)
        if token_start + block_size <= entry.token_end
    )


def _group_metrics(plan: KMeshReusePlan) -> KMeshGroupObserverMetrics:
    request_count = len(plan.group.request_ids)
    shared_segments = len(plan.group.shared_segments)
    shared_tokens = sum(segment.token_length for segment in plan.group.shared_segments)
    structural_full_blocks = 0
    for index in range(shared_segments):
        common_offsets = set(
            _full_block_offsets(
                plan.request_segment_maps[0].entries[index], plan.block_size
            )
        )
        for mapping in plan.request_segment_maps[1:]:
            common_offsets.intersection_update(
                _full_block_offsets(mapping.entries[index], plan.block_size)
            )
        structural_full_blocks += len(common_offsets)
    reuse_multiplier = request_count - 1
    return KMeshGroupObserverMetrics(
        reuse_group_id=plan.group.reuse_group_id,
        request_count=request_count,
        block_size=plan.block_size,
        shared_segments=shared_segments,
        shared_tokens=shared_tokens,
        prospective_reused_segments=shared_segments * reuse_multiplier,
        prospective_reused_tokens=shared_tokens * reuse_multiplier,
        structural_full_blocks=structural_full_blocks,
        structural_full_block_tokens=structural_full_blocks * plan.block_size,
        prospective_reused_full_blocks=structural_full_blocks * reuse_multiplier,
        prospective_reused_full_block_tokens=(
            structural_full_blocks * plan.block_size * reuse_multiplier
        ),
    )


def _reason_counts(
    counts: Counter[KMeshFallbackReason], *, compatibility_order: bool = False
) -> tuple[KMeshReasonCount, ...]:
    if compatibility_order:
        key = lambda item: (compatibility_reason_precedence(item[0]), item[0].value)
    else:
        enum_order = {reason: index for index, reason in enumerate(KMeshFallbackReason)}
        key = lambda item: (enum_order[item[0]], item[0].value)
    return tuple(
        KMeshReasonCount(reason, count)
        for reason, count in sorted(counts.items(), key=key)
        if count
    )


class KMeshPlanner:
    """Stateless group planner for requests in one scheduler step."""

    def __init__(
        self,
        *,
        block_size: int,
        max_group_size: int = 10,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        if not _is_strict_int(block_size) or block_size <= 0:
            raise ValueError("block_size must be a positive integer")
        if not _is_strict_int(max_group_size) or max_group_size < 2:
            raise ValueError("max_group_size must be an integer of at least two")
        self._block_size = block_size
        self._max_group_size = max_group_size
        self._clock_ns = clock_ns

    def observe(
        self, observations: Sequence[KMeshRequestObservation]
    ) -> KMeshPlanBatch:
        """Build transportable plans without changing scheduling or runtime state."""
        start_ns = self._clock_ns()
        ordered = tuple(sorted(observations, key=lambda item: item.request_id))
        request_ids = tuple(observation.request_id for observation in ordered)
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("request IDs must be unique within a planner step")

        outcomes: dict[str, KMeshProspectiveMembership] = {}
        eligible: list[_EligibleRequest] = []
        compatibility_failures: Counter[KMeshFallbackReason] = Counter()

        for observation in ordered:
            result = observation.context_result
            if not result.is_enabled:
                reason = result.fallback_reason or KMeshFallbackReason.INTERNAL_ERROR
                outcomes[observation.request_id] = KMeshProspectiveMembership(
                    observation.request_id, None, None, reason
                )
                continue
            context = result.context
            if context is None:
                outcomes[observation.request_id] = KMeshProspectiveMembership(
                    observation.request_id,
                    None,
                    None,
                    KMeshFallbackReason.INTERNAL_ERROR,
                )
                continue
            if context.group_kind is not KMeshGroupKind.AGENT_ROUND:
                outcomes[observation.request_id] = KMeshProspectiveMembership(
                    observation.request_id,
                    None,
                    None,
                    KMeshFallbackReason.GROUP_KIND_UNSUPPORTED,
                )
                continue
            if context.state_kind is not KMeshStateKind.KV_CACHE:
                outcomes[observation.request_id] = KMeshProspectiveMembership(
                    observation.request_id,
                    None,
                    None,
                    KMeshFallbackReason.STATE_KIND_UNSUPPORTED,
                )
                continue
            key = observation.compatibility_key
            if key is None or not key.has_complete_evidence:
                outcomes[observation.request_id] = KMeshProspectiveMembership(
                    observation.request_id,
                    None,
                    None,
                    KMeshFallbackReason.MISSING_COMPATIBILITY_EVIDENCE,
                )
                continue
            if mismatch_reason := context_key_mismatch_reason(context, key):
                compatibility_failures[mismatch_reason] += 1
                outcomes[observation.request_id] = KMeshProspectiveMembership(
                    observation.request_id, None, None, mismatch_reason
                )
                continue
            if not any(segment.shareable for segment in context.segments):
                outcomes[observation.request_id] = KMeshProspectiveMembership(
                    observation.request_id,
                    None,
                    None,
                    KMeshFallbackReason.NO_SHARED_SEGMENTS,
                )
                continue
            prompt_token_ids = observation.prompt_token_ids
            if prompt_token_ids is None:
                outcomes[observation.request_id] = KMeshProspectiveMembership(
                    observation.request_id,
                    None,
                    None,
                    KMeshFallbackReason.MISSING_PROMPT_TOKEN_EVIDENCE,
                )
                continue
            if len(prompt_token_ids) != key.active_prompt_length:
                reason = KMeshFallbackReason.PROMPT_TOKEN_LENGTH_MISMATCH
                compatibility_failures[reason] += 1
                outcomes[observation.request_id] = KMeshProspectiveMembership(
                    observation.request_id, None, None, reason
                )
                continue
            eligible.append(
                _EligibleRequest(observation, context, key, prompt_token_ids)
            )

        declared_cohorts: dict[tuple[str, str], list[_EligibleRequest]] = defaultdict(
            list
        )
        for record in eligible:
            declared_cohorts[
                (record.context.execution_id, record.context.group_id)
            ].append(record)

        for cohort in declared_cohorts.values():
            for left, right in combinations(cohort, 2):
                if mismatch_reason := compatibility_mismatch_reason(
                    left.key, right.key
                ):
                    compatibility_failures[mismatch_reason] += 1

        plans: list[KMeshReusePlan] = []
        for cohort_id in sorted(declared_cohorts):
            cohort = sorted(
                declared_cohorts[cohort_id], key=lambda record: record.request_id
            )
            exact_buckets: dict[KMeshCompatibilityKey, list[_EligibleRequest]] = (
                defaultdict(list)
            )
            for record in cohort:
                exact_buckets[record.key].append(record)

            for bucket in exact_buckets.values():
                for left, right in combinations(bucket, 2):
                    if _segment_content_mismatch(left, right):
                        compatibility_failures[
                            KMeshFallbackReason.SEGMENT_CONTENT_MISMATCH
                        ] += 1

            for key in sorted(
                exact_buckets, key=lambda item: _canonical_json(item.to_primitive())
            ):
                bucket = exact_buckets[key]
                for group_records, shared_segments in _select_disjoint_groups(
                    bucket, self._max_group_size
                ):
                    group_request_ids = tuple(
                        record.request_id for record in group_records
                    )
                    group_primitive = {
                        "schema": "kmesh-reuse-group-v1",
                        "compatibility_key": key.to_primitive(),
                        "request_ids": list(group_request_ids),
                        "shared_segments": [
                            segment.to_primitive() for segment in shared_segments
                        ],
                    }
                    reuse_group_id = _content_id("kmesh-group-v1", group_primitive)
                    group = KMeshReuseGroup(
                        reuse_group_id=reuse_group_id,
                        compatibility_key=key,
                        request_ids=group_request_ids,
                        shared_segments=shared_segments,
                        master_candidate_ids=group_request_ids,
                    )
                    segment_maps = _make_segment_maps(group_records, shared_segments)
                    plan_primitive = {
                        "schema": "kmesh-reuse-plan-v1",
                        "reuse_group_id": reuse_group_id,
                        "request_segment_maps": [
                            mapping.to_primitive() for mapping in segment_maps
                        ],
                        "block_size": self._block_size,
                    }
                    plan = KMeshReusePlan(
                        plan_id=_content_id("kmesh-plan-v1", plan_primitive),
                        group=group,
                        request_segment_maps=segment_maps,
                        block_size=self._block_size,
                    )
                    plans.append(plan)
                    for request_id in group_request_ids:
                        outcomes[request_id] = KMeshProspectiveMembership(
                            request_id,
                            group.reuse_group_id,
                            plan.plan_id,
                            None,
                        )

            for record in cohort:
                if record.request_id in outcomes:
                    continue
                exact_peers = exact_buckets[record.key]
                if len(exact_peers) > 1:
                    reason = (
                        KMeshFallbackReason.SEGMENT_CONTENT_MISMATCH
                        if _has_content_mismatch(record, exact_peers)
                        else (
                            KMeshFallbackReason.SINGLETON_GROUP
                            if _has_shared_peer(record, exact_peers)
                            else KMeshFallbackReason.NO_SHARED_SEGMENTS
                        )
                    )
                else:
                    mismatch_reasons = [
                        mismatch
                        for peer in cohort
                        if peer.request_id != record.request_id
                        and (
                            mismatch := compatibility_mismatch_reason(
                                record.key, peer.key
                            )
                        )
                        is not None
                    ]
                    reason = (
                        min(mismatch_reasons, key=compatibility_reason_precedence)
                        if mismatch_reasons
                        else KMeshFallbackReason.SINGLETON_GROUP
                    )
                outcomes[record.request_id] = KMeshProspectiveMembership(
                    record.request_id, None, None, reason
                )

        plans_tuple = tuple(sorted(plans, key=lambda plan: plan.plan_id))
        memberships = tuple(outcomes[request_id] for request_id in sorted(outcomes))
        group_metrics = tuple(
            sorted(
                (_group_metrics(plan) for plan in plans_tuple),
                key=lambda metric: metric.reuse_group_id,
            )
        )
        fallback_counts = Counter(
            membership.fallback_reason
            for membership in memberships
            if membership.fallback_reason is not None
        )
        end_ns = self._clock_ns()
        if not _is_strict_int(start_ns) or not _is_strict_int(end_ns):
            raise TypeError("clock_ns must return strict integers")
        if end_ns < start_ns:
            raise ValueError("clock_ns must be monotonic")

        metrics = KMeshObserverMetrics(
            observed_requests=len(ordered),
            candidate_requests=len(eligible),
            formed_groups=len(plans_tuple),
            block_size=self._block_size,
            max_group_size=self._max_group_size,
            group_sizes=tuple(sorted(metric.request_count for metric in group_metrics)),
            shared_segments=sum(metric.shared_segments for metric in group_metrics),
            shared_tokens=sum(metric.shared_tokens for metric in group_metrics),
            prospective_reused_segments=sum(
                metric.prospective_reused_segments for metric in group_metrics
            ),
            prospective_reused_tokens=sum(
                metric.prospective_reused_tokens for metric in group_metrics
            ),
            structural_full_blocks=sum(
                metric.structural_full_blocks for metric in group_metrics
            ),
            structural_full_block_tokens=sum(
                metric.structural_full_block_tokens for metric in group_metrics
            ),
            prospective_reused_full_blocks=sum(
                metric.prospective_reused_full_blocks for metric in group_metrics
            ),
            prospective_reused_full_block_tokens=sum(
                metric.prospective_reused_full_block_tokens for metric in group_metrics
            ),
            compatibility_failures=_reason_counts(
                compatibility_failures, compatibility_order=True
            ),
            fallback_counts=_reason_counts(fallback_counts),
            prospective_membership=memberships,
            group_metrics=group_metrics,
            planner_overhead_ns=end_ns - start_ns,
        )
        return KMeshPlanBatch(plans_tuple, memberships, metrics)
