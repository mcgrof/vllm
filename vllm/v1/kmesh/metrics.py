# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

from vllm.v1.kmesh.types import (
    KMeshFallbackReason,
    _is_strict_int,
    _validate_identifier,
)

KMESH_SELECTED_PLAN_OPPORTUNITY_SCOPE = "deterministic_selected_plan"


@dataclass(frozen=True, slots=True)
class KMeshReasonCount:
    reason: KMeshFallbackReason
    count: int

    def __post_init__(self) -> None:
        if not isinstance(self.reason, KMeshFallbackReason):
            raise TypeError("reason must be KMeshFallbackReason")
        if not _is_strict_int(self.count):
            raise TypeError("count must be an integer")
        if self.count <= 0:
            raise ValueError("count must be positive")

    def to_primitive(self) -> dict[str, Any]:
        return {"reason": self.reason.value, "count": self.count}


@dataclass(frozen=True, slots=True)
class KMeshProspectiveMembership:
    request_id: str
    reuse_group_id: str | None
    plan_id: str | None
    fallback_reason: KMeshFallbackReason | None

    def __post_init__(self) -> None:
        _validate_identifier(self.request_id, "request_id")
        grouped = self.reuse_group_id is not None or self.plan_id is not None
        if grouped:
            if self.reuse_group_id is None or self.plan_id is None:
                raise ValueError("group and plan identities must be set together")
            _validate_identifier(self.reuse_group_id, "reuse_group_id")
            _validate_identifier(self.plan_id, "plan_id")
            if self.fallback_reason is not None:
                raise ValueError("prospective members cannot have a fallback")
        elif not isinstance(self.fallback_reason, KMeshFallbackReason):
            raise ValueError("ungrouped requests require a stable fallback reason")

    @property
    def is_grouped(self) -> bool:
        return self.reuse_group_id is not None

    def to_primitive(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "reuse_group_id": self.reuse_group_id,
            "plan_id": self.plan_id,
            "fallback_reason": (
                self.fallback_reason.value if self.fallback_reason else None
            ),
        }


@dataclass(frozen=True, slots=True)
class KMeshGroupObserverMetrics:
    """Structural overlap for one deterministic prospective group.

    Shared counts describe one logical copy. Prospective reused counts multiply
    that copy by ``request_count - 1``. Full blocks are segment-relative windows
    that are wholly block-aligned in every request-local mapping. These metrics
    describe the selected plan; they are not an optimized or maximum opportunity.
    """

    opportunity_scope: str = dataclass_field(
        init=False, default=KMESH_SELECTED_PLAN_OPPORTUNITY_SCOPE
    )
    reuse_group_id: str
    request_count: int
    block_size: int
    shared_segments: int
    shared_tokens: int
    prospective_reused_segments: int
    prospective_reused_tokens: int
    structural_full_blocks: int
    structural_full_block_tokens: int
    prospective_reused_full_blocks: int
    prospective_reused_full_block_tokens: int

    def __post_init__(self) -> None:
        if self.opportunity_scope != KMESH_SELECTED_PLAN_OPPORTUNITY_SCOPE:
            raise ValueError("group metrics must identify selected-plan opportunity")
        _validate_identifier(self.reuse_group_id, "reuse_group_id")
        if not _is_strict_int(self.request_count) or self.request_count < 2:
            raise ValueError("group metrics require at least two requests")
        if not _is_strict_int(self.block_size) or self.block_size <= 0:
            raise ValueError("block_size must be a positive integer")
        for field in (
            "shared_segments",
            "shared_tokens",
            "prospective_reused_segments",
            "prospective_reused_tokens",
            "structural_full_blocks",
            "structural_full_block_tokens",
            "prospective_reused_full_blocks",
            "prospective_reused_full_block_tokens",
        ):
            value = getattr(self, field)
            if not _is_strict_int(value) or value < 0:
                raise ValueError(f"{field} must be a nonnegative integer")
        multiplier = self.request_count - 1
        expected_reuse = (
            (
                self.prospective_reused_segments,
                self.shared_segments * multiplier,
            ),
            (self.prospective_reused_tokens, self.shared_tokens * multiplier),
            (
                self.prospective_reused_full_blocks,
                self.structural_full_blocks * multiplier,
            ),
            (
                self.prospective_reused_full_block_tokens,
                self.structural_full_block_tokens * multiplier,
            ),
        )
        if any(observed != expected for observed, expected in expected_reuse):
            raise ValueError("prospective reuse counts must use request_count - 1")
        if (
            self.structural_full_block_tokens
            != self.structural_full_blocks * self.block_size
        ):
            raise ValueError("full-block token count must use block_size")

    def to_primitive(self) -> dict[str, Any]:
        return {
            "opportunity_scope": self.opportunity_scope,
            "reuse_group_id": self.reuse_group_id,
            "request_count": self.request_count,
            "block_size": self.block_size,
            "shared_segments": self.shared_segments,
            "shared_tokens": self.shared_tokens,
            "prospective_reused_segments": self.prospective_reused_segments,
            "prospective_reused_tokens": self.prospective_reused_tokens,
            "structural_full_blocks": self.structural_full_blocks,
            "structural_full_block_tokens": self.structural_full_block_tokens,
            "prospective_reused_full_blocks": self.prospective_reused_full_blocks,
            "prospective_reused_full_block_tokens": (
                self.prospective_reused_full_block_tokens
            ),
        }


@dataclass(frozen=True, slots=True)
class KMeshObserverMetrics:
    """Structured metrics emitted by one group-planner call.

    Compatibility failures count failed comparisons, while fallback counts are
    per-request outcomes. Candidate requests are individually eligible requests
    with a declared shareable segment and complete runtime and prompt-token
    evidence. Opportunity counts describe only the deterministic selected plan;
    they are neither optimized maxima nor claims about materialized KV values.
    """

    opportunity_scope: str = dataclass_field(
        init=False, default=KMESH_SELECTED_PLAN_OPPORTUNITY_SCOPE
    )
    observed_requests: int
    candidate_requests: int
    formed_groups: int
    block_size: int
    max_group_size: int
    group_sizes: tuple[int, ...]
    shared_segments: int
    shared_tokens: int
    prospective_reused_segments: int
    prospective_reused_tokens: int
    structural_full_blocks: int
    structural_full_block_tokens: int
    prospective_reused_full_blocks: int
    prospective_reused_full_block_tokens: int
    compatibility_failures: tuple[KMeshReasonCount, ...]
    fallback_counts: tuple[KMeshReasonCount, ...]
    prospective_membership: tuple[KMeshProspectiveMembership, ...]
    group_metrics: tuple[KMeshGroupObserverMetrics, ...]
    planner_overhead_ns: int

    def __post_init__(self) -> None:
        if self.opportunity_scope != KMESH_SELECTED_PLAN_OPPORTUNITY_SCOPE:
            raise ValueError("observer metrics must identify selected-plan opportunity")
        for field in (
            "observed_requests",
            "candidate_requests",
            "formed_groups",
            "shared_segments",
            "shared_tokens",
            "prospective_reused_segments",
            "prospective_reused_tokens",
            "structural_full_blocks",
            "structural_full_block_tokens",
            "prospective_reused_full_blocks",
            "prospective_reused_full_block_tokens",
            "planner_overhead_ns",
        ):
            value = getattr(self, field)
            if not _is_strict_int(value) or value < 0:
                raise ValueError(f"{field} must be a nonnegative integer")
        if self.candidate_requests > self.observed_requests:
            raise ValueError("candidate requests cannot exceed observed requests")
        if not _is_strict_int(self.block_size) or self.block_size <= 0:
            raise ValueError("block_size must be a positive integer")
        if not _is_strict_int(self.max_group_size) or self.max_group_size < 2:
            raise ValueError("max_group_size must be an integer of at least two")
        if any(metric.block_size != self.block_size for metric in self.group_metrics):
            raise ValueError("group metrics must use the observer block size")
        if self.formed_groups != len(self.group_metrics):
            raise ValueError("formed_groups must match group_metrics")
        expected_group_sizes = tuple(
            sorted(metric.request_count for metric in self.group_metrics)
        )
        if self.group_sizes != expected_group_sizes:
            raise ValueError("group_sizes must match group_metrics")
        request_ids = tuple(
            membership.request_id for membership in self.prospective_membership
        )
        if request_ids != tuple(sorted(set(request_ids))):
            raise ValueError("prospective membership must be unique and sorted")
        if len(request_ids) != self.observed_requests:
            raise ValueError("every observed request needs a membership outcome")
        if sum(outcome.is_grouped for outcome in self.prospective_membership) != sum(
            self.group_sizes
        ):
            raise ValueError("group sizes must match prospective membership")

        aggregate_fields = (
            "shared_segments",
            "shared_tokens",
            "prospective_reused_segments",
            "prospective_reused_tokens",
            "structural_full_blocks",
            "structural_full_block_tokens",
            "prospective_reused_full_blocks",
            "prospective_reused_full_block_tokens",
        )
        for field in aggregate_fields:
            if getattr(self, field) != sum(
                getattr(metric, field) for metric in self.group_metrics
            ):
                raise ValueError(f"{field} must equal the group-metric aggregate")
        for field in ("compatibility_failures", "fallback_counts"):
            reason_counts = getattr(self, field)
            reasons = tuple(item.reason for item in reason_counts)
            if len(reasons) != len(set(reasons)):
                raise ValueError(f"{field} must not repeat reasons")
        expected_fallbacks = Counter(
            membership.fallback_reason
            for membership in self.prospective_membership
            if membership.fallback_reason is not None
        )
        if {
            fallback.reason: fallback.count for fallback in self.fallback_counts
        } != expected_fallbacks:
            raise ValueError("fallback_counts must match prospective membership")

    def to_primitive(self) -> dict[str, Any]:
        return {
            "opportunity_scope": self.opportunity_scope,
            "observed_requests": self.observed_requests,
            "candidate_requests": self.candidate_requests,
            "formed_groups": self.formed_groups,
            "block_size": self.block_size,
            "max_group_size": self.max_group_size,
            "group_sizes": list(self.group_sizes),
            "shared_segments": self.shared_segments,
            "shared_tokens": self.shared_tokens,
            "prospective_reused_segments": self.prospective_reused_segments,
            "prospective_reused_tokens": self.prospective_reused_tokens,
            "structural_full_blocks": self.structural_full_blocks,
            "structural_full_block_tokens": self.structural_full_block_tokens,
            "prospective_reused_full_blocks": self.prospective_reused_full_blocks,
            "prospective_reused_full_block_tokens": (
                self.prospective_reused_full_block_tokens
            ),
            "compatibility_failures": [
                failure.to_primitive() for failure in self.compatibility_failures
            ],
            "fallback_counts": [
                fallback.to_primitive() for fallback in self.fallback_counts
            ],
            "prospective_membership": [
                membership.to_primitive() for membership in self.prospective_membership
            ],
            "group_metrics": [metric.to_primitive() for metric in self.group_metrics],
            "planner_overhead_ns": self.planner_overhead_ns,
        }
