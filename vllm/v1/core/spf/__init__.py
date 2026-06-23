# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SPF (Speculative Prefetch) — scheduler-side prefetch controller.

Public API:
    get_spf_controller() -> SPFController | None

Returns an initialized controller when VLLM_SPF_ENABLED=1, else None.
The controller is scheduler-side only and does not touch attention
backends, KV cache tensors, or the forward path.
"""
from __future__ import annotations

from vllm.v1.core.spf.config import (
    MODE_PREFETCH,
    MODE_RETENTION,
    VALID_MODES,
    SPFConfig,
)
from vllm.v1.core.spf.controller import SPFController
from vllm.v1.core.spf.metrics import (
    PrefetchMetrics,
    RetentionMetrics,
    SPFMetrics,
    SPFStepMetrics,
)
from vllm.v1.core.spf.resource import (
    DEFAULT_BLOCK_SIZE,
    ResourceCandidate,
    ResourceId,
    ResourceScope,
    hash_prefix_tokens,
)
from vllm.v1.core.spf.scorer import (
    CandidateFeatures,
    CooldownTracker,
    ExpectedUtilityScorer,
    LearnedScorer,
    Scorer,
    SessionAwareScorer,
)
from vllm.v1.core.spf.transitions import TransitionTable
from vllm.v1.core.spf.utility import (
    UtilityBreakdown,
    UtilityConstants,
    UtilityFeatures,
    expected_utility,
    probability_of_use,
)
from vllm.v1.core.spf.workloads import (
    WORKLOADS,
    WorkloadEvent,
    batched_burst,
    conversation_tree,
    false_shared_first_block,
    long_doc_qa,
    mixed_session_interleave,
    random_no_reuse,
    shared_prefix,
    shuffled_control,
)
from vllm.v1.core.spf.integrations import (
    BlockPool,
    FakeBlockPool,
    FakePrefetchBackend,
    PrefetchBackend,
    PrefetchIntegration,
    PromotionHandle,
    RetentionIntegration,
)
from vllm.v1.core.spf.shadow_baseline import (
    ShadowBaseline,
    ShadowMetrics,
)

__all__ = [
    "get_spf_controller",
    "SPFController",
    "SPFConfig",
    "SPFMetrics",
    "SPFStepMetrics",
    "RetentionMetrics",
    "PrefetchMetrics",
    "ResourceId",
    "ResourceCandidate",
    "ResourceScope",
    "hash_prefix_tokens",
    "DEFAULT_BLOCK_SIZE",
    "MODE_RETENTION",
    "MODE_PREFETCH",
    "VALID_MODES",
    # Phase 2 exports.
    "CandidateFeatures",
    "Scorer",
    "SessionAwareScorer",
    "LearnedScorer",
    "ExpectedUtilityScorer",
    "CooldownTracker",
    "TransitionTable",
    "UtilityFeatures",
    "UtilityConstants",
    "UtilityBreakdown",
    "expected_utility",
    "probability_of_use",
    # Phase 3 exports.
    "WORKLOADS",
    "WorkloadEvent",
    "mixed_session_interleave",
    "batched_burst",
    "shared_prefix",
    "conversation_tree",
    "random_no_reuse",
    "shuffled_control",
    "false_shared_first_block",
    "long_doc_qa",
    # Phase 4 exports.
    "BlockPool",
    "FakeBlockPool",
    "RetentionIntegration",
    "PrefetchBackend",
    "FakePrefetchBackend",
    "PrefetchIntegration",
    "PromotionHandle",
    # Phase 7 instrumentation exports (optional).
    "ShadowBaseline",
    "ShadowMetrics",
]

_controller: SPFController | None = None


def get_spf_controller() -> SPFController | None:
    """Return the singleton SPF controller, or None if disabled.

    Thread-safety: this is called only from the scheduler process,
    which is single-threaded in vLLM v1.
    """
    global _controller
    if _controller is not None:
        return _controller

    config = SPFConfig.from_env()
    if not config.enabled:
        return None

    _controller = SPFController(config)
    return _controller
