# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Routing block selection policies for KV cache routing.

Provides pluggable block selection heuristics that determine which KV
blocks to retain during routed decode. The default policy is the hybrid
anchor+recency policy validated in the W7900/H100 routing experiments.
"""

from vllm.distributed.kv_transfer.kv_connector.v1.routing_policy.hybrid import (
    HybridPolicyConfig, select_blocks_hybrid, select_blocks_recency)

__all__ = [
    "select_blocks_hybrid",
    "select_blocks_recency",
    "HybridPolicyConfig",
]
