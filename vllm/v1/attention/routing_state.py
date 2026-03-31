# SPDX-License-Identifier: Apache-2.0
"""Thread-local routing state for cartridge-based per-head block routing.

The CartridgeConnector sets this state before each forward pass.
FlashAttentionImpl checks it to decide whether to use routed decode.

This is intentionally minimal: a single global that carries per-layer
routing metadata for exactly one request at a time (phase 1 constraint).
"""
import threading
from dataclasses import dataclass

import torch


@dataclass
class RoutingPrior:
    """Per-layer routing prior for first-token decode."""
    # (num_layers, num_kv_heads, num_blocks) — block affinity scores
    block_affinities: torch.Tensor
    # Number of blocks to attend per head
    K: int
    # Routing mode: 'cartridge_prior' or 'cartridge_prior_adjusted'
    mode: str
    # Number of cartridge prefix blocks (routing applies only to these)
    num_prefix_blocks: int
    # Block size
    block_size: int
    # Request ID for safety checks
    request_id: str
    # Flag: only active for first decode token
    active: bool = True


_local = threading.local()


def set_routing_state(state: RoutingPrior | None) -> None:
    """Set the routing state for the current forward pass."""
    _local.routing_state = state


def get_routing_state() -> RoutingPrior | None:
    """Get the routing state, or None if routing is not active."""
    return getattr(_local, 'routing_state', None)


def clear_routing_state() -> None:
    """Clear routing state after forward pass completes."""
    _local.routing_state = None
