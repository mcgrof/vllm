# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Thread-local routing state for cartridge-based per-head block routing.

The CartridgeConnector sets this state before each forward pass. The
state is producer-side scaffolding intended for a routed-decode
attention backend; the backend consumer is not wired on this branch
and is left for follow-up work. With no consumer, set_routing_state
populates a thread-local that is never read; the only observable
effect is in tests that inspect the thread-local directly.

This is intentionally minimal: a single global that carries per-layer
routing metadata for exactly one request at a time (phase 1 constraint).
"""
import threading
from dataclasses import dataclass, field

import torch


@dataclass
class RoutingPrior:
    """Per-layer routing prior for first-token decode."""
    # (num_layers, num_kv_heads, num_blocks) — block affinity scores
    # May be None when routing_policy is "hybrid" or "recency" (no prior needed)
    block_affinities: torch.Tensor | None
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
    # Block selection policy: "topk", "hybrid", or "recency"
    routing_policy: str = "topk"
    # Hybrid policy: number of anchor blocks (beginning of sequence)
    k_anchor: int = 2
    # Hybrid policy: number of recency blocks (end of sequence)
    k_recent: int = 6
    # Eigenvalue-weighted head contributions (topk_eigen policy)
    # Shape: (num_layers, num_kv_heads) — concentration score per head
    head_weights: torch.Tensor | None = None
    # SVD diversity scores (topk_diverse policy)
    # Shape: (num_blocks,) — block importance in the principal attention subspace
    diversity_scores: torch.Tensor | None = None
    # Softmax gain per head (topk_softmax_gain policy)
    # Shape: (num_layers, num_kv_heads) — bits of routing information
    softmax_gains: torch.Tensor | None = None
    # Block mean Keys for query-conditioned routing (topk_query policy)
    # Shape: (num_layers, num_kv_heads, num_blocks, head_dim) — stored in fp16
    k_block_means: torch.Tensor | None = None
    # Scale factor for Q@K (1/sqrt(head_dim))
    qk_scale: float = 0.0
    # Pre-computed K-means block indices (topk_kmeans policy)
    kmeans_blocks: list[int] | None = None

    # === Cached routing artifacts (computed once, reused across layers/steps) ===
    # Selected prefix block indices (within-prefix, not physical)
    _cached_prefix_indices: torch.Tensor | None = field(default=None,
                                                        repr=False)
    # Number of selected blocks
    _cached_actual_K: int | None = field(default=None, repr=False)
    # Cached restricted block table + seqused_k (reused across layers within a step)
    _cached_block_table: torch.Tensor | None = field(default=None, repr=False)
    _cached_seqused_k: torch.Tensor | None = field(default=None, repr=False)
    _cached_kv_len: int | None = field(default=None, repr=False)
    # seqused_k value when block table was cached (invalidate if changed)
    _cached_seqused_k_val: int | None = field(default=None, repr=False)

    def get_cached_selection(
        self,
        device: torch.device,
    ) -> tuple[torch.Tensor, int]:
        """Get cached block selection indices for head-uniform policies.

        For hybrid/recency, the selection is the same for all layers and
        all KV heads, so we compute it once and cache it on the RoutingPrior.
        The cache persists across layers within a forward pass AND across
        decode steps (the same RoutingPrior is re-set each step).
        """
        if self._cached_prefix_indices is not None:
            return self._cached_prefix_indices, self._cached_actual_K

        n = self.num_prefix_blocks
        if self.routing_policy == "hybrid":
            k_a = min(self.k_anchor, n)
            k_r = min(self.k_recent, n)
            if k_a + k_r >= n:
                indices = torch.arange(n, device=device, dtype=torch.long)
            else:
                anchor = torch.arange(k_a, device=device, dtype=torch.long)
                recency = torch.arange(n - k_r,
                                       n,
                                       device=device,
                                       dtype=torch.long)
                indices = torch.cat([anchor, recency])
            actual_K = indices.shape[0]
        elif self.routing_policy == "recency":
            k = min(self.K, n)
            indices = torch.arange(n - k, n, device=device, dtype=torch.long)
            actual_K = k
        elif self.routing_policy == "topk_avg":
            # Content-aware: average affinities across all heads and layers,
            # then take top-K.  Gives a single head-uniform selection that
            # reflects which blocks the model actually attends to.
            if self.block_affinities is None:
                return None, None
            # block_affinities: (num_layers, num_kv_heads, num_blocks)
            avg_aff = self.block_affinities[:, :, :n].mean(dim=(0, 1))
            k = min(self.K, n)
            topk = torch.topk(avg_aff.to(device), k)
            # Sort indices so block table is in position order
            indices = topk.indices.sort().values
            actual_K = k
        elif self.routing_policy == "topk_eigen":
            # Eigenvalue-weighted content-aware selection.  Heads with
            # concentrated attention (high max eigenvalue / high peak/mean
            # ratio) get more influence on which blocks are selected.
            if self.block_affinities is None or self.head_weights is None:
                return None, None
            aff = self.block_affinities[:, :, :n]  # (L, H, n)
            w = self.head_weights  # (L, H)
            # Weighted average: heads with higher concentration score
            # contribute more to the aggregate block importance
            weighted_aff = ((aff * w.unsqueeze(-1)).sum(dim=(0, 1)) / w.sum())
            k = min(self.K, n)
            topk = torch.topk(weighted_aff.to(device), k)
            indices = topk.indices.sort().values
            actual_K = k
        elif self.routing_policy == "topk_diverse":
            # SVD-based diversity selection.
            if self.diversity_scores is None:
                return None, None
            scores = self.diversity_scores[:n]
            k = min(self.K, n)
            topk = torch.topk(scores.to(device), k)
            indices = topk.indices.sort().values
            actual_K = k
        elif self.routing_policy == "topk_softmax_gain":
            # Weight block affinities by softmax information gain per head.
            # Heads where softmax produces sharper routing (more bits of
            # info) contribute more to block selection.
            if self.block_affinities is None or self.softmax_gains is None:
                return None, None
            aff = self.block_affinities[:, :, :n]
            g = self.softmax_gains  # (L, H)
            weighted = (aff * g.unsqueeze(-1)).sum(dim=(0, 1)) / g.sum()
            k = min(self.K, n)
            indices = torch.topk(weighted.to(device), k).indices.sort().values
            actual_K = k
        elif self.routing_policy == "topk_kmeans":
            # K-means on block Key means: select the block nearest each
            # cluster centroid. Gives content-aware structural coverage
            # that is completely independent of position or attention.
            if self.kmeans_blocks is None:
                return None, None
            k = min(self.K, len(self.kmeans_blocks), n)
            # Take first K from the pre-sorted list
            indices = torch.tensor(sorted(self.kmeans_blocks[:k]),
                                   device=device,
                                   dtype=torch.long)
            actual_K = indices.shape[0]
        elif self.routing_policy == "topk_query":
            # Query-conditioned routing: use stored K_block_means to compute
            # routing scores at serving time with the actual query vector.
            # This is computed per decode step, NOT cached across steps.
            # Falls back to softmax-gain-weighted if no query available yet.
            if self.block_affinities is None:
                return None, None
            if self.softmax_gains is not None:
                aff = self.block_affinities[:, :, :n]
                g = self.softmax_gains
                weighted = (aff * g.unsqueeze(-1)).sum(dim=(0, 1)) / g.sum()
            else:
                weighted = self.block_affinities[:, :, :n].mean(dim=(0, 1))
            k = min(self.K, n)
            indices = torch.topk(weighted.to(device), k).indices.sort().values
            actual_K = k
        else:
            # topk is per-head, cannot cache uniformly
            return None, None

        self._cached_prefix_indices = indices
        self._cached_actual_K = actual_K
        return indices, actual_K


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
