# SPDX-License-Identifier: Apache-2.0
"""Per-KV-head routed attention for cartridge routing phase 1.

Implements first-token decode with per-head block selection:
for each KV head, attend only to top-K blocks selected by the routing prior.

This is functionally correct but not throughput-optimized — it runs one
flash_attn call per KV head group instead of one fused call for all heads.
Phase 1 measures quality (logit error, top-token agreement) and KV-touch
reduction, not throughput of the routing path itself.
"""
import time
from dataclasses import dataclass, field

import torch

from vllm.logger import init_logger
from vllm.v1.attention.routing_state import RoutingPrior

logger = init_logger(__name__)


@dataclass
class RoutingMetrics:
    """Metrics collected during routed first-token decode."""
    layer_idx: int = 0
    blocks_touched_per_head: list[int] = field(default_factory=list)
    total_blocks_available: int = 0
    kv_positions_touched: int = 0
    full_kv_positions: int = 0
    wall_time_ns: int = 0


def forward_routed(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    routing_prior: RoutingPrior,
    layer_idx: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    scale: float,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    output: torch.Tensor,
    flash_attn_varlen_func,
    fa_version=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
) -> tuple[torch.Tensor, RoutingMetrics]:
    """Run per-KV-head routed attention for first-token decode.

    For each KV head group:
    1. Select top-K blocks from routing prior
    2. Build a restricted block table
    3. Run flash_attn on that head group with the restricted blocks
    4. Place output in the correct head positions

    Args:
        query: (num_tokens, num_heads, head_size)
        key_cache: (num_blocks, block_size, num_kv_heads, head_size)
        value_cache: (num_blocks, block_size, num_kv_heads, head_size)
        block_table: (batch_size, max_blocks_per_seq) — full block table
        routing_prior: RoutingPrior with block_affinities
        layer_idx: current layer index
        num_heads: total query heads
        num_kv_heads: total KV heads
        head_size: head dimension
        scale: attention scale factor
        cu_seqlens_q: cumulative query sequence lengths
        seqused_k: KV sequence lengths per batch element
        max_seqlen_q: max query length
        max_seqlen_k: max KV length
        output: pre-allocated output tensor
        flash_attn_varlen_func: the flash attention function
        fa_version: flash attention version
        q_descale, k_descale, v_descale: quantization scales

    Returns:
        (output, metrics) tuple
    """
    t0 = time.monotonic_ns()
    heads_per_group = num_heads // num_kv_heads
    num_tokens = query.shape[0]
    K = routing_prior.K
    device = query.device

    # Get block affinities for this layer: (num_kv_heads, num_blocks)
    layer_affinities = routing_prior.block_affinities[layer_idx]

    metrics = RoutingMetrics(
        layer_idx=layer_idx,
        total_blocks_available=block_table.shape[1],
    )

    # For each KV head, select top-K blocks and run attention
    for kv_head_idx in range(num_kv_heads):
        # Get affinities for this head
        head_aff = layer_affinities[kv_head_idx]

        # Only route within prefix blocks, not the full sequence
        prefix_aff = head_aff[:routing_prior.num_prefix_blocks]
        actual_K = min(K, prefix_aff.shape[0])

        # Top-K block indices (within prefix)
        topk_indices = torch.topk(prefix_aff.to(device), actual_K).indices

        # Map to physical block IDs via the original block table
        # block_table[0] is the first (only) sequence in the batch
        prefix_block_ids = block_table[0, :routing_prior.num_prefix_blocks]
        selected_physical_blocks = prefix_block_ids[topk_indices]

        # Also include non-prefix blocks (query/decode blocks) — these are
        # always attended (causal). They come after the prefix blocks.
        non_prefix_blocks = block_table[0, routing_prior.num_prefix_blocks:]
        # Filter out padding (block_id == 0 could be padding or real block 0,
        # but seqused_k handles the actual length)
        num_non_prefix = block_table.shape[1] - routing_prior.num_prefix_blocks
        if num_non_prefix > 0:
            routed_blocks = torch.cat([
                selected_physical_blocks,
                non_prefix_blocks[:num_non_prefix],
            ])
        else:
            routed_blocks = selected_physical_blocks

        # Build restricted block table: (1, num_routed_blocks)
        routed_block_table = routed_blocks.unsqueeze(0)

        # Compute restricted seqused_k:
        # routed KV length = K * block_size + non_prefix_tokens
        routed_kv_len = actual_K * routing_prior.block_size
        # Add non-prefix tokens (from original seqused_k minus prefix tokens)
        prefix_tokens = routing_prior.num_prefix_blocks * routing_prior.block_size
        non_prefix_tokens = max(0, int(seqused_k[0].item()) - prefix_tokens)
        routed_kv_len += non_prefix_tokens
        routed_seqused_k = torch.tensor([routed_kv_len], device=device, dtype=seqused_k.dtype)
        routed_max_seqlen_k = routed_kv_len

        # Extract query heads for this KV group
        q_start = kv_head_idx * heads_per_group
        q_end = q_start + heads_per_group
        q_group = query[:, q_start:q_end, :].contiguous()

        # Output slice for this head group
        out_group = output[:, q_start:q_end, :].contiguous()

        # Extract single KV head from cache
        # key_cache shape: (num_blocks, block_size, num_kv_heads, head_size)
        k_head = key_cache[:, :, kv_head_idx:kv_head_idx+1, :].contiguous()
        v_head = value_cache[:, :, kv_head_idx:kv_head_idx+1, :].contiguous()

        # Build per-head descale if needed
        h_q_descale = None
        h_k_descale = None
        h_v_descale = None
        if q_descale is not None:
            h_q_descale = q_descale[:, kv_head_idx:kv_head_idx+1]
        if k_descale is not None:
            h_k_descale = k_descale[:, kv_head_idx:kv_head_idx+1]
        if v_descale is not None:
            h_v_descale = v_descale[:, kv_head_idx:kv_head_idx+1]

        # Run flash attention for this head group
        flash_attn_varlen_func(
            q=q_group,
            k=k_head,
            v=v_head,
            out=out_group,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=routed_seqused_k,
            max_seqlen_k=routed_max_seqlen_k,
            softmax_scale=scale,
            causal=True,
            block_table=routed_block_table,
            fa_version=fa_version,
            q_descale=h_q_descale,
            k_descale=h_k_descale,
            v_descale=h_v_descale,
        )

        # Copy output back to the full output tensor
        output[:, q_start:q_end, :] = out_group

        metrics.blocks_touched_per_head.append(actual_K)
        metrics.kv_positions_touched += actual_K * routing_prior.block_size

    metrics.full_kv_positions = int(seqused_k[0].item()) * num_kv_heads
    metrics.wall_time_ns = time.monotonic_ns() - t0

    logger.info(
        "Routed attention layer %d: K=%d, %d/%d KV positions touched (%.1f%% reduction), %.2fms",
        layer_idx,
        K,
        metrics.kv_positions_touched,
        metrics.full_kv_positions,
        (1 - metrics.kv_positions_touched / max(metrics.full_kv_positions, 1)) * 100,
        metrics.wall_time_ns / 1e6,
    )

    return output, metrics
