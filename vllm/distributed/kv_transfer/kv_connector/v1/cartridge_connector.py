# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
CartridgeConnector: KVConnectorBase_V1 plugin for injecting pre-trained
cartridge KV caches into vLLM's paged attention system.

Usage:
    vllm serve Qwen/Qwen2.5-7B-Instruct \
        --kv-transfer-config '{
            "kv_connector": "CartridgeConnector",
            "kv_connector_module_path": "vllm.distributed.kv_transfer.kv_connector.v1.cartridge_connector",
            "kv_connector_extra_config": {
                "cartridge_path": "/path/to/cartridge.pt",
                "prefix_token_ids_path": "/path/to/prefix_token_ids.json"
            },
            "kv_role": "kv_both"
        }'

The cartridge .pt file should be a TrainableCache checkpoint containing
per-layer K and V tensors. The prefix_token_ids_path should point to a
JSON file with the token IDs that the cartridge was trained on.

Requests whose prompt starts with these token IDs will have the cartridge
KV injected instead of computing those tokens from scratch.
"""
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger
from vllm.v1.attention.ops.paged_attn import PagedAttention
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


def load_cartridge(path: str) -> dict:
    """Load a cartridge checkpoint and extract per-layer K, V tensors.

    Returns dict with:
        - kv_data: list of (K, V) tuples per layer, each (num_tokens, num_heads, head_dim)
        - num_tokens: int
        - num_layers: int
        - num_kv_heads: int
        - head_dim: int
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    # Handle TrainableCache checkpoint format
    if hasattr(checkpoint, "trainable_keys"):
        cache = checkpoint
    elif isinstance(checkpoint, dict) and "cache" in checkpoint:
        cache = checkpoint["cache"]
    elif isinstance(checkpoint, dict):
        cache = checkpoint
    else:
        raise ValueError(f"Unrecognized cartridge format: {type(checkpoint)}")

    # Extract K, V tensors
    if hasattr(cache, "trainable_keys"):
        keys = cache.trainable_keys
        values = cache.trainable_values
    elif isinstance(cache, dict) and "trainable_keys" in cache:
        keys = cache["trainable_keys"]
        values = cache["trainable_values"]
    else:
        raise ValueError("Cannot find trainable_keys in checkpoint")

    num_layers = len(keys)
    # Shape: (1, num_heads, num_tokens, head_dim)
    kv_data = []
    for l in range(num_layers):
        k = keys[l].data.squeeze(0)  # (num_heads, num_tokens, head_dim)
        v = values[l].data.squeeze(0)
        # Permute to (num_tokens, num_heads, head_dim)
        k = k.permute(1, 0, 2).contiguous()
        v = v.permute(1, 0, 2).contiguous()
        kv_data.append((k, v))
    num_tokens = kv_data[0][0].shape[0]
    num_kv_heads = kv_data[0][0].shape[1]
    head_dim = kv_data[0][0].shape[2]

    return {
        "kv_data": kv_data,
        "num_tokens": num_tokens,
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
    }


def align_to_block_size(num_tokens: int, block_size: int) -> int:
    """Align the number of tokens down to block boundary."""
    return (num_tokens // block_size) * block_size


@dataclass
class CartridgeReqMeta:
    """Metadata for a single request needing cartridge KV injection."""
    slot_mapping: torch.Tensor
    num_tokens: int


@dataclass
class CartridgeConnectorMetadata(KVConnectorMetadata):
    """Metadata passed from scheduler to worker."""
    requests: list[CartridgeReqMeta] = field(default_factory=list)


class CartridgeConnector(KVConnectorBase_V1):
    """
    KV connector that injects pre-trained cartridge KV caches into
    vLLM's paged attention system. Matches against a known prefix
    token sequence and replaces its KV computation with pre-trained
    cartridge KV.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self._block_size = vllm_config.cache_config.block_size
        self._requests_need_load: dict[str, "Request"] = {}

        # Load cartridge
        cartridge_path = self._kv_transfer_config.get_from_extra_config(
            "cartridge_path", None
        )
        if cartridge_path is None:
            raise ValueError(
                "CartridgeConnector requires 'cartridge_path' in "
                "--kv-connector-extra-config"
            )

        # Load prefix token IDs
        prefix_token_ids_path = self._kv_transfer_config.get_from_extra_config(
            "prefix_token_ids_path", None
        )
        if prefix_token_ids_path is None:
            raise ValueError(
                "CartridgeConnector requires 'prefix_token_ids_path' in "
                "--kv-connector-extra-config"
            )

        with open(prefix_token_ids_path) as f:
            self._prefix_token_ids = json.load(f)
        logger.info("Loaded %d prefix token IDs from %s",
                     len(self._prefix_token_ids), prefix_token_ids_path)

        logger.info("Loading cartridge from %s", cartridge_path)
        self._cartridge = load_cartridge(cartridge_path)

        # Align to block size (floor to avoid partial blocks)
        raw_tokens = self._cartridge["num_tokens"]
        self._num_cartridge_tokens = align_to_block_size(
            raw_tokens, self._block_size
        )
        self._num_cartridge_blocks = self._num_cartridge_tokens // self._block_size

        # Also align the prefix token IDs to match
        self._prefix_token_ids = self._prefix_token_ids[:self._num_cartridge_tokens]

        logger.info(
            "Cartridge loaded: %d layers, %d tokens (%d blocks of %d), "
            "%d kv_heads, %d head_dim",
            self._cartridge["num_layers"],
            self._num_cartridge_tokens,
            self._num_cartridge_blocks,
            self._block_size,
            self._cartridge["num_kv_heads"],
            self._cartridge["head_dim"],
        )

        # Pre-stack KV data for efficient injection
        # Shape per layer: (2, num_tokens, num_heads, head_dim)
        self._kv_stacked = []
        for l in range(self._cartridge["num_layers"]):
            k, v = self._cartridge["kv_data"][l]
            k = k[:self._num_cartridge_tokens]
            v = v[:self._num_cartridge_tokens]
            stacked = torch.stack([k, v], dim=0)  # (2, T, H, D)
            self._kv_stacked.append(stacked)

        # Free original kv_data to save memory
        del self._cartridge["kv_data"]

    # ==============================
    # Scheduler-side methods
    # ==============================

    def _match_prefix(self, request: "Request") -> int:
        """Check how many prefix tokens match the request's prompt.

        Returns the number of matched tokens (block-aligned).
        """
        prompt_ids = request.prompt_token_ids
        if prompt_ids is None:
            return 0

        # Check how many prefix tokens match
        match_len = 0
        max_check = min(len(self._prefix_token_ids), len(prompt_ids))
        for i in range(max_check):
            if prompt_ids[i] != self._prefix_token_ids[i]:
                break
            match_len += 1

        # Align to block boundary
        return align_to_block_size(match_len, self._block_size)

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """Report prefix tokens as externally computed if they match."""
        matched = self._match_prefix(request)
        if matched == 0:
            return 0, False

        # How many new external tokens beyond what's already computed
        num_new = matched - num_computed_tokens
        if num_new <= 0:
            return 0, False

        logger.info(
            "Cartridge prefix match: %d tokens matched, %d new (beyond %d computed)",
            matched, num_new, num_computed_tokens
        )
        return num_new, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ):
        """Record that this request needs KV loading."""
        if num_external_tokens > 0:
            self._requests_need_load[request.request_id] = request

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        """Build metadata with slot mappings for KV injection."""
        meta = CartridgeConnectorMetadata()

        for new_req in scheduler_output.scheduled_new_reqs:
            if new_req.req_id in self._requests_need_load:
                block_ids = new_req.block_ids[0]  # first KV cache group
                cartridge_block_ids = block_ids[:self._num_cartridge_blocks]
                block_ids_tensor = torch.tensor(
                    cartridge_block_ids, dtype=torch.long
                )
                block_offsets = torch.arange(
                    0, self._block_size, dtype=torch.long
                )
                slot_mapping = (
                    block_offsets.reshape(1, self._block_size)
                    + block_ids_tensor.reshape(-1, 1) * self._block_size
                ).flatten()

                meta.requests.append(CartridgeReqMeta(
                    slot_mapping=slot_mapping,
                    num_tokens=self._num_cartridge_tokens,
                ))

        self._requests_need_load.clear()
        return meta

    # ==============================
    # Worker-side methods
    # ==============================

    def start_load_kv(
        self, forward_context: "ForwardContext", **kwargs: Any
    ) -> None:
        """Inject cartridge KV into allocated GPU blocks."""
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, CartridgeConnectorMetadata)

        if not metadata.requests:
            return

        for request in metadata.requests:
            slot_mapping = request.slot_mapping.cuda()

            logger.info(
                "Injecting cartridge KV (%d tokens) into GPU cache",
                request.num_tokens,
            )

            for layer_name in forward_context.no_compile_layers:
                layer = forward_context.no_compile_layers[layer_name]
                kv_cache_attr = getattr(layer, "kv_cache", None)
                if kv_cache_attr is None:
                    continue

                # V1: kv_cache is a direct tensor, not indexed by virtual_engine
                kv_cache_layer = kv_cache_attr

                layer_idx = self._extract_layer_idx(layer_name)
                if layer_idx is None or layer_idx >= len(self._kv_stacked):
                    continue

                # Get cartridge KV for this layer: (2, T, H, D)
                src_kv = self._kv_stacked[layer_idx].to(
                    device=kv_cache_layer.device,
                    dtype=kv_cache_layer.dtype,
                )

                # Use reshape_and_cache to write in the correct paged
                # attention layout (key_cache is transposed, not flat).
                src_key = src_kv[0]    # (T, num_kv_heads, head_dim)
                src_value = src_kv[1]  # (T, num_kv_heads, head_dim)

                key_cache, value_cache = PagedAttention.split_kv_cache(
                    kv_cache_layer,
                    self._cartridge["num_kv_heads"],
                    self._cartridge["head_dim"],
                )

                PagedAttention.write_to_paged_cache(
                    src_key,
                    src_value,
                    key_cache,
                    value_cache,
                    slot_mapping,
                    "auto",
                    torch.tensor(1.0, device=kv_cache_layer.device),
                    torch.tensor(1.0, device=kv_cache_layer.device),
                )

    def wait_for_layer_load(self, layer_name: str) -> None:
        """No-op — synchronous load."""
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        """No-op — we don't save cartridge KV back."""
        return

    def wait_for_save(self):
        """No-op."""
        return

    # ==============================
    # Helpers
    # ==============================

    @staticmethod
    def _extract_layer_idx(layer_name: str) -> int | None:
        """Extract layer index from a name like 'model.layers.0.self_attn'."""
        parts = layer_name.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except ValueError:
                    pass
        return None
