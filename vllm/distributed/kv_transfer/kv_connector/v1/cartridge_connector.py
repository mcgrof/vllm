# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
CartridgeConnector: KVConnectorBase_V1 plugin for injecting pre-trained
cartridge KV caches into vLLM's paged attention system.

Cartridges are pre-computed KV caches produced by the HazyResearch
Self-Study training method (arxiv 2504.16106). A cartridge file
contains per-layer K and V tensors that have been optimized via
backpropagation so the model can answer questions about a document
without re-processing it at serve time.

Usage:
    vllm serve meta-llama/Llama-3.2-3B-Instruct \\
        --kv-transfer-config '{
            "kv_connector": "CartridgeConnector",
            "kv_connector_module_path":
                "vllm.distributed.kv_transfer.kv_connector.v1.cartridge_connector",
            "kv_connector_extra_config": {
                "cartridge_path": "/path/to/cartridge.pt"
            },
            "kv_role": "kv_both"
        }'

The cartridge .pt file should be a TrainableCache checkpoint with
trainable_keys, trainable_values, and optionally frozen_keys /
frozen_values for the BOS/system prefix that was held fixed during
Self-Study training.

Every incoming request gets the cartridge KV injected into its first
N positions, skipping prefill for those tokens. The connector reports
num_cartridge_tokens as externally computed to the scheduler, which
allocates blocks and skips prefill accordingly.

This base connector loads one cartridge at init and serves it to all
requests. Multi-cartridge serving (cartridge registry, on-demand
loading, eviction) is future work; see docs/design/cartridge_connector.md.
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
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
)
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Cartridge loading
# ---------------------------------------------------------------------------

def load_cartridge(path: str) -> dict:
    """Load a TrainableCache cartridge checkpoint.

    The checkpoint is a Python object or dict with ``trainable_keys``
    and ``trainable_values`` (per-layer parameter lists), and optionally
    ``frozen_keys`` / ``frozen_values`` for the leading BOS/system
    tokens that were held fixed during Self-Study training.

    Returns dict with:
        - kv_data: list of (K, V) tuples per layer,
          each (num_tokens, num_kv_heads, head_dim)
        - num_tokens: int
        - num_layers: int
        - num_kv_heads: int
        - head_dim: int
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    if hasattr(checkpoint, "trainable_keys"):
        cache = checkpoint
    elif isinstance(checkpoint, dict) and "cache" in checkpoint:
        cache = checkpoint["cache"]
    elif isinstance(checkpoint, dict):
        cache = checkpoint
    else:
        raise ValueError(f"Unrecognized cartridge format: {type(checkpoint)}")

    def _get(attr, obj):
        if hasattr(obj, attr):
            return getattr(obj, attr)
        if isinstance(obj, dict) and attr in obj:
            return obj[attr]
        return None

    trainable_keys = _get("trainable_keys", cache)
    trainable_values = _get("trainable_values", cache)
    frozen_keys = _get("frozen_keys", cache) or []
    frozen_values = _get("frozen_values", cache) or []

    if trainable_keys is None or trainable_values is None:
        raise ValueError("Cannot find trainable_keys in checkpoint")

    num_layers = len(trainable_keys)
    if frozen_keys and len(frozen_keys) != num_layers:
        raise ValueError(
            f"frozen_keys length ({len(frozen_keys)}) != "
            f"trainable_keys length ({num_layers})"
        )

    # Per-layer tensor shape: (1, num_heads, num_tokens, head_dim).
    # Concat frozen (leading BOS/system tokens) + trainable along the
    # token axis. Without this concatenation, the leading positions
    # get overwritten with trained values that were optimized assuming
    # the frozen BOS prefix would be present, which catastrophically
    # corrupts attention and produces garbage output.
    kv_data = []
    for layer_idx in range(num_layers):
        k_trn = trainable_keys[layer_idx]
        v_trn = trainable_values[layer_idx]
        if hasattr(k_trn, "data"):
            k_trn = k_trn.data
        if hasattr(v_trn, "data"):
            v_trn = v_trn.data

        if frozen_keys:
            k_frz = frozen_keys[layer_idx]
            v_frz = frozen_values[layer_idx]
            if hasattr(k_frz, "data"):
                k_frz = k_frz.data
            if hasattr(v_frz, "data"):
                v_frz = v_frz.data
            k_full = torch.cat([k_frz, k_trn], dim=2).squeeze(0)
            v_full = torch.cat([v_frz, v_trn], dim=2).squeeze(0)
        else:
            k_full = k_trn.squeeze(0)
            v_full = v_trn.squeeze(0)

        # (num_heads, num_tokens, head_dim) -> (num_tokens, num_heads, head_dim)
        k = k_full.permute(1, 0, 2).contiguous()
        v = v_full.permute(1, 0, 2).contiguous()

        # Validate shape consistency across layers
        if layer_idx > 0:
            prev_k = kv_data[0][0]
            if k.shape != prev_k.shape:
                raise ValueError(
                    f"Layer {layer_idx} K shape {k.shape} != "
                    f"layer 0 K shape {prev_k.shape}"
                )

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
    """Align the number of tokens down to the nearest block boundary."""
    return (num_tokens // block_size) * block_size


# ---------------------------------------------------------------------------
# Cache injection helper (extracted for testability)
# ---------------------------------------------------------------------------

def inject_kv_into_paged_cache(
    src_key: torch.Tensor,
    src_value: torch.Tensor,
    kv_cache_layer: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Write cartridge K/V into a paged KV cache layer.

    This is the core injection operation, extracted as a pure function
    so it can be tested independently (including with a fake writer
    for CPU-only tests).

    Args:
        src_key: (num_tokens, num_kv_heads, head_dim)
        src_value: (num_tokens, num_kv_heads, head_dim)
        kv_cache_layer: the paged cache tensor for one layer.
            V1 layout is (2, num_blocks, block_size, num_kv_heads, head_dim)
            with K/V split at dim 0 or dim 1.
        slot_mapping: (num_tokens,) int64, mapping each token to a
            flat slot index in the paged cache.
    """
    # Auto-detect K/V split dimension
    if kv_cache_layer.shape[0] == 2:
        key_cache, value_cache = kv_cache_layer.unbind(0)
    elif kv_cache_layer.shape[1] == 2:
        key_cache, value_cache = kv_cache_layer.unbind(1)
    else:
        raise ValueError(
            f"Cannot determine K/V split dim for cache shape "
            f"{kv_cache_layer.shape}. Expected dim 0 or 1 to be 2."
        )

    k_scale = torch.tensor(1.0, dtype=torch.float32,
                            device=kv_cache_layer.device)
    v_scale = torch.tensor(1.0, dtype=torch.float32,
                            device=kv_cache_layer.device)

    triton_reshape_and_cache_flash(
        key=src_key,
        value=src_value,
        key_cache=key_cache,
        value_cache=value_cache,
        slot_mapping=slot_mapping.to(torch.int64),
        kv_cache_dtype="auto",
        k_scale=k_scale,
        v_scale=v_scale,
    )


# ---------------------------------------------------------------------------
# Connector metadata
# ---------------------------------------------------------------------------

@dataclass
class CartridgeReqMeta:
    """Metadata for a single request needing cartridge KV injection."""
    slot_mapping: torch.Tensor
    num_tokens: int


@dataclass
class CartridgeConnectorMetadata(KVConnectorMetadata):
    """Metadata passed from scheduler to worker."""
    requests: list[CartridgeReqMeta] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------

class CartridgeConnector(KVConnectorBase_V1):
    """KV connector that injects a pre-trained cartridge KV cache into
    vLLM's paged attention system.

    Loads one cartridge at init. Every incoming request gets the
    cartridge KV injected into its first N positions, where N is the
    cartridge's token count aligned to block boundaries. The scheduler
    skips prefill for those N tokens.

    This connector always loads the full cartridge. Block-level routing
    (loading K < N blocks) is handled by a separate routing layer.
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

        # Load cartridge via CartridgeStore
        from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_store import (
            CartridgeStore,
        )

        cartridge_path = self._kv_transfer_config.get_from_extra_config(
            "cartridge_path", None
        )
        if cartridge_path is None:
            raise ValueError(
                "CartridgeConnector requires 'cartridge_path' in "
                "--kv-connector-extra-config"
            )

        # Optional manifest path for compatibility validation
        manifest_path = self._kv_transfer_config.get_from_extra_config(
            "manifest_path", None
        )

        # Build or load manifest
        manifest = None
        if manifest_path is not None:
            from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_manifest import (
                CartridgeManifest,
            )
            manifest = CartridgeManifest.from_json(manifest_path)
            model_cfg = vllm_config.model_config.hf_config
            num_layers = getattr(model_cfg, "num_hidden_layers", 0)
            num_kv_heads = getattr(
                model_cfg, "num_key_value_heads",
                getattr(model_cfg, "num_attention_heads", 0),
            )
            head_dim = getattr(
                model_cfg, "head_dim",
                getattr(model_cfg, "hidden_size", 0)
                // max(getattr(model_cfg, "num_attention_heads", 1), 1),
            )

            errors = manifest.validate_against_model(
                model_id=vllm_config.model_config.model,
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
            )
            errors.extend(manifest.validate_against_block_size(
                self._block_size
            ))
            if errors:
                raise ValueError(
                    f"Cartridge manifest validation failed:\n"
                    + "\n".join(f"  - {e}" for e in errors)
                )
            logger.info("Cartridge manifest validated: %s",
                        manifest.cartridge_id)

        # Create a minimal manifest if none provided (for backward compat)
        if manifest is None:
            from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_manifest import (
                CartridgeManifest,
            )
            # Inspect the cartridge to build a minimal manifest
            cartridge_data = load_cartridge(cartridge_path)
            manifest = CartridgeManifest(
                cartridge_id="default",
                model_id="unknown",
                num_layers=cartridge_data["num_layers"],
                num_kv_heads=cartridge_data["num_kv_heads"],
                head_dim=cartridge_data["head_dim"],
                dtype="unknown",
                num_tokens_raw=cartridge_data["num_tokens"],
                num_tokens_aligned=align_to_block_size(
                    cartridge_data["num_tokens"], self._block_size),
                block_size=self._block_size,
                num_blocks=align_to_block_size(
                    cartridge_data["num_tokens"], self._block_size)
                // self._block_size,
                has_frozen_prefix=False,
            )
            del cartridge_data

        # Initialize store and load the cartridge
        self._store = CartridgeStore(block_size=self._block_size)
        self._cartridge_id = manifest.cartridge_id
        self._store.load(
            self._cartridge_id, cartridge_path, manifest, device="cpu"
        )
        self._store.pin(self._cartridge_id)  # prevent accidental eviction

        residency = self._store.get_residency(self._cartridge_id)
        self._num_cartridge_tokens = residency.num_tokens
        self._num_cartridge_blocks = (
            self._num_cartridge_tokens // self._block_size
        )
        self._num_layers = residency.num_layers

        logger.info(
            "Cartridge loaded via store: id=%s, %d layers, "
            "%d tokens (%d blocks of %d)",
            self._cartridge_id,
            self._num_layers,
            self._num_cartridge_tokens,
            self._num_cartridge_blocks,
            self._block_size,
        )

    # ==============================
    # Scheduler-side methods
    # ==============================

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """Report cartridge tokens as externally computed.

        Every request gets the cartridge injected. The number of
        externally-computed tokens is simply the cartridge's aligned
        token count, capped to the prompt length.
        """
        prompt_ids = request.prompt_token_ids
        if prompt_ids is None:
            return 0, False

        # Cap to prompt length so we never claim more positions than
        # the request has.
        matched = align_to_block_size(
            min(self._num_cartridge_tokens, len(prompt_ids)),
            self._block_size,
        )

        num_new = matched - num_computed_tokens
        if num_new <= 0:
            return 0, False

        logger.info(
            "Cartridge: %d tokens externally computed "
            "(%d new beyond %d already computed)",
            matched, num_new, num_computed_tokens,
        )
        return num_new, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ):
        """Record that this request needs KV loading.

        Idempotent: if called twice for the same request (which vLLM's
        connector API permits), the second call is a no-op.
        """
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
                block_ids = new_req.block_ids[0]
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
        from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_store import (
            ChunkKey,
        )

        metadata = self._get_connector_metadata()
        assert isinstance(metadata, CartridgeConnectorMetadata)

        if not metadata.requests:
            return

        # Acquire a ref on the cartridge for this batch of requests
        self._store.acquire(self._cartridge_id)

        try:
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

                    kv_cache_layer = kv_cache_attr[0]

                    layer_idx = self._extract_layer_idx(layer_name)
                    if layer_idx is None or layer_idx >= self._num_layers:
                        continue

                    # Read chunk from the store
                    chunk_key = ChunkKey(self._cartridge_id, layer_idx)
                    src_kv = self._store.get(chunk_key)
                    if src_kv is None:
                        logger.error(
                            "Cartridge chunk missing: %s layer %d",
                            self._cartridge_id, layer_idx,
                        )
                        continue

                    src_kv = src_kv.to(
                        device=kv_cache_layer.device,
                        dtype=kv_cache_layer.dtype,
                    )

                    inject_kv_into_paged_cache(
                        src_key=src_kv[0],
                        src_value=src_kv[1],
                        kv_cache_layer=kv_cache_layer,
                        slot_mapping=slot_mapping,
                    )
        finally:
            self._store.release(self._cartridge_id)

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
        """Extract layer index from ``'model.layers.0.self_attn'``."""
        parts = layer_name.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except ValueError:
                    pass
        return None
