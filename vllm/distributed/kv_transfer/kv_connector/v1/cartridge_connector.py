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

The connector supports two modes:

1. Singleton (``cartridge_path`` in extra_config): loads one cartridge
   at init and serves it to every request. Backward compatible with
   earlier single-cartridge deployments.

2. Multi-cartridge (``cartridges`` + ``router`` in extra_config): loads
   a fixed set of cartridges at init and dispatches per request via a
   ``CartridgeRouter`` (explicit id from request extras, label lookup
   against a CartridgeRegistry, or a composite chain). Per-request
   cartridge identity is carried end-to-end through connector
   metadata to the worker, so two concurrent requests can inject
   different cartridges into different allocated blocks without
   cross-contamination.

In both modes the connector reports per-request num_cartridge_tokens
as externally computed to the scheduler, which allocates blocks and
skips prefill accordingly.
"""
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_gpu_residency import (
    GPUResidencyManager)
from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_router import (
    CartridgeRouter, StaticCartridgeRouter, build_router_from_config)
from vllm.logger import init_logger
from vllm.attention.backends.abstract import AttentionMetadata
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash)
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
        raise ValueError(f"frozen_keys length ({len(frozen_keys)}) != "
                         f"trainable_keys length ({num_layers})")

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
                raise ValueError(f"Layer {layer_idx} K shape {k.shape} != "
                                 f"layer 0 K shape {prev_k.shape}")

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


def _publish_routing_state(
    selected_block_ids: "set[int] | frozenset[int] | None",
    num_blocks: int,
    block_size: int,
    request_id: str,
) -> None:
    """Scaffolding for A.2a step 5 — publish routing state for attention.

    Populates the thread-local ``routing_state`` with a ``RoutingPrior``
    describing the request's per-block selection. A routing-aware
    attention backend can read this and mask non-selected blocks at
    attention time.

    When ``selected_block_ids`` is ``None`` the thread-local is cleared
    so any residual state from a prior request doesn't leak into a
    non-routed request.

    Scope notes (documented here so consumers understand the contract):

    * Thread-local → single-request-at-a-time. Multi-request batches
      with distinct per-request priors would read the wrong state.
      Sufficient for paired A/B benchmarking; multi-request serving
      needs a batched handoff we haven't designed yet.
    * Selection is uniform across layers and heads — the mechanism
      plumbed through A.2a produces a single set. Per-layer/per-head
      variation is a follow-up.
    * No attention backend on this branch currently consumes
      ``routing_state``; this is producer-side scaffolding.
    """
    # Late import: keep the routing_state import off the connector's
    # hot import path when routing isn't configured.
    from vllm.v1.attention.routing_state import (RoutingPrior,
                                                 clear_routing_state,
                                                 set_routing_state)

    if selected_block_ids is None:
        clear_routing_state()
        return

    if not selected_block_ids:
        # Empty selection — nothing to route to. Clear.
        clear_routing_state()
        return

    # Build a RoutingPrior with kmeans_blocks carrying the selection.
    # topk_kmeans is the policy that consumes kmeans_blocks directly
    # without needing per-head block_affinities, so it matches our
    # uniform-across-layers selection mechanism.
    sorted_selection = sorted(selected_block_ids)
    prior = RoutingPrior(
        block_affinities=None,
        K=len(sorted_selection),
        mode="cartridge_prior",
        num_prefix_blocks=num_blocks,
        block_size=block_size,
        request_id=request_id,
        routing_policy="topk_kmeans",
        kmeans_blocks=sorted_selection,
    )
    set_routing_state(prior)


def selected_token_mask_from_block_ids(
    selected_block_ids: "set[int] | list[int]",
    num_tokens: int,
    block_size: int,
) -> torch.Tensor:
    """Expand a set of logical block ids to a per-token boolean mask.

    Pure helper. Token ``t`` is selected iff ``t // block_size`` is in
    ``selected_block_ids``. Trailing tokens that don't fit a full block
    (``num_tokens`` not a multiple of ``block_size``) are handled by
    rounding down — the partial tail only matters if its containing
    block id is selected.

    Args:
        selected_block_ids: logical block ids to keep (0-indexed over
            ``num_tokens // block_size``).
        num_tokens: length of the cartridge's token span.
        block_size: tokens per block.

    Returns:
        bool tensor of shape ``(num_tokens,)``. Safe to index a per-
        token source tensor or ``slot_mapping`` with.
    """
    selected = set(selected_block_ids)
    mask = torch.zeros(num_tokens, dtype=torch.bool)
    if not selected:
        return mask
    for t in range(num_tokens):
        if (t // block_size) in selected:
            mask[t] = True
    return mask


def inject_kv_into_paged_cache(
    src_key: torch.Tensor,
    src_value: torch.Tensor,
    kv_cache_layer: torch.Tensor,
    slot_mapping: torch.Tensor,
    selected_token_mask: "torch.Tensor | None" = None,
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
        selected_token_mask: optional (num_tokens,) bool tensor.
            When provided, only tokens where the mask is True are
            injected; non-selected slots are left untouched. Preserves
            logical positions — the selected tokens are written to
            *their original* slots, not compacted. This matches the
            position-preserving serving invariant: RoPE math depends
            on the logical offset, so selected blocks must be written
            to slots that reflect their original position.
            When None (default), all tokens are injected — same
            behavior as before.
    """
    # Auto-detect K/V split dimension.
    #
    # Three layouts observed across backends:
    #   (2, num_blocks, block_size, num_kv_heads, head_dim)  — K/V stacked at dim 0
    #   (num_blocks, 2, block_size, num_kv_heads, head_dim)  — K/V stacked at dim 1
    #   (num_blocks, block_size, num_kv_heads, head_dim)     — K-only, V lives in a sibling
    #                                                          tensor; kv_cache_layer is
    #                                                          passed as a tuple/list of
    #                                                          (k_cache, v_cache).
    if isinstance(kv_cache_layer, (tuple, list)) and len(kv_cache_layer) == 2:
        key_cache, value_cache = kv_cache_layer[0], kv_cache_layer[1]
    elif kv_cache_layer.dim() == 5 and kv_cache_layer.shape[0] == 2:
        key_cache, value_cache = kv_cache_layer.unbind(0)
    elif kv_cache_layer.dim() == 5 and kv_cache_layer.shape[1] == 2:
        key_cache, value_cache = kv_cache_layer.unbind(1)
    else:
        raise ValueError(
            f"Cannot determine K/V split for cache shape "
            f"{kv_cache_layer.shape if hasattr(kv_cache_layer, 'shape') else type(kv_cache_layer)}."
            " Expected a 5D tensor with K/V stacked at dim 0 or 1, "
            "or a (k_cache, v_cache) tuple.")

    # Align the src tensors to slot_mapping length. A cartridge may
    # carry a few extra tokens beyond the request's aligned
    # num_tokens (e.g. frozen BOS prepended during load), but only
    # the first len(slot_mapping) tokens land in the paged cache.
    # Trimming here keeps the mask validation below simple and the
    # trim is free when the two lengths already match.
    n_inject = slot_mapping.shape[0]
    if src_key.shape[0] > n_inject:
        src_key = src_key[:n_inject]
        src_value = src_value[:n_inject]

    # Apply the optional selection before invoking the kernel.
    # Keep the original slot indices for the selected tokens
    # (position-preserving); we just hand the kernel fewer rows to
    # write.
    if selected_token_mask is not None:
        if selected_token_mask.dtype != torch.bool:
            raise ValueError(f"selected_token_mask must be bool, got "
                             f"{selected_token_mask.dtype}")
        if selected_token_mask.shape[0] != src_key.shape[0]:
            raise ValueError(
                f"selected_token_mask length {selected_token_mask.shape[0]} "
                f"does not match src_key num_tokens {src_key.shape[0]}")
        mask_on_src = selected_token_mask.to(src_key.device)
        # If nothing is selected, there's nothing to write. This is
        # valid (attention backend will see null blocks for all
        # positions, which is a caller-level policy choice).
        if not bool(mask_on_src.any()):
            return
        src_key = src_key[mask_on_src]
        src_value = src_value[mask_on_src]
        slot_mapping = slot_mapping[selected_token_mask.to(
            slot_mapping.device)]

    k_scale = torch.tensor(1.0, dtype=torch.float32, device=key_cache.device)
    v_scale = torch.tensor(1.0, dtype=torch.float32, device=value_cache.device)

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
    """Metadata for a single request needing cartridge KV injection.

    Carries the per-request ``cartridge_id`` so the worker can fetch
    the correct cartridge chunks from the store. Without this, two
    concurrent requests would all receive the same (singleton)
    cartridge regardless of how the scheduler resolved them.

    ``selected_block_ids`` is an optional routing-aware inject
    selection. When set, only the listed logical block ids are
    injected into the paged cache for this request; non-selected
    slots are left untouched (position-preserving: selected blocks
    land at their original logical positions, not compacted). When
    None (default), the full cartridge is injected — same behaviour
    as before.

    This field is a pure mechanism handoff from whatever policy
    decides the selection (today: nobody — the scheduler path still
    leaves it None; tomorrow: a CartridgeKRIProvider consultation in
    the connector's scheduler path). Keeping the field here lets
    step 2 of the A.2a plan ship without any policy wiring, so the
    mechanism is independently benchable.
    """
    cartridge_id: str
    slot_mapping: torch.Tensor
    num_tokens: int
    selected_block_ids: "set[int] | frozenset[int] | None" = None


@dataclass
class CartridgeConnectorMetadata(KVConnectorMetadata):
    """Metadata passed from scheduler to worker."""
    requests: list[CartridgeReqMeta] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------


class CartridgeConnector(KVConnectorBase_V1):
    """KV connector that injects pre-trained cartridge KV caches into
    vLLM's paged attention system.

    Supports two configurations:

    * **Singleton** (``cartridge_path``): one cartridge, injected into
      every request. Backward-compatible with earlier deployments.

    * **Multi-cartridge** (``cartridges`` + ``router``): multiple
      cartridges loaded at init, per-request dispatch via a
      ``CartridgeRouter``. The chosen ``cartridge_id`` is carried
      through scheduler-side state into per-request connector
      metadata, and the worker dispatches to the right cartridge for
      each request. Two concurrent requests with different cartridge
      IDs inject into different allocated blocks with no
      cross-contamination.
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

        # Scheduler-side per-request state: request_id -> cartridge_id
        # resolved by the router at get_num_new_matched_tokens() time
        # and consumed by build_connector_meta().
        self._request_cartridge_ids: dict[str, str] = {}
        # request_id -> aligned token count for the chosen cartridge.
        self._request_num_tokens: dict[str, int] = {}
        # Set of request_ids that have been committed by
        # update_state_after_alloc(); cleared each tick in
        # build_connector_meta().
        self._requests_need_load: set[str] = set()
        # A.2a step 4: per-request routing selection, populated in
        # update_state_after_alloc() after the scheduler has allocated
        # blocks. Consumed by build_connector_meta() to populate
        # CartridgeReqMeta and by get_block_skip_list() to drive
        # scheduler-side null_block replacement.
        self._request_selected_block_ids: dict[str, set[int]] = {}

        self._vllm_config = vllm_config
        self._load_cartridges_from_config()
        self._router: CartridgeRouter = self._build_router_from_config()
        self._residency: GPUResidencyManager = self._build_residency()
        self._preload_cartridges_if_configured()

        # A.2a step 3 — optional routing-prior consultation.
        # Off by default: if the deployment does not configure the
        # "routing" block in extra_config, the connector behaves
        # exactly as before.
        self._kri_provider: Any | None = None
        self._routing_K: int | None = None
        self._load_routing_priors_if_configured()

        logger.info(
            "CartridgeConnector ready: %d cartridge(s) registered, "
            "router=%s, gpu_residency=%s, routing_priors=%s",
            len(self._cartridge_meta),
            type(self._router).__name__,
            self._residency is not None,
            (f"K={self._routing_K}, "
             f"priors={len(self._kri_provider.registered_keys)}"
             if self._kri_provider is not None else "disabled"),
        )

    # ------------------------------------------------------------------
    # Init helpers: cartridge loading + router construction
    # ------------------------------------------------------------------

    def _load_cartridges_from_config(self) -> None:
        """Populate self._store and self._cartridge_meta.

        Accepts either ``cartridge_path`` (singleton) or ``cartridges``
        (a list of ``{cartridge_id, path, manifest_path?}`` dicts). At
        least one of the two must be set.
        """
        from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_store import (
            CartridgeStore)

        singleton_path = self._kv_transfer_config.get_from_extra_config(
            "cartridge_path", None)
        cartridges_list = self._kv_transfer_config.get_from_extra_config(
            "cartridges", None)
        if singleton_path is None and not cartridges_list:
            raise ValueError(
                "CartridgeConnector requires either 'cartridge_path' "
                "(singleton mode) or 'cartridges' (multi-cartridge "
                "mode) in --kv-connector-extra-config")

        self._store = CartridgeStore(block_size=self._block_size)
        # cartridge_id -> {"num_tokens": int, "num_blocks": int,
        #                  "num_layers": int}
        self._cartridge_meta: dict[str, dict] = {}
        # For singleton/default fallback when the router returns None
        # but we need a cartridge_id anyway (legacy behaviour).
        self._default_cartridge_id: Optional[str] = None

        if singleton_path is not None:
            manifest_path = (self._kv_transfer_config.get_from_extra_config(
                "manifest_path", None))
            cart_id = self._load_one(
                cartridge_path=singleton_path,
                manifest_path=manifest_path,
                explicit_cartridge_id=None,
            )
            self._default_cartridge_id = cart_id

        if cartridges_list:
            if not isinstance(cartridges_list, list):
                raise ValueError("'cartridges' must be a list of entries")
            for entry in cartridges_list:
                if not isinstance(entry, dict):
                    raise ValueError("each 'cartridges' entry must be a dict")
                path = entry.get("path")
                if not path:
                    raise ValueError("each 'cartridges' entry requires 'path'")
                self._load_one(
                    cartridge_path=path,
                    manifest_path=entry.get("manifest_path"),
                    explicit_cartridge_id=entry.get("cartridge_id"),
                )
            # First listed cartridge is the default fallback if no
            # singleton was provided.
            if self._default_cartridge_id is None:
                first_entry = cartridges_list[0]
                self._default_cartridge_id = (first_entry.get("cartridge_id")
                                              or next(
                                                  iter(self._cartridge_meta)))

    def _load_one(
        self,
        cartridge_path: str,
        manifest_path: Optional[str],
        explicit_cartridge_id: Optional[str],
    ) -> str:
        """Load a single cartridge, register it in the store, and
        record its per-cartridge sizing in self._cartridge_meta.

        Returns the cartridge_id that was registered.
        """
        from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_manifest import (
            CartridgeManifest)

        manifest: Optional[CartridgeManifest] = None
        if manifest_path is not None:
            manifest = CartridgeManifest.from_json(manifest_path)
            model_cfg = self._vllm_config.model_config.hf_config
            num_layers = getattr(model_cfg, "num_hidden_layers", 0)
            num_kv_heads = getattr(
                model_cfg,
                "num_key_value_heads",
                getattr(model_cfg, "num_attention_heads", 0),
            )
            head_dim = getattr(
                model_cfg,
                "head_dim",
                getattr(model_cfg, "hidden_size", 0) //
                max(getattr(model_cfg, "num_attention_heads", 1), 1),
            )
            errors = manifest.validate_against_model(
                model_id=self._vllm_config.model_config.model,
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
            )
            errors.extend(
                manifest.validate_against_block_size(self._block_size))
            if errors:
                raise ValueError("Cartridge manifest validation failed:\n" +
                                 "\n".join(f"  - {e}" for e in errors))
            logger.info("Cartridge manifest validated: %s",
                        manifest.cartridge_id)
            if (explicit_cartridge_id is not None
                    and explicit_cartridge_id != manifest.cartridge_id):
                raise ValueError(
                    f"cartridge_id mismatch: config says "
                    f"{explicit_cartridge_id!r} but manifest says "
                    f"{manifest.cartridge_id!r}")

        if manifest is None:
            cartridge_data = load_cartridge(cartridge_path)
            aligned = align_to_block_size(cartridge_data["num_tokens"],
                                          self._block_size)
            manifest = CartridgeManifest(
                cartridge_id=explicit_cartridge_id or "default",
                model_id="unknown",
                num_layers=cartridge_data["num_layers"],
                num_kv_heads=cartridge_data["num_kv_heads"],
                head_dim=cartridge_data["head_dim"],
                dtype="unknown",
                num_tokens_raw=cartridge_data["num_tokens"],
                num_tokens_aligned=aligned,
                block_size=self._block_size,
                num_blocks=aligned // self._block_size,
                has_frozen_prefix=False,
            )
            del cartridge_data

        cart_id = manifest.cartridge_id
        if cart_id in self._cartridge_meta:
            raise ValueError(f"duplicate cartridge_id: {cart_id!r}")

        self._store.load(cart_id, cartridge_path, manifest, device="cpu")
        # Intentionally NOT calling store.pin() here. GPU residency
        # is owned by GPUResidencyManager; the CPU store holds the
        # cartridge for the lifetime of the connector, but the GPU
        # tier is bounded/LRU/refcounted by the residency manager.

        residency = self._store.get_residency(cart_id)
        num_tokens = residency.num_tokens
        num_blocks = num_tokens // self._block_size
        self._cartridge_meta[cart_id] = {
            "num_tokens": num_tokens,
            "num_blocks": num_blocks,
            "num_layers": residency.num_layers,
        }
        logger.info(
            "Cartridge loaded: id=%s, %d layers, %d tokens "
            "(%d blocks of %d)",
            cart_id,
            residency.num_layers,
            num_tokens,
            num_blocks,
            self._block_size,
        )
        return cart_id

    def _build_residency(self) -> GPUResidencyManager:
        """Build the GPU residency manager.

        Reads ``gpu_capacity_bytes`` from extra_config. Defaults to
        a conservative value sized to hold ~8 cartridges of the
        largest loaded size, which keeps memory pressure
        unsurprising on the first deployment. Operators who want
        different policy pass an explicit capacity.

        Device defaults to ``cuda`` when CUDA is available, else
        ``cpu`` (the tests exercise the CPU path explicitly).
        """
        explicit = self._kv_transfer_config.get_from_extra_config(
            "gpu_capacity_bytes", None)
        if explicit is not None:
            capacity = int(explicit)
        else:
            # Default: 8 × largest-loaded-cartridge bytes, or 1 GiB,
            # whichever is larger. Callers should set explicitly in
            # production.
            max_cart_bytes = 0
            for cart_id in self._cartridge_meta:
                res = self._store.get_residency(cart_id)
                if res is None:
                    continue
                # Approximate bytes: layers × (2 K+V) × tokens × heads ×
                # dim × 2 (bfloat16). We don't yet know the dtype at
                # connector-init time, use 4 bytes to be conservative.
                model_cfg = self._vllm_config.model_config.hf_config
                heads = getattr(
                    model_cfg,
                    "num_key_value_heads",
                    getattr(model_cfg, "num_attention_heads", 1),
                )
                dim = getattr(
                    model_cfg,
                    "head_dim",
                    getattr(model_cfg, "hidden_size", 0) //
                    max(getattr(model_cfg, "num_attention_heads", 1), 1),
                )
                bytes_est = (res.num_layers * 2 * res.num_tokens * heads *
                             dim * 4)
                max_cart_bytes = max(max_cart_bytes, bytes_est)
            capacity = max(8 * max_cart_bytes, 1 << 30)

        device = self._kv_transfer_config.get_from_extra_config(
            "gpu_residency_device", None)
        if device is None:
            try:
                import torch as _torch  # noqa: F401 (reuse outer import)
                device = ("cuda" if torch.cuda.is_available() else "cpu")
            except Exception:
                device = "cpu"

        return GPUResidencyManager(
            store=self._store,
            capacity_bytes=capacity,
            device=device,
            eviction_policy="lru",
        )

    def _preload_cartridges_if_configured(self) -> None:
        """Optional eager preload for controlled experiments.

        ``preload`` in extra_config is a list of cartridge_ids to
        promote to the GPU tier at init (without pinning). This is
        useful for benchmarks where the first-request latency would
        skew results, but defaults to empty — production paths
        lazily promote on first use, which is the whole point of
        having a residency manager.
        """
        preload = self._kv_transfer_config.get_from_extra_config(
            "preload", None)
        if not preload:
            return
        if not isinstance(preload, list):
            raise ValueError("'preload' must be a list of cartridge_ids")
        for cart_id in preload:
            if cart_id not in self._cartridge_meta:
                logger.warning(
                    "preload: cartridge %s not registered, skipping",
                    cart_id,
                )
                continue
            ok = self._residency.prefetch(str(cart_id))
            logger.info(
                "preload: cartridge=%s resident=%s",
                cart_id,
                ok,
            )

    def _load_routing_priors_if_configured(self) -> None:
        """Opt-in: load routing priors via ``CartridgeKRIProvider``.

        Driven by ``extra_config["routing"]``. Example config::

            "routing": {
                "enabled": true,
                "K": 8,
                "priors": {
                    "cart_medical": "/data/priors/medical.pt",
                    "cart_legal":   "/data/priors/legal.pt"
                }
            }

        Semantics:

        * ``enabled`` omitted or false → no-op. Connector behaves
          exactly as before.
        * ``enabled=true`` but no ``priors`` registered → no-op with a
          warning.
        * ``enabled=true`` + ``priors`` → create the provider, load
          each prior file once, register it under the corresponding
          cartridge id. ``K`` defaults to 8 if unset.

        The provider is queried in ``build_connector_meta`` on a
        per-request basis. A cartridge without a registered prior
        falls through to the full-inject path (``selected_block_ids
        = None``). A cartridge with a prior gets ``selected_block_ids
        = set(manifest.block_indices)``.
        """
        routing_cfg = self._kv_transfer_config.get_from_extra_config(
            "routing", None)
        if not routing_cfg:
            return
        if not isinstance(routing_cfg, dict):
            raise ValueError("'routing' in extra_config must be a dict")
        if not routing_cfg.get("enabled", False):
            return

        K = int(routing_cfg.get("K", 8))
        priors = routing_cfg.get("priors", {}) or {}
        if not isinstance(priors, dict):
            raise ValueError(
                "'routing.priors' must be a dict of cart_id -> path")
        if not priors:
            logger.warning(
                "routing.enabled=true but no routing.priors "
                "configured; cartridge serving will fall through to "
                "the full-inject path for every request")
            return

        # Late import to keep the routing-prior loader off the
        # connector's import graph when routing is disabled.
        from vllm.distributed.kv_transfer.kv_connector.v1.routing_prior.cartridge_kri import (
            CartridgeKRIProvider)

        provider = CartridgeKRIProvider()
        for cart_id, prior_path in priors.items():
            if cart_id not in self._cartridge_meta:
                logger.warning(
                    "routing.priors references unknown cartridge "
                    "id %r; skipping (path=%s)",
                    cart_id,
                    prior_path,
                )
                continue
            try:
                provider.register_prior(
                    prefix_hash=cart_id,
                    prior_path=prior_path,
                )
                logger.info(
                    "routing: registered prior for cartridge %s "
                    "from %s (K=%d)",
                    cart_id,
                    prior_path,
                    K,
                )
            except Exception as e:
                # Fail soft — a bad prior path should not tank the
                # whole connector. Requests for this cartridge will
                # simply miss the prior lookup and fall through to
                # full inject.
                logger.error(
                    "routing: failed to register prior for "
                    "cartridge %s from %s: %s",
                    cart_id,
                    prior_path,
                    e,
                )

        # Only commit provider state if at least one prior loaded.
        if provider.registered_keys:
            self._kri_provider = provider
            self._routing_K = K
        else:
            logger.warning("routing: no priors successfully loaded; "
                           "routing-aware inject disabled for this session")

    def _build_router_from_config(self) -> CartridgeRouter:
        """Build the CartridgeRouter from extra_config.

        If no router config is provided and exactly one cartridge is
        loaded, defaults to StaticCartridgeRouter bound to that id
        (singleton semantics). Otherwise raises.
        """
        router_cfg = self._kv_transfer_config.get_from_extra_config(
            "router", None)
        if router_cfg is None:
            if len(self._cartridge_meta) == 1:
                only_id = next(iter(self._cartridge_meta))
                return StaticCartridgeRouter(only_id)
            raise ValueError("Multi-cartridge mode requires 'router' in "
                             "--kv-connector-extra-config when more than one "
                             "cartridge is loaded")
        if not isinstance(router_cfg, dict):
            raise ValueError("'router' must be a dict describing the router "
                             "configuration")
        return build_router_from_config(
            router_cfg,
            registry=None,  # Registry-based routers must be wired
            # externally via set_registry() for now.
            default_cartridge_id=self._default_cartridge_id,
        )

    # Backwards-compat accessors. Tests and downstream code may still
    # reference these from singleton-era APIs. They reflect the first
    # (or only) loaded cartridge.
    @property
    def _cartridge_id(self) -> str:
        return self._default_cartridge_id or next(iter(self._cartridge_meta))

    @property
    def _num_cartridge_tokens(self) -> int:
        return self._cartridge_meta[self._cartridge_id]["num_tokens"]

    @property
    def _num_cartridge_blocks(self) -> int:
        return self._cartridge_meta[self._cartridge_id]["num_blocks"]

    @property
    def _num_layers(self) -> int:
        return self._cartridge_meta[self._cartridge_id]["num_layers"]

    def set_router(self, router: CartridgeRouter) -> None:
        """Override the router after construction.

        Useful for test injection and for wiring registry-backed
        routers that need a live CartridgeRegistry handle.
        """
        self._router = router

    # ==============================
    # Scheduler-side methods
    # ==============================

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """Resolve the request to a cartridge and report its token
        count as externally computed.

        The router decides *which* cartridge this request should use
        (explicit id in extras, label lookup, static fallback). The
        resolved ``cartridge_id`` is stashed in per-request state so
        that ``build_connector_meta()`` and ``start_load_kv()`` can
        dispatch to the correct cartridge without re-resolving.

        If the router returns None (no cartridge for this request),
        this connector reports zero matched tokens and the request
        falls through to normal prefill.
        """
        prompt_ids = request.prompt_token_ids
        if prompt_ids is None:
            return 0, False

        cart_id = self._router.resolve(request)
        if cart_id is None:
            # Router explicitly declined: no cartridge injection.
            return 0, False
        if cart_id not in self._cartridge_meta:
            logger.warning(
                "Router resolved request %s to unknown cartridge_id "
                "%r; falling through to normal prefill",
                request.request_id,
                cart_id,
            )
            return 0, False

        cart_info = self._cartridge_meta[cart_id]
        # Cap to prompt length so we never claim more positions than
        # the request has. Per-cartridge token count — different
        # cartridges can have different sizes.
        matched = align_to_block_size(
            min(cart_info["num_tokens"], len(prompt_ids)),
            self._block_size,
        )

        num_new = matched - num_computed_tokens
        if num_new <= 0:
            return 0, False

        # Stash the resolved cartridge_id and the *matched* (aligned,
        # prompt-capped) token count. build_connector_meta() will use
        # these to build the slot mapping.
        self._request_cartridge_ids[request.request_id] = cart_id
        self._request_num_tokens[request.request_id] = matched

        logger.info(
            "Cartridge: request %s routed to %s — %d tokens "
            "externally computed (%d new beyond %d already computed)",
            request.request_id,
            cart_id,
            matched,
            num_new,
            num_computed_tokens,
        )
        return num_new, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ):
        """Commit a resolved request for loading.

        vLLM's connector API permits the scheduler to call this twice
        for the same request; the second call is a no-op because our
        state is keyed by request_id. If the request was resolved but
        never allocated (e.g. capacity pressure), its entry is cleared
        in ``build_connector_meta()``.

        A.2a step 4: also consults the routing prior (if configured)
        here so the selection is available before
        ``build_connector_meta`` runs and — more importantly — before
        the scheduler reads ``get_block_skip_list`` to mutate
        block_tables with null_block.
        """
        if num_external_tokens > 0:
            # Sanity check: get_num_new_matched_tokens should have
            # populated our state for this request.
            if (request.request_id not in self._request_cartridge_ids):
                logger.warning(
                    "update_state_after_alloc: no cartridge_id "
                    "recorded for request %s; skipping",
                    request.request_id,
                )
                return
            # Mark as committed for this tick's build_connector_meta.
            self._requests_need_load.add(request.request_id)

            # A.2a step 4: populate the routing selection now so both
            # build_connector_meta and get_block_skip_list can read it.
            if self._kri_provider is not None and self._routing_K is not None:
                cart_id = self._request_cartridge_ids[request.request_id]
                manifest = self._kri_provider.get_manifest(
                    prefix_hash=cart_id,
                    query_hash=None,
                    K=self._routing_K,
                )
                if manifest is not None:
                    self._request_selected_block_ids[request.request_id] = (
                        set(manifest.block_indices))
                    logger.info(
                        "routing: request %s cartridge %s "
                        "selected %d/%d blocks (K=%d)",
                        request.request_id,
                        cart_id,
                        len(manifest.block_indices),
                        manifest.total_blocks,
                        manifest.K,
                    )

    def get_block_skip_list(
        self,
        request_id: str,
        num_logical_blocks: int,
    ) -> "list[int] | None":
        """A.2a step 4: produce the logical blocks the scheduler should
        null-block for this request.

        Reads from the routing selection populated in
        ``update_state_after_alloc``. The skip list is the complement
        of the selection within ``[0, num_logical_blocks)``. Returns
        ``None`` if the request has no routing selection — identical
        to the connector-base default, so a non-routed cartridge
        serve path does not trigger any block_table mutation.
        """
        sel = self._request_selected_block_ids.get(request_id)
        if sel is None:
            return None
        # Complement: every logical block NOT in the selection.
        # Filter against num_logical_blocks so we never return an
        # out-of-range entry even if the prior covered more blocks
        # than the current request actually uses.
        skip: list[int] = [
            i for i in range(num_logical_blocks) if i not in sel
        ]
        return skip

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        """Build metadata with per-request cartridge_id + slot mappings."""
        meta = CartridgeConnectorMetadata()

        for new_req in scheduler_output.scheduled_new_reqs:
            if new_req.req_id not in self._requests_need_load:
                continue

            cart_id = self._request_cartridge_ids.get(new_req.req_id)
            num_tokens = self._request_num_tokens.get(new_req.req_id)
            if cart_id is None or num_tokens is None:
                logger.warning(
                    "build_connector_meta: missing resolved state "
                    "for request %s; skipping",
                    new_req.req_id,
                )
                continue

            num_blocks = num_tokens // self._block_size
            block_ids = new_req.block_ids[0]
            cartridge_block_ids = block_ids[:num_blocks]
            block_ids_tensor = torch.tensor(cartridge_block_ids,
                                            dtype=torch.long)
            block_offsets = torch.arange(0, self._block_size, dtype=torch.long)
            slot_mapping = (
                block_offsets.reshape(1, self._block_size) +
                block_ids_tensor.reshape(-1, 1) * self._block_size).flatten()

            # A.2a: routing selection was populated in
            # update_state_after_alloc (so the scheduler's null_block
            # pass can see it earlier). Read it back here.
            selected_block_ids = self._request_selected_block_ids.get(
                new_req.req_id)

            meta.requests.append(
                CartridgeReqMeta(
                    cartridge_id=cart_id,
                    slot_mapping=slot_mapping,
                    num_tokens=num_tokens,
                    selected_block_ids=selected_block_ids,
                ))

        # Clear tick-local state for consumed AND any stale resolved
        # requests that never got committed (e.g. deferred by the
        # scheduler; they'll be re-resolved next tick).
        self._requests_need_load.clear()
        # Only clear state for requests scheduled this tick — requests
        # that were resolved but not yet scheduled may still be pending.
        # In practice resolved+not-scheduled cleans up on reschedule.
        scheduled_ids = {r.req_id for r in scheduler_output.scheduled_new_reqs}
        for req_id in list(self._request_cartridge_ids.keys()):
            if req_id in scheduled_ids:
                self._request_cartridge_ids.pop(req_id, None)
                self._request_num_tokens.pop(req_id, None)
                self._request_selected_block_ids.pop(req_id, None)

        return meta

    # ==============================
    # Worker-side methods
    # ==============================

    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs: Any) -> None:
        """Inject cartridge KV into allocated GPU blocks, dispatching
        per request to the cartridge chosen by the scheduler-side
        router.

        Different requests in the same batch may have different
        ``cartridge_id`` values; each is loaded from the store and
        written into that request's slot mapping independently. No
        shared state across requests other than the store and the
        paged KV cache.
        """
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, CartridgeConnectorMetadata)

        if not metadata.requests:
            return

        # Determine target device/dtype from the first real paged-cache
        # layer in the forward context. Cartridges are promoted to the
        # GPU tier at that (device, dtype) once per cartridge per
        # promotion; from then on all requests for that cartridge hit
        # the tier without re-converting.
        target_device, target_dtype = self._peek_target_device_dtype(
            forward_context)

        # Acquire GPU residency per unique cartridge — this is the
        # pin. Promotion happens here on first use; subsequent
        # requests for the same cartridge are GPU hits.
        unique_ids = {r.cartridge_id for r in metadata.requests}
        acquired_ids: list[str] = []
        for cart_id in unique_ids:
            try:
                self._residency.acquire(cart_id, dtype=target_dtype)
                acquired_ids.append(cart_id)
            except Exception as e:
                logger.error(
                    "start_load_kv: cannot acquire cartridge %r "
                    "on GPU tier: %s",
                    cart_id,
                    e,
                )
                # Continue — releasing what we already acquired in
                # the finally block; request will fall through.

        try:
            for request in metadata.requests:
                cart_id = request.cartridge_id
                if cart_id not in acquired_ids:
                    continue
                num_layers = self._cartridge_meta.get(cart_id,
                                                      {}).get("num_layers")
                if num_layers is None:
                    logger.error(
                        "start_load_kv: unknown cartridge_id %r; "
                        "skipping request",
                        cart_id,
                    )
                    continue

                self._inject_request(
                    request=request,
                    num_layers=num_layers,
                    forward_context=forward_context,
                    target_device=target_device,
                )

                # A.2c: xa refinement — if a fresh-K/V reference is
                # available for this request, compute HKVD indices and
                # scatter fresh K/V at those positions.
                self._maybe_xa_refine(
                    request=request,
                    num_layers=num_layers,
                    forward_context=forward_context,
                    target_device=target_device,
                )
        finally:
            for cart_id in acquired_ids:
                self._residency.release(cart_id)

    # ------------------------------------------------------------------
    # Injection helpers (separated so the fetch tier can evolve
    # without touching the slot-mapping/write path)
    # ------------------------------------------------------------------

    def _peek_target_device_dtype(
        self,
        forward_context: "ForwardContext",
    ) -> tuple[Any, Any]:
        """Infer the paged-cache device + dtype from the first layer
        in the forward context.

        Returns ``(device, dtype)``; both may be None if no layer
        has a populated ``kv_cache``. The residency manager tolerates
        ``dtype=None`` (means "keep store dtype").
        """
        for layer_name in forward_context.no_compile_layers:
            layer = forward_context.no_compile_layers[layer_name]
            kv_cache_attr = getattr(layer, "kv_cache", None)
            if kv_cache_attr is None:
                continue
            kv = kv_cache_attr[0]
            return kv.device, kv.dtype
        return None, None

    def _inject_request(
        self,
        request: CartridgeReqMeta,
        num_layers: int,
        forward_context: "ForwardContext",
        target_device: Any,
    ) -> None:
        """Write all layers of one request's cartridge into the
        paged cache at the request's slot mapping.

        Source tensors come from the residency manager (GPU-resident
        after acquire). This function only moves the slot mapping to
        device and calls the injection kernel; it does not do any
        CPU->GPU tensor copy on the hot path.

        If ``request.selected_block_ids`` is set, injection is sparse:
        only those logical blocks are written, at their original slot
        positions. Non-selected slots are left untouched. The mask is
        built once per request and reused across all layers.
        """
        cart_id = request.cartridge_id
        # Slot mapping needs to be on the same device as the paged
        # cache. The tensor is small (int64, length ~= num_tokens),
        # so this copy is cheap and unrelated to the cartridge bulk.
        slot_mapping = request.slot_mapping
        if target_device is not None:
            slot_mapping = slot_mapping.to(target_device)
        else:
            slot_mapping = slot_mapping.cuda()

        # A.2a step 2: optional sparse inject. Build the per-token
        # selection mask once — same mask is reused for every layer
        # in this step. Per-layer distinct masks are future work when
        # the policy consultation in step 3 produces per-layer data;
        # for now the selection is uniform across layers.
        selection_mask: torch.Tensor | None = None
        if request.selected_block_ids is not None:
            selection_mask = selected_token_mask_from_block_ids(
                request.selected_block_ids,
                request.num_tokens,
                self._block_size,
            )
            # Move mask to same device as slot_mapping so the
            # per-layer inject doesn't keep re-copying it.
            selection_mask = selection_mask.to(slot_mapping.device)
            logger.info(
                "Injecting cartridge %s KV sparsely "
                "(%d blocks selected of %d total) into GPU cache",
                cart_id,
                len(request.selected_block_ids),
                request.num_tokens // self._block_size,
            )
        else:
            logger.info(
                "Injecting cartridge %s KV (%d tokens) into GPU cache",
                cart_id,
                request.num_tokens,
            )

        # A.2a step 5 (scaffolding): publish the per-request routing
        # selection into the thread-local routing_state so a
        # downstream attention backend that knows how to consume it
        # can mask non-selected blocks at attention time. Today no
        # attention backend on this branch consumes routing_state —
        # this scaffolding establishes the producer side so the
        # consumer side can be added as a separate change without
        # further connector-side work.
        #
        # Scope: thread-local, single-request-at-a-time. Sufficient
        # for paired A/B benchmarking; multi-request batches with
        # distinct per-request priors are not yet supported.
        _publish_routing_state(
            selected_block_ids=request.selected_block_ids,
            num_blocks=request.num_tokens // self._block_size,
            block_size=self._block_size,
            request_id=cart_id,
        )

        for layer_name in forward_context.no_compile_layers:
            layer = forward_context.no_compile_layers[layer_name]
            kv_cache_attr = getattr(layer, "kv_cache", None)
            if kv_cache_attr is None:
                continue

            # Backends provide kv_cache either as [stacked_5d_tensor] (len=1,
            # element is (2, num_blocks, block_size, heads, dim) or similar)
            # or as [k_cache, v_cache] (len=2, both 4D separate tensors).
            # inject_kv_into_paged_cache handles both layouts when given the
            # right shape — pass the tuple if we have separate K/V, else
            # index the stacked element.
            if (len(kv_cache_attr) == 2 and hasattr(kv_cache_attr[0], "dim")
                    and kv_cache_attr[0].dim() == 4):
                kv_cache_layer = (kv_cache_attr[0], kv_cache_attr[1])
            else:
                kv_cache_layer = kv_cache_attr[0]
            layer_idx = self._extract_layer_idx(layer_name)
            if layer_idx is None or layer_idx >= num_layers:
                continue

            src_kv = self._fetch_source_kv(cart_id, layer_idx)
            if src_kv is None:
                continue

            inject_kv_into_paged_cache(
                src_key=src_kv[0],
                src_value=src_kv[1],
                kv_cache_layer=kv_cache_layer,
                slot_mapping=slot_mapping,
                selected_token_mask=selection_mask,
            )

    def _maybe_xa_refine(
        self,
        request: CartridgeReqMeta,
        num_layers: int,
        forward_context: "ForwardContext",
        target_device: Any,
    ) -> None:
        """A.2c cross-attention refinement via HKVD scatter.

        After the main cartridge inject, check for a fresh-K/V reference
        in a well-known file path. If present, compute HKVD positions at
        the check layer and re-inject fresh K/V at those positions.

        The fresh-K/V file is produced externally by the eval harness
        (which runs a separate HF model forward to compute it). This
        file-backed approach works across the vLLM multi-process
        boundary.

        File protocol:
          /tmp/vllm_xa_fresh_kv.pt — a dict with:
            fresh_K_per_layer: list[Tensor (1, heads, T, dim)]
            fresh_V_per_layer: list[Tensor (1, heads, T, dim)]
            hkvd_ratio: float (e.g. 0.25)
            check_layer: int (e.g. 1)
        """
        import os

        # The xa refinement side-channel deserializes a pickled torch
        # file with weights_only=False, which is an arbitrary code
        # execution surface. Require an explicit opt-in env var in
        # addition to a path being set, and never enable by default.
        xa_path = os.environ.get("VLLM_XA_FRESH_KV_PATH")
        if not xa_path:
            return
        if os.environ.get("VLLM_XA_ENABLE_UNSAFE_LOAD", "0") != "1":
            logger.warning(
                "xa_refine: VLLM_XA_FRESH_KV_PATH=%s is set but "
                "VLLM_XA_ENABLE_UNSAFE_LOAD!=1; skipping (the side-"
                "channel uses torch.load with weights_only=False, an "
                "arbitrary code execution surface; opt in only on "
                "trusted inputs).",
                xa_path,
            )
            return
        if not os.path.exists(xa_path):
            return

        import torch

        from vllm.distributed.kv_transfer.kv_connector.v1.xa_refinement import (
            compute_hkvd_indices)

        logger.warning(
            "xa_refine: loading %s with weights_only=False (opt-in via "
            "VLLM_XA_ENABLE_UNSAFE_LOAD=1)",
            xa_path,
        )

        try:
            ref = torch.load(xa_path, map_location="cpu", weights_only=False)
        except Exception as e:
            logger.warning("xa_refine: failed to load %s: %s", xa_path, e)
            return

        fresh_K = ref.get("fresh_K_per_layer", [])
        fresh_V = ref.get("fresh_V_per_layer", [])
        hkvd_ratio = ref.get("hkvd_ratio", 0.25)
        check_layer = ref.get("check_layer", 1)

        if len(fresh_K) < num_layers or len(fresh_V) < num_layers:
            logger.warning(
                "xa_refine: fresh K/V has %d layers, need %d",
                len(fresh_K),
                num_layers,
            )
            return

        # Get the CURRENTLY-INJECTED K at check_layer from the paged
        # cache to diff against fresh K.
        cart_id = request.cartridge_id
        src_kv = self._fetch_source_kv(cart_id, check_layer)
        if src_kv is None:
            return
        cart_K_check = src_kv[0]  # (T, heads, dim) or similar

        fresh_K_check = fresh_K[check_layer]
        # Align shapes for diff. fresh is (1, heads, T, dim), cart may
        # be (T, heads, dim). Normalize.
        if cart_K_check.dim() == 3 and fresh_K_check.dim() == 4:
            fresh_K_check = fresh_K_check.squeeze(0).permute(1, 0, 2)
            # Now both (T, heads, dim)

        # Truncate to common T
        T = min(cart_K_check.shape[0], fresh_K_check.shape[0])
        cart_K_check = cart_K_check[:T]
        fresh_K_check = fresh_K_check[:T].to(cart_K_check.device)

        hkvd_indices = compute_hkvd_indices(
            cart_K_check.unsqueeze(0),
            fresh_K_check.unsqueeze(0),
            hkvd_ratio=hkvd_ratio,
        )

        # Build HKVD-subset slot mapping
        slot_mapping = request.slot_mapping
        if target_device is not None:
            slot_mapping = slot_mapping.to(target_device)
        else:
            slot_mapping = slot_mapping.cuda()
        hkvd_slot_mapping = slot_mapping[hkvd_indices]

        logger.info(
            "xa_refine: cartridge %s — refreshing %d/%d positions "
            "(HKVD ratio %.2f, check_layer=%d)",
            cart_id,
            len(hkvd_indices),
            T,
            hkvd_ratio,
            check_layer,
        )

        # Scatter fresh K/V at HKVD positions for every layer
        layer_idx = 0
        for layer_name in forward_context.no_compile_layers:
            layer = forward_context.no_compile_layers[layer_name]
            kv_cache_attr = getattr(layer, "kv_cache", None)
            if kv_cache_attr is None:
                continue
            parsed_idx = self._extract_layer_idx(layer_name)
            if parsed_idx is None or parsed_idx >= num_layers:
                continue

            if (len(kv_cache_attr) == 2 and hasattr(kv_cache_attr[0], "dim")
                    and kv_cache_attr[0].dim() == 4):
                kv_cache_layer = (kv_cache_attr[0], kv_cache_attr[1])
            else:
                kv_cache_layer = kv_cache_attr[0]

            fk = fresh_K[parsed_idx]
            fv = fresh_V[parsed_idx]
            # Gather HKVD subset
            # fresh shape: (1, heads, T, dim) → extract at hkvd indices
            if fk.dim() == 4:
                fk_sub = fk[:, :, hkvd_indices, :]
                fv_sub = fv[:, :, hkvd_indices, :]
                # Reshape to match inject_kv_into_paged_cache expected
                # src_key shape: (n_tokens, heads, dim) or (1, heads, n, dim)
                fk_sub = fk_sub.squeeze(0).permute(1, 0, 2)  # (n, heads, dim)
                fv_sub = fv_sub.squeeze(0).permute(1, 0, 2)
            else:
                fk_sub = fk[hkvd_indices]
                fv_sub = fv[hkvd_indices]

            inject_kv_into_paged_cache(
                src_key=fk_sub.to(device=hkvd_slot_mapping.device),
                src_value=fv_sub.to(device=hkvd_slot_mapping.device),
                kv_cache_layer=kv_cache_layer,
                slot_mapping=hkvd_slot_mapping,
            )
            layer_idx += 1

    def _fetch_source_kv(
        self,
        cartridge_id: str,
        layer_idx: int,
    ) -> "torch.Tensor | None":
        """Return the device-correct (K,V) stacked tensor for one
        layer.

        Delegates to the residency manager. No CPU->GPU copy on the
        hot path: promotion happened at acquire() time and every
        subsequent call is a cached GPU-side tensor lookup.
        """
        try:
            return self._residency.get_chunk(cartridge_id, layer_idx)
        except Exception as e:
            logger.error(
                "fetch_source_kv: %s layer %d missing: %s",
                cartridge_id,
                layer_idx,
                e,
            )
            return None

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
