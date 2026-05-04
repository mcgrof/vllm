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
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_router import (
    CartridgeRouter,
    StaticCartridgeRouter,
    build_router_from_config,
)
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    pass  # KVCacheConfig not needed on this fork point
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
    # Auto-detect K/V split dimension.
    # Handles both stacked 5D tensors (K/V at dim 0 or dim 1) and
    # backends that pass separate (K, V) tensors as a tuple/list.
    if isinstance(kv_cache_layer, (tuple, list)) and len(kv_cache_layer) == 2:
        key_cache, value_cache = kv_cache_layer[0], kv_cache_layer[1]
    elif hasattr(kv_cache_layer, 'shape') and kv_cache_layer.shape[0] == 2:
        key_cache, value_cache = kv_cache_layer.unbind(0)
    elif hasattr(kv_cache_layer, 'shape') and kv_cache_layer.shape[1] == 2:
        key_cache, value_cache = kv_cache_layer.unbind(1)
    else:
        raise ValueError(
            f"Cannot determine K/V split for cache shape "
            f"{kv_cache_layer.shape if hasattr(kv_cache_layer, 'shape') else type(kv_cache_layer)}."
            " Expected a 5D tensor with K/V stacked at dim 0 or 1, "
            "or a (k_cache, v_cache) tuple."
        )

    # Align src tensors to slot_mapping length.  A cartridge may carry a
    # few extra tokens beyond the request's block-aligned num_tokens
    # (e.g. frozen BOS prepended during load).  Only the first
    # len(slot_mapping) tokens land in the paged cache.
    n_inject = slot_mapping.shape[0]
    if src_key.shape[0] > n_inject:
        src_key = src_key[:n_inject]
        src_value = src_value[:n_inject]

    k_scale = torch.tensor(1.0, dtype=torch.float32,
                            device=key_cache.device)
    v_scale = torch.tensor(1.0, dtype=torch.float32,
                            device=value_cache.device)

    ops.reshape_and_cache_flash(
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
    """
    cartridge_id: str
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
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
        )
        # The base class on this fork point does not store
        # _kv_transfer_config; set it ourselves.
        self._kv_transfer_config = vllm_config.kv_transfer_config
        self._block_size = vllm_config.cache_config.block_size
        self._tp_size = vllm_config.parallel_config.tensor_parallel_size
        self._tp_rank = 0

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

        self._vllm_config = vllm_config
        self._load_cartridges_from_config()
        self._router: CartridgeRouter = self._build_router_from_config()

        logger.info(
            "CartridgeConnector ready: %d cartridge(s) loaded, "
            "router=%s",
            len(self._cartridge_meta),
            type(self._router).__name__,
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
            CartridgeStore,
        )

        singleton_path = self._kv_transfer_config.get_from_extra_config(
            "cartridge_path", None
        )
        cartridges_list = self._kv_transfer_config.get_from_extra_config(
            "cartridges", None
        )
        if singleton_path is None and not cartridges_list:
            raise ValueError(
                "CartridgeConnector requires either 'cartridge_path' "
                "(singleton mode) or 'cartridges' (multi-cartridge "
                "mode) in --kv-connector-extra-config"
            )

        self._store = CartridgeStore(block_size=self._block_size)
        # cartridge_id -> {"num_tokens": int, "num_blocks": int,
        #                  "num_layers": int}
        self._cartridge_meta: dict[str, dict] = {}
        # For singleton/default fallback when the router returns None
        # but we need a cartridge_id anyway (legacy behaviour).
        self._default_cartridge_id: Optional[str] = None

        if singleton_path is not None:
            manifest_path = (
                self._kv_transfer_config.get_from_extra_config(
                    "manifest_path", None
                )
            )
            cart_id = self._load_one(
                cartridge_path=singleton_path,
                manifest_path=manifest_path,
                explicit_cartridge_id=None,
            )
            self._default_cartridge_id = cart_id

        if cartridges_list:
            if not isinstance(cartridges_list, list):
                raise ValueError(
                    "'cartridges' must be a list of entries")
            for entry in cartridges_list:
                if not isinstance(entry, dict):
                    raise ValueError(
                        "each 'cartridges' entry must be a dict")
                path = entry.get("path")
                if not path:
                    raise ValueError(
                        "each 'cartridges' entry requires 'path'")
                self._load_one(
                    cartridge_path=path,
                    manifest_path=entry.get("manifest_path"),
                    explicit_cartridge_id=entry.get("cartridge_id"),
                )
            # First listed cartridge is the default fallback if no
            # singleton was provided.
            if self._default_cartridge_id is None:
                first_entry = cartridges_list[0]
                self._default_cartridge_id = (
                    first_entry.get("cartridge_id")
                    or next(iter(self._cartridge_meta))
                )

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
            CartridgeManifest,
        )

        manifest: Optional[CartridgeManifest] = None
        if manifest_path is not None:
            manifest = CartridgeManifest.from_json(manifest_path)
            model_cfg = self._vllm_config.model_config.hf_config
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
                model_id=self._vllm_config.model_config.model,
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
            )
            errors.extend(
                manifest.validate_against_block_size(self._block_size)
            )
            if errors:
                raise ValueError(
                    "Cartridge manifest validation failed:\n"
                    + "\n".join(f"  - {e}" for e in errors)
                )
            logger.info("Cartridge manifest validated: %s",
                        manifest.cartridge_id)
            if (explicit_cartridge_id is not None
                    and explicit_cartridge_id != manifest.cartridge_id):
                raise ValueError(
                    f"cartridge_id mismatch: config says "
                    f"{explicit_cartridge_id!r} but manifest says "
                    f"{manifest.cartridge_id!r}"
                )

        if manifest is None:
            cartridge_data = load_cartridge(cartridge_path)
            aligned = align_to_block_size(
                cartridge_data["num_tokens"], self._block_size)
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
            raise ValueError(
                f"duplicate cartridge_id: {cart_id!r}")

        self._store.load(cart_id, cartridge_path, manifest, device="cpu")
        self._store.pin(cart_id)  # prevent accidental eviction

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
            cart_id, residency.num_layers,
            num_tokens, num_blocks, self._block_size,
        )
        return cart_id

    def _build_router_from_config(self) -> CartridgeRouter:
        """Build the CartridgeRouter from extra_config.

        If no router config is provided and exactly one cartridge is
        loaded, defaults to StaticCartridgeRouter bound to that id
        (singleton semantics). Otherwise raises.
        """
        router_cfg = self._kv_transfer_config.get_from_extra_config(
            "router", None
        )
        if router_cfg is None:
            if len(self._cartridge_meta) == 1:
                only_id = next(iter(self._cartridge_meta))
                return StaticCartridgeRouter(only_id)
            raise ValueError(
                "Multi-cartridge mode requires 'router' in "
                "--kv-connector-extra-config when more than one "
                "cartridge is loaded"
            )
        if not isinstance(router_cfg, dict):
            raise ValueError(
                "'router' must be a dict describing the router "
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
        return self._default_cartridge_id or next(
            iter(self._cartridge_meta))

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
                request.request_id, cart_id,
            )
            return 0, False

        cart_info = self._cartridge_meta[cart_id]
        # Cap to prompt length so we never claim more positions than
        # the request has. Per-cartridge token count — different
        # cartridges can have different sizes.
        matched = align_to_block_size(
            min(cart_info["num_tokens"], max_claim),
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
            request.request_id, cart_id,
            matched, num_new, num_computed_tokens,
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
        """
        if num_external_tokens > 0:
            # Sanity check: get_num_new_matched_tokens should have
            # populated our state for this request.
            if (request.request_id
                    not in self._request_cartridge_ids):
                logger.warning(
                    "update_state_after_alloc: no cartridge_id "
                    "recorded for request %s; skipping",
                    request.request_id,
                )
                return
            # Mark as committed for this tick's build_connector_meta.
            self._requests_need_load.add(request.request_id)

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
                    "for request %s; skipping", new_req.req_id,
                )
                continue

            num_blocks = num_tokens // self._block_size
            block_ids = new_req.block_ids[0]
            cartridge_block_ids = block_ids[:num_blocks]
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
                cartridge_id=cart_id,
                slot_mapping=slot_mapping,
                num_tokens=num_tokens,
            ))

        # Clear tick-local state for consumed AND any stale resolved
        # requests that never got committed (e.g. deferred by the
        # scheduler; they'll be re-resolved next tick).
        self._requests_need_load.clear()
        # Only clear state for requests scheduled this tick — requests
        # that were resolved but not yet scheduled may still be pending.
        # In practice resolved+not-scheduled cleans up on reschedule.
        scheduled_ids = {r.req_id
                         for r in scheduler_output.scheduled_new_reqs}
        for req_id in list(self._request_cartridge_ids.keys()):
            if req_id in scheduled_ids:
                self._request_cartridge_ids.pop(req_id, None)
                self._request_num_tokens.pop(req_id, None)

        return meta

    # ==============================
    # Worker-side methods
    # ==============================

    def start_load_kv(
        self, forward_context: "ForwardContext", **kwargs: Any
    ) -> None:
        """Inject cartridge KV into allocated GPU blocks, dispatching
        per request to the cartridge chosen by the scheduler-side
        router.

        Different requests in the same batch may have different
        ``cartridge_id`` values; each is loaded from the store and
        written into that request's slot mapping independently. No
        shared state across requests other than the store and the
        paged KV cache.
        """
        from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_store import (
            ChunkKey,
        )

        metadata = self._get_connector_metadata()
        assert isinstance(metadata, CartridgeConnectorMetadata)

        if not metadata.requests:
            return

        # Acquire a ref per unique cartridge used in this batch so
        # that none of them can be evicted mid-injection.
        unique_ids = {r.cartridge_id for r in metadata.requests}
        for cart_id in unique_ids:
            self._store.acquire(cart_id)

        try:
            for request in metadata.requests:
                cart_id = request.cartridge_id
                num_layers = self._cartridge_meta.get(
                    cart_id, {}).get("num_layers")
                if num_layers is None:
                    logger.error(
                        "start_load_kv: unknown cartridge_id %r; "
                        "skipping request", cart_id,
                    )
                    continue

                slot_mapping = request.slot_mapping.cuda()
                logger.info(
                    "Injecting cartridge %s KV (%d tokens) into "
                    "GPU cache", cart_id, request.num_tokens,
                )

                for layer_name in forward_context.no_compile_layers:
                    layer = forward_context.no_compile_layers[layer_name]
                    kv_cache_attr = getattr(layer, "kv_cache", None)
                    if kv_cache_attr is None:
                        continue

                    kv_cache_layer = kv_cache_attr[0]

                    layer_idx = self._extract_layer_idx(layer_name)
                    if layer_idx is None or layer_idx >= num_layers:
                        continue

                    chunk_key = ChunkKey(cart_id, layer_idx)
                    src_kv = self._store.get(chunk_key)
                    if src_kv is None:
                        logger.error(
                            "Cartridge chunk missing: %s layer %d",
                            cart_id, layer_idx,
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
            for cart_id in unique_ids:
                self._store.release(cart_id)

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
