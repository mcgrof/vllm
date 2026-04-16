# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
LMCache StoragePluginInterface wrapper for CartridgeStore.

This module bridges CartridgeStore (our read-only cartridge KV store)
with LMCache's multi-tier cache hierarchy. When configured as an
LMCache storage plugin, cartridge chunks participate in LMCache's
lookup/prefetch/evict lifecycle alongside regular KV cache entries.

Configuration (in LMCache YAML):
    chunk_size: 256
    storage_plugins: cartridge_store
    extra_config:
      storage_plugin.cartridge_store.module_path:
        vllm.distributed.kv_transfer.kv_connector.v1.cartridge_lmcache_plugin
      storage_plugin.cartridge_store.class_name: CartridgeLMCachePlugin
      storage_plugin.cartridge_store.registry_db: /path/to/cartridges.db
      storage_plugin.cartridge_store.cartridge_dir: /path/to/cartridge/files/

This is a thin adapter layer. CartridgeStore owns the cartridge-specific
logic (loading .pt files, splitting into layers, manifest validation,
ref counting). This plugin translates between LMCache's CacheEngineKey
interface and CartridgeStore's ChunkKey interface.

Cartridges are READ-ONLY: put operations are no-ops. The plugin only
serves data that was baked offline.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence, Union

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_store import (
    CartridgeStore,
    ChunkKey,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

# Try to import LMCache types. If LMCache is not installed,
# this module is not usable but should still be importable for
# testing with mocks.
try:
    from lmcache.utils import CacheEngineKey
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.memory_management import MemoryFormat, MemoryObj, MemoryObjMetadata
    from lmcache.v1.metadata import LMCacheMetadata
    from lmcache.v1.storage_backend.abstract_backend import StoragePluginInterface

    if TYPE_CHECKING:
        from lmcache.v1.storage_backend import LocalCPUBackend

    _LMCACHE_AVAILABLE = True
except ImportError:
    _LMCACHE_AVAILABLE = False
    # Define a stub base class so the module can be imported
    class StoragePluginInterface:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass


def _key_to_chunk(key: "CacheEngineKey") -> Optional[ChunkKey]:
    """Extract cartridge_id and layer_idx from a CacheEngineKey.

    Convention: the CacheEngineKey's tags tuple contains:
      tags[0] = cartridge_id
      tags[1] = str(layer_idx)

    Returns None if the key doesn't follow this convention.
    """
    if not hasattr(key, "tags") or key.tags is None or len(key.tags) < 2:
        return None
    try:
        cartridge_id = str(key.tags[0])
        layer_idx = int(key.tags[1])
        return ChunkKey(cartridge_id, layer_idx)
    except (ValueError, TypeError):
        return None


class CartridgeLMCachePlugin(StoragePluginInterface):
    """LMCache storage plugin backed by CartridgeStore.

    Read-only: put operations are no-ops. Only serves cartridge KV
    data that was baked offline and loaded via CartridgeStore.load().
    """

    def __init__(
        self,
        dst_device: str = "cuda",
        config: Optional[Any] = None,
        metadata: Optional[Any] = None,
        local_cpu_backend: Optional[Any] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        if _LMCACHE_AVAILABLE:
            super().__init__(
                dst_device=dst_device,
                config=config,
                metadata=metadata,
                local_cpu_backend=local_cpu_backend,
                loop=loop,
            )
        else:
            self.dst_device = dst_device
            self.config = config
            self.metadata = metadata
            self.local_cpu_backend = local_cpu_backend
            self.loop = loop

        # The store is injected externally or created from config
        self._store: Optional[CartridgeStore] = None

        # Extract config from extra_config if available
        if config is not None and hasattr(config, "extra_config"):
            block_size = getattr(config, "chunk_size", 256)
            self._store = CartridgeStore(block_size=block_size)
            logger.info(
                "CartridgeLMCachePlugin initialized with chunk_size=%d",
                block_size,
            )

    def set_store(self, store: CartridgeStore) -> None:
        """Inject an externally-created CartridgeStore.

        This allows the CartridgeConnector to share its store with
        this plugin, avoiding duplicate loading.
        """
        self._store = store

    # ==============================
    # Read operations
    # ==============================

    def contains(self, key: "CacheEngineKey", pin: bool = False) -> bool:
        if self._store is None:
            return False
        chunk = _key_to_chunk(key)
        if chunk is None:
            return False
        result = self._store.get(chunk) is not None
        if result and pin:
            self._store.pin(chunk.cartridge_id)
        return result

    def exists_in_put_tasks(self, key: "CacheEngineKey") -> bool:
        # Read-only: never has pending put tasks
        return False

    def get_blocking(self, key: "CacheEngineKey") -> Optional[Any]:
        if self._store is None:
            return None
        chunk = _key_to_chunk(key)
        if chunk is None:
            return None
        tensor = self._store.get(chunk)
        if tensor is None:
            return None

        # If LMCache is available, wrap in a MemoryObj
        if _LMCACHE_AVAILABLE:
            meta = MemoryObjMetadata(
                shape=tensor.shape,
                dtype=tensor.dtype,
                address=tensor.data_ptr(),
                phy_size=tensor.nelement() * tensor.element_size(),
                ref_count=1,
                fmt=MemoryFormat.KV_T2D,
            )
            # Create a simple MemoryObj wrapper
            # LMCache expects specific MemoryObj subtypes; for now
            # return the raw tensor. The connector's inject path
            # handles the tensor directly anyway.
            return tensor

        return tensor

    # ==============================
    # Write operations (all no-ops for read-only store)
    # ==============================

    def batched_submit_put_task(
        self,
        keys: Sequence[Any],
        objs: List[Any],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable] = None,
    ) -> Union[List[Future], None]:
        # Read-only: cartridges are never written at serve time
        return None

    async def async_batched_submit_put_task(
        self,
        keys: Sequence[Any],
        objs: List[Any],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable] = None,
    ) -> None:
        # Read-only
        pass

    # ==============================
    # Pin / unpin / remove
    # ==============================

    def pin(self, key: "CacheEngineKey") -> bool:
        if self._store is None:
            return False
        chunk = _key_to_chunk(key)
        if chunk is None:
            return False
        return self._store.pin(chunk.cartridge_id)

    def unpin(self, key: "CacheEngineKey") -> bool:
        if self._store is None:
            return False
        chunk = _key_to_chunk(key)
        if chunk is None:
            return False
        return self._store.unpin(chunk.cartridge_id)

    def remove(self, key: "CacheEngineKey", force: bool = True) -> bool:
        if self._store is None:
            return False
        chunk = _key_to_chunk(key)
        if chunk is None:
            return False
        # Evict the entire cartridge (not just one chunk)
        return self._store.evict(chunk.cartridge_id, force=force)

    # ==============================
    # Allocator (delegate to CPU backend if available)
    # ==============================

    def get_allocator_backend(self) -> Any:
        if self.local_cpu_backend is not None:
            return self.local_cpu_backend
        return self  # fallback

    # ==============================
    # Cache policy
    # ==============================

    def touch_cache(self) -> None:
        # No eviction policy for read-only store
        pass

    # ==============================
    # Lifecycle
    # ==============================

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None
