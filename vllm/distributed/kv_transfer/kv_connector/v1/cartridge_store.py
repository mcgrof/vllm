# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
CartridgeStore: read-only KV store backed by cartridge .pt files.

This module provides a simple storage layer for cartridge KV data.
It loads cartridge checkpoints from disk, splits them into per-layer
chunks, and serves them to the CartridgeConnector on demand.

In a full LMCache-integrated deployment, CartridgeStore would
implement LMCache's StoragePluginInterface and participate in the
multi-tier cache hierarchy. For now, it operates standalone as a
simple dict-backed store with explicit load/evict operations.

Design:
  - Read-only: cartridges are baked offline, never written at serve time.
  - Chunk-keyed: KV data is stored per (cartridge_id, layer_idx) for
    fine-grained memory management.
  - Registry-aware: uses CartridgeManifest to validate before loading.
  - Ref-counted: tracks in-flight requests to prevent eviction of
    active cartridges.

Future:
  - Implement LMCache StoragePluginInterface for multi-tier caching
    (GPU -> CPU -> disk) with automatic eviction and prefetch.
  - Add CacheGen compression for disk bandwidth optimization.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_connector import (
    load_cartridge,
    align_to_block_size,
)
from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_manifest import (
    CartridgeManifest,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class ChunkKey:
    """Key for a single chunk of cartridge KV data."""
    cartridge_id: str
    layer_idx: int

    def __hash__(self):
        return hash((self.cartridge_id, self.layer_idx))

    def __eq__(self, other):
        if not isinstance(other, ChunkKey):
            return False
        return (self.cartridge_id == other.cartridge_id
                and self.layer_idx == other.layer_idx)


@dataclass
class CartridgeResidency:
    """Tracks the residency state of a loaded cartridge."""
    cartridge_id: str
    manifest: CartridgeManifest
    num_layers: int
    num_tokens: int  # aligned token count
    ref_count: int = 0  # in-flight requests using this cartridge
    pinned: bool = False  # prevent eviction

    def acquire(self):
        """Increment ref count (request started using this cartridge)."""
        self.ref_count += 1

    def release(self):
        """Decrement ref count (request finished)."""
        self.ref_count = max(0, self.ref_count - 1)

    @property
    def evictable(self) -> bool:
        return not self.pinned and self.ref_count == 0


class CartridgeStore:
    """Read-only KV store for cartridge chunks.

    Loads cartridge .pt files, splits into per-layer chunks, and serves
    them by (cartridge_id, layer_idx) key. Thread-safe for concurrent
    access from multiple requests.

    Usage:
        store = CartridgeStore(block_size=16)
        store.load("patient_04", "/path/to/cartridge.pt", manifest)
        chunk = store.get(ChunkKey("patient_04", layer_idx=0))
        # chunk is (2, num_tokens, num_kv_heads, head_dim)
        store.acquire("patient_04")  # ref-count for in-flight request
        # ... serve request ...
        store.release("patient_04")
        store.evict("patient_04")  # free memory
    """

    def __init__(self, block_size: int = 16):
        self._block_size = block_size
        self._lock = threading.Lock()

        # chunk storage: ChunkKey -> stacked (2, T, H, D) tensor
        self._chunks: dict[ChunkKey, torch.Tensor] = {}

        # residency tracking: cartridge_id -> CartridgeResidency
        self._residency: dict[str, CartridgeResidency] = {}

    def load(
        self,
        cartridge_id: str,
        cartridge_path: str,
        manifest: CartridgeManifest,
        device: str = "cpu",
    ) -> int:
        """Load a cartridge from disk into the store.

        Splits the cartridge into per-layer chunks and stores them
        in memory. Returns the number of chunks loaded.

        Args:
            cartridge_id: unique identifier for this cartridge
            cartridge_path: path to the .pt checkpoint file
            manifest: manifest for validation
            device: where to store chunks ("cpu" for host RAM)

        Returns:
            Number of layer chunks loaded.

        Raises:
            ValueError: if cartridge_id is already loaded
        """
        with self._lock:
            if cartridge_id in self._residency:
                raise ValueError(
                    f"Cartridge {cartridge_id} is already loaded. "
                    f"Call evict() first."
                )

        cartridge = load_cartridge(cartridge_path)
        num_tokens = align_to_block_size(
            cartridge["num_tokens"], self._block_size
        )

        chunks_loaded = 0
        with self._lock:
            for layer_idx in range(cartridge["num_layers"]):
                k, v = cartridge["kv_data"][layer_idx]
                k = k[:num_tokens].to(device)
                v = v[:num_tokens].to(device)
                stacked = torch.stack([k, v], dim=0)  # (2, T, H, D)

                key = ChunkKey(cartridge_id, layer_idx)
                self._chunks[key] = stacked
                chunks_loaded += 1

            self._residency[cartridge_id] = CartridgeResidency(
                cartridge_id=cartridge_id,
                manifest=manifest,
                num_layers=cartridge["num_layers"],
                num_tokens=num_tokens,
            )

        logger.info(
            "Loaded cartridge %s: %d layers, %d tokens, device=%s",
            cartridge_id, cartridge["num_layers"], num_tokens, device,
        )
        return chunks_loaded

    def get(self, key: ChunkKey) -> Optional[torch.Tensor]:
        """Get a chunk by key. Returns None if not loaded.

        The returned tensor has shape (2, num_tokens, num_kv_heads, head_dim)
        where dim 0 is [K, V].
        """
        with self._lock:
            return self._chunks.get(key)

    def get_all_layers(self, cartridge_id: str) -> Optional[list[torch.Tensor]]:
        """Get all layer chunks for a cartridge, in layer order.

        Returns None if the cartridge is not loaded.
        """
        with self._lock:
            res = self._residency.get(cartridge_id)
            if res is None:
                return None
            return [
                self._chunks[ChunkKey(cartridge_id, li)]
                for li in range(res.num_layers)
            ]

    def contains(self, cartridge_id: str) -> bool:
        """Check if a cartridge is loaded."""
        with self._lock:
            return cartridge_id in self._residency

    def get_residency(self, cartridge_id: str) -> Optional[CartridgeResidency]:
        """Get residency info for a loaded cartridge."""
        with self._lock:
            return self._residency.get(cartridge_id)

    def acquire(self, cartridge_id: str) -> bool:
        """Increment ref count for an in-flight request.

        Returns False if the cartridge is not loaded.
        """
        with self._lock:
            res = self._residency.get(cartridge_id)
            if res is None:
                return False
            res.acquire()
            return True

    def release(self, cartridge_id: str) -> bool:
        """Decrement ref count when a request finishes.

        Returns False if the cartridge is not loaded.
        """
        with self._lock:
            res = self._residency.get(cartridge_id)
            if res is None:
                return False
            res.release()
            return True

    def pin(self, cartridge_id: str) -> bool:
        """Pin a cartridge to prevent eviction."""
        with self._lock:
            res = self._residency.get(cartridge_id)
            if res is None:
                return False
            res.pinned = True
            return True

    def unpin(self, cartridge_id: str) -> bool:
        """Unpin a cartridge to allow eviction."""
        with self._lock:
            res = self._residency.get(cartridge_id)
            if res is None:
                return False
            res.pinned = False
            return True

    def evict(self, cartridge_id: str, force: bool = False) -> bool:
        """Evict a cartridge from the store, freeing memory.

        Returns False if the cartridge is not loaded, is pinned, or
        has in-flight requests (unless force=True).
        """
        with self._lock:
            res = self._residency.get(cartridge_id)
            if res is None:
                return False
            if not force and not res.evictable:
                logger.warning(
                    "Cannot evict %s: ref_count=%d, pinned=%s",
                    cartridge_id, res.ref_count, res.pinned,
                )
                return False

            # Remove all chunks
            for li in range(res.num_layers):
                key = ChunkKey(cartridge_id, li)
                self._chunks.pop(key, None)

            del self._residency[cartridge_id]

        logger.info("Evicted cartridge %s", cartridge_id)
        return True

    def list_loaded(self) -> list[str]:
        """Return IDs of all loaded cartridges."""
        with self._lock:
            return list(self._residency.keys())

    def memory_usage_bytes(self) -> int:
        """Estimate total memory used by loaded chunks."""
        with self._lock:
            total = 0
            for chunk in self._chunks.values():
                total += chunk.nelement() * chunk.element_size()
            return total

    def close(self):
        """Evict all cartridges and release resources."""
        with self._lock:
            self._chunks.clear()
            self._residency.clear()
