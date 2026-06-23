# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV routing-prior manifest provider.

A cartridge ships with one or more pre-baked KV routing priors —
ranked block-selection signals computed offline and bound to the
cartridge's prefix hash. At serve time the CartridgeConnector
consults a provider to resolve ``(prefix_hash, query_hash) ->
BlockManifest``, then uses the manifest's ``block_indices`` to
sparse-inject just the load-bearing blocks instead of the full
cartridge.

Public API:
    BlockManifest         — selection result.
    BlockManifestProvider — Protocol for resolution.
    NullManifestProvider  — no-op default.
    CartridgeKRIProvider  — concrete provider that loads
                            per-cartridge ``.pt`` priors keyed on
                            the cartridge prefix hash.
"""
from __future__ import annotations

from vllm.distributed.kv_transfer.kv_connector.v1.routing_prior.cartridge_kri import (  # noqa: E501
    CartridgeKRIProvider,
)
from vllm.distributed.kv_transfer.kv_connector.v1.routing_prior.manifest import (  # noqa: E501
    BlockManifest,
    BlockManifestProvider,
    NullManifestProvider,
)

__all__ = [
    "BlockManifest",
    "BlockManifestProvider",
    "NullManifestProvider",
    "CartridgeKRIProvider",
]
