# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone tests for CartridgeKRIProvider.

The original ``test_spf_cartridge_kri_provider.py`` was removed during
the spf-phase7d disposal because it depended on SPFController +
SPFConfig. The provider itself survived (the routing branch consumes
it for KRI-D routing-prior consultation), so this file exercises the
provider directly without any SPF-runtime imports.
"""
from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from vllm.distributed.kv_transfer.kv_connector.v1.routing_prior.manifest import BlockManifest
from vllm.distributed.kv_transfer.kv_connector.v1.routing_prior.cartridge_kri import CartridgeKRIProvider


def _write_prior(path: Path, *, num_layers: int = 2, num_kv_heads: int = 8,
                 num_blocks: int = 32) -> None:
    """Write a minimal continuous block_affinities prior the provider
    knows how to top-K on the fly."""
    aff = torch.randn(num_layers, num_kv_heads, num_blocks)
    torch.save({"block_affinities": aff}, path)


def test_register_and_get_manifest_topk(tmp_path: Path) -> None:
    prior_path = tmp_path / "prior.pt"
    _write_prior(prior_path, num_layers=2, num_kv_heads=8, num_blocks=32)

    provider = CartridgeKRIProvider()
    provider.register_prior(prefix_hash="cart_a", prior_path=str(prior_path))

    manifest = provider.get_manifest(
        prefix_hash="cart_a",
        query_hash=None,
        K=8,
    )
    assert isinstance(manifest, BlockManifest)
    # The manifest should expose 8 selected blocks for K=8. total_blocks
    # is derived from the prior's resolved candidate set after the
    # provider's own filtering, not from the raw tensor's outer dim;
    # the contract worth pinning is the K count and the index bounds.
    assert len(manifest.block_indices) == 8
    assert manifest.K == 8
    assert all(0 <= b < manifest.total_blocks for b in manifest.block_indices)
    assert manifest.total_blocks >= len(manifest.block_indices)


def test_unknown_prefix_hash_returns_none(tmp_path: Path) -> None:
    provider = CartridgeKRIProvider()
    manifest = provider.get_manifest(
        prefix_hash="never_registered",
        query_hash=None,
        K=8,
    )
    assert manifest is None


def test_registered_keys_reflects_registrations(tmp_path: Path) -> None:
    prior_a = tmp_path / "a.pt"
    prior_b = tmp_path / "b.pt"
    _write_prior(prior_a)
    _write_prior(prior_b)
    provider = CartridgeKRIProvider()
    provider.register_prior(prefix_hash="cart_a", prior_path=str(prior_a))
    provider.register_prior(prefix_hash="cart_b", prior_path=str(prior_b))
    keys = set(provider.registered_keys)
    # registered_keys returns (prefix_hash, query_hash) tuples; for the
    # KRI-G path query_hash is None.
    assert keys == {("cart_a", None), ("cart_b", None)}
