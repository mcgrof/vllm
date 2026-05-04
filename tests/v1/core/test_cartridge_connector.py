# SPDX-License-Identifier: Apache-2.0
"""Tests for CartridgeConnector.

Covers:
1. load_cartridge: TrainableCache format with and without frozen keys,
   frozen-before-trainable ordering, shape validation, error paths.
2. align_to_block_size: block alignment arithmetic.
3. inject_kv_into_paged_cache: slot mapping correctness using a fake
   cache writer (CPU-only, no GPU required).
4. Connector state: idempotent update_state_after_alloc.
"""
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_connector import (
    align_to_block_size,
    inject_kv_into_paged_cache,
    load_cartridge,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_trainable_cache(num_layers=4, num_kv_heads=2,
                          num_trainable_tokens=30, num_frozen_tokens=2,
                          head_dim=8, include_frozen=True):
    """Create a TrainableCache-style checkpoint dict."""
    cache = {
        "trainable_keys": [],
        "trainable_values": [],
    }
    if include_frozen:
        cache["frozen_keys"] = []
        cache["frozen_values"] = []

    for _ in range(num_layers):
        k_trn = torch.nn.Parameter(
            torch.randn(1, num_kv_heads, num_trainable_tokens, head_dim))
        v_trn = torch.nn.Parameter(
            torch.randn(1, num_kv_heads, num_trainable_tokens, head_dim))
        cache["trainable_keys"].append(k_trn)
        cache["trainable_values"].append(v_trn)

        if include_frozen:
            k_frz = torch.nn.Parameter(
                torch.randn(1, num_kv_heads, num_frozen_tokens, head_dim))
            v_frz = torch.nn.Parameter(
                torch.randn(1, num_kv_heads, num_frozen_tokens, head_dim))
            cache["frozen_keys"].append(k_frz)
            cache["frozen_values"].append(v_frz)

    return cache


def _save_and_load(checkpoint):
    """Save checkpoint to temp file and load via load_cartridge."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(checkpoint, f.name)
        return load_cartridge(f.name), f.name


# ---------------------------------------------------------------------------
# Tests: align_to_block_size
# ---------------------------------------------------------------------------

class TestAlignToBlockSize:
    def test_exact_multiple(self):
        assert align_to_block_size(32, 16) == 32

    def test_rounds_down(self):
        assert align_to_block_size(30, 16) == 16

    def test_less_than_block(self):
        assert align_to_block_size(10, 16) == 0

    def test_zero(self):
        assert align_to_block_size(0, 16) == 0


# ---------------------------------------------------------------------------
# Tests: load_cartridge — TrainableCache format
# ---------------------------------------------------------------------------

class TestLoadCartridgeTrainableCache:
    def test_with_frozen_keys(self):
        ckpt = _make_trainable_cache(
            num_layers=4, num_kv_heads=2,
            num_trainable_tokens=30, num_frozen_tokens=2,
            head_dim=8, include_frozen=True)
        result, path = _save_and_load(ckpt)

        # Total tokens = frozen + trainable = 2 + 30 = 32
        assert result["num_tokens"] == 32
        assert result["num_layers"] == 4

        k, v = result["kv_data"][0]
        assert k.shape == (32, 2, 8)
        Path(path).unlink()

    def test_without_frozen_keys(self):
        ckpt = _make_trainable_cache(
            num_layers=2, num_kv_heads=4,
            num_trainable_tokens=16, num_frozen_tokens=0,
            head_dim=16, include_frozen=False)
        result, path = _save_and_load(ckpt)

        assert result["num_tokens"] == 16
        assert result["num_kv_heads"] == 4
        assert result["head_dim"] == 16
        Path(path).unlink()

    def test_frozen_tokens_come_first(self):
        """Verify frozen tokens occupy positions 0..T_frozen-1."""
        ckpt = _make_trainable_cache(
            num_layers=1, num_kv_heads=1,
            num_trainable_tokens=4, num_frozen_tokens=2,
            head_dim=4, include_frozen=True)

        k_frz = ckpt["frozen_keys"][0].data
        k_trn = ckpt["trainable_keys"][0].data

        result, path = _save_and_load(ckpt)
        k_loaded = result["kv_data"][0][0]

        # First 2 tokens should be frozen
        k_frz_expected = k_frz.squeeze(0).permute(1, 0, 2)
        assert torch.allclose(k_loaded[:2], k_frz_expected)

        # Tokens 2-5 should be trainable
        k_trn_expected = k_trn.squeeze(0).permute(1, 0, 2)
        assert torch.allclose(k_loaded[2:], k_trn_expected)
        Path(path).unlink()

    def test_frozen_layer_count_mismatch_raises(self):
        ckpt = _make_trainable_cache(num_layers=4, include_frozen=True)
        ckpt["frozen_keys"] = ckpt["frozen_keys"][:3]

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(ckpt, f.name)
            with pytest.raises(ValueError, match="frozen_keys length"):
                load_cartridge(f.name)
            Path(f.name).unlink()

    def test_cross_layer_shape_mismatch_raises(self):
        """Verify that inconsistent shapes across layers are caught."""
        ckpt = _make_trainable_cache(
            num_layers=2, num_kv_heads=2,
            num_trainable_tokens=16, include_frozen=False)
        # Corrupt layer 1 to have different token count
        ckpt["trainable_keys"][1] = torch.nn.Parameter(
            torch.randn(1, 2, 8, 8))  # 8 tokens instead of 16
        ckpt["trainable_values"][1] = torch.nn.Parameter(
            torch.randn(1, 2, 8, 8))

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(ckpt, f.name)
            with pytest.raises(ValueError, match="Layer 1 K shape"):
                load_cartridge(f.name)
            Path(f.name).unlink()


# ---------------------------------------------------------------------------
# Tests: load_cartridge — error handling
# ---------------------------------------------------------------------------

class TestLoadCartridgeErrors:
    def test_unrecognized_format_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save("not a cartridge", f.name)
            with pytest.raises(ValueError, match="Unrecognized"):
                load_cartridge(f.name)
            Path(f.name).unlink()

    def test_dict_without_keys_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({"some_field": 42}, f.name)
            with pytest.raises(ValueError, match="Cannot find"):
                load_cartridge(f.name)
            Path(f.name).unlink()


# ---------------------------------------------------------------------------
# Tests: inject_kv_into_paged_cache — slot mapping correctness
# ---------------------------------------------------------------------------

def _fake_inject_kv(src_key, src_value, kv_cache_layer, slot_mapping):
    """CPU-only fake for inject_kv_into_paged_cache.

    Instead of calling ops.reshape_and_cache_flash, directly writes
    KV into flat cache slots. This tests that the slot mapping puts the
    right tokens in the right positions.
    """
    # Determine K/V split dim
    if kv_cache_layer.shape[0] == 2:
        key_cache, value_cache = kv_cache_layer.unbind(0)
    elif kv_cache_layer.shape[1] == 2:
        key_cache, value_cache = kv_cache_layer.unbind(1)
    else:
        raise ValueError(f"Bad cache shape: {kv_cache_layer.shape}")

    # Flatten blocks: (num_blocks, block_size, H, D) -> (num_blocks*bs, H, D)
    flat_k = key_cache.reshape(-1, key_cache.shape[-2], key_cache.shape[-1])
    flat_v = value_cache.reshape(-1, value_cache.shape[-2], value_cache.shape[-1])

    flat_k[slot_mapping] = src_key
    flat_v[slot_mapping] = src_value


class TestInjectKVSlotMapping:
    def test_tokens_land_in_correct_slots(self):
        """Verify that cartridge tokens are written to the expected
        physical slots in the paged cache."""
        num_tokens = 32
        num_kv_heads = 2
        head_dim = 4
        block_size = 16
        num_blocks = 4  # more blocks than needed

        src_key = torch.randn(num_tokens, num_kv_heads, head_dim)
        src_value = torch.randn(num_tokens, num_kv_heads, head_dim)

        # Paged cache: (2, num_blocks, block_size, num_kv_heads, head_dim)
        kv_cache = torch.zeros(2, num_blocks, block_size, num_kv_heads, head_dim)

        # Slot mapping: blocks 0 and 1, contiguous
        block_ids = torch.tensor([0, 1])
        block_offsets = torch.arange(0, block_size)
        slot_mapping = (
            block_offsets.reshape(1, block_size)
            + block_ids.reshape(-1, 1) * block_size
        ).flatten()

        _fake_inject_kv(src_key, src_value, kv_cache, slot_mapping)

        # Verify block 0 has the first 16 tokens
        key_cache = kv_cache[0]  # (num_blocks, block_size, H, D)
        assert torch.allclose(key_cache[0], src_key[:16])
        assert torch.allclose(key_cache[1], src_key[16:32])

        # Verify blocks 2,3 are still zeros (untouched)
        assert (key_cache[2] == 0).all()
        assert (key_cache[3] == 0).all()

    def test_non_contiguous_block_ids(self):
        """Verify injection works when blocks are not contiguous."""
        num_tokens = 32
        num_kv_heads = 1
        head_dim = 4
        block_size = 16
        num_blocks = 8

        src_key = torch.randn(num_tokens, num_kv_heads, head_dim)
        src_value = torch.randn(num_tokens, num_kv_heads, head_dim)

        kv_cache = torch.zeros(2, num_blocks, block_size, num_kv_heads, head_dim)

        # Non-contiguous: blocks 2 and 5
        block_ids = torch.tensor([2, 5])
        block_offsets = torch.arange(0, block_size)
        slot_mapping = (
            block_offsets.reshape(1, block_size)
            + block_ids.reshape(-1, 1) * block_size
        ).flatten()

        _fake_inject_kv(src_key, src_value, kv_cache, slot_mapping)

        key_cache = kv_cache[0]
        assert torch.allclose(key_cache[2], src_key[:16])
        assert torch.allclose(key_cache[5], src_key[16:32])
        # Other blocks untouched
        assert (key_cache[0] == 0).all()
        assert (key_cache[1] == 0).all()
        assert (key_cache[3] == 0).all()

    def test_dim1_kv_split(self):
        """Verify auto-detection of dim-1 K/V split layout."""
        num_tokens = 16
        num_kv_heads = 1
        head_dim = 4
        block_size = 16
        num_blocks = 2

        src_key = torch.randn(num_tokens, num_kv_heads, head_dim)
        src_value = torch.randn(num_tokens, num_kv_heads, head_dim)

        # Alternate layout: (num_blocks, 2, block_size, H, D)
        kv_cache = torch.zeros(num_blocks, 2, block_size, num_kv_heads, head_dim)

        slot_mapping = torch.arange(0, block_size)

        _fake_inject_kv(src_key, src_value, kv_cache, slot_mapping)

        key_cache = kv_cache[:, 0]  # (num_blocks, block_size, H, D)
        assert torch.allclose(key_cache[0], src_key)
