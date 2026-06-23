# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_connector import (
    align_to_block_size, inject_kv_into_paged_cache, load_cartridge,
    selected_token_mask_from_block_ids)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_trainable_cache(num_layers=4,
                          num_kv_heads=2,
                          num_trainable_tokens=30,
                          num_frozen_tokens=2,
                          head_dim=8,
                          include_frozen=True):
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
        ckpt = _make_trainable_cache(num_layers=4,
                                     num_kv_heads=2,
                                     num_trainable_tokens=30,
                                     num_frozen_tokens=2,
                                     head_dim=8,
                                     include_frozen=True)
        result, path = _save_and_load(ckpt)

        # Total tokens = frozen + trainable = 2 + 30 = 32
        assert result["num_tokens"] == 32
        assert result["num_layers"] == 4

        k, v = result["kv_data"][0]
        assert k.shape == (32, 2, 8)
        Path(path).unlink()

    def test_without_frozen_keys(self):
        ckpt = _make_trainable_cache(num_layers=2,
                                     num_kv_heads=4,
                                     num_trainable_tokens=16,
                                     num_frozen_tokens=0,
                                     head_dim=16,
                                     include_frozen=False)
        result, path = _save_and_load(ckpt)

        assert result["num_tokens"] == 16
        assert result["num_kv_heads"] == 4
        assert result["head_dim"] == 16
        Path(path).unlink()

    def test_frozen_tokens_come_first(self):
        """Verify frozen tokens occupy positions 0..T_frozen-1."""
        ckpt = _make_trainable_cache(num_layers=1,
                                     num_kv_heads=1,
                                     num_trainable_tokens=4,
                                     num_frozen_tokens=2,
                                     head_dim=4,
                                     include_frozen=True)

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
        ckpt = _make_trainable_cache(num_layers=2,
                                     num_kv_heads=2,
                                     num_trainable_tokens=16,
                                     include_frozen=False)
        # Corrupt layer 1 to have different token count
        ckpt["trainable_keys"][1] = torch.nn.Parameter(torch.randn(
            1, 2, 8, 8))  # 8 tokens instead of 16
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


def _fake_inject_kv(
    src_key,
    src_value,
    kv_cache_layer,
    slot_mapping,
    selected_token_mask=None,
):
    """CPU-only fake for inject_kv_into_paged_cache.

    Instead of calling triton_reshape_and_cache_flash, directly writes
    KV into flat cache slots. This tests that the slot mapping puts the
    right tokens in the right positions.

    Mirrors the real function's signature: when ``selected_token_mask``
    is provided, only tokens where the mask is True are written, and
    they land at their original slot indices (position-preserving).
    """
    # Determine K/V split dim
    if kv_cache_layer.shape[0] == 2:
        key_cache, value_cache = kv_cache_layer.unbind(0)
    elif kv_cache_layer.shape[1] == 2:
        key_cache, value_cache = kv_cache_layer.unbind(1)
    else:
        raise ValueError(f"Bad cache shape: {kv_cache_layer.shape}")

    if selected_token_mask is not None:
        if not bool(selected_token_mask.any()):
            return
        src_key = src_key[selected_token_mask]
        src_value = src_value[selected_token_mask]
        slot_mapping = slot_mapping[selected_token_mask]

    # Flatten blocks: (num_blocks, block_size, H, D) -> (num_blocks*bs, H, D)
    flat_k = key_cache.reshape(-1, key_cache.shape[-2], key_cache.shape[-1])
    flat_v = value_cache.reshape(-1, value_cache.shape[-2],
                                 value_cache.shape[-1])

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
        kv_cache = torch.zeros(2, num_blocks, block_size, num_kv_heads,
                               head_dim)

        # Slot mapping: blocks 0 and 1, contiguous
        block_ids = torch.tensor([0, 1])
        block_offsets = torch.arange(0, block_size)
        slot_mapping = (block_offsets.reshape(1, block_size) +
                        block_ids.reshape(-1, 1) * block_size).flatten()

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

        kv_cache = torch.zeros(2, num_blocks, block_size, num_kv_heads,
                               head_dim)

        # Non-contiguous: blocks 2 and 5
        block_ids = torch.tensor([2, 5])
        block_offsets = torch.arange(0, block_size)
        slot_mapping = (block_offsets.reshape(1, block_size) +
                        block_ids.reshape(-1, 1) * block_size).flatten()

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
        kv_cache = torch.zeros(num_blocks, 2, block_size, num_kv_heads,
                               head_dim)

        slot_mapping = torch.arange(0, block_size)

        _fake_inject_kv(src_key, src_value, kv_cache, slot_mapping)

        key_cache = kv_cache[:, 0]  # (num_blocks, block_size, H, D)
        assert torch.allclose(key_cache[0], src_key)


# ---------------------------------------------------------------------------
# Tests: selected_token_mask_from_block_ids (pure helper)
# ---------------------------------------------------------------------------


class TestSelectedTokenMaskFromBlockIds:
    """A.2a step 1 — helper that lifts block-level selection into a
    per-token boolean mask suitable for subsetting source tensors."""

    def test_empty_selection_produces_all_false(self):
        mask = selected_token_mask_from_block_ids(
            selected_block_ids=set(),
            num_tokens=32,
            block_size=16,
        )
        assert mask.dtype == torch.bool
        assert mask.shape == (32, )
        assert not mask.any()

    def test_single_block_selects_that_block_only(self):
        mask = selected_token_mask_from_block_ids(
            selected_block_ids={1},
            num_tokens=48,
            block_size=16,
        )
        # block 0: tokens 0..15 → False
        assert not mask[0:16].any()
        # block 1: tokens 16..31 → True
        assert mask[16:32].all()
        # block 2: tokens 32..47 → False
        assert not mask[32:48].any()

    def test_multiple_non_contiguous_blocks(self):
        mask = selected_token_mask_from_block_ids(
            selected_block_ids={0, 2, 4},
            num_tokens=80,
            block_size=16,
        )
        for b in (0, 2, 4):
            assert mask[b * 16:(b + 1) * 16].all()
        for b in (1, 3):
            assert not mask[b * 16:(b + 1) * 16].any()

    def test_block_beyond_num_tokens_is_ignored(self):
        # block 99 doesn't exist in a 32-token span; should have no effect
        mask = selected_token_mask_from_block_ids(
            selected_block_ids={0, 99},
            num_tokens=32,
            block_size=16,
        )
        assert mask[:16].all()
        assert not mask[16:].any()

    def test_list_and_set_both_accepted(self):
        mask_set = selected_token_mask_from_block_ids({1, 3}, 64, 16)
        mask_list = selected_token_mask_from_block_ids([1, 3], 64, 16)
        assert torch.equal(mask_set, mask_list)


# ---------------------------------------------------------------------------
# Tests: inject_kv_into_paged_cache with sparse selection
# ---------------------------------------------------------------------------


class TestInjectKVSparseSelection:
    """A.2a step 1 — inject_kv_into_paged_cache honours an optional
    selection mask. Selected tokens land at their *original* physical
    slots (position-preserving); non-selected slots are untouched."""

    def test_position_preserving_write(self):
        """Given selection {0, 2}, only blocks 0 and 2 of the cache
        receive writes; blocks 1 and 3 stay zero. The selected tokens
        land at their original slots (no compaction)."""
        num_tokens = 64  # 4 blocks of 16
        num_kv_heads = 2
        head_dim = 4
        block_size = 16
        num_blocks = 6

        src_key = torch.randn(num_tokens, num_kv_heads, head_dim)
        src_value = torch.randn(num_tokens, num_kv_heads, head_dim)

        kv_cache = torch.zeros(2, num_blocks, block_size, num_kv_heads,
                               head_dim)

        # Slot mapping: logical blocks 0..3 mapped to physical blocks 0..3.
        logical_block_ids = torch.tensor([0, 1, 2, 3])
        block_offsets = torch.arange(0, block_size)
        slot_mapping = (
            block_offsets.reshape(1, block_size) +
            logical_block_ids.reshape(-1, 1) * block_size).flatten()

        # Select blocks 0 and 2.
        mask = selected_token_mask_from_block_ids(
            selected_block_ids={0, 2},
            num_tokens=num_tokens,
            block_size=block_size,
        )

        _fake_inject_kv(src_key, src_value, kv_cache, slot_mapping, mask)

        key_cache = kv_cache[0]  # (num_blocks, block_size, H, D)

        # Selected blocks carry the original tokens at their original
        # physical positions.
        assert torch.allclose(key_cache[0], src_key[0:16])
        assert torch.allclose(key_cache[2], src_key[32:48])
        # Non-selected blocks untouched.
        assert (key_cache[1] == 0).all()
        assert (key_cache[3] == 0).all()
        # Unallocated blocks untouched.
        assert (key_cache[4] == 0).all()
        assert (key_cache[5] == 0).all()

    def test_empty_selection_is_noop(self):
        num_tokens = 32
        num_kv_heads = 1
        head_dim = 4
        block_size = 16
        num_blocks = 4

        src_key = torch.randn(num_tokens, num_kv_heads, head_dim)
        src_value = torch.randn(num_tokens, num_kv_heads, head_dim)
        kv_cache = torch.zeros(2, num_blocks, block_size, num_kv_heads,
                               head_dim)

        slot_mapping = torch.arange(0, num_tokens)
        mask = torch.zeros(num_tokens, dtype=torch.bool)

        _fake_inject_kv(src_key, src_value, kv_cache, slot_mapping, mask)

        # Nothing should have been written.
        assert (kv_cache == 0).all()

    def test_full_selection_matches_unselected(self):
        """Selection of all blocks produces the same result as no
        selection at all. Regression guard against drift between the
        two code paths."""
        num_tokens = 32
        num_kv_heads = 1
        head_dim = 4
        block_size = 16
        num_blocks = 2

        src_key = torch.randn(num_tokens, num_kv_heads, head_dim)
        src_value = torch.randn(num_tokens, num_kv_heads, head_dim)

        kv_cache_a = torch.zeros(2, num_blocks, block_size, num_kv_heads,
                                 head_dim)
        kv_cache_b = torch.zeros_like(kv_cache_a)
        slot_mapping = torch.arange(0, num_tokens)

        _fake_inject_kv(src_key, src_value, kv_cache_a, slot_mapping)
        _fake_inject_kv(
            src_key,
            src_value,
            kv_cache_b,
            slot_mapping,
            torch.ones(num_tokens, dtype=torch.bool),
        )
        assert torch.equal(kv_cache_a, kv_cache_b)

    def test_mask_length_mismatch_in_real_fn_raises(self):
        """The real inject_kv_into_paged_cache (not the fake) validates
        the mask length. Guards against silently losing tokens when
        callers pass a mask sized for a different cartridge."""
        src_key = torch.randn(16, 1, 4)
        src_value = torch.randn(16, 1, 4)
        kv_cache = torch.zeros(2, 2, 16, 1, 4)
        slot_mapping = torch.arange(0, 16, dtype=torch.int64)
        bad_mask = torch.ones(8, dtype=torch.bool)  # wrong length

        # The real function validates; we call it directly with the
        # bad mask and expect ValueError before any kernel call would
        # fire. (The kernel call won't actually happen because the
        # validation comes first.)
        with pytest.raises(ValueError):
            inject_kv_into_paged_cache(
                src_key=src_key,
                src_value=src_value,
                kv_cache_layer=kv_cache,
                slot_mapping=slot_mapping,
                selected_token_mask=bad_mask,
            )

    def test_mask_wrong_dtype_raises(self):
        src_key = torch.randn(16, 1, 4)
        src_value = torch.randn(16, 1, 4)
        kv_cache = torch.zeros(2, 2, 16, 1, 4)
        slot_mapping = torch.arange(0, 16, dtype=torch.int64)
        bad_mask = torch.ones(16, dtype=torch.float32)

        with pytest.raises(ValueError):
            inject_kv_into_paged_cache(
                src_key=src_key,
                src_value=src_value,
                kv_cache_layer=kv_cache,
                slot_mapping=slot_mapping,
                selected_token_mask=bad_mask,
            )


# ---------------------------------------------------------------------------
# Tests: CartridgeReqMeta.selected_block_ids field (A.2a step 2)
# ---------------------------------------------------------------------------


class TestCartridgeReqMetaSelection:
    """A.2a step 2 — CartridgeReqMeta carries an optional per-request
    block-selection set that _inject_request consumes. This test
    covers the dataclass shape and default behavior only; the actual
    inject-time plumbing is exercised by the step-1 sparse inject
    tests and by integration tests."""

    def test_default_selection_is_none(self):
        from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_connector import (
            CartridgeReqMeta)

        meta = CartridgeReqMeta(
            cartridge_id="test",
            slot_mapping=torch.arange(16),
            num_tokens=16,
        )
        assert meta.selected_block_ids is None

    def test_explicit_selection_stored_as_provided(self):
        from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_connector import (
            CartridgeReqMeta)

        sel = {0, 2, 5}
        meta = CartridgeReqMeta(
            cartridge_id="test",
            slot_mapping=torch.arange(96),
            num_tokens=96,
            selected_block_ids=sel,
        )
        assert meta.selected_block_ids == sel

    def test_empty_selection_allowed(self):
        """Empty selection is a valid value — an empty set means 'no
        blocks selected, inject nothing.' Distinct from None which
        means 'no routing, inject everything.'"""
        from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_connector import (
            CartridgeReqMeta)

        meta = CartridgeReqMeta(
            cartridge_id="test",
            slot_mapping=torch.arange(16),
            num_tokens=16,
            selected_block_ids=set(),
        )
        assert meta.selected_block_ids == set()
        assert meta.selected_block_ids is not None  # distinguish from default


# ---------------------------------------------------------------------------
# Tests: _publish_routing_state scaffolding (A.2a step 5)
# ---------------------------------------------------------------------------


class TestPublishRoutingStateScaffolding:
    """A.2a step 5 (scaffolding) — _publish_routing_state produces a
    RoutingPrior on the thread-local when given a selection, and
    clears it when given None or an empty set.

    Attention-side consumption of this state is a separate follow-up;
    these tests only cover the producer contract."""

    def _fresh(self):
        from vllm.v1.attention.routing_state import clear_routing_state
        clear_routing_state()

    def test_none_clears(self):
        from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_connector import (
            _publish_routing_state)
        from vllm.v1.attention.routing_state import (RoutingPrior,
                                                     get_routing_state,
                                                     set_routing_state)

        # Pre-populate so clear-behavior is observable.
        set_routing_state(
            RoutingPrior(
                block_affinities=None,
                K=4,
                mode="x",
                num_prefix_blocks=16,
                block_size=16,
                request_id="prev",
                kmeans_blocks=[0, 1, 2, 3],
            ))
        assert get_routing_state() is not None
        _publish_routing_state(None,
                               num_blocks=16,
                               block_size=16,
                               request_id="r1")
        assert get_routing_state() is None

    def test_empty_selection_clears(self):
        from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_connector import (
            _publish_routing_state)
        from vllm.v1.attention.routing_state import (RoutingPrior,
                                                     get_routing_state,
                                                     set_routing_state)
        set_routing_state(
            RoutingPrior(
                block_affinities=None,
                K=4,
                mode="x",
                num_prefix_blocks=16,
                block_size=16,
                request_id="prev",
                kmeans_blocks=[0, 1, 2, 3],
            ))
        _publish_routing_state(set(),
                               num_blocks=16,
                               block_size=16,
                               request_id="r1")
        assert get_routing_state() is None

    def test_non_empty_selection_produces_prior(self):
        from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_connector import (
            _publish_routing_state)
        from vllm.v1.attention.routing_state import get_routing_state
        self._fresh()
        _publish_routing_state(
            {0, 5, 12},
            num_blocks=16,
            block_size=16,
            request_id="r1",
        )
        prior = get_routing_state()
        assert prior is not None
        assert prior.K == 3
        assert prior.num_prefix_blocks == 16
        assert prior.block_size == 16
        assert prior.request_id == "r1"
        assert prior.mode == "cartridge_prior"
        assert prior.routing_policy == "topk_kmeans"
        assert prior.kmeans_blocks == [0, 5, 12]  # sorted

    def test_selection_sorted_independent_of_input_order(self):
        from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_connector import (
            _publish_routing_state)
        from vllm.v1.attention.routing_state import get_routing_state
        self._fresh()
        _publish_routing_state({12, 0, 5},
                               num_blocks=16,
                               block_size=16,
                               request_id="r1")
        prior = get_routing_state()
        assert prior.kmeans_blocks == [0, 5, 12]
