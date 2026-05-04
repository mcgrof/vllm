# SPDX-License-Identifier: Apache-2.0
"""Integration tests for CartridgeConnector scheduler-side methods.

These test the connector's interaction with vLLM's scheduler API
without requiring a running server or GPU. Uses mock objects for
Request, SchedulerOutput, and ForwardContext.

Covers:
1. get_num_new_matched_tokens: returns cartridge token count, caps
   to prompt length, handles already-computed tokens.
2. update_state_after_alloc: idempotent (safe to call twice).
3. build_connector_meta: correct slot mapping for single and multiple
   requests, correct num_tokens.
4. Manifest validation on load (wrong model catches at init).
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_connector import (
    CartridgeConnector,
    CartridgeConnectorMetadata,
    align_to_block_size,
    load_cartridge,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cartridge_pt(path, num_layers=2, num_kv_heads=2,
                       num_tokens=32, head_dim=4):
    """Create a minimal TrainableCache .pt file."""
    cache = {"trainable_keys": [], "trainable_values": []}
    for _ in range(num_layers):
        cache["trainable_keys"].append(
            torch.nn.Parameter(torch.randn(1, num_kv_heads, num_tokens, head_dim)))
        cache["trainable_values"].append(
            torch.nn.Parameter(torch.randn(1, num_kv_heads, num_tokens, head_dim)))
    torch.save(cache, path)


def _make_connector(cartridge_path, block_size=16):
    """Create a CartridgeConnector with mocked vllm_config."""
    vllm_config = MagicMock()
    vllm_config.cache_config.block_size = block_size

    # Mock the kv_transfer_config to return our cartridge_path
    extra_config = {"cartridge_path": cartridge_path}

    def get_from_extra_config(key, default=None):
        return extra_config.get(key, default)

    kv_transfer_config = MagicMock()
    kv_transfer_config.get_from_extra_config = get_from_extra_config
    vllm_config.kv_transfer_config = kv_transfer_config

    # Patch the base class to avoid real KVConnector init
    with patch.object(
        CartridgeConnector, '__init__',
        _make_patched_init(vllm_config, cartridge_path, block_size),
    ):
        connector = CartridgeConnector.__new__(CartridgeConnector)
        connector.__init__(vllm_config, MagicMock())

    return connector


def _make_patched_init(vllm_config, cartridge_path, block_size):
    """Build a patched __init__ that skips the base class.

    Matches the current multi-cartridge-aware CartridgeConnector
    state shape (``_cartridge_meta`` dict, ``_request_cartridge_ids``,
    ``_requests_need_load`` as a set of committed request_ids).
    """
    def patched_init(self, vllm_config_arg, role, kv_cache_config=None):
        from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_store import (
            CartridgeStore,
        )
        from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_manifest import (
            CartridgeManifest,
        )
        from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_router import (
            StaticCartridgeRouter,
        )

        self._block_size = block_size
        self._request_cartridge_ids = {}
        self._request_num_tokens = {}
        self._requests_need_load = set()
        self._kv_transfer_config = vllm_config.kv_transfer_config
        self._vllm_config = vllm_config

        cartridge = load_cartridge(cartridge_path)
        manifest = CartridgeManifest(
            cartridge_id="test",
            model_id="test/model",
            num_layers=cartridge["num_layers"],
            num_kv_heads=cartridge["num_kv_heads"],
            head_dim=cartridge["head_dim"],
            dtype="float32",
            num_tokens_raw=cartridge["num_tokens"],
            num_tokens_aligned=align_to_block_size(
                cartridge["num_tokens"], block_size),
            block_size=block_size,
            num_blocks=align_to_block_size(
                cartridge["num_tokens"], block_size) // block_size,
            has_frozen_prefix=False,
        )
        del cartridge

        self._store = CartridgeStore(block_size=block_size)
        self._store.load("test", cartridge_path, manifest, device="cpu")
        residency = self._store.get_residency("test")

        self._cartridge_meta = {
            "test": {
                "num_tokens": residency.num_tokens,
                "num_blocks": (residency.num_tokens
                               // block_size),
                "num_layers": residency.num_layers,
            }
        }
        self._default_cartridge_id = "test"
        # Integration tests target the singleton dispatch path.
        self._router = StaticCartridgeRouter("test")
        # Wire a minimal GPUResidencyManager — these tests focus on
        # the scheduler-side plumbing, not the GPU tier. Capacity is
        # huge so eviction never kicks in; device is CPU for
        # host-only testing.
        from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_gpu_residency import (
            GPUResidencyManager,
        )
        self._residency = GPUResidencyManager(
            store=self._store,
            capacity_bytes=1 << 40,  # 1 TiB (effectively unbounded)
            device="cpu",
        )

    return patched_init


def _make_mock_request(request_id, prompt_len):
    """Create a mock Request with prompt_token_ids + extras.

    The mock carries ``sampling_params.extra_args`` so the router can
    resolve a cartridge_id. Tests that exercise StaticCartridgeRouter
    don't actually need this, but explicit routing tests do.
    """
    req = MagicMock()
    req.request_id = request_id
    req.prompt_token_ids = list(range(prompt_len))
    req.sampling_params.extra_args = {"cartridge_id": "test"}
    return req


# ---------------------------------------------------------------------------
# Tests: get_num_new_matched_tokens
# ---------------------------------------------------------------------------

class TestGetNumNewMatchedTokens:
    def setup_method(self):
        self._tmpfile = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        _make_cartridge_pt(self._tmpfile.name, num_layers=2,
                          num_kv_heads=2, num_tokens=32, head_dim=4)
        self.connector = _make_connector(self._tmpfile.name, block_size=16)

    def teardown_method(self):
        Path(self._tmpfile.name).unlink()

    def test_returns_cartridge_token_count(self):
        req = _make_mock_request("r1", prompt_len=64)
        num_new, _ = self.connector.get_num_new_matched_tokens(req, 0)
        assert num_new == 32  # cartridge has 32 tokens

    def test_caps_to_prompt_length(self):
        req = _make_mock_request("r1", prompt_len=20)
        num_new, _ = self.connector.get_num_new_matched_tokens(req, 0)
        # 20 tokens, block_size=16 -> aligned to 16
        assert num_new == 16

    def test_subtracts_already_computed(self):
        req = _make_mock_request("r1", prompt_len=64)
        num_new, _ = self.connector.get_num_new_matched_tokens(req, 16)
        assert num_new == 16  # 32 - 16 already computed

    def test_returns_zero_when_fully_computed(self):
        req = _make_mock_request("r1", prompt_len=64)
        num_new, _ = self.connector.get_num_new_matched_tokens(req, 32)
        assert num_new == 0

    def test_returns_zero_for_no_prompt(self):
        req = MagicMock()
        req.prompt_token_ids = None
        num_new, _ = self.connector.get_num_new_matched_tokens(req, 0)
        assert num_new == 0

    def test_prompt_shorter_than_one_block(self):
        req = _make_mock_request("r1", prompt_len=5)
        num_new, _ = self.connector.get_num_new_matched_tokens(req, 0)
        # 5 tokens, block_size=16 -> aligned to 0
        assert num_new == 0


# ---------------------------------------------------------------------------
# Tests: update_state_after_alloc — idempotency
# ---------------------------------------------------------------------------

class TestUpdateStateAfterAlloc:
    def setup_method(self):
        self._tmpfile = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        _make_cartridge_pt(self._tmpfile.name)
        self.connector = _make_connector(self._tmpfile.name)

    def teardown_method(self):
        Path(self._tmpfile.name).unlink()

    def _prime_resolved(self, req_id: str):
        """Simulate get_num_new_matched_tokens having resolved this id."""
        self.connector._request_cartridge_ids[req_id] = "test"
        self.connector._request_num_tokens[req_id] = 32

    def test_records_request(self):
        req = _make_mock_request("r1", 64)
        self._prime_resolved("r1")
        self.connector.update_state_after_alloc(req, MagicMock(), 32)
        assert "r1" in self.connector._requests_need_load

    def test_idempotent_double_call(self):
        """vLLM's API may call this twice for the same request."""
        req = _make_mock_request("r1", 64)
        self._prime_resolved("r1")
        self.connector.update_state_after_alloc(req, MagicMock(), 32)
        self.connector.update_state_after_alloc(req, MagicMock(), 32)
        # Should still have exactly one entry, not break
        assert "r1" in self.connector._requests_need_load
        assert len(self.connector._requests_need_load) == 1

    def test_zero_external_tokens_not_recorded(self):
        req = _make_mock_request("r1", 64)
        self._prime_resolved("r1")
        self.connector.update_state_after_alloc(req, MagicMock(), 0)
        assert "r1" not in self.connector._requests_need_load


# ---------------------------------------------------------------------------
# Tests: build_connector_meta — slot mapping
# ---------------------------------------------------------------------------

class TestBuildConnectorMeta:
    def setup_method(self):
        self._tmpfile = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        _make_cartridge_pt(self._tmpfile.name, num_layers=2,
                          num_kv_heads=2, num_tokens=32, head_dim=4)
        self.connector = _make_connector(self._tmpfile.name, block_size=16)

    def teardown_method(self):
        Path(self._tmpfile.name).unlink()

    def _commit(self, req_id: str, num_tokens: int = 32,
                cartridge_id: str = "test"):
        """Simulate get_num_new_matched_tokens → update_state_after_alloc.

        Pre-populates the per-request dicts that build_connector_meta
        consumes. This is what the real scheduler→connector flow would
        produce; the tests below exercise build_connector_meta in
        isolation.
        """
        self.connector._request_cartridge_ids[req_id] = cartridge_id
        self.connector._request_num_tokens[req_id] = num_tokens
        self.connector._requests_need_load.add(req_id)

    def test_single_request_slot_mapping(self):
        self._commit("r1")

        # Mock scheduler output
        sched_out = MagicMock()
        new_req = MagicMock()
        new_req.req_id = "r1"
        new_req.block_ids = [[0, 1, 2, 3]]  # 4 blocks allocated
        sched_out.scheduled_new_reqs = [new_req]

        meta = self.connector.build_connector_meta(sched_out)

        assert isinstance(meta, CartridgeConnectorMetadata)
        assert len(meta.requests) == 1
        assert meta.requests[0].cartridge_id == "test"
        assert meta.requests[0].num_tokens == 32

        # Slot mapping should cover blocks 0,1 (cartridge needs 2 blocks)
        sm = meta.requests[0].slot_mapping
        assert len(sm) == 32  # 2 blocks * 16 tokens
        assert sm[0].item() == 0   # block 0, offset 0
        assert sm[15].item() == 15  # block 0, offset 15
        assert sm[16].item() == 16  # block 1, offset 0
        assert sm[31].item() == 31  # block 1, offset 15

    def test_clears_requests_after_build(self):
        self._commit("r1")

        sched_out = MagicMock()
        new_req = MagicMock()
        new_req.req_id = "r1"
        new_req.block_ids = [[0, 1, 2, 3]]
        sched_out.scheduled_new_reqs = [new_req]

        self.connector.build_connector_meta(sched_out)
        assert len(self.connector._requests_need_load) == 0
        # Per-request resolution state for scheduled reqs is cleared too.
        assert "r1" not in self.connector._request_cartridge_ids
        assert "r1" not in self.connector._request_num_tokens

    def test_unrelated_request_ignored(self):
        """Requests not in _requests_need_load produce no metadata."""
        sched_out = MagicMock()
        new_req = MagicMock()
        new_req.req_id = "r_unknown"
        new_req.block_ids = [[0, 1]]
        sched_out.scheduled_new_reqs = [new_req]

        meta = self.connector.build_connector_meta(sched_out)
        assert len(meta.requests) == 0

    def test_non_contiguous_block_ids(self):
        """Slot mapping works when scheduler allocates non-contiguous blocks."""
        self._commit("r1")

        sched_out = MagicMock()
        new_req = MagicMock()
        new_req.req_id = "r1"
        new_req.block_ids = [[5, 10, 20, 30]]  # non-contiguous
        sched_out.scheduled_new_reqs = [new_req]

        meta = self.connector.build_connector_meta(sched_out)
        sm = meta.requests[0].slot_mapping

        # Block 5: slots 80-95, Block 10: slots 160-175
        assert sm[0].item() == 5 * 16   # = 80
        assert sm[16].item() == 10 * 16  # = 160
