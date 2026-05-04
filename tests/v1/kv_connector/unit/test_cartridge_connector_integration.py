# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration tests for CartridgeConnector scheduler-side flow.

Constructs a real CartridgeConnector (through the real __init__)
against a mocked VllmConfig and exercises the scheduler-side
methods end-to-end without a GPU:

1. get_num_new_matched_tokens: token accounting against prompt
   length, block alignment, and already-computed tokens.
2. update_state_after_alloc: idempotency.
3. build_connector_meta: slot-mapping construction and state
   clearing.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_connector import (
    CartridgeConnector,
    CartridgeConnectorMetadata,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cartridge_pt(path, num_layers=2, num_kv_heads=2, num_tokens=32, head_dim=4):
    """Create a minimal TrainableCache .pt file."""
    cache = {"trainable_keys": [], "trainable_values": []}
    for _ in range(num_layers):
        cache["trainable_keys"].append(
            torch.nn.Parameter(torch.randn(1, num_kv_heads, num_tokens, head_dim))
        )
        cache["trainable_values"].append(
            torch.nn.Parameter(torch.randn(1, num_kv_heads, num_tokens, head_dim))
        )
    torch.save(cache, path)


def _make_connector(cartridge_path, block_size=16):
    """Create a CartridgeConnector through the real __init__.

    The VllmConfig is mocked, but every field the connector reads is
    set to a real value so no MagicMock leaks into arithmetic or
    comparisons.
    """
    vllm_config = MagicMock()
    vllm_config.cache_config.block_size = block_size
    vllm_config.parallel_config.tensor_parallel_size = 1

    extra_config = {"cartridge_path": cartridge_path}

    def get_from_extra_config(key, default=None):
        return extra_config.get(key, default)

    kv_transfer_config = MagicMock()
    kv_transfer_config.get_from_extra_config = get_from_extra_config
    vllm_config.kv_transfer_config = kv_transfer_config

    return CartridgeConnector(vllm_config, KVConnectorRole.SCHEDULER, MagicMock())


def _make_mock_request(request_id, prompt_len):
    """Create a mock Request with prompt_token_ids."""
    req = MagicMock()
    req.request_id = request_id
    req.prompt_token_ids = list(range(prompt_len))
    return req


# ---------------------------------------------------------------------------
# Tests: get_num_new_matched_tokens
# ---------------------------------------------------------------------------


class TestGetNumNewMatchedTokens:
    def setup_method(self):
        fd, self._tmppath = tempfile.mkstemp(suffix=".pt")
        os.close(fd)
        _make_cartridge_pt(
            self._tmppath, num_layers=2, num_kv_heads=2, num_tokens=32, head_dim=4
        )
        self.connector = _make_connector(self._tmppath, block_size=16)

    def teardown_method(self):
        Path(self._tmppath).unlink()

    def test_returns_cartridge_token_count(self):
        req = _make_mock_request("r1", prompt_len=64)
        num_new, _ = self.connector.get_num_new_matched_tokens(req, 0)
        assert num_new == 32  # cartridge has 32 tokens

    def test_caps_to_prompt_length(self):
        req = _make_mock_request("r1", prompt_len=20)
        num_new, _ = self.connector.get_num_new_matched_tokens(req, 0)
        # 20 tokens, minus one for the scheduler, block_size=16
        # -> aligned to 16
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
        fd, self._tmppath = tempfile.mkstemp(suffix=".pt")
        os.close(fd)
        _make_cartridge_pt(self._tmppath)
        self.connector = _make_connector(self._tmppath)

    def teardown_method(self):
        Path(self._tmppath).unlink()

    def test_records_request(self):
        req = _make_mock_request("r1", 64)
        self.connector.update_state_after_alloc(req, MagicMock(), 32)
        assert "r1" in self.connector._requests_need_load

    def test_idempotent_double_call(self):
        """vLLM's API may call this twice for the same request."""
        req = _make_mock_request("r1", 64)
        self.connector.update_state_after_alloc(req, MagicMock(), 32)
        self.connector.update_state_after_alloc(req, MagicMock(), 32)
        # Should still have exactly one entry, not break
        assert "r1" in self.connector._requests_need_load
        assert len(self.connector._requests_need_load) == 1

    def test_zero_external_tokens_not_recorded(self):
        req = _make_mock_request("r1", 64)
        self.connector.update_state_after_alloc(req, MagicMock(), 0)
        assert "r1" not in self.connector._requests_need_load


# ---------------------------------------------------------------------------
# Tests: build_connector_meta — slot mapping
# ---------------------------------------------------------------------------


class TestBuildConnectorMeta:
    def setup_method(self):
        fd, self._tmppath = tempfile.mkstemp(suffix=".pt")
        os.close(fd)
        _make_cartridge_pt(
            self._tmppath, num_layers=2, num_kv_heads=2, num_tokens=32, head_dim=4
        )
        self.connector = _make_connector(self._tmppath, block_size=16)

    def teardown_method(self):
        Path(self._tmppath).unlink()

    def _commit(self, req_id: str):
        """Simulate update_state_after_alloc having committed req_id."""
        self.connector._requests_need_load[req_id] = _make_mock_request(req_id, 64)

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
        assert meta.requests[0].num_tokens == 32

        # Slot mapping should cover blocks 0,1 (cartridge needs 2 blocks)
        sm = meta.requests[0].slot_mapping
        assert len(sm) == 32  # 2 blocks * 16 tokens
        assert sm[0].item() == 0  # block 0, offset 0
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
        """Slot mapping works when scheduler allocates non-contiguous
        blocks."""
        self._commit("r1")

        sched_out = MagicMock()
        new_req = MagicMock()
        new_req.req_id = "r1"
        new_req.block_ids = [[5, 10, 20, 30]]  # non-contiguous
        sched_out.scheduled_new_reqs = [new_req]

        meta = self.connector.build_connector_meta(sched_out)
        sm = meta.requests[0].slot_mapping

        # Block 5: slots 80-95, Block 10: slots 160-175
        assert sm[0].item() == 5 * 16  # = 80
        assert sm[16].item() == 10 * 16  # = 160
