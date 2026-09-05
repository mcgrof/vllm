# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for fail-closed provenance keys in the LMCache MP connector."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1 import (
    lmcache_mp_connector as connector_module,
)
from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_connector import (
    LMCacheMPConnectorUpstream,
    LMCacheMPRequestMetadata,
    LMCacheMPRequestState,
    LMCacheMPRequestTracker,
    get_bound_external_kv_keys,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request, RequestStatus


def make_request(
    keys: list[bytes] | None,
    *,
    request_id: str = "request",
) -> SimpleNamespace:
    """Build the request fields consumed by the connector tracker."""
    return SimpleNamespace(
        request_id=request_id,
        cache_salt="tenant-salt",
        all_token_ids=list(range(512)),
        block_hashes=[bytes([index]) * 32 for index in range(32)],
        kv_provenance_keys=None if keys is None else tuple(keys),
        status=RequestStatus.WAITING,
    )


def make_tracker(keys: list[bytes]) -> LMCacheMPRequestTracker:
    """Build a two-chunk provenance-required tracker."""
    return LMCacheMPRequestTracker(
        make_request(keys),
        blocks_in_chunk=16,
        vllm_block_size=16,
        require_provenance=True,
    )


def test_get_bound_external_kv_keys_accepts_engine_binding() -> None:
    """The connector wraps the engine-internal key binding for LMCache."""
    keys = get_bound_external_kv_keys(make_request([b"a" * 32, b"b" * 32]))

    assert keys is not None
    assert keys.keys == (b"a" * 32, b"b" * 32)


def test_request_provenance_binding_is_internal_and_immutable() -> None:
    """The binding is not sourced from transfer params and cannot be replaced."""
    request = Request(
        request_id="request",
        prompt_token_ids=list(range(256)),
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
    )
    request.kv_transfer_params = {"lmcache_external_kv_keys": [b"wrong" * 7]}

    assert get_bound_external_kv_keys(request) is None
    request.bind_kv_provenance_keys((b"a" * 32,))
    assert get_bound_external_kv_keys(request).keys == (b"a" * 32,)
    with pytest.raises(RuntimeError, match="already bound"):
        request.bind_kv_provenance_keys((b"b" * 32,))


@pytest.mark.parametrize(
    "keys",
    [None, [b"short"], [b"a" * 32]],
)
def test_required_tracker_rejects_missing_or_mismatched_keys(
    keys: list[bytes] | None,
) -> None:
    """A request cannot enter provenance mode with incomplete key material."""
    with pytest.raises(ValueError):
        LMCacheMPRequestTracker(
            make_request(keys),
            blocks_in_chunk=16,
            vllm_block_size=16,
            require_provenance=True,
        )


def test_store_metadata_carries_only_its_external_key_range() -> None:
    """The scheduler-to-worker operation preserves the corresponding keys."""
    tracker = make_tracker([b"a" * 32, b"b" * 32])
    tracker.allocated_block_ids = list(range(32))
    tracker.num_scheduled_tokens = 512

    metadata = LMCacheMPRequestMetadata.GetStoreMetadata(tracker, 16, 16)

    assert metadata is not None
    assert metadata.op.require_provenance
    assert metadata.op.external_keys is not None
    assert metadata.op.external_keys.keys == (b"a" * 32, b"b" * 32)


def test_retrieve_metadata_carries_the_external_hit_range() -> None:
    """A retrieve addresses exactly the keys found by external lookup."""
    tracker = make_tracker([b"a" * 32, b"b" * 32])
    tracker.allocated_block_ids = list(range(32))
    tracker.num_vllm_hit_blocks = 16
    tracker.num_lmcache_hit_blocks = 32
    tracker.state = LMCacheMPRequestState.WAITING_FOR_LOAD

    metadata = LMCacheMPRequestMetadata.GetRetrieveMetadata(tracker, 16, 16)

    assert metadata is not None
    assert metadata.op.external_keys is not None
    assert metadata.op.external_keys.keys == (b"b" * 32,)
    assert metadata.op.skip_first_n_tokens == 0


def test_legacy_metadata_omits_new_load_store_arguments(monkeypatch) -> None:
    """An older LMCache LoadStoreOp remains usable outside provenance mode."""

    class LegacyLoadStoreOp:
        def __init__(
            self,
            token_ids,
            block_ids,
            start=0,
            end=0,
            skip_first_n_tokens=0,
        ) -> None:
            self.token_ids = token_ids
            self.block_ids = block_ids
            self.start = start
            self.end = end
            self.skip_first_n_tokens = skip_first_n_tokens

    monkeypatch.setattr(connector_module, "LoadStoreOp", LegacyLoadStoreOp)
    tracker = LMCacheMPRequestTracker(
        make_request(None),
        blocks_in_chunk=16,
        vllm_block_size=16,
    )
    tracker.allocated_block_ids = list(range(32))
    tracker.num_scheduled_tokens = 512

    metadata = LMCacheMPRequestMetadata.GetStoreMetadata(tracker, 16, 16)

    assert metadata is not None
    assert metadata.op.end == 512


def test_lookup_records_external_hit_separately_from_vllm_hit() -> None:
    """The positive control exposes an external-only hit in tracker counters."""
    request = make_request([b"a" * 32, b"b" * 32])
    connector = LMCacheMPConnectorUpstream.__new__(LMCacheMPConnectorUpstream)
    connector.request_trackers = {}
    connector.require_provenance = True
    connector.vllm_block_size = 16
    connector.scheduler_adapter = MagicMock()
    connector.scheduler_adapter.num_blocks_per_chunk.return_value = 16
    connector.scheduler_adapter.check_lookup_result.return_value = 512

    result = connector.get_num_new_matched_tokens(request, num_computed_tokens=0)

    assert result == (512, True)
    tracker = connector.request_trackers[request.request_id]
    assert tracker.num_vllm_hit_blocks == 0
    assert tracker.num_lmcache_hit_blocks == 32
    lookup = connector.scheduler_adapter.maybe_submit_lookup_request.call_args.kwargs
    assert lookup["require_provenance"]
    assert lookup["external_keys"].keys == (b"a" * 32, b"b" * 32)


def test_lock_release_reuses_external_keys_and_access_scope() -> None:
    """The early APC lock-release path cannot reconstruct a token-only key."""
    request = make_request([b"a" * 32, b"b" * 32])
    tracker = make_tracker([b"a" * 32, b"b" * 32])
    tracker.num_vllm_hit_blocks = 16
    tracker.num_lmcache_hit_blocks = 16
    connector = LMCacheMPConnectorUpstream.__new__(LMCacheMPConnectorUpstream)
    connector.request_trackers = {request.request_id: tracker}
    connector.vllm_block_size = 16
    connector.scheduler_adapter = MagicMock()
    connector.scheduler_adapter.num_blocks_per_chunk.return_value = 16
    blocks = MagicMock()
    blocks.get_block_ids.return_value = (list(range(16)),)

    connector.update_state_after_alloc(request, blocks, num_external_tokens=0)

    release = connector.scheduler_adapter.free_lookup_locks.call_args.kwargs
    assert release["cache_salt"] == "tenant-salt"
    assert release["require_provenance"]
    assert release["external_keys"].keys == (b"a" * 32,)
