# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.kmesh import KMeshSegmentKind, parse_kmesh_extra_args
from vllm.v1.request import Request
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

pytestmark = pytest.mark.skip_global_cleanup


def _sideband_context() -> dict:
    return {
        "schema_version": 1,
        "enabled": True,
        "execution_id": "execution-1",
        "group_kind": "agent_round",
        "group_id": "group-1",
        "round_id": "round-1",
        "branch_id": "agent-0",
        "parent_branch_id": None,
        "state_kind": "kv_cache",
        "cache_domain": "tenant-a",
        "segment_encoding": "sideband",
        "segments": [
            {
                "segment_id": "private-0",
                "kind": "private_history",
                "token_start": 0,
                "token_end": 2,
                "logical_order": 0,
                "source_id": "agent-0",
                "shareable": False,
                "lineage_id": "lineage-private-0",
            },
            {
                "segment_id": "shared-0",
                "kind": "shared_update",
                "token_start": 2,
                "token_end": 5,
                "logical_order": 1,
                "source_id": "coordinator",
                "shareable": True,
                "lineage_id": "lineage-shared-0",
            },
        ],
    }


def test_kmesh_sideband_survives_engine_core_msgpack_and_request() -> None:
    prompt_token_ids = [10, 11, 20, 21, 22]
    extra_args = {"kmesh": _sideband_context(), "unrelated_plugin": "kept"}
    core_request = EngineCoreRequest(
        request_id="request-1",
        prompt_token_ids=prompt_token_ids,
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=1, extra_args=extra_args),
        pooling_params=None,
        arrival_time=1.0,
        lora_request=None,
        cache_salt="cache-salt",
        data_parallel_rank=None,
    )

    encoded = MsgpackEncoder().encode(core_request)
    decoded = MsgpackDecoder(EngineCoreRequest).decode(encoded)

    assert decoded.sampling_params is not None
    assert decoded.sampling_params.extra_args == extra_args
    scheduler_request = Request.from_engine_core_request(decoded, block_hasher=None)
    assert scheduler_request.sampling_params.extra_args == extra_args

    parsed = parse_kmesh_extra_args(
        scheduler_request.sampling_params.extra_args,
        scheduler_request.prompt_token_ids,
    )
    assert parsed.is_enabled
    assert parsed.context is not None
    assert parsed.context.cache_domain == "tenant-a"
    assert parsed.context.segments[1].kind is KMeshSegmentKind.SHARED_UPDATE
