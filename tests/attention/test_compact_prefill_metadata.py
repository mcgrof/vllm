# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The metadata contract is CPU-only and independent of CUDA imports."""

import importlib.util
import sys
from pathlib import Path

import pytest

path = (
    Path(__file__).parents[2] / "vllm/v1/attention/backends/compact_prefill_metadata.py"
)
spec = importlib.util.spec_from_file_location("compact_prefill_metadata", path)
assert spec is not None and spec.loader is not None
metadata = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = metadata
spec.loader.exec_module(metadata)


def make(lengths, **kwargs):
    ptr = [0]
    for length in lengths:
        ptr.append(ptr[-1] + length)
    return metadata.plan_whole_prompt(ptr, lengths, max_sequences=96, **kwargs)


def test_allocation_arithmetic():
    assert metadata.PLANE_BYTES == 128 * 2**20
    assert metadata.PAYLOAD_BYTES == 256 * 2**20
    plan, reason = make([8192])
    assert reason == "candidate" and plan.pages == 512
    assert plan.pages * 16 * 8 * 128 * 4 == 32 * 2**20


def test_unequal_tails_and_repeated_page_occurrences():
    plan, _ = make([8191, 8187, 17, 1])
    assert plan.page_indptr == (0, 512, 1024, 1026, 1027)
    assert plan.last_page_lengths == (15, 11, 1, 1)
    # Source IDs are opaque. Zero, sparse/permuted IDs, and shared pages
    # retain one destination occurrence for every source occurrence.
    ids = [0, 79, 3, 79, 0]
    compact, _ = make([33, 17])
    assert compact.page_indptr == (0, 3, 5)
    expected = [
        [ids[i] for i in range(a, b)]
        for a, b in zip(compact.page_indptr, compact.page_indptr[1:])
    ]
    assert expected == [[0, 79, 3], [79, 0]]


def test_budget_boundary():
    plan, _ = make([8192] * 8)
    assert plan.pages == 4096
    assert make([8192] * 8 + [1])[1] == "page_budget"


def test_variable_batch_uses_configured_limit():
    plan, reason = make([31 + i for i in range(32)])
    assert reason == "candidate" and len(plan.query_lengths) == 32
    assert make([16] * 96)[0].pages == 96
    assert make([16] * 97)[1] == "request_count"


@pytest.mark.parametrize(
    "kw, reason",
    [
        ({"num_decodes": 1}, "decode_or_mixed"),
        ({"causal": False}, "mask_or_prefix"),
        ({"prefix": True}, "mask_or_prefix"),
        ({"supported": False}, "unsupported_geometry"),
    ],
)
def test_unsupported_routes(kw, reason):
    assert make([64], **kw) == (None, reason)


def test_existing_context_and_chunked_prefill():
    assert (
        metadata.plan_whole_prompt([0, 32], [64], max_sequences=96)[1]
        == "context_or_chunk"
    )
    assert make([8193])[1] == "query_length"
    assert make([0])[1] == "query_length"
    assert (
        metadata.plan_whole_prompt([1, 33], [32], max_sequences=96)[1]
        == "request_count"
    )
