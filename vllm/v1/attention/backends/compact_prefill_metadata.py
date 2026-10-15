# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU eligibility and page accounting for bounded whole-prompt prefill."""

from dataclasses import dataclass
from itertools import accumulate

MAX_PAGES = 4096
PAGE_SIZE = 16
PAGE_ELEMENTS = PAGE_SIZE * 8 * 128
PLANE_BYTES = MAX_PAGES * PAGE_ELEMENTS * 2
PAYLOAD_BYTES = 2 * PLANE_BYTES


@dataclass(frozen=True)
class CompactPagePlan:
    query_lengths: tuple[int, ...]
    page_indptr: tuple[int, ...]
    last_page_lengths: tuple[int, ...]

    @property
    def pages(self) -> int:
        return self.page_indptr[-1]


def plan_whole_prompt(
    query_indptr,
    kv_lengths,
    *,
    max_sequences: int,
    num_decodes: int = 0,
    causal: bool = True,
    prefix: bool = False,
    supported: bool = True,
) -> tuple[CompactPagePlan | None, str]:
    """Check CPU metadata without reading a device tensor or source page IDs."""
    if not supported:
        return None, "unsupported_geometry"
    if not causal or prefix:
        return None, "mask_or_prefix"
    if num_decodes:
        return None, "decode_or_mixed"
    q = tuple(int(x) for x in query_indptr)
    kv = tuple(int(x) for x in kv_lengths)
    if not kv or len(kv) > max_sequences or len(q) != len(kv) + 1 or q[0] != 0:
        return None, "request_count"
    lengths = tuple(b - a for a, b in zip(q, q[1:]))
    if any(n <= 0 or n > 8192 for n in lengths):
        return None, "query_length"
    if lengths != kv:
        return None, "context_or_chunk"
    counts = tuple((n + PAGE_SIZE - 1) // PAGE_SIZE for n in kv)
    ptr = (0, *accumulate(counts))
    if ptr[-1] > MAX_PAGES:
        return None, "page_budget"
    tails = tuple((n - 1) % PAGE_SIZE + 1 for n in kv)
    return CompactPagePlan(lengths, ptr, tails), "candidate"
