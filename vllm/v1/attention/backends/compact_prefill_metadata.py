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
    source_page_start: int = 0

    @property
    def source_page_end(self) -> int:
        return self.source_page_start + self.pages

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


def plan_prefill_subset(
    query_indptr,
    kv_lengths,
    page_indptr,
    *,
    max_sequences: int,
    num_decodes: int = 0,
    causal: bool = True,
    prefix: bool = False,
    supported: bool = True,
) -> tuple[CompactPagePlan | None, str]:
    """Plan only whole fresh prompts after the scheduler's decode prefix."""
    q = tuple(int(x) for x in query_indptr)
    kv = tuple(int(x) for x in kv_lengths)
    pages = tuple(int(x) for x in page_indptr)
    count = len(kv)
    if (
        not 0 <= num_decodes <= count
        or count > max_sequences
        or len(q) != count + 1
        or not q
        or q[0] != 0
        or any(b <= a for a, b in zip(q, q[1:]))
    ):
        return None, "request_count"
    if (
        len(pages) != count + 1
        or pages[0] != 0
        or any(n <= 0 for n in kv)
        or tuple(b - a for a, b in zip(pages, pages[1:]))
        != tuple((n + PAGE_SIZE - 1) // PAGE_SIZE for n in kv)
    ):
        return None, "page_mapping"
    if num_decodes == count:
        return None, "decode_only"
    offset = q[num_decodes]
    plan, reason = plan_whole_prompt(
        tuple(x - offset for x in q[num_decodes:]),
        kv[num_decodes:],
        max_sequences=max_sequences,
        causal=causal,
        prefix=prefix,
        supported=supported,
    )
    if plan is None:
        return None, reason
    return CompactPagePlan(
        plan.query_lengths,
        plan.page_indptr,
        plan.last_page_lengths,
        source_page_start=pages[num_decodes],
    ), reason
