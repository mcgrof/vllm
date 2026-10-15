# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in serial-stream prefill using a bounded native K/V gather workspace."""

import os
import threading
from collections import Counter
from contextlib import contextmanager

import torch
from flashinfer import BatchPrefillWithPagedKVCacheWrapper

from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.compact_prefill_metadata import (
    MAX_PAGES,
    PAGE_ELEMENTS,
    PAGE_SIZE,
    PAYLOAD_BYTES,
    CompactPagePlan,
)


@triton.jit
def _gather(
    K,
    V,
    IDs,
    DK,
    DV,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    KS2: tl.constexpr,
    KS3: tl.constexpr,
    VS0: tl.constexpr,
    VS1: tl.constexpr,
    VS2: tl.constexpr,
    VS3: tl.constexpr,
    ELEMENTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    page = tl.program_id(0)
    x = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    src = tl.load(IDs + page).to(tl.int64)
    token, head, dim = x // 1024, (x // 128) % 8, x % 128
    k = tl.load(K + src * KS0 + token * KS1 + head * KS2 + dim * KS3)
    v = tl.load(V + src * VS0 + token * VS1 + head * VS2 + dim * VS3)
    tl.store(DK + page * ELEMENTS + x, k)
    tl.store(DV + page * ELEMENTS + x, v.to(tl.float16))


class CompactPrefillWorkspace:
    """One worker owns this storage and submits all accesses on one stream."""

    def __init__(self, device):
        self.backing = torch.empty(
            PAYLOAD_BYTES // 2, dtype=torch.float16, device=device
        )
        self.k, self.v = self.backing.view(2, MAX_PAGES, PAGE_SIZE, 8, 128).unbind(0)
        self.indices = torch.arange(MAX_PAGES, dtype=torch.int32, device=device)
        self.float_workspace = torch.empty(32 * 2**20, dtype=torch.uint8, device=device)
        self.wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.float_workspace, "NHD", backend="fa3"
        )
        self._lock = threading.Lock()
        self._stream = None
        self._thread = None
        self.calls: Counter[str] = Counter()
        self.queries = self.read_pages = 0
        self.last_plan: CompactPagePlan | None = None
        self.allocation_before_profile: dict[str, int] | None = None

    @contextmanager
    def serial(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("compact prefill does not support overlapping forwards")
        try:
            thread = threading.get_ident()
            if self._thread is None:
                self._thread = thread
            if self._thread != thread:
                raise RuntimeError(
                    "compact prefill requires one worker submission thread"
                )
            stream = torch.cuda.current_stream(self.backing.device).cuda_stream
            if self._stream is None:
                self._stream = stream
            if self._stream != stream:
                raise RuntimeError("compact prefill requires one serial CUDA stream")
            yield
        finally:
            self._lock.release()

    def plan(self, page_plan, qo, indptr, tails, **kwargs):
        with self.serial():
            self.wrapper.plan(
                qo_indptr=qo,
                paged_kv_indptr=indptr,
                paged_kv_indices=self.indices[: page_plan.pages],
                paged_kv_last_page_len=tails,
                num_qo_heads=32,
                num_kv_heads=8,
                head_dim_qk=128,
                page_size=PAGE_SIZE,
                causal=True,
                q_data_type=torch.float16,
                kv_data_type=torch.float16,
                k_data_type=torch.float16,
                v_data_type=torch.float16,
                o_data_type=torch.float16,
                **kwargs,
            )
            self.last_plan = page_plan

    def gather(self, cache, source_indices):
        k, v = cache
        if k.dtype != torch.float16 or v.dtype != torch.float8_e4m3fn:
            raise RuntimeError("compact prefill requires stored K16/V8")
        if k.shape != v.shape or tuple(k.shape[1:]) != (PAGE_SIZE, 8, 128):
            raise RuntimeError("compact prefill requires matching NHD cache planes")
        n = source_indices.numel()
        if not 0 < n <= MAX_PAGES:
            raise RuntimeError("compact prefill page budget exceeded")
        _gather[(n, PAGE_ELEMENTS // 4096)](
            k,
            v,
            source_indices,
            self.k,
            self.v,
            *k.stride(),
            *v.stride(),
            ELEMENTS=PAGE_ELEMENTS,
            BLOCK=4096,
            num_warps=8,
        )
        return self.k[:n], self.v[:n]

    def run(self, query, cache, source_indices, page_plan: CompactPagePlan, **kwargs):
        with self.serial():
            if page_plan is not self.last_plan:
                raise RuntimeError("compact prefill plan was replaced before execution")
            compact = self.gather(cache, source_indices)
            result = self.wrapper.run(query, compact, **kwargs)
            self.calls["candidate"] += 1
            self.queries += query.shape[0]
            self.read_pages += page_plan.pages
            return result

    def receipt(self):
        return {
            "calls": dict(self.calls),
            "total_queries": self.queries,
            "read_pages": self.read_pages,
            "workspace_ptr": self.backing.data_ptr(),
            "payload_capacity_bytes": self.backing.untyped_storage().nbytes(),
            "float_workspace_bytes": self.float_workspace.numel(),
            "compact_index_bytes": self.indices.numel() * self.indices.element_size(),
            "active_payload_bytes": 0
            if self.last_plan is None
            else self.last_plan.pages * PAGE_ELEMENTS * 4,
            "stream": self._stream,
            "submission_thread": self._thread,
            "allocation_before_KV_profile": self.allocation_before_profile,
        }


def initialize_workspace(config, device):
    """Allocate before memory profiling so the automatic KV budget counts it."""
    if os.environ.get("VLLM_COMPACT_PREFILL", "0") != "1":
        return None
    parallel = config.parallel_config
    if (
        parallel.tensor_parallel_size != 1
        or parallel.pipeline_parallel_size != 1
        or parallel.data_parallel_size != 1
        or parallel.use_ubatching
    ):
        raise ValueError("compact prefill requires TP1 and one worker stream")
    workspace = getattr(config, "_compact_prefill_workspace", None)
    if workspace is None:
        before = torch.accelerator.memory_allocated(device)
        workspace = CompactPrefillWorkspace(device)
        workspace.allocation_before_profile = {
            "allocated_before": before,
            "allocated_after": torch.accelerator.memory_allocated(device),
        }
        config._compact_prefill_workspace = workspace
    return workspace
