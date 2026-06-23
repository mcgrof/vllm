# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SPF ↔ Scheduler integration bridge.

This module encapsulates all the wiring between the SPFController and
the vLLM v1 Scheduler.  The Scheduler calls a small number of methods
here; the module translates between the Scheduler's internal types
(BlockHash bytes, KVCacheBlock objects) and the SPFController's string-
based prefix_hash / session_id interface.

Design goals:
  - Scheduler diff stays minimal (< 30 lines).
  - All SPF-specific complexity lives here, not in scheduler.py.
  - Enable/disable is a no-op when controller is None.
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vllm.v1.core.spf import get_spf_controller
from vllm.v1.core.spf.controller import PrefetchHint, SPFController

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.request import Request

logger = logging.getLogger("vllm.spf.integration")


def _block_hash_to_str(bh: bytes) -> str:
    """Convert a BlockHash (bytes) to the hex string SPFController uses."""
    return bh.hex()


def _derive_session_id(request: "Request") -> str:
    """Derive a session identifier from a request.

    Strategy: hash the first block's worth of prompt tokens.  Requests
    that share the same system-prompt prefix will cluster into the same
    session, which is exactly the pattern SPF is designed to exploit.

    Falls back to request_id if prompt is too short for even one block.
    """
    prompt = getattr(request, "prompt_token_ids", None)
    if prompt and len(prompt) >= 16:
        # Use first 128 tokens (or all if shorter) as fingerprint.
        prefix_tokens = prompt[: min(128, len(prompt))]
        raw = ",".join(str(t) for t in prefix_tokens).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]
    return request.request_id


# Number of trailing prompt tokens to feed into the query_hash rule.
# This is the configurable "query region extraction" point that the
# SPF design doc describes — KRI-Q providers use the resulting hash to
# key their per-(cartridge, query) priors.  256 tokens covers a typical
# user question with room for a few framing tokens; tune via
# VLLM_SPF_QUERY_TAIL_TOKENS if a deployment needs a different region.
_DEFAULT_QUERY_TAIL_TOKENS = 256


def _derive_query_hash(
    request: "Request",
    *,
    tail_tokens: int = _DEFAULT_QUERY_TAIL_TOKENS,
) -> str | None:
    """Hash the trailing query region of a request prompt.

    Returns ``None`` if the prompt is too short to meaningfully
    distinguish a query from the system prompt.  In that case the
    request will be a manifest miss for KRI-Q providers (correct
    behavior — there is no query to route by) and a manifest hit for
    KRI-G providers (also correct — KRI-G ignores queries).

    The extraction rule is intentionally simple and configurable.  Per
    the SPF integration design (see docs/design/spf_kri_integration.md),
    the rule lives at the request-ingestion boundary so that:

    1. SPF's scoring loop never sees query tokens — sessions still
       cluster on shared prefix prompts.
    2. Providers stay opaque to prompt format — they receive an
       already-hashed token.
    3. Future KRI-x variants that want a different query region can
       swap this function (or override via config) without touching
       SPF's controller.
    """
    prompt = getattr(request, "prompt_token_ids", None)
    if not prompt:
        return None
    # Need enough total length that the tail is distinct from the
    # prefix used by _derive_session_id.  If the prompt is shorter than
    # roughly two prefix windows, treat it as having no extractable
    # query region.
    if len(prompt) < 2 * 128:
        return None
    tail = prompt[-min(tail_tokens, len(prompt)):]
    raw = ",".join(str(t) for t in tail).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


@dataclass
class SPFStepResult:
    """Result of one SPF integration step."""
    hints_issued: int = 0
    blocks_touched: int = 0
    step_time_ms: float = 0.0


@dataclass
class SPFIntegrationMetrics:
    """Accumulated integration-level metrics (separate from controller metrics)."""
    total_observe_calls: int = 0
    total_step_calls: int = 0
    total_hints_issued: int = 0
    total_blocks_touched: int = 0
    total_ttft_samples: int = 0
    ttft_sum_ms: float = 0.0
    ttft_values: list[float] = field(default_factory=list)
    _last_flush_step: int = 0

    def record_ttft(self, ttft_ms: float) -> None:
        self.total_ttft_samples += 1
        self.ttft_sum_ms += ttft_ms
        self.ttft_values.append(ttft_ms)

    def flush_summary(self, step: int) -> dict:
        """Return a summary dict and reset interval counters."""
        if not self.ttft_values:
            p50 = p95 = mean = 0.0
        else:
            sorted_v = sorted(self.ttft_values)
            n = len(sorted_v)
            p50 = sorted_v[n // 2]
            p95 = sorted_v[int(n * 0.95)]
            mean = sum(sorted_v) / n

        summary = {
            "step": step,
            "observe_calls": self.total_observe_calls,
            "step_calls": self.total_step_calls,
            "hints_issued": self.total_hints_issued,
            "blocks_touched": self.total_blocks_touched,
            "ttft_samples": len(self.ttft_values),
            "ttft_p50_ms": round(p50, 3),
            "ttft_p95_ms": round(p95, 3),
            "ttft_mean_ms": round(mean, 3),
        }
        self.ttft_values = []
        self._last_flush_step = step
        return summary


class SPFSchedulerBridge:
    """Bridge between Scheduler and SPFController.

    Instantiated once per Scheduler.  If SPF is disabled (env var),
    the controller is None and all methods are no-ops.
    """

    def __init__(self):
        self.controller: SPFController | None = get_spf_controller()
        self.metrics = SPFIntegrationMetrics()
        self._step_count = 0
        self._request_arrival_times: dict[str, float] = {}

        if self.controller is not None:
            logger.info("SPF scheduler bridge initialized (SPF ENABLED)")
        else:
            logger.info("SPF scheduler bridge initialized (SPF DISABLED)")

    @property
    def enabled(self) -> bool:
        return self.controller is not None

    def observe_request(self, request: "Request") -> None:
        """Called when a new request is added to the scheduler."""
        if self.controller is None:
            return

        self.metrics.total_observe_calls += 1
        self._request_arrival_times[request.request_id] = time.monotonic()

        session_id = _derive_session_id(request)

        # Use the first block hash as the prefix identifier.
        # If no block hashes yet, use a hash of the prompt tokens.
        if request.block_hashes:
            prefix_hash = _block_hash_to_str(request.block_hashes[0])
        elif request.prompt_token_ids:
            raw = ",".join(str(t) for t in request.prompt_token_ids[:64]).encode()
            prefix_hash = hashlib.sha256(raw).hexdigest()[:16]
        else:
            return

        # Compute the query_hash here, at the ingestion boundary.  This
        # is the architectural seam: query-region extraction lives at
        # the edge so SPF's scoring loop and any future scoring policy
        # can keep using prefix_hash exclusively.  KRI-Q providers will
        # consume this; KRI-G providers will ignore it.
        query_hash = _derive_query_hash(request)

        self.controller.observe_request(session_id, prefix_hash, query_hash)

    def step(self, kv_cache_manager: "KVCacheManager") -> SPFStepResult:
        """Called once per scheduling round after scheduling decisions.

        Generates prefetch hints and touches the corresponding blocks
        in the GPU prefix cache to prevent their eviction.
        """
        if self.controller is None:
            return SPFStepResult()

        self._step_count += 1
        self.metrics.total_step_calls += 1
        t0 = time.monotonic()

        block_pool = kv_cache_manager.block_pool

        # Gather free block count.
        free_gpu_blocks = block_pool.get_num_free_blocks()

        # Gather resident prefix set (cached block hashes).
        # Only rebuild every 10 steps to avoid O(N) iteration overhead
        # on large caches (300K+ blocks on 80GB H100).
        if not hasattr(self, '_cached_resident_prefixes'):
            self._cached_resident_prefixes: set[str] = set()
            self._resident_rebuild_step = 0

        if self._step_count - self._resident_rebuild_step >= 10:
            self._cached_resident_prefixes.clear()
            for bh_with_gid in block_pool.cached_block_hash_to_block._cache:
                # BlockHashWithGroupId is bytes: BlockHash + 4-byte group_id.
                if isinstance(bh_with_gid, bytes) and len(bh_with_gid) > 4:
                    block_hash_bytes = bh_with_gid[:-4]
                    self._cached_resident_prefixes.add(
                        _block_hash_to_str(block_hash_bytes)
                    )
            self._resident_rebuild_step = self._step_count

        resident_prefixes = self._cached_resident_prefixes

        # Get LRU victim (tail of free queue) if cache is under pressure.
        gpu_lru_victim: str | None = None
        if free_gpu_blocks < block_pool.num_gpu_blocks * 0.1:
            # Cache is >90% full — check eviction candidate.
            try:
                tail = block_pool.free_block_queue.tail
                if tail is not None and hasattr(tail, 'block_hash') and tail.block_hash is not None:
                    gpu_lru_victim = _block_hash_to_str(tail.block_hash)
            except (AttributeError, IndexError):
                pass  # free queue impl may vary

        # Run the controller.
        hints = self.controller.step(
            free_gpu_blocks=free_gpu_blocks,
            resident_prefixes=resident_prefixes if resident_prefixes else None,
            gpu_lru_victim=gpu_lru_victim,
        )

        # Process hints: touch matching blocks in the prefix cache.
        blocks_touched = 0
        for hint in hints:
            self._touch_hint_blocks(hint, block_pool, kv_cache_manager)
            blocks_touched += hint.num_blocks

        self.metrics.total_hints_issued += len(hints)
        self.metrics.total_blocks_touched += blocks_touched

        elapsed_ms = (time.monotonic() - t0) * 1000.0

        # Periodic logging.
        if self._step_count % 50 == 0:
            summary = self.metrics.flush_summary(self._step_count)
            logger.info("SPF integration step=%d summary=%s", self._step_count, summary)

        return SPFStepResult(
            hints_issued=len(hints),
            blocks_touched=blocks_touched,
            step_time_ms=elapsed_ms,
        )

    def _touch_hint_blocks(
        self,
        hint: PrefetchHint,
        block_pool: "BlockPool",
        kv_cache_manager: "KVCacheManager",
    ) -> None:
        """Touch blocks in the prefix cache that match a prefetch hint.

        This promotes the block in LRU ordering, preventing eviction of
        blocks that SPF predicts will be needed soon.

        When the hint carries a :class:`BlockManifest`, the bridge logs
        that fact and reports the K-bounded promotion in metrics.  The
        actual "translate manifest block indices into physical GPU
        block IDs" step happens on the connector side (cartridge
        connector / RAG-KRI loader / etc.) — SPF's job is to attach
        the manifest to the hint and pass it through unchanged so the
        downstream consumer can honor it.
        """
        # Convert hint prefix_hash back to bytes for lookup.
        try:
            target_hash = bytes.fromhex(hint.prefix_hash)
        except ValueError:
            return

        # Look up across all KV cache groups.
        num_groups = len(kv_cache_manager.kv_cache_config.kv_cache_groups)
        group_ids = list(range(num_groups))
        cached_blocks = block_pool.get_cached_block(target_hash, group_ids)

        if cached_blocks is not None:
            block_pool.touch(cached_blocks)
            if hint.manifest is not None:
                m = hint.manifest
                logger.debug(
                    "SPF touch (KRI %s): promoted %d cached blocks "
                    "for prefix %s, manifest K=%d/N=%d "
                    "(savings=%d, %.1f%%) session=%s",
                    m.prior_type,
                    len(cached_blocks),
                    hint.prefix_hash[:8],
                    m.K,
                    m.total_blocks,
                    m.savings_blocks,
                    m.savings_fraction * 100.0,
                    hint.session_id[:8],
                )
            else:
                logger.debug(
                    "SPF touch: promoted %d blocks for prefix %s (session %s)",
                    len(cached_blocks),
                    hint.prefix_hash[:8],
                    hint.session_id[:8],
                )

    def record_first_token(self, request_id: str) -> None:
        """Called when the first token is generated for a request.

        Records TTFT for SPF analysis.
        """
        if self.controller is None:
            return

        arrival = self._request_arrival_times.pop(request_id, None)
        if arrival is not None:
            ttft_ms = (time.monotonic() - arrival) * 1000.0
            self.metrics.record_ttft(ttft_ms)

    def request_finished(self, request_id: str) -> None:
        """Cleanup when a request finishes."""
        self._request_arrival_times.pop(request_id, None)

    def get_integration_stats(self) -> dict:
        """Return current integration stats for external consumption."""
        return {
            "spf_enabled": self.enabled,
            "observe_calls": self.metrics.total_observe_calls,
            "step_calls": self.metrics.total_step_calls,
            "hints_issued": self.metrics.total_hints_issued,
            "blocks_touched": self.metrics.total_blocks_touched,
            "ttft_samples": self.metrics.total_ttft_samples,
            "ttft_mean_ms": round(
                self.metrics.ttft_sum_ms / max(1, self.metrics.total_ttft_samples), 3
            ),
        }
