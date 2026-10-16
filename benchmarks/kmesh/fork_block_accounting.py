# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure exact-prefix fork accounting through the real KVCacheManager.

Each child extends its parent's token sequence with an independent tail.
Drive allocator metadata without a model, scheduler, tensors, or GPU. Treat
successful allocation as completed prefill, as in prefix-caching unit tests.
Compare cached and uncached allocation with a tree oracle, and measure lost
ancestor hits after unrelated traffic evicts inactive cached blocks.

Written blocks count simulated prefill work over the entire run, not peak
residency. Peak usage is pool size minus free-queue length, including the
reserved null block. Byte counts project Qwen2.5-0.5B geometry (24 layers,
2 KV heads, head size 64, BF16); they are not measured HBM allocations.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from typing import Any

import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request

BLOCK_SIZE = 16
# Qwen2.5-0.5B: K+V, 24 layers, 2 KV heads, head size 64, BF16.
BYTES_PER_TOKEN = 2 * 24 * 2 * 64 * 2
BYTES_PER_BLOCK = BYTES_PER_TOKEN * BLOCK_SIZE
VOCAB_LOW, VOCAB_HIGH = 4, 50000


def make_manager(num_blocks: int, enable_caching: bool) -> KVCacheManager:
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=BLOCK_SIZE,
                    num_kv_heads=2,
                    head_size=64,
                    dtype=torch.bfloat16,
                ),
            )
        ],
    )
    return KVCacheManager(
        config,
        max_model_len=1 << 20,
        scheduler_block_size=BLOCK_SIZE,
        hash_block_size=BLOCK_SIZE,
        enable_caching=enable_caching,
    )


def make_request(request_id: str, tokens: list[int]) -> Request:
    params = SamplingParams(max_tokens=1)
    params.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id=request_id,
        prompt_token_ids=tokens,
        sampling_params=params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )


@dataclass
class TreeNode:
    """One tree member: full token sequence plus its fork point."""

    label: str
    tokens: list[int]
    divergence: int


def build_fork_tree(
    seed: int,
    ancestor_len: int,
    branching: int,
    tail_len: int,
    levels: int,
) -> list[TreeNode]:
    """Build a breadth-first tree of exact ancestors and independent tails."""
    rng = random.Random(seed)

    def fresh(n: int) -> list[int]:
        return [rng.randrange(VOCAB_LOW, VOCAB_HIGH) for _ in range(n)]

    root = TreeNode("d0", fresh(ancestor_len), 0)
    nodes = [root]
    frontier = [root]
    for level in range(1, levels + 1):
        next_frontier = []
        for parent in frontier:
            for child_idx in range(branching):
                child = TreeNode(
                    f"{parent.label}.b{child_idx}",
                    parent.tokens + fresh(tail_len),
                    len(parent.tokens),
                )
                nodes.append(child)
                next_frontier.append(child)
        frontier = next_frontier
    return nodes


@dataclass
class Accounting:
    """Tree-wide accounting over one manager lifetime."""

    manager: KVCacheManager
    pool_blocks: int
    seen_hashes: set = field(default_factory=set)
    rows: list[dict] = field(default_factory=list)
    peak_used_blocks: int = 0
    live: dict[str, Request] = field(default_factory=dict)

    def _sample_usage(self) -> None:
        used = self.pool_blocks - (
            self.manager.block_pool.free_block_queue.num_free_blocks
        )
        self.peak_used_blocks = max(self.peak_used_blocks, used)

    def admit(self, node: TreeNode) -> bool:
        req = make_request(node.label, node.tokens)
        computed, num_hit, _ = self.manager.get_computed_blocks(req)
        num_new = req.num_tokens - num_hit
        blocks = self.manager.allocate_slots(req, num_new, num_hit, computed)
        n_tokens = req.num_tokens
        n_full = n_tokens // BLOCK_SIZE
        hit_blocks = num_hit // BLOCK_SIZE
        row: dict[str, Any] = {
            "label": node.label,
            "tokens": n_tokens,
            "divergence": node.divergence,
            "hit_tokens": num_hit,
            "allocated": blocks is not None,
        }
        if blocks is None:
            # Admission failure: nothing allocated, nothing to free.
            row.update(
                written_full_blocks=0,
                duplicated_full_blocks=0,
                new_full_blocks=0,
                partial_blocks=0,
                duplicated_boundary_tokens=0,
            )
            self.rows.append(row)
            return False
        self.live[node.label] = req
        row["block_ids"] = self.manager.get_block_ids(node.label)[0]
        written_idx = range(hit_blocks, n_full)
        written_hashes = [req.block_hashes[i] for i in written_idx]
        dup_full = sum(1 for h in written_hashes if h in self.seen_hashes)
        self.seen_hashes.update(written_hashes)
        partial = 1 if n_tokens % BLOCK_SIZE else 0
        # Ancestor-content tokens this child recomputes: everything below
        # the fork point that the prefix cache failed to serve.
        dup_boundary = max(0, node.divergence - num_hit)
        row.update(
            written_full_blocks=len(written_hashes),
            duplicated_full_blocks=dup_full,
            new_full_blocks=len(written_hashes) - dup_full,
            partial_blocks=partial,
            duplicated_boundary_tokens=dup_boundary,
        )
        self.rows.append(row)
        self._sample_usage()
        return True

    def finish(self, label: str) -> None:
        req = self.live.pop(label)
        self.manager.free(req)
        self._sample_usage()

    def finish_all(self) -> None:
        for label in list(self.live):
            self.finish(label)

    def churn(self, total_tokens: int, request_tokens: int, seed: int) -> int:
        """Run unrelated single-shot requests totalling ``total_tokens``
        through the pool, each admitted and freed immediately.  Returns
        the number of churn requests that failed admission."""
        rng = random.Random(seed)
        failures = 0
        idx = 0
        remaining = total_tokens
        while remaining > 0:
            n = min(request_tokens, remaining)
            tokens = [rng.randrange(VOCAB_LOW, VOCAB_HIGH) for _ in range(n)]
            req = make_request(f"churn-{seed}-{idx}", tokens)
            computed, num_hit, _ = self.manager.get_computed_blocks(req)
            blocks = self.manager.allocate_slots(
                req, req.num_tokens - num_hit, num_hit, computed
            )
            if blocks is None:
                failures += 1
            else:
                self._sample_usage()
                self.manager.free(req)
            remaining -= n
            idx += 1
        return failures


def oracles(nodes: list[TreeNode]) -> dict:
    """Content and fork oracles for the tree, in blocks."""
    content_blocks = 0
    fork_clone_blocks = 0
    for node in nodes:
        div_floor = node.divergence // BLOCK_SIZE
        n_blocks = (len(node.tokens) + BLOCK_SIZE - 1) // BLOCK_SIZE
        # Blocks whose content does not exist in any ancestor: everything
        # from the first block touching post-divergence content onward.
        content_blocks += n_blocks - div_floor
        if node.divergence % BLOCK_SIZE:
            # The fork oracle clones the parent's partial boundary block
            # instead of recomputing it; content-wise it is not new, so
            # subtract it from content and count it as a clone.
            content_blocks -= 1
            fork_clone_blocks += 1
    logical_blocks = sum(
        (len(node.tokens) + BLOCK_SIZE - 1) // BLOCK_SIZE for node in nodes
    )
    logical_tokens = sum(len(node.tokens) for node in nodes)
    return {
        "logical_tokens": logical_tokens,
        "logical_blocks": logical_blocks,
        "content_oracle_blocks": content_blocks,
        "fork_oracle_blocks": content_blocks + fork_clone_blocks,
    }


def summarize(acct: Accounting, nodes: list[TreeNode]) -> dict:
    oracle = oracles(nodes)
    written = sum(r["written_full_blocks"] + r["partial_blocks"] for r in acct.rows)
    duplicated = sum(r["duplicated_full_blocks"] for r in acct.rows)
    hit_tokens = sum(r["hit_tokens"] for r in acct.rows)
    dup_boundary = sum(r["duplicated_boundary_tokens"] for r in acct.rows)
    failures = sum(1 for r in acct.rows if not r["allocated"])
    summary = {
        **oracle,
        "written_blocks": written,
        "duplicated_full_blocks": duplicated,
        "duplicated_boundary_tokens": dup_boundary,
        "hit_tokens": hit_tokens,
        "admission_failures": failures,
        "peak_used_blocks": acct.peak_used_blocks,
        "written_bytes": written * BYTES_PER_BLOCK,
        "content_oracle_bytes": oracle["content_oracle_blocks"] * BYTES_PER_BLOCK,
        "logical_bytes": oracle["logical_tokens"] * BYTES_PER_TOKEN,
    }
    if oracle["fork_oracle_blocks"]:
        summary["written_over_fork_oracle"] = round(
            written / oracle["fork_oracle_blocks"], 4
        )
    return summary


def scenario_sequential(
    nodes: list[TreeNode], pool_blocks: int, enable_caching: bool
) -> dict:
    """Admit and free each node before exploring the next node."""
    acct = Accounting(make_manager(pool_blocks, enable_caching), pool_blocks)
    for node in nodes:
        if acct.admit(node):
            acct.finish(node.label)
    return {"rows": acct.rows, "summary": summarize(acct, nodes)}


def scenario_concurrent(
    nodes: list[TreeNode], pool_blocks: int, enable_caching: bool
) -> dict:
    """Keep a level's siblings live together, freeing parents beforehand."""
    acct = Accounting(make_manager(pool_blocks, enable_caching), pool_blocks)
    by_level: dict[int, list[TreeNode]] = {}
    for node in nodes:
        by_level.setdefault(node.label.count("."), []).append(node)
    for level in sorted(by_level):
        admitted = [n for n in by_level[level] if acct.admit(n)]
        if level + 1 not in by_level:
            break
        for node in admitted:
            acct.finish(node.label)
    acct.finish_all()
    return {"rows": acct.rows, "summary": summarize(acct, nodes)}


def scenario_churn(
    nodes: list[TreeNode],
    pool_blocks: int,
    churn_fraction: float,
    seed: int,
) -> dict:
    """Parent completes, unrelated traffic churns the pool, then the
    children arrive together.  Measures how much of the common prefix
    survives the free-queue window."""
    acct = Accounting(make_manager(pool_blocks, True), pool_blocks)
    root, children = nodes[0], nodes[1:]
    if acct.admit(root):
        acct.finish(root.label)
    churn_tokens = int(pool_blocks * BLOCK_SIZE * churn_fraction)
    churn_failures = acct.churn(churn_tokens, 2048, seed + 7)
    for node in children:
        acct.admit(node)
    acct.finish_all()
    result = {"rows": acct.rows, "summary": summarize(acct, nodes)}
    result["summary"]["churn_fraction"] = churn_fraction
    result["summary"]["churn_tokens"] = churn_tokens
    result["summary"]["churn_admission_failures"] = churn_failures
    hits = [r["hit_tokens"] for r in result["rows"][1:]]
    result["summary"]["child_hit_tokens"] = hits
    result["summary"]["ancestor_survival"] = (
        round(sum(hits) / (len(hits) * len(root.tokens)), 4) if hits else 0.0
    )
    return result


def scenario_pool_pressure(nodes: list[TreeNode], enable_caching: bool) -> dict:
    """Pool sized between the shared-tree footprint and the dense
    footprint: 1.25x the fork oracle.  Concurrent admission then shows
    whether block sharing lets the whole tree fit where dense copies
    cannot."""
    oracle = oracles(nodes)
    pool_blocks = max(64, int(oracle["fork_oracle_blocks"] * 1.25))
    result = scenario_concurrent(nodes, pool_blocks, enable_caching)
    result["summary"]["pool_blocks"] = pool_blocks
    return result


def run_matrix(seed: int) -> dict:
    ample_pool = 8192
    trees = {
        "a4096-b5-t512": build_fork_tree(seed, 4096, 5, 512, 1),
        "a2048-b8-t256": build_fork_tree(seed + 1, 2048, 8, 256, 1),
        "a4096-b3-t256-l2": build_fork_tree(seed + 2, 4096, 3, 256, 2),
        # Exercise fork points inside a block as well as aligned forks.
        "a4101-b5-t509": build_fork_tree(seed + 3, 4101, 5, 509, 1),
        "a4101-b3-t253-l2": build_fork_tree(seed + 4, 4101, 3, 253, 2),
    }
    results: dict[str, Any] = {
        "block_size": BLOCK_SIZE,
        "bytes_per_token": BYTES_PER_TOKEN,
        "seed": seed,
        "scenarios": {},
    }
    for name, nodes in trees.items():
        entry: dict[str, Any] = {
            "members": len(nodes),
            "sequential_cached": scenario_sequential(nodes, ample_pool, True),
            "concurrent_cached": scenario_concurrent(nodes, ample_pool, True),
            "concurrent_uncached": scenario_concurrent(nodes, ample_pool, False),
            "pool_pressure_cached": scenario_pool_pressure(nodes, True),
            "pool_pressure_uncached": scenario_pool_pressure(nodes, False),
        }
        results["scenarios"][name] = entry

    # Loss-window sweep on the primary tree with a deliberately small
    # pool so churn can actually evict the ancestor.
    nodes = trees["a4096-b5-t512"]
    window_pool = 1024
    sweep = []
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5):
        out = scenario_churn(nodes, window_pool, fraction, seed)
        sweep.append(out["summary"])
    results["loss_window_sweep"] = {
        "tree": "a4096-b5-t512",
        "pool_blocks": window_pool,
        "points": sweep,
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    init_none_hash(sha256)
    if args.smoke:
        nodes = build_fork_tree(args.seed, 128, 3, 48, 1)
        results = {
            "smoke": scenario_concurrent(nodes, 256, True),
            "smoke_uncached": scenario_concurrent(nodes, 256, False),
        }
    else:
        results = run_matrix(args.seed)

    import vllm

    results["vllm_version"] = vllm.__version__
    results["python"] = sys.version.split()[0]
    results["schema_version"] = 1
    results["torch_version"] = torch.__version__
    results["measurement"] = "cpu-allocator-metadata"
    with open(args.output, "x") as sink:
        json.dump(results, sink, indent=2, sort_keys=True)
        sink.write("\n")
    print(json.dumps({"wrote": args.output}, indent=2))


if __name__ == "__main__":
    main()
