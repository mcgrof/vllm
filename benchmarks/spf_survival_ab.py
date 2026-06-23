#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SPF Q4 survival-gate A/B benchmark.

Five arms across one model under cache pressure:

    1. lru             — vLLM default prefix-caching LRU (SPF disabled)
    2. session_ttl     — session-TTL + pinning (SessionTTLScorer)
    3. frequency       — frequency-only admission (FrequencyAdmissionScorer)
    4. spf             — ExpectedUtilityScorer (current SPF)
    5. oracle          — Belady-style reuse-distance oracle

Workload: multi-session synthetic conversations with a shared system
prefix. Each session has N turns; sessions are interleaved to force
cache eviction. The shared prefix is the part SPF is supposed to keep
warm; per-session content is the noise that should fall out of cache.

Measures per request:
  - TTFT (first-token latency)
  - num_cached_tokens   (from RequestMetrics)
  - num_total_tokens
  - prefill_tokens_recomputed = total - cached

Aggregates per arm × seed:
  - p50 / p95 / p99 TTFT
  - sum prefill_tokens_recomputed
  - throughput (req/s)
  - wasted promoted bytes (proxy: hints_issued - hints_used) * block_bytes

Hard survival assertion (between SPF and session_ttl baseline):
  - p95 TTFT improvement >= 5%  OR  prefill tokens reduction >= 10%
  - throughput regression < 1%
  - wasted promoted bytes < 10% of total promoted

Usage:
    python benchmarks/spf_survival_ab.py \\
        --model meta-llama/Llama-3.2-3B-Instruct \\
        --num-sessions 32 \\
        --turns-per-session 4 \\
        --shared-prefix-tokens 4096 \\
        --gpu-memory-utilization 0.5 \\
        --seeds 42,43,44 \\
        --output spf_survival_results.json
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


ARMS = ["lru", "session_ttl", "frequency", "spf", "oracle"]


@dataclass
class ArmConfig:
    name: str
    spf_enabled: bool
    spf_scorer: str | None = None
    extra_env: dict[str, str] = field(default_factory=dict)


def arm_configs() -> dict[str, ArmConfig]:
    return {
        "lru": ArmConfig(name="lru", spf_enabled=False),
        "session_ttl": ArmConfig(
            name="session_ttl",
            spf_enabled=True,
            spf_scorer="session_ttl",
            extra_env={"VLLM_SPF_SESSION_TTL_STEPS": "8"},
        ),
        "frequency": ArmConfig(
            name="frequency",
            spf_enabled=True,
            spf_scorer="frequency",
        ),
        "spf": ArmConfig(
            name="spf",
            spf_enabled=True,
            spf_scorer="session_aware",
        ),
        "oracle": ArmConfig(
            name="oracle",
            spf_enabled=True,
            spf_scorer="oracle",
        ),
    }


# ---------------------------------------------------------------------------
# Workload generator
# ---------------------------------------------------------------------------


@dataclass
class WorkloadRequest:
    """A single LLM request in the benchmark trace."""

    step_idx: int
    session_id: str
    prefix_hash: str
    prompt_tokens: list[int]
    expected_output_tokens: int = 32


def _gen_shared_prefix(
    num_tokens: int, vocab_size: int = 32000, seed: int = 0
) -> list[int]:
    rng = random.Random(seed)
    return [rng.randint(100, vocab_size - 100) for _ in range(num_tokens)]


def _gen_session_query(
    session_idx: int,
    turn_idx: int,
    length: int = 128,
    vocab_size: int = 32000,
    seed: int = 0,
) -> list[int]:
    rng = random.Random(hash((seed, session_idx, turn_idx)) & 0xFFFFFFFF)
    return [rng.randint(100, vocab_size - 100) for _ in range(length)]


def _prefix_hash(tokens: list[int]) -> str:
    h = hashlib.sha256()
    h.update(b"".join(t.to_bytes(4, "little") for t in tokens))
    return h.hexdigest()[:16]


def build_workload(
    *,
    num_sessions: int,
    turns_per_session: int,
    shared_prefix_tokens: int,
    per_query_tokens: int,
    num_personas: int = 1,
    interleave: str = "round_robin",
    seed: int = 42,
) -> tuple[list[WorkloadRequest], list[list[int]], dict[str, list[int]]]:
    """Generate a multi-persona multi-session workload.

    Each of ``num_personas`` personas owns a distinct shared prefix.
    Sessions are partitioned across personas in a balanced fashion.
    A persona's shared prefix is the same across all turns of every
    session bound to it; per-session/per-turn content is the unique
    tail.

    Returns (requests, per_persona_shared_prefixes, oracle_reuse_map).
    """
    personas = [
        _gen_shared_prefix(shared_prefix_tokens,
                           seed=(seed * 17_001 + p))
        for p in range(num_personas)
    ]
    session_to_persona = [s % num_personas for s in range(num_sessions)]
    requests: list[WorkloadRequest] = []
    rng = random.Random(seed)
    # Round-robin: s0 t0, s1 t0, ..., sN t0, s0 t1, s1 t1, ...
    # Shuffled: random session, random turn (with caps)
    def _build(s: int, t: int) -> WorkloadRequest:
        persona_idx = session_to_persona[s]
        return _make_req(
            len(requests), s, t, personas[persona_idx],
            per_query_tokens, seed,
        )

    if interleave == "round_robin":
        for turn in range(turns_per_session):
            for s in range(num_sessions):
                requests.append(_build(s, turn))
    elif interleave == "shuffled":
        plan: list[tuple[int, int]] = [
            (s, t) for t in range(turns_per_session) for s in range(num_sessions)
        ]
        rng.shuffle(plan)
        for s, t in plan:
            requests.append(_build(s, t))
    else:
        raise ValueError(f"unknown interleave: {interleave}")

    # Build oracle reuse map: prefix_hash -> sorted list of step indices
    # where that prefix appears. Each persona's shared prefix appears in
    # every request bound to one of its sessions.
    oracle: dict[str, list[int]] = {}
    for req in requests:
        oracle.setdefault(req.prefix_hash, []).append(req.step_idx)
    return requests, personas, oracle


def _make_req(
    step_idx: int,
    session_idx: int,
    turn_idx: int,
    shared: list[int],
    per_query_tokens: int,
    seed: int,
) -> WorkloadRequest:
    query = _gen_session_query(session_idx, turn_idx, per_query_tokens, seed=seed)
    full = shared + query
    return WorkloadRequest(
        step_idx=step_idx,
        session_id=f"s{session_idx:04d}",
        prefix_hash=_prefix_hash(full[: len(shared)]),
        prompt_tokens=full,
    )


# ---------------------------------------------------------------------------
# Run one arm
# ---------------------------------------------------------------------------


@dataclass
class RequestResult:
    step_idx: int
    session_id: str
    ttft_ms: float
    num_cached_tokens: int
    num_total_tokens: int
    output_tokens: int
    wall_ms: float


@dataclass
class ArmResult:
    arm: str
    seed: int
    num_requests: int
    requests: list[RequestResult]
    p50_ttft_ms: float
    p95_ttft_ms: float
    p99_ttft_ms: float
    total_prefill_recomputed: int
    total_prefill_offered: int
    cache_hit_rate: float
    wall_seconds: float
    throughput_rps: float
    spf_hints_issued: int = 0
    spf_hints_used: int = 0


def run_arm(
    *,
    arm: ArmConfig,
    seed: int,
    model: str,
    workload: list[WorkloadRequest],
    oracle_path: Path | None,
    gpu_memory_utilization: float,
    max_model_len: int,
    output_tokens: int,
    enforce_eager: bool,
) -> ArmResult:
    """Spawn a vLLM offline LLM with the right env, run the workload."""
    # Configure env per arm. Inherit current env.
    env = os.environ.copy()
    env["VLLM_SPF_ENABLED"] = "1" if arm.spf_enabled else "0"
    if arm.spf_scorer:
        env["VLLM_SPF_SCORER"] = arm.spf_scorer
    for k, v in arm.extra_env.items():
        env[k] = v
    if arm.name == "oracle":
        assert oracle_path is not None
        env["VLLM_SPF_ORACLE_TRACE_PATH"] = str(oracle_path)

    # Import vllm INSIDE the function so child processes pick up
    # env vars correctly. NOTE: we run each arm in a subprocess to
    # avoid vLLM state leaking between arms (the engine doesn't
    # cleanly tear down inside one process).
    print(f"[{arm.name} seed={seed}] starting subprocess", flush=True)
    workload_path = Path(f"/tmp/spf_q4_workload_{arm.name}_{seed}.json")
    workload_path.write_text(
        json.dumps(
            [
                {
                    "step_idx": r.step_idx,
                    "session_id": r.session_id,
                    "prefix_hash": r.prefix_hash,
                    "prompt_tokens": r.prompt_tokens,
                    "expected_output_tokens": output_tokens,
                }
                for r in workload
            ]
        )
    )

    result_path = Path(f"/tmp/spf_q4_result_{arm.name}_{seed}.json")
    if result_path.exists():
        result_path.unlink()

    cmd = [
        sys.executable,
        "-m",
        "benchmarks.spf_survival_ab",
        "--worker",
        "--worker-arm",
        arm.name,
        "--worker-seed",
        str(seed),
        "--worker-model",
        model,
        "--worker-workload",
        str(workload_path),
        "--worker-output",
        str(result_path),
        "--worker-gpu-mem-util",
        str(gpu_memory_utilization),
        "--worker-max-model-len",
        str(max_model_len),
        "--worker-output-tokens",
        str(output_tokens),
    ]
    if enforce_eager:
        cmd.append("--worker-enforce-eager")
    proc = subprocess.run(cmd, env=env, check=False)
    if proc.returncode != 0 or not result_path.exists():
        raise RuntimeError(f"arm {arm.name} seed {seed} failed (rc={proc.returncode})")

    data = json.loads(result_path.read_text())
    return ArmResult(**data)


def _worker_main(args: argparse.Namespace) -> None:
    """Inner subprocess: load vLLM, run requests, write result JSON."""
    # Import here so SPF env vars are read by SPFConfig.from_env()
    # at LLM construction time.
    import numpy as np
    from vllm import LLM, SamplingParams

    arm = arm_configs()[args.worker_arm]
    seed = args.worker_seed

    raw_workload = json.loads(Path(args.worker_workload).read_text())
    workload = [
        WorkloadRequest(
            step_idx=r["step_idx"],
            session_id=r["session_id"],
            prefix_hash=r["prefix_hash"],
            prompt_tokens=r["prompt_tokens"],
            expected_output_tokens=r["expected_output_tokens"],
        )
        for r in raw_workload
    ]

    llm = LLM(
        model=args.worker_model,
        gpu_memory_utilization=args.worker_gpu_mem_util,
        max_model_len=args.worker_max_model_len,
        enable_prefix_caching=True,
        enforce_eager=args.worker_enforce_eager,
        seed=seed,
        # Avoid downloading bigger tokenizers than the model.
        trust_remote_code=False,
    )
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=args.worker_output_tokens,
        ignore_eos=True,
    )

    # vLLM offline API runs all prompts together when passed as a
    # list. For pressure-aware comparison we want them processed in
    # workload-order one at a time, with the cache state evolving
    # between requests. So we loop.
    results: list[RequestResult] = []
    t0 = time.perf_counter()
    for req in workload:
        req_start = time.perf_counter()
        outputs = llm.generate(
            [{"prompt_token_ids": req.prompt_tokens}],
            sampling_params=sp,
            use_tqdm=False,
        )
        wall_ms = (time.perf_counter() - req_start) * 1000.0
        out = outputs[0]
        # vLLM RequestOutput has metrics with first_token_time etc.
        ttft_ms = float("nan")
        cached = 0
        try:
            metrics = out.metrics
            if metrics is not None:
                if (
                    metrics.first_token_time is not None
                    and metrics.arrival_time is not None
                ):
                    ttft_ms = (metrics.first_token_time - metrics.arrival_time) * 1000.0
            cached = out.num_cached_tokens or 0
        except Exception:
            pass
        total = len(req.prompt_tokens)
        produced = len(out.outputs[0].token_ids) if out.outputs else 0
        results.append(
            RequestResult(
                step_idx=req.step_idx,
                session_id=req.session_id,
                ttft_ms=ttft_ms,
                num_cached_tokens=int(cached),
                num_total_tokens=int(total),
                output_tokens=int(produced),
                wall_ms=wall_ms,
            )
        )
    wall_s = time.perf_counter() - t0

    ttfts = [r.ttft_ms for r in results if r.ttft_ms == r.ttft_ms]  # filter NaN
    if not ttfts:
        ttfts = [r.wall_ms for r in results]

    ttfts_sorted = sorted(ttfts)

    def _pct(p: float) -> float:
        if not ttfts_sorted:
            return float("nan")
        idx = max(0, min(len(ttfts_sorted) - 1, int(p * len(ttfts_sorted))))
        return ttfts_sorted[idx]

    total_offered = sum(r.num_total_tokens for r in results)
    total_cached = sum(r.num_cached_tokens for r in results)
    total_recomputed = max(0, total_offered - total_cached)

    # Pull SPF metrics if available.
    hints_issued = 0
    hints_used = 0
    try:
        # Reach into the scheduler bridge if exposed. vLLM doesn't
        # expose this directly via LLM; we read from the SPFController
        # singleton via the package.
        from vllm.v1.core.spf import get_spf_controller

        ctrl = get_spf_controller()
        if ctrl is not None:
            snap = ctrl.metrics_snapshot
            hints_issued = int(snap.get("hints_issued", 0))
            hints_used = int(snap.get("hints_used", 0) + snap.get("prefetch_used", 0))
    except Exception:
        pass

    out_data = {
        "arm": arm.name,
        "seed": seed,
        "num_requests": len(results),
        "requests": [asdict(r) for r in results],
        "p50_ttft_ms": _pct(0.50),
        "p95_ttft_ms": _pct(0.95),
        "p99_ttft_ms": _pct(0.99),
        "total_prefill_recomputed": int(total_recomputed),
        "total_prefill_offered": int(total_offered),
        "cache_hit_rate": (float(total_cached) / float(max(total_offered, 1))),
        "wall_seconds": wall_s,
        "throughput_rps": len(results) / max(wall_s, 1e-9),
        "spf_hints_issued": hints_issued,
        "spf_hints_used": hints_used,
    }
    Path(args.worker_output).write_text(json.dumps(out_data, indent=2))
    print(f"[{arm.name} seed={seed}] wrote {args.worker_output}")


# ---------------------------------------------------------------------------
# Survival assertion + report
# ---------------------------------------------------------------------------


def survival_verdict(
    spf_results: list[ArmResult],
    session_ttl_results: list[ArmResult],
    lru_results: list[ArmResult],
) -> dict[str, Any]:
    """Apply the hard survival gate per Codex's Q4 spec."""
    import statistics as st

    def mean_p95(results: list[ArmResult]) -> float:
        return st.mean(r.p95_ttft_ms for r in results)

    def mean_recompute(results: list[ArmResult]) -> int:
        return int(st.mean(r.total_prefill_recomputed for r in results))

    def mean_throughput(results: list[ArmResult]) -> float:
        return st.mean(r.throughput_rps for r in results)

    spf_p95 = mean_p95(spf_results)
    ttl_p95 = mean_p95(session_ttl_results)
    spf_recomp = mean_recompute(spf_results)
    ttl_recomp = mean_recompute(session_ttl_results)
    spf_rps = mean_throughput(spf_results)
    lru_rps = mean_throughput(lru_results)

    ttft_pct = ((ttl_p95 - spf_p95) / max(ttl_p95, 1e-9)) * 100.0
    recomp_pct = ((ttl_recomp - spf_recomp) / max(ttl_recomp, 1)) * 100.0
    tput_pct = ((lru_rps - spf_rps) / max(lru_rps, 1e-9)) * 100.0

    issued = sum(r.spf_hints_issued for r in spf_results)
    used = sum(r.spf_hints_used for r in spf_results)
    waste_pct = 100.0 * (issued - used) / max(issued, 1)

    ttft_pass = ttft_pct >= 5.0
    recomp_pass = recomp_pct >= 10.0
    tput_pass = tput_pct < 1.0
    waste_pass = waste_pct < 10.0

    survives = (ttft_pass or recomp_pass) and tput_pass and waste_pass

    return {
        "survives": survives,
        "spf_p95_ttft_ms": spf_p95,
        "session_ttl_p95_ttft_ms": ttl_p95,
        "p95_ttft_improvement_pct": ttft_pct,
        "spf_prefill_recomputed": spf_recomp,
        "session_ttl_prefill_recomputed": ttl_recomp,
        "prefill_reduction_pct": recomp_pct,
        "lru_throughput_rps": lru_rps,
        "spf_throughput_rps": spf_rps,
        "throughput_regression_pct": tput_pct,
        "spf_wasted_pct": waste_pct,
        "ttft_pass": ttft_pass,
        "recomp_pass": recomp_pass,
        "tput_pass": tput_pass,
        "waste_pass": waste_pass,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument(
        "--arms", default=",".join(ARMS), help="Comma-separated arm names."
    )
    parser.add_argument("--seeds", default="42", help="Comma-separated seeds.")
    parser.add_argument("--num-sessions", type=int, default=32)
    parser.add_argument("--turns-per-session", type=int, default=4)
    parser.add_argument("--shared-prefix-tokens", type=int, default=4096)
    parser.add_argument("--num-personas", type=int, default=1,
                        help="Number of distinct shared-prefix personas; "
                             "sessions are partitioned across personas. "
                             "num_personas=1 reproduces the old single-"
                             "prefix workload (NOT pressured).")
    parser.add_argument("--per-query-tokens", type=int, default=128)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument(
        "--interleave", choices=["round_robin", "shuffled"], default="round_robin"
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--output", default="spf_survival_results.json")

    # Worker (inner subprocess) args
    parser.add_argument(
        "--worker", action="store_true", help="(internal) run as inner worker."
    )
    parser.add_argument("--worker-arm")
    parser.add_argument("--worker-seed", type=int)
    parser.add_argument("--worker-model")
    parser.add_argument("--worker-workload")
    parser.add_argument("--worker-output")
    parser.add_argument("--worker-gpu-mem-util", type=float)
    parser.add_argument("--worker-max-model-len", type=int)
    parser.add_argument("--worker-output-tokens", type=int)
    parser.add_argument("--worker-enforce-eager", action="store_true")

    args = parser.parse_args()

    if args.worker:
        _worker_main(args)
        return

    seeds = [int(s) for s in args.seeds.split(",")]
    arms = [a.strip() for a in args.arms.split(",")]
    bad = [a for a in arms if a not in ARMS]
    if bad:
        parser.error(f"unknown arm(s): {bad} (valid: {ARMS})")

    all_arm_cfgs = arm_configs()
    all_results: dict[str, list[ArmResult]] = {a: [] for a in arms}

    for seed in seeds:
        workload, shared, oracle = build_workload(
            num_sessions=args.num_sessions,
            turns_per_session=args.turns_per_session,
            shared_prefix_tokens=args.shared_prefix_tokens,
            per_query_tokens=args.per_query_tokens,
            interleave=args.interleave,
            seed=seed,
        )
        oracle_path = Path(f"/tmp/spf_q4_oracle_{seed}.json")
        oracle_path.write_text(json.dumps(oracle))
        print(
            f"workload seed={seed}: {len(workload)} requests, "
            f"shared_prefix={len(shared)} tokens, "
            f"unique_prefixes={len(oracle)}"
        )

        for arm_name in arms:
            arm_cfg = all_arm_cfgs[arm_name]
            try:
                r = run_arm(
                    arm=arm_cfg,
                    seed=seed,
                    model=args.model,
                    workload=workload,
                    oracle_path=oracle_path if arm_name == "oracle" else None,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    max_model_len=args.max_model_len,
                    output_tokens=args.output_tokens,
                    enforce_eager=args.enforce_eager,
                )
                all_results[arm_name].append(r)
            except Exception as e:
                print(f"[{arm_name} seed={seed}] FAILED: {e}", file=sys.stderr)

    # Summarize.
    summary: dict[str, Any] = {
        "model": args.model,
        "config": {
            "num_sessions": args.num_sessions,
            "turns_per_session": args.turns_per_session,
            "shared_prefix_tokens": args.shared_prefix_tokens,
            "per_query_tokens": args.per_query_tokens,
            "output_tokens": args.output_tokens,
            "interleave": args.interleave,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "seeds": seeds,
            "arms": arms,
        },
        "arm_summary": {},
        "per_arm_per_seed": {},
    }
    for arm_name in arms:
        rs = all_results[arm_name]
        if not rs:
            continue
        import statistics as st

        summary["arm_summary"][arm_name] = {
            "n_seeds": len(rs),
            "mean_p50_ttft_ms": st.mean(r.p50_ttft_ms for r in rs),
            "mean_p95_ttft_ms": st.mean(r.p95_ttft_ms for r in rs),
            "mean_p99_ttft_ms": st.mean(r.p99_ttft_ms for r in rs),
            "mean_prefill_recomputed": st.mean(r.total_prefill_recomputed for r in rs),
            "mean_prefill_offered": st.mean(r.total_prefill_offered for r in rs),
            "mean_cache_hit_rate": st.mean(r.cache_hit_rate for r in rs),
            "mean_throughput_rps": st.mean(r.throughput_rps for r in rs),
            "total_spf_hints_issued": sum(r.spf_hints_issued for r in rs),
            "total_spf_hints_used": sum(r.spf_hints_used for r in rs),
        }
        summary["per_arm_per_seed"][arm_name] = [
            {k: v for k, v in asdict(r).items() if k != "requests"} for r in rs
        ]

    # Survival verdict if both SPF and session_ttl ran.
    if (
        all_results.get("spf")
        and all_results.get("session_ttl")
        and all_results.get("lru")
    ):
        summary["survival_verdict"] = survival_verdict(
            spf_results=all_results["spf"],
            session_ttl_results=all_results["session_ttl"],
            lru_results=all_results["lru"],
        )

    Path(args.output).write_text(json.dumps(summary, indent=2))
    print(f"\nFinal summary written to {args.output}")
    if "survival_verdict" in summary:
        v = summary["survival_verdict"]
        print(f"\n=== SURVIVAL VERDICT ===")
        print(f"  survives: {v['survives']}")
        print(
            f"  p95 TTFT improvement: {v['p95_ttft_improvement_pct']:.2f}% "
            f"(threshold >=5%, pass={v['ttft_pass']})"
        )
        print(
            f"  prefill recomputed reduction: "
            f"{v['prefill_reduction_pct']:.2f}% "
            f"(threshold >=10%, pass={v['recomp_pass']})"
        )
        print(
            f"  throughput regression vs LRU: "
            f"{v['throughput_regression_pct']:.2f}% "
            f"(threshold <1%, pass={v['tput_pass']})"
        )
        print(
            f"  wasted hints: {v['spf_wasted_pct']:.2f}% "
            f"(threshold <10%, pass={v['waste_pass']})"
        )


if __name__ == "__main__":
    main()
