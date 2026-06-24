#!/usr/bin/env python3
"""A100 fused INT4 serving test ladder.

Runs a strict, bounded inference-serving test ladder on A100:
  Stage 0: Baseline controls (FP16 direct, FP16 server, INT4 direct, fused-off)
  Stage 1: Minimal reproducible server-path failure
  Stage 2: Direct LLM(...) vs API-server comparison
  Stage 3: Post-prefill cache state comparison (if divergence found)

All outputs saved to a durable artifact directory.

Usage:
    python benchmarks/a100_serving_ladder.py --stage 0
    python benchmarks/a100_serving_ladder.py --stage 1
    python benchmarks/a100_serving_ladder.py --stage 2
    python benchmarks/a100_serving_ladder.py --stage all
"""

import argparse
import gc
import json
import os
import signal
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone

# ---- Config ----
MODEL = "Qwen/Qwen2.5-7B-Instruct"
MAX_MODEL_LEN = 2048
GPU_MEM_UTIL = 0.80
MAX_TOKENS = 30  # Must be enough to catch continuation garbage
TEMPERATURE = 0.0
SERVER_PORT = 8192
SERVER_TIMEOUT = 300
MSL = int(os.environ.get("VLLM_FUSED_INT4_MIN_SEQ_LEN", "48"))

# Diverse prompt set spanning below/at/above MSL boundary
PROMPTS = [
    # Very short (well below MSL)
    {"label": "short_1", "text": "What is 2 + 2?"},
    {"label": "short_2", "text": "Hi"},
    {"label": "short_3", "text": "The capital of France is"},
    {"label": "short_4", "text": "List the first 5 prime numbers."},
    # Medium (near MSL boundary)
    {"label": "medium_1", "text": "Explain quantum computing in one sentence."},
    {"label": "medium_2", "text": "Write a haiku about the ocean."},
    {"label": "medium_3", "text": "Tell me a short joke about programmers."},
    # Long (above MSL=48 — should enter fused decode)
    {"label": "long_1", "text": (
        "The quick brown fox jumps over the lazy dog. "
        "The quick brown fox jumps over the lazy dog. "
        "The quick brown fox jumps over the lazy dog. "
        "Summarize the above.")},
    {"label": "long_2", "text": (
        "In the year 2024, artificial intelligence continued to advance "
        "at a rapid pace. Large language models became more capable and "
        "efficient, enabling new applications across many industries. "
        "What is the main topic of this passage?")},
    {"label": "long_3", "text": (
        "Write a Python function that calculates the factorial of a number. "
        "The function should handle edge cases like negative numbers and zero. "
        "Include type hints and a docstring. Show the implementation.")},
]


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)


def save_artifact(artifact_dir, name, data):
    path = os.path.join(artifact_dir, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    log(f"  Artifact saved: {path}")
    return path


# ---- Direct LLM(...) path ----

def run_direct(prompts, kv_dtype, label, max_num_seqs=1):
    """Run via direct LLM() API."""
    from vllm import LLM, SamplingParams

    log(f"[direct/{label}] Loading LLM kv_cache_dtype={kv_dtype} "
        f"max_num_seqs={max_num_seqs}...")
    llm = LLM(
        model=MODEL,
        dtype="float16",
        kv_cache_dtype=kv_dtype,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM_UTIL,
        disable_log_stats=True,
        enforce_eager=True,
        max_num_seqs=max_num_seqs,
    )
    sp = SamplingParams(max_tokens=MAX_TOKENS, temperature=TEMPERATURE)
    results = []
    for p in prompts:
        out = llm.generate([p["text"]], sp)[0].outputs[0]
        ids = list(out.token_ids)
        txt = out.text
        log(f"  [{label}] {p['label']:12s} -> ids={ids[:8]}... text={txt[:60]!r}")
        results.append({
            "label": p["label"],
            "prompt": p["text"][:80],
            "token_ids": ids,
            "text": txt,
            "num_tokens": len(ids),
        })
    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    time.sleep(3)
    return results


# ---- API Server path ----

def wait_for_server(port, timeout):
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = urllib.request.urlopen(
                urllib.request.Request(url, method="GET"), timeout=5)
            if resp.status == 200:
                return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2)
    return False


def server_generate(port, prompt_text, max_tokens):
    url = f"http://localhost:{port}/v1/completions"
    payload = json.dumps({
        "model": MODEL,
        "prompt": prompt_text,
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=120)
    body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]


def start_server(kv_dtype, label, extra_env=None):
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--dtype", "float16",
        "--kv-cache-dtype", kv_dtype,
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", str(GPU_MEM_UTIL),
        "--disable-log-stats",
        "--enforce-eager",
        "--max-num-seqs", "1",
        "--port", str(SERVER_PORT),
    ]
    log_path = f"/tmp/vllm_server_{label}.log"
    log_file = open(log_path, "w")
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    log(f"[server/{label}] Starting: {' '.join(cmd)}")
    log(f"[server/{label}] Log: {log_path}")
    proc = subprocess.Popen(
        cmd, stdout=log_file, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid, env=env,
    )
    return proc, log_file, log_path


def stop_server(proc, log_file):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
    log_file.close()
    time.sleep(5)


def run_server(prompts, kv_dtype, label, extra_env=None):
    proc, log_file, log_path = start_server(kv_dtype, label, extra_env)
    try:
        log(f"[server/{label}] Waiting for health...")
        if not wait_for_server(SERVER_PORT, SERVER_TIMEOUT):
            log(f"[server/{label}] TIMEOUT!")
            with open(log_path) as f:
                lines = f.readlines()
                for line in lines[-30:]:
                    log(f"  SERVER LOG: {line.rstrip()}")
            return None, log_path

        log(f"[server/{label}] Ready. Running prompts...")
        results = []
        for p in prompts:
            try:
                choice = server_generate(SERVER_PORT, p["text"], MAX_TOKENS)
                txt = choice["text"]
                log(f"  [{label}] {p['label']:12s} -> text={txt[:60]!r}")
                results.append({
                    "label": p["label"],
                    "prompt": p["text"][:80],
                    "text": txt,
                    "num_tokens": len(txt.split()),
                })
            except Exception as e:
                log(f"  [{label}] {p['label']:12s} -> ERROR: {e}")
                results.append({
                    "label": p["label"],
                    "prompt": p["text"][:80],
                    "text": f"ERROR: {e}",
                    "error": True,
                })
        return results, log_path
    finally:
        log(f"[server/{label}] Stopping...")
        stop_server(proc, log_file)


# ---- Comparison ----

def compare(baseline, test, path_label):
    """Strict comparison. No prefix-only passes."""
    comparisons = []
    for b, t in zip(baseline, test):
        if t.get("error"):
            comparisons.append({
                "label": b["label"],
                "path": path_label,
                "verdict": "ERROR",
                "detail": t.get("text"),
            })
            continue

        # Token ID comparison (direct path)
        if "token_ids" in b and "token_ids" in t:
            b_ids = b["token_ids"]
            t_ids = t["token_ids"]
            exact = b_ids == t_ids
            prefix_len = 0
            for i in range(min(len(b_ids), len(t_ids))):
                if b_ids[i] == t_ids[i]:
                    prefix_len += 1
                else:
                    break
            is_prefix_only = prefix_len > 0 and not exact

            # Garbage detection: triple repeat
            garbage = []
            if len(t_ids) >= 3:
                for k in range(2, len(t_ids)):
                    if t_ids[k] == t_ids[k-1] == t_ids[k-2]:
                        garbage.append(f"triple_repeat@{k}(id={t_ids[k]})")
                        break

            verdict = "PASS" if exact else "FAIL"
            reason = None
            if not exact:
                reason = (f"prefix_only({prefix_len}/{len(b_ids)})"
                          if is_prefix_only else "token_mismatch")
                if garbage:
                    reason += f" {garbage}"

            comparisons.append({
                "label": b["label"],
                "path": path_label,
                "verdict": verdict,
                "reason": reason,
                "exact_match": exact,
                "prefix_match_len": prefix_len,
                "baseline_ids": b_ids,
                "test_ids": t_ids,
                "baseline_text": b["text"],
                "test_text": t["text"],
                "garbage": garbage,
            })
        else:
            # Text-only comparison (server path)
            exact = b["text"] == t["text"]
            # Check for garbage in test text
            garbage = []
            words = t["text"].split()
            if len(words) >= 3:
                for k in range(2, len(words)):
                    if words[k] == words[k-1] == words[k-2]:
                        garbage.append(f"word_triple@{k}({words[k]!r})")
                        break

            verdict = "PASS" if exact else "FAIL"
            reason = None
            if not exact:
                reason = "text_mismatch"
                if garbage:
                    reason += f" {garbage}"
                # Check for obvious corruption patterns
                if "pérdida" in t["text"] or len(set(words)) < len(words) * 0.3:
                    reason += " CORRUPTION_DETECTED"

            comparisons.append({
                "label": b["label"],
                "path": path_label,
                "verdict": verdict,
                "reason": reason,
                "exact_match": exact,
                "baseline_text": b["text"],
                "test_text": t["text"],
                "garbage": garbage,
            })

    return comparisons


def summarize(comparisons, label):
    total = len(comparisons)
    passes = sum(1 for c in comparisons if c["verdict"] == "PASS")
    fails = sum(1 for c in comparisons if c["verdict"] == "FAIL")
    errors = sum(1 for c in comparisons if c["verdict"] == "ERROR")
    log(f"\n{'='*60}")
    log(f"SUMMARY [{label}]: {passes}/{total} PASS, {fails} FAIL, {errors} ERROR")
    for c in comparisons:
        if c["verdict"] != "PASS":
            log(f"  FAIL: {c['label']} - {c.get('reason', 'unknown')}")
            if "baseline_text" in c:
                log(f"    baseline: {c['baseline_text'][:80]!r}")
                log(f"    test:     {c['test_text'][:80]!r}")
    log(f"{'='*60}\n")
    return {"total": total, "passes": passes, "fails": fails,
            "errors": errors, "all_pass": fails == 0 and errors == 0}


# ---- Stages ----

def stage0(artifact_dir):
    """Stage 0: Baseline controls."""
    log("=" * 60)
    log("STAGE 0: Baseline controls")
    log("=" * 60)

    results = {}

    # 0a: FP16 direct LLM(...)
    log("\n--- 0a: FP16 direct LLM(...) ---")
    fp16_direct = run_direct(PROMPTS, "auto", "fp16_direct")
    results["fp16_direct"] = fp16_direct
    save_artifact(artifact_dir, "stage0/fp16_direct.json", fp16_direct)

    # 0b: INT4 fused direct LLM(...)
    log("\n--- 0b: INT4 fused direct LLM(...) ---")
    int4_direct = run_direct(PROMPTS, "int4_fused", "int4_fused_direct")
    results["int4_fused_direct"] = int4_direct
    save_artifact(artifact_dir, "stage0/int4_fused_direct.json", int4_direct)

    # Compare direct paths
    direct_comp = compare(fp16_direct, int4_direct, "direct_fused")
    save_artifact(artifact_dir, "stage0/direct_comparison.json", direct_comp)
    direct_summary = summarize(direct_comp, "DIRECT fused vs FP16")
    results["direct_comparison_summary"] = direct_summary

    # 0c: FP16 server
    log("\n--- 0c: FP16 API server ---")
    fp16_server, fp16_server_log = run_server(PROMPTS, "auto", "fp16_server")
    if fp16_server is not None:
        results["fp16_server"] = fp16_server
        save_artifact(artifact_dir, "stage0/fp16_server.json", fp16_server)
    else:
        results["fp16_server"] = "STARTUP_FAIL"
        log("FP16 server failed to start!")

    # 0d: INT4 fused server
    log("\n--- 0d: INT4 fused API server ---")
    int4_server, int4_server_log = run_server(
        PROMPTS, "int4_fused", "int4_fused_server")
    if int4_server is not None:
        results["int4_fused_server"] = int4_server
        save_artifact(artifact_dir, "stage0/int4_fused_server.json", int4_server)
    else:
        results["int4_fused_server"] = "STARTUP_FAIL"
        log("INT4 fused server failed to start!")

    # Compare server paths
    if fp16_server and int4_server and not isinstance(fp16_server, str):
        server_comp = compare(fp16_server, int4_server, "server_fused")
        save_artifact(artifact_dir, "stage0/server_comparison.json", server_comp)
        server_summary = summarize(server_comp, "SERVER fused vs FP16")
        results["server_comparison_summary"] = server_summary
    else:
        results["server_comparison_summary"] = {"error": "server_startup_fail"}

    # Copy server logs
    for lbl in ["fp16_server", "int4_fused_server"]:
        src = f"/tmp/vllm_server_{lbl}.log"
        if os.path.exists(src):
            import shutil
            dst = os.path.join(artifact_dir, "stage0", f"{lbl}.log")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)

    # Stage 0 summary
    stage_summary = {
        "stage": 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "msl": MSL,
        "model": MODEL,
        "direct_fused_summary": results.get("direct_comparison_summary"),
        "server_fused_summary": results.get("server_comparison_summary"),
    }
    save_artifact(artifact_dir, "stage0/summary.json", stage_summary)
    return results


def stage1(artifact_dir, stage0_results=None):
    """Stage 1: Minimal reproducible server-path failure case.

    Start with B=1, max_num_seqs=1, prefix caching OFF, eager mode.
    Test boundary prompts around MSL.
    """
    log("=" * 60)
    log("STAGE 1: Minimal reproducible server-path case")
    log("=" * 60)

    # Use a targeted prompt set for boundary testing
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    filler = ("The quick brown fox jumps over the lazy dog. " * 20)
    boundary_lengths = [1, MSL - 1, MSL, MSL + 1, MSL + 8, 96]

    boundary_prompts = []
    for tl in boundary_lengths:
        tokens = tokenizer.encode(filler)[:tl]
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        actual_len = len(tokenizer.encode(text))
        boundary_prompts.append({
            "label": f"boundary_len_{tl}",
            "text": text,
            "target_len": tl,
            "actual_len": actual_len,
        })

    # FP16 baseline (direct)
    log("\n--- 1a: FP16 direct baseline for boundary prompts ---")
    fp16_baseline = run_direct(boundary_prompts, "auto", "fp16_boundary")
    save_artifact(artifact_dir, "stage1/fp16_boundary_direct.json", fp16_baseline)

    # INT4 fused direct
    log("\n--- 1b: INT4 fused direct for boundary prompts ---")
    int4_direct = run_direct(boundary_prompts, "int4_fused", "int4_boundary")
    save_artifact(artifact_dir, "stage1/int4_boundary_direct.json", int4_direct)

    # Compare
    direct_comp = compare(fp16_baseline, int4_direct, "boundary_direct")
    save_artifact(artifact_dir, "stage1/boundary_direct_comparison.json", direct_comp)
    direct_summary = summarize(direct_comp, "BOUNDARY DIRECT fused vs FP16")

    # INT4 fused server
    log("\n--- 1c: INT4 fused server for boundary prompts ---")
    int4_server, _ = run_server(boundary_prompts, "int4_fused", "int4_boundary_srv")
    if int4_server is not None:
        save_artifact(artifact_dir, "stage1/int4_boundary_server.json", int4_server)
        server_comp = compare(fp16_baseline, int4_server, "boundary_server")
        save_artifact(artifact_dir, "stage1/boundary_server_comparison.json",
                      server_comp)
        server_summary = summarize(server_comp, "BOUNDARY SERVER fused vs FP16")
    else:
        server_summary = {"error": "server_startup_fail"}

    # Quick B=2 sanity
    log("\n--- 1d: B=2 direct sanity check ---")
    int4_b2 = run_direct(boundary_prompts[:3], "int4_fused",
                          "int4_b2", max_num_seqs=2)
    save_artifact(artifact_dir, "stage1/int4_b2_direct.json", int4_b2)
    b2_comp = compare(fp16_baseline[:3], int4_b2, "b2_direct")
    b2_summary = summarize(b2_comp, "B=2 DIRECT fused vs FP16")

    # Copy server logs
    for lbl in ["int4_boundary_srv"]:
        src = f"/tmp/vllm_server_{lbl}.log"
        if os.path.exists(src):
            import shutil
            dst = os.path.join(artifact_dir, "stage1", f"{lbl}.log")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)

    summary = {
        "stage": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "msl": MSL,
        "boundary_lengths": boundary_lengths,
        "direct_boundary_summary": direct_summary,
        "server_boundary_summary": server_summary,
        "b2_summary": b2_summary,
    }
    save_artifact(artifact_dir, "stage1/summary.json", summary)
    return summary


def stage2(artifact_dir):
    """Stage 2: Direct LLM(...) vs API-server comparison.

    Focus on the long prompts that should trigger fused decode.
    Compare token-level output between direct and server paths.
    """
    log("=" * 60)
    log("STAGE 2: Direct LLM(...) vs API-server comparison")
    log("=" * 60)

    # Use prompts above MSL that should activate fused path
    long_prompts = [p for p in PROMPTS if p["label"].startswith("long")]

    # FP16 baseline direct
    log("\n--- 2a: FP16 direct baseline ---")
    fp16_direct = run_direct(long_prompts, "auto", "fp16_long_direct")
    save_artifact(artifact_dir, "stage2/fp16_long_direct.json", fp16_direct)

    # INT4 fused direct
    log("\n--- 2b: INT4 fused direct ---")
    int4_direct = run_direct(long_prompts, "int4_fused", "int4_long_direct")
    save_artifact(artifact_dir, "stage2/int4_long_direct.json", int4_direct)

    # FP16 server
    log("\n--- 2c: FP16 server baseline ---")
    fp16_server, _ = run_server(long_prompts, "auto", "fp16_long_srv")
    if fp16_server:
        save_artifact(artifact_dir, "stage2/fp16_long_server.json", fp16_server)

    # INT4 fused server
    log("\n--- 2d: INT4 fused server ---")
    int4_server, _ = run_server(long_prompts, "int4_fused", "int4_long_srv")
    if int4_server:
        save_artifact(artifact_dir, "stage2/int4_long_server.json", int4_server)

    # Comparisons
    results = {}

    # Direct fused vs FP16
    direct_comp = compare(fp16_direct, int4_direct, "long_direct_fused")
    save_artifact(artifact_dir, "stage2/direct_comparison.json", direct_comp)
    results["direct_fused_vs_fp16"] = summarize(
        direct_comp, "LONG DIRECT fused vs FP16")

    # Server fused vs FP16
    if fp16_server and int4_server:
        server_comp = compare(fp16_server, int4_server, "long_server_fused")
        save_artifact(artifact_dir, "stage2/server_comparison.json", server_comp)
        results["server_fused_vs_fp16"] = summarize(
            server_comp, "LONG SERVER fused vs FP16")

    # Direct vs Server (same config) — are they producing same output?
    if int4_server:
        # Text-only comparison since server doesn't return token IDs easily
        cross_comp = []
        for d, s in zip(int4_direct, int4_server):
            match = d["text"] == s["text"]
            cross_comp.append({
                "label": d["label"],
                "path": "direct_vs_server_int4",
                "verdict": "PASS" if match else "FAIL",
                "reason": None if match else "text_mismatch",
                "direct_text": d["text"],
                "server_text": s["text"],
            })
        save_artifact(artifact_dir, "stage2/direct_vs_server.json", cross_comp)
        results["direct_vs_server_int4"] = summarize(
            cross_comp, "INT4 DIRECT vs SERVER")

    # Fused-off control: disable fused by setting MSL very high
    log("\n--- 2e: Fused-OFF control (MSL=99999 to force FP16 fallback) ---")
    fused_off_env = {"VLLM_FUSED_INT4_MIN_SEQ_LEN": "99999"}
    fused_off_server, _ = run_server(
        long_prompts, "int4_fused", "fused_off_ctrl", extra_env=fused_off_env)
    if fused_off_server:
        save_artifact(artifact_dir, "stage2/fused_off_server.json", fused_off_server)
        if fp16_server:
            fused_off_comp = compare(
                fp16_server, fused_off_server, "fused_off_control")
            save_artifact(artifact_dir, "stage2/fused_off_comparison.json",
                          fused_off_comp)
            results["fused_off_vs_fp16"] = summarize(
                fused_off_comp, "FUSED-OFF CONTROL vs FP16")

    summary = {
        "stage": 2,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "msl": MSL,
        "results": results,
    }
    save_artifact(artifact_dir, "stage2/summary.json", summary)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="A100 fused INT4 serving test ladder")
    parser.add_argument("--stage", default="all",
                        help="Stage to run: 0, 1, 2, or all")
    parser.add_argument("--artifact-dir", default=None,
                        help="Artifact directory (auto-generated if not set)")
    args = parser.parse_args()

    if args.artifact_dir:
        artifact_dir = args.artifact_dir
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        artifact_dir = f"/tmp/a100-serving-ladder-{ts}"
    os.makedirs(artifact_dir, exist_ok=True)

    log(f"Artifact dir: {artifact_dir}")
    log(f"Model: {MODEL}")
    log(f"MSL: {MSL}")
    log(f"MAX_TOKENS: {MAX_TOKENS}")

    # Save environment info
    env_info = {
        "model": MODEL,
        "msl": MSL,
        "max_tokens": MAX_TOKENS,
        "max_model_len": MAX_MODEL_LEN,
        "gpu_mem_util": GPU_MEM_UTIL,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "env": {
            k: v for k, v in os.environ.items()
            if k.startswith("VLLM_") or k in ("CUDA_VISIBLE_DEVICES",)
        },
    }
    save_artifact(artifact_dir, "env_info.json", env_info)

    stages = args.stage
    if stages == "all":
        stages = ["0", "1", "2"]
    else:
        stages = [stages]

    results = {}
    for s in stages:
        try:
            if s == "0":
                results["stage0"] = stage0(artifact_dir)
            elif s == "1":
                results["stage1"] = stage1(artifact_dir)
            elif s == "2":
                results["stage2"] = stage2(artifact_dir)
            else:
                log(f"Unknown stage: {s}")
        except Exception as e:
            log(f"STAGE {s} FAILED: {e}")
            log(traceback.format_exc())
            results[f"stage{s}"] = {"error": str(e),
                                     "traceback": traceback.format_exc()}

    # Final summary
    save_artifact(artifact_dir, "final_summary.json", results)
    log(f"\nAll artifacts in: {artifact_dir}")


if __name__ == "__main__":
    main()
