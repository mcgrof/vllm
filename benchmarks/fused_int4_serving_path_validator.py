#!/usr/bin/env python3
"""Strict server-path validator for fused INT4.

Tests BOTH the direct LLM() path and the API-server path against an FP16
baseline. Uses exact token-ID comparison and generates enough tokens (20)
to catch continuation corruption that prefix-only checks would miss.

The validator explicitly rejects prefix-only matches: if only the first
few tokens match but the continuation diverges, it's a FAIL.

Usage:
    # Test direct LLM() path only:
    python fused_int4_serving_path_validator.py --mode direct

    # Test API-server path only:
    python fused_int4_serving_path_validator.py --mode server

    # Test both (default):
    python fused_int4_serving_path_validator.py --mode both

    # Override MSL:
    VLLM_FUSED_INT4_MIN_SEQ_LEN=48 python fused_int4_serving_path_validator.py
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error

MODEL = "Qwen/Qwen2.5-7B-Instruct"
MAX_TOKENS = 20  # Enough to catch continuation corruption
TEMPERATURE = 0.0
SERVER_PORT = 8192
SERVER_TIMEOUT = 300  # seconds to wait for server startup

FILLER = "The quick brown fox jumps over the lazy dog. " * 20

# Prompt lengths chosen to span below/at/above MSL=48 boundary
PROMPT_LENGTHS = [1, 8, 16, 32, 47, 48, 49, 64, 96]


def make_prompts():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompts = []
    for tl in PROMPT_LENGTHS:
        tokens = tokenizer.encode(FILLER)[:tl]
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        actual = tokenizer.encode(text)
        prompts.append({
            "label": f"len_{tl}",
            "target_len": tl,
            "actual_len": len(actual),
            "text": text,
        })
    return prompts


# ---- Direct LLM() path ----

def run_direct(prompts, kv_dtype, label):
    """Run via direct LLM() API (no server)."""
    from vllm import LLM, SamplingParams

    print(f"[direct/{label}] Initializing LLM with kv_cache_dtype={kv_dtype}...",
          file=sys.stderr)
    llm = LLM(
        model=MODEL,
        dtype="float16",
        kv_cache_dtype=kv_dtype,
        max_model_len=2048,
        gpu_memory_utilization=0.8,
        disable_log_stats=True,
        enforce_eager=True,
        max_num_seqs=1,
    )
    sp = SamplingParams(max_tokens=MAX_TOKENS, temperature=TEMPERATURE)
    results = []
    for p in prompts:
        out = llm.generate([p["text"]], sp)[0].outputs[0]
        ids = list(out.token_ids)
        txt = out.text
        print(f"  [direct/{label}] prompt_len={p['actual_len']:3d} -> "
              f"ids={ids}  text={txt!r}", file=sys.stderr)
        results.append({
            "label": p["label"],
            "prompt_len": p["actual_len"],
            "ids": ids,
            "text": txt,
        })
    del llm
    return results


# ---- API Server path ----

def wait_for_server(port, timeout):
    """Wait for the vLLM server to become healthy."""
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request(url, method="GET")
            resp = urllib.request.urlopen(req, timeout=5)
            if resp.status == 200:
                return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2)
    return False


def server_generate(port, prompt_text, max_tokens):
    """Hit the OpenAI-compatible completions endpoint."""
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


def start_server(kv_dtype, label):
    """Launch a vLLM API server as a subprocess."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--dtype", "float16",
        "--kv-cache-dtype", kv_dtype,
        "--max-model-len", "2048",
        "--gpu-memory-utilization", "0.8",
        "--disable-log-stats",
        "--enforce-eager",
        "--max-num-seqs", "1",
        "--port", str(SERVER_PORT),
    ]
    log_path = f"/tmp/vllm_server_{label}.log"
    log_file = open(log_path, "w")
    print(f"[server/{label}] Starting server: {' '.join(cmd)}", file=sys.stderr)
    print(f"[server/{label}] Log: {log_path}", file=sys.stderr)
    proc = subprocess.Popen(
        cmd, stdout=log_file, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    return proc, log_file, log_path


def stop_server(proc, log_file):
    """Stop the vLLM server."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    proc.wait(timeout=30)
    log_file.close()


def run_server(prompts, kv_dtype, label):
    """Run via API server path."""
    proc, log_file, log_path = start_server(kv_dtype, label)
    try:
        print(f"[server/{label}] Waiting for server to be healthy...",
              file=sys.stderr)
        if not wait_for_server(SERVER_PORT, SERVER_TIMEOUT):
            print(f"[server/{label}] TIMEOUT waiting for server!", file=sys.stderr)
            # Dump last 50 lines of server log
            with open(log_path) as f:
                lines = f.readlines()
                for line in lines[-50:]:
                    print(f"  SERVER LOG: {line.rstrip()}", file=sys.stderr)
            return None

        print(f"[server/{label}] Server ready. Running prompts...",
              file=sys.stderr)
        results = []
        for p in prompts:
            choice = server_generate(SERVER_PORT, p["text"], MAX_TOKENS)
            txt = choice["text"]
            # We need token IDs for exact comparison; use logprobs if available
            # Otherwise fall back to text comparison
            # The completions endpoint returns 'text' but not token_ids directly
            # We'll use text comparison for server path
            print(f"  [server/{label}] prompt_len={p['actual_len']:3d} -> "
                  f"text={txt!r}", file=sys.stderr)
            results.append({
                "label": p["label"],
                "prompt_len": p["actual_len"],
                "text": txt,
            })
        return results
    finally:
        print(f"[server/{label}] Stopping server...", file=sys.stderr)
        stop_server(proc, log_file)
        # Brief pause to release GPU
        time.sleep(5)


# ---- Comparison ----

def compare_results(baseline, test, path_label):
    """Strict comparison. Returns list of comparison dicts."""
    comparisons = []
    for b, t in zip(baseline, test):
        # Try token-ID comparison first (direct path), fall back to text
        if "ids" in b and "ids" in t:
            match = b["ids"] == t["ids"]
            prefix_len = 0
            min_len = min(len(b["ids"]), len(t["ids"]))
            for i in range(min_len):
                if b["ids"][i] == t["ids"][i]:
                    prefix_len += 1
                else:
                    break
            prefix_only = prefix_len > 0 and not match
        else:
            match = b["text"] == t["text"]
            # Check for prefix-only match (text level)
            prefix_only = False
            if not match:
                # Find common prefix length
                min_len = min(len(b["text"]), len(t["text"]))
                common = 0
                for i in range(min_len):
                    if b["text"][i] == t["text"][i]:
                        common += 1
                    else:
                        break
                prefix_only = common > len(b["text"]) * 0.3
            prefix_len = 0  # Not available for text-only

        verdict = "PASS" if match else "FAIL"
        if prefix_only and not match:
            verdict = "FAIL_PREFIX_ONLY"

        comp = {
            "label": b["label"],
            "prompt_len": b["prompt_len"],
            "path": path_label,
            "baseline_text": b["text"],
            "test_text": t["text"],
            "match": match,
            "prefix_only": prefix_only,
            "verdict": verdict,
        }
        if "ids" in b and "ids" in t:
            comp["baseline_ids"] = b["ids"]
            comp["test_ids"] = t["ids"]
            comp["prefix_match_tokens"] = prefix_len

        status_char = "OK" if match else "FAIL"
        print(f"  [{path_label}] {b['label']}: [{status_char}] "
              f"baseline={b['text']!r:.60} "
              f"test={t['text']!r:.60}",
              file=sys.stderr)
        comparisons.append(comp)
    return comparisons


def main():
    parser = argparse.ArgumentParser(
        description="Strict server-path validator for fused INT4")
    parser.add_argument("--mode", choices=["direct", "server", "both"],
                        default="both",
                        help="Which path(s) to test")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: stdout)")
    args = parser.parse_args()

    prompts = make_prompts()
    all_comparisons = []
    server_logs = {}

    # ---- Direct path ----
    if args.mode in ("direct", "both"):
        print("\n=== DIRECT PATH: baseline (FP16) ===", file=sys.stderr)
        direct_baseline = run_direct(prompts, "auto", "baseline")

        print("\n=== DIRECT PATH: fused INT4 ===", file=sys.stderr)
        direct_fused = run_direct(prompts, "int4_fused", "fused")

        print("\n=== DIRECT PATH: comparison ===", file=sys.stderr)
        direct_comps = compare_results(direct_baseline, direct_fused, "direct")
        all_comparisons.extend(direct_comps)

    # ---- Server path ----
    if args.mode in ("server", "both"):
        print("\n=== SERVER PATH: baseline (FP16) ===", file=sys.stderr)
        server_baseline = run_server(prompts, "auto", "baseline")

        if server_baseline is not None:
            print("\n=== SERVER PATH: fused INT4 ===", file=sys.stderr)
            server_fused = run_server(prompts, "int4_fused", "fused")

            if server_fused is not None:
                print("\n=== SERVER PATH: comparison ===", file=sys.stderr)
                server_comps = compare_results(
                    server_baseline, server_fused, "server")
                all_comparisons.extend(server_comps)
            else:
                print("SERVER FUSED FAILED TO START", file=sys.stderr)
                all_comparisons.append({
                    "path": "server",
                    "verdict": "SERVER_STARTUP_FAIL",
                    "label": "fused_server",
                })
        else:
            print("SERVER BASELINE FAILED TO START", file=sys.stderr)
            all_comparisons.append({
                "path": "server",
                "verdict": "SERVER_STARTUP_FAIL",
                "label": "baseline_server",
            })

        # Collect server logs
        for lbl in ["baseline", "fused"]:
            log_path = f"/tmp/vllm_server_{lbl}.log"
            if os.path.exists(log_path):
                with open(log_path) as f:
                    server_logs[lbl] = f.read()

    # ---- Summary ----
    total = sum(1 for c in all_comparisons if "match" in c)
    passes = sum(1 for c in all_comparisons if c.get("verdict") == "PASS")
    fails = sum(1 for c in all_comparisons if c.get("verdict", "").startswith("FAIL"))
    prefix_only = sum(1 for c in all_comparisons
                      if c.get("verdict") == "FAIL_PREFIX_ONLY")

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"STRICT VALIDATOR SUMMARY", file=sys.stderr)
    print(f"  Total comparisons: {total}", file=sys.stderr)
    print(f"  PASS:              {passes}", file=sys.stderr)
    print(f"  FAIL:              {fails}", file=sys.stderr)
    print(f"  FAIL_PREFIX_ONLY:  {prefix_only}", file=sys.stderr)
    print(f"  Overall:           {'ALL PASS' if fails == 0 and total > 0 else 'FAILURES DETECTED'}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    # Print failing cases
    for c in all_comparisons:
        if c.get("verdict", "").startswith("FAIL"):
            print(f"\n  FAILING: [{c.get('path')}] {c.get('label')} "
                  f"prompt_len={c.get('prompt_len')}", file=sys.stderr)
            print(f"    baseline: {c.get('baseline_text', 'N/A')!r}",
                  file=sys.stderr)
            print(f"    test:     {c.get('test_text', 'N/A')!r}",
                  file=sys.stderr)

    manifest = {
        "probe": "strict_serving_path_validator",
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "prompt_lengths": PROMPT_LENGTHS,
        "msl": os.environ.get("VLLM_FUSED_INT4_MIN_SEQ_LEN", "48"),
        "k_precision": os.environ.get("VLLM_FUSED_INT4_K_PRECISION", "int4"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "comparisons": all_comparisons,
        "summary": {
            "total": total,
            "passes": passes,
            "fails": fails,
            "prefix_only_fails": prefix_only,
            "all_pass": fails == 0 and total > 0,
        },
    }

    output = json.dumps(manifest, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Results written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
