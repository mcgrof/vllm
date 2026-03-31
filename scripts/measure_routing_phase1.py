#!/usr/bin/env python3
"""Phase 1 routing measurement — A/B comparison.

Runs vLLM with cartridge connector in two modes:
A) routing_mode=full (baseline dense attention)
B) routing_mode=cartridge_prior with K sweep

Measures TTFT and compares output tokens between modes.

Usage:
    python measure_routing_phase1.py \
        --model marin-8b-instruct \
        --cartridge-dir /path/to/cartridge_dir \
        --port 8100
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time

import requests


def wait_for_server(port, timeout=120):
    """Wait for vLLM server to become healthy."""
    for i in range(timeout):
        try:
            r = requests.get(f"http://localhost:{port}/health", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def run_requests(port, prompt, num_runs=5, max_tokens=32):
    """Send requests and measure wall time."""
    url = f"http://localhost:{port}/v1/chat/completions"
    results = []
    for i in range(num_runs):
        t0 = time.monotonic()
        resp = requests.post(url, json={
            "model": "default",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
        }, timeout=120)
        t1 = time.monotonic()
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = data.get("usage", {})
        results.append({
            "run": i,
            "wall_time_ms": (t1 - t0) * 1000,
            "output_text": content[:200],
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        })
        print(f"  Run {i}: {(t1-t0)*1000:.1f}ms, "
              f"tokens={usage.get(completion_tokens, 0)}")
    return results


def start_server(model, port, kv_config, log_path):
    """Start vLLM server and return process."""
    cmd = [
        "vllm", "serve", model,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.40",
        "--kv-transfer-config", json.dumps(kv_config),
    ]
    log_f = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
    return proc, log_f


def stop_server(proc, log_f):
    """Stop vLLM server gracefully."""
    try:
        os.kill(proc.pid, signal.SIGTERM)
        proc.wait(timeout=15)
    except Exception:
        proc.kill()
    log_f.close()
    time.sleep(2)


def run_variant(label, model, cart_dir, port, results_dir,
                routing_mode="full", routing_k=4):
    """Run one experiment variant."""
    print(f"\n--- {label}: mode={routing_mode}, K={routing_k} ---")

    extra_config = {
        "cartridge_path": os.path.join(cart_dir, "cartridge.pt"),
        "prefix_token_ids_path": os.path.join(cart_dir, "prefix_token_ids.json"),
    }
    if routing_mode != "full":
        extra_config["routing_prior_path"] = os.path.join(cart_dir, "routing_prior.pt")
        extra_config["routing_mode"] = routing_mode
        extra_config["routing_K"] = routing_k

    kv_config = {
        "kv_connector": "CartridgeConnector",
        "kv_connector_module_path": (
            "vllm.distributed.kv_transfer.kv_connector.v1.cartridge_connector"
        ),
        "kv_connector_extra_config": extra_config,
        "kv_role": "kv_both",
    }

    log_path = os.path.join(results_dir, f"server_{label}.log")
    proc, log_f = start_server(model, port, kv_config, log_path)

    try:
        print(f"Waiting for server (pid={proc.pid})...")
        if not wait_for_server(port):
            print(f"ERROR: Server not ready. Check {log_path}")
            return None

        prompt = ("What is the main contribution of this document? "
                  "Provide a concise summary.")
        results = run_requests(port, prompt)

        result_path = os.path.join(results_dir, f"results_{label}.json")
        with open(result_path, "w") as f:
            json.dump({"label": label, "routing_mode": routing_mode,
                       "routing_K": routing_k, "runs": results}, f, indent=2)
        print(f"Results saved to {result_path}")

        avg_ms = sum(r["wall_time_ms"] for r in results) / len(results)
        return {"label": label, "avg_ms": avg_ms, "runs": results}

    finally:
        stop_server(proc, log_f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--cartridge-dir", required=True)
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()

    ts = time.strftime("%Y%m%dT%H%M%SZ")
    results_dir = f"/data/knlp-key-results/paper-router/vllm-cartridge-routing-phase1-{ts}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results directory: {results_dir}")

    # Validate artifacts
    for name in ["cartridge.pt", "prefix_token_ids.json", "routing_prior.pt"]:
        path = os.path.join(args.cartridge_dir, name)
        if not os.path.exists(path):
            print(f"ERROR: Missing {path}")
            sys.exit(1)

    summary = []

    # Baseline: full dense attention
    r = run_variant("baseline", args.model, args.cartridge_dir,
                    args.port, results_dir, "full", 0)
    if r:
        summary.append(r)

    # K sweep with cartridge_prior
    for K in [1, 2, 4, 8]:
        r = run_variant(f"cartridge_K{K}", args.model, args.cartridge_dir,
                        args.port, results_dir, "cartridge_prior", K)
        if r:
            summary.append(r)

    # Print summary
    print("\n=== Results Summary ===")
    print(f"Results: {results_dir}")
    for s in summary:
        print(f"  {s[label]}: avg {s[avg_ms]:.1f}ms")

    # Save summary
    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone. Inspect server logs for per-layer routing metrics:")
    print(f"  grep Routed attention {results_dir}/server_*.log")


if __name__ == "__main__":
    main()
