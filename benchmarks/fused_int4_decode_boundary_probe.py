#!/usr/bin/env python3
"""Decode-boundary probe for fused INT4 vs baseline correctness.

Phase 1 probe: isolates the first decode step by running baseline vs fused
with max_tokens=2 across prompt lengths near block boundaries.

Usage:
    # Start baseline server (in another terminal):
    python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-7B-Instruct \
        --dtype float16

    # Start fused server (in another terminal):
    python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-7B-Instruct \
        --dtype float16 \
        --kv-cache-dtype int4_fused \
        --port 8001

    # Run probe:
    python benchmarks/fused_int4_decode_boundary_probe.py \
        --baseline-url http://localhost:8000 \
        --fused-url http://localhost:8001

    # Or run against a single server to just collect outputs:
    python benchmarks/fused_int4_decode_boundary_probe.py \
        --baseline-url http://localhost:8000

Output:
    JSON results to stdout. Save with:
        python benchmarks/fused_int4_decode_boundary_probe.py ... > probe.json
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MAX_TOKENS = 2
PROMPT_LENGTHS = [1, 2, 15, 16, 17]

# Short filler words to build prompts of specific token counts.
# We use single-token ASCII words to get predictable tokenization.
FILLER_WORDS = [
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
    "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
]


def make_prompt(n_tokens: int) -> str:
    """Build a prompt that is approximately n_tokens long.

    Uses single-character words which are reliably single tokens
    in most tokenizers.
    """
    if n_tokens <= 0:
        return ""
    words = [FILLER_WORDS[i % len(FILLER_WORDS)] for i in range(n_tokens)]
    return " ".join(words)


def query_server(
    base_url: str,
    prompt: str,
    model: str,
    max_tokens: int,
    temperature: float = 0.0,
) -> dict:
    """Send a completions request and return parsed JSON response."""
    url = f"{base_url}/v1/completions"
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.URLError as e:
        return {"error": str(e)}


def extract_result(resp: dict) -> dict:
    """Pull token IDs and text from a completions response."""
    if "error" in resp:
        return {"error": resp["error"], "token_ids": [], "text": ""}
    try:
        choice = resp["choices"][0]
        text = choice["text"]
        # Token IDs may or may not be in the response depending on server
        # config. We include logprobs if available.
        logprobs = choice.get("logprobs")
        token_ids = []
        if logprobs and "tokens" in logprobs:
            token_ids = logprobs["tokens"]
        return {"text": text, "token_ids": token_ids}
    except (KeyError, IndexError) as e:
        return {"error": f"parse error: {e}", "token_ids": [], "text": ""}


def run_probe(
    baseline_url: str | None,
    fused_url: str | None,
    model: str,
    max_tokens: int,
    prompt_lengths: list[int],
) -> dict:
    """Run the full probe matrix and return results."""
    results = []

    for n_tok in prompt_lengths:
        prompt = make_prompt(n_tok)
        point = {
            "target_prompt_tokens": n_tok,
            "prompt": prompt,
            "max_tokens": max_tokens,
        }

        if baseline_url:
            resp = query_server(baseline_url, prompt, model, max_tokens)
            point["baseline"] = extract_result(resp)

        if fused_url:
            resp = query_server(fused_url, prompt, model, max_tokens)
            point["fused"] = extract_result(resp)

        # Compare if both present
        if baseline_url and fused_url:
            b = point.get("baseline", {})
            f = point.get("fused", {})
            point["text_match"] = b.get("text") == f.get("text")
            # If token IDs available, compare those too
            b_ids = b.get("token_ids", [])
            f_ids = f.get("token_ids", [])
            if b_ids and f_ids:
                point["token_id_match"] = b_ids == f_ids
            else:
                point["token_id_match"] = None

        results.append(point)

    # Summary
    if baseline_url and fused_url:
        n_match = sum(1 for r in results if r.get("text_match"))
        n_total = len(results)
        summary = {
            "text_matches": n_match,
            "total": n_total,
            "all_match": n_match == n_total,
        }
    else:
        summary = {"note": "only one server tested, no comparison"}

    return {
        "probe": "decode_boundary_probe",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "model": model,
            "max_tokens": max_tokens,
            "prompt_lengths": prompt_lengths,
            "batch_size": 1,
            "baseline_url": baseline_url,
            "fused_url": fused_url,
        },
        "summary": summary,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Decode-boundary probe: fused INT4 vs baseline"
    )
    parser.add_argument(
        "--baseline-url",
        default=None,
        help="Base URL for baseline vLLM server (e.g. http://localhost:8000)",
    )
    parser.add_argument(
        "--fused-url",
        default=None,
        help="Base URL for fused INT4 vLLM server (e.g. http://localhost:8001)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=MAX_TOKENS,
        help=f"Max tokens to generate (default: {MAX_TOKENS})",
    )
    parser.add_argument(
        "--prompt-lengths",
        type=int,
        nargs="+",
        default=PROMPT_LENGTHS,
        help=f"Prompt lengths to test (default: {PROMPT_LENGTHS})",
    )
    args = parser.parse_args()

    if not args.baseline_url and not args.fused_url:
        parser.error("At least one of --baseline-url or --fused-url required")

    result = run_probe(
        baseline_url=args.baseline_url,
        fused_url=args.fused_url,
        model=args.model,
        max_tokens=args.max_tokens,
        prompt_lengths=args.prompt_lengths,
    )

    json.dump(result, sys.stdout, indent=2)
    print()  # trailing newline

    # Print summary to stderr
    if "all_match" in result["summary"]:
        s = result["summary"]
        status = "PASS" if s["all_match"] else "FAIL"
        print(
            f"[{status}] {s['text_matches']}/{s['total']} prompts matched",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
