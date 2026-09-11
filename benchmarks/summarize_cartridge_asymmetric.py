#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize cartridge-asymmetric benchmark JSON files as Markdown."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def median(rows: list[dict[str, Any]], field: str) -> float:
    return statistics.median(float(row[field]) for row in rows)


def main() -> None:
    args = parse_args()
    results = [json.loads(path.read_text()) for path in args.results]
    by_mode = {result["mode"]: result for result in results}
    control_modes = {
        "cart-k16-v8": "bf16-cart",
        "cart-global-k16-v8": "global-k16-v8",
    }
    lines = [
        (
            "| Mode | Control | Quality | Batch | req/s | throughput delta | "
            "tok/s | TTFT p50 | TTFT delta | cached tokens |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in result["performance"]:
            grouped[int(row["batch_size"])].append(row)
        for batch_size in sorted(grouped):
            rows = grouped[batch_size]
            req_s = median(rows, "request_per_s")
            control_mode = control_modes.get(result["mode"])
            control_req_s = None
            control_ttft = None
            if control_mode is not None and control_mode in by_mode:
                control_rows = [
                    row
                    for row in by_mode[control_mode]["performance"]
                    if int(row["batch_size"]) == batch_size
                ]
                if control_rows:
                    control_req_s = median(control_rows, "request_per_s")
                    control_ttft = median(control_rows, "ttft_ms_p50")
            throughput_delta = (
                (req_s / control_req_s - 1.0) * 100.0 if control_req_s else None
            )
            ttft = median(rows, "ttft_ms_p50")
            ttft_delta = (ttft / control_ttft - 1.0) * 100.0 if control_ttft else None
            lines.append(
                "| {mode} | {control} | {quality:.3f} | {batch} | {req_s:.2f} | "
                "{throughput_delta} | {tok_s:.1f} | {ttft:.1f} ms | "
                "{ttft_delta} | {cached:.0f} |".format(
                    mode=result["mode"],
                    control=control_mode or "—",
                    quality=float(result["quality"]["accuracy"]),
                    batch=batch_size,
                    req_s=req_s,
                    throughput_delta=(
                        f"{throughput_delta:+.2f}%"
                        if throughput_delta is not None
                        else "—"
                    ),
                    tok_s=median(rows, "output_token_per_s"),
                    ttft=ttft,
                    ttft_delta=(
                        f"{ttft_delta:+.2f}%" if ttft_delta is not None else "—"
                    ),
                    cached=median(rows, "num_cached_tokens_min"),
                )
            )
    text = "\n".join(lines) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
