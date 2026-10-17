# TokenDance-style shared-update controls

`generate_all_gather_trace.py` creates synthetic, tokenizer-independent request
traces for testing segment-aware grouping before a model or GPU is involved.
Each run writes canonical JSONL plus a deterministic manifest and structural
overlap oracle. These are ordinary and observer controls for the
[shared-update recovery baseline](../../docs/design/kmesh.md). The serial and
collective GPU driver is a separate experiment input; this runner does not
execute those recovery modes.

Generate one controlled case:

```bash
.venv/bin/python benchmarks/kmesh/generate_all_gather_trace.py \
  --case same_order_different_offsets \
  --agents 5 \
  --shared-fraction 0.8 \
  --alignment unaligned \
  --block-size 32 \
  --qps 2 \
  --mode prefill_focus \
  --seed 20260731 \
  --output-dir /tmp/kmesh-trace
```

The command replaces only `/tmp/kmesh-trace/trace.jsonl` and
`/tmp/kmesh-trace/manifest.json`. Repeating it with the same arguments produces
byte-identical files. Supported axes are:

- cases: `same_order_different_offsets`, `different_shared_order`,
  `missing_shared_segment`, `different_private_history_lengths`,
  `incompatible_round_ids`, `mixed_cache_domains`, and `cot_branch_tree`;
- agents: `1`, `3`, `5`, or `10`;
- shared-token fractions: `0`, `0.5`, `0.8`, `0.9`, or `0.95`;
- shared segment lengths: `aligned` or `unaligned` to the selected block size;
- QPS: `1`, `2`, or `4`;
- modes: `prefill_focus` (one generated token) or `round_e2e` (32 generated
  tokens).

`--shared-fraction` is a nominal workload target, recorded as
`manifest.config.nominal_shared_fraction`. The observed fraction can differ due
to block quantization, the fixed task/private reserve, case-specific replacement
or variable-history transformations, and the branch-local thought added by the
CoT case. Per-request observed values and ten-percentage-point bucket counts are
recorded under `manifest.observed`; the min, max, and mean are also retained.

Single-agent and zero-shared-token runs are intentional controls. The manifest
reports `metadata_exercised` separately from `overlap_gate_exercised`, and
`fully_exercised` requires both. A control without a request pair or structurally
shared tokens is therefore marked degenerate with a machine-readable
`degenerate_reason`, even when its metadata distinction was generated correctly.

## JSONL record

Each line is one request and contains:

- deterministic synthetic `prompt_token_ids`;
- a versioned `kmesh_context` accepted directly by the strict KMesh request
  parser;
- request, agent, execution, group, round, and branch lineage;
- a complete compatibility key, including round, cache domain, model/layout,
  parallelism, prompt length, cached span, and prefill chunk span;
- private-history, shared-update, task, and optional thought segments;
- zero-based, half-open token spans and stable content digests;
- segment-relative and absolute-prompt block geometry;
- workload mode, QPS, and generated-token count.

`kmesh_context` uses lowercase wire enums, `segment_encoding: "sideband"`, and
only the canonical sideband segment fields (`segment_id`, `kind`,
`token_start`, `token_end`, `logical_order`, `source_id`, `shareable`, and
`lineage_id`). The separate top-level `segments` collection retains analytic
fields such as content digests, source lineage, reuse eligibility, and block
geometry. The synthetic prompt does not contain separator tokens and the
generator does not import vLLM, a model, a tokenizer, or a GPU library.

## Manifest and oracle

`manifest.json` records the full configuration, trace SHA-256, case-specific
checks, counts, observed fractions, branch tree, compatibility groups, pairwise
overlap, and aggregate overlap.

The oracle re-reads every emitted segment span from each request's
`prompt_token_ids`; it does not use KV values or generator-side expected counts.
It reports two distinct quantities:

- **raw structural overlap** requires the same segment ID and exact token slice,
  independent of absolute prompt position;
- **eligible structural overlap** additionally requires exact compatibility-key
  equality.

It also reports ordered shared-prefix tokens, segment-relative full blocks,
matching absolute-prompt-aligned full blocks, residual tokens, and shared tokens
at identical absolute offsets. Consequently, the incompatible-round and
mixed-cache-domain controls can have raw overlap while correctly reporting no
eligible overlap across their compatibility boundary.

This generator is a structural workload source and oracle only. It does not
measure KV similarity, execute a model, issue load at the requested QPS, or claim
that structurally matching blocks are numerically reusable.

## Baseline replay

Validate a trace and create a machine-readable result bundle without importing
the model runtime:

```bash
python benchmarks/kmesh/benchmark_all_gather.py \
  --trace /tmp/kmesh-trace/trace.jsonl \
  --manifest /tmp/kmesh-trace/manifest.json \
  --output-dir /tmp/kmesh-b0-dry \
  --mode b0 \
  --dry-run
```

For a real run, omit `--dry-run`. Mode `b0` disables automatic prefix caching;
mode `b1` enables it. Mode `b2` also disables prefix caching and attaches each
record's canonical `kmesh_context` to its own `SamplingParams`. It enables the
strict `KMeshConnector` observer with the `noop` backend, so it parses, groups,
and reports opportunities but never loads, saves, aliases, or mutates KV. B2
rejects asynchronous scheduling and chunked prefill. All three modes explicitly
disable the hybrid KV-cache manager because the observer does not implement its
allocation contract.

`--admission-mode batch` submits the complete group in one offline call and is
the primary B2 same-step grouping run. `serial_paced` issues requests one at a
time at the trace's QPS; for B2 this is the intentional zero-same-step control.
It reports service time separately from pacing time. Engine statistics are
enabled in all three modes for a fair control path. Every bundle retains source,
runtime, model, trace, configuration, token-output, cache-hit, and failure data.
The runner rejects an output directory containing prior results.

B2 snapshots the public `LLM.get_metrics()` API after warmup and after replay.
`summary.json.observer_metrics` contains their counter delta, with
`source: vllm_kv_connector_stats_prometheus_delta`,
`semantics: sum_of_scheduler_step_prospective_observations`, and
`executed_kv_reuse: false`. Shared-token and block counts are selected-plan,
structural opportunities—not materialized KV hits. The runner fails the run if
planner calls are absent, group/request accounting does not close, or the
runtime block size differs from the trace manifest. It also requires zero cache
hits and checks group presence against a generated support-aware expectation
derived from the independent structural oracle. Unsupported CoT inputs require
typed fallbacks, while `serial_paced` intentionally expects no same-step group.

The analyzer exposes a separate `b0_b2` behavior gate. It opens only when B0
and B2 have identical deterministic outputs, every B2 run reports zero cached
tokens, and every B2 observer invariant passes. This is a behavior-isolation
gate, not a performance or PIC claim.

At least five independent completed bundles per mode are needed before the
analyzer marks timing as claim-ready:

```bash
python benchmarks/kmesh/analyze_results.py \
  /tmp/kmesh-results \
  --output-json /tmp/kmesh-analysis.json \
  --output-markdown /tmp/kmesh-analysis.md
```

The analyzer compares only bundles with identical trace, model, runtime,
scheduler, and source identity. It requires output equivalence before reporting
a timing ratio. For batch runs it compares replay time; for serial-paced runs it
compares mean request service time so the configured arrival delay cannot appear
as inference speedup.
