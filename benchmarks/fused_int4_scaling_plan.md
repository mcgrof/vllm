# Fused INT4 scaling plan after the March 30 A100 root cause analysis

This note records the current working interpretation and the next scaling steps
for fused INT4 testing inside `vllm`.

## Root cause identified (2026-03-30)

The serving-path corruption is caused by **activation outliers** in specific
KV heads. Qwen2.5-7B kv_head 1 produces isolated K-cache values of ±400
while the per-group (GROUP_SIZE=32) average sits at ±2. Symmetric INT4
quantization (`scale = amax/7 ≈ 57`) makes nearly all values quantize to
zero, destroying the attention signal for all query heads sharing that KV head.

A trimmed-mean outlier clipping fix has been implemented in the write kernel
(`_reshape_and_cache_int4_kernel`). This must be validated before scaling.

## Current working interpretation

The most likely explanation for the earlier serving-corruption confusion is that
we mixed together:
- standalone kernel correctness,
- NVIDIA A100 serving tests,
- and prior ROCm / W7900 instability.

The important practical conclusion for `vllm` test planning is:
- **do not let the W7900 lane block NVIDIA scaling**,
- treat NVIDIA A100/H100 as the canonical fused-INT4 validation lane,
- keep the correctness bar strict while scaling.

## What changed in the test protocol

The test protocol fix is conceptual but important:

- do **not** count answer-prefix matches as success,
- do **not** accept outputs with obviously corrupted continuations,
- use exact text / token agreement or an explicitly documented semantic rule,
- keep micro-benchmark speedup evidence separate from serving correctness.

This avoids repeating the earlier mistake where a run looked successful because
its leading answer tokens were plausible even though the continuation quality was
clearly broken.

## What is considered solved vs unsolved

### Solved enough to scale
- fused INT4 kernel math on NVIDIA,
- grouped-scale / packed-contract fear at the standalone diagnostic level,
- basic fused smoke and micro-benchmark bring-up on A100.

### Not yet solved enough to declare production-ready
- full serving-path correctness across the planned model sweep,
- model-specific MSL policy calibration on the final target GPUs,
- API-server + lm-eval validation on the chosen policy settings.

## Planned scaling order inside `vllm`

### Phase 1 — strict single-GPU policy calibration on NVIDIA
Use the existing sweep scripts and keep exact-match comparison rules:

1. `benchmarks/fused_int4_qwen2_h100_sweep.py`
2. `benchmarks/fused_int4_mistral_h100_sweep.py`
3. the existing Qwen2.5 H100 policy lane documented in
   `benchmarks/fused_int4_qwen_h100_policy.md`

Goal:
- identify the lowest safe `VLLM_FUSED_INT4_MIN_SEQ_LEN` per model,
- verify whether symmetric / asymmetric and group-size choices materially
  change the correctness envelope,
- keep the comparison criterion strict.

### Phase 2 — bounded decode/throughput scaling on NVIDIA
Once a per-model MSL policy exists, run bounded throughput checks on the same
NVIDIA lane:
- `B=1,2,4,8,16`
- representative decode lengths / prompt lengths,
- record where fused becomes throughput-positive versus FP16 fallback.

Goal:
- convert correctness-safe policies into actual serving guidance.

### Phase 3 — API-server and lm-eval validation
After the single-GPU policy looks stable:
- run a real vLLM API server,
- compare against FP16 baseline,
- run bounded lm-eval / task checks,
- verify that server-path behavior matches the direct `LLM(...)` benchmark path.

Goal:
- prove the serving integration path, not just the kernel path.

### Phase 4 — broader model / hardware generalization
Only after the above succeeds cleanly:
- expand to additional supported models,
- confirm the H100 policy generalization envelope,
- consider whether A100 and H100 need different defaults.

## Operational guidance

- NVIDIA scaling should happen on **A100/H100**, not ROCm/W7900.
- Keep result bundles small, strict, and reproducible.
- Treat any output corruption under the server path as an integration failure
  until proven otherwise, even if the standalone kernel diagnostics still pass.
- Do not reopen low-level kernel panic unless the strict NVIDIA path shows a
  real numerical mismatch again.
