# Cartridges 8xH100 Validation Handoff

## Purpose

This document is the handoff for running cartridge serving validation
on an 8xH100 node. The vLLM integration, tests, and W7900 (ROCm)
validation are complete. This document describes what still needs to
be done on H100 and what is already done.

**Target hardware**: 8x NVIDIA H100 80GB (640 GB aggregate HBM)

## Current state (branch `20260415-cartridges-v0.4`)

### Source code — complete

All under `vllm/distributed/kv_transfer/kv_connector/v1/`:

| File | Lines | Purpose |
|------|------:|---------|
| `cartridge_connector.py` | 1011 | vLLM `KVConnectorBase_V1` plugin, multi-cartridge routing, residency-aware dispatch |
| `cartridge_router.py` | 257 | Request → cartridge_id resolution (Explicit / Label / Static / Composite) |
| `cartridge_gpu_residency.py` | 567 | Bounded GPU tier with LRU eviction, refcount pinning, metrics |
| `cartridge_manifest.py` | 217 | Per-cartridge metadata + validation |
| `cartridge_registry.py` | 177 | SQLite-backed manifest index |
| `cartridge_store.py` | 295 | Read-only CPU chunk store + ref counting |
| `cartridge_lmcache_plugin.py` | 260 | LMCache `StoragePluginInterface` adapter (read-only) |

### Tests — complete

Under `tests/v1/core/`:

| File | Tests | What it covers |
|------|------:|----------------|
| `test_cartridge_connector.py` | 14 | load, slot mapping, inject |
| `test_cartridge_connector_integration.py` | 13 | scheduler API, idempotency |
| `test_cartridge_manifest.py` | 14 | compat, JSON, from_cartridge |
| `test_cartridge_registry.py` | 12 | CRUD, label/model lookup |
| `test_cartridge_store.py` | 25 | lifecycle, refcount, pin, threads |
| `test_cartridge_lmcache_plugin.py` | 20 | key translation, delegation |
| `test_cartridge_fault.py` | 17 | corrupt, partial, shape, faults |
| `test_cartridge_router.py` | 28 | all 4 router types + config builder |
| `test_cartridge_routing.py` | 8 | per-request dispatch, isolation, dedup |
| `test_cartridge_gpu_residency.py` | 27 | tier logic: refcount / LRU / pin / metrics / prefetch |
| `test_cartridge_gpu_residency_cuda.py` | 7 | real-GPU evidence via `torch.cuda.memory_allocated()` |
| **Total** | **185** | host-only suite runs in ~30s; CUDA suite skips unless GPU present |

### Phase harnesses — complete and validated on W7900

Under `tools/`:

| Harness | Purpose | W7900 result |
|---------|---------|--------------|
| `cartridge_phase0_smoke.py` | End-to-end pipeline, TTFT, integrity | ✓ 5.5x avg TTFT speedup (3 patients) |
| `cartridge_phase0_5_lmcache.py` | LMCache plugin integration | ✓ 9/9 checks pass |
| `cartridge_phase1_multi.py` | Multi-cartridge + Zipfian + registry | ✓ 8/8 checks pass |
| `cartridge_phase2_perf.py` | TTFT percentiles at scale | ✓ 7.0x p50 speedup (n=50) |
| `cartridge_phase3_routed.py` | Multi-cartridge request routing (serving boundary) | ✓ 8/8 checks pass |
| `cartridge_phase4_residency.py` | Bounded GPU residency under Zipfian load | ✓ 9/9 checks pass on `--device cuda:0` (W7900) |

Phase 1 validated the storage plumbing (Store, Registry, LMCache
plugin). Phase 3 validates the **serving boundary** — the
scheduler-visible dispatch layer that resolves requests to
cartridge_ids and carries the chosen id through
`get_num_new_matched_tokens()` → `update_state_after_alloc()` →
`build_connector_meta()` → `start_load_kv()`. Phase 4 validates
the **GPU residency tier** — bounded capacity, LRU eviction,
refcount pinning, metrics, and no per-request PCIe copy on the
hot path. See `docs/design/cartridge_gpu_residency.md`.

### Phase 4 — real-GPU eviction evidence

Phase 4 has two complementary suites:

- **`tools/cartridge_phase4_residency.py --device cuda:0`** — 9
  stress checks against a real GPU: working set vs capacity,
  Zipfian hit rate, hot-cartridge residency, bytes accounting,
  pin protection, ref-leak safety, metrics consistency,
  `GPU_MEMORY_EVIDENCE` (`torch.cuda.memory_allocated()` deltas),
  and re-promotion.
- **`tests/v1/core/test_cartridge_gpu_residency_cuda.py`** — 7
  unit tests that directly measure GPU memory:
  `promote_increases_gpu_memory`,
  `evict_releases_gpu_memory`,
  `evict_without_replace_shrinks_gpu_memory`,
  `re_promotion_is_new_allocation` (external reference pins the
  old region so the allocator can't reuse it),
  `returned_tensor_is_on_device`,
  `no_unbounded_gpu_growth_under_pressure`,
  `pinned_cartridge_memory_survives_pressure`.
  Skipped unless CUDA/ROCm is available.

W7900 (AMD Radeon Pro W7900, ROCm 6.4.3, gfx1100) numbers —
H100 runs should reproduce or improve:

| Measurement | W7900 value |
|---|---|
| `cartridge_phase4_residency.py --device cuda:0` | 9/9 PASS |
| GPU_MEMORY_EVIDENCE baseline → promote | 0 → 234 881 024 bytes (+235 MB for 1 cartridge) |
| evict + same-size replace | Δ = 0 (evict freed 235 MB, reclaimed by replacement) |
| after `close()` | 0 bytes (full reclaim) |
| re-promoted data_ptr differs | True |
| Zipfian (500 req, 20 carts, cap=5, s=1.2) hit rate | 57.0 % (285 hits / 215 misses) |
| `evict_count` | 42 |
| Ref leaks across 4 workers × 125 requests | 0 |
| `test_cartridge_gpu_residency_cuda.py` on `cuda:0` | 7/7 PASS |

### Build — complete

- `VLLM_BUILD_PROFILE=cartridges_rocm` for AMD GPUs
- gfx1100 LDS overflow fix in `csrc/sampler.cu`
- `skinny_gemm` disabled on gfx1x

### Docs — complete

- `docs/design/cartridge_connector.md` — architecture, config, production roadmap
- `/home/mcgrof/devel/cartridges-engineering-guide/` — separate repo with full engineering guide (PDF + HTML)

## What 8xH100 enables that 1xH100 doesn't

The in-tree `CartridgeConnector` + `CartridgeRouter` +
`GPUResidencyManager` make multi-cartridge scale-out *safe* on top of
the TP story: residency is bounded, unused cartridges evict, pinned
cartridges survive, and metrics prove the accounting is closed. That
lifts several 1xGPU limits that used to cap our testing ambition.

1. **Bigger models**: Llama-3.1-70B and Qwen2.5-72B fit across 8 GPUs
   with tensor parallelism. Our W7900 work was stuck on Llama-3.2-3B.
2. **Longer context cartridges**: MTOB-style 128k+ token documents
   can be baked since KV cache at that length requires more HBM
   than a single H100.
3. **Parallel training**: train 8 cartridges simultaneously, one
   per GPU. Cuts total benchmark training time by 8x.
4. **Safe multi-cartridge working sets**: 640 GB aggregate HBM plus
   `GPUResidencyManager`'s LRU eviction means the tier can be told
   about *hundreds* of cartridges and it won't DoS the GPU — the
   hot set stays resident, the cold set lives in the CPU store, and
   capacity pressure is handled by eviction of unpinned entries.
   Phase 4 already demonstrated this at working-set=4× capacity on
   W7900; 8xH100 lets us go 10-100× bigger.
5. **vLLM tensor parallelism**: exercise the `tensor_parallel_size=8`
   serving path with cartridge injection. The per-request cartridge_id
   path is orthogonal to TP, but the slot-mapping code has not been
   tested under TP > 1 and may need adjustment for per-rank slices.
6. **Multi-tenant realism**: with the registry handling label lookup
   and residency handling placement/eviction, an 8xH100 node can
   simulate many independent tenants (each with their own cartridge
   set) interleaving requests — a workload shape W7900's 48 GB
   simply couldn't host.

## H100 scale-out validation work

Phases 0 through 4 are written and green on W7900. The work items
below are what the scaling lab needs to produce on H100 — a mix of
*running existing harnesses* against H100 hardware and *writing
new code* for TP support, benchmark loaders, and cross-GPU
residency. Each sub-phase says explicitly which.

### Phase 3a — Infrastructure parity (required)

Goal: prove the W7900 tests produce identical results on H100.

**Already implemented, just need to run:**
- `tools/cartridge_phase0_smoke.py --device cuda:0` (3 patients, TTFT, integrity)
- `tools/cartridge_phase0_5_lmcache.py` (LMCache plugin on CUDA)
- `tools/cartridge_phase1_multi.py --device cuda:0` (multi-cartridge)
- `tools/cartridge_phase2_perf.py --device cuda:0 --n-queries 100` (percentiles)

**Expected outputs:**
- TTFT speedup ≥ 7.0x (should be higher than W7900)
- KV checksums bit-identical to W7900 runs (cartridge loading is deterministic)
- All 8 Phase 1 checks pass
- All 9 Phase 0.5 checks pass

**Additional 8xH100-specific checks (need to be added):**
- Run Phase 0 on multiple GPUs simultaneously (one cartridge per GPU),
  verify KV checksums are identical across ranks
- Smoke test vLLM with `tensor_parallel_size=2, 4, 8` and the
  cartridge connector. **This may expose TP bugs in the connector
  that the W7900 single-GPU path masked.**
- Expected failure mode: per-rank slot mapping under TP. The current
  `build_connector_meta()` returns a single slot mapping; under TP
  each rank needs its slice of the KV injected into its local
  paged cache. Check whether vLLM's V1 KVConnector API handles this
  automatically or whether we need to split by rank.

**Deliverable:** `h100_phase3a_parity.log` with side-by-side W7900 vs
single-H100 numbers, plus a separate `h100_phase3a_tp.log` documenting
TP compatibility status.

**Estimated GPU time:** 1 hour (model load + single GPU + TP smoke).

### Phase 3b — LongHealth cartridge retrain (required)

Goal: verify Self-Study training reproduces on H100 and produces a
cartridge with matching quality.

**What exists (on the original development host):**
- A pre-trained `cache-step2694.pt` (Llama-3.2-3B, 10 LongHealth
  patients, 2048 tokens). Reference checkpoint to compare against.
- HazyResearch `cartridges` training package
  (https://github.com/HazyResearch/cartridges).
- Training config `examples/benchmarks/longhealth/longhealth_train.py`
  from the cartridges package (10 patients, 2 epochs, batch_size=32).

**What needs to be done:**
1. Launch H100 pod (see RunPod setup below)
2. Clone `/data/cartridges` and `/data/vllm-cartridges` onto the pod
3. Run `python examples/benchmarks/longhealth/longhealth_train.py`
   with `NUM_PATIENTS=10` to reproduce the W7900 cartridge
4. Compare:
   - Per-layer KV tensor distributions (mean, std, max) between
     W7900 and H100 trained cartridges
   - LongHealth MC accuracy via `eval_cartridge_hf_positions.py`
     (should match W7900's 54% at K=full)

**Deliverable:** `h100_phase3b_retrain.log` with a trained
`cache-step2694.h100.pt` file and a comparison report against the
W7900 cartridge.

**Estimated GPU time (1xH100):** 1-2 hours for training + 30 min eval.

**8xH100 opportunity**: Train cartridges for 8 different documents
in parallel (one per GPU), each taking ~1 hour. This lets us produce
a 10-patient-per-GPU cartridge farm (~80 cartridges) in a single
batch instead of serializing.

**Risk:** Training hyperparameter sensitivity. The existing cartridge
used ROCm bfloat16; H100 may need slightly different precision
handling. Have a fallback plan to train with bf16 on H100 first,
then test fp16 if bf16 fails.

### Phase 3c — Benchmark cartridge training (discretionary)

Goal: train cartridges for Musique and HotPotQA to compare against
CacheBlend (arxiv 2405.16444) and FusionRAG (arxiv 2601.12904).

**What needs to be built:**

1. **Benchmark data loaders** (not yet written):
   - `cartridges/examples/benchmarks/musique/musique_train.py`
   - `cartridges/examples/benchmarks/hotpotqa/hotpotqa_train.py`
   - Load from HuggingFace: `dgslibisey/MuSiQue` and `hotpot_qa`
   - For each sample, concatenate retrieved passages as the
     training document (per CacheBlend's Fig 1 chunked-RAG setup)

2. **Per-benchmark training scripts** (not yet written):
   - Train 10 cartridges per benchmark (one per sample set)
   - Use same Self-Study objective as LongHealth
   - Save with manifest (use `CartridgeManifest.from_cartridge()`)

3. **Benchmark eval harnesses** (not yet written):
   - `tools/cartridge_phase3_musique.py`
   - `tools/cartridge_phase3_hotpotqa.py`
   - Report: accuracy (F1 / EM), TTFT speedup, compared to
     (a) full prefill, (b) CacheBlend's published numbers

4. **Registry population** (can reuse `CartridgeRegistry`):
   - Register each trained cartridge with labels
     `{"benchmark": "musique", "sample_id": "..."}`
   - Route queries via `lookup_by_label()`

**Deliverable files:**
- Trained cartridges: `musique_*.pt` and `hotpotqa_*.pt`
- Manifests: `musique_*.json` and `hotpotqa_*.json`
- Eval results: `h100_phase3c_musique.json` and
  `h100_phase3c_hotpotqa.json`
- Comparison doc vs published CacheBlend/FusionRAG numbers

**Estimated GPU time (1xH100):**
- 10 Musique cartridges × ~30 min each = 5 hours training
- 10 HotPotQA cartridges × ~30 min each = 5 hours training
- Eval: 1-2 hours total

**8xH100 speedup**: Run 8 cartridge trainings in parallel per batch.
- 10 Musique cartridges in ~2 batches = ~1 hour wall clock
- 10 HotPotQA cartridges in ~2 batches = ~1 hour wall clock
- Total Phase 3c with 8xH100: ~3 hours including eval

**Risk:** Unknown how long Self-Study training takes on Musique-style
contexts (~17k tokens vs LongHealth's 2048 tokens). Starting with
1 Musique cartridge as a smoke test before committing to 10 is
safer.

### Phase 3d — Production pressure test (discretionary)

Goal: stress the residency tier and registry at production scale.

**What exists (single-GPU):**

- `cartridge_phase4_residency.py` already stresses the tier with
  working-set=4× capacity, Zipfian s=1.2, n_workers × n_requests
  concurrency, and 8 accounting/invariant checks plus
  `GPU_MEMORY_EVIDENCE` on real GPU. W7900 hit rate: 57%,
  evict_count: 42, 0 ref leaks across 4 workers × 125 requests.
- `cartridge_phase1_multi.py` exercises registry + LMCache plugin
  at 3 cartridges.

**What needs to be built for the 8xH100 variant:**

Scale the existing Phase 4 pattern up, and add the multi-GPU
dimension:

- `tools/cartridge_phase3d_pressure_8xh100.py` (new), which:
  - Registers 500+ cartridges (multi-tenant mix: 10 tenants ×
    50 cartridges each, distinct labels).
  - Per-GPU `GPUResidencyManager` with capacity sized for ~50
    cartridges (~10 × W7900 Phase 4's capacity).
  - Zipfian s=1.0 and s=1.2 sweeps, 100k+ requests, 1-8 hour
    runs to catch long-tail leaks / memory creep.
  - Cross-GPU routing: requests route to the GPU holding the
    cartridge; `lookup_by_label()` returns an `(tenant_id → GPU
    index)` affinity map. When a cartridge is not resident on
    any GPU, promote to the least-loaded one.
  - Measures: p50 / p95 / p99 TTFT, hit rate, eviction rate,
    per-tenant fairness, long-run stability (hit rate drift over
    time), pinned-entry count vs total-resident count.
  - Failure detection: wrong-cartridge incidents, ref leaks
    (`ref_count` != 0 at end), GPU memory drift via
    `torch.cuda.memory_allocated()`.
- Cross-GPU *migration*: optional — when Zipfian shifts the hot
  set across GPUs, migrate a cartridge between `GPUResidencyManager`
  instances without re-reading from the `CartridgeStore`. Today
  the store is the only CPU intermediary; migration would go
  through it, which is correct but adds a copy. A direct GPU ↔
  GPU NVLink path is future work.

**Deliverable:** `h100_phase3d_pressure.json` with TTFT
distribution, hit rate, per-tenant metrics, and a stability
timeseries. Plus a short report comparing to the W7900 Phase 4
numbers.

**Estimated GPU time:** 2-4 hours for the 100k-request sweep;
optional 8-hour run for long-tail stability.

**What residency unlocks here**: before the GPU tier existed,
"production pressure test" meant "pray we fit in memory." With
bounded capacity + LRU + refcount pinning + accounting invariants
(see `docs/design/cartridge_gpu_residency.md`), the pressure test
can cheerfully register 10× what fits and still complete — the
tier simply cycles hot cartridges in and out. That's the whole
point of having a cache tier instead of immortal residency.

### Phase 3e — Multi-cartridge request routing (required)

Goal: prove the scheduler-visible dispatch layer is correct on H100.

`cartridge_phase3_routed.py` exercises the serving boundary — the
piece that turns the Phase 1 storage infrastructure into an actually
routable serving path. Phase 1 ran on W7900 and validated the
Store/Registry/LMCache plumbing; this harness validates the
connector API path: router → per-request metadata → per-request
dispatch → cross-cartridge isolation.

**What runs out of the box:**

```bash
python tools/cartridge_phase3_routed.py \
    --cartridge /path/to/cache-step2694.pt \
    --device cuda:0 --n-cartridges 3
```

Eight checks:

1. **CONFIG** — connector loads N cartridges
2. **ROUTER** — explicit / static / composite all resolve
3. **LABEL_ROUTER** — registry-backed label lookup works
4. **METADATA** — per-request cartridge_id propagates to meta
5. **DISPATCH + ISOLATION** — untargeted cartridges never read;
   two concurrent requests → two cartridges → no cross-contamination
6. **BATCH_DEDUP** — N requests / K unique ids → K acquire/release
7. **UNKNOWN_ID** — router miss falls through to prefill, no crash
8. **CLEANUP** — tick-local state hygiene

**Expected outputs:**

- All 8 checks pass with identical semantics to W7900 (dispatch
  logic is GPU-agnostic; the harness runs on CPU paths via a stubbed
  triton kernel, but also sanity-checks against the real kernel on
  CUDA hosts).
- Bit-identical slot mapping output across W7900 and H100 runs with
  the same input cartridges and block allocations.

**8xH100-specific additions:**

- Run `cartridge_phase3_routed.py` with `tensor_parallel_size > 1`
  (same env changes as Phase 3a). The per-request cartridge_id path
  is orthogonal to TP, but the slot mapping code may need adjustment
  for per-rank slices — verify by running the harness against a
  TP-enabled VllmConfig.
- Load ≥ 16 cartridges across multiple GPUs (1 per device) and run
  a mixed-routing workload: half explicit_id, half label lookup via
  registry, 1k requests. Confirm acquire/release counts match unique
  cartridge counts per batch.

**Deliverable:** `h100_phase3e_routing.log` with 8-check pass log
plus a `--n-cartridges 16` run on 8 GPUs.

**Estimated GPU time:** 30 minutes.

## Scale-out ambitions — what the new tier unlocks

With the connector's request routing and `GPUResidencyManager` in
place, the things below are now *safe to try at scale* on 8xH100.
None of them were attempted on W7900 because 48 GB + immortal CPU
pinning would have OOMed or thrashed long before reaching an
interesting regime.

1. **Cartridge farms (100-1000)**. Register 500-1000 LongHealth /
   Musique cartridges simultaneously; stress the registry, the
   store, and the GPU tier's eviction. Expected behaviour: linear
   CPU RAM growth (the store), bounded GPU residency, per-tenant
   hit rate that tracks Zipfian skew.
2. **Long-run stability**. 8-24 hour runs at steady Zipfian load.
   Measure: hit rate drift, `ref_count` balance (every cartridge
   at 0 at the end), GPU memory stability (no creep vs baseline),
   metrics counters consistent with accounting invariants across
   the whole run.
3. **Tenant-shape probing**. Pathological workloads to confirm the
   tier degrades gracefully:
   - *Uniform*: every cartridge hit exactly once per round (no
     Zipfian locality) — expect ~0 % hit rate, maxed eviction,
     nothing crashes.
   - *All-pinned*: pin N > capacity cartridges concurrently —
     expect `GPUResidencyError` on the overflow, no corruption,
     no stale state after release.
   - *Flip-flop*: alternate between two disjoint hot sets at the
     residency boundary — expect thrash, measure throughput
     floor, confirm no infinite loops.
4. **Prefetch wins**. Use `GPUResidencyManager.prefetch()` driven
   by a router predictor; measure first-request TTFT vs cold
   miss. Expected improvement: remove the ~235 MB/cart PCIe copy
   from the critical path for predicted-next-cartridge requests.
5. **Benchmark breadth on one node**. Musique, HotPotQA, QASPER,
   LongBench, MTOB — 10 cartridges per benchmark, 500+ eval
   queries per cartridge. Previously node-hour-bounded; now
   memory-safe because working sets larger than GPU capacity
   just cycle through the tier.
6. **Cross-GPU migration (Tier 3 work)**. Move a cartridge from
   one GPU's tier to another under demand shift. The store acts
   as the transfer medium; eventual GPU↔GPU NVLink path is
   future work but the CPU-roundtrip variant is validatable today
   once the code is written.

Each of these is framed as a hypothesis about the tier's
behaviour. Failure of any one of them is a *signal*, not a
problem with the test — that's the point of going bananas.

## Execution plan (8xH100 node-hours)

1. **Tier 1 (required for upstream PR)**: Phase 3a + 3b + 3e.
   - 1xH100 portion (parity + retrain): 3 hours
   - TP smoke test: 1 hour
   - Parallel cartridge training for multiple docs: 1-2 hours
   - Phase 3e routing harness (single + TP): 30 min
   - Estimated node-time: **3-4 hours**
2. **Tier 2 (for benchmark validation paper)**: Add Phase 3c for
   one benchmark (Musique).
   - 10 cartridges across 8 GPUs, ~2 batches of 30min each: 1 hour
   - Eval: 1 hour
   - Estimated node-time: **+2 hours**
3. **Tier 3 (for production readiness)**: Add Phase 3d cross-GPU
   residency test. Requires new code for cross-GPU migration.
   - Code implementation: multi-day off-GPU
   - Test: 2-3 hours on 8xH100
   - Estimated node-time: **+3 hours**
4. **Tier 4 (paper-grade)**: Phase 3c for HotPotQA + QASPER + MTOB.
   - HotPotQA: 1 hour (parallel across 8 GPUs)
   - QASPER: 1 hour
   - MTOB (70B model, large context): 4 hours
   - Estimated node-time: **+6 hours**

## H100 setup reference

### Single H100 (cloud-style)

Setup that previously worked on a RunPod H100:

```bash
apt-get update && apt-get install -y git
cd /root
git clone <vllm fork> && cd vllm && git checkout 20260415-cartridges-v0.4
pip install -e . --no-build-isolation
cd /root && git clone https://github.com/HazyResearch/cartridges && \
    cd cartridges && pip install -e .
```

### 8xH100 node

Additional setup needed:
- NCCL: verify `torch.distributed.init_process_group(backend='nccl')`
  works across all 8 ranks
- vLLM TP: launch with `--tensor-parallel-size 8` (or 4, 2 for smaller
  smoke tests first)
- For parallel cartridge training: each rank can run its own
  cartridge training independently. Use `CUDA_VISIBLE_DEVICES=N`
  per process rather than DDP for independent cartridges.
- Storage: verify the cartridges training pipeline can write to a
  shared directory without rank collisions. Prefer one directory
  per cartridge with the rank-independent cartridge_id as the name.

### Expected dependencies

- `torch==2.10.0+cu124` (matches cartridges training requirements)
- `transformers>=4.50`
- `vllm==0.17.2rc1` (or match what the W7900 venv has)
- `nccl>=2.19` for 8-GPU comms

## What to send to the scaling lab

### Links to give them

1. **Branch**: `20260415-cartridges-v0.4` on [their public fork URL]
2. **Engineering guide PDF**: from
   `~/devel/cartridges-engineering-guide/cartridges_engineering_guide.pdf`
3. **This handoff doc**: `docs/design/cartridge_h100_handoff.md`

### What they need to add

**Code they should write** (doesn't exist in the branch):

- [ ] Phase 3a TP smoke test (`tools/cartridge_phase3a_tp.py`):
      verify cartridge injection works under `tensor_parallel_size >
      1`. This is likely to expose bugs in per-rank slot mapping.
- [ ] Phase 3b parallel training driver: run N cartridge trainings
      across 8 GPUs using `CUDA_VISIBLE_DEVICES`.
- [ ] Phase 3c benchmark data loaders:
      - `cartridges/examples/benchmarks/musique/musique_train.py`
      - `cartridges/examples/benchmarks/hotpotqa/hotpotqa_train.py`
- [ ] Phase 3c training scripts (per benchmark, parallelized).
- [ ] Phase 3c eval harness:
      - `tools/cartridge_phase3c_musique.py`
      - `tools/cartridge_phase3c_hotpotqa.py`
- [ ] Phase 3d production pressure harness (8xH100 variant):
      - `tools/cartridge_phase3d_8xh100_pressure.py`
      - Cross-GPU residency manager (new module, not yet written)

**Code they can run as-is** (already in branch):

- [x] Phase 0 smoke test (`tools/cartridge_phase0_smoke.py`)
- [x] Phase 0.5 LMCache integration (`tools/cartridge_phase0_5_lmcache.py`)
- [x] Phase 1 multi-cartridge infrastructure (`tools/cartridge_phase1_multi.py`)
- [x] Phase 2 TTFT percentiles (`tools/cartridge_phase2_perf.py`)
- [x] Phase 3 multi-cartridge routing (`tools/cartridge_phase3_routed.py`)
- [x] Phase 4 GPU residency stress (`tools/cartridge_phase4_residency.py --device cuda:0`)
- [x] 185 unit tests (`pytest tests/v1/core/test_cartridge_*.py`)
      including 7 CUDA-only residency evidence tests

### What to flag as known limitations

1. **Synchronous cartridge load at connector init**. `CartridgeConnector.__init__`
   loads every configured cartridge into the `CartridgeStore` (CPU
   source of truth) before serving. The GPU tier is populated lazily
   on first request, so GPU memory is not an issue, but CPU RAM
   grows linearly with the number of registered cartridges. A
   future on-demand lazy-load driven by the registry would remove
   this.

2. **Block-level routing is a separate layer**. Loading K < N blocks
   per cartridge for memory savings is handled on top of this
   connector, not here. The GPU residency tier evicts at cartridge
   granularity; finer-grained chunk residency would require changes
   to the eviction API.

3. **Synchronous KV injection at prefill time**. `start_load_kv`
   blocks until every layer is written to the paged cache. Async
   injection — overlap the PCIe copy with prefill of other
   requests — is the obvious next optimisation and drops out
   naturally once the residency tier gets a background promotion
   queue.

4. **LMCache plugin is read-only by design**. `put` operations are
   no-ops. Cartridges are produced offline by the Self-Study
   training pipeline; nothing at serve time writes to the
   cartridge tier. This is intentional, not a bug, but worth
   flagging for anyone expecting bidirectional behaviour.

5. **TP > 1 not yet exercised**. The per-request `cartridge_id`
   path is orthogonal to tensor parallelism, but the slot-mapping
   code assumes a single rank today. Phase 3a's TP smoke test is
   how we find out whether it needs per-rank splits.

## Success criteria for the scaling lab

The scaling lab's work is "done" when:

1. [ ] **Tier 1** (3a + 3b + 3e): H100 produces Phase 0-4 results
       matching W7900 within tolerance; LongHealth cartridge
       retrained on H100 matches existing cartridge's accuracy;
       vLLM TP=8 serving path works with cartridge injection (or,
       if not, the TP bug is identified and scoped via the Phase
       3a / 3e harnesses).
2. [ ] **Tier 2** (3c Musique): cartridge serving matches or beats
       CacheBlend's 2.2-3.3x TTFT speedup on Musique (our W7900
       number is 7.0x) with quality degradation <5pp vs full
       attention.
3. [ ] **Tier 3** (3d scaled-up pressure): 500+ registered
       cartridges across 8 GPUs sustain >95% quality, <1%
       wrong-cartridge incidents, bounded GPU memory (every
       checkpoint stays below per-GPU capacity), and 0 ref leaks
       over a multi-hour run. Cross-GPU migration validated with
       the Phase 4 metrics invariants holding across the whole
       run.
4. [ ] **Tier 4** (paper-grade): multi-benchmark validation across
       LongHealth, Musique, HotPotQA, QASPER; optional MTOB for
       context-extension story. All trained cartridges registered
       with manifests and routable via the Registry's
       `lookup_by_label()`.

---

Branch: `20260415-cartridges-v0.4`
Last W7900 validation: 2026-04-16 — all 6 phase harnesses green
(Phase 0/0.5/1/2/3/4), 185/185 unit tests passing including 7
CUDA-only residency evidence tests on cuda:0, 7.0x TTFT p50 speedup
over full prefill, 57% hit rate at working-set=4× capacity under
Zipfian s=1.2.
