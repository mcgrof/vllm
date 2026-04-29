# Cartridge GPU residency tier

## Motivation

Before this change, `CartridgeConnector` loaded every configured
cartridge into CPU memory at init, pinned each one, and did an
unconditional per-request `.to(device=...)` copy inside
`start_load_kv()` to move the chunks to GPU. At scale this meant:

- **No GPU reuse.** Two consecutive requests for the same cartridge
  both paid the full CPU→GPU PCIe copy.
- **Immortal CPU residency.** Every routed cartridge sat pinned in
  CPU RAM for the lifetime of the connector even if only a handful
  were hot.
- **No capacity bound.** Nothing stopped a multi-cartridge
  deployment from promoting 100 cartridges onto GPU if the
  connector happened to route there in sequence.

The fifth production-architecture component (GPUResidencyManager)
fixes all three.

## Ownership

```
┌─────────────────────────┐       ┌─────────────────────────┐
│   CartridgeStore        │       │   CartridgeLMCachePlugin│
│ (CPU / disk source of   │◄──────│    read-only bridge     │
│   truth; read-only      │       │     into LMCache        │
│   from serve path)      │       └─────────────────────────┘
└───────────┬─────────────┘
            │ borrows CPU tensors
            ▼
┌─────────────────────────┐
│   GPUResidencyManager   │     ← GPU tier (this component)
│   - bounded capacity    │
│   - LRU eviction        │
│   - refcount pinning    │
│   - per-cartridge       │
│     promote/evict       │
└───────────┬─────────────┘
            │ get_chunk(id, layer)
            ▼
┌─────────────────────────┐
│   CartridgeConnector    │     ← request-time consumer
│   start_load_kv():      │
│   - acquire, inject,    │
│     release per batch   │
└─────────────────────────┘
```

- **`CartridgeStore`** remains the durable CPU/disk source of truth
  and the manifest owner. It is read-only from the serve path.
- **`GPUResidencyManager`** owns GPU placement. It borrows tensors
  from the store, caches GPU copies, enforces capacity, tracks
  in-flight refcounts, and evicts LRU under pressure.
- **`CartridgeLMCachePlugin`** is unchanged — still read-only,
  still zero-patch against LMCache. The residency manager is
  orthogonal: both components borrow from the same `CartridgeStore`.
- **`CartridgeConnector`** is a pure request-time consumer. It
  calls `acquire / get_chunk / release`. It does not do
  `.to(device=...)` on the hot path.

## Lifecycle

A request's cartridge chunks move through three states:

### 1. Uninitialized

Cartridge is registered in `CartridgeStore` (via config), loaded
as CPU tensors, but has no GPU copy. `is_resident(id) == False`.
Cartridges default to this state at connector init — no eager
promotion.

### 2. Promoted (acquired)

On first `acquire(id)`, the manager:

1. Reads every layer chunk from the store (CPU tensors).
2. Computes required bytes.
3. If required bytes > free capacity, evicts LRU unpinned entries
   until there's room, or raises `GPUResidencyError` if the
   needed cartridge can't fit even after evicting all unpinned.
4. Copies each chunk to the GPU device (dtype conversion at this
   step if requested).
5. Inserts into an LRU `OrderedDict` keyed by cartridge_id,
   increments `ref_count`, bumps `last_used`.

Subsequent `acquire(id)` calls while resident are O(1):
`ref_count += 1`, touch LRU, no PCIe copy. These are the **GPU
hits** that motivate the whole design.

`get_chunk(id, layer_idx)` returns the cached GPU tensor
(expected to be called only after acquire).

### 3. Released / evictable

`release(id)` decrements `ref_count`. A cartridge with
`ref_count == 0` is **eligible** for eviction but is NOT evicted
eagerly — only when another acquire triggers capacity pressure.

## Eviction

### Granularity

Chunk keys are `(cartridge_id, layer_idx)`. However, a request
using cartridge A needs every layer chunk of A; partial residency
of A is useless for serving. So **eviction operates at cartridge
granularity** (N chunks together). Metrics and internal indexing
still use chunk keys for precision but the pin/evict unit is the
whole cartridge.

### Policy

`lru` only. Walk the LRU `OrderedDict` oldest-first; for each
unpinned cartridge, evict all its chunks and remove the entry.
Stop when `resident_bytes + needed_bytes <= capacity_bytes`. If
walking all entries still doesn't free enough (all pinned), raise
`GPUResidencyError` from `acquire`; prefetch returns False.

Future: LFU, size-aware, or ARC would slot in as alternate
`_try_evict_until_fits` implementations.

### Pins are strict

A pinned cartridge (ref_count > 0) is **never** evicted, even
under memory pressure. If the working set of pinned cartridges
exceeds capacity, the offending acquire fails fast; the caller
decides whether to block, back off, or fall through to regular
prefill. The manager does not make that policy decision.

## Config surface (connector)

Added in `kv_connector_extra_config`:

| Key | Type | Default | Purpose |
|-----|------|---------|---------|
| `gpu_capacity_bytes` | int | `max(8×largest_cartridge, 1 GiB)` | GPU tier budget |
| `gpu_residency_device` | str | `cuda` if available else `cpu` | Target device |
| `preload` | list[str] | `[]` | Cartridge IDs to prefetch at init (unpinned) |

Singleton config (`cartridge_path`) and multi-cartridge config
(`cartridges` + `router`) are both unchanged. The residency
manager is always present; in the singleton case it degenerates
to a tier of size 1.

## Metrics

`GPUResidencyManager.snapshot_metrics()` returns:

- `gpu_hit` — acquires where the cartridge was already resident.
- `gpu_miss` — acquires that required promotion.
- `cpu_hit` — chunks read from the backing store (per promotion,
  summed across layers).
- `disk_hit` — reserved for a future tier where the store spills
  to disk; currently always 0.
- `promote_count` — number of cartridge promotions.
- `demote_count` — number of cartridge evictions.
- `evict_count` — same (1:1 with demote in the current design;
  distinct fields kept for future LFU/ARC variants that may
  demote without evicting).
- `bytes_promoted` — total bytes moved CPU→GPU.
- `bytes_evicted` — total bytes freed.
- `request_wait_on_promotion_count` — requests that blocked on a
  promotion instead of hitting; equals `gpu_miss` for this
  synchronous design.
- `current_gpu_resident_bytes` — live gauge.
- `current_gpu_resident_cartridges` — live gauge.
- `pinned_entries` — live gauge.

### Accounting invariant

After any sequence of acquire/release/prefetch,

```
bytes_promoted - bytes_evicted == current_gpu_resident_bytes
```

This is asserted by the Phase 4 `NO_UNBOUNDED_GROWTH` and
`METRICS_CONSISTENT` checks.

## Tests

### Unit tests (`tests/v1/core/test_cartridge_gpu_residency.py`)

Seven categories, 27 tests:

1. **Residency basics** — miss→promote, hit (no re-copy),
   device-correct tensor, error paths.
2. **Refcount safety** — concurrent pins, eviction blocked while
   pinned, eviction allowed at ref==0.
3. **LRU eviction** — oldest unpinned evicted, access promotes
   to MRU, re-promotion after eviction.
4. **Pinned protection** — pinned entries never evicted, all-pin
   raises, oversized cartridge raises.
5. **Multi-cartridge routing** — per-cartridge isolation,
   dtype conversion at promote time.
6. **No eager residency** — 100 cartridges registered, 0 resident
   until requested.
7. **Metrics coverage** — hit/miss/promote/evict counters,
   bytes accounting, wait counter.
8. **Prefetch** — promotes without pinning, respects pinned
   entries, next acquire is a hit.
9. **Thread safety smoke** — concurrent acquire/release across
   worker threads without corruption.

### Integration (`tools/cartridge_phase4_residency.py`)

Eight stress checks under realistic load:

1. Working set exceeds capacity (N > K).
2. Zipfian hits dominate (s=1.2, hit rate ≥ 50%).
3. Hot cartridges dominate residency.
4. No unbounded growth (resident ≤ capacity; accounting balances).
5. Eviction respects pins under pressure.
6. No ref leaks across `n_workers × n_requests`.
7. Metrics consistency (hit+miss=acquires, promote=miss, balance).
8. Re-promotion after eviction (evicted cartridges promotable).

### Validated on W7900 (prune)

- 178 cartridge unit tests (all passing, including 27 new residency
  tests and the full prior cartridge suite with no regressions).
- Phase 3 routing: 8/8 checks, unchanged after refactor.
- Phase 4 residency: 8/8 checks, 500-request Zipfian workload,
  57% hit rate at 5/20 capacity ratio.

## What the connector gave up

Lines removed from `start_load_kv()` fast path:

- `self._store.acquire(cart_id)` / `self._store.release(cart_id)`
  (replaced by residency acquire/release).
- `src_kv = self._store.get(chunk_key)` (replaced by
  `self._residency.get_chunk(...)`).
- `src_kv.to(device=kv_cache_layer.device, dtype=kv_cache_layer.dtype)`
  — the PCIe copy that motivated this whole exercise.

`start_load_kv()` is now two things: `acquire/release` bracket
around a helper (`_inject_request`) that handles slot mapping and
invokes the paged-cache writer. The "fetch device-correct source
tensors" step is `_fetch_source_kv()`, a one-line delegation to
the residency manager. Future tier changes (e.g. an LMCache-backed
GPU backend) replace only that helper.

## What stayed read-only

`CartridgeLMCachePlugin.batched_submit_put_task` and
`async_batched_submit_put_task` are still no-ops. The residency
manager does not push cartridge data into LMCache; it only
borrows CPU tensors from the same `CartridgeStore` that the plugin
exposes to LMCache. Cartridge content is still produced
exclusively offline by the Self-Study training pipeline.

## What's next

The residency tier is capacity-bounded, LRU, refcounted, and
tested. Follow-ups:

- **Async promotion** — today acquire is synchronous; a miss
  blocks the caller. A background promotion queue + per-cartridge
  readiness future would let the scheduler overlap PCIe copies
  with prefill of other requests.
- **LMCache-backed tier** — the current `GPUResidencyManager`
  lives in vLLM code. It could be re-implemented as an LMCache
  storage plugin so LMCache's controller runs placement/eviction
  and the connector only consumes. `CartridgeLMCachePlugin`'s
  interface is already prepared for this.
- **Tensor parallelism** — per-rank KV placement needs testing
  with the residency tier; residency is currently per-process and
  each TP worker will maintain its own view. Validated in Phase 3e
  of the H100 handoff.
