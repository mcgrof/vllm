# CartridgeConnector Design

## Overview

CartridgeConnector is a `KVConnectorBase_V1` plugin that injects
pre-trained KV caches ("cartridges") into vLLM's paged attention system.
Cartridges are produced by the [HazyResearch Self-Study training
method](https://arxiv.org/abs/2504.16106), which optimizes a KV cache's
values so the model can answer questions about a document without
re-processing it at serve time.

## Architecture

```
User prompt: "What is patient_01's diagnosis?"
         │
         ▼
┌─────────────────┐    prefix match?    ┌──────────────────┐
│   vLLM V1       │───────────────────►│ CartridgeConnector│
│   Scheduler     │   yes: N tokens     │                  │
│                 │◄────────────────────│ get_num_new_     │
│                 │   "N externally     │ matched_tokens() │
│   allocate N    │    computed"        └──────────────────┘
│   blocks for    │
│   prefix        │
│                 │
│   schedule      │    build_connector_meta()
│   prefill for   │──────────────────►┌──────────────────┐
│   remaining     │   slot_mapping     │ CartridgeConnector│
│   tokens only   │                   │ (worker side)    │
│                 │                   │                  │
└─────────────────┘                   │ start_load_kv(): │
                                      │ write KV via the │
                                      │ backend's cache- │
                                      │ update hook      │
                                      └──────────────────┘
```

## Cartridge format

The connector loads TrainableCache checkpoints, the format produced
by Self-Study training. A checkpoint is a Python object or dict with:

- `trainable_keys`: per-layer parameter list, each `(1, H, T_train, D)`
- `trainable_values`: same shape
- `frozen_keys` (optional): per-layer BOS/system prefix, each `(1, H, T_frozen, D)`
- `frozen_values` (optional): same shape

The frozen prefix is concatenated before the trainable tokens to
reconstruct the full KV cache. Without this, the leading positions
get overwritten with trained values that were optimized assuming the
frozen BOS prefix would be present, corrupting attention.

## Configuration

```json
{
    "kv_connector": "CartridgeConnector",
    "kv_connector_module_path":
        "vllm.distributed.kv_transfer.kv_connector.v1.cartridge_connector",
    "kv_connector_extra_config": {
        "cartridge_path": "/path/to/cartridge.pt"
    },
    "kv_role": "kv_both"
}
```

### Parameters

- **cartridge_path** (required): path to the `.pt` cartridge checkpoint.
- **manifest_path** (optional): path to a cartridge manifest JSON for
  model-compatibility validation at startup.

## KV injection path

Cartridge injection is valid iff the connector's write layout equals
the attention backend's read layout. The paged-cache byte layout is
owned by the attention backend and is not always flat token-major:
ROCM_ATTN, for example, keeps KV HIP-swizzled behind a nominally flat
allocation (`PagedAttention.split_kv_cache` views), so a flat write
into it corrupts silently — injected tokens are read back scrambled
while natively computed tokens stay fine.

The injector therefore routes each layer's write through the owning
attention impl's `do_kv_cache_update` hook when the impl is a
standard (non-MLA) `AttentionImpl` exposing it. That is the exact
call the backend uses for natively computed tokens, so the write
layout matches the read layout by construction on every backend.
MLA impls are excluded because their `do_kv_cache_update` takes
`(kv_c_normed, k_pe)`, not `(key, value)`.

For layers without the hook, the injector falls back to a flat
token-major `reshape_and_cache_flash` write, auto-detecting the K/V
split among the layouts observed across backends:

```
(2, num_blocks, block_size, num_kv_heads, head_dim)   K/V at dim 0
(num_blocks, 2, block_size, num_kv_heads, head_dim)   K/V at dim 1
(k_cache, v_cache) tuple of separate 4D tensors
```

The fallback is only correct for flat-layout backends.

## Storage components

- **CartridgeStore** — read-only KV chunk store with acquire/release
  ref counting. Loads cartridge layers on demand, caches them keyed
  by `ChunkKey(cartridge_id, layer_idx)`, and tracks per-cartridge
  residency metadata. The connector holds a ref for the duration of
  each batch injection so a cartridge cannot be evicted mid-write.

- **CartridgeRegistry** — SQLite-backed index mapping cartridge IDs
  to manifest paths, labels, and metadata. Supports insert, lookup
  by ID or label, and listing. The registry is the durable catalog
  for multi-cartridge deployments; the store is the in-memory data
  plane.

## Multi-cartridge routing

The connector supports two configurations:

### Singleton mode (backward-compatible)

One cartridge loaded at startup, injected into every request. The
existing `cartridge_path` config produces this shape.

### Multi-cartridge mode

Multiple cartridges loaded at startup, per-request dispatch via a
`CartridgeRouter`. The router resolves each incoming request to a
`cartridge_id` before block allocation, and the chosen ID is carried
end-to-end through scheduler-side state into per-request connector
metadata. The worker dispatches to the correct cartridge per request
within a single batch.

```
Request R0 (kv_transfer_params: cartridge_id=patient_00)  ──┐
Request R1 (kv_transfer_params: cartridge_id=patient_01)  ──┤
Request R2 (kv_transfer_params: cartridge_id=patient_02)  ──┘
                     │
                     ▼
        ┌─────────────────────────┐
        │   CartridgeRouter       │
        │  (Explicit/Label/etc.)  │
        └─────────────────────────┘
                     │ per-request cartridge_id
                     ▼
        get_num_new_matched_tokens  ──► stash (req_id → id, num_tokens)
                     │
                     ▼
        update_state_after_alloc    ──► commit to _requests_need_load
                     │
                     ▼
        build_connector_meta        ──► CartridgeReqMeta[] with id
                     │
                     ▼
        start_load_kv (worker)      ──► per-request store.get(id, layer)
                                        per-request slot_mapping write
```

Config for multi-cartridge mode:

```json
{
    "kv_connector": "CartridgeConnector",
    "kv_connector_module_path":
        "vllm.distributed.kv_transfer.kv_connector.v1.cartridge_connector",
    "kv_connector_extra_config": {
        "cartridges": [
            {"cartridge_id": "patient_00", "path": "/carts/p00.pt"},
            {"cartridge_id": "patient_01", "path": "/carts/p01.pt",
             "manifest_path": "/carts/p01.manifest.json"}
        ],
        "router": {
            "type": "composite",
            "routers": [
                {"type": "explicit", "key": "cartridge_id"},
                {"type": "static", "cartridge_id": "patient_00"}
            ]
        }
    },
    "kv_role": "kv_both"
}
```

Each `cartridges[*]` entry takes `path` (required), `cartridge_id`
(inferred from the manifest if omitted), and `manifest_path` (optional
— enables full model-compat validation).

### Router types

- **ExplicitCartridgeRouter**: reads `cartridge_id` from the request's
  `kv_transfer_params` (with a fallback to the raw
  `sampling_params.extra_args`, top level or nested, following vLLM
  convention). Use this when the caller knows which cartridge they
  want.

- **LabelCartridgeRouter**: reads a label value (e.g. `patient_id`,
  `doc_type`) from extras and queries `CartridgeRegistry` by label
  for a matching manifest. Use this for domain-level routing where
  callers speak labels, not cartridge IDs.

- **StaticCartridgeRouter**: always returns the same ID. Used as a
  fallback and for singleton-mode compat.

- **CompositeRouter**: chains routers in order; the first non-None
  resolution wins. Typical pattern is explicit → label → static
  fallback.

### Isolation guarantees

`CartridgeReqMeta` carries `cartridge_id` per request, so two
concurrent requests bound to different cartridges inject into their
own allocated slot ranges with no cross-contamination. The worker
deduplicates `store.acquire()`/`store.release()` to a single pair
per unique cartridge in a batch. Tests exercise this end-to-end
without a GPU.

## LMCache integration

`CartridgeLMCachePlugin`
(`vllm/distributed/kv_transfer/kv_connector/v1/cartridge_lmcache_plugin.py`)
lets cartridge chunks participate in LMCache's multi-tier KV cache
hierarchy (GPU → CPU → disk) with automatic eviction, prefetch, and
controller operations.

### Zero-patch plugin design

The integration requires **no changes to LMCache itself**. LMCache
exposes a first-class plugin extension point —
`lmcache.v1.storage_backend.abstract_backend.StoragePluginInterface` —
so third-party storage backends register via LMCache's YAML config
without forking the project:

```yaml
chunk_size: 256
storage_plugins: cartridge_store
extra_config:
  storage_plugin.cartridge_store.module_path:
    vllm.distributed.kv_transfer.kv_connector.v1.cartridge_lmcache_plugin
  storage_plugin.cartridge_store.class_name: CartridgeLMCachePlugin
  storage_plugin.cartridge_store.registry_db: /path/to/cartridges.db
  storage_plugin.cartridge_store.cartridge_dir: /path/to/cartridge/files/
```

LMCache imports the module, instantiates `CartridgeLMCachePlugin`,
and calls `contains()`, `get_blocking()`, `pin()`, `unpin()`,
`remove()`, etc. per the plugin contract. All cartridge-specific
logic — `.pt` loading, layer splitting, ref counting, manifest
validation — stays in the vLLM tree.

### Key translation

`CacheEngineKey` ↔ `ChunkKey` mapping uses the `tags` tuple:

| LMCache side             | CartridgeStore side                |
| ------------------------ | ---------------------------------- |
| `key.tags[0]` (str)      | `ChunkKey.cartridge_id`            |
| `key.tags[1]` (str(int)) | `ChunkKey.layer_idx`               |

Keys that do not follow this convention return `None` from the
plugin's read operations (i.e. the plugin acts as a miss for
non-cartridge traffic, letting LMCache fall back to its default
backend chain).

### Read-only semantics

Cartridges are baked offline and served unchanged. All write
operations (`batched_submit_put_task`,
`async_batched_submit_put_task`) are no-ops. This is correct: LMCache
should never write to the cartridge tier because cartridge contents
are produced by the Self-Study training pipeline, not by serve-time
KV production. Pin/unpin/remove do delegate to `CartridgeStore`'s
ref counting so that LMCache's controller can still coordinate
memory pressure with in-flight requests.

### Store sharing

`CartridgeLMCachePlugin.set_store()` lets `CartridgeConnector`
inject its own `CartridgeStore` instance into the plugin, so a
single cartridge file is loaded once and served through both the
direct connector path and the LMCache tiering path.

### Version compatibility

The plugin targets the `StoragePluginInterface` extension point
(LMCache ≥ 0.3.13, where put tasks carry the unified
`on_complete_callback`). When LMCache is not installed, the module
still imports cleanly — a stub `StoragePluginInterface` base class
lets unit tests run without LMCache as a dependency.

## Current limitations

- Cartridges are loaded synchronously at connector init. On-demand
  lazy loading driven by the registry is future work.
- GPU residency management (bounded GPU tier, eviction) is future
  work; cartridge chunks are moved to the paged cache's device at
  injection time.
- Block-level routing (loading K < N blocks per cartridge for memory
  savings) is out of scope for this connector.
- KV injection is synchronous at prefill time.
- At tensor parallelism > 1, the cartridge stores full KV and each
  rank slices its head range at injection time.
