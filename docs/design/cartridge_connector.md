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
                                      │ write KV via     │
                                      │ triton_reshape_  │
                                      │ and_cache_flash  │
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

## KV injection path

The connector writes cartridge KV through
`triton_reshape_and_cache_flash`, which writes to the unified V1
paged-KV layout:

```
(2, num_blocks, block_size, num_kv_heads, head_dim)
```

This is the same write path used by the attention backends
(flash_attn on CUDA, triton_attn, rocm_attn on ROCm), ensuring
correct memory layout on all platforms without per-backend branching.

## Current limitations

- One cartridge per vLLM instance, loaded at startup. Every request
  gets the same cartridge injected.
- Block-level routing (loading K < N blocks) is handled by a separate
  routing layer, not this base connector.
- Synchronous KV loading at prefill time.

## Production architecture (future work)

The current single-cartridge connector is stage 0. The production
architecture has five components:

1. **CartridgeManifest**: per-cartridge metadata (model ID, shapes,
   checksums, routing labels like patient/document/topic).
2. **CartridgeRegistry**: SQLite index over manifests. Answers
   "which cartridge should this request use?" via exact key lookup,
   full-text search, or embedding similarity.
3. **CartridgeStore**: chunked serving format with checksums.
   Converts raw `.pt` into `(cartridge_id, layer_idx, chunk_idx)`
   storage keys. LMCache fits here as the multi-tier (GPU/CPU/SSD)
   storage backend with prefetch and eviction.
4. **GPUResidencyManager**: tracks which cartridge chunks are on GPU,
   refcounts in-flight requests, evicts under memory pressure using
   LRU/LFU. Separate from the registry.
5. **RoutedCartridgeConnector**: receives `cartridge_id` from a
   request router, loads on demand from the store, injects into
   allocated slots.

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

| Requirement           | Version   | Why                                               |
| --------------------- | --------- | ------------------------------------------------- |
| `StoragePluginInterface` | ≥ 0.3.10.post1 | Plugin extensibility refactor (LMCache PR #2118)  |
| Unified `on_complete_callback` on put tasks | ≥ 0.3.13 | Matches our `batched_submit_put_task` signature (LMCache PR #2393) |

**Minimum supported**: LMCache `v0.3.13`.
**Tested against**: LMCache `v0.4.x` (mainline as of April 2026).

When LMCache is not installed, the plugin module still imports
cleanly — a stub `StoragePluginInterface` base class lets unit tests
run without LMCache as a dependency. The plugin is exercised
end-to-end against a real LMCache runtime by
`tools/cartridge_phase0_5_lmcache.py`.
