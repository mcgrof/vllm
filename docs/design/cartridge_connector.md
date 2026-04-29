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

LMCache integration targets steps 3 and 4: its LocalCPUBackend +
LocalDiskBackend provide tiered storage, its controller handles
lookup/move/pin/evict operations, and its chunk-based API maps
naturally to the cartridge storage format.
