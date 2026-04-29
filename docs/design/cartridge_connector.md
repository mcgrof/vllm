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
Request R0 (extras: cartridge_id=patient_00)  ──┐
Request R1 (extras: cartridge_id=patient_01)  ──┤
Request R2 (extras: cartridge_id=patient_02)  ──┘
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
  `sampling_params.extra_args` (either at the top level or nested in
  `kv_transfer_params`, following vLLM convention). Use this when the
  caller knows which cartridge they want.

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
per unique cartridge in a batch, matching the per-request count
seen by ref counting. Tests exercise this end-to-end without a GPU.

## Current limitations

- Block-level routing (loading K < N blocks per cartridge for memory
  savings) is handled by a separate routing layer on top of this
  connector, not here.
- Cartridges are loaded synchronously at connector init. On-demand
  lazy loading driven by the registry is future work (see
  Production architecture below).
- KV injection is synchronous at prefill time.

## Production architecture (future work)

The in-tree connector covers four of the five production components.
Remaining work is the GPU residency manager:

1. **CartridgeManifest** — in tree.
2. **CartridgeRegistry** — in tree (SQLite index, label lookup).
3. **CartridgeStore** — in tree (chunked, ref-counted, thread-safe).
4. **RoutedCartridgeConnector** — in tree as the multi-cartridge
   mode of this connector (Explicit / Label / Composite routers).
5. **GPUResidencyManager** — not yet. Cartridges are currently
   pinned in CPU memory at init. A GPU-resident tier with LRU/LFU
   eviction under memory pressure is the next piece, and will plug
   in as an LMCache backend via the existing
   `CartridgeLMCachePlugin`.

LMCache integration targets steps 3 and 4: its LocalCPUBackend +
LocalDiskBackend provide tiered storage, its controller handles
lookup/move/pin/evict operations, and its chunk-based API maps
naturally to the cartridge storage format.
