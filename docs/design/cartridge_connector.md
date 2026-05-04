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

## Current limitations

- One cartridge per server: the connector loads a single cartridge at
  init and injects it into every request. Multi-cartridge serving
  (per-request routing, GPU residency management) is future work.
- Block-level routing (loading K < N blocks per cartridge for memory
  savings) is out of scope for this connector.
- KV injection is synchronous at prefill time.
- At tensor parallelism > 1, the cartridge stores full KV and each
  rank slices its head range at injection time.
