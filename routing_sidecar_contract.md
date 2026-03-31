# Routing Sidecar Artifact Contract — Phase 1

## Purpose
Store per-head block affinity priors alongside cartridge assets for
first-token decode routing in vLLM.

## File layout
A cartridge directory with routing enabled contains:

    cartridge_dir/
    ├── cartridge.pt                  # existing: TrainableCache KV
    ├── prefix_token_ids.json         # existing: tokenized prefix
    └── routing_prior.pt              # NEW: routing sidecar

## routing_prior.pt format

A dict saved via torch.save() with these keys:

    {
        "version": 1,
        "model_id": str,              # e.g. "marin-8b-instruct"
        "block_size": int,             # must match vLLM block_size
        "doc_len": int,                # document token count
        "num_blocks": int,             # ceil(doc_len / block_size)
        "num_layers": int,
        "num_kv_heads": int,
        "context_length": int,         # context length used to build priors

        # Core routing data — per-layer, per-KV-head block affinities
        # Shape: (num_layers, num_kv_heads, num_blocks)
        # Higher value = stronger affinity = should be attended first
        "block_affinities": Tensor,

        # Optional: query-adjustment coefficients
        # Shape: (num_layers, num_kv_heads, num_blocks) or None
        "query_adjust_weights": Tensor | None,

        # Provenance
        "experiment_root": str,        # path to experiment that produced this
        "timestamp": str,              # ISO-8601
        "cartridge_path": str,         # path to cartridge.pt used
    }

## How the connector uses the sidecar

1. At startup, if routing_prior_path is set in kv_connector_extra_config,
   load routing_prior.pt alongside cartridge.pt
2. Validate block_size and num_layers match the model config
3. At first-token decode for a matched request, expose block_affinities
   to the attention layer via connector metadata
4. The attention layer uses topk(block_affinities[layer][head], K) to
   select which blocks each KV head group attends

## Routing modes (feature flag)

routing_mode in kv_connector_extra_config:
  - "full":                 standard dense attention (default)
  - "cartridge_prior":      all-head routing using block_affinities topk
  - "cartridge_prior_adjusted": routing with 0.7*cart + 0.3*query adjustment

## K parameter

routing_K in kv_connector_extra_config:
  - int, default 4
  - number of blocks per head to attend during routed decode
  - sweep: K={1,2,4,8}

