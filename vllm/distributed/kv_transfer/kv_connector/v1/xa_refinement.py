# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-attention refinement (xa) side-channel + helpers.

This is the vLLM-side infrastructure for HKVD-based K/V refresh after a
connector has populated the paged cache. Composable with
CartridgeConnector (cartridge inject path) and the LMCache sparse
dispatch path (retrieve_chunks).

Design: Option 1 from docs/design/cartridge_xa_refinement.md — the
caller precomputes fresh K/V per request (via a separate HF
transformers forward), stashes it in this module's side channel, and
the connector consults the side channel after its inject step. The
connector then scatters fresh K/V at HKVD positions using existing
paged-cache inject machinery.

Why a side channel: vLLM's model forward is driven by the worker
runloop, not by connector methods. Running a fresh reference forward
inside start_load_kv is hard to do cleanly. A side channel keeps the
vLLM flow intact; the fresh forward happens outside.

Usage from an eval harness:

    from vllm.distributed.kv_transfer.kv_connector.v1 import xa_refinement

    # Precompute fresh K/V for request_id via your own HF forward.
    xa_refinement.register(
        request_id="req123",
        fresh_K_per_layer=fresh_K_list,   # list[Tensor], one per layer
        fresh_V_per_layer=fresh_V_list,
        hkvd_ratio=0.25,
        check_layer=1,
        reference_source="first_doc_tokens",
    )

    # Submit request to vLLM with kv_transfer_params carrying the
    # xa_ref_id matching the registered request_id. CartridgeConnector
    # reads the entry during start_load_kv.

    # After the request completes:
    xa_refinement.unregister("req123")

The side channel is process-local, so this works for `LLM.generate()`
usage (single-process) but will NOT propagate across multi-process
engine workers. For multi-process support see Phase X3 (Option 2) in
the design doc.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import torch


@dataclass
class XaEntry:
    """One queued xa refinement job keyed by request_id."""
    fresh_K_per_layer: list[torch.Tensor]
    fresh_V_per_layer: list[torch.Tensor]
    hkvd_ratio: float
    check_layer: int
    reference_source: Literal["first_doc_tokens", "init_tokens", "custom"]
    # Optional: per-layer KV mask for which cache positions to consider for
    # HKVD selection (e.g. restricting to cartridge-injected positions vs
    # LMCache-retrieved positions). None = consider all positions
    # in the paged cache that correspond to the fresh tensors.
    position_mask: torch.Tensor | None = None


_lock = threading.Lock()
_registry: dict[str, XaEntry] = {}


def register(
    request_id: str,
    fresh_K_per_layer: list[torch.Tensor],
    fresh_V_per_layer: list[torch.Tensor],
    hkvd_ratio: float = 0.25,
    check_layer: int = 1,
    reference_source: str = "first_doc_tokens",
    position_mask: torch.Tensor | None = None,
) -> None:
    """Register fresh K/V for a pending vLLM request.

    The connector will consult this registry during start_load_kv for
    the matching request_id (via `sampling_params.extra_args
    ["kv_transfer_params"]["cartridge_xa_ref_id"]` or equivalent hook
    in the connector implementation).
    """
    if not isinstance(hkvd_ratio, float) or not 0.0 < hkvd_ratio <= 1.0:
        raise ValueError(f"hkvd_ratio must be in (0.0, 1.0], got {hkvd_ratio}")
    if not isinstance(check_layer, int) or check_layer < 0:
        raise ValueError(
            f"check_layer must be non-negative int, got {check_layer}")
    if len(fresh_K_per_layer) != len(fresh_V_per_layer):
        raise ValueError(
            f"K/V layer count mismatch: {len(fresh_K_per_layer)} vs "
            f"{len(fresh_V_per_layer)}")
    entry = XaEntry(
        fresh_K_per_layer=fresh_K_per_layer,
        fresh_V_per_layer=fresh_V_per_layer,
        hkvd_ratio=hkvd_ratio,
        check_layer=check_layer,
        reference_source=reference_source,  # type: ignore[arg-type]
        position_mask=position_mask,
    )
    with _lock:
        _registry[request_id] = entry


def consume(request_id: str) -> XaEntry | None:
    """Atomically fetch and remove an xa entry for a request_id.

    Called by the connector during start_load_kv. Returns None if no
    entry is registered for this request_id.
    """
    with _lock:
        return _registry.pop(request_id, None)


def unregister(request_id: str) -> None:
    """Remove a registered xa entry without consuming it."""
    with _lock:
        _registry.pop(request_id, None)


def clear() -> None:
    """Drop all registered xa entries. Useful between evaluations."""
    with _lock:
        _registry.clear()


def size() -> int:
    """Current number of pending xa entries."""
    with _lock:
        return len(_registry)


def compute_hkvd_indices(
    cached_K_at_check: torch.Tensor,
    fresh_K_at_check: torch.Tensor,
    hkvd_ratio: float,
    refresh_policy: str = "hkvd",
    rng_seed: int = 42,
) -> torch.Tensor:
    """Identify HKVD positions by ||fresh_K - cache_K||^2 per token.

    Both tensors shape (..., L, head_dim). Returns sorted indices into
    [0, L) of shape (n_refresh,).

    Mirrors the HF harness implementation in eval_xattn_hybrid.py so
    vLLM and HF paths produce identical HKVD indices given identical
    inputs. Parity matters for validation.
    """
    import torch

    assert cached_K_at_check.shape == fresh_K_at_check.shape, (
        f"shape mismatch: cache={cached_K_at_check.shape} "
        f"fresh={fresh_K_at_check.shape}")
    L = cached_K_at_check.shape[-2]
    n_refresh = max(1, int(L * hkvd_ratio))

    if refresh_policy == "random":
        g = torch.Generator(
            device=cached_K_at_check.device).manual_seed(rng_seed)
        perm = torch.randperm(L, generator=g, device=cached_K_at_check.device)
        return perm[:n_refresh].sort().values

    # HKVD: top-K by squared L2 deviation, summed over head dims
    diff = (cached_K_at_check.float() - fresh_K_at_check.float())**2
    # Sum over everything except the token dim
    reduce_dims = list(range(diff.ndim))
    reduce_dims.remove(diff.ndim - 2)  # keep token dim
    per_token_diff = diff.sum(dim=reduce_dims)  # shape: (L,)
    top = torch.topk(per_token_diff, n_refresh).indices
    return top.sort().values
