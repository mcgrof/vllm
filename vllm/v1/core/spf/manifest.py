# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SPF block-manifest abstraction.

A *block manifest* is a deterministic claim of the form "out of N blocks
in this KV resource, only these K blocks matter."  SPF treats manifests
as an opaque optimization on top of its prefetch decisions: when a
manifest is available, SPF can land a prediction for K-blocks worth of
budget instead of N-blocks worth, making the prefetch budget go N/K
times further at the same prediction quality.

The canonical source of manifests today is the **KRI family** (Key
Routing Index — see /data/plans/briefs/kri-state-and-scope-2026-04-06.md
for the full taxonomy):

* **KRI-G** — Geometric, K-means on cartridge block-mean Keys.
  Query-agnostic.  Precomputed once per cartridge.
* **KRI-Q** — Query-conditioned, model's W_q applied to the actual user
  query, scored against block-mean Keys.  Precomputed per (cartridge,
  query).  No training.
* **KRI-x** — Open-ended placeholder for future variants (multi-layer Q,
  batch routing, learned re-ranking, temporal routing for streaming, …).

SPF deliberately knows **none** of this.  The whole point of this module
is to keep SPF's controller variant-agnostic: any future routing
primitive — RAG-KRI, learned KRI, or anything else — that can answer
"give me the K blocks that matter for this resource" plugs in by
implementing :class:`BlockManifestProvider` and emitting
:class:`BlockManifest` objects in the same shape that the routing-side
cartridge connector reads from disk (``kmeans_blocks_perK[K]`` with
``kmeans_blocks`` as a legacy fallback).

Layering rules (do not violate):

1. SPF's scoring loop never sees ``query_hash`` — scoring stays
   prefix-based so multi-turn sessions still cluster correctly.
2. SPF computes ``query_hash`` at the request-ingestion boundary, not
   inside the controller, and passes it through to providers as an
   opaque string.
3. Providers decide their own keying precision.  KRI-G ignores
   ``query_hash``; KRI-Q must require it.
4. The provider abstraction is variant-agnostic: callers do not branch
   on ``prior_type``.  The string is informational only (logging,
   metrics).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class BlockManifest:
    """A KRI-bounded slice of a larger KV resource.

    Mirrors the on-disk routing-prior format used by
    ``vllm/distributed/kv_transfer/kv_connector/v1/cartridge_connector.py``
    on the routing branch (commit ``3b2a58934`` and following).  The
    fields here are the strict subset SPF needs at scheduling time;
    providers may carry additional state internally.

    Attributes:
        block_indices: Sorted block indices in the K slice
            (``0..total_blocks-1``).  Length equals ``K``.
        total_blocks: ``N`` — the total number of blocks in the
            underlying resource.  Used for savings reporting.
        K: Convenience: ``len(block_indices)``.  Stored explicitly so
            metrics can read it without re-measuring.
        prior_type: Informational tag identifying the variant that
            produced this manifest (``"kri_g"``, ``"kri_q"``,
            ``"synthetic"``, future ``"kri_x_*"``).  SPF must NOT
            branch on this — it exists for logs and metrics only.
        resource_id: Opaque identifier for the underlying KV resource
            (cartridge name, document hash, RAG corpus key, …).  SPF
            uses it only for log lines.
    """

    block_indices: tuple[int, ...]
    total_blocks: int
    K: int
    prior_type: str = "unknown"
    resource_id: str = ""

    def __post_init__(self) -> None:
        # Defensive checks — these are cheap and catch malformed
        # manifests at the scheduler boundary instead of letting them
        # poison the prefetch queue.
        if self.K != len(self.block_indices):
            raise ValueError(
                f"BlockManifest K={self.K} disagrees with "
                f"len(block_indices)={len(self.block_indices)}"
            )
        if self.total_blocks <= 0:
            raise ValueError(
                f"BlockManifest total_blocks must be positive, "
                f"got {self.total_blocks}"
            )
        if any(b < 0 or b >= self.total_blocks for b in self.block_indices):
            raise ValueError(
                f"BlockManifest block_indices out of range "
                f"[0, {self.total_blocks}): {self.block_indices}"
            )

    @property
    def savings_blocks(self) -> int:
        """How many blocks SPF avoids prefetching by honoring this manifest."""
        return self.total_blocks - self.K

    @property
    def savings_fraction(self) -> float:
        """``savings_blocks / total_blocks``, in ``[0.0, 1.0]``."""
        if self.total_blocks == 0:
            return 0.0
        return self.savings_blocks / self.total_blocks

    @classmethod
    def from_kri_prior(
        cls,
        *,
        prior: dict,
        K: int,
        prior_type: str | None = None,
        resource_id: str = "",
    ) -> "BlockManifest | None":
        """Build a manifest from an on-disk KRI prior dict.

        Mirrors the dispatch logic in the routing-side cartridge
        connector: prefer ``kmeans_blocks_perK[K]`` (principled per-K
        priors), fall back to ``kmeans_blocks`` truncated to K (legacy
        priors generated before the per-K dispatch fix).

        Returns ``None`` if neither key is present — that signals "this
        prior cannot serve K=<K>" and the caller should treat the
        request as a manifest miss (full prefetch).
        """
        per_k = prior.get("kmeans_blocks_perK")
        legacy = prior.get("kmeans_blocks")

        block_list: list[int] | None = None
        if per_k is not None and K in per_k:
            block_list = sorted(per_k[K])
        elif legacy is not None:
            # Legacy fallback: sorted truncation.  Mirrors the
            # connector dispatch in cartridge_connector.py — known to
            # be lossy compared to per-K priors but acceptable for
            # back-compat with older .pt files.
            block_list = sorted(legacy[: min(K, len(legacy))])

        if block_list is None:
            return None

        total = int(prior.get("num_blocks", max(block_list) + 1))
        return cls(
            block_indices=tuple(block_list),
            total_blocks=total,
            K=len(block_list),
            prior_type=prior_type or str(prior.get("prior_type", "unknown")),
            resource_id=resource_id,
        )


@runtime_checkable
class BlockManifestProvider(Protocol):
    """Resolve a (resource, query) pair into a K-bounded block manifest.

    Providers are the integration seam between SPF and any source of
    deterministic block manifests — KRI-G, KRI-Q, RAG-KRI, learned
    KRI-x variants, etc.  SPF instantiates one provider (or a chain),
    asks it for manifests at scheduling time, and stays otherwise
    ignorant of the variant.

    Keying contract:

    * ``prefix_hash`` — SPF's existing short prefix fingerprint,
      identifying the underlying KV resource (a cartridge, a RAG
      corpus, a document blob).  Always present.
    * ``query_hash`` — Optional opaque hash of the query region of the
      prompt.  SPF computes this at the request-ingestion boundary
      (NOT inside the scoring loop) and passes it through.  Providers
      that do not need it (KRI-G) must accept ``None``.  Providers
      that require it (KRI-Q) must raise :class:`ValueError` on
      ``None`` so caller bugs surface loudly instead of silently
      degrading to a cache miss.
    * ``K`` — The block count SPF wants.  Providers may return ``None``
      if they have no manifest at exactly this K (and SPF will treat
      the candidate as a manifest miss → fall back to full prefetch).

    Return value:

    * :class:`BlockManifest` — the K-bounded slice for this request.
    * ``None`` — manifest miss (no prior on file for this resource, or
      no per-K entry at the requested K).  SPF treats this as legacy
      prefetch behavior.
    """

    def get_manifest(
        self,
        *,
        prefix_hash: str,
        query_hash: str | None,
        K: int,
    ) -> BlockManifest | None:
        ...


class NullManifestProvider:
    """Provider that always reports a manifest miss.

    Used as the default when no real provider has been registered, so
    SPF callsites do not need ``provider is not None`` guards.  Also
    useful as a baseline arm in A/B comparisons (KRI off vs KRI on).
    """

    def get_manifest(
        self,
        *,
        prefix_hash: str,
        query_hash: str | None,
        K: int,
    ) -> BlockManifest | None:
        return None


@dataclass
class _MultiProviderEntry:
    name: str
    provider: BlockManifestProvider


class MultiManifestProvider:
    """Chain multiple providers, returning the first hit.

    Order matters: list more specific providers first (e.g. KRI-Q
    before KRI-G), so a query-conditioned hit takes precedence over
    a query-agnostic one for the same resource.  Providers that raise
    on missing ``query_hash`` are *skipped* (not propagated) when the
    caller did not supply a query — this lets a chain mix KRI-G and
    KRI-Q gracefully without forcing every caller to pass a query.
    """

    def __init__(self) -> None:
        self._entries: list[_MultiProviderEntry] = []

    def register(self, name: str, provider: BlockManifestProvider) -> None:
        self._entries.append(_MultiProviderEntry(name=name, provider=provider))

    def get_manifest(
        self,
        *,
        prefix_hash: str,
        query_hash: str | None,
        K: int,
    ) -> BlockManifest | None:
        for entry in self._entries:
            try:
                manifest = entry.provider.get_manifest(
                    prefix_hash=prefix_hash,
                    query_hash=query_hash,
                    K=K,
                )
            except ValueError:
                # Provider required query_hash but caller didn't have
                # one — skip this provider, try the next.  This is the
                # "graceful chain" behavior: a KRI-Q provider declining
                # because no query was extracted is not an error, it's
                # a no-match, and the chain should fall through to a
                # KRI-G provider that can serve the same resource.
                continue
            if manifest is not None:
                return manifest
        return None
