# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cartridge KRI provider — reads routing_prior_kmeans*.pt files.

This is the first production :class:`BlockManifestProvider` for SPF.
It consumes the on-disk prior format that the routing team's
``scripts/generate_kmeans_prior.py --per-k`` flow produces, mirrored
into the SPF branch via the squash-pick at commit ``4ae52f638``.

Scope (what this provider does):

* Loads one or more ``routing_prior_kmeans*.pt`` files from disk at
  registration time and caches the contents in memory.  No per-request
  disk I/O.
* Maps each prior to a ``prefix_hash`` supplied by the caller.  SPF's
  ingestion boundary computes ``prefix_hash`` from the prompt; the
  provider does not invent its own keying scheme.
* Returns :class:`BlockManifest` objects via the standard
  ``from_kri_prior`` parser, so the dispatch logic stays in the
  manifest module instead of being copy-pasted here.
* Accepts ``query_hash=None`` (KRI-G is query-agnostic).  KRI-Q
  selections also live in this same on-disk shape — see the per-query
  registration helper for the hybrid case where the same provider
  serves both query-keyed and query-agnostic priors.

Scope (what this provider does NOT do):

* Generate priors.  Run ``scripts/generate_kmeans_prior.py --per-k``
  for that.
* Compute ``prefix_hash``.  That's the SPF bridge's job
  (``_derive_session_id`` / first-block-hash logic).
* Translate manifest block indices into physical GPU block IDs.  That
  is the cartridge connector's slot-mapping path on the routing side.
* Validate cartridge content against the prior.  We trust the
  ``num_blocks`` field in the prior dict.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch

from vllm.v1.core.spf.manifest import BlockManifest, BlockManifestProvider

logger = logging.getLogger("vllm.spf.providers.cartridge_kri")


@dataclass
class _CartridgeKRIEntry:
    """Cached prior for one (prefix_hash, query_hash) pair."""

    prefix_hash: str
    query_hash: str | None
    prior: dict
    source_path: str
    prior_type: str


class CartridgeKRIProvider:
    """Serve KRI manifests from on-disk routing_prior_kmeans*.pt files.

    Lifecycle::

        provider = CartridgeKRIProvider()
        provider.register_prior(
            prefix_hash="abc123",
            prior_path="/path/to/routing_prior_kmeans_perK_16384.pt",
            prior_type="kri_g",
        )
        # Optionally for KRI-Q, register multiple priors against the
        # same prefix_hash with distinct query_hash keys:
        provider.register_prior(
            prefix_hash="abc123",
            prior_path="/path/to/routing_prior_offlineq_16384_warranty.pt",
            prior_type="kri_q",
            query_hash="qwarranty",
        )

        controller = SPFController(config, manifest_provider=provider)

    Lookup::

        get_manifest(prefix_hash="abc123", query_hash=None, K=8)
            → KRI-G manifest from the first prior, query_hash ignored
        get_manifest(prefix_hash="abc123", query_hash="qwarranty", K=8)
            → KRI-Q manifest from the second prior

    Lookup precedence within a single ``prefix_hash``:

    1. Exact ``(prefix_hash, query_hash)`` match — used for KRI-Q.
    2. Query-agnostic fallback ``(prefix_hash, None)`` — used for KRI-G.
    3. ``None`` if neither is registered → manifest miss → SPF falls
       back to legacy unbounded prefetch for this candidate.

    Note: this provider does NOT raise on missing ``query_hash``.  It
    is permissive (KRI-G semantics) by default because it primarily
    serves KRI-G priors.  Wrap it in a strict KRI-Q-only adapter if a
    deployment needs the loud-failure contract for query-agnostic
    callers.
    """

    def __init__(self) -> None:
        # (prefix_hash, query_hash) -> entry
        # query_hash=None is the KRI-G slot for a given prefix.
        self._store: dict[tuple[str, str | None], _CartridgeKRIEntry] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_prior(
        self,
        *,
        prefix_hash: str,
        prior_path: str | Path,
        prior_type: str = "kri_g",
        query_hash: str | None = None,
    ) -> _CartridgeKRIEntry:
        """Load a prior file and bind it to ``(prefix_hash, query_hash)``.

        ``prior_type`` is informational and propagates onto the emitted
        manifest's ``prior_type`` field for logs and metrics.  SPF does
        not branch on it.

        Returns the cached entry for inspection (mostly so test code
        can assert on the loaded shape).
        """
        path = Path(prior_path)
        prior = torch.load(str(path), map_location="cpu", weights_only=False)
        if not isinstance(prior, dict):
            raise ValueError(
                f"Routing prior at {path} is not a dict "
                f"(got {type(prior).__name__})"
            )
        # Validate that the dispatch path will actually find
        # something. The manifest builder accepts
        # kmeans_blocks_perK, legacy kmeans_blocks, or a continuous
        # block_affinities tensor (KRI-D-kv-sum style — top-K is
        # computed on the fly via torch.topk).
        if (
            "kmeans_blocks_perK" not in prior
            and "kmeans_blocks" not in prior
            and prior.get("block_affinities") is None
        ):
            raise ValueError(
                f"Routing prior at {path} has neither "
                f"kmeans_blocks_perK, kmeans_blocks, nor block_affinities"
            )
        entry = _CartridgeKRIEntry(
            prefix_hash=prefix_hash,
            query_hash=query_hash,
            prior=prior,
            source_path=str(path),
            prior_type=prior_type,
        )
        self._store[(prefix_hash, query_hash)] = entry
        per_k_keys = sorted(
            (prior.get("kmeans_blocks_perK") or {}).keys()
        )
        legacy_len = (
            len(prior["kmeans_blocks"])
            if "kmeans_blocks" in prior else 0
        )
        logger.info(
            "CartridgeKRI registered prefix=%s query=%s type=%s "
            "from %s (per-K=%s, legacy=%d, num_blocks=%s)",
            prefix_hash[:8],
            query_hash if query_hash else "(none)",
            prior_type,
            path.name,
            per_k_keys,
            legacy_len,
            prior.get("num_blocks"),
        )
        return entry

    def register_directory(
        self,
        *,
        directory: str | Path,
        prefix_hash: str,
        glob: str = "routing_prior_kmeans*.pt",
        prior_type: str = "kri_g",
    ) -> int:
        """Convenience: register every prior matching ``glob`` in a
        directory under one ``prefix_hash``.

        Used in benchmarks where one cartridge has several K-specific
        priors all sitting next to each other.  Returns the number of
        files registered.
        """
        directory = Path(directory)
        count = 0
        for pt in sorted(directory.glob(glob)):
            self.register_prior(
                prefix_hash=prefix_hash,
                prior_path=pt,
                prior_type=prior_type,
            )
            count += 1
        return count

    # ------------------------------------------------------------------
    # Provider interface
    # ------------------------------------------------------------------

    def get_manifest(
        self,
        *,
        prefix_hash: str,
        query_hash: str | None,
        K: int,
    ) -> BlockManifest | None:
        # Try the exact (prefix, query) key first.  KRI-Q uses this.
        entry = self._store.get((prefix_hash, query_hash))
        if entry is None and query_hash is not None:
            # KRI-G fallback: same prefix, query-agnostic prior.
            entry = self._store.get((prefix_hash, None))
        if entry is None:
            return None
        return BlockManifest.from_kri_prior(
            prior=entry.prior,
            K=K,
            prior_type=entry.prior_type,
            resource_id=entry.source_path,
        )

    # ------------------------------------------------------------------
    # Introspection (for test/debug only)
    # ------------------------------------------------------------------

    @property
    def registered_keys(self) -> list[tuple[str, str | None]]:
        return list(self._store.keys())

    def stats(self) -> dict:
        """Return a small dict summarizing what's loaded — used in logs."""
        return {
            "num_priors": len(self._store),
            "prefix_hashes": sorted({k[0] for k in self._store}),
            "prior_types": sorted({e.prior_type for e in self._store.values()}),
        }
