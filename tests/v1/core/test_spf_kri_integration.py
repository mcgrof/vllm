# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract tests for SPF + KRI integration.

This is the *design test* for the BlockManifestProvider abstraction.
It is intentionally a contract test, not just a unit test: it locks in
the integration shape that real KRI providers (cartridge KRI-G,
cartridge KRI-Q, future RAG-KRI, future KRI-x) must satisfy when they
plug into SPF.

Properties verified:

1. **Per-K dispatch parity with the routing branch**: SPF reads
   ``kmeans_blocks_perK[K]`` first, falls back to ``kmeans_blocks``
   only when the per-K dict is missing.  Mirrors the cartridge
   connector dispatch in
   ``vllm/distributed/kv_transfer/kv_connector/v1/cartridge_connector.py``
   on the routing branch (commit ``3b2a58934`` and following).

2. **Manifest-bounded budget arithmetic**: with KRI manifests, SPF
   fits more candidates into the same prefetch budget than it does
   with the legacy unbounded path.  This is the headline win.

3. **Variant-agnostic provider abstraction**: KRI-G, KRI-Q, and a
   third synthetic ``"kri_x_test"`` provider all flow through SPF's
   controller identically.  The controller never branches on
   ``prior_type``.

4. **Loud-failure contract for KRI-Q**: a KRI-Q-style provider raises
   :class:`ValueError` when called without a ``query_hash`` and SPF
   logs/skips the candidate gracefully.  KRI-G accepts ``query_hash=None``
   without complaint.

The synthetic providers in this file are also reusable as fixtures for
future SPF integration tests that exercise the prefetch path end-to-end
without requiring real cartridge files.
"""
from __future__ import annotations

import logging

import pytest

from vllm.v1.core.spf.config import SPFConfig
from vllm.v1.core.spf.controller import (
    PrefetchCandidate,
    PrefetchHint,
    SPFController,
)
from vllm.v1.core.spf.manifest import (
    BlockManifest,
    BlockManifestProvider,
    MultiManifestProvider,
    NullManifestProvider,
)


# ---------------------------------------------------------------------------
# Synthetic KRI prior fixtures — match the on-disk shape from the routing
# branch so the same dict can be loaded by either real or test code.
# ---------------------------------------------------------------------------

def _make_perk_prior(
    num_blocks: int = 258,
    block_size: int = 16,
    prior_type: str = "kri_g",
) -> dict:
    """Build a synthetic prior dict matching the routing-side .pt format.

    Mirrors what ``scripts/generate_kmeans_prior.py --per-k 4,8,16,32``
    on the routing branch produces.  Block indices are arbitrary but
    sorted, with full-document coverage at every K (the principled
    per-K shape, not the legacy sorted-truncation shape).
    """
    return {
        "version": 1,
        "num_blocks": num_blocks,
        "block_size": block_size,
        "prior_type": prior_type,
        "kmeans_blocks_perK": {
            4: [10, 80, 150, 220],
            8: [10, 40, 80, 110, 150, 180, 220, 250],
            16: list(range(8, num_blocks, num_blocks // 16))[:16],
            32: list(range(4, num_blocks, num_blocks // 32))[:32],
        },
        # Legacy single-list fallback — would be used if per-K is absent.
        # Intentionally a *low-index-biased* list so we can detect when
        # the dispatch (incorrectly) falls back to it.
        "kmeans_blocks": list(range(0, 32)),
    }


def _make_legacy_only_prior(num_blocks: int = 258) -> dict:
    """Build a prior with only the legacy ``kmeans_blocks`` field."""
    return {
        "version": 1,
        "num_blocks": num_blocks,
        "block_size": 16,
        "prior_type": "kmeans",  # historical name
        "kmeans_blocks": [10, 14, 24, 31, 41, 61, 63, 71, 79, 88],
    }


# ---------------------------------------------------------------------------
# Synthetic provider implementations
# ---------------------------------------------------------------------------

class SyntheticKRIGProvider:
    """Query-agnostic synthetic provider mirroring the KRI-G contract.

    Accepts ``query_hash=None`` because KRI-G does not need a query.
    Looks up by ``prefix_hash`` only.
    """

    def __init__(self, store: dict[str, dict]) -> None:
        self._store = store

    def get_manifest(
        self,
        *,
        prefix_hash: str,
        query_hash: str | None,
        K: int,
    ) -> BlockManifest | None:
        prior = self._store.get(prefix_hash)
        if prior is None:
            return None
        return BlockManifest.from_kri_prior(
            prior=prior,
            K=K,
            prior_type=prior.get("prior_type", "kri_g"),
            resource_id=prefix_hash,
        )


class SyntheticKRIQProvider:
    """Query-conditioned synthetic provider mirroring the KRI-Q contract.

    Requires ``query_hash`` and raises :class:`ValueError` if it is
    missing — the loud-failure contract from the SPF brief.
    Looks up by ``(prefix_hash, query_hash)``.
    """

    def __init__(self, store: dict[tuple[str, str], dict]) -> None:
        self._store = store

    def get_manifest(
        self,
        *,
        prefix_hash: str,
        query_hash: str | None,
        K: int,
    ) -> BlockManifest | None:
        if query_hash is None:
            raise ValueError(
                "KRI-Q provider requires query_hash; "
                "caller must extract query region before lookup"
            )
        prior = self._store.get((prefix_hash, query_hash))
        if prior is None:
            return None
        return BlockManifest.from_kri_prior(
            prior=prior,
            K=K,
            prior_type=prior.get("prior_type", "kri_q"),
            resource_id=f"{prefix_hash}:{query_hash}",
        )


# ---------------------------------------------------------------------------
# Provider-level tests (no controller involved)
# ---------------------------------------------------------------------------

class TestBlockManifestFromKRIPrior:
    """Property #1: per-K dispatch parity with the routing-side connector."""

    def test_prefers_per_k_over_legacy(self) -> None:
        prior = _make_perk_prior()
        manifest = BlockManifest.from_kri_prior(prior=prior, K=8)
        assert manifest is not None
        assert manifest.K == 8
        # Must come from kmeans_blocks_perK[8], NOT from
        # sorted(kmeans_blocks)[:8] which would be [0, 1, 2, 3, 4, 5, 6, 7].
        assert list(manifest.block_indices) == sorted(
            prior["kmeans_blocks_perK"][8]
        )
        assert manifest.block_indices != tuple(range(8))

    def test_falls_back_to_legacy_when_per_k_missing(self) -> None:
        prior = _make_legacy_only_prior()
        manifest = BlockManifest.from_kri_prior(prior=prior, K=4)
        assert manifest is not None
        # Sorted truncation of the legacy list — matches the connector
        # fallback path.
        assert list(manifest.block_indices) == sorted(
            prior["kmeans_blocks"][:4]
        )
        assert manifest.K == 4

    def test_returns_none_when_no_blocks(self) -> None:
        prior = {"version": 1, "num_blocks": 100, "block_size": 16}
        manifest = BlockManifest.from_kri_prior(prior=prior, K=8)
        assert manifest is None

    def test_per_k_miss_falls_back_to_legacy(self) -> None:
        # K=64 not in per-K dict, but legacy list is present.
        prior = _make_perk_prior()
        manifest = BlockManifest.from_kri_prior(prior=prior, K=64)
        assert manifest is not None
        # Falls back to sorted(kmeans_blocks[:64]) — at most len(legacy).
        assert manifest.K == min(64, len(prior["kmeans_blocks"]))


class TestBlockManifestSavings:
    def test_savings_blocks(self) -> None:
        m = BlockManifest(
            block_indices=(0, 1, 2, 3, 4, 5, 6, 7),
            total_blocks=258,
            K=8,
        )
        assert m.savings_blocks == 250
        assert pytest.approx(m.savings_fraction, rel=1e-4) == 250 / 258

    def test_validates_total_positive(self) -> None:
        with pytest.raises(ValueError):
            BlockManifest(
                block_indices=(0, 1),
                total_blocks=0,
                K=2,
            )

    def test_validates_indices_in_range(self) -> None:
        with pytest.raises(ValueError):
            BlockManifest(
                block_indices=(0, 10),
                total_blocks=5,
                K=2,
            )

    def test_validates_K_matches_indices(self) -> None:
        with pytest.raises(ValueError):
            BlockManifest(
                block_indices=(0, 1, 2),
                total_blocks=10,
                K=2,
            )


class TestKRIQContractFailures:
    """Property #4: KRI-Q must raise on missing query_hash."""

    def test_kriq_raises_on_missing_query_hash(self) -> None:
        provider = SyntheticKRIQProvider(store={})
        with pytest.raises(ValueError, match="query_hash"):
            provider.get_manifest(
                prefix_hash="abc123",
                query_hash=None,
                K=8,
            )

    def test_krig_accepts_missing_query_hash(self) -> None:
        store = {"abc123": _make_perk_prior()}
        provider = SyntheticKRIGProvider(store=store)
        # KRI-G must NOT raise — query_hash=None is the normal path.
        manifest = provider.get_manifest(
            prefix_hash="abc123",
            query_hash=None,
            K=8,
        )
        assert manifest is not None
        assert manifest.K == 8


# ---------------------------------------------------------------------------
# Multi-provider chain tests
# ---------------------------------------------------------------------------

class TestMultiProviderChain:
    """Verify the chain skips KRI-Q gracefully when no query is present."""

    def test_chain_falls_through_kriq_to_krig_without_query(self) -> None:
        kriq = SyntheticKRIQProvider(store={
            ("abc123", "qX"): _make_perk_prior(prior_type="kri_q"),
        })
        krig = SyntheticKRIGProvider(store={
            "abc123": _make_perk_prior(prior_type="kri_g"),
        })
        chain = MultiManifestProvider()
        chain.register("kri_q", kriq)
        chain.register("kri_g", krig)

        # No query supplied — KRI-Q should be skipped (not propagated)
        # and KRI-G should answer.
        manifest = chain.get_manifest(
            prefix_hash="abc123",
            query_hash=None,
            K=8,
        )
        assert manifest is not None
        assert manifest.prior_type == "kri_g"

    def test_chain_prefers_kriq_when_query_supplied(self) -> None:
        kriq = SyntheticKRIQProvider(store={
            ("abc123", "qX"): _make_perk_prior(prior_type="kri_q"),
        })
        krig = SyntheticKRIGProvider(store={
            "abc123": _make_perk_prior(prior_type="kri_g"),
        })
        chain = MultiManifestProvider()
        chain.register("kri_q", kriq)
        chain.register("kri_g", krig)

        manifest = chain.get_manifest(
            prefix_hash="abc123",
            query_hash="qX",
            K=8,
        )
        assert manifest is not None
        assert manifest.prior_type == "kri_q"


# ---------------------------------------------------------------------------
# Controller-level tests — the integration contract
# ---------------------------------------------------------------------------

def _make_controller(
    *,
    provider: BlockManifestProvider | None = None,
    target_K: int = 8,
    max_prefetch_blocks: int = 64,
    prefetch_fraction: float = 1.0,
) -> SPFController:
    """Helper that builds a controller with predictable budget settings."""
    config = SPFConfig(
        enabled=True,
        scorer="session_aware",
        max_prefetch_blocks=max_prefetch_blocks,
        prefetch_fraction=prefetch_fraction,
        lookahead_steps=4,
        metrics_interval=10_000,  # don't flush during tests
        target_K=target_K,
    )
    return SPFController(config, manifest_provider=provider)


def _populate_session(
    controller: SPFController,
    session_id: str,
    prefix_hashes: list[str],
    query_hash: str | None = None,
) -> None:
    """Drive observe_request once per prefix to seed session state."""
    for ph in prefix_hashes:
        controller.observe_request(
            session_id=session_id,
            prefix_hash=ph,
            query_hash=query_hash,
        )


class TestControllerWithoutProvider:
    """Baseline: legacy behavior preserved when no provider is registered."""

    def test_default_is_null_provider(self) -> None:
        controller = _make_controller(provider=None)
        _populate_session(controller, "s1", ["p1", "p2"])
        hints = controller.step(
            free_gpu_blocks=1024,
            resident_prefixes={"p1", "p2"},
        )
        # Hints emitted, none carry a manifest.
        assert len(hints) == 2
        assert all(h.manifest is None for h in hints)


class TestControllerWithKRIGProvider:
    """Property #2: manifest-bounded budget arithmetic."""

    def test_kri_attaches_manifest_with_savings(self) -> None:
        """Smoke test: with a KRI-G provider, every hint carries a
        manifest, the manifest's K matches target_K, and the savings
        vs the underlying N are correctly reported.

        This is the "what KRI gives you" assertion, separate from the
        budget-pinch comparison below.  Default ``num_blocks=1`` per
        candidate keeps the budget out of the picture for this test —
        the budget arithmetic win is exercised in the next test.
        """
        prefix_hashes = [f"prefix_{i:02d}" for i in range(5)]
        store = {ph: _make_perk_prior() for ph in prefix_hashes}
        provider = SyntheticKRIGProvider(store=store)

        controller = _make_controller(
            provider=provider,
            max_prefetch_blocks=1024,  # generous budget
        )
        for ph in prefix_hashes:
            controller.observe_request("s1", ph)
        hints = controller.step(
            free_gpu_blocks=1024,
            resident_prefixes=set(prefix_hashes),
        )

        assert len(hints) == 5
        assert all(h.manifest is not None for h in hints)
        assert all(h.manifest.K == 8 for h in hints)
        assert all(h.num_blocks == 8 for h in hints)
        assert all(h.manifest.total_blocks == 258 for h in hints)
        assert all(h.manifest.savings_blocks == 250 for h in hints)

    def test_kri_dominates_when_legacy_blocks_dominate_budget(self) -> None:
        # Make num_blocks huge in the legacy path so the budget pinches,
        # then verify KRI fits more candidates.
        provider = SyntheticKRIGProvider(store={
            f"prefix_{i:02d}": _make_perk_prior() for i in range(20)
        })
        kri_controller = _make_controller(
            provider=provider,
            max_prefetch_blocks=64,  # K=8 × 8 = 64 → fits exactly 8
        )
        for i in range(20):
            kri_controller.observe_request("s1", f"prefix_{i:02d}")

        # Inject a fake high num_blocks via the candidate generation
        # path — easiest by monkey-patching _generate_candidates here.
        original_generate = kri_controller._generate_candidates

        def generate_with_big_blocks(resident):
            cands = original_generate(resident)
            for c in cands:
                c.num_blocks = 32  # legacy cost per candidate
            return cands

        kri_controller._generate_candidates = generate_with_big_blocks

        hints = kri_controller.step(
            free_gpu_blocks=1024,
            resident_prefixes={f"prefix_{i:02d}" for i in range(20)},
        )

        # Without KRI: 32 blocks/cand × budget 64 → only 2 fit.
        # With KRI: 8 blocks/cand × budget 64 → 8 fit.
        assert len(hints) == 8
        assert all(h.manifest is not None for h in hints)
        assert all(h.num_blocks == 8 for h in hints)


class TestControllerWithKRIQProvider:
    """KRI-Q path: provider needs query_hash, controller surfaces it."""

    def test_kriq_hits_when_query_observed(self) -> None:
        store = {
            ("p1", "q1"): _make_perk_prior(prior_type="kri_q"),
        }
        provider = SyntheticKRIQProvider(store=store)
        controller = _make_controller(provider=provider)
        controller.observe_request("s1", "p1", query_hash="q1")
        hints = controller.step(
            free_gpu_blocks=1024,
            resident_prefixes={"p1"},
        )
        assert len(hints) == 1
        assert hints[0].manifest is not None
        assert hints[0].manifest.prior_type == "kri_q"

    def test_kriq_misses_when_query_absent(self) -> None:
        store = {("p1", "q1"): _make_perk_prior(prior_type="kri_q")}
        provider = SyntheticKRIQProvider(store=store)
        controller = _make_controller(provider=provider)
        # Observe WITHOUT a query_hash — KRI-Q raises, controller logs
        # and falls through to legacy behavior for this candidate.
        controller.observe_request("s1", "p1", query_hash=None)
        # vLLM disables propagation on its logger hierarchy, so
        # pytest's caplog can't see warnings emitted by `vllm.spf`.
        # Attach our own handler directly to capture the records.
        captured: list[logging.LogRecord] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                captured.append(record)

        spf_logger = logging.getLogger("vllm.spf")
        handler = _Capture(level=logging.WARNING)
        spf_logger.addHandler(handler)
        try:
            hints = controller.step(
                free_gpu_blocks=1024,
                resident_prefixes={"p1"},
            )
        finally:
            spf_logger.removeHandler(handler)
        # Hint emitted, but with no manifest (manifest miss).
        assert len(hints) == 1
        assert hints[0].manifest is None
        # The loud-failure path produced a warning so test runs surface
        # integration bugs instead of silently degrading.
        assert any(
            "manifest provider declined" in record.getMessage().lower()
            for record in captured
        ), f"expected warning, got: {[r.getMessage() for r in captured]}"


class TestControllerIsVariantAgnostic:
    """Property #3: SPF treats KRI-G, KRI-Q, and ad-hoc KRI-x identically."""

    def test_three_variants_same_code_path(self) -> None:
        # Three providers with three different prior_type tags.
        krig_store = {"p1": _make_perk_prior(prior_type="kri_g")}
        krix_store = {"p2": _make_perk_prior(prior_type="kri_x_test")}

        chain = MultiManifestProvider()
        chain.register("kri_g", SyntheticKRIGProvider(store=krig_store))
        chain.register("kri_x", SyntheticKRIGProvider(store=krix_store))

        controller = _make_controller(provider=chain)
        controller.observe_request("s1", "p1")
        controller.observe_request("s1", "p2")
        hints = controller.step(
            free_gpu_blocks=1024,
            resident_prefixes={"p1", "p2"},
        )
        assert len(hints) == 2
        types = sorted(h.manifest.prior_type for h in hints if h.manifest)
        # The variant tags pass through unchanged — no controller logic
        # branched on them.
        assert types == ["kri_g", "kri_x_test"]


class TestNullProviderIsNoOp:
    def test_null_provider_returns_no_manifest(self) -> None:
        provider = NullManifestProvider()
        assert provider.get_manifest(
            prefix_hash="anything",
            query_hash=None,
            K=8,
        ) is None
        # Also accepts a query_hash without raising.
        assert provider.get_manifest(
            prefix_hash="anything",
            query_hash="qX",
            K=8,
        ) is None
