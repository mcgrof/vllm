# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diagnostic test for the first-block-only identity bug.

Scenario:

  Many requests share an identical first block (e.g. a common chat
  template or system prompt prefix) but diverge arbitrarily after
  the first 16 tokens. Under the **legacy** first-block-only
  identity, the policy cannot distinguish them — they all collapse
  to one identity, outcomes cross-contaminate, and the scorer is
  effectively lying about what it observes.

  Under the honest :class:`ResourceId` — which hashes the *full*
  reusable region — each request produces a distinct resource_id
  and the scorer / outcome tracker can tell them apart.

This test exists so the bug is never quietly reintroduced. It is
referenced by name from the phase-1 design note and from any future
identity refactor.
"""
from __future__ import annotations

from vllm.v1.core.spf.resource import (DEFAULT_BLOCK_SIZE, ResourceId,
                                       hash_prefix_tokens,
                                       legacy_first_block_hash)

# A realistic system-prompt-sized shared prefix. 128 tokens is a
# common chat-template preamble size. The test does not depend on
# the specific values — only on the divergence structure.
SHARED_PREFIX = list(range(128))


def _divergent_prompts(n: int, tail_base: int) -> list[list[int]]:
    """Build ``n`` prompts sharing the same prefix + distinct tails."""
    return [
        SHARED_PREFIX + [tail_base + i, tail_base + i + 1, tail_base + i + 2]
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Diagnosis: what goes wrong under the legacy identity
# ---------------------------------------------------------------------------


class TestLegacyIdentityCollapses:
    """Show that the legacy first-block-only hash collapses many
    distinct prompts into a single identity. This is the baseline
    the honest identity is supposed to improve on."""

    def test_legacy_collapses_128_token_prefix(self):
        prompts = _divergent_prompts(n=50, tail_base=10_000)
        legacy_ids = {
            legacy_first_block_hash(p, block_size=DEFAULT_BLOCK_SIZE)
            for p in prompts
        }
        # All 50 collapse to the same legacy id — this is the bug.
        assert len(legacy_ids) == 1, (
            f"legacy identity should collapse all {len(prompts)} "
            f"prompts; got {len(legacy_ids)} distinct ids")

    def test_legacy_does_not_distinguish_divergent_tails(self):
        """Even a single-token divergence after the first block
        leaves the legacy id unchanged."""
        a = SHARED_PREFIX + [1]
        b = SHARED_PREFIX + [2]
        assert legacy_first_block_hash(
            a, block_size=DEFAULT_BLOCK_SIZE) == legacy_first_block_hash(
                b, block_size=DEFAULT_BLOCK_SIZE)


# ---------------------------------------------------------------------------
# Fix: the honest identity distinguishes them
# ---------------------------------------------------------------------------


class TestHonestIdentityDistinguishes:
    """Show that :func:`hash_prefix_tokens` — the function the new
    :class:`ResourceId` uses — does the right thing on the same
    workload."""

    def test_honest_identity_distinguishes_all_50(self):
        prompts = _divergent_prompts(n=50, tail_base=10_000)
        honest_ids = {hash_prefix_tokens(p) for p in prompts}
        assert len(honest_ids) == 50, (
            f"honest identity should give {len(prompts)} distinct "
            f"resource_ids; got {len(honest_ids)}")

    def test_honest_distinguishes_single_token_divergence(self):
        a = SHARED_PREFIX + [1]
        b = SHARED_PREFIX + [2]
        assert hash_prefix_tokens(a) != hash_prefix_tokens(b)

    def test_resource_id_wrapper_also_distinguishes(self):
        """Going through the :class:`ResourceId.for_prefix`
        factory (the path real callers will use) preserves the
        distinguishing property."""
        prompts = _divergent_prompts(n=20, tail_base=500)
        rids = {
            ResourceId.for_prefix(p, session_id="s").resource_id
            for p in prompts
        }
        assert len(rids) == 20

    def test_same_prompt_same_id_across_sessions(self):
        """Two sessions asking for the same prompt share a
        resource_id (the resource IS the prefix). Session identity
        lives separately on the struct."""
        p = SHARED_PREFIX + [9, 9, 9]
        a = ResourceId.for_prefix(p, session_id="alpha")
        b = ResourceId.for_prefix(p, session_id="beta")
        assert a.resource_id == b.resource_id
        assert a.session_id != b.session_id


# ---------------------------------------------------------------------------
# Side-by-side: the honest identity and the legacy field coexist
# ---------------------------------------------------------------------------


class TestCoexistence:

    def test_resource_id_struct_carries_both(self):
        p = SHARED_PREFIX + [42]
        r = ResourceId.for_prefix(p, session_id="s")
        # Resource id is full-region.
        assert r.resource_id == hash_prefix_tokens(p)
        # Legacy first-block hash is carried verbatim.
        assert r.first_block_hash == legacy_first_block_hash(p)

    def test_two_prompts_same_legacy_hash_different_resource_id(self):
        """This is the invariant the rest of SPF depends on.

        If you ever see this assertion fail, the identity has
        regressed and scoring / outcome tracking are lying again.
        """
        a_tokens = SHARED_PREFIX + [1]
        b_tokens = SHARED_PREFIX + [2]
        a = ResourceId.for_prefix(a_tokens, session_id="s")
        b = ResourceId.for_prefix(b_tokens, session_id="s")

        # Legacy: same (the bug).
        assert a.first_block_hash == b.first_block_hash
        # Honest: different (the fix).
        assert a.resource_id != b.resource_id
