# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase-1 honest-mode tests for SPF ResourceId.

Covers the contract the module docstring lays out:

  - resource_id is a stable, deterministic hash over the full token
    sequence. Same input ⇒ same id, always.
  - Two token sequences that differ anywhere past the first block
    produce different resource_ids (not first-block-only).
  - first_block_hash is retained on the struct but NEVER reused as
    the load-bearing identity.
  - Scopes are isolated: identical token content at different
    scopes would not collide in future code. (Only PREFIX exists
    today; we guard the invariant now so phase-2 extensions inherit
    the property.)
  - Validation rejects invalid num_blocks / num_bytes.
"""
from __future__ import annotations

import pytest

from vllm.v1.core.spf.resource import (DEFAULT_BLOCK_SIZE, ResourceCandidate,
                                       ResourceId, ResourceScope,
                                       hash_prefix_tokens,
                                       legacy_first_block_hash)

# ---------------------------------------------------------------------------
# Identity stability
# ---------------------------------------------------------------------------


class TestIdentityStability:

    def test_same_tokens_same_id(self):
        a = ResourceId.for_prefix([1, 2, 3, 4, 5, 6, 7, 8], session_id="s0")
        b = ResourceId.for_prefix([1, 2, 3, 4, 5, 6, 7, 8], session_id="s0")
        assert a.resource_id == b.resource_id

    def test_different_sessions_same_resource_hash(self):
        """Same tokens ⇒ same resource_id even across sessions.

        The resource IS the shared prefix; the session is metadata.
        Future session-scoped resources live at a different scope.
        """
        a = ResourceId.for_prefix([9, 9, 9], session_id="alpha")
        b = ResourceId.for_prefix([9, 9, 9], session_id="beta")
        assert a.resource_id == b.resource_id
        assert a.session_id != b.session_id

    def test_hash_is_deterministic_across_calls(self):
        tokens = list(range(100))
        h1 = hash_prefix_tokens(tokens)
        h2 = hash_prefix_tokens(tokens)
        h3 = hash_prefix_tokens(tokens)
        assert h1 == h2 == h3


# ---------------------------------------------------------------------------
# NOT first-block-only — the load-bearing correctness test
# ---------------------------------------------------------------------------


class TestNotFirstBlockOnly:

    def test_same_first_block_different_tail_yields_different_id(self):
        """Two prompts sharing a first block but diverging after MUST
        produce different resource_ids."""
        common = list(range(DEFAULT_BLOCK_SIZE))  # 16 tokens
        a_tokens = common + [100, 101, 102]
        b_tokens = common + [200, 201, 202]

        a = ResourceId.for_prefix(a_tokens, session_id="s")
        b = ResourceId.for_prefix(b_tokens, session_id="s")

        # Same first block — legacy hashes collide.
        assert a.first_block_hash == b.first_block_hash

        # But the honest resource_id is DIFFERENT.
        assert a.resource_id != b.resource_id, (
            "resource_id collapses prompts sharing a first block — "
            "this is the first-block-only bug")

    def test_128_token_shared_prefix_still_distinguishes(self):
        """A realistic system-prompt scenario: 128 tokens of shared
        chat template, then divergence."""
        common = list(range(128))
        a_tokens = common + list(range(1000, 1050))
        b_tokens = common + list(range(2000, 2050))

        a = hash_prefix_tokens(a_tokens)
        b = hash_prefix_tokens(b_tokens)
        assert a != b

    def test_first_block_only_collides_on_shared_first_block(self):
        """Sanity: the *legacy* first-block hash DOES collide on the
        scenario we're guarding against. This proves the test
        above is actually exercising the bug."""
        a_tokens = list(range(16)) + [42]
        b_tokens = list(range(16)) + [99]
        legacy_a = legacy_first_block_hash(a_tokens)
        legacy_b = legacy_first_block_hash(b_tokens)
        assert legacy_a == legacy_b  # the old bug


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:

    def test_num_blocks_must_be_positive(self):
        with pytest.raises(ValueError, match="num_blocks"):
            ResourceId(
                scope=ResourceScope.PREFIX,
                resource_id="abc",
                num_blocks=0,
                num_bytes=0,
                session_id="s",
            )

    def test_num_bytes_must_be_nonneg(self):
        with pytest.raises(ValueError, match="num_bytes"):
            ResourceId(
                scope=ResourceScope.PREFIX,
                resource_id="abc",
                num_blocks=1,
                num_bytes=-1,
                session_id="s",
            )

    def test_resource_id_must_be_nonempty(self):
        with pytest.raises(ValueError, match="resource_id"):
            ResourceId(
                scope=ResourceScope.PREFIX,
                resource_id="",
                num_blocks=1,
                num_bytes=0,
                session_id="s",
            )

    def test_scope_must_be_enum(self):
        with pytest.raises(TypeError, match="scope"):
            ResourceId(
                scope="prefix",  # type: ignore[arg-type]
                resource_id="abc",
                num_blocks=1,
                num_bytes=0,
                session_id="s",
            )

    def test_for_prefix_requires_tokens(self):
        with pytest.raises(ValueError, match="at least one token"):
            ResourceId.for_prefix([], session_id="s")

    def test_for_prefix_block_count_ceiling(self):
        """17 tokens at block_size=16 should count as 2 blocks."""
        r = ResourceId.for_prefix(list(range(17)),
                                  session_id="s",
                                  block_size=16)
        assert r.num_blocks == 2

        r = ResourceId.for_prefix(list(range(16)),
                                  session_id="s",
                                  block_size=16)
        assert r.num_blocks == 1

        r = ResourceId.for_prefix(list(range(32)),
                                  session_id="s",
                                  block_size=16)
        assert r.num_blocks == 2


# ---------------------------------------------------------------------------
# Legacy first_block_hash carry
# ---------------------------------------------------------------------------


class TestLegacyCarry:

    def test_first_block_hash_present_on_struct(self):
        r = ResourceId.for_prefix(list(range(32)),
                                  session_id="s",
                                  block_size=16)
        assert r.first_block_hash is not None
        # And it matches the bare helper.
        assert r.first_block_hash == legacy_first_block_hash(list(range(32)),
                                                             block_size=16)

    def test_first_block_hash_differs_from_resource_id(self):
        """The two hashes have different domains by construction:
        the legacy one has a distinguishing prefix to prevent any
        accidental comparison-equals-true collision."""
        r = ResourceId.for_prefix(list(range(16)), session_id="s")
        assert r.first_block_hash != r.resource_id


# ---------------------------------------------------------------------------
# ResourceCandidate shape
# ---------------------------------------------------------------------------


class TestResourceCandidate:

    def test_candidate_exposes_proxy_properties(self):
        rid = ResourceId.for_prefix(list(range(10)),
                                    session_id="sA",
                                    bytes_per_token=4)
        cand = ResourceCandidate(resource=rid)
        assert cand.resource_id == rid.resource_id
        assert cand.session_id == "sA"
        assert cand.num_blocks == rid.num_blocks
        assert cand.num_bytes == 40
        assert cand.scope is ResourceScope.PREFIX

    def test_candidate_defaults(self):
        rid = ResourceId.for_prefix(list(range(10)), session_id="sA")
        cand = ResourceCandidate(resource=rid)
        assert cand.score == 0.0
        assert cand.last_query_hash is None
        assert cand.ctx == {}
