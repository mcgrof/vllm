# SPDX-License-Identifier: Apache-2.0
"""Structural tests for SPF synthetic workload generators.

Per SPF instruction item 15, each generator has a documented
structural property. This file locks each one in so a future
refactor can't silently change the workload shape.

Tests cover:
  - Seed determinism (same seed + params ⇒ same output).
  - Event count / arrival-step monotonicity.
  - Per-generator invariant (what each workload is *supposed* to
    produce):
      mixed_session_interleave:  a small set of recurring prefixes
      batched_burst:             one session owns each burst
      shared_prefix:             common early prefix across all
      conversation_tree:         children extend parents
      random_no_reuse:           every prefix unique
      shuffled_control:          prefix multiset preserved, order
                                 randomised
      false_shared_first_block:  shared first block, unique tail
      long_doc_qa:               configurable hit:miss mix
"""
from __future__ import annotations

import pytest

from vllm.v1.core.spf.resource import hash_prefix_tokens
from vllm.v1.core.spf.workloads import (
    WORKLOADS,
    WorkloadEvent,
    batched_burst,
    conversation_tree,
    false_shared_first_block,
    long_doc_qa,
    mixed_session_interleave,
    random_no_reuse,
    shared_prefix,
    shuffled_control,
)


# ---------------------------------------------------------------------------
# Determinism — every generator must respect its seed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,fn,params", [
    ("mixed_session_interleave", mixed_session_interleave,
     dict(n_events=30, seed=42)),
    ("batched_burst", batched_burst,
     dict(n_bursts=3, burst_size=5, seed=42)),
    ("shared_prefix", shared_prefix,
     dict(n_events=30, seed=42)),
    ("conversation_tree", conversation_tree,
     dict(n_conversations=2, depth=3, branches=2, seed=42)),
    ("random_no_reuse", random_no_reuse,
     dict(n_events=30, seed=42)),
    ("false_shared_first_block", false_shared_first_block,
     dict(n_events=30, seed=42)),
    ("long_doc_qa", long_doc_qa,
     dict(n_docs=3, queries_per_doc=5, seed=42)),
])
def test_seed_determinism(name, fn, params):
    a = fn(**params)
    b = fn(**params)
    assert len(a) == len(b)
    for ea, eb in zip(a, b):
        assert ea.session_id == eb.session_id
        assert ea.token_ids == eb.token_ids
        assert ea.arrival_step == eb.arrival_step


# ---------------------------------------------------------------------------
# Universal property: arrival_step is non-decreasing 0..N-1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn,params", [
    (mixed_session_interleave, dict(n_events=30, seed=1)),
    (batched_burst,            dict(n_bursts=3, burst_size=5, seed=1)),
    (shared_prefix,            dict(n_events=30, seed=1)),
    (conversation_tree,
     dict(n_conversations=2, depth=3, branches=2, seed=1)),
    (random_no_reuse,          dict(n_events=30, seed=1)),
    (false_shared_first_block, dict(n_events=30, seed=1)),
    (long_doc_qa,
     dict(n_docs=3, queries_per_doc=5, seed=1)),
])
def test_arrival_steps_are_contiguous(fn, params):
    events = fn(**params)
    assert events[0].arrival_step == 0
    steps = [e.arrival_step for e in events]
    # Non-decreasing and exactly 0..N-1 (our generators produce
    # dense contiguous steps).
    assert steps == list(range(len(events)))


# ---------------------------------------------------------------------------
# mixed_session_interleave
# ---------------------------------------------------------------------------

class TestMixedSessionInterleave:
    def test_small_set_of_recurring_prefixes(self):
        events = mixed_session_interleave(
            n_sessions=4, n_prefixes=5, n_events=50,
            prefix_len=32, tail_len=8, seed=0)
        # Every event has a "prefix_idx" in meta, range [0, 5).
        idxs = {e.meta["prefix_idx"] for e in events}
        assert idxs.issubset(set(range(5)))
        # At least some recurrence: a 50-event workload with 5
        # prefixes must reuse at least one prefix.
        from collections import Counter
        c = Counter(e.meta["prefix_idx"] for e in events)
        assert any(v > 1 for v in c.values())

    def test_sessions_interleave(self):
        events = mixed_session_interleave(
            n_sessions=3, n_events=30, seed=0)
        # All 3 sessions appear.
        sessions = {e.session_id for e in events}
        assert sessions == {"s0", "s1", "s2"}
        # Interleaved: no single session dominates a 10-event window.
        window = [e.session_id for e in events[:10]]
        assert len(set(window)) > 1


# ---------------------------------------------------------------------------
# batched_burst
# ---------------------------------------------------------------------------

class TestBatchedBurst:
    def test_one_session_owns_each_burst(self):
        events = batched_burst(
            n_sessions=3, burst_size=10, n_bursts=4, seed=0)
        # Within a single burst, all events have the same session.
        for burst_idx in range(4):
            burst = [e for e in events
                     if e.meta["burst_idx"] == burst_idx]
            assert len(burst) == 10
            sessions = {e.session_id for e in burst}
            assert len(sessions) == 1

    def test_sessions_rotate_across_bursts(self):
        events = batched_burst(
            n_sessions=3, burst_size=5, n_bursts=6, seed=0)
        # Bursts 0-5 map to sessions [s0, s1, s2, s0, s1, s2].
        by_burst = {}
        for e in events:
            by_burst.setdefault(
                e.meta["burst_idx"], set()).add(e.session_id)
        order = [next(iter(by_burst[i])) for i in range(6)]
        assert order == ["s0", "s1", "s2", "s0", "s1", "s2"]


# ---------------------------------------------------------------------------
# shared_prefix
# ---------------------------------------------------------------------------

class TestSharedPrefix:
    def test_all_events_share_identical_prefix(self):
        events = shared_prefix(
            n_events=25, shared_len=64, tail_len=16, seed=0)
        first_prefix = events[0].token_ids[:64]
        for e in events:
            assert e.token_ids[:64] == first_prefix

    def test_tails_diverge(self):
        events = shared_prefix(
            n_events=10, shared_len=32, tail_len=16, seed=0)
        tails = {tuple(e.token_ids[32:]) for e in events}
        # All 10 tails distinct (rng makes collision negligible).
        assert len(tails) == 10


# ---------------------------------------------------------------------------
# conversation_tree
# ---------------------------------------------------------------------------

class TestConversationTree:
    def test_children_extend_parent(self):
        """For every non-root event, there exists an earlier event
        whose token_ids is a proper prefix of this one AND belongs
        to the same conversation."""
        events = conversation_tree(
            n_conversations=1, depth=4, branches=2,
            segment_len=8, seed=0)
        by_conv = {}
        for e in events:
            by_conv.setdefault(e.meta["conv"], []).append(e)
        for conv_events in by_conv.values():
            for i, e in enumerate(conv_events):
                if e.meta["depth"] == 0:
                    continue
                # Find an ancestor whose token_ids is a prefix.
                ancestors = [
                    prev for prev in conv_events[:i]
                    if len(prev.token_ids) < len(e.token_ids)
                    and e.token_ids[:len(prev.token_ids)]
                    == prev.token_ids
                ]
                assert ancestors, (
                    f"event at depth {e.meta['depth']} has no "
                    f"ancestor prefix")

    def test_depth_structure(self):
        events = conversation_tree(
            n_conversations=2, depth=3, branches=2,
            segment_len=4, seed=0)
        depths = {e.meta["depth"] for e in events}
        assert depths == {0, 1, 2}


# ---------------------------------------------------------------------------
# random_no_reuse
# ---------------------------------------------------------------------------

class TestRandomNoReuse:
    def test_every_prefix_unique(self):
        events = random_no_reuse(
            n_events=50, prefix_len=64, seed=0)
        prefixes = {tuple(e.token_ids[:64]) for e in events}
        assert len(prefixes) == 50

    def test_no_hash_collisions(self):
        """Honest resource_id must also distinguish every event."""
        events = random_no_reuse(n_events=30, seed=0)
        ids = {hash_prefix_tokens(e.token_ids) for e in events}
        assert len(ids) == 30


# ---------------------------------------------------------------------------
# shuffled_control — preservation properties
# ---------------------------------------------------------------------------

class TestShuffledControl:
    def test_same_event_count(self):
        base = mixed_session_interleave(n_events=40, seed=1)
        shuf = shuffled_control(base, seed=2)
        assert len(shuf) == len(base)

    def test_multiset_of_prefixes_preserved(self):
        """Shuffling must not invent or drop any prefix."""
        base = mixed_session_interleave(n_events=40, seed=1)
        shuf = shuffled_control(base, seed=2)
        base_ids = sorted(tuple(e.token_ids) for e in base)
        shuf_ids = sorted(tuple(e.token_ids) for e in shuf)
        assert base_ids == shuf_ids

    def test_order_actually_changes(self):
        base = mixed_session_interleave(n_events=40, seed=1)
        shuf = shuffled_control(base, seed=2)
        # Probability that a random shuffle preserves every
        # position is vanishingly small.
        preserved = sum(
            1 for a, b in zip(base, shuf)
            if a.token_ids == b.token_ids
        )
        assert preserved < len(base)

    def test_arrival_steps_reindexed(self):
        base = mixed_session_interleave(n_events=10, seed=1)
        shuf = shuffled_control(base, seed=2)
        assert [e.arrival_step for e in shuf] == list(range(10))


# ---------------------------------------------------------------------------
# false_shared_first_block
# ---------------------------------------------------------------------------

class TestFalseSharedFirstBlock:
    def test_all_share_first_block(self):
        events = false_shared_first_block(
            n_events=30, shared_block_size=16, tail_len=100,
            seed=0)
        first = events[0].token_ids[:16]
        for e in events:
            assert e.token_ids[:16] == first

    def test_tails_are_unique(self):
        events = false_shared_first_block(
            n_events=30, shared_block_size=16, tail_len=64,
            seed=0)
        tails = {tuple(e.token_ids[16:]) for e in events}
        assert len(tails) == 30

    def test_honest_identity_distinguishes_all(self):
        """Even though the first block collides by construction,
        the honest resource_id must disambiguate."""
        events = false_shared_first_block(
            n_events=30, seed=0)
        ids = {hash_prefix_tokens(e.token_ids) for e in events}
        assert len(ids) == 30


# ---------------------------------------------------------------------------
# long_doc_qa
# ---------------------------------------------------------------------------

class TestLongDocQA:
    def test_introduces_each_doc_once(self):
        events = long_doc_qa(n_docs=4, queries_per_doc=5, seed=0)
        intros = [e for e in events
                  if e.meta.get("phase") == "intro"]
        assert len(intros) == 4
        # doc_id 0..3 all seen once.
        doc_ids = {e.meta["doc_id"] for e in intros}
        assert doc_ids == {0, 1, 2, 3}

    def test_query_count_matches_param(self):
        events = long_doc_qa(n_docs=3, queries_per_doc=7, seed=0)
        queries = [e for e in events
                   if e.meta.get("phase") == "query"]
        assert len(queries) == 3 * 7

    def test_hit_ratio_respected_statistically(self):
        """Over many queries the observed hit fraction should be
        close to the requested ratio. Use enough samples that the
        variance is tight."""
        events = long_doc_qa(
            n_docs=5, queries_per_doc=100,
            hit_ratio=0.8, seed=0)
        queries = [e for e in events
                   if e.meta.get("phase") == "query"]
        n_hit = sum(1 for e in queries
                    if e.meta["hit_miss"] == "hit")
        frac = n_hit / len(queries)
        # 500 trials at p=0.8 has std ≈ 0.018 — 4σ window is
        # plenty.
        assert 0.72 <= frac <= 0.88

    def test_hit_query_reuses_doc_prefix(self):
        events = long_doc_qa(
            n_docs=2, doc_len=128, queries_per_doc=10,
            hit_ratio=1.0, seed=0)
        intros = {e.meta["doc_id"]: e.token_ids
                  for e in events if e.meta.get("phase") == "intro"}
        queries = [e for e in events
                   if e.meta.get("phase") == "query"]
        for q in queries:
            assert q.meta["hit_miss"] == "hit"
            doc = intros[q.meta["doc_id"]]
            assert q.token_ids[:128] == doc

    def test_miss_query_uses_new_prefix(self):
        events = long_doc_qa(
            n_docs=2, doc_len=32, queries_per_doc=5,
            hit_ratio=0.0, seed=0)
        queries = [e for e in events
                   if e.meta.get("phase") == "query"]
        intros_tokens = {
            tuple(e.token_ids) for e in events
            if e.meta.get("phase") == "intro"
        }
        for q in queries:
            assert q.meta["hit_miss"] == "miss"
            # Miss queries use prefixes that are NOT any intro doc.
            assert tuple(q.token_ids[:32]) not in intros_tokens

    def test_invalid_hit_ratio_raises(self):
        with pytest.raises(ValueError, match="hit_ratio"):
            long_doc_qa(hit_ratio=-0.1)
        with pytest.raises(ValueError, match="hit_ratio"):
            long_doc_qa(hit_ratio=1.1)


# ---------------------------------------------------------------------------
# Registry exposure
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_all_named_workloads_in_registry(self):
        assert set(WORKLOADS.keys()) >= {
            "mixed_session_interleave",
            "batched_burst",
            "shared_prefix",
            "conversation_tree",
            "random_no_reuse",
            "false_shared_first_block",
            "long_doc_qa",
        }

    def test_registry_callables_produce_events(self):
        for name, fn in WORKLOADS.items():
            events = fn(seed=7)
            assert events, f"{name} produced no events"
            assert isinstance(events[0], WorkloadEvent)
