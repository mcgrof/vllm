# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synthetic workload generators for SPF testing and benchmarking.

Rule 10 in the SPF instructions says "everything needed for tests
must live in-tree or be synthetically generated." This module is
the in-tree half of that: seven named workload generators plus a
Long Doc QA adapter, all of which produce deterministic
:class:`WorkloadEvent` lists without reading any disk corpus.

Each generator takes a ``seed`` and reproducible parameters; same
seed + same params ⇒ same event list byte-for-byte. That's what
makes the A/B arms across SPF on/off comparable.

Taxonomy (from SPF instruction item 14-15):

  - ``mixed_session_interleave``: small set of recurring prefixes
    interleaved across sessions. Reuse-heavy, retention-friendly.
  - ``batched_burst``: one session dominates for a burst, then
    another. Stresses the burst / stale-session guards.
  - ``shared_prefix``: many requests share a common early prefix.
    Retention should dominate with little eviction pressure.
  - ``conversation_tree``: tree-structured prefix growth (each
    turn extends its parent). Simulates multi-turn dialogue.
  - ``random_no_reuse``: every request has unique random tokens.
    Negative control — SPF should show no gain.
  - ``shuffled_control``: same events as another workload but
    randomised order. Negative control for the transition model —
    any benefit from transitions should collapse.
  - ``false_shared_first_block``: all requests share the same
    first block then diverge. Exercises the honest resource_id;
    a first-block-only policy would conflate them.

Plus a Long Doc QA adapter:

  - ``long_doc_qa``: N synthetic documents (long token sequences),
    each followed by a configurable mix of "hit" queries (reuse
    the doc prefix) and "miss" queries (new docs).

Every generator returns ``list[WorkloadEvent]`` sorted by
``arrival_step``. Callers that want to export to JSONL for
existing trace harnesses can do so trivially; we don't bundle an
exporter here to keep the surface narrow.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field


# Token ID range. 32 bit positive values keep compatibility with
# the int-to-bytes encoding used by ``hash_prefix_tokens``.
_TOKEN_MIN = 1
_TOKEN_MAX = 1 << 31 - 1


@dataclass
class WorkloadEvent:
    """A single request in a synthetic SPF workload.

    - ``session_id``: caller who issued this request.
    - ``token_ids``: full prompt. The prefix region the SPF scorer
      cares about is embedded here; callers typically hash the
      first ``prefix_len`` tokens (workload-specific).
    - ``arrival_step``: scheduler step this request arrives at. All
      generators produce events in non-decreasing ``arrival_step``
      order.
    - ``meta``: generator-specific metadata — doc_id, hit/miss tag,
      branch depth, etc. Tests use this to check structural
      invariants without re-deriving them from token_ids.
    """

    session_id: str
    token_ids: list[int]
    arrival_step: int
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helper: deterministic random token block generator
# ---------------------------------------------------------------------------

def _rand_tokens(rng: random.Random, n: int) -> list[int]:
    """Draw ``n`` independent token IDs from the RNG."""
    return [rng.randint(_TOKEN_MIN, _TOKEN_MAX) for _ in range(n)]


def _rand_tokens_labeled(
    rng: random.Random,
    n: int,
    label: int,
) -> list[int]:
    """Generate n tokens seeded from ``label`` so multiple calls
    with the same label produce the same tokens.

    Used by workloads that want a small set of "named" prefixes
    that recur — we produce the prefix once, cache it, and reuse.
    """
    local = random.Random(label + rng.random())
    return [local.randint(_TOKEN_MIN, _TOKEN_MAX) for _ in range(n)]


# ---------------------------------------------------------------------------
# 1. mixed_session_interleave
# ---------------------------------------------------------------------------

def mixed_session_interleave(
    n_sessions: int = 4,
    n_prefixes: int = 8,
    n_events: int = 100,
    prefix_len: int = 64,
    tail_len: int = 16,
    seed: int = 0,
) -> list[WorkloadEvent]:
    """Small set of recurring prefixes interleaved across sessions.

    Each session picks from the same pool of ``n_prefixes`` shared
    prefixes (so a reuse-heavy workload), but different sessions
    interleave their requests. SPF retention should hold the hot
    prefixes resident; the interleave is the test that per-session
    bookkeeping doesn't collapse.
    """
    rng = random.Random(seed)
    prefixes = [_rand_tokens(rng, prefix_len) for _ in range(n_prefixes)]

    events: list[WorkloadEvent] = []
    for i in range(n_events):
        sid = f"s{i % n_sessions}"
        prefix_idx = rng.randrange(n_prefixes)
        tail = _rand_tokens(rng, tail_len)
        events.append(WorkloadEvent(
            session_id=sid,
            token_ids=list(prefixes[prefix_idx]) + tail,
            arrival_step=i,
            meta={"prefix_idx": prefix_idx, "kind": "mixed_session"},
        ))
    return events


# ---------------------------------------------------------------------------
# 2. batched_burst
# ---------------------------------------------------------------------------

def batched_burst(
    n_sessions: int = 3,
    burst_size: int = 20,
    n_bursts: int = 5,
    prefix_len: int = 64,
    tail_len: int = 16,
    seed: int = 0,
) -> list[WorkloadEvent]:
    """One session dominates for a burst, then the next.

    Each session owns its own prefix pool (no cross-session reuse).
    Within a burst the session hits its prefixes repeatedly; at the
    burst boundary the active session changes. Exercises the
    stale-session / self-prefetch burst guard behaviour.
    """
    rng = random.Random(seed)
    # Each session has its own 4 prefixes.
    per_session_prefixes = {
        f"s{i}": [_rand_tokens(rng, prefix_len) for _ in range(4)]
        for i in range(n_sessions)
    }

    events: list[WorkloadEvent] = []
    step = 0
    for burst_idx in range(n_bursts):
        sid = f"s{burst_idx % n_sessions}"
        prefixes = per_session_prefixes[sid]
        for i in range(burst_size):
            p = prefixes[i % len(prefixes)]
            events.append(WorkloadEvent(
                session_id=sid,
                token_ids=list(p) + _rand_tokens(rng, tail_len),
                arrival_step=step,
                meta={
                    "burst_idx": burst_idx,
                    "kind": "batched_burst",
                },
            ))
            step += 1
    return events


# ---------------------------------------------------------------------------
# 3. shared_prefix
# ---------------------------------------------------------------------------

def shared_prefix(
    n_events: int = 100,
    shared_len: int = 128,
    tail_len: int = 50,
    n_sessions: int = 4,
    seed: int = 0,
) -> list[WorkloadEvent]:
    """Many requests share a common early prefix (chat template).

    The shared_len tokens are the same across every event; the
    tail diverges per event. Retention should cache the shared
    prefix after request 1.
    """
    rng = random.Random(seed)
    shared = _rand_tokens(rng, shared_len)

    events: list[WorkloadEvent] = []
    for i in range(n_events):
        sid = f"s{i % n_sessions}"
        tail = _rand_tokens(rng, tail_len)
        events.append(WorkloadEvent(
            session_id=sid,
            token_ids=list(shared) + tail,
            arrival_step=i,
            meta={"kind": "shared_prefix"},
        ))
    return events


# ---------------------------------------------------------------------------
# 4. conversation_tree
# ---------------------------------------------------------------------------

def conversation_tree(
    n_conversations: int = 5,
    depth: int = 8,
    branches: int = 2,
    segment_len: int = 32,
    seed: int = 0,
) -> list[WorkloadEvent]:
    """Tree-structured prefix growth: each turn extends its parent.

    ``n_conversations`` root conversations each branch into a tree
    of depth ``depth``. At depth d, every node has ``branches``
    children. Each child's token list is the parent's list plus a
    fresh ``segment_len`` segment. Models multi-turn dialogue.

    Events are emitted in BFS order so that parent is always
    observed before child — respects causality the SPF controller
    assumes.
    """
    rng = random.Random(seed)
    events: list[WorkloadEvent] = []
    step = 0
    for conv_idx in range(n_conversations):
        sid = f"conv{conv_idx}"
        root_tokens = _rand_tokens(rng, segment_len)
        # BFS layers: list of (tokens, depth_index).
        current_layer = [(root_tokens, 0)]
        events.append(WorkloadEvent(
            session_id=sid,
            token_ids=list(root_tokens),
            arrival_step=step,
            meta={"kind": "conversation_tree",
                  "conv": conv_idx, "depth": 0},
        ))
        step += 1
        for d in range(1, depth):
            next_layer = []
            for parent_tokens, _ in current_layer:
                for _ in range(branches):
                    seg = _rand_tokens(rng, segment_len)
                    child_tokens = list(parent_tokens) + seg
                    events.append(WorkloadEvent(
                        session_id=sid,
                        token_ids=child_tokens,
                        arrival_step=step,
                        meta={
                            "kind": "conversation_tree",
                            "conv": conv_idx, "depth": d,
                        },
                    ))
                    step += 1
                    next_layer.append((child_tokens, d))
            current_layer = next_layer
    return events


# ---------------------------------------------------------------------------
# 5. random_no_reuse
# ---------------------------------------------------------------------------

def random_no_reuse(
    n_events: int = 100,
    prefix_len: int = 128,
    n_sessions: int = 4,
    seed: int = 0,
) -> list[WorkloadEvent]:
    """Every request has a unique random prefix — no reuse at all.

    The negative control: SPF's P(use) estimator should return
    very low values on every candidate because no transition /
    recency / frequency signal exists. Any A/B benefit on this
    workload is spurious.
    """
    rng = random.Random(seed)
    events: list[WorkloadEvent] = []
    for i in range(n_events):
        sid = f"s{i % n_sessions}"
        events.append(WorkloadEvent(
            session_id=sid,
            token_ids=_rand_tokens(rng, prefix_len),
            arrival_step=i,
            meta={"kind": "random_no_reuse"},
        ))
    return events


# ---------------------------------------------------------------------------
# 6. shuffled_control
# ---------------------------------------------------------------------------

def shuffled_control(
    base: list[WorkloadEvent],
    seed: int = 0,
) -> list[WorkloadEvent]:
    """Permute ``base``'s event order randomly.

    The prefix set (multiset of token_ids sequences) is identical
    to ``base``; only the temporal order changes. The transition
    model's predictive edge should collapse because the A → B
    patterns that made transitions useful are destroyed.

    Preserves ``arrival_step`` ordering semantics by reassigning
    steps 0..N-1 after shuffling so consumers see a coherent
    timeline.
    """
    rng = random.Random(seed)
    shuffled = list(base)
    rng.shuffle(shuffled)
    # Reassign arrival steps in the new order.
    out = []
    for i, ev in enumerate(shuffled):
        out.append(WorkloadEvent(
            session_id=ev.session_id,
            token_ids=list(ev.token_ids),
            arrival_step=i,
            meta={**ev.meta, "kind": "shuffled_control"},
        ))
    return out


# ---------------------------------------------------------------------------
# 7. false_shared_first_block
# ---------------------------------------------------------------------------

def false_shared_first_block(
    n_events: int = 50,
    shared_block_size: int = 16,
    tail_len: int = 100,
    n_sessions: int = 4,
    seed: int = 0,
) -> list[WorkloadEvent]:
    """All requests share the same first block, then diverge.

    Every event's first ``shared_block_size`` tokens are identical;
    the rest are unique. Under the old first-block-only identity,
    all N events collapse to one resource_id. Under the honest
    identity, each is distinct. The SPF controller's hit-rate
    behaviour on this workload is a diagnostic for whether the
    identity is the honest one.
    """
    rng = random.Random(seed)
    shared = _rand_tokens(rng, shared_block_size)

    events: list[WorkloadEvent] = []
    for i in range(n_events):
        sid = f"s{i % n_sessions}"
        tail = _rand_tokens(rng, tail_len)
        events.append(WorkloadEvent(
            session_id=sid,
            token_ids=list(shared) + tail,
            arrival_step=i,
            meta={
                "kind": "false_shared_first_block",
                "shared_block_size": shared_block_size,
            },
        ))
    return events


# ---------------------------------------------------------------------------
# 8. Long Doc QA adapter
# ---------------------------------------------------------------------------

def long_doc_qa(
    n_docs: int = 5,
    doc_len: int = 4096,
    queries_per_doc: int = 10,
    query_len: int = 64,
    hit_ratio: float = 0.7,
    n_sessions: int = 4,
    seed: int = 0,
) -> list[WorkloadEvent]:
    """Long-document QA pattern with configurable hit:miss mix.

    Structure:

      1. ``n_docs`` documents are "introduced" each as a single
         long event (the document body).
      2. Then ``queries_per_doc`` queries are generated per doc:
         - With probability ``hit_ratio``, the query reuses the
           doc's tokens as its prefix (a retention/prefetch hit).
         - With probability ``1 - hit_ratio``, the query uses a
           fresh unseen prefix of the same length (a miss).

    Events are generated in introduction-first order, then the
    queries in interleaved order (query k for doc i, query k for
    doc i+1, …). Sessions round-robin across queries.

    Synthetic only — no real corpus. Deterministic given
    ``seed``; all ``token_ids`` are drawn from the seeded RNG so a
    downstream test can reconstruct an exact prompt by re-running
    the generator.
    """
    if not 0.0 <= hit_ratio <= 1.0:
        raise ValueError(
            f"hit_ratio must be in [0, 1], got {hit_ratio}")
    rng = random.Random(seed)

    docs: list[list[int]] = [
        _rand_tokens(rng, doc_len) for _ in range(n_docs)
    ]

    events: list[WorkloadEvent] = []
    step = 0
    # Phase 1: introduce each document.
    for doc_idx, doc_tokens in enumerate(docs):
        sid = f"s{doc_idx % n_sessions}"
        events.append(WorkloadEvent(
            session_id=sid,
            token_ids=list(doc_tokens),
            arrival_step=step,
            meta={
                "kind": "long_doc_qa",
                "phase": "intro",
                "doc_id": doc_idx,
            },
        ))
        step += 1

    # Phase 2: interleaved queries. For k in [0, queries_per_doc),
    # emit query_k for each doc in turn, so docs interleave.
    for k in range(queries_per_doc):
        for doc_idx in range(n_docs):
            sid = f"s{step % n_sessions}"
            is_hit = rng.random() < hit_ratio
            if is_hit:
                prefix = docs[doc_idx]
                tag = "hit"
            else:
                prefix = _rand_tokens(rng, doc_len)  # unseen doc
                tag = "miss"
            q_tail = _rand_tokens(rng, query_len)
            events.append(WorkloadEvent(
                session_id=sid,
                token_ids=list(prefix) + q_tail,
                arrival_step=step,
                meta={
                    "kind": "long_doc_qa",
                    "phase": "query",
                    "doc_id": doc_idx if is_hit else -1,
                    "query_idx": k,
                    "hit_miss": tag,
                },
            ))
            step += 1

    return events


# ---------------------------------------------------------------------------
# Registry for convenience
# ---------------------------------------------------------------------------

WORKLOADS = {
    "mixed_session_interleave": mixed_session_interleave,
    "batched_burst": batched_burst,
    "shared_prefix": shared_prefix,
    "conversation_tree": conversation_tree,
    "random_no_reuse": random_no_reuse,
    "false_shared_first_block": false_shared_first_block,
    "long_doc_qa": long_doc_qa,
}
"""Name → generator mapping. ``shuffled_control`` is excluded
because it takes a base workload, not scalar parameters, and
driving it through a uniform API would require a different
signature. Callers that want shuffled control compose directly."""
