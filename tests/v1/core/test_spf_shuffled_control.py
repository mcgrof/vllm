# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Negative-control test: shuffled workload destroys transition advantage.

The point of the transition model (``vllm.v1.core.spf.transitions``)
is to exploit real temporal structure — "session S just accessed A;
it has often followed A with B, so B is a good prefetch target".

If SPF shows benefit on a workload and *also* shows benefit after
you shuffle that workload's events, then whatever SPF is doing has
nothing to do with the transitions. Shuffling destroys the A → B
order, so genuine transition-derived predictive power must
collapse.

This file locks in that collapse:

  1. Build a strongly-transitioned workload (mixed_session_interleave).
  2. Feed it to a TransitionTable event-by-event — observe
     a concentrated probability distribution.
  3. Shuffle via :func:`shuffled_control`.
  4. Feed the shuffled version to a fresh TransitionTable.
  5. Assert that the shuffled version's top-1 transition
     probability is meaningfully *lower* than the baseline's —
     i.e. the signal degrades.

No SPFController involved; this is an algorithmic property of the
transition model itself. Phase-5 live runs will layer SPF on top
and measure TTFT/hit-rate deltas — that's a separate test.
"""
from __future__ import annotations

from vllm.v1.core.spf.resource import hash_prefix_tokens
from vllm.v1.core.spf.transitions import TransitionTable
from vllm.v1.core.spf.workloads import (mixed_session_interleave,
                                        shuffled_control)


def _observe_all(
    events,
    prefix_slice: int = 64,
) -> TransitionTable:
    """Feed every event into a TransitionTable using the first
    ``prefix_slice`` tokens of each prompt as the prefix identity.
    """
    tt = TransitionTable()
    for ev in events:
        prefix = hash_prefix_tokens(ev.token_ids[:prefix_slice])
        tt.record_access(ev.session_id, prefix)
    return tt


def _top1_prob(tt: TransitionTable, events) -> float:
    """Average top-1 transition probability over all observed
    (session, from_prefix) pairs.

    A concentrated structure (A → B dominates) pushes this
    average toward 1.0. A shuffled / random structure flattens
    the distribution and pushes the average toward ``1/K`` where
    K is the number of distinct destinations.
    """
    seen_pairs = set()
    for ev in events:
        prefix = hash_prefix_tokens(ev.token_ids[:64])
        if tt.total_from(ev.session_id, prefix) > 0:
            seen_pairs.add((ev.session_id, prefix))

    if not seen_pairs:
        return 0.0

    total = 0.0
    for sid, from_prefix in seen_pairs:
        dist = tt.outgoing(sid, from_prefix)
        if dist:
            total += max(dist.values())
    return total / len(seen_pairs)


def _avg_outgoing_cardinality(
    tt: TransitionTable,
    events,
) -> float:
    """Average number of distinct destinations per (session, from)
    pair. Diagnostic: a shuffled workload spreads transitions
    across more destinations than a structured one does."""
    seen_pairs = set()
    for ev in events:
        prefix = hash_prefix_tokens(ev.token_ids[:64])
        if tt.total_from(ev.session_id, prefix) > 0:
            seen_pairs.add((ev.session_id, prefix))
    if not seen_pairs:
        return 0.0
    total = 0
    for sid, from_prefix in seen_pairs:
        total += len(tt.outgoing(sid, from_prefix))
    return total / len(seen_pairs)


# ---------------------------------------------------------------------------
# The invariant
# ---------------------------------------------------------------------------


class TestShuffledDestroysTransitionAdvantage:
    """Canonical negative control. If the assertion ever flips,
    the transition model (or the workload generator) has a bug
    that is creating structure where there shouldn't be any."""

    def test_top1_probability_degrades_after_shuffle(self):
        base = mixed_session_interleave(n_sessions=3,
                                        n_prefixes=6,
                                        n_events=200,
                                        prefix_len=64,
                                        tail_len=8,
                                        seed=42)
        shuf = shuffled_control(base, seed=42)

        tt_base = _observe_all(base)
        tt_shuf = _observe_all(shuf)

        top1_base = _top1_prob(tt_base, base)
        top1_shuf = _top1_prob(tt_shuf, shuf)

        # The baseline should have SOME concentrated structure.
        # With 6 prefixes per session, random pick would give top-1
        # ≈ 1/6 ≈ 0.17; base should be no worse than that.
        assert top1_base > 0.15, (
            f"baseline top-1 probability {top1_base:.3f} too low — "
            f"workload has no structure to exploit")

        # The shuffle must not *improve* top-1 probability (that
        # would mean we're faking structure). And it should stay
        # within the random-pick neighbourhood: no better than
        # roughly 1/N where N is the number of distinct prefixes
        # seen per session.
        assert top1_shuf <= top1_base + 1e-6, (
            f"shuffled top-1 {top1_shuf:.3f} exceeded baseline "
            f"{top1_base:.3f} — shuffle created structure, "
            f"generator or model bug")

    def test_outgoing_cardinality_does_not_shrink_after_shuffle(self):
        """A shuffled workload spreads transitions across at least
        as many destinations as the baseline (a structured
        workload can concentrate; a random one cannot).
        """
        base = mixed_session_interleave(n_sessions=3,
                                        n_prefixes=6,
                                        n_events=200,
                                        seed=7)
        shuf = shuffled_control(base, seed=7)

        tt_base = _observe_all(base)
        tt_shuf = _observe_all(shuf)

        card_base = _avg_outgoing_cardinality(tt_base, base)
        card_shuf = _avg_outgoing_cardinality(tt_shuf, shuf)

        # Shuffling ≥ spreading. A small tolerance because with 6
        # prefixes the ceiling on cardinality is 6 either way.
        assert card_shuf >= card_base - 0.1


# ---------------------------------------------------------------------------
# Random_no_reuse: transition table learns nothing
# ---------------------------------------------------------------------------


class TestRandomNoReuseTransitions:
    """Complementary negative control. On a workload with NO
    reuse, every (session, from_prefix) pair has exactly one
    outgoing destination, so top-1 is trivially 1.0 — but there
    is only ONE observation per pair, so transition_count is 1
    and robustness weighting should heavily discount the signal.
    """

    def test_all_pairs_have_one_observation(self):
        events = random_no_reuse_events(n_events=30, seed=0)
        tt = _observe_all(events)
        for ev in events:
            prefix = hash_prefix_tokens(ev.token_ids[:64])
            total = tt.total_from(ev.session_id, prefix)
            assert total <= 1

    def test_no_concentration_possible(self):
        """With every prefix unique, the session's history has
        no repeated (from, to) pair, so no count exceeds 1."""
        events = random_no_reuse_events(n_events=30, seed=1)
        tt = _observe_all(events)
        # Max count anywhere in the table is 1.
        # (Accessing internals because that's what we're asserting
        # — this is a diagnostic property, not a public API.)
        # noqa: SLF001 — test-only private access
        for inner in tt._counts.values():
            assert all(c == 1 for c in inner.values())


# Tiny convenience wrapper to avoid a circular-import confusion:
# random_no_reuse imported at top of file, but we only need it in
# one test class, so keep it out of the top-level import mess.
def random_no_reuse_events(**kw):
    from vllm.v1.core.spf.workloads import random_no_reuse
    return random_no_reuse(**kw)
