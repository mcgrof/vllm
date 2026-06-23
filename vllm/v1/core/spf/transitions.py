# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-session prefix transition model for SPF.

When session S accesses prefix A and then accesses prefix B, we
record ``A → B`` in that session's table. Later, when scoring a
candidate resource B for "next request from session S", we can ask
"given the session's current prefix is A, how often does A
transition to B?" and use that as a causal next-resource feature.

This is deliberately the tiniest possible causal predictor:

  - Per-session counts, not shared across sessions (a different
    session's transitions tell us nothing about this session's
    next prefix).
  - First-order only (one-step lookback). Second-order chains are
    expensive to maintain and overfit on short sessions.
  - No smoothing, no Bayesian priors, no neural anything. The
    whole point of phase 2 is to be a defensible baseline — "count
    what you saw and divide" is exactly that.
  - Updated causally: the transition is only recorded after both
    sides have been observed, so the table never tells the future.

Properties tests exercise:

  - Update causality (A → B requires observing B after A, never
    from B alone).
  - Per-session isolation (two sessions with identical prefix
    histories don't share counts).
  - Probability normalization (outgoing probs for a given
    (session, from_prefix) sum to 1.0 when totals > 0).
  - Unseen transitions return 0.0.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TransitionTable:
    """Causal, per-session prefix transition counts.

    State:

      * ``_counts[(session_id, from_prefix)] = {to_prefix: count}``
      * ``_totals[(session_id, from_prefix)] = sum of outgoing counts``

    ``_totals`` is kept separate so probability lookups are O(1)
    without rebuilding the sum on every call.
    """

    # Populated directly by ``record_transition``. Keys are left
    # untyped in the annotation to keep the dataclass simple; tests
    # pin the expected shapes.
    _counts: dict = field(default_factory=dict)
    _totals: dict = field(default_factory=dict)

    # Most recently observed prefix per session, used so callers
    # that just know the current prefix can record a one-step
    # transition implicitly via :meth:`record_access`.
    _last_prefix: dict = field(default_factory=dict)

    def record_access(
        self,
        session_id: str,
        prefix: str,
    ) -> None:
        """Record that session ``session_id`` just accessed
        ``prefix``.

        If this session has a previous access, record a transition
        ``previous → prefix``. Otherwise just remember this access.
        No-op self-loops (accessing the same prefix twice in a row)
        are recorded as transitions to keep the counting consistent
        with how the controller's existing session_history behaves.
        """
        last = self._last_prefix.get(session_id)
        if last is not None:
            self.record_transition(session_id, last, prefix)
        self._last_prefix[session_id] = prefix

    def record_transition(
        self,
        session_id: str,
        from_prefix: str,
        to_prefix: str,
    ) -> None:
        """Explicitly record ``from_prefix → to_prefix`` for a session."""
        key = (session_id, from_prefix)
        inner = self._counts.setdefault(key, {})
        inner[to_prefix] = inner.get(to_prefix, 0) + 1
        self._totals[key] = self._totals.get(key, 0) + 1

    def probability(
        self,
        session_id: str,
        from_prefix: str,
        to_prefix: str,
    ) -> float:
        """Return P(to_prefix | session_id accessed from_prefix).

        0.0 if there's no record of the (session, from) pair or
        if ``to_prefix`` has never followed ``from_prefix`` for
        this session. No smoothing — a genuinely unseen transition
        returns exactly zero.
        """
        key = (session_id, from_prefix)
        total = self._totals.get(key, 0)
        if total == 0:
            return 0.0
        return self._counts.get(key, {}).get(to_prefix, 0) / total

    def count(
        self,
        session_id: str,
        from_prefix: str,
        to_prefix: str,
    ) -> int:
        """Raw count of a transition, 0 if never observed."""
        return self._counts.get(
            (session_id, from_prefix), {},
        ).get(to_prefix, 0)

    def total_from(
        self,
        session_id: str,
        from_prefix: str,
    ) -> int:
        """Sum of outgoing transitions from ``from_prefix`` for
        this session."""
        return self._totals.get((session_id, from_prefix), 0)

    def current_prefix(self, session_id: str) -> "str | None":
        """The most recent prefix this session accessed, or None."""
        return self._last_prefix.get(session_id)

    def outgoing(
        self,
        session_id: str,
        from_prefix: str,
    ) -> "dict[str, float]":
        """Return the probability distribution over next-prefixes
        for ``(session_id, from_prefix)``.

        Empty dict if the pair has never been observed. Used by the
        controller to nominate likely-next candidates without
        materialising every possible target prefix.
        """
        key = (session_id, from_prefix)
        total = self._totals.get(key, 0)
        if total == 0:
            return {}
        return {
            to_prefix: count / total
            for to_prefix, count in self._counts.get(key, {}).items()
        }

    def reset(self) -> None:
        """Forget all state. Used by tests and by per-session
        eviction logic when a session ends."""
        self._counts.clear()
        self._totals.clear()
        self._last_prefix.clear()
