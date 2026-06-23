# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the SPF cooldown / suppression tracker.

The cooldown exists so a resource that stays hot across many
scheduler steps doesn't get hinted on every step. Without it,
``hints_issued`` pumps up but residency state doesn't change —
the counters lie about how much work the policy actually did.
"""
from __future__ import annotations

from vllm.v1.core.spf.scorer import CooldownTracker

# ---------------------------------------------------------------------------
# Basic semantics
# ---------------------------------------------------------------------------


class TestCooldownBasics:

    def test_unknown_resource_is_not_suppressed(self):
        c = CooldownTracker(period_steps=3)
        assert c.should_suppress("r", current_step=0) is False

    def test_after_issue_suppresses_within_period(self):
        c = CooldownTracker(period_steps=3)
        c.mark_issued("r", current_step=10)
        assert c.should_suppress("r", current_step=10) is True
        assert c.should_suppress("r", current_step=11) is True
        assert c.should_suppress("r", current_step=12) is True

    def test_after_period_allows_reissue(self):
        c = CooldownTracker(period_steps=3)
        c.mark_issued("r", current_step=10)
        # Steps 10, 11, 12 suppressed. Step 13 allowed.
        assert c.should_suppress("r", current_step=13) is False

    def test_period_zero_never_suppresses(self):
        c = CooldownTracker(period_steps=0)
        c.mark_issued("r", current_step=10)
        assert c.should_suppress("r", current_step=10) is False
        assert c.should_suppress("r", current_step=11) is False

    def test_per_resource_isolation(self):
        c = CooldownTracker(period_steps=3)
        c.mark_issued("r1", current_step=10)
        # r2 has never been issued → not suppressed.
        assert c.should_suppress("r2", current_step=10) is False
        assert c.should_suppress("r1", current_step=10) is True


# ---------------------------------------------------------------------------
# The "don't re-hint every step" behaviour (the whole point)
# ---------------------------------------------------------------------------


class TestRepeatedSteps:

    def test_hot_resource_only_hinted_once_per_period(self):
        """Simulate a resource at the top of the ranking for 10
        consecutive steps. With cooldown=3 we should issue a hint
        every 3 steps, not every step."""
        c = CooldownTracker(period_steps=3)
        issued: list[int] = []
        for step in range(10):
            if not c.should_suppress("hot", current_step=step):
                issued.append(step)
                c.mark_issued("hot", current_step=step)
        # Expected: 0, 3, 6, 9 → 4 hints across 10 steps.
        assert issued == [0, 3, 6, 9]

    def test_hot_resource_across_many_steps_is_bounded(self):
        """Across N steps with cooldown=3, issue count is ~N/3."""
        c = CooldownTracker(period_steps=3)
        count = 0
        for step in range(30):
            if not c.should_suppress("r", current_step=step):
                count += 1
                c.mark_issued("r", current_step=step)
        assert count == 10  # 0, 3, 6, …, 27


# ---------------------------------------------------------------------------
# Manual clearing (post-outcome use case)
# ---------------------------------------------------------------------------


class TestClearing:

    def test_clear_allows_immediate_reissue(self):
        c = CooldownTracker(period_steps=5)
        c.mark_issued("r", current_step=10)
        assert c.should_suppress("r", current_step=11) is True
        c.clear("r")
        assert c.should_suppress("r", current_step=11) is False

    def test_clear_of_unknown_is_noop(self):
        c = CooldownTracker(period_steps=3)
        c.clear("never_issued")  # must not raise

    def test_reset_clears_all(self):
        c = CooldownTracker(period_steps=3)
        c.mark_issued("a", current_step=1)
        c.mark_issued("b", current_step=2)
        c.reset()
        assert c.should_suppress("a", current_step=2) is False
        assert c.should_suppress("b", current_step=3) is False


# ---------------------------------------------------------------------------
# Interaction with the mark/should-suppress ordering
# ---------------------------------------------------------------------------


class TestOrdering:

    def test_mark_before_should_suppress_in_same_step(self):
        """A caller that marks on step N and then checks on step N
        sees the resource as suppressed — i.e. mark takes effect
        immediately. Matches the controller's in-step semantics."""
        c = CooldownTracker(period_steps=2)
        c.mark_issued("r", current_step=5)
        assert c.should_suppress("r", current_step=5) is True

    def test_mark_idempotent(self):
        """Marking twice on the same step is effectively a no-op
        beyond the last-issued bookkeeping."""
        c = CooldownTracker(period_steps=3)
        c.mark_issued("r", current_step=5)
        c.mark_issued("r", current_step=5)
        # Still in cooldown for steps 5, 6, 7.
        assert c.should_suppress("r", current_step=7) is True
        assert c.should_suppress("r", current_step=8) is False
