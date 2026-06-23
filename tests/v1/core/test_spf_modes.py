# SPDX-License-Identifier: Apache-2.0
"""Phase-1 honest-mode tests for SPF mode separation.

Covers:

  - Config's ``mode`` default is ``retention``.
  - Invalid modes are rejected (no silent fallback).
  - Mode propagates into metrics / log output so experiments never
    have to guess which code path produced the numbers.
  - Retention-mode counters fill retention fields; prefetch-mode
    counters fill prefetch fields; they do not cross-contaminate.
  - Outcome tracking (hint_used / hint_wasted /
    harmful_evictions_avoided) routes to the correct bucket.
  - Lifecycle hooks on the controller tie timestamps and outcomes
    to ``resource_id``.
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from vllm.v1.core.spf.config import (
    MODE_PREFETCH,
    MODE_RETENTION,
    VALID_MODES,
    SPFConfig,
)
from vllm.v1.core.spf.controller import SPFController
from vllm.v1.core.spf.metrics import (
    PrefetchMetrics,
    RetentionMetrics,
    SPFMetrics,
    SPFStepMetrics,
)
from vllm.v1.core.spf.resource import ResourceId


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestConfigMode:
    def test_default_mode_is_retention(self):
        cfg = SPFConfig()
        assert cfg.mode == MODE_RETENTION

    def test_valid_modes_are_two(self):
        assert set(VALID_MODES) == {"retention", "prefetch"}

    def test_env_retention(self):
        with patch.dict(os.environ, {"VLLM_SPF_MODE": "retention"},
                        clear=False):
            cfg = SPFConfig.from_env()
            assert cfg.mode == MODE_RETENTION

    def test_env_prefetch(self):
        with patch.dict(os.environ, {"VLLM_SPF_MODE": "prefetch"},
                        clear=False):
            cfg = SPFConfig.from_env()
            assert cfg.mode == MODE_PREFETCH

    def test_invalid_mode_raises(self):
        with patch.dict(os.environ, {"VLLM_SPF_MODE": "cartridge"},
                        clear=False):
            with pytest.raises(ValueError, match="VLLM_SPF_MODE"):
                SPFConfig.from_env()

    def test_case_insensitive(self):
        with patch.dict(os.environ, {"VLLM_SPF_MODE": "RETENTION"},
                        clear=False):
            cfg = SPFConfig.from_env()
            assert cfg.mode == MODE_RETENTION

    def test_legacy_aliases(self):
        cfg = SPFConfig(max_hint_blocks=128, hint_fraction=0.25)
        assert cfg.max_prefetch_blocks == 128
        assert cfg.prefetch_fraction == 0.25


# ---------------------------------------------------------------------------
# Metrics routing
# ---------------------------------------------------------------------------

class TestMetricsRouting:
    def test_retention_hints_go_to_retention_bucket(self):
        m = SPFMetrics(_mode="retention", _interval=0)
        step = SPFStepMetrics(
            mode="retention",
            retention=RetentionMetrics(
                hints_issued=3, hints_used=2, hints_wasted=1),
            budget_blocks=10,
        )
        m.record_step(step)
        snap = m.snapshot()
        assert snap["mode"] == "retention"
        assert snap["hints_issued"] == 3
        assert snap["hints_used"] == 2
        assert snap["hints_wasted"] == 1
        # No prefetch-flavour counters leaked through.
        assert "prefetch_used" not in snap
        assert "bytes_promoted" not in snap

    def test_prefetch_hints_go_to_prefetch_bucket(self):
        m = SPFMetrics(_mode="prefetch", _interval=0)
        step = SPFStepMetrics(
            mode="prefetch",
            prefetch=PrefetchMetrics(
                hints_issued=4, prefetch_used=3, prefetch_wasted=1,
                bytes_promoted=12345, promote_latency_ms=7.5,
            ),
            budget_blocks=10,
        )
        m.record_step(step)
        snap = m.snapshot()
        assert snap["mode"] == "prefetch"
        assert snap["hints_issued"] == 4
        assert snap["prefetch_used"] == 3
        assert snap["bytes_promoted"] == 12345
        assert snap["promote_latency_ms"] == 7.5
        assert "hints_used" not in snap

    def test_mode_always_in_snapshot(self):
        for mode in ("retention", "prefetch"):
            m = SPFMetrics(_mode=mode, _interval=0)
            assert m.snapshot()["mode"] == mode

    def test_hint_used_routes_by_mode(self):
        m_ret = SPFMetrics(_mode="retention", _interval=0)
        m_ret.record_hint_issued("r1")
        m_ret.record_hint_used("r1")
        s = m_ret.snapshot()
        assert s["hints_used"] == 1
        assert s["outstanding"] == 0

        m_pf = SPFMetrics(_mode="prefetch", _interval=0)
        m_pf.record_hint_issued("r1")
        m_pf.record_hint_used("r1")
        s = m_pf.snapshot()
        assert s["prefetch_used"] == 1

    def test_hint_wasted_routes_by_mode(self):
        m = SPFMetrics(_mode="retention", _interval=0)
        m.record_hint_issued("r1")
        m.record_hint_wasted("r1")
        assert m.snapshot()["hints_wasted"] == 1

    def test_harmful_eviction_counter_retention_only(self):
        m = SPFMetrics(_mode="retention", _interval=0)
        m.record_harmful_eviction_avoided()
        m.record_harmful_eviction_avoided()
        assert m.snapshot()["harmful_evictions_avoided"] == 2


# ---------------------------------------------------------------------------
# Step-level legacy name compatibility
# ---------------------------------------------------------------------------

class TestLegacyStepNames:
    def test_prefetch_issued_alias_retention(self):
        step = SPFStepMetrics(mode="retention")
        step.prefetch_issued = 5
        assert step.retention.hints_issued == 5
        assert step.prefetch_issued == 5

    def test_prefetch_issued_alias_prefetch(self):
        step = SPFStepMetrics(mode="prefetch")
        step.prefetch_issued = 7
        assert step.prefetch.hints_issued == 7

    def test_prefetch_hit_alias(self):
        step = SPFStepMetrics(mode="retention")
        step.prefetch_hit = 3
        assert step.retention.hints_used == 3

    def test_prefetch_waste_alias(self):
        step = SPFStepMetrics(mode="prefetch")
        step.prefetch_waste = 2
        assert step.prefetch.prefetch_wasted == 2


# ---------------------------------------------------------------------------
# Controller lifecycle hooks
# ---------------------------------------------------------------------------

def _make_resource(rid_tail: int = 0, session: str = "s0"):
    tokens = list(range(32)) + [rid_tail]
    return ResourceId.for_prefix(tokens, session_id=session)


class TestControllerLifecycle:
    def _ctrl(self, mode: str = "retention") -> SPFController:
        cfg = SPFConfig(enabled=True, mode=mode, metrics_interval=0)
        return SPFController(cfg)

    def test_controller_exposes_mode(self):
        c = self._ctrl("retention")
        assert c.mode == "retention"
        c = self._ctrl("prefetch")
        assert c.mode == "prefetch"

    def test_snapshot_tagged_with_mode(self):
        c = self._ctrl("prefetch")
        snap = c.metrics_snapshot
        assert snap["mode"] == "prefetch"

    def test_mark_hint_and_used_flow(self):
        c = self._ctrl("retention")
        r = _make_resource(1)
        c.mark_hint_issued(r.resource_id)
        # After finishing a request that used it, should be counted.
        c.record_request_finished(r)
        snap = c.metrics_snapshot
        assert snap["hints_issued"] == 1
        assert snap["hints_used"] == 1
        assert snap["outstanding"] == 0

    def test_mark_hint_wasted_flow(self):
        c = self._ctrl("retention")
        r = _make_resource(2)
        c.mark_hint_issued(r.resource_id)
        c.record_hint_wasted(r.resource_id)
        snap = c.metrics_snapshot
        assert snap["hints_issued"] == 1
        assert snap["hints_wasted"] == 1
        assert snap["hints_used"] == 0

    def test_arrival_and_first_token_tracked(self):
        c = self._ctrl("retention")
        r = _make_resource(3)
        c.record_request_arrival(r, arrival_time=1000.0)
        c.record_first_token(r, first_token_time=1000.5)
        # Internal structures populated.
        assert c._request_arrival[r.resource_id] == 1000.0  # noqa: SLF001
        assert c._request_first_token[r.resource_id] == 1000.5  # noqa: SLF001
        c.record_request_finished(r)
        # Lifecycle entries cleared on finish.
        assert r.resource_id not in c._request_arrival  # noqa: SLF001
        assert r.resource_id not in c._request_first_token  # noqa: SLF001

    def test_unhinted_finish_is_noop(self):
        """Finishing a request we never hinted on does not bump
        used / wasted."""
        c = self._ctrl("retention")
        r = _make_resource(4)
        c.record_request_finished(r)
        snap = c.metrics_snapshot
        assert snap["hints_issued"] == 0
        assert snap["hints_used"] == 0

    def test_observe_resource_uses_full_id_not_first_block(self):
        """observe_resource must feed the legacy observe_request
        path with the full-region resource_id, not the legacy
        first_block_hash — otherwise the controller's session state
        would still collapse shared-first-block prompts."""
        c = self._ctrl("retention")
        r = _make_resource(5, session="sX")
        c.observe_resource(r)
        state = c._sessions["sX"]  # noqa: SLF001
        assert state.prefix_history[-1] == r.resource_id
        assert state.prefix_history[-1] != r.first_block_hash
