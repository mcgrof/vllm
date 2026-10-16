# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.worker import startup_plan
from vllm.v1.worker.gpu_worker import Worker
from vllm.v1.worker.startup_plan import (
    maybe_apply_startup_plan,
    maybe_save_startup_plan,
)

# Startup-plan persistence (vllm/v1/worker/startup_plan.py), applied and
# saved by Worker.determine_available_memory / compile_or_warm_up_model.


def _plan_worker(config_hash="abc123", free_memory=78 * GiB_bytes, kv_bytes=None):
    """The minimal Worker surface the startup-plan entry points touch."""
    return SimpleNamespace(
        vllm_config=SimpleNamespace(compute_hash=lambda: config_hash),
        rank=0,
        parallel_config=SimpleNamespace(world_size=1),
        init_snapshot=SimpleNamespace(free_memory=free_memory),
        cache_config=SimpleNamespace(kv_cache_memory_bytes=kv_bytes),
    )


def _plan_platform(name="NVIDIA H100 PCIe"):
    return SimpleNamespace(
        get_device_name=lambda device_id=0: name,
        get_device_total_memory=lambda device_id=0: 80 * GiB_bytes,
        get_device_capability=lambda device_id=0: (9, 0),
    )


@pytest.fixture
def plan_env(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Enable the startup plan, isolated under a tmp cache root."""
    monkeypatch.setenv("VLLM_ENABLE_STARTUP_PLAN", "1")
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    with patch.object(startup_plan, "current_platform", _plan_platform()):
        yield


def test_startup_plan_fingerprint_sensitivity(plan_env):
    """The fingerprint is the OOM-safety key: stable for identical inputs,
    different for anything the profiled value depends on."""
    fp = startup_plan.compute_plan_fingerprint
    base = fp(_plan_worker().vllm_config, 0, 1)
    assert base == fp(_plan_worker().vllm_config, 0, 1)
    assert base != fp(_plan_worker("other").vllm_config, 0, 1)
    assert base != fp(_plan_worker().vllm_config, 1, 2)
    with patch.object(startup_plan, "current_platform", _plan_platform("NVIDIA A100")):
        assert base != fp(_plan_worker().vllm_config, 0, 1)
    with patch("vllm.__version__", "0.0.0+plan-test"):
        assert base != fp(_plan_worker().vllm_config, 0, 1)


def test_startup_plan_apply_gate(plan_env):
    """Only a fingerprint-matching, memory-safe plan is ever applied."""
    maybe_save_startup_plan(_plan_worker(), 50 * GiB_bytes)

    applied = _plan_worker()
    maybe_apply_startup_plan(applied)
    assert applied.cache_config.kv_cache_memory_bytes == 50 * GiB_bytes

    less_memory = _plan_worker(free_memory=60 * GiB_bytes)
    other_config = _plan_worker(config_hash="zzz999")
    for refused in (less_memory, other_config):
        maybe_apply_startup_plan(refused)
        assert refused.cache_config.kv_cache_memory_bytes is None

    # An explicit --kv-cache-memory is never overridden.
    explicit = _plan_worker(kv_bytes=7 * GiB_bytes)
    maybe_apply_startup_plan(explicit)
    assert explicit.cache_config.kv_cache_memory_bytes == 7 * GiB_bytes


@pytest.mark.skip_global_cleanup
def test_initialize_registers_loaded_model_before_kv_cache() -> None:
    events: list[str] = []
    model = MagicMock()
    connector = MagicMock()
    model_runner = MagicMock()
    model_runner.get_model.return_value = model
    connector.register_model.side_effect = lambda _: events.append("register_model")
    model_runner.initialize_kv_cache.side_effect = lambda _: events.append(
        "initialize_kv_cache"
    )
    worker = SimpleNamespace(
        cache_config=SimpleNamespace(num_gpu_blocks=None),
        vllm_config=MagicMock(),
        model_runner=model_runner,
        model_config=SimpleNamespace(enable_return_routed_experts=False),
        _maybe_get_memory_pool_context=lambda **_: nullcontext(),
    )
    kv_cache_config = SimpleNamespace(
        num_blocks=8,
        needs_kv_cache_zeroing=False,
    )

    with (
        patch(
            "vllm.v1.worker.gpu_worker.ensure_kv_transfer_initialized"
        ) as initialize_connector,
        patch("vllm.v1.worker.gpu_worker.has_kv_transfer_group", return_value=True),
        patch(
            "vllm.v1.worker.gpu_worker.get_kv_transfer_group",
            return_value=connector,
        ),
    ):
        Worker.initialize_from_config(worker, kv_cache_config)  # type: ignore[arg-type]

    initialize_connector.assert_called_once_with(worker.vllm_config, kv_cache_config)
    connector.register_model.assert_called_once_with(model)
    assert events == ["register_model", "initialize_kv_cache"]
