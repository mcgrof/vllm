# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def _scheduler_output(num_computed_tokens: int, num_output_tokens: int):
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData(
            req_ids=["req"],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[None],
            num_computed_tokens=[num_computed_tokens],
            num_output_tokens=[num_output_tokens],
        ),
        num_scheduled_tokens={"req": 512},
        total_num_scheduled_tokens=512,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def _runner(previous_num_computed_tokens: int):
    runner = GPUModelRunner.__new__(GPUModelRunner)
    num_computed_tokens = Mock()
    total_len = Mock()
    output_bin_counts = torch.zeros(1, 32, dtype=torch.int32)
    output_bin_counts[0, 7] = 1
    runner.req_states = SimpleNamespace(
        req_id_to_index={"req": 0},
        num_computed_tokens_np=np.array([previous_num_computed_tokens], dtype=np.int32),
        num_computed_prefill_tokens=np.array(
            [previous_num_computed_tokens], dtype=np.int32
        ),
        prefill_len=SimpleNamespace(np=np.array([512], dtype=np.int32)),
        num_computed_tokens=num_computed_tokens,
        total_len=total_len,
        last_sampled_tokens=torch.tensor([[24363]], dtype=torch.int64),
        draft_tokens=torch.tensor([[11, 12]], dtype=torch.int64),
    )
    runner.sampler = SimpleNamespace(
        penalties_state=SimpleNamespace(output_bin_counts=output_bin_counts)
    )
    runner.scheduler_config = SimpleNamespace(async_scheduling=False)
    runner.block_tables = Mock()
    runner.kv_block_zeroer = None
    runner.kv_caches = []
    runner.kv_cache_config = SimpleNamespace(num_blocks=0)
    return runner


def test_update_requests_repairs_context_rewind_before_input_preparation():
    runner = _runner(previous_num_computed_tokens=440)

    GPUModelRunner.update_requests(runner, _scheduler_output(0, 0))

    runner.req_states.num_computed_tokens.stage_write_elem.assert_called_once_with(0, 0)
    runner.req_states.num_computed_tokens.apply_write.assert_called_once_with()
    runner.req_states.total_len.stage_write_elem.assert_called_once_with(0, 512)
    runner.req_states.total_len.apply_write.assert_called_once_with()
    assert runner.req_states.num_computed_tokens_np.tolist() == [0]
    assert runner.req_states.num_computed_prefill_tokens.tolist() == [0]
    assert runner.req_states.last_sampled_tokens.tolist() == [[0]]
    assert runner.req_states.draft_tokens.tolist() == [[0, 0]]
    assert not runner.sampler.penalties_state.output_bin_counts.any()


@pytest.mark.parametrize(
    ("previous_num_computed_tokens", "num_output_tokens", "async_scheduling"),
    [(0, 0, False), (440, 0, True), (440, 1, False)],
)
def test_update_requests_leaves_unsupported_rewinds_untouched(
    previous_num_computed_tokens: int,
    num_output_tokens: int,
    async_scheduling: bool,
):
    runner = _runner(previous_num_computed_tokens=previous_num_computed_tokens)
    runner.scheduler_config.async_scheduling = async_scheduling

    GPUModelRunner.update_requests(runner, _scheduler_output(0, num_output_tokens))

    runner.req_states.num_computed_tokens.stage_write_elem.assert_not_called()
    runner.req_states.num_computed_tokens.apply_write.assert_not_called()
    runner.req_states.total_len.stage_write_elem.assert_not_called()
    runner.req_states.total_len.apply_write.assert_not_called()
    assert runner.req_states.last_sampled_tokens.tolist() == [[24363]]
    assert runner.req_states.draft_tokens.tolist() == [[11, 12]]
    assert runner.sampler.penalties_state.output_bin_counts[0, 7] == 1
