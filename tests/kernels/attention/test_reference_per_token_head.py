# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact E4M3 code/scale, masked-slot and repeated-slot writer checks."""

import numpy as np
import pytest
import torch

from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash_per_token_head_quant,
)
from vllm.v1.kv_cache_interface import KVQuantMode


def reference(operand):
    """Decode the E4M3 codebook arithmetically; do not use a GPU FP8 cast."""
    x = operand.float().cpu().numpy()
    scales = np.maximum(np.abs(x).max(axis=-1), np.float32(1e-12))
    scales = np.multiply(scales, np.float32(1 / 448), dtype=np.float32)
    normalized = np.clip(np.divide(x, scales[..., None], dtype=np.float32), -448, 448)
    codebook = np.array(
        [
            (code % 8) * 2.0**-9
            if code < 8
            else (1 + (code % 8) / 8) * 2.0 ** (code // 8 - 7)
            for code in range(127)
        ],
        dtype=np.float32,
    )
    magnitude = np.abs(normalized)
    upper = np.searchsorted(codebook, magnitude).clip(0, 126)
    lower = (upper - 1).clip(0, 126)
    lo, hi = magnitude - codebook[lower], codebook[upper] - magnitude
    nearest = np.where((hi < lo) | ((hi == lo) & (upper % 2 == 0)), upper, lower)
    codes = nearest.astype(np.uint8) | (np.signbit(normalized).astype(np.uint8) * 128)
    return scales, codes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA writer qualification")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_reference_writer_exact_pages_and_repeated_updates(dtype):
    device = "cuda"
    # One padded token must not touch storage; real tokens use shuffled pages.
    mapping = torch.tensor([17, -1, 0, 34, 7, 23], device=device, dtype=torch.int64)
    base = torch.linspace(-1.5, 1.5, 128, device=device, dtype=torch.float32)
    key = base.expand(6, 8, 128).clone().to(dtype)
    key[0, 0].zero_()
    key[2, 1] = -0.0
    key[3, 2].fill_(torch.finfo(torch.float16).smallest_normal / 1024)
    key[4, 3] = (base.sign() * torch.finfo(torch.float16).max).to(dtype)
    key[5, 4] = torch.tensor(
        [448.0, 1.0625, 1.1875, -1.0625, -1.1875, 0.0, -0.0, 0.0009765625],
        device=device,
        dtype=dtype,
    ).repeat(16)
    value = key.flip(-1).contiguous()
    backing = torch.full((3, 2, 16, 8, 132), 165, dtype=torch.uint8, device=device)
    kcache, vcache = backing.unbind(1)
    scales = [
        plane[..., 128:].view(torch.float32).squeeze(-1) for plane in (kcache, vcache)
    ]
    expected = backing.cpu().numpy().copy()
    for iteration in range(2):
        if iteration:
            # Change the assigned slots and fresh operands; old pages stay intact.
            mapping = torch.tensor(
                [2, -1, 17, 33, 8, 22], device=device, dtype=torch.int64
            )
            key = (key.float() * 0.5).to(dtype)
            value = (value.float() * -0.25).to(dtype)
        for token, slot in enumerate(mapping.cpu().tolist()):
            if slot < 0:
                continue
            for component, operand in enumerate((key, value)):
                scale, codes = reference(operand[token])
                expected[slot // 16, component, slot % 16, :, :128] = codes
                expected[slot // 16, component, slot % 16, :, 128:] = scale.view(
                    np.uint8
                ).reshape(8, 4)
        triton_reshape_and_cache_flash_per_token_head_quant(
            key,
            value,
            kcache.view(torch.float8_e4m3fn),
            vcache.view(torch.float8_e4m3fn),
            *scales,
            mapping,
            KVQuantMode.FP8_PER_TOKEN_HEAD,
            reference_fp8=True,
        )
        np.testing.assert_array_equal(backing.cpu().numpy(), expected)


@pytest.mark.parametrize(
    "mode,dtype",
    [
        (KVQuantMode.INT4_PER_TOKEN_HEAD, torch.uint8),
        (KVQuantMode.INT8_PER_TOKEN_HEAD, torch.int8),
        (KVQuantMode.FP8_PER_TOKEN_HEAD, torch.float16),
    ],
)
def test_reference_policy_rejects_other_formats_before_dispatch(mode, dtype):
    cache = torch.empty(1, dtype=dtype)
    with pytest.raises(ValueError, match="requires FP8 E4M3FN"):
        triton_reshape_and_cache_flash_per_token_head_quant(
            None, None, cache, cache, None, None, None, mode, reference_fp8=True
        )
