# SPDX-License-Identifier: Apache-2.0
"""Byte-level comparison of the single-launch mixed writer."""

import pytest
import torch

from vllm.v1.attention.ops.mixed_reshape_and_cache import mixed_reshape_and_cache


@pytest.mark.parametrize("scale", [0.03125, 0.37, 1.7, 13.0])
@pytest.mark.parametrize("strided", [False, True])
def test_writer_bytes_guards_and_scales(scale, strided):
    torch.manual_seed(173)
    tokens, heads, dim, blocks, page = 49, 8, 128, 6, 16
    if strided:
        key = torch.randn(tokens * 2, heads, dim, device="cuda", dtype=torch.float16)[
            ::2
        ]
        value = torch.randn(tokens * 3, heads, dim, device="cuda", dtype=torch.float16)[
            ::3
        ]
    else:
        key = torch.randn(tokens, heads, dim, device="cuda", dtype=torch.float16)
        value = torch.randn_like(key)
    edges = torch.tensor(
        [
            0.0,
            -0.0,
            65504.0,
            -65504.0,
            448.0,
            -448.0,
            1.0625,
            1.1875,
            2**-24,
            -(2**-24),
        ],
        device="cuda",
        dtype=torch.float16,
    )
    value[0, 0, : edges.numel()] = edges
    # Actual region-split allocation with disjoint typed views of one storage.
    count = blocks * page * heads * dim
    raw = torch.full((count * 3 + 128,), 91, device="cuda", dtype=torch.uint8)
    k = raw[64 : 64 + 2 * count].view(torch.float16).view(blocks, page, heads, dim)
    v = raw[64 + 2 * count : 64 + 3 * count].view(torch.float8_e4m3fn).view_as(k)
    initial = raw.clone()
    old_k, old_v = k.clone(), v.clone()
    slots = torch.randperm(blocks * page, device="cuda")[:tokens].to(torch.int64)
    slots[-2:] = -1
    vs = torch.tensor(scale, dtype=torch.float32, device="cuda")
    ks = torch.tensor(0.123, dtype=torch.float32, device="cuda")
    torch.ops._C_cache_ops.reshape_and_cache_flash(
        key, key, old_k, old_k, slots, "auto", ks, ks
    )
    torch.ops._C_cache_ops.reshape_and_cache_flash(
        value, value, old_v, old_v, slots, "fp8_e4m3", vs, vs
    )
    mixed_reshape_and_cache(key, value, k, v, slots, vs)
    assert torch.equal(k.view(torch.uint8), old_k.view(torch.uint8))
    assert torch.equal(v.view(torch.uint8), old_v.view(torch.uint8))
    valid = slots >= 0
    reference = (value.float() / vs).clamp(-448, 448).to(torch.float8_e4m3fn)
    assert torch.equal(k.flatten(0, 1)[slots[valid]], key[valid])
    assert torch.equal(
        v.flatten(0, 1)[slots[valid]].view(torch.uint8),
        reference[valid].view(torch.uint8),
    )
    written = torch.zeros(blocks * page, device="cuda", dtype=torch.bool)
    written[slots[valid]] = True
    assert torch.equal(
        k.flatten(0, 1)[~written].view(torch.uint8),
        initial[64 : 64 + 2 * count]
        .view(torch.float16)
        .view_as(k)
        .flatten(0, 1)[~written]
        .view(torch.uint8),
    )
    assert torch.equal(raw[:64], initial[:64])
    assert torch.equal(raw[-64:], initial[-64:])


def test_writer_independent_destination_strides():
    key = torch.randn(17, 8, 128, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    k = torch.zeros(4, 16, 8, 128, device="cuda", dtype=torch.float16)[::2]
    v = torch.zeros(6, 16, 8, 128, device="cuda", dtype=torch.float8_e4m3fn)[::3]
    slots = torch.arange(17, device="cuda", dtype=torch.int64)
    scale = torch.tensor(0.3, device="cuda")
    mixed_reshape_and_cache(key, value, k, v, slots, scale)
    assert torch.equal(k[0], key[:16])
    assert torch.equal(
        v[0].view(torch.uint8),
        (value[:16].float() / scale)
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8),
    )
    assert torch.equal(k[1, 0], key[16])


@pytest.mark.parametrize("scale", [0.03125, 0.37, 1.7])
def test_writer_preserves_exceptional_encodings(scale):
    key = torch.randn(4, 8, 128, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    bits = (
        torch.tensor(
            [0x7C00, 0xFC00, 0x7E00, 0xFE00, 0x7C01, 0xFC01, 0x7FFF, 0xFFFF],
            device="cuda",
            dtype=torch.int32,
        )
        .to(torch.int16)
        .view(torch.float16)
    )
    key[0, 0, :8] = bits
    value[0, 0, :8] = bits
    k = torch.zeros(1, 16, 8, 128, device="cuda", dtype=torch.float16)
    v = torch.zeros_like(k, dtype=torch.float8_e4m3fn)
    old_k, old_v = k.clone(), v.clone()
    slots = torch.arange(4, device="cuda", dtype=torch.int64)
    vs = torch.tensor(scale, device="cuda", dtype=torch.float32)
    ks = torch.tensor(0.123, device="cuda", dtype=torch.float32)
    torch.ops._C_cache_ops.reshape_and_cache_flash(
        key, key, old_k, old_k, slots, "auto", ks, ks
    )
    torch.ops._C_cache_ops.reshape_and_cache_flash(
        value, value, old_v, old_v, slots, "fp8_e4m3", vs, vs
    )
    mixed_reshape_and_cache(key, value, k, v, slots, vs)
    assert torch.equal(k.view(torch.uint8), old_k.view(torch.uint8))
    assert torch.equal(v.view(torch.uint8), old_v.view(torch.uint8))
