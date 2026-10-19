# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The metadata contract is CPU-only and independent of CUDA imports."""

import importlib.util
import sys
from pathlib import Path

import pytest

path = (
    Path(__file__).parents[2] / "vllm/v1/attention/backends/compact_prefill_metadata.py"
)
spec = importlib.util.spec_from_file_location("compact_prefill_metadata", path)
assert spec is not None and spec.loader is not None
metadata = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = metadata
spec.loader.exec_module(metadata)


def make(lengths, **kwargs):
    ptr = [0]
    for length in lengths:
        ptr.append(ptr[-1] + length)
    return metadata.plan_whole_prompt(ptr, lengths, max_sequences=96, **kwargs)


def test_allocation_arithmetic():
    assert metadata.PLANE_BYTES == 128 * 2**20
    assert metadata.PAYLOAD_BYTES == 256 * 2**20
    plan, reason = make([8192])
    assert reason == "candidate" and plan.pages == 512
    assert plan.pages * 16 * 8 * 128 * 4 == 32 * 2**20


def test_unequal_tails_and_repeated_page_occurrences():
    plan, _ = make([8191, 8187, 17, 1])
    assert plan.page_indptr == (0, 512, 1024, 1026, 1027)
    assert plan.last_page_lengths == (15, 11, 1, 1)
    # Source IDs are opaque. Zero, sparse/permuted IDs, and shared pages
    # retain one destination occurrence for every source occurrence.
    ids = [0, 79, 3, 79, 0]
    compact, _ = make([33, 17])
    assert compact.page_indptr == (0, 3, 5)
    expected = [
        [ids[i] for i in range(a, b)]
        for a, b in zip(compact.page_indptr, compact.page_indptr[1:])
    ]
    assert expected == [[0, 79, 3], [79, 0]]


def test_budget_boundary():
    plan, _ = make([8192] * 8)
    assert plan.pages == 4096
    assert make([8192] * 8 + [1])[1] == "page_budget"


def test_variable_batch_uses_configured_limit():
    plan, reason = make([31 + i for i in range(32)])
    assert reason == "candidate" and len(plan.query_lengths) == 32
    assert make([16] * 96)[0].pages == 96
    assert make([16] * 97)[1] == "request_count"


@pytest.mark.parametrize(
    "kw, reason",
    [
        ({"num_decodes": 1}, "decode_or_mixed"),
        ({"causal": False}, "mask_or_prefix"),
        ({"prefix": True}, "mask_or_prefix"),
        ({"supported": False}, "unsupported_geometry"),
    ],
)
def test_unsupported_routes(kw, reason):
    assert make([64], **kw) == (None, reason)


def test_existing_context_and_chunked_prefill():
    assert (
        metadata.plan_whole_prompt([0, 32], [64], max_sequences=96)[1]
        == "context_or_chunk"
    )
    assert make([8193])[1] == "query_length"
    assert make([0])[1] == "query_length"
    assert (
        metadata.plan_whole_prompt([1, 33], [32], max_sequences=96)[1]
        == "request_count"
    )


def subset(query_lengths, kv_lengths=None, *, decodes=0, **kwargs):
    q = [0]
    pages = [0]
    kv_lengths = query_lengths if kv_lengths is None else kv_lengths
    for qn, kn in zip(query_lengths, kv_lengths):
        q.append(q[-1] + qn)
        pages.append(pages[-1] + (kn + 15) // 16)
    plan, reason = metadata.plan_prefill_subset(
        q, kv_lengths, pages, num_decodes=decodes, max_sequences=96, **kwargs
    )
    return plan, reason


def test_mixed_subset_rebases_both_address_spaces():
    plan, reason = subset([1, 2, 33, 17], [8193, 17, 33, 17], decodes=2)
    assert reason == "candidate"
    assert plan.query_lengths == (33, 17)
    assert plan.page_indptr == (0, 3, 5)
    assert plan.last_page_lengths == (1, 1)
    assert (plan.source_page_start, plan.source_page_end) == (515, 520)
    ids = [99] * 515 + [0, 79, 3, 79, 0] + [123]
    selected = ids[plan.source_page_start : plan.source_page_end]
    assert selected == [0, 79, 3, 79, 0]


def test_recorded_admission_subsets_fit_unchanged_payload():
    for decodes, fresh in [(0, 8), (8, 7), (15, 7), (22, 7), (29, 3)]:
        plan, reason = subset(
            [1] * decodes + [8192] * fresh,
            [8193] * decodes + [8192] * fresh,
            decodes=decodes,
        )
        assert reason == "candidate" and plan.pages == fresh * 512
        assert plan.source_page_start == decodes * 513
    assert subset([1] + [8192] * 8, [8193] + [8192] * 8, decodes=1)[0].pages == 4096
    assert (
        subset([1] + [8192] * 8 + [1], [8193] + [8192] * 8 + [1], decodes=1)[1]
        == "page_budget"
    )


@pytest.mark.parametrize(
    "lengths,decodes,reason",
    [([], 0, "decode_only"), ([1, 2], 2, "decode_only"), ([8192], 0, "candidate")],
)
def test_empty_decode_only_and_prefill_only(lengths, decodes, reason):
    assert subset(lengths, decodes=decodes)[1] == reason


def test_subset_preserves_fallbacks_and_rejects_bad_metadata():
    assert subset([1, 32], [65, 64], decodes=1)[1] == "context_or_chunk"
    assert subset([1, 32], [65, 32], decodes=1, causal=False)[1] == "mask_or_prefix"
    assert subset([1, 32], [65, 32], decodes=1, prefix=True)[1] == "mask_or_prefix"
    assert (
        subset([1, 32], [65, 32], decodes=1, supported=False)[1]
        == "unsupported_geometry"
    )
    assert (
        metadata.plan_prefill_subset(
            [0, 1, 33], [65, 32], [0, 5, 6], num_decodes=1, max_sequences=96
        )[1]
        == "page_mapping"
    )
    assert (
        metadata.plan_prefill_subset(
            [0, 1], [1], [0, 1], num_decodes=2, max_sequences=96
        )[1]
        == "request_count"
    )


@pytest.mark.parametrize("shape", [(32, 8, 128), (40, 10, 128), (16, 4, 256)])
def test_production_metadata_branch_slices_actual_source_ids(monkeypatch, shape):
    """Execute the builder's compact branch with real CPU metadata tensors."""
    import ast
    from types import SimpleNamespace

    import torch

    source = path.with_name("flashinfer.py").read_text()
    tree = ast.parse(source)
    branches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "compact is not None"
        and any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "plan_prefill_subset"
            for child in ast.walk(node)
        )
    ]
    assert len(branches) == 1
    fn = ast.FunctionDef(
        name="production",
        args=ast.arguments(
            posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
        ),
        body=branches[0].body,
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    monkeypatch.setitem(
        sys.modules, "vllm.v1.attention.backends.compact_prefill_metadata", metadata
    )
    calls = []
    compact = SimpleNamespace(
        geometry=metadata.select_geometry(*shape),
        calls=__import__("collections").Counter(),
        plan=lambda *args, **kw: calls.append((args, kw)),
    )
    cfg = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=96, enable_chunked_prefill=False)
    )
    owner = SimpleNamespace(
        _is_asymmetric=True,
        use_dcp=False,
        k_cache_dtype=torch.float16,
        v_cache_dtype=torch.float8_e4m3fn,
        q_data_type_prefill=torch.float16,
        model_config=SimpleNamespace(dtype=torch.float16),
        page_size=16,
        num_qo_heads=shape[0],
        num_kv_heads=shape[1],
        head_dim=shape[2],
        window_left=-1,
        logits_soft_cap=None,
        has_sinks=False,
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        vllm_config=cfg,
        sm_scale=128**-0.5,
        prefill_fixed_split_size=-1,
        disable_split_kv=False,
        paged_kv_indptr=SimpleNamespace(cpu=torch.tensor([0, 5, 8, 10])),
    )
    ptr = owner.paged_kv_indptr.cpu
    original = ptr.clone()
    attn = SimpleNamespace(prefill=SimpleNamespace())
    env = dict(
        self=owner,
        compact=compact,
        current_platform=SimpleNamespace(is_device_capability=lambda n: n == 90),
        get_kv_cache_layout=lambda: "NHD",
        torch=torch,
        qo_indptr_cpu=torch.tensor([0, 1, 34, 51]),
        seq_lens_np=[65, 33, 17],
        num_reqs=3,
        num_decodes=1,
        causal=True,
        use_cascade=False,
        qo_indptr_prefill_cpu=torch.tensor([0, 33, 50]),
        paged_kv_indptr_prefill_cpu=ptr[1:],
        paged_kv_last_page_len_prefill_cpu=torch.tensor([1, 1]),
        paged_kv_indices=torch.tensor([99] * 5 + [0, 79, 3, 79, 0]),
        attn_metadata=attn,
    )
    exec(compile(module, str(path.with_name("flashinfer.py")), "exec"), env)
    env["production"]()
    assert len(calls) == 1
    args, _ = calls[0]
    assert args[1].tolist() == [0, 33, 50]
    assert args[2].tolist() == [0, 3, 5]
    assert attn.prefill.source_indices.tolist() == [0, 79, 3, 79, 0]
    assert torch.equal(ptr, original)
    assert args[0] is attn.prefill.compact_plan and attn.prefill.compact is compact
    owner.head_dim = 64
    env["production"]()
    assert len(calls) == 1
    assert compact.calls["batch_unsupported_geometry"] == 1


def test_workspace_lifetime_guards_and_output_placement():
    import ast
    import threading
    from contextlib import contextmanager
    from types import SimpleNamespace

    import torch

    source = path.with_name("compact_prefill.py").read_text()
    cls = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.ClassDef) and n.name == "CompactPrefillWorkspace"
    )
    cls.body = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name in ("serial", "run")
    ]
    tree = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    stream = SimpleNamespace(value=1)
    identity = SimpleNamespace(value=5)
    env = dict(
        contextmanager=contextmanager,
        threading=SimpleNamespace(get_ident=lambda: identity.value),
        torch=SimpleNamespace(
            cuda=SimpleNamespace(
                current_stream=lambda device: SimpleNamespace(cuda_stream=stream.value)
            )
        ),
        CompactPagePlan=metadata.CompactPagePlan,
    )
    exec(compile(tree, str(path.with_name("compact_prefill.py")), "exec"), env)
    workspace = env["CompactPrefillWorkspace"]()
    workspace._lock = threading.Lock()
    workspace._stream = workspace._thread = None
    workspace.backing = SimpleNamespace(device="unused")
    with (
        workspace.serial(),
        pytest.raises(RuntimeError, match="overlapping"),
        workspace.serial(),
    ):
        pass
    stream.value = 2
    with pytest.raises(RuntimeError, match="serial CUDA stream"), workspace.serial():
        pass
    stream.value = 1
    identity.value = 6
    with pytest.raises(RuntimeError, match="submission thread"), workspace.serial():
        pass
    identity.value = 5
    workspace.last_plan = object()
    with pytest.raises(RuntimeError, match="replaced"):
        workspace.run(None, None, None, object())
    assert not workspace._lock.locked()

    # Use the actual forward slices: decode tokens need not equal decode requests.
    forward = ast.parse(path.with_name("flashinfer.py").read_text())
    assignments: list[ast.stmt] = []
    for wanted in [
        "prefill_query = query[num_decode_tokens:]",
        "out_prefill = output[num_decode_tokens:]",
    ]:
        matches = [
            n
            for n in ast.walk(forward)
            if isinstance(n, ast.Assign) and ast.unparse(n) == wanted
        ]
        assert len(matches) == 1
        assignments.append(matches[0])
    q = torch.arange(30).reshape(10, 3)
    out = torch.full_like(q, -1)
    env = dict(query=q, output=out, num_decode_tokens=3)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=assignments, type_ignores=[])),
            "production-forward-slices",
            "exec",
        ),
        env,
    )
    assert torch.equal(env["prefill_query"], q[3:])
    env["out_prefill"].fill_(9)
    assert torch.equal(out[:3], torch.full_like(out[:3], -1))
    assert torch.equal(out[3:], torch.full_like(out[3:], 9))


@pytest.mark.parametrize(
    "shape,payload_mib",
    [((32, 8, 128), 256), ((40, 10, 128), 320), ((16, 4, 256), 256)],
)
def test_geometry_payload_and_gather_coordinates(shape, payload_mib):
    """Every flattened tile element maps to one valid source coordinate."""
    geometry = metadata.select_geometry(*shape)
    assert geometry.payload_bytes == payload_mib * 2**20
    coordinates = [
        (
            x // (geometry.kv_heads * geometry.head_dim),
            (x // geometry.head_dim) % geometry.kv_heads,
            x % geometry.head_dim,
        )
        for x in range(geometry.page_elements)
    ]
    assert len(set(coordinates)) == 16 * shape[1] * shape[2]
    assert coordinates[0] == (0, 0, 0)
    assert coordinates[-1] == (15, shape[1] - 1, shape[2] - 1)


@pytest.mark.parametrize("shape", [(32, 8, 64), (16, 8, 256), (16, 1, 512)])
def test_unqualified_geometries_remain_closed(shape):
    with pytest.raises(ValueError, match="unsupported"):
        metadata.select_geometry(*shape)
