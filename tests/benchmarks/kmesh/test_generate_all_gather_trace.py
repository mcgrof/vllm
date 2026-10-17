# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from benchmarks.kmesh import generate_all_gather_trace as generator

_CONTEXT_KEYS = {
    "schema_version",
    "enabled",
    "execution_id",
    "group_kind",
    "group_id",
    "round_id",
    "branch_id",
    "parent_branch_id",
    "state_kind",
    "cache_domain",
    "segment_encoding",
    "segments",
}
_WIRE_SEGMENT_KEYS = {
    "segment_id",
    "kind",
    "token_start",
    "token_end",
    "logical_order",
    "source_id",
    "shareable",
    "lineage_id",
}
_GROUP_KINDS = {"agent_round", "cot_branch"}
_SEGMENT_KINDS = {"private_history", "shared_update", "task", "thought"}


def _assert_wire_contract(record: dict[str, Any]) -> None:
    context = record["kmesh_context"]
    assert isinstance(context, dict)
    assert set(context) == _CONTEXT_KEYS
    assert type(context["schema_version"]) is int
    assert context["schema_version"] == 1
    assert type(context["enabled"]) is bool
    assert context["enabled"] is True
    assert context["group_kind"] in _GROUP_KINDS
    assert context["state_kind"] == "kv_cache"
    assert context["segment_encoding"] == "sideband"
    assert context["group_kind"] == record["lineage"]["group_kind"]
    assert context["round_id"] == record["lineage"]["round_id"]

    wire_segments = context["segments"]
    analytic_segments = record["segments"]
    assert isinstance(wire_segments, list)
    assert len(wire_segments) == len(analytic_segments) > 0
    for logical_order, (wire, analytic) in enumerate(
        zip(wire_segments, analytic_segments)
    ):
        assert set(wire) == _WIRE_SEGMENT_KEYS
        assert wire["kind"] in _SEGMENT_KINDS
        assert type(wire["token_start"]) is int
        assert type(wire["token_end"]) is int
        assert type(wire["logical_order"]) is int
        assert type(wire["shareable"]) is bool
        assert wire["logical_order"] == logical_order
        assert wire["token_start"] < wire["token_end"]
        assert wire == {
            key: analytic[key]
            for key in (
                "segment_id",
                "kind",
                "token_start",
                "token_end",
                "logical_order",
                "source_id",
                "shareable",
                "lineage_id",
            )
        }


@pytest.fixture
def should_do_global_cleanup_after_test() -> bool:
    return False


@pytest.mark.parametrize("case", generator.CASES)
def test_controlled_case_exercises_its_declared_structure(case: str) -> None:
    config = generator.TraceConfig(
        case=case,
        agents=3,
        shared_fraction=0.8,
        alignment="unaligned",
    )

    records, manifest = generator.generate_trace(config)

    assert len(records) == config.agents
    assert manifest["case_expectations"]["fully_exercised"]
    assert manifest["case_expectations"]["metadata_exercised"]
    assert manifest["case_expectations"]["overlap_gate_exercised"]
    assert all(manifest["case_expectations"]["checks"].values())
    assert manifest["counts"]["requests"] == config.agents
    assert manifest["counts"]["records"] == config.agents
    assert all(record["case"] == case for record in records)
    assert all(record["segments"] for record in records)
    for record in records:
        _assert_wire_contract(record)
    assert all(
        segment["length"] == segment["token_end"] - segment["token_start"]
        for record in records
        for segment in record["segments"]
    )


@pytest.mark.parametrize("agents", generator.AGENT_COUNTS)
@pytest.mark.parametrize("shared_fraction", generator.SHARED_FRACTIONS)
@pytest.mark.parametrize("alignment", generator.ALIGNMENTS)
def test_agent_fraction_and_alignment_axes_are_supported(
    agents: int, shared_fraction: float, alignment: str
) -> None:
    config = generator.TraceConfig(
        case="same_order_different_offsets",
        agents=agents,
        shared_fraction=shared_fraction,
        alignment=alignment,
    )

    records, manifest = generator.generate_trace(config)

    assert len(records) == agents
    observed = manifest["observed"]["shareable_fraction_mean"]
    assert observed == pytest.approx(shared_fraction, abs=0.04)
    shared_segments = [
        segment
        for record in records
        for segment in record["segments"]
        if segment["kind"] == "shared_update"
    ]
    if shared_fraction == 0.0:
        assert not shared_segments
    elif alignment == "aligned":
        assert all(
            segment["length"] % config.block_size == 0 for segment in shared_segments
        )
    else:
        assert all(
            segment["length"] % config.block_size != 0 for segment in shared_segments
        )


def test_wire_contract_across_complete_supported_axis_product() -> None:
    for case, agents, fraction, alignment, qps, mode in itertools.product(
        generator.CASES,
        generator.AGENT_COUNTS,
        generator.SHARED_FRACTIONS,
        generator.ALIGNMENTS,
        generator.QPS_VALUES,
        generator.MODES,
    ):
        records, manifest = generator.generate_trace(
            generator.TraceConfig(
                case=case,
                agents=agents,
                shared_fraction=fraction,
                alignment=alignment,
                qps=qps,
                mode=mode,
            )
        )

        assert manifest["config"]["nominal_shared_fraction"] == fraction
        assert "shared_fraction" not in manifest["config"]
        for record in records:
            _assert_wire_contract(record)


def test_deterministic_geometry_fuzz() -> None:
    rng = random.Random(20260731)
    for _ in range(64):
        block_size = rng.choice((8, 16, 32, 64))
        config = generator.TraceConfig(
            case=rng.choice(generator.CASES),
            agents=rng.choice(generator.AGENT_COUNTS),
            shared_fraction=rng.choice(generator.SHARED_FRACTIONS),
            alignment=rng.choice(generator.ALIGNMENTS),
            block_size=block_size,
            qps=rng.choice(generator.QPS_VALUES),
            mode=rng.choice(generator.MODES),
            seed=rng.randrange(0, 2**31),
            tokens_per_agent=max(512, block_size * 8),
        )

        first_records, first_manifest = generator.generate_trace(config)
        second_records, second_manifest = generator.generate_trace(config)

        assert generator._serialize_jsonl(first_records) == generator._serialize_jsonl(
            second_records
        )
        assert generator._canonical_json(first_manifest) == generator._canonical_json(
            second_manifest
        )
        for record in first_records:
            _assert_wire_contract(record)
            generator._validate_record(record, block_size)


def test_live_parser_handoff_when_kmesh_runtime_is_present() -> None:
    segments_module = pytest.importorskip("vllm.v1.kmesh.segments")
    for case in generator.CASES:
        records, _ = generator.generate_trace(
            generator.TraceConfig(
                case=case,
                agents=3,
                shared_fraction=0.8,
                alignment="unaligned",
            )
        )
        for record in records:
            result = segments_module.parse_kmesh_request_context(
                record["kmesh_context"], record["prompt_token_ids"]
            )
            assert result.is_enabled, result.to_primitive()


@pytest.mark.parametrize("qps", generator.QPS_VALUES)
@pytest.mark.parametrize(
    ("mode", "expected_max_new_tokens"),
    (("prefill_focus", 1), ("round_e2e", 32)),
)
def test_load_and_mode_axes_are_recorded(
    qps: int, mode: str, expected_max_new_tokens: int
) -> None:
    records, _ = generator.generate_trace(
        generator.TraceConfig(
            case="mixed_cache_domains",
            agents=3,
            qps=qps,
            mode=mode,
        )
    )

    assert all(record["workload"]["qps"] == qps for record in records)
    assert all(record["workload"]["mode"] == mode for record in records)
    assert all(
        record["workload"]["max_new_tokens"] == expected_max_new_tokens
        for record in records
    )


def test_written_trace_and_manifest_are_byte_deterministic(tmp_path: Path) -> None:
    config = generator.TraceConfig(
        case="different_shared_order",
        agents=5,
        shared_fraction=0.9,
        alignment="unaligned",
        qps=4,
        mode="round_e2e",
    )

    first_trace, first_manifest = generator.write_trace(config, tmp_path / "first")
    second_trace, second_manifest = generator.write_trace(config, tmp_path / "second")

    assert first_trace.read_bytes() == second_trace.read_bytes()
    assert first_manifest.read_bytes() == second_manifest.read_bytes()
    manifest = json.loads(first_manifest.read_text())
    assert (
        manifest["trace_sha256"] == hashlib.sha256(first_trace.read_bytes()).hexdigest()
    )


def test_seed_changes_tokens_but_not_structural_counts() -> None:
    first_records, first_manifest = generator.generate_trace(
        generator.TraceConfig(
            case="missing_shared_segment",
            agents=3,
            seed=generator.DEFAULT_SEED,
        )
    )
    second_records, second_manifest = generator.generate_trace(
        generator.TraceConfig(
            case="missing_shared_segment",
            agents=3,
            seed=generator.DEFAULT_SEED + 1,
        )
    )

    assert first_records[0]["trace_id"] != second_records[0]["trace_id"]
    assert first_records[0]["prompt_token_ids"] != second_records[0]["prompt_token_ids"]
    assert first_manifest["counts"] == second_manifest["counts"]
    assert (
        first_manifest["oracle"]["aggregate"] == second_manifest["oracle"]["aggregate"]
    )


def test_nominal_fraction_is_distinct_from_observed_distribution() -> None:
    records, manifest = generator.generate_trace(
        generator.TraceConfig(
            case="cot_branch_tree",
            agents=3,
            shared_fraction=0.95,
            alignment="unaligned",
        )
    )

    assert manifest["config"]["nominal_shared_fraction"] == 0.95
    assert "shared_fraction" not in manifest["config"]
    observed = manifest["observed"]
    by_request = observed["shareable_fraction_by_request"]
    assert set(by_request) == {record["request_id"] for record in records}
    assert observed["shareable_fraction_mean"] == pytest.approx(
        sum(by_request.values()) / len(by_request)
    )
    assert sum(observed["shareable_fraction_bucket_counts"].values()) == len(records)


@pytest.mark.parametrize("case", ("incompatible_round_ids", "mixed_cache_domains"))
def test_compatibility_controls_gate_raw_structural_overlap(case: str) -> None:
    records, manifest = generator.generate_trace(
        generator.TraceConfig(case=case, agents=3, shared_fraction=0.8)
    )

    assert all(
        record["prompt_token_ids"] == records[0]["prompt_token_ids"]
        for record in records[1:]
    )
    controlled_field = (
        "round_id" if case == "incompatible_round_ids" else "cache_domain"
    )
    differing_fields = {
        key
        for key in records[0]["compatibility_key"]
        if len(
            {
                generator._canonical_json(record["compatibility_key"][key])
                for record in records
            }
        )
        > 1
    }
    assert differing_fields == {controlled_field}
    aggregate = manifest["oracle"]["aggregate"]
    assert aggregate["raw_structural_shared_tokens_total"] > 0
    assert (
        0
        < aggregate["eligible_structural_shared_tokens_total"]
        < aggregate["raw_structural_shared_tokens_total"]
    )
    assert aggregate["compatible_pair_count"] < aggregate["pair_count"]
    assert len(manifest["oracle"]["compatibility_groups"]) == 2


@pytest.mark.parametrize("case", ("incompatible_round_ids", "mixed_cache_domains"))
def test_zero_sharing_exercises_metadata_but_not_overlap_gate(case: str) -> None:
    records, manifest = generator.generate_trace(
        generator.TraceConfig(case=case, agents=3, shared_fraction=0.0)
    )

    expectations = manifest["case_expectations"]
    assert expectations["metadata_exercised"]
    assert not expectations["overlap_gate_exercised"]
    assert not expectations["fully_exercised"]
    assert "structurally shared tokens" in expectations["degenerate_reason"]
    assert all(
        record["prompt_token_ids"] == records[0]["prompt_token_ids"]
        for record in records[1:]
    )


def test_single_agent_is_not_reported_as_exercising_overlap() -> None:
    _, manifest = generator.generate_trace(
        generator.TraceConfig(case="cot_branch_tree", agents=1, shared_fraction=0.8)
    )

    expectations = manifest["case_expectations"]
    assert not expectations["metadata_exercised"]
    assert not expectations["overlap_gate_exercised"]
    assert not expectations["fully_exercised"]
    assert "at least two requests" in expectations["degenerate_reason"]


def test_cot_shared_lineage_is_rooted_and_branch_local_lineage_stays_leaf() -> None:
    records, manifest = generator.generate_trace(
        generator.TraceConfig(case="cot_branch_tree", agents=3, shared_fraction=0.8)
    )

    expectation = manifest["observer_expectation"]
    assert expectation["supported_request_count"] == 0
    assert expectation["required_fallback_counts"] == {"group_kind_unsupported": 3}
    assert expectation["batch_group_presence_expected"] is False
    assert expectation["serial_paced_group_presence_expected"] is False

    shared_lineage_by_id: dict[str, set[str]] = {}
    for record in records:
        leaf_branch = record["lineage"]["branch_id"]
        for segment in record["segments"]:
            if segment["kind"] == "shared_update":
                assert segment["source_id"] == "all_gather_coordinator"
                assert segment["source_lineage"]["branch_id"] == "root"
                shared_lineage_by_id.setdefault(segment["segment_id"], set()).add(
                    segment["lineage_id"]
                )
            elif segment["kind"] in {"private_history", "thought"}:
                assert segment["source_id"] == record["agent_id"]
                assert segment["source_lineage"]["branch_id"] == leaf_branch
                assert f":{leaf_branch}:" in segment["lineage_id"]
    assert shared_lineage_by_id
    assert all(len(lineages) == 1 for lineages in shared_lineage_by_id.values())


def test_different_private_lengths_are_structural_but_not_step_compatible() -> None:
    _, manifest = generator.generate_trace(
        generator.TraceConfig(
            case="different_private_history_lengths",
            agents=3,
            shared_fraction=0.8,
        )
    )

    aggregate = manifest["oracle"]["aggregate"]
    assert aggregate["raw_structural_shared_tokens_total"] > 0
    assert aggregate["eligible_structural_shared_tokens_total"] == 0
    assert aggregate["compatible_pair_count"] == 0


def test_order_control_preserves_overlap_but_reduces_ordered_prefix() -> None:
    _, manifest = generator.generate_trace(
        generator.TraceConfig(
            case="different_shared_order",
            agents=3,
            shared_fraction=0.8,
        )
    )

    aggregate = manifest["oracle"]["aggregate"]
    assert aggregate["eligible_structural_shared_tokens_total"] > 0
    assert (
        aggregate["ordered_shared_prefix_tokens_total"]
        < aggregate["raw_structural_shared_tokens_total"]
    )


def test_offset_control_distinguishes_position_independent_overlap() -> None:
    _, manifest = generator.generate_trace(
        generator.TraceConfig(
            case="same_order_different_offsets",
            agents=3,
            shared_fraction=0.8,
        )
    )

    aggregate = manifest["oracle"]["aggregate"]
    expectation = manifest["observer_expectation"]
    assert expectation["supported_request_count"] == 3
    assert expectation["required_fallback_counts"] == {}
    assert expectation["batch_group_presence_expected"] is True
    assert expectation["serial_paced_group_presence_expected"] is False
    assert aggregate["eligible_structural_shared_tokens_total"] > 0
    assert (
        aggregate["same_absolute_offset_shared_tokens_total"]
        < aggregate["raw_structural_shared_tokens_total"]
    )
    assert (
        aggregate["raw_prompt_aligned_full_blocks_total"]
        < aggregate["raw_segment_relative_full_blocks_total"]
    )


def test_oracle_recomputes_overlap_from_emitted_prompt_slices() -> None:
    records, _ = generator.generate_trace(
        generator.TraceConfig(
            case="same_order_different_offsets",
            agents=3,
            shared_fraction=0.8,
        )
    )
    baseline = generator.compute_structural_overlap(records, block_size=32)
    mutated = copy.deepcopy(records)
    changed_segment = next(
        segment
        for segment in mutated[1]["segments"]
        if segment["kind"] == "shared_update"
    )
    token_index = changed_segment["token_start"]
    mutated[1]["prompt_token_ids"][token_index] += 1
    changed_tokens = mutated[1]["prompt_token_ids"][
        changed_segment["token_start"] : changed_segment["token_end"]
    ]
    changed_segment["content_sha256"] = generator._content_sha256(changed_tokens)

    recomputed = generator.compute_structural_overlap(mutated, block_size=32)

    baseline_pair = baseline["pairwise"][0]
    recomputed_pair = recomputed["pairwise"][0]
    assert (
        recomputed_pair["raw_structural_shared_tokens"]
        == baseline_pair["raw_structural_shared_tokens"] - changed_segment["length"]
    )
    assert changed_segment["segment_id"] not in recomputed_pair["shared_segment_ids"]


def test_cli_reproduces_one_case_without_runtime_imports(
    tmp_path: Path,
) -> None:
    script = Path(generator.__file__)
    output_dir = tmp_path / "trace"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--case",
            "cot_branch_tree",
            "--agents",
            "3",
            "--shared-fraction",
            "0.5",
            "--alignment",
            "unaligned",
            "--qps",
            "2",
            "--mode",
            "round_e2e",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(result.stdout)
    assert Path(summary["trace"]) == output_dir / generator.TRACE_FILE
    assert Path(summary["manifest"]) == output_dir / generator.MANIFEST_FILE
    manifest = json.loads((output_dir / generator.MANIFEST_FILE).read_text())
    assert manifest["case_expectations"]["fully_exercised"]


@pytest.mark.parametrize(
    "kwargs",
    (
        {"case": "unknown"},
        {"case": generator.CASES[0], "agents": 2},
        {"case": generator.CASES[0], "shared_fraction": 0.7},
        {"case": generator.CASES[0], "alignment": "partial"},
        {"case": generator.CASES[0], "block_size": 4},
        {"case": generator.CASES[0], "qps": 3},
        {"case": generator.CASES[0], "mode": "decode_only"},
        {"case": generator.CASES[0], "tokens_per_agent": 128},
    ),
)
def test_invalid_axis_values_fail_closed(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        generator.TraceConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"case": True},
        {"case": generator.CASES[0], "agents": True},
        {"case": generator.CASES[0], "agents": 3.0},
        {"case": generator.CASES[0], "shared_fraction": True},
        {"case": generator.CASES[0], "shared_fraction": 1},
        {"case": generator.CASES[0], "shared_fraction": "0.8"},
        {"case": generator.CASES[0], "alignment": False},
        {"case": generator.CASES[0], "block_size": True},
        {"case": generator.CASES[0], "block_size": 32.0},
        {"case": generator.CASES[0], "qps": True},
        {"case": generator.CASES[0], "mode": 1},
        {"case": generator.CASES[0], "seed": False},
        {"case": generator.CASES[0], "tokens_per_agent": 512.0},
    ),
)
def test_programmatic_scalar_types_fail_closed(kwargs: dict[str, object]) -> None:
    with pytest.raises(TypeError):
        generator.TraceConfig(**kwargs)
