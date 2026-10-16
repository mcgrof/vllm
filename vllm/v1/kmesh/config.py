# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from numbers import Real
from typing import TYPE_CHECKING, Any, cast

import torch

from vllm.v1.attention.backends.registry import AttentionBackendEnum

if TYPE_CHECKING:
    from vllm.config import VllmConfig


KMESH_TARGET_PROFILE_KEY = "kmesh_target_profile"
KMESH_SERIAL_PIC_V1_PROFILE = "serial_pic_v1"
KMESH_COLLECTIVE_LOCKSTEP_V1_PROFILE = "collective_lockstep_v1"
KMESH_COLLECTIVE_BATCHED_POSITION_RECOVERY_V1_PROFILE = (
    "collective_batched_position_recovery_v1"
)
KMESH_COLLECTIVE_BATCHED_QKV_V1_PROFILE = "collective_batched_qkv_v1"
KMESH_SERIAL_PIC_V1_MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
KMESH_SERIAL_PIC_V1_MODEL_PATH = (
    "/scratch/hf/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/"
    + KMESH_SERIAL_PIC_V1_MODEL_REVISION
)

_KMESH_TARGET_PROFILE_QUALIFICATION_ATTR = (
    "_kmesh_internal_target_profile_qualification"
)
_KMESH_TARGET_PROFILE_AUTHORITY = object()

_KMESH_TARGET_PROFILE_SHARED_CONFIG_FIELDS = (
    "model_config",
    "cache_config",
    "parallel_config",
    "scheduler_config",
    "device_config",
    "load_config",
    "offload_config",
    "attention_config",
    "mamba_config",
    "kernel_config",
    "lora_config",
    "speculative_config",
    "diffusion_config",
    "compilation_config",
    "structured_outputs_config",
    "observability_config",
    "quant_config",
    "profiler_config",
    "kv_events_config",
    "ec_transfer_config",
    "ec_manager_config",
    "reasoning_config",
    "additional_config",
    "weight_transfer_config",
)


@dataclass(frozen=True, slots=True)
class _KMeshTargetProfileQualification:
    """Unforgeable authority plus frozen shared and child evidence."""

    authority: object
    target_profile: str
    shared_config_ids: tuple[tuple[str, int], ...]
    shared_state: tuple[tuple[str, object], ...]
    child_kv_transfer_config_id: int
    child_kv_transfer_state: object


class KMeshMode(str, Enum):
    OFF = "off"
    OBSERVE = "observe"
    SERIAL_PIC = "serial_pic"
    COLLECTIVE_LOCKSTEP = "collective_lockstep"
    COLLECTIVE_BATCHED = "collective_batched"
    MASTER_MIRROR_REFERENCE = "master_mirror_reference"


_KMESH_TARGET_PROFILE_MODES = {
    KMESH_SERIAL_PIC_V1_PROFILE: KMeshMode.SERIAL_PIC,
    KMESH_COLLECTIVE_LOCKSTEP_V1_PROFILE: KMeshMode.COLLECTIVE_LOCKSTEP,
    KMESH_COLLECTIVE_BATCHED_POSITION_RECOVERY_V1_PROFILE: (
        KMeshMode.COLLECTIVE_BATCHED
    ),
    KMESH_COLLECTIVE_BATCHED_QKV_V1_PROFILE: KMeshMode.COLLECTIVE_BATCHED,
}

# These profiles share a collective control plane. The external worker runs
# per-member operations or batches cached-key recovery, with optional batching
# of model-side query/key rotary encoding and check-layer differences.
KMESH_COLLECTIVE_PROFILES = (
    KMESH_COLLECTIVE_LOCKSTEP_V1_PROFILE,
    KMESH_COLLECTIVE_BATCHED_POSITION_RECOVERY_V1_PROFILE,
    KMESH_COLLECTIVE_BATCHED_QKV_V1_PROFILE,
)


_CONFIG_FIELDS = frozenset(
    {
        "enabled",
        "mode",
        "segment_encoding",
        "pic_backend",
        "check_layer",
        "recompute_ratio",
        "max_group_size",
        "max_group_wait_ms",
        "require_same_active_length",
        "fail_open",
        "strict_validation",
        "master_selection",
        "storage_mode",
    }
)


def _strict_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"kmesh.{field} must be a boolean")
    return value


def _strict_int(value: object, field: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"kmesh.{field} must be an integer")
    if value < minimum:
        raise ValueError(f"kmesh.{field} must be at least {minimum}")
    return value


def _bounded_ratio(value: object) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError("kmesh.recompute_ratio must be a real number")
    ratio = float(value)
    if not 0.0 < ratio <= 1.0:
        raise ValueError("kmesh.recompute_ratio must be in (0, 1]")
    return ratio


def _choice(value: object, field: str, choices: frozenset[str]) -> str:
    if not isinstance(value, str):
        raise TypeError(f"kmesh.{field} must be a string")
    if value not in choices:
        expected = ", ".join(sorted(choices))
        raise ValueError(f"kmesh.{field} must be one of: {expected}")
    return value


def _target_profile_mode(profile: object) -> KMeshMode:
    if type(profile) is not str or profile not in _KMESH_TARGET_PROFILE_MODES:
        raise ValueError(f"unsupported KMesh target profile: {profile!r}")
    return _KMESH_TARGET_PROFILE_MODES[profile]


def _target_profile_error(profile: str, requirement: str, actual: object) -> ValueError:
    return ValueError(
        f"KMesh target profile {profile!r} requires {requirement}; got {actual!r}"
    )


def _require_target_profile(
    profile: str, actual: object, expected: object, requirement: str
) -> None:
    if isinstance(expected, Enum):
        matches = actual is expected
    elif expected is None:
        matches = actual is None
    elif type(expected) in (bool, int, float, str):
        matches = type(actual) is type(expected) and actual == expected
    else:
        matches = actual == expected
    if not matches:
        raise _target_profile_error(profile, requirement, actual)


def _resolved_model_flag(model_config: object, name: str) -> object:
    value = getattr(model_config, name, None)
    return value() if callable(value) else value


def _enum_value(value: object) -> object:
    return value.value if isinstance(value, Enum) else value


def _freeze_qualification_value(value: object) -> object:
    if value is None:
        return ("builtins", "NoneType", None)
    if type(value) in (bool, int, float, str):
        return (type(value).__module__, type(value).__qualname__, value)
    if isinstance(value, torch.dtype):
        return ("torch", "dtype", str(value))
    if isinstance(value, Enum):
        return (
            type(value).__module__,
            type(value).__qualname__,
            _freeze_qualification_value(value.value),
        )
    if isinstance(value, (list, tuple)):
        return (
            type(value).__module__,
            type(value).__qualname__,
            tuple(_freeze_qualification_value(item) for item in value),
        )
    if isinstance(value, Mapping):
        frozen_items = tuple(
            sorted(
                (
                    (
                        _freeze_qualification_value(key),
                        _freeze_qualification_value(item),
                    )
                    for key, item in value.items()
                ),
                key=repr,
            )
        )
        return (type(value).__module__, type(value).__qualname__, frozen_items)
    if isinstance(value, (set, frozenset)):
        return (
            type(value).__module__,
            type(value).__qualname__,
            tuple(
                sorted((_freeze_qualification_value(item) for item in value), key=repr)
            ),
        )
    if is_dataclass(value) and not isinstance(value, type):
        return (
            type(value).__module__,
            type(value).__qualname__,
            tuple(
                (
                    field.name,
                    _freeze_qualification_value(getattr(value, field.name)),
                )
                for field in fields(value)
            ),
        )
    if isinstance(value, type):
        return (value.__module__, value.__qualname__)
    return (type(value).__module__, type(value).__qualname__, repr(value))


def _snapshot_resolved_hf_config(config: object) -> object:
    """Freeze the complete resolved HF object, not only selected aliases."""

    if config is None:
        return _freeze_qualification_value(None)
    to_dict = getattr(config, "to_dict", None)
    contents = to_dict() if callable(to_dict) else getattr(config, "__dict__", config)
    return (
        type(config).__module__,
        type(config).__qualname__,
        _freeze_qualification_value(contents),
    )


_KMESH_SERIAL_PIC_V1_ABSENT_HF_EXECUTION_FIELDS = (
    # Generic ModelArchConfigConvertor geometry aliases. Presence matters for
    # the KV-head aliases because getattr_iter() stops at the first attribute,
    # even when its value is None.
    "head_dim",
    "hidden_size_per_head",
    "n_head_kv",
    "num_kv_heads",
    "multi_query_group_num",
    "num_attention_groups",
    # Generic MoE/loader selection inputs.
    "num_experts",
    "moe_num_experts",
    "n_routed_experts",
    "num_local_experts",
    "block_configs",
    # Live checkpoint quantization inputs read by both the architecture
    # convertor and weight loaders, independently of ModelConfig's aliases.
    "quantization_config",
    "compression_config",
    "text_config",
    "decoder",
    "generator",
    "text_encoder",
    # Generic attention/backend and input-buffer selection inputs.
    "is_mm_prefix_lm",
    "rswa_window",
    "index_topk",
    "projection_dim",
    "projection_size",
    # Alternate maximum-length inputs consumed by the generic convertor.
    "n_positions",
    "max_seq_len",
    "seq_length",
    "model_max_length",
    "max_target_positions",
    "max_sequence_length",
    "seq_len",
)


def _serial_pic_v1_resolved_head_dim(hf_text_config: object) -> object:
    """Resolve the generic cache-layout head dimension without loose coercion."""

    head_dim = getattr(hf_text_config, "head_dim", None)
    if type(head_dim) is int and head_dim > 0:
        return head_dim
    if head_dim is not None:
        return head_dim

    hidden_size_per_head = getattr(hf_text_config, "hidden_size_per_head", None)
    if hidden_size_per_head is not None:
        return hidden_size_per_head

    hidden_size = getattr(hf_text_config, "hidden_size", None)
    num_attention_heads = getattr(hf_text_config, "num_attention_heads", None)
    if (
        type(hidden_size) is int
        and type(num_attention_heads) is int
        and num_attention_heads > 0
    ):
        return hidden_size // num_attention_heads
    return None


def _serial_pic_v1_resolved_num_kv_heads(hf_text_config: object) -> object:
    """Mirror the generic convertor's presence-sensitive KV-head aliases."""

    for field_name in (
        "n_head_kv",
        "num_kv_heads",
        "num_key_value_heads",
        "multi_query_group_num",
        "num_attention_groups",
    ):
        if hasattr(hf_text_config, field_name):
            return getattr(hf_text_config, field_name)
    return getattr(hf_text_config, "num_attention_heads", None)


def _serial_pic_v1_absent_hf_execution_state(
    hf_text_config: object,
) -> tuple[tuple[str, object], ...]:
    """Freeze exact absence plus value for generic execution aliases."""

    return tuple(
        item
        for field_name in _KMESH_SERIAL_PIC_V1_ABSENT_HF_EXECUTION_FIELDS
        for item in (
            (
                f"{field_name}.present",
                _freeze_qualification_value(hasattr(hf_text_config, field_name)),
            ),
            (
                field_name,
                _freeze_qualification_value(getattr(hf_text_config, field_name, None)),
            ),
        )
    )


def _serial_pic_v1_hf_execution_state(hf_text_config: object) -> object:
    """Return every resolved Qwen2 field that can select or alter execution."""

    resolved_defaults = {
        "dual_chunk_attention_config": None,
        "is_causal": True,
        "qk_norm": False,
    }
    field_names = (
        "architectures",
        "model_type",
        "attention_dropout",
        "bos_token_id",
        "eos_token_id",
        "hidden_act",
        "hidden_size",
        "initializer_range",
        "intermediate_size",
        "max_position_embeddings",
        "max_window_layers",
        "num_attention_heads",
        "num_hidden_layers",
        "num_key_value_heads",
        "rms_norm_eps",
        "rope_parameters",
        "sliding_window",
        "tie_word_embeddings",
        "dtype",
        "use_cache",
        "use_sliding_window",
        "vocab_size",
        "layer_types",
        "dual_chunk_attention_config",
        "is_causal",
        "qk_norm",
    )
    resolved_state = tuple(
        (
            field_name,
            _freeze_qualification_value(
                getattr(
                    hf_text_config,
                    field_name,
                    resolved_defaults.get(field_name),
                )
            ),
        )
        for field_name in field_names
    )
    return (
        *resolved_state,
        (
            "resolved_head_dim",
            _freeze_qualification_value(
                _serial_pic_v1_resolved_head_dim(hf_text_config)
            ),
        ),
        (
            "resolved_num_kv_heads",
            _freeze_qualification_value(
                _serial_pic_v1_resolved_num_kv_heads(hf_text_config)
            ),
        ),
        *_serial_pic_v1_absent_hf_execution_state(hf_text_config),
    )


def _serial_pic_v1_expected_model_arch_config() -> object:
    """Return the exact generic architecture conversion for the target model."""

    from vllm.config.model_arch import ModelArchitectureConfig

    return ModelArchitectureConfig(
        architectures=["Qwen2ForCausalLM"],
        model_type="qwen2",
        text_model_type="qwen2",
        hidden_size=896,
        total_num_hidden_layers=24,
        total_num_attention_heads=14,
        head_size=64,
        vocab_size=151936,
        total_num_kv_heads=2,
        num_experts=0,
        quantization_config=None,
        is_deepseek_mla=False,
        is_mm_prefix_lm=False,
        rswa_window=None,
        derived_max_model_len_and_key=(32768.0, "max_position_embeddings"),
    )


_KMESH_SERIAL_PIC_V1_HF_EXECUTION_STATE = (
    ("architectures", _freeze_qualification_value(["Qwen2ForCausalLM"])),
    ("model_type", _freeze_qualification_value("qwen2")),
    ("attention_dropout", _freeze_qualification_value(0.0)),
    ("bos_token_id", _freeze_qualification_value(151643)),
    ("eos_token_id", _freeze_qualification_value(151645)),
    ("hidden_act", _freeze_qualification_value("silu")),
    ("hidden_size", _freeze_qualification_value(896)),
    ("initializer_range", _freeze_qualification_value(0.02)),
    ("intermediate_size", _freeze_qualification_value(4864)),
    ("max_position_embeddings", _freeze_qualification_value(32768)),
    ("max_window_layers", _freeze_qualification_value(21)),
    ("num_attention_heads", _freeze_qualification_value(14)),
    ("num_hidden_layers", _freeze_qualification_value(24)),
    ("num_key_value_heads", _freeze_qualification_value(2)),
    ("rms_norm_eps", _freeze_qualification_value(1e-6)),
    (
        "rope_parameters",
        _freeze_qualification_value(
            {"rope_theta": 1_000_000.0, "rope_type": "default"}
        ),
    ),
    ("sliding_window", _freeze_qualification_value(None)),
    ("tie_word_embeddings", _freeze_qualification_value(True)),
    ("dtype", _freeze_qualification_value(torch.bfloat16)),
    ("use_cache", _freeze_qualification_value(True)),
    ("use_sliding_window", _freeze_qualification_value(False)),
    ("vocab_size", _freeze_qualification_value(151936)),
    ("layer_types", _freeze_qualification_value(["full_attention"] * 24)),
    ("dual_chunk_attention_config", _freeze_qualification_value(None)),
    ("is_causal", _freeze_qualification_value(True)),
    ("qk_norm", _freeze_qualification_value(False)),
    ("resolved_head_dim", _freeze_qualification_value(64)),
    ("resolved_num_kv_heads", _freeze_qualification_value(2)),
) + tuple(
    item
    for field_name in _KMESH_SERIAL_PIC_V1_ABSENT_HF_EXECUTION_FIELDS
    for item in (
        (f"{field_name}.present", _freeze_qualification_value(False)),
        (field_name, _freeze_qualification_value(None)),
    )
)


def _snapshot_fields(
    label: str, config: object, field_names: tuple[str, ...]
) -> tuple[tuple[str, object], ...]:
    return tuple(
        (
            f"{label}.{field_name}",
            _freeze_qualification_value(getattr(config, field_name, None)),
        )
        for field_name in field_names
    )


def _resolved_scheduler_class(scheduler_config: object) -> object:
    resolver = getattr(scheduler_config, "get_scheduler_cls", None)
    if callable(resolver):
        return resolver()
    scheduler_cls = getattr(scheduler_config, "scheduler_cls", None)
    if scheduler_cls is not None:
        return scheduler_cls
    from vllm.v1.core.sched.scheduler import Scheduler

    return Scheduler


def _serial_pic_v1_shared_state(
    vllm_config: VllmConfig,
) -> tuple[tuple[str, object], ...]:
    model_config = getattr(vllm_config, "model_config", None)
    cache_config = getattr(vllm_config, "cache_config", None)
    parallel_config = getattr(vllm_config, "parallel_config", None)
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    device_config = getattr(vllm_config, "device_config", None)
    load_config = getattr(vllm_config, "load_config", None)
    offload_config = getattr(vllm_config, "offload_config", None)
    attention_config = getattr(vllm_config, "attention_config", None)
    kernel_config = getattr(vllm_config, "kernel_config", None)
    compilation_config = getattr(vllm_config, "compilation_config", None)
    structured = getattr(vllm_config, "structured_outputs_config", None)
    hf_text_config = getattr(model_config, "hf_text_config", None)
    rope_parameters = getattr(hf_text_config, "rope_parameters", None)
    if rope_parameters is None:
        rope_parameters = getattr(hf_text_config, "rope_scaling", None)

    state: list[tuple[str, object]] = []
    state.extend(
        _snapshot_fields(
            "model_config",
            model_config,
            (
                "model",
                "model_weights",
                "tokenizer",
                "served_model_name",
                "revision",
                "tokenizer_revision",
                "code_revision",
                "hf_config_path",
                "dtype",
                "enforce_eager",
                "max_model_len",
                "architectures",
                "runner",
                "runner_type",
                "model_impl",
                "override_attention_dtype",
                "seed",
                "quantization",
                "quantization_config",
                "enable_prompt_embeds",
                "trust_remote_code",
                "hf_overrides",
                "skip_tokenizer_init",
                "model_class_overrides",
                "disable_cascade_attn",
            ),
        )
    )
    state.extend(
        (
            f"model_config.{name}",
            _freeze_qualification_value(_resolved_model_flag(model_config, name)),
        )
        for name in (
            "is_encoder_decoder",
            "is_diffusion",
            "uses_alibi",
            "uses_mrope",
            "uses_xdrope_dim",
            "is_multimodal_model",
            "use_mla",
            "is_hybrid",
            "get_sliding_window",
        )
    )
    state.append(
        ("model_config.rope_parameters", _freeze_qualification_value(rope_parameters))
    )
    state.extend(
        (
            (
                "model_config.hf_config",
                _snapshot_resolved_hf_config(getattr(model_config, "hf_config", None)),
            ),
            (
                "model_config.hf_text_config",
                _snapshot_resolved_hf_config(hf_text_config),
            ),
            (
                "model_config.hf_execution_state",
                _serial_pic_v1_hf_execution_state(hf_text_config),
            ),
            (
                "model_config.model_arch_config",
                _freeze_qualification_value(
                    getattr(model_config, "model_arch_config", None)
                ),
            ),
        )
    )
    state.extend(
        _snapshot_fields(
            "cache_config",
            cache_config,
            (
                "cache_dtype",
                "block_size",
                "gpu_memory_utilization",
                "enable_prefix_caching",
                "sliding_window",
                "mamba_cache_mode",
                "calculate_kv_scales",
                "kv_cache_dtype_skip_layers",
                "kv_offloading_size",
                "kv_cache_memory_bytes",
                "kv_sharing_fast_prefill",
                "use_replayssm",
                "is_attention_free",
            ),
        )
    )
    state.extend(
        _snapshot_fields(
            "parallel_config",
            parallel_config,
            (
                "tensor_parallel_size",
                "pipeline_parallel_size",
                "data_parallel_size",
                "decode_context_parallel_size",
                "prefill_context_parallel_size",
                "data_parallel_size_local",
                "data_parallel_rank",
                "data_parallel_rank_local",
                "distributed_executor_backend",
                "world_size",
                "rank",
                "nnodes",
                "node_rank",
                "data_parallel_external_lb",
                "data_parallel_hybrid_lb",
                "enable_expert_parallel",
                "enable_elastic_ep",
                "enable_fault_tolerance",
                "disable_custom_all_reduce",
                "enable_dbo",
                "ubatch_size",
                "worker_cls",
                "worker_extension_cls",
            ),
        )
    )
    state.extend(
        _snapshot_fields(
            "scheduler_config",
            scheduler_config,
            (
                "async_scheduling",
                "enable_chunked_prefill",
                "disable_hybrid_kv_cache_manager",
                "scheduler_cls",
            ),
        )
    )
    state.append(
        (
            "scheduler_config.resolved_scheduler_cls",
            _freeze_qualification_value(_resolved_scheduler_class(scheduler_config)),
        )
    )
    state.extend(_snapshot_fields("device_config", device_config, ("device_type",)))
    state.extend(
        _snapshot_fields(
            "load_config", load_config, ("load_format", "model_loader_extra_config")
        )
    )
    state.append(
        (
            "offload_config.complete_state",
            _freeze_qualification_value(offload_config),
        )
    )
    offload_hash = getattr(offload_config, "compute_hash", None)
    state.append(
        (
            "offload_config.resolved_hash",
            _freeze_qualification_value(
                offload_hash() if callable(offload_hash) else None
            ),
        )
    )
    state.extend(
        _snapshot_fields(
            "attention_config",
            attention_config,
            (
                "backend",
                "backend_per_kind",
                "flash_attn_version",
                "use_prefill_decode_attention",
                "flash_attn_max_num_splits_for_cuda_graph",
                "tq_max_kv_splits_for_cuda_graph",
                "use_trtllm_attention",
                "disable_flashinfer_q_quantization",
                "mla_prefill_backend",
                "use_prefill_query_quantization",
                "use_fp4_indexer_cache",
                "indexer_kv_dtype",
                "use_non_causal",
                "sparse_mla_force_mqa",
                "flex_attn_block_m",
                "flex_attn_block_n",
                "flex_attn_q_block_size",
                "flex_attn_kv_block_size",
            ),
        )
    )
    state.extend(_snapshot_fields("kernel_config", kernel_config, ("linear_backend",)))
    state.extend(
        _snapshot_fields(
            "compilation_config", compilation_config, ("mode", "cudagraph_mode")
        )
    )
    state.extend(
        _snapshot_fields(
            "structured_outputs_config",
            structured,
            (
                "backend",
                "disable_any_whitespace",
                "disable_additional_properties",
                "reasoning_parser",
                "reasoning_parser_plugin",
                "enable_in_reasoning",
            ),
        )
    )
    state.extend(
        (
            (
                "vllm_config.use_v2_model_runner",
                _freeze_qualification_value(
                    _resolved_model_flag(vllm_config, "use_v2_model_runner")
                ),
            ),
            (
                "vllm_config.additional_config",
                _freeze_qualification_value(
                    getattr(vllm_config, "additional_config", None)
                ),
            ),
        )
    )
    for field in (
        "speculative_config",
        "lora_config",
        "diffusion_config",
        "quant_config",
        "kv_events_config",
        "ec_transfer_config",
        "weight_transfer_config",
        "reasoning_config",
    ):
        state.append(
            (
                f"vllm_config.{field}",
                _freeze_qualification_value(getattr(vllm_config, field, None)),
            )
        )
    return tuple(state)


def _target_profile_shared_config_ids(
    vllm_config: VllmConfig,
) -> tuple[tuple[str, int], ...]:
    return tuple(
        (field, id(getattr(vllm_config, field, None)))
        for field in _KMESH_TARGET_PROFILE_SHARED_CONFIG_FIELDS
    )


def _qualify_kmesh_target_profile_child(
    vllm_config: VllmConfig,
    target_profile: str,
    shared_vllm_config: VllmConfig | None = None,
) -> None:
    """Attach provisional authority and immutable shared-config evidence.

    Only ``MultiConnector`` calls this, after validating the complete outer
    profile and before constructing either exact built-in profile child. The
    KMesh child remains non-actionable until ``MultiConnector`` validates every
    child again and explicitly arms it.
    """

    _target_profile_mode(target_profile)
    source = shared_vllm_config or vllm_config
    child_kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    setattr(
        vllm_config,
        _KMESH_TARGET_PROFILE_QUALIFICATION_ATTR,
        _KMeshTargetProfileQualification(
            authority=_KMESH_TARGET_PROFILE_AUTHORITY,
            target_profile=target_profile,
            shared_config_ids=_target_profile_shared_config_ids(source),
            shared_state=_serial_pic_v1_shared_state(source),
            child_kv_transfer_config_id=id(child_kv_transfer_config),
            child_kv_transfer_state=_freeze_qualification_value(
                child_kv_transfer_config
            ),
        ),
    )


def _clear_kmesh_target_profile_qualification(vllm_config: VllmConfig) -> None:
    setattr(vllm_config, _KMESH_TARGET_PROFILE_QUALIFICATION_ATTR, None)


def _is_kmesh_target_profile_child_qualified(
    vllm_config: VllmConfig, target_profile: str
) -> bool:
    qualification = getattr(vllm_config, _KMESH_TARGET_PROFILE_QUALIFICATION_ATTR, None)
    return (
        type(qualification) is _KMeshTargetProfileQualification
        and qualification.authority is _KMESH_TARGET_PROFILE_AUTHORITY
        and qualification.target_profile == target_profile
    )


def _revalidate_kmesh_target_profile_child_qualification(
    vllm_config: VllmConfig,
) -> str:
    qualification = getattr(vllm_config, _KMESH_TARGET_PROFILE_QUALIFICATION_ATTR, None)
    if (
        type(qualification) is not _KMeshTargetProfileQualification
        or qualification.authority is not _KMESH_TARGET_PROFILE_AUTHORITY
    ):
        raise ValueError(
            "KMesh target profile requires an internally qualified outer MultiConnector"
        )
    profile = qualification.target_profile
    _target_profile_mode(profile)
    current_ids = _target_profile_shared_config_ids(vllm_config)
    if current_ids != qualification.shared_config_ids:
        raise ValueError(
            f"KMesh {profile} engine-global configuration ownership drifted after "
            "qualification"
        )
    current_state = _serial_pic_v1_shared_state(vllm_config)
    if current_state != qualification.shared_state:
        raise ValueError(
            f"KMesh {profile} engine-global configuration drifted after qualification"
        )
    child_kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    if id(child_kv_transfer_config) != qualification.child_kv_transfer_config_id:
        raise ValueError(
            f"KMesh {profile} child KV-transfer configuration ownership drifted "
            "after qualification"
        )
    if (
        _freeze_qualification_value(child_kv_transfer_config)
        != qualification.child_kv_transfer_state
    ):
        raise ValueError(
            f"KMesh {profile} child KV-transfer configuration drifted after "
            "qualification"
        )
    return profile


def _validate_kmesh_target_profile_child_runtime(vllm_config: VllmConfig) -> str:
    """Revalidate the qualified KMesh child without trusting outer aliases."""

    profile = _revalidate_kmesh_target_profile_child_qualification(vllm_config)
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    for field, expected, requirement in (
        ("kv_connector", "KMeshConnector", "KMesh child connector identity"),
        ("kv_connector_module_path", None, "KMesh child module path unset"),
        ("kv_buffer_device", "cuda", "KMesh child kv_buffer_device='cuda'"),
        ("kv_buffer_size", 1e9, "KMesh child kv_buffer_size=1e9"),
        ("kv_role", "kv_both", "KMesh child kv_role='kv_both'"),
        ("kv_parallel_size", 1, "KMesh child kv_parallel_size=1"),
        ("kv_rank", None, "KMesh child kv_rank unset"),
        ("kv_ip", "127.0.0.1", "KMesh child kv_ip='127.0.0.1'"),
        ("kv_port", 14579, "KMesh child kv_port=14579"),
        (
            "enable_permute_local_kv",
            False,
            "KMesh child enable_permute_local_kv=false",
        ),
        (
            "kv_load_failure_policy",
            "recompute",
            "KMesh child kv_load_failure_policy='recompute'",
        ),
    ):
        _require_target_profile(
            profile, getattr(kv_transfer_config, field, None), expected, requirement
        )
    engine_id = getattr(kv_transfer_config, "engine_id", None)
    if type(engine_id) is not str or not engine_id or engine_id != engine_id.strip():
        raise _target_profile_error(
            profile, "a nonempty KMesh child engine_id", engine_id
        )
    extra_config = getattr(kv_transfer_config, "kv_connector_extra_config", None)
    if type(extra_config) is not dict:
        raise _target_profile_error(
            profile, "a KMesh child extra-config mapping", extra_config
        )
    _require_target_profile(
        profile,
        set(extra_config),
        {"kmesh", KMESH_TARGET_PROFILE_KEY},
        "only kmesh and the propagated profile in KMesh child extra config",
    )
    _require_target_profile(
        profile,
        extra_config.get(KMESH_TARGET_PROFILE_KEY),
        profile,
        f"the propagated {profile} KMesh child profile",
    )
    _validate_target_profile_kmesh_config(
        KMeshEngineConfig.from_connector_extra_config(extra_config), profile
    )
    return profile


def validate_lmcache_target_profile_child_runtime(vllm_config: VllmConfig) -> str:
    """Revalidate the exact external LMCache child and frozen evidence."""

    profile = _revalidate_kmesh_target_profile_child_qualification(vllm_config)
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    for field, expected, requirement in (
        ("kv_connector", "LMCacheConnectorV1", "LMCache child connector identity"),
        ("kv_connector_module_path", None, "LMCache child module path unset"),
        ("kv_buffer_device", "cuda", "LMCache child kv_buffer_device='cuda'"),
        ("kv_buffer_size", 1e9, "LMCache child kv_buffer_size=1e9"),
        ("kv_role", "kv_both", "LMCache child kv_role='kv_both'"),
        ("kv_parallel_size", 1, "LMCache child kv_parallel_size=1"),
        ("kv_rank", None, "LMCache child kv_rank unset"),
        ("kv_ip", "127.0.0.1", "LMCache child kv_ip='127.0.0.1'"),
        ("kv_port", 14579, "LMCache child kv_port=14579"),
        (
            "enable_permute_local_kv",
            False,
            "LMCache child enable_permute_local_kv=false",
        ),
        (
            "kv_load_failure_policy",
            "recompute",
            "LMCache child kv_load_failure_policy='recompute'",
        ),
    ):
        _require_target_profile(
            profile, getattr(kv_transfer_config, field, None), expected, requirement
        )
    engine_id = getattr(kv_transfer_config, "engine_id", None)
    if type(engine_id) is not str or not engine_id or engine_id != engine_id.strip():
        raise _target_profile_error(
            profile, "a nonempty LMCache child engine_id", engine_id
        )
    extra_config = getattr(kv_transfer_config, "kv_connector_extra_config", None)
    if type(extra_config) is not dict:
        raise _target_profile_error(
            profile, "an LMCache child extra-config mapping", extra_config
        )
    allowed_keys = {"use_native", KMESH_TARGET_PROFILE_KEY, "lmcache_rpc_port"}
    unexpected_keys = sorted(
        repr(key)
        for key in extra_config
        if type(key) is not str or key not in allowed_keys
    )
    if unexpected_keys:
        raise _target_profile_error(
            profile,
            "only use_native, the propagated profile, and optional "
            "lmcache_rpc_port in LMCache child extra config",
            unexpected_keys,
        )
    _require_target_profile(
        profile, extra_config.get("use_native"), False, "LMCache child use_native=false"
    )
    _require_target_profile(
        profile,
        extra_config.get(KMESH_TARGET_PROFILE_KEY),
        profile,
        f"the propagated {profile} LMCache child profile",
    )
    if "lmcache_rpc_port" in extra_config:
        lmcache_rpc_port = extra_config["lmcache_rpc_port"]
        if (
            type(lmcache_rpc_port) is not int
            or lmcache_rpc_port < 0
            or lmcache_rpc_port > 65535
        ):
            raise _target_profile_error(
                profile,
                "LMCache child lmcache_rpc_port from 0 through 65535",
                lmcache_rpc_port,
            )
    return profile


def _validate_target_profile_kmesh_config(
    config: KMeshEngineConfig, profile: str
) -> None:
    expected_mode = _target_profile_mode(profile)
    exact = (
        ("enabled", config.enabled, True, "kmesh.enabled=true"),
        (
            "mode",
            config.mode,
            expected_mode,
            f"kmesh.mode={expected_mode.value!r}",
        ),
        (
            "segment_encoding",
            config.segment_encoding,
            "sideband",
            "kmesh.segment_encoding='sideband'",
        ),
        (
            "pic_backend",
            config.pic_backend,
            "lmcache",
            "kmesh.pic_backend='lmcache'",
        ),
        ("check_layer", config.check_layer, 1, "kmesh.check_layer=1"),
        (
            "recompute_ratio",
            config.recompute_ratio,
            0.15,
            "kmesh.recompute_ratio=0.15",
        ),
        ("max_group_size", config.max_group_size, 10, "kmesh.max_group_size=10"),
        (
            "max_group_wait_ms",
            config.max_group_wait_ms,
            0,
            "kmesh.max_group_wait_ms=0",
        ),
        (
            "require_same_active_length",
            config.require_same_active_length,
            True,
            "kmesh.require_same_active_length=true",
        ),
        ("fail_open", config.fail_open, False, "kmesh.fail_open=false"),
        (
            "strict_validation",
            config.strict_validation,
            True,
            "kmesh.strict_validation=true",
        ),
        (
            "master_selection",
            config.master_selection,
            "min_total_changed_blocks",
            "kmesh.master_selection='min_total_changed_blocks'",
        ),
        (
            "storage_mode",
            config.storage_mode,
            "none",
            "kmesh.storage_mode='none'",
        ),
    )
    for _field, actual, expected, requirement in exact:
        _require_target_profile(profile, actual, expected, requirement)


def validate_kmesh_target_profile(vllm_config: VllmConfig) -> str | None:
    """Fail closed on either exact KMesh research-runtime profile.

    The validator is deliberately dormant unless the outer ``MultiConnector``
    extra config names a supported profile. ``collective_lockstep_v1`` reuses
    the complete frozen ``serial_pic_v1`` runtime envelope and changes only the
    KMesh execution mode. Both profiles admit either automatic attention
    selection or the exact ``TRITON_ATTN`` backend needed for a same-kernel
    numerical oracle; the resolved choice remains part of the frozen shared
    state. The external LMCache adapter independently validates its resolved
    storage and lookup configuration.
    """

    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    extra_config = getattr(kv_transfer_config, "kv_connector_extra_config", None)
    if not isinstance(extra_config, dict):
        return None
    profile = extra_config.get(KMESH_TARGET_PROFILE_KEY)
    if profile is None:
        return None
    _target_profile_mode(profile)

    def require(actual: object, expected: object, requirement: str) -> None:
        _require_target_profile(profile, actual, expected, requirement)

    def profile_error(requirement: str, actual: object) -> ValueError:
        return _target_profile_error(profile, requirement, actual)

    require(
        set(extra_config),
        {KMESH_TARGET_PROFILE_KEY, "connectors"},
        "only kmesh_target_profile and connectors in the outer extra config",
    )

    require(
        getattr(kv_transfer_config, "kv_connector", None),
        "MultiConnector",
        "outer kv_connector='MultiConnector'",
    )
    outer_engine_id = getattr(kv_transfer_config, "engine_id", None)
    if (
        not isinstance(outer_engine_id, str)
        or not outer_engine_id
        or outer_engine_id != outer_engine_id.strip()
    ):
        raise profile_error("a nonempty outer engine_id", outer_engine_id)
    for field, expected, requirement in (
        ("kv_connector_module_path", None, "outer kv_connector_module_path unset"),
        ("kv_buffer_device", "cuda", "outer kv_buffer_device='cuda'"),
        ("kv_buffer_size", 1e9, "outer kv_buffer_size=1e9"),
        ("kv_role", "kv_both", "outer kv_role='kv_both'"),
        ("kv_parallel_size", 1, "outer kv_parallel_size=1"),
        ("kv_rank", None, "outer kv_rank unset"),
        ("kv_ip", "127.0.0.1", "outer kv_ip='127.0.0.1'"),
        ("kv_port", 14579, "outer kv_port=14579"),
        (
            "enable_permute_local_kv",
            False,
            "outer enable_permute_local_kv=false",
        ),
    ):
        require(getattr(kv_transfer_config, field, None), expected, requirement)
    connectors = extra_config.get("connectors")
    if not isinstance(connectors, list) or len(connectors) != 2:
        raise profile_error("exactly two connector mappings", connectors)
    if any(
        not isinstance(connector, dict)
        or any(not isinstance(key, str) for key in connector)
        for connector in connectors
    ):
        raise profile_error("string-keyed connector mappings", connectors)
    connector_names = tuple(connector.get("kv_connector") for connector in connectors)
    require(
        connector_names,
        ("KMeshConnector", "LMCacheConnectorV1"),
        "connector order [KMeshConnector, LMCacheConnectorV1]",
    )

    for index, connector in enumerate(connectors):
        admitted_child_fields = {
            "kv_connector",
            "engine_id",
            "kv_buffer_device",
            "kv_buffer_size",
            "kv_role",
            "kv_rank",
            "kv_parallel_size",
            "kv_ip",
            "kv_port",
            "kv_connector_extra_config",
            "kv_connector_module_path",
            "enable_permute_local_kv",
            "kv_load_failure_policy",
        }
        unexpected_child_fields = sorted(
            repr(key)
            for key in connector
            if type(key) is not str or key not in admitted_child_fields
        )
        if unexpected_child_fields:
            raise profile_error(
                f"no unreviewed child {index} KV-transfer fields",
                unexpected_child_fields,
            )
        child_extra = connector.get("kv_connector_extra_config", {})
        if not isinstance(child_extra, dict):
            raise profile_error("child kv_connector_extra_config mappings", child_extra)
        child_profile = child_extra.get(KMESH_TARGET_PROFILE_KEY)
        if child_profile is not None:
            raise profile_error(
                "target profile declared only on the outer MultiConnector",
                child_profile,
            )
        child_policy = connector.get("kv_load_failure_policy")
        if child_policy not in (None, "recompute"):
            raise profile_error(
                "no conflicting child KV load failure policy", child_policy
            )
        child_engine_id = connector.get("engine_id", outer_engine_id)
        require(
            child_engine_id,
            outer_engine_id,
            f"child {index} engine_id inherited from the outer connector",
        )
        for field, expected, requirement in (
            (
                "kv_connector_module_path",
                None,
                f"child {index} kv_connector_module_path unset",
            ),
            ("kv_buffer_device", "cuda", f"child {index} kv_buffer_device='cuda'"),
            ("kv_buffer_size", 1e9, f"child {index} kv_buffer_size=1e9"),
            ("kv_role", "kv_both", f"child {index} kv_role='kv_both'"),
            ("kv_parallel_size", 1, f"child {index} kv_parallel_size=1"),
            ("kv_rank", None, f"child {index} kv_rank unset"),
            ("kv_ip", "127.0.0.1", f"child {index} kv_ip='127.0.0.1'"),
            ("kv_port", 14579, f"child {index} kv_port=14579"),
            (
                "enable_permute_local_kv",
                False,
                f"child {index} enable_permute_local_kv=false",
            ),
        ):
            require(connector.get(field, expected), expected, requirement)

    kmesh_extra = connectors[0].get("kv_connector_extra_config", {})
    assert isinstance(kmesh_extra, dict)
    require(set(kmesh_extra), {"kmesh"}, "only kmesh in the KMesh child extra config")
    kmesh_config = KMeshEngineConfig.from_connector_extra_config(kmesh_extra)
    _validate_target_profile_kmesh_config(kmesh_config, profile)

    lmcache_extra = connectors[1].get("kv_connector_extra_config", {})
    assert isinstance(lmcache_extra, dict)
    unexpected_lmcache_extra = sorted(
        repr(key)
        for key in lmcache_extra
        if type(key) is not str or key not in {"use_native", "lmcache_rpc_port"}
    )
    if unexpected_lmcache_extra:
        raise profile_error(
            "only use_native and optional lmcache_rpc_port in the LMCache child "
            "extra config",
            unexpected_lmcache_extra,
        )
    require(lmcache_extra.get("use_native"), False, "LMCache use_native=false")
    if "lmcache_rpc_port" in lmcache_extra:
        lmcache_rpc_port = lmcache_extra["lmcache_rpc_port"]
        if (
            type(lmcache_rpc_port) is not int
            or lmcache_rpc_port < 0
            or lmcache_rpc_port > 65535
        ):
            raise profile_error(
                "LMCache lmcache_rpc_port from 0 through 65535", lmcache_rpc_port
            )
    require(
        getattr(kv_transfer_config, "kv_load_failure_policy", None),
        "recompute",
        "kv_load_failure_policy='recompute'",
    )

    parallel_config = getattr(vllm_config, "parallel_config", None)
    for field, label in (
        ("tensor_parallel_size", "TP"),
        ("pipeline_parallel_size", "PP"),
        ("data_parallel_size", "DP"),
        ("decode_context_parallel_size", "DCP"),
        ("prefill_context_parallel_size", "PCP"),
    ):
        require(getattr(parallel_config, field, None), 1, f"{label}=1")
    for field, expected, requirement in (
        ("data_parallel_size_local", 1, "local DP size=1"),
        ("data_parallel_rank", 0, "DP rank=0"),
        ("data_parallel_rank_local", 0, "local DP rank=0"),
        ("world_size", 1, "world_size=1"),
        ("rank", 0, "global rank=0"),
        ("nnodes", 1, "nnodes=1"),
        ("node_rank", 0, "node_rank=0"),
        ("data_parallel_external_lb", False, "external DP load balancing disabled"),
        ("data_parallel_hybrid_lb", False, "hybrid DP load balancing disabled"),
        ("enable_expert_parallel", False, "expert parallelism disabled"),
        ("enable_elastic_ep", False, "elastic expert parallelism disabled"),
        ("enable_fault_tolerance", False, "fault tolerance disabled"),
        ("disable_custom_all_reduce", True, "custom all-reduce disabled"),
    ):
        require(getattr(parallel_config, field, None), expected, requirement)
    require(
        getattr(parallel_config, "distributed_executor_backend", None),
        "uni",
        "distributed_executor_backend='uni'",
    )
    require(getattr(parallel_config, "enable_dbo", None), False, "DBO disabled")
    require(getattr(parallel_config, "ubatch_size", None), 0, "ubatching disabled")
    require(
        getattr(parallel_config, "worker_cls", None),
        "vllm.v1.worker.gpu_worker.Worker",
        "the resolved built-in GPU worker class",
    )
    require(
        getattr(parallel_config, "worker_extension_cls", None),
        "",
        "no worker extension class",
    )

    model_config = getattr(vllm_config, "model_config", None)
    for field, expected, requirement in (
        (
            "model",
            KMESH_SERIAL_PIC_V1_MODEL_PATH,
            f"model path {KMESH_SERIAL_PIC_V1_MODEL_PATH}",
        ),
        (
            "tokenizer",
            KMESH_SERIAL_PIC_V1_MODEL_PATH,
            f"tokenizer path {KMESH_SERIAL_PIC_V1_MODEL_PATH}",
        ),
        (
            "served_model_name",
            KMESH_SERIAL_PIC_V1_MODEL_PATH,
            f"served model identity {KMESH_SERIAL_PIC_V1_MODEL_PATH}",
        ),
        ("model_weights", "", "model_weights unset"),
        ("model_impl", "auto", "model_impl='auto'"),
        ("override_attention_dtype", None, "attention dtype override unset"),
        ("seed", 20260731, "model seed=20260731"),
        ("runner", "auto", "model runner selection='auto'"),
        ("hf_config_path", None, "HF config path override unset"),
        ("code_revision", None, "model code revision override unset"),
    ):
        require(getattr(model_config, field, None), expected, requirement)
    require(getattr(model_config, "dtype", None), torch.bfloat16, "BF16 model dtype")
    require(getattr(model_config, "enforce_eager", None), True, "eager execution")
    require(getattr(model_config, "max_model_len", None), 2048, "max_model_len=2048")
    require(
        tuple(getattr(model_config, "architectures", ()) or ()),
        ("Qwen2ForCausalLM",),
        "the Qwen2ForCausalLM architecture",
    )
    require(
        getattr(model_config, "revision", None),
        KMESH_SERIAL_PIC_V1_MODEL_REVISION,
        f"model revision {KMESH_SERIAL_PIC_V1_MODEL_REVISION}",
    )
    require(
        getattr(model_config, "tokenizer_revision", None),
        KMESH_SERIAL_PIC_V1_MODEL_REVISION,
        f"tokenizer revision {KMESH_SERIAL_PIC_V1_MODEL_REVISION}",
    )
    model_requirements: tuple[tuple[str, object, str], ...] = (
        ("runner_type", "generate", "the generate runner"),
        ("quantization", None, "model quantization disabled"),
        ("quantization_config", None, "model quantization configuration unset"),
        ("enable_prompt_embeds", False, "prompt embeddings disabled globally"),
        ("trust_remote_code", False, "trust_remote_code=false"),
        ("hf_overrides", {}, "no Hugging Face config overrides"),
        ("skip_tokenizer_init", False, "tokenizer initialization enabled"),
        ("model_class_overrides", {}, "no model class overrides"),
    )
    for field, model_expected, requirement in model_requirements:
        require(getattr(model_config, field, None), model_expected, requirement)
    for field, expected, requirement in (
        ("is_encoder_decoder", False, "a decoder-only model"),
        ("is_diffusion", False, "a non-diffusion model"),
        ("uses_alibi", False, "ALiBi disabled"),
        ("uses_mrope", False, "M-RoPE disabled"),
        ("uses_xdrope_dim", 0, "XDRoPE disabled"),
    ):
        require(_resolved_model_flag(model_config, field), expected, requirement)
    require(
        _resolved_model_flag(model_config, "is_multimodal_model"),
        False,
        "a text-only model",
    )
    require(_resolved_model_flag(model_config, "use_mla"), False, "MLA disabled")
    require(
        _resolved_model_flag(model_config, "is_hybrid"),
        False,
        "hybrid/Mamba models disabled",
    )
    require(
        _resolved_model_flag(model_config, "get_sliding_window"),
        None,
        "sliding-window attention disabled",
    )
    require(
        getattr(model_config, "disable_cascade_attn", None),
        True,
        "cascade attention disabled",
    )
    hf_text_config = getattr(model_config, "hf_text_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    from transformers import Qwen2Config

    require(
        type(getattr(model_config, "hf_overrides", None)),
        dict,
        "a plain Hugging Face override mapping",
    )
    require(type(hf_config), Qwen2Config, "the resolved Qwen2 Hugging Face config")
    hf_config = cast(Qwen2Config, hf_config)
    require(
        hf_text_config is hf_config,
        True,
        "one resolved text-only Hugging Face config object",
    )
    require(
        hf_config.get_text_config() is hf_config,
        True,
        "Qwen2 get_text_config() resolving to that same config object",
    )
    require(
        _serial_pic_v1_resolved_head_dim(hf_text_config),
        64,
        "resolved Qwen2 head_dim=64 for model and KV-cache geometry",
    )
    require(
        _serial_pic_v1_resolved_num_kv_heads(hf_text_config),
        2,
        "resolved Qwen2 KV-head count=2 for model and KV-cache geometry",
    )
    rope_parameters = getattr(hf_text_config, "rope_parameters", None)
    if rope_parameters is None:
        rope_parameters = getattr(hf_text_config, "rope_scaling", None)
    require(
        rope_parameters,
        {"rope_theta": 1_000_000.0, "rope_type": "default"},
        "the frozen Qwen2.5 default RoPE parameters",
    )
    require(
        _serial_pic_v1_hf_execution_state(hf_text_config),
        _KMESH_SERIAL_PIC_V1_HF_EXECUTION_STATE,
        "the complete frozen Qwen2.5-0.5B resolved computation identity",
    )
    expected_model_arch_config = _serial_pic_v1_expected_model_arch_config()
    model_arch_config = getattr(model_config, "model_arch_config", None)
    require(
        type(model_arch_config),
        type(expected_model_arch_config),
        "the built-in resolved ModelArchitectureConfig",
    )
    require(
        _freeze_qualification_value(model_arch_config),
        _freeze_qualification_value(expected_model_arch_config),
        "the exact resolved Qwen2 model/cache architecture geometry",
    )

    cache_config = getattr(vllm_config, "cache_config", None)
    require(
        getattr(cache_config, "gpu_memory_utilization", None),
        0.3,
        "gpu_memory_utilization=0.3",
    )
    require(
        getattr(cache_config, "cache_dtype", None),
        "auto",
        "cache_dtype='auto' (effective BF16)",
    )
    require(getattr(cache_config, "block_size", None), 32, "KV block_size=32")
    require(
        getattr(cache_config, "enable_prefix_caching", None),
        False,
        "automatic prefix caching disabled",
    )
    require(
        getattr(cache_config, "sliding_window", None),
        None,
        "sliding-window KV cache disabled",
    )
    require(
        getattr(cache_config, "mamba_cache_mode", None),
        "none",
        "Mamba cache disabled",
    )
    cache_requirements: tuple[tuple[str, object, str], ...] = (
        ("calculate_kv_scales", False, "dynamic KV scale calculation disabled"),
        ("kv_cache_dtype_skip_layers", [], "no KV dtype skip layers"),
        ("kv_offloading_size", None, "native KV offloading disabled"),
        ("kv_cache_memory_bytes", None, "manual KV cache byte override unset"),
        ("kv_sharing_fast_prefill", False, "KV-sharing fast prefill disabled"),
        ("use_replayssm", False, "ReplaySSM disabled"),
        ("is_attention_free", False, "an attention model"),
    )
    for field, cache_expected, requirement in cache_requirements:
        require(getattr(cache_config, field, None), cache_expected, requirement)

    device_config = getattr(vllm_config, "device_config", None)
    require(
        getattr(device_config, "device_type", None),
        "cuda",
        "the pinned ROCm CUDA-compatible device path",
    )

    attention_config = getattr(vllm_config, "attention_config", None)
    attention_backend = getattr(attention_config, "backend", None)
    if attention_backend not in (None, AttentionBackendEnum.TRITON_ATTN):
        raise profile_error(
            "the automatic backend or exact TRITON_ATTN backend",
            attention_backend,
        )
    require(
        getattr(attention_config, "backend_per_kind", None),
        {},
        "no per-kind attention backend overrides",
    )
    require(
        getattr(attention_config, "use_prefill_decode_attention", None),
        False,
        "unified prefill/decode attention",
    )
    attention_requirements: tuple[tuple[str, object, str], ...] = (
        ("flash_attn_version", None, "no forced FlashAttention version"),
        (
            "flash_attn_max_num_splits_for_cuda_graph",
            32,
            "FlashAttention CUDA-graph split default=32",
        ),
        (
            "tq_max_kv_splits_for_cuda_graph",
            32,
            "TurboQuant CUDA-graph split default=32",
        ),
        ("use_trtllm_attention", None, "TRT-LLM attention override unset"),
        (
            "disable_flashinfer_q_quantization",
            False,
            "FlashInfer query quantization override disabled",
        ),
        ("mla_prefill_backend", None, "MLA prefill backend override unset"),
        (
            "use_prefill_query_quantization",
            False,
            "prefill query quantization disabled",
        ),
        ("use_fp4_indexer_cache", False, "FP4 indexer cache disabled"),
        ("indexer_kv_dtype", "bf16", "indexer KV dtype='bf16'"),
        ("use_non_causal", False, "non-causal attention disabled"),
        ("sparse_mla_force_mqa", False, "sparse MLA force-MQA disabled"),
        ("flex_attn_block_m", None, "FlexAttention BLOCK_M override unset"),
        ("flex_attn_block_n", None, "FlexAttention BLOCK_N override unset"),
        (
            "flex_attn_q_block_size",
            None,
            "FlexAttention query block-size override unset",
        ),
        (
            "flex_attn_kv_block_size",
            None,
            "FlexAttention KV block-size override unset",
        ),
    )
    for (
        attention_field,
        attention_expected,
        attention_requirement,
    ) in attention_requirements:
        require(
            getattr(attention_config, attention_field, None),
            attention_expected,
            attention_requirement,
        )

    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    require(
        getattr(scheduler_config, "async_scheduling", None),
        False,
        "async scheduling disabled",
    )
    require(
        getattr(scheduler_config, "enable_chunked_prefill", None),
        False,
        "chunked prefill disabled",
    )
    require(
        getattr(scheduler_config, "disable_hybrid_kv_cache_manager", None),
        True,
        "hybrid KV cache manager disabled",
    )
    require(
        getattr(scheduler_config, "scheduler_cls", None),
        None,
        "scheduler_cls unset",
    )
    from vllm.v1.core.sched.scheduler import Scheduler

    require(
        _resolved_scheduler_class(scheduler_config),
        Scheduler,
        "the built-in vLLM Scheduler",
    )

    load_config = getattr(vllm_config, "load_config", None)
    require(getattr(load_config, "load_format", None), "auto", "load_format='auto'")
    require(
        getattr(load_config, "model_loader_extra_config", None),
        {},
        "empty model_loader_extra_config",
    )

    offload_config = getattr(vllm_config, "offload_config", None)
    from vllm.config.offload import (
        OffloadConfig,
        PrefetchOffloadConfig,
        UVAOffloadConfig,
    )

    require(type(offload_config), OffloadConfig, "the built-in OffloadConfig")
    require(
        getattr(offload_config, "offload_backend", None),
        "auto",
        "automatic model offloading with every backend disabled",
    )
    uva_offload = getattr(offload_config, "uva", None)
    require(type(uva_offload), UVAOffloadConfig, "the built-in UVA offload config")
    cpu_offload_gb = getattr(uva_offload, "cpu_offload_gb", None)
    if type(cpu_offload_gb) not in (int, float) or cpu_offload_gb != 0:
        raise profile_error("UVA model offloading disabled", cpu_offload_gb)
    require(
        type(getattr(uva_offload, "cpu_offload_params", None)),
        set,
        "a plain UVA offload parameter set",
    )
    require(
        getattr(uva_offload, "cpu_offload_params", None),
        set(),
        "no UVA-offloaded parameters",
    )
    prefetch_offload = getattr(offload_config, "prefetch", None)
    require(
        type(prefetch_offload),
        PrefetchOffloadConfig,
        "the built-in prefetch offload config",
    )
    for field, expected, requirement in (
        ("offload_group_size", 0, "prefetch model offloading disabled"),
        ("offload_num_in_group", 1, "default prefetch offload group count"),
        ("offload_prefetch_step", 1, "default prefetch lookahead"),
    ):
        require(getattr(prefetch_offload, field, None), expected, requirement)
    require(
        type(getattr(prefetch_offload, "offload_params", None)),
        set,
        "a plain prefetch offload parameter set",
    )
    require(
        getattr(prefetch_offload, "offload_params", None),
        set(),
        "no prefetch-offloaded parameters",
    )

    kernel_config = getattr(vllm_config, "kernel_config", None)
    require(
        getattr(kernel_config, "linear_backend", None),
        "auto",
        "kernel linear_backend='auto'",
    )
    compilation_config = getattr(vllm_config, "compilation_config", None)
    require(
        _enum_value(getattr(compilation_config, "mode", None)),
        0,
        "compilation mode NONE",
    )
    require(
        _enum_value(getattr(compilation_config, "cudagraph_mode", None)),
        0,
        "CUDA graph mode NONE",
    )

    require(
        getattr(vllm_config, "speculative_config", None),
        None,
        "speculative decoding disabled",
    )
    require(getattr(vllm_config, "lora_config", None), None, "LoRA disabled")
    for field, requirement in (
        ("diffusion_config", "diffusion runtime disabled"),
        ("quant_config", "resolved quantization disabled"),
        ("kv_events_config", "KV event publishing disabled"),
        ("ec_transfer_config", "encoder-cache transfer disabled"),
        ("weight_transfer_config", "weight transfer disabled"),
        ("reasoning_config", "reasoning parser runtime disabled"),
    ):
        require(getattr(vllm_config, field, None), None, requirement)

    require(
        _resolved_model_flag(vllm_config, "use_v2_model_runner"),
        True,
        "Model Runner V2 enabled",
    )
    require(
        getattr(vllm_config, "additional_config", None),
        {},
        "empty additional_config",
    )

    structured = getattr(vllm_config, "structured_outputs_config", None)
    structured_state = (
        getattr(structured, "backend", None),
        getattr(structured, "disable_any_whitespace", None),
        getattr(structured, "disable_additional_properties", None),
        getattr(structured, "reasoning_parser", None),
        getattr(structured, "reasoning_parser_plugin", None),
        getattr(structured, "enable_in_reasoning", None),
    )
    require(
        structured_state,
        ("auto", False, False, "", "", False),
        "default structured-output controls (structured requests remain forbidden)",
    )
    return profile


@dataclass(frozen=True, slots=True)
class KMeshEngineConfig:
    """Experimental engine controls carried in connector extra config.

    The public request schema remains in ``SamplingParams.extra_args``. These
    controls are deliberately connector-local so KMesh does not widen the
    OpenAI-compatible API or add a scheduler implementation.
    """

    enabled: bool = False
    mode: KMeshMode = KMeshMode.OFF
    segment_encoding: str = "sideband"
    pic_backend: str = "lmcache"
    check_layer: int = 1
    recompute_ratio: float = 0.15
    max_group_size: int = 10
    max_group_wait_ms: int = 0
    require_same_active_length: bool = True
    fail_open: bool = True
    strict_validation: bool = False
    master_selection: str = "min_total_changed_blocks"
    storage_mode: str = "none"

    def __post_init__(self) -> None:
        _strict_bool(self.enabled, "enabled")
        if not isinstance(self.mode, KMeshMode):
            raise TypeError("kmesh.mode must be a KMeshMode")
        _choice(
            self.segment_encoding,
            "segment_encoding",
            frozenset({"sideband", "ttsep"}),
        )
        _choice(self.pic_backend, "pic_backend", frozenset({"lmcache", "noop"}))
        _strict_int(self.check_layer, "check_layer")
        _bounded_ratio(self.recompute_ratio)
        _strict_int(self.max_group_size, "max_group_size", minimum=2)
        _strict_int(self.max_group_wait_ms, "max_group_wait_ms")
        _strict_bool(self.require_same_active_length, "require_same_active_length")
        _strict_bool(self.fail_open, "fail_open")
        _strict_bool(self.strict_validation, "strict_validation")
        _choice(
            self.master_selection,
            "master_selection",
            frozenset({"min_total_changed_blocks"}),
        )
        _choice(self.storage_mode, "storage_mode", frozenset({"none", "reference"}))
        if self.enabled == (self.mode is KMeshMode.OFF):
            raise ValueError(
                "kmesh.enabled=true requires a non-off mode, and "
                "kmesh.enabled=false requires mode=off"
            )
        if self.max_group_wait_ms != 0:
            raise ValueError("kmesh.max_group_wait_ms is reserved; v0 supports only 0")
        if not self.require_same_active_length:
            raise ValueError(
                "kmesh.require_same_active_length=false is outside "
                "the v0 safety boundary"
            )

    @classmethod
    def from_connector_extra_config(cls, extra_config: object) -> KMeshEngineConfig:
        if extra_config is None:
            return cls()
        if not isinstance(extra_config, dict):
            raise TypeError("kv_connector_extra_config must be a mapping")
        raw = extra_config.get("kmesh")
        if raw is None:
            return cls()
        if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
            raise TypeError(
                "kv_connector_extra_config.kmesh must be a string-keyed mapping"
            )
        if unknown := sorted(raw.keys() - _CONFIG_FIELDS):
            raise ValueError(f"unknown kmesh config field: {unknown[0]}")

        enabled = _strict_bool(raw.get("enabled", False), "enabled")
        default_mode = "observe" if enabled else "off"
        raw_mode = raw.get("mode", default_mode)
        if not isinstance(raw_mode, str):
            raise TypeError("kmesh.mode must be a string")
        try:
            mode = KMeshMode(raw_mode)
        except ValueError as exc:
            expected = ", ".join(mode.value for mode in KMeshMode)
            raise ValueError(f"kmesh.mode must be one of: {expected}") from exc

        return cls(
            enabled=enabled,
            mode=mode,
            segment_encoding=_choice(
                raw.get("segment_encoding", "sideband"),
                "segment_encoding",
                frozenset({"sideband", "ttsep"}),
            ),
            pic_backend=_choice(
                raw.get("pic_backend", "lmcache"),
                "pic_backend",
                frozenset({"lmcache", "noop"}),
            ),
            check_layer=_strict_int(raw.get("check_layer", 1), "check_layer"),
            recompute_ratio=_bounded_ratio(raw.get("recompute_ratio", 0.15)),
            max_group_size=_strict_int(
                raw.get("max_group_size", 10), "max_group_size", minimum=2
            ),
            max_group_wait_ms=_strict_int(
                raw.get("max_group_wait_ms", 0), "max_group_wait_ms"
            ),
            require_same_active_length=_strict_bool(
                raw.get("require_same_active_length", True),
                "require_same_active_length",
            ),
            fail_open=_strict_bool(raw.get("fail_open", True), "fail_open"),
            strict_validation=_strict_bool(
                raw.get("strict_validation", False), "strict_validation"
            ),
            master_selection=_choice(
                raw.get("master_selection", "min_total_changed_blocks"),
                "master_selection",
                frozenset({"min_total_changed_blocks"}),
            ),
            storage_mode=_choice(
                raw.get("storage_mode", "none"),
                "storage_mode",
                frozenset({"none", "reference"}),
            ),
        )

    def to_primitive(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode.value,
            "segment_encoding": self.segment_encoding,
            "pic_backend": self.pic_backend,
            "check_layer": self.check_layer,
            "recompute_ratio": self.recompute_ratio,
            "max_group_size": self.max_group_size,
            "max_group_wait_ms": self.max_group_wait_ms,
            "require_same_active_length": self.require_same_active_length,
            "fail_open": self.fail_open,
            "strict_validation": self.strict_validation,
            "master_selection": self.master_selection,
            "storage_mode": self.storage_mode,
        }
