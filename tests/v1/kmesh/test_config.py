# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import copy
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from transformers import Qwen2Config

import vllm.platforms as vllm_platforms
from vllm.config import (
    AttentionConfig,
    CacheConfig,
    DeviceConfig,
    KVTransferConfig,
    ModelConfig,
    OffloadConfig,
    ParallelConfig,
    SchedulerConfig,
    StructuredOutputsConfig,
    VllmConfig,
)
from vllm.config import vllm as vllm_config_module
from vllm.config.model_arch import ModelArchitectureConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.kmesh_connector import (
    KMeshConnector,
    KMeshConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
    MultiKVConnectorMetadata,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.kmesh import config as kmesh_config_module
from vllm.v1.kmesh.config import (
    KMESH_COLLECTIVE_BATCHED_POSITION_RECOVERY_V1_PROFILE,
    KMESH_COLLECTIVE_BATCHED_QKV_V1_PROFILE,
    KMESH_COLLECTIVE_LOCKSTEP_V1_PROFILE,
    KMESH_SERIAL_PIC_V1_MODEL_PATH,
    KMESH_SERIAL_PIC_V1_PROFILE,
    KMESH_TARGET_PROFILE_KEY,
    KMeshEngineConfig,
    KMeshMode,
    validate_kmesh_target_profile,
)

pytestmark = pytest.mark.skip_global_cleanup


def test_absent_config_is_disabled() -> None:
    config = KMeshEngineConfig.from_connector_extra_config({"use_native": True})

    assert config == KMeshEngineConfig()
    assert config.mode is KMeshMode.OFF


def test_enabled_config_defaults_to_observer() -> None:
    config = KMeshEngineConfig.from_connector_extra_config({"kmesh": {"enabled": True}})

    assert config.enabled
    assert config.mode is KMeshMode.OBSERVE
    assert config.max_group_wait_ms == 0


def test_full_observer_config_round_trips() -> None:
    raw = {
        "enabled": True,
        "mode": "observe",
        "segment_encoding": "ttsep",
        "pic_backend": "noop",
        "check_layer": 3,
        "recompute_ratio": 0.25,
        "max_group_size": 5,
        "max_group_wait_ms": 0,
        "require_same_active_length": True,
        "fail_open": False,
        "strict_validation": True,
        "master_selection": "min_total_changed_blocks",
        "storage_mode": "reference",
    }

    config = KMeshEngineConfig.from_connector_extra_config({"kmesh": raw})

    assert config.to_primitive() == raw


@pytest.mark.parametrize(
    ("raw", "error"),
    [
        ({"enabled": 1}, TypeError),
        ({"enabled": True, "mode": False}, TypeError),
        ({"enabled": True, "mode": "bogus"}, ValueError),
        ({"enabled": False, "mode": "observe"}, ValueError),
        ({"enabled": True, "mode": "off"}, ValueError),
        ({"enabled": True, "check_layer": True}, TypeError),
        ({"enabled": True, "recompute_ratio": True}, TypeError),
        ({"enabled": True, "recompute_ratio": 0}, ValueError),
        ({"enabled": True, "max_group_size": 1}, ValueError),
        ({"enabled": True, "max_group_wait_ms": 1}, ValueError),
        ({"enabled": True, "require_same_active_length": False}, ValueError),
        ({"enabled": True, "surprise": 1}, ValueError),
    ],
)
def test_rejects_invalid_or_unsupported_config(
    raw: dict[str, object], error: type[Exception]
) -> None:
    with pytest.raises(error):
        KMeshEngineConfig.from_connector_extra_config({"kmesh": raw})


@pytest.mark.parametrize("extra_config", [[], {"kmesh": []}, {"kmesh": {1: True}}])
def test_rejects_non_mapping_config(extra_config: object) -> None:
    with pytest.raises(TypeError):
        KMeshEngineConfig.from_connector_extra_config(extra_config)


def _serial_pic_v1_vllm_config() -> SimpleNamespace:
    engine_id = "kmesh-serial-pic-v1"
    hf_config = Qwen2Config(
        architectures=["Qwen2ForCausalLM"],
        attention_dropout=0.0,
        bos_token_id=151643,
        eos_token_id=151645,
        hidden_act="silu",
        hidden_size=896,
        initializer_range=0.02,
        intermediate_size=4864,
        max_position_embeddings=32768,
        max_window_layers=21,
        num_attention_heads=14,
        num_hidden_layers=24,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        sliding_window=32768,
        tie_word_embeddings=True,
        dtype=torch.bfloat16,
        use_cache=True,
        use_sliding_window=False,
        vocab_size=151936,
    )
    connectors = [
        {
            "kv_connector": "KMeshConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "kmesh": {
                    "enabled": True,
                    "mode": "serial_pic",
                    "segment_encoding": "sideband",
                    "pic_backend": "lmcache",
                    "check_layer": 1,
                    "recompute_ratio": 0.15,
                    "max_group_size": 10,
                    "max_group_wait_ms": 0,
                    "require_same_active_length": True,
                    "fail_open": False,
                    "strict_validation": True,
                    "master_selection": "min_total_changed_blocks",
                    "storage_mode": "none",
                }
            },
        },
        {
            "kv_connector": "LMCacheConnectorV1",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {"use_native": False},
        },
    ]
    return SimpleNamespace(
        kv_transfer_config=KVTransferConfig(
            kv_connector="MultiConnector",
            kv_role="kv_both",
            engine_id=engine_id,
            kv_buffer_device="cuda",
            kv_connector_module_path=None,
            enable_permute_local_kv=False,
            kv_load_failure_policy="recompute",
            kv_connector_extra_config={
                KMESH_TARGET_PROFILE_KEY: KMESH_SERIAL_PIC_V1_PROFILE,
                "connectors": connectors,
            },
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            data_parallel_size_local=1,
            data_parallel_rank=0,
            data_parallel_rank_local=0,
            distributed_executor_backend="uni",
            world_size=1,
            rank=0,
            nnodes=1,
            node_rank=0,
            data_parallel_external_lb=False,
            data_parallel_hybrid_lb=False,
            enable_expert_parallel=False,
            enable_elastic_ep=False,
            enable_fault_tolerance=False,
            disable_custom_all_reduce=True,
            enable_dbo=False,
            ubatch_size=0,
            worker_cls="vllm.v1.worker.gpu_worker.Worker",
            worker_extension_cls="",
        ),
        model_config=SimpleNamespace(
            model=KMESH_SERIAL_PIC_V1_MODEL_PATH,
            model_weights="",
            tokenizer=KMESH_SERIAL_PIC_V1_MODEL_PATH,
            served_model_name=KMESH_SERIAL_PIC_V1_MODEL_PATH,
            runner="auto",
            model_impl="auto",
            override_attention_dtype=None,
            seed=20260731,
            hf_config_path=None,
            code_revision=None,
            dtype=torch.bfloat16,
            enforce_eager=True,
            max_model_len=2048,
            architectures=["Qwen2ForCausalLM"],
            revision="7ae557604adf67be50417f59c2c2f167def9a775",
            tokenizer_revision="7ae557604adf67be50417f59c2c2f167def9a775",
            runner_type="generate",
            quantization=None,
            quantization_config=None,
            enable_prompt_embeds=False,
            trust_remote_code=False,
            hf_overrides={},
            skip_tokenizer_init=False,
            model_class_overrides={},
            is_encoder_decoder=False,
            is_diffusion=False,
            uses_alibi=False,
            uses_mrope=False,
            is_multimodal_model=False,
            use_mla=False,
            is_hybrid=False,
            get_sliding_window=lambda: None,
            disable_cascade_attn=True,
            hf_config=hf_config,
            hf_text_config=hf_config,
            model_arch_config=ModelArchitectureConfig(
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
                num_experts_per_token=0,
                quantization_config=None,
                is_deepseek_mla=False,
                is_mm_prefix_lm=False,
                rswa_window=None,
                derived_max_model_len_and_key=(
                    32768.0,
                    "max_position_embeddings",
                ),
            ),
        ),
        cache_config=SimpleNamespace(
            cache_dtype="auto",
            block_size=32,
            gpu_memory_utilization=0.3,
            enable_prefix_caching=False,
            sliding_window=None,
            mamba_cache_mode="none",
            kv_cache_dtype_skip_layers=[],
            kv_offloading_size=None,
            kv_cache_memory_bytes=None,
            kv_sharing_fast_prefill=False,
            use_replayssm=False,
            is_attention_free=False,
        ),
        scheduler_config=SimpleNamespace(
            async_scheduling=False,
            enable_chunked_prefill=False,
            disable_hybrid_kv_cache_manager=True,
            is_encoder_decoder=False,
            scheduler_cls=None,
        ),
        device_config=SimpleNamespace(device_type="cuda"),
        load_config=SimpleNamespace(load_format="auto", model_loader_extra_config={}),
        offload_config=OffloadConfig(),
        attention_config=SimpleNamespace(
            backend=None,
            backend_per_kind={},
            flash_attn_version=None,
            flash_attn_max_num_splits_for_cuda_graph=32,
            tq_max_kv_splits_for_cuda_graph=32,
            use_trtllm_attention=None,
            disable_flashinfer_q_quantization=False,
            mla_prefill_backend=None,
            use_prefill_query_quantization=False,
            indexer_kv_dtype="auto",
            use_non_causal=False,
            sparse_mla_force_mqa=False,
            flex_attn_block_m=None,
            flex_attn_block_n=None,
            flex_attn_q_block_size=None,
            flex_attn_kv_block_size=None,
            compute_hash=lambda: "serial-pic-v1-attention",
        ),
        kernel_config=SimpleNamespace(linear_backend="auto"),
        compilation_config=SimpleNamespace(mode=0, cudagraph_mode=0),
        use_v2_model_runner=True,
        additional_config={},
        speculative_config=None,
        lora_config=None,
        diffusion_config=None,
        quant_config=None,
        kv_events_config=None,
        ec_transfer_config=None,
        weight_transfer_config=None,
        reasoning_config=None,
        structured_outputs_config=SimpleNamespace(
            backend="auto",
            disable_any_whitespace=False,
            disable_additional_properties=False,
            reasoning_parser="",
            reasoning_parser_plugin="",
            enable_in_reasoning=False,
        ),
    )


def _collective_lockstep_v1_vllm_config() -> SimpleNamespace:
    config = _serial_pic_v1_vllm_config()
    config.kv_transfer_config.engine_id = "kmesh-collective-lockstep-v1"
    extra_config = config.kv_transfer_config.kv_connector_extra_config
    extra_config[KMESH_TARGET_PROFILE_KEY] = KMESH_COLLECTIVE_LOCKSTEP_V1_PROFILE
    extra_config["connectors"][0]["kv_connector_extra_config"]["kmesh"]["mode"] = (
        "collective_lockstep"
    )
    return config


def _empty_scheduler_output() -> SchedulerOutput:
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def _install_external_adapter(
    monkeypatch: pytest.MonkeyPatch, adapter_class: type[object]
) -> None:
    module_name = "lmcache.integration.vllm.vllm_v1_adapter"
    adapter_module = ModuleType(module_name)
    adapter_module.LMCacheConnectorV1Impl = adapter_class  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module_name, adapter_module)


class _SerialPICExternalAdapter:
    """Minimal public protocol-v2 adapter double for composite tests."""

    KMESH_HANDOFF_PROTOCOL_VERSION = 2

    def __init__(self, *args: object) -> None:
        pass

    def prepare_request_lookup(self, *args: object) -> None:
        pass

    def validate_peer_connector_metadata(
        self, peer_connector_metadata: tuple[KVConnectorMetadata, ...]
    ) -> None:
        pass

    def bind_peer_connector_metadata(
        self, peer_connector_metadata: tuple[KVConnectorMetadata, ...]
    ) -> None:
        pass

    def clear_peer_connector_metadata(self) -> None:
        pass

    def begin_connector_metadata_transaction(self) -> None:
        pass

    def commit_connector_metadata_transaction(self) -> None:
        pass

    def finalize_connector_metadata_transaction(self) -> None:
        pass

    def abort_connector_metadata_transaction(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def build_connector_meta(self, *args: object) -> KVConnectorMetadata:
        return KVConnectorMetadata()


class _TransactionalSerialPICAdapter(_SerialPICExternalAdapter):
    """Replay-aware adapter double for cross-child atomicity tests."""

    instances: list[_TransactionalSerialPICAdapter] = []

    def __init__(self, *args: object) -> None:
        self.last_generation = 0
        self.pending_generation: int | None = None
        self.rollback_generation: int | None = None
        self.transaction_active = False
        self.begin_calls = 0
        self.commit_calls = 0
        self.finalize_calls = 0
        self.abort_calls = 0
        self.clear_calls = 0
        self.commit_error: BaseException | None = None
        self.drift_protocol_on_commit = False
        self.finalize_error: BaseException | None = None
        self.abort_error: BaseException | None = None
        self.abort_error_after_restore = False
        type(self).instances.append(self)

    @classmethod
    def reset(cls) -> None:
        cls.instances = []

    def _validated_generation(
        self, peer_connector_metadata: tuple[KVConnectorMetadata, ...]
    ) -> int:
        kmesh_metadata = peer_connector_metadata[0]
        assert isinstance(kmesh_metadata, KMeshConnectorMetadata)
        generation = kmesh_metadata.step_generation
        if generation <= self.last_generation:
            raise ValueError("external KMesh metadata is stale")
        return generation

    def validate_peer_connector_metadata(
        self, peer_connector_metadata: tuple[KVConnectorMetadata, ...]
    ) -> None:
        self._validated_generation(peer_connector_metadata)

    def bind_peer_connector_metadata(
        self, peer_connector_metadata: tuple[KVConnectorMetadata, ...]
    ) -> None:
        generation = self._validated_generation(peer_connector_metadata)
        if self.transaction_active:
            self.pending_generation = generation
        else:
            self.last_generation = generation

    def clear_peer_connector_metadata(self) -> None:
        self.clear_calls += 1
        self.pending_generation = None

    def begin_connector_metadata_transaction(self) -> None:
        if self.transaction_active:
            raise RuntimeError("external metadata transaction is already active")
        self.begin_calls += 1
        self.rollback_generation = self.last_generation
        self.pending_generation = None
        self.transaction_active = True

    def commit_connector_metadata_transaction(self) -> None:
        self.commit_calls += 1
        if not self.transaction_active or self.pending_generation is None:
            raise RuntimeError("external transaction has no pending generation")
        self.last_generation = self.pending_generation
        if self.drift_protocol_on_commit:
            self.KMESH_HANDOFF_PROTOCOL_VERSION = 3
            self.drift_protocol_on_commit = False
        error = self.commit_error
        self.commit_error = None
        if error is not None:
            raise error

    def finalize_connector_metadata_transaction(self) -> None:
        self.finalize_calls += 1
        self.pending_generation = None
        self.transaction_active = False
        error = self.finalize_error
        self.finalize_error = None
        if error is not None:
            raise error

    def abort_connector_metadata_transaction(self) -> None:
        self.abort_calls += 1
        error = self.abort_error
        self.abort_error = None
        if error is not None and not self.abort_error_after_restore:
            raise error
        if self.rollback_generation is not None:
            self.last_generation = self.rollback_generation
        self.pending_generation = None
        self.rollback_generation = None
        self.transaction_active = False
        if error is not None:
            raise error


def test_serial_pic_v1_target_profile_accepts_exact_outer_runtime() -> None:
    config = _serial_pic_v1_vllm_config()

    assert validate_kmesh_target_profile(config) == KMESH_SERIAL_PIC_V1_PROFILE


@pytest.mark.parametrize("substitute_class", [False, True])
def test_mp_child_inspection_requires_the_resolved_connector_class(
    monkeypatch: pytest.MonkeyPatch, substitute_class: bool
) -> None:
    """Permit inspection without granting a lookalike MP class admission."""

    class ResolvedMPConnector:
        pass

    class LookalikeMPConnector:
        pass

    module_name = "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_connector"
    module = ModuleType(module_name)
    module.LMCacheMPConnector = ResolvedMPConnector  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module_name, module)
    original_resolve = KVConnectorFactory.get_connector_class

    def resolve(cls: type, transfer: KVTransferConfig) -> type:
        if transfer.kv_connector == "LMCacheMPConnector":
            return LookalikeMPConnector if substitute_class else ResolvedMPConnector
        return original_resolve(transfer)

    monkeypatch.setattr(KVConnectorFactory, "get_connector_class", classmethod(resolve))
    config = _serial_pic_v1_vllm_config()
    config.kv_transfer_config.kv_connector_extra_config["connectors"][1][
        "kv_connector"
    ] = "LMCacheMPConnector"
    assert validate_kmesh_target_profile(config) == KMESH_SERIAL_PIC_V1_PROFILE
    if substitute_class:
        with pytest.raises(ValueError, match="exact built-in LMCacheMPConnector"):
            MultiConnector.inspect_connector_classes_and_configs(config)
    else:
        children = MultiConnector.inspect_connector_classes_and_configs(config)
        assert children[1][0] is ResolvedMPConnector


@pytest.mark.parametrize("child", ["UnknownConnector", [], False, None])
def test_target_profile_rejects_unadmitted_child_names(child: object) -> None:
    config = _serial_pic_v1_vllm_config()
    config.kv_transfer_config.kv_connector_extra_config["connectors"][1][
        "kv_connector"
    ] = child
    with pytest.raises(ValueError, match="connector order"):
        validate_kmesh_target_profile(config)


def _collective_batched_position_recovery_v1_vllm_config() -> SimpleNamespace:
    config = _serial_pic_v1_vllm_config()
    config.kv_transfer_config.engine_id = "kmesh-collective-batched-pr-v1"
    extra_config = config.kv_transfer_config.kv_connector_extra_config
    extra_config[KMESH_TARGET_PROFILE_KEY] = (
        KMESH_COLLECTIVE_BATCHED_POSITION_RECOVERY_V1_PROFILE
    )
    extra_config["connectors"][0]["kv_connector_extra_config"]["kmesh"]["mode"] = (
        "collective_batched"
    )
    return config


def _collective_batched_qkv_v1_vllm_config() -> SimpleNamespace:
    config = _serial_pic_v1_vllm_config()
    config.kv_transfer_config.engine_id = "kmesh-collective-batched-qkv-v1"
    extra_config = config.kv_transfer_config.kv_connector_extra_config
    extra_config[KMESH_TARGET_PROFILE_KEY] = KMESH_COLLECTIVE_BATCHED_QKV_V1_PROFILE
    extra_config["connectors"][0]["kv_connector_extra_config"]["kmesh"]["mode"] = (
        "collective_batched"
    )
    return config


def test_collective_batched_qkv_profile_accepts_exact_sibling_envelope() -> None:
    batched = _collective_batched_position_recovery_v1_vllm_config()
    qkv = _collective_batched_qkv_v1_vllm_config()

    assert validate_kmesh_target_profile(qkv) == KMESH_COLLECTIVE_BATCHED_QKV_V1_PROFILE
    batched_kmesh = batched.kv_transfer_config.kv_connector_extra_config["connectors"][
        0
    ]["kv_connector_extra_config"]["kmesh"]
    qkv_kmesh = qkv.kv_transfer_config.kv_connector_extra_config["connectors"][0][
        "kv_connector_extra_config"
    ]["kmesh"]
    assert qkv_kmesh == batched_kmesh

    children = MultiConnector.inspect_connector_classes_and_configs(qkv)
    assert all(
        child.kv_transfer_config.kv_connector_extra_config[KMESH_TARGET_PROFILE_KEY]
        == KMESH_COLLECTIVE_BATCHED_QKV_V1_PROFILE
        for _, child in children
    )


def test_collective_batched_profile_accepts_exact_sibling_envelope() -> None:
    lockstep = _collective_lockstep_v1_vllm_config()
    batched = _collective_batched_position_recovery_v1_vllm_config()

    assert (
        validate_kmesh_target_profile(batched)
        == KMESH_COLLECTIVE_BATCHED_POSITION_RECOVERY_V1_PROFILE
    )
    lockstep_kmesh = lockstep.kv_transfer_config.kv_connector_extra_config[
        "connectors"
    ][0]["kv_connector_extra_config"]["kmesh"]
    batched_kmesh = batched.kv_transfer_config.kv_connector_extra_config["connectors"][
        0
    ]["kv_connector_extra_config"]["kmesh"]
    assert {
        **batched_kmesh,
        "mode": "collective_lockstep",
    } == lockstep_kmesh

    children = MultiConnector.inspect_connector_classes_and_configs(batched)
    assert all(
        child.kv_transfer_config.kv_connector_extra_config[KMESH_TARGET_PROFILE_KEY]
        == KMESH_COLLECTIVE_BATCHED_POSITION_RECOVERY_V1_PROFILE
        for _, child in children
    )


def test_collective_lockstep_v1_accepts_exact_serial_sibling_envelope() -> None:
    serial = _serial_pic_v1_vllm_config()
    collective = _collective_lockstep_v1_vllm_config()

    assert (
        validate_kmesh_target_profile(collective)
        == KMESH_COLLECTIVE_LOCKSTEP_V1_PROFILE
    )
    serial_kmesh = serial.kv_transfer_config.kv_connector_extra_config["connectors"][0][
        "kv_connector_extra_config"
    ]["kmesh"]
    collective_kmesh = collective.kv_transfer_config.kv_connector_extra_config[
        "connectors"
    ][0]["kv_connector_extra_config"]["kmesh"]
    assert {
        **collective_kmesh,
        "mode": "serial_pic",
    } == serial_kmesh

    children = MultiConnector.inspect_connector_classes_and_configs(collective)
    assert all(
        child.kv_transfer_config.kv_connector_extra_config[KMESH_TARGET_PROFILE_KEY]
        == KMESH_COLLECTIVE_LOCKSTEP_V1_PROFILE
        for _, child in children
    )


@pytest.mark.parametrize(
    ("factory", "profile"),
    [
        (_serial_pic_v1_vllm_config, KMESH_SERIAL_PIC_V1_PROFILE),
        (
            _collective_lockstep_v1_vllm_config,
            KMESH_COLLECTIVE_LOCKSTEP_V1_PROFILE,
        ),
        (
            _collective_batched_position_recovery_v1_vllm_config,
            KMESH_COLLECTIVE_BATCHED_POSITION_RECOVERY_V1_PROFILE,
        ),
        (
            _collective_batched_qkv_v1_vllm_config,
            KMESH_COLLECTIVE_BATCHED_QKV_V1_PROFILE,
        ),
    ],
)
def test_target_profiles_accept_exact_triton_attention_backend(
    factory: Callable[[], SimpleNamespace], profile: str
) -> None:
    config = factory()
    config.attention_config = AttentionConfig(backend="TRITON_ATTN")

    assert validate_kmesh_target_profile(config) == profile


def test_collective_lockstep_v1_rejects_serial_mode_alias() -> None:
    config = _collective_lockstep_v1_vllm_config()
    kmesh = config.kv_transfer_config.kv_connector_extra_config["connectors"][0][
        "kv_connector_extra_config"
    ]["kmesh"]
    kmesh["mode"] = "serial_pic"

    with pytest.raises(ValueError, match="kmesh.mode='collective_lockstep'"):
        validate_kmesh_target_profile(config)


def test_collective_lockstep_v1_rejects_non_lmcache_backend() -> None:
    config = _collective_lockstep_v1_vllm_config()
    kmesh = config.kv_transfer_config.kv_connector_extra_config["connectors"][0][
        "kv_connector_extra_config"
    ]["kmesh"]
    kmesh["pic_backend"] = "noop"

    with pytest.raises(ValueError, match="pic_backend='lmcache'"):
        validate_kmesh_target_profile(config)


def test_absent_target_profile_leaves_unrelated_configs_unchanged() -> None:
    config = SimpleNamespace(
        kv_transfer_config=KVTransferConfig(
            kv_connector="MultiConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"connectors": []},
        )
    )

    assert validate_kmesh_target_profile(config) is None


def test_target_profile_is_propagated_to_copied_child_configs() -> None:
    config = _serial_pic_v1_vllm_config()
    original_children = copy.deepcopy(
        config.kv_transfer_config.kv_connector_extra_config["connectors"]
    )

    children = MultiConnector.inspect_connector_classes_and_configs(config)

    assert len(children) == 2
    for _, child_config in children:
        assert child_config.kv_transfer_config is not None
        assert (
            child_config.kv_transfer_config.kv_connector_extra_config[
                KMESH_TARGET_PROFILE_KEY
            ]
            == KMESH_SERIAL_PIC_V1_PROFILE
        )
        assert child_config.kv_transfer_config.kv_load_failure_policy == "recompute"
        assert (
            child_config.kv_transfer_config.engine_id
            == config.kv_transfer_config.engine_id
        )
    assert (
        config.kv_transfer_config.kv_connector_extra_config["connectors"]
        == original_children
    )
    kmesh_cls, kmesh_config = children[0]
    with pytest.raises(ValueError, match="internally qualified"):
        kmesh_cls(
            kmesh_config,
            KVConnectorRole.SCHEDULER,
            SimpleNamespace(kv_cache_groups=[]),
        )

    kmesh_child = children[0][1]
    child_kmesh = kmesh_child.kv_transfer_config.kv_connector_extra_config["kmesh"]
    child_kmesh["check_layer"] = 99
    original_kmesh = config.kv_transfer_config.kv_connector_extra_config["connectors"][
        0
    ]["kv_connector_extra_config"]["kmesh"]
    assert original_kmesh["check_layer"] == 1


def test_serial_pic_v1_with_real_vllm_config_dataclasses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setattr(vllm_config_module, "HAS_TRITON", True)
    Qwen2Config(
        architectures=["Qwen2ForCausalLM"],
        attention_dropout=0.0,
        bos_token_id=151643,
        eos_token_id=151645,
        hidden_act="silu",
        hidden_size=896,
        initializer_range=0.02,
        intermediate_size=4864,
        max_position_embeddings=32768,
        max_window_layers=21,
        num_attention_heads=14,
        num_hidden_layers=24,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        sliding_window=32768,
        tie_word_embeddings=True,
        dtype=torch.bfloat16,
        use_cache=True,
        use_sliding_window=False,
        vocab_size=151936,
    ).save_pretrained(tmp_path)
    model_config = ModelConfig(
        model=str(tmp_path),
        tokenizer=str(tmp_path),
        dtype="bfloat16",
        seed=20260731,
        enforce_eager=True,
        max_model_len=2048,
        revision="7ae557604adf67be50417f59c2c2f167def9a775",
        tokenizer_revision="7ae557604adf67be50417f59c2c2f167def9a775",
        skip_tokenizer_init=False,
    )
    parallel_config = ParallelConfig(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
        distributed_executor_backend="uni",
        disable_custom_all_reduce=True,
        enable_dbo=False,
        ubatch_size=0,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=4,
        max_num_batched_tokens=2048,
        max_model_len=2048,
        enable_chunked_prefill=False,
        async_scheduling=False,
        is_encoder_decoder=False,
        disable_hybrid_kv_cache_manager=True,
    )
    cache_config = CacheConfig(
        block_size=32,
        cache_dtype="auto",
        enable_prefix_caching=False,
        gpu_memory_utilization=0.3,
    )
    simple_config = _serial_pic_v1_vllm_config()
    kv_transfer_config = copy.deepcopy(simple_config.kv_transfer_config)
    config = VllmConfig(
        model_config=model_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        cache_config=cache_config,
        device_config=DeviceConfig("cuda"),
        structured_outputs_config=StructuredOutputsConfig(),
        kv_transfer_config=kv_transfer_config,
    )
    config.model_config.model = KMESH_SERIAL_PIC_V1_MODEL_PATH
    config.model_config.tokenizer = KMESH_SERIAL_PIC_V1_MODEL_PATH
    config.model_config.served_model_name = KMESH_SERIAL_PIC_V1_MODEL_PATH
    config.parallel_config.worker_cls = "vllm.v1.worker.gpu_worker.Worker"

    assert config.model_config.dtype is torch.bfloat16
    assert config.model_config.enforce_eager is True
    assert config.model_config.max_model_len == 2048
    assert config.parallel_config.decode_context_parallel_size == 1
    assert config.parallel_config.prefill_context_parallel_size == 1
    assert config.parallel_config.distributed_executor_backend == "uni"
    assert config.cache_config.mamba_cache_mode == "none"
    assert config.structured_outputs_config == StructuredOutputsConfig()
    assert validate_kmesh_target_profile(config) == KMESH_SERIAL_PIC_V1_PROFILE

    children = MultiConnector.inspect_connector_classes_and_configs(config)
    assert [child.kv_transfer_config.kv_connector for _, child in children] == [
        "KMeshConnector",
        "LMCacheConnectorV1",
    ]
    assert all(
        child.kv_transfer_config.kv_connector_extra_config[KMESH_TARGET_PROFILE_KEY]
        == KMESH_SERIAL_PIC_V1_PROFILE
        for _, child in children
    )

    overridden_model_config = ModelConfig(
        model=str(tmp_path),
        tokenizer=str(tmp_path),
        dtype="bfloat16",
        seed=20260731,
        enforce_eager=True,
        max_model_len=2048,
        revision="7ae557604adf67be50417f59c2c2f167def9a775",
        tokenizer_revision="7ae557604adf67be50417f59c2c2f167def9a775",
        skip_tokenizer_init=False,
        hf_overrides={"rms_norm_eps": 0.01},
    )
    overridden_model_config.model = KMESH_SERIAL_PIC_V1_MODEL_PATH
    overridden_model_config.tokenizer = KMESH_SERIAL_PIC_V1_MODEL_PATH
    overridden_model_config.served_model_name = KMESH_SERIAL_PIC_V1_MODEL_PATH
    assert overridden_model_config.hf_text_config.rms_norm_eps == 0.01
    config.model_config = overridden_model_config

    with pytest.raises(ValueError, match="no Hugging Face config overrides"):
        validate_kmesh_target_profile(config)


def test_serial_pic_v1_with_resolved_engine_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setattr(vllm_config_module, "HAS_TRITON", True)
    monkeypatch.setattr(vllm_platforms.current_platform, "device_type", "cuda")

    def resolve_gpu_worker(config: VllmConfig) -> None:
        if config.parallel_config.worker_cls == "auto":
            config.parallel_config.worker_cls = "vllm.v1.worker.gpu_worker.Worker"

    monkeypatch.setattr(
        vllm_platforms.current_platform,
        "check_and_update_config",
        resolve_gpu_worker,
    )
    simple_config = _serial_pic_v1_vllm_config()
    simple_config.model_config.hf_config.save_pretrained(tmp_path)
    engine_args = EngineArgs(
        model=str(tmp_path),
        tokenizer=str(tmp_path),
        dtype="bfloat16",
        seed=20260731,
        max_model_len=2048,
        enforce_eager=True,
        revision="7ae557604adf67be50417f59c2c2f167def9a775",
        tokenizer_revision="7ae557604adf67be50417f59c2c2f167def9a775",
        block_size=32,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.3,
        distributed_executor_backend="uni",
        disable_custom_all_reduce=True,
        enable_chunked_prefill=False,
        async_scheduling=False,
        disable_hybrid_kv_cache_manager=True,
        cpu_offload_gb=0.0,
        kv_transfer_config=copy.deepcopy(simple_config.kv_transfer_config),
    )

    config = engine_args.create_engine_config()
    config.model_config.model = KMESH_SERIAL_PIC_V1_MODEL_PATH
    config.model_config.tokenizer = KMESH_SERIAL_PIC_V1_MODEL_PATH
    config.model_config.served_model_name = KMESH_SERIAL_PIC_V1_MODEL_PATH

    assert config.parallel_config.worker_cls == "vllm.v1.worker.gpu_worker.Worker"
    assert type(config.offload_config.uva.cpu_offload_gb) is float
    assert config.offload_config.uva.cpu_offload_gb == 0.0
    assert validate_kmesh_target_profile(config) == KMESH_SERIAL_PIC_V1_PROFILE


def test_profile_validation_precedes_connector_resolution(monkeypatch) -> None:
    config = _serial_pic_v1_vllm_config()
    _reverse_connectors(config)
    get_connector_class = MagicMock(
        side_effect=AssertionError("connector resolution must not run")
    )
    monkeypatch.setattr(KVConnectorFactory, "get_connector_class", get_connector_class)

    with pytest.raises(ValueError, match="connector order"):
        MultiConnector.inspect_connector_classes_and_configs(config)

    get_connector_class.assert_not_called()


@pytest.mark.parametrize(
    "profile",
    [True, "serial_pic", "collective_lockstep_v2", [], {}],
)
def test_rejects_unknown_target_profile(profile: object) -> None:
    config = _serial_pic_v1_vllm_config()
    config.kv_transfer_config.kv_connector_extra_config[KMESH_TARGET_PROFILE_KEY] = (
        profile
    )

    with pytest.raises(ValueError, match="unsupported KMesh target profile"):
        validate_kmesh_target_profile(config)


def _reverse_connectors(config: SimpleNamespace) -> None:
    config.kv_transfer_config.kv_connector_extra_config["connectors"].reverse()


def _drop_lmcache_connector(config: SimpleNamespace) -> None:
    config.kv_transfer_config.kv_connector_extra_config["connectors"].pop()


def _add_nonstring_connector_key(config: SimpleNamespace) -> None:
    connectors = config.kv_transfer_config.kv_connector_extra_config["connectors"]
    connectors[0][1] = "invalid"  # type: ignore[index]


def _set_child_extra(
    config: SimpleNamespace, child_index: int, key: str, value: object
) -> None:
    children = config.kv_transfer_config.kv_connector_extra_config["connectors"]
    children[child_index]["kv_connector_extra_config"][key] = value


def _set_child_field(
    config: SimpleNamespace, child_index: int, key: str, value: object
) -> None:
    children = config.kv_transfer_config.kv_connector_extra_config["connectors"]
    children[child_index][key] = value


def _set_kmesh_field(config: SimpleNamespace, key: str, value: object) -> None:
    children = config.kv_transfer_config.kv_connector_extra_config["connectors"]
    children[0]["kv_connector_extra_config"]["kmesh"][key] = value


@pytest.mark.parametrize(
    ("mutation", "requirement"),
    [
        (_reverse_connectors, "connector order"),
        (_drop_lmcache_connector, "exactly two"),
        (_add_nonstring_connector_key, "string-keyed connector mappings"),
        (
            lambda config: _set_child_extra(config, 1, "use_native", True),
            "use_native=false",
        ),
        (
            lambda config: _set_child_extra(
                config, 1, KMESH_TARGET_PROFILE_KEY, "serial_pic_v2"
            ),
            "declared only on the outer",
        ),
        (
            lambda config: setattr(
                config.kv_transfer_config, "kv_load_failure_policy", "fail"
            ),
            "kv_load_failure_policy",
        ),
        (
            lambda config: config.kv_transfer_config.kv_connector_extra_config[
                "connectors"
            ][1].__setitem__("kv_load_failure_policy", "fail"),
            "child KV load failure policy",
        ),
        (
            lambda config: setattr(config.parallel_config, "tensor_parallel_size", 2),
            "TP=1",
        ),
        (
            lambda config: setattr(config.parallel_config, "pipeline_parallel_size", 2),
            "PP=1",
        ),
        (
            lambda config: setattr(config.parallel_config, "data_parallel_size", 2),
            "DP=1",
        ),
        (
            lambda config: setattr(
                config.parallel_config, "decode_context_parallel_size", 2
            ),
            "DCP=1",
        ),
        (
            lambda config: setattr(
                config.parallel_config, "prefill_context_parallel_size", 2
            ),
            "PCP=1",
        ),
        (
            lambda config: setattr(
                config.parallel_config, "distributed_executor_backend", "mp"
            ),
            "distributed_executor_backend",
        ),
        (
            lambda config: setattr(config.parallel_config, "enable_dbo", True),
            "DBO disabled",
        ),
        (
            lambda config: setattr(config.parallel_config, "ubatch_size", 2),
            "ubatching disabled",
        ),
        (
            lambda config: setattr(config.model_config, "dtype", torch.float16),
            "BF16",
        ),
        (
            lambda config: setattr(config.model_config, "enforce_eager", False),
            "eager execution",
        ),
        (
            lambda config: setattr(config.model_config, "max_model_len", 4096),
            "max_model_len=2048",
        ),
        (
            lambda config: setattr(config.model_config, "is_multimodal_model", True),
            "text-only model",
        ),
        (
            lambda config: setattr(config.model_config, "use_mla", True),
            "MLA disabled",
        ),
        (
            lambda config: setattr(config.model_config, "is_hybrid", True),
            "hybrid/Mamba models disabled",
        ),
        (
            lambda config: setattr(
                config.model_config, "get_sliding_window", lambda: 128
            ),
            "sliding-window attention disabled",
        ),
        (
            lambda config: setattr(config.model_config, "disable_cascade_attn", False),
            "cascade attention disabled",
        ),
        (
            lambda config: setattr(config.cache_config, "enable_prefix_caching", True),
            "prefix caching disabled",
        ),
        (
            lambda config: setattr(config.cache_config, "sliding_window", 128),
            "sliding-window KV cache disabled",
        ),
        (
            lambda config: setattr(config.cache_config, "mamba_cache_mode", "all"),
            "Mamba cache disabled",
        ),
        (
            lambda config: setattr(config.scheduler_config, "async_scheduling", True),
            "async scheduling disabled",
        ),
        (
            lambda config: setattr(
                config.scheduler_config, "enable_chunked_prefill", True
            ),
            "chunked prefill disabled",
        ),
        (
            lambda config: setattr(
                config.scheduler_config, "disable_hybrid_kv_cache_manager", False
            ),
            "hybrid KV cache manager disabled",
        ),
        (
            lambda config: setattr(config, "speculative_config", object()),
            "speculative decoding disabled",
        ),
        (
            lambda config: setattr(config, "lora_config", object()),
            "LoRA disabled",
        ),
        (
            lambda config: setattr(
                config.structured_outputs_config, "backend", "xgrammar"
            ),
            "structured-output controls",
        ),
    ],
)
def test_serial_pic_v1_target_profile_rejects_global_mutations(
    mutation: Callable[[SimpleNamespace], None], requirement: str
) -> None:
    config = _serial_pic_v1_vllm_config()
    mutation(config)

    with pytest.raises(ValueError, match=requirement):
        validate_kmesh_target_profile(config)


@pytest.mark.parametrize(
    ("mutation", "requirement"),
    [
        (
            lambda config: setattr(
                config.kv_transfer_config,
                "kv_connector_module_path",
                "untrusted.connector",
            ),
            "outer kv_connector_module_path unset",
        ),
        (
            lambda config: setattr(config.kv_transfer_config, "kv_role", "kv_producer"),
            "outer kv_role",
        ),
        (
            lambda config: setattr(config.kv_transfer_config, "kv_parallel_size", 2),
            "retired KV-transfer field kv_parallel_size",
        ),
        (
            lambda config: setattr(config.kv_transfer_config, "kv_rank", 0),
            "retired KV-transfer field kv_rank",
        ),
        (
            lambda config: setattr(
                config.kv_transfer_config, "enable_permute_local_kv", True
            ),
            "outer enable_permute_local_kv=false",
        ),
        (
            lambda config: setattr(config.kv_transfer_config, "engine_id", ""),
            "nonempty outer engine_id",
        ),
        (
            lambda config: _set_child_extra(
                config,
                0,
                KMESH_TARGET_PROFILE_KEY,
                KMESH_SERIAL_PIC_V1_PROFILE,
            ),
            "declared only on the outer",
        ),
        (
            lambda config: _set_child_field(
                config, 0, "kv_connector_module_path", "untrusted.kmesh"
            ),
            "child 0 kv_connector_module_path unset",
        ),
        (
            lambda config: _set_child_field(config, 0, "kv_role", "kv_producer"),
            "child 0 kv_role",
        ),
        (
            lambda config: _set_child_field(config, 0, "kv_parallel_size", 2),
            "no unreviewed child 0 KV-transfer fields",
        ),
        (
            lambda config: _set_child_field(config, 0, "kv_rank", 0),
            "no unreviewed child 0 KV-transfer fields",
        ),
        (
            lambda config: _set_child_field(config, 0, "enable_permute_local_kv", True),
            "child 0 enable_permute_local_kv=false",
        ),
        (
            lambda config: _set_child_field(config, 0, "engine_id", "other"),
            "child 0 engine_id inherited",
        ),
        (
            lambda config: _set_child_field(
                config, 1, "kv_connector_module_path", "untrusted.lmcache"
            ),
            "child 1 kv_connector_module_path unset",
        ),
        (
            lambda config: _set_child_field(config, 1, "kv_role", "kv_consumer"),
            "child 1 kv_role",
        ),
        (
            lambda config: _set_child_field(config, 1, "kv_parallel_size", 2),
            "no unreviewed child 1 KV-transfer fields",
        ),
        (
            lambda config: _set_child_field(config, 1, "kv_rank", 0),
            "no unreviewed child 1 KV-transfer fields",
        ),
        (
            lambda config: _set_child_field(config, 1, "enable_permute_local_kv", True),
            "child 1 enable_permute_local_kv=false",
        ),
        (
            lambda config: _set_child_field(config, 1, "engine_id", "other"),
            "child 1 engine_id inherited",
        ),
    ],
)
def test_serial_pic_v1_rejects_transport_and_child_bypasses(
    mutation: Callable[[SimpleNamespace], None], requirement: str
) -> None:
    config = _serial_pic_v1_vllm_config()
    mutation(config)

    with pytest.raises(ValueError, match=requirement):
        validate_kmesh_target_profile(config)


def test_serial_pic_v1_requires_exact_builtin_connector_classes(monkeypatch) -> None:
    config = _serial_pic_v1_vllm_config()
    monkeypatch.setattr(
        KVConnectorFactory,
        "get_connector_class",
        MagicMock(return_value=object),
    )

    with pytest.raises(ValueError, match="exact built-in KMeshConnector"):
        MultiConnector.inspect_connector_classes_and_configs(config)


@pytest.mark.parametrize(
    ("field", "value", "requirement"),
    [
        ("enabled", False, "enabled=true"),
        ("segment_encoding", "ttsep", "segment_encoding='sideband'"),
        ("check_layer", 2, "check_layer=1"),
        ("recompute_ratio", 0.2, "recompute_ratio=0.15"),
        ("max_group_size", 9, "max_group_size=10"),
        ("max_group_wait_ms", 1, "supports only 0"),
        ("require_same_active_length", False, "outside the v0 safety boundary"),
        ("fail_open", True, "fail_open=false"),
        ("strict_validation", False, "strict_validation=true"),
        ("master_selection", "first", "must be one of"),
        ("storage_mode", "reference", "storage_mode='none'"),
    ],
)
def test_serial_pic_v1_rejects_kmesh_correctness_knob_drift(
    field: str, value: object, requirement: str
) -> None:
    config = _serial_pic_v1_vllm_config()
    _set_kmesh_field(config, field, value)

    with pytest.raises((TypeError, ValueError), match=requirement):
        validate_kmesh_target_profile(config)


@pytest.mark.parametrize(
    ("mutation", "requirement"),
    [
        (
            lambda config: setattr(
                config.parallel_config, "data_parallel_size_local", 2
            ),
            "local DP size=1",
        ),
        (
            lambda config: setattr(config.parallel_config, "data_parallel_rank", 1),
            "DP rank=0",
        ),
        (
            lambda config: setattr(
                config.parallel_config, "data_parallel_rank_local", 1
            ),
            "local DP rank=0",
        ),
        (
            lambda config: setattr(config.parallel_config, "world_size", 2),
            "world_size=1",
        ),
        (
            lambda config: setattr(config.parallel_config, "rank", 1),
            "global rank=0",
        ),
        (
            lambda config: setattr(config.parallel_config, "nnodes", 2),
            "nnodes=1",
        ),
        (
            lambda config: setattr(config.parallel_config, "node_rank", 1),
            "node_rank=0",
        ),
        (
            lambda config: setattr(
                config.parallel_config, "data_parallel_external_lb", True
            ),
            "external DP load balancing disabled",
        ),
        (
            lambda config: setattr(
                config.parallel_config, "data_parallel_hybrid_lb", True
            ),
            "hybrid DP load balancing disabled",
        ),
        (
            lambda config: setattr(
                config.parallel_config, "enable_expert_parallel", True
            ),
            "expert parallelism disabled",
        ),
        (
            lambda config: setattr(config.parallel_config, "enable_elastic_ep", True),
            "elastic expert parallelism disabled",
        ),
        (
            lambda config: setattr(
                config.parallel_config, "enable_fault_tolerance", True
            ),
            "fault tolerance disabled",
        ),
        (
            lambda config: setattr(
                config.parallel_config, "disable_custom_all_reduce", False
            ),
            "custom all-reduce disabled",
        ),
        (
            lambda config: setattr(
                config.parallel_config, "worker_cls", "custom.Worker"
            ),
            "resolved built-in GPU worker class",
        ),
        (
            lambda config: setattr(config.parallel_config, "worker_cls", "auto"),
            "resolved built-in GPU worker class",
        ),
        (
            lambda config: setattr(
                config.parallel_config, "worker_extension_cls", "custom.Extension"
            ),
            "no worker extension class",
        ),
    ],
)
def test_serial_pic_v1_rejects_rank_and_executor_bypasses(
    mutation: Callable[[SimpleNamespace], None], requirement: str
) -> None:
    config = _serial_pic_v1_vllm_config()
    mutation(config)

    with pytest.raises(ValueError, match=requirement):
        validate_kmesh_target_profile(config)


@pytest.mark.parametrize(
    ("mutation", "requirement"),
    [
        (
            lambda config: setattr(config.model_config, "architectures", ["Other"]),
            "Qwen2ForCausalLM architecture",
        ),
        (
            lambda config: setattr(config.model_config, "revision", "other"),
            "model revision",
        ),
        (
            lambda config: setattr(config.model_config, "tokenizer_revision", "other"),
            "tokenizer revision",
        ),
        (
            lambda config: setattr(config.model_config, "runner_type", "pooling"),
            "generate runner",
        ),
        (
            lambda config: setattr(config.model_config, "quantization", "fp8"),
            "model quantization disabled",
        ),
        (
            lambda config: setattr(config.model_config, "quantization_config", {}),
            "quantization configuration unset",
        ),
        (
            lambda config: setattr(config.model_config, "enable_prompt_embeds", True),
            "prompt embeddings disabled globally",
        ),
        (
            lambda config: setattr(config.model_config, "trust_remote_code", True),
            "trust_remote_code=false",
        ),
        (
            lambda config: setattr(config.model_config, "skip_tokenizer_init", True),
            "tokenizer initialization enabled",
        ),
        (
            lambda config: setattr(
                config.model_config,
                "model_class_overrides",
                {"Qwen2ForCausalLM": "x:y"},
            ),
            "no model class overrides",
        ),
        (
            lambda config: setattr(
                config.model_config, "hf_overrides", {"rms_norm_eps": 0.01}
            ),
            "no Hugging Face config overrides",
        ),
        (
            lambda config: setattr(
                config.model_config.hf_text_config, "rms_norm_eps", 0.01
            ),
            "complete frozen Qwen2.5-0.5B resolved computation identity",
        ),
        (
            lambda config: setattr(config.model_config, "is_encoder_decoder", True),
            "decoder-only model",
        ),
        (
            lambda config: setattr(config.model_config, "is_diffusion", True),
            "non-diffusion model",
        ),
        (
            lambda config: setattr(config.model_config, "uses_alibi", True),
            "ALiBi disabled",
        ),
        (
            lambda config: setattr(config.model_config, "uses_mrope", True),
            "M-RoPE disabled",
        ),
        (
            lambda config: setattr(config.model_config, "uses_xdrope_dim", 4),
            "XDRoPE disabled",
        ),
        (
            lambda config: setattr(
                config.model_config.hf_text_config,
                "rope_parameters",
                {"rope_theta": 10_000.0, "rope_type": "default"},
            ),
            "frozen Qwen2.5 default RoPE",
        ),
    ],
)
def test_serial_pic_v1_rejects_model_identity_and_mode_bypasses(
    mutation: Callable[[SimpleNamespace], None], requirement: str
) -> None:
    config = _serial_pic_v1_vllm_config()
    mutation(config)

    with pytest.raises(ValueError, match=requirement):
        validate_kmesh_target_profile(config)


@pytest.mark.parametrize(
    ("field", "value", "requirement"),
    [
        ("head_dim", 128, "resolved Qwen2 head_dim=64"),
        ("hidden_size_per_head", 128, "resolved Qwen2 head_dim=64"),
        ("n_head_kv", 4, "resolved Qwen2 KV-head count=2"),
        ("n_head_kv", None, "resolved Qwen2 KV-head count=2"),
        ("num_kv_heads", 4, "resolved Qwen2 KV-head count=2"),
        (
            "multi_query_group_num",
            4,
            "complete frozen Qwen2.5-0.5B resolved computation identity",
        ),
        (
            "num_attention_groups",
            4,
            "complete frozen Qwen2.5-0.5B resolved computation identity",
        ),
    ],
)
def test_serial_pic_v1_rejects_generic_cache_geometry_aliases(
    field: str, value: object, requirement: str
) -> None:
    config = _serial_pic_v1_vllm_config()

    setattr(config.model_config.hf_text_config, field, value)

    with pytest.raises(ValueError, match=requirement):
        validate_kmesh_target_profile(config)


def test_serial_pic_v1_rejects_resolved_model_arch_geometry_drift() -> None:
    config = _serial_pic_v1_vllm_config()

    config.model_config.model_arch_config.head_size = 128

    with pytest.raises(ValueError, match="exact resolved Qwen2 model/cache"):
        validate_kmesh_target_profile(config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_experts", 8),
        ("moe_num_experts", 8),
        ("n_routed_experts", 8),
        ("num_local_experts", 8),
        ("block_configs", [{"block_type": "moe", "n_routed_experts": 8}]),
        ("quantization_config", {"quant_method": "fp8"}),
        ("compression_config", {"quant_method": "compressed-tensors"}),
        ("is_mm_prefix_lm", True),
        ("rswa_window", 128),
        ("index_topk", 128),
        ("projection_dim", 128),
        ("projection_size", 128),
        ("n_positions", 4096),
        ("max_seq_len", 4096),
        ("seq_length", 4096),
        ("model_max_length", 4096),
        ("max_target_positions", 4096),
        ("max_sequence_length", 4096),
        ("seq_len", 4096),
    ],
)
def test_serial_pic_v1_rejects_generic_arch_backend_and_loader_inputs(
    field: str, value: object
) -> None:
    config = _serial_pic_v1_vllm_config()

    setattr(config.model_config.hf_text_config, field, value)

    with pytest.raises(
        ValueError,
        match="complete frozen Qwen2.5-0.5B resolved computation identity",
    ):
        validate_kmesh_target_profile(config)


@pytest.mark.parametrize(
    "field", ["text_config", "decoder", "generator", "text_encoder"]
)
def test_serial_pic_v1_rejects_nested_hf_text_config_aliases(field: str) -> None:
    config = _serial_pic_v1_vllm_config()

    setattr(config.model_config.hf_config, field, Qwen2Config(hidden_size=128))

    with pytest.raises(ValueError, match=r"get_text_config\(\)"):
        validate_kmesh_target_profile(config)


@pytest.mark.parametrize(
    ("mutation", "requirement"),
    [
        (
            lambda config: setattr(config.cache_config, "cache_dtype", "bfloat16"),
            "cache_dtype='auto'",
        ),
        (
            lambda config: setattr(config.cache_config, "block_size", 16),
            "block_size=32",
        ),
        (
            lambda config: setattr(config.cache_config, "calculate_kv_scales", True),
            "KV scale calculation disabled",
        ),
        (
            lambda config: setattr(
                config.cache_config, "kv_cache_dtype_skip_layers", ["1"]
            ),
            "no KV dtype skip layers",
        ),
        (
            lambda config: setattr(config.cache_config, "kv_offloading_size", 1.0),
            "native KV offloading disabled",
        ),
        (
            lambda config: setattr(config.cache_config, "kv_cache_memory_bytes", 1024),
            "manual KV cache byte override unset",
        ),
        (
            lambda config: setattr(
                config.cache_config, "kv_sharing_fast_prefill", True
            ),
            "KV-sharing fast prefill disabled",
        ),
        (
            lambda config: setattr(config.cache_config, "use_replayssm", True),
            "ReplaySSM disabled",
        ),
        (
            lambda config: setattr(config.cache_config, "is_attention_free", True),
            "attention model",
        ),
        (
            lambda config: setattr(config.device_config, "device_type", "cpu"),
            "pinned ROCm",
        ),
        (
            lambda config: setattr(config.offload_config, "offload_backend", "uva"),
            "offloading with every backend disabled",
        ),
        (
            lambda config: setattr(config.offload_config.uva, "cpu_offload_gb", 1),
            "UVA model offloading disabled",
        ),
        (
            lambda config: setattr(config.offload_config.uva, "cpu_offload_gb", 0.1),
            "UVA model offloading disabled",
        ),
        (
            lambda config: setattr(config.offload_config.uva, "cpu_offload_gb", False),
            "UVA model offloading disabled",
        ),
        (
            lambda config: config.offload_config.uva.cpu_offload_params.add("layers"),
            "no UVA-offloaded parameters",
        ),
        (
            lambda config: setattr(
                config.offload_config.prefetch, "offload_group_size", 2
            ),
            "prefetch model offloading disabled",
        ),
        (
            lambda config: setattr(
                config.offload_config.prefetch, "offload_num_in_group", 2
            ),
            "default prefetch offload group count",
        ),
        (
            lambda config: setattr(
                config.offload_config.prefetch, "offload_prefetch_step", 2
            ),
            "default prefetch lookahead",
        ),
        (
            lambda config: config.offload_config.prefetch.offload_params.add("layers"),
            "no prefetch-offloaded parameters",
        ),
        (
            lambda config: setattr(
                config.attention_config,
                "backend",
                AttentionBackendEnum.ROCM_ATTN,
            ),
            "automatic backend or exact TRITON_ATTN backend",
        ),
        (
            lambda config: setattr(config.attention_config, "backend", "TRITON_ATTN"),
            "automatic backend or exact TRITON_ATTN backend",
        ),
        (
            lambda config: setattr(
                config.attention_config, "backend_per_kind", {"full_attention": "x"}
            ),
            "no per-kind attention backend overrides",
        ),
        (
            lambda config: setattr(
                config.attention_config, "use_prefill_decode_attention", True
            ),
            "unified prefill/decode attention",
        ),
        (
            lambda config: setattr(config.compilation_config, "mode", 1),
            "compilation mode NONE",
        ),
        (
            lambda config: setattr(config.compilation_config, "cudagraph_mode", 1),
            "CUDA graph mode NONE",
        ),
        (
            lambda config: setattr(config, "diffusion_config", object()),
            "diffusion runtime disabled",
        ),
        (
            lambda config: setattr(config, "quant_config", object()),
            "resolved quantization disabled",
        ),
        (
            lambda config: setattr(config, "kv_events_config", object()),
            "KV event publishing disabled",
        ),
        (
            lambda config: setattr(config, "ec_transfer_config", object()),
            "encoder-cache transfer disabled",
        ),
        (
            lambda config: setattr(config, "weight_transfer_config", object()),
            "weight transfer disabled",
        ),
        (
            lambda config: setattr(config, "reasoning_config", object()),
            "reasoning parser runtime disabled",
        ),
    ],
)
def test_serial_pic_v1_rejects_cache_and_runtime_bypasses(
    mutation: Callable[[SimpleNamespace], None], requirement: str
) -> None:
    config = _serial_pic_v1_vllm_config()
    mutation(config)

    with pytest.raises(ValueError, match=requirement):
        validate_kmesh_target_profile(config)


@pytest.mark.parametrize(
    ("mode", "enabled"),
    [
        ("off", False),
        ("observe", True),
        ("collective_lockstep", True),
        ("collective_batched", True),
        ("master_mirror_reference", True),
    ],
)
def test_serial_pic_v1_target_profile_rejects_nonserial_kmesh_modes(
    mode: str, enabled: bool
) -> None:
    config = _serial_pic_v1_vllm_config()
    kmesh = config.kv_transfer_config.kv_connector_extra_config["connectors"][0][
        "kv_connector_extra_config"
    ]["kmesh"]
    kmesh["mode"] = mode
    kmesh["enabled"] = enabled

    with pytest.raises(ValueError, match="kmesh.(enabled|mode)"):
        validate_kmesh_target_profile(config)


def test_serial_pic_v1_target_profile_rejects_noop_pic_backend() -> None:
    config = _serial_pic_v1_vllm_config()
    kmesh = config.kv_transfer_config.kv_connector_extra_config["connectors"][0][
        "kv_connector_extra_config"
    ]["kmesh"]
    kmesh["pic_backend"] = "noop"

    with pytest.raises(ValueError, match="pic_backend='lmcache'"):
        validate_kmesh_target_profile(config)


@pytest.mark.parametrize(
    ("mutation", "requirement"),
    [
        (
            lambda config: setattr(config.model_config, "model", "/tmp/other"),
            "model path",
        ),
        (
            lambda config: setattr(config.model_config, "tokenizer", "/tmp/other"),
            "tokenizer path",
        ),
        (
            lambda config: setattr(
                config.model_config, "served_model_name", "other-name"
            ),
            "served model identity",
        ),
        (
            lambda config: setattr(config.model_config, "model_weights", "remote"),
            "model_weights unset",
        ),
        (
            lambda config: setattr(config.model_config, "model_impl", "transformers"),
            "model_impl='auto'",
        ),
        (
            lambda config: setattr(
                config.model_config, "override_attention_dtype", "float16"
            ),
            "attention dtype override unset",
        ),
        (
            lambda config: setattr(config.model_config, "seed", 0),
            "model seed=20260731",
        ),
        (
            lambda config: setattr(config.load_config, "load_format", "dummy"),
            "load_format='auto'",
        ),
        (
            lambda config: setattr(
                config.load_config, "model_loader_extra_config", {"x": 1}
            ),
            "empty model_loader_extra_config",
        ),
        (
            lambda config: setattr(
                config.scheduler_config, "scheduler_cls", "custom.Scheduler"
            ),
            "scheduler_cls unset",
        ),
        (
            lambda config: setattr(
                config.attention_config, "use_prefill_query_quantization", True
            ),
            "prefill query quantization disabled",
        ),
        (
            lambda config: setattr(config.cache_config, "gpu_memory_utilization", 0.9),
            "gpu_memory_utilization=0.3",
        ),
        (
            lambda config: setattr(config.kernel_config, "linear_backend", "torch"),
            "linear_backend='auto'",
        ),
        (
            lambda config: setattr(config, "use_v2_model_runner", False),
            "Model Runner V2 enabled",
        ),
        (
            lambda config: setattr(config, "additional_config", {"x": 1}),
            "empty additional_config",
        ),
        (
            lambda config: (
                config.kv_transfer_config.kv_connector_extra_config.__setitem__(
                    "unreviewed_outer", True
                )
            ),
            "only kmesh_target_profile and connectors",
        ),
        (
            lambda config: _set_child_extra(config, 0, "unreviewed_kmesh", True),
            "only kmesh in the KMesh child",
        ),
    ],
)
def test_serial_pic_v1_rejects_reviewed_v3_profile_bypasses(
    mutation: Callable[[SimpleNamespace], None], requirement: str
) -> None:
    config = _serial_pic_v1_vllm_config()
    mutation(config)

    with pytest.raises(ValueError, match=requirement):
        validate_kmesh_target_profile(config)


def test_serial_pic_v1_children_share_globals_but_own_kv_transfer_state() -> None:
    config = _serial_pic_v1_vllm_config()

    children = MultiConnector.inspect_connector_classes_and_configs(config)

    for _, child in children:
        assert child.model_config is config.model_config
        assert child.cache_config is config.cache_config
        assert child.parallel_config is config.parallel_config
        assert child.scheduler_config is config.scheduler_config
        assert child.kv_transfer_config is not config.kv_transfer_config
        assert (
            child.kv_transfer_config.kv_connector_extra_config
            is not config.kv_transfer_config.kv_connector_extra_config
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda config: setattr(
            config.model_config.hf_text_config, "rms_norm_eps", 0.01
        ),
        lambda config: setattr(config.model_config.hf_text_config, "head_dim", 128),
        lambda config: setattr(config.model_config.model_arch_config, "head_size", 128),
        lambda config: setattr(config.offload_config.prefetch, "offload_group_size", 2),
    ],
)
def test_serial_pic_v1_child_rejects_resolved_computation_drift(
    mutation: Callable[[SimpleNamespace], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExternalAdapter(_SerialPICExternalAdapter):
        pass

    _install_external_adapter(monkeypatch, ExternalAdapter)
    config = _serial_pic_v1_vllm_config()
    connector = MultiConnector(
        config,
        KVConnectorRole.SCHEDULER,
        SimpleNamespace(kv_cache_groups=[]),
    )
    monkeypatch.setattr(
        kmesh_config_module,
        "validate_kmesh_target_profile",
        lambda _config: KMESH_SERIAL_PIC_V1_PROFILE,
    )

    mutation(config)

    with pytest.raises(ValueError, match="configuration drifted"):
        connector.build_connector_meta(_empty_scheduler_output())


def test_serial_pic_v1_composite_arms_only_with_capable_lmcache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExternalAdapter(_SerialPICExternalAdapter):
        build_calls = 0

        def __init__(self, *args: object) -> None:
            self.init_args = args

        def prepare_request_lookup(self, *args: object) -> None:
            pass

        def validate_peer_connector_metadata(self, *args: object) -> None:
            pass

        def bind_peer_connector_metadata(self, *args: object) -> None:
            pass

        def clear_peer_connector_metadata(self) -> None:
            pass

        def build_connector_meta(self, *args: object) -> KVConnectorMetadata:
            type(self).build_calls += 1
            return KVConnectorMetadata()

    _install_external_adapter(monkeypatch, ExternalAdapter)
    config = _serial_pic_v1_vllm_config()

    connector = MultiConnector(
        config,
        KVConnectorRole.SCHEDULER,
        SimpleNamespace(kv_cache_groups=[]),
    )

    metadata = connector.build_connector_meta(_empty_scheduler_output())

    assert isinstance(metadata, MultiKVConnectorMetadata)
    assert len(metadata.metadata) == 2
    assert ExternalAdapter.build_calls == 1


def test_collective_lockstep_v1_constructs_and_arms_composite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExternalAdapter(_SerialPICExternalAdapter):
        build_calls = 0

        def build_connector_meta(self, *args: object) -> KVConnectorMetadata:
            type(self).build_calls += 1
            return KVConnectorMetadata()

    _install_external_adapter(monkeypatch, ExternalAdapter)
    config = _collective_lockstep_v1_vllm_config()

    connector = MultiConnector(
        config,
        KVConnectorRole.SCHEDULER,
        SimpleNamespace(kv_cache_groups=[]),
    )
    metadata = connector.build_connector_meta(_empty_scheduler_output())

    assert isinstance(metadata, MultiKVConnectorMetadata)
    kmesh_metadata = metadata.metadata[0]
    assert isinstance(kmesh_metadata, KMeshConnectorMetadata)
    assert kmesh_metadata.plan_batch is not None
    assert ExternalAdapter.build_calls == 1


def test_serial_pic_v1_revalidates_after_lmcache_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MutatingExternalAdapter(_SerialPICExternalAdapter):
        shutdown_calls = 0

        def __init__(self, vllm_config, *args: object) -> None:
            vllm_config.parallel_config.tensor_parallel_size = 7

        def prepare_request_lookup(self, *args: object) -> None:
            pass

        def validate_peer_connector_metadata(self, *args: object) -> None:
            pass

        def bind_peer_connector_metadata(self, *args: object) -> None:
            pass

        def clear_peer_connector_metadata(self) -> None:
            pass

        def shutdown(self) -> None:
            type(self).shutdown_calls += 1

    _install_external_adapter(monkeypatch, MutatingExternalAdapter)
    arm = MagicMock()
    monkeypatch.setattr(KMeshConnector, "arm_serial_pic_v1", arm)
    config = _serial_pic_v1_vllm_config()

    with pytest.raises(ValueError, match="TP=1"):
        MultiConnector(
            config,
            KVConnectorRole.SCHEDULER,
            SimpleNamespace(kv_cache_groups=[]),
        )

    arm.assert_not_called()
    assert MutatingExternalAdapter.shutdown_calls == 1


def test_serial_pic_v1_rejects_child_global_config_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ReplacingExternalAdapter(_SerialPICExternalAdapter):
        def __init__(self, vllm_config, *args: object) -> None:
            vllm_config.parallel_config = copy.copy(vllm_config.parallel_config)

        def prepare_request_lookup(self, *args: object) -> None:
            pass

        def validate_peer_connector_metadata(self, *args: object) -> None:
            pass

        def bind_peer_connector_metadata(self, *args: object) -> None:
            pass

        def clear_peer_connector_metadata(self) -> None:
            pass

    _install_external_adapter(monkeypatch, ReplacingExternalAdapter)
    arm = MagicMock()
    monkeypatch.setattr(KMeshConnector, "arm_serial_pic_v1", arm)
    config = _serial_pic_v1_vllm_config()

    with pytest.raises(ValueError, match="share read-only engine-global"):
        MultiConnector(
            config,
            KVConnectorRole.SCHEDULER,
            SimpleNamespace(kv_cache_groups=[]),
        )

    arm.assert_not_called()


def test_serial_pic_v1_peer_preflight_preserves_same_generation_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExternalAdapter(_SerialPICExternalAdapter):
        reject_peer_metadata = True
        bind_calls = 0

        def __init__(self, *args: object) -> None:
            pass

        def prepare_request_lookup(self, *args: object) -> None:
            pass

        def validate_peer_connector_metadata(self, *args: object) -> None:
            if self.reject_peer_metadata:
                raise ValueError("peer metadata rejected")

        def bind_peer_connector_metadata(self, *args: object) -> None:
            type(self).bind_calls += 1

        def clear_peer_connector_metadata(self) -> None:
            pass

        def build_connector_meta(self, *args: object) -> KVConnectorMetadata:
            return KVConnectorMetadata()

    _install_external_adapter(monkeypatch, ExternalAdapter)
    config = _serial_pic_v1_vllm_config()
    kv_cache_config = SimpleNamespace(kv_cache_groups=[])
    scheduler = MultiConnector(
        config,
        KVConnectorRole.SCHEDULER,
        kv_cache_config,
    )
    metadata = scheduler.build_connector_meta(_empty_scheduler_output())
    worker = MultiConnector(config, KVConnectorRole.WORKER, kv_cache_config)

    with pytest.raises(ValueError, match="peer metadata rejected"):
        worker.bind_connector_metadata(metadata)

    assert not worker.has_connector_metadata()
    assert ExternalAdapter.bind_calls == 0

    ExternalAdapter.reject_peer_metadata = False
    worker.bind_connector_metadata(metadata)

    assert worker.has_connector_metadata()
    assert ExternalAdapter.bind_calls == 1


@pytest.mark.parametrize("drift_scope", ["class", "instance"])
def test_serial_pic_v1_revalidates_adapter_protocol_before_bind(
    monkeypatch: pytest.MonkeyPatch, drift_scope: str
) -> None:
    class ExternalAdapter(_SerialPICExternalAdapter):
        instances: list[ExternalAdapter] = []
        bind_calls = 0

        def __init__(self, *args: object) -> None:
            type(self).instances.append(self)

        def prepare_request_lookup(self, *args: object) -> None:
            pass

        def validate_peer_connector_metadata(self, *args: object) -> None:
            pass

        def bind_peer_connector_metadata(self, *args: object) -> None:
            type(self).bind_calls += 1

        def clear_peer_connector_metadata(self) -> None:
            pass

        def build_connector_meta(self, *args: object) -> KVConnectorMetadata:
            return KVConnectorMetadata()

    _install_external_adapter(monkeypatch, ExternalAdapter)
    config = _serial_pic_v1_vllm_config()
    kv_cache_config = SimpleNamespace(kv_cache_groups=[])
    scheduler = MultiConnector(config, KVConnectorRole.SCHEDULER, kv_cache_config)
    metadata = scheduler.build_connector_meta(_empty_scheduler_output())
    worker = MultiConnector(config, KVConnectorRole.WORKER, kv_cache_config)
    worker_adapter = ExternalAdapter.instances[-1]
    if drift_scope == "class":
        ExternalAdapter.KMESH_HANDOFF_PROTOCOL_VERSION = 3
    else:
        worker_adapter.KMESH_HANDOFF_PROTOCOL_VERSION = 3

    with pytest.raises(ValueError, match="handoff protocol version 2"):
        worker.bind_connector_metadata(metadata)

    assert not worker.has_connector_metadata()
    assert ExternalAdapter.bind_calls == 0
    if drift_scope == "class":
        ExternalAdapter.KMESH_HANDOFF_PROTOCOL_VERSION = 2
    else:
        del worker_adapter.KMESH_HANDOFF_PROTOCOL_VERSION
    worker.bind_connector_metadata(metadata)
    assert worker.has_connector_metadata()


def test_serial_pic_v1_revalidates_frozen_lmcache_child_before_bind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExternalAdapter(_SerialPICExternalAdapter):
        child_configs: list[SimpleNamespace] = []
        bind_calls = 0

        def __init__(self, vllm_config, *args: object) -> None:
            type(self).child_configs.append(vllm_config)

        def prepare_request_lookup(self, *args: object) -> None:
            pass

        def validate_peer_connector_metadata(self, *args: object) -> None:
            pass

        def bind_peer_connector_metadata(self, *args: object) -> None:
            type(self).bind_calls += 1

        def clear_peer_connector_metadata(self) -> None:
            pass

        def build_connector_meta(self, *args: object) -> KVConnectorMetadata:
            return KVConnectorMetadata()

    _install_external_adapter(monkeypatch, ExternalAdapter)
    config = _serial_pic_v1_vllm_config()
    kv_cache_config = SimpleNamespace(kv_cache_groups=[])
    scheduler = MultiConnector(config, KVConnectorRole.SCHEDULER, kv_cache_config)
    metadata = scheduler.build_connector_meta(_empty_scheduler_output())
    worker = MultiConnector(config, KVConnectorRole.WORKER, kv_cache_config)
    worker_config = ExternalAdapter.child_configs[-1]
    child_extra = worker_config.kv_transfer_config.kv_connector_extra_config
    child_extra["use_native"] = True

    with pytest.raises(ValueError, match="configuration drifted"):
        worker.bind_connector_metadata(metadata)

    assert not worker.has_connector_metadata()
    assert ExternalAdapter.bind_calls == 0
    child_extra["use_native"] = False
    worker.bind_connector_metadata(metadata)
    assert worker.has_connector_metadata()


def test_serial_pic_v1_protocol_drift_clear_is_exhaustive_and_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExternalAdapter(_SerialPICExternalAdapter):
        instances: list[ExternalAdapter] = []
        clear_calls = 0

        def __init__(self, *args: object) -> None:
            type(self).instances.append(self)

        def prepare_request_lookup(self, *args: object) -> None:
            pass

        def validate_peer_connector_metadata(self, *args: object) -> None:
            pass

        def bind_peer_connector_metadata(self, *args: object) -> None:
            pass

        def clear_peer_connector_metadata(self) -> None:
            type(self).clear_calls += 1

        def build_connector_meta(self, *args: object) -> KVConnectorMetadata:
            return KVConnectorMetadata()

    _install_external_adapter(monkeypatch, ExternalAdapter)
    config = _serial_pic_v1_vllm_config()
    kv_cache_config = SimpleNamespace(kv_cache_groups=[])
    scheduler = MultiConnector(config, KVConnectorRole.SCHEDULER, kv_cache_config)
    metadata = scheduler.build_connector_meta(_empty_scheduler_output())
    worker = MultiConnector(config, KVConnectorRole.WORKER, kv_cache_config)
    worker.bind_connector_metadata(metadata)
    worker_adapter = ExternalAdapter.instances[-1]
    worker_adapter.KMESH_HANDOFF_PROTOCOL_VERSION = 3

    with pytest.raises(RuntimeError, match="metadata cleanup failed"):
        worker.clear_connector_metadata()

    assert not worker.has_connector_metadata()
    assert ExternalAdapter.clear_calls == 0

    del worker_adapter.KMESH_HANDOFF_PROTOCOL_VERSION
    worker.clear_connector_metadata()
    assert ExternalAdapter.clear_calls == 1


def test_serial_pic_v1_protocol_drift_abort_remains_visible_after_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _TransactionalSerialPICAdapter.reset()
    _install_external_adapter(monkeypatch, _TransactionalSerialPICAdapter)
    config = _serial_pic_v1_vllm_config()
    kv_cache_config = SimpleNamespace(kv_cache_groups=[])
    scheduler = MultiConnector(config, KVConnectorRole.SCHEDULER, kv_cache_config)
    metadata = scheduler.build_connector_meta(_empty_scheduler_output())
    worker = MultiConnector(config, KVConnectorRole.WORKER, kv_cache_config)
    worker_adapter = _TransactionalSerialPICAdapter.instances[-1]

    worker.bind_connector_metadata(metadata)
    worker.clear_connector_metadata()
    worker_adapter.KMESH_HANDOFF_PROTOCOL_VERSION = 3

    with pytest.raises(RuntimeError, match="metadata abort failed"):
        worker.abort_connector_metadata_transaction()

    assert worker_adapter.last_generation == 0
    assert worker_adapter.abort_calls == 1
    with pytest.raises(RuntimeError, match="metadata abort failed"):
        worker.abort_connector_metadata_transaction()
    assert worker_adapter.abort_calls == 1

    del worker_adapter.KMESH_HANDOFF_PROTOCOL_VERSION
    worker.abort_connector_metadata_transaction()
    worker.bind_connector_metadata(metadata)
    assert worker.has_connector_metadata()


def test_serial_pic_v1_commit_interrupt_restores_same_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _TransactionalSerialPICAdapter.reset()
    _install_external_adapter(monkeypatch, _TransactionalSerialPICAdapter)
    config = _serial_pic_v1_vllm_config()
    kv_cache_config = SimpleNamespace(kv_cache_groups=[])
    scheduler = MultiConnector(config, KVConnectorRole.SCHEDULER, kv_cache_config)
    metadata = scheduler.build_connector_meta(_empty_scheduler_output())
    worker = MultiConnector(config, KVConnectorRole.WORKER, kv_cache_config)
    worker_adapter = _TransactionalSerialPICAdapter.instances[-1]
    kmesh_metadata = metadata.metadata[0]
    assert isinstance(kmesh_metadata, KMeshConnectorMetadata)
    real_commit = KMeshConnector.commit_connector_metadata_transaction
    interrupt_once = True

    def commit_then_interrupt(connector: KMeshConnector) -> None:
        nonlocal interrupt_once
        real_commit(connector)
        if interrupt_once:
            interrupt_once = False
            raise KeyboardInterrupt("after KMesh tentative commit")

    monkeypatch.setattr(
        KMeshConnector,
        "commit_connector_metadata_transaction",
        commit_then_interrupt,
    )

    with pytest.raises(KeyboardInterrupt, match="tentative commit"):
        worker.bind_connector_metadata(metadata)

    assert not worker.has_connector_metadata()
    assert worker_adapter.last_generation == 0
    assert worker_adapter.begin_calls == 1
    assert worker_adapter.commit_calls == 1
    assert worker_adapter.abort_calls == 1
    assert worker_adapter.finalize_calls == 0
    worker.bind_connector_metadata(metadata)
    assert worker.has_connector_metadata()
    assert worker_adapter.last_generation == kmesh_metadata.step_generation
    assert worker_adapter.begin_calls == 2
    assert worker_adapter.commit_calls == 2
    assert worker_adapter.abort_calls == 1
    assert worker_adapter.finalize_calls == 1


@pytest.mark.parametrize("failure_scope", ["external", "kmesh"])
def test_serial_pic_v1_finalize_interrupt_restores_every_child(
    monkeypatch: pytest.MonkeyPatch,
    failure_scope: str,
) -> None:
    _TransactionalSerialPICAdapter.reset()
    _install_external_adapter(monkeypatch, _TransactionalSerialPICAdapter)
    config = _serial_pic_v1_vllm_config()
    kv_cache_config = SimpleNamespace(kv_cache_groups=[])
    scheduler = MultiConnector(config, KVConnectorRole.SCHEDULER, kv_cache_config)
    metadata = scheduler.build_connector_meta(_empty_scheduler_output())
    worker = MultiConnector(config, KVConnectorRole.WORKER, kv_cache_config)
    worker_adapter = _TransactionalSerialPICAdapter.instances[-1]
    kmesh_metadata = metadata.metadata[0]
    assert isinstance(kmesh_metadata, KMeshConnectorMetadata)

    if failure_scope == "external":
        worker_adapter.finalize_error = KeyboardInterrupt("after external finalize")
    else:
        real_finalize = KMeshConnector.finalize_connector_metadata_transaction
        interrupt_once = True

        def finalize_then_interrupt(connector: KMeshConnector) -> None:
            nonlocal interrupt_once
            real_finalize(connector)
            if interrupt_once:
                interrupt_once = False
                raise KeyboardInterrupt("after KMesh finalize")

        monkeypatch.setattr(
            KMeshConnector,
            "finalize_connector_metadata_transaction",
            finalize_then_interrupt,
        )

    with pytest.raises(KeyboardInterrupt, match="after .* finalize"):
        worker.bind_connector_metadata(metadata)

    assert not worker.has_connector_metadata()
    assert worker_adapter.last_generation == 0
    assert worker_adapter.begin_calls == 1
    assert worker_adapter.commit_calls == 1
    assert worker_adapter.finalize_calls == 1
    assert worker_adapter.abort_calls == 1

    worker.bind_connector_metadata(metadata)

    assert worker.has_connector_metadata()
    assert worker_adapter.last_generation == kmesh_metadata.step_generation
    assert worker_adapter.begin_calls == 2
    assert worker_adapter.commit_calls == 2
    assert worker_adapter.finalize_calls == 2
    assert worker_adapter.abort_calls == 1


@pytest.mark.parametrize("recovery_path", ["next_bind", "public_abort"])
@pytest.mark.parametrize("abort_after_restore", [False, True])
def test_serial_pic_v1_failed_abort_is_retried_before_peer_preflight(
    monkeypatch: pytest.MonkeyPatch,
    recovery_path: str,
    abort_after_restore: bool,
) -> None:
    _TransactionalSerialPICAdapter.reset()
    _install_external_adapter(monkeypatch, _TransactionalSerialPICAdapter)
    config = _serial_pic_v1_vllm_config()
    kv_cache_config = SimpleNamespace(kv_cache_groups=[])
    scheduler = MultiConnector(config, KVConnectorRole.SCHEDULER, kv_cache_config)
    metadata = scheduler.build_connector_meta(_empty_scheduler_output())
    worker = MultiConnector(config, KVConnectorRole.WORKER, kv_cache_config)
    worker_adapter = _TransactionalSerialPICAdapter.instances[-1]
    kmesh_metadata = metadata.metadata[0]
    assert isinstance(kmesh_metadata, KMeshConnectorMetadata)
    finalize_error = KeyboardInterrupt("finalize interrupted after publication")
    abort_error = KeyboardInterrupt("abort interrupted before rollback")
    worker_adapter.finalize_error = finalize_error
    worker_adapter.abort_error = abort_error
    worker_adapter.abort_error_after_restore = abort_after_restore

    with pytest.raises(KeyboardInterrupt) as exc_info:
        worker.bind_connector_metadata(metadata)

    assert exc_info.value is finalize_error
    assert exc_info.value.__cause__ is abort_error
    assert not worker.has_connector_metadata()
    expected_failed_generation = (
        0 if abort_after_restore else kmesh_metadata.step_generation
    )
    assert worker_adapter.last_generation == expected_failed_generation
    assert worker_adapter.abort_calls == 1

    if recovery_path == "public_abort":
        worker.abort_connector_metadata_transaction()
        assert worker_adapter.last_generation == 0
        assert worker_adapter.abort_calls == 2

    worker.bind_connector_metadata(metadata)

    assert worker.has_connector_metadata()
    assert worker_adapter.last_generation == kmesh_metadata.step_generation
    assert worker_adapter.begin_calls == 2
    assert worker_adapter.commit_calls == 2
    assert worker_adapter.finalize_calls == 2
    assert worker_adapter.abort_calls == 2


def test_serial_pic_v1_public_abort_restores_finalized_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _TransactionalSerialPICAdapter.reset()
    _install_external_adapter(monkeypatch, _TransactionalSerialPICAdapter)
    config = _serial_pic_v1_vllm_config()
    kv_cache_config = SimpleNamespace(kv_cache_groups=[])
    scheduler = MultiConnector(config, KVConnectorRole.SCHEDULER, kv_cache_config)
    metadata = scheduler.build_connector_meta(_empty_scheduler_output())
    worker = MultiConnector(config, KVConnectorRole.WORKER, kv_cache_config)
    worker_adapter = _TransactionalSerialPICAdapter.instances[-1]
    kmesh_metadata = metadata.metadata[0]
    assert isinstance(kmesh_metadata, KMeshConnectorMetadata)

    worker.bind_connector_metadata(metadata)
    worker.clear_connector_metadata()
    worker.abort_connector_metadata_transaction()

    assert worker_adapter.last_generation == 0
    assert worker_adapter.abort_calls == 1
    worker.bind_connector_metadata(metadata)
    assert worker.has_connector_metadata()
    assert worker_adapter.last_generation == kmesh_metadata.step_generation


def test_serial_pic_v1_new_preflight_disarms_finalized_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _TransactionalSerialPICAdapter.reset()
    _install_external_adapter(monkeypatch, _TransactionalSerialPICAdapter)
    config = _serial_pic_v1_vllm_config()
    kv_cache_config = SimpleNamespace(kv_cache_groups=[])
    scheduler = MultiConnector(config, KVConnectorRole.SCHEDULER, kv_cache_config)
    first_metadata = scheduler.build_connector_meta(_empty_scheduler_output())
    second_metadata = scheduler.build_connector_meta(_empty_scheduler_output())
    worker = MultiConnector(config, KVConnectorRole.WORKER, kv_cache_config)
    worker_adapter = _TransactionalSerialPICAdapter.instances[-1]
    first_kmesh_metadata = first_metadata.metadata[0]
    second_kmesh_metadata = second_metadata.metadata[0]
    assert isinstance(first_kmesh_metadata, KMeshConnectorMetadata)
    assert isinstance(second_kmesh_metadata, KMeshConnectorMetadata)

    worker.bind_connector_metadata(first_metadata)
    worker.clear_connector_metadata()
    with pytest.raises(ValueError, match="stale"):
        worker.bind_connector_metadata(first_metadata)
    worker.abort_connector_metadata_transaction()

    assert worker_adapter.last_generation == first_kmesh_metadata.step_generation
    assert worker_adapter.abort_calls == 0
    worker.bind_connector_metadata(second_metadata)
    assert worker_adapter.last_generation == second_kmesh_metadata.step_generation


def test_serial_pic_v1_drifted_commit_uses_begin_captured_abort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _TransactionalSerialPICAdapter.reset()
    _install_external_adapter(monkeypatch, _TransactionalSerialPICAdapter)
    config = _serial_pic_v1_vllm_config()
    kv_cache_config = SimpleNamespace(kv_cache_groups=[])
    scheduler = MultiConnector(config, KVConnectorRole.SCHEDULER, kv_cache_config)
    metadata = scheduler.build_connector_meta(_empty_scheduler_output())
    worker = MultiConnector(config, KVConnectorRole.WORKER, kv_cache_config)
    worker_adapter = _TransactionalSerialPICAdapter.instances[-1]
    worker_adapter.drift_protocol_on_commit = True
    worker_adapter.commit_error = KeyboardInterrupt("external commit interrupted")

    with pytest.raises(KeyboardInterrupt, match="external commit interrupted"):
        worker.bind_connector_metadata(metadata)

    assert not worker.has_connector_metadata()
    assert worker_adapter.last_generation == 0
    assert worker_adapter.abort_calls == 1
    assert worker_adapter.clear_calls == 0

    del worker_adapter.KMESH_HANDOFF_PROTOCOL_VERSION
    worker.bind_connector_metadata(metadata)

    assert worker.has_connector_metadata()
    assert worker_adapter.begin_calls == 2
    assert worker_adapter.commit_calls == 2
    assert worker_adapter.abort_calls == 1
    assert worker_adapter.finalize_calls == 1


def test_serial_pic_v1_next_begin_checkpoints_committed_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _TransactionalSerialPICAdapter.reset()
    _install_external_adapter(monkeypatch, _TransactionalSerialPICAdapter)
    config = _serial_pic_v1_vllm_config()
    kv_cache_config = SimpleNamespace(kv_cache_groups=[])
    scheduler = MultiConnector(config, KVConnectorRole.SCHEDULER, kv_cache_config)
    first_metadata = scheduler.build_connector_meta(_empty_scheduler_output())
    second_metadata = scheduler.build_connector_meta(_empty_scheduler_output())
    worker = MultiConnector(config, KVConnectorRole.WORKER, kv_cache_config)
    worker_adapter = _TransactionalSerialPICAdapter.instances[-1]
    first_kmesh_metadata = first_metadata.metadata[0]
    second_kmesh_metadata = second_metadata.metadata[0]
    assert isinstance(first_kmesh_metadata, KMeshConnectorMetadata)
    assert isinstance(second_kmesh_metadata, KMeshConnectorMetadata)

    worker.bind_connector_metadata(first_metadata)
    worker.clear_connector_metadata()
    assert worker_adapter.abort_calls == 0
    assert worker_adapter.last_generation == first_kmesh_metadata.step_generation

    worker_adapter.commit_error = KeyboardInterrupt("second commit interrupted")
    with pytest.raises(KeyboardInterrupt, match="second commit interrupted"):
        worker.bind_connector_metadata(second_metadata)

    assert worker_adapter.last_generation == first_kmesh_metadata.step_generation
    assert worker_adapter.abort_calls == 1

    worker.bind_connector_metadata(second_metadata)
    assert worker_adapter.last_generation == second_kmesh_metadata.step_generation


def test_serial_pic_v1_failed_next_begin_never_aborts_prior_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _TransactionalSerialPICAdapter.reset()
    _install_external_adapter(monkeypatch, _TransactionalSerialPICAdapter)
    config = _serial_pic_v1_vllm_config()
    kv_cache_config = SimpleNamespace(kv_cache_groups=[])
    scheduler = MultiConnector(config, KVConnectorRole.SCHEDULER, kv_cache_config)
    first_metadata = scheduler.build_connector_meta(_empty_scheduler_output())
    second_metadata = scheduler.build_connector_meta(_empty_scheduler_output())
    worker = MultiConnector(config, KVConnectorRole.WORKER, kv_cache_config)
    worker_adapter = _TransactionalSerialPICAdapter.instances[-1]
    first_kmesh_metadata = first_metadata.metadata[0]
    second_kmesh_metadata = second_metadata.metadata[0]
    assert isinstance(first_kmesh_metadata, KMeshConnectorMetadata)
    assert isinstance(second_kmesh_metadata, KMeshConnectorMetadata)

    worker.bind_connector_metadata(first_metadata)
    worker.clear_connector_metadata()
    real_begin = KMeshConnector.begin_connector_metadata_transaction
    drift_once = True

    def begin_then_drift(connector: KMeshConnector) -> None:
        nonlocal drift_once
        real_begin(connector)
        if drift_once:
            drift_once = False
            worker_adapter.KMESH_HANDOFF_PROTOCOL_VERSION = 3

    monkeypatch.setattr(
        KMeshConnector,
        "begin_connector_metadata_transaction",
        begin_then_drift,
    )

    with pytest.raises(ValueError, match="handoff protocol version 2"):
        worker.bind_connector_metadata(second_metadata)

    assert worker_adapter.last_generation == first_kmesh_metadata.step_generation
    assert worker_adapter.begin_calls == 1
    assert worker_adapter.abort_calls == 0

    del worker_adapter.KMESH_HANDOFF_PROTOCOL_VERSION
    worker.bind_connector_metadata(second_metadata)

    assert worker_adapter.last_generation == second_kmesh_metadata.step_generation
    assert worker_adapter.begin_calls == 2
    assert worker_adapter.abort_calls == 0
