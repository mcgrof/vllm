# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import filecmp
import shutil
import tempfile
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from tests.v1.kv_connector.unit.utils import create_vllm_config
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRequestSidecar,
    SupportsHMA,
    supports_hma,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
    MultiKVConnectorMetadata,
    MultiKVConnectorPromMetrics,
    MultiKVConnectorStats,
    MultiKVConnectorWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    NixlKVConnectorStats,
)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import KVConnectorOutput, KVConnectorWorkerMetadata

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

PROMPT_CONTEXT = "Hi " * 100
PROMPTS = [
    PROMPT_CONTEXT + "Hello, my name is",
    PROMPT_CONTEXT + "The capital of France is",
]

SAMPLING_PARAMS = SamplingParams(temperature=0, max_tokens=20)


# Test connector with custom stats for testing MultiConnector
class MockConnectorStats(KVConnectorStats):
    """Mock stats class for testing."""

    pass


class MockConnector(KVConnectorBase_V1):
    """Mock connector for testing."""

    def __new__(cls, *args, **kwargs):
        # mock all KVConnectorBase_V1 functions
        mock = MagicMock(spec_set=KVConnectorBase_V1)
        # Override just build_kv_connector_stats
        mock.build_kv_connector_stats = cls.build_kv_connector_stats
        mock.get_kv_connector_stats.return_value = None
        mock.export_request_sidecar.return_value = None
        return mock

    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> KVConnectorStats | None:
        return MockConnectorStats(data=data) if data is not None else None

    def start_load_kv(self, forward_context, **kwargs):
        pass

    def wait_for_layer_load(self, layer_name):
        pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        pass

    def wait_for_save(self):
        pass

    def build_connector_meta(self, scheduler_output):
        return None

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return (0, False)

    def update_state_after_alloc(self, request, blocks, num_tokens) -> None:
        pass


class MockHMAConnector(KVConnectorBase_V1, SupportsHMA):
    """Mock connector that supports HMA for testing."""

    def __new__(cls, *args, **kwargs):
        mock = MagicMock(spec_set=cls)
        mock.get_kv_connector_stats.return_value = None
        mock.export_request_sidecar.return_value = None
        return mock

    def start_load_kv(self, forward_context, **kwargs):
        pass

    def wait_for_layer_load(self, layer_name):
        pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        pass

    def wait_for_save(self):
        pass

    def build_connector_meta(self, scheduler_output):
        return None

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return (0, False)

    def update_state_after_alloc(self, request, blocks, num_tokens) -> None:
        pass

    def request_finished_all_groups(self, request, block_ids):
        return (False, None)


class MetadataProbeConnector(MockConnector):
    """Publicly observable connector used for lifecycle fault injection."""

    events: list[tuple[str, str]] = []
    failure: tuple[str, str] | None = None
    failure_error: BaseException | None = None
    failure_remaining = 0
    queued_failures: dict[tuple[str, str], list[BaseException]] = {}

    def __new__(cls, *args: object, **kwargs: object):
        return object.__new__(cls)

    def __init__(self, vllm_config, role, kv_cache_config) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        self.name = self._kv_transfer_config.kv_connector_extra_config["name"]
        type(self).events.append(("construct", self.name))

    @classmethod
    def reset_probe(cls) -> None:
        cls.events = []
        cls.failure = None
        cls.failure_error = None
        cls.failure_remaining = 0
        cls.queued_failures = {}

    def _record(self, phase: str) -> None:
        type(self).events.append((phase, self.name))
        queued_failures = type(self).queued_failures.get((phase, self.name))
        if queued_failures:
            raise queued_failures.pop(0)
        if (
            type(self).failure == (phase, self.name)
            and type(self).failure_remaining > 0
        ):
            type(self).failure_remaining -= 1
            error = type(self).failure_error
            assert error is not None
            raise error

    def begin_connector_metadata_transaction(self) -> None:
        self._record("begin")

    def commit_connector_metadata_transaction(self) -> None:
        self._record("commit")

    def finalize_connector_metadata_transaction(self) -> None:
        self._record("finalize")

    def abort_connector_metadata_transaction(self) -> None:
        self._record("abort")

    def validate_connector_metadata(
        self, connector_metadata: KVConnectorMetadata
    ) -> None:
        self._record("validate-own")

    def validate_peer_connector_metadata(
        self, peer_connector_metadata: tuple[KVConnectorMetadata, ...]
    ) -> None:
        self._record("validate-peer")

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        self._record("bind-own")
        super().bind_connector_metadata(connector_metadata)

    def bind_peer_connector_metadata(
        self, peer_connector_metadata: tuple[KVConnectorMetadata, ...]
    ) -> None:
        self._record("bind-peer")

    def clear_peer_connector_metadata(self) -> None:
        self._record("clear-peer")

    def clear_connector_metadata(self) -> None:
        self._record("clear-own")
        super().clear_connector_metadata()

    def shutdown(self) -> None:
        self._record("shutdown")


class ConstructionProbeConnector(MetadataProbeConnector):
    """Connector that fails construction and one owned shutdown attempt."""

    construction_error = RuntimeError("third child construction failed")
    cleanup_error = KeyboardInterrupt("second child shutdown interrupted")

    def __init__(self, vllm_config, role, kv_cache_config) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        if self.name == "third":
            raise type(self).construction_error

    def shutdown(self) -> None:
        type(self).events.append(("shutdown", self.name))
        if self.name == "second" and type(self).failure_remaining > 0:
            type(self).failure_remaining -= 1
            raise type(self).cleanup_error


# Register mock connectors
KVConnectorFactory.register_connector("MockConnector", __name__, MockConnector.__name__)
KVConnectorFactory.register_connector(
    "MockHMAConnector", __name__, MockHMAConnector.__name__
)


@pytest.fixture
def mc() -> MultiConnector:
    """MultiConnector using two mocked connectors"""
    mock_connector_config = {
        "kv_connector": "MockConnector",
        "kv_role": "kv_both",
        "kv_connector_module_path": "tests.v1.kv_connector.unit.test_multi_connector",
    }

    vllm_config = create_vllm_config(
        kv_connector="MultiConnector",
        kv_connector_extra_config={
            "connectors": [mock_connector_config, mock_connector_config],
        },
    )

    kv_cache_config = KVCacheConfig(
        num_blocks=0, kv_cache_tensors=[], kv_cache_groups=[]
    )

    mc = MultiConnector(
        vllm_config=vllm_config,
        role=KVConnectorRole.WORKER,
        kv_cache_config=kv_cache_config,
    )

    return mc


def _probe_multi_connector(
    connector_name: str = "MetadataProbeConnector",
    names: tuple[str, ...] = ("first", "second"),
) -> MultiConnector:
    connectors = [
        {
            "kv_connector": connector_name,
            "kv_role": "kv_both",
            "kv_connector_module_path": __name__,
            "kv_connector_extra_config": {"name": name},
        }
        for name in names
    ]
    config = create_vllm_config(
        kv_connector="MultiConnector",
        kv_connector_extra_config={"connectors": connectors},
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=0, kv_cache_tensors=[], kv_cache_groups=[]
    )
    return MultiConnector(config, KVConnectorRole.WORKER, kv_cache_config)


@pytest.mark.skip_global_cleanup
def test_register_model_is_forwarded_to_every_child(mc: MultiConnector) -> None:
    model = MagicMock(spec=torch.nn.Module)

    mc.register_model(model)

    for connector in mc._connectors:
        connector.register_model.assert_called_once_with(model)


@pytest.mark.skip_global_cleanup
def test_request_sidecar_is_immutable_and_tensor_free() -> None:
    sidecar = KVConnectorRequestSidecar("provider", ("value", (1, True, None)))

    with pytest.raises(FrozenInstanceError):
        sidecar.payload = "changed"  # type: ignore[misc]

    with pytest.raises(TypeError, match="scalar primitives and tuples"):
        KVConnectorRequestSidecar("provider", torch.empty(0))  # type: ignore[arg-type]


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("provider_index", [0, 1])
def test_request_sidecars_are_delivered_before_lookup_independent_of_order(
    mc: MultiConnector, provider_index: int
) -> None:
    request = MagicMock()
    request.request_id = "request"
    sidecar = KVConnectorRequestSidecar("provider", ("descriptor", 7))
    events: list[tuple[str, int]] = []

    for i, connector in enumerate(mc._connectors):

        def export_sidecar(request, *, index=i):
            events.append(("export_sidecar", index))
            return sidecar if index == provider_index else None

        def prepare_lookup(request, peer_sidecars, *, index=i):
            events.append(("prepare_lookup", index))
            assert peer_sidecars == (sidecar,)

        def get_matches(request, num_computed_tokens, *, index=i):
            events.append(("lookup", index))
            return (7, False) if index != provider_index else (0, False)

        connector.export_request_sidecar.side_effect = export_sidecar
        connector.prepare_request_lookup.side_effect = prepare_lookup
        connector.get_num_new_matched_tokens.side_effect = get_matches

    assert mc.get_num_new_matched_tokens(request, 0) == (7, False)
    assert events == [
        ("export_sidecar", 0),
        ("export_sidecar", 1),
        ("prepare_lookup", 0),
        ("prepare_lookup", 1),
        ("lookup", 0),
        ("lookup", 1),
    ]


@pytest.mark.skip_global_cleanup
def test_request_sidecar_hooks_preserve_noop_connector_behavior(
    mc: MultiConnector,
) -> None:
    request = MagicMock()
    request.request_id = "request"
    mc._connectors[0].get_num_new_matched_tokens.return_value = (3, False)
    mc._connectors[1].get_num_new_matched_tokens.return_value = (5, True)

    assert mc.get_num_new_matched_tokens(request, 0) == (3, False)
    for connector in mc._connectors:
        connector.prepare_request_lookup.assert_called_once_with(request, ())


@pytest.mark.skip_global_cleanup
def test_duplicate_request_sidecar_providers_are_rejected(mc: MultiConnector) -> None:
    request = MagicMock()
    request.request_id = "request"
    for i, connector in enumerate(mc._connectors):
        connector.export_request_sidecar.return_value = KVConnectorRequestSidecar(
            "duplicate", i
        )

    with pytest.raises(
        ValueError,
        match=(
            "Duplicate request sidecar provider 'duplicate' at connector "
            "indices 0 and 1"
        ),
    ):
        mc.get_num_new_matched_tokens(request, 0)

    for connector in mc._connectors:
        connector.prepare_request_lookup.assert_not_called()
        connector.get_num_new_matched_tokens.assert_not_called()


@pytest.mark.skip_global_cleanup
def test_peer_metadata_is_bound_after_own_metadata_and_cleared_with_it(
    mc: MultiConnector,
) -> None:
    metadata = (KVConnectorMetadata(), KVConnectorMetadata())
    multi_metadata = MultiKVConnectorMetadata(metadata)
    events: list[tuple[str, int]] = []
    own_metadata_bound = [False, False]

    for i, connector in enumerate(mc._connectors):

        def bind_own(connector_metadata, *, index=i):
            events.append(("bind_own", index))
            assert connector_metadata is metadata[index]
            own_metadata_bound[index] = True

        def bind_peers(peer_metadata, *, index=i):
            events.append(("bind_peers", index))
            assert all(own_metadata_bound)
            assert peer_metadata == metadata

        def clear_peers(*, index=i):
            events.append(("clear_peers", index))

        def clear_own(*, index=i):
            events.append(("clear_own", index))

        connector.bind_connector_metadata.side_effect = bind_own
        connector.bind_peer_connector_metadata.side_effect = bind_peers
        connector.clear_peer_connector_metadata.side_effect = clear_peers
        connector.clear_connector_metadata.side_effect = clear_own

    mc.bind_connector_metadata(multi_metadata)
    assert mc.has_connector_metadata()
    assert events == [
        ("bind_own", 0),
        ("bind_own", 1),
        ("bind_peers", 0),
        ("bind_peers", 1),
    ]

    mc.clear_connector_metadata()
    assert not mc.has_connector_metadata()
    assert events[4:] == [
        ("clear_peers", 0),
        ("clear_peers", 1),
        ("clear_own", 0),
        ("clear_own", 1),
    ]


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("phase", ["own", "peer"])
@pytest.mark.parametrize("failure_index", [0, 1])
def test_metadata_bind_failure_rolls_back_every_child(
    phase: str, failure_index: int
) -> None:
    MetadataProbeConnector.reset_probe()
    connector = _probe_multi_connector()
    metadata = (KVConnectorMetadata(), KVConnectorMetadata())
    multi_metadata = MultiKVConnectorMetadata(metadata)
    bind_error = RuntimeError(f"{phase} bind {failure_index} failed")
    failure_name = ("first", "second")[failure_index]
    MetadataProbeConnector.failure = (f"bind-{phase}", failure_name)
    MetadataProbeConnector.failure_error = bind_error
    MetadataProbeConnector.failure_remaining = 1

    with pytest.raises(RuntimeError) as exc_info:
        connector.bind_connector_metadata(multi_metadata)

    assert exc_info.value is bind_error
    assert not connector.has_connector_metadata()
    for name in ("first", "second"):
        assert ("clear-peer", name) in MetadataProbeConnector.events
        assert ("clear-own", name) in MetadataProbeConnector.events
        assert ("abort", name) in MetadataProbeConnector.events


@pytest.mark.skip_global_cleanup
def test_metadata_abort_failure_retains_only_failed_children() -> None:
    MetadataProbeConnector.reset_probe()
    connector = _probe_multi_connector()
    metadata = MultiKVConnectorMetadata((KVConnectorMetadata(), KVConnectorMetadata()))
    bind_error = RuntimeError("peer bind interrupted")
    first_abort_error = SystemExit("first abort interrupted")
    second_abort_error = KeyboardInterrupt("second abort interrupted")
    retry_abort_error = KeyboardInterrupt("second abort retry interrupted")
    MetadataProbeConnector.queued_failures = {
        ("bind-peer", "second"): [bind_error],
        ("abort", "first"): [first_abort_error],
        ("abort", "second"): [second_abort_error, retry_abort_error],
    }

    with pytest.raises(RuntimeError) as exc_info:
        connector.bind_connector_metadata(metadata)

    assert exc_info.value is bind_error
    assert not connector.has_connector_metadata()
    assert MetadataProbeConnector.events.count(("abort", "second")) == 1
    assert MetadataProbeConnector.events.count(("abort", "first")) == 1

    MetadataProbeConnector.events = []
    with pytest.raises(KeyboardInterrupt) as exc_info:
        connector.abort_connector_metadata_transaction()

    assert exc_info.value is retry_abort_error
    assert MetadataProbeConnector.events == [
        ("abort", "second"),
        ("abort", "first"),
    ]

    MetadataProbeConnector.events = []
    connector.abort_connector_metadata_transaction()
    assert MetadataProbeConnector.events == [("abort", "second")]

    MetadataProbeConnector.events = []
    connector.abort_connector_metadata_transaction()
    assert MetadataProbeConnector.events == []


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("phase", ["peer", "own"])
@pytest.mark.parametrize("failure_index", [0, 1])
def test_metadata_clear_attempts_every_cleanup_and_is_retryable(
    phase: str, failure_index: int
) -> None:
    MetadataProbeConnector.reset_probe()
    connector = _probe_multi_connector()
    metadata = (KVConnectorMetadata(), KVConnectorMetadata())
    multi_metadata = MultiKVConnectorMetadata(metadata)
    cleanup_error: BaseException
    if phase == "peer" and failure_index == 0:
        cleanup_error = KeyboardInterrupt("peer cleanup interrupted")
    else:
        cleanup_error = RuntimeError(f"{phase} clear {failure_index} failed")
    failure_name = ("first", "second")[failure_index]

    connector.bind_connector_metadata(multi_metadata)
    MetadataProbeConnector.events = []
    MetadataProbeConnector.failure = (f"clear-{phase}", failure_name)
    MetadataProbeConnector.failure_error = cleanup_error
    MetadataProbeConnector.failure_remaining = 1
    with pytest.raises(type(cleanup_error)) as exc_info:
        connector.clear_connector_metadata()

    assert exc_info.value is cleanup_error
    assert not connector.has_connector_metadata()
    for name in ("first", "second"):
        assert MetadataProbeConnector.events.count(("clear-peer", name)) == 1
        assert MetadataProbeConnector.events.count(("clear-own", name)) == 1

    connector.clear_connector_metadata()
    for name in ("first", "second"):
        assert MetadataProbeConnector.events.count(("clear-peer", name)) == 2
        assert MetadataProbeConnector.events.count(("clear-own", name)) == 2


@pytest.mark.skip_global_cleanup
def test_construction_failure_shuts_down_created_children_in_reverse() -> None:
    ConstructionProbeConnector.reset_probe()
    ConstructionProbeConnector.failure_remaining = 1
    pending_before = MultiConnector.pending_construction_cleanup_count()

    with pytest.raises(RuntimeError) as exc_info:
        _probe_multi_connector(
            "ConstructionProbeConnector", ("first", "second", "third")
        )

    assert exc_info.value is ConstructionProbeConnector.construction_error
    assert exc_info.value.__cause__ is ConstructionProbeConnector.cleanup_error
    assert ConstructionProbeConnector.events == [
        ("construct", "first"),
        ("construct", "second"),
        ("construct", "third"),
        ("shutdown", "second"),
        ("shutdown", "first"),
    ]
    assert MultiConnector.pending_construction_cleanup_count() == pending_before + 1

    MultiConnector.retry_failed_construction_cleanup()

    assert ConstructionProbeConnector.events[-1] == ("shutdown", "second")
    assert MultiConnector.pending_construction_cleanup_count() == pending_before


# Helper function to compare directories recursively
def _compare_directories(dir1: Path, dir2: Path) -> bool:
    """Compares two directories recursively for identical content."""
    dcmp = filecmp.dircmp(dir1, dir2)
    if dcmp.left_only or dcmp.right_only or dcmp.diff_files:
        print(f"Differences found between {dir1} and {dir2}:")
        print(f"  Left only: {dcmp.left_only}")
        print(f"  Right only: {dcmp.right_only}")
        print(f"  Different files: {dcmp.diff_files}")
        return False
    for sub_dir in dcmp.common_dirs:
        if not _compare_directories(dir1 / sub_dir, dir2 / sub_dir):
            return False
    return True


def test_multi_example_connector_consistency():
    """
    Tests that MultiConnector with two ExampleConnectors saves
    identical KV cache data to separate storage locations.
    """
    storage_1_path = Path("storage_1/")
    storage_2_path = Path("storage_2/")
    shutil.rmtree(storage_1_path, ignore_errors=True)
    shutil.rmtree(storage_2_path, ignore_errors=True)
    storage_1_path.mkdir()
    storage_2_path.mkdir()

    # Configure MultiConnector with two ExampleConnectors
    kv_transfer_config = KVTransferConfig(
        kv_connector="MultiConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "connectors": [
                {
                    "kv_connector": "TestExampleConnector",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {
                        "shared_storage_path": str(storage_1_path),
                        "name": "storage1",
                    },
                    "kv_connector_module_path": "tests.v1.kv_connector.unit.utils",
                },
                {
                    "kv_connector": "TestExampleConnector",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {
                        "shared_storage_path": str(storage_2_path),
                        "name": "storage2",
                    },
                    "kv_connector_module_path": "tests.v1.kv_connector.unit.utils",
                },
            ]
        },
    )

    llm = LLM(
        model=MODEL_NAME,
        enforce_eager=True,
        gpu_memory_utilization=0.5,
        kv_transfer_config=kv_transfer_config,
        async_scheduling=False,
    )
    # Run generation - this should trigger saving KV cache
    # Use a single prompt to avoid race conditions depending on the order of scheduling
    _ = llm.generate(PROMPTS[0], SAMPLING_PARAMS)

    # --- Verification ---

    # Check that both storage directories were populated
    local_subdirs = list(storage_1_path.iterdir())
    external_subdirs = list(storage_2_path.iterdir())

    assert len(local_subdirs) > 0, (
        f"Local storage path {storage_1_path} is empty after generation."
    )
    assert len(external_subdirs) > 0, (
        f"External storage path {storage_2_path} is empty after generation."
    )
    assert len(local_subdirs) == len(external_subdirs), (
        f"Mismatch in number of cache entries: "
        f"Local={len(local_subdirs)}, External={len(external_subdirs)}"
    )

    # The subdirectories should correspond to the prompt hashes
    # Since prompts are the same, the hash directories should be the same name
    local_subdir_names = sorted([d.name for d in local_subdirs])
    external_subdir_names = sorted([d.name for d in external_subdirs])
    assert local_subdir_names == external_subdir_names, (
        "Cache directory names do not match between local and external storage"
    )

    # Compare the contents of each corresponding cache directory
    for subdir_name in local_subdir_names:
        print(f"Comparing contents of cache directory: {subdir_name}")
        assert _compare_directories(
            storage_1_path / subdir_name, storage_2_path / subdir_name
        ), (
            f"Contents differ for cache directory '{subdir_name}' between "
            f"{storage_1_path} and {storage_2_path}"
        )

    events = get_connector_events()
    storage1_scheduler_events = _ignore_event_collection(events["storage1-SCHEDULER"])
    storage2_scheduler_events = _ignore_event_collection(events["storage2-SCHEDULER"])
    # First event is bind_gpu_block_pool from initialization, then
    # set_xfer_handshake_metadata_pp_aware, then on_new_request when the request is
    # enqueued, then get_num_new_matched_tokens and update_state_after_alloc from
    # generate().
    assert storage1_scheduler_events[:6] == [
        "bind_gpu_block_pool",
        "set_xfer_handshake_metadata_pp_aware",
        "on_new_request",
        "get_num_new_matched_tokens 0",
        "update_state_after_alloc num_blocks=[7] 0",
        "build_connector_meta",
    ]
    # First three events are from initialization (register_kv_caches,
    # set_host_xfer_buffer_ops, get_handshake_metadata), then generate() events.
    storage1_worker_events = _ignore_event_collection(events["storage1-WORKER"])
    storage2_worker_events = _ignore_event_collection(events["storage2-WORKER"])
    assert storage1_worker_events[:8] == [
        "register_kv_caches",
        "set_host_xfer_buffer_ops",
        "get_handshake_metadata",
        "handle_preemptions",
        "bind_connector_metadata",
        "start_load_kv",
        "wait_for_layer_load",
        "save_kv_layer",
    ]
    assert storage2_scheduler_events[:6] == [
        "bind_gpu_block_pool",
        "set_xfer_handshake_metadata_pp_aware",
        "on_new_request",
        "get_num_new_matched_tokens 0",
        "update_state_after_alloc num_blocks=[7] 0",
        "build_connector_meta",
    ]
    assert storage2_worker_events[:8] == [
        "register_kv_caches",
        "set_host_xfer_buffer_ops",
        "get_handshake_metadata",
        "handle_preemptions",
        "bind_connector_metadata",
        "start_load_kv",
        "wait_for_layer_load",
        "save_kv_layer",
    ]

    # Reset prefix cache or else we'll just get the tokens back from there.
    llm.reset_prefix_cache()

    # Run generation again - this should trigger loading from the first
    # connector.
    _ = llm.generate(PROMPTS[1], SAMPLING_PARAMS)

    events = get_connector_events()
    # get_num_new_matched_tokens will return new tokens from the first
    # connector (first nonzero match is chosen), so update_state_after_alloc
    # will report those external tokens on that one. Other connectors still
    # receive the request's real blocks but with 0 external tokens.
    storage1_scheduler_events = _events_from_request(
        _ignore_event_collection(events["storage1-SCHEDULER"])
    )
    storage2_scheduler_events = _events_from_request(
        _ignore_event_collection(events["storage2-SCHEDULER"])
    )
    assert storage1_scheduler_events[:4] == [
        "on_new_request",
        "get_num_new_matched_tokens 0",
        "update_state_after_alloc num_blocks=[7] 96",
        "build_connector_meta",
    ]
    assert storage2_scheduler_events[:4] == [
        "on_new_request",
        "get_num_new_matched_tokens 0",
        "update_state_after_alloc num_blocks=[7] 0",
        "build_connector_meta",
    ]

    # Delete storage1 connector state
    shutil.rmtree(storage_1_path)

    # Reset prefix cache or else we'll just get the tokens back from there.
    llm.reset_prefix_cache()

    # Run generation again - this should trigger loading from the first
    # connector.
    _ = llm.generate(PROMPTS[0], SAMPLING_PARAMS)

    events = get_connector_events()
    # get_num_new_matched_tokens will be called for both connectors but will
    # return 0 from the first connector, while the second connector has a hit.
    # Both connectors receive the request's real blocks, but only the chosen
    # (second) connector reports external tokens.
    storage1_scheduler_events = _events_from_request(
        _ignore_event_collection(events["storage1-SCHEDULER"])
    )
    storage2_scheduler_events = _events_from_request(
        _ignore_event_collection(events["storage2-SCHEDULER"])
    )
    assert storage1_scheduler_events[:4] == [
        "on_new_request",
        "get_num_new_matched_tokens 0",
        "update_state_after_alloc num_blocks=[7] 0",
        "build_connector_meta",
    ]
    assert storage2_scheduler_events[:4] == [
        "on_new_request",
        "get_num_new_matched_tokens 0",
        "update_state_after_alloc num_blocks=[7] 96",
        "build_connector_meta",
    ]

    # Clean up
    shutil.rmtree(storage_1_path)
    shutil.rmtree(storage_2_path)


def _ignore_event_collection(events: list[str]) -> list[str]:
    # Filter out per-step polling hooks that the scheduler calls repeatedly
    # and which are not meaningful state transitions for these assertions.
    ignored = {
        "bind_peer_connector_metadata",
        "clear_peer_connector_metadata",
        "export_request_sidecar",
        "get_kv_connector_stats",
        "has_pending_push_work",
        "prepare_request_lookup",
        "take_events",
    }
    return [event for event in events if event not in ignored]


def _events_from_request(events: list[str]) -> list[str]:
    # The async engine-core can emit a trailing build_connector_meta from the
    # previous generation's final (idle) scheduler step. Depending on timing it
    # may be flushed into this window ahead of on_new_request, so anchor the
    # comparison on the new request's first event to avoid a flaky ordering.
    if "on_new_request" in events:
        return events[events.index("on_new_request") :]
    return events


def get_connector_events() -> dict[str, list[str]]:
    # Read in connector events and reset the files.
    import glob

    event_files = glob.glob(tempfile.gettempdir() + "/connector_*_events.log")
    connector_events = {}
    for fname in event_files:
        name = fname.split("connector_")[1].split("_events.log")[0]
        try:
            with open(fname, "r+") as f:
                connector_events[name] = [line.strip() for line in f if line.strip()]
                f.truncate(0)
        except Exception as e:
            print(f"[ERROR] Could not read connector events for {name}: {e}")

    return connector_events


def test_engine_id_conflict():
    configs = [KVTransferConfig() for _ in range(2)]
    ids = [config.engine_id for config in configs]
    assert ids[0] != ids[1], (
        f"Engine IDs should be different for different configs. Got {ids}"
    )


def test_multi_connector_handle_preemptions_integration():
    """
    Integration test: verify MultiConnector delegates handle_preemptions
    to all sub-connectors.

    Uses TestExampleConnector which logs all method calls to temp files.
    This test directly calls handle_preemptions on a MultiConnector with
    TestExampleConnector sub-connectors and verifies the calls are logged.
    """
    from tests.v1.kv_connector.unit.utils import (
        create_scheduler,
        create_vllm_config,
    )

    storage_path = Path(tempfile.mkdtemp())

    try:
        # Configure MultiConnector with two TestExampleConnectors
        connectors_extra_config = {
            "connectors": [
                {
                    "kv_connector": "TestExampleConnector",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {
                        "shared_storage_path": str(storage_path / "s1"),
                        "name": "preempt1",
                    },
                    "kv_connector_module_path": "tests.v1.kv_connector.unit.utils",
                },
                {
                    "kv_connector": "TestExampleConnector",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {
                        "shared_storage_path": str(storage_path / "s2"),
                        "name": "preempt2",
                    },
                    "kv_connector_module_path": "tests.v1.kv_connector.unit.utils",
                },
            ]
        }

        vllm_config = create_vllm_config(
            block_size=16,
            max_num_batched_tokens=100,
            kv_connector="MultiConnector",
            kv_connector_extra_config=connectors_extra_config,
        )

        # Create scheduler - this initializes the MultiConnector with SCHEDULER role
        scheduler = create_scheduler(vllm_config, num_blocks=10)

        # Clear any events from initialization
        get_connector_events()

        # Directly call handle_preemptions on the scheduler's connector
        # Note: handle_preemptions is normally a worker-side method, but we're
        # testing the delegation behavior of MultiConnector here.
        # The connector attribute contains the KV connector.
        assert scheduler.connector is not None, "Scheduler should have a connector"
        connector_md = scheduler.connector.build_connector_meta(scheduler.schedule())
        scheduler.connector.handle_preemptions(connector_md)

        # Verify both connectors received the handle_preemptions call
        events = get_connector_events()

        # Both SCHEDULER-role connectors should have logged handle_preemptions
        assert "handle_preemptions" in events.get("preempt1-SCHEDULER", []), (
            f"preempt1-SCHEDULER should have handle_preemptions call. "
            f"Got events: {events}"
        )
        assert "handle_preemptions" in events.get("preempt2-SCHEDULER", []), (
            f"preempt2-SCHEDULER should have handle_preemptions call. "
            f"Got events: {events}"
        )

    finally:
        # Cleanup
        shutil.rmtree(storage_path, ignore_errors=True)


class TestMultiConnectorStats:
    """Tests for MultiConnector stats reconstruction and operations."""

    @pytest.mark.skip_global_cleanup
    def test_prometheus_projection_unwraps_live_child_stats(self):
        """Prometheus receives the child stats objects held by Multi stats."""
        child_prom = MagicMock()
        prom = MultiKVConnectorPromMetrics.__new__(MultiKVConnectorPromMetrics)
        prom._prom_metrics = {"MockConnector": child_prom}
        stats = MockConnectorStats(data={"mock_field": [1, 2, 3]})

        prom.observe({"MockConnector": stats}, engine_idx=2)

        child_prom.observe.assert_called_once_with(stats.data, 2)

    @pytest.mark.skip_global_cleanup
    def test_prometheus_projection_unwraps_serialized_child_stats(self):
        """Prometheus also accepts child stats serialized during transport."""
        child_prom = MagicMock()
        prom = MultiKVConnectorPromMetrics.__new__(MultiKVConnectorPromMetrics)
        prom._prom_metrics = {"MockConnector": child_prom}
        stats_data = {"mock_field": [1, 2, 3]}

        prom.observe({"MockConnector": {"data": stats_data}}, engine_idx=2)

        child_prom.observe.assert_called_once_with(stats_data, 2)

    @pytest.mark.skip_global_cleanup
    @pytest.mark.parametrize("stats_data", [object(), {}, {"data": []}])
    def test_prometheus_projection_rejects_malformed_child_stats(self, stats_data):
        child_prom = MagicMock()
        prom = MultiKVConnectorPromMetrics.__new__(MultiKVConnectorPromMetrics)
        prom._prom_metrics = {"MockConnector": child_prom}

        with pytest.raises(
            TypeError, match="Stats for connector 'MockConnector' must be"
        ):
            prom.observe({"MockConnector": stats_data})

        child_prom.observe.assert_not_called()

    def test_build_kv_connector_stats_with_none(self):
        """Test that build_kv_connector_stats returns empty stats when given None."""
        stats = MultiConnector.build_kv_connector_stats(data=None)

        assert stats is not None
        assert isinstance(stats, MultiKVConnectorStats)
        assert len(stats.data) == 0
        assert stats.is_empty()

    def test_build_kv_connector_stats_with_empty_dict(self):
        """Test that build_kv_connector_stats returns empty stats with empty dict."""
        stats = MultiConnector.build_kv_connector_stats(data={})

        assert stats is not None
        assert isinstance(stats, MultiKVConnectorStats)
        assert len(stats.data) == 0
        assert stats.is_empty()

    def test_build_kv_connector_stats_reconstructs_nixl_stats(self):
        """Test that NixlConnector stats are properly reconstructed with
        correct data."""
        serialized_data = {
            "NixlConnector": {
                "data": {
                    "transfer_duration": [1.5, 2.3],
                    "post_duration": [0.1, 0.2],
                    "bytes_transferred": [1024, 2048],
                    "num_descriptors": [10, 20],
                    "num_failed_transfers": [],
                    "num_failed_notifications": [],
                }
            }
        }

        stats = MultiConnector.build_kv_connector_stats(data=serialized_data)

        assert "NixlConnector" in stats.data
        nixl_stats = stats.data["NixlConnector"]
        assert isinstance(nixl_stats, NixlKVConnectorStats)
        assert nixl_stats.data["transfer_duration"] == [1.5, 2.3]
        assert nixl_stats.data["post_duration"] == [0.1, 0.2]
        assert nixl_stats.data["bytes_transferred"] == [1024, 2048]
        assert nixl_stats.data["num_descriptors"] == [10, 20]

    def test_build_kv_connector_stats_with_multiple_connectors(self):
        """Test reconstruction with multiple connector types that have custom stats."""
        serialized_data = {
            "NixlConnector": {
                "data": {
                    "transfer_duration": [1.5],
                    "post_duration": [0.1],
                    "bytes_transferred": [1024],
                    "num_descriptors": [10],
                    "num_failed_transfers": [],
                    "num_failed_notifications": [],
                }
            },
            "MockConnector": {"data": {"mock_field": [1, 2, 3]}},
        }

        stats = MultiConnector.build_kv_connector_stats(data=serialized_data)

        assert stats is not None
        assert isinstance(stats, MultiKVConnectorStats)
        # Both connectors should be reconstructed
        assert len(stats.data) == 2
        assert "NixlConnector" in stats.data
        assert "MockConnector" in stats.data
        assert isinstance(stats.data["NixlConnector"], NixlKVConnectorStats)
        assert isinstance(stats.data["MockConnector"], MockConnectorStats)
        # Verify data is preserved
        assert stats.data["MockConnector"].data == {"mock_field": [1, 2, 3]}

    def test_build_kv_connector_stats_raises_error_for_unknown_connector(self):
        """Test that unknown connectors raise an error."""
        serialized_data = {
            "UnknownConnector": {"data": {"some_field": [1, 2, 3]}},
            "NixlConnector": {
                "data": {
                    "transfer_duration": [1.5],
                    "post_duration": [0.1],
                    "bytes_transferred": [1024],
                    "num_descriptors": [10],
                    "num_failed_transfers": [],
                    "num_failed_notifications": [],
                }
            },
        }

        with pytest.raises(
            ValueError, match="Connector 'UnknownConnector' is not registered."
        ):
            MultiConnector.build_kv_connector_stats(data=serialized_data)

    def test_build_kv_connector_stats_with_already_instantiated_objects(self):
        """Test that already-instantiated stats objects are preserved (same process)."""
        # This simulates the in-process case where stats are not serialized
        nixl_stats = NixlKVConnectorStats(
            data={
                "transfer_duration": [1.5],
                "post_duration": [0.1],
                "bytes_transferred": [1024],
                "num_descriptors": [10],
                "num_failed_transfers": [],
                "num_failed_notifications": [],
            }
        )
        mock_stats = MockConnectorStats(data={"mock_field": [1, 2, 3]})

        data_with_objects = {
            "NixlConnector": nixl_stats,
            "MockConnector": mock_stats,
        }

        stats = MultiConnector.build_kv_connector_stats(data=data_with_objects)

        assert stats is not None
        assert isinstance(stats, MultiKVConnectorStats)
        assert len(stats.data) == 2
        # Verify objects are preserved as-is
        assert stats.data["NixlConnector"] is nixl_stats
        assert stats.data["MockConnector"] is mock_stats

    def test_build_kv_connector_stats_with_mixed_objects_and_dicts(self):
        """Test handling mixed already-instantiated and serialized stats."""
        # This can happen during transition or partial serialization
        nixl_stats = NixlKVConnectorStats(
            data={
                "transfer_duration": [1.5],
                "post_duration": [0.1],
                "bytes_transferred": [1024],
                "num_descriptors": [10],
                "num_failed_transfers": [],
                "num_failed_notifications": [],
            }
        )

        mixed_data = {
            "NixlConnector": nixl_stats,  # Already instantiated
            "MockConnector": {"data": {"mock_field": [1, 2, 3]}},  # Serialized
        }

        stats = MultiConnector.build_kv_connector_stats(data=mixed_data)

        assert stats is not None
        assert isinstance(stats, MultiKVConnectorStats)
        assert len(stats.data) == 2
        # Instantiated object preserved
        assert stats.data["NixlConnector"] is nixl_stats
        # Serialized object reconstructed
        assert isinstance(stats.data["MockConnector"], MockConnectorStats)
        assert stats.data["MockConnector"].data == {"mock_field": [1, 2, 3]}

    def test_build_kv_connector_stats_skips_connectors_without_custom_stats(self):
        """Test that connectors without custom stats (return None) are skipped."""
        # ExampleConnector doesn't override build_kv_connector_stats,
        # so it returns None and should be skipped
        serialized_data = {
            "NixlConnector": {
                "data": {
                    "transfer_duration": [1.5],
                    "post_duration": [0.1],
                    "bytes_transferred": [1024],
                    "num_descriptors": [10],
                    "num_failed_transfers": [],
                    "num_failed_notifications": [],
                }
            },
            "ExampleConnector": {"data": {"some_field": [1, 2, 3]}},
        }

        stats = MultiConnector.build_kv_connector_stats(data=serialized_data)

        assert stats is not None
        assert isinstance(stats, MultiKVConnectorStats)
        # Only NixlConnector should be reconstructed
        assert len(stats.data) == 1
        assert "NixlConnector" in stats.data
        assert isinstance(stats.data["NixlConnector"], NixlKVConnectorStats)
        # ExampleConnector should be skipped (returns None)
        assert "ExampleConnector" not in stats.data

    def test_build_kv_connector_stats_handles_malformed_data(self):
        """Test that malformed data raises appropriate errors."""
        serialized_data = {
            "NixlConnector": {"wrong_field": {"transfer_duration": [1.5]}}
        }

        with pytest.raises(AssertionError, match="Expected a dict with a 'data' field"):
            MultiConnector.build_kv_connector_stats(data=serialized_data)

    def test_aggregate_same_connector(self):
        """Test aggregating stats from the same connector type."""
        stats1 = MultiKVConnectorStats(
            data={
                "NixlConnector": NixlKVConnectorStats(
                    data={
                        "transfer_duration": [1.0],
                        "post_duration": [0.1],
                        "bytes_transferred": [1024],
                        "num_descriptors": [10],
                        "num_failed_transfers": [],
                        "num_failed_notifications": [],
                    }
                )
            }
        )

        stats2 = MultiKVConnectorStats(
            data={
                "NixlConnector": NixlKVConnectorStats(
                    data={
                        "transfer_duration": [2.0],
                        "post_duration": [0.2],
                        "bytes_transferred": [2048],
                        "num_descriptors": [20],
                        "num_failed_transfers": [],
                        "num_failed_notifications": [],
                    }
                )
            }
        )

        result = stats1.aggregate(stats2)

        assert result is stats1  # Should return self
        assert "NixlConnector" in result.data
        nixl_stats = result.data["NixlConnector"]
        assert nixl_stats.data["transfer_duration"] == [1.0, 2.0]
        assert nixl_stats.data["post_duration"] == [0.1, 0.2]
        assert nixl_stats.data["bytes_transferred"] == [1024, 2048]
        assert nixl_stats.data["num_descriptors"] == [10, 20]

    def test_aggregate_new_connector(self):
        """Test aggregating stats when a new connector type appears."""
        from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
            KVConnectorStats,
        )

        stats1 = MultiKVConnectorStats(
            data={
                "NixlConnector": NixlKVConnectorStats(
                    data={
                        "transfer_duration": [1.0],
                        "post_duration": [0.1],
                        "bytes_transferred": [1024],
                        "num_descriptors": [10],
                        "num_failed_transfers": [],
                        "num_failed_notifications": [],
                    }
                )
            }
        )

        stats2 = MultiKVConnectorStats(
            data={"ExampleConnector": KVConnectorStats(data={"field": [1, 2]})}
        )

        result = stats1.aggregate(stats2)

        assert "NixlConnector" in result.data
        assert "ExampleConnector" in result.data

    def test_reduce(self):
        """Test that reduce() correctly reduces all nested connector stats."""
        stats = MultiKVConnectorStats(
            data={
                "NixlConnector": NixlKVConnectorStats(
                    data={
                        "transfer_duration": [1.0, 2.0],
                        "post_duration": [0.1, 0.2],
                        "bytes_transferred": [1024, 2048],
                        "num_descriptors": [10, 20],
                        "num_failed_transfers": [],
                        "num_failed_notifications": [],
                    }
                )
            }
        )

        reduced = stats.reduce()

        assert "NixlConnector" in reduced
        assert isinstance(reduced["NixlConnector"], dict)
        # Check that the stats were reduced (should have aggregated values)
        assert "Num successful transfers" in reduced["NixlConnector"]
        assert reduced["NixlConnector"]["Num successful transfers"] == 2

    def test_reset(self):
        """Test that reset() resets all nested connector stats."""
        stats = MultiKVConnectorStats(
            data={
                "NixlConnector": NixlKVConnectorStats(
                    data={
                        "transfer_duration": [1.0, 2.0],
                        "post_duration": [0.1, 0.2],
                        "bytes_transferred": [1024, 2048],
                        "num_descriptors": [10, 20],
                        "num_failed_transfers": [],
                        "num_failed_notifications": [],
                    }
                )
            }
        )

        assert not stats.is_empty()

        stats.reset()

        # After reset, stats should be empty
        assert stats.is_empty()
        nixl_stats = stats.data["NixlConnector"]
        assert len(nixl_stats.data["transfer_duration"]) == 0

    def test_is_empty_with_multiple_connectors(self):
        """Test is_empty() returns correct value with multiple connectors."""
        # All empty
        stats = MultiKVConnectorStats(
            data={
                "NixlConnector": NixlKVConnectorStats(data={}),
            }
        )
        # Initialize empty stats
        stats.data["NixlConnector"].reset()
        assert stats.is_empty()

        # One non-empty
        stats.data["NixlConnector"].data["transfer_duration"].append(1.0)
        assert not stats.is_empty()


def test_multi_connector_overrides_all_base_methods():
    """
    Ensure MultiConnector overrides all public methods from KVConnectorBase_V1.
    """
    # These are fine to inherit from KVConnectorBase_V1
    # TODO(https://github.com/vllm-project/vllm/pull/31811): Remove
    # get_kv_connector_kv_cache_events from INHERITED_OK once implemented.
    INHERITED_OK = {
        "bind_peer_connector_metadata",
        "clear_peer_connector_metadata",
        "export_request_sidecar",
        "role",
        "has_connector_metadata",
        "get_kv_connector_kv_cache_events",
        "prepare_request_lookup",
    }

    base_members = {
        name for name in dir(KVConnectorBase_V1) if not name.startswith("_")
    } - KVConnectorBase_V1.__abstractmethods__

    missing = [
        name
        for name in sorted(base_members)
        if name not in INHERITED_OK and name not in MultiConnector.__dict__
    ]

    if missing:
        pytest.fail(f"""
MultiConnector does not override these KVConnectorBase_V1 methods: {missing}

MultiConnector wraps other connectors and must delegate all methods.
Please add overrides that delegate to self._connectors.

Options:
  1. Add delegation in MultiConnector (preferred)
  2. Add to INHERITED_OK if the base implementation works correctly
""")


def test_multi_connector_prefer_cross_layer_blocks(mc):
    mc._connectors[0].prefer_cross_layer_blocks = False
    mc._connectors[1].prefer_cross_layer_blocks = True
    assert mc.prefer_cross_layer_blocks is False

    mc._connectors[0].prefer_cross_layer_blocks = True
    mc._connectors[1].prefer_cross_layer_blocks = True
    assert mc.prefer_cross_layer_blocks is True


def test_multi_connector_worker_metadata(mc):
    class MockConnectorWorkerMetadata(KVConnectorWorkerMetadata):
        def __init__(self, data: set[str]):
            self.data = data

    class MockConnectorWorkerMetadata0(MockConnectorWorkerMetadata):
        def aggregate(
            self, other: KVConnectorWorkerMetadata
        ) -> KVConnectorWorkerMetadata:
            assert isinstance(other, MockConnectorWorkerMetadata)
            return MockConnectorWorkerMetadata0(data=self.data | other.data)

    class MockConnectorWorkerMetadata1(MockConnectorWorkerMetadata):
        def aggregate(
            self, other: KVConnectorWorkerMetadata
        ) -> KVConnectorWorkerMetadata:
            assert isinstance(other, MockConnectorWorkerMetadata)
            return MockConnectorWorkerMetadata1(data=self.data | other.data)

    # -------------------- test build_worker_connector_meta -------------------

    # both connectors return None
    mc._connectors[0].build_connector_worker_meta.return_value = None
    mc._connectors[1].build_connector_worker_meta.return_value = None
    assert mc.build_connector_worker_meta() is None

    # only first connector returns None
    worker_meta1a = MockConnectorWorkerMetadata1({"1a"})
    mc._connectors[0].build_connector_worker_meta.return_value = None
    mc._connectors[1].build_connector_worker_meta.return_value = worker_meta1a
    mc_worker_meta_none_1a = mc.build_connector_worker_meta()
    assert isinstance(mc_worker_meta_none_1a, MultiKVConnectorWorkerMetadata)
    assert mc_worker_meta_none_1a.metadata == (None, worker_meta1a)

    # only second connector returns None
    worker_meta0a = MockConnectorWorkerMetadata0({"0a"})
    mc._connectors[0].build_connector_worker_meta.return_value = worker_meta0a
    mc._connectors[1].build_connector_worker_meta.return_value = None
    mc_worker_meta_0a_none = mc.build_connector_worker_meta()
    assert isinstance(mc_worker_meta_0a_none, MultiKVConnectorWorkerMetadata)
    assert mc_worker_meta_0a_none.metadata == (worker_meta0a, None)

    # both connectors do not return None
    worker_meta0b = MockConnectorWorkerMetadata0({"0b"})
    worker_meta1b = MockConnectorWorkerMetadata1({"1b"})
    mc._connectors[0].build_connector_worker_meta.return_value = worker_meta0b
    mc._connectors[1].build_connector_worker_meta.return_value = worker_meta1b
    mc_worker_meta_0b_1b = mc.build_connector_worker_meta()
    assert isinstance(mc_worker_meta_0b_1b, MultiKVConnectorWorkerMetadata)
    assert mc_worker_meta_0b_1b.metadata == (worker_meta0b, worker_meta1b)

    # ----------------------------- test aggregate ----------------------------

    # aggregate ({"0a"}, None) and (None, {"1a"}) -> ({"0a"}, {"1a"})
    mc_worker_meta_0a_1a = mc_worker_meta_0a_none.aggregate(mc_worker_meta_none_1a)
    assert isinstance(mc_worker_meta_0a_1a, MultiKVConnectorWorkerMetadata)
    assert mc_worker_meta_0a_1a.metadata == (worker_meta0a, worker_meta1a)

    # aggregate ({"0a"}, None) and ({"0b"}, None) -> ({"0a", "0b"}, None)
    mc._connectors[0].build_connector_worker_meta.return_value = worker_meta0b
    mc._connectors[1].build_connector_worker_meta.return_value = None
    mc_worker_meta_0b_none = mc.build_connector_worker_meta()
    mc_worker_meta_0a_0b = mc_worker_meta_0a_none.aggregate(mc_worker_meta_0b_none)
    assert isinstance(mc_worker_meta_0a_0b, MultiKVConnectorWorkerMetadata)
    assert mc_worker_meta_0a_0b.metadata[1] is None
    connector0_md = mc_worker_meta_0a_0b.metadata[0]
    assert isinstance(connector0_md, MockConnectorWorkerMetadata0)
    assert connector0_md.data == {"0a", "0b"}

    # aggregate ({"0a"}, {"1a"}) and ({"0b"}, {"1b"}) -> ({"0a", "0b"}, {"1a", "1b"})
    mc_worker_meta_01a_01b = mc_worker_meta_0a_1a.aggregate(mc_worker_meta_0b_1b)
    assert isinstance(mc_worker_meta_01a_01b, MultiKVConnectorWorkerMetadata)
    metadata = mc_worker_meta_01a_01b.metadata
    assert len(metadata) == 2
    connector0_md, connector1_md = metadata
    assert isinstance(connector0_md, MockConnectorWorkerMetadata0)
    assert isinstance(connector1_md, MockConnectorWorkerMetadata1)
    assert connector0_md.data == {"0a", "0b"}
    assert connector1_md.data == {"1a", "1b"}

    # ---------------------- test update_connector_output ---------------------

    def verify_worker_metadata(expected_metadata: MockConnectorWorkerMetadata | None):
        def _verify_worker_metadata(connector_output: KVConnectorOutput):
            worker_meta = connector_output.kv_connector_worker_meta
            if expected_metadata is None:
                assert worker_meta is None
                return

            assert isinstance(worker_meta, MockConnectorWorkerMetadata)
            assert type(worker_meta) is type(expected_metadata)
            assert expected_metadata.data == worker_meta.data

        return _verify_worker_metadata

    def assert_update_connector_output_called(mc: MultiConnector):
        for c in mc._connectors:
            c.update_connector_output.assert_called_once()
            c.update_connector_output.reset_mock()

    # no worker meta
    kv_connector_output = KVConnectorOutput()
    mc._connectors[0].update_connector_output.side_effect = verify_worker_metadata(None)
    mc._connectors[1].update_connector_output.side_effect = verify_worker_metadata(None)
    mc.update_connector_output(kv_connector_output)
    assert_update_connector_output_called(mc)

    # multi worker meta
    kv_connector_output.kv_connector_worker_meta = mc_worker_meta_01a_01b
    mc._connectors[0].update_connector_output.side_effect = verify_worker_metadata(
        connector0_md
    )
    mc._connectors[1].update_connector_output.side_effect = verify_worker_metadata(
        connector1_md
    )
    mc.update_connector_output(kv_connector_output)
    assert_update_connector_output_called(mc)
    assert kv_connector_output.kv_connector_worker_meta == mc_worker_meta_01a_01b


def _make_multi_connector(connector_names: list[str]) -> MultiConnector:
    """Build a MultiConnector wrapping the given registered connectors."""
    connectors = [
        {
            "kv_connector": name,
            "kv_role": "kv_both",
            "kv_connector_module_path": "tests.v1.kv_connector.unit.test_multi_connector",  # noqa: E501
        }
        for name in connector_names
    ]
    vllm_config = create_vllm_config(
        kv_connector="MultiConnector",
        kv_connector_extra_config={"connectors": connectors},
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=0,
        kv_cache_tensors=[],
        kv_cache_groups=[],
    )
    return MultiConnector(
        vllm_config=vllm_config,
        role=KVConnectorRole.WORKER,
        kv_cache_config=kv_cache_config,
    )


def test_multi_connector_hma_support_detection():
    """
    At runtime, _all_support_hma is True only when every sub-connector
    implements SupportsHMA. Test all combinations of HMA / non-HMA
    sub-connectors.
    """

    assert supports_hma(MultiConnector)

    # -- All non-HMA connectors => _all_support_hma is False --
    mc_none = _make_multi_connector(["MockConnector", "MockConnector"])
    assert not supports_hma(mc_none._connectors[0])
    assert not supports_hma(mc_none._connectors[1])
    assert mc_none._all_support_hma is False

    # -- All HMA connectors => _all_support_hma is True --
    mc_all = _make_multi_connector(["MockHMAConnector", "MockHMAConnector"])
    assert supports_hma(mc_all._connectors[0])
    assert supports_hma(mc_all._connectors[1])
    assert mc_all._all_support_hma is True

    # -- Mixed: first HMA, second non-HMA => _all_support_hma is False --
    mc_mixed1 = _make_multi_connector(["MockHMAConnector", "MockConnector"])
    assert supports_hma(mc_mixed1._connectors[0])
    assert not supports_hma(mc_mixed1._connectors[1])
    assert mc_mixed1._all_support_hma is False

    # -- Mixed: first non-HMA, second HMA => _all_support_hma is False --
    mc_mixed2 = _make_multi_connector(["MockConnector", "MockHMAConnector"])
    assert not supports_hma(mc_mixed2._connectors[0])
    assert supports_hma(mc_mixed2._connectors[1])
    assert mc_mixed2._all_support_hma is False


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires GPU to instantiate LLM"
)
def test_multi_connector_mixed_hma_disables_hybrid_kv_cache(monkeypatch):
    """
    When MultiConnector wraps a mix of HMA (NixlConnector) and non-HMA
    (MockConnector) sub-connectors, verify that:
    1. The scheduler's MultiConnector has _all_support_hma == False.
    2. vLLM auto-disables the hybrid KV cache manager (no preference expressed by user)
    """
    from unittest.mock import patch

    from tests.v1.kv_connector.unit.test_nixl_connector import FakeNixlWrapper

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    kv_transfer_config = KVTransferConfig(
        kv_connector="MultiConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "connectors": [
                {
                    "kv_connector": "NixlConnector",
                    "kv_role": "kv_consumer",
                },
                {
                    "kv_connector": "MockConnector",
                    "kv_role": "kv_both",
                    "kv_connector_module_path": (
                        "tests.v1.kv_connector.unit.test_multi_connector"
                    ),
                },
            ],
        },
    )

    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker.NixlWrapper",
        FakeNixlWrapper,
    ):
        llm = LLM(
            model="Qwen/Qwen3-0.6B",
            enforce_eager=True,
            gpu_memory_utilization=0.3,
            max_model_len=128,
            max_num_seqs=1,
            max_num_batched_tokens=128,
            kv_transfer_config=kv_transfer_config,
        )
        try:
            # HMA should be auto-disabled when user has not expressed a preference.
            assert (
                llm.llm_engine.vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
                is True
            )
            # The scheduler-side MultiConnector should detect the mixed
            # HMA support among its sub-connectors.
            scheduler = llm.llm_engine.engine_core.engine_core.scheduler
            mc = scheduler.connector
            assert isinstance(mc, MultiConnector)
            assert mc._all_support_hma is False
        finally:
            llm.llm_engine.engine_core.shutdown()
