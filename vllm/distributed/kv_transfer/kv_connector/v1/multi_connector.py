# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, Any, cast

import torch

from vllm.config import VllmConfig
from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBaseType
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    CopyBlocksOp,
    KVConnectorBase_V1,
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
    KVConnectorRequestSidecar,
    KVConnectorRole,
    KVConnectorTransferResults,
    KVConnectorWorkerMetadata,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput

if TYPE_CHECKING:
    from vllm.distributed.kv_events import KVCacheEvent
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionBackend
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)

_construction_cleanup_lock = Lock()
_construction_cleanup_owners: list[KVConnectorBase_V1] = []


def _retain_construction_cleanup_owner(connector: KVConnectorBase_V1) -> None:
    with _construction_cleanup_lock:
        if not any(owner is connector for owner in _construction_cleanup_owners):
            _construction_cleanup_owners.append(connector)


def _release_construction_cleanup_owner(connector: KVConnectorBase_V1) -> None:
    with _construction_cleanup_lock:
        _construction_cleanup_owners[:] = [
            owner for owner in _construction_cleanup_owners if owner is not connector
        ]


class _ConnectorCleanupError(RuntimeError):
    def __init__(self, message: str, errors: list[BaseException]) -> None:
        super().__init__(message)
        self.errors = tuple(errors)


class _ConnectorCleanupBaseError(BaseException):
    def __init__(self, message: str, errors: list[BaseException]) -> None:
        super().__init__(message)
        self.errors = tuple(errors)


def _combine_cleanup_errors(message: str, errors: list[BaseException]) -> BaseException:
    if len(errors) == 1:
        return errors[0]
    if all(isinstance(error, Exception) for error in errors):
        return _ConnectorCleanupError(message, errors)
    return _ConnectorCleanupBaseError(message, errors)


@dataclass
class MultiKVConnectorMetadata(KVConnectorMetadata):
    metadata: tuple[KVConnectorMetadata, ...]
    extra_async_saves: dict[str, int] | None = None


@dataclass
class MultiKVConnectorWorkerMetadata(KVConnectorWorkerMetadata):
    metadata: tuple[KVConnectorWorkerMetadata | None, ...]

    def aggregate(self, other: KVConnectorWorkerMetadata) -> KVConnectorWorkerMetadata:
        assert isinstance(other, MultiKVConnectorWorkerMetadata)

        assert len(self.metadata) == len(other.metadata)
        metadata_list = []
        for metadata1, metadata2 in zip(self.metadata, other.metadata):
            if metadata1 is None:
                metadata_list.append(metadata2)
            elif metadata2 is None:
                metadata_list.append(metadata1)
            else:
                metadata_list.append(metadata1.aggregate(metadata2))

        return MultiKVConnectorWorkerMetadata(metadata=tuple(metadata_list))


@dataclass
class MultiKVConnectorStats(KVConnectorStats):
    """Maintain a dict of KVConnectorStats objects, one for each connector.
    This is used to aggregate the stats from all connectors separately.
    """

    def to_dict(self) -> dict[str, Any]:
        return {
            connector_id: stats.to_dict() for connector_id, stats in self.data.items()
        }

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        for connector_id, stats in other.data.items():
            if connector_id not in self.data:
                self[connector_id] = stats
            else:
                assert isinstance(stats, type(self.data[connector_id]))
                self[connector_id] = self[connector_id].aggregate(stats)
        return self

    def reset(self):
        for stats in self.data.values():
            stats.reset()

    def reduce(self) -> dict[str, Any]:
        # TODO (NickLucche) Adjust for logging on separate lines
        return {
            connector_id: stats.reduce() for connector_id, stats in self.data.items()
        }

    def is_empty(self) -> bool:
        return all(stats.is_empty() for stats in self.data.values())

    def __getitem__(self, connector_id: str) -> KVConnectorStats:
        return self.data[connector_id]

    def __setitem__(self, connector_id: str, stats: KVConnectorStats):
        self.data[connector_id] = stats


class MultiKVConnectorPromMetrics(KVConnectorPromMetrics):
    def __init__(
        self,
        vllm_config: "VllmConfig",
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
        prom_metrics: dict[str, KVConnectorPromMetrics],
    ):
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)
        self._prom_metrics = prom_metrics

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        for connector_id, stats_data in transfer_stats_data.items():
            assert connector_id in self._prom_metrics, (
                f"{connector_id} is not contained in the list of registered connectors "
                f"with Prometheus metrics support: {self._prom_metrics.keys()}"
            )
            self._prom_metrics[connector_id].observe(stats_data, engine_idx)


class MultiConnector(KVConnectorBase_V1, SupportsHMA):
    """A wrapper for using multiple KVConnectors at the same time.

    The current logic is:
    - Load KV from the first connector that advertises available tokens from
      get_num_new_matched_tokens(), based on the order in the config.
    - Save to all connectors.
    """

    @classmethod
    def requires_piecewise_for_cudagraph(cls, extra_config: dict[str, Any]) -> bool:
        """MultiConnector requires PIECEWISE CUDA graph mode if any of its
        child connectors require it.
        """
        connectors_config = extra_config.get("connectors", [])
        for conn_config in connectors_config:
            temp_ktc = KVTransferConfig(**conn_config)
            connector_cls = KVConnectorFactory.get_connector_class(temp_ktc)
            child_extra_config = conn_config.get("kv_connector_extra_config", {})
            if connector_cls.requires_piecewise_for_cudagraph(child_extra_config):
                return True
        return False

    @classmethod
    def all_children_support_hma(cls, kv_transfer_config: "KVTransferConfig") -> bool:
        """Return True only if every configured child connector supports HMA."""
        connectors_config = kv_transfer_config.kv_connector_extra_config.get(
            "connectors", []
        )
        if not connectors_config:
            return False
        for conn_config in connectors_config:
            child_config = KVTransferConfig(
                **{"engine_id": kv_transfer_config.engine_id, **conn_config}
            )
            if not KVConnectorFactory.supports_hma_config(child_config):
                return False
        return True

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(
            vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config
        )

        from vllm.v1.kmesh.config import validate_kmesh_target_profile

        self._kmesh_target_profile = validate_kmesh_target_profile(vllm_config)
        self._connectors: list[KVConnectorBase_V1] = []
        self._ktc_kv_transfer_config = []
        try:
            for connector_cls, temp_config in self._get_connector_classes_and_configs(
                vllm_config
            ):
                connector = connector_cls(temp_config, role, kv_cache_config)
                self._connectors.append(connector)
                self._ktc_kv_transfer_config.append(temp_config.kv_transfer_config)

            assert vllm_config.kv_transfer_config is not None
            self._all_support_hma = MultiConnector.all_children_support_hma(
                vllm_config.kv_transfer_config
            )
            assert (
                vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
                or self._all_support_hma
            ), "HMA should not be enabled unless all sub-connectors support it"
        except BaseException as construction_error:
            cleanup_errors: list[BaseException] = []
            for connector in reversed(self._connectors):
                _retain_construction_cleanup_owner(connector)
                try:
                    connector.shutdown()
                except BaseException as error:
                    cleanup_errors.append(error)
                else:
                    _release_construction_cleanup_owner(connector)
            if cleanup_errors:
                cleanup_cause = _combine_cleanup_errors(
                    "MultiConnector construction rollback failed", cleanup_errors
                )
                raise construction_error from cleanup_cause
            raise

        # A mapping from request id to the index of the connector chosen to
        # load the request from (if any).
        self._requests_to_connector: dict[str, int] = {}

        # Keeps track of *additional* remaining async saves (beyond 1) to be
        # finished per request. Not needed for async loads since we only allow
        # a single connector to load.
        # Propagated from scheduler to worker side via the connector metadata.
        self._extra_async_saves: dict[str, int] = {}
        self._metadata_transaction_state = "idle"
        self._metadata_transaction_children: tuple[KVConnectorBase_V1, ...] = ()

    @classmethod
    def pending_construction_cleanup_count(cls) -> int:
        """Return the number of children retained after failed construction."""
        with _construction_cleanup_lock:
            return len(_construction_cleanup_owners)

    @classmethod
    def retry_failed_construction_cleanup(cls) -> None:
        """Retry shutdown for every child retained after failed construction."""
        with _construction_cleanup_lock:
            owners = tuple(_construction_cleanup_owners)
        cleanup_errors: list[BaseException] = []
        for connector in reversed(owners):
            try:
                connector.shutdown()
            except BaseException as error:
                cleanup_errors.append(error)
            else:
                _release_construction_cleanup_owner(connector)
        if cleanup_errors:
            raise _combine_cleanup_errors(
                "MultiConnector construction cleanup retry failed", cleanup_errors
            )

    def _validate_target_profile_runtime(self, *, arm: bool) -> None:
        if self._kmesh_target_profile is None:
            return
        from vllm.distributed.kv_transfer.kv_connector.v1 import (
            kmesh_connector as kmesh_connector_module,
        )
        from vllm.v1.kmesh.config import (
            _KMESH_TARGET_PROFILE_SHARED_CONFIG_FIELDS,
            KMESH_COLLECTIVE_BATCHED_POSITION_RECOVERY_V1_PROFILE,
            KMESH_COLLECTIVE_BATCHED_QKV_V1_PROFILE,
            KMESH_COLLECTIVE_LOCKSTEP_V1_PROFILE,
            KMESH_SERIAL_PIC_V1_PROFILE,
            validate_kmesh_target_profile,
        )

        profile = validate_kmesh_target_profile(self._vllm_config)
        if profile != self._kmesh_target_profile:
            raise ValueError("KMesh target profile drifted after construction")
        for connector in self._connectors:
            connector.validate_shared_vllm_config(
                self._vllm_config, _KMESH_TARGET_PROFILE_SHARED_CONFIG_FIELDS
            )
            connector.validate_runtime_profile(self._vllm_config)
        if not arm:
            return
        kmesh_child = self._connectors[0]
        if type(kmesh_child) is not kmesh_connector_module.KMeshConnector:
            raise ValueError(
                f"KMesh {profile} requires the exact built-in KMeshConnector"
            )
        if profile == KMESH_SERIAL_PIC_V1_PROFILE:
            kmesh_child.arm_serial_pic_v1()
        elif profile == KMESH_COLLECTIVE_LOCKSTEP_V1_PROFILE:
            kmesh_child.arm_collective_lockstep_v1()
        elif profile == KMESH_COLLECTIVE_BATCHED_POSITION_RECOVERY_V1_PROFILE:
            kmesh_child.arm_collective_batched_position_recovery_v1()
        elif profile == KMESH_COLLECTIVE_BATCHED_QKV_V1_PROFILE:
            kmesh_child.arm_collective_batched_qkv_v1()
        else:  # pragma: no cover - validate_kmesh_target_profile rejects this.
            raise AssertionError(f"unreachable KMesh target profile: {profile!r}")

    @property
    def sub_connectors(self) -> list[KVConnectorBase_V1]:
        return list(self._connectors)

    @property
    def supports_divergent_local_hybrid_hits(self) -> bool:
        return bool(self._connectors) and all(
            c.supports_divergent_local_hybrid_hits for c in self._connectors
        )

    @property
    def requires_pre_forward_load(self) -> bool:
        return any(c.requires_pre_forward_load for c in self._connectors)

    @property
    def requires_kv_delivery(self) -> bool:
        return any(c.requires_kv_delivery for c in self._connectors)

    @classmethod
    def _get_connector_classes_and_configs(
        cls, vllm_config: "VllmConfig"
    ) -> list[tuple[type[KVConnectorBaseType], "VllmConfig"]]:
        """Resolve and internally qualify child connector configurations."""
        assert vllm_config.kv_transfer_config is not None
        from vllm.v1.kmesh.config import (
            KMESH_TARGET_PROFILE_KEY,
            _clear_kmesh_target_profile_qualification,
            _qualify_kmesh_target_profile_child,
            validate_kmesh_target_profile,
        )

        profile = validate_kmesh_target_profile(vllm_config)
        outer_kv_transfer_config = vllm_config.kv_transfer_config
        ktcs = outer_kv_transfer_config.kv_connector_extra_config.get("connectors")
        assert ktcs is not None
        ret: list[tuple[type[KVConnectorBaseType], VllmConfig]] = []
        for raw_ktc in ktcs:
            # Each child owns only its mutable KV-transfer configuration.
            # Engine-global configs remain shared read-only truth and are
            # snapshotted by KMesh qualification below.
            ktc = copy.deepcopy(raw_ktc) if profile is not None else dict(raw_ktc)
            if profile is not None:
                child_extra = copy.deepcopy(ktc.get("kv_connector_extra_config", {}))
                child_profile = child_extra.get(KMESH_TARGET_PROFILE_KEY)
                if child_profile is not None:
                    raise ValueError(
                        "KMesh target profile must be declared only on the outer "
                        "MultiConnector"
                    )
                child_extra[KMESH_TARGET_PROFILE_KEY] = profile
                ktc["kv_connector_extra_config"] = child_extra

                child_policy = ktc.get(
                    "kv_load_failure_policy",
                    outer_kv_transfer_config.kv_load_failure_policy,
                )
                if child_policy != outer_kv_transfer_config.kv_load_failure_policy:
                    raise ValueError(
                        "child KV load failure policy conflicts with outer "
                        "KMesh target profile"
                    )
                ktc["kv_load_failure_policy"] = child_policy
                ktc["engine_id"] = outer_kv_transfer_config.engine_id
                for field in (
                    "kv_buffer_device",
                    "kv_role",
                    "kv_port",
                    "kv_connector_module_path",
                    "enable_permute_local_kv",
                ):
                    ktc[field] = getattr(outer_kv_transfer_config, field)
            temp_config = copy.copy(vllm_config)
            _clear_kmesh_target_profile_qualification(temp_config)
            engine_id = ktc.pop("engine_id", outer_kv_transfer_config.engine_id)
            temp_config.kv_transfer_config = KVTransferConfig(
                **ktc, engine_id=engine_id
            )
            connector_cls = KVConnectorFactory.get_connector_class(
                temp_config.kv_transfer_config
            )
            if profile is not None:
                from vllm.distributed.kv_transfer.kv_connector.v1 import (
                    kmesh_connector,
                    lmcache_connector,
                )

                expected_classes = {
                    "KMeshConnector": kmesh_connector.KMeshConnector,
                    "LMCacheConnectorV1": lmcache_connector.LMCacheConnectorV1,
                }
                connector_name = cast(str, temp_config.kv_transfer_config.kv_connector)
                if connector_cls is not expected_classes.get(connector_name):
                    raise ValueError(
                        "KMesh target profile requires the exact built-in "
                        f"{connector_name} connector class"
                    )
                assert profile is not None
                _qualify_kmesh_target_profile_child(temp_config, profile, vllm_config)
            ret.append(
                (
                    connector_cls,
                    temp_config,
                )
            )
        return ret

    @classmethod
    def inspect_connector_classes_and_configs(
        cls, vllm_config: "VllmConfig"
    ) -> list[tuple[type[KVConnectorBaseType], "VllmConfig"]]:
        """Return resolved child configs without construction authority.

        This diagnostic interface exposes the copied connector settings while
        stripping the private qualification that permits actionable KMesh
        construction. Only ``MultiConnector`` construction consumes qualified
        child configs.
        """
        from vllm.v1.kmesh.config import _clear_kmesh_target_profile_qualification

        resolved = cls._get_connector_classes_and_configs(vllm_config)
        for _, child_config in resolved:
            _clear_kmesh_target_profile_qualification(child_config)
        return resolved

    def register_cross_layers_kv_cache(
        self, kv_cache: torch.Tensor, attn_backend: type["AttentionBackend"]
    ):
        self._validate_target_profile_runtime(arm=True)
        # Register on all connectors
        for c in self._connectors:
            c.register_cross_layers_kv_cache(kv_cache, attn_backend)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self._validate_target_profile_runtime(arm=True)
        for c in self._connectors:
            c.register_kv_caches(kv_caches)

    def bind_kv_cache_manager(self, kv_cache_manager: "KVCacheManager") -> None:
        for connector in self._connectors:
            connector.bind_kv_cache_manager(kv_cache_manager)

    def register_model(self, model: torch.nn.Module) -> None:
        self._validate_target_profile_runtime(arm=True)
        for connector in self._connectors:
            connector.register_model(model)

    def bind_gpu_block_pool(self, gpu_block_pool: "BlockPool") -> None:
        self._validate_target_profile_runtime(arm=True)
        for c in self._connectors:
            c.bind_gpu_block_pool(gpu_block_pool)

    def validate_shared_vllm_config(
        self, parent_config: "VllmConfig", fields: tuple[str, ...]
    ) -> None:
        super().validate_shared_vllm_config(parent_config, fields)

    def validate_runtime_profile(self, parent_config: "VllmConfig") -> None:
        self._validate_target_profile_runtime(arm=True)

    def validate_connector_metadata(
        self, connector_metadata: KVConnectorMetadata
    ) -> None:
        self._validate_target_profile_runtime(arm=True)
        if not isinstance(connector_metadata, MultiKVConnectorMetadata):
            raise TypeError("MultiConnector requires MultiKVConnectorMetadata")
        metadata = connector_metadata.metadata
        if len(metadata) != len(self._connectors):
            raise ValueError("MultiConnector metadata count does not match children")
        for connector, own_metadata in zip(self._connectors, metadata):
            connector.validate_connector_metadata(own_metadata)
        for connector in self._connectors:
            connector.validate_peer_connector_metadata(metadata)

    def validate_peer_connector_metadata(
        self, peer_connector_metadata: tuple[KVConnectorMetadata, ...]
    ) -> None:
        self._validate_target_profile_runtime(arm=True)
        return super().validate_peer_connector_metadata(peer_connector_metadata)

    def begin_connector_metadata_transaction(self) -> None:
        self._validate_target_profile_runtime(arm=True)
        return super().begin_connector_metadata_transaction()

    def commit_connector_metadata_transaction(self) -> None:
        self._validate_target_profile_runtime(arm=True)
        return super().commit_connector_metadata_transaction()

    def finalize_connector_metadata_transaction(self) -> None:
        self._validate_target_profile_runtime(arm=True)
        return super().finalize_connector_metadata_transaction()

    def abort_connector_metadata_transaction(self) -> None:
        transaction_children = self._metadata_transaction_children
        cleanup_errors: list[BaseException] = []
        try:
            self._validate_target_profile_runtime(arm=False)
        except BaseException as error:
            cleanup_errors.append(error)
        if transaction_children:
            cleanup_errors.extend(
                self._abort_connector_metadata_children(transaction_children)
            )
        else:
            self._metadata_transaction_state = "idle"
        if cleanup_errors:
            raise _combine_cleanup_errors(
                "MultiConnector metadata abort failed", cleanup_errors
            )

    def _abort_connector_metadata_children(
        self, transaction_children: Iterable[KVConnectorBase_V1]
    ) -> list[BaseException]:
        ordered_children = tuple(transaction_children)
        failed_children: list[KVConnectorBase_V1] = []
        cleanup_errors: list[BaseException] = []
        for connector in reversed(ordered_children):
            try:
                connector.abort_connector_metadata_transaction()
            except BaseException as error:
                failed_children.append(connector)
                cleanup_errors.append(error)
        failed_children.reverse()
        self._metadata_transaction_children = tuple(failed_children)
        self._metadata_transaction_state = (
            "abort_pending" if failed_children else "idle"
        )
        return cleanup_errors

    def _prepare_connector_metadata_transaction(self) -> None:
        if self._metadata_transaction_state == "abort_pending":
            cleanup_errors = self._abort_connector_metadata_children(
                self._metadata_transaction_children
            )
            if cleanup_errors:
                raise _combine_cleanup_errors(
                    "MultiConnector metadata recovery failed", cleanup_errors
                )
        if self._metadata_transaction_state == "finalized":
            self._metadata_transaction_children = ()
            self._metadata_transaction_state = "idle"
        if self._metadata_transaction_state != "idle":
            raise RuntimeError("MultiConnector metadata transaction is already active")
        self._metadata_transaction_state = "begun"

    # We must override the base class method here because we need to bind
    # the metadata to each connector in the order of the connectors in the
    # MultiKVConnectorMetadata.
    #
    # Note: Call the base class method to ensure metadata is also set on the
    # MultiConnector instance itself; otherwise, `has_connector_metadata()` will
    # always return False.
    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        self._prepare_connector_metadata_transaction()
        try:
            self.validate_connector_metadata(connector_metadata)
        except BaseException:
            self._metadata_transaction_state = "idle"
            raise
        assert isinstance(connector_metadata, MultiKVConnectorMetadata)
        metadata = connector_metadata.metadata
        extra_async_saves = dict(connector_metadata.extra_async_saves or {})

        transaction_children: list[KVConnectorBase_V1] = []
        try:
            for connector in self._connectors:
                transaction_children.append(connector)
                self._metadata_transaction_children = tuple(transaction_children)
                connector.begin_connector_metadata_transaction()
            for connector, own_metadata in zip(self._connectors, metadata):
                connector.bind_connector_metadata(own_metadata)
            for connector in self._connectors:
                connector.bind_peer_connector_metadata(metadata)
            super().bind_connector_metadata(connector_metadata)
            for connector in reversed(transaction_children):
                connector.commit_connector_metadata_transaction()
            for connector in reversed(transaction_children):
                connector.finalize_connector_metadata_transaction()
        except BaseException as bind_error:
            cleanup_errors = self._clear_connector_metadata_best_effort()
            cleanup_errors.extend(
                self._abort_connector_metadata_children(transaction_children)
            )
            if cleanup_errors:
                cleanup_cause = _combine_cleanup_errors(
                    "MultiConnector metadata rollback failed", cleanup_errors
                )
                raise bind_error from cleanup_cause
            raise

        self._metadata_transaction_children = tuple(transaction_children)
        self._metadata_transaction_state = "finalized"
        if extra_async_saves:
            self._extra_async_saves.update(extra_async_saves)

    def _clear_connector_metadata_best_effort(self) -> list[BaseException]:
        cleanup_errors: list[BaseException] = []
        for connector in self._connectors:
            try:
                connector.clear_peer_connector_metadata()
            except BaseException as error:
                cleanup_errors.append(error)
        for connector in self._connectors:
            try:
                connector.clear_connector_metadata()
            except BaseException as error:
                cleanup_errors.append(error)
        try:
            super().clear_connector_metadata()
        except BaseException as error:
            cleanup_errors.append(error)
        return cleanup_errors

    def clear_connector_metadata(self) -> None:
        cleanup_errors: list[BaseException] = []
        try:
            self._validate_target_profile_runtime(arm=False)
        except BaseException as error:
            cleanup_errors.append(error)
        cleanup_errors.extend(self._clear_connector_metadata_best_effort())
        if not cleanup_errors:
            return
        raise _combine_cleanup_errors(
            "MultiConnector metadata cleanup failed", cleanup_errors
        )

    def shutdown(self):
        cleanup_errors: list[BaseException] = []
        for c in reversed(self._connectors):
            try:
                c.shutdown()
            except BaseException as error:
                logger.exception(
                    "Exception during connector %s shutdown.", c.__class__.__name__
                )
                cleanup_errors.append(error)
        if cleanup_errors:
            raise _combine_cleanup_errors(
                "MultiConnector shutdown failed", cleanup_errors
            )

    # ==============================
    # Worker-side methods
    # ==============================
    def get_mem_pool_context(self) -> AbstractContextManager | None:
        """Forward a custom KV cache memory pool from a child connector.

        KV cache is allocated once and can only live in one pool, so at
        most one child may provide a context.
        """
        found: list[tuple[str, AbstractContextManager]] = []
        for connector in self._connectors:
            if (ctx := connector.get_mem_pool_context()) is not None:
                found.append((type(connector).__name__, ctx))

        if len(found) > 1:
            names = [name for name, _ in found]
            raise ValueError(
                f"Multiple connectors provide a KV cache memory pool {names}; "
                "KV cache is allocated once and can only live in one pool. "
                "Configure custom_mem_pool on at most one connector."
            )
        return found[0][1] if found else None

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        self._validate_target_profile_runtime(arm=True)
        for c in self._connectors:
            c.start_load_kv(forward_context, **kwargs)

    def finish_forward(self) -> None:
        for c in self._connectors:
            c.finish_forward()

    def reset_capture_state(self) -> None:
        for c in self._connectors:
            c.reset_capture_state()

    def wait_for_layer_load(self, layer_name: str) -> None:
        self._validate_target_profile_runtime(arm=True)
        for c in self._connectors:
            c.wait_for_layer_load(layer_name)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> None:
        self._validate_target_profile_runtime(arm=True)
        for c in self._connectors:
            c.save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)

    def wait_for_save(self):
        self._validate_target_profile_runtime(arm=True)
        for c in self._connectors:
            c.wait_for_save()

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        results = self.get_transfer_results(finished_req_ids)
        return (
            results.finished_sending or None,
            results.finished_recving or None,
        )

    def get_transfer_results(
        self, finished_req_ids: set[str]
    ) -> KVConnectorTransferResults:
        self._validate_target_profile_runtime(arm=True)
        results = KVConnectorTransferResults()
        for connector in self._connectors:
            child_results = connector.get_transfer_results(finished_req_ids)
            results.finished_recving.update(child_results.finished_recving)
            results.failed_recving.update(child_results.failed_recving)
            for req_id in child_results.finished_sending:
                extra_pending = self._extra_async_saves.get(req_id)
                if extra_pending is None:
                    results.finished_sending.add(req_id)
                else:
                    assert extra_pending > 0
                    if extra_pending == 1:
                        del self._extra_async_saves[req_id]
                    else:
                        self._extra_async_saves[req_id] = extra_pending - 1
        return results

    def get_block_ids_with_load_errors(self) -> set[int]:
        self._validate_target_profile_runtime(arm=True)
        agg_block_ids: set[int] = set()
        for c in self._connectors:
            agg_block_ids |= c.get_block_ids_with_load_errors()
        return agg_block_ids

    def set_host_xfer_buffer_ops(self, copy_operation: CopyBlocksOp):
        """Set xPU-specific copy ops for all sub-connectors."""
        self._validate_target_profile_runtime(arm=True)
        for c in self._connectors:
            c.set_host_xfer_buffer_ops(copy_operation)

    def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata):
        """Handle preempted requests for all sub-connectors."""
        self._validate_target_profile_runtime(arm=True)
        assert isinstance(kv_connector_metadata, MultiKVConnectorMetadata)
        for c, cm in zip(self._connectors, kv_connector_metadata.metadata):
            c.handle_preemptions(cm)

    def get_finished_count(self) -> int | None:
        child_counts = [
            connector.get_finished_count() for connector in self._connectors
        ]
        if any(count is None for count in child_counts):
            return None
        counts = {count for count in child_counts if count is not None}
        if len(counts) > 1:
            raise ValueError(
                "MultiConnector children returned incompatible finished counts: "
                f"{sorted(counts)}"
            )
        return next(iter(counts), None)

    def build_connector_worker_meta(self) -> KVConnectorWorkerMetadata | None:
        self._validate_target_profile_runtime(arm=True)
        metadata_list: list[KVConnectorWorkerMetadata | None] | None = None
        for i, c in enumerate(self._connectors):
            kv_connector_worker_meta = c.build_connector_worker_meta()
            if metadata_list is None and kv_connector_worker_meta is not None:
                metadata_list = [None] * i
            if metadata_list is not None:
                metadata_list.append(kv_connector_worker_meta)
        if metadata_list is None:
            return None
        return MultiKVConnectorWorkerMetadata(metadata=tuple(metadata_list))

    # TODO: Add a generic implementation of 'get_kv_connector_kv_cache_events'
    # method for the MultiConnector. It should be able to get events from
    # multiple connectors, handling the case where only a subset of the
    # requested connectors implements the 'get_kv_connector_kv_cache_events'
    # WIP: https://github.com/vllm-project/vllm/pull/31811

    # ==============================
    # Scheduler-side methods
    # ==============================
    def get_loaded_kv_cache_group_ids(self, request: "Request") -> tuple[int, ...]:
        connector = self._connectors[self._requests_to_connector[request.request_id]]
        return connector.get_loaded_kv_cache_group_ids(request)

    def _collect_request_sidecars(
        self, request: "Request"
    ) -> tuple[KVConnectorRequestSidecar, ...]:
        self._validate_target_profile_runtime(arm=True)
        sidecars: list[KVConnectorRequestSidecar] = []
        provider_indices: dict[str, int] = {}
        for i, connector in enumerate(self._connectors):
            sidecar = connector.export_request_sidecar(request)
            if sidecar is None:
                continue
            if not isinstance(sidecar, KVConnectorRequestSidecar):
                raise TypeError(
                    f"Connector at index {i} returned an invalid request sidecar: "
                    f"{type(sidecar).__name__}"
                )
            if sidecar.provider_id in provider_indices:
                first_index = provider_indices[sidecar.provider_id]
                raise ValueError(
                    f"Duplicate request sidecar provider {sidecar.provider_id!r} "
                    f"at connector indices {first_index} and {i}"
                )
            provider_indices[sidecar.provider_id] = i
            sidecars.append(sidecar)
        return tuple(sidecars)

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        sidecars = self._collect_request_sidecars(request)
        for connector in self._connectors:
            connector.prepare_request_lookup(request, sidecars)

        to_return = (0, False)
        for i, c in enumerate(self._connectors):
            toks, load_async = c.get_num_new_matched_tokens(
                request, num_computed_tokens
            )
            # If there is a connector still looking up the matches,
            # we return None to indicate that we are not done yet.
            if toks is None:
                return (None, False)
            # The first connector that has new matched tokens will be assigned
            # to this request.
            if to_return[0] == 0 and toks > 0:
                self._requests_to_connector[request.request_id] = i
                to_return = (toks, load_async)
        return to_return

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        self._validate_target_profile_runtime(arm=True)
        chosen_connector = self._requests_to_connector.get(request.request_id, -1)
        for i, c in enumerate(self._connectors):
            if i == chosen_connector:
                # Forward call to the chosen connector (if any).
                c.update_state_after_alloc(request, blocks, num_external_tokens)
            else:
                # Other connectors still receive the request's real blocks
                c.update_state_after_alloc(request, blocks, 0)

    def on_new_request(self, request: "Request") -> None:
        self._validate_target_profile_runtime(arm=True)
        for c in self._connectors:
            c.on_new_request(request)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> MultiKVConnectorMetadata:
        self._validate_target_profile_runtime(arm=True)
        metadata = MultiKVConnectorMetadata(
            metadata=tuple(
                c.build_connector_meta(scheduler_output) for c in self._connectors
            )
        )
        if self._extra_async_saves:
            metadata.extra_async_saves = self._extra_async_saves
            self._extra_async_saves = {}
        return metadata

    def update_connector_output(self, connector_output: KVConnectorOutput):
        self._validate_target_profile_runtime(arm=True)
        multi_connector_worker_meta: MultiKVConnectorWorkerMetadata | None = None
        if connector_output.kv_connector_worker_meta is not None:
            assert isinstance(
                connector_output.kv_connector_worker_meta,
                MultiKVConnectorWorkerMetadata,
            )
            multi_connector_worker_meta = connector_output.kv_connector_worker_meta

        try:
            for i, c in enumerate(self._connectors):
                if multi_connector_worker_meta is not None:
                    # set the connector-specific worker metadata
                    connector_output.kv_connector_worker_meta = (
                        multi_connector_worker_meta.metadata[i]
                    )
                c.update_connector_output(connector_output)
        finally:
            # restore kv_connector_worker_meta
            connector_output.kv_connector_worker_meta = multi_connector_worker_meta

    def get_handshake_metadata(self) -> KVConnectorHandshakeMetadata | None:
        """Get the KVConnector handshake metadata from sub-connectors.
        Returns the first non-None metadata from sub-connectors.
        """
        self._validate_target_profile_runtime(arm=True)
        for c in self._connectors:
            metadata = c.get_handshake_metadata()
            if metadata is not None:
                return metadata
        return None

    def set_xfer_handshake_metadata(
        self, metadata: dict[int, KVConnectorHandshakeMetadata]
    ) -> None:
        """Set the KV connector handshake metadata for all sub-connectors.
        This is needed to start the NIXL listener thread for NixlConnector.
        """
        self._validate_target_profile_runtime(arm=True)
        for c in self._connectors:
            c.set_xfer_handshake_metadata(metadata)

    def set_xfer_handshake_metadata_pp_aware(
        self, metadata: dict[tuple[int, int], KVConnectorHandshakeMetadata]
    ) -> None:
        self._validate_target_profile_runtime(arm=True)
        for c in self._connectors:
            c.set_xfer_handshake_metadata_pp_aware(metadata)

    def _aggregate_request_finished(
        self,
        request: "Request",
        per_connector_fn: Callable[
            [KVConnectorBase_V1], tuple[bool, dict[str, Any] | None]
        ],
    ) -> tuple[bool, dict[str, Any] | None]:
        self._validate_target_profile_runtime(arm=True)
        async_saves = 0
        kv_txfer_params = None
        for c in self._connectors:
            async_save, txfer_params = per_connector_fn(c)
            if async_save:
                async_saves += 1
            if txfer_params is not None:
                if kv_txfer_params is not None:
                    clashes = set(kv_txfer_params) & set(txfer_params)
                    if clashes:
                        raise RuntimeError(
                            "Key clash in kv_transfer_params from multiple "
                            f"connectors: {clashes}"
                        )
                    kv_txfer_params.update(txfer_params)
                else:
                    kv_txfer_params = txfer_params
        if async_saves > 1:
            self._extra_async_saves[request.request_id] = async_saves - 1

        self._requests_to_connector.pop(request.request_id, None)

        return async_saves > 0, kv_txfer_params

    def request_finished(
        self,
        request: "Request",
        blocks: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        return self._aggregate_request_finished(
            request,
            lambda c: c.request_finished(request, blocks),
        )

    def register_finished_partial_tail(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
        partial_tail_offloads: list[tuple[int, int, int]],
    ) -> bool:
        accepted = [
            c.register_finished_partial_tail(request, block_ids, partial_tail_offloads)
            for c in self._connectors
        ]
        return any(accepted)

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        if not self._all_support_hma:
            assert len(block_ids) == 1, (
                "HMA with multiple kv_cache_groups requires all "
                "sub-connectors to support HMA"
            )
            return self.request_finished(request, block_ids[0])

        return self._aggregate_request_finished(
            request,
            lambda c: cast(SupportsHMA, c).request_finished_all_groups(
                request, block_ids
            ),
        )

    def take_events(self) -> Iterable["KVCacheEvent"]:
        self._validate_target_profile_runtime(arm=True)
        for c in self._connectors:
            yield from c.take_events()

    def has_pending_block_frees(self) -> bool:
        return any(c.has_pending_block_frees() for c in self._connectors)

    def has_pending_push_work(self) -> bool:
        self._validate_target_profile_runtime(arm=True)
        return any(c.has_pending_push_work() for c in self._connectors)

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: "VllmConfig") -> str | None:
        """Get the required KV cache layout for this connector.

        Args:
            vllm_config (VllmConfig): the vllm config.

        Returns:
            str: the required KV cache layout. e.g. HND, or NHD.
            None if the connector does not require a specific layout.

        """
        assert vllm_config.kv_transfer_config is not None
        layouts: set[str] = set()
        for connector_cls, temp_config in cls._get_connector_classes_and_configs(
            vllm_config
        ):
            required_kvcache_layout = connector_cls.get_required_kvcache_layout(
                temp_config
            )
            if required_kvcache_layout is not None:
                layouts.add(required_kvcache_layout)

        if len(layouts) > 1:
            raise ValueError(
                f"KV cache layout mismatch: "
                f"found {len(layouts)} different layouts "
                f"({', '.join(layouts)})."
                f"All connectors must use the same layout."
            )
        return next(iter(layouts), None)

    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> KVConnectorStats | None:
        if data is None:
            return MultiKVConnectorStats()

        # data is a dict mapping connector name to their stats data.
        # The stats data can be either:
        # 1. Already-instantiated KVConnectorStats objects (same process)
        # 2. Serialized dicts (cross-process after serialization)
        # We need to reconstruct proper KVConnectorStats objects from dicts
        reconstructed_data = {}
        for connector_name, stats_value in data.items():
            # If already a KVConnectorStats object, use it directly
            if isinstance(stats_value, KVConnectorStats):
                reconstructed_data[connector_name] = stats_value
                continue

            # Otherwise, reconstruct from serialized dict
            # Get the connector class to reconstruct its stats
            connector_cls = KVConnectorFactory.get_connector_class_by_name(
                connector_name
            )

            # Use the connector's build_kv_connector_stats to reconstruct
            if reconstructed_stats := connector_cls.build_kv_connector_stats(
                data=stats_value
            ):
                reconstructed_data[connector_name] = reconstructed_stats

        return MultiKVConnectorStats(data=reconstructed_data)

    def get_kv_connector_stats(self) -> MultiKVConnectorStats | None:
        self._validate_target_profile_runtime(arm=True)
        # Group connector stats by connector type.
        stats_by_connector: MultiKVConnectorStats | None = None
        for c in self._connectors:
            stats = c.get_kv_connector_stats()
            if stats is None:
                continue
            if stats_by_connector is None:
                # Lazy init to allow optional return value.
                stats_by_connector = MultiKVConnectorStats()
            connector_id = c.__class__.__name__
            if connector_id in stats_by_connector.data:
                stats_by_connector[connector_id] = stats_by_connector[
                    connector_id
                ].aggregate(stats)
            else:
                stats_by_connector[connector_id] = stats
        return stats_by_connector

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: "VllmConfig",
        metric_types: dict[type["PromMetric"], type["PromMetricT"]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> KVConnectorPromMetrics:
        prom_metrics: dict[str, KVConnectorPromMetrics] = {}
        seen_classes: set[type] = set()
        for connector_cls, temp_config in cls._get_connector_classes_and_configs(
            vllm_config
        ):
            if connector_cls in seen_classes:
                continue
            seen_classes.add(connector_cls)
            connector_prom = connector_cls.build_prom_metrics(
                temp_config, metric_types, labelnames, per_engine_labelvalues
            )
            if connector_prom is not None:
                prom_metrics[connector_cls.__name__] = connector_prom
        return MultiKVConnectorPromMetrics(
            vllm_config,
            metric_types,
            labelnames,
            per_engine_labelvalues,
            prom_metrics,
        )

    def reset_cache(self) -> bool:
        self._validate_target_profile_runtime(arm=True)
        results = [c.reset_cache() is not False for c in self._connectors]
        return all(results)
