# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-to-cartridge routing for CartridgeConnector.

Scheduler-visible dispatch layer that resolves an incoming request to
a specific cartridge_id before block allocation. This is the piece
that turns the multi-cartridge infrastructure (Store, Registry,
LMCache plugin) into an actually routable serving path.

Resolution modes:

  - ExplicitCartridgeRouter: reads cartridge_id directly from
    ``sampling_params.extra_args["cartridge_id"]`` (or the
    ``kv_transfer_params`` nested dict, following vLLM convention).
    Use this when the caller knows which cartridge they want.

  - LabelCartridgeRouter: reads a label value from extras and queries
    the CartridgeRegistry by label. Use this for patient_id /
    document_id / topic-style routing where the caller does not know
    cartridge IDs directly.

  - StaticCartridgeRouter: always returns the same cartridge_id.
    Used for the backward-compatible singleton mode and as a fallback
    in CompositeRouter chains.

  - CompositeRouter: tries a list of routers in order, returns the
    first non-None resolution. Used to layer explicit-id lookups over
    label lookups over a static default.

Returning None from resolve() means "no cartridge for this request" —
the connector should treat this as a normal prefill request with no
KV injection.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Sequence

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_registry import (
        CartridgeRegistry,
    )
    from vllm.v1.request import Request

logger = init_logger(__name__)


def _get_extras(request: "Request") -> dict:
    """Pull extra_args dict from a request, defensively.

    Works across SamplingParams + PoolingParams + missing fields.
    Returns empty dict when nothing is set, never raises.
    """
    sp = getattr(request, "sampling_params", None)
    if sp is None:
        return {}
    extras = getattr(sp, "extra_args", None)
    if not isinstance(extras, dict):
        return {}
    return extras


def _get_extras_value(request: "Request", key: str) -> Optional[str]:
    """Look up a key in extras, checking both the top level and the
    ``kv_transfer_params`` nested dict (the canonical vLLM location
    for connector-bound request metadata).
    """
    extras = _get_extras(request)
    if not extras:
        return None
    value = extras.get(key)
    if value is not None:
        return str(value)
    ktp = extras.get("kv_transfer_params")
    if isinstance(ktp, dict):
        value = ktp.get(key)
        if value is not None:
            return str(value)
    return None


class CartridgeRouter(ABC):
    """Resolves a Request to a cartridge_id.

    Returns None if this router cannot make a decision for the
    request; downstream routers (in a CompositeRouter) may still
    succeed. None at the top level means "no cartridge injection."
    """

    @abstractmethod
    def resolve(self, request: "Request") -> Optional[str]:
        ...


class ExplicitCartridgeRouter(CartridgeRouter):
    """Reads cartridge_id directly from sampling_params.extra_args.

    Looks for a key named ``cartridge_id`` at the top of extras, and
    if missing, inside ``extras["kv_transfer_params"]``. Returns None
    if neither is set.
    """

    def __init__(self, key: str = "cartridge_id"):
        self._key = key

    def resolve(self, request: "Request") -> Optional[str]:
        return _get_extras_value(request, self._key)


class LabelCartridgeRouter(CartridgeRouter):
    """Resolves via label lookup against a CartridgeRegistry.

    The request carries a label VALUE in extras under ``extras_key``
    (default: ``cartridge_label``). The router queries the registry
    for cartridges whose manifest has ``labels[label_key] == value``
    and returns the first match (ordered by registration time).

    If the registry has no match, returns None.
    """

    def __init__(
        self,
        registry: "CartridgeRegistry",
        label_key: str,
        extras_key: str = "cartridge_label",
    ):
        self._registry = registry
        self._label_key = label_key
        self._extras_key = extras_key

    def resolve(self, request: "Request") -> Optional[str]:
        label_value = _get_extras_value(request, self._extras_key)
        if label_value is None:
            return None
        matches = self._registry.lookup_by_label(
            self._label_key, label_value)
        if not matches:
            logger.debug(
                "LabelCartridgeRouter: no match for %s=%s",
                self._label_key, label_value,
            )
            return None
        return matches[0].cartridge_id


class StaticCartridgeRouter(CartridgeRouter):
    """Always returns the same cartridge_id.

    Used for singleton-mode backward compatibility and as a default
    fallback at the end of a CompositeRouter chain.
    """

    def __init__(self, cartridge_id: str):
        self._cartridge_id = cartridge_id

    def resolve(self, request: "Request") -> Optional[str]:
        return self._cartridge_id


class CompositeRouter(CartridgeRouter):
    """Tries a sequence of routers in order.

    Returns the first non-None resolution. If all routers return
    None, returns None. Order matters: put specific routers (explicit
    id) before generic ones (label lookup) before defaults (static).
    """

    def __init__(self, routers: Sequence[CartridgeRouter]):
        self._routers = list(routers)

    def resolve(self, request: "Request") -> Optional[str]:
        for router in self._routers:
            cart_id = router.resolve(request)
            if cart_id is not None:
                return cart_id
        return None


def build_router_from_config(
    config: dict,
    registry: Optional["CartridgeRegistry"] = None,
    default_cartridge_id: Optional[str] = None,
) -> CartridgeRouter:
    """Build a CartridgeRouter from a dict config.

    Config shape::

        {
            "type": "explicit" | "label" | "static" | "composite",
            # type=explicit:
            "key": "cartridge_id",        # optional, defaults to cartridge_id
            # type=label:
            "label_key": "patient_id",    # required
            "extras_key": "patient_id",   # optional, defaults to cartridge_label
            # type=static:
            "cartridge_id": "...",        # required
            # type=composite:
            "routers": [ {...}, {...} ],  # list of router configs
        }

    If config is empty/None and default_cartridge_id is set, returns
    a StaticCartridgeRouter bound to that id (singleton mode).

    Raises ValueError for unknown types or missing required fields.
    """
    if not config:
        if default_cartridge_id is None:
            raise ValueError(
                "router config is empty and no default_cartridge_id "
                "provided; cannot build a router"
            )
        return StaticCartridgeRouter(default_cartridge_id)

    router_type = config.get("type", "explicit")

    if router_type == "explicit":
        return ExplicitCartridgeRouter(
            key=config.get("key", "cartridge_id"),
        )

    if router_type == "label":
        if registry is None:
            raise ValueError(
                "LabelCartridgeRouter requires a CartridgeRegistry")
        label_key = config.get("label_key")
        if not label_key:
            raise ValueError("label router requires 'label_key'")
        return LabelCartridgeRouter(
            registry=registry,
            label_key=label_key,
            extras_key=config.get("extras_key", "cartridge_label"),
        )

    if router_type == "static":
        cart_id = config.get("cartridge_id") or default_cartridge_id
        if not cart_id:
            raise ValueError(
                "static router requires 'cartridge_id' "
                "(or a default_cartridge_id)")
        return StaticCartridgeRouter(cart_id)

    if router_type == "composite":
        sub_configs = config.get("routers", [])
        if not sub_configs:
            raise ValueError("composite router requires 'routers' list")
        sub_routers = [
            build_router_from_config(
                sub, registry=registry,
                default_cartridge_id=default_cartridge_id,
            )
            for sub in sub_configs
        ]
        return CompositeRouter(sub_routers)

    raise ValueError(f"unknown router type: {router_type}")
