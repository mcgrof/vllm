# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The handoff manifest: what a stored KV object is and where it lives."""

from __future__ import annotations

import enum
import json
from dataclasses import asdict, dataclass, field


class ObjectState(str, enum.Enum):
    """Lifecycle of one stored object; transitions are the coordinator's."""

    RESERVED = "RESERVED"  # extents allocated, nothing written
    WRITING = "WRITING"  # producer I/O in flight
    READY = "READY"  # every write completed and the manifest is published
    READING = "READING"  # at least one consumer holds a lease
    ACKED = "ACKED"  # every consumer acknowledged consumption
    DRAINING = "DRAINING"  # cancelled or failed; waiting for I/O to cease
    RECLAIMABLE = "RECLAIMABLE"  # extents may be handed out again


@dataclass(frozen=True)
class Extent:
    """A byte range of the namespace, LBA aligned."""

    offset: int
    length: int

    @property
    def end(self) -> int:
        return self.offset + self.length


@dataclass
class LayerSlice:
    """Where one layer's blocks of the object live: one extent, K then V
    per block in request block order (a single region for MLA layers)."""

    layer_name: str
    extent: Extent
    bytes_per_block: int  # one K or one V region of one block
    regions_per_block: int  # 2 for separate K and V, 1 for MLA


@dataclass
class HandoffManifest:
    """Everything a consumer needs to load the object and prove it is the
    one it asked for. GPU block ids of the producer are deliberately absent:
    they are not storage keys."""

    session_epoch: str
    request_id: str
    transfer_id: str
    generation: int
    model_id: str
    kv_dtype: str
    kv_scales: str | None
    tp_rank: int
    tp_size: int
    block_size: int
    num_tokens: int  # valid tokens
    num_blocks: int  # blocks stored, the last possibly padded
    layout: str
    namespace_serial: str
    namespace_wwid: str
    producer: str
    layers: list[LayerSlice] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return sum(layer.extent.length for layer in self.layers)

    @property
    def extents(self) -> list[Extent]:
        return [layer.extent for layer in self.layers]

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> HandoffManifest:
        raw = json.loads(text)
        layers = [
            LayerSlice(
                layer_name=item["layer_name"],
                extent=Extent(**item["extent"]),
                bytes_per_block=item["bytes_per_block"],
                regions_per_block=item["regions_per_block"],
            )
            for item in raw.pop("layers")
        ]
        return cls(layers=layers, **raw)

    def same_identity(self, other: HandoffManifest) -> bool:
        """The fields a consumer must match before restoring bytes into
        its own KV cache."""
        return (
            self.session_epoch == other.session_epoch
            and self.model_id == other.model_id
            and self.kv_dtype == other.kv_dtype
            and self.kv_scales == other.kv_scales
            and self.tp_rank == other.tp_rank
            and self.tp_size == other.tp_size
            and self.block_size == other.block_size
            and self.layout == other.layout
            and self.namespace_serial == other.namespace_serial
            and self.namespace_wwid == other.namespace_wwid
            and [
                (layer.layer_name, layer.bytes_per_block, layer.regions_per_block)
                for layer in self.layers
            ]
            == [
                (layer.layer_name, layer.bytes_per_block, layer.regions_per_block)
                for layer in other.layers
            ]
        )
