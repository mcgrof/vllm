# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
CartridgeRegistry: SQLite-backed index of cartridge manifests.

The registry answers one question: "which cartridge should this
request use?" It indexes CartridgeManifest entries by exact keys
(cartridge_id, labels) and returns the manifest for a matching
cartridge.

Usage:
    registry = CartridgeRegistry("/path/to/cartridges.db")
    registry.register(manifest)
    manifest = registry.lookup(cartridge_id="patient_04_longhealth")
    manifests = registry.lookup_by_label("patient_id", "patient_04")
    all_manifests = registry.list_all()

The registry stores manifest JSON in SQLite. It does NOT store
cartridge KV data — that is handled by the CartridgeStore (future).
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_manifest import (
    CartridgeManifest,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cartridges (
    cartridge_id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    num_layers INTEGER NOT NULL,
    num_kv_heads INTEGER NOT NULL,
    head_dim INTEGER NOT NULL,
    dtype TEXT NOT NULL,
    num_tokens_raw INTEGER NOT NULL,
    num_tokens_aligned INTEGER NOT NULL,
    block_size INTEGER NOT NULL,
    num_blocks INTEGER NOT NULL,
    has_frozen_prefix INTEGER NOT NULL,
    num_frozen_tokens INTEGER NOT NULL DEFAULT 0,
    checksum TEXT NOT NULL DEFAULT '',
    file_size_bytes INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT '',
    training_steps INTEGER NOT NULL DEFAULT 0,
    source_document TEXT NOT NULL DEFAULT '',
    labels_json TEXT NOT NULL DEFAULT '{}',
    manifest_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_model_id ON cartridges(model_id);
"""


class CartridgeRegistry:
    """SQLite-backed cartridge manifest registry."""

    def __init__(self, db_path: str | Path):
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self):
        self._conn.close()

    def register(self, manifest: CartridgeManifest) -> None:
        """Add or update a cartridge manifest in the registry."""
        self._conn.execute(
            """
            INSERT OR REPLACE INTO cartridges (
                cartridge_id, model_id, num_layers, num_kv_heads, head_dim,
                dtype, num_tokens_raw, num_tokens_aligned, block_size,
                num_blocks, has_frozen_prefix, num_frozen_tokens,
                checksum, file_size_bytes, created_at, training_steps,
                source_document, labels_json, manifest_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                manifest.cartridge_id,
                manifest.model_id,
                manifest.num_layers,
                manifest.num_kv_heads,
                manifest.head_dim,
                manifest.dtype,
                manifest.num_tokens_raw,
                manifest.num_tokens_aligned,
                manifest.block_size,
                manifest.num_blocks,
                int(manifest.has_frozen_prefix),
                manifest.num_frozen_tokens,
                manifest.checksum,
                manifest.file_size_bytes,
                manifest.created_at,
                manifest.training_steps,
                manifest.source_document,
                json.dumps(manifest.labels),
                json.dumps(manifest.to_dict()),
            ),
        )
        self._conn.commit()
        logger.info("Registered cartridge: %s", manifest.cartridge_id)

    def unregister(self, cartridge_id: str) -> bool:
        """Remove a cartridge from the registry. Returns True if found."""
        cursor = self._conn.execute(
            "DELETE FROM cartridges WHERE cartridge_id = ?",
            (cartridge_id,),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def lookup(self, cartridge_id: str) -> Optional[CartridgeManifest]:
        """Look up a cartridge by its unique ID."""
        row = self._conn.execute(
            "SELECT manifest_json FROM cartridges WHERE cartridge_id = ?",
            (cartridge_id,),
        ).fetchone()
        if row is None:
            return None
        return CartridgeManifest.from_dict(json.loads(row["manifest_json"]))

    def lookup_by_label(
        self, key: str, value: str
    ) -> list[CartridgeManifest]:
        """Find cartridges whose labels contain key=value.

        Uses JSON extraction on the labels_json column. Returns all
        matches (may be empty).
        """
        rows = self._conn.execute(
            """
            SELECT manifest_json FROM cartridges
            WHERE json_extract(labels_json, ?) = ?
            """,
            (f"$.{key}", value),
        ).fetchall()
        return [
            CartridgeManifest.from_dict(json.loads(r["manifest_json"]))
            for r in rows
        ]

    def lookup_by_model(self, model_id: str) -> list[CartridgeManifest]:
        """Find all cartridges for a given model."""
        rows = self._conn.execute(
            "SELECT manifest_json FROM cartridges WHERE model_id = ?",
            (model_id,),
        ).fetchall()
        return [
            CartridgeManifest.from_dict(json.loads(r["manifest_json"]))
            for r in rows
        ]

    def list_all(self) -> list[CartridgeManifest]:
        """Return all registered cartridges."""
        rows = self._conn.execute(
            "SELECT manifest_json FROM cartridges"
        ).fetchall()
        return [
            CartridgeManifest.from_dict(json.loads(r["manifest_json"]))
            for r in rows
        ]

    def count(self) -> int:
        """Return the number of registered cartridges."""
        row = self._conn.execute(
            "SELECT COUNT(*) as n FROM cartridges"
        ).fetchone()
        return row["n"]
