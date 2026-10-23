# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-independent catalog for immutable external context artifacts."""

import json
import sqlite3
from pathlib import Path


class ArtifactRegistry:
    """Store typed artifact records in the cartridge registry database.

    Payloads live in the storage backend. This catalog is safe to use from
    storage worker threads; each operation owns a separate connection.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS context_artifacts ("
                "id TEXT PRIMARY KEY, record TEXT NOT NULL)"
            )

    def register_artifact(self, artifact_id: str, record: dict) -> None:
        """Publish an immutable typed artifact record beside cartridge entries.

        The storage owner must verify and synchronize payload bytes first.
        Repeated publication must have identical identity and metadata.
        A separate connection allows reads from storage worker threads.
        """
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":"))
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT record FROM context_artifacts WHERE id=?", (artifact_id,)
            ).fetchone()
            if row is not None and row[0] != encoded:
                raise ValueError("artifact ID is already bound to another record")
            conn.execute(
                "INSERT OR IGNORE INTO context_artifacts VALUES (?, ?)",
                (artifact_id, encoded),
            )

    def lookup_artifact(self, artifact_id: str) -> dict | None:
        """Resolve an exact typed artifact ID without reading its payload."""
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT record FROM context_artifacts WHERE id=?", (artifact_id,)
            ).fetchone()
        return None if row is None else json.loads(row[0])

    def close(self) -> None:
        """Close the catalog; operations do not retain connections."""
