# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Thread-safe catalog operations close their SQLite connections eagerly."""

import importlib.util
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


@pytest.fixture
def catalog_type(monkeypatch):
    path = Path(__file__).parents[2] / "vllm/artifact_registry.py"
    spec = importlib.util.spec_from_file_location("catalog_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    connect = sqlite3.connect
    live = set()
    guard = threading.Lock()

    class Connection(sqlite3.Connection):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            with guard:
                live.add(id(self))

        def close(self):
            try:
                super().close()
            finally:
                with guard:
                    live.discard(id(self))

    def tracked_connect(*args, **kwargs):
        return connect(*args, factory=Connection, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", tracked_connect)
    return module.ArtifactRegistry, live


def test_connections_close_after_success_and_conflict(tmp_path, catalog_type):
    cls, live = catalog_type
    catalog = cls(tmp_path / "catalog.sqlite3")
    assert not live
    catalog.register_artifact("a", {"value": 1})
    catalog.register_artifact("a", {"value": 1})
    assert catalog.lookup_artifact("a") == {"value": 1}
    assert catalog.lookup_artifact("missing") is None
    assert not live
    with pytest.raises(ValueError, match="already bound"):
        catalog.register_artifact("a", {"value": 2})
    assert not live
    assert catalog.lookup_artifact("a") == {"value": 1}
    assert not live
    catalog.close()


def test_threaded_operations_leave_no_open_connections(tmp_path, catalog_type):
    cls, live = catalog_type
    catalog = cls(tmp_path / "catalog.sqlite3")
    catalog.register_artifact("a", {"value": 1})

    def request(_):
        assert catalog.lookup_artifact("a") == {"value": 1}

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(request, range(512)))
    assert not live
    catalog.close()
