"""Shared fixtures; this folder is on sys.path for them, so test files can `import kit`."""

from __future__ import annotations

import pytest

from histor.engine.store import EventStore


@pytest.fixture
def store(tmp_path):
    """An empty store in the test's own folder."""
    s = EventStore(tmp_path / "db.sqlite")
    yield s
    s.close()
