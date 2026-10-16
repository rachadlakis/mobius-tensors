"""Unit tests for shared/persistence.py — backend selection and URL rewriting."""

import logging

import pytest
from google.adk.sessions import InMemorySessionService

from shared import persistence


# --------------------------------------------------------------------------- #
# _async_db_url
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("sqlite:///./x.db", "sqlite+aiosqlite:///./x.db"),
        ("sqlite:////abs/path.db", "sqlite+aiosqlite:////abs/path.db"),
        ("postgresql://u:p@h:5432/db", "postgresql+asyncpg://u:p@h:5432/db"),
        ("postgres://u:p@h/db", "postgresql+asyncpg://u:p@h/db"),
        ("mysql://u:p@h/db", "mysql+aiomysql://u:p@h/db"),
        # already async / already has a driver -> untouched
        ("sqlite+aiosqlite:///./x.db", "sqlite+aiosqlite:///./x.db"),
        ("postgresql+asyncpg://u:p@h/db", "postgresql+asyncpg://u:p@h/db"),
        ("postgresql+psycopg://u:p@h/db", "postgresql+psycopg://u:p@h/db"),
    ],
)
def test_async_db_url_rewrites_sync_drivers(raw, expected):
    assert persistence._async_db_url(raw) == expected


def test_async_db_url_leaves_unknown_scheme_alone():
    assert persistence._async_db_url("weirddb://host/x") == "weirddb://host/x"


# --------------------------------------------------------------------------- #
# _build_session_service — backend selection
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("alias", ["memory", "in_memory", "inmemory", "in-memory"])
def test_memory_aliases_build_in_memory_service(monkeypatch, alias):
    monkeypatch.setattr(persistence, "SESSION_BACKEND", alias)
    svc = persistence._build_session_service()
    assert isinstance(svc, InMemorySessionService)


@pytest.mark.parametrize("alias", ["database", "db", "sqlite", "postgres", "postgresql"])
def test_database_aliases_build_database_service(monkeypatch, tmp_path, alias):
    from google.adk.sessions import DatabaseSessionService

    monkeypatch.setattr(persistence, "SESSION_BACKEND", alias)
    monkeypatch.setattr(
        persistence, "SESSION_DB_URL", f"sqlite:///{tmp_path / 'sel.db'}"
    )
    svc = persistence._build_session_service()
    assert isinstance(svc, DatabaseSessionService)


def test_unknown_backend_falls_back_to_memory_with_warning(monkeypatch, caplog):
    monkeypatch.setattr(persistence, "SESSION_BACKEND", "banana")
    with caplog.at_level(logging.WARNING):
        svc = persistence._build_session_service()
    assert isinstance(svc, InMemorySessionService)
    assert any("banana" in r.message for r in caplog.records)


def test_database_backend_upgrades_plain_sqlite_url(monkeypatch, tmp_path):
    """A bare sqlite:// URL must be handed to ADK as sqlite+aiosqlite://."""
    captured = {}

    class _Spy:
        def __init__(self, db_url):
            captured["db_url"] = db_url

    monkeypatch.setattr(persistence, "SESSION_BACKEND", "database")
    monkeypatch.setattr(persistence, "SESSION_DB_URL", f"sqlite:///{tmp_path / 'up.db'}")
    monkeypatch.setattr(
        "google.adk.sessions.DatabaseSessionService", _Spy, raising=True
    )
    persistence._build_session_service()
    assert captured["db_url"].startswith("sqlite+aiosqlite:///")


# --------------------------------------------------------------------------- #
# get_session_service — caching
# --------------------------------------------------------------------------- #
def test_get_session_service_is_cached(reset_persistence_cache, monkeypatch):
    monkeypatch.setattr(persistence, "SESSION_BACKEND", "memory")
    first = persistence.get_session_service()
    second = persistence.get_session_service()
    assert first is second


def test_cache_reset_yields_new_instance(reset_persistence_cache, monkeypatch):
    monkeypatch.setattr(persistence, "SESSION_BACKEND", "memory")
    first = persistence.get_session_service()
    persistence._session_service = None
    assert persistence.get_session_service() is not first
