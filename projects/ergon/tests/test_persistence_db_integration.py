"""
Integration tests for the database session backend.

These exercise a real ADK DatabaseSessionService against a throwaway SQLite file
(via aiosqlite) — create/get/list/delete sessions, state round-tripping, and the
key property the DB backend exists for: state survives a fresh service instance
pointed at the same file.
"""

import pytest

from shared import persistence

pytestmark = pytest.mark.asyncio


@pytest.fixture
def db_service(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'integration.db'}"
    monkeypatch.setattr(persistence, "SESSION_BACKEND", "database")
    monkeypatch.setattr(persistence, "SESSION_DB_URL", db_url)
    return persistence._build_session_service(), db_url


async def test_create_and_get_session_roundtrip(db_service):
    svc, _ = db_service
    created = await svc.create_session(app_name="App", user_id="u1")
    fetched = await svc.get_session(
        app_name="App", user_id="u1", session_id=created.id
    )
    assert fetched is not None
    assert fetched.id == created.id
    assert fetched.app_name == "App"
    assert fetched.user_id == "u1"


async def test_get_missing_session_returns_none(db_service):
    svc, _ = db_service
    assert (
        await svc.get_session(app_name="App", user_id="u1", session_id="missing")
        is None
    )


async def test_initial_state_is_persisted(db_service):
    svc, _ = db_service
    created = await svc.create_session(
        app_name="App", user_id="u1", state={"role": "admin", "count": 3}
    )
    fetched = await svc.get_session(
        app_name="App", user_id="u1", session_id=created.id
    )
    assert fetched.state == {"role": "admin", "count": 3}


async def test_list_sessions_scoped_to_user(db_service):
    svc, _ = db_service
    await svc.create_session(app_name="App", user_id="u1")
    await svc.create_session(app_name="App", user_id="u1")
    await svc.create_session(app_name="App", user_id="u2")

    u1 = await svc.list_sessions(app_name="App", user_id="u1")
    u2 = await svc.list_sessions(app_name="App", user_id="u2")
    assert len(u1.sessions) == 2
    assert len(u2.sessions) == 1


async def test_delete_session(db_service):
    svc, _ = db_service
    created = await svc.create_session(app_name="App", user_id="u1")
    await svc.delete_session(
        app_name="App", user_id="u1", session_id=created.id
    )
    assert (
        await svc.get_session(
            app_name="App", user_id="u1", session_id=created.id
        )
        is None
    )


async def test_state_survives_a_new_service_instance(tmp_path, monkeypatch):
    """The whole point of the DB backend: restart-safety."""
    db_url = f"sqlite:///{tmp_path / 'restart.db'}"
    monkeypatch.setattr(persistence, "SESSION_BACKEND", "database")
    monkeypatch.setattr(persistence, "SESSION_DB_URL", db_url)

    svc1 = persistence._build_session_service()
    created = await svc1.create_session(
        app_name="App", user_id="u1", state={"greeting": "hello"}
    )

    # Simulate a process restart: brand-new service, same file.
    svc2 = persistence._build_session_service()
    fetched = await svc2.get_session(
        app_name="App", user_id="u1", session_id=created.id
    )
    assert fetched is not None
    assert fetched.state["greeting"] == "hello"


async def test_in_memory_backend_does_not_share_across_instances(monkeypatch):
    monkeypatch.setattr(persistence, "SESSION_BACKEND", "memory")
    svc1 = persistence._build_session_service()
    created = await svc1.create_session(app_name="App", user_id="u1")

    svc2 = persistence._build_session_service()
    assert (
        await svc2.get_session(
            app_name="App", user_id="u1", session_id=created.id
        )
        is None
    )
