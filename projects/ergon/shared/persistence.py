"""
Session storage for Ergon employees.

Every agent's ADK ``Runner`` needs a ``session_service``. This module hands one
back based on ``SESSION_BACKEND`` (see ``shared/config.py``):

    SESSION_BACKEND=memory     -> InMemorySessionService   (default)
    SESSION_BACKEND=database   -> DatabaseSessionService(SESSION_DB_URL)

Usage (in each agent's __main__.py / the orchestrator):

    from shared.persistence import get_session_service

    runner = Runner(
        app_name=agent_card.name,
        agent=adk_agent,
        artifact_service=InMemoryArtifactService(),
        session_service=get_session_service(),
        memory_service=InMemoryMemoryService(),
    )

The service is created once and cached, so if several runners are built in the
same process they share a single DB connection pool.
"""

import logging

from shared.config import SESSION_BACKEND, SESSION_DB_URL

logger = logging.getLogger(__name__)

_MEMORY_ALIASES = {"memory", "in_memory", "inmemory", "in-memory"}
_DATABASE_ALIASES = {"database", "db", "sqlite", "postgres", "postgresql"}

_session_service = None

# ADK's DatabaseSessionService runs on SQLAlchemy's async engine, which rejects
# sync drivers. Upgrade the common sync URLs to their async equivalent so a
# plain "sqlite:///..." / "postgresql://..." in .env just works.
_ASYNC_DRIVER_UPGRADES = {
    "sqlite://": "sqlite+aiosqlite://",
    "postgresql://": "postgresql+asyncpg://",
    "postgres://": "postgresql+asyncpg://",
    "mysql://": "mysql+aiomysql://",
}


def _async_db_url(url: str) -> str:
    """Rewrite a sync SQLAlchemy URL to use an async driver (leaves URLs that
    already specify a driver, e.g. 'sqlite+aiosqlite://', untouched)."""
    for sync_prefix, async_prefix in _ASYNC_DRIVER_UPGRADES.items():
        if url.startswith(sync_prefix):
            return async_prefix + url[len(sync_prefix):]
    return url


def _build_session_service():
    """Build the session service selected by SESSION_BACKEND."""
    backend = SESSION_BACKEND

    if backend in _DATABASE_ALIASES:
        from google.adk.sessions import DatabaseSessionService

        db_url = _async_db_url(SESSION_DB_URL)
        logger.info("Session backend: database (%s)", db_url)
        return DatabaseSessionService(db_url=db_url)

    if backend not in _MEMORY_ALIASES:
        logger.warning(
            "Unknown SESSION_BACKEND=%r; falling back to in-memory sessions.",
            backend,
        )
    else:
        logger.info("Session backend: in-memory (sessions are lost on restart)")

    from google.adk.sessions import InMemorySessionService

    return InMemorySessionService()


def get_session_service():
    """Return the process-wide session service (cached after first call)."""
    global _session_service
    if _session_service is None:
        _session_service = _build_session_service()
    return _session_service
