"""
Shared pytest fixtures / test environment setup for Ergon.

These tests are all offline: no real web, no MCP server, no LLM calls.
Anything that would reach the network is monkeypatched in the individual tests.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# The system temp dir isn't writable in this environment; point pytest's tmp_path
# (and tempfile in general) at a repo-local, git-ignored directory.
_TMP_ROOT = REPO_ROOT / ".pytest-tmp"
_TMP_ROOT.mkdir(exist_ok=True)
tempfile.tempdir = str(_TMP_ROOT)
for _var in ("TMPDIR", "TEMP", "TMP"):
    os.environ[_var] = str(_TMP_ROOT)

# The agent packages do `sys.path.insert(0, ".."/parent)` at import time, but make
# the repo root importable up front so `import shared...` works from a clean run.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Deterministic, obviously-fake credentials so importing modules that read env at
# import time (config, orchestrator) never picks up real secrets and
# never tries to auto-initialise against live services.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-anthropic-key")
os.environ.setdefault("USER_ID", "test_user")
os.environ.setdefault("TENANT_ID", "test_tenant")
# Keep the orchestrator module importable without network round trips.
os.environ["ERGON_ORCHESTRATOR_AUTOINIT"] = "0"


@pytest.fixture
def reset_persistence_cache():
    """Clear the cached session service before and after a test."""
    from shared import persistence

    persistence._session_service = None
    yield persistence
    persistence._session_service = None


@pytest.fixture
def reload_config(monkeypatch):
    """Reload `shared.config` with a patched environment; restore afterwards.

    `load_dotenv` is stubbed during the reload so the patched env is the whole
    truth - otherwise the repo's real .env would re-populate vars a test just
    cleared (python-dotenv's load_dotenv doesn't override, but it *does* re-add
    anything a test deleted). Pass env vars as kwargs; value `None` deletes.
    """
    import importlib

    from shared import config

    # Config-controlled vars that .env may have loaded into the process before a
    # test ran: clear them so `reload_config(LLM_PROVIDER=...)` starts from a
    # clean slate and only the kwargs a test passes take effect.
    _managed = (
        "LLM_PROVIDER", "DEFAULT_MODEL", "ORCHESTRATOR_MODEL", "WORKER_MODEL",
        "ANTHROPIC_API_BASE",
        "AZURE_API_KEY", "AZURE_API_BASE", "AZURE_API_VERSION",
        "AZURE_AI_API_KEY", "AZURE_AI_API_BASE", "AZURE_AI_API_VERSION",
        "GEMINI_API_KEY",
    )

    def _reload(**env):
        for key in _managed:
            monkeypatch.delenv(key, raising=False)
        for key, value in env.items():
            if value is None:
                monkeypatch.delenv(key, raising=False)
            else:
                monkeypatch.setenv(key, value)
        monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
        return importlib.reload(config)

    yield _reload
    monkeypatch.undo()
    importlib.reload(config)
