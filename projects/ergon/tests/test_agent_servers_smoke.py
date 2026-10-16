"""
Smoke tests for the agent server entrypoints (`*/__main__.py`).

Each `main()` is run with `uvicorn.run` stubbed out, so the A2A Starlette app is
built but never bound to a socket. We then drive it with Starlette's TestClient:
no network, no LLM.

This is the only coverage of AgentCard construction, the Runner wiring (including
`session_service=get_session_service()`), and the A2A app for each agent.
"""

import importlib.util
import sys

import pytest
from starlette.testclient import TestClient

from tests.conftest import REPO_ROOT

AGENT_CARD_PATH = "/.well-known/agent-card.json"
DEPRECATED_CARD_PATH = "/.well-known/agent.json"

# (package dir, expected AgentCard name, expected skill id, config port attr)
AGENTS = [
    ("data_agent_adk", "Data Analyst", "data_analysis", "DATA_PORT"),
    ("web_intel_agent_adk", "Researcher", "web_research", "WEB_INTEL_PORT"),
]


@pytest.fixture
def load_main(monkeypatch):
    """Import an agent's __main__.py in isolation.

    The entrypoints use bare `from agent import ...` / `from agent_executor
    import ...`, so the agent's own directory has to lead sys.path and the
    cached `agent` / `agent_executor` modules must be dropped between agents.
    """

    def _load(pkg_dir):
        agent_dir = REPO_ROOT / pkg_dir
        monkeypatch.syspath_prepend(str(agent_dir))
        for name in ("agent", "agent_executor"):
            monkeypatch.delitem(sys.modules, name, raising=False)
        spec = importlib.util.spec_from_file_location(
            f"{pkg_dir}__main__", agent_dir / "__main__.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    return _load


def _build_app(mod, monkeypatch):
    captured = {}

    def _fake_run(app, **kwargs):
        captured["app"] = app
        captured["kwargs"] = kwargs

    monkeypatch.setattr(mod.uvicorn, "run", _fake_run)
    mod.main()
    return captured["app"], captured["kwargs"]


@pytest.mark.parametrize("pkg_dir, card_name, skill_id, port_attr", AGENTS)
def test_server_serves_agent_card(
    monkeypatch, load_main, pkg_dir, card_name, skill_id, port_attr
):
    mod = load_main(pkg_dir)
    app, run_kwargs = _build_app(mod, monkeypatch)

    client = TestClient(app)
    resp = client.get(AGENT_CARD_PATH)
    assert resp.status_code == 200

    card = resp.json()
    assert card["name"] == card_name
    assert card["capabilities"]["streaming"] is True
    assert card["defaultInputModes"] == ["text/plain"]
    assert [s["id"] for s in card["skills"]] == [skill_id]
    assert card["skills"][0]["examples"]  # examples are populated


@pytest.mark.parametrize("pkg_dir, card_name, skill_id, port_attr", AGENTS)
def test_server_binds_configured_host_and_port(
    monkeypatch, load_main, pkg_dir, card_name, skill_id, port_attr
):
    from shared import config

    mod = load_main(pkg_dir)
    _, run_kwargs = _build_app(mod, monkeypatch)

    assert run_kwargs["host"] == config.AGENT_HOST
    assert run_kwargs["port"] == getattr(config, port_attr)


@pytest.mark.parametrize("pkg_dir, card_name, skill_id, port_attr", AGENTS)
def test_card_url_matches_bind_target(
    monkeypatch, load_main, pkg_dir, card_name, skill_id, port_attr
):
    mod = load_main(pkg_dir)
    app, run_kwargs = _build_app(mod, monkeypatch)
    card = TestClient(app).get(AGENT_CARD_PATH).json()
    assert card["url"] == f"http://{run_kwargs['host']}:{run_kwargs['port']}/"


@pytest.mark.parametrize("pkg_dir, card_name, skill_id, port_attr", AGENTS)
def test_deprecated_card_path_still_served(
    monkeypatch, load_main, pkg_dir, card_name, skill_id, port_attr
):
    mod = load_main(pkg_dir)
    app, _ = _build_app(mod, monkeypatch)
    resp = TestClient(app).get(DEPRECATED_CARD_PATH)
    assert resp.status_code == 200
    assert resp.json()["name"] == card_name


@pytest.mark.parametrize("pkg_dir, card_name, skill_id, port_attr", AGENTS)
def test_unknown_path_is_404(monkeypatch, load_main, pkg_dir, card_name, skill_id, port_attr):
    mod = load_main(pkg_dir)
    app, _ = _build_app(mod, monkeypatch)
    assert TestClient(app).get("/does-not-exist").status_code == 404


@pytest.mark.parametrize("pkg_dir, card_name, skill_id, port_attr", AGENTS)
def test_jsonrpc_unknown_method_returns_error(
    monkeypatch, load_main, pkg_dir, card_name, skill_id, port_attr
):
    mod = load_main(pkg_dir)
    app, _ = _build_app(mod, monkeypatch)
    resp = TestClient(app).post(
        "/",
        json={"jsonrpc": "2.0", "id": "1", "method": "bogus/method", "params": {}},
    )
    body = resp.json()
    assert body.get("error") is not None
    assert body["error"]["code"] < 0
