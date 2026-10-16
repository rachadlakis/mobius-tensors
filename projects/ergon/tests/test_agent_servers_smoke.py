"""
Smoke tests for the employee servers (shared/employee_server.py).

Each employee's A2A Starlette app is built from its YAML and driven with
Starlette's TestClient: no network, no LLM. `main()` is run with `uvicorn.run`
stubbed out, so nothing binds to a socket.

This covers AgentCard construction, the Runner wiring (including
`session_service=get_session_service()`), and the A2A app for every employee.
"""

import pytest
from starlette.testclient import TestClient

from shared import employee_server
from shared.config import AGENT_HOST
from shared.employees import load_all

AGENT_CARD_PATH = "/.well-known/agent-card.json"
DEPRECATED_CARD_PATH = "/.well-known/agent.json"

EMPLOYEES = load_all()
IDS = [e.id for e in EMPLOYEES]


def test_the_team_is_what_we_expect():
    assert {e.id: (e.name, e.skill.id, e.port) for e in EMPLOYEES} == {
        "data_analyst": ("Data Analyst", "data_analysis", 10003),
        "researcher": ("Researcher", "web_research", 10005),
        "writer": ("Writer", "content_writing", 10006),
        "sales_rep": ("Sales Rep", "sales", 10007),
        "support_agent": ("Support Agent", "customer_support", 10008),
        "engineer": ("Engineer", "software_engineering", 10009),
    }


def test_only_file_tool_employees_take_files():
    from shared.a2a_files import FILE_INPUT_MODE

    takes_files = {e.id for e in EMPLOYEES
                   if FILE_INPUT_MODE in employee_server.build_card(e).default_input_modes}
    assert takes_files == {"data_analyst"}


@pytest.mark.parametrize("emp", EMPLOYEES, ids=IDS)
def test_server_serves_agent_card(emp):
    client = TestClient(employee_server.build_app(emp))
    resp = client.get(AGENT_CARD_PATH)
    assert resp.status_code == 200

    card = resp.json()
    assert card["name"] == emp.name
    assert card["description"] == emp.description
    assert card["capabilities"]["streaming"] is True
    assert card["defaultInputModes"][0] == "text/plain"
    assert [s["id"] for s in card["skills"]] == [emp.skill.id]
    assert card["skills"][0]["examples"]  # examples are populated
    assert card["url"] == f"http://{AGENT_HOST}:{emp.port}/"


@pytest.mark.parametrize("emp", EMPLOYEES, ids=IDS)
def test_deprecated_card_path_still_served(emp):
    resp = TestClient(employee_server.build_app(emp)).get(DEPRECATED_CARD_PATH)
    assert resp.status_code == 200
    assert resp.json()["name"] == emp.name


@pytest.mark.parametrize("emp", EMPLOYEES, ids=IDS)
def test_unknown_path_is_404(emp):
    assert TestClient(employee_server.build_app(emp)).get("/does-not-exist").status_code == 404


@pytest.mark.parametrize("emp", EMPLOYEES, ids=IDS)
def test_jsonrpc_unknown_method_returns_error(emp):
    resp = TestClient(employee_server.build_app(emp)).post(
        "/",
        json={"jsonrpc": "2.0", "id": "1", "method": "bogus/method", "params": {}},
    )
    body = resp.json()
    assert body.get("error") is not None
    assert body["error"]["code"] < 0


@pytest.mark.parametrize("emp", EMPLOYEES, ids=IDS)
def test_main_binds_configured_host_and_port(monkeypatch, emp):
    captured = {}
    monkeypatch.setattr(
        employee_server.uvicorn, "run",
        lambda app, **kw: captured.update(app=app, **kw),
    )
    assert employee_server.main([emp.id]) == 0
    assert captured["host"] == AGENT_HOST
    assert captured["port"] == emp.port


def test_main_unknown_employee_fails_cleanly(monkeypatch):
    monkeypatch.setattr(employee_server.uvicorn, "run", lambda *a, **k: pytest.fail("ran"))
    assert employee_server.main(["nobody"]) == 1


def test_main_missing_credentials_fails_cleanly(monkeypatch):
    monkeypatch.setattr(employee_server, "missing_llm_credentials", lambda: ["SOME_KEY"])
    monkeypatch.setattr(employee_server.uvicorn, "run", lambda *a, **k: pytest.fail("ran"))
    assert employee_server.main(["researcher"]) == 1
