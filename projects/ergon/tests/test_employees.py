"""Tests for loading employees from YAML (shared/employees.py) and the tool
registry (shared/tool_registry.py)."""

import textwrap

import pytest

from shared import employee_server
from shared.config import AGENT_HOST
from shared.employees import EmployeeConfigError, find, load_all, load_employee
from shared.tool_registry import TOOLS, resolve_tools

VALID = """\
name: Sales Rep
port: 10006
description: Wins customers.
model: worker
tools: [write_pdf_report]
skill:
  id: sales
  name: Sales
  description: Prospecting and proposals.
  tags: [sales]
  examples: ["Draft a proposal for Acme."]
instruction: |
  **Role:** You are the Sales Rep.
"""


def _write(directory, name, text):
    path = directory / name
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


class TestLoadEmployee:
    def test_valid_file(self, tmp_path):
        emp = load_employee(_write(tmp_path, "sales_rep.yaml", VALID))
        assert emp.id == "sales_rep"
        assert emp.name == "Sales Rep"
        assert emp.agent_name == "Sales_Rep"
        assert emp.port == 10006
        assert emp.tools == ["write_pdf_report"]
        assert emp.skill.examples == ["Draft a proposal for Acme."]
        assert emp.instruction.startswith("**Role:**")
        assert emp.url == f"http://{AGENT_HOST}:10006"

    def test_model_defaults_to_worker(self, tmp_path):
        emp = load_employee(_write(tmp_path, "s.yaml", VALID.replace("model: worker\n", "")))
        assert emp.model == "worker"

    @pytest.mark.parametrize("missing", ["name", "description", "instruction", "skill"])
    def test_missing_required_field(self, tmp_path, missing):
        lines = VALID.splitlines(keepends=True)
        start = next(i for i, ln in enumerate(lines) if ln.startswith(f"{missing}:"))
        end = start + 1
        while end < len(lines) and lines[end].startswith("  "):
            end += 1
        text = "".join(lines[:start] + lines[end:])
        with pytest.raises(EmployeeConfigError, match=missing):
            load_employee(_write(tmp_path, "s.yaml", text))

    def test_bad_port(self, tmp_path):
        with pytest.raises(EmployeeConfigError, match="port"):
            load_employee(_write(tmp_path, "s.yaml", VALID.replace("10006", "soon")))

    def test_port_env_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SALES_REP_PORT", "20006")
        assert load_employee(_write(tmp_path, "sales_rep.yaml", VALID)).port == 20006

    def test_bad_file_name(self, tmp_path):
        with pytest.raises(EmployeeConfigError, match="file name"):
            load_employee(_write(tmp_path, "Sales-Rep.yaml", VALID))

    def test_tools_must_be_a_list_of_strings(self, tmp_path):
        text = VALID.replace("tools: [write_pdf_report]", "tools: write_pdf_report")
        with pytest.raises(EmployeeConfigError, match="tools"):
            load_employee(_write(tmp_path, "s.yaml", text))

    def test_invalid_yaml(self, tmp_path):
        with pytest.raises(EmployeeConfigError, match="YAML"):
            load_employee(_write(tmp_path, "s.yaml", "name: [unclosed\n"))

    def test_not_a_mapping(self, tmp_path):
        with pytest.raises(EmployeeConfigError, match="mapping"):
            load_employee(_write(tmp_path, "s.yaml", "- a\n- b\n"))


class TestLoadAll:
    def test_sorted_by_id(self, tmp_path):
        _write(tmp_path, "b.yaml", VALID.replace("Sales Rep", "B"))
        _write(tmp_path, "a.yaml", VALID.replace("Sales Rep", "A").replace("10006", "10007"))
        assert [e.id for e in load_all(tmp_path)] == ["a", "b"]

    def test_duplicate_port(self, tmp_path):
        _write(tmp_path, "a.yaml", VALID.replace("Sales Rep", "A"))
        _write(tmp_path, "b.yaml", VALID.replace("Sales Rep", "B"))
        with pytest.raises(EmployeeConfigError, match="port"):
            load_all(tmp_path)

    def test_duplicate_name(self, tmp_path):
        _write(tmp_path, "a.yaml", VALID)
        _write(tmp_path, "b.yaml", VALID.replace("10006", "10007"))
        with pytest.raises(EmployeeConfigError, match="name"):
            load_all(tmp_path)

    def test_empty_directory(self, tmp_path):
        assert load_all(tmp_path) == []

    def test_real_team_loads_and_every_tool_exists(self):
        for emp in load_all():
            resolve_tools(emp.tools)  # raises on an unknown tool


class TestFind:
    def test_by_id_and_by_path(self, tmp_path):
        path = _write(tmp_path, "sales_rep.yaml", VALID)
        assert find("sales_rep", tmp_path).name == "Sales Rep"
        assert find(str(path)).name == "Sales Rep"

    def test_unknown_lists_known(self, tmp_path):
        _write(tmp_path, "sales_rep.yaml", VALID)
        with pytest.raises(EmployeeConfigError, match="sales_rep"):
            find("nobody", tmp_path)


class TestToolRegistry:
    def test_resolves_in_order(self):
        assert resolve_tools(["fetch_page", "web_search"]) == [
            TOOLS["fetch_page"], TOOLS["web_search"],
        ]

    def test_unknown_tool_names_every_typo(self):
        with pytest.raises(KeyError, match="nope.*also_nope"):
            resolve_tools(["web_search", "nope", "also_nope"])

    def test_registry_names_match_function_names(self):
        assert all(fn.__name__ == name for name, fn in TOOLS.items())

    def test_unknown_tool_in_yaml_fails_at_build(self, tmp_path):
        emp = load_employee(_write(tmp_path, "s.yaml", VALID.replace(
            "tools: [write_pdf_report]", "tools: [send_rocket]")))
        with pytest.raises(KeyError, match="send_rocket"):
            employee_server.build_agent(emp)


class TestModelChoice:
    def test_role_names_pick_the_role_model(self, tmp_path):
        from shared import config
        emp = load_employee(_write(tmp_path, "s.yaml", VALID.replace(
            "model: worker", "model: orchestrator")))
        assert config.model_for_role("orchestrator") in employee_server.build_model(emp).model

    def test_explicit_model_name(self, tmp_path):
        emp = load_employee(_write(tmp_path, "s.yaml", VALID.replace(
            "model: worker", "model: openai/gpt-4o-mini")))
        assert employee_server.build_model(emp).model == "openai/gpt-4o-mini"
