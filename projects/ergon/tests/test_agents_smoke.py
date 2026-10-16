"""
Smoke tests: each employee builds a valid ADK agent with the right tools.

No LLM is contacted - LlmAgent construction is lazy.
"""

from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool import McpToolset

from shared import callbacks as cb
from shared.employee_server import build_agent
from shared.employees import find
from shared.tool_registry import TOOLS


def test_data_analyst_has_python_and_document_tools():
    agent = build_agent(find("data_analyst"))
    assert isinstance(agent, LlmAgent)
    assert agent.name == "Data_Analyst"
    assert agent.tools[:3] == [TOOLS["run_python"], TOOLS["calculate"], TOOLS["write_spreadsheet"]]
    assert TOOLS["search_company_knowledge"] in agent.tools
    # no web tools -> no fetch budget
    assert agent.before_tool_callback == [cb.log_before_tool]


def test_researcher_has_web_tools_fetch_budget_and_no_mcp():
    agent = build_agent(find("researcher"))
    assert isinstance(agent, LlmAgent)
    assert agent.name == "Researcher"
    assert agent.tools == [
        TOOLS["web_search"], TOOLS["fetch_page"], TOOLS["get_weather"],
        TOOLS["search_company_knowledge"], TOOLS["read_company_document"],
    ]
    assert not any(isinstance(t, McpToolset) for t in agent.tools)
    assert agent.before_tool_callback == [cb.enforce_scrape_budget, cb.log_before_tool]
    assert agent.after_agent_callback is cb.clear_scrape_budget


def test_instruction_is_company_profile_then_the_yaml():
    emp = find("researcher")
    agent = build_agent(emp)
    text = agent.instruction(None)  # a provider: recomposed every turn
    assert text.startswith("## Your company: Example Co.")
    assert text.endswith(emp.instruction)
    assert "search_company_knowledge" in text
    assert agent.description == emp.description


def test_drafting_employees_cannot_reach_the_web_or_run_code():
    """Writer, Sales Rep and Support Agent draft from the company documents only."""
    for emp_id in ("writer", "sales_rep", "support_agent"):
        tools = set(find(emp_id).tools)
        assert {"search_company_knowledge", "read_company_document"} <= tools, emp_id
        assert not tools & {"web_search", "fetch_page", "run_python"}, emp_id


def test_every_employee_with_prices_has_the_calculator():
    for emp_id in ("data_analyst", "sales_rep", "support_agent"):
        assert "calculate" in find(emp_id).tools, emp_id


def test_engineer_codes_only_through_its_branch_tool():
    tools = set(find("engineer").tools)
    assert {"run_coding_task", "list_repositories"} <= tools
    assert not tools & {"run_python", "web_search", "fetch_page"}
