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


def test_data_analyst_has_document_tools():
    agent = build_agent(find("data_analyst"))
    assert isinstance(agent, LlmAgent)
    assert agent.name == "Data_Analyst"
    assert agent.tools == [TOOLS["write_pdf_report"], TOOLS["list_workspace_files"]]
    # no web tools -> no fetch budget
    assert agent.before_tool_callback == [cb.log_before_tool]


def test_researcher_has_web_tools_fetch_budget_and_no_mcp():
    agent = build_agent(find("researcher"))
    assert isinstance(agent, LlmAgent)
    assert agent.name == "Researcher"
    assert agent.tools == [TOOLS["web_search"], TOOLS["fetch_page"], TOOLS["get_weather"]]
    assert not any(isinstance(t, McpToolset) for t in agent.tools)
    assert agent.before_tool_callback == [cb.enforce_scrape_budget, cb.log_before_tool]
    assert agent.after_agent_callback is cb.clear_scrape_budget


def test_instruction_and_description_come_from_the_yaml():
    emp = find("researcher")
    agent = build_agent(emp)
    assert agent.instruction == emp.instruction
    assert agent.description == emp.description
