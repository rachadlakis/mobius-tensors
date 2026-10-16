"""
Smoke tests: each employee builds a valid ADK agent with the right tools.

No LLM is contacted - LlmAgent construction is lazy.
"""

import importlib

from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool import McpToolset


def test_data_analyst_has_document_tools():
    mod = importlib.import_module("data_agent_adk.agent")
    from shared.documents import DOCUMENT_TOOLS

    agent = mod.create_agent()
    assert isinstance(agent, LlmAgent)
    assert agent.name == "Data_Analyst"
    assert agent.tools == list(DOCUMENT_TOOLS)


def test_researcher_has_web_tools_and_no_mcp():
    """The Researcher is web-only: no MCP toolset, just the web_tools."""
    mod = importlib.import_module("web_intel_agent_adk.agent")
    from shared.web_tools import WEB_INTEL_TOOLS

    agent = mod.create_agent()
    assert isinstance(agent, LlmAgent)
    assert agent.name == "Researcher"
    assert agent.tools == list(WEB_INTEL_TOOLS)
    assert not any(isinstance(t, McpToolset) for t in agent.tools)
