"""
Run any employee as an A2A server, from its YAML file.

    python -m shared.employee_server researcher
    python -m shared.employee_server employees/researcher.yaml

Every employee has the same shape: Agent Card -> A2A server -> BaseA2AExecutor
-> ADK Runner -> LlmAgent -> tools. Only the YAML differs.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make `shared` importable when run as a script, not just with -m.
sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from google.adk.agents import LlmAgent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner

from shared.callbacks import FETCH_TOOLS, WEB_INTEL_CALLBACKS, WORKER_CALLBACKS
from shared.config import AGENT_HOST, LLM_PROVIDER, get_llm_model, missing_llm_credentials
from shared.employees import Employee, EmployeeConfigError, find
from shared.executor import BaseA2AExecutor
from shared.persistence import get_session_service
from shared.tool_registry import resolve_tools

logger = logging.getLogger(__name__)

_ROLES = {"worker", "orchestrator"}


def build_model(emp: Employee):
    """`model: worker|orchestrator` picks that role's model; anything else is
    taken as an explicit model name."""
    if emp.model in _ROLES:
        return get_llm_model(role=emp.model)
    return get_llm_model(emp.model)


def build_agent(emp: Employee) -> LlmAgent:
    """The ADK agent for an employee. An employee with web-fetching tools also
    gets the hard fetch budget (shared/callbacks.py)."""
    callbacks = WEB_INTEL_CALLBACKS if FETCH_TOOLS & set(emp.tools) else WORKER_CALLBACKS
    return LlmAgent(
        model=build_model(emp),
        name=emp.agent_name,
        description=emp.description,
        instruction=emp.instruction,
        tools=resolve_tools(emp.tools),
        **callbacks,
    )


def build_card(emp: Employee, host: str = AGENT_HOST) -> AgentCard:
    return AgentCard(
        name=emp.name,
        description=emp.description,
        url=f"http://{host}:{emp.port}/",
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[AgentSkill(
            id=emp.skill.id,
            name=emp.skill.name,
            description=emp.skill.description,
            tags=emp.skill.tags,
            examples=emp.skill.examples,
        )],
    )


def build_app(emp: Employee, host: str = AGENT_HOST):
    """The Starlette A2A app for an employee (not yet bound to a socket)."""
    card = build_card(emp, host)
    runner = Runner(
        app_name=card.name,
        agent=build_agent(emp),
        artifact_service=InMemoryArtifactService(),
        session_service=get_session_service(),
        memory_service=InMemoryMemoryService(),
    )
    handler = DefaultRequestHandler(
        agent_executor=BaseA2AExecutor(runner),
        task_store=InMemoryTaskStore(),
    )
    return A2AStarletteApplication(agent_card=card, http_handler=handler).build()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one Ergon employee as an A2A server.")
    parser.add_argument("employee", help="employee id (e.g. researcher) or path to its YAML")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    missing = missing_llm_credentials()
    if missing:
        logger.error(
            "LLM_PROVIDER=%s needs %s set in .env (or change LLM_PROVIDER).",
            LLM_PROVIDER, ", ".join(missing),
        )
        return 1
    try:
        emp = find(args.employee)
        app = build_app(emp)
    except (EmployeeConfigError, KeyError) as e:
        logger.error("Could not start %s: %s", args.employee, e)
        return 1

    logger.info("%s on http://%s:%s", emp.name, AGENT_HOST, emp.port)
    uvicorn.run(app, host=AGENT_HOST, port=emp.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
