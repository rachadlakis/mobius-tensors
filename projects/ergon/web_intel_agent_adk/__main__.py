import logging
import os
import sys

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from agent import create_agent
from agent_executor import ResearcherExecutor
from dotenv import load_dotenv
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner

# Add shared module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.config import (
    AGENT_HOST,
    LLM_PROVIDER,
    WEB_INTEL_PORT,
    missing_llm_credentials,
)
from shared.persistence import get_session_service

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingAPIKeyError(Exception):
    """Exception for missing API key."""
    pass


def main():
    """Starts the Researcher server."""
    host = AGENT_HOST
    port = WEB_INTEL_PORT
    try:
        missing = missing_llm_credentials()
        if missing:
            raise MissingAPIKeyError(
                f"LLM_PROVIDER={LLM_PROVIDER} needs {', '.join(missing)} set in .env "
                "(or change LLM_PROVIDER)."
            )

        capabilities = AgentCapabilities(streaming=True)
        skill = AgentSkill(
            id="web_research",
            name="Web Research",
            description=(
                "Researches the public web: searches (DuckDuckGo), reads pages, "
                "follows recent news and looks up weather (Open-Meteo). Markets, "
                "competitors, potential customers and partners, suppliers, "
                "technology, regulations and prices. Public pages only (no logins "
                "or paywalls), respects robots.txt, and cites its sources."
            ),
            tags=["web", "search", "research", "news", "competitors", "leads"],
            examples=[
                "Who are the main competitors in video analytics for retail in the GCC?",
                "Find 10 security system integrators in the UAE with their websites.",
                "What changed in the EU AI Act this month?",
                "Summarise the latest news about edge AI chips.",
                "What's the weather forecast for Dubai tomorrow?",
            ],
        )
        agent_card = AgentCard(
            name="Researcher",
            description=(
                "Researcher: web search and page reading for market, competitor, "
                "lead and technology questions. Read-only, public pages only, "
                "respects robots.txt, cites sources."
            ),
            url=f"http://{host}:{port}/",
            version="1.0.0",
            default_input_modes=["text/plain"],
            default_output_modes=["text/plain"],
            capabilities=capabilities,
            skills=[skill],
        )

        adk_agent = create_agent()
        runner = Runner(
            app_name=agent_card.name,
            agent=adk_agent,
            artifact_service=InMemoryArtifactService(),
            session_service=get_session_service(),
            memory_service=InMemoryMemoryService(),
        )
        agent_executor = ResearcherExecutor(runner)

        request_handler = DefaultRequestHandler(
            agent_executor=agent_executor,
            task_store=InMemoryTaskStore(),
        )
        server = A2AStarletteApplication(
            agent_card=agent_card, http_handler=request_handler
        )

        uvicorn.run(server.build(), host=host, port=port)
    except MissingAPIKeyError as e:
        logger.error(f"Error: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"An error occurred during server startup: {e}")
        exit(1)


if __name__ == "__main__":
    main()
