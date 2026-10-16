import os
from typing import Callable

import httpx
from a2a.client import A2AClient
from a2a.types import (
    AgentCard,
    SendMessageRequest,
    SendMessageResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
)
from dotenv import load_dotenv

load_dotenv()

TaskCallbackArg = Task | TaskStatusUpdateEvent | TaskArtifactUpdateEvent
TaskUpdateCallback = Callable[[TaskCallbackArg, AgentCard], Task]


class RemoteAgentConnections:
    """A class to hold the connections to the remote agents."""

    def __init__(self, agent_card: AgentCard, agent_url: str):
        print(f"RemoteAgentConnections: {agent_card.name} @ {agent_url}")
        # A single employee request can chain many slow steps: for the Researcher,
        # a web search plus several page fetches, each with its own robots.txt
        # lookup and politeness delay, plus the LLM turns in between.
        # A multi-part research question ("weather AND competitors") easily runs
        # 2-4 minutes, so the read timeout is generous. Override with
        # A2A_CLIENT_TIMEOUT_SECONDS if needed.
        read_timeout = float(os.getenv("A2A_CLIENT_TIMEOUT_SECONDS", "300"))
        timeout = httpx.Timeout(read_timeout, connect=10.0, write=30.0, pool=read_timeout)
        self._httpx_client = httpx.AsyncClient(timeout=timeout)
        self.agent_client = A2AClient(self._httpx_client, agent_card, url=agent_url)
        self.card = agent_card
        self.conversation_name = None
        self.conversation = None
        self.pending_tasks: set[str] = set()

    def get_agent(self) -> AgentCard:
        return self.card

    async def send_message(
        self, message_request: SendMessageRequest
    ) -> SendMessageResponse:
        return await self.agent_client.send_message(message_request)
