import asyncio
import base64
import binascii
import inspect
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterable, List

# Remote agent replies can contain arbitrary Unicode (emoji, non-Latin text);
# on Windows, stdout defaults to the legacy console codepage (e.g. cp1252),
# which raises UnicodeEncodeError on those characters and crashes mid-request.
# Non-terminal streams (e.g. Jupyter's ipykernel stream, used when this agent
# is run via adk web/notebook) don't expose reconfigure(), so probe for it.
for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if callable(_reconfigure) and getattr(_stream, "encoding", "").lower() != "utf-8":
        _reconfigure(encoding="utf-8", errors="replace")

import httpx
import nest_asyncio
from a2a.client import A2ACardResolver
from a2a.client.errors import (
    A2AClientError,
    A2AClientHTTPError,
    A2AClientTimeoutError,
)
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
    SendMessageResponse,
    SendMessageSuccessResponse,
    Task,
)
from dotenv import load_dotenv
from google.adk import Agent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from .remote_agent_connection import RemoteAgentConnections

# Load .env from parent directories
parent_dir = Path(__file__).parent.parent.parent
load_dotenv(parent_dir / ".env")
nest_asyncio.apply()

# Add shared module to path
sys.path.insert(0, str(parent_dir))
from shared import a2a_files, attachments
from shared.callbacks import WORKER_CALLBACKS
from shared.config import (
    A2A_HEALTHCHECK_TIMEOUT_SECONDS,
    A2A_MAX_RETRIES,
    A2A_RETRY_BACKOFF_SECONDS,
    agent_url,
    get_llm_model,
    require_llm_credentials,
)
from shared.persistence import get_session_service

# Fail fast (and clearly) if the configured LLM_PROVIDER has no credentials.
require_llm_credentials()

# The orchestrator makes the routing/delegation decisions, so it runs on the
# capable ORCHESTRATOR_MODEL; the workers run bounded tasks on a cheaper
# WORKER_MODEL.
CLAUDE_MODEL = get_llm_model(role="orchestrator")


# --------------------------------------------------------------------------- #
# Files between agents (hub-and-spoke: everything passes through the orchestrator)
#
# The orchestrator keeps a per-session file store in `tool_context.state`
# ("session_files"): every file the user uploaded in `adk web`, plus every file
# a worker produced and returned as an A2A artifact. On each delegation those
# files are forwarded to the chosen worker as A2A message parts, so Worker B can
# use a file Worker A made. On the way back, files a worker returned are saved to
# the orchestrator's own artifact store (downloadable in adk web) and replaced,
# in the LLM's view, with a short note (plus an inline preview for images) - the
# base64 bytes never enter the model context.
# --------------------------------------------------------------------------- #
_MAX_FORWARDED_FILE_BYTES = 5 * 1024 * 1024      # per delegation, all files summed
_MAX_SESSION_FILE_BYTES = 5 * 1024 * 1024        # largest single file we retain
_MAX_INLINE_IMAGE_BYTES = 3 * 1024 * 1024        # largest image shown to the LLM

_MIME_EXT = {
    "text/csv": ".csv",
    "text/plain": ".txt",
    "text/tab-separated-values": ".tsv",
    "application/json": ".json",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/pdf": ".pdf",
    "image/png": ".png",
    "image/jpeg": ".jpg",
}


def _blob_name(blob: Any, index: int) -> str:
    """Best-effort filename for an inline blob (adk web sets display_name)."""
    name = getattr(blob, "display_name", None)
    if name:
        return str(name)
    ext = _MIME_EXT.get(getattr(blob, "mime_type", "") or "") or ".bin"
    return f"upload_{index + 1}{ext}"


def _safe_name(name: Any) -> str:
    """A basename with no path separators."""
    return Path(str(name or "")).name.strip()


def _session_files(tool_context: ToolContext) -> list[dict]:
    return tool_context.state.get("session_files") or []


def _store_session_file(
    tool_context: ToolContext, name: str, mime: str, raw: bytes, source: str
) -> None:
    """Add/replace a file in the per-session store (deduped by name)."""
    if len(raw) > _MAX_SESSION_FILE_BYTES:
        print(f"send_message: {name!r} is {a2a_files.human_size(len(raw))}, not retained")
        return
    files = [f for f in _session_files(tool_context) if f.get("name") != name]
    files.append({
        "name": name,
        "mime": mime or "application/octet-stream",
        "b64": base64.b64encode(raw).decode("ascii"),
        "source": source,  # "user" or the worker agent name
    })
    tool_context.state["session_files"] = files


def _remember_uploads(tool_context: ToolContext) -> None:
    """Stash any files attached to the current user turn into the session store."""
    content = getattr(tool_context, "user_content", None)
    for i, part in enumerate(getattr(content, "parts", None) or []):
        blob = getattr(part, "inline_data", None)
        if not blob or not getattr(blob, "data", None):
            continue
        raw = blob.data if isinstance(blob.data, bytes) else str(blob.data).encode()
        _store_session_file(
            tool_context, _blob_name(blob, i),
            getattr(blob, "mime_type", "") or "application/octet-stream",
            raw, source="user",
        )


def _delegation_parts(agent_name: str, task: str, tool_context: ToolContext) -> list[dict]:
    """The A2A message parts to send a worker: the task text + every session file
    (spreadsheets/CSVs rendered as CSV text, everything else as a raw file part)."""
    parts: list[dict] = [{"kind": "text", "text": task}]
    total = 0
    for f in _session_files(tool_context):
        if f.get("source") == agent_name:
            continue  # don't hand a file back to the agent that produced it
        raw = base64.b64decode(f["b64"])
        total += len(raw)
        if total > _MAX_FORWARDED_FILE_BYTES:
            print("send_message: hit the forwarded-file size cap, skipping the rest")
            break
        name, mime = f["name"], f["mime"]
        if attachments.looks_tabular(mime, name):
            text = None
            try:
                text = attachments.to_text(raw, mime, name)
            except Exception as e:  # a bad file must not kill the delegation
                print(f"send_message: could not render {name!r} as text: {e}")
            if text:
                parts.append({
                    "kind": "text",
                    "text": f"\n\n--- file: {name} ({mime}) ---\n{text}\n--- end of {name} ---",
                })
                continue
        parts.append(a2a_files.file_part_dict(name, raw, mime))
    return parts


async def _absorb_file_part(
    file_part: dict, agent_name: str, tool_context: ToolContext
) -> list:
    """One A2A file part a worker returned -> what the Orchestrator LLM should see.

    Side effects: the file is saved to the orchestrator's artifact service (so
    the adk web user can download it) and added to the session store so it can
    be forwarded onward. Returns a text note, plus an inline image Part for a
    small enough image.
    """
    f = file_part.get("file") or {}
    name = _safe_name(f.get("name")) or "output.bin"
    mime = f.get("mimeType") or f.get("mime_type") or "application/octet-stream"

    uri = f.get("uri")
    if uri and not f.get("bytes"):
        return [{"kind": "text", "text": f"[file from {agent_name}] {name} ({mime}) at {uri}"}]

    try:
        raw = base64.b64decode(f.get("bytes") or "", validate=True)
    except (binascii.Error, ValueError):
        raw = (f.get("bytes") or "").encode("utf-8")

    saved = False
    try:
        result = tool_context.save_artifact(
            name, types.Part(inline_data=types.Blob(data=raw, mime_type=mime))
        )
        if inspect.isawaitable(result):
            await result
        saved = True
    except Exception as e:  # no artifact service / backend error
        print(f"send_message: could not save artifact {name!r}: {e}")

    _store_session_file(tool_context, name, mime, raw, source=agent_name)

    note = (
        f"[file from {agent_name}] {name} ({mime}, {a2a_files.human_size(len(raw))})"
        + (" - saved, downloadable in the UI" if saved else " - not saved")
    )
    entries: list = [{"kind": "text", "text": note}]
    if mime.startswith("image/") and len(raw) <= _MAX_INLINE_IMAGE_BYTES:
        entries.append(types.Part(inline_data=types.Blob(data=raw, mime_type=mime)))
    return entries


# --------------------------------------------------------------------------- #
# Delegation resilience
#
# The orchestrator delegates over a single blocking A2A call. A worker that is
# slow, wedged, crashed, or restarting must not take down the orchestrator turn:
# `_send_to_worker` converts every failure into a short error string, which
# `send_message` hands back to the orchestrator LLM as an ordinary tool result
# so it can route around the outage (proceed with the other agents' answers,
# retry once, or tell the user which part is missing). Knobs in shared/config.py.
# --------------------------------------------------------------------------- #
_TRANSIENT_HTTP_STATUS = {502, 503, 504}


async def _worker_is_reachable(base_url: str) -> bool:
    """Fast liveness probe: GET the worker's agent-card. A dead / wedged worker
    then fails in seconds instead of blocking for the full request timeout.
    Returns True when the check is disabled or the card responds < 500."""
    if A2A_HEALTHCHECK_TIMEOUT_SECONDS <= 0 or not base_url:
        return True
    url = base_url.rstrip("/") + "/.well-known/agent-card.json"
    try:
        async with httpx.AsyncClient(timeout=A2A_HEALTHCHECK_TIMEOUT_SECONDS) as client:
            resp = await client.get(url)
        return resp.status_code < 500
    except Exception:
        return False


async def _send_to_worker(
    client: RemoteAgentConnections, message_request: SendMessageRequest, agent_name: str
) -> tuple[SendMessageResponse | None, str | None]:
    """Send a delegation to a worker, resiliently.

    Returns ``(response, None)`` on success, or ``(None, error_text)`` when the
    worker was unreachable, timed out, or errored - never raises.
    """
    base_url = getattr(getattr(client, "card", None), "url", "") or ""
    if not await _worker_is_reachable(base_url):
        return None, (
            f"{agent_name} is not responding to a health check "
            f"(checked for {A2A_HEALTHCHECK_TIMEOUT_SECONDS:.0f}s) - it looks "
            f"down or is restarting."
        )

    attempts = max(1, A2A_MAX_RETRIES + 1)
    last_err = "unknown error"
    for i in range(attempts):
        transient = False
        try:
            return await client.send_message(message_request), None
        except A2AClientTimeoutError:
            # Retrying would just time out again - stop here.
            return None, (
                f"{agent_name} did not finish within the time limit "
                f"(A2A_CLIENT_TIMEOUT_SECONDS). The task may be too big for one "
                f"call, or the agent is stuck."
            )
        except A2AClientHTTPError as e:
            last_err = f"HTTP {e.status_code} from {agent_name}"
            transient = e.status_code in _TRANSIENT_HTTP_STATUS
        except A2AClientError as e:
            last_err = f"{type(e).__name__} from {agent_name}: {e}"
        except Exception as e:  # noqa: BLE001 - a delegation must never crash the turn
            last_err = f"{type(e).__name__} talking to {agent_name}: {e}"

        if not transient or i == attempts - 1:
            break
        backoff = A2A_RETRY_BACKOFF_SECONDS * (2 ** i)
        print(f"send_message: transient error ({last_err}); retrying {agent_name} in {backoff:.1f}s")
        await asyncio.sleep(backoff)

    return None, f"Delegation to {agent_name} failed: {last_err}."


# Whether the orchestrator may reach out to the worker agents on its own. Off
# (ERGON_ORCHESTRATOR_AUTOINIT=0) for the unit tests, which inject fake
# connections by hand; on everywhere else.
_AUTO_CONNECT = os.getenv("ERGON_ORCHESTRATOR_AUTOINIT", "1") != "0"

_DEFAULT_WORKER_URLS = [
    agent_url("data"),
    agent_url("web_intel"),
]


class OrchestratorAgent:
    """The Chief of Staff: takes the user's request and delegates it to the employees."""

    def __init__(self, remote_agent_addresses: List[str] | None = None) -> None:
        self.remote_agent_connections: dict[str, RemoteAgentConnections] = {}
        self.cards: dict[str, AgentCard] = {}
        self.agents: str = ""
        # Worker URLs are remembered but NOT contacted here - the cards are
        # fetched lazily on the first turn (see _ensure_connected), so importing
        # this module and constructing the agent never does network I/O and
        # start order stops mattering.
        self._addresses: list[str] = list(
            remote_agent_addresses if remote_agent_addresses is not None
            else _DEFAULT_WORKER_URLS
        )
        self._connected_addresses: set[str] = set()
        self._connect_lock = asyncio.Lock()
        self._agent = self.create_agent()
        self._user_id = os.getenv("USER_ID", "demo_user_001")
        # session_service is in-memory or database-backed depending on
        # SESSION_BACKEND in .env (see shared/persistence.py).
        self._runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),
            session_service=get_session_service(),
            memory_service=InMemoryMemoryService(),
        )

    async def _connect(self, addresses: List[str]) -> None:
        """Fetch each worker's agent-card and register a connection. A worker
        that can't be reached is logged and skipped - it will be retried on the
        next turn."""
        async with httpx.AsyncClient(timeout=30) as client:
            for address in addresses:
                try:
                    card = await A2ACardResolver(client, address).get_agent_card()
                except httpx.ConnectError as e:
                    print(f"ERROR: Failed to get agent card from {address}: {e}")
                    continue
                except Exception as e:
                    print(f"ERROR: Failed to initialize connection for {address}: {e}")
                    continue
                self.remote_agent_connections[card.name] = RemoteAgentConnections(
                    agent_card=card, agent_url=address
                )
                self.cards[card.name] = card
                self._connected_addresses.add(address)
        self._refresh_agent_index()

    def _refresh_agent_index(self) -> None:
        """Rebuild the `agents` blob the root instruction shows the LLM."""
        agent_info = [
            json.dumps({
                "name": card.name,
                "description": card.description,
                "skills": [
                    {
                        "name": skill.name,
                        "description": skill.description,
                        "examples": skill.examples,
                    }
                    for skill in card.skills
                ],
            })
            for card in self.cards.values()
        ]
        print("Connected agents:", agent_info)
        self.agents = "\n".join(agent_info) if agent_info else "No agents found"

    async def _ensure_connected(self) -> None:
        """Connect to any worker not yet reached. The first call does the
        initial card fetch; later calls retry workers that were down. A no-op
        once every address is connected, or when auto-connect is disabled."""
        if not _AUTO_CONNECT:
            return
        if all(a in self._connected_addresses for a in self._addresses):
            return
        async with self._connect_lock:
            pending = [a for a in self._addresses if a not in self._connected_addresses]
            if pending:
                await self._connect(pending)

    async def _async_init_components(self, remote_agent_addresses: List[str]) -> None:
        """Eager connect - used by the async `create()` constructor and callers
        that want the cards fetched up front rather than on first use."""
        self._addresses = list(remote_agent_addresses)
        self._connected_addresses.clear()
        self.remote_agent_connections.clear()
        self.cards.clear()
        await self._connect(self._addresses)
        if not self.cards:
            print(
                "WARNING: Orchestrator started with zero connected agents. "
                "Make sure the employees are running (python run_all.py) "
                "before sending requests."
            )

    @classmethod
    async def create(cls, remote_agent_addresses: List[str]):
        instance = cls(remote_agent_addresses)
        await instance._async_init_components(remote_agent_addresses)
        return instance

    def create_agent(self) -> Agent:
        return Agent(
            model=CLAUDE_MODEL,
            name="Chief_of_Staff",
            instruction=self.root_instruction,
            description="Chief of Staff: splits a request into tasks, delegates them to the right AI employees and merges their work.",
            tools=[self.send_message],
            # Connect to the workers on the first turn (not at import), then log
            # every delegation + token usage per turn. See shared/callbacks.py.
            before_agent_callback=self._connect_before_run,
            **WORKER_CALLBACKS,
        )

    async def _connect_before_run(self, callback_context) -> None:
        """before_agent_callback: make sure the worker cards are loaded before
        the LLM renders its instruction / picks an agent to delegate to."""
        await self._ensure_connected()
        return None

    def root_instruction(self, context: ReadonlyContext) -> str:
        return f"""
        **Role:** You are the Chief of Staff of a team of AI employees.
        The user is your manager. You take their request, split it into tasks, hand
        each task to the right employee, and return one finished result. You do not
        do the specialist work yourself.

        **Core Directives:**

        *   **Task Delegation:** Use the `send_message` tool to delegate tasks to the appropriate
            employee. Check each employee's `description` and `skills` (including `examples`)
            below to pick the right one BEFORE calling `send_message` - do not guess-and-check
            across employees.
        *   **Files:** `send_message` carries files for you automatically. Any
            file the user uploaded, or that a worker produced, is forwarded to
            whichever agent you delegate to next (spreadsheets/CSVs as CSV text,
            other files as bytes) - you do NOT copy file contents into the
            `task`, just say what to do with the file. When a worker returns a
            file you'll get a short `[file from ...]` note; it is already saved
            for the user to download, so just tell them it's ready - never try
            to paste a file's bytes back to the user.
        *   **Act, don't just announce.** Naming the agents you'll use or
            outlining your plan is NOT progress - it is worthless to the user on
            its own. NEVER end a turn with only a plan, an intention, or a
            "here's how I'll approach this" message. If you describe a step, you
            MUST issue the matching `send_message` call in the SAME turn. The
            only turn you may end without a further `send_message` is one where
            every part of the request already has its answer (or a stated
            failure) and you are delivering the final consolidated result.
        *   **Multi-part requests:** break the request into its parts, then
            delegate EVERY part with `send_message` before you reply. Independent
            parts (e.g. analysing an uploaded spreadsheet and unrelated web research) should be
            sent in the same step so they run in parallel; a part that needs an
            earlier part's output waits for it. Keep delegating until no part is
            left unanswered. Then synthesize once.
        *   **Conditional / gated steps:** when the user says "if it's a GO, also
            do X" (or any "if ... then ..."), you evaluate the condition yourself
            from the gathered results and, if it holds, carry out X in the same
            run via the appropriate agent - do not stop and hand the decision
            back to the user unless the request is genuinely ambiguous or the
            gated action is destructive and unclear in scope.
        *   **Don't ask permission to do what was already asked.** If the user
            requested the whole workflow, execute the whole workflow. No "shall I
            proceed?" / "let me know if you want me to continue" mid-task.
        *   **Analyze Responses:** Collect and synthesize responses from agents
            into ONE final answer that follows the structure the user asked for.
        *   **Transparent Communication:** Brief progress notes are fine, but a
            progress note never substitutes for the tool calls and never ends the
            turn - see "Act, don't just announce."
        *   **Tool Reliance:** Use available tools. Do not make assumptions.
        *   **A failed delegation is not a dead end.** If `send_message` comes
            back with a "⚠️ ..." error (an agent timed out, is down, or errored),
            do NOT abandon the whole request and do NOT retry that agent more
            than once. Deliver every part you CAN get from the other agents, then
            tell the user plainly which piece failed, which agent it needed, and
            that they can retry it. Never invent the missing answer yourself
            (e.g. never guess figures or facts an employee was meant to
            return).
        *   **Readability:** Respond in a concise, easy to read format.

        **Today's Date:** {datetime.now().strftime("%Y-%m-%d")}

        <Available Employees>
        {self.agents}
        </Available Employees>
        """

    async def stream(
        self, query: str, session_id: str
    ) -> AsyncIterable[dict[str, Any]]:
        """Streams the agent's response to a given query."""
        session = await self._runner.session_service.get_session(
            app_name=self._agent.name,
            user_id=self._user_id,
            session_id=session_id,
        )
        content = types.Content(role="user", parts=[types.Part.from_text(text=query)])
        if session is None:
            # ---------------------------------------------------------------------
            # TODO: PERSISTENT STORAGE - Add initial state for new sessions:
            #   state={
            #       "user_id": self._user_id,
            #       "conversation_history": [],
            #       "preferences": {},
            #   }
            # This state will be auto-saved to DB when using DatabaseSessionService
            # ---------------------------------------------------------------------
            session = await self._runner.session_service.create_session(
                app_name=self._agent.name,
                user_id=self._user_id,
                state={},  # TODO: Add initial state (see above)
                session_id=session_id,
            )
        async for event in self._runner.run_async(
            user_id=self._user_id, session_id=session.id, new_message=content
        ):
            if event.is_final_response():
                response = ""
                if (
                    event.content
                    and event.content.parts
                    and event.content.parts[0].text
                ):
                    response = "\n".join(
                        [p.text for p in event.content.parts if p.text]
                    )
                yield {
                    "is_task_complete": True,
                    "content": response,
                }
            else:
                yield {
                    "is_task_complete": False,
                    "updates": "Chief of Staff is coordinating...",
                }

    async def send_message(self, agent_name: str, task: str, tool_context: ToolContext):
        """Sends a task to a remote agent."""
        if agent_name not in self.remote_agent_connections:
            # A worker may have come up since startup - try (re)connecting once.
            await self._ensure_connected()
        if agent_name not in self.remote_agent_connections:
            raise ValueError(f"Agent {agent_name} not found")
        client = self.remote_agent_connections[agent_name]

        if not client:
            raise ValueError(f"Client not available for {agent_name}")

        # ---------------------------------------------------------------------
        # PERSISTENCE NOTE: tool_context.state is automatically persisted
        # when using DatabaseSessionService. You can store data like:
        #   tool_context.state["last_agent_called"] = agent_name
        #   tool_context.state["call_history"] = [...]
        # This data will survive application restarts!
        # ---------------------------------------------------------------------
        # Task/context IDs are per-remote-agent: each agent has its own task
        # store, so a task_id from one agent is meaningless (and rejected) by
        # another. Only include taskId once that agent has actually returned
        # one for us to continue; a brand-new task must omit it entirely or
        # the server responds with TaskNotFoundError.
        state = tool_context.state
        task_id = state.get(f"task_id:{agent_name}")
        context_id = state.get(f"context_id:{agent_name}", str(uuid.uuid4()))
        message_id = str(uuid.uuid4())

        # Pick up any files attached to the current user turn, then forward the
        # whole session file set (user uploads + files earlier workers produced)
        # to this worker alongside the task text.
        _remember_uploads(tool_context)
        parts = _delegation_parts(agent_name, task, tool_context)
        if len(parts) > 1:
            print(f"send_message: forwarding {len(parts) - 1} file part(s) to {agent_name}")

        message: dict = {
            "role": "user",
            "parts": parts,
            "messageId": message_id,
            "contextId": context_id,
        }
        if task_id:
            message["taskId"] = task_id

        message_request = SendMessageRequest(
            id=message_id, params=MessageSendParams.model_validate({"message": message})
        )
        send_response, delegation_error = await _send_to_worker(
            client, message_request, agent_name
        )
        if delegation_error is not None or send_response is None:
            error = delegation_error or f"{agent_name} returned no response."
            print(f"send_message: {error}")
            return [{
                "kind": "text",
                "text": (
                    f"⚠️ {error}\n\n"
                    f"This sub-task did not complete. Do NOT retry {agent_name} "
                    f"more than once. If the user's request can still be partly "
                    f"answered by other agents, continue with their results and "
                    f"state clearly which part (handled by {agent_name}) is "
                    f"missing and why."
                ),
            }]
        # Don't print the whole response - a file reply carries base64 bytes.
        print(f"send_message: {agent_name} responded ({type(send_response.root).__name__})")

        if not isinstance(
            send_response.root, SendMessageSuccessResponse
        ) or not isinstance(send_response.root.result, Task):
            print("Received a non-success or non-task response. Cannot proceed.")
            return [{
                "kind": "text",
                "text": f"Error: no usable response from {agent_name} "
                        f"(non-success or non-task response from its A2A server).",
            }]

        result_task = send_response.root.result
        # A task in a terminal state (completed/canceled/failed/rejected) can't
        # accept follow-up messages - the remote agent's task store rejects
        # them. Only keep the task_id around for reuse if it's still open
        # (submitted/working/input-required/auth-required); otherwise the next
        # call to this agent must start a fresh task, or every second message
        # to the same agent within a session fails with "task in terminal
        # state". context_id is a conversation-level correlator that persists
        # across tasks, so it's kept either way.
        TERMINAL_TASK_STATES = {"completed", "canceled", "failed", "rejected"}
        if result_task.status.state not in TERMINAL_TASK_STATES:
            state[f"task_id:{agent_name}"] = result_task.id
        else:
            # ADK's tool_context.state doesn't support pop(); None is falsy so
            # the `if task_id:` check above naturally omits it next call.
            state[f"task_id:{agent_name}"] = None
        state[f"context_id:{agent_name}"] = result_task.context_id

        response_content = send_response.root.model_dump_json(exclude_none=True)
        json_content = json.loads(response_content)

        # Text parts pass straight through. A file part a worker returned is
        # saved to the orchestrator's artifact store (+ remembered for onward
        # forwarding) and replaced with a short note - see _absorb_file_part.
        resp: list = []
        for artifact in json_content.get("result", {}).get("artifacts") or []:
            for part in artifact.get("parts") or []:
                if (part.get("kind") or part.get("type")) == "file":
                    resp.extend(await _absorb_file_part(part, agent_name, tool_context))
                else:
                    resp.append(part)
        return resp


# `adk web` / `adk api_server` import this module and expect a module-level
# `root_agent`. Building it does NO network I/O: the orchestrator fetches the
# worker agent-cards lazily on its first turn (and retries any that were down),
# so start order no longer matters and importing the module is cheap. Tests set
# ERGON_ORCHESTRATOR_AUTOINIT=0, which also disables that lazy connect (they
# inject fake connections by hand); root_agent stays None in that case.
root_agent = None
if _AUTO_CONNECT:
    print("Building Chief of Staff (employees connect on first request)...")
    root_agent = OrchestratorAgent().create_agent()
