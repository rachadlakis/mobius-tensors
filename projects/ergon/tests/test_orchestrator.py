"""
Tests for the Orchestrator agent.

conftest.py sets ERGON_ORCHESTRATOR_AUTOINIT=0 so importing the module does
not try to reach the worker agents. We drive OrchestratorAgent directly with a
fake remote connection.
"""

import base64
import importlib

import pytest
from a2a.types import (
    Artifact,
    FilePart,
    FileWithBytes,
    Part,
    SendMessageResponse,
    SendMessageSuccessResponse,
    Task,
    TaskState,
    TaskStatus,
    TextPart,
)
from google.adk.agents import BaseAgent

agent_mod = importlib.import_module("orchestrator_agent.orchestrator.agent")
OrchestratorAgent = agent_mod.OrchestratorAgent


def _task(state=TaskState.working, artifacts=None, task_id="task-1", context_id="ctx-1"):
    return Task(
        id=task_id,
        context_id=context_id,
        status=TaskStatus(state=state),
        artifacts=artifacts or [],
    )


def _success(task):
    return SendMessageResponse(
        root=SendMessageSuccessResponse(id="1", jsonrpc="2.0", result=task)
    )


class FakeConn:
    """Stand-in for RemoteAgentConnections."""

    def __init__(self, response):
        self.response = response
        self.requests = []

    async def send_message(self, message_request):
        self.requests.append(message_request)
        return self.response(self) if callable(self.response) else self.response


class FakeToolContext:
    def __init__(self, state=None, user_content=None):
        self.state = state if state is not None else {}
        self.user_content = user_content


def _turn_with_upload(data: bytes, mime_type: str, display_name=None):
    """A types.Content shaped like what adk web puts on a user turn."""
    from google.genai import types as genai_types

    blob = genai_types.Blob(data=data, mime_type=mime_type)
    if display_name is not None:
        blob.display_name = display_name
    return genai_types.Content(role="user", parts=[genai_types.Part(inline_data=blob)])


@pytest.fixture
def orch():
    return OrchestratorAgent()


# --------------------------------------------------------------------------- #
# Construction / instruction
# --------------------------------------------------------------------------- #
def test_module_import_did_not_autoinit():
    assert agent_mod.root_agent is None


def test_construction_does_no_network(orch):
    # __init__ remembers the worker URLs but must not contact them - the cards
    # are fetched lazily on the first turn (see test_lazy_connect_* below).
    assert orch.remote_agent_connections == {}
    assert orch._connected_addresses == set()
    assert len(orch._addresses) == 2  # the two employees, not yet connected


def test_create_agent_shape(orch):
    agent = orch.create_agent()
    assert isinstance(agent, BaseAgent)
    assert agent.name == "Chief_of_Staff"
    assert orch.send_message in agent.tools
    assert agent.before_agent_callback is not None  # lazy-connect hook


# --------------------------------------------------------------------------- #
# Lazy worker connection
# --------------------------------------------------------------------------- #
async def test_lazy_connect_on_first_use(orch, monkeypatch):
    monkeypatch.setattr(agent_mod, "_AUTO_CONNECT", True)
    orch._addresses = ["http://w1", "http://w2"]

    connected = []

    async def fake_connect(addresses):
        connected.extend(addresses)
        orch._connected_addresses.update(addresses)

    monkeypatch.setattr(orch, "_connect", fake_connect)

    await orch._ensure_connected()
    assert connected == ["http://w1", "http://w2"]

    connected.clear()
    await orch._ensure_connected()  # all connected -> no-op
    assert connected == []


async def test_ensure_connected_retries_only_the_missing_workers(orch, monkeypatch):
    monkeypatch.setattr(agent_mod, "_AUTO_CONNECT", True)
    orch._addresses = ["http://w1", "http://w2"]
    calls = []

    async def fake_connect(addresses):
        calls.append(list(addresses))
        if "http://w1" in addresses:  # w1 comes up, w2 stays down
            orch._connected_addresses.add("http://w1")

    monkeypatch.setattr(orch, "_connect", fake_connect)

    await orch._ensure_connected()
    await orch._ensure_connected()
    assert calls == [["http://w1", "http://w2"], ["http://w2"]]


async def test_ensure_connected_is_a_noop_when_auto_connect_is_off(orch, monkeypatch):
    monkeypatch.setattr(agent_mod, "_AUTO_CONNECT", False)
    called = []
    monkeypatch.setattr(orch, "_connect", lambda a: called.append(a))
    await orch._ensure_connected()
    assert called == []


async def test_before_agent_callback_connects(orch, monkeypatch):
    hit = []

    async def fake_ensure():
        hit.append(True)

    monkeypatch.setattr(orch, "_ensure_connected", fake_ensure)
    assert await orch._connect_before_run(callback_context=None) is None
    assert hit == [True]


async def test_send_message_to_unknown_agent_tries_reconnecting_first(orch, monkeypatch):
    tried = []

    async def fake_ensure():
        tried.append(True)  # a worker might have come up since startup

    monkeypatch.setattr(orch, "_ensure_connected", fake_ensure)
    with pytest.raises(ValueError):
        await orch.send_message("Ghost Agent", "do it", FakeToolContext())
    assert tried == [True]


def test_root_instruction_lists_connected_agents(orch):
    orch.agents = '{"name": "Data Analyst"}'
    text = orch.root_instruction(None)
    assert "Chief of Staff" in text
    assert "Data Analyst" in text
    assert "send_message" in text


# --------------------------------------------------------------------------- #
# send_message
# --------------------------------------------------------------------------- #
async def test_unknown_agent_raises(orch):
    with pytest.raises(ValueError):
        await orch.send_message("Ghost Agent", "do it", FakeToolContext())


async def test_new_task_omits_task_id_and_stores_ids(orch):
    conn = FakeConn(_success(_task(state=TaskState.working)))
    orch.remote_agent_connections["Data Analyst"] = conn
    ctx = FakeToolContext()

    await orch.send_message("Data Analyst", "run report", ctx)

    sent = conn.requests[0].params.message
    assert sent.task_id is None  # brand-new task must not send taskId
    assert sent.context_id  # a fresh context id is generated
    # Non-terminal -> keep the task id for follow-ups.
    assert ctx.state["task_id:Data Analyst"] == "task-1"
    assert ctx.state["context_id:Data Analyst"] == "ctx-1"


async def test_existing_non_terminal_task_id_is_reused(orch):
    conn = FakeConn(_success(_task(state=TaskState.input_required)))
    orch.remote_agent_connections["Data Analyst"] = conn
    ctx = FakeToolContext(
        {"task_id:Data Analyst": "prev-task", "context_id:Data Analyst": "prev-ctx"}
    )

    await orch.send_message("Data Analyst", "more", ctx)

    assert conn.requests[0].params.message.task_id == "prev-task"


async def test_terminal_task_state_clears_stored_task_id(orch):
    conn = FakeConn(_success(_task(state=TaskState.completed)))
    orch.remote_agent_connections["Data Analyst"] = conn
    ctx = FakeToolContext({"task_id:Data Analyst": "old"})

    await orch.send_message("Data Analyst", "finish", ctx)

    assert ctx.state["task_id:Data Analyst"] is None
    assert ctx.state["context_id:Data Analyst"] == "ctx-1"


async def test_artifact_parts_are_returned(orch):
    task = _task(
        state=TaskState.completed,
        artifacts=[
            Artifact(
                artifact_id="a1",
                parts=[Part(root=TextPart(text="the report body"))],
            )
        ],
    )
    conn = FakeConn(_success(task))
    orch.remote_agent_connections["Data Analyst"] = conn

    result = await orch.send_message("Data Analyst", "report", FakeToolContext())

    assert any(p.get("text") == "the report body" for p in result)


# --------------------------------------------------------------------------- #
# uploaded-file forwarding
# --------------------------------------------------------------------------- #
async def test_no_upload_sends_only_the_task_text(orch):
    conn = FakeConn(_success(_task()))
    orch.remote_agent_connections["Data Analyst"] = conn

    await orch.send_message("Data Analyst", "just text", FakeToolContext())

    parts = conn.requests[0].params.message.parts
    assert len(parts) == 1
    assert parts[0].root.text == "just text"


async def test_uploaded_csv_is_forwarded_as_text(orch):
    conn = FakeConn(_success(_task()))
    orch.remote_agent_connections["Researcher"] = conn
    raw = "الاسم,المدينة\nالفرع,دبي\n".encode("utf-8")
    ctx = FakeToolContext(user_content=_turn_with_upload(raw, "text/csv", "branches.csv"))

    await orch.send_message("Researcher", "summarise this", ctx)

    parts = conn.requests[0].params.message.parts
    texts = [p.root.text for p in parts if p.root.kind == "text"]
    assert any("summarise this" in t for t in texts)
    joined = "\n".join(texts)
    assert "branches.csv" in joined
    assert "الفرع,دبي" in joined
    # no binary file part for a converted spreadsheet
    assert all(p.root.kind == "text" for p in parts)


async def test_uploaded_xlsx_is_converted_to_text(orch):
    import io

    from openpyxl import Workbook

    wb = Workbook()
    wb.active.append(["name", "city"])
    wb.active.append(["Branch A", "Dubai"])
    buf = io.BytesIO()
    wb.save(buf)

    conn = FakeConn(_success(_task()))
    orch.remote_agent_connections["Data Analyst"] = conn
    ctx = FakeToolContext(user_content=_turn_with_upload(
        buf.getvalue(),
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "branches.xlsx",
    ))

    await orch.send_message("Data Analyst", "summarise", ctx)

    parts = conn.requests[0].params.message.parts
    joined = "\n".join(p.root.text for p in parts if p.root.kind == "text")
    assert "Branch A,Dubai" in joined


async def test_uploaded_pdf_is_forwarded_as_base64_file_part(orch):
    import base64

    conn = FakeConn(_success(_task()))
    orch.remote_agent_connections["Data Analyst"] = conn
    raw = b"%PDF-1.4 fake pdf bytes \x00\x01"
    ctx = FakeToolContext(user_content=_turn_with_upload(raw, "application/pdf", "doc.pdf"))

    await orch.send_message("Data Analyst", "read it", ctx)

    parts = conn.requests[0].params.message.parts
    file_parts = [p.root for p in parts if p.root.kind == "file"]
    assert len(file_parts) == 1
    assert base64.b64decode(file_parts[0].file.bytes) == raw
    assert file_parts[0].file.mime_type == "application/pdf"


# --------------------------------------------------------------------------- #
# files a worker returns  ->  absorbed + carried onward
# --------------------------------------------------------------------------- #
class RecordingCtx(FakeToolContext):
    """FakeToolContext that also records save_artifact() calls."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.saved = []

    def save_artifact(self, filename, part):
        self.saved.append((filename, part.inline_data.data, part.inline_data.mime_type))
        return 1


def _file_artifact(name, data: bytes, mime="application/pdf"):
    return Artifact(
        artifact_id="a1",
        parts=[Part(root=FilePart(
            file=FileWithBytes(bytes=base64.b64encode(data).decode(), mime_type=mime, name=name)
        ))],
    )


async def test_returned_file_is_saved_and_noted_not_dumped(orch):
    pdf = b"%PDF-1.4 report bytes"
    conn = FakeConn(_success(_task(state=TaskState.completed,
                                   artifacts=[_file_artifact("report.pdf", pdf)])))
    orch.remote_agent_connections["Data Analyst"] = conn
    ctx = RecordingCtx()

    result = await orch.send_message("Data Analyst", "make a report", ctx)

    # saved to the orchestrator's artifact store for download
    assert ctx.saved == [("report.pdf", pdf, "application/pdf")]
    # the LLM gets a short note, NOT the base64 blob
    assert len(result) == 1
    assert result[0]["kind"] == "text"
    assert "report.pdf" in result[0]["text"]
    assert base64.b64encode(pdf).decode() not in result[0]["text"]
    # remembered in session state for onward forwarding
    assert ctx.state["session_files"][0]["name"] == "report.pdf"
    assert ctx.state["session_files"][0]["source"] == "Data Analyst"


async def test_returned_image_is_inlined_for_the_llm(orch):
    png = b"\x89PNG\r\n\x1a\n" + b"x" * 50
    conn = FakeConn(_success(_task(state=TaskState.completed,
                                   artifacts=[_file_artifact("chart.png", png, "image/png")])))
    orch.remote_agent_connections["Data Analyst"] = conn
    ctx = RecordingCtx()

    result = await orch.send_message("Data Analyst", "chart it", ctx)

    assert any(getattr(p, "inline_data", None) and p.inline_data.data == png for p in result)


async def test_file_from_one_agent_is_forwarded_to_the_next(orch):
    pdf = b"%PDF-1.4 shared"
    data_conn = FakeConn(_success(_task(state=TaskState.completed,
                                        artifacts=[_file_artifact("report.pdf", pdf)])))
    research_conn = FakeConn(_success(_task()))
    orch.remote_agent_connections["Data Analyst"] = data_conn
    orch.remote_agent_connections["Researcher"] = research_conn
    ctx = RecordingCtx()

    await orch.send_message("Data Analyst", "make a report", ctx)
    await orch.send_message("Researcher", "use the report", ctx)

    parts = research_conn.requests[0].params.message.parts
    file_parts = [p.root for p in parts if p.root.kind == "file"]
    assert len(file_parts) == 1
    assert file_parts[0].file.name == "report.pdf"
    assert base64.b64decode(file_parts[0].file.bytes) == pdf


async def test_file_is_not_sent_back_to_the_agent_that_made_it(orch):
    pdf = b"%PDF-1.4 x"
    conn = FakeConn(_success(_task(state=TaskState.completed,
                                   artifacts=[_file_artifact("r.pdf", pdf)])))
    orch.remote_agent_connections["Data Analyst"] = conn
    ctx = RecordingCtx()

    await orch.send_message("Data Analyst", "make it", ctx)
    await orch.send_message("Data Analyst", "now revise it", ctx)

    parts = conn.requests[1].params.message.parts
    assert all(p.root.kind == "text" for p in parts)  # no file echoed back


async def test_non_success_response_returns_error_payload(orch):
    import types as _types

    # send_response.root is not a SendMessageSuccessResponse -> error branch.
    bad = _types.SimpleNamespace(root=_types.SimpleNamespace(result=None))
    conn = FakeConn(bad)
    orch.remote_agent_connections["Data Analyst"] = conn

    result = await orch.send_message("Data Analyst", "x", FakeToolContext())
    assert len(result) == 1
    assert "Error" in result[0]["text"]
    assert "Data Analyst" in result[0]["text"]


# --------------------------------------------------------------------------- #
# Delegation resilience - a slow / down / broken worker must not crash the turn
# --------------------------------------------------------------------------- #
from a2a.client.errors import A2AClientHTTPError, A2AClientTimeoutError  # noqa: E402


class RaisingConn:
    """A worker connection whose send_message raises (or raises then succeeds)."""

    def __init__(self, *effects):
        # each effect: an Exception to raise, or a SendMessageResponse to return
        self.effects = list(effects)
        self.calls = 0

    async def send_message(self, message_request):
        self.calls += 1
        effect = self.effects[min(self.calls - 1, len(self.effects) - 1)]
        if isinstance(effect, BaseException):
            raise effect
        return effect

    card = None  # -> _worker_is_reachable short-circuits to True in tests


@pytest.fixture
def fast_retries(monkeypatch):
    monkeypatch.setattr(agent_mod, "A2A_RETRY_BACKOFF_SECONDS", 0.0)
    monkeypatch.setattr(agent_mod, "A2A_MAX_RETRIES", 1)


async def test_worker_timeout_is_returned_not_raised(orch, fast_retries):
    conn = RaisingConn(A2AClientTimeoutError("Client Request timed out"))
    orch.remote_agent_connections["Researcher"] = conn

    result = await orch.send_message("Researcher", "research X", FakeToolContext())

    assert len(result) == 1
    assert result[0]["text"].startswith("⚠️")
    assert "time limit" in result[0]["text"]
    assert conn.calls == 1  # a timeout is NOT retried


async def test_transient_http_error_is_retried_then_recovers(orch, fast_retries):
    conn = RaisingConn(A2AClientHTTPError(503, "Service Unavailable"), _success(_task()))
    orch.remote_agent_connections["Data Analyst"] = conn

    result = await orch.send_message("Data Analyst", "kpis", FakeToolContext())

    assert conn.calls == 2  # retried once, second call succeeded
    assert not any(str(p.get("text", "")).startswith("⚠️") for p in result)


async def test_transient_http_error_exhausts_retries_and_reports(orch, fast_retries):
    conn = RaisingConn(A2AClientHTTPError(503, "down"), A2AClientHTTPError(503, "down"))
    orch.remote_agent_connections["Data Analyst"] = conn

    result = await orch.send_message("Data Analyst", "kpis", FakeToolContext())

    assert conn.calls == 2
    assert result[0]["text"].startswith("⚠️")
    assert "Data Analyst" in result[0]["text"]


async def test_unexpected_exception_does_not_crash_the_turn(orch, fast_retries):
    conn = RaisingConn(RuntimeError("boom in the transport"))
    orch.remote_agent_connections["Researcher"] = conn

    result = await orch.send_message("Researcher", "find leads", FakeToolContext())

    assert result[0]["text"].startswith("⚠️")
    assert "Researcher" in result[0]["text"]


async def test_unreachable_worker_fails_fast_without_sending(orch, monkeypatch, fast_retries):
    async def _down(base_url):
        return False

    monkeypatch.setattr(agent_mod, "_worker_is_reachable", _down)
    conn = RaisingConn(_success(_task()))
    orch.remote_agent_connections["Researcher"] = conn

    result = await orch.send_message("Researcher", "x", FakeToolContext())

    assert conn.calls == 0  # never attempted the real call
    assert "health check" in result[0]["text"]


async def test_failed_delegation_leaves_task_id_state_untouched(orch, fast_retries):
    conn = RaisingConn(A2AClientTimeoutError("timed out"))
    orch.remote_agent_connections["Data Analyst"] = conn
    ctx = FakeToolContext({"task_id:Data Analyst": "prev", "context_id:Data Analyst": "prev-ctx"})

    await orch.send_message("Data Analyst", "x", ctx)

    # a failed call must not clobber a still-valid task id from an earlier turn
    assert ctx.state["task_id:Data Analyst"] == "prev"
