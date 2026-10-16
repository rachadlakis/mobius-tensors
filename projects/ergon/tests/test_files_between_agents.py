"""
Integration tests for files travelling between the Chief of Staff and employees.

No LLM and no HTTP server: every file-plumbing function is real and wired
together with a real ``InMemoryArtifactService``, but the model turn itself
(which would decide *which* tool to call) is not exercised - the tests call the
tool directly, the way the employee's LLM would.

The pipe under test, end to end:

    orchestrator._delegation_parts        (build the A2A message + attach files)
      -> a2a_files.convert_a2a_parts_to_genai   (employee executor decodes them)
      -> BaseA2AExecutor._save_incoming_files   (saved as a named artifact)
      -> documents.write_pdf_report         (a tool produces a result file)
      -> BaseA2AExecutor._new_file_artifacts    (harvest what the turn made)
      -> orchestrator._absorb_file_part     (save for download + remember onward)
"""

import base64
import importlib

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
)
from google.adk.artifacts import InMemoryArtifactService
from google.genai import types

from shared import documents
from shared.a2a_files import convert_a2a_parts_to_genai
from shared.config import USER_ID

agent_mod = importlib.import_module("orchestrator_agent.orchestrator.agent")
dae = importlib.import_module("data_agent_adk.agent_executor")

SESSION_ID = "sess-1"
DATA_ANALYST = "Data Analyst"
RESEARCHER = "Researcher"


# --------------------------------------------------------------------------- #
# Fakes / helpers
# --------------------------------------------------------------------------- #
class FakeToolContext:
    def __init__(self, state=None, user_content=None):
        self.state = state if state is not None else {}
        self.user_content = user_content


class RecordingCtx(FakeToolContext):
    """Orchestrator-side tool context that records save_artifact() calls."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.saved = []

    def save_artifact(self, filename, part):
        self.saved.append((filename, part.inline_data.data, part.inline_data.mime_type))
        return 1


class FakeRunner:
    def __init__(self, artifact_service, app_name=DATA_ANALYST):
        self.artifact_service = artifact_service
        self.app_name = app_name


class ArtifactServiceCtx:
    """The slice of ToolContext a file-producing tool needs, backed by a
    real InMemoryArtifactService scoped to one (app, user, session)."""

    def __init__(self, svc, app_name, session_id):
        self._svc = svc
        self._app = app_name
        self._session = session_id
        self.state = {}

    async def load_artifact(self, filename):
        return await self._svc.load_artifact(
            app_name=self._app, user_id=USER_ID,
            session_id=self._session, filename=filename,
        )

    async def save_artifact(self, filename, part):
        return await self._svc.save_artifact(
            app_name=self._app, user_id=USER_ID,
            session_id=self._session, filename=filename, artifact=part,
        )

    async def list_artifacts(self):
        return await self._svc.list_artifact_keys(
            app_name=self._app, user_id=USER_ID, session_id=self._session,
        )


def turn_with_upload(data: bytes, mime: str, name: str):
    blob = types.Blob(data=data, mime_type=mime)
    blob.display_name = name
    return types.Content(role="user", parts=[types.Part(inline_data=blob)])


def _task(state=TaskState.completed, artifacts=None):
    return Task(
        id="t1", context_id="c1",
        status=TaskStatus(state=state), artifacts=artifacts or [],
    )


def _success(task):
    return SendMessageResponse(
        root=SendMessageSuccessResponse(id="1", jsonrpc="2.0", result=task)
    )


class FakeConn:
    def __init__(self, response):
        self.response = response
        self.requests = []

    async def send_message(self, message_request):
        self.requests.append(message_request)
        return self.response


def _file_artifact(name, data: bytes, mime):
    return Artifact(artifact_id="a1", parts=[Part(root=FilePart(
        file=FileWithBytes(bytes=base64.b64encode(data).decode(), mime_type=mime, name=name)
    ))])


# --------------------------------------------------------------------------- #
# 1. Full round trip: upload a PDF -> Data Analyst -> PDF report -> back
# --------------------------------------------------------------------------- #
async def test_upload_and_report_round_trip_through_data_analyst(monkeypatch, tmp_path):
    monkeypatch.setattr(documents, "WORKSPACE_DIR", tmp_path)
    brief = b"%PDF-1.4 brief"

    # -- Chief of Staff side: remember the upload, build the delegation message --
    orch_ctx = FakeToolContext(
        user_content=turn_with_upload(brief, "application/pdf", "brief.pdf")
    )
    agent_mod._remember_uploads(orch_ctx)
    parts = agent_mod._delegation_parts(DATA_ANALYST, "Summarise the brief.", orch_ctx)
    # a non-tabular upload is handed over as a real file part
    file_parts = [p for p in parts if p.get("kind") == "file"]
    assert [p["file"]["name"] for p in file_parts] == ["brief.pdf"]
    assert base64.b64decode(file_parts[0]["file"]["bytes"]) == brief

    # -- employee executor side: decode the A2A parts, save incoming files ------
    a2a_parts = [Part.model_validate(p) for p in parts]
    content = types.UserContent(parts=convert_a2a_parts_to_genai(a2a_parts))

    svc = InMemoryArtifactService()
    ex = dae.DataAnalystExecutor(FakeRunner(svc))
    before = await ex._artifact_versions(SESSION_ID)
    await ex._save_incoming_files(content, SESSION_ID)
    assert "brief.pdf" in await ex._artifact_keys(SESSION_ID)

    # -- the tool the employee's LLM would call produces a report ---------------
    tool_ctx = ArtifactServiceCtx(svc, ex.runner.app_name, SESSION_ID)
    out = await documents.write_pdf_report(
        "Brief summary", [{"heading": "Summary", "text": "All good."}],
        tool_context=tool_ctx,
    )
    assert out["success"] is True

    # -- executor harvests the file the tool produced --------------------------
    produced = await ex._new_file_artifacts(SESSION_ID, before)
    produced_names = {name for name, _ in produced}
    assert out["file_name"] in produced_names
    name, part = next(p for p in produced if p[0] == out["file_name"])

    # -- Chief of Staff absorbs the returned file ------------------------------
    file_dict = {
        "kind": "file",
        "file": {
            "name": part.root.file.name,
            "mimeType": part.root.file.mime_type,
            "bytes": part.root.file.bytes,
        },
    }
    back_ctx = RecordingCtx()
    entries = await agent_mod._absorb_file_part(file_dict, DATA_ANALYST, back_ctx)

    # saved for the adk-web download button
    assert len(back_ctx.saved) == 1
    saved_name, saved_bytes, saved_mime = back_ctx.saved[0]
    assert saved_name == out["file_name"]
    assert saved_mime == "application/pdf"
    assert saved_bytes.startswith(b"%PDF")
    # remembered in the session drawer, tagged with the producing employee
    sf = back_ctx.state["session_files"][0]
    assert sf["name"] == out["file_name"]
    assert sf["source"] == DATA_ANALYST
    # the LLM gets a note, never the bytes
    assert entries[0]["kind"] == "text"
    assert out["file_name"] in entries[0]["text"]
    assert base64.b64encode(saved_bytes).decode() not in entries[0]["text"]


# --------------------------------------------------------------------------- #
# 2. A spreadsheet one employee produced reaches the NEXT employee as CSV text
#    with its contents intact.
# --------------------------------------------------------------------------- #
async def test_employee_produced_csv_is_forwarded_as_text():
    orch = agent_mod.OrchestratorAgent()
    csv_bytes = "company,website\nAcme,acme.example\nGlobex,globex.example\n".encode("utf-8")

    research_conn = FakeConn(_success(_task(
        artifacts=[_file_artifact("leads.csv", csv_bytes, "text/csv")]
    )))
    data_conn = FakeConn(_success(_task()))
    orch.remote_agent_connections[RESEARCHER] = research_conn
    orch.remote_agent_connections[DATA_ANALYST] = data_conn
    ctx = RecordingCtx()

    await orch.send_message(RESEARCHER, "Find leads.", ctx)
    await orch.send_message(DATA_ANALYST, "Clean up the leads sheet.", ctx)

    parts = data_conn.requests[0].params.message.parts
    assert not [p for p in parts if p.root.kind == "file"]
    joined = "\n".join(p.root.text for p in parts if p.root.kind == "text")
    assert "--- file: leads.csv" in joined
    assert "Globex,globex.example" in joined


# --------------------------------------------------------------------------- #
# 3. Accumulation: after an employee returns a file, the NEXT delegation
#    carries both the original upload and the produced file.
# --------------------------------------------------------------------------- #
async def test_upload_and_produced_file_both_forwarded_on_next_hop():
    orch = agent_mod.OrchestratorAgent()
    upload = "company,city\nAcme,Dubai\n".encode("utf-8")
    produced = b"company,city,website\nAcme,Dubai,acme.example\n"

    research_conn = FakeConn(_success(_task(
        artifacts=[_file_artifact("leads_enriched.csv", produced, "text/csv")]
    )))
    data_conn = FakeConn(_success(_task()))
    orch.remote_agent_connections[RESEARCHER] = research_conn
    orch.remote_agent_connections[DATA_ANALYST] = data_conn

    ctx = RecordingCtx(user_content=turn_with_upload(upload, "text/csv", "leads.csv"))
    await orch.send_message(RESEARCHER, "Find the website of each company.", ctx)
    await orch.send_message(DATA_ANALYST, "Summarise the enriched file.", ctx)

    parts = data_conn.requests[0].params.message.parts
    texts = "\n".join(p.root.text for p in parts if p.root.kind == "text")
    assert "leads.csv" in texts
    assert "leads_enriched.csv" in texts
    assert "Acme,Dubai,acme.example" in texts
