"""The worker executors harvest files their tools saved during a turn and
attach them to the A2A task (so the Orchestrator receives them)."""

import importlib

import pytest
from a2a.types import FilePart
from google.adk.artifacts import InMemoryArtifactService
from google.genai import types

from shared.config import USER_ID

de = importlib.import_module("data_agent_adk.agent_executor")


class FakeRunner:
    def __init__(self, artifact_service, app_name="test-app"):
        self.artifact_service = artifact_service
        self.app_name = app_name


async def _save(svc, session_id, name, data, mime):
    await svc.save_artifact(
        app_name="test-app", user_id=USER_ID, session_id=session_id,
        filename=name, artifact=types.Part(inline_data=types.Blob(data=data, mime_type=mime)),
    )


async def test_artifact_keys_empty_when_none_saved():
    ex = de.DataAnalystExecutor(FakeRunner(InMemoryArtifactService()))
    assert await ex._artifact_keys("s1") == set()


async def test_new_file_artifacts_are_returned_as_a2a_file_parts():
    svc = InMemoryArtifactService()
    ex = de.DataAnalystExecutor(FakeRunner(svc))

    known = await ex._artifact_versions("s1")
    await _save(svc, "s1", "chart.png", b"\x89PNGdata", "image/png")

    new = await ex._new_file_artifacts("s1", known)
    assert len(new) == 1
    name, part = new[0]
    assert name == "chart.png"
    assert isinstance(part.root, FilePart)
    assert part.root.file.name == "chart.png"
    assert part.root.file.mime_type == "image/png"

    import base64
    assert base64.b64decode(part.root.file.bytes) == b"\x89PNGdata"


async def test_pre_existing_artifacts_are_not_re_harvested():
    svc = InMemoryArtifactService()
    ex = de.DataAnalystExecutor(FakeRunner(svc))

    await _save(svc, "s1", "old.pdf", b"%PDF-old", "application/pdf")
    known = await ex._artifact_versions("s1")  # snapshot AFTER old.pdf exists

    await _save(svc, "s1", "new.pdf", b"%PDF-new", "application/pdf")

    new = await ex._new_file_artifacts("s1", known)
    assert [n for n, _ in new] == ["new.pdf"]


async def test_overwritten_artifact_is_re_harvested():
    """A tool that rewrites a fixed filename (say, always writing
    results.csv) bumps the version, not the key set. Turn 2 of a
    conversation must still forward it."""
    svc = InMemoryArtifactService()
    ex = de.DataAnalystExecutor(FakeRunner(svc))
    v1, v2 = b"id,total;1,25.1", b"id,total;2,26.2"

    # Turn 1 produces the file and forwards it.
    turn1_known = await ex._artifact_versions("s1")
    await _save(svc, "s1", "results.csv", v1, "text/csv")
    assert [n for n, _ in await ex._new_file_artifacts("s1", turn1_known)] == [
        "results.csv"
    ]

    # Turn 2, same session: the name already exists, only the version moves.
    turn2_known = await ex._artifact_versions("s1")
    assert turn2_known == {"results.csv": 0}
    await _save(svc, "s1", "results.csv", v2, "text/csv")

    new = await ex._new_file_artifacts("s1", turn2_known)
    assert [n for n, _ in new] == ["results.csv"]

    import base64
    assert base64.b64decode(new[0][1].root.file.bytes) == v2


async def test_untouched_artifact_is_not_re_harvested_when_another_is_written():
    svc = InMemoryArtifactService()
    ex = de.DataAnalystExecutor(FakeRunner(svc))

    await _save(svc, "s1", "old.pdf", b"%PDF-old", "application/pdf")
    known = await ex._artifact_versions("s1")
    await _save(svc, "s1", "chart.png", b"PNGbytes", "image/png")

    assert [n for n, _ in await ex._new_file_artifacts("s1", known)] == ["chart.png"]



async def test_missing_artifact_service_is_tolerated():
    ex = de.DataAnalystExecutor(FakeRunner(None))
    assert await ex._artifact_keys("s1") == set()
    assert await ex._artifact_versions("s1") == {}
    assert await ex._new_file_artifacts("s1", {}) == []


@pytest.mark.parametrize("module_name, cls_name", [
    ("data_agent_adk.agent_executor", "DataAnalystExecutor"),
    ("web_intel_agent_adk.agent_executor", "ResearcherExecutor"),
])
def test_every_executor_has_the_harvest_helpers(module_name, cls_name):
    mod = importlib.import_module(module_name)
    cls = getattr(mod, cls_name)
    assert hasattr(cls, "_artifact_keys")
    assert hasattr(cls, "_artifact_versions")
    assert hasattr(cls, "_new_file_artifacts")
