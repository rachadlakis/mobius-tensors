"""The MCP server: the chat's tools over MCP, through the SDK's in-process client."""

from __future__ import annotations

import json
from datetime import datetime

import anyio
import numpy as np
import pytest

pytest.importorskip("mcp")
from mcp import Client  # noqa: E402

from argus.engine.search import SearchEngine  # noqa: E402
from argus.engine.store import EventStore  # noqa: E402
from argus.mcp_server import build_toolbox, create_server  # noqa: E402
from argus.tools import Toolbox  # noqa: E402

NOW = datetime(2026, 10, 9, 12, 0)


class FakeClip:
    def embed_query(self, text: str) -> np.ndarray:
        return np.array([1, 0, 0, 0], np.float32)


class FakeVision:
    model, max_images = "fake-vl", 4

    def __init__(self) -> None:
        self.questions: list[str] = []

    def ask(self, images, question, system=None):
        self.questions.append(question)
        return f"{len(images)} image(s): two people."


@pytest.fixture
def store(tmp_path):
    s = EventStore(tmp_path / "db.sqlite")
    t = NOW.replace(hour=9).timestamp()
    snap = s.save_image(np.full((120, 160, 3), 90, np.uint8), "snapshots", "yard_enter", t)
    event_id = s.add_event("gate", t, {"type": "enter", "rule": "yard", "label": "person",
                                       "track_id": 3}, snap)
    s.set_verification(event_id, "yes", "A person walks into the yard.")
    yield s
    s.close()


def call(server, steps):
    async def go():
        async with Client(server) as client:
            return [await step(client) for step in steps]
    return anyio.run(go)


def test_lists_the_chat_tools_read_only(store):
    toolbox = Toolbox(SearchEngine(store, FakeClip(), camera_names=["gate"]), now=NOW,
                      vision=FakeVision())
    server = create_server(toolbox)
    (listed,) = call(server, [lambda c: c.list_tools()])
    names = [t.name for t in listed.tools]
    assert names == [t.name for t in toolbox.tools] and "look" in names and "face" not in names
    alerts = next(t for t in listed.tools if t.name == "alerts")
    assert alerts.input_schema["type"] == "object" and "rules" in alerts.input_schema["properties"]
    assert all(t.annotations.read_only_hint for t in listed.tools)


def test_calls_return_the_chat_json_with_site_time_and_look_works(store):
    vision = FakeVision()
    toolbox = Toolbox(SearchEngine(store, FakeClip(), camera_names=["gate"]), now=NOW,
                      vision=vision)
    server = create_server(toolbox)
    alerts, look, bad = call(server, [
        lambda c: c.call_tool("alerts", {"when": "today"}),
        lambda c: c.call_tool("look", {"images": [], "question": "How many?"}),
        lambda c: c.call_tool("objects", {"labels": ["unicorn"]}),
    ])
    out = json.loads(alerts.content[0].text)
    assert "site_time" in out and out["total"] == 1
    (event,) = out["events"]
    assert (event["rule"], event["verified"]) == ("yard", "yes") and event["snapshot"]
    assert look.is_error and "images" in look.content[0].text  # no paths given
    assert bad.is_error and "unicorn" in bad.content[0].text

    snapshot = event["snapshot"]  # a path an earlier result listed: look may open it
    (seen,) = call(server, [lambda c: c.call_tool("look", {"images": [snapshot],
                                                            "question": "How many people?"})])
    assert not seen.is_error
    assert json.loads(seen.content[0].text)["answer"] == "1 image(s): two people."
    assert vision.questions == ["How many people?"]


def test_instructions_describe_the_site(store):
    toolbox = Toolbox(SearchEngine(store, FakeClip(), camera_names=["gate"]), now=NOW)
    server = create_server(toolbox)
    assert "yard" in server.instructions and "site_time" in server.instructions


def test_build_toolbox_leaves_face_search_out(tmp_path, capsys):
    config = {"store": {"path": str(tmp_path / "s.db")},
              "search": {"clip": {"text": str(tmp_path / "none.onnx"),
                                  "vocab": str(tmp_path / "none.gz")}},
              "faces": {"enabled": True}}
    toolbox = build_toolbox(config)
    assert toolbox.engine.faces is None and "face" not in [t.name for t in toolbox.tools]
    assert "search" not in [t.name for t in toolbox.tools]  # no CLIP here
    assert capsys.readouterr().out == ""  # stdout is the MCP channel: nothing printed on it
    toolbox.close()
