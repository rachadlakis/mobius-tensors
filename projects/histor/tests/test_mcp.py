"""The MCP server: the chat's tools over MCP, through the SDK's in-process client."""

from __future__ import annotations

import json
from datetime import datetime

import anyio
import numpy as np
import pytest

pytest.importorskip("mcp")
from mcp import Client  # noqa: E402

from histor.engine.search import SearchEngine  # noqa: E402
from histor.engine.store import EventStore  # noqa: E402
from histor.mcp_server import build_toolbox, create_server  # noqa: E402
from histor.tools import Toolbox  # noqa: E402

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


# ------------------------------------------------------------------ face search over MCP

class FakeFaces:
    """A bright photo has one usable face; a dark one has none."""

    def photo_faces(self, image):
        from histor.engine.faces import FaceQuality, PhotoFace

        if image.mean() < 50:
            return []
        v = np.zeros(128, np.float32)
        v[0] = 1
        return [PhotoFace(v, (0, 0, 80, 80), FaceQuality(0.9, 80, 0.1, 120), [])]


def photo(path, value):
    import cv2

    cv2.imwrite(str(path), np.full((120, 120, 3), value, np.uint8))
    return str(path)


def test_face_off_by_default_and_the_model_is_told_it_exists_elsewhere(store):
    toolbox = Toolbox(SearchEngine(store, FakeClip(), FakeFaces(), camera_names=["gate"]), now=NOW)
    server = create_server(toolbox)  # face_search off
    listed, called = call(server, [lambda c: c.list_tools(),
                                   lambda c: c.call_tool("face", {"photo_path": "x.jpg"})])
    assert "face" not in [t.name for t in listed.tools]
    assert called.is_error and "not available" in called.content[0].text
    assert "Never say Histor can't match faces" in server.instructions
    assert "attached (photo1" not in server.instructions  # the chat's /photo help is left out


def test_face_search_from_a_photo_file(store, tmp_path):
    seen = NOW.replace(hour=10).timestamp()
    v = np.zeros(128, np.float32)
    v[0] = 1
    store.add_faces([{"camera": "gate", "time": seen + k, "track_id": 5, "embedding": v}
                     for k in range(3)])
    toolbox = Toolbox(SearchEngine(store, FakeClip(), FakeFaces(), camera_names=["gate"]), now=NOW)
    server = create_server(toolbox, face_search=True, photo_dirs=[str(tmp_path / "photos")])
    (tmp_path / "photos").mkdir()
    good = photo(tmp_path / "photos" / "visitor.jpg", 200)
    dark = photo(tmp_path / "photos" / "dark.jpg", 10)
    outside = photo(tmp_path / "elsewhere.jpg", 200)
    (tmp_path / "photos" / "notes.txt").write_text("hello")
    listed, found, refused, not_photo, missing, out_of_dirs = call(server, [
        lambda c: c.list_tools(),
        lambda c: c.call_tool("face", {"photo_path": good, "when": "today"}),
        lambda c: c.call_tool("face", {"photo_path": dark}),
        lambda c: c.call_tool("face", {"photo_path": str(tmp_path / "photos" / "notes.txt")}),
        lambda c: c.call_tool("face", {"photo_path": str(tmp_path / "photos" / "none.jpg")}),
        lambda c: c.call_tool("face", {"photo_path": outside}),
    ])
    face = next(t for t in listed.tools if t.name == "face")
    assert face.input_schema["required"] == ["photo_path"]
    assert "file on the Histor server" in server.instructions
    out = json.loads(found.content[0].text)
    assert not found.is_error and out["likely"] == 1 and out["visits"][0]["camera"] == "gate"
    assert refused.is_error and "Error:" in refused.content[0].text
    assert not_photo.is_error and "not a photo" in not_photo.content[0].text
    assert missing.is_error and "does not exist" in missing.content[0].text
    assert out_of_dirs.is_error and "only photos in these folders" in out_of_dirs.content[0].text
