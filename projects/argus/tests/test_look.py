"""The `look` tool: a local vision model reads images that earlier results listed.

A fake OpenAI-compatible server stands in for Ollama, so the real HTTP request (images as
base64 JPEG parts) is checked end to end."""

from __future__ import annotations

import base64
import json
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np
import pytest

from argus.chat import LOOK_HELP, answer, system_prompt
from argus.engine.search import SearchEngine
from argus.engine.store import EventStore
from argus.llm import Step, ToolCall
from argus.tools import LOOK_TOOL, Toolbox, _media
from argus.vision import VisionModel, encode_jpeg

NOW = datetime(2026, 10, 9, 12, 0)


class FakeClip:
    def embed_query(self, text: str) -> np.ndarray:
        v = np.zeros(4, np.float32)
        v[0] = 1
        return v


class VisionServer:
    """Minimal /v1/chat/completions that records requests and replies with canned text."""

    def __init__(self, reply: str = "Two people, one holding a red umbrella.",
                 status: int = 200) -> None:
        self.requests: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append({"path": self.path, "body": body,
                                       "auth": self.headers.get("Authorization")})
                payload = json.dumps({"choices": [{"message": {
                    "role": "assistant", "content": reply}}]}).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload if status == 200 else b'{"error": "model not found"}')

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def server():
    s = VisionServer()
    yield s
    s.close()


@pytest.fixture
def store(tmp_path):
    s = EventStore(tmp_path / "db.sqlite")
    yield s
    s.close()


def add_scene(store: EventStore, t: float, name: str, size=(480, 640), colour=(0, 0, 200)) -> str:
    image = np.full((*size, 3), colour, np.uint8)
    rel = store.save_image(image, "scenes", name, t)
    store.add_sightings([{"camera": "gate", "time": t, "kind": "scene", "thumb": rel,
                          "box": [0, 0, size[1], size[0]],
                          "embedding": np.array([1, 0, 0, 0], np.float32)}])
    return rel


def make_toolbox(store, vision) -> Toolbox:
    return Toolbox(SearchEngine(store, FakeClip(), camera_names=["gate"]), now=NOW,
                   vision=vision)


def decode_parts(request: dict) -> list[np.ndarray]:
    parts = request["body"]["messages"][1]["content"]
    images = []
    for p in parts[1:]:
        assert p["type"] == "image_url"
        prefix = "data:image/jpeg;base64,"
        assert p["image_url"]["url"].startswith(prefix)
        raw = base64.b64decode(p["image_url"]["url"][len(prefix):])
        images.append(cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR))
    return images


# ------------------------------------------------------------------ VisionModel

def test_from_config():
    assert VisionModel.from_config(None) is None
    assert VisionModel.from_config({}) is None
    assert VisionModel.from_config({"model": "qwen3.5:4b", "enabled": False}) is None
    v = VisionModel.from_config({"model": "qwen3.5:4b", "note": "ui only", "max_images": 0,
                                 "base_url": "http://x/v1/"})
    assert v.model == "qwen3.5:4b" and v.url == "http://x/v1/chat/completions"
    assert v.max_images == 1  # at least one
    with pytest.raises(ValueError, match="OpenAI-compatible"):
        VisionModel.from_config({"model": "m", "api": "anthropic"})


@pytest.mark.parametrize("shape, max_side, expected", [
    ((1080, 1920), 640, (360, 640)),
    ((1920, 1080), 640, (640, 360)),
    ((100, 200), 640, (100, 200)),  # never enlarged
])
def test_encode_jpeg_shrinks_longest_side(shape, max_side, expected):
    raw = base64.b64decode(encode_jpeg(np.zeros((*shape, 3), np.uint8), max_side))
    assert cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR).shape[:2] == expected


def test_ask_sends_question_and_images(server):
    v = VisionModel("qwen3.5:4b", base_url=server.url, max_side=320, api_key="k")
    out = v.ask([np.full((480, 640, 3), 255, np.uint8)], "How many people?")
    assert out == "Two people, one holding a red umbrella."
    req = server.requests[0]
    assert req["path"] == "/v1/chat/completions" and req["auth"] == "Bearer k"
    body = req["body"]
    assert body["model"] == "qwen3.5:4b" and body["stream"] is False
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][1]["content"][0] == {"type": "text", "text": "How many people?"}
    (image,) = decode_parts(req)
    assert image.shape == (240, 320, 3) and image.mean() > 240


def test_ask_strips_thinking():
    s = VisionServer(reply="<think>count them</think>\nThree.")
    try:
        assert VisionModel("m", base_url=s.url).ask([np.zeros((8, 8, 3), np.uint8)], "q") == "Three."
    finally:
        s.close()


def test_ask_limits():
    v = VisionModel("m", max_images=2)
    with pytest.raises(ValueError, match="No image"):
        v.ask([], "q")
    with pytest.raises(ValueError, match="At most 2"):
        v.ask([np.zeros((8, 8, 3), np.uint8)] * 3, "q")


def test_ask_server_errors_are_actionable():
    down = VisionModel("qwen3.5:4b", base_url="http://127.0.0.1:9/v1", timeout=2)
    with pytest.raises(RuntimeError, match="ollama pull qwen3.5:4b"):
        down.ask([np.zeros((8, 8, 3), np.uint8)], "q")
    s = VisionServer(status=404)
    try:
        with pytest.raises(RuntimeError, match="404"):
            VisionModel("m", base_url=s.url).ask([np.zeros((8, 8, 3), np.uint8)], "q")
    finally:
        s.close()


# ------------------------------------------------------------------ the tool

def test_media_paths_found_at_any_depth():
    out = {"results": [{"thumb": "a.jpg", "with": [{"thumb": "b.jpg"}]},
                       {"thumb": None}, {"snapshot": "s.jpg", "box": [1, 2]}],
           "thumb_note": "x.jpg"}
    assert sorted(_media(out)) == ["a.jpg", "b.jpg", "s.jpg"]


def test_tool_offered_only_with_vision(store):
    without = make_toolbox(store, None)
    with_ = make_toolbox(store, VisionModel("m"))
    try:
        assert "look" not in [t.name for t in without.tools]
        assert [t.name for t in with_.tools][-1] == "look"
        assert LOOK_HELP.strip() in system_prompt(with_, NOW)
        assert "`look`" not in system_prompt(without, NOW)
        r = without.run(ToolCall("1", "look", {"images": ["a.jpg"], "question": "q"}))
        assert r.is_error and "Unknown tool" in r.output
    finally:
        without.close()
        with_.close()


def test_look_after_search(store, server):
    t = NOW.timestamp() - 300
    rel = add_scene(store, t, "gate_1")
    box = make_toolbox(store, VisionModel("qwen3.5:4b", base_url=server.url, max_side=320))
    try:
        found = json.loads(box.run(ToolCall("1", "search", {"query": "people", "when": "today"})).output)
        assert found["results"][0]["thumb"] == rel
        box.evidence.clear()
        r = box.run(ToolCall("2", "look", {"images": [rel], "question": "How many people?"}))
        assert not r.is_error
        out = json.loads(r.output)
        assert out == {"answer": "Two people, one holding a red umbrella.", "images": [rel],
                       "model": "qwen3.5:4b"}
        assert box.evidence == [rel]
        (image,) = decode_parts(server.requests[0])
        assert image.shape == (240, 320, 3)
        assert image[..., 2].mean() > 150 and image[..., 0].mean() < 60  # red stays red (BGR)
    finally:
        box.close()


def test_look_refuses_paths_not_shown(store, server):
    rel = add_scene(store, NOW.timestamp() - 300, "gate_1")
    box = make_toolbox(store, VisionModel("m", base_url=server.url))
    try:
        for images in ([rel], ["../../etc/passwd"], ["C:/Windows/win.ini"]):
            r = box.run(ToolCall("1", "look", {"images": images, "question": "q"}))
            assert r.is_error and "not in any earlier result" in r.output
        assert server.requests == []  # nothing was sent
    finally:
        box.close()


def test_look_argument_errors(store, server):
    t = NOW.timestamp() - 300
    rels = [add_scene(store, t + i, f"gate_{i}") for i in range(3)]
    box = make_toolbox(store, VisionModel("m", base_url=server.url, max_images=2))
    try:
        box.run(ToolCall("1", "search", {"query": "x", "when": "today"}))
        cases = [({"images": [], "question": "q"}, "1 or more"),
                 ({"images": rels[:1], "question": "  "}, "question"),
                 ({"images": rels, "question": "q"}, "at most 2"),
                 ({"question": "q"}, "Error")]  # missing argument
        for args, message in cases:
            r = box.run(ToolCall("1", "look", args))
            assert r.is_error and message in r.output, (args, r.output)
        # a single string is accepted, duplicates collapse
        out = json.loads(box.run(ToolCall("2", "look", {"images": rels[0], "question": "q"})).output)
        assert out["images"] == [rels[0]]
        out = json.loads(box.run(ToolCall("3", "look", {"images": [rels[0], rels[0]],
                                                       "question": "q"})).output)
        assert out["images"] == [rels[0]]
    finally:
        box.close()


def test_look_expired_image(store, server):
    rel = add_scene(store, NOW.timestamp() - 300, "gate_1")
    box = make_toolbox(store, VisionModel("m", base_url=server.url))
    try:
        box.run(ToolCall("1", "search", {"query": "x", "when": "today"}))
        store.media_path(rel).unlink()
        r = box.run(ToolCall("2", "look", {"images": [rel], "question": "q"}))
        assert r.is_error and "no longer stored" in r.output
    finally:
        box.close()


def test_look_with_vision_server_down_tells_the_model(store):
    rel = add_scene(store, NOW.timestamp() - 300, "gate_1")
    box = make_toolbox(store, VisionModel("qwen3.5:4b", base_url="http://127.0.0.1:9/v1",
                                          timeout=2))
    try:
        box.run(ToolCall("1", "search", {"query": "x", "when": "today"}))
        r = box.run(ToolCall("2", "look", {"images": [rel], "question": "q"}))
        assert r.is_error and "vision model failed" in r.output and "ollama pull" in r.output
    finally:
        box.close()


def test_alert_snapshots_can_be_looked_at(store, server):
    t = NOW.timestamp() - 300
    snap = store.save_image(np.zeros((90, 160, 3), np.uint8), "events", "gate_ev", t)
    store.add_event("gate", t, {"type": "enter", "rule": "yard", "label": "person",
                                "track_id": 3}, snap)
    box = make_toolbox(store, VisionModel("m", base_url=server.url))
    try:
        alerts = json.loads(box.run(ToolCall("1", "alerts", {"when": "today"})).output)
        assert snap in _media(alerts)
        r = box.run(ToolCall("2", "look", {"images": [snap], "question": "Who is it?"}))
        assert not r.is_error
    finally:
        box.close()


class ScriptedBackend:
    """A chat model that searches, then looks, then answers."""

    name = "scripted"

    def __init__(self) -> None:
        self.results: list = []
        self.steps = iter([
            Step("", [ToolCall("a", "search", {"query": "people at the gate", "when": "today"})]),
            None,  # filled with a look at whatever the search returned
            Step("Two people; one had a red umbrella (looked at the image)."),
        ])

    def add_user(self, text: str) -> None:
        self.question = text

    def step(self) -> Step:
        s = next(self.steps)
        if s is None:
            thumb = json.loads(self.results[-1][0].output)["results"][0]["thumb"]
            s = Step("", [ToolCall("b", "look", {"images": [thumb],
                                                 "question": "How many people?"})])
        return s

    def add_results(self, results) -> None:
        self.results.append(results)


def test_chat_loop_search_then_look(store, server):
    add_scene(store, NOW.timestamp() - 300, "gate_1")
    box = make_toolbox(store, VisionModel("m", base_url=server.url))
    try:
        backend = ScriptedBackend()
        text = answer(backend, box, "how many people were at the gate today?")
        assert text.startswith("Two people")
        look_result = backend.results[1][0]
        assert not look_result.is_error and "red umbrella" in look_result.output
        assert len(server.requests) == 1
    finally:
        box.close()


def test_from_config_builds_vision_and_expands_env(store, monkeypatch):
    monkeypatch.setenv("VISION_KEY", "secret")
    engine = SearchEngine(store, None)
    config = {"chat": {"vision": {"model": "qwen3.5:4b", "api_key": "${VISION_KEY}"}}}
    box = Toolbox.from_config(engine, config)
    try:
        assert box.vision is not None and box.vision.api_key == "secret"
        assert "look" in [t.name for t in box.tools] and "search" not in [t.name for t in box.tools]
    finally:
        box.close()
    box = Toolbox.from_config(engine, {"chat": {"vision": {"model": "m", "enabled": False}}})
    try:
        assert box.vision is None
    finally:
        box.close()


def test_look_tool_schema():
    assert LOOK_TOOL.parameters["required"] == ["images", "question"]
    assert LOOK_TOOL.parameters["properties"]["images"]["type"] == "array"
