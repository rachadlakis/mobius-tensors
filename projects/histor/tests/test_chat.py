import json
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest
from histor.chat import FACE_HELP, answer, model_config, parse_photo_command, system_prompt
from histor.engine.faces import FaceQuality, PhotoFace
from histor.llm import Claude, OpenAICompatible, Step, Tool, ToolCall, ToolResult, create_backend
from histor.engine.search import SearchEngine
from histor.engine.store import EventStore
from histor.tools import TOOLS, Toolbox

NOW = datetime(2026, 10, 3, 12, 0)
YESTERDAY = datetime(2026, 10, 2, 18, 0).timestamp()


def unit(i: int) -> np.ndarray:
    v = np.zeros(512, dtype=np.float32)
    v[i] = 1
    return v


class FakeClip:
    def embed_query(self, text: str) -> np.ndarray:
        return unit(0) if "van" in text else unit(1)


def row(camera, t, track_id, label, thumb, vector):
    return {"camera": camera, "time": t, "kind": "object", "track_id": track_id, "label": label,
            "box": [0, 0, 9, 9], "thumb": thumb, "embedding": vector}


@pytest.fixture
def toolbox(tmp_path):
    store = EventStore(tmp_path / "h.db")
    y = YESTERDAY  # 18:00
    store.add_sightings([
        row("gate", y, 1, "car", "a.jpg", unit(1)),
        row("gate", y + 5, 1, "car", "a.jpg", unit(1)),
        row("gate", y + 60, 2, "car", "b.jpg", unit(1)),
        row("gate", y + 90, 3, "truck", "van.jpg", unit(0)),
        row("gate", y + 100, 4, "person", "p.jpg", unit(1)),   # person walking a dog
        row("gate", y + 130, 4, "person", "p.jpg", unit(1)),
        row("gate", y + 110, 5, "dog", "d.jpg", unit(2)),
        row("gate", y + 120, 5, "dog", "d.jpg", unit(2)),
        row("gate", y + 600, 6, "person", "p2.jpg", unit(1)),  # alone
        row("yard", y + 100, 1, "dog", "yd.jpg", unit(2)),     # same time, other camera
        row("gate", NOW.timestamp() - 600, 9, "car", "c.jpg", unit(1)),  # today
    ])
    box = Toolbox(SearchEngine(store, FakeClip(), camera_names=["gate", "yard"]), now=NOW)
    yield box
    box.close()
    store.close()


def test_objects_by_label_and_time(toolbox):
    out = toolbox.objects(labels=["car"], when="yesterday")
    assert out["total"] == 2 and [v["track_id"] for v in out["visits"]] == [1, 2]
    assert toolbox.objects(labels="Car ")["total"] == 3  # any time, label normalized


def test_objects_by_time_only(toolbox):
    out = toolbox.objects(when="yesterday")
    assert out["counts"] == {"car": 2, "person": 2, "dog": 2, "truck": 1}
    assert [v["first_seen"][11:] for v in out["visits"][:2]] == ["18:00:00", "18:01:00"]
    assert toolbox.objects(when="yesterday on yard")["counts"] == {"dog": 1}
    assert toolbox.objects(when="yesterday", camera="gate")["counts"]["dog"] == 1


def test_objects_exact_iso_range(toolbox):
    out = toolbox.objects(labels=["car", "truck"], since="2026-10-02 18:01",
                          until="2026-10-02T18:03")
    assert [v["label"] for v in out["visits"]] == ["car", "truck"]
    r = toolbox.run(ToolCall("1", "objects", {"since": "friday evening"}))
    assert r.is_error and "ISO" in r.output


def test_objects_together(toolbox):
    out = toolbox.objects(labels=["person"], with_labels=["dog"])
    assert [v["track_id"] for v in out["visits"]] == [4]
    assert out["visits"][0]["with"] == [{"label": "dog", "track_id": 5,
                                         "first_seen": "2026-10-02 18:01:50"}]
    assert toolbox.objects(labels=["car"], with_labels=["dog"])["total"] == 0
    assert toolbox.objects(labels=["car"], with_labels=["dog"], within_seconds=60)["total"] == 1


def test_search_with_time_and_objects(toolbox):
    out = json.loads(toolbox.run(ToolCall("1", "search", {"query": "white van yesterday"})).output)
    assert out["results"][0]["thumb"] == "van.jpg" and out["looked_for"] == "white van"
    assert toolbox.evidence[0] == "van.jpg"
    out = toolbox.search("someone", with_labels=["dog"])
    assert [r["track_id"] for r in out["results"]] == [4]
    out = toolbox.search("car", since="2026-10-03 00:00")
    assert [r["thumb"] for r in out["results"]] == ["c.jpg"]


def test_tool_errors_go_back_to_the_model(toolbox):
    r = toolbox.run(ToolCall("1", "objects", {"labels": ["van"]}))
    assert r.is_error and "search tool" in r.output
    assert toolbox.run(ToolCall("2", "objects", {"with_labels": ["unicorn"]})).is_error
    assert toolbox.run(ToolCall("3", "nope", {})).is_error
    assert toolbox.run(ToolCall("4", "search", {"wrong": 1})).is_error


def test_coverage(toolbox):
    assert [c["camera"] for c in toolbox.coverage()] == ["gate", "yard"]


class ScriptedBackend:
    name = "fake"

    def __init__(self, steps):
        self.steps, self.results, self.users = list(steps), [], []

    def add_user(self, text):
        self.users.append(text)

    def step(self):
        return self.steps.pop(0)

    def add_results(self, results):
        self.results.append(results)


def test_answer_runs_tools_until_text(toolbox):
    backend = ScriptedBackend([
        Step("", [ToolCall("1", "objects", {"labels": ["car"], "when": "yesterday"})]),
        Step("Two cars came in yesterday."),
    ])
    assert answer(backend, toolbox, "how many cars yesterday?") == "Two cars came in yesterday."
    assert json.loads(backend.results[0][0].output)["total"] == 2


def test_answer_stops_after_max_rounds(toolbox):
    loop = Step("", [ToolCall("1", "objects", {"labels": ["car"]})])
    assert "too many tool calls" in answer(ScriptedBackend([loop] * 10), toolbox, "?")


def test_openai_compatible_parses_tool_calls_and_strips_thinking(monkeypatch):
    backend = OpenAICompatible("sys", TOOLS, model="qwen3:4b")
    replies = iter([
        {"choices": [{"message": {"role": "assistant", "content": "<think>hmm</think>", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "objects", "arguments": '{"labels": ["car"]}'}}]}}]},
        {"choices": [{"message": {"role": "assistant", "content": "<think>ok</think>\nThree cars."}}]},
    ])
    sent = []
    monkeypatch.setattr(backend, "_post", lambda body: sent.append(body) or next(replies))
    backend.add_user("how many cars?")
    step = backend.step()
    assert step.text == "" and step.calls == [ToolCall("c1", "objects", {"labels": ["car"]})]
    backend.add_results([ToolResult(step.calls[0], '{"count": 3}')])
    assert backend.step().text == "Three cars."
    assert [t["function"]["name"] for t in sent[0]["tools"]] == ["objects", "search", "alerts"]
    assert [m["role"] for m in backend.messages] == ["system", "user", "assistant", "tool",
                                                     "assistant"]


def test_claude_backend_message_shapes():
    backend = Claude.__new__(Claude)
    backend.model, backend.effort, backend.max_tokens, backend.system = "claude-opus-5-5", "low", 100, "s"
    backend.tools = [{"name": "objects", "description": "d", "input_schema": {}}]
    backend.messages = []
    calls = []
    content = [SimpleNamespace(type="text", text="Let me check."),
               SimpleNamespace(type="tool_use", id="t1", name="objects", input={"label": "dog"})]
    backend.client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(
        create=lambda **kw: calls.append(kw) or SimpleNamespace(stop_reason="tool_use",
                                                                content=content))))
    backend.add_user("any dogs?")
    step = backend.step()
    assert step.calls == [ToolCall("t1", "objects", {"label": "dog"})]
    assert calls[0]["fallbacks"] == "default" and calls[0]["output_config"] == {"effort": "low"}
    assert backend.messages[1] == {"role": "assistant", "content": content}  # unchanged blocks
    backend.add_results([ToolResult(step.calls[0], "err", is_error=True)])
    assert backend.messages[2]["content"][0] == {"type": "tool_result", "tool_use_id": "t1",
                                                 "content": "err", "is_error": True}


def test_model_switch_config(monkeypatch):
    monkeypatch.setenv("KEY", "secret")
    config = {"chat": {"models": {"hosted": {"api": "openai", "model": "m",
                                             "base_url": "https://x/v1", "api_key": "${KEY}"}}}}
    backend = create_backend(model_config(config, "hosted"), "sys", [Tool("t", "d", {})])
    assert isinstance(backend, OpenAICompatible) and backend.api_key == "secret"
    assert backend.url == "https://x/v1/chat/completions"
    with pytest.raises(SystemExit):
        model_config(config, "missing")
    with pytest.raises(ValueError):
        create_backend({"api": "nope"}, "s", [])


# ------------------------------------------------------------------ face search + camera context

CAMERAS = {
    "office": {"url": "rtsp://a", "about": {
        "description": "Inside the front office, above the door, looking at the desk",
        "aliases": ["front office"], "covers": "entrance", "faces": "good",
        "hours": "08:00-18:00", "zones": {"door": "the office door"}}},
    "lobby": {"url": "rtsp://b", "about": {"description": "Reception", "faces": "poor"}},
}


def face_vec(cos):
    v = np.zeros(128, dtype=np.float32)
    v[0], v[1] = cos, np.sqrt(1 - cos * cos)
    return v


class FakeFaces:
    """A bright photo has one usable face (print = face_vec(1)); a dark one has none."""

    def photo_faces(self, image):
        if image.mean() < 50:
            return []
        return [PhotoFace(face_vec(1.0), (0, 0, 80, 80), FaceQuality(0.9, 80, 0.1, 120), [])]


TODAY_1340 = datetime(2026, 10, 3, 13, 40).timestamp()


@pytest.fixture
def face_toolbox(tmp_path):
    store = EventStore(tmp_path / "f.db")
    t = TODAY_1340
    store.add_sightings([row("office", t + s, 7, "person", "o.jpg", unit(1)) for s in (0, 300)])
    store.add_faces([
        {"camera": "office", "time": t + 240, "track_id": 7, "thumb": "f7.jpg",
         "embedding": face_vec(0.8)},
        {"camera": "office", "time": t + 250, "track_id": 7, "embedding": face_vec(0.7)},
        {"camera": "lobby", "time": t + 30, "track_id": 3, "embedding": face_vec(0.1)},
    ])
    engine = SearchEngine(store, FakeClip(), FakeFaces(), camera_names=["lobby", "office"],
                          camera_aliases={"front office": "office"})
    box = Toolbox(engine, now=datetime(2026, 10, 3, 17, 0), cameras=CAMERAS,
                  faces_config={"enabled": True, "cameras": ["office"]})
    yield box
    box.close()
    store.close()


def test_attach_photo(face_toolbox):
    assert face_toolbox.add_photo([np.zeros((50, 50, 3), np.uint8)]) == (
        None, "No face found in the photo.")
    bright = np.full((50, 50, 3), 200, np.uint8)
    assert face_toolbox.add_photo([bright]) == ("photo1", "Attached as photo1")
    assert face_toolbox.add_photo([bright, bright])[0] == "photo2"
    assert len(face_toolbox.photos["photo2"]) == 2
    assert [t.name for t in face_toolbox.tools][-1] == "face"


def test_face_tool(face_toolbox):
    face_toolbox.add_photo([np.full((50, 50, 3), 200, np.uint8)])
    r = face_toolbox.run(ToolCall("1", "face", {
        "photo": "photo1", "since": "2026-10-03 13:00", "until": "2026-10-03 16:00",
        "camera": "Front Office"}))  # an alias
    out = json.loads(r.output)
    assert not r.is_error and out["cameras"] == ["office"] and out["likely"] == 1
    visit = out["visits"][0]
    assert (visit["first_seen"], visit["last_seen"]) == ("2026-10-03 13:40:00",
                                                         "2026-10-03 13:45:00")
    assert visit["face_seen"] == "2026-10-03 13:44:00" and visit["frames_matched"] == 2
    assert out["checked"] == [{"camera": "office", "faces_recorded": 2,
                               "face_recording_on": True, "face_view": "good"}]
    assert face_toolbox.evidence == ["f7.jpg"]
    # all cameras: the lobby recorded a face, but not this person, and can't see faces well
    out = face_toolbox.face("photo1", when="today")
    assert out["likely"] == 1 and out["checked"][0] == {
        "camera": "lobby", "faces_recorded": 1, "face_recording_on": False, "face_view": "poor"}
    assert "Attached: photo1" in face_toolbox.run(ToolCall("2", "face", {"photo": "photo9"})).output
    assert face_toolbox.run(ToolCall("3", "face", {"photo": "photo1", "camera": "roof"})).is_error


def test_face_tool_only_with_face_models(toolbox):
    assert [t.name for t in toolbox.tools] == ["objects", "search", "alerts"]
    assert toolbox.run(ToolCall("1", "face", {"photo": "photo1"})).is_error
    assert FACE_HELP not in system_prompt(toolbox)


def test_system_prompt_describes_cameras(face_toolbox):
    prompt = system_prompt(face_toolbox, now=datetime(2026, 10, 3, 17, 0))
    assert ("- office (also: front office): Inside the front office, above the door, looking at "
            "the desk. covers: entrance. faces visible: good. usual hours: 08:00-18:00. "
            "areas: door = the office door. Recorded 2026-10-03 13:40:00 to 2026-10-03 13:45:00."
            ) in prompt
    assert "- lobby: Reception. faces visible: poor. Nothing recorded yet." in prompt
    assert FACE_HELP in prompt and "Saturday 2026-10-03 17:00" in prompt


def test_parse_photo_command():
    assert parse_photo_command(' a.jpg "my photos/b.jpg"') == (["a.jpg", "my photos/b.jpg"], None)
    assert parse_photo_command(" group.jpg --face 2") == (["group.jpg"], 2)
    with pytest.raises(ValueError):
        parse_photo_command(" --face x")
    with pytest.raises(ValueError):
        parse_photo_command("")
