"""Alert verification (verify.py): a vision model looks at each alert before it is sent."""

from __future__ import annotations

import json
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from histor.engine import alerts
from histor.engine.alerts import AlertManager, Sink, alert_message
from histor.engine.tracker import Track
from histor.engine.verify import QUESTIONS, AlertVerifier, close_up, parse_answer
from histor.engine.zones import ZoneEvent

NIGHT = datetime(2026, 10, 10, 23, 0).timestamp()


class Collect(Sink):
    got: list[dict] = []

    def __init__(self) -> None:
        Collect.got = []

    def send(self, payload, jpeg):
        Collect.got.append(payload)


class FakeModel:
    """Stands in for VisionModel.ask: canned answer (or exception), records what it saw."""

    def __init__(self, answer: str | Exception = "YES\nA person is lying on the floor.") -> None:
        self.answer = answer
        self.calls: list[tuple[list[np.ndarray], str, str]] = []

    def __call__(self, images, question, system):
        self.calls.append((images, question, system))
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


@pytest.fixture(autouse=True)
def collect_channel(monkeypatch):
    monkeypatch.setitem(alerts.CHANNELS, "collect", Collect)


def person(track_id: int = 7) -> Track:
    return Track(track_id, np.array([40.0, 60.0, 80.0, 150.0]), 0.9, label_votes={"person": 1.0})


def manager(store, model: FakeModel, **verify) -> AlertManager:
    verifier = AlertVerifier(model, **verify)
    return AlertManager(console=False, store=store, source="hall", verify=verifier,
                        channels=[{"type": "collect"}])


def fire(m: AlertManager, event: ZoneEvent, tracks=None) -> None:
    frame = np.full((240, 320, 3), 90, np.uint8)
    m.handle([event], frame, tracks if tracks is not None else [person()], NIGHT)
    m.close()


def stored(store) -> list[dict]:
    return store.list_events()


@pytest.mark.parametrize("text, verdict, reason", [
    ("YES\nSomeone is lying on the floor.", "yes", "Someone is lying on the floor."),
    ("No. It is a coat on a chair.", "no", "It is a coat on a chair."),
    ("**UNSURE**\nToo dark.", "unsure", "Too dark."),
    ("Maybe a person", "unsure", "Maybe a person"),
    ("", "unsure", ""),
    ("yes - both hands are up", "yes", "both hands are up"),
])
def test_parse_answer(text, verdict, reason):
    assert parse_answer(text) == (verdict, reason)


def test_close_up_pads_clamps_and_enlarges():
    frame = np.zeros((100, 200, 3), np.uint8)
    crop = close_up(frame, [10, 10, 30, 50], pad=0.5, min_side=96)
    # box 20x40 widened by half each way: x 0..40 (clamped at 0), y 0..70 -> 40x70, then x2.4
    assert crop.shape[:2] == (168, 96)
    assert close_up(frame, [0, 0, 200, 100], pad=0.5).shape[:2] == (100, 200)  # whole frame


def test_yes_is_sent_with_the_verdict_and_stored(store):
    model = FakeModel()
    fire(manager(store, model), ZoneEvent(1.0, "down", "person_down", 7, "person"))
    (sent,) = Collect.got
    assert sent["verified"] == "yes" and sent["verification"] == "A person is lying on the floor."
    (row,) = stored(store)
    assert row["verified"] == "yes" and row["verification"].startswith("A person")
    images, question, system = model.calls[0]
    assert len(images) == 2  # snapshot + close-up of the person
    assert question == QUESTIONS["down"] and "YES, NO or UNSURE" in system
    assert "Check:  yes (A person is lying on the floor.)" in alert_message(sent)[1]


def test_no_holds_the_alert_back_but_keeps_it_in_the_store(store):
    fire(manager(store, FakeModel("NO\nA coat on the floor.")),
         ZoneEvent(1.0, "down", "person_down", 7, "person"))
    assert Collect.got == []
    (row,) = stored(store)
    assert (row["verified"], row["verification"]) == ("no", "A coat on the floor.")
    assert row["snapshot"]  # the evidence is kept for a person to check


def test_tag_mode_sends_a_no_too(store):
    fire(manager(store, FakeModel("NO\nA shadow."), mode="tag"),
         ZoneEvent(1.0, "enter", "yard", 7, "person"))
    assert [p["verified"] for p in Collect.got] == ["no"]


def test_model_errors_never_lose_an_alert(store):
    fire(manager(store, FakeModel(RuntimeError("Can't reach the model server"))),
         ZoneEvent(1.0, "down", "person_down", 7, "person"))
    (sent,) = Collect.got
    assert sent["verified"] == "error" and "Can't reach" in sent["verification"]
    assert stored(store)[0]["verified"] == "error"


def test_only_listed_events_and_rules_are_checked(store):
    model = FakeModel("NO\nNothing there.")
    m = manager(store, model, events=["enter"], rules=["yard"])
    frame = np.full((240, 320, 3), 90, np.uint8)
    m.handle([ZoneEvent(1.0, "enter", "gate", 7, "person"),       # other rule: not checked
              ZoneEvent(2.0, "dwell", "yard", 7, "person"),       # other type: not checked
              ZoneEvent(3.0, "enter", "yard", 7, "person")],      # checked: NO -> held back
             frame, [person()], NIGHT)
    m.close()
    assert [(p["rule"], p["type"]) for p in Collect.got] == [("gate", "enter"), ("yard", "dwell")]
    assert "verified" not in Collect.got[0] and len(model.calls) == 1


def test_events_without_a_question_or_an_image_go_straight_out(store):
    model = FakeModel("NO")
    m = manager(store, model)
    m.handle([ZoneEvent(1.0, "sound", "smoke_alarm", None, "Smoke detector, smoke alarm")],
             np.zeros((240, 320, 3), np.uint8), [], NIGHT)
    m.handle([ZoneEvent(2.0, "down", "person_down", 7, "person")], None, None, NIGHT)  # no frame
    m.close()
    assert [p["type"] for p in Collect.got] == ["sound", "down"] and model.calls == []


def test_crowd_question_uses_the_count_and_only_the_snapshot(store):
    model = FakeModel("YES\nSix people.")
    fire(manager(store, model), ZoneEvent(1.0, "crowd", "till", None, "person", count=6), [])
    images, question, _ = model.calls[0]
    assert len(images) == 1 and question == "Are there at least 6 people in the picture?"


def test_full_queue_sends_unverified(store):
    gate, busy = threading.Event(), threading.Event()

    def slow(images, question, system):
        busy.set()
        gate.wait(5)
        return "YES\nok"

    m = AlertManager(console=False, store=store, source="hall", channels=[{"type": "collect"}],
                     verify=AlertVerifier(slow, max_queue=1))
    frame = np.full((240, 320, 3), 90, np.uint8)
    events = [ZoneEvent(float(i), "down", "person_down", 7, "person") for i in range(3)]
    for e in events:  # 1st is being checked, 2nd waits in the queue, 3rd finds it full
        m.handle([e], frame, [person()], NIGHT + e.timestamp)
        busy.wait(5)
    gate.set()
    m.close()
    assert sorted(p["verified"] for p in Collect.got) == ["skipped", "yes", "yes"]


def test_config_errors():
    assert AlertVerifier.from_config(None) is None
    assert AlertVerifier.from_config({"enabled": False}) is None
    with pytest.raises(ValueError, match="needs a vision model"):
        AlertVerifier.from_config({"enabled": True})
    with pytest.raises(ValueError, match="no question for"):
        AlertVerifier(FakeModel(), events=["tamper"])
    with pytest.raises(ValueError, match="veto or tag"):
        AlertVerifier(FakeModel(), mode="drop")
    custom = AlertVerifier(FakeModel(), events=["tamper"],
                           questions={"tamper": "Is the view blocked?"})
    assert custom.question({"type": "tamper", "rule": "blocked"}) == "Is the view blocked?"
    custom.close()


def test_real_http_request_with_chat_vision_defaults(store):
    """From config through VisionModel to an OpenAI-compatible server, like Ollama."""
    requests: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            body = json.dumps({"choices": [{"message": {"content": "<think>hm</think>NO\n"
                                                        "A jacket on the floor."}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    vision = {"model": "qwen3.5:4b", "max_images": 4, "max_side": 320,
              "base_url": f"http://127.0.0.1:{server.server_port}/v1"}
    m = AlertManager.from_config({"console": False, "channels": [{"type": "collect"}],
                                  "verify": {"enabled": True, "events": ["down"]}},
                                 source="hall", store=store, vision=vision)
    fire(m, ZoneEvent(1.0, "down", "person_down", 7, "person"))
    server.shutdown()
    assert Collect.got == [] and stored(store)[0]["verified"] == "no"
    (req,) = requests
    assert req["model"] == "qwen3.5:4b"
    assert req["messages"][0]["content"].startswith("You check alarms")
    parts = req["messages"][1]["content"]
    assert parts[0]["text"] == QUESTIONS["down"]
    assert [p["type"] for p in parts[1:]] == ["image_url", "image_url"]
