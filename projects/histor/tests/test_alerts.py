import json
from datetime import datetime

import pytest
from histor.chat import system_prompt
from histor.llm import ToolCall
from histor.engine.search import SearchEngine
from histor.engine.store import EventStore
from histor.tools import TOOLS, Toolbox

NOW = datetime(2026, 10, 6, 9, 0)  # Tuesday morning


def at(day: int, hour: int, minute: int = 0) -> float:
    return datetime(2026, 10, day, hour, minute).timestamp()


@pytest.fixture
def toolbox(tmp_path):
    store = EventStore(tmp_path / "h.db")
    for camera, t, event, snapshot in [
        ("yard", at(5, 22, 10), {"type": "enter", "rule": "yard", "label": "person", "track_id": 1},
         "snapshots/y1.jpg"),
        ("yard", at(5, 22, 40), {"type": "exit", "rule": "yard", "label": "person", "track_id": 1,
                                 "dwell_seconds": 1800.0}, None),
        ("gate", at(6, 2, 0), {"type": "cross", "rule": "gate_line", "label": "car", "track_id": 7,
                               "direction": "in"}, "snapshots/g7.jpg"),
        ("yard", at(5, 14, 0), {"type": "dwell", "rule": "fire_lane", "label": "car",
                                "track_id": 3, "dwell_seconds": 61.0}, "snapshots/f3.jpg"),
    ]:
        store.add_event(camera, t, event, snapshot)
    box = Toolbox(SearchEngine(store, None, camera_names=["gate", "yard"]), now=NOW)
    yield box
    box.close()
    store.close()


def test_alerts_tool_is_offered():
    assert [t.name for t in TOOLS] == ["objects", "search", "alerts"]


def test_alerts_last_night_in_time_order(toolbox):
    out = toolbox.alerts(when="last night")  # Mon 21:00 - Tue 06:00, asked Tue morning
    assert out["total"] == 3
    assert [(e["rule"], e["type"]) for e in out["events"]] == [
        ("yard", "enter"), ("yard", "exit"), ("gate_line", "cross")]
    assert out["events"][2]["direction"] == "in" and out["events"][0]["time"] == "2026-10-05 22:10:00"
    assert out["counts"] == {"yard enter": 1, "yard exit": 1, "gate_line cross": 1}
    assert toolbox.evidence == ["snapshots/y1.jpg", "snapshots/g7.jpg"]


def test_alerts_filters(toolbox):
    assert toolbox.alerts(rules=["yard"], types=["enter"])["total"] == 1
    assert toolbox.alerts(labels=["car"])["total"] == 2
    assert toolbox.alerts(camera="gate")["total"] == 1
    assert toolbox.alerts(when="yesterday", types="dwell")["events"][0]["dwell_seconds"] == 61.0


def test_alerts_bad_type_goes_back_to_the_model(toolbox):
    r = toolbox.run(ToolCall("1", "alerts", {"types": ["fire"]}))
    assert r.is_error and "enter, exit, dwell, cross" in r.output
    ok = json.loads(toolbox.run(ToolCall("2", "alerts", {"rules": ["gate_line"]})).output)
    assert ok["total"] == 1


def test_system_prompt_lists_alert_rules(toolbox):
    prompt = system_prompt(toolbox)
    assert "- gate: gate_line (1 events)" in prompt and "- yard: yard (2 events)" in prompt
