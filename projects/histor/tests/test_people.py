"""The `people` tool: people by body pose and clothing colour, and pose alerts in `alerts`."""

import json
from datetime import datetime

import pytest
from histor.chat import system_prompt
from histor.llm import ToolCall
from histor.engine.search import SearchEngine
from histor.engine.store import EventStore
from histor.tools import Toolbox

NOW = datetime(2026, 10, 6, 9, 0)  # Tuesday morning


def at(day: int, hour: int, minute: int = 0, second: int = 0) -> float:
    return datetime(2026, 10, day, hour, minute, second).timestamp()


def rows(camera: str, track_id: int, start: float, **facts) -> list[dict]:
    """30 s of one person with the same facts: a row every 10 s (heartbeat)."""
    return [{"camera": camera, "time": start + i, "track_id": track_id, "posture": "standing",
             "arms": "down", "facing": "camera", "upper_colour": None, "lower_colour": None,
             **facts} for i in range(0, 30, 10)]


@pytest.fixture
def store(tmp_path):
    s = EventStore(tmp_path / "p.db")
    s.add_poses(rows("lobby", 1, at(5, 23, 10), posture="lying", upper_colour="red")
                + rows("lobby", 2, at(6, 8, 0), upper_colour="red", lower_colour="blue")
                + rows("gate", 3, at(6, 8, 30), arms="both_up", upper_colour="black",
                       motion="running", carrying="backpack"))
    s.add_sightings([{"camera": "lobby", "time": at(6, 8, 0, 5), "kind": "object", "track_id": 2,
                      "label": "person", "thumb": "thumbs/p2.jpg", "embedding": None},
                     {"camera": "lobby", "time": at(6, 8, 0, 5), "kind": "object", "track_id": 9,
                      "label": "dog", "thumb": "thumbs/d9.jpg", "embedding": None}])
    s.add_event("lobby", at(5, 23, 10, 20), {"type": "down", "rule": "person_down",
                                             "label": "person", "track_id": 1,
                                             "dwell_seconds": 10.0}, "snapshots/down1.jpg")
    yield s
    s.close()


@pytest.fixture
def toolbox(store):
    box = Toolbox(SearchEngine(store, None, camera_names=["gate", "lobby"]), now=NOW)
    yield box
    box.close()


def test_people_tool_offered_when_poses_exist(toolbox, tmp_path):
    assert "people" in [t.name for t in toolbox.tools]
    assert "`people` finds people by body pose" in system_prompt(toolbox)
    empty = EventStore(tmp_path / "empty.db")
    box = Toolbox(SearchEngine(empty, None))
    assert "people" not in [t.name for t in box.tools]
    assert "people" in [t.name for t in Toolbox(SearchEngine(empty, None),
                                                pose_config={"enabled": True}).tools]
    box.close()
    empty.close()


def test_people_by_posture_colour_and_arms(toolbox):
    out = toolbox.people(postures=["lying"], when="last night")
    assert out["total"] == 1 and out["visits"][0]["camera"] == "lobby"
    assert out["visits"][0]["posture_seconds"] == {"lying": 21.0}
    assert out["visits"][0]["first_seen"] == "2026-10-05 23:10:00"
    out = toolbox.people(upper_colour="red", lower_colour="blue", when="today")
    assert [v["track_id"] for v in out["visits"]] == [2]
    assert out["visits"][0]["thumb"] == "thumbs/p2.jpg" and toolbox.evidence == ["thumbs/p2.jpg"]
    assert [v["track_id"] for v in toolbox.people(arms="up")["visits"]] == [3]
    assert toolbox.people(upper_colour="red", camera="gate")["total"] == 0
    assert toolbox.people(upper_colour="red", with_labels=["dog"])["total"] == 1
    out = toolbox.people(motion="running", carrying=["backpack"])
    assert [v["track_id"] for v in out["visits"]] == [3]
    assert out["visits"][0]["carrying"] == ["backpack"]
    assert toolbox.people(head_down=True)["total"] == 0


def test_people_bad_values_go_back_to_the_model(toolbox):
    r = toolbox.run(ToolCall("1", "people", {"upper_colour": "navy"}))
    assert r.is_error and "blue" in r.output
    ok = json.loads(toolbox.run(ToolCall("2", "people", {"postures": ["lying"]})).output)
    assert ok["total"] == 1


def test_pose_alerts_in_alerts_tool(toolbox):
    out = toolbox.alerts(types=["down"], when="last night")
    assert out["total"] == 1 and out["events"][0]["rule"] == "person_down"
