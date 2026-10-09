"""Body poses: facts from keypoints, clothing colours, storage on change, person-down events and
people search. Synthetic keypoints and a fake model; the real models only in the last test."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from argus.engine.pose import (L_ANK, L_EYE, L_HIP, L_KNEE, L_SHO, L_WRI, NOSE, R_ANK, R_EYE, R_HIP, R_KNEE,
                  R_SHO, R_WRI, _decode_heatmaps, describe, pack_keypoints, pose_enabled,
                  unpack_keypoints)
from argus.engine.pose_index import PoseIndexer, _carrying, _motion
from argus.engine.search import SearchEngine, match_people, person_visits
from argus.engine.store import EventStore
from argus.engine.tracker import Track

ROOT = Path(__file__).resolve().parents[2]


def person(**moves: tuple[float, float]) -> np.ndarray:
    """An upright person facing the camera (torso 60 px), with some joints moved."""
    pts = {
        "nose": (50, 20), "left_eye": (54, 18), "right_eye": (46, 18), "left_ear": (58, 20),
        "right_ear": (42, 20), "left_shoulder": (65, 40), "right_shoulder": (35, 40),
        "left_elbow": (68, 60), "right_elbow": (32, 60), "left_wrist": (70, 80),
        "right_wrist": (30, 80), "left_hip": (60, 100), "right_hip": (40, 100),
        "left_knee": (60, 140), "right_knee": (40, 140), "left_ankle": (60, 180),
        "right_ankle": (40, 180)}
    pts.update(moves)
    return np.array([[x, y, 0.9] for x, y in pts.values()], dtype=np.float32)


@pytest.fixture
def store(tmp_path):
    s = EventStore(tmp_path / "db.sqlite")
    yield s
    s.close()


def test_postures():
    assert describe(person()).posture == "standing"
    sitting = person(left_knee=(60, 105), right_knee=(40, 105), left_ankle=(60, 150),
                     right_ankle=(40, 150))
    assert describe(sitting).posture == "sitting"
    crouching = person(left_knee=(60, 110), right_knee=(40, 110), left_ankle=(60, 125),
                       right_ankle=(40, 125))
    assert describe(crouching).posture == "crouching"
    lying = person()[:, [1, 0, 2]]  # turned 90 degrees: head on the left, feet on the right
    facts = describe(lying)
    assert facts.posture == "lying" and facts.arms == "unknown"
    upper_body = person()
    upper_body[[L_KNEE, R_KNEE, L_ANK, R_ANK], 2] = 0.05  # behind a desk
    assert describe(upper_body).posture == "unknown"
    # seen from the side, bent over at the hips with straight legs: not lying
    bent = person(nose=(-20, 100), left_shoulder=(-5, 95), right_shoulder=(-5, 95))
    assert describe(bent).posture == "bending"
    leaning = person(left_shoulder=(25, 55), right_shoulder=(5, 55))  # ~45 degrees forward
    assert describe(leaning).posture == "bending"


def test_head_down():
    assert describe(person()).head == "up"
    assert describe(person(nose=(50, 32))).head == "down"  # nose well below the ears
    no_ears = person()
    no_ears[[3, 4], 2] = 0.1
    assert describe(no_ears).head == "unknown"
    assert describe(person()[:, [1, 0, 2]]).head == "unknown"  # lying


def test_arms_and_facing():
    assert describe(person()).arms == "down"
    assert describe(person(left_wrist=(70, 0))).arms == "one_up"
    assert describe(person(left_wrist=(70, 0), right_wrist=(30, 2))).arms == "both_up"
    assert describe(person(left_wrist=(70, 15))).arms == "down"  # level with the eyes
    assert describe(person()).facing == "camera"
    back = person(left_shoulder=(35, 40), right_shoulder=(65, 40))  # left shoulder image-left
    assert describe(back).facing == "away"  # even with face points on the back of the head
    profile = person()
    profile[[R_EYE], 2] = 0.1
    assert describe(profile).facing == "side"
    nothing = person()
    nothing[:, 2] = 0.1
    facts = describe(nothing)
    assert (facts.posture, facts.arms, facts.facing, facts.visible) == (
        "unknown", "unknown", "unknown", 0)


def test_clothing_colours():
    image = np.full((200, 120, 3), 128, np.uint8)
    kp = person()
    torso = kp[[L_SHO, R_SHO, R_HIP, L_HIP], :2].astype(np.int32)
    cv2.fillConvexPoly(image, torso, (0, 0, 200))                     # red top
    for hip, ankle in ((L_HIP, L_ANK), (R_HIP, R_ANK)):
        cv2.line(image, tuple(kp[hip, :2].astype(int)), tuple(kp[ankle, :2].astype(int)),
                 (200, 60, 0), 14)                                    # blue trousers
    facts = describe(kp, image)
    assert (facts.upper_colour, facts.lower_colour) == ("red", "blue")
    assert describe(kp, np.full((200, 120, 3), 128, np.uint8)).upper_colour == "grey"
    assert describe(kp).upper_colour is None  # no image, no colours


def test_pack_and_decode():
    kp = person()
    box = np.array([20, 10, 80, 190])
    again = unpack_keypoints(pack_keypoints(kp, box), box)
    assert np.abs(again[:, :2] - kp[:, :2]).max() < 0.2 and len(pack_keypoints(kp, box)) == 102
    # the same keypoints on a box twice as big (main stream): twice the coordinates
    assert np.allclose(unpack_keypoints(pack_keypoints(kp, box), box * 2)[:, :2],
                       kp[:, :2] * 2, atol=0.5)
    heatmaps = np.zeros((1, 17, 64, 48), np.float32)
    heatmaps[0, NOSE, 10, 20] = 0.9
    heatmaps[0, NOSE, 10, 21] = 0.5  # pulls the peak a quarter cell right
    decoded = _decode_heatmaps(heatmaps)
    assert decoded[0, NOSE, 2] == pytest.approx(0.9)
    assert decoded[0, NOSE, 0] == pytest.approx((20.25 + 0.5) * 4 - 0.5)
    assert decoded[0, NOSE, 1] == pytest.approx((10 + 0.5) * 4 - 0.5)


def test_pose_enabled_rules():
    assert not pose_enabled(None, "gate") and not pose_enabled({"enabled": False}, "gate")
    assert pose_enabled({"enabled": True}, "gate")
    assert not pose_enabled({"enabled": True, "cameras": ["lobby"]}, "gate")


class FakePose:
    """`pose_for(box)` decides each person's keypoints (relative to the box). The worker runs
    later than update(), so the pose must come from the job itself, not from a shared value."""

    min_confidence = 0.3

    def __init__(self, pose_for=lambda box: person()) -> None:
        self.pose_for = pose_for
        self.calls = 0

    def estimate(self, image, boxes):
        self.calls += 1
        return [self.pose_for(b) + np.array([b[0], b[1], 0], np.float32) for b in boxes]


def track(box: list[float]) -> Track:
    return Track(id=7, box=np.array(box, dtype=float), score=0.9, label_votes={"person": 1.0},
                 hits=5)


def test_indexer_stores_changes_and_fires_down(store):
    lying = person()[:, [1, 0, 2]]
    fake = FakePose(lambda box: person() if box[0] < 110 else lying)  # lying from the 6th second
    indexer = PoseIndexer(store, "lobby", fake, interval_seconds=1, still_factor=3,
                          heartbeat_seconds=10, down_seconds=5, blocking=True)
    frame = np.zeros((300, 800, 3), np.uint8)  # wide enough for the whole walk
    t0 = datetime(2026, 10, 8, 9).timestamp()
    for i in range(30):
        box = [10 + i * 20, 10, 110 + i * 20, 210]  # walking: sampled every second
        indexer.update(frame, [track(box)], t0 + i)
    indexer.close()
    rows = store.pose_rows()
    postures = [r["posture"] for r in rows]
    # standing (motion unknown on the first sample, then still: 0.1 heights a second), lying
    # once 2 of the last 3 samples agree, then a heartbeat every 10 s
    assert postures == ["standing", "standing", "lying", "lying", "lying"]
    assert [r["motion"] for r in rows[:2]] == ["unknown", "still"]
    assert rows[2]["time"] == t0 + 6 and rows[3]["time"] == t0 + 16
    events = indexer.take_events()
    assert [(e[1], e[2]) for e in events] == [(7, "down")] and events[0][0] == t0 + 11
    assert indexer.overlay(t0 + 29) and not indexer.overlay(t0 + 60)


def test_motion_and_carrying():
    box = np.array([0, 0, 100, 200], dtype=float)
    assert _motion(box, None, 10.0) == "unknown"
    assert _motion(box, (9.0, box), 10.0) == "still"
    assert _motion(box + [100, 0, 100, 0], (9.0, box), 10.0) == "walking"   # 0.5 heights/s
    assert _motion(box + [400, 0, 400, 0], (9.0, box), 10.0) == "running"   # 2 heights/s
    closer = np.array([-25, -50, 125, 250], dtype=float)  # 1.5x taller: coming towards us
    assert _motion(closer, (9.0, box), 10.0) == "walking"
    assert _motion(box, (0.0, box), 30.0) == "unknown"  # too long ago to tell
    bag = ("backpack", np.array([20, 40, 80, 120], dtype=float))
    far = ("suitcase", np.array([300, 100, 360, 190], dtype=float))
    beside = ("handbag", np.array([95, 100, 125, 140], dtype=float))  # centre just outside
    assert _carrying(box, [bag, far, beside]) == "backpack,handbag"
    assert _carrying(box, [far]) is None


def test_indexer_records_motion_and_carrying(store):
    indexer = PoseIndexer(store, "gate", FakePose(), interval_seconds=1, heartbeat_seconds=2,
                          blocking=True)
    frame = np.zeros((300, 1200, 3), np.uint8)
    for i in range(6):
        x = 10 + i * 150  # 0.75 body heights a second
        person_track = track([x, 10, x + 100, 210])
        suitcase = Track(id=8, box=np.array([x + 60, 120, x + 110, 200], dtype=float), score=0.8,
                         label_votes={"suitcase": 1.0}, hits=5)
        indexer.update(frame, [person_track, suitcase], 1000.0 + i)
    indexer.close()
    rows = store.pose_rows()
    assert rows[0]["motion"] == "unknown" and rows[-1]["motion"] == "walking"
    assert {r["carrying"] for r in rows} == {"suitcase"}
    visit = person_visits(rows)[0]
    assert match_people([visit], motion=["walking"], carrying=["suitcase"]) == [visit]
    assert match_people([visit], motion=["running"]) == []
    assert visit.to_dict()["carrying"] == ["suitcase"]


def test_old_store_gets_the_new_columns(tmp_path):
    import sqlite3

    path = tmp_path / "old.db"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE poses (id INTEGER PRIMARY KEY, camera TEXT NOT NULL, time REAL NOT "
               "NULL, track_id INTEGER, posture TEXT, arms TEXT, facing TEXT, upper_colour TEXT, "
               "lower_colour TEXT, score REAL, box TEXT, keypoints BLOB)")
    db.execute("INSERT INTO poses (camera, time, posture) VALUES ('a', 1.0, 'standing')")
    db.commit()
    db.close()
    s = EventStore(path)
    s.add_poses([{"camera": "a", "time": 2.0, "posture": "bending", "motion": "running"}])
    rows = s.pose_rows()
    s.close()
    assert [(r["posture"], r["motion"]) for r in rows] == [("standing", None),
                                                          ("bending", "running")]


def test_indexer_samples_still_people_less(store):
    fake = FakePose()
    indexer = PoseIndexer(store, "lobby", fake, interval_seconds=1, still_factor=3,
                          blocking=True)
    frame = np.zeros((300, 300, 3), np.uint8)
    for i in range(12):
        indexer.update(frame, [track([10, 10, 110, 210])], 1000.0 + i)  # not moving
    indexer.close()
    assert fake.calls == 4  # every 3 s instead of every second
    small = PoseIndexer(store, "lobby", fake, min_box_height=80, blocking=True)
    small.update(frame, [track([10, 10, 40, 60])], 2000.0)  # 50 px tall: skipped
    small.close()
    assert fake.calls == 4


def test_person_visits_and_matching():
    t = 1000.0
    rows = [
        {"camera": "lobby", "track_id": 1, "time": t, "posture": "standing", "arms": "down",
         "facing": "camera", "upper_colour": "red", "lower_colour": "blue"},
        {"camera": "lobby", "track_id": 1, "time": t + 4, "posture": "lying", "arms": "unknown",
         "facing": "unknown", "upper_colour": "red", "lower_colour": None},
        {"camera": "lobby", "track_id": 1, "time": t + 14, "posture": "lying", "arms": "unknown",
         "facing": "unknown", "upper_colour": "red", "lower_colour": None},
        {"camera": "lobby", "track_id": 2, "time": t + 1, "posture": "standing",
         "arms": "both_up", "facing": "away", "upper_colour": "black", "lower_colour": "black"},
        {"camera": "lobby", "track_id": 1, "time": t + 3600, "posture": "sitting",
         "arms": "down", "facing": "side", "upper_colour": "white", "lower_colour": "blue"},
    ]
    visits = person_visits(rows, heartbeat=10)
    assert [(v.track_id, round(v.first_seen - t)) for v in visits] == [(1, 0), (2, 1), (1, 3600)]
    first = visits[0]
    assert first.seconds["posture"] == {"standing": 4, "lying": 11}  # 10 s + last row 1 s
    assert first.main("upper_colour") == "red" and first.main("lower_colour") == "blue"
    assert match_people(visits, postures=["lying"]) == [first]
    assert [v.track_id for v in match_people(visits, arms="up")] == [2]
    assert [v.track_id for v in match_people(visits, upper_colour="red",
                                             lower_colour="blue")] == [1]
    assert match_people(visits, facing="camera") == [first]
    assert match_people(visits, head="down") == []  # rows without the newer facts still work
    d = first.to_dict()
    assert d["posture_seconds"] == {"lying": 11, "standing": 4} and d["upper_colour"] == "red"


def test_store_purges_poses(store):
    old = datetime(2026, 1, 1).timestamp()
    store.add_poses([{"camera": "a", "time": old, "track_id": 1, "posture": "standing",
                      "box": [0, 0, 10, 20], "keypoints": b"x" * 102}])
    assert store.counts()["poses"] == 1 and store.pose_keypoints(1) is not None
    assert store.purge(now=old + 31 * 86400)["poses"] == 1


def test_api_people(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from argus.engine.server import create_app

    app = create_app({"store": {"path": str(tmp_path / "db.sqlite")}, "cameras": {"lobby": {}}})
    s: EventStore = app.state.store
    t = datetime(2026, 10, 7, 22).timestamp()
    s.add_poses([{"camera": "lobby", "time": t + i, "track_id": 3, "posture": "lying",
                  "upper_colour": "green"} for i in range(0, 30, 10)])
    client = TestClient(app)
    r = client.get("/people", params={"posture": "lying", "when": "2026-10-07"}).json()
    assert r["total"] == 1 and r["results"][0]["upper_colour"] == "green"
    assert client.get("/people", params={"upper": "red"}).json()["total"] == 0
    assert client.get("/people", params={"posture": "flying"}).status_code == 400
    assert client.get("/people", params={"carrying": "pram"}).status_code == 400
    assert client.get("/people", params={"motion": "running"}).json()["total"] == 0
    engine: SearchEngine = app.state.engine
    assert engine.people(postures=["standing"]) == []


@pytest.mark.skipif(not (ROOT / "models" / "pose_rtmpose_t.onnx").exists(),
                    reason="uv run -m argus.engine.download_model pose_rtmpose_t")
def test_real_model_on_sample_image():
    from argus.engine.pose import PoseEstimator

    image = cv2.imread(str(ROOT / "data" / "basketball2.png"))
    estimator = PoseEstimator(ROOT / "models" / "pose_rtmpose_t.onnx")
    left, right = estimator.estimate(image, [np.array([30, 83, 169, 479]),
                                             np.array([431, 17, 637, 440])])
    assert describe(left, image).posture == "standing"
    assert describe(left, image).facing == "camera" and describe(right, image).facing == "away"
    assert left[L_EYE, 0] > left[R_EYE, 0] and left[L_WRI, 2] > 0.3 and left[R_WRI, 2] > 0.3
