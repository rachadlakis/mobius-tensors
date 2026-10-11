"""Rules that need no model: camera tampering, abandoned objects, crowding in an area."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from histor.engine.abandoned import AbandonedWatch, abandoned_enabled
from histor.engine.alerts import AlertManager, alert_message
from histor.engine.detector import Detection
from histor.engine.pipeline import Pipeline
from histor.engine.search import SearchEngine
from histor.engine.store import EventStore
from histor.engine.tamper import TamperCheck, measure, similarity, tamper_enabled
from histor.engine.tracker import Track, Tracker
from histor.engine.zones import Zone, ZoneMonitor
from histor.tools import Toolbox

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def store(tmp_path):
    s = EventStore(tmp_path / "db.sqlite")
    yield s
    s.close()


def scene(seed: int = 0, shape=(360, 640)) -> np.ndarray:
    """A textured view: rectangles and lines, like a street with buildings."""
    rng = np.random.default_rng(seed)
    image = np.full((*shape, 3), 110, np.uint8)
    for _ in range(60):
        x, y = int(rng.integers(0, shape[1] - 60)), int(rng.integers(0, shape[0] - 40))
        w, h = int(rng.integers(15, 80)), int(rng.integers(10, 60))
        cv2.rectangle(image, (x, y), (x + w, y + h), tuple(int(c) for c in rng.integers(0, 255, 3)), -1)
    for _ in range(30):
        p = tuple(int(v) for v in rng.integers(0, 640, 2))
        q = tuple(int(v) for v in rng.integers(0, 360, 2))
        cv2.line(image, (p[0], q[0]), (p[1], q[1]), (240, 240, 240), 2)
    return image


def noisy(image: np.ndarray, seed: int) -> np.ndarray:
    noise = np.random.default_rng(seed).normal(0, 3, image.shape)
    return np.clip(image + noise, 0, 255).astype(np.uint8)


def feed(check: TamperCheck, frames, start: float, seconds: float, step: float = 1.0):
    events = []
    t = start
    i = 0
    while t < start + seconds:
        frame = frames(i) if callable(frames) else frames
        events += check.update(frame, t)
        t += step
        i += 1
    return events, t


def learned(**options) -> tuple[TamperCheck, np.ndarray, float]:
    """A check that has learned its normal view (60 s of a noisy textured scene)."""
    check = TamperCheck(interval_seconds=1, hold_seconds=10, learn_seconds=30, **options)
    view = scene()
    events, t = feed(check, lambda i: noisy(view, i), 0, 60)
    assert events == [] and check.sharpness is not None and check.reference is not None
    return check, view, t


# ------------------------------------------------------------------ tamper

def test_normal_view_with_noise_and_people_raises_nothing():
    check, view, t = learned()

    def busy(i):
        frame = noisy(view, i)
        x = (i * 23) % 560
        cv2.rectangle(frame, (x, 150), (x + 50, 330), (30, 30, 200), -1)  # someone walking by
        return frame

    events, _ = feed(check, busy, t, 600)
    assert events == []


def test_covered_lens_is_blocked_once_until_it_clears():
    check, view, t = learned()
    covered = np.full_like(view, 40)
    events, t = feed(check, lambda i: noisy(covered, i)[..., :1].repeat(3, 2) // 1, t, 9)
    assert events == []  # not for long enough
    events, t = feed(check, lambda i: covered, t, 30)
    assert [e.kind for e in events] == ["blocked"] and events[0].value < 6
    events, t = feed(check, lambda i: noisy(view, i), t, 20)  # uncovered
    assert events == []
    events, t = feed(check, lambda i: covered, t, 15)
    assert [e.kind for e in events] == ["blocked"]  # re-armed after it cleared


def test_dark_with_a_few_lights():
    check, view, t = learned()
    night = np.zeros_like(view)
    for x in range(40, 640, 120):
        cv2.circle(night, (x, 60), 12, (255, 255, 255), -1)  # street lights
    m = measure(night)
    assert m.mean < 12 and m.std >= 6
    events, _ = feed(check, night, t, 15)
    assert [e.kind for e in events] == ["dark"]


def test_defocused_is_blurred():
    check, view, t = learned()
    events, _ = feed(check, lambda i: cv2.GaussianBlur(noisy(view, i), (31, 31), 12), t, 15)
    assert [e.kind for e in events] == ["blurred"] and events[0].value < 0.25


def test_turned_camera_is_moved_once_then_the_new_view_is_normal():
    check, view, t = learned()
    other = scene(seed=7)
    events, t = feed(check, lambda i: noisy(other, i), t, 40)
    assert [e.kind for e in events] == ["moved"] and events[0].value < 0.5
    events, _ = feed(check, lambda i: noisy(other, i), t, 60)
    assert events == []  # the new view became the reference


def test_frozen_stream():
    check, view, t = learned()
    still = noisy(view, 999)
    events, _ = feed(check, still, t, 15)
    assert [e.kind for e in events] == ["frozen"]


def test_no_blur_verdict_while_learning_and_checks_subset():
    check = TamperCheck(interval_seconds=1, hold_seconds=5, learn_seconds=1000,
                        checks=["blurred", "blocked"])
    view = scene()
    events, t = feed(check, lambda i: noisy(view, i), 0, 20)
    events2, _ = feed(check, lambda i: cv2.GaussianBlur(view, (31, 31), 12), t, 20)
    assert events == events2 == [] and check.sharpness is None
    events, _ = feed(check, view, t + 20, 20)  # frozen, but not checked
    assert events == []


def test_interval_limits_work():
    check = TamperCheck(interval_seconds=2)
    view = scene()
    for i in range(30):
        check.update(view, i * 0.1)
    assert check._last == pytest.approx(2.0)


def test_tamper_config_errors_and_enabled():
    with pytest.raises(ValueError, match="unknown"):
        TamperCheck(checks=["smoke"])
    assert tamper_enabled({"enabled": True}, "gate")
    assert not tamper_enabled({"enabled": True, "cameras": ["yard"]}, "gate")
    assert not tamper_enabled(None, "gate")


def test_similarity_of_views():
    view = scene()
    same = similarity(measure(view).gradient, measure(noisy(view, 1)).gradient)
    shifted = similarity(measure(view).gradient, measure(np.roll(view, 60, 1)).gradient)
    other = similarity(measure(view).gradient, measure(scene(7)).gradient)
    darker = similarity(measure(view).gradient, measure((view * 0.6).astype(np.uint8)).gradient)
    assert same > 0.95 and darker > 0.9 and shifted < 0.5 and other < 0.4
    flat = np.zeros((45, 80), np.float32)
    assert similarity(flat, flat) == 1.0  # nothing to compare


def test_pointed_at_a_wall_is_blocked_and_learns_no_reference():
    check = TamperCheck(interval_seconds=1, hold_seconds=3, learn_seconds=2)
    wall = np.full((360, 640, 3), 128, np.uint8)
    wall[::40] = 140  # faint stripes
    events, t = feed(check, lambda i: noisy(wall, i), 0, 20)
    assert [e.kind for e in events] == ["blocked"] and check.reference is None
    events, _ = feed(check, lambda i: noisy(scene(), i), t, 20)
    assert events == [] and check.reference is not None  # a real view: learned, not "moved"


VIDEO = ROOT / "data" / "vtest.avi"


@pytest.mark.skipif(not VIDEO.exists(), reason="sample video data/vtest.avi")
def test_real_street_video_no_false_alarm_then_covered():
    cap = cv2.VideoCapture(str(VIDEO))
    frames = []
    while len(frames) < 400:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    assert len(frames) >= 300
    check = TamperCheck(interval_seconds=0.5, hold_seconds=10, learn_seconds=20)
    events = []
    for i, frame in enumerate(frames):  # ~10 fps: ~40 s of pedestrians
        events += check.update(frame, i * 0.1)
    assert events == []
    t = len(frames) * 0.1
    hand = np.full_like(frames[-1], 25)
    for i in range(150):
        events += check.update(hand, t + i * 0.1)
    assert [e.kind for e in events] == ["blocked"]


# ------------------------------------------------------------------ abandoned

def tr(tid: int, box, label: str, missed: int = 0) -> Track:
    t = Track(id=tid, box=np.array(box, float), score=0.9, label_votes={label: 1.0})
    t.missed = missed
    return t


BAG = [300, 300, 340, 340]


def run_watch(watch: AbandonedWatch, seconds: float, people, bag=BAG, start: float = 0.0,
              step: float = 1.0, bag_id: int = 9):
    events = []
    for i in range(int(seconds / step)):
        t = start + i * step
        tracks = [tr(bag_id, bag(t) if callable(bag) else bag, "suitcase")]
        tracks += [tr(100 + k, p(t), "person") for k, p in enumerate(people)]
        events += watch.update(tracks, [], t)
    return events


def walker(t: float) -> list[float]:
    x = 280 + 40 * t  # walks away from the bag at 40 px/s
    return [x, 200, x + 50, 350]


def test_bag_left_behind_fires_once():
    watch = AbandonedWatch(seconds=30)
    events = run_watch(watch, 120, [walker])
    (event,) = events
    assert event.track_id == 9 and event.label == "suitcase"
    assert 30 <= event.seconds <= 32 and event.time < 40


def test_bag_with_its_owner_nearby_is_fine():
    watch = AbandonedWatch(seconds=30)
    sitting = lambda t: [345, 210, 395, 360]  # noqa: E731
    assert run_watch(watch, 200, [sitting]) == []


def test_carried_bag_is_never_still():
    watch = AbandonedWatch(seconds=10)
    moving = lambda t: [300 + 20 * t, 300, 340 + 20 * t, 340]  # noqa: E731
    far = lambda t: [0, 0, 20, 40]  # noqa: E731
    assert run_watch(watch, 60, [far], bag=moving) == []


def test_small_jitter_still_counts_as_still():
    watch = AbandonedWatch(seconds=20, still_ratio=0.25)
    jitter = lambda t: [300 + (t % 2) * 4, 300, 340 + (t % 2) * 4, 340]  # 4 px on a 40 px bag  # noqa: E731
    assert len(run_watch(watch, 40, [], bag=jitter)) == 1


def test_passer_by_resets_the_clock():
    watch = AbandonedWatch(seconds=20)

    def passer(t):
        x = -500 if not 15 <= t < 17 else 300  # walks past the bag at 15-17 s
        return [x, 200, x + 50, 350]

    (event,) = run_watch(watch, 60, [passer])
    assert event.time >= 37  # 17 s + 20 s


def test_missed_frames_keep_state_and_removed_tracks_forget():
    watch = AbandonedWatch(seconds=10)
    for t in range(8):
        watch.update([tr(9, BAG, "backpack")], [], t)
    watch.update([tr(9, BAG, "backpack", missed=1)], [], 8)  # not detected for a frame
    events = watch.update([tr(9, BAG, "backpack")], [], 10)
    assert len(events) == 1
    watch.update([], [tr(9, BAG, "backpack")], 11)
    assert 9 not in watch._state


def test_only_inside_named_zones():
    zone = Zone("platform", [[0, 0], [200, 0], [200, 200], [0, 200]])
    zone.resolve((640, 360))
    watch = AbandonedWatch(seconds=5, zones=[zone])
    assert run_watch(watch, 20, []) == []  # BAG is outside the platform
    inside = [50, 50, 90, 90]
    assert len(run_watch(watch, 20, [], bag=inside, bag_id=3)) == 1


def test_abandoned_from_config():
    zone = Zone("platform", [[0, 0], [1, 0], [1, 1]])
    watch = AbandonedWatch.from_config({"enabled": True, "seconds": 45, "zones": ["platform"],
                                        "classes": ["suitcase"]}, [zone])
    assert watch.seconds == 45 and watch.zones == [zone] and watch.classes == {"suitcase"}
    with pytest.raises(ValueError, match="not areas"):
        AbandonedWatch.from_config({"zones": ["lobby"]}, [zone])
    assert abandoned_enabled({"enabled": True}, "x") and not abandoned_enabled({}, "x")


# ------------------------------------------------------------------ crowd

def people(n: int, start: int = 0) -> list[Track]:
    return [tr(start + i, [100 + 30 * i, 100, 120 + 30 * i, 200], "person") for i in range(n)]


def test_crowd_fires_after_count_seconds_and_rearms():
    zone = Zone("queue", classes=["person"], max_count=3, count_seconds=5, confirm_frames=1)
    zone.resolve((640, 360))
    events = []
    for t in range(10):
        events += zone.update(people(3), [], t)
    crowd = [e for e in events if e.type == "crowd"]
    assert len(crowd) == 1
    assert crowd[0].count == 3 and crowd[0].label == "person" and crowd[0].timestamp == 5
    assert crowd[0].track_id is None and crowd[0].dwell_seconds == 5
    # two leave: below the limit; then three again -> a new crowd
    events = zone.update(people(1), people(3)[1:], 10)
    assert not [e for e in events if e.type == "crowd"]
    events = []
    for t in range(11, 20):
        events += zone.update(people(3, start=10) + people(1), [], t)
    assert [e.count for e in events if e.type == "crowd"] == [4]


def test_crowd_counts_only_its_classes_and_inside():
    zone = Zone("queue", [[0, 0], [0.5, 0], [0.5, 1], [0, 1]], classes=["person"],
                max_count=2, count_seconds=0, confirm_frames=1)
    zone.resolve((640, 360))
    outside = [tr(50, [500, 100, 520, 200], "person")]
    cars = [tr(60, [10, 10, 60, 60], "car"), tr(61, [70, 10, 120, 60], "car")]
    events = zone.update(people(1) + outside + cars, [], 0)
    assert not [e for e in events if e.type == "crowd"]


def test_crowd_config():
    monitor = ZoneMonitor.from_config([{"name": "q", "max_count": 4, "count_seconds": 3}])
    assert monitor.rules[0].max_count == 4
    with pytest.raises(ValueError, match="max_count"):
        Zone("q", max_count=0)


# ------------------------------------------------------------------ pipeline, store, chat

class Scripted:
    def __init__(self, script) -> None:
        self.script = script
        self.i = 0

    def detect(self, frame):
        out = self.script(self.i)
        self.i += 1
        return out


def test_pipeline_tamper_without_any_tracker(store):
    alerts = AlertManager(console=False, store=store, source="gate")
    pipe = Pipeline(Scripted(lambda i: []), alerts=alerts, store=store, start_time=0.0,
                    tamper=TamperCheck(interval_seconds=1, hold_seconds=5, learn_seconds=5))
    assert pipe.tracker is None
    view, covered = scene(), np.full((360, 640, 3), 30, np.uint8)
    for i in range(20):
        pipe.process(noisy(view, i), float(i))
    for i in range(20, 30):
        pipe.process(covered, float(i))
    pipe.close()
    rows = store._db.execute("SELECT type, rule, label, track_id, snapshot FROM events").fetchall()
    assert [tuple(r[:4]) for r in rows] == [("tamper", "blocked", "camera", None)] and rows[0][4]


def test_pipeline_abandoned_and_crowd_reach_the_store(store):
    def script(i):
        person_x = 280 + 40 * i
        dets = [Detection(300, 300, 340, 340, 0.9, 28, "suitcase"),
                Detection(person_x, 200, person_x + 50, 350, 0.9, 0, "person")]
        if i >= 5:
            dets += [Detection(20 + 40 * k, 20, 50 + 40 * k, 120, 0.9, 0, "person")
                     for k in range(3)]
        return dets

    alerts = AlertManager(console=False, store=store, source="hall")
    monitor = ZoneMonitor([Zone("lobby", [[0, 0], [0.3, 0], [0.3, 0.5], [0, 0.5]],
                                classes=["person"], max_count=3, count_seconds=3)])
    pipe = Pipeline(Scripted(script), Tracker(), monitor, alerts, store, start_time=0.0,
                    abandoned=AbandonedWatch(seconds=10))
    frame = np.zeros((360, 640, 3), np.uint8)
    for i in range(20):
        pipe.process(frame, float(i))
    pipe.close()
    rows = store._db.execute("SELECT type, rule, label, count FROM events WHERE type IN "
                             "('abandoned', 'crowd') ORDER BY time").fetchall()
    assert [tuple(r) for r in rows] == [("crowd", "lobby", "person", 3),
                                        ("abandoned", "abandoned_object", "suitcase", None)]
    box = Toolbox(SearchEngine(store))
    try:
        out = box.alerts(types=["crowd", "abandoned"])
        crowd = next(e for e in out["events"] if e["type"] == "crowd")
        assert crowd["count"] == 3
        assert box.alerts(types=["tamper"])["total"] == 0
    finally:
        box.close()


def test_alert_messages_for_crowd_and_tamper():
    _, crowd = alert_message({"type": "crowd", "rule": "queue", "label": "person", "count": 7,
                              "camera": "shop", "iso_time": "x"})
    assert "Event:  crowd (queue) 7 inside" in crowd
    title, tamper = alert_message({"type": "tamper", "rule": "blocked", "label": "camera",
                                   "camera": "gate", "iso_time": "x"})
    assert title == "tamper blocked - camera on gate" and "#None" not in tamper


def test_pipeline_from_config_builds_the_rules(tmp_path):
    model = ROOT / "models" / "yolox_nano.onnx"
    if not model.exists():
        pytest.skip("uv run -m histor.engine.download_model yolox_nano")
    config = {"model": {"path": str(model)}, "store": {"path": str(tmp_path / "r.db")},
              "zones": [{"name": "platform", "points": [[0, 0], [1, 0], [1, 1]]}],
              "abandoned": {"enabled": True, "zones": ["platform"]},
              "tamper": {"enabled": True, "cameras": ["gate"], "hold_seconds": 5}}
    pipe = Pipeline.from_config(config, source="gate", live=False)
    assert pipe.tamper is not None and pipe.abandoned is not None and pipe.tracker is not None
    assert pipe.alerts is not None and pipe.abandoned.zones[0].name == "platform"
    pipe.close()
    pipe.store.close()
    other = Pipeline.from_config({**config, "store": {"path": str(tmp_path / "o.db")}},
                                 source="yard", live=False)
    assert other.tamper is None  # not listed
    other.close()
    other.store.close()
    with pytest.raises(ValueError, match="not areas"):
        Pipeline.from_config({**config, "abandoned": {"enabled": True, "zones": ["lobby"]}},
                             source="gate", live=False)
