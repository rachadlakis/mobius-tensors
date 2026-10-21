"""Rules that need no model: camera tampering, abandoned objects, crowding in an area."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from histor.engine.abandoned import AbandonedWatch
from histor.engine.alerts import AlertManager, alert_message
from histor.engine.change import ChangeAnalyser
from histor.engine.detector import Detection
from histor.engine.pipeline import Pipeline
from histor.engine.search import SearchEngine
from histor.engine.tamper import TamperCheck, measure, similarity
from histor.engine.tracker import Track, Tracker
from histor.engine.zones import Zone, ZoneMonitor
from histor.tools import Toolbox
from kit import Scripted, feed, learned, noisy, scene_shapes, track

ROOT = Path(__file__).resolve().parents[2]


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
    other = scene_shapes(seed=7)
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
    view = scene_shapes()
    events, t = feed(check, lambda i: noisy(view, i), 0, 20)
    events2, _ = feed(check, lambda i: cv2.GaussianBlur(view, (31, 31), 12), t, 20)
    assert events == events2 == [] and check.sharpness is None
    events, _ = feed(check, view, t + 20, 20)  # frozen, but not checked
    assert events == []


def test_interval_limits_work():
    check = TamperCheck(interval_seconds=2)
    view = scene_shapes()
    for i in range(30):
        check.update(view, i * 0.1)
    assert check._last == pytest.approx(2.0)


def test_tamper_config_errors():
    with pytest.raises(ValueError, match="unknown"):
        TamperCheck(checks=["smoke"])


def test_similarity_of_views():
    view = scene_shapes()
    same = similarity(measure(view).gradient, measure(noisy(view, 1)).gradient)
    shifted = similarity(measure(view).gradient, measure(np.roll(view, 60, 1)).gradient)
    other = similarity(measure(view).gradient, measure(scene_shapes(7)).gradient)
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
    events, _ = feed(check, lambda i: noisy(scene_shapes(), i), t, 20)
    assert events == [] and check.reference is not None  # a real view: learned, not "moved"


# --- moved or obstructed: a truck parked in front is not a turned camera

def truck(view: np.ndarray, x1: int, x2: int, seed: int = 0) -> np.ndarray:
    """A truck's side from x1 to x2, full height: plain paint, a logo stripe, two windows."""
    frame = view.copy()
    frame[:, x1:x2] = (60, 70, 150)
    cv2.rectangle(frame, (x1 + 10, 120), (x2 - 10, 150), (240, 240, 240), -1)
    for wx in (x1 + 20, (x1 + x2) // 2):
        cv2.rectangle(frame, (wx, 40), (wx + 50, 90), (20, 20, 20), -1)
    return noisy(frame, seed)  # the sensor's noise is over the truck too


def shifted(view: np.ndarray, dx: int, dy: int = 0) -> np.ndarray:
    return cv2.warpAffine(view, np.float32([[1, 0, dx], [0, 1, dy]]), view.shape[1::-1],
                          borderMode=cv2.BORDER_REPLICATE)


def blocking(**options) -> tuple[TamperCheck, np.ndarray, float]:
    return learned(obstructed_seconds=30, **options)


def test_truck_parked_over_half_the_view_is_obstructed_not_moved():
    check, view, t = blocking()
    events, t = feed(check, lambda i: truck(view, 0, 380, i), t, 120)
    assert [e.kind for e in events] == ["obstructed"]
    assert events[0].value >= 0.5 and events[0].label == "camera"  # no detector: unknown cause
    events, _ = feed(check, lambda i: noisy(view, i), t, 60)  # it drives away
    assert events == [] and check.last_state == "home"


def test_obstruction_says_what_the_detector_saw():
    check, view, t = blocking()
    seen = [("truck", (0, 0, 380, 360)), ("person", (500, 100, 540, 300))]
    events = []
    for i in range(60):
        events += check.update(truck(view, 0, 380, i), t + i, seen)
    assert [(e.kind, e.label) for e in events] == [("obstructed", "truck")]


def test_near_full_cover_seen_by_the_detector_is_obstructed():
    check, view, t = blocking()
    events = []
    for i in range(60):  # 95% of the view, a truck box over it
        events += check.update(truck(view, 0, 610, i), t + i, [("truck", (0, 0, 610, 360))])
    assert [(e.kind, e.label) for e in events] == [("obstructed", "truck")]


def test_near_full_cover_that_grew_from_a_partial_one_is_obstructed():
    check, view, t = blocking()
    events, t = feed(check, lambda i: truck(view, 0, 380, i), t, 6)  # it pulls in...
    events2, _ = feed(check, lambda i: truck(view, 0, 610, i), t, 60)  # ...and stops
    assert events == [] and [e.kind for e in events2] == ["obstructed"]


def test_unexplained_full_cover_raises_once_but_its_leaving_does_not():
    check, view, t = blocking()
    events, t = feed(check, lambda i: truck(view, 0, 640, i), t, 40)  # sudden, no detection
    assert len(events) == 1 and events[0].kind in ("moved", "blurred")  # can't tell: say so
    events, _ = feed(check, lambda i: noisy(view, i), t, 60)  # the first view is known again
    assert events == [] and check.last_state == "home"


def test_wrong_moved_is_undone_when_the_first_view_comes_back():
    check, view, t = blocking(blur_ratio=0.0)  # only the position judged
    events, t = feed(check, lambda i: truck(view, 0, 640, i), t, 40)
    assert [e.kind for e in events] == ["moved"] and len(check.views) == 2
    events, _ = feed(check, lambda i: noisy(view, i), t, 60)
    assert events == [] and check.last_state == "home"


def test_small_pan_is_moved():
    check, view, t = learned()
    events, _ = feed(check, lambda i: noisy(shifted(view, 40, 10), i), t, 30)
    assert [e.kind for e in events] == ["moved"]


def test_wind_sway_is_not_moved():
    check, view, t = learned()
    events, _ = feed(check, lambda i: noisy(shifted(view, (-3, 3)[i % 2], (2, -2)[i % 2]), i),
                     t, 120)
    assert events == []


def test_crowd_passing_is_not_obstructed():
    check, view, t = blocking()

    def crowd(i):
        frame = noisy(view, i)
        rng = np.random.default_rng(i)
        for _ in range(14):  # people everywhere, somewhere else each sample
            x, y = int(rng.integers(0, 600)), int(rng.integers(0, 200))
            cv2.rectangle(frame, (x, y), (x + 60, y + 160),
                          tuple(int(c) for c in rng.integers(0, 255, 3)), -1)
        return frame

    events, _ = feed(check, crowd, t, 120)
    assert events == []


def test_on_screen_clock_does_not_hide_a_turn():
    def clock(frame: np.ndarray, i: int) -> np.ndarray:  # burnt in after the sensor's noise
        cv2.putText(frame, f"2026-10-20 14:{i // 60 % 60:02d}:{i % 60:02d}  GATE CAMERA 01",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        return frame

    check = TamperCheck(interval_seconds=1, hold_seconds=10, learn_seconds=30)
    view, other = scene_shapes(), scene_shapes(seed=7)
    events, t = feed(check, lambda i: clock(noisy(view, i), i), 0, 60)
    assert events == [] and check._overlay is not None and check._overlay[:30].any()
    events, _ = feed(check, lambda i: clock(noisy(other, i), i), t, 30)
    assert [e.kind for e in events] == ["moved"]


def test_night_look_of_the_same_view_is_not_moved():
    check, view, t = learned()
    grey = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
    infrared = cv2.cvtColor(cv2.convertScaleAbs(grey, alpha=0.6, beta=30), cv2.COLOR_GRAY2BGR)
    events, _ = feed(check, lambda i: noisy(infrared, i), t, 120)
    assert events == []


# On the real street (data/vtest.avi): covered, a van parking, the camera turned are scenario
# bench specs (tests/scenarios/specs/street: street_cover, street_van_parks, street_pan).


# ------------------------------------------------------------------ abandoned

BAG = [300, 300, 340, 340]


def run_watch(watch: AbandonedWatch, seconds: float, people, bag=BAG, start: float = 0.0,
              step: float = 1.0, bag_id: int = 9):
    events = []
    for i in range(int(seconds / step)):
        t = start + i * step
        tracks = [track(bag_id, bag(t) if callable(bag) else bag, "suitcase")]
        tracks += [track(100 + k, p(t), "person") for k, p in enumerate(people)]
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
        watch.update([track(9, BAG, "backpack")], [], t)
    watch.update([track(9, BAG, "backpack", missed=1)], [], 8)  # not detected for a frame
    events = watch.update([track(9, BAG, "backpack")], [], 10)
    assert len(events) == 1
    watch.update([], [track(9, BAG, "backpack")], 11)
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


# ------------------------------------------------------------------ crowd

def people(n: int, start: int = 0) -> list[Track]:
    return [track(start + i, [100 + 30 * i, 100, 120 + 30 * i, 200], "person") for i in range(n)]


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
    outside = [track(50, [500, 100, 520, 200], "person")]
    cars = [track(60, [10, 10, 60, 60], "car"), track(61, [70, 10, 120, 60], "car")]
    events = zone.update(people(1) + outside + cars, [], 0)
    assert not [e for e in events if e.type == "crowd"]


def test_crowd_config():
    monitor = ZoneMonitor.from_config([{"name": "q", "max_count": 4, "count_seconds": 3}])
    assert monitor.rules[0].max_count == 4
    with pytest.raises(ValueError, match="max_count"):
        Zone("q", max_count=0)


# ------------------------------------------------------------------ pipeline, store, chat

def test_tamper_with_shared_change_gives_identical_events():
    view = scene_shapes()
    frozen = noisy(view, 999)

    def frame(i: int) -> np.ndarray:
        if i < 60:
            return noisy(view, i)
        if i < 80:
            return noisy(np.roll(view, 80, 1), i)  # turned
        if i < 100:
            return np.full((360, 640, 3), 30, np.uint8)  # covered
        return frozen  # a stuck stream

    runs = []
    for shared in (False, True):
        check = TamperCheck(interval_seconds=1, hold_seconds=10, learn_seconds=30)
        analyser = ChangeAnalyser()
        events = []
        for i in range(130):
            change = analyser.update(frame(i), float(i)) if shared else None
            events += check.update(frame(i), float(i), change=change)
        runs.append([(e.time, e.kind, e.value, e.label) for e in events])
    assert runs[0] == runs[1] and {kind for _, kind, _, _ in runs[0]} >= {
        "moved", "blocked", "frozen"}


def test_pipeline_tamper_without_any_tracker(store):
    alerts = AlertManager(console=False, store=store, source="gate")
    pipe = Pipeline(Scripted(lambda i: []), alerts=alerts, store=store, start_time=0.0,
                    tamper=TamperCheck(interval_seconds=1, hold_seconds=5, learn_seconds=5))
    assert pipe.tracker is None
    view, covered = scene_shapes(), np.full((360, 640, 3), 30, np.uint8)
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
    assert "Event:  crowd (queue) 7 inside" in crowd and "Object: person" in crowd
    title, tamper = alert_message({"type": "tamper", "rule": "blocked", "label": "camera",
                                   "camera": "gate", "iso_time": "x"})
    assert title == "tamper blocked - camera on gate" and "#None" not in tamper
    assert "Heard" not in crowd + tamper  # only sounds are heard


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
