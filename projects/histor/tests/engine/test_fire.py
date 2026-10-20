"""Fire and smoke: the YOLOv9 decoding, the persistence rule, the pipeline and the safety rules
around the vision model's second opinion."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from histor.engine.alerts import AlertManager
from histor.engine.detector import Detection, Yolov9OnnxDetector
from histor.engine.fire import FireWatch, fire_enabled
from histor.engine.pipeline import Pipeline
from histor.engine.store import EventStore
from histor.engine.verify import AlertVerifier

ROOT = Path(__file__).resolve().parents[2]
SMALL = ROOT / "models" / "fire_smoke_yolov9s_640.onnx"


@pytest.fixture
def store(tmp_path):
    s = EventStore(tmp_path / "db.sqlite")
    yield s
    s.close()


def fire_box(label: str = "fire", score: float = 0.8) -> Detection:
    return Detection(100, 100, 200, 220, score, 0 if label == "fire" else 1, label)


class Scripted:
    """A detector that returns script(call number)."""

    def __init__(self, script):
        self.script, self.calls = script, 0

    def detect(self, frame):
        out = self.script(self.calls)
        self.calls += 1
        return out


# ------------------------------------------------------------------ decoding

def make_postprocessor(**kwargs) -> Yolov9OnnxDetector:
    det = Yolov9OnnxDetector.__new__(Yolov9OnnxDetector)
    det.input_size = (640, 640)
    det.class_names = ("fire", "smoke")
    det.conf_threshold = 0.25
    det.nms_threshold = 0.45
    det.allowed_ids = None
    det.__dict__.update(kwargs)
    return det


def test_postprocess_scales_boxes_and_keeps_both_classes_of_one_anchor():
    raw = np.zeros((6, 4), dtype=np.float32)  # x1 y1 x2 y2 fire smoke, 4 anchors
    raw[:, 0] = [10, 20, 110, 220, 0.9, 0.6]   # fire and smoke on the same anchor
    raw[:, 1] = [12, 22, 112, 222, 0.8, 0.1]   # a duplicate fire box: suppressed
    raw[:, 2] = [300, 300, 400, 400, 0.1, 0.2]  # below the threshold
    raw[:, 3] = [600, 10, 700, 50, 0.5, 0.0]    # crosses the image edge: clipped
    dets = make_postprocessor().postprocess(raw, ratio=0.5, image_shape=(400, 1300))
    assert [(d.label, round(d.score, 2)) for d in dets] == [("fire", 0.9), ("smoke", 0.6),
                                                           ("fire", 0.5)]
    assert (dets[0].x1, dets[0].y1, dets[0].x2, dets[0].y2) == (20, 40, 220, 400)  # clipped to the 400-pixel height
    assert dets[2].x2 == 1300 and dets[2].class_id == 0


def test_classes_filter():
    raw = np.zeros((6, 1), dtype=np.float32)
    raw[:, 0] = [10, 20, 110, 220, 0.9, 0.6]
    dets = make_postprocessor(allowed_ids={1}).postprocess(raw, 1.0, (640, 640))
    assert [d.label for d in dets] == ["smoke"]


@pytest.mark.skipif(not SMALL.exists(), reason="fire model not installed (download_model fire_smoke)")
def test_real_model_loads_with_names_from_metadata():
    # 2 threads and freed at once: an idle session's thread pool slows the timing-sensitive tests
    det = Yolov9OnnxDetector(SMALL, device="cpu", num_threads=2)
    try:
        assert det.class_names == ("fire", "smoke") and det.input_size == (640, 640)
        assert det.detect(np.full((480, 640, 3), 128, np.uint8)) == []  # a grey wall: no fire
    finally:
        del det.session


# ------------------------------------------------------------------ persistence

def run(watch: FireWatch, seen, start: float = 0.0, step: float = 0.5):
    frame = np.zeros((10, 10, 3), np.uint8)
    events = []
    for i, _ in enumerate(seen):
        events += watch.update(frame, start + i * step)
    return events


def test_fire_seen_on_3_of_5_checks_raises_one_event():
    seen = [1, 0, 1, 1, 1, 1, 1, 1]
    watch = FireWatch(Scripted(lambda i: [fire_box()] if seen[i] else []))
    events = run(watch, seen)
    assert [(e.kind, e.time) for e in events] == [("fire", 1.5)] and events[0].score == 0.8


def test_flicker_below_the_rule_and_low_scores_raise_nothing():
    seen = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1]  # never 3 in 5
    assert run(FireWatch(Scripted(lambda i: [fire_box()] if seen[i] else [])), seen) == []
    weak = FireWatch(Scripted(lambda i: [fire_box(score=0.2)]), min_score=0.25)
    assert run(weak, range(10)) == []


def test_smoke_and_fire_are_separate_and_a_new_episode_alerts_again():
    def script(i):
        if i < 6:
            return [fire_box("smoke", 0.4)]
        if i < 10:
            return []
        return [fire_box("smoke", 0.7)]
    watch = FireWatch(Scripted(script), clear_seconds=1.5)
    events = run(watch, range(16))
    assert [(e.kind, e.time) for e in events] == [("smoke", 1.0), ("smoke", 6.0)]


def test_checks_only_every_interval():
    calls = Scripted(lambda i: [])
    watch = FireWatch(calls, interval_seconds=1.0)
    run(watch, range(10), step=0.25)  # 2.5 s of frames
    assert calls.calls == 3


class Clock:
    """A detector that sees fire when burning(now) says so; counts its checks."""

    def __init__(self, burning):
        self.burning, self.now, self.times = burning, 0.0, []

    def detect(self, frame):
        self.times.append(self.now)
        return [fire_box()] if self.burning(self.now) else []


def live(watch: FireWatch, clock: Clock, seconds: float, fps: float = 25.0):
    """Frames at `fps` for `seconds`, like a camera: the events with their times."""
    frame = np.zeros((10, 10, 3), np.uint8)
    events = []
    for i in range(int(seconds * fps)):
        clock.now = round(i / fps, 3)
        events += watch.update(frame, clock.now)
    return events


def test_one_sighting_and_the_next_frames_are_checked_at_once():
    clock = Clock(lambda t: t >= 1.0)  # flames from 1.0 s
    events = live(FireWatch(clock), clock, 3.0)
    first = min(t for t in clock.times if t >= 1.0)  # the calm check that first saw them
    assert [e.kind for e in events] == ["fire"]
    assert 0.4 <= events[0].time - first <= 0.5  # 0.4 s after the first sighting, not 1-2.5 s
    gaps = np.diff([t for t in clock.times if first <= t <= events[0].time])
    assert gaps.max() <= 0.13  # every ~0.1 s (the next frame after it) while suspected
    assert np.diff([t for t in clock.times if t < first]).min() >= 0.5  # calm: every 0.5 s


def test_a_glint_checked_closely_raises_nothing_and_checks_calm_down():
    clock = Clock(lambda t: 2.0 <= t < 2.3)  # a headlight's reflection, 0.3 s
    watch = FireWatch(clock)
    assert live(watch, clock, 8.0) == []
    assert np.diff([t for t in clock.times if t > 5.0]).min() >= 0.5  # calm again
    assert not watch.suspected


def test_a_stray_sighting_does_not_slow_a_fire_that_follows():
    clock = Clock(lambda t: 0.5 <= t < 0.56 or t >= 1.5)  # one stray box, then a fire
    events = live(FireWatch(clock), clock, 4.0)
    first = min(t for t in clock.times if t >= 1.5)
    assert len(events) == 1 and events[0].time - first <= 0.5  # the stray misses don't count


def test_config():
    with pytest.raises(ValueError):
        FireWatch(Scripted(lambda i: []), share=1.5)
    with pytest.raises(ValueError):
        FireWatch(Scripted(lambda i: []), min_seconds=3, window_seconds=2.5)
    assert FireWatch(Scripted(lambda i: []), window=5).window == 2.5  # old configs still load
    with pytest.raises(ValueError):
        FireWatch(Scripted(lambda i: []), classes=["flames"])
    assert fire_enabled({"enabled": True}, "gate")
    assert not fire_enabled({"enabled": True, "cameras": ["yard"]}, "gate")
    assert not fire_enabled(None, "gate")


# ------------------------------------------------------------------ pipeline

def test_pipeline_fire_reaches_the_store_with_a_boxed_snapshot(store):
    alerts = AlertManager(console=False, store=store, source="kitchen")
    watch = FireWatch(Scripted(lambda i: [fire_box()] if i >= 2 else []))
    pipe = Pipeline(Scripted(lambda i: []), alerts=alerts, store=store, start_time=0.0, fire=watch)
    frame = np.full((360, 640, 3), 90, np.uint8)
    for i in range(20):
        pipe.process(frame, i * 0.25)
    shown = pipe.draw(frame.copy(), pipe.process(frame, 5.0))
    pipe.close()
    rows = store._db.execute("SELECT type, rule, label, track_id, snapshot FROM events").fetchall()
    assert [tuple(r[:4]) for r in rows] == [("fire", "fire", "fire", None)]
    snapshot = cv2.imread(str(store.media_path(rows[0][4])))
    edge = snapshot[110:210, 100].astype(int)  # the box's left edge, drawn in red (BGR)
    assert ((edge[:, 2] > 180) & (edge[:, 1] < 90) & (edge[:, 0] < 90)).mean() > 0.8
    assert (shown[100:220, 100] != 90).any()  # the live overlay shows the box too


# ------------------------------------------------------------------ second opinion

def test_vision_model_never_checks_fire_unless_asked_and_never_holds_it_back():
    verifier = AlertVerifier(lambda images, q, s: "NO\nlooks like a lamp")
    try:
        assert not verifier.applies({"type": "fire", "rule": "fire"})
        assert verifier.applies({"type": "down", "rule": "person_down"})
    finally:
        verifier.close()
    opted = AlertVerifier(lambda images, q, s: "NO", events=["fire"], mode="veto")
    try:
        assert opted.applies({"type": "fire", "rule": "smoke"})
        assert "smoke" in opted.question({"type": "fire", "rule": "smoke", "label": "smoke"})
        assert opted.sends("no", "fire") and not opted.sends("no", "down")
    finally:
        opted.close()
