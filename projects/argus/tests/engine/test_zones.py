import numpy as np

from argus.engine.alerts import AlertManager
from argus.engine.detector import Detection
from argus.engine.tracker import Tracker, iou_matrix
from argus.engine.zones import Line, Zone, ZoneMonitor, resolve_points

FRAME = (200, 100)  # w, h


def det(x: float, y: float, label: str = "car", score: float = 0.9, size: float = 30) -> Detection:
    """Box whose bottom-center (the default anchor) is at (x, y)."""
    return Detection(x - size / 2, y - size, x + size / 2, y, score, 2, label)


def run(monitor: ZoneMonitor, frames: list[list[Detection]], fps: float = 10.0, tracker=None):
    tracker = tracker or Tracker(min_hits=1)
    events = []
    for i, detections in enumerate(frames):
        tracks, removed = tracker.update(detections)
        events += monitor.update(tracks, removed, i / fps, FRAME)
    return events


def test_iou_matrix():
    a = np.array([[0, 0, 10, 10]], dtype=float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], dtype=float)
    assert np.allclose(iou_matrix(a, b), [[1.0, 1 / 3, 0.0]])


def test_tracker_keeps_id_while_moving_and_through_short_occlusion():
    tracker = Tracker(min_hits=2, max_missed=3)
    ids = []
    for i in range(10):
        dets = [] if i in (5, 6) else [det(20 + 3 * i, 50)]
        tracks, _ = tracker.update(dets)
        ids.append(tracks[0].id if tracks else None)
    assert ids[0] is None  # not confirmed yet
    assert set(ids[1:]) == {1}  # same ID all along, including coasting frames 5-6


def test_tracker_separate_objects_and_removal():
    tracker = Tracker(min_hits=1, max_missed=2)
    tracks, _ = tracker.update([det(20, 50), det(150, 50)])
    assert sorted(t.id for t in tracks) == [1, 2]
    removed = []
    for _ in range(4):
        _, gone = tracker.update([det(20, 50)])
        removed += gone
    assert [t.id for t in removed] == [2]


def test_label_vote_is_stable_under_flicker():
    tracker = Tracker(min_hits=1)
    for label in ["car", "truck", "car", "car", "truck"]:
        tracks, _ = tracker.update([det(50, 50, label=label)])
    assert tracks[0].label == "car"


def test_zone_enter_dwell_exit():
    zone = Zone("spot", [[50, 20], [100, 20], [100, 80], [50, 80]], dwell_seconds=1.0,
                confirm_frames=1)
    walk_in = [[det(x, 50)] for x in (30, 40, 50, 60, 70)]  # anchor enters at x=50 (t=0.2)
    park = [[det(70, 50)]] * 15
    walk_out = [[det(x, 50)] for x in (80, 90, 100, 110, 120)]  # leaves after x=100 (t=2.3)
    events = run(ZoneMonitor([zone]), walk_in + park + walk_out)
    assert [e.type for e in events] == ["enter", "dwell", "exit"]
    assert events[0].timestamp == 0.2
    assert events[1].dwell_seconds == 1.0
    assert events[2].timestamp == 2.3 and events[2].dwell_seconds == 2.1


def test_zone_hysteresis_ignores_single_frame_jitter():
    zone = Zone("spot", [[50, 20], [100, 20], [100, 80], [50, 80]], confirm_frames=2)
    frames = [[det(45, 50)], [det(55, 50)], [det(45, 50)], [det(55, 50)], [det(55, 50)]]
    events = run(ZoneMonitor([zone]), frames)
    assert [e.type for e in events] == ["enter"]


def test_zone_class_filter():
    zone = Zone("spot", [[50, 20], [100, 20], [100, 80], [50, 80]], classes=["person"],
                confirm_frames=1)
    assert run(ZoneMonitor([zone]), [[det(70, 50)]] * 3) == []


def test_zone_exit_when_track_lost_inside():
    zone = Zone("spot", [[50, 20], [100, 20], [100, 80], [50, 80]], confirm_frames=1)
    tracker = Tracker(min_hits=1, max_missed=1)
    events = run(ZoneMonitor([zone]), [[det(70, 50)]] * 3 + [[]] * 3, tracker=tracker)
    assert [e.type for e in events] == ["enter", "exit"]


def test_line_crossing_directions():
    # Vertical line at x=100 drawn top -> bottom: its arrow points left, so right->left is "in".
    line = Line("gate", [[100, 10], [100, 90]])
    right_to_left = [[det(130 - 10 * i, 50)] for i in range(6)]
    events = run(ZoneMonitor([line]), right_to_left)
    assert [(e.type, e.direction) for e in events] == [("cross", "in")]
    assert line.counts == {"in": 1, "out": 0}


def test_line_ignores_passing_beside_the_segment():
    line = Line("gate", [[100, 10], [100, 40]])  # short segment; object passes below it
    frames = [[det(130 - 10 * i, 80)] for i in range(6)]
    assert run(ZoneMonitor([line]), frames) == []


def test_line_direction_filter():
    line = Line("gate", [[100, 10], [100, 90]], direction="out")
    frames = [[det(130 - 10 * i, 50)] for i in range(6)]
    assert run(ZoneMonitor([line]), frames) == []
    assert line.counts["in"] == 1  # still counted, just not alerted


def test_normalized_points_and_config():
    assert np.allclose(resolve_points([[0.5, 0.5], [1, 1]], (200, 100)), [[100, 50], [200, 100]])
    monitor = ZoneMonitor.from_config([
        {"name": "a", "type": "zone", "points": [[0, 0], [0.5, 0], [0.5, 1]]},
        {"name": "b", "type": "line", "points": [[0.5, 0], [0.5, 1]], "direction": "in"},
    ])
    assert [r.kind for r in monitor.rules] == ["zone", "line"]


def test_alert_filter_and_cooldown(tmp_path):
    line = Line("gate", [[100, 10], [100, 90]])
    back_and_forth = [[det(x, 50)] for x in [125, 115, 105, 95, 85, 95, 105, 115, 105, 95]]
    events = run(ZoneMonitor([line]), back_and_forth)
    assert [e.timestamp for e in events] == [0.3, 0.6, 0.9]
    out = tmp_path / "events.jsonl"
    alerts = AlertManager(console=False, jsonl=out, events=["cross"], cooldown_seconds=0.5)
    emitted = alerts.handle(events)
    assert [e.timestamp for e in emitted] == [0.3, 0.9]  # 0.6 is within the cooldown
    assert len(out.read_text().splitlines()) == 2
