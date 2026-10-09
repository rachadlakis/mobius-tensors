"""Record body poses while recording: what people were doing, as searchable facts in the store.

Per camera, on its own background thread (pose is the slowest model per person, so it never
holds up detection or the CLIP index):

  * sampling   each tracked person about every `interval_seconds`, from the main-stream keyframe
               when the person was matched in one (more pixels per joint); someone standing
               still is sampled `still_factor` times less often
  * motion     still / walking / running, from how far the person's box moved since their last
               sample, in body heights per second (so it doesn't depend on the distance); moving
               straight towards or away from the camera counts through the box growing or
               shrinking, but far away that is slow and reads as still
  * carrying   backpack / handbag / suitcase / umbrella (pose.CARRIED) when the detector found
               one on the person: its box centre inside the person's (a bit widened). A bag set
               down right next to someone counts too, until they walk away from it
  * smoothing  a fact (posture, arms, facing, head, motion, carrying, colours) counts once it is
               the majority of the last 3 samples, so one bad frame doesn't make a standing
               person "lying"
  * storing    a row only when the smoothed facts change, plus a heartbeat every
               `heartbeat_seconds`: a person sitting at a desk for an hour is ~360 rows, not 3600,
               and durations are still exact to the heartbeat (see search.person_visits)
  * events     `down` when a person stays lying for `down_seconds`, `hands_up` when both hands
               stay above the head for `hands_up_seconds` (only the types listed in `alerts`).
               The pipeline takes them with take_events() and sends them through the alert
               channels like area and line events.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections import Counter, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from argus.engine.pose import CARRIED, PoseEstimator, describe, pack_keypoints
from argus.engine.tracker import iou_matrix

if TYPE_CHECKING:
    from argus.engine.indexer import HiResFrame
    from argus.engine.store import EventStore
    from argus.engine.tracker import Track

log = logging.getLogger(__name__)
FIELDS = ("posture", "arms", "facing", "head", "motion", "carrying", "upper_colour",
          "lower_colour")
WALKING, RUNNING = 0.15, 1.5  # body heights per second (walking ~0.8, jogging 2+)
EVENT_TYPES = ("down", "hands_up")
EVENT_RULES = {"down": "person_down", "hands_up": "hands_up"}  # rule name in the event log


@dataclass
class _Job:
    frame: np.ndarray
    time: float
    # track id, box, hi-res box, facts from the tracks (motion, carrying)
    people: list[tuple[int, np.ndarray, np.ndarray | None, dict[str, Any]]]
    hires: np.ndarray | None


@dataclass
class _TrackState:
    recent: deque[dict[str, Any]]
    written: tuple[Any, ...] | None = None
    written_at: float = -1e18
    seen: float = 0.0
    lying_since: float | None = None
    hands_since: float | None = None
    fired: set[str] | None = None


class PoseIndexer:
    def __init__(
        self,
        store: EventStore,
        camera: str,
        estimator: Any,
        interval_seconds: float = 1.0,
        still_factor: float = 3.0,
        min_box_height: int = 80,
        min_visible: int = 6,
        heartbeat_seconds: float = 10.0,
        store_keypoints: bool = True,
        alerts: list[str] | None = None,
        down_seconds: float = 10.0,
        hands_up_seconds: float = 3.0,
        blocking: bool = False,
        max_queue: int = 2,
    ) -> None:
        """
        estimator: anything with estimate(image, boxes) -> keypoints and min_confidence
                   (PoseEstimator, or a fake in tests).
        min_box_height: smaller people give unreliable keypoints (pixels, in whichever stream
                        the person is cropped from).
        min_visible: samples with fewer visible keypoints are not stored.
        alerts: event types to raise, from EVENT_TYPES (default: down only).
        """
        self.store = store
        self.camera = camera
        self.estimator = estimator
        self.interval = interval_seconds
        self.still_factor = still_factor
        self.min_box_height = min_box_height
        self.min_visible = min_visible
        self.heartbeat = heartbeat_seconds
        self.store_keypoints = store_keypoints
        self.alert_types = set(EVENT_TYPES[:1] if alerts is None else alerts)
        unknown = self.alert_types - set(EVENT_TYPES)
        if unknown:
            raise ValueError(f"pose.alerts: unknown {sorted(unknown)}; use {list(EVENT_TYPES)}")
        self.down_seconds = down_seconds
        self.hands_up_seconds = hands_up_seconds
        self.blocking = blocking
        self.min_confidence = float(getattr(estimator, "min_confidence", 0.3))
        self.dropped = 0
        self.stored = 0
        # main thread: when each track was last sampled, and its box then
        self._last: dict[int, tuple[float, np.ndarray]] = {}
        # worker thread: smoothing and event state per track
        self._state: dict[int, _TrackState] = {}
        self._lock = threading.Lock()  # guards the two below (read by the main thread)
        self._events: list[tuple[float, int, str, float]] = []
        self._overlay: dict[int, tuple[float, bytes]] = {}
        self._queue: queue.Queue[_Job | None] = queue.Queue(max_queue)
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"pose-{camera}")
        self._thread.start()

    @classmethod
    def from_config(cls, config: dict[str, Any], store: EventStore, camera: str,
                    blocking: bool) -> PoseIndexer:
        """config: the `pose` section."""
        estimator = PoseEstimator.from_config(config)
        options = {k: v for k, v in config.items() if k not in {
            "enabled", "cameras", "model", "min_confidence", "num_threads", "batch_size", "device"}}
        return cls(store, camera, estimator, blocking=blocking, **options)

    # ------------------------------------------------------------ main thread

    def update(self, frame: np.ndarray, tracks: list[Track], t: float,
               hires: HiResFrame | None = None) -> None:
        people = []
        carried = [(t.label, np.asarray(t.box, dtype=float)) for t in tracks
                   if not t.missed and t.label in CARRIED]
        for track in tracks:
            if track.missed or track.label != "person":
                continue
            box = np.asarray(track.box, dtype=float)
            hbox = hires.boxes.get(track.id) if hires is not None else None
            hi_ok = hbox is not None and hbox[3] - hbox[1] >= self.min_box_height
            if not hi_ok and box[3] - box[1] < self.min_box_height:
                continue
            last = self._last.get(track.id)
            interval = self.interval
            if last is not None and iou_matrix(box[None], last[1][None])[0, 0] > 0.85:
                interval *= self.still_factor  # not moving: its pose rarely changes
            if last is not None and t - last[0] < interval and not hi_ok:
                continue
            if last is not None and t - last[0] < self.interval:
                continue  # a keyframe match is welcome, but not more often than the interval
            self._last[track.id] = (t, box)
            extra = {"motion": _motion(box, last, t), "carrying": _carrying(box, carried)}
            people.append((track.id, box, hbox if hi_ok else None, extra))
        if len(self._last) > 2000:  # forget people not seen for 10 minutes
            self._last = {k: v for k, v in self._last.items() if t - v[0] < 600}
        if not people:
            return
        job = _Job(frame.copy(), t, people, hires.image if hires is not None else None)
        if self.blocking:
            self._queue.put(job)
            return
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            self.dropped += 1  # live: skip a sample rather than fall behind

    def take_events(self) -> list[tuple[float, int, str, float]]:
        """Events raised since the last call: (unix time, track id, type, seconds held)."""
        with self._lock:
            events, self._events = self._events, []
        return events

    def overlay(self, t: float, max_age: float = 2.0) -> dict[int, bytes]:
        """Latest packed keypoints per track (pose.pack_keypoints: relative to the box), if
        sampled in the last `max_age` seconds. Unpacked on the track's current box, the
        skeleton moves with the person between samples."""
        with self._lock:
            return {k: kp for k, (when, kp) in self._overlay.items() if t - when <= max_age}

    def close(self, timeout: float = 120.0) -> None:
        self._queue.put(None)
        self._thread.join(timeout=timeout)

    # ------------------------------------------------------------ worker

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            try:
                self._process(job)
            except Exception:  # never let one bad frame stop pose recording
                log.exception("Pose failed for camera %s", self.camera)

    def _process(self, job: _Job) -> None:
        t = job.time
        hi = [(i, box, hbox, extra) for i, box, hbox, extra in job.people
              if hbox is not None and job.hires is not None]
        sub = [(i, box, None, extra) for i, box, hbox, extra in job.people
               if hbox is None or job.hires is None]
        rows = []
        for source, group in ((job.hires, hi), (job.frame, sub)):
            if not group:
                continue
            crops = [hbox if hbox is not None else box for _, box, hbox, _ in group]
            for (track_id, box, _, extra), crop_box, kp in zip(
                    group, crops, self.estimator.estimate(source, crops)):
                facts = describe(kp, source, self.min_confidence)
                if facts.visible < self.min_visible:
                    continue
                blob = pack_keypoints(kp, crop_box)  # relative to the box: any stream
                with self._lock:
                    self._overlay[track_id] = (t, blob)
                row = self._smooth(track_id, t, {**facts.to_dict(), **extra}, box, blob)
                if row is not None:
                    rows.append(row)
        if rows:
            self.store.add_poses(rows)
            self.stored += len(rows)
        if len(self._state) > 500:
            gone = [k for k, s in self._state.items() if t - s.seen > 600]
            with self._lock:
                for k in gone:
                    del self._state[k]
                    self._overlay.pop(k, None)

    def _smooth(self, track_id: int, t: float, facts: dict[str, Any], box: np.ndarray,
                blob: bytes) -> dict[str, Any] | None:
        """Add a sample; return a row to store if the smoothed facts changed (or heartbeat)."""
        state = self._state.setdefault(track_id, _TrackState(deque(maxlen=3), fired=set()))
        state.recent.append(facts)
        state.seen = t
        stable = {f: _majority([s[f] for s in state.recent]) for f in FIELDS}
        self._check_events(track_id, state, stable, t)
        key = tuple(stable[f] for f in FIELDS)
        if key == state.written and t - state.written_at < self.heartbeat:
            return None
        state.written, state.written_at = key, t
        return {"camera": self.camera, "time": t, "track_id": track_id, **stable,
                "score": facts["score"], "box": box,
                "keypoints": blob if self.store_keypoints else b""}

    def _check_events(self, track_id: int, state: _TrackState, stable: dict[str, Any],
                      t: float) -> None:
        assert state.fired is not None
        for kind, active, attr, hold in (
                ("down", stable["posture"] == "lying", "lying_since", self.down_seconds),
                ("hands_up", stable["arms"] == "both_up", "hands_since", self.hands_up_seconds)):
            if not active:
                setattr(state, attr, None)
                state.fired.discard(kind)  # a new episode can fire again
                continue
            since = getattr(state, attr)
            if since is None:
                setattr(state, attr, t)
                since = t
            if kind in self.alert_types and kind not in state.fired and t - since >= hold:
                state.fired.add(kind)
                with self._lock:
                    self._events.append((t, track_id, kind, t - since))


def _motion(box: np.ndarray, last: tuple[float, np.ndarray] | None, t: float) -> str:
    """still / walking / running from the move since the person's last sample: the box centre
    shift and its change of size (walking towards the camera), in body heights per second."""
    if last is None or not 0 < t - last[0] <= 10:
        return "unknown"
    before = last[1]
    h0, h1 = before[3] - before[1], box[3] - box[1]
    if h0 < 1 or h1 < 1:
        return "unknown"
    shift = np.hypot((box[0] + box[2] - before[0] - before[2]) / 2,
                     (box[1] + box[3] - before[1] - before[3]) / 2) / ((h0 + h1) / 2)
    speed = np.hypot(shift, np.log(h1 / h0)) / (t - last[0])
    return "running" if speed >= RUNNING else "walking" if speed >= WALKING else "still"


def _carrying(box: np.ndarray, carried: list[tuple[str, np.ndarray]]) -> str | None:
    """Comma-separated CARRIED labels on this person (sorted), or None."""
    x1, y1, x2, y2 = box
    pad = 0.15 * (x2 - x1)
    found = {label for label, b in carried
             if x1 - pad <= (b[0] + b[2]) / 2 <= x2 + pad and y1 <= (b[1] + b[3]) / 2 <= y2
             and (b[2] - b[0]) * (b[3] - b[1]) < (x2 - x1) * (y2 - y1)}
    return ",".join(sorted(found)) or None


def _majority(values: list[Any]) -> Any:
    """Most common value of the last samples; on a tie, the latest."""
    counts = Counter(values)
    best = max(counts.values())
    return next(v for v in reversed(values) if counts[v] == best)

