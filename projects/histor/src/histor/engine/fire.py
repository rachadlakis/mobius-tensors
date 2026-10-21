"""Fire and smoke: Histor's own detector (training/fire_smoke) run on the picture, a `fire` alert
when flames or smoke persist.

Every `interval_seconds` the frame goes through the fire/smoke model (YOLOv9, ONNX; installed by
`download_model fire_smoke`). A class is *seen* on a check when any box of it scores at least
`min_score`. Once it is seen, nothing waits: the next frames are checked every
`suspect_interval_seconds` (about every frame on a CPU) until it is confirmed or gone for
`window_seconds`.

It raises an event once it has been seen on at least `hits` checks spread over at least
`min_seconds`, on at least `share` of the checks since the first of them (within the last
`window_seconds`). Flames flicker and smoke drifts, so a real fire keeps being seen, while a
glint, a reflection or a passing headlight is gone within a few tenths of a second and does
not add up. With the defaults (seen 3 times over 0.4 s, 60% of the checks) the alarm comes
~0.4 s after the fire is first seen, plus up to one `interval_seconds` before that first
sighting.

One event per episode and class: it can fire again once that class has not been seen for
`clear_seconds`. Events are type `fire`, rule `fire` or `smoke`, with the best score, through
the same channels as the other alerts; the snapshot has the boxes drawn on it.

Missing a fire is the costly mistake, so the defaults favour catching it: a fairly low
`min_score` and a short window, with persistence filtering the extra false alarms. The model
misses fires that fill the whole frame and tiny flames, and it is not a certified detector:
it adds early warning to smoke and heat detectors, it does not replace them.

    fire:
      enabled: true
      cameras: []              # empty = every camera
      model:
        path: models/fire_smoke_yolov9s_640.onnx   # yolov9t for weak CPUs (~3x faster)
      interval_seconds: 0.5
      suspect_interval_seconds: 0.1
      min_score: 0.25
      hits: 3
      min_seconds: 0.4
      share: 0.6
      window_seconds: 2.5
      clear_seconds: 60
      classes: [fire, smoke]
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from histor.engine.detector import Detection, Detector, create_detector

EVENT_TYPE = "fire"
CLASSES = ("fire", "smoke")
COLORS = {"fire": (0, 0, 255), "smoke": (200, 120, 40)}  # BGR: red, steel blue


@dataclass
class FireEvent:
    time: float
    kind: str     # fire | smoke
    score: float  # best score over the checks that raised it
    boxes: list[Detection] = field(default_factory=list)  # that kind's boxes on the last check


class FireWatch:
    def __init__(self, detector: Detector, interval_seconds: float = 0.5, min_score: float = 0.25,
                 hits: int = 3, clear_seconds: float = 60.0,
                 classes: list[str] | tuple[str, ...] = CLASSES,
                 suspect_interval_seconds: float = 0.1, min_seconds: float = 0.4,
                 share: float = 0.6, window_seconds: float = 2.5,
                 window: int | None = None) -> None:
        """window: the old rule's window in checks (`hits` of the last `window`); read as
        `window_seconds` = window x interval_seconds."""
        if window is not None:
            window_seconds = window * interval_seconds
        if hits < 1 or not 0 < share <= 1 or not 0 <= min_seconds < window_seconds:
            raise ValueError(f"fire: need hits >= 1 ({hits}), 0 < share <= 1 ({share}) and "
                             f"0 <= min_seconds ({min_seconds}) < window_seconds "
                             f"({window_seconds})")
        unknown = set(classes) - set(CLASSES)
        if unknown:
            raise ValueError(f"fire.classes: unknown {sorted(unknown)}; use {list(CLASSES)}")
        self.detector = detector
        self.interval = interval_seconds
        self.suspect_interval = min(suspect_interval_seconds, interval_seconds)
        self.min_score = min_score
        self.hits = hits
        self.min_seconds = min_seconds
        self.share = share
        self.window = window_seconds
        self.clear = clear_seconds
        self.classes = tuple(classes)
        # per class, the checks of the last `window_seconds`: (time, seen, best score)
        self._history: dict[str, deque[tuple[float, bool, float]]] = {
            c: deque() for c in self.classes}
        self._last_seen: dict[str, float] = {}
        self._fired: set[str] = set()
        self._last = -1e18
        self.last_detections: list[Detection] = []  # the latest check's boxes (for drawing)
        self.last_check = -1e18

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> FireWatch:
        model = {"backend": "yolov9_onnx", "path": "models/fire_smoke_yolov9s_640.onnx",
                 **(config.get("model") or {})}
        model.setdefault("conf_threshold", config.get("min_score", 0.25))
        own = {k: v for k, v in config.items() if k not in {"enabled", "cameras", "model"}}
        return cls(create_detector(model), **own)

    @property
    def suspected(self) -> bool:
        """Fire or smoke seen lately and not confirmed yet: checking every frame."""
        return any(kind not in self._fired and any(seen for _, seen, _ in history)
                   for kind, history in self._history.items())

    def update(self, frame: np.ndarray, t: float) -> list[FireEvent]:
        wait = self.suspect_interval if self.suspected else self.interval
        if t - self._last < wait - 1e-6:  # 9.1 - 9.0 is 0.0999...: still the next frame
            return []
        self._last = t
        detections = [d for d in self.detector.detect(frame)
                      if d.label in self.classes and d.score >= self.min_score]
        self.last_detections, self.last_check = detections, t
        events = []
        for kind in self.classes:
            mine = [d for d in detections if d.label == kind]
            history = self._history[kind]
            history.append((t, bool(mine), max((d.score for d in mine), default=0.0)))
            while history and t - history[0][0] > self.window:
                history.popleft()
            if mine:
                self._last_seen[kind] = t
            elif kind in self._fired and t - self._last_seen.get(kind, t) >= self.clear:
                self._fired.discard(kind)  # gone long enough: a new episode can alert again
            if kind not in self._fired and mine and self._confirmed(history, t):
                self._fired.add(kind)
                events.append(FireEvent(t, kind, round(max(s for _, _, s in history), 3), mine))
        return events

    def _confirmed(self, history: deque[tuple[float, bool, float]], t: float) -> bool:
        """From some sighting on: `hits` sightings over `min_seconds`, on `share` of the
        checks. Every sighting is tried as the start, so an old stray one does not dilute a
        fire that starts after it."""
        checks = list(history)
        for i, (start, seen, _) in enumerate(checks):
            if not seen or t - start < self.min_seconds:
                continue
            since = checks[i:]
            hits = sum(s for _, s, _ in since)
            if hits >= self.hits and hits >= self.share * len(since):
                return True
        return False

    def boxes(self, t: float, max_age: float = 2.0) -> list[Detection]:
        """The latest check's boxes, if it is recent (for the live overlay)."""
        return self.last_detections if t - self.last_check <= max_age else []


def draw_fire(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Fire/smoke boxes with their score (on the frame itself)."""
    from histor.engine.detector import draw_box

    for d in detections:
        draw_box(frame, (d.x1, d.y1, d.x2, d.y2), f"{d.label} {d.score:.2f}",
                 COLORS.get(d.label, (0, 0, 255)))
    return frame


def snapshot_with_boxes(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    image = draw_fire(frame.copy(), detections)
    cv2.putText(image, "FIRE/SMOKE", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                cv2.LINE_AA)
    return image
