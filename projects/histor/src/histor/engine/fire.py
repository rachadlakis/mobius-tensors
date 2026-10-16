"""Fire and smoke: Histor's own detector (training/fire_smoke) run on the picture, a `fire` alert
when flames or smoke persist.

Every `interval_seconds` the frame goes through the fire/smoke model (YOLOv9, ONNX; installed by
`download_model fire_smoke`). A class is *seen* on a check when any box of it scores at least
`min_score`. It raises an event once it has been seen on `hits` of the last `window` checks:
flames flicker and smoke drifts, so a real fire keeps being seen, while a single odd frame (a
lamp, a reflection, a passing headlight) does not add up. With the defaults (every 0.5 s,
3 of 5) the alarm comes 1-2.5 s after the fire is in view.

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
      min_score: 0.25
      hits: 3
      window: 5
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
                 hits: int = 3, window: int = 5, clear_seconds: float = 60.0,
                 classes: list[str] | tuple[str, ...] = CLASSES) -> None:
        if not 1 <= hits <= window:
            raise ValueError(f"fire: need 1 <= hits ({hits}) <= window ({window})")
        unknown = set(classes) - set(CLASSES)
        if unknown:
            raise ValueError(f"fire.classes: unknown {sorted(unknown)}; use {list(CLASSES)}")
        self.detector = detector
        self.interval = interval_seconds
        self.min_score = min_score
        self.hits = hits
        self.clear = clear_seconds
        self.classes = tuple(classes)
        self._seen = {c: deque(maxlen=window) for c in self.classes}
        self._best = {c: deque(maxlen=window) for c in self.classes}
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

    def update(self, frame: np.ndarray, t: float) -> list[FireEvent]:
        if t - self._last < self.interval:
            return []
        self._last = t
        detections = [d for d in self.detector.detect(frame)
                      if d.label in self.classes and d.score >= self.min_score]
        self.last_detections, self.last_check = detections, t
        events = []
        for kind in self.classes:
            mine = [d for d in detections if d.label == kind]
            self._seen[kind].append(bool(mine))
            self._best[kind].append(max((d.score for d in mine), default=0.0))
            if mine:
                self._last_seen[kind] = t
            elif kind in self._fired and t - self._last_seen.get(kind, t) >= self.clear:
                self._fired.discard(kind)  # gone long enough: a new episode can alert again
            if kind not in self._fired and sum(self._seen[kind]) >= self.hits:
                self._fired.add(kind)
                events.append(FireEvent(t, kind, round(max(self._best[kind]), 3), mine))
        return events

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


def fire_enabled(config: dict[str, Any] | None, camera: str) -> bool:
    if not config or not config.get("enabled"):
        return False
    cameras = config.get("cameras") or []
    return not cameras or camera in cameras
