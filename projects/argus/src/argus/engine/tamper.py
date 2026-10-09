"""Camera tampering and health, from the picture alone (no model): a `tamper` event when a
camera is covered, sprayed, turned or broken, so a site is never silently blind.

Every `interval_seconds`, on a 320-pixel-wide grey copy of the frame:

    blocked   almost no contrast (grey-level spread below `flat_std`): lens covered, sprayed,
              pointed at a wall
    dark      mean brightness below `dark_mean` with some contrast left: lights off, IR broken
    blurred   sharpness (variance of the Laplacian) below `blur_ratio` of this camera's own
              usual sharpness, learned over its first `learn_seconds`: defocused, fogged, smeared
    moved     the picture's structure no longer matches the reference: correlation of the two
              gradient maps (80x45, edges' strength) below `moved_similarity`. The same view
              stays around 0.8 with people walking through or the light changing; a camera
              turned by an eighth of its width falls to ~0.4, a different view to ~0.2
    frozen    the picture has not changed at all (to the last grey level): a stuck stream
              (only judged on a usable picture: a covered lens is `blocked`, not frozen)

A condition must hold for `hold_seconds` before the event, and fires once until it clears.
After `moved`, the new view becomes the reference. The reference also follows slow changes
(daylight, shadows) every `refresh_minutes` while the view still matches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

EVENT_TYPE = "tamper"
CHECKS = ("blocked", "dark", "blurred", "moved", "frozen")


@dataclass
class TamperEvent:
    time: float
    kind: str      # one of CHECKS
    value: float   # the measurement that tripped it (std, mean, sharpness ratio, overlap...)


@dataclass
class Measure:
    std: float
    mean: float
    sharpness: float
    gradient: np.ndarray = field(repr=False)  # 80x45 map of edge strength
    grey: np.ndarray = field(repr=False)


def measure(frame: np.ndarray, width: int = 320) -> Measure:
    h, w = frame.shape[:2]
    scale = width / w
    small = cv2.resize(frame, (width, max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    blurred = cv2.GaussianBlur(grey.astype(np.float32), (0, 0), 2)
    magnitude = cv2.magnitude(cv2.Sobel(blurred, cv2.CV_32F, 1, 0),
                              cv2.Sobel(blurred, cv2.CV_32F, 0, 1))
    gradient = cv2.resize(magnitude, (80, 45), interpolation=cv2.INTER_AREA)
    return Measure(float(grey.std()), float(grey.mean()),
                   float(cv2.Laplacian(grey, cv2.CV_64F).var()), gradient, grey)


def similarity(reference: np.ndarray, current: np.ndarray) -> float:
    """Pearson correlation of two gradient maps: 1 = same structure, ~0 = unrelated."""
    a = reference - reference.mean()
    b = current - current.mean()
    denominator = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum()) / denominator if denominator > 1e-9 else 1.0


class TamperCheck:
    def __init__(self, interval_seconds: float = 2.0, hold_seconds: float = 20.0,
                 flat_std: float = 6.0, dark_mean: float = 12.0, blur_ratio: float = 0.25,
                 moved_similarity: float = 0.5, learn_seconds: float = 60.0,
                 refresh_minutes: float = 30.0, checks: list[str] | None = None,
                 min_structure: float = 2.0) -> None:
        """min_structure: a view whose gradient map varies less than this (a blank wall) can't
        tell `moved`."""
        self.checks = set(CHECKS if checks is None else checks)
        unknown = self.checks - set(CHECKS)
        if unknown:
            raise ValueError(f"tamper.checks: unknown {sorted(unknown)}; use {list(CHECKS)}")
        self.interval = interval_seconds
        self.hold = hold_seconds
        self.flat_std = flat_std
        self.dark_mean = dark_mean
        self.blur_ratio = blur_ratio
        self.moved_similarity = moved_similarity
        self.learn = learn_seconds
        self.refresh = refresh_minutes * 60
        self.min_structure = min_structure
        self.started: float | None = None
        self.sharpness: float | None = None  # this camera's usual sharpness
        self._learning: list[float] = []
        self.reference: np.ndarray | None = None
        self._reference_at = 0.0
        self._previous: np.ndarray | None = None
        self._last = -1e18
        self._since: dict[str, float] = {}
        self._fired: set[str] = set()
        self.last_measure: Measure | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> TamperCheck:
        return cls(**{k: v for k, v in config.items() if k not in {"enabled", "cameras"}})

    def update(self, frame: np.ndarray, t: float) -> list[TamperEvent]:
        if t - self._last < self.interval:
            return []
        self._last = t
        m = measure(frame)
        self.last_measure = m
        if self.started is None:
            self.started = t
        active = self._conditions(m, t)
        self._previous = m.grey
        events = []
        for kind in CHECKS:
            if kind not in self.checks:
                continue
            if kind not in active:
                self._since.pop(kind, None)
                self._fired.discard(kind)
                continue
            since = self._since.setdefault(kind, t)
            if kind not in self._fired and t - since >= self.hold:
                self._fired.add(kind)
                events.append(TamperEvent(t, kind, round(active[kind], 3)))
                if kind == "moved":  # the new view is the reference from now on
                    self.reference, self._reference_at = m.gradient, t
        return events

    def _conditions(self, m: Measure, t: float) -> dict[str, float]:
        active: dict[str, float] = {}
        if m.std < self.flat_std:
            active["blocked"] = m.std
        elif m.mean < self.dark_mean:
            active["dark"] = m.mean
        if active:
            return active  # no usable picture: frozen, sharpness and structure mean nothing
        if self._previous is not None and self._previous.shape == m.grey.shape and \
                not cv2.absdiff(self._previous, m.grey).any():
            active["frozen"] = 0.0
        # this camera's usual sharpness: median of the first samples, then a slow average
        if self.sharpness is None:
            self._learning.append(m.sharpness)
            started = self.started if self.started is not None else t
            if t - started >= self.learn and self._learning:
                self.sharpness = float(np.median(self._learning))
        elif m.sharpness < self.blur_ratio * self.sharpness:
            active["blurred"] = m.sharpness / self.sharpness
        if self.sharpness is not None and "blurred" not in active:
            self.sharpness = 0.99 * self.sharpness + 0.01 * m.sharpness
        if "blurred" in active:
            return active
        if self.reference is None:
            if float(m.gradient.std()) >= self.min_structure:
                self.reference, self._reference_at = m.gradient, t
            return active
        same = similarity(self.reference, m.gradient)
        if same < self.moved_similarity and float(self.reference.std()) >= self.min_structure:
            active["moved"] = same
        elif same >= 0.7 and t - self._reference_at >= self.refresh:
            self.reference, self._reference_at = m.gradient, t  # follow daylight and shadows
        return active


def tamper_enabled(config: dict[str, Any] | None, camera: str) -> bool:
    if not config or not config.get("enabled"):
        return False
    cameras = config.get("cameras") or []
    return not cameras or camera in cameras
