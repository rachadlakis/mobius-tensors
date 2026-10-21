"""Test helpers shared by the test files (`import kit`: tests/ is on the path, conftest.py)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from histor.engine.detector import Detection
from histor.engine.tamper import TamperCheck
from histor.engine.tracker import Track


class Scripted:
    """A detector that returns script(call number); counts its calls."""

    def __init__(self, script: Callable[[int], list[Detection]]) -> None:
        self.script, self.calls = script, 0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        out = self.script(self.calls)
        self.calls += 1
        return out


def fire_box(label: str = "fire", score: float = 0.8) -> Detection:
    return Detection(100, 100, 200, 220, score, 0 if label == "fire" else 1, label)


class Clock:
    """A detector that sees fire when burning(now) says so; records when it was asked."""

    def __init__(self, burning: Callable[[float], bool]) -> None:
        self.burning, self.now, self.times = burning, 0.0, list[float]()

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self.times.append(self.now)
        return [fire_box()] if self.burning(self.now) else []


def live(watch: Any, clock: Clock, seconds: float, fps: float = 25.0) -> list[Any]:
    """Frames at `fps` for `seconds` through watch.update, like a camera: the events."""
    frame = np.zeros((10, 10, 3), np.uint8)
    events = []
    for i in range(int(seconds * fps)):
        clock.now = round(i / fps, 3)
        events += watch.update(frame, clock.now)
    return events


def track(tid: int, box: Any, label: str = "person", **fields: Any) -> Track:
    """A Track with that box and label; `fields` set others (missed, hits)."""
    t = Track(id=tid, box=np.array(box, float), score=0.9, label_votes={label: 1.0})
    for name, value in fields.items():
        setattr(t, name, value)
    return t


def unit(*values: float) -> np.ndarray:
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)


class FakeEmbedder:
    """CLIP stand-in: each image's vector follows its mean brightness; counts the images."""

    def __init__(self) -> None:
        self.calls = 0

    def embed_images(self, images: list[np.ndarray]) -> np.ndarray:
        self.calls += len(images)
        return np.stack([unit(float(im.mean()) + 1, 1) for im in images])


# ------------------------------------------------------------------ textured views

def scene_shapes(seed: int = 0, shape: tuple[int, int] = (360, 640)) -> np.ndarray:
    """A textured view: rectangles and lines, like a street with buildings."""
    rng = np.random.default_rng(seed)
    image = np.full((*shape, 3), 110, np.uint8)
    for _ in range(60):
        x, y = int(rng.integers(0, shape[1] - 60)), int(rng.integers(0, shape[0] - 40))
        w, h = int(rng.integers(15, 80)), int(rng.integers(10, 60))
        cv2.rectangle(image, (x, y), (x + w, y + h),
                      tuple(int(c) for c in rng.integers(0, 255, 3)), -1)
    for _ in range(30):
        p = tuple(int(v) for v in rng.integers(0, 640, 2))
        q = tuple(int(v) for v in rng.integers(0, 360, 2))
        cv2.line(image, (p[0], q[0]), (p[1], q[1]), (240, 240, 240), 2)
    return image


def scene_noise(seed: int = 0, shape: tuple[int, int] = (360, 640)) -> np.ndarray:
    """A textured still view: smooth colour noise, edges everywhere (like a real yard)."""
    h, w = shape
    noise = np.random.default_rng(seed).integers(0, 255, (h // 8, w // 8, 3), dtype=np.uint8)
    return cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)


def noisy(image: np.ndarray, seed: int) -> np.ndarray:
    """The image with camera noise (sigma 3)."""
    noise = np.random.default_rng(seed).normal(0, 3, image.shape)
    return np.clip(image + noise, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ tamper

def feed(check: TamperCheck, frames: Any, start: float, seconds: float, step: float = 1.0
         ) -> tuple[list[Any], float]:
    """`frames` (an image, or i -> image) for `seconds` from `start`: the events, the end time."""
    events = []
    t = start
    i = 0
    while t < start + seconds:
        frame = frames(i) if callable(frames) else frames
        events += check.update(frame, t)
        t += step
        i += 1
    return events, t


def learned(**options: Any) -> tuple[TamperCheck, np.ndarray, float]:
    """A check that has learned its normal view (60 s of a noisy textured scene)."""
    check = TamperCheck(interval_seconds=1, hold_seconds=10, learn_seconds=30, **options)
    view = scene_shapes()
    events, t = feed(check, lambda i: noisy(view, i), 0, 60)
    assert events == [] and check.sharpness is not None and check.reference is not None
    return check, view, t
