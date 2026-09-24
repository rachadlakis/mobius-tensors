"""Frame sources: webcam index, image file, video file, or stream URL (rtsp/http)."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from cameras import LiveCapture, is_stream_url, mask_url

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Frame:
    image: np.ndarray
    index: int
    timestamp: float
    fps: float


def is_live(source: str | int) -> bool:
    return isinstance(source, int) or str(source).isdigit() or "://" in str(source)


def iter_frames(source: str | int) -> Iterator[Frame]:
    """Yield frames from `source`. A digit string like "0" means a webcam index."""
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    if is_stream_url(source):
        yield from iter_stream(str(source))
        return

    if isinstance(source, str) and Path(source).suffix.lower() in IMAGE_EXTENSIONS:
        image = cv2.imread(source)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {source}")
        yield Frame(image, 0, 0.0, 0.0)
        return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {mask_url(source)}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    live = is_live(source)
    start = time.monotonic()
    index = 0
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            timestamp = time.monotonic() - start if live else index / fps
            yield Frame(image, index, timestamp, fps)
            index += 1
    finally:
        cap.release()


def iter_stream(url: str) -> Iterator[Frame]:
    """IP camera / network stream: never ends, skips stale frames, reconnects on drops."""
    with LiveCapture(url) as cap:
        start = time.monotonic()
        index = 0
        while True:
            # Short timeout keeps Ctrl+C responsive while the camera is reconnecting.
            image = cap.read(timeout=1.0)
            if image is None:
                continue
            yield Frame(image, index, time.monotonic() - start, cap.fps)
            index += 1
