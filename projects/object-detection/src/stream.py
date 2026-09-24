"""Frame sources: webcam index, image file, video file, or stream URL (rtsp/http)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_frames(source: str | int) -> Iterator[np.ndarray]:
    """Yield BGR frames from `source`. A digit string like "0" means a webcam index."""
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    if isinstance(source, str) and Path(source).suffix.lower() in IMAGE_EXTENSIONS:
        frame = cv2.imread(source)
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {source}")
        yield frame
        return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()
