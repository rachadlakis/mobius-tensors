"""Live view: read a camera on a background thread, detect, and serve annotated frames as MJPEG.

The camera is opened by this process (webcam index or RTSP/HTTP URL), which is how it will run
on a server too. Browsers display `multipart/x-mixed-replace` JPEG streams natively in <img>.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import Any

import cv2

from histor.engine.detector import Detector
from histor.engine.pipeline import Pipeline
from histor.engine.stream import iter_frames

BOUNDARY = "frame"


class LiveView:
    def __init__(self, jpeg_quality: int = 80) -> None:
        self.jpeg_quality = jpeg_quality
        self._pipeline: Pipeline | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._cond = threading.Condition()
        self._jpeg: bytes | None = None
        self._seq = 0
        self.stats: dict[str, Any] = {}

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, source: str | int, pipeline: Pipeline) -> None:
        self.stop()
        self._pipeline = pipeline
        self._stop.clear()
        self.stats = {"source": str(source), "status": "opening camera..."}
        self._thread = threading.Thread(target=self._run, args=(source,), daemon=True)
        self._thread.start()

    def set_detector(self, detector: Detector) -> None:
        """Swap model/thresholds while running; takes effect on the next frame."""
        if self._pipeline is not None:
            self._pipeline.detector = detector

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._thread = None
        with self._cond:
            self._cond.notify_all()

    def _run(self, source: str | int) -> None:
        frames, last = 0, time.perf_counter()
        total = 0
        try:
            for frame in iter_frames(source):  # breaking out closes (releases) the camera
                if self._stop.is_set():
                    break
                pipeline = self._pipeline
                assert pipeline is not None
                start = time.perf_counter()
                result = pipeline.process(frame.image, frame.timestamp)
                infer_ms = (time.perf_counter() - start) * 1000
                annotated = pipeline.draw(frame.image, result)
                ok, jpeg = cv2.imencode(
                    ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                )
                if not ok:
                    continue
                with self._cond:
                    self._jpeg = jpeg.tobytes()
                    self._seq += 1
                    self._cond.notify_all()

                frames += 1
                total += 1
                now = time.perf_counter()
                self.stats.update(
                    status="live",
                    infer_ms=round(infer_ms, 1),
                    size=f"{frame.image.shape[1]}x{frame.image.shape[0]}",
                    detections=[f"{d.label} {d.score:.2f}" for d in result.detections],
                )
                if now - last >= 1.0:
                    self.stats["fps"] = round(frames / (now - last), 1)
                    frames, last = 0, now
            else:
                self.stats["status"] = "source ended" if total else (
                    "error: the camera opened but sent no frames. It is probably in use by "
                    "another app or browser tab: close that and press Start again."
                )
        except Exception as exc:  # surfaced in the UI instead of dying silently
            self.stats["status"] = f"error: {exc}"
        finally:
            if self._stop.is_set():
                self.stats["status"] = "stopped"

    def mjpeg(self) -> Iterator[bytes]:
        """multipart/x-mixed-replace body: yields each new frame once, ends when stopped."""
        seen = -1
        while True:
            with self._cond:
                self._cond.wait_for(
                    lambda: self._seq != seen or self._stop.is_set() or not self.running,
                    timeout=2.0,
                )
                if self._seq == seen:
                    if self._stop.is_set() or not self.running:
                        return
                    continue
                seen, jpeg = self._seq, self._jpeg
            if jpeg is not None:
                yield (
                    f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                    f"Content-Length: {len(jpeg)}\r\n\r\n"
                ).encode() + jpeg + b"\r\n"
