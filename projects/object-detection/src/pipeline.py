"""detect -> track -> zone rules -> alerts, per frame. Shared by the CLI, the UI and (later) the server."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alerts import AlertManager
from detector import Detection, Detector, create_detector, draw_detections
from tracker import Track, Tracker
from zones import ZoneEvent, ZoneMonitor


@dataclass
class FrameResult:
    detections: list[Detection]
    tracks: list[Track] = field(default_factory=list)
    events: list[ZoneEvent] = field(default_factory=list)
    timestamp: float = 0.0


class Pipeline:
    def __init__(
        self,
        detector: Detector,
        tracker: Tracker | None = None,
        monitor: ZoneMonitor | None = None,
        alerts: AlertManager | None = None,
    ) -> None:
        if monitor and monitor.rules and tracker is None:
            tracker = Tracker()  # zones need object IDs
        self.detector = detector
        self.tracker = tracker
        self.monitor = monitor
        self.alerts = alerts

    @classmethod
    def from_config(cls, config: dict[str, Any], source: str | None = None) -> Pipeline:
        tracking = config.get("tracking") or {}
        tracker = Tracker(**{k: v for k, v in tracking.items() if k != "enabled"}) if (
            tracking.get("enabled") or config.get("zones")
        ) else None
        monitor = ZoneMonitor.from_config(config.get("zones")) if config.get("zones") else None
        alerts = AlertManager.from_config(config.get("alerts"), source=source) if monitor else None
        return cls(create_detector(config["model"]), tracker, monitor, alerts)

    def process(self, frame: np.ndarray, timestamp: float) -> FrameResult:
        result = FrameResult(self.detector.detect(frame), timestamp=timestamp)
        if self.tracker is None:
            return result
        result.tracks, removed = self.tracker.update(result.detections)
        if self.monitor:
            h, w = frame.shape[:2]
            events = self.monitor.update(result.tracks, removed, timestamp, (w, h))
            result.events = self.alerts.handle(events) if self.alerts else events
        return result

    def draw(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:
        if self.tracker is None:
            return draw_detections(frame, result.detections)
        visible = [t for t in result.tracks if not t.missed]
        if self.monitor:
            return self.monitor.draw(frame, visible, result.timestamp)
        return ZoneMonitor([]).draw(frame, visible, result.timestamp)
