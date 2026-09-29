"""detect -> track -> zone rules -> alerts (+ search index), per frame.

Shared by the CLI, the UI and the server. With a `store` section in the config, events,
snapshots and the search index are saved to SQLite for the API (src/server.py).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from alerts import AlertManager
from detector import Detection, Detector, create_detector, draw_detections
from tracker import Track, Tracker
from zones import ZoneEvent, ZoneMonitor

if TYPE_CHECKING:
    from indexer import SearchIndexer
    from store import EventStore


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
        store: EventStore | None = None,
        indexer: SearchIndexer | None = None,
        start_time: float | None = None,
    ) -> None:
        """start_time: unix time of timestamp 0 (live: now; a recorded file: when it was filmed)."""
        if (monitor and monitor.rules or indexer) and tracker is None:
            tracker = Tracker()  # zones and the search index need object IDs
        self.detector = detector
        self.tracker = tracker
        self.monitor = monitor
        self.alerts = alerts
        self.store = store
        self.indexer = indexer
        self.start_time = time.time() if start_time is None else start_time

    def wall_time(self, timestamp: float) -> float:
        return self.start_time + timestamp

    @classmethod
    def from_config(cls, config: dict[str, Any], source: str | None = None,
                    start_time: float | None = None, live: bool = True) -> Pipeline:
        """source: the camera *name* (used in events and the index, never the URL).
        live: False for video files, so indexing processes every step instead of skipping."""
        from store import EventStore

        store = EventStore.from_config(config.get("store"))
        search = config.get("search") or {}
        indexer = None
        if store is not None and search.get("enabled"):
            from indexer import SearchIndexer

            indexer = SearchIndexer.from_config(search, config.get("faces"), store,
                                                camera=str(source or "camera"), blocking=not live)
        tracking = config.get("tracking") or {}
        tracker = Tracker(**{k: v for k, v in tracking.items() if k != "enabled"}) if (
            tracking.get("enabled") or config.get("zones") or indexer
        ) else None
        monitor = ZoneMonitor.from_config(config.get("zones")) if config.get("zones") else None
        alerts = AlertManager.from_config(config.get("alerts"), source=source,
                                          store=store) if monitor else None
        return cls(create_detector(config["model"]), tracker, monitor, alerts, store, indexer,
                   start_time)

    def close(self) -> None:
        """Flush the search index and pending alerts (call when the source ends)."""
        if self.indexer is not None:
            self.indexer.close()
        if self.alerts is not None:
            self.alerts.close()

    def process(self, frame: np.ndarray, timestamp: float) -> FrameResult:
        result = FrameResult(self.detector.detect(frame), timestamp=timestamp)
        if self.tracker is None:
            return result
        result.tracks, removed = self.tracker.update(result.detections)
        wall = self.wall_time(timestamp)
        if self.monitor:
            h, w = frame.shape[:2]
            events = self.monitor.update(result.tracks, removed, timestamp, (w, h))
            result.events = (self.alerts.handle(events, frame, result.tracks, wall)
                             if self.alerts else events)
        if self.indexer is not None:
            self.indexer.update(frame, result.tracks, wall)
        return result

    def draw(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:
        if self.tracker is None:
            return draw_detections(frame, result.detections)
        visible = [t for t in result.tracks if not t.missed]
        if self.monitor:
            return self.monitor.draw(frame, visible, result.timestamp)
        return ZoneMonitor([]).draw(frame, visible, result.timestamp)
