"""detect -> track -> zone rules -> alerts (+ search index), per frame.

Shared by the CLI, the UI and the server. With a `store` section in the config, events,
snapshots and the search index are saved to SQLite for the API (src/server.py).

With a `recording` section and a live camera, the camera's main stream is also opened
(src/mainstream.py): saved to disk, and each keyframe (about 1/s) is run through the detector
once and matched to the tracks, so the search index can crop from full resolution.

With a `pose` section (src/pose_index.py), tracked people also get body keypoints about once a
second; the facts derived from them (posture, arms, facing, clothing colours) go to the store,
and `down` / `hands_up` events go through the alert channels like area and line events.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from alerts import AlertManager
from detector import Detection, Detector, create_detector, draw_detections
from tracker import Track, Tracker, iou_matrix
from zones import ZoneEvent, ZoneMonitor

if TYPE_CHECKING:
    from indexer import HiResFrame, SearchIndexer
    from pose_index import PoseIndexer
    from mainstream import MainStream
    from store import EventStore

log = logging.getLogger(__name__)
KEYFRAME_MIN_IOU = 0.3  # keyframe detection <-> track; the two streams are not frame-synced


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
        mainstream: MainStream | None = None,
        pose: PoseIndexer | None = None,
    ) -> None:
        """start_time: unix time of timestamp 0 (live: now; a recorded file: when it was filmed).
        mainstream: the camera's main stream (recording + keyframes for hi-res crops)."""
        if (monitor and monitor.rules or indexer or pose) and tracker is None:
            tracker = Tracker()  # zones, the search index and poses need object IDs
        self.detector = detector
        self.tracker = tracker
        self.monitor = monitor
        self.alerts = alerts
        self.store = store
        self.indexer = indexer
        self.mainstream = mainstream
        self.pose = pose
        self.start_time = time.time() if start_time is None else start_time

    def wall_time(self, timestamp: float) -> float:
        return self.start_time + timestamp

    @classmethod
    def from_config(cls, config: dict[str, Any], source: str | None = None,
                    start_time: float | None = None, live: bool = True,
                    main_url: str | None = None) -> Pipeline:
        """source: the camera *name* (used in events and the index, never the URL).
        live: False for video files, so indexing processes every step instead of skipping.
        main_url: the camera's main stream, for the `recording` section (live cameras only)."""
        from store import EventStore

        store = EventStore.from_config(config.get("store"))
        search = config.get("search") or {}
        indexer = None
        if store is not None and search.get("enabled"):
            from indexer import SearchIndexer

            indexer = SearchIndexer.from_config(search, config.get("faces"), store,
                                                camera=str(source or "camera"), blocking=not live)
        pose = None
        if store is not None:
            from pose import pose_enabled

            camera = str(source or "camera")
            if pose_enabled(config.get("pose"), camera):
                from pose_index import PoseIndexer

                try:
                    pose = PoseIndexer.from_config(config["pose"], store, camera,
                                                   blocking=not live)
                except FileNotFoundError as e:  # record the rest anyway
                    log.warning("No pose for camera %s: %s", camera, e)
        tracking = config.get("tracking") or {}
        tracker = Tracker(**{k: v for k, v in tracking.items() if k != "enabled"}) if (
            tracking.get("enabled") or config.get("zones") or indexer or pose
        ) else None
        monitor = ZoneMonitor.from_config(config.get("zones")) if config.get("zones") else None
        alerts = AlertManager.from_config(config.get("alerts"), source=source, store=store) if (
            monitor or (pose and pose.alert_types)) else None
        mainstream = None
        if live and main_url and config.get("recording"):
            from mainstream import MainStream

            mainstream = MainStream.from_config(config["recording"], main_url,
                                                camera=str(source or "camera"),
                                                keyframes=indexer is not None or pose is not None)
        return cls(create_detector(config["model"]), tracker, monitor, alerts, store, indexer,
                   start_time, mainstream, pose)

    def close(self) -> None:
        """Flush the search index and pending alerts (call when the source ends)."""
        if self.mainstream is not None:
            self.mainstream.close()
        if self.indexer is not None:
            self.indexer.close()
        if self.pose is not None:
            self.pose.close()
            self._pose_events(None, [])  # events from the last samples
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
        hires = self._match_keyframe(frame, result) if self.indexer or self.pose else None
        if self.indexer is not None:
            self.indexer.update(frame, result.tracks, wall, hires)
        if self.pose is not None:
            self.pose.update(frame, result.tracks, wall, hires)
            result.events += self._pose_events(frame, result.tracks)
        return result

    def _pose_events(self, frame: np.ndarray | None, tracks: list[Track]) -> list[ZoneEvent]:
        """Pose events (person down, hands up) through the alert channels and the event log."""
        from pose_index import EVENT_RULES

        assert self.pose is not None
        events = [ZoneEvent(wall - self.start_time, kind, EVENT_RULES[kind], track_id, "person",
                            dwell_seconds=round(held, 1))
                  for wall, track_id, kind, held in self.pose.take_events()]
        if self.alerts is None:
            return events
        # snapshots use the current frame: a pose event is a second or two behind it
        return [e for event in events for e in self.alerts.handle(
            [event], frame, tracks, self.start_time + event.timestamp)]

    def _match_keyframe(self, frame: np.ndarray, result: FrameResult) -> HiResFrame | None:
        """Detect on a new main-stream keyframe (if one arrived) and pair its boxes with the
        current tracks. Detection on the keyframe itself gives exact hi-res boxes, whatever
        the delay between the two streams; costs one extra detection per keyframe."""
        keyframe = self.mainstream.take_keyframe() if self.mainstream is not None else None
        if keyframe is None:
            return None
        from indexer import HiResFrame

        tracks = [t for t in result.tracks if not t.missed]
        detections = self.detector.detect(keyframe.image) if tracks else []
        boxes: dict[int, np.ndarray] = {}
        if detections:
            h, w = frame.shape[:2]
            kh, kw = keyframe.image.shape[:2]
            hires = np.array([[d.x1, d.y1, d.x2, d.y2] for d in detections], dtype=float)
            # sub streams may be squeezed (704x576 of a 16:9 view): scale x and y separately
            scaled = hires * np.array([w / kw, h / kh, w / kw, h / kh])
            iou = iou_matrix(np.array([t.box for t in tracks], dtype=float), scaled)
            used: set[int] = set()
            for ti, di in sorted(np.ndindex(iou.shape), key=lambda ij: -iou[ij]):  # greedy
                if iou[ti, di] < KEYFRAME_MIN_IOU:
                    break
                if tracks[ti].id in boxes or di in used:
                    continue
                boxes[tracks[ti].id] = hires[di]
                used.add(di)
        return HiResFrame(keyframe.time, keyframe.image, boxes)

    def draw(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:
        if self.tracker is None:
            return draw_detections(frame, result.detections)
        visible = [t for t in result.tracks if not t.missed]
        if self.pose is not None:
            from pose import draw_skeleton, unpack_keypoints

            skeletons = self.pose.overlay(self.wall_time(result.timestamp))
            for t in visible:
                if t.id in skeletons:
                    draw_skeleton(frame, unpack_keypoints(skeletons[t.id], t.box),
                                  self.pose.min_confidence)
        if self.monitor:
            return self.monitor.draw(frame, visible, result.timestamp)
        return ZoneMonitor([]).draw(frame, visible, result.timestamp)
