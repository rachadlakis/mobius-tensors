"""detect -> track -> zone rules -> alerts (+ search index), per frame.

Shared by the CLI, the UI and the server. With a `store` section in the config, events,
snapshots and the search index are saved to SQLite for the API (server.py).

With a `recording` section and a live camera, the camera's main stream is also opened
(mainstream.py): saved to disk, and each keyframe (about 1/s) is run through the detector
once and matched to the tracks, so the search index can crop from full resolution.

With a `pose` section (pose_index.py), tracked people also get body keypoints about once a
second; the facts derived from them (posture, arms, facing, clothing colours) go to the store,
and `down` / `hands_up` events go through the alert channels like area and line events.

With a `plates` section (plate_index.py), tracked vehicles get their number plate read about
once a second (from the main-stream keyframe when there is one); reads go to the store, and a
plate on the watchlist raises a `plate` event through the same channels.

With a `sounds` section (sound.py) and a source with audio, the camera's microphone is
classified on its own thread (YAMNet): smoke alarms, breaking glass, gunshots, screams and
sirens become `sound` events through the same channels. The audio itself is never kept.

With a `fire` section (fire.py), the frame also goes through Histor's own fire/smoke model
about twice a second; flames or smoke seen on several checks in a row raise a `fire` event
through the same channels, with the boxes drawn on the snapshot.

When search motion or fire is on, each frame is also analysed for change once (change.py:
shake, background, light, flicker), and every part that needs it reads that one analysis.

Two rules need no model: `tamper` (tamper.py: the picture covered, dark, blurred, turned or
frozen) and `abandoned` (abandoned.py: a bag or suitcase left still with nobody near it).

With `alerts.verify` (verify.py), a local vision model checks each alert's snapshot before it
is sent, and can hold back false alarms.

With `timings=True` (the scenario bench, scenarios.py), process() records how long each stage
took (ms, the latest TIMINGS_KEPT per stage) in `timings`; off by default.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from histor.engine.alerts import AlertManager
from histor.engine.detector import Detection, Detector, create_detector, draw_detections
from histor.engine.boxes import greedy_match, iou_matrix
from histor.engine.tracker import Track, Tracker
from histor.engine.zones import ZoneEvent, ZoneMonitor

if TYPE_CHECKING:
    from histor.engine.change import Change, ChangeAnalyser
    from histor.engine.indexer import HiResFrame, SearchIndexer
    from histor.engine.plate_index import PlateIndexer
    from histor.engine.pose_index import PoseIndexer
    from histor.engine.sound import SoundListener
    from histor.engine.abandoned import AbandonedWatch
    from histor.engine.tamper import TamperCheck
    from histor.engine.fire import FireWatch
    from histor.engine.mainstream import MainStream
    from histor.engine.store import EventStore

log = logging.getLogger(__name__)
KEYFRAME_MIN_IOU = 0.3  # keyframe detection <-> track; the two streams are not frame-synced
TIMINGS_KEPT = 10_000  # per stage, with `timings` on


@dataclass
class FrameResult:
    detections: list[Detection]
    tracks: list[Track] = field(default_factory=list)
    events: list[ZoneEvent] = field(default_factory=list)
    timestamp: float = 0.0
    change: Change | None = None  # the frame's change analysis, when the Pipeline has one


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
        plates: PlateIndexer | None = None,
        sounds: SoundListener | None = None,
        tamper: TamperCheck | None = None,
        abandoned: AbandonedWatch | None = None,
        fire: FireWatch | None = None,
        timings: bool = False,
        change: ChangeAnalyser | None = None,
    ) -> None:
        """start_time: unix time of timestamp 0 (live: now; a recorded file: when it was filmed).
        mainstream: the camera's main stream (recording + keyframes for hi-res crops).
        timings: record each stage's time per frame in `timings` (stage -> ms).
        change: the camera's change analysis (change.py), read by search motion and tamper."""
        if (monitor and monitor.rules or indexer or pose or plates or abandoned) and \
                tracker is None:
            tracker = Tracker()  # zones, the search index and poses need object IDs
        self.detector = detector
        self.tracker = tracker
        self.monitor = monitor
        self.alerts = alerts
        self.store = store
        self.indexer = indexer
        self.mainstream = mainstream
        self.pose = pose
        self.plates = plates
        self.sounds = sounds
        self.tamper = tamper
        self.abandoned = abandoned
        self.fire = fire
        self.change = change
        self.start_time = time.time() if start_time is None else start_time
        self.timings: dict[str, deque[float]] | None = {} if timings else None

    def wall_time(self, timestamp: float) -> float:
        return self.start_time + timestamp

    @classmethod
    def from_config(cls, config: dict[str, Any], source: str | None = None,
                    start_time: float | None = None, live: bool = True,
                    main_url: str | None = None,
                    audio_source: str | None = None, timings: bool = False) -> Pipeline:
        """source: the camera *name* (used in events and the index, never the URL).
        live: False for video files, so indexing processes every step instead of skipping.
        main_url: the camera's main stream, for the `recording` section (live cameras only).
        audio_source: where the `sounds` section listens (camera URL or video file)."""
        from histor.engine.store import EventStore

        store = EventStore.from_config(config.get("store"))
        search = config.get("search") or {}
        indexer = None
        if store is not None and search.get("enabled"):
            from histor.engine.indexer import SearchIndexer

            indexer = SearchIndexer.from_config(search, config.get("faces"), store,
                                                camera=str(source or "camera"), blocking=not live)
        from histor.engine.cameras import expand_env, section_enabled

        camera = str(source or "camera")
        pose = None
        if store is not None:
            if section_enabled(config.get("pose"), camera):
                from histor.engine.pose_index import PoseIndexer

                try:
                    pose = PoseIndexer.from_config(config["pose"], store, camera,
                                                   blocking=not live)
                except FileNotFoundError as e:  # record the rest anyway
                    log.warning("No pose for camera %s: %s", camera, e)
        plates = None
        if store is not None:
            if section_enabled(config.get("plates"), camera):
                from histor.engine.plate_index import PlateIndexer

                try:
                    plates = PlateIndexer.from_config(config["plates"], store, camera,
                                                      blocking=not live)
                except FileNotFoundError as e:  # record the rest anyway
                    log.warning("No plate reading for camera %s: %s", camera, e)
        sounds = None
        if section_enabled(config.get("sounds"), camera):
            if not audio_source:
                log.warning("sounds: no audio for camera %s (a webcam, or no stream URL)", source)
            else:
                from histor.engine.sound import SoundListener

                try:
                    sounds = SoundListener.from_config(config["sounds"], audio_source, live)
                except FileNotFoundError as e:
                    log.warning("No sound events for camera %s: %s", source, e)
        tamper = None
        if section_enabled(config.get("tamper"), camera):
            from histor.engine.tamper import TamperCheck

            tamper = TamperCheck.from_config(config["tamper"])
        fire = None
        if section_enabled(config.get("fire"), camera):
            from histor.engine.fire import FireWatch

            try:
                fire = FireWatch.from_config(config["fire"])
            except FileNotFoundError as e:  # record the rest anyway, but say it loudly
                log.error("NO FIRE DETECTION for camera %s: %s", camera, e)
        monitor = ZoneMonitor.from_config(config.get("zones")) if config.get("zones") else None
        abandoned = None
        if section_enabled(config.get("abandoned"), camera):
            from histor.engine.abandoned import AbandonedWatch
            from histor.engine.zones import Zone

            zones = [r for r in monitor.rules if isinstance(r, Zone)] if monitor else []
            abandoned = AbandonedWatch.from_config(config["abandoned"], zones)
        tracking = config.get("tracking") or {}
        tracker = Tracker(**{k: v for k, v in tracking.items() if k != "enabled"}) if (
            tracking.get("enabled") or config.get("zones") or indexer or pose or plates
            or abandoned
        ) else None
        vision = {k: expand_env(v) if isinstance(v, str) else v  # alerts.verify's defaults
                  for k, v in ((config.get("chat") or {}).get("vision") or {}).items()}
        alerts = AlertManager.from_config(config.get("alerts"), source=source, store=store,
                                          vision=vision, recording=config.get("recording")) if (
            monitor or (pose and pose.alert_types) or (plates and plates.watchlist) or sounds
            or tamper or abandoned or fire
        ) else None
        mainstream = None
        if live and main_url and config.get("recording"):
            from histor.engine.mainstream import MainStream

            mainstream = MainStream.from_config(config["recording"], main_url,
                                                camera=str(source or "camera"),
                                                keyframes=indexer is not None or pose is not None
                                                or plates is not None)
        change = None
        if (indexer is not None and indexer.motion is not None) or fire is not None:
            from histor.engine.change import ChangeAnalyser

            change = ChangeAnalyser.from_config(config, fire_masks=fire is not None)
        return cls(create_detector(config["model"]), tracker, monitor, alerts, store, indexer,
                   start_time, mainstream, pose, plates, sounds, tamper, abandoned, fire, timings,
                   change)

    def close(self) -> None:
        """Flush the search index and pending alerts (call when the source ends)."""
        if self.mainstream is not None:
            self.mainstream.close()
        if self.indexer is not None:
            self.indexer.close()
        if self.pose is not None:
            self.pose.close()
            self._pose_events(None, [])  # events from the last samples
        if self.plates is not None:
            self.plates.close()
            self._plate_events(None, [])
        if self.sounds is not None:
            self.sounds.close()
            self._sound_events(None, [], None)  # what the end of a file still held
        if self.alerts is not None:
            self.alerts.close()

    def process(self, frame: np.ndarray, timestamp: float) -> FrameResult:
        lap = self._lap("", 0.0)
        result = FrameResult(self.detector.detect(frame), timestamp=timestamp)
        lap = self._lap("detect", lap)
        if self.change is not None:
            result.change = self.change.update(frame, timestamp)
            lap = self._lap("change", lap)
        if self.tracker is not None:
            self._track(frame, result)
            lap = self._lap("track", lap)
        if self.tamper is not None:
            result.events += self._tamper_events(frame, result, timestamp)
            lap = self._lap("tamper", lap)
        if self.fire is not None:
            result.events += self._fire_events(frame, result.tracks, timestamp)
            lap = self._lap("fire", lap)
        if self.sounds is not None:
            result.events += self._sound_events(frame, result.tracks, timestamp)
            self._lap("sounds", lap)
        return result

    def _lap(self, stage: str, since: float) -> float:
        """With `timings` on: the time since `since` goes to `stage` (none for ""); now."""
        if self.timings is None:
            return 0.0
        now = time.perf_counter()
        if stage:
            self.timings.setdefault(stage, deque(maxlen=TIMINGS_KEPT)).append(
                (now - since) * 1000)
        return now

    def _track(self, frame: np.ndarray, result: FrameResult) -> None:
        """Tracking and everything that needs object IDs."""
        assert self.tracker is not None
        timestamp = result.timestamp
        result.tracks, removed = self.tracker.update(result.detections)
        wall = self.wall_time(timestamp)
        if self.monitor:
            h, w = frame.shape[:2]
            events = self.monitor.update(result.tracks, removed, timestamp, (w, h))
            result.events = (self.alerts.handle(events, frame, result.tracks, wall)
                             if self.alerts else events)
        hires = self._match_keyframe(frame, result) if (
            self.indexer or self.pose or self.plates) else None
        if self.indexer is not None:
            self.indexer.update(frame, result.tracks, wall, hires, result.change)
        if self.pose is not None:
            self.pose.update(frame, result.tracks, wall, hires)
            result.events += self._pose_events(frame, result.tracks)
        if self.plates is not None:
            self.plates.update(frame, result.tracks, wall, hires)
            result.events += self._plate_events(frame, result.tracks)
        if self.abandoned is not None:
            from histor.engine.abandoned import EVENT_TYPE, RULE

            found = [ZoneEvent(timestamp, EVENT_TYPE, RULE, e.track_id, e.label,
                               dwell_seconds=e.seconds)
                     for e in self.abandoned.update(result.tracks, removed, timestamp)]
            result.events += (self.alerts.handle(found, frame, result.tracks, wall)
                              if self.alerts else found)

    def _tamper_events(self, frame: np.ndarray, result: FrameResult,
                       timestamp: float) -> list[ZoneEvent]:
        """The detections tell a truck parked in front of the camera from a turned camera."""
        from histor.engine.tamper import EVENT_TYPE

        assert self.tamper is not None
        objects = [(d.label, (d.x1, d.y1, d.x2, d.y2)) for d in result.detections]
        found = [ZoneEvent(timestamp, EVENT_TYPE, e.kind, None, e.label or "camera",
                           score=e.value)
                 for e in self.tamper.update(frame, timestamp, objects, result.change)]
        if self.alerts is None or not found:
            return found
        return self.alerts.handle(found, frame, result.tracks, self.wall_time(timestamp))

    def _fire_events(self, frame: np.ndarray, tracks: list[Track],
                     timestamp: float) -> list[ZoneEvent]:
        """Fire/smoke through the alert channels; the snapshot shows the boxes that raised it."""
        from histor.engine.fire import EVENT_TYPE, snapshot_with_boxes

        assert self.fire is not None
        found = self.fire.update(frame, timestamp)
        events = [ZoneEvent(timestamp, EVENT_TYPE, e.kind, None, e.kind, score=e.score)
                  for e in found]
        if self.alerts is None or not events:
            return events
        boxed = snapshot_with_boxes(frame, self.fire.last_detections)
        return self.alerts.handle(events, boxed, tracks, self.wall_time(timestamp))

    def _sound_events(self, frame: np.ndarray | None, tracks: list[Track],
                      timestamp: float | None) -> list[ZoneEvent]:
        """Sound events through the alert channels; from a file, each when the video reaches it
        (so the snapshot shows that moment). timestamp None: all that are left."""
        from histor.engine.sound import EVENT_TYPE

        assert self.sounds is not None
        live = self.sounds.live
        found = self.sounds.take_events(None if live or timestamp is None else timestamp)
        events = [ZoneEvent((e.time - self.start_time) if live else e.time, EVENT_TYPE, e.group,
                            None, e.label, score=e.score) for e in found]
        if self.alerts is None:
            return events
        return [x for event in events for x in self.alerts.handle(
            [event], frame, tracks, self.start_time + event.timestamp)]

    def _plate_events(self, frame: np.ndarray | None, tracks: list[Track]) -> list[ZoneEvent]:
        """Watchlist plates through the alert channels and the event log."""
        from histor.engine.plate_index import EVENT_TYPE

        assert self.plates is not None
        events = [ZoneEvent(wall - self.start_time, EVENT_TYPE, rule, track_id, label,
                            plate=text)
                  for wall, track_id, rule, label, text in self.plates.take_events()]
        if self.alerts is None:
            return events
        return [e for event in events for e in self.alerts.handle(
            [event], frame, tracks, self.start_time + event.timestamp)]

    def _pose_events(self, frame: np.ndarray | None, tracks: list[Track]) -> list[ZoneEvent]:
        """Pose events (person down, hands up) through the alert channels and the event log."""
        from histor.engine.pose_index import EVENT_RULES

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
        from histor.engine.indexer import HiResFrame

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
            for ti, di in greedy_match(iou, KEYFRAME_MIN_IOU):
                boxes[tracks[ti].id] = hires[di]
        return HiResFrame(keyframe.time, keyframe.image, boxes)

    def draw(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:
        if self.fire is not None:
            from histor.engine.fire import draw_fire

            draw_fire(frame, self.fire.boxes(result.timestamp))
        if self.tracker is None:
            return draw_detections(frame, result.detections)
        visible = [t for t in result.tracks if not t.missed]
        if self.pose is not None:
            from histor.engine.pose import draw_skeleton, unpack_keypoints

            skeletons = self.pose.overlay(self.wall_time(result.timestamp))
            for t in visible:
                if t.id in skeletons:
                    draw_skeleton(frame, unpack_keypoints(skeletons[t.id], t.box),
                                  self.pose.min_confidence)
        out = (self.monitor or ZoneMonitor([])).draw(frame, visible, result.timestamp)
        if self.plates is not None:  # the voted plate under each vehicle's box
            from histor.engine.detector import draw_label

            plates = self.plates.plates(self.wall_time(result.timestamp))
            for t in visible:
                if t.id in plates:
                    draw_label(out, plates[t.id], (int(t.box[0]), int(t.box[3]) + 18),
                               (255, 255, 255))
        return out
