"""Read number plates while recording: each vehicle's plate, as searchable text in the store.

Per camera, on its own background thread (like poses, so it never holds up detection):

  * sampling   each tracked vehicle (plates.VEHICLES) about every `interval_seconds`, from the
               main-stream keyframe when the vehicle was matched in one: plates are small, and
               the sub stream rarely has enough pixels to read them
  * finding    the plate detector runs on the vehicle's crop (a little padded), not the frame;
               the best plate box is read if it is at least `min_plate_width` pixels wide
  * reading    a read counts when every character is at least `min_confidence` sure
  * voting     a vehicle's plate is the text with the highest summed confidence over its reads,
               so one bad frame doesn't rename it
  * storing    a row per new text read on a vehicle, or a clearly better read of the same text,
               at most `max_per_track` rows per vehicle, each with the plate crop as thumbnail
  * watchlist  `plate` event (rule = the entry's name) once per vehicle when its voted plate,
               read at least `min_reads` times, is within `max_distance` edits of a listed plate
               (look-alike characters forgiven: plates.plate_distance)
"""

from __future__ import annotations

import logging
import queue
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from argus.engine.plates import (VEHICLES, PlateDetector, PlateRead, PlateReader, crop_box,
                                 normalize_plate, plate_distance)

if TYPE_CHECKING:
    from argus.engine.indexer import HiResFrame
    from argus.engine.store import EventStore
    from argus.engine.tracker import Track

log = logging.getLogger(__name__)
EVENT_TYPE = "plate"


@dataclass
class _Job:
    frame: np.ndarray
    time: float
    vehicles: list[tuple[int, str, np.ndarray, np.ndarray | None]]  # id, label, box, hi-res box
    hires: np.ndarray | None


@dataclass
class _TrackState:
    label: str
    votes: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    reads: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    best: dict[str, float] = field(default_factory=dict)  # stored text -> its best confidence
    rows: int = 0
    seen: float = 0.0
    fired: set[str] = field(default_factory=set)

    def voted(self) -> tuple[str | None, int]:
        if not self.votes:
            return None, 0
        text = max(self.votes, key=lambda k: self.votes[k])
        return text, self.reads[text]


def parse_watchlist(entries: Any) -> list[tuple[str, str]]:
    """[(plate, rule name)] from a list of {plate, name} / plain strings, or a {plate: name}
    mapping."""
    if not entries:
        return []
    if isinstance(entries, dict):
        entries = [{"plate": k, "name": v} for k, v in entries.items()]
    out = []
    for e in entries:
        plate, name = (e, None) if isinstance(e, str) else (e.get("plate"), e.get("name"))
        text = normalize_plate(plate or "")
        if not text:
            raise ValueError(f"plates.watchlist: entry {e!r} has no plate")
        out.append((text, str(name or "watchlist")))
    return out


class PlateIndexer:
    def __init__(
        self,
        store: EventStore,
        camera: str,
        detector: Any,
        reader: Any,
        interval_seconds: float = 1.0,
        min_vehicle_height: int = 40,
        min_plate_width: int = 40,
        min_confidence: float = 0.6,
        max_per_track: int = 5,
        watchlist: Any = None,
        max_distance: int = 1,
        min_reads: int = 2,
        blocking: bool = False,
        max_queue: int = 2,
    ) -> None:
        """
        detector: anything with detect(image) -> [(x1, y1, x2, y2, score)], best first.
        reader: anything with read(crops) -> [PlateRead].
        min_plate_width: narrower plates (pixels, in the stream they are cut from) are not read.
        """
        self.store = store
        self.camera = camera
        self.detector = detector
        self.reader = reader
        self.interval = interval_seconds
        self.min_vehicle_height = min_vehicle_height
        self.min_plate_width = min_plate_width
        self.min_confidence = min_confidence
        self.max_per_track = max_per_track
        self.watchlist = parse_watchlist(watchlist)
        self.max_distance = max_distance
        self.min_reads = max(1, int(min_reads))
        self.blocking = blocking
        self.dropped = 0
        self.stored = 0
        self._last: dict[int, float] = {}  # main thread: when each vehicle was last sampled
        self._state: dict[int, _TrackState] = {}  # worker thread
        self._lock = threading.Lock()
        self._events: list[tuple[float, int, str, str, str]] = []
        self._latest: dict[int, tuple[float, str]] = {}  # track -> (time, voted plate): overlay
        self._queue: queue.Queue[_Job | None] = queue.Queue(max_queue)
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"plates-{camera}")
        self._thread.start()

    @classmethod
    def from_config(cls, config: dict[str, Any], store: EventStore, camera: str,
                    blocking: bool) -> PlateIndexer:
        """config: the `plates` section."""
        det_cfg = dict(config.get("detector") or {})
        ocr_cfg = dict(config.get("reader") or {})
        device = config.get("device", "auto")
        detector = PlateDetector(**{"device": device, **det_cfg})
        reader = PlateReader(**{"device": device, **ocr_cfg})
        options = {k: v for k, v in config.items() if k not in {
            "enabled", "cameras", "detector", "reader", "device"}}
        return cls(store, camera, detector, reader, blocking=blocking, **options)

    # ------------------------------------------------------------ main thread

    def update(self, frame: np.ndarray, tracks: list[Track], t: float,
               hires: HiResFrame | None = None) -> None:
        vehicles = []
        for track in tracks:
            if track.missed or track.label not in VEHICLES:
                continue
            box = np.asarray(track.box, dtype=float)
            hbox = hires.boxes.get(track.id) if hires is not None else None
            if hbox is None and box[3] - box[1] < self.min_vehicle_height:
                continue
            last = self._last.get(track.id)
            if last is not None and t - last < self.interval:
                continue
            self._last[track.id] = t
            vehicles.append((track.id, str(track.label), box, hbox))
        if len(self._last) > 2000:  # forget vehicles not seen for 10 minutes
            self._last = {k: v for k, v in self._last.items() if t - v < 600}
        if not vehicles:
            return
        job = _Job(frame.copy(), t, vehicles, hires.image if hires is not None else None)
        if self.blocking:
            self._queue.put(job)
            return
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            self.dropped += 1  # live: skip a sample rather than fall behind

    def take_events(self) -> list[tuple[float, int, str, str, str]]:
        """Watchlist hits since the last call: (unix time, track id, rule, vehicle label, plate)."""
        with self._lock:
            events, self._events = self._events, []
        return events

    def plates(self, t: float, max_age: float = 5.0) -> dict[int, str]:
        """Voted plate per vehicle track, read in the last `max_age` seconds (for the live view)."""
        with self._lock:
            return {k: text for k, (when, text) in self._latest.items() if t - when <= max_age}

    def close(self, timeout: float = 120.0) -> None:
        self._queue.put(None)
        self._thread.join(timeout=timeout)

    # ------------------------------------------------------------ worker

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            try:
                self._process(job)
            except Exception:  # never let one bad frame stop plate reading
                log.exception("Plate reading failed for camera %s", self.camera)

    def _process(self, job: _Job) -> None:
        t = job.time
        found: list[tuple[int, str, np.ndarray, np.ndarray]] = []  # id, label, frame box, crop
        for track_id, label, box, hbox in job.vehicles:
            source, vbox = (job.hires, hbox) if hbox is not None and job.hires is not None \
                else (job.frame, box)
            vehicle = crop_box(source, vbox, pad=0.05)
            if vehicle is None:
                continue
            boxes = self.detector.detect(vehicle)
            if not boxes:
                continue
            x1, y1, x2, y2, _ = boxes[0]
            if x2 - x1 < self.min_plate_width:
                continue
            plate = crop_box(vehicle, (x1, y1, x2, y2), pad=0.06)
            if plate is not None:
                found.append((track_id, label, box, plate))
        if not found:
            return
        reads = self.reader.read([p for *_, p in found])
        rows = []
        for (track_id, label, box, plate), read in zip(found, reads):
            row = self._vote(track_id, label, t, read, box, plate)
            if row is not None:
                rows.append(row)
        if rows:
            self.store.add_plates(rows)
            self.stored += len(rows)
        if len(self._state) > 500:
            gone = [k for k, s in self._state.items() if t - s.seen > 600]
            with self._lock:
                for k in gone:
                    del self._state[k]
                    self._latest.pop(k, None)

    def _vote(self, track_id: int, label: str, t: float, read: PlateRead, box: np.ndarray,
              plate: np.ndarray) -> dict[str, Any] | None:
        """Count a read; return a row to store if it is new or clearly better."""
        if not read.text or read.confidence < self.min_confidence:
            return None
        state = self._state.setdefault(track_id, _TrackState(label))
        state.seen = t
        state.votes[read.text] += read.confidence
        state.reads[read.text] += 1
        voted, support = state.voted()
        assert voted is not None
        with self._lock:
            self._latest[track_id] = (t, voted)
        self._check_watchlist(track_id, state, voted, support, t)
        best = state.best.get(read.text)
        if state.rows >= self.max_per_track or (best is not None and read.confidence < best + 0.05):
            return None
        state.best[read.text] = read.confidence
        state.rows += 1
        stamp = f"{int(t * 1000)}"
        thumb = self.store.save_image(plate, "plates", f"{self.camera}_{stamp}_{track_id}", t,
                                      quality=90)
        return {"camera": self.camera, "time": t, "track_id": track_id, "label": label,
                "text": read.text, "confidence": round(read.confidence, 4),
                "region": read.region, "box": box, "thumb": thumb}

    def _check_watchlist(self, track_id: int, state: _TrackState, voted: str, support: int,
                         t: float) -> None:
        if support < self.min_reads:
            return
        for plate, rule in self.watchlist:
            if rule in state.fired or plate_distance(voted, plate) > self.max_distance:
                continue
            state.fired.add(rule)
            with self._lock:
                self._events.append((t, track_id, rule, state.label, voted))
