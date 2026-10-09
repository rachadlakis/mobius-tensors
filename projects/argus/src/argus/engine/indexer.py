"""Build the search index while recording: what the camera saw, as vectors in the store.

Per camera, on a background thread (detection never waits for it on live cameras):
  * objects - each tracked person/vehicle is embedded with CLIP when it appears and then at
              most every `object_interval_seconds` (so one person is not stored 300 times)
  * scenes  - every `scene_interval_seconds` the whole frame plus a grid of tiles, so things
              the detector ignores (a box on the floor, an open gate, smoke) are searchable too
  * faces   - only if enabled for this camera: face prints of tracked people, checked every
              `face_interval_seconds`; blurred / tiny / side-on faces are skipped, and only a
              track's best `max_faces_per_track` are kept (an hour at a desk is not 3600 rows)

Without the CLIP image model, tracked objects are still stored (thumbnail, time, label: what
the chat's `objects` tool and face search need) but not embedded, and scenes are skipped.

With a main stream (mainstream.py), crops come from its full-resolution keyframes when the
object was matched in one (`HiResFrame`); otherwise from the sub-stream frame, so an object
that passes between two keyframes is still indexed. A track embedded once in hi-res is then
re-embedded from keyframes only.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    from argus.engine.store import EventStore
    from argus.engine.tracker import Track

log = logging.getLogger(__name__)
PEOPLE_AND_VEHICLES = ["person", "car", "truck", "bus", "motorcycle", "bicycle"]


class ImageEmbedder(Protocol):
    def embed_images(self, images: list[np.ndarray]) -> np.ndarray: ...


class FaceModel(Protocol):
    def detect(self, image: np.ndarray) -> list[Any]: ...
    def analyze(self, image: np.ndarray, face: Any) -> tuple[np.ndarray, Any]: ...
    def problems(self, quality: Any) -> list[str]: ...
    def crop(self, image: np.ndarray, face: Any) -> np.ndarray: ...


@dataclass
class HiResFrame:
    """A main-stream keyframe and the tracks found in it (boxes in keyframe pixels)."""

    time: float
    image: np.ndarray
    boxes: dict[int, np.ndarray]


@dataclass
class _Job:
    frame: np.ndarray
    time: float
    objects: list[tuple[int, str, np.ndarray, np.ndarray | None]]  # id, label, box, hi-res box
    faces: list[tuple[int, np.ndarray, bool]]  # person track id, box, box is hi-res
    scene: bool
    hires: np.ndarray | None = None  # keyframe the hi-res boxes refer to
    scene_image: np.ndarray | None = None  # frame for the scene + tiles (None: `frame`)


class SearchIndexer:
    def __init__(
        self,
        store: EventStore,
        camera: str,
        embedder: ImageEmbedder | None,
        faces: FaceModel | None = None,
        classes: list[str] | None = None,
        object_interval_seconds: float = 2.0,
        scene_interval_seconds: float = 5.0,
        scene_tiles: int = 2,
        min_box_height: int = 40,
        face_interval_seconds: float = 1.0,
        max_faces_per_track: int = 5,
        blocking: bool = False,
        max_queue: int = 4,
        keyframe_max_age: float = 3.0,
    ) -> None:
        """
        scene_tiles: n -> the frame is also embedded as an n x n grid (0 = whole frame only).
        blocking: wait when the worker is busy (video files: index everything) instead of
                  dropping the job (live cameras: stay real-time).
        keyframe_max_age: scenes use the latest main-stream keyframe if it is this recent.
        """
        self.store = store
        self.camera = camera
        self.embedder = embedder
        self.faces = faces
        self.classes = set(classes or PEOPLE_AND_VEHICLES)
        self.object_interval = object_interval_seconds
        self.scene_interval = scene_interval_seconds
        self.scene_tiles = scene_tiles
        self.min_box_height = min_box_height
        self.face_interval = face_interval_seconds
        self.max_faces_per_track = max_faces_per_track
        self.blocking = blocking
        self.keyframe_max_age = keyframe_max_age
        self._keyframe: HiResFrame | None = None
        self._hires_tracks: set[int] = set()  # tracks seen in a keyframe at least once
        self._last_object: dict[int, float] = {}
        self._last_face: dict[int, float] = {}
        self._last_scene = -1e18
        # worker thread only: per track, the kept faces as (quality, row id), best first
        self._best_faces: dict[int, list[tuple[float, int]]] = {}
        self._face_seen: dict[int, float] = {}
        self.dropped = 0
        self.faces_skipped = 0  # faces too blurred / small / turned away to store
        self._queue: queue.Queue[_Job | None] = queue.Queue(max_queue)
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"index-{camera}")
        self._thread.start()

    @classmethod
    def from_config(cls, config: dict[str, Any], faces_config: dict[str, Any] | None,
                    store: EventStore, camera: str, blocking: bool) -> SearchIndexer:
        from argus.engine.clip_onnx import ClipEncoder

        clip_cfg = dict(config.get("clip") or {})
        image_model = clip_cfg.get("image", "models/clip_image.onnx")
        embedder = None
        if Path(image_model).exists():
            embedder = ClipEncoder(image=image_model, text=None,
                                   num_threads=clip_cfg.get("num_threads", 0),
                                   device=clip_cfg.get("device", "auto"))
        else:
            log.warning("%s not found: no visual search for camera %s (objects and faces are "
                        "still recorded). Export it: uv run --group export -m argus.engine.export_clip",
                        image_model, camera)
        face_model = None
        if faces_enabled(faces_config, camera):
            from argus.engine.faces import FaceEngine

            face_model = FaceEngine.from_config(faces_config or {})
            log.info("Face indexing ON for camera %s", camera)
        options = {k: v for k, v in config.items() if k not in {"enabled", "clip"}}
        for key, option in (("interval_seconds", "face_interval_seconds"),
                            ("max_per_track", "max_faces_per_track")):
            if faces_config and key in faces_config:
                options[option] = faces_config[key]
        return cls(store, camera, embedder, face_model, blocking=blocking, **options)

    # ------------------------------------------------------------ called per frame

    def update(self, frame: np.ndarray, tracks: list[Track], t: float,
               hires: HiResFrame | None = None) -> None:
        """hires: a main-stream keyframe that arrived since the last call, if any."""
        if hires is not None:
            self._keyframe = hires
        objects, faces = [], []
        for track in tracks:
            if track.missed or track.label not in self.classes:
                continue
            box = np.asarray(track.box, dtype=float)
            hbox = hires.boxes.get(track.id) if hires is not None else None
            sub_ok = box[3] - box[1] >= self.min_box_height
            hi_ok = hbox is not None and hbox[3] - hbox[1] >= self.min_box_height
            if not (sub_ok or hi_ok):  # a far object can be big enough in the main stream only
                continue
            if self._due(self._last_object, track.id, t, self.object_interval, hi_ok):
                self._last_object[track.id] = t
                objects.append((track.id, track.label, box, hbox if hi_ok else None))
            if self.faces is not None and track.label == "person" and self._due(
                    self._last_face, track.id, t, self.face_interval, hi_ok):
                self._last_face[track.id] = t
                faces.append((track.id, hbox, True) if hi_ok else (track.id, box, False))
            if hi_ok:
                self._hires_tracks.add(track.id)
        if len(self._last_object) > 5000:  # forget objects not seen for 10 minutes
            self._last_object = {k: v for k, v in self._last_object.items() if t - v < 600}
            self._last_face = {k: v for k, v in self._last_face.items() if t - v < 600}
            self._hires_tracks &= set(self._last_object)
        scene = self.embedder is not None and t - self._last_scene >= self.scene_interval
        if scene:
            self._last_scene = t
        if not (objects or faces or scene):
            return
        keyframe = self._keyframe
        fresh = keyframe is not None and abs(t - keyframe.time) <= self.keyframe_max_age
        job = _Job(frame.copy(), t, objects, faces, scene,
                   hires=hires.image if hires is not None else None,
                   scene_image=keyframe.image if scene and fresh else None)
        if self.blocking:
            self._queue.put(job)
            return
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            self.dropped += 1  # live: better to skip an index step than to lag behind

    def _due(self, last: dict[int, float], track_id: int, t: float, interval: float,
             hi_ok: bool) -> bool:
        """Before any keyframe match: every `interval` from whatever frame we have, plus right
        away on the first match (upgrade to hi-res). After it: from keyframes only, unless
        none matched for 3 intervals (e.g. the object now moves too fast to be matched)."""
        elapsed = t - last.get(track_id, -1e18)
        if track_id not in self._hires_tracks:
            return elapsed >= interval or hi_ok
        return (hi_ok and elapsed >= interval) or elapsed >= 3 * interval

    def close(self, timeout: float = 120.0) -> None:
        """Finish queued work (e.g. the end of a video file)."""
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
            except Exception:  # never let one bad frame stop indexing
                log.exception("Indexing failed for camera %s", self.camera)

    def _process(self, job: _Job) -> None:
        frame, t = job.frame, job.time
        h, w = frame.shape[:2]
        images, rows = [], []
        stamp = int(t * 1000)

        for track_id, label, box, hbox in job.objects:
            if hbox is not None and job.hires is not None:
                crop = _crop(job.hires, hbox, pad=0.1)
            else:
                crop = _crop(frame, box, pad=0.1)
            if crop is None:
                continue
            images.append(crop)
            thumb = self.store.save_thumb(crop, "thumbs", f"{self.camera}_{stamp}_{track_id}", t)
            rows.append({"camera": self.camera, "time": t, "kind": "object",
                         "track_id": track_id, "label": label, "box": box, "thumb": thumb})

        if job.scene:
            if job.scene_image is not None:  # main-stream keyframe: tiles keep their detail
                frame = job.scene_image
                h, w = frame.shape[:2]
            thumb = self.store.save_image(frame, "scenes", f"{self.camera}_{stamp}", t,
                                          max_height=360, quality=75)
            regions = [np.array([0, 0, w, h], dtype=float)]
            n = self.scene_tiles
            for i in range(n):
                for j in range(n):  # tiles overlap by 25% so objects on a border are not cut
                    tw, th = w / n, h / n
                    x1, y1 = max(0, j * tw - tw * 0.25), max(0, i * th - th * 0.25)
                    x2, y2 = min(w, (j + 1) * tw + tw * 0.25), min(h, (i + 1) * th + th * 0.25)
                    regions.append(np.array([x1, y1, x2, y2]))
            for k, box in enumerate(regions):
                crop = _crop(frame, box)
                if crop is None:
                    continue
                images.append(crop)
                rows.append({"camera": self.camera, "time": t, "kind": "scene" if k == 0 else "tile",
                             "box": box, "thumb": thumb})

        if rows:
            if self.embedder is not None:
                vectors = self.embedder.embed_images(images)
                model = getattr(self.embedder, "model", None)
                for row, vector in zip(rows, vectors):
                    row["embedding"] = vector
                    row["model"] = model
            self.store.add_sightings(rows)

        if self.faces is not None and job.faces:
            for track_id, box, is_hires in job.faces:
                source = job.hires if is_hires and job.hires is not None else job.frame
                person = _crop(source, box, pad=0.05)
                if person is None:
                    continue
                upper = person[: max(1, int(person.shape[0] * 0.6))]  # faces are in the top part
                found = self.faces.detect(upper)
                if not found:
                    continue
                face = found[0]
                vector, quality = self.faces.analyze(upper, face)
                if self.faces.problems(quality):
                    self.faces_skipped += 1  # matches everyone a little: worse than nothing
                    continue
                self._keep_face(track_id, t, float(quality.value), vector,
                                self.faces.crop(upper, face), stamp)
            if len(self._face_seen) > 500:  # forget tracks not seen for 10 minutes
                gone = [k for k, v in self._face_seen.items() if t - v > 600]
                for k in gone:
                    del self._face_seen[k], self._best_faces[k]

    def _keep_face(self, track_id: int, t: float, quality: float, vector: np.ndarray,
                   crop: np.ndarray, stamp: int) -> None:
        """Store the face if it is among the track's best `max_faces_per_track`, and delete the
        one it pushes out."""
        kept = self._best_faces.setdefault(track_id, [])
        self._face_seen[track_id] = t
        if len(kept) >= self.max_faces_per_track and quality <= kept[-1][0]:
            return
        thumb = self.store.save_thumb(crop, "faces", f"{self.camera}_{stamp}_{track_id}", t)
        [row_id] = self.store.add_faces([{"camera": self.camera, "time": t, "track_id": track_id,
                                          "quality": quality, "thumb": thumb,
                                          "embedding": vector}])
        kept.append((quality, row_id))
        kept.sort(key=lambda k: -k[0])
        if len(kept) > self.max_faces_per_track:
            self.store.delete_faces([kept.pop()[1]])


def faces_enabled(config: dict[str, Any] | None, camera: str) -> bool:
    """Faces are off unless `faces.enabled: true`; `faces.cameras` narrows it to some cameras."""
    if not config or not config.get("enabled"):
        return False
    cameras = config.get("cameras")
    return not cameras or camera in cameras


def _crop(frame: np.ndarray, box: np.ndarray, pad: float = 0.0) -> np.ndarray | None:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box)
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1 = int(max(0, x1 - px)), int(max(0, y1 - py))
    x2, y2 = int(min(w, x2 + px)), int(min(h, y2 + py))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return frame[y1:y2, x1:x2]
