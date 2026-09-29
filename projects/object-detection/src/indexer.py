"""Build the search index while recording: what the camera saw, as vectors in the store.

Per camera, on a background thread (detection never waits for it on live cameras):
  * objects - each tracked person/vehicle is embedded with CLIP when it appears and then at
              most every `object_interval_seconds` (so one person is not stored 300 times)
  * scenes  - every `scene_interval_seconds` the whole frame plus a grid of tiles, so things
              the detector ignores (a box on the floor, an open gate, smoke) are searchable too
  * faces   - only if enabled for this camera: face prints of tracked people
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    from store import EventStore
    from tracker import Track

log = logging.getLogger(__name__)
PEOPLE_AND_VEHICLES = ["person", "car", "truck", "bus", "motorcycle", "bicycle"]


class ImageEmbedder(Protocol):
    def embed_images(self, images: list[np.ndarray]) -> np.ndarray: ...


class FaceModel(Protocol):
    def detect(self, image: np.ndarray) -> list[Any]: ...
    def embed(self, image: np.ndarray, face: Any) -> np.ndarray: ...
    def crop(self, image: np.ndarray, face: Any) -> np.ndarray: ...


@dataclass
class _Job:
    frame: np.ndarray
    time: float
    objects: list[tuple[int, str, np.ndarray]]  # track id, label, box
    faces: list[tuple[int, np.ndarray]]  # person track id, box
    scene: bool


class SearchIndexer:
    def __init__(
        self,
        store: EventStore,
        camera: str,
        embedder: ImageEmbedder,
        faces: FaceModel | None = None,
        classes: list[str] | None = None,
        object_interval_seconds: float = 2.0,
        scene_interval_seconds: float = 5.0,
        scene_tiles: int = 2,
        min_box_height: int = 40,
        face_interval_seconds: float = 1.0,
        blocking: bool = False,
        max_queue: int = 4,
    ) -> None:
        """
        scene_tiles: n -> the frame is also embedded as an n x n grid (0 = whole frame only).
        blocking: wait when the worker is busy (video files: index everything) instead of
                  dropping the job (live cameras: stay real-time).
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
        self.blocking = blocking
        self._last_object: dict[int, float] = {}
        self._last_face: dict[int, float] = {}
        self._last_scene = -1e18
        self.dropped = 0
        self._queue: queue.Queue[_Job | None] = queue.Queue(max_queue)
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"index-{camera}")
        self._thread.start()

    @classmethod
    def from_config(cls, config: dict[str, Any], faces_config: dict[str, Any] | None,
                    store: EventStore, camera: str, blocking: bool) -> SearchIndexer:
        from clip_onnx import ClipEncoder

        clip_cfg = dict(config.get("clip") or {})
        embedder = ClipEncoder(image=clip_cfg.get("image", "models/clip_image.onnx"), text=None,
                               num_threads=clip_cfg.get("num_threads", 0))
        face_model = None
        if faces_enabled(faces_config, camera):
            from faces import FaceEngine

            face_model = FaceEngine(**{k: v for k, v in (faces_config or {}).items()
                                       if k in {"detector", "recognizer", "score_threshold",
                                                "min_face_size"}})
            log.info("Face indexing ON for camera %s", camera)
        options = {k: v for k, v in config.items() if k not in {"enabled", "clip"}}
        if faces_config and "interval_seconds" in faces_config:
            options["face_interval_seconds"] = faces_config["interval_seconds"]
        return cls(store, camera, embedder, face_model, blocking=blocking, **options)

    # ------------------------------------------------------------ called per frame

    def update(self, frame: np.ndarray, tracks: list[Track], t: float) -> None:
        objects, faces = [], []
        for track in tracks:
            if track.missed or track.label not in self.classes:
                continue
            box = np.asarray(track.box, dtype=float)
            if box[3] - box[1] < self.min_box_height:
                continue
            if t - self._last_object.get(track.id, -1e18) >= self.object_interval:
                self._last_object[track.id] = t
                objects.append((track.id, track.label, box))
            if (self.faces is not None and track.label == "person"
                    and t - self._last_face.get(track.id, -1e18) >= self.face_interval):
                self._last_face[track.id] = t
                faces.append((track.id, box))
        if len(self._last_object) > 5000:  # forget objects not seen for 10 minutes
            self._last_object = {k: v for k, v in self._last_object.items() if t - v < 600}
            self._last_face = {k: v for k, v in self._last_face.items() if t - v < 600}
        scene = t - self._last_scene >= self.scene_interval
        if scene:
            self._last_scene = t
        if not (objects or faces or scene):
            return
        job = _Job(frame.copy(), t, objects, faces, scene)
        if self.blocking:
            self._queue.put(job)
            return
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            self.dropped += 1  # live: better to skip an index step than to lag behind

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

        for track_id, label, box in job.objects:
            crop = _crop(frame, box, pad=0.1)
            if crop is None:
                continue
            images.append(crop)
            thumb = self.store.save_thumb(crop, "thumbs", f"{self.camera}_{stamp}_{track_id}", t)
            rows.append({"camera": self.camera, "time": t, "kind": "object",
                         "track_id": track_id, "label": label, "box": box, "thumb": thumb})

        if job.scene:
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

        if images:
            vectors = self.embedder.embed_images(images)
            for row, vector in zip(rows, vectors):
                row["embedding"] = vector
            self.store.add_sightings(rows)

        if self.faces is not None and job.faces:
            face_rows = []
            for track_id, box in job.faces:
                person = _crop(frame, box, pad=0.05)
                if person is None:
                    continue
                upper = person[: max(1, int(person.shape[0] * 0.6))]  # faces are in the top part
                found = self.faces.detect(upper)
                if not found:
                    continue
                face = found[0]
                vector = self.faces.embed(upper, face)
                thumb = self.store.save_thumb(self.faces.crop(upper, face), "faces",
                                              f"{self.camera}_{stamp}_{track_id}", t)
                face_rows.append({"camera": self.camera, "time": t, "track_id": track_id,
                                  "quality": float(face.score) * min(1.0, face.size / 80),
                                  "thumb": thumb, "embedding": vector})
            self.store.add_faces(face_rows)


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
