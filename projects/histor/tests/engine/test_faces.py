"""Face prints: quality checks, choosing the face in a photo, keeping a track's best faces,
scoring visits, and camera aliases. Fake vectors; the real models only in the last test."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from histor.engine.cameras import camera_about, camera_aliases
from histor.engine.faces import Face, FaceEngine, FaceQuality, PhotoFace, choose_face
from histor.engine.indexer import SearchIndexer
from histor.engine.query import parse_query
from histor.engine.search import SearchEngine, face_answer, group_face_visits
from histor.engine.store import EventStore
from histor.engine.tracker import Track

ROOT = Path(__file__).resolve().parents[2]


def unit(*values: float) -> np.ndarray:
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)


def at(cos: float) -> np.ndarray:
    """A 2-d unit vector whose cosine with unit(1, 0) is `cos`."""
    return unit(cos, float(np.sqrt(1 - cos * cos)))


@pytest.fixture
def store(tmp_path):
    s = EventStore(tmp_path / "db.sqlite")
    yield s
    s.close()


# ------------------------------------------------------------------ quality

def face_with_nose(nose_x: float) -> Face:
    row = np.zeros(15, np.float32)
    row[:4] = [30, 30, 40, 40]
    row[4:10] = [40, 45, 60, 45, nose_x, 55]  # right eye, left eye, nose tip
    return Face((30, 30, 40, 40), 0.9, row)


def test_turn_and_quality_value():
    assert face_with_nose(50).turn == 0
    assert face_with_nose(58).turn == pytest.approx(0.4)
    good = FaceQuality(score=0.9, size=80, turn=0.1, sharpness=150)
    assert good.value > FaceQuality(0.9, 80, 0.5, 150).value  # turned
    assert good.value > FaceQuality(0.9, 40, 0.1, 150).value  # small
    assert good.value > FaceQuality(0.9, 80, 0.1, 20).value   # soft


def photo_face(size: float, problems: list[str] | None = None) -> PhotoFace:
    return PhotoFace(None if problems else unit(1, 0), (0, 0, size, size),
                     FaceQuality(0.9, size, 0.1, 80), problems or [])


def test_choose_face_in_photo():
    assert choose_face([]) == "No face found in the photo."
    one = photo_face(60)
    assert choose_face([one]) is one
    two = [photo_face(60), photo_face(50)]
    assert "say which one" in choose_face(two)
    assert choose_face(two, choice=2) is two[1]
    assert "pick 1 to 2" in choose_face(two, choice=3)
    subject = [photo_face(25), photo_face(90)]  # a big face and one far behind it
    assert choose_face(subject) is subject[1]
    msg = choose_face([photo_face(30, ["too small (30 px, needs 40)"])])
    assert msg.startswith("The face in the photo can't be used: too small")


# ------------------------------------------------------------------ store

def test_store_face_helpers(store):
    t0 = datetime(2026, 10, 7, 13).timestamp()
    thumb = store.save_thumb(np.zeros((40, 40, 3), np.uint8), "faces", "x", t0)
    ids = store.add_faces([
        {"camera": "office", "time": t0, "track_id": 1, "thumb": thumb, "embedding": unit(1, 0)},
        {"camera": "office", "time": t0 + 5, "track_id": 1, "embedding": unit(0, 1)},
        {"camera": "lobby", "time": t0 + 9000, "track_id": 2, "embedding": unit(0, 1)},
    ])
    assert len(ids) == 3 and ids[0] < ids[1]
    # several reference photos: each row scores its best match among them
    hits = store.search("faces", np.stack([unit(1, 0), unit(0, 1)]), top_k=None)
    assert len(hits) == 3 and all(h.score > 0.99 for h in hits)
    assert store.face_counts(t0, t0 + 60) == {"office": 2}
    assert store.face_counts(cameras=["lobby"]) == {"lobby": 1}
    store.delete_faces([ids[0]])
    assert store.media_path(thumb) is None and store.counts()["faces"] == 2


# ------------------------------------------------------------------ indexer

class FakeEmbedder:
    def embed_images(self, images):
        return np.stack([unit(1, 1) for _ in images])


class FakeFaceModel:
    """Faces whose quality follows a script; quality < 0 means "blurred"."""

    def __init__(self, qualities):
        self.qualities = list(qualities)

    def detect(self, image):
        return ["face"]

    def analyze(self, image, face):
        q = self.qualities.pop(0)
        return unit(1, 0), SimpleNamespace(value=abs(q), blurred=q < 0)

    def problems(self, quality):
        return ["blurred"] if quality.blurred else []

    def crop(self, image, face):
        return image


def test_indexer_keeps_best_faces_per_track(store):
    qualities = [0.2, 0.5, 0.1, 0.9, 0.3, 0.8, -0.95, 0.4, 0.6, 0.7]
    idx = SearchIndexer(store, "office", FakeEmbedder(), FakeFaceModel(qualities),
                        face_interval_seconds=1, max_faces_per_track=3, scene_interval_seconds=1e9,
                        blocking=True)
    frame = np.full((240, 320, 3), 100, np.uint8)
    for i in range(len(qualities)):
        person = Track(id=1, box=np.array([10, 10, 60, 120], float), score=0.9,
                       label_votes={"person": 1.0})
        idx.update(frame, [person], 1000.0 + i)
    idx.close()
    kept = store._db.execute("SELECT quality, thumb FROM faces ORDER BY quality DESC").fetchall()
    assert [round(q, 2) for q, _ in kept] == [0.9, 0.8, 0.7]  # the blurred 0.95 is not stored
    assert idx.faces_skipped == 1
    on_disk = sorted(p.relative_to(store.media_dir).as_posix()
                     for p in (store.media_dir / "faces").rglob("*.jpg"))
    assert on_disk == sorted(t for _, t in kept)  # pushed-out faces lose their thumbnail too


# ------------------------------------------------------------------ search

def test_face_visits_levels_and_whole_track(store):
    t0 = datetime(2026, 10, 7, 13, 40).timestamp()
    store.add_sightings([{"camera": "office", "time": t0 + s, "kind": "object", "track_id": 7,
                          "label": "person", "embedding": unit(1, 1)} for s in (0, 150, 300)])
    store.add_faces([
        {"camera": "office", "time": t0 + 240, "track_id": 7, "embedding": at(0.8)},
        {"camera": "office", "time": t0 + 250, "track_id": 7, "embedding": at(0.6)},
        {"camera": "office", "time": t0 + 260, "track_id": 7, "embedding": at(0.1)},
        {"camera": "lobby", "time": t0 + 30, "track_id": 8, "embedding": at(0.4)},
        {"camera": "lobby", "time": t0 + 60, "track_id": 9, "embedding": at(0.2)},
        # recorder restarted: track id 7 again, two hours later, someone else's visit
        {"camera": "office", "time": t0 + 7200, "track_id": 7, "embedding": at(0.9)},
    ])
    engine = SearchEngine(store)
    visits = engine.face_search([unit(1, 0)])
    summary = [(v.camera, v.track_id, v.level, round(v.score, 2)) for v in visits]
    assert summary == [("office", 7, "likely", 0.9), ("office", 7, "likely", 0.5),
                       ("lobby", 8, "possible", 0.4)]  # track 9 (0.2) is not a match
    visit = visits[1]
    assert (visit.frames, visit.checked) == (2, 3)
    # the face showed at 13:44, but the person was tracked from 13:40 to 13:45
    assert (visit.first_seen, visit.last_seen) == (t0, t0 + 300)
    assert visit.to_dict()["face_seen"].endswith("13:44:00")
    assert (visits[0].first_seen, visits[0].last_seen) == (t0 + 7200, t0 + 7200)

    since, until = t0 - 600, t0 + 3600
    within = engine.face_search([unit(1, 0)], since, until, ["office"])
    assert len(within) == 1 and within[0].level == "likely"
    assert face_answer(within, {"office": 3}).startswith("Yes, likely: 1 visit(s), first on office")
    possible = group_face_visits(store.search("faces", unit(1, 0), cameras=["lobby"], top_k=None))
    assert face_answer(possible, {"lobby": 2}).startswith("Possibly: 1 weak match(es)")
    assert face_answer([], {"office": 37}, "Wed 07 Oct 13:00 - 16:00").startswith(
        "No match among 37 recorded faces (Wed 07 Oct 13:00 - 16:00)")
    assert face_answer([], {}).startswith("No faces were recorded")


# ------------------------------------------------------------------ camera context

CAMERAS = {
    "lobby": {"url": "rtsp://x", "about": {"description": "Reception desk and main door",
                                            "aliases": ["Reception", "front desk"],
                                            "faces": "good"}},
    "gate": {"url": "rtsp://y"},
}


def test_camera_about_and_aliases():
    assert camera_about(CAMERAS)["gate"] == {}
    assert camera_aliases(CAMERAS) == {"reception": "lobby", "front desk": "lobby"}
    with pytest.raises(ValueError, match="used by cameras"):
        camera_aliases({**CAMERAS, "yard": {"about": {"aliases": ["reception"]}}})
    with pytest.raises(ValueError, match="about.faces"):
        camera_about({"x": {"about": {"faces": "great"}}})


def test_parse_query_resolves_aliases():
    now = datetime(2026, 10, 7, 17, 0)
    q = parse_query("anyone at the front desk between 1 and 4pm", now=now,
                    camera_names=["lobby", "gate"], camera_aliases=camera_aliases(CAMERAS))
    assert q.cameras == ["lobby"]
    assert (q.since, q.until) == (datetime(2026, 10, 7, 13), datetime(2026, 10, 7, 16))
    q = parse_query("red car on gate or at reception", now=now, camera_names=["lobby", "gate"],
                    camera_aliases=camera_aliases(CAMERAS))
    assert sorted(q.cameras) == ["gate", "lobby"]


# ------------------------------------------------------------------ real models

@pytest.mark.skipif(not (ROOT / "models/face_recognition_sface_2021dec.onnx").exists(),
                    reason="face models not downloaded")
def test_real_models_quality_and_photo_check():
    engine = FaceEngine(ROOT / "models/face_detection_yunet_2023mar.onnx",
                        ROOT / "models/face_recognition_sface_2021dec.onnx", min_face_size=20)
    image = cv2.imread(str(ROOT / "data/messi5.jpg"))
    faces = engine.photo_faces(image)
    assert faces and faces[0].quality.sharpness > 30
    # a ~30 px face is fine on camera but too small to search with from an upload
    assert "too small" in choose_face(faces)
    face = engine.detect(image)[0]
    blurred = cv2.GaussianBlur(image, (9, 9), 0)
    _, sharp_q = engine.analyze(image, face)
    _, soft_q = engine.analyze(blurred, face)
    assert soft_q.sharpness < sharp_q.sharpness / 3 and "blurred" in engine.problems(soft_q)


@pytest.mark.skipif(not (ROOT / "models/face_recognition_sface_2021dec.onnx").exists(),
                    reason="face models not downloaded")
def test_real_models_find_big_and_small_faces_in_big_photos():
    engine = FaceEngine(ROOT / "models/face_detection_yunet_2023mar.onnx",
                        ROOT / "models/face_recognition_sface_2021dec.onnx", min_face_size=20)
    image = cv2.imread(str(ROOT / "data/messi5.jpg"))
    x, y, w, h = (int(v) for v in engine.detect(image)[0].box)

    # a selfie: the face fills a 2400 px photo. YuNet misses it at full size.
    crop = image[max(0, y - h // 2):y + h + h // 2, max(0, x - w // 2):x + w + w // 2]
    selfie = cv2.resize(crop, None, fx=2400 / crop.shape[0], fy=2400 / crop.shape[0],
                        interpolation=cv2.INTER_CUBIC)
    assert not engine._detect_at(selfie, 1.0)
    (face,) = engine.detect(selfie)
    fx, fy, fw, fh = face.box  # in the selfie's own pixels
    assert fw > 600 and 0 <= fx < selfie.shape[1] / 2 < fx + fw and fy < selfie.shape[0] / 2 < fy + fh

    # a ~30 px face in a 3200 px photo: too small at 640 px, found by the full-size retry
    group = np.full((2400, 3200, 3), 128, np.uint8)
    group[:image.shape[0], :image.shape[1]] = image
    assert not engine.detect(group)
    faces = engine.detect(group, retry=True)
    assert faces and abs(faces[0].box[0] - x) < 5


def test_indexer_without_clip_still_records_objects_and_faces(store):
    idx = SearchIndexer(store, "laptop", None, FakeFaceModel([0.5, 0.6]), face_interval_seconds=1,
                        object_interval_seconds=1, blocking=True)
    frame = np.full((240, 320, 3), 100, np.uint8)
    for i in range(2):
        person = Track(id=3, box=np.array([10, 10, 60, 120], float), score=0.9,
                       label_votes={"person": 1.0})
        idx.update(frame, [person], 1000.0 + i)
    idx.close()
    counts = store.counts()
    assert counts["sightings"] == 2 and counts["faces"] == 2  # no scenes without CLIP
    assert store.track_span("laptop", 3, 1000.5, 1000.5) == (1000.0, 1001.0)
    assert store.search("sightings", unit(1, 0)) == []  # rows without a vector are not searched
