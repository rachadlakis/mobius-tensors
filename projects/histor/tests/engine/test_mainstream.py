"""Main stream: ffmpeg command, keyframe pipe, recordings lookup/retention, hi-res crops."""

from __future__ import annotations

import io
import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from histor.engine.detector import Detection
from histor.engine.indexer import HiResFrame, SearchIndexer
from histor.engine.mainstream import Keyframe, MainStream, Recordings, prune_recordings, read_bmp
from histor.engine.pipeline import Pipeline
from histor.engine.store import EventStore
from histor.engine.tracker import Track, Tracker

URL = "rtsp://admin:s3cret@10.50.0.11:554/Streaming/Channels/101"


def test_command_copies_recording_and_decodes_only_keyframes(tmp_path):
    cmd = MainStream(URL, "gate", record_dir=tmp_path).command()
    i = cmd.index("-i")
    assert cmd[cmd.index("-rtsp_transport") + 1] == "tcp"
    assert cmd.index("-skip_frame") < i and cmd[cmd.index("-skip_frame") + 1] == "nokey"
    assert cmd[cmd.index("-c") + 1] == "copy"
    assert cmd[-1] == "pipe:1" and "bmp" in cmd and "segment" in cmd
    assert str(tmp_path / "gate" / "%Y%m%d-%H%M%S.mp4") in cmd

    record_only = MainStream(URL, "gate", record_dir=tmp_path, keyframes=False).command()
    assert "-skip_frame" not in record_only and "pipe:1" not in record_only
    with pytest.raises(ValueError):
        MainStream(URL, "gate", record_dir=None, keyframes=False)


def test_read_bmp_stream():
    a = np.random.default_rng(0).integers(0, 255, (7, 13, 3), dtype=np.uint8)  # row padding
    b = np.full((4, 6, 3), 200, np.uint8)
    pipe = io.BytesIO(cv2.imencode(".bmp", a)[1].tobytes() + cv2.imencode(".bmp", b)[1].tobytes())
    assert (read_bmp(pipe) == a).all() and (read_bmp(pipe) == b).all()
    assert read_bmp(pipe) is None


def test_ffmpeg_errors_never_log_the_password(caplog):
    stream = MainStream(URL, "gate", record_dir=None)
    with caplog.at_level(logging.WARNING):
        stream._log_errors(io.BytesIO(f"{URL}: 401 Unauthorized\n".encode()))
    assert "s3cret" not in caplog.text and "***" in caplog.text


def _segment(folder: Path, when: datetime) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{when:%Y%m%d-%H%M%S}.mp4"
    path.write_bytes(b"x")
    return path


def test_recordings_find_offset_and_reject_traversal(tmp_path):
    _segment(tmp_path / "gate", datetime(2026, 10, 5, 14, 0, 0))
    _segment(tmp_path / "gate", datetime(2026, 10, 5, 14, 1, 0))
    rec = Recordings(tmp_path, segment_seconds=60)
    found = rec.find("gate", datetime(2026, 10, 5, 14, 1, 37).timestamp())
    assert found is not None
    rel, offset = found
    assert rel == "gate/20261005-140100.mp4" and offset == pytest.approx(37)
    assert rec.find("gate", datetime(2026, 10, 5, 13, 59).timestamp()) is None  # before
    assert rec.find("gate", datetime(2026, 10, 5, 15, 0).timestamp()) is None  # gap
    assert rec.find("../gate", datetime(2026, 10, 5, 14, 0, 5).timestamp()) is None
    assert rec.path(rel) is not None and rec.path("../../etc/passwd") is None


def test_stopping_asks_ffmpeg_to_finish_its_file():
    class Proc:  # terminate() is a hard kill on Windows: the last file would be cut
        def __init__(self) -> None:
            self.stdin, self.code, self.terminated = io.BytesIO(), None, False

        def poll(self):
            return self.code

        def wait(self, timeout=None):
            self.code = 0

        def terminate(self):
            self.terminated = True

    stream = MainStream("rtsp://cam/main", "cam", keyframes=True)
    stream._proc = proc = Proc()  # type: ignore[assignment]
    stream._kill()
    assert proc.stdin.getvalue() == b"q" and not proc.terminated
    assert "-nostdin" not in stream.command()


def test_recordings_lead_before_an_alert(tmp_path):
    _segment(tmp_path / "gate", datetime(2026, 10, 5, 14, 0, 0))
    rec = Recordings(tmp_path, segment_seconds=60)
    at = datetime(2026, 10, 5, 14, 0, 10).timestamp()
    assert rec.find_lead("gate", at, 3) == ("gate/20261005-140000.mp4", pytest.approx(7))
    # an alert 1 s into the recording (it had just started): its clip starts with the file
    soon = datetime(2026, 10, 5, 14, 0, 1).timestamp()
    assert rec.find("gate", soon - 3) is None
    assert rec.find_lead("gate", soon, 3) == ("gate/20261005-140000.mp4", 0.0)
    assert rec.find_lead("gate", datetime(2026, 10, 5, 13, 0).timestamp(), 3) is None


def test_prune_by_age_and_free_space_keeps_the_file_being_written(tmp_path):
    folder = tmp_path / "gate"
    now = datetime(2026, 10, 5, 12).timestamp()
    old = _segment(folder, datetime(2026, 9, 20, 10))
    kept = _segment(folder, datetime(2026, 10, 4, 10))
    newest = _segment(folder, datetime(2026, 10, 5, 11, 59))
    assert prune_recordings(folder, retention_days=7, min_free_gb=None, now=now) == 1
    assert not old.exists() and kept.exists()
    assert prune_recordings(folder, None, min_free_gb=1e12, now=now) == 1  # disk "full"
    assert not kept.exists() and newest.exists()


# ------------------------------------------------------------------ hi-res crops in the index

class ShapeEmbedder:
    def __init__(self):
        self.shapes: list[tuple[int, int]] = []

    def embed_images(self, images):
        self.shapes += [im.shape[:2] for im in images]
        return np.tile(np.array([1, 0], np.float32), (len(images), 1))


def track(tid, box, label="person"):
    return Track(id=tid, box=np.array(box, float), score=0.9, label_votes={label: 1.0})


def test_indexer_upgrades_to_hires_and_keeps_fast_objects(tmp_path):
    store = EventStore(tmp_path / "db.sqlite")
    emb = ShapeEmbedder()
    idx = SearchIndexer(store, "gate", emb, object_interval_seconds=2, scene_interval_seconds=1e9,
                        scene_tiles=0, blocking=True)
    sub = np.full((240, 320, 3), 50, np.uint8)
    key = np.full((960, 1280, 3), 50, np.uint8)  # main stream, 4x the sub stream
    near, far = track(1, [10, 10, 60, 120]), track(2, [200, 100, 205, 110])  # far: 10 px high

    def hires(t, boxes):
        return HiResFrame(t, key, {k: np.array(v, float) for k, v in boxes.items()})

    idx.update(sub, [near, far], 1000.0)  # first sight: sub crop (far is too small)
    idx.update(sub, [near, far], 1000.5, hires(1000.5, {1: [40, 40, 240, 480],
                                                        2: [800, 400, 820, 440]}))
    idx.update(sub, [near, far], 1002.6)  # due by interval, but waits for a keyframe
    idx.update(sub, [near, far], 1003.0, hires(1003.0, {1: [44, 40, 244, 480]}))
    idx.close()
    assert emb.shapes == [
        (131, 60), (240, 320),  # t=1000: `near` from the sub stream right away + scene
        (524, 240), (48, 24),   # t=1000.5: `near` upgraded; `far` is only big enough in hi-res
        (524, 240),             # t=1003: next keyframe (t=1002.6 waited for it)
    ]
    store.close()


class SizeDetector:
    """Returns the same scene at either resolution (+ a parked car only seen in hi-res)."""

    def detect(self, frame):
        s = frame.shape[0] / 240
        dets = [Detection(10 * s + (3 if s > 1 else 0), 10 * s, 60 * s, 120 * s, 0.9, 0, "person")]
        if s > 1:
            dets.append(Detection(500, 300, 600, 380, 0.8, 2, "car"))
        return dets


class FakeMain:
    def __init__(self):
        self.pending: Keyframe | None = None

    def take_keyframe(self):
        frame, self.pending = self.pending, None
        return frame

    def close(self):
        pass


class CaptureIndexer:
    def __init__(self):
        self.hires = []

    def update(self, frame, tracks, t, hires=None):
        self.hires.append(hires)

    def close(self):
        pass


def test_pipeline_detects_on_keyframe_and_matches_tracks():
    main, idx = FakeMain(), CaptureIndexer()
    pipe = Pipeline(SizeDetector(), Tracker(), indexer=idx, mainstream=main)  # type: ignore[arg-type]
    sub = np.zeros((240, 320, 3), np.uint8)
    for i in range(4):
        pipe.process(sub, i / 10)
    main.pending = Keyframe(time.time(), np.zeros((960, 1280, 3), np.uint8))
    result = pipe.process(sub, 0.4)
    hires = idx.hires[-1]
    assert hires is not None and len(result.tracks) == 1
    (tid, box), = hires.boxes.items()  # the person matched, the hi-res-only car did not
    assert tid == result.tracks[0].id and box.tolist() == [43, 40, 240, 480]
    assert idx.hires[:-1] == [None] * 4


def test_api_links_results_to_recordings(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from histor.engine.server import create_app

    rec_dir = tmp_path / "recordings"
    _segment(rec_dir / "gate", datetime(2026, 10, 5, 14, 1))
    config = {"store": {"path": str(tmp_path / "db.sqlite")},
              "recording": {"dir": str(rec_dir)}, "cameras": {"gate": {}}}
    app = create_app(config, clip=None, faces=None)
    app.state.store.add_event("gate", datetime(2026, 10, 5, 14, 1, 40).timestamp(),
                              {"type": "cross", "rule": "line"}, None)
    app.state.store.add_event("gate", datetime(2026, 10, 5, 16).timestamp(),
                              {"type": "cross", "rule": "line"}, None)
    client = TestClient(app)
    later, earlier = client.get("/events").json()["events"]
    assert earlier["video_url"] == "/recordings/gate/20261005-140100.mp4#t=37"
    assert later["video_url"] is None  # nothing was recording then
    assert client.get("/recordings/gate/20261005-140100.mp4").status_code == 200
    assert client.get("/recordings/../db.sqlite").status_code == 404


# ------------------------------------------------------------------ real ffmpeg (if installed)

FFMPEG = os.environ.get("FFMPEG") or shutil.which("ffmpeg")


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not installed")
def test_ffmpeg_records_and_streams_keyframes(tmp_path):
    video = tmp_path / "in.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter.fourcc(*"mp4v"), 10, (320, 240))
    for i in range(30):
        writer.write(np.full((240, 320, 3), i * 8, np.uint8))
    writer.release()

    stream = MainStream(str(video), "cam", record_dir=tmp_path / "rec", ffmpeg=FFMPEG).start()
    try:
        deadline = time.monotonic() + 20
        while stream.latest() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        keyframe = stream.take_keyframe()
        assert keyframe is not None and keyframe.image.shape == (240, 320, 3)
        assert stream.take_keyframe() is None  # only new keyframes are returned
    finally:
        stream.close()
    files = list((tmp_path / "rec" / "cam").glob("*.mp4"))
    assert files and cv2.VideoCapture(str(files[0])).read()[0]
