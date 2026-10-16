"""Number plates: text normalisation and fuzzy matching, the OCR decoding, the plate finder's
box mapping, the indexer (sampling, voting, watchlist), the store, search, the chat tool, the
API and the pipeline. The real models run in the last tests when they are downloaded."""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from histor.engine import download_model
from histor.engine.detector import Detection
from histor.engine.indexer import HiResFrame
from histor.engine.pipeline import Pipeline
from histor.engine.plate_index import PlateIndexer, parse_watchlist
from histor.engine.plates import (PlateDetector, PlateRead, PlateReader, crop_box, letterbox,
                                 normalize_plate, plate_distance, plate_key, plates_enabled)
from histor.engine.search import SearchEngine, plate_visits
from histor.engine.store import EventStore
from histor.engine.tracker import Track, Tracker
from histor.llm import ToolCall
from histor.tools import Toolbox

ROOT = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 10, 9, 12, 0)
T0 = NOW.timestamp() - 3600  # 11:00 today


@pytest.fixture
def store(tmp_path):
    s = EventStore(tmp_path / "db.sqlite")
    yield s
    s.close()


def track(tid: int, box, label: str = "car", missed: int = 0) -> Track:
    t = Track(id=tid, box=np.array(box, float), score=0.9, label_votes={label: 1.0})
    t.missed = missed
    return t


class FakeDetector:
    """Finds a plate in the bottom middle of any vehicle crop, `width` pixels wide."""

    def __init__(self, width: float | None = None, score: float = 0.9) -> None:
        self.width = width
        self.score = score
        self.calls: list[tuple[int, int]] = []

    def detect(self, image):
        h, w = image.shape[:2]
        self.calls.append((h, w))
        pw = self.width if self.width is not None else w * 0.4
        x1 = (w - pw) / 2
        return [(x1, h * 0.7, x1 + pw, h * 0.85, self.score)]


class FakeReader:
    """Reads the texts given in order (cycling), with their confidences."""

    def __init__(self, reads: list[tuple[str, float]]) -> None:
        self.reads = reads
        self.i = 0
        self.crops: list[np.ndarray] = []

    def read(self, crops):
        out = []
        for crop in crops:
            self.crops.append(crop)
            text, conf = self.reads[self.i % len(self.reads)]
            self.i += 1
            out.append(PlateRead(text, conf, "Unknown", 0.5))
        return out


def indexer(store, reads, **options) -> PlateIndexer:
    options = {"blocking": True, "interval_seconds": 1.0, "min_plate_width": 20, **options}
    return PlateIndexer(store, "gate", options.pop("detector", FakeDetector()),
                        FakeReader(reads), **options)


def run(idx: PlateIndexer, seconds: int, tracks=None, start: float = T0, fps: int = 2):
    frame = np.full((480, 640, 3), 120, np.uint8)
    tracks = tracks if tracks is not None else [track(1, [100, 100, 400, 300])]
    for i in range(seconds * fps):
        idx.update(frame, tracks, start + i / fps)


# ------------------------------------------------------------------ text

@pytest.mark.parametrize("raw, expected", [
    ("ab-12 3", "AB123"), (" KL 5521 ", "KL5521"), ("é1!", "1"), ("", ""), (None, "")])
def test_normalize_plate(raw, expected):
    assert normalize_plate(raw) == expected


@pytest.mark.parametrize("a, b, distance", [
    ("AB123CD", "AB 123 CD", 0),
    ("AB123CD", "A8123CD", 0),     # 8 / B look alike
    ("0O0", "OOD", 0),             # 0 / O / D
    ("SI2", "512", 0),             # S/5, I/1, Z/2... 2 is itself
    ("AB123CD", "AB124CD", 1),
    ("AB123CD", "AB123C", 1),      # a character missed
    ("AB123CD", "XY987ZW", 7),
])
def test_plate_distance(a, b, distance):
    assert plate_distance(a, b) == distance


def test_plate_key_only_merges_lookalikes():
    assert plate_key("ABCD") == "A8C0" and plate_key("XYZ") == "XY2"


# ------------------------------------------------------------------ OCR decoding

def bare_reader(alphabet="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_", regions=("Spain", "Unknown")):
    r = object.__new__(PlateReader)
    r.alphabet, r.pad, r.regions = alphabet, "_", list(regions)
    r.size, r.rgb, r.keep_aspect = (128, 64), True, False
    return r


def slots_for(text: str, alphabet: str, slots: int = 10, sure: float = 0.97) -> np.ndarray:
    out = np.full((slots, len(alphabet)), (1 - sure) / (len(alphabet) - 1), np.float32)
    for i, ch in enumerate(text.ljust(slots, "_")):
        out[i, alphabet.index(ch)] = sure
    return out


def test_decode_drops_pad_and_takes_lowest_char_confidence():
    r = bare_reader()
    slots = slots_for("AB123", r.alphabet)
    slots[2] = 0.01
    slots[2, r.alphabet.index("1")] = 0.55  # one doubtful character
    read = r.decode(slots, np.array([0.9, 0.1]))
    assert read.text == "AB123" and read.confidence == pytest.approx(0.55)
    assert read.region == "Spain" and read.region_confidence == pytest.approx(0.9)


def test_decode_empty_and_inner_pad():
    r = bare_reader()
    assert r.decode(slots_for("", r.alphabet)).text == ""
    assert r.decode(slots_for("", r.alphabet)).confidence == 0.0
    # a pad in the middle (model unsure of a gap) is dropped, not kept as "_"
    assert r.decode(slots_for("AB_12", r.alphabet)).text == "AB12"


def test_reader_preprocess_shapes():
    r = bare_reader()
    crop = np.zeros((40, 200, 3), np.uint8)
    crop[..., 2] = 255  # red in BGR
    x = r.preprocess(crop)
    assert x.shape == (64, 128, 3) and x.dtype == np.uint8 and x[..., 0].mean() == 255  # RGB
    r.rgb = False
    assert r.preprocess(crop).shape == (64, 128, 1)
    r.rgb, r.keep_aspect = True, True
    x = r.preprocess(crop)
    assert x.shape == (64, 128, 3) and x[-1].sum() == 0  # letterboxed: bottom rows padded


def test_reader_missing_model(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_model plates"):
        PlateReader(tmp_path / "no.onnx", tmp_path / "no.yaml")


# ------------------------------------------------------------------ plate finder

def test_letterbox_and_box_mapping_back():
    image = np.zeros((300, 600, 3), np.uint8)
    tensor, r, (dw, dh) = letterbox(image, 384)
    assert tensor.shape == (1, 3, 384, 384) and tensor.dtype == np.float32
    assert r == pytest.approx(384 / 600) and dw == 0 and dh == pytest.approx((384 - 192) / 2)
    det = object.__new__(PlateDetector)
    det.type, det.min_score, det.input, det.input_size = "yolov9", 0.4, "images", 384
    # a plate at x 300..400, y 200..250 in the original -> model coordinates
    model_box = [300 * r + dw, 200 * r + dh, 400 * r + dw, 250 * r + dh]
    rows = np.array([[0, *model_box, 0, 0.9],
                     [0, 0, 0, 5, 5, 0, 0.95],         # tiny after mapping: dropped
                     [0, *model_box, 0, 0.2]], np.float32)  # below min_score
    det.session = SimpleNamespace(run=lambda _, feeds: [rows])
    (box,) = det.detect(image)
    assert box[:4] == pytest.approx((300, 200, 400, 250), abs=0.5) and box[4] == pytest.approx(0.9)


def test_detector_clips_to_image_and_sorts():
    det = object.__new__(PlateDetector)
    det.type, det.min_score, det.input, det.input_size = "yolov9", 0.1, "images", 100
    rows = np.array([[0, -10, 10, 50, 40, 0, 0.5], [0, 60, 60, 120, 90, 0, 0.8]], np.float32)
    det.session = SimpleNamespace(run=lambda _, feeds: [rows])
    out = det.detect(np.zeros((100, 100, 3), np.uint8))
    assert [b[4] for b in out] == pytest.approx([0.8, 0.5])
    assert out[0][2] == 100 and out[1][0] == 0


def test_detector_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_model plates"):
        PlateDetector(tmp_path / "no.onnx")
    (tmp_path / "m.onnx").write_bytes(b"")
    with pytest.raises(ValueError, match="yolov9 or yolox"):
        PlateDetector(tmp_path / "m.onnx", type="ssd")


def test_crop_box_padding_and_edges():
    image = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
    assert crop_box(image, (10, 10, 30, 20)).shape == (10, 20, 3)
    assert crop_box(image, (10, 10, 30, 20), pad=0.5).shape == (20, 40, 3)
    assert crop_box(image, (90, 90, 130, 130)).shape == (10, 10, 3)
    assert crop_box(image, (50, 50, 51, 51)) is None


def test_plates_enabled():
    assert not plates_enabled(None, "gate") and not plates_enabled({"enabled": False}, "gate")
    assert plates_enabled({"enabled": True}, "gate")
    assert plates_enabled({"enabled": True, "cameras": ["gate"]}, "gate")
    assert not plates_enabled({"enabled": True, "cameras": ["yard"]}, "gate")


def test_download_names_match_config_defaults():
    names = set(download_model.PLATE_MODELS)
    assert {"plate_ocr_cct_s_v2.onnx", "plate_ocr_cct_s_v2.yaml", "plate_yolov9_t_384.onnx"} == names
    assert all(u.startswith("https://github.com/ankandrew/") for u in download_model.PLATE_MODELS.values())


# ------------------------------------------------------------------ indexer

def test_watchlist_parsing():
    assert parse_watchlist(None) == []
    assert parse_watchlist(["ab 123"]) == [("AB123", "watchlist")]
    assert parse_watchlist([{"plate": "xy-9", "name": "van"}]) == [("XY9", "van")]
    assert parse_watchlist({"KL 55": "boss"}) == [("KL55", "boss")]
    with pytest.raises(ValueError, match="no plate"):
        parse_watchlist([{"name": "x"}])


def test_samples_vehicles_once_per_interval(store):
    idx = indexer(store, [("AB123", 0.9)])
    run(idx, 5, [track(1, [100, 100, 400, 300]), track(2, [10, 10, 60, 200], "person"),
                 track(3, [0, 0, 300, 20], "truck"),               # too small (20 px tall)
                 track(4, [300, 100, 600, 300], "bus", missed=1)])  # not seen this frame
    idx.close()
    assert len(idx.detector.calls) == 5  # vehicle 1, once a second for 5 s; nothing else


def test_votes_and_stores_new_or_better_reads(store):
    reads = [("AB123", 0.7), ("AB123", 0.72), ("A8123", 0.65), ("AB123", 0.9), ("XX", 0.3)]
    idx = indexer(store, reads)
    run(idx, 5)
    idx.close()
    rows = store.plate_rows()
    # 0.7 stored; 0.72 not clearly better; A8123 new text; 0.9 better; 0.3 below min_confidence
    assert [(r["text"], r["confidence"]) for r in rows] == [
        ("AB123", 0.7), ("A8123", 0.65), ("AB123", 0.9)]
    assert all(r["label"] == "car" and r["track_id"] == 1 for r in rows)
    assert all(store.media_path(r["thumb"]) is not None for r in rows)
    assert idx.plates(T0 + 4.5) == {1: "AB123"}
    assert idx.plates(T0 + 60) == {}  # too old for the live view


def test_max_rows_per_vehicle(store):
    idx = indexer(store, [(f"AB{i}", 0.9) for i in range(10)], max_per_track=3)
    run(idx, 10)
    idx.close()
    assert len(store.plate_rows()) == 3


def test_narrow_plates_are_not_read(store):
    idx = indexer(store, [("AB123", 0.9)], detector=FakeDetector(width=10), min_plate_width=40)
    run(idx, 3)
    idx.close()
    assert idx.reader.i == 0 and store.plate_rows() == []


def test_reads_from_hires_keyframe_when_matched(store):
    idx = indexer(store, [("AB123", 0.9)])
    frame = np.full((360, 640, 3), 50, np.uint8)
    key = np.full((1080, 1920, 3), 200, np.uint8)
    hires = HiResFrame(T0, key, {1: np.array([300, 300, 1200, 900])})
    idx.update(frame, [track(1, [100, 100, 400, 300])], T0, hires)
    idx.close()
    h, w = idx.detector.calls[0]
    assert (h, w) == (660, 990)  # the hi-res box, padded 5%: not the sub-stream one
    assert idx.reader.crops[0].mean() == pytest.approx(200)


def test_watchlist_event_once_after_min_reads(store):
    idx = indexer(store, [("A8 123 CD".replace(" ", ""), 0.9)],
                  watchlist=[{"plate": "AB 123 CD", "name": "banned_van"}, "ZZ999"],
                  min_reads=2)
    run(idx, 1)
    assert idx.take_events() == []  # one read is not enough
    run(idx, 5, start=T0 + 1)
    idx.close()
    events = idx.take_events()
    assert len(events) == 1
    t, track_id, rule, label, text = events[0]
    assert (track_id, rule, label, text) == (1, "banned_van", "car", "A8123CD")
    assert idx.take_events() == []


def test_watchlist_distance(store):
    far = indexer(store, [("AB124CE", 0.9)], watchlist=["AB123CD"], max_distance=1)
    run(far, 3)
    far.close()
    assert far.take_events() == []
    near = indexer(store, [("AB124CD", 0.9)], watchlist=["AB123CD"], max_distance=1)
    run(near, 3)
    near.close()
    assert [e[2] for e in near.take_events()] == ["watchlist"]


def test_live_queue_drops_instead_of_blocking(store):
    import threading

    gate = threading.Event()

    class SlowDetector(FakeDetector):
        def detect(self, image):
            gate.wait(5)
            return super().detect(image)

    idx = PlateIndexer(store, "gate", SlowDetector(), FakeReader([("AB1", 0.9)]),
                       interval_seconds=0, blocking=False, max_queue=1, min_plate_width=1)
    frame = np.zeros((480, 640, 3), np.uint8)
    for i in range(10):
        idx.update(frame, [track(1, [100, 100, 400, 300])], T0 + i)
    gate.set()
    idx.close()
    assert idx.dropped > 0


def test_reader_failure_does_not_stop_the_thread(store, caplog):
    class Broken(FakeReader):
        def read(self, crops):
            if self.i == 0:
                self.i += 1
                raise RuntimeError("boom")
            return super().read(crops)

    idx = PlateIndexer(store, "gate", FakeDetector(), Broken([("AB1", 0.9)]), blocking=True,
                       min_plate_width=1)
    run(idx, 3)
    idx.close()
    assert "Plate reading failed" in caplog.text and len(store.plate_rows()) == 1


# ------------------------------------------------------------------ store

def test_store_plate_rows_filters_and_purge(tmp_path):
    s = EventStore(tmp_path / "db.sqlite", retention_days=30, plate_retention_days=7)
    try:
        now = NOW.timestamp()
        s.add_plates([
            {"camera": "gate", "time": now - 10 * 86400, "track_id": 1, "text": "OLD1",
             "confidence": 0.9, "box": [0, 0, 1, 1]},
            {"camera": "gate", "time": now - 60, "track_id": 2, "text": "NEW1", "confidence": 0.8},
            {"camera": "yard", "time": now - 30, "track_id": 3, "text": "YARD1"},
        ])
        assert [r["text"] for r in s.plate_rows(cameras=["gate"])] == ["OLD1", "NEW1"]
        assert [r["text"] for r in s.plate_rows(since=now - 120)] == ["NEW1", "YARD1"]
        assert s.plate_rows()[0]["box"] == [0, 0, 1, 1]
        assert s.counts()["plates"] == 3
        assert s.purge(now)["plates"] == 1  # 10 days > 7, though the rest keeps 30
    finally:
        s.close()


def test_plate_retention_defaults_to_retention(tmp_path):
    s = EventStore(tmp_path / "db.sqlite", retention_days=12)
    try:
        assert s.plate_retention_days == 12
    finally:
        s.close()


def test_old_events_table_gets_plate_column(tmp_path):
    path = tmp_path / "old.db"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE events (id INTEGER PRIMARY KEY, camera TEXT NOT NULL, time REAL NOT "
               "NULL, type TEXT NOT NULL, rule TEXT NOT NULL, label TEXT, track_id INTEGER, "
               "direction TEXT, dwell_seconds REAL, snapshot TEXT)")
    db.commit()
    db.close()
    s = EventStore(path)
    try:
        s.add_event("gate", 1.0, {"type": "plate", "rule": "van", "plate": "AB1"}, None)
        assert s._db.execute("SELECT plate FROM events").fetchone()[0] == "AB1"
    finally:
        s.close()


# ------------------------------------------------------------------ search

def read_row(camera, t, track_id, text, conf=0.9, thumb=None, label="car"):
    return {"camera": camera, "time": t, "track_id": track_id, "text": text, "confidence": conf,
            "thumb": thumb or f"{text}.jpg", "label": label}


def test_plate_visits_group_and_vote():
    rows = [read_row("gate", 0, 1, "AB123", 0.7), read_row("gate", 1, 1, "A8123", 0.95),
            read_row("gate", 2, 1, "AB123", 0.8), read_row("gate", 5, 2, "XY9"),
            read_row("gate", 2000, 1, "AB123")]  # same id much later: a new visit
    visits = plate_visits(rows)
    assert [(v.track_id, v.plate) for v in visits] == [(1, "AB123"), (2, "XY9"), (1, "AB123")]
    v = visits[0]
    assert v.reads == pytest.approx({"AB123": 1.5, "A8123": 0.95})
    assert v.confidence == pytest.approx(0.8) and v.thumb == "AB123.jpg"
    d = v.to_dict()
    assert d["plate"] == "AB123" and d["also_read"] == ["A8123"] and "match" not in d


def test_plate_visits_query():
    rows = [read_row("gate", 0, 1, "AB123CD"), read_row("gate", 10, 2, "AB124CD"),
            read_row("gate", 20, 3, "XY987ZW"), read_row("gate", 30, 4, "QQAB12"),
            read_row("gate", 40, 5, "A8123CD")]
    visits = plate_visits(rows, "ab 123 cd")
    assert [(v.track_id, v.distance) for v in visits] == [(1, 0), (5, 0), (2, 1)]
    assert [v.to_dict()["match"] for v in visits] == ["exact", "exact", "close"]
    assert [v.track_id for v in plate_visits(rows, "AB12")] == [1, 2, 4, 5]  # contained
    assert plate_visits(rows, "AB123CD", max_distance=0)[-1].track_id == 5
    assert plate_visits(rows, "ZZZZZZZ") == []


def test_engine_plates_span_whole_vehicle_visit(store):
    store.add_sightings([
        {"camera": "gate", "time": T0 - 30, "kind": "object", "track_id": 1, "label": "car"},
        {"camera": "gate", "time": T0 + 90, "kind": "object", "track_id": 1, "label": "car"}])
    store.add_plates([read_row("gate", T0, 1, "AB123"), read_row("gate", T0 + 2, 1, "AB123")])
    (v,) = SearchEngine(store).plates("AB123")
    assert (v.first_seen, v.last_seen) == (T0 - 30, T0 + 90)


# ------------------------------------------------------------------ chat tool

@pytest.fixture
def toolbox(store):
    store.add_plates([read_row("gate", T0, 1, "AB123CD", thumb="p1.jpg"),
                      read_row("gate", T0 + 1, 1, "AB123CD", thumb="p1.jpg"),
                      read_row("yard", T0 + 600, 7, "XY987", label="truck")])
    store.add_event("gate", T0 + 2, {"type": "plate", "rule": "banned_van", "label": "car",
                                     "track_id": 1, "plate": "AB123CD"}, "ev.jpg")
    box = Toolbox(SearchEngine(store, camera_names=["gate", "yard"]), now=NOW)
    yield box
    box.close()


def test_plates_tool_offered_when_rows_or_config(store, toolbox):
    from histor.chat import PLATES_HELP, system_prompt

    assert "plates" in [t.name for t in toolbox.tools]
    assert PLATES_HELP.strip() in system_prompt(toolbox, NOW)
    empty = EventStore(Path(store.path).parent / "empty.db")
    try:
        off = Toolbox(SearchEngine(empty), now=NOW)
        on = Toolbox(SearchEngine(empty), now=NOW, plates_config={"enabled": True})
        assert "plates" not in [t.name for t in off.tools] and "plates" in [t.name for t in on.tools]
        assert off.run(ToolCall("1", "plates", {})).is_error
        off.close()
        on.close()
    finally:
        empty.close()


def test_plates_tool_find_and_list(toolbox):
    out = json.loads(toolbox.run(ToolCall("1", "plates", {"plate": "ab-123-cd",
                                                          "when": "today"})).output)
    assert out["total"] == 1 and out["looked_for"] == "AB123CD"
    (v,) = out["vehicles"]
    assert v["plate"] == "AB123CD" and v["match"] == "exact" and v["camera"] == "gate"
    assert v["first_seen"].endswith("11:00:00") and "p1.jpg" in toolbox.evidence
    out = toolbox.plates(when="today")
    assert [v["plate"] for v in out["vehicles"]] == ["AB123CD", "XY987"]
    assert [v["plate"] for v in toolbox.plates(camera="yard")["vehicles"]] == ["XY987"]


def test_plates_tool_no_match_says_what_was_read(toolbox):
    out = toolbox.plates(plate="ZZ000", when="today")
    assert out["total"] == 0 and out["plates_read_in_range"] == 2
    r = toolbox.run(ToolCall("1", "plates", {"plate": "--"}))
    assert r.is_error and "letters and digits" in r.output


def test_alerts_tool_lists_plate_events(toolbox):
    out = toolbox.alerts(types=["plate"])
    (ev,) = out["events"]
    assert ev["rule"] == "banned_van" and ev["plate"] == "AB123CD"
    assert toolbox.run(ToolCall("1", "alerts", {"types": ["plate"]})).is_error is False


def test_look_can_open_plate_thumbs(toolbox):
    from histor.tools import _media

    out = toolbox.plates(plate="AB123CD")
    assert "p1.jpg" in _media(out)


# ------------------------------------------------------------------ API

def test_api_plates(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from histor.engine.server import create_app

    app = create_app({"store": {"path": str(tmp_path / "db.sqlite")},
                      "cameras": {"gate": {}, "yard": {}}})
    s: EventStore = app.state.store
    s.add_plates([read_row("gate", T0, 1, "AB123CD"), read_row("yard", T0 + 5, 2, "XY987")])
    client = TestClient(app)
    r = client.get("/plates", params={"plate": "A8 123 CD"}).json()
    assert r["total"] == 1 and r["results"][0]["plate"] == "AB123CD"
    assert r["results"][0]["match"] == "exact"
    r = client.get("/plates", params={"camera": "yard"}).json()
    assert [x["plate"] for x in r["results"]] == ["XY987"]
    assert client.get("/plates", params={"max_distance": 9}).status_code == 422
    assert client.get("/health").json()["counts"]["plates"] == 2


# ------------------------------------------------------------------ pipeline

class OneCar:
    def detect(self, frame):
        return [Detection(100, 100, 400, 300, 0.9, 2, "car")]


def test_pipeline_watchlist_event_reaches_store_and_live_view(store):
    from histor.engine.alerts import AlertManager

    idx = indexer(store, [("AB123", 0.9)], watchlist={"AB 123": "boss"}, min_reads=2)
    alerts = AlertManager(console=False, store=store, source="gate")
    pipe = Pipeline(OneCar(), Tracker(), alerts=alerts, store=store, start_time=T0, plates=idx)
    frame = np.full((480, 640, 3), 90, np.uint8)
    events = []
    for i in range(8):
        result = pipe.process(frame, i * 0.5)
        events += result.events
        if i == 6:
            drawn = pipe.draw(frame.copy(), result)
        time.sleep(0.02)  # reads run on the indexer's thread; a real camera gives it ~100 ms
                          # a frame, so without a pause the event may only come out at close()
    pipe.close()
    rows = store._db.execute("SELECT type, rule, label, plate FROM events").fetchall()
    assert [tuple(r) for r in rows] == [("plate", "boss", "car", "AB123")]
    assert [e.plate for e in events] == ["AB123"]
    assert not np.array_equal(drawn[300:330], frame[300:330])  # plate label under the box


def test_pipeline_from_config_without_models_records_the_rest(store, tmp_path, caplog):
    config = {"model": {"path": str(ROOT / "models" / "yolox_nano.onnx")},
              "store": {"path": str(tmp_path / "p.db")},
              "plates": {"enabled": True, "detector": {"model": str(tmp_path / "no.onnx")}}}
    if not Path(config["model"]["path"]).exists():
        pytest.skip("uv run -m histor.engine.download_model yolox_nano")
    pipe = Pipeline.from_config(config, source="gate", live=False)
    assert pipe.plates is None and "No plate reading" in caplog.text
    pipe.close()
    pipe.store.close()


# ------------------------------------------------------------------ the real models

OCR = ROOT / "models" / "plate_ocr_cct_s_v2.onnx"
OCR_CFG = ROOT / "models" / "plate_ocr_cct_s_v2.yaml"
FINDER = ROOT / "models" / "plate_yolov9_t_384.onnx"


def synthetic_plate(text: str, w: int = 260, h: int = 60) -> np.ndarray:
    plate = np.full((h, w, 3), 255, np.uint8)
    cv2.rectangle(plate, (2, 2), (w - 3, h - 3), (0, 0, 0), 3)
    cv2.putText(plate, text, (14, h - 15), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 3)
    return plate


@pytest.mark.skipif(not (OCR.exists() and OCR_CFG.exists()),
                    reason="uv run -m histor.engine.download_model plates")
@pytest.mark.parametrize("text, expected", [("ABC1234", "ABC1234"), ("7XYZ42", "7XYZ42"),
                                            ("KL 5521", "KL5521")])
def test_real_reader_reads_synthetic_plates(text, expected):
    reader = PlateReader(OCR, OCR_CFG)
    (read,) = reader.read([synthetic_plate(text)])
    assert read.text == expected and read.confidence > 0.8


@pytest.mark.skipif(not (OCR.exists() and OCR_CFG.exists()),
                    reason="uv run -m histor.engine.download_model plates")
def test_real_reader_batches_match_single_reads():
    reader = PlateReader(OCR, OCR_CFG)
    plates = [synthetic_plate(t) for t in ("AB12CD", "9988XY", "QWE77")]
    batch = reader.read(plates)
    singles = [reader.read([p])[0] for p in plates]
    assert [b.text for b in batch] == [s.text for s in singles] == ["AB12CD", "9988XY", "QWE77"]


def synthetic_car(text: str) -> np.ndarray:
    car = np.full((300, 500, 3), (90, 90, 90), np.uint8)
    cv2.rectangle(car, (60, 60), (440, 215), (40, 40, 160), -1)  # body
    car[220:280, 120:380] = synthetic_plate(text)
    return car


@pytest.mark.skipif(not FINDER.exists(), reason="uv run -m histor.engine.download_model plates")
def test_real_finder_locates_the_plate():
    det = PlateDetector(FINDER)
    (x1, y1, x2, y2, score), *_ = det.detect(synthetic_car("AB12CD"))
    assert (x1, y1, x2, y2) == pytest.approx((120, 220, 380, 280), abs=8) and score > 0.6
    assert det.detect(np.full((60, 90, 3), 90, np.uint8)) == []  # no plate, small image


@pytest.mark.skipif(not (FINDER.exists() and OCR.exists()),
                    reason="uv run -m histor.engine.download_model plates")
def test_real_models_end_to_end_in_the_indexer(store):
    idx = PlateIndexer(store, "gate", PlateDetector(FINDER), PlateReader(OCR, OCR_CFG),
                       watchlist=["AB 12 CD"], blocking=True)
    frame = np.full((720, 1280, 3), 60, np.uint8)
    frame[300:600, 400:900] = synthetic_car("AB12CD")
    for i in range(3):
        idx.update(frame, [track(5, [400, 300, 900, 600])], T0 + i)
    idx.close()
    rows = store.plate_rows()
    assert rows and {r["text"] for r in rows} == {"AB12CD"}
    assert [(e[2], e[4]) for e in idx.take_events()] == [("watchlist", "AB12CD")]
    (v,) = SearchEngine(store).plates("A8 12 CD")
    assert v.plate == "AB12CD" and v.distance == 0


# ------------------------------------------------------------------ spacing never matters

SPELLINGS = ["AB123CD", "AB 123 CD", "AB12 3CD", "A B 1 2 3 C D", "ab-123-cd", "AB.123.CD",
             " AB 123CD ", "AB\t123\nCD", "ab 12-3 cd"]


@pytest.mark.parametrize("spelling", SPELLINGS)
def test_spaces_and_separators_are_ignored(spelling):
    assert normalize_plate(spelling) == "AB123CD"
    assert plate_distance(spelling, "AB123CD") == 0


@pytest.mark.parametrize("spelling", SPELLINGS)
def test_search_finds_the_plate_however_it_is_typed(store, spelling):
    store.add_plates([read_row("gate", T0, 1, "AB123CD")])
    box = Toolbox(SearchEngine(store, camera_names=["gate"]), now=NOW)
    try:
        (v,) = box.plates(plate=spelling)["vehicles"]
        assert v["plate"] == "AB123CD" and v["match"] == "exact"
    finally:
        box.close()


@pytest.mark.parametrize("listed", ["AB 123 CD", "AB12 3CD", "ab-123-cd"])
def test_watchlist_entry_however_it_is_typed(store, listed):
    idx = indexer(store, [("AB123CD", 0.9)], watchlist=[{"plate": listed, "name": "van"}])
    run(idx, 3)
    idx.close()
    assert [(e[2], e[4]) for e in idx.take_events()] == [("van", "AB123CD")]


def test_api_plate_with_spaces(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from histor.engine.server import create_app

    app = create_app({"store": {"path": str(tmp_path / "db.sqlite")}})
    app.state.store.add_plates([read_row("gate", T0, 1, "AB123CD")])
    client = TestClient(app)
    for spelling in ("AB 123 CD", "AB12 3CD", "ab-123-cd"):
        (r,) = client.get("/plates", params={"plate": spelling}).json()["results"]
        assert r["plate"] == "AB123CD" and r["match"] == "exact"
