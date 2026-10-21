"""CLIP model tags: export presets, model names read from the ONNX metadata, stored vectors tagged
with their model, and search that never compares vectors of two models."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from histor.engine import export_clip
from histor.engine.clip_onnx import LEGACY_MODEL, ClipEncoder, preprocess, session_model
from histor.engine.indexer import SearchIndexer
from histor.engine.search import SearchEngine
from histor.engine.store import EventStore
from histor.engine.tracker import Track

ROOT = Path(__file__).resolve().parents[2]
NEW = "ViT-B-32/laion2b_s34b_b79k"
NOW = datetime(2026, 10, 9, 12)


def unit(*values: float) -> np.ndarray:
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)


class FakeSession:
    def __init__(self, meta: dict[str, str] | None) -> None:
        self.meta = meta

    def get_modelmeta(self):
        return type("Meta", (), {"custom_metadata_map": self.meta})()


class TaggedEmbedder:
    def __init__(self, model: str | None) -> None:
        if model is not None:
            self.model = model

    def embed_images(self, images):
        return np.stack([unit(1, 0, 0) for _ in images])


class TaggedClip:
    """Text side of a CLIP model: every query points along x."""

    def __init__(self, model: str) -> None:
        self.model = model

    def embed_query(self, text: str) -> np.ndarray:
        return unit(1, 0, 0)


def sighting(t: float, model: str | None, track_id: int, vector=None) -> dict:
    row = {"camera": "gate", "time": t, "kind": "object", "track_id": track_id,
           "label": "person", "box": [0, 0, 10, 20],
           "embedding": unit(1, 0, 0) if vector is None else vector}
    if model is not None:
        row["model"] = model
    return row


# ------------------------------------------------------------------ export presets

def test_default_preset_is_laion2b_b32():
    assert export_clip.DEFAULT_PRESET == "laion2b"
    assert export_clip.model_name("laion2b") == NEW
    assert export_clip.model_name("laion400m") == LEGACY_MODEL


@pytest.mark.parametrize("preset", sorted(export_clip.PRESETS))
def test_presets_are_b32_with_known_source(preset):
    arch, pretrained, url = export_clip.PRESETS[preset]
    assert arch.startswith("ViT-B-32")  # same size: clip_onnx's 224 px input and 512-d vectors
    assert pretrained and (url is None or url.startswith("https://"))


# ------------------------------------------------------------------ model name from ONNX

@pytest.mark.parametrize("meta, expected", [
    ({"clip_model": NEW}, NEW),
    ({}, LEGACY_MODEL),         # exported before the tag existed
    (None, LEGACY_MODEL),
    ({"clip_model": ""}, LEGACY_MODEL),
])
def test_session_model(meta, expected):
    assert session_model(FakeSession(meta)) == expected


def test_encoder_refuses_mixed_image_and_text_models(monkeypatch, tmp_path):
    for name in ("image.onnx", "text.onnx"):
        (tmp_path / name).write_bytes(b"")
    sessions = {"image.onnx": FakeSession({"clip_model": NEW}),
                "text.onnx": FakeSession({})}
    monkeypatch.setattr("histor.engine.onnx_device.create_session",
                        lambda path, device, threads: sessions[Path(path).name])
    monkeypatch.setattr("histor.engine.clip_onnx.Tokenizer", lambda vocab: None)
    with pytest.raises(ValueError, match="differ"):
        ClipEncoder(tmp_path / "image.onnx", tmp_path / "text.onnx", tmp_path / "vocab")
    enc = ClipEncoder(tmp_path / "image.onnx", None, tmp_path / "vocab")
    assert enc.model == NEW


def test_encoder_reports_missing_model(tmp_path):
    with pytest.raises(FileNotFoundError, match="export_clip"):
        ClipEncoder(tmp_path / "nope.onnx", None)


def test_preprocess_pads_instead_of_cropping():
    tall = np.zeros((300, 100, 3), np.uint8)
    tall[:10] = 255  # a white head at the very top must survive
    out = preprocess(tall)
    assert out.shape == (3, 224, 224) and out.dtype == np.float32
    assert out[:, 0, 112].mean() > out[:, 112, 0].mean()  # top-centre white, side padding grey


# ------------------------------------------------------------------ store tags

def test_store_tags_and_filters_by_model(store):
    store.add_sightings([sighting(1000, LEGACY_MODEL, 1), sighting(1001, NEW, 2),
                         sighting(1002, None, 3)])  # untagged = legacy
    assert {h.track_id for h in store.search("sightings", unit(1, 0, 0), model=NEW)} == {2}
    assert {h.track_id for h in store.search("sightings", unit(1, 0, 0),
                                             model=LEGACY_MODEL)} == {1, 3}
    assert len(store.search("sightings", unit(1, 0, 0))) == 3  # no model: everything
    assert store.embedding_models() == {LEGACY_MODEL: 2, NEW: 1}


def test_store_does_not_tag_rows_without_vectors(store):
    store.add_sightings([{"camera": "gate", "time": 1, "kind": "scene", "model": NEW}])
    row = store._db.execute("SELECT model, length(embedding) FROM sightings").fetchone()
    assert row[0] is None and row[1] == 0
    assert store.embedding_models() == {}


def test_old_store_gets_model_column(tmp_path):
    path = tmp_path / "old.db"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE sightings (id INTEGER PRIMARY KEY, camera TEXT NOT NULL, "
               "time REAL NOT NULL, kind TEXT NOT NULL, track_id INTEGER, label TEXT, box TEXT, "
               "thumb TEXT, embedding BLOB NOT NULL)")
    db.execute("INSERT INTO sightings (camera, time, kind, track_id, embedding) "
               "VALUES ('gate', 5, 'object', 7, ?)", (unit(1, 0, 0).tobytes(),))
    db.commit()
    db.close()
    s = EventStore(path)
    try:
        assert [h.track_id for h in s.search("sightings", unit(1, 0, 0), model=LEGACY_MODEL)] == [7]
        assert s.search("sightings", unit(1, 0, 0), model=NEW) == []
    finally:
        s.close()


def test_purge_drops_old_model_rows_like_any_other(store):
    now = NOW.timestamp()
    store.add_sightings([sighting(now - 40 * 86400, LEGACY_MODEL, 1), sighting(now, NEW, 2)])
    store.purge(now)
    assert store.embedding_models() == {NEW: 1}


# ------------------------------------------------------------------ indexer and search

@pytest.mark.parametrize("model, stored", [(NEW, NEW), (None, None)])
def test_indexer_tags_rows_with_embedder_model(store, model, stored):
    idx = SearchIndexer(store, "gate", TaggedEmbedder(model), object_interval_seconds=1,
                        scene_interval_seconds=1000, blocking=True)
    frame = np.full((240, 320, 3), 90, np.uint8)
    track = Track(id=1, box=np.array([10, 10, 80, 200], float), score=0.9,
                  label_votes={"person": 1.0})
    idx.update(frame, [track], 1000.0)
    idx.close()
    models = {r[0] for r in store._db.execute("SELECT model FROM sightings")}
    assert models == {stored}


def test_text_search_skips_other_model_vectors(store, caplog):
    t = NOW.timestamp() - 600
    # The old model's vector is the better match; it must not be compared at all.
    store.add_sightings([sighting(t, LEGACY_MODEL, 1, unit(1, 0, 0)),
                         sighting(t + 1, NEW, 2, unit(0.8, 0.6, 0))])
    with caplog.at_level("INFO"):
        engine = SearchEngine(store, TaggedClip(NEW))
    assert "1 older vectors" in caplog.text
    results = engine.text("person", now=NOW)["results"]
    assert [r["track_id"] for r in results] == [2]


def test_chat_search_tool_skips_other_model_vectors(store):
    from histor.tools import Toolbox

    t = NOW.timestamp() - 600
    store.add_sightings([sighting(t, LEGACY_MODEL, 1), sighting(t + 1, NEW, 2)])
    box = Toolbox(SearchEngine(store, TaggedClip(NEW), camera_names=["gate"]), now=NOW)
    out = box.search(query="person", when="today")
    assert [v["track_id"] for v in out["results"]] == [2]


def test_search_without_model_tag_uses_everything(store):
    class Untagged:
        def embed_query(self, text):
            return unit(1, 0, 0)

    t = NOW.timestamp() - 600
    store.add_sightings([sighting(t, LEGACY_MODEL, 1), sighting(t + 60, NEW, 2)])
    engine = SearchEngine(store, Untagged())
    assert engine.clip_model is None
    assert {r["track_id"] for r in engine.text("person", now=NOW)["results"]} == {1, 2}


# ------------------------------------------------------------------ the real exported model

CLIP_FILES = [ROOT / "models" / n for n in ("clip_image.onnx", "clip_text.onnx",
                                             "clip_bpe_vocab.txt.gz")]


@pytest.mark.skipif(not all(p.exists() for p in CLIP_FILES),
                    reason="uv run --group export -m histor.engine.export_clip")
def test_exported_model_matches_text_to_images():
    enc = ClipEncoder(*CLIP_FILES)
    images = {name: cv2.imread(str(ROOT / "data" / name))
              for name in ("dog.jpg", "messi5.jpg", "basketball2.png")}
    assert all(im is not None for im in images.values())
    vectors = enc.embed_images(list(images.values()))
    assert vectors.shape == (3, 512)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4)
    names = list(images)
    for query, expected in (("a dog", "dog.jpg"), ("people playing basketball", "basketball2.png"),
                            ("a footballer kicking a ball", "messi5.jpg")):
        scores = vectors @ enc.embed_query(query)
        assert names[int(np.argmax(scores))] == expected, (query, scores)
    # Same image twice gives the same vector; batching doesn't change it.
    single = enc.embed_images([images["dog.jpg"]])[0]
    assert float(single @ vectors[0]) > 0.999


def test_tag_model_writes_the_name_onnx_runtime_reads(tmp_path):
    onnx = pytest.importorskip("onnx")  # the `export` group
    import onnxruntime as ort
    from onnx import TensorProto, helper

    graph = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "g",
                              [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])])
    path = tmp_path / "m.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9),
              str(path))
    assert session_model(ort.InferenceSession(str(path))) == LEGACY_MODEL
    export_clip.tag_model(path, NEW)
    export_clip.tag_model(path, NEW)  # tagging twice keeps one entry
    assert session_model(ort.InferenceSession(str(path))) == NEW
    assert [p.key for p in onnx.load(str(path)).metadata_props] == ["clip_model"]
