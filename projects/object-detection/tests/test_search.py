"""Store, query parsing, indexing, alert sinks and the API, without the real ML models."""

from __future__ import annotations

import json
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from alerts import AlertManager, EmailSink
from indexer import SearchIndexer, faces_enabled
from query import parse_query
from search import SearchEngine, group_visits
from store import EventStore
from tracker import Track
from zones import ZoneEvent

NOW = datetime(2026, 9, 29, 15, 30)  # a Tuesday


def unit(*values: float) -> np.ndarray:
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def store(tmp_path):
    s = EventStore(tmp_path / "db.sqlite", retention_days=30, face_retention_days=7)
    yield s
    s.close()


# ------------------------------------------------------------------ query parsing

@pytest.mark.parametrize("text, description, since, until", [
    ("red shirt past friday", "red shirt", datetime(2026, 9, 25), datetime(2026, 9, 26)),
    ("search for red shirt last tuesday", "red shirt", datetime(2026, 9, 22), datetime(2026, 9, 23)),
    ("white van yesterday 6pm to 9pm", "white van", datetime(2026, 9, 28, 18),
     datetime(2026, 9, 28, 21)),
    ("man with a backpack between 2 and 4pm on friday", "man with a backpack",
     datetime(2026, 9, 25, 14), datetime(2026, 9, 25, 16)),
    ("person in the last 2 hours", "person", datetime(2026, 9, 29, 13, 30), NOW),
    ("truck after 10pm yesterday", "truck", datetime(2026, 9, 28, 22), datetime(2026, 9, 29)),
    ("dog 2026-09-20", "dog", datetime(2026, 9, 20), datetime(2026, 9, 21)),
    ("woman in black dress this morning", "woman in black dress", datetime(2026, 9, 29, 6),
     datetime(2026, 9, 29, 12)),
])
def test_parse_query(text, description, since, until):
    q = parse_query(text, now=NOW)
    assert (q.text, q.since, q.until) == (description, since, until)


def test_parse_query_cameras_and_no_time():
    q = parse_query("blue car on gate", now=NOW, camera_names=["gate", "shop"])
    assert q.cameras == ["gate"] and q.text == "blue car" and q.since is None


# ------------------------------------------------------------------ store

def test_store_search_filters_and_ranks(store):
    t0 = datetime(2026, 9, 25, 10).timestamp()
    store.add_sightings([
        {"camera": "gate", "time": t0, "kind": "object", "track_id": 1, "label": "person",
         "box": [0, 0, 10, 20], "embedding": unit(1, 0, 0)},
        {"camera": "gate", "time": t0 + 5, "kind": "object", "track_id": 2, "label": "person",
         "box": [0, 0, 10, 20], "embedding": unit(0.6, 0.8, 0)},
        {"camera": "shop", "time": t0 + 86400, "kind": "scene", "embedding": unit(1, 0.1, 0)},
    ])
    hits = store.search("sightings", unit(1, 0, 0))
    assert [h.track_id for h in hits][:2] == [1, None]
    hits = store.search("sightings", unit(1, 0, 0), since=t0, until=t0 + 60, cameras=["gate"])
    assert [h.track_id for h in hits] == [1, 2] and hits[0].score > 0.99
    assert store.search("sightings", unit(1, 0, 0), min_score=0.9, cameras=["gate"])[0].track_id == 1


def test_store_purge_respects_face_retention(store, tmp_path):
    now = datetime(2026, 9, 29).timestamp()
    thumb = store.save_thumb(np.zeros((50, 30, 3), np.uint8), "faces", "a/b", now - 10 * 86400)
    assert "/" not in thumb.split("/")[-1] and store.media_path(thumb) is not None
    store.add_faces([{"camera": "shop", "time": now - 10 * 86400, "thumb": thumb,
                      "embedding": unit(1, 0)}])
    store.add_sightings([{"camera": "shop", "time": now - 10 * 86400, "kind": "scene",
                          "embedding": unit(1, 0)}])
    deleted = store.purge(now)
    assert deleted["faces"] == 1 and deleted["sightings"] == 0  # 10 days: faces 7, rest 30
    assert store.media_path(thumb) is None
    assert store.media_path("../../etc/passwd") is None


# ------------------------------------------------------------------ indexer

class FakeEmbedder:
    def __init__(self):
        self.calls = 0

    def embed_images(self, images):
        self.calls += len(images)
        return np.stack([unit(float(im.mean()) + 1, 1) for im in images])


def track(tid, box, label="person"):
    return Track(id=tid, box=np.array(box, float), score=0.9, label_votes={label: 1.0})


def test_indexer_intervals_scene_tiles_and_no_faces_by_default(store):
    emb = FakeEmbedder()
    idx = SearchIndexer(store, "gate", emb, object_interval_seconds=2, scene_interval_seconds=5,
                        scene_tiles=2, blocking=True)
    frame = np.full((240, 320, 3), 100, np.uint8)
    for i in range(31):  # 3 s at 10 fps
        t = 1000 + i / 10
        idx.update(frame, [track(1, [10, 10, 60, 120]), track(2, [100, 10, 110, 20])], t)
    idx.close()
    counts = store.counts()
    # track 1 at t=0 and t=2 (track 2 is too small); one scene = 1 frame + 4 tiles
    assert counts["sightings"] == 2 + 5 and counts["faces"] == 0


def test_faces_enabled_rules():
    assert not faces_enabled(None, "shop")
    assert not faces_enabled({"enabled": False}, "shop")
    assert faces_enabled({"enabled": True}, "shop")
    assert faces_enabled({"enabled": True, "cameras": ["shop"]}, "shop")
    assert not faces_enabled({"enabled": True, "cameras": ["shop"]}, "gate")


def test_group_visits_merges_track(store):
    t0 = 1_000_000.0
    store.add_sightings([{"camera": "gate", "time": t0 + i, "kind": "object", "track_id": 7,
                          "label": "person", "embedding": unit(1, i * 0.01)} for i in range(5)])
    visits = group_visits(store.search("sightings", unit(1, 0)))
    assert len(visits) == 1 and visits[0].matches == 5
    assert visits[0].last_seen - visits[0].first_seen == 4


# ------------------------------------------------------------------ alerts

class _Hook(BaseHTTPRequestHandler):
    received: list[dict] = []

    def do_POST(self):  # noqa: N802
        body = self.rfile.read(int(self.headers["Content-Length"]))
        _Hook.received.append({"json": json.loads(body), "auth": self.headers.get("Authorization")})
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass


def test_alerts_store_snapshot_webhook(store, monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), _Hook)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("HOOK_TOKEN", "secret")
    alerts = AlertManager(
        console=False, source="gate", store=store,
        webhook={"url": f"http://127.0.0.1:{server.server_port}/hook",
                 "headers": {"Authorization": "Bearer ${HOOK_TOKEN}"}},
    )
    frame = np.zeros((120, 160, 3), np.uint8)
    event = ZoneEvent(12.0, "cross", "gate_line", 3, "person", direction="in")
    emitted = alerts.handle([event], frame, [track(3, [10, 10, 50, 100])], wall_time=1.79e9)
    alerts.close()
    server.shutdown()
    assert emitted == [event]
    stored = store.list_events()
    assert stored[0]["rule"] == "gate_line" and store.media_path(stored[0]["snapshot"])
    hook = _Hook.received[-1]
    assert hook["auth"] == "Bearer secret"
    assert hook["json"]["camera"] == "gate" and hook["json"]["direction"] == "in"


def test_email_sink_builds_message(monkeypatch):
    sent = {}

    class FakeSMTP:
        def __init__(self, host, port, timeout):
            sent["host"] = (host, port)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def starttls(self):
            sent["tls"] = True

        def login(self, user, password):
            sent["login"] = user

        def send_message(self, msg):
            sent["msg"] = msg

    monkeypatch.setattr("smtplib.SMTP", FakeSMTP)
    sink = EmailSink(host="smtp.example.com", sender="a@x.com", to="b@y.com", username="u",
                     password="p")
    sink.send({"type": "dwell", "rule": "fire_lane", "label": "car", "camera": "yard",
               "iso_time": "2026-09-29T10:00:00", "track_id": 4, "dwell_seconds": 61.0},
              b"\xff\xd8jpeg")
    msg = sent["msg"]
    assert sent["tls"] and sent["login"] == "u" and "fire_lane" in msg["Subject"]
    assert [p.get_filename() for p in msg.iter_attachments()] == ["snapshot.jpg"]


# ------------------------------------------------------------------ API

class FakeClip:
    def embed_query(self, text):
        return unit(1, 0) if "red" in text else unit(0, 1)


class FakeFaces:
    def best_face_print(self, image):
        return (unit(1, 0), None) if image.mean() > 100 else None


def test_api_text_and_face_search(tmp_path):
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from server import create_app

    assert fastapi
    config = {"store": {"path": str(tmp_path / "db.sqlite")}, "cameras": {"shop": {}, "gate": {}}}
    friday = datetime(2026, 9, 25, 11).timestamp()
    app = create_app(config, clip=FakeClip(), faces=FakeFaces())
    s: EventStore = app.state.store
    s.add_sightings([
        {"camera": "shop", "time": friday, "kind": "object", "track_id": 1, "label": "person",
         "embedding": unit(1, 0.05)},
        {"camera": "gate", "time": friday, "kind": "object", "track_id": 2, "label": "person",
         "embedding": unit(0.1, 1)},
    ])
    s.add_faces([{"camera": "shop", "time": friday, "track_id": 1, "embedding": unit(1, 0.1)},
                 {"camera": "gate", "time": friday, "track_id": 2, "embedding": unit(0, 1)}])
    client = TestClient(app)

    assert client.get("/health").json()["counts"]["faces"] == 2
    r = client.get("/search", params={"q": "red shirt"}).json()
    assert r["results"][0]["camera"] == "shop"
    r = client.get("/search", params={"q": "red shirt on gate"}).json()
    assert r["query"]["cameras"] == ["gate"] and all(x["camera"] == "gate" for x in r["results"])

    import cv2
    photo = cv2.imencode(".jpg", np.full((64, 64, 3), 200, np.uint8))[1].tobytes()
    dark = cv2.imencode(".jpg", np.zeros((64, 64, 3), np.uint8))[1].tobytes()
    r = client.post("/search/face", files={"photo": ("p.jpg", photo, "image/jpeg")},
                    data={"when": "2026-09-25"}).json()
    assert r["face_found"] and [x["camera"] for x in r["results"]] == ["shop"]
    assert r["answer"].startswith("Yes")
    r = client.post("/search/face", files={"photo": ("p.jpg", dark, "image/jpeg")}).json()
    assert r["face_found"] is False
    assert client.get("/media/../../secret").status_code == 404
    engine: SearchEngine = app.state.engine
    assert engine.camera_names == ["gate", "shop"]
