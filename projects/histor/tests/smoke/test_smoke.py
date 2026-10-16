"""Smoke tests: every entry point end to end, with the real models in models/ and
data/vtest.avi. Slow (a few minutes on a laptop CPU), so not in the default run:

    uv run pytest -m smoke

One site is recorded once (`histor.record`, every feature on: conftest.py), then each command,
API and bot runs against it. Language models, Telegram and webhooks are local fakes
(smoke_kit.py): nothing leaves the machine. A live camera (main stream, sounds, video clips:
ffmpeg) is test_live_camera.py; without ffmpeg the recorder must carry on, which is checked
here. Not covered: export_clip / export_pose (they need the `export` group's PyTorch).
"""

from __future__ import annotations

import functools
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import wave
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from smoke_kit import DATA, MISSING, MODELS, PROJECT, TOKEN, make_clip, run

pytestmark = [pytest.mark.smoke,
              pytest.mark.skipif(bool(MISSING), reason=f"needs models/ and data/: {MISSING}")]


def noop(*args, **kwargs) -> None:  # a Gradio progress callback outside Gradio
    pass


@pytest.fixture(scope="module")
def files(site):
    """site.tmp over HTTP: a "network camera" for the camera tools."""
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(site.tmp))
    handler.log_message = noop  # type: ignore[attr-defined]
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_port
    server.shutdown()
    server.server_close()


@pytest.fixture(scope="module")
def short_clip(tmp_path_factory):
    return make_clip(tmp_path_factory.mktemp("short") / "short.avi", frames=40, black=0)


# ------------------------------------------------------------------ recording

def test_record_stores_every_kind_of_fact_and_sends_alerts(site, fakes):
    from histor.engine.store import EventStore

    r = site.record
    assert r.returncode == 0, r.stderr[-4000:]
    assert "Store " in r.stdout
    # worker threads log their failures instead of raising: none allowed but missing ffmpeg
    errors = [line for line in r.stderr.splitlines() if line.startswith(("ERROR", "Traceback"))]
    assert all("ffmpeg" in line for line in errors), errors
    store = EventStore(site.store)
    try:
        counts = store.counts()
        assert counts["sightings"] > 50 and counts["poses"] > 20 and counts["events"] > 0, counts
        assert len(store.embedding_models()) == 1  # every vector tagged with the CLIP model
        events = store.list_events(limit=10000)
        types = {e["type"] for e in events}
        assert {"enter", "crowd", "tamper"} <= types, types
        tamper = [e for e in events if e["type"] == "tamper"]
        assert tamper[0]["rule"] in ("blocked", "dark")  # the black frames at the end
        checked = [e for e in events if e["type"] in ("enter", "crowd")]
        assert checked and all(e.get("verified") in ("yes", "skipped") for e in checked)
        assert all(store.media_path(e["snapshot"]) for e in events if e.get("snapshot"))
        kinds = {row[0] for row in store._db.execute("SELECT DISTINCT kind FROM sightings")}
        assert kinds == {"object", "scene", "tile"}
    finally:
        store.close()

    lines = (site.tmp / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(events)
    hooks = [h["body"] for h in fakes.webhook.requests if h["path"] == "/hook"]
    assert any(b["type"] == "tamper" for b in hooks)
    assert all(b["camera"] == "street" and b["iso_time"] for b in hooks)
    photos = fakes.telegram.sent("sendPhoto")  # crowd / tamper only, with the snapshot
    assert photos and all(p["files"]["photo"][:2] == b"\xff\xd8" for p in photos)
    assert any("tamper" in p["fields"]["caption"] for p in photos)
    vision = [q for q in fakes.models.requests if not q.get("tools")]
    assert vision and "alarms" in vision[0]["messages"][0]["content"]  # alert checking ran
    if shutil.which("ffmpeg") is None:  # sounds read through ffmpeg: logged, recording goes on
        assert "ffmpeg not found" in r.stderr


def test_engine_main_saves_annotated_image_and_video(site, tmp_path):
    config = {"model": {"path": str(MODELS / "yolox_nano.onnx")}}
    path = tmp_path / "plain.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    r = run("-m", "histor.engine.main", "--config", path, "--source", DATA / "dog.jpg",
            "--save", tmp_path / "dog.jpg")
    assert r.returncode == 0, r.stderr[-2000:]
    assert "dog" in r.stdout and (tmp_path / "dog.jpg").stat().st_size > 0

    zones = {**config, "zones": site.config["zones"][:2]}  # no store: alerts on the console
    path.write_text(yaml.safe_dump(zones), encoding="utf-8")
    r = run("-m", "histor.engine.main", "--config", path, "--source", site.clip,
            "--save", tmp_path / "out.mp4")
    assert r.returncode == 0, r.stderr[-2000:]
    assert (tmp_path / "out.mp4").stat().st_size > 0 and "ENTER junction" in r.stdout


# ------------------------------------------------------------------ asking

def test_ask_cli(site):
    r = run("-m", "histor.ask", "--config", site.config_path, "--top", "3",
            "person walking on road")
    assert r.returncode == 0, r.stderr[-2000:]
    assert "looking for 'person walking' on ['street']" in r.stdout  # "road" is an alias
    assert " 1. " in r.stdout and ".jpg" in r.stdout
    r = run("-m", "histor.ask", "--config", site.config_path, "person 30/2")  # not a date
    assert r.returncode == 0 and "looking for 'person 30/2'" in r.stdout


def test_chat_cli_runs_tools_and_looks(site, fakes):
    r = run("-m", "histor.chat", "--config", site.config_path,
            "how many people walked by? look at one")
    assert r.returncode == 0, r.stderr[-2000:]
    assert "-> objects" in r.stderr and "-> look" in r.stderr
    assert "SMOKE ANSWER after 2 lookups" in r.stdout and "Two people walk past." in r.stdout
    assert "Evidence:" in r.stdout


def test_chat_cli_with_a_photo(site):
    r = run("-m", "histor.chat", "--config", site.config_path, "--photo", site.face,
            "did this person come today?")
    assert r.returncode == 0, r.stderr[-2000:]
    assert "Attached as photo1" in r.stdout and "-> face" in r.stderr
    assert "SMOKE ANSWER after 1 lookups" in r.stdout
    r = run("-m", "histor.chat", "--config", site.config_path, "--photo", DATA / "messi5.jpg",
            "did this person come today?")
    assert r.returncode == 0 and "too small" in r.stdout  # refused with the reason


def test_report_cli(site, fakes):
    r = run("-m", "histor.report", "--config", site.config_path, "--now", "smoke")
    assert r.returncode == 0, r.stderr[-2000:]
    assert "# smoke: hourly report" in r.stdout and "## Counts" in r.stdout
    assert "SMOKE ANSWER after 2 lookups" in r.stdout  # objects + alerts, then the summary
    (saved,) = (site.store.parent / "reports" / "smoke").glob("*.json")
    report = json.loads(saved.read_text(encoding="utf-8"))
    assert report["sent"] == {"1:webhook": "ok"} and report["error"] is None
    (street,) = report["facts"]["cameras"]
    assert street["recorded"] and street["counts"].get("person", 0) > 0
    sent = [h["body"] for h in fakes.webhook.requests if h["path"] == "/report"]
    assert sent and sent[-1]["kind"] == "report" and sent[-1]["summary"]
    assert "smoke" in run("-m", "histor.report", "--config", site.config_path, "--list").stdout
    shown = run("-m", "histor.report", "--config", site.config_path, "--show", "smoke").stdout
    assert shown.startswith("# smoke: hourly report")


# ------------------------------------------------------------------ API, MCP, Telegram

def test_api_every_endpoint(site):
    from fastapi.testclient import TestClient

    from histor.engine.server import create_app

    app = create_app(site.config)
    with TestClient(app) as c:
        health = c.get("/health").json()
        assert health["text_search"] and health["face_search"]
        assert health["counts"]["sightings"] > 0
        assert c.get("/cameras").json() == {"cameras": ["street"]}
        events = c.get("/events", params={"limit": 1000}).json()["events"]
        assert events and {e["type"] for e in c.get(
            "/events", params={"type": "tamper"}).json()["events"]} == {"tamper"}
        shot = next(e for e in events if e.get("snapshot_url"))
        image = c.get(shot["snapshot_url"])
        assert image.status_code == 200 and image.headers["content-type"] == "image/jpeg"
        found = c.get("/search", params={"q": "a person walking", "limit": 5}).json()
        assert found["results"] and c.get(found["results"][0]["thumb_url"]).status_code == 200
        assert c.get("/search", params={"q": "person 30/2"}).status_code == 200
        with open(site.face, "rb") as f:
            face = c.post("/search/face", files={"photo": ("face.jpg", f, "image/jpeg")},
                          data={"when": "today"}).json()
        assert face["face_found"] and "answer" in face
        with open(DATA / "dog.jpg", "rb") as f:
            dog = c.post("/search/face", files={"photo": ("dog.jpg", f, "image/jpeg")}).json()
        assert dog["face_found"] is False and "No face" in dog["answer"]
        assert c.post("/search/face", files={"photo": ("x.jpg", b"not an image")}).status_code \
            == 400
        people = c.get("/people", params={"posture": "standing"}).json()
        assert people["total"] > 0 and people["results"][0]["thumb_url"]
        assert c.get("/people", params={"arms": "down"}).status_code == 400
        assert c.get("/plates", params={"plate": "AB123CD"}).status_code == 200
        status = c.get("/status").json()
        assert [cam["camera"] for cam in status["cameras"]] == ["street"]
        assert c.post("/maintenance/purge").json()["deleted"]["sightings"] == 0
        assert "Histor" in c.get("/").text
        assert c.get("/recordings/street/none.mp4").status_code == 404
        assert c.get("/media/../../site.yaml").status_code == 404


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_api_server_cli(site):
    refused = run("-m", "histor.engine.server", "--config", site.config_path,
                  "--host", "0.0.0.0")
    assert refused.returncode != 0 and "without a login" in refused.stderr
    port = free_port()
    proc = subprocess.Popen([sys.executable, "-m", "histor.engine.server", "--config",
                             str(site.config_path), "--port", str(port)], cwd=PROJECT,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 120
        while True:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                    health = json.loads(r.read())
                break
            except OSError:
                assert proc.poll() is None and time.monotonic() < deadline, "server didn't start"
                time.sleep(0.5)
        assert health["status"] == "ok" and health["counts"]["events"] > 0
    finally:
        proc.terminate()
        proc.wait(timeout=30)


def test_mcp_server_over_stdio(site):
    import anyio
    from mcp import Client, StdioServerParameters

    params = StdioServerParameters(
        command=sys.executable, args=["-m", "histor.mcp_server", "--config",
                                      str(site.config_path)],
        cwd=str(PROJECT), env={**os.environ, "PYTHONIOENCODING": "utf-8"})

    async def go():
        async with Client(params, read_timeout_seconds=120) as client:
            listed = await client.list_tools()
            objects = await client.call_tool("objects", {"labels": ["person"]})
            face = await client.call_tool("face", {"photo_path": str(site.face)})
            outside = await client.call_tool("face", {"photo_path": str(DATA / "dog.jpg")})
            return listed, objects, face, outside

    listed, objects, face, outside = anyio.run(go)
    names = {t.name for t in listed.tools}
    assert {"objects", "search", "alerts", "people", "plates", "look", "face"} <= names
    out = json.loads(objects.content[0].text)
    assert "site_time" in out and out["total"] > 0
    assert not face.is_error and "checked" in json.loads(face.content[0].text)
    assert outside.is_error and "only photos in these folders" in outside.content[0].text


def test_telegram_bot(site, fakes):
    from histor.ask import build_engine
    from histor.chat import model_config, system_prompt
    from histor.engine.telegram import TelegramAPI
    from histor.llm import create_backend
    from histor.telegram_bot import TelegramBot
    from histor.tools import Toolbox

    engine = build_engine(site.config)
    toolbox = Toolbox.from_config(engine, site.config)
    model = model_config(site.config, "fake")
    bot = TelegramBot(TelegramAPI(TOKEN, fakes.telegram.url), toolbox,
                      lambda: create_backend(model, system_prompt(toolbox), toolbox.tools),
                      {"42"}, images=True, cameras=["street"], face_search=True)
    tg = fakes.telegram
    try:
        def say(update: dict) -> list[str]:
            tg.requests.clear()
            bot.handle({"update_id": 1, "message": {"chat": {"id": 42}, **update}})
            return [r["fields"]["text"] for r in tg.sent("sendMessage")]

        assert "/status" in say({"text": "/start"})[0]
        assert say({"text": "/status"})[0].startswith("street: ")
        assert say({"text": "how many people walked by?"})[0].startswith("SMOKE ANSWER")
        assert tg.sent("sendPhoto") or tg.sent("sendMediaGroup")  # evidence, images on
        tg.file = site.face.read_bytes()
        (reply,) = say({"photo": [{"file_id": "big"}], "caption": "today"})
        assert reply.startswith(("Yes", "Possibly", "No "))
        tg.requests.clear()
        bot.handle({"update_id": 2, "message": {"chat": {"id": 99}, "text": "hi"}})  # stranger
        assert tg.requests == []
    finally:
        toolbox.close()
        engine.store.close()


# ------------------------------------------------------------------ UIs

def test_ui_builds_and_records(site, tmp_path):
    pytest.importorskip("gradio")
    from histor.ui import build_ui

    config = {**site.config, "alerts": {"console": False},  # no channels, no checking
              "store": {"path": str(tmp_path / "ui.db"), "media_dir": str(tmp_path / "media")}}
    demo = build_ui(config, "street", str(site.clip))
    recorder = demo.recorder
    try:
        recorder.start(str(site.clip))
        deadline = time.monotonic() + 300
        while recorder.running and time.monotonic() < deadline:
            time.sleep(0.5)
        assert not recorder.running and recorder.frame_no > 100, recorder.status
        assert recorder.status == "The camera stopped sending frames."  # the file ended
        assert recorder.frame is not None and recorder.frame.shape[2] == 3
        assert demo.toolbox.engine.store.counts()["sightings"] > 0
    finally:
        recorder.stop()
        demo.toolbox.close()
        if demo.reporter:
            demo.reporter.close()


def test_bench(site, short_clip):
    pytest.importorskip("gradio")
    from histor.engine import app as bench

    names = bench.available_models()
    assert "yolox_nano" in names and not [n for n in names if n.startswith(bench.NOT_DETECTORS)]
    dog = cv2.cvtColor(cv2.imread(str(DATA / "dog.jpg")), cv2.COLOR_BGR2RGB)
    _, table, info = bench.detect_image(dog, "yolox_nano", 0.35, 0.45, [])
    assert "dog" in [row[0] for row in table] and "yolox_nano" in info
    out, info = bench.detect_video(str(short_clip), "yolox_nano", 0.35, 0.45, ["person"],
                                   progress=noop)
    assert Path(out).stat().st_size > 0 and "person" in info

    result = bench.faces_detect([str(site.face), str(DATA / "messi5.jpg")])
    faces, pairs = result[0], result[7]
    assert len(faces) == 2 and faces[1]["problems"]  # the small one is flagged
    assert pairs[0][3] == "likely same person"  # the same face, at two sizes

    shapes = [{"name": "junction", "type": "zone",
               "points": [[560, 190], [767, 150], [767, 420], [560, 350]],
               "classes": ["person"], "dwell_seconds": 2.0, "direction": "both"},
              {"name": "path", "type": "line", "points": [[350, 190], [350, 330]],
               "classes": ["person"], "dwell_seconds": None, "direction": "both"}]
    assert "junction" in bench.shapes_yaml(shapes, dog)
    out, rows, summary = bench.zones_run(str(short_clip), shapes, "yolox_nano", 0.25, 0.45, [],
                                         progress=noop)
    assert Path(out).stat().st_size > 0 and "**junction** (zone)" in summary

    html, _ = bench.live_start(str(short_clip), "yolox_nano", 0.35, 0.45, [])
    try:
        assert "/live/stream.mjpg" in html
        frame = next(bench.LIVE.mjpeg())
        assert frame.startswith(b"--frame\r\nContent-Type: image/jpeg") and b"\xff\xd8" in frame
    finally:
        bench.live_stop()


def test_live_capture_and_camera_tools(site, files, tmp_path):
    cams = tmp_path / "cams.yaml"
    cams.write_text(yaml.safe_dump({"cameras": {
        "gate": {"vendor": "hikvision", "host": "10.0.0.5", "username": "admin",
                 "password": "${SMOKE_CAM_PASSWORD}", "about": {"description": "البوابة"}},
        "lobby": {"url": f"http://127.0.0.1:{files}/clip.avi"}}}, allow_unicode=True),
        encoding="utf-8")
    listed = run("-m", "histor.engine.cameras", "--config", cams, "list",
                 env={"SMOKE_CAM_PASSWORD": "p@ss"})
    assert listed.returncode == 0, listed.stderr[-2000:]
    assert "admin:***@10.0.0.5:554/Streaming/Channels/101" in listed.stdout
    assert "p@ss" not in listed.stdout and "p%40ss" not in listed.stdout
    scan = run("-m", "histor.engine.cameras", "scan", "127.0.0.1/32", "--ports", files)
    assert f"127.0.0.1:{files} open" in scan.stdout
    snap = tmp_path / "snap.jpg"
    tested = run("-m", "histor.engine.cameras", "--config", cams, "test", "lobby",
                 "--seconds", "1", "--save", snap, timeout=120)
    assert tested.returncode == 0, tested.stdout + tested.stderr[-2000:]
    assert "OK: 768x576" in tested.stdout and cv2.imread(str(snap)) is not None


def test_download_model_finds_what_is_there():
    for name in ("yolox_nano", "faces", "plates", "sounds", "pose_rtmpose_t"):
        r = run("-m", "histor.engine.download_model", name)
        assert r.returncode == 0 and "Already present" in r.stdout, (name, r.stderr[-1000:])


# ------------------------------------------------------------------ each model on its own

def test_clip_ranks_the_right_words_first():
    from histor.engine.clip_onnx import ClipEncoder

    clip = ClipEncoder(MODELS / "clip_image.onnx", MODELS / "clip_text.onnx",
                       MODELS / "clip_bpe_vocab.txt.gz")
    dog = clip.embed_images([cv2.imread(str(DATA / "dog.jpg"))])[0]
    assert dog @ clip.embed_query("a dog") > dog @ clip.embed_query("a red sports car")


@pytest.mark.parametrize("model", ["pose_vitpose_b.onnx", "pose_rtmpose_t.onnx"])
def test_pose_models(model):
    from histor.engine.detector import YoloxOnnxDetector
    from histor.engine.pose import POSTURES, PoseEstimator, describe

    image = cv2.imread(str(DATA / "basketball2.png"))
    boxes = [np.array([d.x1, d.y1, d.x2, d.y2]) for d in
             YoloxOnnxDetector(MODELS / "yolox_nano.onnx", classes=["person"]).detect(image)]
    assert boxes
    keypoints = PoseEstimator(MODELS / model).estimate(image, boxes)
    assert len(keypoints) == len(boxes) and keypoints[0].shape == (17, 3)
    facts = [describe(kp, image) for kp in keypoints]
    assert all(f.posture in POSTURES for f in facts) and max(f.visible for f in facts) >= 10


def test_plate_models_read_a_plate():
    from histor.engine.plates import PlateDetector, PlateReader

    plate = np.full((60, 240, 3), 255, np.uint8)
    cv2.rectangle(plate, (1, 1), (238, 58), (0, 0, 0), 2)
    cv2.putText(plate, "AB123CD", (12, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
    (read,) = PlateReader(MODELS / "plate_ocr_cct_s_v2.onnx",
                          MODELS / "plate_ocr_cct_s_v2.yaml").read([plate])
    assert read.text.isalnum() and 0 <= read.confidence <= 1
    car = np.full((300, 400, 3), 90, np.uint8)
    car[200:260, 80:320] = plate
    found = PlateDetector(MODELS / "plate_yolov9_t_384.onnx").detect(car)
    assert all(len(b) == 5 for b in found)


def test_sound_model_on_a_wav(tmp_path):
    from histor.engine.sound import SAMPLE_RATE, SoundClassifier, SoundListener, SoundRules

    t = np.arange(6 * SAMPLE_RATE) / SAMPLE_RATE
    beeps = 0.5 * np.sin(2 * np.pi * 3100 * t) * ((t % 1.0) < 0.5)  # a smoke-alarm pattern
    wav = tmp_path / "alarm.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes((beeps * 32767).astype(np.int16).tobytes())
    classifier = SoundClassifier(MODELS / "sound_yamnet.onnx", MODELS / "sound_yamnet_classes.csv")
    listener = SoundListener(classifier, SoundRules(classifier.classes), str(wav), live=False)
    listener.close()
    assert listener.error is None and listener.seconds >= 4
    assert all(e.group and 0 <= e.time <= 6 for e in listener.take_events())


FIRE_VIDEO = DATA / "test_videos" / "fire_loop.mp4"


@pytest.mark.skipif(not FIRE_VIDEO.exists() or not (MODELS / "fire_smoke_yolov9s_640.onnx").exists(),
                    reason="training/fire_smoke/make_test_video.py and download_model fire_smoke")
@pytest.mark.parametrize("size", ["s", "t"])
def test_fire_alerts_on_a_video(tmp_path, size):
    """Held-out fire and smoke pictures in a video: each alerts within 3 s of coming into view,
    once, with a snapshot; the clean stretches raise nothing."""
    from histor.engine.pipeline import Pipeline

    timeline = json.loads((DATA / "test_videos" / "fire_loop.json").read_text(encoding="utf-8"))
    config = {"model": {"path": str(MODELS / "yolox_nano.onnx")},
              "store": {"path": str(tmp_path / "db.sqlite")},
              "fire": {"enabled": True,
                       "model": {"path": str(MODELS / f"fire_smoke_yolov9{size}_640.onnx")}},
              "alerts": {"console": False}}
    pipe = Pipeline.from_config(config, source="kitchen", live=False, start_time=0.0)
    cap = cv2.VideoCapture(str(FIRE_VIDEO))
    fps, i = cap.get(cv2.CAP_PROP_FPS) or 10, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        pipe.process(frame, i / fps)
        i += 1
    pipe.close()
    rows = pipe.store._db.execute("SELECT time, rule, snapshot FROM events WHERE type = 'fire' "
                                  "ORDER BY time").fetchall()
    pipe.store.close()
    for kind in ("fire", "smoke"):
        start = min(t["from"] for t in timeline if t["kind"] == kind)
        hits = [r for r in rows if r[1] == kind]
        assert len(hits) == 1 and start <= hits[0][0] <= start + 3 and hits[0][2], (kind, rows)
    assert len(rows) == 2  # nothing on the clean stretches
