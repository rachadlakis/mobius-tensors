"""Smoke test of a live IP camera, played by ffmpeg: data/vtest.avi as H.264 with a smoke-alarm
beep on its audio track, served in real time as three HTTP streams (the camera's low-res sub
stream, its full-resolution main stream, and the main stream again for the microphone: ffmpeg
serves one client per stream).

The recorder reads them as it would a camera: detection on the sub stream, the main stream
recorded to disk with its keyframes decoded for hi-res crops, the audio classified for sound
events, and alerts sent with a video clip cut from the recording. Needs ffmpeg.

    uv run pytest -m smoke tests/smoke/test_live_camera.py
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import time
from pathlib import Path

import cv2
import pytest
from smoke_kit import DATA, MISSING, MODELS, TOKEN

FFMPEG = shutil.which("ffmpeg")
pytestmark = [pytest.mark.smoke,
              pytest.mark.skipif(bool(MISSING), reason=f"needs models/ and data/: {MISSING}"),
              pytest.mark.skipif(FFMPEG is None, reason="needs ffmpeg")]
SECONDS = 30  # how long the recorder runs
BEEPS = "aevalsrc=0.5*sin(2*PI*3100*t)*lt(mod(t\\,1)\\,0.5):s=16000:d=60"  # 0.5 s on / off


def ffmpeg(*args: str) -> None:
    assert FFMPEG is not None
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *args], check=True,
                   timeout=300)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def camera(tmp_path_factory):
    """URLs of the sub stream, the main stream and the audio, each served once, in real time."""
    tmp = tmp_path_factory.mktemp("camera")
    main, sub = tmp / "main.mp4", tmp / "sub.mp4"
    ffmpeg("-i", str(DATA / "vtest.avi"), "-f", "lavfi", "-i", BEEPS, "-map", "0:v",
           "-map", "1:a", "-t", "45", "-c:v", "libx264", "-preset", "veryfast", "-g", "10",
           "-pix_fmt", "yuv420p", "-c:a", "aac", str(main))  # 768x576, a keyframe a second
    ffmpeg("-i", str(main), "-map", "0:v", "-vf", "scale=384:288", "-c:v", "libx264",
           "-preset", "veryfast", "-g", "10", str(sub))
    servers, urls = [], []
    for source, name in ((sub, "sub"), (main, "main"), (main, "audio")):
        url = f"http://127.0.0.1:{free_port()}/{name}.ts"
        servers.append(subprocess.Popen(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-re", "-i", str(source), "-c",
             "copy", "-f", "mpegts", "-listen", "1", url],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        urls.append(url)
    time.sleep(1.0)  # listening
    yield urls
    for server in servers:
        server.kill()
        server.wait()


def live_config(tmp: Path, fakes) -> dict:
    def m(name: str) -> str:
        return str(MODELS / name)

    return {
        "model": {"path": m("yolox_nano.onnx"), "conf_threshold": 0.25,
                  "class_agnostic_nms": True},
        "cameras": {"live": {}},
        "store": {"path": str(tmp / "live.db"), "media_dir": str(tmp / "media")},
        "recording": {"enabled": True, "dir": str(tmp / "rec"), "segment_seconds": 10,
                      "hires_crops": True, "min_free_gb": None},
        "search": {"enabled": True,
                   "clip": {"image": m("clip_image.onnx"), "text": m("clip_text.onnx"),
                            "vocab": m("clip_bpe_vocab.txt.gz")},
                   "object_interval_seconds": 1, "scene_interval_seconds": 3},
        "pose": {"enabled": True, "model": m("pose_rtmpose_t.onnx"), "min_box_height": 40},
        "sounds": {"enabled": True, "model": m("sound_yamnet.onnx"),
                   "classes": m("sound_yamnet_classes.csv")},
        "zones": [{"name": "crowd", "classes": ["person"], "max_count": 3, "count_seconds": 2}],
        "alerts": {"console": False, "channels": [
            {"type": "webhook", "url": fakes.webhook.url + "/live"},
            {"type": "telegram", "token": TOKEN, "chat_id": 7, "api_url": fakes.telegram.url,
             "images": True, "video": True, "video_seconds": 4, "events": ["sound", "crowd"],
             "cooldown_seconds": 60}]},
    }


def test_live_camera_records_hires_sound_and_clips(camera, fakes, tmp_path):
    from fastapi.testclient import TestClient

    from histor.engine.pipeline import Pipeline
    from histor.engine.server import create_app
    from histor.engine.stream import iter_frames

    sub_url, main_url, audio_url = camera
    config = live_config(tmp_path, fakes)
    pipeline = Pipeline.from_config(config, source="live", live=True, main_url=main_url,
                                    audio_source=audio_url)
    assert pipeline.mainstream is not None and pipeline.sounds is not None
    frames, hires = 0, False
    deadline = time.monotonic() + SECONDS
    try:
        for frame in iter_frames(sub_url):
            pipeline.process(frame.image, frame.timestamp)
            frames += 1
            hires = hires or bool(pipeline.indexer and pipeline.indexer._hires_tracks)
            if time.monotonic() > deadline:
                break
        keyframe = pipeline.mainstream.latest()
    finally:
        pipeline.close()
    assert frames > 50 and frame.image.shape[:2] == (288, 384)  # the sub stream
    assert keyframe is not None and keyframe.image.shape[:2] == (576, 768)  # the main stream
    assert hires  # tracks matched in a keyframe: crops from full resolution

    store = pipeline.store
    assert store is not None
    events = store.list_events(limit=1000)
    sounds = [e for e in events if e["type"] == "sound"]
    assert sounds and sounds[0]["rule"] == "smoke_alarm", [e["type"] for e in events]
    assert any(e["type"] == "crowd" for e in events)
    recordings = sorted((tmp_path / "rec" / "live").glob("*.mp4"))
    assert recordings and cv2.VideoCapture(str(recordings[0])).read()[0]

    deadline = time.monotonic() + 30  # clips go out once recorded (a few seconds after)
    while not fakes.telegram.sent("sendVideo") and time.monotonic() < deadline:
        time.sleep(0.5)
    videos = fakes.telegram.sent("sendVideo")
    assert videos and videos[0]["files"]["video"][4:8] == b"ftyp"  # an MP4
    assert any(p["fields"]["chat_id"] == "7" for p in fakes.telegram.sent("sendPhoto"))
    assert any(h["body"]["type"] == "sound" for h in fakes.webhook.requests
               if h["path"] == "/live")

    store.close()
    with TestClient(create_app(config)) as client:
        listed = client.get("/events", params={"type": "sound"}).json()["events"]
        assert listed[0]["video_url"]  # the moment in the recording
        video = client.get(listed[0]["video_url"].split("#")[0])
        assert video.status_code == 200 and video.headers["content-type"] == "video/mp4"
