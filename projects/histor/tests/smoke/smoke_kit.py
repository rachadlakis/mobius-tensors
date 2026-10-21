"""What the smoke tests share: a site recorded from data/vtest.avi with every feature on, and
local stand-ins for what Histor talks to over the network.

    FakeModels    an OpenAI-compatible server: the chat model (a fixed plan of tool calls picked
                  from the question's words, then an answer quoting the last result) and the
                  vision model (`look`, alert checking)
    FakeTelegram  the Bot API: records each call, serves one photo for downloads
    FakeWebhook   records every request: method, path and the JSON body (None when empty)
    FakePing      a dead-man's switch (Healthchecks.io style): records path, method and time

Nothing leaves this machine. The real ONNX models in models/ do the rest.
"""

from __future__ import annotations

import email
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import yaml

PROJECT = Path(__file__).resolve().parents[2]
MODELS = PROJECT / "models"
DATA = PROJECT / "data"
TOKEN = "123456:SMOKE-token"
NEEDED = ["yolox_nano.onnx", "clip_image.onnx", "clip_text.onnx", "clip_bpe_vocab.txt.gz",
          "face_detection_yunet_2023mar.onnx", "face_recognition_sface_2021dec.onnx",
          "pose_rtmpose_t.onnx", "pose_vitpose_b.onnx", "plate_yolov9_t_384.onnx",
          "plate_ocr_cct_s_v2.onnx", "plate_ocr_cct_s_v2.yaml", "sound_yamnet.onnx",
          "sound_yamnet_classes.csv"]


MISSING = [n for n in NEEDED if not (MODELS / n).exists()] + [
    n for n in ("vtest.avi", "messi5.jpg", "dog.jpg", "basketball2.png") if not (DATA / n).exists()]


def run(*args: Any, timeout: float = 300, env: dict[str, str] | None = None
        ) -> subprocess.CompletedProcess[str]:
    """`python <args>` in the project folder, as a user would run it (UTF-8 output)."""
    return subprocess.run([sys.executable, *map(str, args)], cwd=PROJECT, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=timeout,
                          env={**os.environ, "PYTHONIOENCODING": "utf-8", **(env or {})})


# ------------------------------------------------------------------ fake servers

class Fake:
    """An HTTP server on a free local port; `reply(method, path, headers, body)` answers."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _answer(self, method: str) -> None:
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                status, content_type, data = outer.reply(method, self.path, self.headers, body)
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:  # noqa: N802
                self._answer("GET")

            def do_POST(self) -> None:  # noqa: N802
                self._answer("POST")

            def log_message(self, *args: Any) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def reply(self, method: str, path: str, headers: Any, body: bytes) -> tuple[int, str, bytes]:
        raise NotImplementedError

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def _json(data: Any, status: int = 200) -> tuple[int, str, bytes]:
    return status, "application/json", json.dumps(data).encode()


class FakeModels(Fake):
    def reply(self, method: str, path: str, headers: Any, body: bytes) -> tuple[int, str, bytes]:
        if method == "GET" and path.endswith("/models"):
            return _json({"data": [{"id": "fake"}, {"id": "fake-vl"}]})
        request = json.loads(body)
        self.requests.append(request)
        messages = request["messages"]
        if not request.get("tools"):  # the vision model: alert checking or `look`
            text = ("YES\nA person is in the red box." if "alarms" in messages[0]["content"]
                    else "Two people walk past.")
            return _json(_message(text))
        last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
        results = [m["content"] for m in messages[last_user + 1:] if m["role"] == "tool"]
        plan = self.plan(messages[last_user]["content"], results)
        if len(results) < len(plan):
            name, args = plan[len(results)]
            return _json(_message("", [{"id": f"call{len(messages)}", "type": "function",
                                        "function": {"name": name,
                                                     "arguments": json.dumps(args)}}]))
        last = results[-1][:200] if results else "no lookup"
        return _json(_message(f"SMOKE ANSWER after {len(results)} lookups: {last}"))

    @staticmethod
    def plan(question: str, results: list[str]) -> list[tuple[str, dict[str, Any]]]:
        """The tool calls for a question, in order (later ones may use earlier results)."""
        if "photo1" in question:
            return [("face", {"photo": "photo1"})]
        if m := re.search(r'since="([^"]+)" and until="([^"]+)"', question):  # a report
            span = {"since": m[1], "until": m[2]}
            return [("objects", span), ("alerts", span)]
        if "look" in question:
            thumb = re.search(r'"thumb": "([^"]+)"', results[0]) if results else None
            return [("objects", {"labels": ["person"]}),
                    ("look", {"images": [thumb[1] if thumb else "none"],
                              "question": "How many people?"})]
        return [("objects", {"labels": ["person"]})]


def _message(text: str, calls: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if calls:
        message["tool_calls"] = calls
    return {"choices": [{"message": message}]}


class FakeTelegram(Fake):
    def __init__(self) -> None:
        super().__init__()
        self.file = b""  # what a photo sent to the bot downloads as

    def reply(self, method: str, path: str, headers: Any, body: bytes) -> tuple[int, str, bytes]:
        if method == "GET":  # a file download
            self.requests.append({"method": "download", "path": path})
            return 200, "image/jpeg", self.file
        name = path.rsplit("/", 1)[-1]
        fields: dict[str, Any] = {}
        files: dict[str, bytes] = {}
        if headers["Content-Type"].startswith("multipart/"):
            msg = email.message_from_bytes(
                f"Content-Type: {headers['Content-Type']}\r\n\r\n".encode() + body)
            parts: list[Any] = msg.get_payload()  # type: ignore[assignment]
            for part in parts:
                key = part.get_param("name", header="content-disposition")
                if part.get_filename():
                    files[key] = part.get_payload(decode=True)
                else:
                    fields[key] = part.get_payload(decode=True).decode()
        else:
            fields = json.loads(body or b"{}")
        self.requests.append({"method": name, "fields": fields, "files": files})
        if name == "getMe":
            return _json({"ok": True, "result": {"username": "histor_smoke_bot"}})
        if name == "getFile":
            return _json({"ok": True, "result": {"file_path": "photos/p.jpg",
                                                 "file_size": len(self.file)}})
        return _json({"ok": True, "result": {"message_id": len(self.requests)}})

    def sent(self, method: str) -> list[dict[str, Any]]:
        return [r for r in self.requests if r["method"] == method]


class FakeWebhook(Fake):
    def reply(self, method: str, path: str, headers: Any, body: bytes) -> tuple[int, str, bytes]:
        self.requests.append({"method": method, "path": path,
                              "body": json.loads(body) if body else None})
        return _json({"ok": True})


class FakePing(Fake):
    def reply(self, method: str, path: str, headers: Any, body: bytes) -> tuple[int, str, bytes]:
        self.requests.append({"path": path, "method": method, "time": time.time()})
        return 200, "text/plain", b"OK"


# ------------------------------------------------------------------ the recorded site

def make_clip(path: Path, frames: int = 100, black: int = 40) -> Path:
    """The first `frames` of vtest.avi (10 fps), then `black` frames of a covered lens."""
    cap = cv2.VideoCapture(str(DATA / "vtest.avi"))
    writer = None
    for _ in range(frames):
        ok, image = cap.read()
        assert ok
        if writer is None:
            h, w = image.shape[:2]
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), 10, (w, h))
        writer.write(image)
    cap.release()
    assert writer is not None
    for _ in range(black):
        writer.write(np.zeros((h, w, 3), np.uint8))
    writer.release()
    return path


def site_config(tmp: Path, fakes: SimpleNamespace) -> dict[str, Any]:
    def m(name: str) -> str:
        return str(MODELS / name)

    llm = fakes.models.url + "/v1"
    return {
        "model": {"path": m("yolox_nano.onnx"), "conf_threshold": 0.25,
                  "class_agnostic_nms": True},
        "tracking": {"track_threshold": 0.45, "max_missed": 30, "min_hits": 3},
        "cameras": {"street": {"about": {
            "description": "A pedestrian street seen from a first floor", "aliases": ["road"],
            "covers": "outdoor", "faces": "poor", "zones": {"junction": "where the paths meet"}}}},
        "store": {"path": str(tmp / "store" / "histor.db"),
                  "media_dir": str(tmp / "store" / "media")},
        "search": {"enabled": True,
                   "clip": {"image": m("clip_image.onnx"), "text": m("clip_text.onnx"),
                            "vocab": m("clip_bpe_vocab.txt.gz")},
                   "classes": ["person", "bicycle", "car", "backpack", "handbag", "suitcase",
                               "umbrella"],
                   "object_interval_seconds": 1, "scene_interval_seconds": 3,
                   "motion": {"enabled": True}},
        "faces": {"enabled": True, "detector": m("face_detection_yunet_2023mar.onnx"),
                  "recognizer": m("face_recognition_sface_2021dec.onnx")},
        "pose": {"enabled": True, "model": m("pose_rtmpose_t.onnx"), "min_box_height": 40,
                 "alerts": ["down", "hands_up"]},
        "plates": {"enabled": True, "detector": {"model": m("plate_yolov9_t_384.onnx")},
                   "reader": {"model": m("plate_ocr_cct_s_v2.onnx"),
                              "config": m("plate_ocr_cct_s_v2.yaml")},
                   "watchlist": [{"plate": "AB 123 CD", "name": "banned_van"}]},
        # the clip's audio needs ffmpeg: without it the recorder must carry on regardless
        "sounds": {"enabled": True, "model": m("sound_yamnet.onnx"),
                   "classes": m("sound_yamnet_classes.csv")},
        "tamper": {"enabled": True, "hold_seconds": 2},
        "abandoned": {"enabled": True, "seconds": 3},
        "zones": [
            {"name": "junction", "points": [[0.729, 0.330], [1.0, 0.260], [1.0, 0.729],
                                            [0.729, 0.608]], "classes": ["person"],
             "dwell_seconds": 2},
            {"name": "path_line", "type": "line", "points": [[0.456, 0.330], [0.456, 0.573]],
             "classes": ["person"]},
            {"name": "street_crowd", "classes": ["person"], "max_count": 3, "count_seconds": 1},
        ],
        "alerts": {
            "console": True, "jsonl": str(tmp / "events.jsonl"),
            "channels": [
                {"type": "webhook", "url": fakes.webhook.url + "/hook"},
                {"type": "telegram", "token": TOKEN, "chat_id": 42, "images": True,
                 "api_url": fakes.telegram.url, "events": ["tamper", "crowd"]},
            ],
            "verify": {"enabled": True, "timeout": 20,
                       "events": ["enter", "dwell", "cross", "crowd", "down", "abandoned"]},
        },
        "chat": {"model": "fake",
                 "models": {"fake": {"api": "openai", "base_url": llm, "model": "fake"}},
                 "vision": {"model": "fake-vl", "base_url": llm}},
        "reports": [{"name": "smoke", "every": "hourly", "delay_minutes": 0,
                     "interests": ["people at the junction"],
                     "channels": [{"type": "webhook", "url": fakes.webhook.url + "/report"}]}],
        "telegram": {"token": TOKEN, "allowed_chats": ["42"], "images": True,
                     "face_search": True},
        "mcp": {"face_search": True, "photo_dirs": [str(tmp)]},
        "server": {"token": ""},
    }


def record_site(tmp: Path, fakes: SimpleNamespace) -> SimpleNamespace:
    """data/vtest.avi recorded with `histor.record`, every feature on. The clip starts 50
    minutes before this hour, so the last hourly period holds all of it."""
    clip = make_clip(tmp / "clip.avi")
    face = tmp / "face.jpg"  # messi5.jpg's face is too small at 1x (31 px): 2x is usable
    image = cv2.imread(str(DATA / "messi5.jpg"))
    assert image is not None
    cv2.imwrite(str(face), cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
    config = site_config(tmp, fakes)
    path = tmp / "site.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
                    encoding="utf-8")
    start = datetime.now().replace(minute=0, second=0, microsecond=0) - timedelta(minutes=50)
    result = run("-m", "histor.record", "--config", path, "--source", clip, "--camera", "street",
                 "--start-time", start.isoformat(timespec="seconds"), timeout=900)
    return SimpleNamespace(tmp=tmp, clip=clip, face=face, config=config, config_path=path,
                           start=start, record=result, store=Path(config["store"]["path"]))
