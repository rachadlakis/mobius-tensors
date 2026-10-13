"""Telegram: the API client, the alert channel and the chat bot, against a fake Bot API server."""

from __future__ import annotations

import email
import json
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

import cv2

from histor.engine.alerts import AlertManager, TelegramSink, make_channel
from histor.engine.faces import FaceQuality, PhotoFace
from histor.engine.search import SearchEngine
from histor.engine.store import EventStore
from histor.engine.telegram import CAPTION_LIMIT, TelegramAPI, TelegramError, chunks
from histor.engine.zones import ZoneEvent
from histor.llm import Step, ToolCall
from histor.telegram_bot import TelegramBot
from histor.tools import Toolbox

TOKEN = "123456:TEST-token"


class FakeTelegram:
    """Records every Bot API call: method, form fields (JSON or multipart) and attached files."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.fail: str | None = None
        self.file = b""  # what a sent photo downloads as
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                body = self.rfile.read(int(self.headers["Content-Length"]))
                method = self.path.rsplit("/", 1)[-1]
                fields, files = {}, {}
                if self.headers["Content-Type"].startswith("multipart/"):
                    msg = email.message_from_bytes(
                        f"Content-Type: {self.headers['Content-Type']}\r\n\r\n".encode() + body)
                    for part in msg.get_payload():
                        name = part.get_param("name", header="content-disposition")
                        if part.get_filename():
                            files[name] = part.get_payload(decode=True)
                        else:
                            fields[name] = part.get_payload(decode=True).decode()
                else:
                    fields = json.loads(body or b"{}")
                outer.calls.append({"method": method, "path": self.path, "fields": fields,
                                    "files": files})
                if outer.fail:
                    reply = {"ok": False, "description": outer.fail}
                elif method == "getMe":
                    reply = {"ok": True, "result": {"username": "histor_test_bot"}}
                elif method == "getFile":
                    reply = {"ok": True, "result": {"file_path": "photos/file_1.jpg",
                                                    "file_size": len(outer.file)}}
                else:
                    reply = {"ok": True, "result": {"message_id": len(outer.calls)}}
                data = json.dumps(reply).encode()
                self.send_response(200 if reply["ok"] else 400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):  # noqa: N802  (file downloads)
                outer.calls.append({"method": "download", "path": self.path, "fields": {},
                                    "files": {}})
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.end_headers()
                self.wfile.write(outer.file)

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def methods(self) -> list[str]:
        return [c["method"] for c in self.calls]


@pytest.fixture
def tg():
    server = FakeTelegram()
    yield server
    server.server.shutdown()


def payload(**extra):
    t = datetime(2026, 10, 9, 23, 15).timestamp()  # in the past: clips are due at once
    return {"type": "down", "rule": "person_down", "label": "person", "camera": "warehouse",
            "track_id": 4, "time": t, "iso_time": datetime.fromtimestamp(t).isoformat(), **extra}


# ------------------------------------------------------------------ client

def test_chunks_split_long_text_on_lines():
    text = "\n".join(f"line {i} " + "x" * 50 for i in range(200))
    parts = chunks(text, 1000)
    assert all(len(p) <= 1000 for p in parts) and "\n".join(parts) == text
    assert chunks("y" * 2500, 1000) == ["y" * 1000, "y" * 1000, "y" * 500]
    assert chunks("") == [" "]


def test_api_errors_never_show_the_token(tg):
    api = TelegramAPI(TOKEN, tg.url)
    tg.fail = "Unauthorized"
    with pytest.raises(TelegramError, match="Unauthorized") as e:
        api.send_message(1, "hi")
    assert TOKEN not in str(e.value)
    with pytest.raises(ValueError, match="BotFather"):
        TelegramAPI("not-a-token")
    assert tg.calls[0]["path"] == f"/bot{TOKEN}/sendMessage"


# ------------------------------------------------------------------ alert channel

def test_alert_is_text_only_by_default(tg):
    sink = make_channel({"type": "telegram", "token": TOKEN, "chat_id": "42", "api_url": tg.url})
    assert isinstance(sink, TelegramSink) and not sink.images and not sink.video
    sink.send(payload(verified="yes", verification="Someone lies on the floor."), b"\xff\xd8jpeg")
    (call,) = tg.calls
    assert call["method"] == "sendMessage" and call["fields"]["chat_id"] == "42"
    text = call["fields"]["text"]
    assert "down person_down" in text and "warehouse" in text and "Check:  yes" in text


def test_images_on_sends_the_snapshot_with_a_caption_to_every_chat(tg):
    sink = make_channel({"type": "telegram", "token": TOKEN, "chat_id": [1, "-100200"],
                         "images": True, "api_url": tg.url, "rules": ["person_down"]})
    long = "a very long reason " * 100
    sink.send(payload(verified="unsure", verification=long), b"\xff\xd8jpeg")
    assert tg.methods() == ["sendPhoto", "sendPhoto"]
    assert [c["fields"]["chat_id"] for c in tg.calls] == ["1", "-100200"]
    assert tg.calls[0]["files"]["photo"] == b"\xff\xd8jpeg"
    assert len(tg.calls[0]["fields"]["caption"]) <= CAPTION_LIMIT
    assert sink.filter.rules == {"person_down"}  # the usual channel filters apply
    sink.send(payload(), None)  # no snapshot (snapshots off): text instead
    assert tg.methods()[-1] == "sendMessage"


def test_video_clip_is_sent_after_the_alert(tg, monkeypatch, tmp_path):
    from histor.engine import telegram

    sink = TelegramSink(TOKEN, 7, video=True, video_seconds=4, api_url=tg.url)
    sink.use_recordings({"dir": str(tmp_path)})
    found = (tmp_path / "warehouse").mkdir() or tmp_path / "warehouse" / "x.mp4"
    found.write_bytes(b"mp4")
    monkeypatch.setattr(sink.recordings, "find", lambda camera, t: ("warehouse/x.mp4", 12.0))
    cut = []
    monkeypatch.setattr(telegram, "cut_clip",
                        lambda path, start, seconds: cut.append((path.name, start, seconds))
                        or b"clipdata")
    done = threading.Event()
    real = sink._send_clip
    monkeypatch.setattr(sink, "_send_clip", lambda *a: (real(*a), done.set()))
    sink.send(payload(), None)  # an alert from the past: the clip is ready at once
    assert done.wait(5)
    assert tg.methods() == ["sendMessage", "sendVideo"]
    assert cut == [("x.mp4", 12.0, 4.0)] and tg.calls[1]["files"]["video"] == b"clipdata"


def test_video_needs_recording_on(tg, caplog):
    sink = TelegramSink(TOKEN, 7, video=True, api_url=tg.url)
    sink.use_recordings({"enabled": False})
    assert sink.recordings is None and "recording" in caplog.text


def test_reports_go_out_with_images_only_when_allowed(tg):
    text_only = TelegramSink(TOKEN, 9, api_url=tg.url)
    text_only.send_report("Morning report", "All quiet.", {}, [("a.jpg", b"1"), ("b.jpg", b"2")])
    assert tg.methods() == ["sendMessage"]
    with_images = TelegramSink(TOKEN, 9, images=True, api_url=tg.url)
    with_images.send_report("Morning report", "All quiet.", {}, [("a.jpg", b"1"), ("b.jpg", b"2")])
    assert tg.methods()[1:] == ["sendMessage", "sendMediaGroup"]
    media = json.loads(tg.calls[-1]["fields"]["media"])
    assert [m["media"] for m in media] == ["attach://p0", "attach://p1"]


def test_alert_manager_passes_recording_to_telegram(tg, tmp_path):
    manager = AlertManager(console=False, channels=[{"type": "telegram", "token": TOKEN,
                                                     "chat_id": 5, "video": True,
                                                     "api_url": tg.url}],
                           recording={"dir": str(tmp_path)})
    assert manager.sinks[0].recordings is not None
    manager.handle([ZoneEvent(1.0, "enter", "yard", 3, "person")],
                   wall_time=datetime(2026, 10, 10, 9).timestamp())
    manager.close()
    assert tg.methods() == ["sendMessage"]  # no recording at that time: no clip, no error


# ------------------------------------------------------------------ chat bot

class FakeBackend:
    """Calls `alerts` once, then answers."""
    name = "fake"

    def __init__(self) -> None:
        self.questions: list[str] = []
        self._called = False

    def add_user(self, text):
        self.questions.append(text)
        self._called = False

    def step(self, tools=True):
        if not self._called:
            self._called = True
            return Step("", [ToolCall("1", "alerts", {"when": "today"})])
        return Step("One person fell at the warehouse at 23:15.")

    def add_results(self, results):
        pass


@pytest.fixture
def bot_setup(tg, tmp_path):
    store = EventStore(tmp_path / "db.sqlite")
    t = datetime.now().replace(hour=0, minute=1).timestamp()
    snap = store.save_image(np.full((60, 80, 3), 90, np.uint8), "snapshots", "down", t)
    store.add_event("warehouse", t, {"type": "down", "rule": "person_down", "label": "person",
                                     "track_id": 4}, snap)
    toolbox = Toolbox(SearchEngine(store, None, camera_names=["warehouse"]))
    backends: list[FakeBackend] = []

    def new_backend():
        backends.append(FakeBackend())
        return backends[-1]

    def make(images=False, face_search=False):
        return TelegramBot(TelegramAPI(TOKEN, tg.url), toolbox, new_backend, {"42"},
                           images=images, cameras=["warehouse"], face_search=face_search)

    yield make, backends
    toolbox.close()
    store.close()


def msg(chat, text):
    return {"update_id": 1, "message": {"chat": {"id": chat}, "text": text}}


def test_strangers_only_get_their_id(tg, bot_setup):
    make, backends = bot_setup
    bot = make()
    bot.handle(msg(99, "what happened last night?"))
    assert tg.calls == [] and backends == []  # ignored
    bot.handle(msg(99, "/start"))
    (call,) = tg.calls
    assert "99" in call["fields"]["text"] and "allowed_chats" in call["fields"]["text"]


def test_allowed_chat_gets_an_answer_text_only_by_default(tg, bot_setup):
    make, backends = bot_setup
    bot = make()
    bot.handle(msg(42, "did anyone fall?"))
    sent = [c for c in tg.calls if c["method"] != "sendChatAction"]
    assert [c["method"] for c in sent] == ["sendMessage"]
    assert sent[0]["fields"]["text"] == "One person fell at the warehouse at 23:15."
    assert backends[0].questions == ["did anyone fall?"]
    bot.handle(msg(42, "and before that?"))  # same conversation
    assert len(backends) == 1 and len(backends[0].questions) == 2
    bot.handle(msg(42, "/new"))
    bot.handle(msg(42, "hello"))
    assert len(backends) == 2


def test_evidence_images_when_enabled(tg, bot_setup):
    make, _ = bot_setup
    make(images=True).handle(msg(42, "did anyone fall?"))
    sent = [c for c in tg.calls if c["method"] != "sendChatAction"]
    assert [c["method"] for c in sent] == ["sendMessage", "sendPhoto"]
    assert sent[1]["files"]["photo"][:2] == b"\xff\xd8"  # the alert's snapshot, as JPEG


def test_a_photo_gets_why_not_instead_of_a_search(tg, bot_setup):
    make, backends = bot_setup
    make().handle({"update_id": 2, "message": {"chat": {"id": 42}, "caption": "did he come?",
                                               "photo": [{"file_id": "x"}]}})
    (call,) = tg.calls
    assert "biometric" in call["fields"]["text"] and backends == []  # no model, no face search


class FakeFaces:
    """A bright photo has one usable face; a dark one has none."""

    def photo_faces(self, image):
        if image.mean() < 50:
            return []
        v = np.zeros(128, np.float32)
        v[0] = 1
        return [PhotoFace(v, (0, 0, 80, 80), FaceQuality(0.9, 80, 0.1, 120), [])]


def photo_msg(caption=None, document=False):
    message = {"chat": {"id": 42}}
    if document:  # sent "as a file"
        message["document"] = {"file_id": "doc1", "mime_type": "image/png"}
    else:  # Telegram sends several sizes, the largest last
        message["photo"] = [{"file_id": "small"}, {"file_id": "large"}]
    if caption:
        message["caption"] = caption
    return {"update_id": 3, "message": message}


def jpeg(value):
    return cv2.imencode(".jpg", np.full((120, 120, 3), value, np.uint8))[1].tobytes()


def test_face_search_off_or_without_faces(tg, bot_setup):
    make, backends = bot_setup
    make(face_search=True).handle(photo_msg("did he come?"))  # faces.enabled is off here
    (call,) = tg.calls
    assert "faces.enabled" in call["fields"]["text"] and backends == []


def face_print():
    v = np.zeros(128, np.float32)
    v[0] = 1
    return v


def test_a_photo_runs_face_search_without_the_chat_model(tg, bot_setup):
    make, backends = bot_setup
    bot = make(face_search=True)
    bot.toolbox.engine.faces = FakeFaces()
    store = bot.toolbox.engine.store
    seen = datetime(2026, 10, 7, 0, 30).timestamp()  # a fixed day: "today 00:30" fails at 00:10
    store.add_faces([{"camera": "warehouse", "time": seen + k, "track_id": 8,
                      "embedding": face_print()} for k in range(3)])
    tg.file = jpeg(200)
    bot.handle(photo_msg("Did this man come to the warehouse on 2026-10-07?"))
    get_file = next(c for c in tg.calls if c["method"] == "getFile")
    assert get_file["fields"]["file_id"] == "large"  # the biggest size Telegram has
    assert any(c["method"] == "download" and c["path"].endswith("/photos/file_1.jpg")
               for c in tg.calls)
    reply = [c for c in tg.calls if c["method"] == "sendMessage"][0]["fields"]["text"]
    assert reply.startswith("Yes, likely: 1 visit") and "warehouse" in reply and "00:30" in reply
    assert backends == []  # no language model needed
    assert list(bot.toolbox.photos) == ["photo1"]  # a face print, not the photo

    bot.handle(msg(42, "and yesterday?"))  # a follow-up goes to the chat model with the photo
    assert backends[0].questions == ["[Attached for face search: photo1]\nand yesterday?"]


def test_no_match_says_what_was_checked(tg, bot_setup):
    make, _ = bot_setup
    bot = make(face_search=True, images=True)
    bot.toolbox.engine.faces = FakeFaces()
    tg.file = jpeg(200)
    bot.handle(photo_msg(document=True))  # sent as a file, no caption: "today"
    reply = [c for c in tg.calls if c["method"] == "sendMessage"][0]["fields"]["text"]
    assert reply.startswith("No faces were recorded")
    assert "sendPhoto" not in [c["method"] for c in tg.calls]  # nothing to show


def test_a_photo_after_a_question_sends_no_leftover_evidence(tg, bot_setup):
    make, _ = bot_setup
    bot = make(face_search=True, images=True)
    bot.toolbox.engine.faces = FakeFaces()
    bot.handle(msg(42, "did anyone fall?"))  # its evidence: the alert snapshot
    tg.calls.clear()
    tg.file = jpeg(200)
    bot.handle(photo_msg("today"))  # no face recorded: nothing to show
    assert "sendPhoto" not in [c["method"] for c in tg.calls]


def test_an_unusable_photo_is_refused_with_the_reason(tg, bot_setup):
    make, backends = bot_setup
    bot = make(face_search=True)
    bot.toolbox.engine.faces = FakeFaces()
    tg.file = jpeg(10)  # too dark: no face found
    bot.handle(photo_msg("did he come?"))
    reply = [c for c in tg.calls if c["method"] == "sendMessage"][-1]["fields"]["text"]
    assert "clearer photo" in reply and backends == [] and bot.toolbox.photos == {}
    tg.file = b"not an image"
    bot.handle(photo_msg("did he come?"))
    reply = [c for c in tg.calls if c["method"] == "sendMessage"][-1]["fields"]["text"]
    assert "can't read that image" in reply


def test_status_and_help(tg, bot_setup):
    make, _ = bot_setup
    bot = make()
    bot.handle(msg(42, "/status"))
    bot.handle(msg(42, "/help@histor_test_bot"))
    status, help_text = (c["fields"]["text"] for c in tg.calls)
    assert status.startswith("warehouse: ") and "1 alerts" in status
    assert "/status" in help_text


def test_model_errors_are_told_and_the_conversation_reset(tg, bot_setup):
    make, backends = bot_setup

    class Down(FakeBackend):
        def step(self, tools=True):
            raise RuntimeError("Can't reach the model server")

    bot = make()
    bot.new_backend = Down
    bot.handle(msg(42, "anything?"))
    sent = [c for c in tg.calls if c["method"] == "sendMessage"]
    text = sent[0]["fields"]["text"]
    assert "can't answer questions right now" in text  # plain words for the user...
    assert "Can't reach" not in text and "127.0.0.1" not in text  # ...the details only in the log
    assert "42" not in bot.conversations
