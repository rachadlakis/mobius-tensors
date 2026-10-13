"""Ask Histor from Telegram: the chat (chat.py) behind a Telegram bot.

    uv run -m histor.telegram_bot --config configs/default.yaml

    telegram:
      token: ${TELEGRAM_BOT_TOKEN}     # from @BotFather, in .env
      allowed_chats: []                # chat ids that may ask; send /id to the bot to get yours
      images: false                    # also send the evidence thumbnails (they leave the site)
      face_search: false               # a photo sent to the bot searches the recorded faces
      # model: local                   # chat.models entry (default: chat.model)

The bot only answers chats listed in `allowed_chats`; any other chat only gets its own id back
for /start or /id, so it can be added. It polls Telegram (outgoing HTTPS only): the server needs
no open port and no public address. Questions are answered one at a time, by the same model and
tools as the chat. Each chat keeps its conversation for 30 minutes of quiet, or until /new.

What reaches Telegram: the questions and answers (text), and with `images: true` the evidence
thumbnails (for face search: the face crops from the cameras).

Face search (`face_search: true`, and `faces.enabled`): a photo sent to the bot, with the
question as its caption, is downloaded into memory, turned into a face print exactly as in the
chat (`/photo`), and searched directly, like `POST /search/face`: the caption's time and camera
words ("today", "yesterday 6pm to 9pm", "office") set the range, no language model is needed,
and the answer lists "likely" / "possible" visits. The photo then stays attached, so a follow-up
question to the chat model can use it. The photo is never
written to disk on the server, but it stays in the Telegram chat, on Telegram's servers: face
prints are biometric data, so turn this on only where the client agreed in writing.
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from dotenv import load_dotenv

from histor.ask import build_engine
from histor.chat import answer, model_config, system_prompt
from histor.engine.cameras import expand_env
from histor.engine.status import camera_status
from histor.engine.telegram import TelegramAPI, TelegramError
from histor.llm import Backend, create_backend
from histor.tools import Toolbox

log = logging.getLogger("histor.telegram")

HELP = """Ask me what the cameras saw, in plain words. For example:
• what happened at the gate last night?
• how many cars came in today?
• did any alert go off this morning?
• was anyone running in the car park?

/status  cameras online, today's counts and alerts
/new     start a new conversation
/id      this chat's id"""
PHOTO_REPLY = ("I can't look for a person from a photo here: face search from Telegram is off "
               "on this site (telegram.face_search). Face prints are biometric data, and a photo "
               "sent here stays on Telegram's servers. I can still search by description: \"a "
               "man in a dark blue jacket at the door today\".")
FACES_OFF = ("Face search is off on this site (faces.enabled), so there are no recorded faces to "
             "compare a photo with. I can still search by description.")
PHOTO_QUESTION = "Did this person come here today?"
# what people in Telegram see when something fails: no addresses, ports or error codes
MODEL_DOWN = ("I can't answer questions right now: the assistant on the Histor server isn't "
              "available. /status still works, and alerts keep coming. Please try again later.")
IDLE_SECONDS = 30 * 60


class TelegramBot:
    def __init__(self, api: TelegramAPI, toolbox: Toolbox, new_backend: Callable[[], Backend],
                 allowed: set[str], images: bool = False, max_images: int = 4,
                 cameras: list[str] | None = None, face_search: bool = False) -> None:
        self.api = api
        self.face_search = face_search
        self.toolbox = toolbox
        self.new_backend = new_backend
        self.allowed = allowed
        self.images = images
        self.max_images = max_images
        self.cameras = cameras or []
        self.conversations: dict[str, tuple[Backend, float]] = {}
        self.untold: dict[str, list[str]] = {}  # photos the chat model hasn't heard of yet

    def handle(self, update: dict[str, Any]) -> None:
        message = update.get("message") or {}
        chat = (message.get("chat") or {}).get("id")
        if chat is None:
            return
        text = (message.get("text") or "").strip()
        command = text.split()[0].split("@")[0].lower() if text.startswith("/") else None
        if str(chat) not in self.allowed:
            log.warning("Message from chat %s, which is not in telegram.allowed_chats", chat)
            if command in ("/start", "/id"):
                self.api.send_message(chat, f"This chat's id is {chat}. To use Histor here, add it "
                                            "to telegram.allowed_chats on the Histor server and "
                                            "restart the bot.")
            return
        if command in ("/start", "/help"):
            self.api.send_message(chat, HELP)
        elif command == "/id":
            self.api.send_message(chat, f"This chat's id is {chat}.")
        elif command == "/new":
            self.conversations.pop(str(chat), None)
            self.api.send_message(chat, "New conversation.")
        elif command == "/status":
            self.api.send_message(chat, self.status())
        elif message.get("photo") or _image_document(message):
            if not self.face_search:
                self.api.send_message(chat, PHOTO_REPLY)
            elif self.toolbox.engine.faces is None:
                self.api.send_message(chat, FACES_OFF)
            else:
                self.ask_with_photo(chat, message)
        elif message.get("document") or message.get("video"):
            self.api.send_message(chat, "Send me a question as text, or a photo of a person "
                                        "with your question as its caption.")
        elif not text or command:
            self.api.send_message(chat, "Send me a question as text. /help for examples.")
        else:
            self.ask(chat, text)

    def ask(self, chat: int | str, question: str) -> None:
        key = str(chat)
        backend, last = self.conversations.get(key, (None, 0.0))
        if backend is None or time.time() - last > IDLE_SECONDS:
            backend = self.new_backend()  # fresh instructions, with the current time
        self.conversations[key] = (backend, time.time())
        if self.untold.get(key):
            question = f"[Attached for face search: {', '.join(self.untold.pop(key))}]\n{question}"
        stop = threading.Event()
        threading.Thread(target=self._typing, args=(chat, stop), daemon=True).start()
        self.toolbox.evidence.clear()
        try:
            reply = answer(backend, self.toolbox, question, log)
        except RuntimeError as e:  # the model server is down or the model is missing
            log.error("Chat model failed for chat %s: %s", chat, e)  # details for the admin only
            reply = MODEL_DOWN
            self.conversations.pop(key, None)
        finally:
            stop.set()
        self.api.send_message(chat, reply or "(no answer)")
        if self.images and self.toolbox.evidence:
            jpegs = self._evidence()
            if jpegs:
                self.api.send_photos(chat, jpegs, "Evidence")

    def ask_with_photo(self, chat: int | str, message: dict[str, Any]) -> None:
        """Face search: the photo (in memory only) -> a face print -> the question."""
        file_id = (message["photo"][-1] if message.get("photo") else message["document"])["file_id"]
        try:
            data = self.api.download(file_id)  # the largest size Telegram has
        except TelegramError as e:
            log.error("Downloading a photo from chat %s failed: %s", chat, e)
            self.api.send_message(chat, "I couldn't get the photo. Please send it again.")
            return
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        del data
        if image is None:
            self.api.send_message(chat, "I can't read that image: send a JPEG or PNG photo.")
            return
        name, note = self.toolbox.add_photo([image])
        del image  # only the face print is kept, in memory, for this bot's session
        if name is None:
            self.api.send_message(chat, f"{note} Send a clearer photo with one face, facing the "
                                        "camera.")
            return
        log.info("Chat %s attached a photo for face search as %s", chat, name)
        question = (message.get("caption") or "").strip() or PHOTO_QUESTION
        self.toolbox.evidence.clear()  # else the last question's thumbnails go out as faces
        try:
            found = self.toolbox.face(name, when=question)
        except ValueError as e:
            log.error("Face search for chat %s failed: %s", chat, e)
            self.api.send_message(chat, "I couldn't run the search. Please try again later.")
            return
        self.api.send_message(chat, face_reply(found))
        self.untold.setdefault(str(chat), []).append(name)  # for follow-up questions
        if self.images and self.toolbox.evidence:
            jpegs = self._evidence()
            if jpegs:
                self.api.send_photos(chat, jpegs, "Best face of each visit")

    def status(self) -> str:
        s = camera_status(self.toolbox.engine.store, self.cameras)
        if not s["cameras"]:
            return "No camera has recorded anything yet."
        lines = []
        for c in s["cameras"]:
            state = "online" if c["online"] else "OFFLINE"
            seen = c["last_seen"][11:16] if c["last_seen"] else "never"
            counts = ", ".join(f"{k} {n}" for k, n in list(c["tracked_today"].items())[:4])
            alerts = c["alerts_today"]
            line = f"{c['camera']}: {state} (last {seen})"
            if counts:
                line += f" · today: {counts}"
            line += f" · {alerts['total']} alerts"
            if alerts["held_back"]:
                line += f" ({alerts['held_back']} held back)"
            if c["tamper"]:
                line += f" · TAMPER: {c['tamper']['rule']} at {c['tamper']['iso_time'][11:16]}"
            lines.append(line)
        return "\n".join(lines)

    def _evidence(self) -> list[bytes]:
        jpegs = []
        for rel in list(dict.fromkeys(self.toolbox.evidence))[: self.max_images]:
            path = self.toolbox.engine.store.media_path(rel)
            if path is not None:
                jpegs.append(Path(path).read_bytes())
        return jpegs

    def _typing(self, chat: int | str, stop: threading.Event) -> None:
        while not stop.is_set():  # "typing…" lasts 5 s on Telegram: renew it while answering
            try:
                self.api.typing(chat)
            except TelegramError:
                return
            stop.wait(4.5)

    def run(self) -> None:
        offset = None
        while True:
            try:
                updates = self.api.updates(offset)
            except TelegramError as e:
                log.warning("%s; retrying in 5 s", e)
                time.sleep(5)
                continue
            for update in updates:
                offset = update["update_id"] + 1
                try:
                    self.handle(update)
                except Exception:  # one bad message must not stop the bot
                    log.exception("Handling a Telegram message failed")


def face_reply(found: dict[str, Any], top: int = 5) -> str:
    """The face tool's result in a few lines: likely / possible visits, or what was checked."""
    when = found.get("when") or "any time"
    visits = found.get("visits") or []
    if visits:
        likely = found.get("likely", 0)
        head = ("Yes, likely" if likely else "Possibly") + \
            f": {len(visits)} visit{'s' if len(visits) != 1 else ''} ({when})."
        lines = [head]
        for v in visits[:top]:
            seen = str(v.get("face_seen", ""))[11:16]
            lines.append(f"• {v['camera']}, {str(v['first_seen'])[:16]} to "
                         f"{str(v['last_seen'])[11:16]}: {v['level']} (match {v['score']:.2f}, "
                         f"face clearly seen at {seen})")
        if len(visits) > top:
            lines.append(f"…and {len(visits) - top} more.")
        lines.append("Check the faces before acting: a match is a lead, not proof.")
        return "\n".join(lines)
    checked = found.get("checked") or []
    recorded = sum(c.get("faces_recorded", 0) for c in checked)
    if not recorded:
        return (f"No faces were recorded {when if when != 'any time' else 'yet'}, so I can't "
                "tell. Faces need a camera that sees them big and frontal enough.")
    per_camera = ", ".join(f"{c['camera']} {c['faces_recorded']}" for c in checked
                           if c.get("faces_recorded"))
    return (f"No match ({when}). Faces recorded and compared: {per_camera}. No match is not "
            "proof the person wasn't there: faces turned away or too far are not recorded.")


def _image_document(message: dict[str, Any]) -> bool:
    """A photo sent "as a file" (uncompressed) arrives as a document."""
    doc = message.get("document") or {}
    return str(doc.get("mime_type", "")).startswith("image/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask Histor from Telegram")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args()
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    settings = config.get("telegram") or {}
    if not settings.get("token"):
        raise SystemExit("Set telegram.token in the config (e.g. ${TELEGRAM_BOT_TOKEN} in .env)")
    api = TelegramAPI(expand_env(str(settings["token"])))
    allowed = {expand_env(str(c)).strip() for c in settings.get("allowed_chats") or []}
    face_search = bool(settings.get("face_search"))
    if not face_search:  # no photos over Telegram: don't load the face models at all
        config = {**config, "faces": {**(config.get("faces") or {}), "enabled": False}}
    engine = build_engine(config)
    toolbox = Toolbox.from_config(engine, config)
    name = settings.get("model") or (config.get("chat") or {}).get("model", "local")
    model = model_config(config, name)

    def new_backend() -> Backend:
        return create_backend(model, system_prompt(toolbox), toolbox.tools)

    bot = TelegramBot(api, toolbox, new_backend, allowed, images=bool(settings.get("images")),
                      cameras=sorted(config.get("cameras") or {}), face_search=face_search)
    me = api.me()
    log.info("Telegram bot @%s ready, model %s, images %s, face search %s", me.get("username"),
             name, "on" if bot.images else "off",
             "on" if face_search and engine.faces is not None else "off")
    if not allowed:
        log.warning("Nobody can ask yet: send /id to @%s, then add the id to "
                    "telegram.allowed_chats and restart", me.get("username"))
    try:
        bot.run()
    except KeyboardInterrupt:
        pass
    finally:
        toolbox.close()
        engine.store.close()


if __name__ == "__main__":
    main()
