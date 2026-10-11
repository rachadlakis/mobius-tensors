"""A small Telegram Bot API client (plain HTTPS, no SDK), for the alert channel (alerts.py) and
the chat bot (histor/telegram_bot.py).

Images and video only go to Telegram when a setting says so: they leave the site and are stored
on Telegram's servers, so a client agrees to it first (see DOCUMENTATION.md, section 17.1).
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

API_URL = "https://api.telegram.org"
MESSAGE_LIMIT = 4096   # characters in a text message
CAPTION_LIMIT = 1024   # characters in a photo / video caption
MEDIA_GROUP_LIMIT = 10


class TelegramError(RuntimeError):
    pass


def chunks(text: str, limit: int = MESSAGE_LIMIT) -> list[str]:
    """Split long text on line breaks (or hard, for a single huge line) into messages."""
    out, current = [], ""
    for line in text.splitlines(keepends=True):
        while len(line) > limit:
            if current:
                out.append(current)
                current = ""
            out.append(line[:limit])
            line = line[limit:]
        if len(current) + len(line) > limit:
            out.append(current)
            current = ""
        current += line
    if current.strip() or not out:
        out.append(current)
    return [c.rstrip("\n") or " " for c in out]


def caption(text: str) -> str:
    return text if len(text) <= CAPTION_LIMIT else text[:CAPTION_LIMIT - 1] + "…"


def multipart(fields: dict[str, Any], files: list[tuple[str, str, bytes, str]]
              ) -> tuple[bytes, str]:
    """fields + files [(field, filename, data, content type)] -> (body, content type header)."""
    boundary = uuid.uuid4().hex
    parts: list[bytes] = []
    for name, value in fields.items():
        if value is None:
            continue
        text = value if isinstance(value, str) else json.dumps(value)
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
                     f'{text}\r\n'.encode())
    for name, filename, data, content_type in files:
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; '
                     f'filename="{filename}"\r\nContent-Type: {content_type}\r\n\r\n'.encode()
                     + data + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


class TelegramAPI:
    def __init__(self, token: str, api_url: str = API_URL, timeout: float = 30.0) -> None:
        if not token or ":" not in token:
            raise ValueError("telegram token looks wrong: it comes from @BotFather and looks like "
                             "123456789:AA... (put it in .env, e.g. TELEGRAM_BOT_TOKEN)")
        self._base = f"{api_url.rstrip('/')}/bot{token}/"
        self._files = f"{api_url.rstrip('/')}/file/bot{token}/"
        self.timeout = timeout

    def call(self, method: str, params: dict[str, Any] | None = None,
             files: list[tuple[str, str, bytes, str]] | None = None,
             timeout: float | None = None) -> Any:
        """One Bot API method; the token is never in an error message."""
        if files:
            body, content_type = multipart(params or {}, files)
        else:
            body = json.dumps({k: v for k, v in (params or {}).items() if v is not None}).encode()
            content_type = "application/json"
        request = urllib.request.Request(self._base + method, body,
                                         {"Content-Type": content_type})
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
                reply = json.loads(response.read())
        except urllib.error.HTTPError as e:
            try:
                reply = json.loads(e.read())
            except ValueError:
                raise TelegramError(f"Telegram {method}: HTTP {e.code}") from None
        except urllib.error.URLError as e:
            raise TelegramError(f"Can't reach Telegram ({e.reason})") from None
        if not reply.get("ok"):
            raise TelegramError(f"Telegram {method}: {reply.get('description', 'failed')}")
        return reply.get("result")

    # ------------------------------------------------------------------ the methods Histor uses

    def send_message(self, chat_id: int | str, text: str) -> list[Any]:
        return [self.call("sendMessage", {"chat_id": chat_id, "text": part,
                                          "disable_web_page_preview": True})
                for part in chunks(text)]

    def send_photo(self, chat_id: int | str, jpeg: bytes, text: str = "") -> Any:
        return self.call("sendPhoto", {"chat_id": chat_id, "caption": caption(text) or None},
                         [("photo", "snapshot.jpg", jpeg, "image/jpeg")])

    def send_photos(self, chat_id: int | str, jpegs: list[bytes], text: str = "") -> Any:
        """Up to 10 photos as one album; the caption goes on the first."""
        jpegs = jpegs[:MEDIA_GROUP_LIMIT]
        if len(jpegs) == 1:
            return self.send_photo(chat_id, jpegs[0], text)
        media = [{"type": "photo", "media": f"attach://p{i}",
                  **({"caption": caption(text)} if i == 0 and text else {})}
                 for i in range(len(jpegs))]
        return self.call("sendMediaGroup", {"chat_id": chat_id, "media": media},
                         [(f"p{i}", f"p{i}.jpg", j, "image/jpeg") for i, j in enumerate(jpegs)])

    def send_video(self, chat_id: int | str, mp4: bytes, text: str = "") -> Any:
        return self.call("sendVideo", {"chat_id": chat_id, "caption": caption(text) or None,
                                       "supports_streaming": True},
                         [("video", "clip.mp4", mp4, "video/mp4")], timeout=120)

    def typing(self, chat_id: int | str) -> None:
        self.call("sendChatAction", {"chat_id": chat_id, "action": "typing"})

    def updates(self, offset: int | None, wait: int = 30) -> list[dict[str, Any]]:
        """Long polling: returns as soon as a message arrives, or after `wait` seconds."""
        return self.call("getUpdates", {"offset": offset, "timeout": wait,
                                        "allowed_updates": ["message"]}, timeout=wait + 10)

    def me(self) -> dict[str, Any]:
        return self.call("getMe")

    def download(self, file_id: str, max_bytes: int = 20_000_000) -> bytes:
        """A file someone sent the bot (a photo), into memory: it is never written to disk."""
        info = self.call("getFile", {"file_id": file_id})
        if (info.get("file_size") or 0) > max_bytes:
            raise TelegramError("That file is too big")
        try:
            with urllib.request.urlopen(self._files + info["file_path"],
                                        timeout=self.timeout) as response:
                return response.read(max_bytes + 1)[:max_bytes]
        except urllib.error.URLError as e:
            raise TelegramError(f"Can't download the file from Telegram "
                                f"({getattr(e, 'reason', e)})") from None


def cut_clip(recording: Path, start: float, seconds: float, ffmpeg: str = "ffmpeg"
             ) -> bytes | None:
    """`seconds` of a main-stream recording from `start` (seconds into the file), copied without
    re-encoding. None when ffmpeg is missing or the cut fails."""
    exe = shutil.which(ffmpeg)
    if exe is None:
        log.warning("Video clips need ffmpeg (sudo apt install ffmpeg): sending without one")
        return None
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "clip.mp4"
        cmd = [exe, "-hide_banner", "-loglevel", "error", "-ss", f"{max(0.0, start):.2f}",
               "-i", str(recording), "-t", f"{seconds:.2f}", "-c", "copy",
               "-movflags", "+faststart", "-y", str(out)]
        try:
            subprocess.run(cmd, check=True, timeout=60, capture_output=True)
        except (subprocess.SubprocessError, OSError) as e:
            log.warning("Could not cut a clip from %s: %s", recording.name, e)
            return None
        data = out.read_bytes() if out.exists() else b""
        return data or None
