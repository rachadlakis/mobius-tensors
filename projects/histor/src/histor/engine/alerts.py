"""Turn zone/line events into alerts: console, JSON-lines file, event store, and channels.

Filtering (event types, cooldown) happens once; every emitted event then goes to each channel
whose own filter (rules, events, labels, hours) accepts it. Channels are sent from a background
thread so a slow mail server never stalls detection. Each event can carry a JPEG snapshot of the
frame with the object highlighted.

Channels (config `alerts.channels`, a list; `type` picks the class in CHANNELS):

    webhook   POST the event as JSON (n8n, Zapier, Make, Slack/Teams workflows, your backend)
    email     SMTP, snapshot attached
    telegram  a Telegram bot: the alert text, with the snapshot (`images`) and a short clip from
              the recording (`video`), both optional: they leave the site (telegram.py)

With `verify` (verify.py), a local vision model checks each alert's snapshot first: a NO holds it
back from the channels (it stays in the store, marked), anything else is sent with the verdict.

A new channel (Telegram, WhatsApp, SMS, a siren relay...) is one Sink subclass with
`send(payload, jpeg)`, registered in CHANNELS; `alert_message()` gives it the same title/text
as email. `send_report()` carries longer messages on the same channels (Histor's periodic
summaries).
"""

from __future__ import annotations

import json
import logging
import queue
import smtplib
import threading
import urllib.request
from datetime import datetime, time as dtime
from email.message import EmailMessage
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from histor.engine.cameras import expand_env
from histor.engine.zones import ZoneEvent

if TYPE_CHECKING:
    from histor.engine.store import EventStore
    from histor.engine.tracker import Track
    from histor.engine.verify import AlertVerifier

log = logging.getLogger(__name__)


class AlertManager:
    def __init__(
        self,
        console: bool = True,
        jsonl: str | Path | None = None,
        events: list[str] | None = None,
        cooldown_seconds: float = 0.0,
        source: str | None = None,
        webhook: dict[str, Any] | None = None,
        email: dict[str, Any] | None = None,
        channels: list[dict[str, Any]] | None = None,
        snapshots: bool = True,
        store: EventStore | None = None,
        verify: dict[str, Any] | AlertVerifier | None = None,
        vision: dict[str, Any] | None = None,
        recording: dict[str, Any] | None = None,
    ) -> None:
        """
        events: only alert on these event types (e.g. [dwell, cross]); None = all.
        cooldown_seconds: minimum time between two alerts for the same rule + event type.
        channels: [{type: webhook | email, ...settings, rules?, events?, labels?, hours?,
            cooldown_seconds?}]: the optional filters limit what that one channel receives,
            e.g. hours: "22:00-06:00". A channel cooldown holds back messages only, while the
            global cooldown_seconds also skips storing the event.
        webhook / email: shorthand for one channel of that type (older configs).
        snapshots: save a JPEG per alert (needs a store) and attach it to emails.
        verify: the `alerts.verify` section, or a ready AlertVerifier (verify.py); vision: the
            `chat.vision` section, whose model settings `verify` defaults to.
        recording: the `recording` section, for channels that send video clips (telegram).
        """
        self.console = console
        self.jsonl = Path(jsonl) if jsonl else None
        self.event_types = set(events) if events else None
        self.cooldown = cooldown_seconds
        self.source = source
        self.store = store
        self.snapshots = snapshots
        configs = list(channels or [])
        if webhook:
            configs.append({"type": "webhook", **webhook})
        if email:
            configs.append({"type": "email", **email})
        self.sinks: list[Sink] = [make_channel(c) for c in configs]
        for sink in self.sinks:
            if isinstance(sink, TelegramSink):
                sink.use_recordings(recording)
        self._dispatcher = Dispatcher(self.sinks) if self.sinks else None
        if isinstance(verify, dict):
            from histor.engine.verify import AlertVerifier

            verify = AlertVerifier.from_config(verify, vision)
        self.verifier = verify
        self._last: dict[tuple[str, str], float] = {}
        if self.jsonl:
            self.jsonl.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, config: dict[str, Any] | None, source: str | None = None,
                    store: EventStore | None = None,
                    vision: dict[str, Any] | None = None,
                    recording: dict[str, Any] | None = None) -> AlertManager:
        """vision: the `chat.vision` section (defaults for `verify`)."""
        return cls(**(config or {}), source=source, store=store, vision=vision,
                   recording=recording)

    def handle(self, events: list[ZoneEvent], frame: np.ndarray | None = None,
               tracks: list[Track] | None = None, wall_time: float | None = None) -> list[ZoneEvent]:
        """Filter events, emit them to the sinks, and return the ones emitted.

        frame/tracks: used for the snapshot; wall_time: unix time of this frame.
        """
        emitted = []
        for event in events:
            if self.event_types and event.type not in self.event_types:
                continue
            key = (event.rule, event.type)
            if self.cooldown and event.timestamp - self._last.get(key, -1e18) < self.cooldown:
                continue
            self._last[key] = event.timestamp
            emitted.append(event)
            self._emit(event, frame, tracks, wall_time)
        return emitted

    def close(self) -> None:
        """Wait for pending checks and queued webhooks/emails to go out."""
        if self.verifier:
            self.verifier.close()
        if self._dispatcher:
            self._dispatcher.close()

    def _emit(self, event: ZoneEvent, frame: np.ndarray | None, tracks: list[Track] | None,
              wall_time: float | None) -> None:
        if self.console:
            extra = event.direction or event.plate or (
                f"{event.count} inside" if event.count else None) or (
                f"{event.dwell_seconds:.1f}s" if event.dwell_seconds is not None else ""
            )
            who = event.label if event.track_id is None else f"{event.label} #{event.track_id}"
            print(f"[{event.timestamp:8.2f}s] {event.type.upper():5s} {event.rule}: {who} "
                  f"{extra}".rstrip())
        if self.jsonl:
            record = {**event.to_dict(), "source": self.source}
            with self.jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        if self.store is None and not self.sinks:
            return

        t = wall_time if wall_time is not None else datetime.now().timestamp()
        camera = self.source or "camera"
        jpeg, snapshot, image = None, None, None
        if frame is not None and (self.snapshots or self.verifier):
            image = snapshot_image(frame, event, tracks)
        if self.snapshots and image is not None:
            ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            jpeg = buf.tobytes() if ok else None
            if self.store is not None:
                name = f"{camera}_{int(t * 1000)}_{event.rule}_{event.type}_{event.track_id}"
                snapshot = self.store.save_image(image, "snapshots", _safe(name), t)
        event_id = None
        if self.store is not None:
            event_id = self.store.add_event(camera, t, event.to_dict(), snapshot)
        payload = {
            **event.to_dict(), "id": event_id, "camera": camera, "time": t,
            "iso_time": datetime.fromtimestamp(t).isoformat(timespec="seconds"),
            "snapshot": snapshot,
        }
        if (self.verifier and frame is not None and image is not None
                and self.verifier.applies(payload)):
            from histor.engine.verify import close_up

            images = [image] + [close_up(frame, tr.box) for tr in tracks or []
                                if tr.id == event.track_id][:1]
            self.verifier.submit(payload, images,
                                 lambda p, verdict, reason: self._verified(p, verdict, reason,
                                                                           jpeg))
        elif self._dispatcher:
            self._dispatcher.submit(payload, jpeg)

    def _verified(self, payload: dict[str, Any], verdict: str, reason: str,
                  jpeg: bytes | None) -> None:
        """The vision model's verdict: stored with the event; sent unless it vetoes."""
        assert self.verifier is not None
        payload["verified"], payload["verification"] = verdict, reason
        if self.store is not None and payload.get("id"):
            self.store.set_verification(payload["id"], verdict, reason)
        sent = self.verifier.sends(verdict)
        log.info("%s %s/%s #%s checked: %s (%s)%s", payload["camera"], payload["rule"],
                 payload["type"], payload.get("track_id"), verdict, reason,
                 "" if sent else ", not sent")
        if sent and self._dispatcher:
            self._dispatcher.submit(payload, jpeg)


def snapshot_image(frame: np.ndarray, event: ZoneEvent, tracks: list[Track] | None) -> np.ndarray:
    """Copy of the frame with the object that caused the event boxed and labelled."""
    image = frame.copy()
    for t in tracks or []:
        if t.id != event.track_id:
            continue
        x1, y1, x2, y2 = (int(v) for v in t.box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        text = f"{event.type} {event.rule}: {t.label} #{t.id}"
        cv2.putText(image, text, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2, cv2.LINE_AA)
    return image


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def alert_message(payload: dict[str, Any], prefix: str = "") -> tuple[str, str]:
    """(title, text) for a person: shared by email and any future chat/SMS channel."""
    what = (payload.get("direction") or payload.get("plate")
            or (f"{payload['count']} inside" if payload.get("count") else None)
            or payload.get("dwell_seconds") or "")
    title = (f"{prefix} {payload['type']} {payload['rule']} - "
             f"{payload.get('label')} on {payload['camera']}").strip()
    text = "\n".join([
        f"Camera: {payload['camera']}",
        f"Time:   {payload['iso_time']}",
        f"Event:  {payload['type']} ({payload['rule']}) {what}".rstrip(),
        f"Heard:  {payload.get('label')}" if payload.get("type") == "sound" else
        f"Object: {payload.get('label')}" + (f" #{payload['track_id']}"  # crowd, tamper: no id
                                             if payload.get("track_id") is not None else ""),
    ] + ([f"Check:  {payload['verified']} ({payload.get('verification') or 'no reason'})"]
         if payload.get("verified") else []))
    return title, text


def _hours(spec: str) -> tuple[dtime, dtime]:
    try:
        start, end = (dtime.fromisoformat(part.strip()) for part in spec.split("-"))
    except ValueError:
        raise ValueError(f"hours must look like '22:00-06:00', got {spec!r}") from None
    return start, end


class ChannelFilter:
    """Which alerts one channel receives. Empty = all of them.

    cooldown_seconds: at most one message per rule + event type in that time, for this channel
    only (the event store still gets every event).
    """

    def __init__(self, rules: list[str] | None = None, events: list[str] | None = None,
                 labels: list[str] | None = None, hours: str | None = None,
                 cooldown_seconds: float = 0.0) -> None:
        self.rules = set(rules or [])
        self.events = set(events or [])
        self.labels = set(labels or [])
        self.hours = _hours(hours) if hours else None  # wraps past midnight when start > end
        self.cooldown = cooldown_seconds
        self._last: dict[tuple[str, str], float] = {}

    def accepts(self, payload: dict[str, Any]) -> bool:
        if self.rules and payload.get("rule") not in self.rules:
            return False
        if self.events and payload.get("type") not in self.events:
            return False
        if self.labels and payload.get("label") not in self.labels:
            return False
        if self.hours:
            start, end = self.hours
            now = datetime.fromtimestamp(payload["time"]).time()
            inside = start <= now < end if start <= end else (now >= start or now < end)
            if not inside:
                return False
        if self.cooldown:
            key = (payload.get("rule", ""), payload.get("type", ""))
            if payload["time"] - self._last.get(key, -1e18) < self.cooldown:
                return False
            self._last[key] = payload["time"]
        return True


class Sink:
    filter = ChannelFilter()

    def accepts(self, payload: dict[str, Any]) -> bool:
        return self.filter.accepts(payload)

    def send(self, payload: dict[str, Any], jpeg: bytes | None) -> None:
        raise NotImplementedError

    def send_report(self, title: str, text: str, payload: dict[str, Any],
                    images: list[tuple[str, bytes]] | None = None) -> None:
        """A longer message (a periodic summary): title, plain text, its data as JSON, and a few
        JPEGs as (filename, bytes). Not filtered: the report chose this channel."""
        raise NotImplementedError(f"{type(self).__name__} can't send reports")


class WebhookSink(Sink):
    """POST the event as JSON. Point it at n8n/Zapier/Slack workflows or your own backend."""

    def __init__(self, url: str, headers: dict[str, str] | None = None, timeout: float = 10.0):
        self.url = expand_env(url)
        self.headers = {k: expand_env(v) for k, v in (headers or {}).items()}
        self.timeout = timeout

    def send(self, payload: dict[str, Any], jpeg: bytes | None) -> None:
        self._post(payload)

    def send_report(self, title: str, text: str, payload: dict[str, Any],
                    images: list[tuple[str, bytes]] | None = None) -> None:
        self._post({**payload, "title": title, "text": text})  # images: paths are in payload

    def _post(self, body: dict[str, Any]) -> None:
        request = urllib.request.Request(
            self.url, data=json.dumps(body, ensure_ascii=False).encode(), method="POST",
            headers={"Content-Type": "application/json", **self.headers},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            response.read()


class EmailSink(Sink):
    def __init__(self, host: str, sender: str, to: list[str] | str, port: int = 587,
                 username: str | None = None, password: str | None = None,
                 starttls: bool = True, ssl: bool = False,
                 subject_prefix: str = "[Histor]", timeout: float = 20.0):
        self.host, self.port = expand_env(host), port
        self.sender = expand_env(sender)
        self.to = [to] if isinstance(to, str) else list(to)
        self.username = expand_env(username) if username else None
        self.password = expand_env(password) if password else None
        self.starttls, self.ssl = starttls, ssl
        self.subject_prefix = subject_prefix
        self.timeout = timeout

    def send(self, payload: dict[str, Any], jpeg: bytes | None) -> None:
        msg = EmailMessage()
        title, text = alert_message(payload, self.subject_prefix)
        msg["Subject"] = title
        msg["From"], msg["To"] = self.sender, ", ".join(self.to)
        msg.set_content(text)
        if jpeg:
            msg.add_attachment(jpeg, maintype="image", subtype="jpeg", filename="snapshot.jpg")
        self._deliver(msg)

    def send_report(self, title: str, text: str, payload: dict[str, Any],
                    images: list[tuple[str, bytes]] | None = None) -> None:
        msg = EmailMessage()
        msg["Subject"] = title
        msg["From"], msg["To"] = self.sender, ", ".join(self.to)
        msg.set_content(text)
        for filename, data in images or []:
            msg.add_attachment(data, maintype="image", subtype="jpeg", filename=filename)
        self._deliver(msg)

    def _deliver(self, msg: EmailMessage) -> None:
        smtp_cls = smtplib.SMTP_SSL if self.ssl else smtplib.SMTP
        with smtp_cls(self.host, self.port, timeout=self.timeout) as smtp:
            if self.starttls and not self.ssl:
                smtp.starttls()
            if self.username:
                smtp.login(self.username, self.password or "")
            smtp.send_message(msg)


class TelegramSink(Sink):
    """A Telegram bot posts each alert to one or more chats (a person, a group, a channel).

    images: attach the snapshot. video: then send a clip of the recording, from 3 s before the
    alert, once it is recorded (needs `recording:` and ffmpeg). Both leave the site and stay on
    Telegram's servers: turn them on only where the client agreed; off, only text goes out."""

    def __init__(self, token: str, chat_id: int | str | list[int | str], images: bool = False,
                 video: bool = False, video_seconds: float = 10.0,
                 api_url: str = "https://api.telegram.org", timeout: float = 30.0) -> None:
        from histor.engine.telegram import TelegramAPI

        self.api = TelegramAPI(expand_env(token), api_url, timeout)
        ids = chat_id if isinstance(chat_id, list) else [chat_id]
        self.chat_ids = [expand_env(str(c)) for c in ids]
        self.images, self.video = images, video
        self.video_seconds = max(2.0, float(video_seconds))
        self.recordings: Any = None

    def use_recordings(self, recording: dict[str, Any] | None) -> None:
        if not self.video:
            return
        if not recording or recording.get("enabled") is False:
            log.warning("telegram video: needs the `recording:` section on; sending without clips")
            return
        from histor.engine.mainstream import Recordings

        self.recordings = Recordings(recording.get("dir", "data/store/recordings"),
                                     recording.get("segment_seconds", 60))

    def send(self, payload: dict[str, Any], jpeg: bytes | None) -> None:
        title, text = alert_message(payload)
        message = f"{title}\n{text}"
        for chat in self.chat_ids:
            if self.images and jpeg:
                self.api.send_photo(chat, jpeg, message)
            else:
                self.api.send_message(chat, message)
        if self.video and self.recordings is not None:
            start = payload["time"] - VIDEO_LEAD_SECONDS
            ready = start + self.video_seconds + 2.0 - datetime.now().timestamp()
            timer = threading.Timer(max(0.0, ready), self._send_clip, (payload, title))
            timer.daemon = True
            timer.start()

    def _send_clip(self, payload: dict[str, Any], title: str) -> None:
        from histor.engine.telegram import cut_clip

        try:
            found = self.recordings.find_lead(payload["camera"], payload["time"],
                                              VIDEO_LEAD_SECONDS)
            path = self.recordings.path(found[0]) if found else None
            clip = cut_clip(path, found[1], self.video_seconds) if path and found else None
            if clip is None:
                log.info("telegram: no recording for %s at %s, no clip", payload["camera"],
                         payload.get("iso_time"))
                return
            for chat in self.chat_ids:
                self.api.send_video(chat, clip, f"Clip: {title}")
        except Exception as exc:  # a clip must never take anything else down
            log.error("telegram clip failed for %s/%s: %s", payload.get("rule"),
                      payload.get("type"), exc)

    def send_report(self, title: str, text: str, payload: dict[str, Any],
                    images: list[tuple[str, bytes]] | None = None) -> None:
        for chat in self.chat_ids:
            self.api.send_message(chat, f"{title}\n\n{text}")
            if self.images and images:
                self.api.send_photos(chat, [data for _, data in images])


VIDEO_LEAD_SECONDS = 3  # clips start this long before the alert, to see it happen
CHANNELS: dict[str, type[Sink]] = {"webhook": WebhookSink, "email": EmailSink,
                                   "telegram": TelegramSink}
FILTER_KEYS = ("rules", "events", "labels", "hours", "cooldown_seconds")


def make_channel(config: dict[str, Any]) -> Sink:
    """{type, ...settings, rules?, events?, labels?, hours?, cooldown_seconds?} -> a Sink."""
    config = dict(config)
    kind = config.pop("type", None)
    if kind not in CHANNELS:
        raise ValueError(f"Unknown alert channel {kind!r}; available: {', '.join(CHANNELS)}")
    filters = {k: config.pop(k) for k in FILTER_KEYS if k in config}
    sink = CHANNELS[kind](**config)
    sink.filter = ChannelFilter(**filters)
    return sink


class Dispatcher:
    """Background sender with a bounded queue and a couple of retries per sink."""

    def __init__(self, sinks: list[Sink], max_queue: int = 100, retries: int = 2) -> None:
        self.sinks = sinks
        self.retries = retries
        self._queue: queue.Queue[tuple[dict[str, Any], bytes | None] | None] = queue.Queue(max_queue)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, payload: dict[str, Any], jpeg: bytes | None) -> None:
        try:
            self._queue.put_nowait((payload, jpeg))
        except queue.Full:
            log.warning("Alert queue full: dropping alert %s/%s", payload["rule"], payload["type"])

    def close(self, timeout: float = 30.0) -> None:
        self._queue.put(None)
        self._thread.join(timeout=timeout)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            payload, jpeg = item
            for sink in self.sinks:
                if not sink.accepts(payload):
                    continue
                for attempt in range(self.retries + 1):
                    try:
                        sink.send(payload, jpeg)
                        break
                    except Exception as exc:  # network errors must never kill detection
                        if attempt == self.retries:
                            log.error("%s failed for %s/%s: %s", type(sink).__name__,
                                      payload["rule"], payload["type"], exc)
                        else:
                            threading.Event().wait(1.0 * (attempt + 1))
