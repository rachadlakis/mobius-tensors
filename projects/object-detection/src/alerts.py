"""Turn zone/line events into alerts: console, JSON-lines file, event store, webhook, email.

Filtering (event types, cooldown) happens once; every emitted event then goes to all sinks.
Webhook and email are sent from a background thread so a slow mail server never stalls
detection. Each event can carry a JPEG snapshot of the frame with the object highlighted.
"""

from __future__ import annotations

import json
import logging
import queue
import smtplib
import threading
import urllib.request
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from cameras import expand_env
from zones import ZoneEvent

if TYPE_CHECKING:
    from store import EventStore
    from tracker import Track

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
        snapshots: bool = True,
        store: EventStore | None = None,
    ) -> None:
        """
        events: only alert on these event types (e.g. [dwell, cross]); None = all.
        cooldown_seconds: minimum time between two alerts for the same rule + event type.
        webhook: {url, headers?, timeout?} -> POST JSON (+ snapshot as base64-free URL path).
        email: {host, port?, username?, password?, sender, to: [..], starttls?, subject_prefix?}
        snapshots: save a JPEG per alert (needs a store) and attach it to emails.
        """
        self.console = console
        self.jsonl = Path(jsonl) if jsonl else None
        self.event_types = set(events) if events else None
        self.cooldown = cooldown_seconds
        self.source = source
        self.store = store
        self.snapshots = snapshots
        self.sinks: list[Sink] = []
        if webhook:
            self.sinks.append(WebhookSink(**webhook))
        if email:
            self.sinks.append(EmailSink(**email))
        self._dispatcher = Dispatcher(self.sinks) if self.sinks else None
        self._last: dict[tuple[str, str], float] = {}
        if self.jsonl:
            self.jsonl.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, config: dict[str, Any] | None, source: str | None = None,
                    store: EventStore | None = None) -> AlertManager:
        return cls(**(config or {}), source=source, store=store)

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
        """Wait for queued webhooks/emails to go out."""
        if self._dispatcher:
            self._dispatcher.close()

    def _emit(self, event: ZoneEvent, frame: np.ndarray | None, tracks: list[Track] | None,
              wall_time: float | None) -> None:
        if self.console:
            extra = event.direction or (
                f"{event.dwell_seconds:.1f}s" if event.dwell_seconds is not None else ""
            )
            print(
                f"[{event.timestamp:8.2f}s] {event.type.upper():5s} {event.rule}: "
                f"{event.label} #{event.track_id} {extra}".rstrip()
            )
        if self.jsonl:
            record = {**event.to_dict(), "source": self.source}
            with self.jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        if self.store is None and not self.sinks:
            return

        t = wall_time if wall_time is not None else datetime.now().timestamp()
        camera = self.source or "camera"
        jpeg, snapshot = None, None
        if self.snapshots and frame is not None:
            image = snapshot_image(frame, event, tracks)
            ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            jpeg = buf.tobytes() if ok else None
            if self.store is not None:
                name = f"{camera}_{int(t * 1000)}_{event.rule}_{event.type}_{event.track_id}"
                snapshot = self.store.save_image(image, "snapshots", _safe(name), t)
        event_id = None
        if self.store is not None:
            event_id = self.store.add_event(camera, t, event.to_dict(), snapshot)
        if self._dispatcher:
            payload = {
                **event.to_dict(), "id": event_id, "camera": camera, "time": t,
                "iso_time": datetime.fromtimestamp(t).isoformat(timespec="seconds"),
                "snapshot": snapshot,
            }
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


class Sink:
    def send(self, payload: dict[str, Any], jpeg: bytes | None) -> None:
        raise NotImplementedError


class WebhookSink(Sink):
    """POST the event as JSON. Point it at n8n/Zapier/Slack workflows or your own backend."""

    def __init__(self, url: str, headers: dict[str, str] | None = None, timeout: float = 10.0):
        self.url = expand_env(url)
        self.headers = {k: expand_env(v) for k, v in (headers or {}).items()}
        self.timeout = timeout

    def send(self, payload: dict[str, Any], jpeg: bytes | None) -> None:
        request = urllib.request.Request(
            self.url, data=json.dumps(payload).encode(), method="POST",
            headers={"Content-Type": "application/json", **self.headers},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            response.read()


class EmailSink(Sink):
    def __init__(self, host: str, sender: str, to: list[str] | str, port: int = 587,
                 username: str | None = None, password: str | None = None,
                 starttls: bool = True, ssl: bool = False,
                 subject_prefix: str = "[Mobius Tensors]", timeout: float = 20.0):
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
        what = payload.get("direction") or payload.get("dwell_seconds") or ""
        msg["Subject"] = (f"{self.subject_prefix} {payload['type']} {payload['rule']} - "
                          f"{payload.get('label')} on {payload['camera']}").strip()
        msg["From"], msg["To"] = self.sender, ", ".join(self.to)
        lines = [
            f"Camera: {payload['camera']}",
            f"Time:   {payload['iso_time']}",
            f"Event:  {payload['type']} ({payload['rule']}) {what}".rstrip(),
            f"Object: {payload.get('label')} #{payload.get('track_id')}",
        ]
        msg.set_content("\n".join(lines))
        if jpeg:
            msg.add_attachment(jpeg, maintype="image", subtype="jpeg", filename="snapshot.jpg")
        smtp_cls = smtplib.SMTP_SSL if self.ssl else smtplib.SMTP
        with smtp_cls(self.host, self.port, timeout=self.timeout) as smtp:
            if self.starttls and not self.ssl:
                smtp.starttls()
            if self.username:
                smtp.login(self.username, self.password or "")
            smtp.send_message(msg)


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
