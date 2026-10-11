"""Alert verification: a local vision model looks at each alert before it is sent.

Rules (areas, lines, poses, bags left behind) fire on boxes and keypoints; a shadow, a poster or a
person half hidden behind a desk can trip them. Before an alert goes to its channels, the vision
model behind the chat's `look` tool (vision.py) gets the alert's snapshot and a close-up of the
object, and a yes/no question for that event type ("Is the person in the red box lying on the
ground?"). It answers YES, NO or UNSURE and one reason.

- The event is stored at once, as before; the verdict is added to its row when it comes
  (`verified`: yes | no | unsure | error | skipped, `verification`: the model's reason).
- mode `veto` (default): a NO holds the alert back from the channels. Everything else is sent,
  with the verdict in the message. mode `tag`: everything is sent, with the verdict.
- Never a silent loss: model down, an answer later than `timeout`, or a full queue sends the
  alert unverified (`error` / `skipped`). The model can veto, it never raises an alert itself.
- Runs on its own thread: detection never waits. On a CPU each check takes seconds, so the
  alert arrives that much later; limit `events` to the alerts that matter.

    alerts:
      verify:
        enabled: true
        events: [down, abandoned, enter]   # event types to check (default: all with a question)
        rules: []                          # only these rules (zone/line names); empty = all
        mode: veto                         # veto | tag
        timeout: 60                        # seconds per check
        questions:                         # override or add per event type
          enter: "Is there a real person (not a shadow, reflection or poster) in the red box?"
        # model, base_url, max_side...: default to chat.vision
"""

from __future__ import annotations

import logging
import queue
import re
import threading
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

log = logging.getLogger(__name__)

SYSTEM = (
    "You check alarms from security cameras. Image 1 is the camera's picture with the object "
    "that raised the alarm in a red box. Image 2, when given, is a close-up of that box. Answer "
    "only from what you can see. First line: YES, NO or UNSURE. Second line: one short reason. "
    "Say UNSURE when the image is too small, dark or blurred to tell.")

# {label} (and its plural {labels}), {rule}, {count} come from the event. Types without a question are not checked
# (tamper and sound: the picture alone can't confirm them; plate: the reader already voted).
QUESTIONS: dict[str, str] = {
    "enter": "Is there a real {label} in the red box, not a shadow, reflection, poster or "
             "other object?",
    "exit": "Is there a real {label} in the red box, not a shadow, reflection, poster or "
            "other object?",
    "dwell": "Is there a real {label} in the red box, not a shadow, reflection, poster or "
             "other object?",
    "cross": "Is there a real {label} in the red box, not a shadow, reflection, poster or "
             "other object?",
    "crowd": "Are there at least {count} {labels} in the picture?",
    "down": "Is the person in the red box lying or fallen on the ground?",
    "hands_up": "Does the person in the red box have both hands raised above their head?",
    "abandoned": "Is the object in the red box a bag or suitcase left on its own, with nobody "
                 "holding or standing next to it?",
}
PLURALS = {"person": "people", "bus": "buses"}
VERDICTS = ("yes", "no", "unsure")
MODES = ("veto", "tag")
_FIRST_WORD = re.compile(r"[a-z]+")


def parse_answer(text: str) -> tuple[str, str]:
    """Model answer -> (yes | no | unsure, reason). Anything unclear is unsure."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return "unsure", ""
    word = _FIRST_WORD.search(lines[0].lower())
    verdict = word.group() if word and word.group() in VERDICTS else "unsure"
    reason = " ".join(lines[1:]) if len(lines) > 1 else lines[0]
    reason = re.sub(r"^(yes|no|unsure)\b[\s.,:;-]*", "", reason, flags=re.IGNORECASE)
    return verdict, reason.strip()[:300]


def close_up(frame: np.ndarray, box: Any, pad: float = 0.5, min_side: int = 96) -> np.ndarray:
    """The box widened by `pad` of its size each way (some context around the object),
    enlarged so its shorter side is at least min_side pixels."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box)
    bw, bh = x2 - x1, y2 - y1
    x1, x2 = max(0, int(x1 - pad * bw)), min(w, int(x2 + pad * bw))
    y1, y2 = max(0, int(y1 - pad * bh)), min(h, int(y2 + pad * bh))
    crop = frame[y1:max(y1 + 1, y2), x1:max(x1 + 1, x2)]
    scale = min_side / max(1, min(crop.shape[:2]))
    if scale > 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return crop


class AlertVerifier:
    def __init__(self, ask: Callable[[list[np.ndarray], str, str], str],
                 events: list[str] | None = None, rules: list[str] | None = None,
                 mode: str = "veto", questions: dict[str, str] | None = None,
                 max_queue: int = 20) -> None:
        """ask(images, question, system) -> the model's answer (VisionModel.ask).
        events: event types to check; None = every type that has a question."""
        if mode not in MODES:
            raise ValueError(f"alerts.verify.mode {mode!r}: use {' or '.join(MODES)}")
        self.questions = {**QUESTIONS, **(questions or {})}
        unknown = [e for e in events or [] if e not in self.questions]
        if unknown:
            raise ValueError(f"alerts.verify.events: no question for {unknown}; add one under "
                             "alerts.verify.questions")
        self.ask = ask
        self.events = set(events) if events else set(self.questions)
        self.rules = set(rules or [])
        self.mode = mode
        self._queue: queue.Queue[tuple[Any, ...] | None] = queue.Queue(max_queue)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @classmethod
    def from_config(cls, config: dict[str, Any] | None,
                    vision: dict[str, Any] | None = None) -> AlertVerifier | None:
        """`alerts.verify`, with the model settings of `chat.vision` as defaults. None when
        missing, not enabled, or no model is set."""
        if not config or not config.get("enabled"):
            return None
        from histor.vision import VisionModel

        own = {k: config[k] for k in ("events", "rules", "mode", "questions", "max_queue")
               if k in config}
        model_keys = ("model", "base_url", "api_key", "max_side", "temperature", "timeout", "api")
        settings = {k: v for k, v in (vision or {}).items() if k in model_keys}
        settings.update({k: config[k] for k in model_keys if k in config})
        settings.setdefault("timeout", 60)
        model = VisionModel.from_config({**settings, "max_images": 2})
        if model is None:
            raise ValueError("alerts.verify needs a vision model: set alerts.verify.model "
                             "or chat.vision.model")
        log.info("Alert verification with %s (%s)", model.model, own.get("mode", "veto"))
        return cls(model.ask, **own)

    def applies(self, payload: dict[str, Any]) -> bool:
        return payload.get("type") in self.events and (
            not self.rules or payload.get("rule") in self.rules)

    def question(self, payload: dict[str, Any]) -> str:
        label = payload.get("label") or "object"
        fields = {"label": label, "labels": PLURALS.get(label, label + "s"),
                  "rule": payload.get("rule", ""), "count": payload.get("count") or "several"}
        return self.questions[payload["type"]].format(**fields)

    def submit(self, payload: dict[str, Any], images: list[np.ndarray],
               done: Callable[[dict[str, Any], str, str], None]) -> None:
        """Check in the background; done(payload, verdict, reason) gets the outcome. With no
        room in the queue, done() is called at once with `skipped`."""
        try:
            self._queue.put_nowait((payload, images, done))
        except queue.Full:
            log.warning("Verification queue full: %s/%s sent unverified",
                        payload.get("rule"), payload.get("type"))
            done(payload, "skipped", "verification queue full")

    def close(self, timeout: float = 120.0) -> None:
        self._queue.put(None)
        self._thread.join(timeout=timeout)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            payload, images, done = item
            try:
                verdict, reason = parse_answer(self.ask(images, self.question(payload), SYSTEM))
            except Exception as exc:  # the model must never stop an alert
                log.error("Alert verification failed for %s/%s: %s", payload.get("rule"),
                          payload.get("type"), exc)
                verdict, reason = "error", str(exc)[:300]
            try:
                done(payload, verdict, reason)
            except Exception:
                log.exception("Handling the verdict of %s/%s failed", payload.get("rule"),
                              payload.get("type"))

    def sends(self, verdict: str) -> bool:
        """Whether an alert with this verdict goes to the channels."""
        return not (self.mode == "veto" and verdict == "no")
