"""Turn zone/line events into alerts: console log, JSON-lines file (webhooks etc. later)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from zones import ZoneEvent


class AlertManager:
    def __init__(
        self,
        console: bool = True,
        jsonl: str | Path | None = None,
        events: list[str] | None = None,
        cooldown_seconds: float = 0.0,
        source: str | None = None,
    ) -> None:
        """
        events: only alert on these event types (e.g. [dwell, cross]); None = all.
        cooldown_seconds: minimum time between two alerts for the same rule + event type.
        """
        self.console = console
        self.jsonl = Path(jsonl) if jsonl else None
        self.event_types = set(events) if events else None
        self.cooldown = cooldown_seconds
        self.source = source
        self._last: dict[tuple[str, str], float] = {}
        if self.jsonl:
            self.jsonl.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, config: dict[str, Any] | None, source: str | None = None) -> AlertManager:
        return cls(**(config or {}), source=source)

    def handle(self, events: list[ZoneEvent]) -> list[ZoneEvent]:
        """Filter events, emit them to the sinks, and return the ones emitted."""
        emitted = []
        for event in events:
            if self.event_types and event.type not in self.event_types:
                continue
            key = (event.rule, event.type)
            if self.cooldown and event.timestamp - self._last.get(key, -1e18) < self.cooldown:
                continue
            self._last[key] = event.timestamp
            emitted.append(event)
            self._emit(event)
        return emitted

    def _emit(self, event: ZoneEvent) -> None:
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
