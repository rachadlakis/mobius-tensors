"""The tools the chat model can call, over the Histor store.

    objects   tracked objects by time and/or class: "what happened 6pm to 9pm", "how many cars
              today", "people seen with a dog" (exact detector labels, no guessing)
    search    visual search by description (CLIP): "man in a red jacket", "open gate"

Both take the same filters: a time range (plain words or exact ISO times), a camera, and
`with_labels`: keep only what was on the same camera at the same time as these other objects.

The model never sees the database, only these JSON results; thumbnails of what it was shown are
kept in `evidence` so the answer can point at them.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import shared  # noqa: F401
from detector import COCO_CLASSES
from llm import Tool, ToolCall, ToolResult
from query import parse_query
from search import SearchEngine, group_visits

_FILTERS: dict[str, Any] = {
    "when": {"type": "string", "description":
             "Time range in plain words: 'yesterday', 'today 6pm to 9pm', 'last 2 hours', "
             "'this morning', 'last friday', '2026-10-02'"},
    "since": {"type": "string", "description": "Exact start, ISO 'YYYY-MM-DD HH:MM' (overrides when)"},
    "until": {"type": "string", "description": "Exact end, ISO 'YYYY-MM-DD HH:MM' (overrides when)"},
    "camera": {"type": "string", "description": "Only this camera"},
    "with_labels": {"type": "array", "items": {"type": "string"}, "description":
                    "Keep only results seen on the same camera at the same time as ALL of these "
                    "detector classes, e.g. ['dog'] for 'with a dog'"},
    "within_seconds": {"type": "number",
                       "description": "How close in time counts as 'together' (default 15)"},
}

TOOLS = [
    Tool("objects", (
        "Tracked objects from the detector, in time order, with a count per class. Exact, not a "
        "guess. Use it for: what happened in a time range (no labels), how many / when "
        "(labels=['car']), and combinations (labels=['person'], with_labels=['dog']). One tracked "
        "object = one visit: a car parked all day counts once, a person who leaves and comes back "
        f"counts twice. Detector classes: {', '.join(COCO_CLASSES)}."),
        {"type": "object", "properties": {
            "labels": {"type": "array", "items": {"type": "string"},
                       "description": "Only these classes (default: all)"},
            **_FILTERS,
            "limit": {"type": "integer", "description": "Max visits listed, default 30"},
        }}),
    Tool("search", (
        "Visual search by description (CLIP image-text similarity): appearance, colors, "
        "clothing, actions, and things with no detector class. Results are ranked guesses; score "
        "is relative (about 0.2-0.35), so low-ranked hits can be wrong. Combine with the same "
        "time / camera / with_labels filters as `objects`."),
        {"type": "object", "properties": {
            "query": {"type": "string", "description": "What it looks like, e.g. 'white van'"},
            **_FILTERS,
            "limit": {"type": "integer", "description": "Max results, default 8"},
        }, "required": ["query"]}),
]


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t).isoformat(sep=" ", timespec="seconds")


def _timestamp(value: str, name: str) -> float:
    try:
        return datetime.fromisoformat(value.strip()).timestamp()
    except ValueError:
        raise ValueError(f"{name}={value!r} is not an ISO time like '2026-10-02 18:00'") from None


def _labels(values: list[str] | str | None, name: str) -> list[str]:
    if not values:
        return []
    labels = [v.strip().lower() for v in ([values] if isinstance(values, str) else values)]
    unknown = [v for v in labels if v not in COCO_CLASSES]
    if unknown:
        raise ValueError(f"{name}: {unknown} not detector classes. Classes: {', '.join(COCO_CLASSES)}."
                         " For anything else use the search tool.")
    return labels


@dataclass
class Visit:
    camera: str
    track_id: int | None
    label: str | None
    first: float
    last: float
    thumb: str | None
    score: float | None = None
    with_: list[Visit] = field(default_factory=list)

    def overlaps(self, other: Visit, slack: float) -> bool:
        return (self.camera == other.camera and other.first <= self.last + slack
                and other.last >= self.first - slack
                and not (self.track_id is not None and self.track_id == other.track_id))

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"camera": self.camera, "label": self.label, "track_id": self.track_id,
                             "first_seen": _iso(self.first), "last_seen": _iso(self.last),
                             "thumb": self.thumb}
        if self.score is not None:
            d["score"] = round(self.score, 4)
        if self.with_:
            d["with"] = [{"label": w.label, "track_id": w.track_id,
                          "first_seen": _iso(w.first)} for w in self.with_[:5]]
        return d


@dataclass
class Range:
    text: str  # the query with the time and camera words removed
    since: float | None
    until: float | None
    cameras: list[str]

    def describe(self) -> str:
        if self.since is None and self.until is None:
            return "any time"
        return f"{_iso(self.since) if self.since else 'start'} - {_iso(self.until) if self.until else 'now'}"


class Toolbox:
    def __init__(self, engine: SearchEngine, now: datetime | None = None) -> None:
        self.engine = engine
        self.now = now  # fixed clock for tests; None = real time
        db = Path(engine.store.path).resolve()
        self._db = sqlite3.connect(db.as_uri() + "?mode=ro", uri=True, check_same_thread=False)
        self.evidence: list[str] = []  # thumbnails (relative to media_dir) shown to the model

    def close(self) -> None:
        self._db.close()

    def coverage(self) -> list[dict[str, Any]]:
        """What has been recorded: per camera, first and last indexed time."""
        rows = self._db.execute("SELECT camera, MIN(time), MAX(time), COUNT(*) FROM sightings "
                                "GROUP BY camera ORDER BY camera").fetchall()
        return [{"camera": c, "from": _iso(a), "to": _iso(b), "rows": n} for c, a, b, n in rows]

    def run(self, call: ToolCall) -> ToolResult:
        try:
            if call.name == "objects":
                out = self.objects(**call.args)
            elif call.name == "search":
                out = self.search(**call.args)
            else:
                raise ValueError(f"Unknown tool {call.name!r}; use one of: objects, search")
            return ToolResult(call, json.dumps(out, ensure_ascii=False))
        except (TypeError, ValueError) as e:  # bad arguments: tell the model so it can retry
            return ToolResult(call, f"Error: {e}", is_error=True)

    # ------------------------------------------------------------------ tools

    def objects(self, labels: list[str] | str | None = None, when: str | None = None,
                since: str | None = None, until: str | None = None, camera: str | None = None,
                with_labels: list[str] | str | None = None, within_seconds: float = 15,
                limit: int = 30) -> dict[str, Any]:
        wanted = _labels(labels, "labels")
        rng = self._range(when or "", since, until, camera)
        visits = self._visits(wanted, rng.since, rng.until, rng.cameras)
        visits = self._together(visits, with_labels, within_seconds)
        shown = visits[: max(1, min(int(limit), 200))]
        self.evidence += [v.thumb for v in shown[:3] if v.thumb]
        return {"when": rng.describe(), "cameras": rng.cameras or "all",
                "labels": wanted or "all", "with_labels": with_labels or None,
                "counts": dict(Counter(v.label for v in visits).most_common()),
                "total": len(visits), "listed": len(shown),
                "visits": [v.to_dict() for v in shown]}

    def search(self, query: str, when: str | None = None, since: str | None = None,
               until: str | None = None, camera: str | None = None,
               with_labels: list[str] | str | None = None, within_seconds: float = 15,
               limit: int = 8) -> dict[str, Any]:
        if self.engine.clip is None:
            raise ValueError("Visual search is not available (no CLIP text model)")
        # Time and camera words may be in the query itself ("red car yesterday on gate").
        rng = self._range(f"{query} {when or ''}", since, until, camera)
        limit = max(1, min(int(limit), 20))
        vector = self.engine.clip.embed_query(rng.text)
        matches = self.engine.store.search("sightings", vector, rng.since, rng.until,
                                           rng.cameras or None, top_k=limit * 50)
        visits = [Visit(v.camera, v.track_id, v.label or v.best.kind, v.first_seen, v.last_seen,
                        v.best.thumb, v.score) for v in group_visits(matches)]
        visits = self._together(visits, with_labels, within_seconds)[:limit]
        self.evidence += [v.thumb for v in visits[:3] if v.thumb]
        return {"looked_for": rng.text, "when": rng.describe(), "cameras": rng.cameras or "all",
                "with_labels": with_labels or None, "results": [v.to_dict() for v in visits]}

    # ------------------------------------------------------------------ helpers

    def _range(self, text: str, since: str | None, until: str | None,
               camera: str | None) -> Range:
        parsed = parse_query(text, now=self.now, camera_names=self.engine.camera_names)
        start, end = parsed.range()
        if since:
            start = _timestamp(since, "since")
        if until:
            end = _timestamp(until, "until")
        return Range(parsed.text, start, end, [camera] if camera else parsed.cameras)

    def _visits(self, labels: list[str], since: float | None, until: float | None,
                cameras: list[str]) -> list[Visit]:
        """Tracked objects (one row per track and label), in time order."""
        sql = ("SELECT camera, track_id, label, MIN(time), MAX(time), MAX(thumb) FROM sightings "
               "WHERE kind = 'object'")
        args: list[Any] = []
        for column, values in (("label", labels), ("camera", cameras)):
            if values:
                sql += f" AND {column} IN ({','.join('?' * len(values))})"
                args += values
        if since is not None:
            sql += " AND time >= ?"
            args.append(since)
        if until is not None:
            sql += " AND time < ?"
            args.append(until)
        sql += " GROUP BY camera, track_id, label ORDER BY MIN(time)"
        return [Visit(*row) for row in self._db.execute(sql, args).fetchall()]

    def _together(self, visits: list[Visit], with_labels: list[str] | str | None,
                  within: float) -> list[Visit]:
        """Keep visits that overlap (same camera, +-within seconds) a visit of every label."""
        required = _labels(with_labels, "with_labels")
        if not required or not visits:
            return visits
        within = max(0.0, float(within))
        lo = min(v.first for v in visits) - within
        hi = max(v.last for v in visits) + within
        others = self._visits(required, lo, hi + 1e-3, sorted({v.camera for v in visits}))
        kept = []
        for v in visits:
            near = [o for o in others if v.overlaps(o, within)]
            if all(any(o.label == label for o in near) for label in required):
                v.with_ = near
                kept.append(v)
        return kept
