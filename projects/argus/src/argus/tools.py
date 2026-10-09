"""The tools the chat model can call, over the Argus store.

    objects   tracked objects by time and/or class: "what happened 6pm to 9pm", "how many cars
              today", "people seen with a dog" (exact detector labels, no guessing)
    search    visual search by description (CLIP): "man in a red jacket", "open gate"
    alerts    what the site's alert rules fired (areas and lines): "did any alert go off last
              night", "who entered the yard"
    face      a person from a photo the user attached: "did this person come to the office
              between 1 and 4?" (only when face search is enabled)
    people    people by body pose and clothing colour: "anyone lying on the floor last night",
              "hands up near the till", "red top and blue trousers" (when poses are recorded)

All take a time range (plain words or exact ISO times) and a camera (its name or an alias from
`about.aliases`). objects and search also take `with_labels`: keep only what was on the same
camera at the same time as these other objects.

The model never sees the database, only these JSON results; thumbnails of what it was shown are
kept in `evidence` so the answer can point at them. It never sees an attached photo either: the
photo's face print stays here and the model refers to it by name (photo1).
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from argus.engine.cameras import camera_about
from argus.engine.detector import COCO_CLASSES
from argus.engine.faces import choose_face
from argus.engine.indexer import faces_enabled
from argus.llm import Tool, ToolCall, ToolResult
from argus.engine.pose import CARRIED, COLOURS, MOTIONS, POSTURES
from argus.engine.search import SearchEngine, group_visits

_FILTERS: dict[str, Any] = {
    "when": {"type": "string", "description":
             "Time range in plain words: 'yesterday', 'today 6pm to 9pm', 'last 2 hours', "
             "'this morning', 'last friday', '2026-10-02'"},
    "since": {"type": "string", "description": "Exact start, ISO 'YYYY-MM-DD HH:MM' (overrides when)"},
    "until": {"type": "string", "description": "Exact end, ISO 'YYYY-MM-DD HH:MM' (overrides when)"},
    "camera": {"type": "string", "description": "Only this camera (name or alias)"},
    "with_labels": {"type": "array", "items": {"type": "string"}, "description":
                    "Keep only results seen on the same camera at the same time as ALL of these "
                    "detector classes, e.g. ['dog'] for 'with a dog'"},
    "within_seconds": {"type": "number",
                       "description": "How close in time counts as 'together' (default 15)"},
}

ALERT_TYPES = ("enter", "exit", "dwell", "cross", "down", "hands_up")
_ISO_KEYS = {"first_seen", "last_seen", "face_seen"}  # same time format as the other tools
_WHEN = {k: _FILTERS[k] for k in ("when", "since", "until", "camera")}

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
    Tool("alerts", (
        "Alerts fired by the site's rules: an object entering or leaving an area, staying longer "
        "than allowed (dwell) or crossing a line (cross, with direction in/out), and from body "
        "poses a person lying down for a while (down, rule person_down) or holding both hands "
        "up (hands_up). Exact, from the event log. Use it for: did any alert go off, who "
        "entered an area, what crossed a line, did anyone fall. Rule names are the areas and "
        "lines configured on each camera."),
        {"type": "object", "properties": {
            **_WHEN,
            "rules": {"type": "array", "items": {"type": "string"},
                      "description": "Only these rule (area / line) names"},
            "types": {"type": "array", "items": {"type": "string", "enum": list(ALERT_TYPES)},
                      "description": "Only these event types"},
            "labels": {"type": "array", "items": {"type": "string"},
                       "description": "Only these object labels, e.g. ['person']"},
            "limit": {"type": "integer", "description": "Max events listed, default 30"},
        }}),
]

PEOPLE_TOOL = Tool("people", (
    "People by body pose and clothing colour, from body keypoints recorded about once a second "
    "per person. Exact facts per visit, not ranked guesses: seconds spent standing / sitting / "
    "crouching / bending over / lying, hands raised above the head, facing the camera, head "
    "down (looking at a phone or a desk), still / walking / running, a backpack, handbag, "
    "suitcase or umbrella on the person, and the main colour of the top (torso) and of the "
    "bottom (legs). Use it for: someone lying on the floor, hands up, anyone running, who came "
    "in with a suitcase, a man in a red top, blue trousers. 'A bag' = handbag or backpack. "
    "Running is fast movement across the picture; someone far away walking straight towards "
    "the camera can read as still. Colours are only these names: "
    f"{', '.join(COLOURS)}; map others to the nearest (navy -> blue, beige -> white or brown). "
    "Infrared night images have no colour (everything grey or black). Only people tall enough "
    "in the picture get a pose; for other clothing details use `search`."),
    {"type": "object", "properties": {
        "postures": {"type": "array", "items": {"type": "string", "enum": list(POSTURES[:-1])},
                     "description": "Any of these postures for at least 2 seconds"},
        "arms": {"type": "string", "enum": ["up", "both_up"],
                 "description": "up: one or both hands above the head; both_up: both"},
        "facing_camera": {"type": "boolean", "description": "Only people who faced the camera"},
        "head_down": {"type": "boolean",
                      "description": "Only people who looked down (phone, desk) for 2+ seconds"},
        "motion": {"type": "array", "items": {"type": "string", "enum": list(MOTIONS[:-1])},
                   "description": "Any of these for at least 2 seconds"},
        "carrying": {"type": "array", "items": {"type": "string", "enum": list(CARRIED)},
                     "description": "Any of these on the person for at least 2 seconds"},
        "upper_colour": {"type": "string", "enum": list(COLOURS), "description": "Top colour"},
        "lower_colour": {"type": "string", "enum": list(COLOURS),
                         "description": "Trousers / skirt colour"},
        **_FILTERS,
        "limit": {"type": "integer", "description": "Max visits listed, default 20"},
    }})

FACE_TOOL = Tool("face", (
    "Find a person from a photo the user attached (photo1, photo2, ...; you only get its name). "
    "Compares the photo's face with the faces recorded on the cameras. Each visit has a level: "
    "likely (strong match over several frames) or possible (weak: the user should check the "
    "thumbnail). first_seen / last_seen cover the whole visit, face_seen is when the face was "
    "clear. `checked` says how many faces each camera recorded in the range and whether it can "
    "see faces: no match is not proof the person was not there."),
    {"type": "object", "properties": {
        "photo": {"type": "string", "description": "Name of the attached photo, e.g. 'photo1'"},
        **_WHEN,
        "limit": {"type": "integer", "description": "Max visits listed, default 10"},
    }, "required": ["photo"]})


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


def _strings(values: list[str] | str | None, lower: bool = True) -> list[str]:
    if not values:
        return []
    items = [values] if isinstance(values, str) else list(values)
    return [v.strip().lower() if lower else v.strip() for v in items]


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
    def __init__(self, engine: SearchEngine, now: datetime | None = None,
                 cameras: dict[str, Any] | None = None,
                 faces_config: dict[str, Any] | None = None,
                 pose_config: dict[str, Any] | None = None) -> None:
        """cameras / faces_config / pose_config: the config's `cameras`, `faces` and `pose`
        sections (camera descriptions, which cameras record faces, whether poses are recorded)."""
        self.engine = engine
        self.now = now  # fixed clock for tests; None = real time
        self.about = camera_about(cameras)
        self.faces_config = faces_config
        db = Path(engine.store.path).resolve()
        self._db = sqlite3.connect(db.as_uri() + "?mode=ro", uri=True, check_same_thread=False)
        self.evidence: list[str] = []  # thumbnails (relative to media_dir) shown to the model
        self.photos: dict[str, list[np.ndarray]] = {}  # name -> face prints (one per photo)
        self.only_cameras: list[str] = []  # set by a report: every lookup stays on these
        # the people tool: poses recorded now (config) or before (rows in the store)
        has_rows = self._db.execute("SELECT 1 FROM sqlite_master WHERE name = 'poses'").fetchone() \
            and self._db.execute("SELECT 1 FROM poses LIMIT 1").fetchone()
        self.people_on = bool((pose_config or {}).get("enabled") or has_rows)

    @property
    def tools(self) -> list[Tool]:
        tools = [t for t in TOOLS if t.name != "search" or self.engine.clip is not None]
        tools += [PEOPLE_TOOL] if self.people_on else []
        return tools + ([FACE_TOOL] if self.engine.faces is not None else [])

    def close(self) -> None:
        self._db.close()

    def coverage(self) -> list[dict[str, Any]]:
        """What has been recorded: per camera, first and last indexed time."""
        rows = self._db.execute("SELECT camera, MIN(time), MAX(time), COUNT(*) FROM sightings "
                                "GROUP BY camera ORDER BY camera").fetchall()
        return [{"camera": c, "from": _iso(a), "to": _iso(b), "rows": n} for c, a, b, n in rows]

    def recorded(self, since: float, until: float) -> dict[str, int]:
        """Rows indexed per camera in a time range: a camera missing here recorded nothing."""
        rows = self._db.execute("SELECT camera, COUNT(*) FROM sightings WHERE time >= ? AND "
                                "time < ? GROUP BY camera", (since, until)).fetchall()
        return dict(rows)

    def cameras(self) -> list[str]:
        """One line per camera for the model: what it looks at (`about:`) and what has been
        recorded."""
        recorded = {c["camera"]: c for c in self.coverage()}
        lines = []
        for name in sorted(set(self.about) | set(recorded)):
            about = self.about.get(name, {})
            line = name
            if about.get("aliases"):
                line += f" (also: {', '.join(str(a) for a in about['aliases'])})"
            parts = [str(about[k]).rstrip(".") for k in ("description", "location") if about.get(k)]
            parts += [f"{label}: {about[k]}" for k, label in (
                ("covers", "covers"), ("faces", "faces visible"), ("hours", "usual hours"))
                if about.get(k)]
            if about.get("notes"):
                parts.append(str(about["notes"]).rstrip("."))
            if about.get("zones"):
                parts.append("areas: " + "; ".join(f"{z} = {d}" for z, d in about["zones"].items()))
            text = ". ".join(parts)
            if len(text) > 300:  # small local models have small context windows
                text = text[:297].rsplit(" ", 1)[0] + "..."
            line += f": {text}." if text else ":"
            c = recorded.get(name)
            line += f" Recorded {c['from']} to {c['to']}." if c else " Nothing recorded yet."
            lines.append(line)
        return lines

    def add_photo(self, images: list[np.ndarray],
                  choice: int | None = None) -> tuple[str | None, str]:
        """Attach one or more photos of the same person for the `face` tool. Returns its name
        (None if refused) and a message for the user. choice: which face, if a photo has
        several (1 = leftmost)."""
        if self.engine.faces is None:
            return None, "Face search is off (faces.enabled in the config)."
        prints = []
        for i, image in enumerate(images, 1):
            chosen = choose_face(self.engine.faces.photo_faces(image), choice)
            if isinstance(chosen, str):
                return None, (f"Photo {i}: " if len(images) > 1 else "") + chosen
            prints.append(chosen.print)
        name = f"photo{len(self.photos) + 1}"
        self.photos[name] = prints
        return name, f"Attached as {name}" + (f" ({len(prints)} photos)" if len(prints) > 1 else "")

    def alert_rules(self) -> list[dict[str, Any]]:
        """Rules that have fired so far, per camera, with a count."""
        rows = self._db.execute("SELECT camera, rule, COUNT(*) FROM events GROUP BY camera, rule "
                                "ORDER BY camera, rule").fetchall()
        return [{"camera": c, "rule": r, "events": n} for c, r, n in rows]

    def run(self, call: ToolCall) -> ToolResult:
        try:
            if call.name == "objects":
                out = self.objects(**call.args)
            elif call.name == "search":
                out = self.search(**call.args)
            elif call.name == "alerts":
                out = self.alerts(**call.args)
            elif call.name == "people" and self.people_on:
                out = self.people(**call.args)
            elif call.name == "face" and self.engine.faces is not None:
                out = self.face(**call.args)
            else:
                names = ", ".join(t.name for t in self.tools)
                raise ValueError(f"Unknown tool {call.name!r}; use one of: {names}")
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

    def alerts(self, when: str | None = None, since: str | None = None, until: str | None = None,
               camera: str | None = None, rules: list[str] | str | None = None,
               types: list[str] | str | None = None, labels: list[str] | str | None = None,
               limit: int = 30) -> dict[str, Any]:
        rng = self._range(when or "", since, until, camera)
        wanted_types = _strings(types)
        bad = [t for t in wanted_types if t not in ALERT_TYPES]
        if bad:
            raise ValueError(f"types: {bad} unknown; use {', '.join(ALERT_TYPES)}")
        sql = ("SELECT camera, time, type, rule, label, track_id, direction, dwell_seconds, "
               "snapshot FROM events WHERE 1=1")
        args: list[Any] = []
        for column, values in (("camera", rng.cameras), ("rule", _strings(rules, lower=False)),
                               ("type", wanted_types), ("label", _strings(labels))):
            if values:
                sql += f" AND {column} IN ({','.join('?' * len(values))})"
                args += values
        if rng.since is not None:
            sql += " AND time >= ?"
            args.append(rng.since)
        if rng.until is not None:
            sql += " AND time < ?"
            args.append(rng.until)
        rows = self._db.execute(sql + " ORDER BY time", args).fetchall()
        shown = rows[: max(1, min(int(limit), 200))]
        self.evidence += [r[8] for r in shown[:3] if r[8]]
        keys = ("camera", "time", "type", "rule", "label", "track_id", "direction",
                "dwell_seconds", "snapshot")
        events = [{k: (_iso(v) if k == "time" else v) for k, v in zip(keys, r) if v is not None}
                  for r in shown]
        return {"when": rng.describe(), "cameras": rng.cameras or "all", "total": len(rows),
                "counts": dict(Counter(f"{r[3]} {r[2]}" for r in rows).most_common()),
                "listed": len(shown), "events": events}

    def people(self, postures: list[str] | str | None = None, arms: str | None = None,
               facing_camera: bool = False, upper_colour: str | None = None,
               lower_colour: str | None = None, head_down: bool = False,
               motion: list[str] | str | None = None, carrying: list[str] | str | None = None,
               when: str | None = None,
               since: str | None = None, until: str | None = None, camera: str | None = None,
               with_labels: list[str] | str | None = None, within_seconds: float = 15,
               limit: int = 20) -> dict[str, Any]:
        wanted, moving, carried = _strings(postures), _strings(motion), _strings(carrying)
        for name, values, allowed in (("postures", wanted, POSTURES),
                                      ("motion", moving, MOTIONS), ("carrying", carried, CARRIED),
                                      ("arms", _strings(arms), ("up", "both_up")),
                                      ("upper_colour", _strings(upper_colour), COLOURS),
                                      ("lower_colour", _strings(lower_colour), COLOURS)):
            bad = [v for v in values if v not in allowed]
            if bad:
                raise ValueError(f"{name}: {bad} unknown; use {', '.join(allowed)}")
        rng = self._range(when or "", since, until, camera)
        found = self.engine.people(rng.since, rng.until, rng.cameras or None, wanted or None,
                                   arms.lower() if arms else None,
                                   "camera" if facing_camera else None,
                                   upper_colour.lower() if upper_colour else None,
                                   lower_colour.lower() if lower_colour else None,
                                   head="down" if head_down else None,
                                   motion=moving or None, carrying=carried or None)
        by_key = {(v.camera, v.track_id, v.first_seen): v for v in found}
        visits = [Visit(v.camera, v.track_id, "person", v.first_seen, v.last_seen, None)
                  for v in found]
        visits = self._together(visits, with_labels, within_seconds)
        shown = [by_key[(v.camera, v.track_id, v.first)]
                 for v in visits[: max(1, min(int(limit), 100))]]
        self.engine.add_thumbs(shown)
        self.evidence += [v.thumb for v in shown[:3] if v.thumb]
        filters = {"postures": wanted, "arms": arms, "facing_camera": facing_camera or None,
                   "head_down": head_down or None, "motion": moving, "carrying": carried,
                   "upper_colour": upper_colour, "lower_colour": lower_colour,
                   "with_labels": with_labels}
        return {"when": rng.describe(), "cameras": rng.cameras or "all",
                "filters": {k: v for k, v in filters.items() if v},
                "total": len(visits), "listed": len(shown),
                "visits": [{k: (x.replace("T", " ") if k in _ISO_KEYS else x)
                            for k, x in v.to_dict().items() if k != "time"} for v in shown]}

    def face(self, photo: str, when: str | None = None, since: str | None = None,
             until: str | None = None, camera: str | None = None,
             limit: int = 10) -> dict[str, Any]:
        prints = self.photos.get(str(photo).strip().lower())
        if prints is None:
            raise ValueError(f"No photo {photo!r}. Attached: {', '.join(self.photos) or 'none'}"
                             " (the user attaches one with /photo)")
        rng = self._range(when or "", since, until, camera)
        cams = rng.cameras or None
        visits = self.engine.face_search(prints, rng.since, rng.until, cams,
                                         top_k=max(1, min(int(limit), 50)))
        counts = self.engine.store.face_counts(rng.since, rng.until, cams)
        checked = []
        for name in rng.cameras or sorted(set(self.about) | set(counts)):
            c: dict[str, Any] = {"camera": name, "faces_recorded": counts.get(name, 0)}
            if self.faces_config is not None:
                c["face_recording_on"] = faces_enabled(self.faces_config, name)
            if self.about.get(name, {}).get("faces"):
                c["face_view"] = self.about[name]["faces"]
            checked.append(c)
        self.evidence += [v.best.thumb for v in visits[:3] if v.best.thumb]
        return {"photo": photo, "when": rng.describe(), "cameras": rng.cameras or "all",
                "likely": sum(v.level == "likely" for v in visits),
                "possible": sum(v.level == "possible" for v in visits),
                "visits": [{k: x.replace("T", " ") if k in _ISO_KEYS else x
                            for k, x in v.to_dict().items() if k != "time"} for v in visits],
                "checked": checked}

    # ------------------------------------------------------------------ helpers

    def _range(self, text: str, since: str | None, until: str | None,
               camera: str | None) -> Range:
        parsed = self.engine.parse(text, now=self.now)
        start, end = parsed.range()
        if since:
            start = _timestamp(since, "since")
        if until:
            end = _timestamp(until, "until")
        cameras = [self._camera(camera)] if camera else parsed.cameras
        if self.only_cameras:
            if cameras and not set(cameras) & set(self.only_cameras):
                raise ValueError(f"Only these cameras can be checked: {', '.join(self.only_cameras)}")
            cameras = [c for c in cameras if c in self.only_cameras] or list(self.only_cameras)
        return Range(parsed.text, start, end, cameras)

    def _camera(self, camera: str) -> str:
        """A camera name, or an alias of one ("front desk" -> lobby)."""
        name = self.engine.camera_aliases.get(camera.strip().lower(), camera.strip())
        known = self.engine.camera_names or sorted(set(self.about))
        if known and name not in known:
            raise ValueError(f"No camera {camera!r}. Cameras: {', '.join(known)}")
        return name

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
