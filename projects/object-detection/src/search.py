"""Search the index: by words ("red shirt last friday") or by a photo of a face.

Results are grouped into *visits*: all matches of the same tracked object on the same camera
become one result (first seen, last seen, best thumbnail), so one person walking past the
camera for 40 seconds is one answer, not twenty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from faces import SAME_PERSON_THRESHOLD
from query import ParsedQuery, parse_query
from store import EventStore, Match


@dataclass
class Visit:
    camera: str
    track_id: int | None
    first_seen: float
    last_seen: float
    score: float
    best: Match
    matches: int = 1
    label: str | None = None
    kinds: set[str] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        iso = lambda t: datetime.fromtimestamp(t).isoformat(timespec="seconds")  # noqa: E731
        return {
            "camera": self.camera, "track_id": self.track_id, "label": self.label,
            "first_seen": iso(self.first_seen), "last_seen": iso(self.last_seen),
            "score": round(self.score, 4), "matches": self.matches,
            "thumb": self.best.thumb, "box": self.best.box, "kind": self.best.kind,
            "time": self.best.time,
        }


def group_visits(matches: list[Match], gap_seconds: float = 60.0) -> list[Visit]:
    """Merge matches of the same track on the same camera; scene/tile matches that are close in
    time on the same camera are merged too (same moment seen from several tiles)."""
    visits: list[Visit] = []
    by_key: dict[tuple[str, Any], Visit] = {}
    for m in sorted(matches, key=lambda m: -m.score):
        if m.track_id is not None:
            key: tuple[str, Any] = (m.camera, m.track_id)
        else:
            key = (m.camera, round(m.time / gap_seconds))
        v = by_key.get(key)
        if v is None:
            v = Visit(m.camera, m.track_id, m.time, m.time, m.score, m, label=m.label)
            by_key[key] = v
            visits.append(v)
        else:
            v.first_seen, v.last_seen = min(v.first_seen, m.time), max(v.last_seen, m.time)
            v.matches += 1
        if m.kind:
            v.kinds.add(m.kind)
    return visits


class SearchEngine:
    def __init__(self, store: EventStore, clip: Any = None, faces: Any = None,
                 camera_names: list[str] | None = None) -> None:
        """clip: ClipEncoder with a text model; faces: FaceEngine. Either may be None."""
        self.store = store
        self.clip = clip
        self.faces = faces
        self.camera_names = camera_names or []

    def text(self, query: str, cameras: list[str] | None = None, top_k: int = 20,
             now: datetime | None = None) -> dict[str, Any]:
        if self.clip is None:
            raise RuntimeError("Text search needs the CLIP models (search.clip in the config)")
        parsed = parse_query(query, now=now, camera_names=self.camera_names)
        cams = cameras or parsed.cameras or None
        since, until = parsed.range()
        vector = self.clip.embed_query(parsed.text)
        matches = self.store.search("sightings", vector, since, until, cams, top_k=top_k * 50)
        visits = group_visits(matches)[:top_k]
        return {"query": _parsed_dict(parsed, cams), "results": [v.to_dict() for v in visits]}

    def face(self, image: np.ndarray, when: str | None = None, cameras: list[str] | None = None,
             threshold: float = SAME_PERSON_THRESHOLD, top_k: int = 20,
             now: datetime | None = None) -> dict[str, Any]:
        """"Did this man come into my shop on Friday?" -> photo + when="friday" + cameras."""
        if self.faces is None:
            raise RuntimeError("Face search needs the face models (faces section in the config)")
        parsed = parse_query(when or "", now=now, camera_names=self.camera_names)
        cams = cameras or parsed.cameras or None
        since, until = parsed.range()
        found = self.faces.best_face_print(image)
        if found is None:
            return {"face_found": False, "answer": "No face found in the photo.",
                    "query": _parsed_dict(parsed, cams), "results": []}
        vector, _ = found
        matches = self.store.search("faces", vector, since, until, cams,
                                    min_score=threshold, top_k=top_k * 20)
        visits = group_visits(matches)[:top_k]
        if visits:
            first = min(visits, key=lambda v: v.first_seen)
            answer = (f"Yes: seen {len(visits)} time(s), first on {first.camera} at "
                      f"{datetime.fromtimestamp(first.first_seen):%a %d %b %H:%M}.")
        else:
            answer = "No match found" + (f" ({parsed.when})" if parsed.when else "") + "."
        return {"face_found": True, "answer": answer, "threshold": threshold,
                "query": _parsed_dict(parsed, cams), "results": [v.to_dict() for v in visits]}


def _parsed_dict(parsed: ParsedQuery, cameras: list[str] | None) -> dict[str, Any]:
    return {
        "text": parsed.text, "cameras": cameras or "all", "when": parsed.when or "any time",
        "since": parsed.since.isoformat(timespec="seconds") if parsed.since else None,
        "until": parsed.until.isoformat(timespec="seconds") if parsed.until else None,
    }
