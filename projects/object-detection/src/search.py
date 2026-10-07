"""Search the index: by words ("red shirt last friday") or by a photo of a face.

Results are grouped into *visits*: all matches of the same tracked object on the same camera
become one result (first seen, last seen, best thumbnail), so one person walking past the
camera for 40 seconds is one answer, not twenty.

Face search scores a visit by its best 3 face matches (one lucky frame is weak evidence, four
good ones are strong), labels it "likely" or "possible", and stretches it to the whole person
track: someone seen from behind at 13:40 who shows their face at 13:44 arrived at 13:40.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from faces import LIKELY_SAME_PERSON, SAME_PERSON_THRESHOLD, choose_face
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


@dataclass
class FaceVisit:
    camera: str
    track_id: int | None
    first_seen: float  # the whole visit (person track), not only the frames with a face
    last_seen: float
    score: float       # mean of the visit's best 3 face matches
    best: Match        # best-matching face: its thumbnail is the evidence
    frames: int        # face prints of this visit at or above the threshold
    checked: int       # face prints of this visit compared
    level: str         # likely | possible

    def to_dict(self) -> dict[str, Any]:
        iso = lambda t: datetime.fromtimestamp(t).isoformat(timespec="seconds")  # noqa: E731
        return {
            "camera": self.camera, "track_id": self.track_id, "level": self.level,
            "first_seen": iso(self.first_seen), "last_seen": iso(self.last_seen),
            "score": round(self.score, 4), "best_score": round(self.best.score, 4),
            "frames_matched": self.frames, "frames_checked": self.checked,
            "face_seen": iso(self.best.time), "thumb": self.best.thumb, "time": self.best.time,
        }


def group_face_visits(matches: list[Match], threshold: float = SAME_PERSON_THRESHOLD,
                      likely: float = LIKELY_SAME_PERSON, top: int = 3,
                      gap_seconds: float = 1800.0) -> list[FaceVisit]:
    """matches: every face print compared (not only those above the threshold), so a visit's
    score also counts its frames that did not match. Best visits first."""
    groups: dict[tuple[str, Any], list[list[Match]]] = {}
    for m in sorted(matches, key=lambda m: m.time):
        runs = groups.setdefault((m.camera, m.track_id if m.track_id is not None else m.id), [])
        if runs and m.time - runs[-1][-1].time <= gap_seconds:  # a reused track id is a new run
            runs[-1].append(m)
        else:
            runs.append([m])
    visits = []
    for (camera, _), runs in groups.items():
        for run in runs:
            ranked = sorted(run, key=lambda m: -m.score)
            if ranked[0].score < threshold:
                continue
            score = float(np.mean([m.score for m in ranked[:top]]))
            visits.append(FaceVisit(
                camera, ranked[0].track_id, run[0].time, run[-1].time, score, ranked[0],
                sum(m.score >= threshold for m in run), len(run),
                "likely" if score >= likely else "possible"))
    return sorted(visits, key=lambda v: -v.score)


class SearchEngine:
    def __init__(self, store: EventStore, clip: Any = None, faces: Any = None,
                 camera_names: list[str] | None = None,
                 camera_aliases: dict[str, str] | None = None) -> None:
        """clip: ClipEncoder with a text model; faces: FaceEngine. Either may be None.
        camera_aliases: other names people use for a camera ("front office" -> office)."""
        self.store = store
        self.clip = clip
        self.faces = faces
        self.camera_names = camera_names or []
        self.camera_aliases = camera_aliases or {}

    def parse(self, text: str, now: datetime | None = None) -> ParsedQuery:
        return parse_query(text, now=now, camera_names=self.camera_names,
                           camera_aliases=self.camera_aliases)

    def text(self, query: str, cameras: list[str] | None = None, top_k: int = 20,
             now: datetime | None = None) -> dict[str, Any]:
        if self.clip is None:
            raise RuntimeError("Text search needs the CLIP models (search.clip in the config)")
        parsed = self.parse(query, now)
        cams = cameras or parsed.cameras or None
        since, until = parsed.range()
        vector = self.clip.embed_query(parsed.text)
        matches = self.store.search("sightings", vector, since, until, cams, top_k=top_k * 50)
        visits = group_visits(matches)[:top_k]
        return {"query": _parsed_dict(parsed, cams), "results": [v.to_dict() for v in visits]}

    def face(self, image: np.ndarray, when: str | None = None, cameras: list[str] | None = None,
             threshold: float = SAME_PERSON_THRESHOLD, likely: float = LIKELY_SAME_PERSON,
             top_k: int = 20, now: datetime | None = None,
             choice: int | None = None) -> dict[str, Any]:
        """"Did this man come into my shop on Friday?" -> photo + when="friday" + cameras.
        choice: which face in a photo with several (1-based, left to right)."""
        if self.faces is None:
            raise RuntimeError("Face search needs the face models (faces section in the config)")
        parsed = self.parse(when or "", now)
        cams = cameras or parsed.cameras or None
        since, until = parsed.range()
        faces = self.faces.photo_faces(image)
        chosen = choose_face(faces, choice)
        query = _parsed_dict(parsed, cams)
        if isinstance(chosen, str):
            return {"face_found": False, "answer": chosen, "faces_in_photo": len(faces),
                    "query": query, "results": []}
        visits = self.face_search([chosen.print], since, until, cams, threshold, likely, top_k)
        checked = self.store.face_counts(since, until, cams)
        return {"face_found": True, "answer": face_answer(visits, checked, parsed.when),
                "faces_in_photo": len(faces), "threshold": threshold, "likely": likely,
                "checked": checked, "query": query, "results": [v.to_dict() for v in visits]}

    def face_search(self, prints: list[np.ndarray], since: float | None = None,
                    until: float | None = None, cameras: list[str] | None = None,
                    threshold: float = SAME_PERSON_THRESHOLD,
                    likely: float = LIKELY_SAME_PERSON, top_k: int = 20) -> list[FaceVisit]:
        """Visits of the person in `prints` (one per reference photo), best first."""
        matches = self.store.search("faces", np.stack(prints), since, until, cameras, top_k=None)
        visits = group_face_visits(matches, threshold, likely)[:top_k]
        for v in visits:
            if v.track_id is not None:
                v.first_seen, v.last_seen = self.store.track_span(v.camera, v.track_id,
                                                                  v.first_seen, v.last_seen)
        return visits


def face_answer(visits: list[FaceVisit], checked: dict[str, int], when: str = "") -> str:
    fmt = "%a %d %b %H:%M"
    likely = [v for v in visits if v.level == "likely"]
    if likely:
        first = min(likely, key=lambda v: v.first_seen)
        return (f"Yes, likely: {len(likely)} visit(s), first on {first.camera} at "
                f"{datetime.fromtimestamp(first.first_seen):{fmt}} (match {first.score:.2f}).")
    if visits:
        best = visits[0]
        return (f"Possibly: {len(visits)} weak match(es), best on {best.camera} at "
                f"{datetime.fromtimestamp(best.first_seen):{fmt}} (match {best.score:.2f}). "
                "Check the face thumbnails.")
    span = f" ({when})" if when else ""
    if not checked:
        return f"No faces were recorded{span}, so this can't be answered."
    return (f"No match among {sum(checked.values())} recorded faces{span}. That is not proof "
            "the person wasn't there: faces turned away or too far are not recorded.")


def _parsed_dict(parsed: ParsedQuery, cameras: list[str] | None) -> dict[str, Any]:
    return {
        "text": parsed.text, "cameras": cameras or "all", "when": parsed.when or "any time",
        "since": parsed.since.isoformat(timespec="seconds") if parsed.since else None,
        "until": parsed.until.isoformat(timespec="seconds") if parsed.until else None,
    }
