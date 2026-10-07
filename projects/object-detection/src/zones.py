"""Zone (polygon) and line (tripwire) rules evaluated on tracked objects.

Zone events:  enter, exit, dwell (inside longer than `dwell_seconds`, e.g. a parked car)
Line events:  cross (with direction "in" / "out"; "in" = the side the drawn arrow points to)

Points may be pixels or normalized 0..1 (resolution independent); normalized points are
scaled to the frame size on the first frame. Each object is represented by one anchor
point of its box: bottom-center (where it touches the ground) by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from detector import color_for, draw_box, draw_label
from tracker import Track

ANCHORS = {
    "bottom_center": lambda b: ((b[0] + b[2]) / 2, b[3]),
    "center": lambda b: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2),
}


@dataclass(frozen=True)
class ZoneEvent:
    timestamp: float  # seconds (video time for files, wall-clock offset for live sources)
    type: str  # enter | exit | dwell | cross
    rule: str  # zone / line name
    track_id: int
    label: str
    direction: str | None = None  # lines only: in | out
    dwell_seconds: float | None = None  # dwell/exit only

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


def resolve_points(points: list[list[float]], frame_size: tuple[int, int]) -> np.ndarray:
    """(w, h) frame size. Treat all-0..1 coordinates as normalized."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if pts.max() <= 1.0:
        pts = pts * np.array(frame_size, dtype=float)
    return pts


class Rule:
    kind = ""

    def __init__(
        self, name: str, points: list[list[float]], classes: list[str] | None = None,
        anchor: str = "bottom_center",
    ) -> None:
        if anchor not in ANCHORS:
            raise ValueError(f"Unknown anchor {anchor!r}, expected one of {list(ANCHORS)}")
        self.name = name
        self.raw_points = points
        self.points = np.zeros((0, 2))
        self.classes = set(classes or [])
        self.anchor = ANCHORS[anchor]

    def resolve(self, frame_size: tuple[int, int]) -> None:
        self.points = resolve_points(self.raw_points, frame_size)

    def applies_to(self, track: Track) -> bool:
        return not self.classes or track.label in self.classes

    def update(self, tracks: list[Track], removed: list[Track], t: float) -> list[ZoneEvent]:
        raise NotImplementedError


FULL_FRAME = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]


class Zone(Rule):
    kind = "zone"

    def __init__(
        self, name: str, points: list[list[float]] | None = None, classes: list[str] | None = None,
        dwell_seconds: float | None = None, confirm_frames: int = 2,
        anchor: str = "bottom_center",
    ) -> None:
        points = points or FULL_FRAME  # no points = the whole picture ("seen anywhere")
        super().__init__(name, points, classes, anchor)
        if len(points) < 3:
            raise ValueError(f"Zone {name!r} needs at least 3 points")
        self.dwell_seconds = dwell_seconds or None
        self.confirm_frames = confirm_frames  # hysteresis against jitter at the border
        self.inside: dict[int, float] = {}  # track id -> time it entered
        self.dwell_fired: set[int] = set()
        self._pending: dict[int, int] = {}  # track id -> consecutive frames in the other state

    def contains(self, point: tuple[float, float]) -> bool:
        contour = self.points.astype(np.float32).reshape(-1, 1, 2)
        return cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0

    def dwell_time(self, track_id: int, t: float) -> float | None:
        return t - self.inside[track_id] if track_id in self.inside else None

    def update(self, tracks: list[Track], removed: list[Track], t: float) -> list[ZoneEvent]:
        events: list[ZoneEvent] = []
        for track in tracks:
            if track.missed or not self.applies_to(track):
                continue  # coasting tracks keep their last state
            is_in = self.contains(self.anchor(track.box))
            was_in = track.id in self.inside
            if is_in == was_in:
                self._pending.pop(track.id, None)
            else:
                self._pending[track.id] = self._pending.get(track.id, 0) + 1
                if self._pending[track.id] >= self.confirm_frames:
                    self._pending.pop(track.id)
                    if is_in:
                        self.inside[track.id] = t
                        events.append(ZoneEvent(t, "enter", self.name, track.id, track.label))
                    else:
                        events.append(self._exit(track, t))

            if (
                self.dwell_seconds
                and track.id in self.inside
                and track.id not in self.dwell_fired
                and t - self.inside[track.id] >= self.dwell_seconds
            ):
                self.dwell_fired.add(track.id)
                events.append(ZoneEvent(
                    t, "dwell", self.name, track.id, track.label,
                    dwell_seconds=round(t - self.inside[track.id], 2),
                ))

        for track in removed:  # lost for good while inside -> treat as exit
            self._pending.pop(track.id, None)
            if track.id in self.inside:
                events.append(self._exit(track, t))
        return events

    def _exit(self, track: Track, t: float) -> ZoneEvent:
        entered = self.inside.pop(track.id)
        self.dwell_fired.discard(track.id)
        return ZoneEvent(
            t, "exit", self.name, track.id, track.label, dwell_seconds=round(t - entered, 2)
        )


class Line(Rule):
    kind = "line"

    def __init__(
        self, name: str, points: list[list[float]], classes: list[str] | None = None,
        direction: str = "both", anchor: str = "bottom_center",
    ) -> None:
        super().__init__(name, points, classes, anchor)
        if len(points) != 2:
            raise ValueError(f"Line {name!r} needs exactly 2 points")
        if direction not in {"both", "in", "out"}:
            raise ValueError(f"Line {name!r}: direction must be both, in or out")
        self.direction = direction
        self.counts = {"in": 0, "out": 0}
        self.last: dict[int, tuple[float, float]] = {}  # track id -> last anchor point

    def side(self, p: tuple[float, float]) -> float:
        (x1, y1), (x2, y2) = self.points
        return (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1)

    def _crosses_segment(self, a: tuple[float, float], b: tuple[float, float]) -> bool:
        """Movement a->b crosses the drawn segment (not just its infinite extension)."""
        s1, s2 = self.side(a), self.side(b)
        if s1 * s2 >= 0:
            return False
        p1, p2 = self.points
        d = np.array(b) - np.array(a)
        o1 = d[0] * (p1[1] - a[1]) - d[1] * (p1[0] - a[0])
        o2 = d[0] * (p2[1] - a[1]) - d[1] * (p2[0] - a[0])
        return o1 * o2 <= 0

    def update(self, tracks: list[Track], removed: list[Track], t: float) -> list[ZoneEvent]:
        events: list[ZoneEvent] = []
        for track in tracks:
            if track.missed or not self.applies_to(track):
                continue
            point = self.anchor(track.box)
            if self.side(point) == 0:
                continue  # exactly on the line: wait until it is clearly on one side
            prev = self.last.get(track.id)  # last point that was clearly on one side
            self.last[track.id] = point
            if prev is None or not self._crosses_segment(prev, point):
                continue
            direction = "in" if self.side(point) > 0 else "out"
            self.counts[direction] += 1
            if self.direction in ("both", direction):
                events.append(
                    ZoneEvent(t, "cross", self.name, track.id, track.label, direction=direction)
                )
        for track in removed:
            self.last.pop(track.id, None)
        return events


class ZoneMonitor:
    """Evaluates all rules for one camera."""

    def __init__(self, rules: list[Rule]) -> None:
        names = [r.name for r in rules]
        if len(names) != len(set(names)):
            raise ValueError(f"Zone/line names must be unique: {names}")
        self.rules = rules
        self._frame_size: tuple[int, int] | None = None

    @classmethod
    def from_config(cls, items: list[dict[str, Any]] | None) -> ZoneMonitor:
        rules: list[Rule] = []
        for item in items or []:
            item = dict(item)
            kind = item.pop("type", "zone")
            if kind == "zone":
                rules.append(Zone(**item))
            elif kind == "line":
                rules.append(Line(**item))
            else:
                raise ValueError(f"Unknown rule type {kind!r} (expected zone or line)")
        return cls(rules)

    def update(
        self, tracks: list[Track], removed: list[Track], t: float, frame_size: tuple[int, int]
    ) -> list[ZoneEvent]:
        if frame_size != self._frame_size:
            for rule in self.rules:
                rule.resolve(frame_size)
            self._frame_size = frame_size
        events: list[ZoneEvent] = []
        for rule in self.rules:
            events.extend(rule.update(tracks, removed, t))
        return events

    def draw(self, frame: np.ndarray, tracks: list[Track], t: float) -> np.ndarray:
        h, w = frame.shape[:2]
        if self._frame_size != (w, h):
            for rule in self.rules:
                rule.resolve((w, h))
            self._frame_size = (w, h)
        draw_rules(frame, self.rules)
        for track in tracks:
            text = f"{track.label} #{track.id}"
            dwell = [
                z.dwell_time(track.id, t) or 0.0
                for z in self.rules
                if isinstance(z, Zone) and track.id in z.inside
            ]
            if dwell:
                text += f" {max(dwell):.0f}s"
            draw_box(frame, track.box, text, color_for(track.id))
        return frame


def draw_rules(frame: np.ndarray, rules: list[Rule]) -> None:
    zone_colors = {
        r.name: (0, 0, 255) if r.inside else (0, 200, 0) for r in rules if isinstance(r, Zone)
    }
    # Translucent fills first, then crisp outlines and labels on top.
    overlay = frame.copy()
    for rule in rules:
        if isinstance(rule, Zone):
            cv2.fillPoly(overlay, [rule.points.astype(np.int32)], zone_colors[rule.name])
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, dst=frame)

    for rule in rules:
        pts = rule.points.astype(np.int32)
        if isinstance(rule, Zone):
            color = zone_colors[rule.name]
            cv2.polylines(frame, [pts], True, color, 2)
            draw_label(frame, f"{rule.name}: {len(rule.inside)}", (int(pts[0][0]), int(pts[0][1])), color)
        elif isinstance(rule, Line):
            color = (255, 160, 0)
            p1 = (int(pts[0][0]), int(pts[0][1]))
            p2 = (int(pts[1][0]), int(pts[1][1]))
            cv2.line(frame, p1, p2, color, 3)
            mid = (pts[0] + pts[1]) / 2
            normal = np.array([-(pts[1][1] - pts[0][1]), pts[1][0] - pts[0][0]], dtype=float)
            normal = normal / (np.linalg.norm(normal) + 1e-9) * 25
            tip = mid + normal
            cv2.arrowedLine(
                frame, (int(mid[0]), int(mid[1])), (int(tip[0]), int(tip[1])), color, 2,
                tipLength=0.4,
            )
            draw_label(
                frame, f"{rule.name} in:{rule.counts['in']} out:{rule.counts['out']}", p1, color
            )
