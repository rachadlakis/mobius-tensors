"""Moving things the detector has no class for (a cart, a drone, a ladder being carried, a
thrown object), found by motion alone so they are searchable even when they cross the
picture in a second.

What moved comes from the camera's change analysis (change.py: shake cancelled, background
learned, light changes left out). Here it is only followed:

    regions     what differs from the background, at least `min_area` of the frame, and not
                already covered by a tracked object (those are embedded by the indexer)
    blobs       regions are followed from frame to frame by overlap, or by a nearby centre
                (a fast object jumps far between two frames). A blob seen in `confirm_frames`
                frames is reported at once, then again when it shows clearly bigger (a better
                view) or every `refresh_seconds` while it stays; at most `max_per_second`
                crops a second in all, first sightings first. A turned camera (a new view)
                forgets them all
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from histor.engine.boxes import area, centre, inside, iou_matrix
from histor.engine.change import ANALYSER_KEYS

if TYPE_CHECKING:
    from histor.engine.change import Change


@dataclass
class _Blob:
    id: int
    box: np.ndarray            # frame pixels
    last_seen: float
    hits: int = 1
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))  # centre, px per frame
    embedded_area: float = 0.0  # 0: never reported
    last_embed: float = -1e18
    reported: bool = False


class MotionWatch:
    def __init__(self, min_area: float = 0.002, max_area: float = 0.6, confirm_frames: int = 2,
                 growth: float = 1.5, refresh_seconds: float = 5.0, min_gap_seconds: float = 1.0,
                 max_per_second: int = 4, forget_seconds: float = 1.0, max_jump: float = 3.0,
                 **analysis: Any) -> None:
        """min_area / max_area: share of the frame a region covers; max_jump: how far (in object
        sizes) a blob may move between two frames and still be the same one. analysis: the
        change analysis keys older configs have here (change.ANALYSER_KEYS), read there."""
        unknown = set(analysis) - set(ANALYSER_KEYS)
        if unknown:
            raise TypeError(f"search.motion: unknown {sorted(unknown)}")
        self.min_area, self.max_area = min_area, max_area
        self.confirm_frames = confirm_frames
        self.growth = growth
        self.refresh = refresh_seconds
        self.min_gap = min_gap_seconds
        self.max_per_second = max_per_second
        self.forget = forget_seconds
        self.max_jump = max_jump
        self._blobs: dict[int, _Blob] = {}
        self._next_id = 1
        self._sent: list[float] = []
        self.reports = 0  # blobs reported, in all
        self.blobs = 0    # distinct blobs reported

    def update(self, change: Change, t: float,
               covered: list[np.ndarray] | None = None) -> list[tuple[int, np.ndarray]]:
        """Blobs to embed now: (blob id, box in frame pixels). covered: boxes of tracked
        objects that are embedded anyway."""
        if change.restarted:
            self._blobs.clear()
        if not change.ready or change.light:
            return []
        found = [box for box in change.regions(change.fg, self.min_area, self.max_area)
                 if not any(inside(box, c) >= 0.6 for c in covered or [])]
        return self._follow(found, t)

    def retry(self, ids: list[int]) -> None:
        """Blobs reported but not embedded (the indexer was busy): report them again."""
        for i in ids:
            if i in self._blobs:
                self._blobs[i].embedded_area, self._blobs[i].last_embed = 0.0, -1e18

    def _follow(self, boxes: list[np.ndarray], t: float) -> list[tuple[int, np.ndarray]]:
        free = dict(self._blobs)
        for box in sorted(boxes, key=area, reverse=True):
            blob = self._match(box, free)
            if blob is None:
                blob = _Blob(self._next_id, box, t)
                self._blobs[blob.id] = blob
                self._next_id += 1
                continue
            del free[blob.id]
            blob.velocity = centre(box) - centre(blob.box)
            blob.box, blob.last_seen, blob.hits = box, t, blob.hits + 1
        self._blobs = {i: b for i, b in self._blobs.items() if t - b.last_seen <= self.forget}

        self._sent = [s for s in self._sent if t - s < 1.0]
        due = [b for b in self._blobs.values()
               if b.last_seen == t and b.hits >= self.confirm_frames and (
                   not b.embedded_area or (t - b.last_embed >= self.min_gap and (
                       area(b.box) >= self.growth * b.embedded_area
                       or t - b.last_embed >= self.refresh)))]
        due.sort(key=lambda b: (b.embedded_area > 0, -area(b.box)))  # first sightings first
        out = []
        for blob in due[:max(0, self.max_per_second - len(self._sent))]:
            if not blob.reported:
                blob.reported = True
                self.blobs += 1
            blob.embedded_area, blob.last_embed = area(blob.box), t
            self._sent.append(t)
            self.reports += 1
            out.append((blob.id, blob.box.copy()))
        return out

    def _match(self, box: np.ndarray, free: dict[int, _Blob]) -> _Blob | None:
        best, best_score = None, 0.0
        for blob in free.values():
            guess = blob.box + np.tile(blob.velocity, 2)  # where it should be now
            size = max(box[2] - box[0], box[3] - box[1], guess[2] - guess[0], guess[3] - guess[1])
            jump = float(np.linalg.norm(centre(box) - centre(guess))) / max(size, 1.0)
            overlap = float(iou_matrix(box[None], guess[None])[0, 0])
            score = 1.0 + overlap if overlap > 0 else 1.0 - jump / self.max_jump
            if score > best_score:
                best, best_score = blob, score
        return best


