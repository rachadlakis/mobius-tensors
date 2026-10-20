"""Moving things the detector has no class for (a cart, a drone, a ladder being carried, a
thrown object), found by motion alone so they are searchable even when they cross the
picture in a second.

On a small grey copy of each frame (`width` px wide, ~4 ms on a laptop CPU):

    shake       the frame is first lined up with the learned background (phase correlation),
                so a camera swaying in the wind by a few pixels is not motion; a bigger jump
                (the camera was turned) starts the background again
    background  OpenCV MOG2 learns, per pixel, what is normal there, including what always
                moves (leaves, water, a flag); shadows are dropped
    light       when most of the picture changes at once (clouds, lamps switched on), it is
                light, not an object: nothing is reported while the background catches up
    regions     what differs from the background, at least `min_area` of the frame, and not
                already covered by a tracked object (those are embedded by the indexer)
    blobs       regions are followed from frame to frame by overlap, or by a nearby centre
                (a fast object jumps far between two frames). A blob seen in `confirm_frames`
                frames is reported at once, then again when it shows clearly bigger (a better
                view) or every `refresh_seconds` while it stays; at most `max_per_second`
                crops a second in all, first sightings first
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class _Blob:
    id: int
    box: np.ndarray            # frame pixels
    last_seen: float
    hits: int = 1
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))  # centre, px per frame
    embedded_area: float = 0.0  # 0: never reported
    last_embed: float = -1e18


class MotionWatch:
    def __init__(self, width: int = 320, min_area: float = 0.002, max_area: float = 0.6,
                 light_change: float = 0.4, max_shift: float = 0.08, confirm_frames: int = 2,
                 growth: float = 1.5, refresh_seconds: float = 5.0, min_gap_seconds: float = 1.0,
                 max_per_second: int = 4, forget_seconds: float = 1.0, max_jump: float = 3.0,
                 warmup_frames: int = 15, history: int = 300,
                 var_threshold: float = 25.0) -> None:
        """min_area / max_area: share of the frame a region covers; light_change: share of the
        frame changed at once that means light, not an object; max_shift: a camera jump bigger
        than this share of the frame restarts the background; max_jump: how far (in object
        sizes) a blob may move between two frames and still be the same one."""
        self.width = width
        self.min_area, self.max_area = min_area, max_area
        self.light_change = light_change
        self.max_shift = max_shift
        self.confirm_frames = confirm_frames
        self.growth = growth
        self.refresh = refresh_seconds
        self.min_gap = min_gap_seconds
        self.max_per_second = max_per_second
        self.forget = forget_seconds
        self.max_jump = max_jump
        self.warmup = warmup_frames
        self.history, self.var_threshold = history, var_threshold
        self._blobs: dict[int, _Blob] = {}
        self._next_id = 1
        self._sent: list[float] = []
        self._window: np.ndarray | None = None
        self.light_changes = 0  # frames skipped as a change of light
        self.restarts = 0       # camera moved: background learned again
        self._restart()
        self.restarts = 0

    def _restart(self) -> None:
        self._bg = cv2.createBackgroundSubtractorMOG2(self.history, self.var_threshold,
                                                      detectShadows=True)
        self._frames = 0
        self._blobs.clear()
        self.restarts += 1

    def update(self, frame: np.ndarray, t: float,
               covered: list[np.ndarray] | None = None) -> list[tuple[int, np.ndarray]]:
        """Blobs to embed now: (blob id, box in frame pixels). covered: boxes of tracked
        objects that are embedded anyway."""
        h, w = frame.shape[:2]
        scale = self.width / w
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        small = cv2.resize(grey, (self.width, max(1, round(h * scale))),
                           interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (5, 5), 0)
        if self._frames:
            small = self._align(small)
        self._frames += 1
        mask = self._bg.apply(small)
        if self._frames <= self.warmup:
            return []
        fg: np.ndarray = (mask > 200).astype(np.uint8)  # 255 = foreground, 127 = shadow
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))  # a blurred streak
        if fg.mean() > self.light_change:
            self.light_changes += 1
            return []
        sh, sw = fg.shape
        count, _, stats, _ = cv2.connectedComponentsWithStats(fg)
        boxes = []
        for x, y, bw, bh, area in stats[1:count]:
            if not self.min_area * sw * sh <= area <= self.max_area * sw * sh:
                continue
            box = np.array([x, y, x + bw, y + bh], dtype=float) / scale
            if any(_inside(box, c) >= 0.6 for c in covered or []):
                continue
            boxes.append(box)
        return self._follow(boxes, t)

    def retry(self, ids: list[int]) -> None:
        """Blobs reported but not embedded (the indexer was busy): report them again."""
        for i in ids:
            if i in self._blobs:
                self._blobs[i].embedded_area, self._blobs[i].last_embed = 0.0, -1e18

    def _align(self, small: np.ndarray) -> np.ndarray:
        ref = self._bg.getBackgroundImage()
        if ref is None or ref.shape[:2] != small.shape[:2]:
            return small
        if ref.ndim == 3:
            ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        if self._window is None or self._window.shape != small.shape:
            self._window = cv2.createHanningWindow(small.shape[::-1], cv2.CV_32F)
        (dx, dy), response = cv2.phaseCorrelate(ref.astype(np.float32),
                                                small.astype(np.float32), self._window)
        if (abs(dx) < 0.5 and abs(dy) < 0.5) or response < 0.1:  # still, or no clear shift
            return small
        sh, sw = small.shape
        if max(abs(dx) / sw, abs(dy) / sh) > self.max_shift:  # turned: a new view
            self._restart()
            return small
        shift = np.array([[1, 0, -dx], [0, 1, -dy]], dtype=np.float32)
        return cv2.warpAffine(small, shift, (sw, sh), borderMode=cv2.BORDER_REPLICATE)

    def _follow(self, boxes: list[np.ndarray], t: float) -> list[tuple[int, np.ndarray]]:
        free = dict(self._blobs)
        for box in sorted(boxes, key=_area, reverse=True):
            blob = self._match(box, free)
            if blob is None:
                blob = _Blob(self._next_id, box, t)
                self._blobs[blob.id] = blob
                self._next_id += 1
                continue
            del free[blob.id]
            blob.velocity = _centre(box) - _centre(blob.box)
            blob.box, blob.last_seen, blob.hits = box, t, blob.hits + 1
        self._blobs = {i: b for i, b in self._blobs.items() if t - b.last_seen <= self.forget}

        self._sent = [s for s in self._sent if t - s < 1.0]
        due = [b for b in self._blobs.values()
               if b.last_seen == t and b.hits >= self.confirm_frames and (
                   not b.embedded_area or (t - b.last_embed >= self.min_gap and (
                       _area(b.box) >= self.growth * b.embedded_area
                       or t - b.last_embed >= self.refresh)))]
        due.sort(key=lambda b: (b.embedded_area > 0, -_area(b.box)))  # first sightings first
        out = []
        for blob in due[:max(0, self.max_per_second - len(self._sent))]:
            blob.embedded_area, blob.last_embed = _area(blob.box), t
            self._sent.append(t)
            out.append((blob.id, blob.box.copy()))
        return out

    def _match(self, box: np.ndarray, free: dict[int, _Blob]) -> _Blob | None:
        best, best_score = None, 0.0
        for blob in free.values():
            guess = blob.box + np.tile(blob.velocity, 2)  # where it should be now
            size = max(box[2] - box[0], box[3] - box[1], guess[2] - guess[0], guess[3] - guess[1])
            jump = float(np.linalg.norm(_centre(box) - _centre(guess))) / max(size, 1.0)
            overlap = _iou(box, guess)
            score = 1.0 + overlap if overlap > 0 else 1.0 - jump / self.max_jump
            if score > best_score:
                best, best_score = blob, score
        return best


def _area(box: np.ndarray) -> float:
    return float(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))


def _centre(box: np.ndarray) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


def _intersection(a: np.ndarray, b: np.ndarray) -> float:
    return _area(np.array([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = _intersection(a, b)
    return inter / max(1e-6, _area(a) + _area(b) - inter)


def _inside(box: np.ndarray, cover: np.ndarray) -> float:
    """Share of `box` that lies inside `cover`."""
    return _intersection(box, np.asarray(cover, dtype=float)) / max(1e-6, _area(box))
