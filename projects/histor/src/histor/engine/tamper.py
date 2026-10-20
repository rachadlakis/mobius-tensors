"""Camera tampering and health, from the picture alone (no model): a `tamper` event when a
camera is covered, sprayed, turned or broken, so a site is never silently blind.

Every `interval_seconds`, on a 320-pixel-wide grey copy of the frame:

    blocked     almost no contrast (grey-level spread below `flat_std`): lens covered,
                sprayed, pointed at a wall
    dark        mean brightness below `dark_mean` with some contrast left: lights off, IR
                broken
    blurred     sharpness (variance of the Laplacian) below `blur_ratio` of this camera's own
                usual sharpness, learned over its first `learn_seconds`, and its crispest
                edges (99.9th percentile) soft too: defocused, fogged. A plain truck's side in
                front of the camera has little detail but crisp edges (its logo, a door): not
                blurred, nor while the detector sees an object filling the frame or a
                partial cover grows into a full one
    moved       the camera was turned, tilted or zoomed. Judged on landmarks: corners of the
                usual view (ORB features) found again in the picture.
                  * landmarks still in place, spread over the picture: not moved, whatever
                    covers the rest (a truck parked in front, a crowd)
                  * landmarks found all shifted / turned / zoomed together: moved
                  * no landmark found: a new view, or something covering nearly all of it.
                    Covered when the detector sees (or saw in the last `big_memory_seconds`)
                    an object filling `big_object` of the frame, or when the cover grew from
                    a partial one; otherwise moved
                A view with few corners (night, plain walls) is judged on its structure: the
                correlation of the gradient maps (80x45, edges' strength) with the usual view,
                below `moved_similarity` (same view ~0.8, turned by an eighth ~0.4, another
                ~0.2), unless some parts of it still match in place
    obstructed  the camera did not move, but something still (a parked truck, a pallet, a
                sign) hides at least `obstructed_share` of the usual view for
                `obstructed_seconds`: the camera is partly blind. A crowd passing is not
                still, so it is not this. The event's label says what blocks the view when
                the detector saw it ("truck"), else "camera"
    frozen      the picture has not changed at all (to the last grey level): a stuck stream
                (only judged on a usable picture: a covered lens is `blocked`, not frozen)

A condition must hold for `hold_seconds` (obstructed: `obstructed_seconds`) before the event,
and fires once until it clears.

The usual view is kept as up to `views` references: day, night (infrared), and the views
before a move. So the first view coming back (a truck that covered everything driving away)
is known again, and never raises a second "moved". After `moved`, the new view is added as
the main reference. References follow slow changes (daylight, shadows) every
`refresh_minutes` while the camera is in place and its view clear.

On-screen text burnt into the picture (a clock, the camera's name) stays in place when the
camera turns, so it must not count as landmarks. It is learned over `learn_seconds`: pixels
at full white or black that never change.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

EVENT_TYPE = "tamper"
CHECKS = ("blocked", "dark", "blurred", "moved", "obstructed", "frozen")
GRID = 4  # the picture is judged in GRID x GRID cells


@dataclass
class TamperEvent:
    time: float
    kind: str      # one of CHECKS
    value: float   # the measurement that tripped it (std, mean, sharpness ratio, share...)
    label: str | None = None  # obstructed: what blocks the view, when the detector saw it


@dataclass
class Measure:
    std: float
    mean: float
    sharpness: float
    gradient: np.ndarray = field(repr=False)  # 80x45 map of edge strength
    grey: np.ndarray = field(repr=False)
    peak: float = 0.0  # the crispest edges: 99.9th percentile of |Laplacian|


def measure(frame: np.ndarray, width: int = 320) -> Measure:
    h, w = frame.shape[:2]
    scale = width / w
    small = cv2.resize(frame, (width, max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    blurred = cv2.GaussianBlur(grey.astype(np.float32), (0, 0), 2)
    magnitude = cv2.magnitude(cv2.Sobel(blurred, cv2.CV_32F, 1, 0),
                              cv2.Sobel(blurred, cv2.CV_32F, 0, 1))
    gradient = cv2.resize(magnitude, (80, 45), interpolation=cv2.INTER_AREA)
    laplacian = cv2.Laplacian(grey, cv2.CV_64F)
    return Measure(float(grey.std()), float(grey.mean()), float(laplacian.var()), gradient,
                   grey, float(np.percentile(np.abs(laplacian), 99.9)))


def similarity(reference: np.ndarray, current: np.ndarray) -> float:
    """Pearson correlation of two gradient maps: 1 = same structure, ~0 = unrelated."""
    a = reference - reference.mean()
    b = current - current.mean()
    denominator = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum()) / denominator if denominator > 1e-9 else 1.0


@dataclass
class _View:
    """One reference of the usual view."""

    gradient: np.ndarray
    points: np.ndarray               # landmark positions, N x 2 (pixels of the grey copy)
    descriptors: np.ndarray | None   # their ORB descriptors
    added: float
    used: float
    size: tuple[int, ...]            # shape of the grey copy it was taken from


class TamperCheck:
    def __init__(self, interval_seconds: float = 2.0, hold_seconds: float = 20.0,
                 flat_std: float = 6.0, dark_mean: float = 12.0, blur_ratio: float = 0.25,
                 moved_similarity: float = 0.5, learn_seconds: float = 60.0,
                 refresh_minutes: float = 30.0, checks: list[str] | None = None,
                 min_structure: float = 2.0, obstructed_share: float = 0.5,
                 obstructed_seconds: float = 300.0, max_shift: float = 0.015,
                 min_landmarks: int = 15, min_cells: int = 3, views: int = 4,
                 big_object: float = 0.5, big_memory_seconds: float = 30.0) -> None:
        """min_structure: a view (or cell) whose gradient map varies less than this (a blank
        wall) can't tell `moved`. max_shift: landmarks within this share of the width of
        where they were are in place (wind sway, compression). min_landmarks / min_cells:
        how many landmarks, over how many of the GRID x GRID cells, prove a position. big_object:
        share of the frame a detected object must fill to explain a cover of the whole view."""
        self.checks = set(CHECKS if checks is None else checks)
        unknown = self.checks - set(CHECKS)
        if unknown:
            raise ValueError(f"tamper.checks: unknown {sorted(unknown)}; use {list(CHECKS)}")
        self.interval = interval_seconds
        self.hold = hold_seconds
        self.flat_std = flat_std
        self.dark_mean = dark_mean
        self.blur_ratio = blur_ratio
        self.moved_similarity = moved_similarity
        self.learn = learn_seconds
        self.refresh = refresh_minutes * 60
        self.min_structure = min_structure
        self.obstructed_share = obstructed_share
        self.obstructed_seconds = obstructed_seconds
        self.max_shift = max_shift
        self.min_landmarks = min_landmarks
        self.min_cells = min_cells
        self.max_views = views
        self.big_object = big_object
        self.big_memory = big_memory_seconds
        self.started: float | None = None
        self.sharpness: float | None = None  # this camera's usual sharpness
        self.peak = 0.0                      # and its usual crispest edges
        self._learning: list[float] = []
        self._peaks: list[float] = []
        self.views: list[_View] = []  # references of the usual view, the main one first
        self._previous: np.ndarray | None = None           # last grey copy (frozen)
        self._previous_gradient: np.ndarray | None = None  # last gradient map (still cover)
        self._overlay: np.ndarray | None = None  # on-screen text: no landmarks there
        self._range: tuple[np.ndarray, np.ndarray] | None = None  # grey min / max, learning
        self._big: tuple[float, str] = (-1e18, "")  # last time a frame-filling object was seen
        self._covered_at = -1e18  # last sample with at least `obstructed_share` covered
        self._added_at = -1e18
        self._label: str | None = None  # what blocks the view now
        self._last = -1e18
        self._since: dict[str, float] = {}
        self._fired: set[str] = set()
        self.last_measure: Measure | None = None
        self.last_state = ""  # home / moved / covered: how the last sample was judged
        self._orb = cv2.ORB.create(nfeatures=500, edgeThreshold=15, patchSize=15)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

    @property
    def reference(self) -> np.ndarray | None:
        """The main reference's gradient map (None until a view with structure was seen)."""
        return self.views[0].gradient if self.views else None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> TamperCheck:
        return cls(**{k: v for k, v in config.items() if k not in {"enabled", "cameras"}})

    def update(self, frame: np.ndarray, t: float,
               objects: Sequence[tuple[str, Sequence[float]]] | None = None
               ) -> list[TamperEvent]:
        """objects: what the detector found in this frame, as (label, box in frame pixels)."""
        h, w = frame.shape[:2]
        seen = [(name, [box[0] / w, box[1] / h, box[2] / w, box[3] / h])
                for name, box in objects or []]  # boxes as shares of the frame
        for name, (x1, y1, x2, y2) in seen:  # every frame: it may pass between samples
            if (x2 - x1) * (y2 - y1) >= self.big_object:
                self._big = (t, name)
        if t - self._last < self.interval:
            return []
        self._last = t
        m = measure(frame)
        self.last_measure = m
        if self.started is None:
            self.started = t
        active = self._conditions(m, t, seen)
        self._previous = m.grey
        events = []
        for kind in CHECKS:
            if kind not in self.checks:
                continue
            if kind not in active:
                self._since.pop(kind, None)
                self._fired.discard(kind)
                continue
            since = self._since.setdefault(kind, t)
            hold = self.obstructed_seconds if kind == "obstructed" else self.hold
            if kind not in self._fired and t - since >= hold:
                self._fired.add(kind)
                label = (self._label or "camera") if kind == "obstructed" else None
                events.append(TamperEvent(t, kind, round(active[kind], 3), label))
                if kind == "moved":  # the new view is the main reference from now on
                    self._add_view(m, t)
                    self._since.pop("obstructed", None)
        return events

    # ------------------------------------------------------------ judging a sample

    def _conditions(self, m: Measure, t: float,
                    objects: list[tuple[str, list[float]]]) -> dict[str, float]:
        active: dict[str, float] = {}
        if m.std < self.flat_std:
            active["blocked"] = m.std
        elif m.mean < self.dark_mean:
            active["dark"] = m.mean
        if active:
            return active  # no usable picture: frozen, sharpness and structure mean nothing
        if self._previous is not None and self._previous.shape == m.grey.shape and \
                not cv2.absdiff(self._previous, m.grey).any():
            active["frozen"] = 0.0
        # this camera's usual sharpness: median of the first samples, then a slow average
        started = self.started if self.started is not None else t
        if self.sharpness is None:
            self._learning.append(m.sharpness)
            self._peaks.append(m.peak)
            if t - started >= self.learn and self._learning:
                self.sharpness = float(np.median(self._learning))
                self.peak = float(np.median(self._peaks))
        elif m.sharpness < self.blur_ratio * self.sharpness and m.peak < 0.5 * self.peak \
                and t - self._big[0] > self.big_memory \
                and t - self._covered_at > 3 * self.interval:  # not a cover that grew
            active["blurred"] = m.sharpness / self.sharpness
        elif m.sharpness >= self.blur_ratio * self.sharpness:  # not while something covers it
            self.sharpness = 0.99 * self.sharpness + 0.01 * m.sharpness
            self.peak = 0.99 * self.peak + 0.01 * m.peak
        if "blurred" in active:
            return active
        self._learn_overlay(m.grey, t - started)
        if self.views and self.views[0].size != m.grey.shape:
            self.views = []  # another picture size: another camera, learn again
        if not self.views:
            if float(m.gradient.std()) >= self.min_structure:
                self._add_view(m, t)
            self._previous_gradient = m.gradient
            return active
        active.update(self._position(m, t, objects))
        self._previous_gradient = m.gradient
        return active

    def _position(self, m: Measure, t: float,
                  objects: list[tuple[str, list[float]]]) -> dict[str, float]:
        """moved / obstructed for a usable picture, against the references."""
        points, descriptors = self._landmarks(m.grey)
        home, moved_from = None, None
        for view in sorted(self.views, key=lambda v: -v.used):  # most recently seen first
            a, b = self._pairs(view, points, descriptors)
            if len(a) < self.min_landmarks:
                continue
            still = np.linalg.norm(b - a, axis=1) <= self._tolerance(m)
            if still.sum() >= self.min_landmarks and _cells(a[still], m.grey.shape) >= \
                    self.min_cells:
                home = view
                break
            if moved_from is None and self._shifted(a, b, m):
                moved_from = view
        by_structure = False
        if home is None and moved_from is None:  # no landmarks to go by
            home = self._by_structure(m)
            by_structure = home is not None
        if home is not None:
            self.last_state = "home"
            home.used = t
            share, cells = self._blocked(home.gradient, m.gradient)
            if share >= self.obstructed_share:
                self._covered_at = t
                self._label = _what(cells, objects)
                return {"obstructed": share}
            if by_structure and t - self._added_at >= 60 and \
                    similarity(home.gradient, m.gradient) >= 0.7:
                self._add_view(m, t, main=False)  # e.g. the infrared look of the same view
            elif t - home.added >= self.refresh and share < 0.1 and \
                    similarity(home.gradient, m.gradient) >= 0.7:
                self._replace(home, m, t)  # follow daylight and shadows
            return {}
        main = moved_from or self.views[0]
        if moved_from is None:  # nothing known in sight: covered, or a new view
            recent_big = t - self._big[0] <= self.big_memory
            grew = t - self._covered_at <= 3 * self.interval
            if (recent_big or grew) and self._still(m.gradient):
                self.last_state = "covered"
                self._covered_at = t
                self._label = self._big[1] if recent_big else self._label
                return {"obstructed": 1.0}
        self.last_state = "moved"
        return {"moved": similarity(main.gradient, m.gradient)}

    def _shifted(self, a: np.ndarray, b: np.ndarray, m: Measure) -> bool:
        """The landmarks moved together (pan, tilt, zoom, rotation) and are spread out."""
        if len(a) < 3:
            return False
        matrix, inliers = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC,
                                                      ransacReprojThreshold=3.0)
        if matrix is None or inliers is None:
            return False
        keep = inliers.ravel().astype(bool)
        if keep.sum() < self.min_landmarks or _cells(a[keep], m.grey.shape) < self.min_cells:
            return False
        moved = float(np.median(np.linalg.norm(b[keep] - a[keep], axis=1)))
        return moved > self._tolerance(m)

    def _by_structure(self, m: Measure) -> _View | None:
        """For views with few landmarks: the reference whose structure still matches, as a
        whole or in at least `min_cells` cells (and 15% of its structured cells) in place."""
        for view in self.views:
            if similarity(view.gradient, m.gradient) >= self.moved_similarity:
                return view
            if float(view.gradient.std()) < self.min_structure:
                continue
            structured = same = 0
            for ref, cur in zip(_split(view.gradient), _split(m.gradient)):
                if float(ref.std()) < self.min_structure:
                    continue
                structured += 1
                c = _correlation(ref, cur)
                same += c is not None and c >= 0.7
            if same >= self.min_cells and same >= 0.15 * structured:
                return view
        return None

    def _blocked(self, reference: np.ndarray, current: np.ndarray) -> tuple[float, list[int]]:
        """Share of the reference's structured cells hidden by something still: changed
        from the reference, unchanged since the last sample. Also which cells."""
        previous = self._previous_gradient
        if previous is None or previous.shape != current.shape:
            return 0.0, []
        structured, hidden = 0, []
        cells = zip(_split(reference), _split(current), _split(previous))
        for i, (ref, cur, prev) in enumerate(cells):
            if float(ref.std()) < self.min_structure:
                continue
            structured += 1
            c = _correlation(ref, cur)
            changed = c is None or c < 0.5  # None: the cell went plain (a truck's side)
            p = _correlation(prev, cur)
            still = (p is None and _plain(prev) and _plain(cur)) or (p is not None and p >= 0.8)
            if changed and still:
                hidden.append(i)
        return (len(hidden) / structured if structured else 0.0), hidden

    def _still(self, current: np.ndarray) -> bool:
        previous = self._previous_gradient
        if previous is None or previous.shape != current.shape:
            return False
        c = _correlation(previous, current)
        return c is None or c >= 0.8

    # ------------------------------------------------------------ references

    def _landmarks(self, grey: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        keypoints, descriptors = self._orb.detectAndCompute(self._clahe.apply(grey), None)
        points = np.array([k.pt for k in keypoints], np.float32).reshape(-1, 2)
        return points, descriptors

    def _pairs(self, view: _View, points: np.ndarray,
               descriptors: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        """Landmarks of `view` found again: (positions in the view, positions now)."""
        empty = np.zeros((0, 2), np.float32)
        if view.descriptors is None or descriptors is None or len(points) < 2 or \
                len(view.points) < 2:
            return empty, empty
        good = [p[0] for p in self._matcher.knnMatch(view.descriptors, descriptors, k=2)
                if len(p) == 2 and p[0].distance < 0.8 * p[1].distance]
        if not good:
            return empty, empty
        a = view.points[[g.queryIdx for g in good]]
        b = points[[g.trainIdx for g in good]]
        if self._overlay is not None:  # on-screen text never moves with the camera
            h, w = self._overlay.shape
            x = np.clip(a[:, 0].astype(int), 0, w - 1)
            y = np.clip(a[:, 1].astype(int), 0, h - 1)
            keep = ~self._overlay[y, x]
            a, b = a[keep], b[keep]
        return a, b

    def _tolerance(self, m: Measure) -> float:
        return max(3.0, self.max_shift * m.grey.shape[1])

    def _add_view(self, m: Measure, t: float, main: bool = True) -> None:
        points, descriptors = self._landmarks(m.grey)
        view = _View(m.gradient, points, descriptors, t, t, m.grey.shape)
        self.views.insert(0 if main else len(self.views), view)
        self._added_at = t
        while len(self.views) > self.max_views:  # forget the one seen least recently
            self.views.remove(min(self.views[1:], key=lambda v: v.used))

    def _replace(self, view: _View, m: Measure, t: float) -> None:
        view.points, view.descriptors = self._landmarks(m.grey)
        view.gradient, view.added, view.used = m.gradient, t, t

    def _learn_overlay(self, grey: np.ndarray, age: float) -> None:
        """Over the first `learn_seconds`: pixels at full white or black that never change
        are on-screen text (dilated a little: its edges are corners too)."""
        if self._overlay is not None:
            return
        if self._range is None or self._range[0].shape != grey.shape:
            self._range = (grey.copy(), grey.copy())
        low, high = np.minimum(self._range[0], grey), np.maximum(self._range[1], grey)
        self._range = (low, high)
        if age >= self.learn:
            fixed = (high.astype(int) - low.astype(int) <= 4) & ((high >= 235) | (low <= 20))
            self._overlay = cv2.dilate(fixed.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
            self._range = None


def _split(gradient: np.ndarray) -> list[np.ndarray]:
    """The map's GRID x GRID cells, row by row."""
    return [cell for row in np.array_split(gradient, GRID, axis=0)
            for cell in np.array_split(row, GRID, axis=1)]


def _plain(cell: np.ndarray) -> bool:
    return float(cell.std()) < 1.0


def _correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    """Correlation of two cells; None when either is plain (nothing to compare)."""
    if _plain(a) or _plain(b):
        return None
    return similarity(a, b)


def _cells(points: np.ndarray, shape: tuple[int, ...]) -> int:
    """In how many of the GRID x GRID cells the points lie."""
    h, w = shape[:2]
    cols = np.clip((points[:, 0] * GRID / w).astype(int), 0, GRID - 1)
    rows = np.clip((points[:, 1] * GRID / h).astype(int), 0, GRID - 1)
    return len(set(zip(rows.tolist(), cols.tolist())))


def _what(cells: list[int], objects: list[tuple[str, list[float]]]) -> str | None:
    """The detected object covering most of the hidden cells (half of them at least)."""
    if not cells:
        return None
    centres = [((i % GRID + 0.5) / GRID, (i // GRID + 0.5) / GRID) for i in cells]
    best, best_count = None, 0
    for label, (x1, y1, x2, y2) in objects:
        count = sum(x1 <= x <= x2 and y1 <= y <= y2 for x, y in centres)
        if count > best_count:
            best, best_count = label, count
    return best if best_count * 2 >= len(cells) else None


def tamper_enabled(config: dict[str, Any] | None, camera: str) -> bool:
    if not config or not config.get("enabled"):
        return False
    cameras = config.get("cameras") or []
    return not cameras or camera in cameras
