"""Lightweight multi-object tracker (ByteTrack-style two-stage IoU association).

Gives each object a stable ID across frames so zone/line rules can reason about
"car #7 has been parked for 60 s". Pure numpy + greedy matching so it ports 1:1 to C++.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from argus.engine.detector import Detection


@dataclass
class Track:
    id: int  # 0 until confirmed
    box: np.ndarray  # x1, y1, x2, y2 in image pixels
    score: float
    hits: int = 1
    missed: int = 0  # consecutive frames without a matching detection
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4))
    label_votes: dict[str, float] = field(default_factory=dict)

    @property
    def label(self) -> str:
        """Most confident label seen so far (robust to car/truck flicker)."""
        return max(self.label_votes, key=lambda k: self.label_votes[k])


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N, 4) and (M, 4) xyxy boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def greedy_match(iou: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Match rows to columns by descending IoU; each row/column used at most once."""
    pairs: list[tuple[int, int]] = []
    if iou.size == 0:
        return pairs
    used_rows, used_cols = set(), set()
    for flat in np.argsort(-iou, axis=None):
        r, c = divmod(int(flat), iou.shape[1])
        if iou[r, c] < threshold:
            break
        if r in used_rows or c in used_cols:
            continue
        pairs.append((r, c))
        used_rows.add(r)
        used_cols.add(c)
    return pairs


class Tracker:
    def __init__(
        self,
        track_threshold: float = 0.45,
        match_iou: float = 0.3,
        low_match_iou: float = 0.5,
        max_missed: int = 30,
        min_hits: int = 3,
    ) -> None:
        """
        track_threshold: detections at/above this can start tracks; lower ones only extend them.
        max_missed: frames a track survives without detections (occlusion tolerance).
        min_hits: matches needed before a track gets an ID and is reported.
        """
        self.track_threshold = track_threshold
        self.match_iou = match_iou
        self.low_match_iou = low_match_iou
        self.max_missed = max_missed
        self.min_hits = min_hits
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, detections: list[Detection]) -> tuple[list[Track], list[Track]]:
        """Advance one frame. Returns (confirmed live tracks, confirmed tracks removed now).

        Live tracks include ones coasting through a short occlusion (`missed > 0`).
        """
        for t in self.tracks:
            t.box = t.box + t.velocity  # constant-velocity prediction

        det_boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in detections]).reshape(-1, 4)
        scores = np.array([d.score for d in detections])
        high = [i for i in range(len(detections)) if scores[i] >= self.track_threshold]
        low = [i for i in range(len(detections)) if scores[i] < self.track_threshold]

        # Stage 1: confident detections vs all tracks.
        track_idx = list(range(len(self.tracks)))
        matched_tracks: set[int] = set()
        unmatched_high = set(high)
        for r, c in greedy_match(self._iou(track_idx, det_boxes[high]), self.match_iou):
            self._apply(self.tracks[track_idx[r]], detections[high[c]])
            matched_tracks.add(track_idx[r])
            unmatched_high.discard(high[c])

        # Stage 2: weak detections only keep existing tracks alive (reduces ID switches).
        remaining = [i for i in track_idx if i not in matched_tracks]
        for r, c in greedy_match(self._iou(remaining, det_boxes[low]), self.low_match_iou):
            self._apply(self.tracks[remaining[r]], detections[low[c]])
            matched_tracks.add(remaining[r])

        removed: list[Track] = []
        kept: list[Track] = []
        for i, t in enumerate(self.tracks):
            if i not in matched_tracks:
                t.missed += 1
                t.velocity *= 0.9
                # Tentative tracks die on first miss; confirmed ones survive occlusions.
                if t.id == 0 or t.missed > self.max_missed:
                    if t.id:
                        removed.append(t)
                    continue
            kept.append(t)
        self.tracks = kept

        for i in sorted(unmatched_high):
            d = detections[i]
            self.tracks.append(
                Track(id=0, box=det_boxes[i].astype(float), score=d.score,
                      label_votes={d.label: d.score})
            )
            self._maybe_confirm(self.tracks[-1])

        return [t for t in self.tracks if t.id], removed

    def _iou(self, track_indices: list[int], boxes: np.ndarray) -> np.ndarray:
        track_boxes = np.array([self.tracks[i].box for i in track_indices]).reshape(-1, 4)
        return iou_matrix(track_boxes, boxes)

    def _apply(self, track: Track, det: Detection) -> None:
        new_box = np.array([det.x1, det.y1, det.x2, det.y2], dtype=float)
        observed = new_box - (track.box - track.velocity)  # motion since last frame
        track.velocity = 0.6 * track.velocity + 0.4 * observed
        track.box = new_box
        track.score = det.score
        track.hits += 1
        track.missed = 0
        track.label_votes[det.label] = track.label_votes.get(det.label, 0.0) + det.score
        self._maybe_confirm(track)

    def _maybe_confirm(self, track: Track) -> None:
        if track.id == 0 and track.hits >= self.min_hits:
            track.id = self._next_id
            self._next_id += 1
