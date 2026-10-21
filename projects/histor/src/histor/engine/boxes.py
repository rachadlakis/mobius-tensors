"""Box geometry shared by the engine (one implementation each). Boxes are x1, y1, x2, y2 in
pixels; a box with x2 <= x1 or y2 <= y1 is empty (area 0)."""

from __future__ import annotations

from typing import Any

import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N, 4) and (M, 4) xyxy boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def greedy_match(iou: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Match rows to columns by descending IoU; each row/column used at most once."""
    pairs: list[tuple[int, int]] = []
    if iou.size == 0:
        return pairs
    used_rows, used_cols = set(), set()
    for flat in np.argsort(-iou, axis=None, kind="stable"):
        r, c = divmod(int(flat), iou.shape[1])
        if iou[r, c] < threshold:
            break
        if r in used_rows or c in used_cols:
            continue
        pairs.append((r, c))
        used_rows.add(r)
        used_cols.add(c)
    return pairs


def area(box: Any) -> float:
    return float(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))


def centre(box: Any) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], dtype=float)


def _intersection(a: Any, b: Any) -> float:
    return area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def inside(box: Any, cover: Any) -> float:
    """Share of `box` that lies inside `cover` (0 for an empty box)."""
    return _intersection(box, cover) / max(1e-6, area(box))


def loose_overlap(a: Any, b: Any, iou: float = 0.1, grow: float = 0.5) -> bool:
    """The same thing seen twice, loosely: IoU at least `iou`, or the centre of either inside
    the other grown by `grow` of its size on each side."""
    if iou_matrix(np.asarray([a], float), np.asarray([b], float))[0, 0] >= iou:
        return True
    for box, other in ((a, b), (b, a)):
        gx, gy = (other[2] - other[0]) * grow, (other[3] - other[1]) * grow
        x, y = centre(box)
        if area(other) > 0 and other[0] - gx <= x <= other[2] + gx and \
                other[1] - gy <= y <= other[3] + gy:
            return True
    return False


def scale_box(box: Any, sx: float, sy: float | None = None) -> np.ndarray:
    """The box in another picture's pixels (sy: sx when not given)."""
    sy = sx if sy is None else sy
    return np.asarray(box, dtype=float) * np.array([sx, sy, sx, sy])


def clip_box(box: Any, w: float, h: float) -> np.ndarray:
    """The box cut to a w x h picture."""
    return np.clip(np.asarray(box, dtype=float), 0, [w, h, w, h])
