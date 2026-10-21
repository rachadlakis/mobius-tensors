"""Body keypoints (pose) for tracked people, and the facts worth storing from them.

Top-down: the person boxes the detector and tracker already have are cropped, and a pose model
returns the 17 COCO keypoints of each crop (nose, eyes, ears, shoulders, elbows, wrists, hips,
knees, ankles), each with a confidence. Two ONNX formats, told apart by their outputs:

    heatmaps  1 output (N, 17, 64, 48)   ViTPose-B "simple": Apache-2.0, fine-tuned on COCO only.
                                         The model to ship. export_pose.py
    SimCC     2 outputs (N, 17, 384/512) RTMPose-t (OpenMMLab): ~10x faster, but the published
                                         weights were trained on 7 datasets, some research-only.
                                         Trials only. download_model.py pose_rtmpose_t

`describe()` turns keypoints into a few searchable facts (rules, no extra model):

    posture   standing | sitting | crouching | bending | lying | unknown
    arms      down | one_up | both_up | unknown     (a wrist above the head)
    facing    camera | side | away | unknown
    head      up | down | unknown                   (looking down: a phone, a desk)
    upper / lower colour   dominant colour of the torso (shoulders to hips) and of the legs,
              one of COLOURS; None when unclear. At night (infrared) everything is grey.

Two more facts come from the tracks rather than the keypoints (pose_index.py): `motion`
(MOTIONS, from how far the person moved since the last sample) and `carrying` (CARRIED objects
the detector found on the person).

The store keeps these per sample (about once a second per person) plus the keypoints themselves
(float16, relative to the person box: 102 bytes), so a new fact can be derived later from the
stored keypoints without running the model again. Keypoints say much less about who someone is
than a face print, but a walk can still be recognised: same retention as the search index.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from histor.engine import colours

KEYPOINTS = ("nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
             "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
             "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle")
NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI, L_HIP, R_HIP, \
    L_KNEE, R_KNEE, L_ANK, R_ANK = range(17)
SKELETON = ((L_SHO, R_SHO), (L_SHO, L_ELB), (L_ELB, L_WRI), (R_SHO, R_ELB), (R_ELB, R_WRI),
            (L_SHO, L_HIP), (R_SHO, R_HIP), (L_HIP, R_HIP), (L_HIP, L_KNEE), (L_KNEE, L_ANK),
            (R_HIP, R_KNEE), (R_KNEE, R_ANK), (NOSE, L_EYE), (NOSE, R_EYE), (L_EYE, L_EAR),
            (R_EYE, R_EAR))
POSTURES = ("standing", "sitting", "crouching", "bending", "lying", "unknown")
ARMS = ("down", "one_up", "both_up", "unknown")
FACING = ("camera", "side", "away", "unknown")
HEAD = ("up", "down", "unknown")
MOTIONS = ("still", "walking", "running", "unknown")
CARRIED = ("backpack", "handbag", "suitcase", "umbrella")  # COCO labels
COLOURS = ("black", "white", "grey", "red", "orange", "yellow", "green", "blue", "purple", "pink",
           "brown")

INPUT_W, INPUT_H = 192, 256  # both model families take 256x192 crops
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)  # ImageNet, RGB
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
BOX_PADDING = 1.25  # crop a bit around the person box, as both models were trained


@dataclass
class PoseFacts:
    posture: str
    arms: str
    facing: str
    upper_colour: str | None
    lower_colour: str | None
    score: float   # mean confidence of the visible keypoints
    visible: int   # keypoints above min_confidence
    head: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class PoseEstimator:
    def __init__(self, model: str | Path = "models/pose_vitpose_b.onnx",
                 min_confidence: float = 0.3, num_threads: int = 0, batch_size: int = 8,
                 device: str = "auto") -> None:
        """min_confidence: keypoints below it count as not visible. ViTPose heatmap peaks and
        RTMPose SimCC peaks are on similar 0-1 scales; 0.3 suits both."""
        from histor.engine.onnx_device import create_session

        if not Path(model).exists():
            raise FileNotFoundError(
                f"{model} not found. Export it with: uv run --group export -m histor.engine.export_pose")
        self.session = create_session(model, device, num_threads)
        self.input_name = self.session.get_inputs()[0].name
        outputs = self.session.get_outputs()
        self.simcc = len(outputs) == 2
        self.min_confidence = min_confidence
        self.batch_size = batch_size

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> PoseEstimator:
        """config: the `pose` section."""
        options = {k: v for k, v in config.items()
                   if k in {"min_confidence", "num_threads", "batch_size", "device"}}
        return cls(config.get("model", "models/pose_vitpose_b.onnx"), **options)

    def estimate(self, image: np.ndarray, boxes: list[np.ndarray]) -> list[np.ndarray]:
        """Keypoints (17, 3) per person box (x1, y1, x2, y2) of a BGR image: x, y in image
        pixels and a confidence."""
        if not boxes:
            return []
        out: list[np.ndarray] = []
        for i in range(0, len(boxes), self.batch_size):
            chunk = boxes[i:i + self.batch_size]
            warps = [_affine(np.asarray(b, dtype=float)) for b in chunk]
            crops = [cv2.warpAffine(image, m, (INPUT_W, INPUT_H), flags=cv2.INTER_LINEAR)
                     for m in warps]
            batch = np.stack([((c[:, :, ::-1].astype(np.float32) - MEAN) / STD).transpose(2, 0, 1)
                              for c in crops])
            raw = self.session.run(None, {self.input_name: batch})
            points = _decode_simcc(*raw) if self.simcc else _decode_heatmaps(raw[0])
            for m, kp in zip(warps, points):
                inverse = cv2.invertAffineTransform(m)
                xy = np.c_[kp[:, :2], np.ones(len(kp))] @ inverse.T
                out.append(np.c_[xy, kp[:, 2]].astype(np.float32))
        return out


def _affine(box: np.ndarray) -> np.ndarray:
    """Box -> 2x3 matrix mapping it (padded, at the model's aspect ratio) onto the input."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    aspect = INPUT_W / INPUT_H
    if w > h * aspect:
        h = w / aspect
    else:
        w = h * aspect
    w, h = w * BOX_PADDING, h * BOX_PADDING
    src = np.float32([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2], [cx - w / 2, cy + h / 2]])
    dst = np.float32([[0, 0], [INPUT_W, 0], [0, INPUT_H]])
    return cv2.getAffineTransform(src, dst)


def _decode_heatmaps(heatmaps: np.ndarray) -> np.ndarray:
    """(N, K, H, W) -> (N, K, 3) in input pixels: peak, refined a quarter pixel towards the
    higher neighbour (the classic heatmap decoding)."""
    n, k, h, w = heatmaps.shape
    flat = heatmaps.reshape(n, k, -1)
    idx = flat.argmax(-1)
    conf = flat.max(-1)
    ys, xs = (idx // w).astype(np.float32), (idx % w).astype(np.float32)
    for b in range(n):
        for j in range(k):
            x, y = int(xs[b, j]), int(ys[b, j])
            hm = heatmaps[b, j]
            if 0 < x < w - 1:
                xs[b, j] += 0.25 * np.sign(hm[y, x + 1] - hm[y, x - 1])
            if 0 < y < h - 1:
                ys[b, j] += 0.25 * np.sign(hm[y + 1, x] - hm[y - 1, x])
    scale_x, scale_y = INPUT_W / w, INPUT_H / h
    return np.stack([(xs + 0.5) * scale_x - 0.5, (ys + 0.5) * scale_y - 0.5, conf], -1)


def _decode_simcc(simcc_x: np.ndarray, simcc_y: np.ndarray, split_ratio: float = 2.0) -> np.ndarray:
    """RTMPose: one 1-D distribution per axis and keypoint; confidence = the weaker axis."""
    xs = simcc_x.argmax(-1) / split_ratio
    ys = simcc_y.argmax(-1) / split_ratio
    conf = np.minimum(simcc_x.max(-1), simcc_y.max(-1))
    return np.stack([xs, ys, conf], -1).astype(np.float32)


# ------------------------------------------------------------------ facts from keypoints


def describe(keypoints: np.ndarray, image: np.ndarray | None = None,
             min_confidence: float = 0.3) -> PoseFacts:
    """Searchable facts from one person's keypoints (17, 3). image: the BGR image the
    keypoints are in, for the clothing colours (None: no colours)."""
    kp = np.asarray(keypoints, dtype=float)
    seen = kp[:, 2] >= min_confidence

    def mid(*ids: int) -> np.ndarray | None:
        pts = [kp[i, :2] for i in ids if seen[i]]
        return np.mean(pts, axis=0) if pts else None

    shoulders, hips = mid(L_SHO, R_SHO), mid(L_HIP, R_HIP)
    knees, ankles = mid(L_KNEE, R_KNEE), mid(L_ANK, R_ANK)
    torso = float(np.linalg.norm(hips - shoulders)) if shoulders is not None and hips is not None \
        else 0.0

    posture = "unknown"
    if torso > 1:
        dx, dy = hips - shoulders  # image y grows downwards: upright means dy > 0
        tilt = np.degrees(np.arctan2(abs(dx), dy))  # 0 upright, 90 horizontal, 180 upside down
        # Legs straight down from the hips (feet well below them): a torso this tilted is
        # someone bending over, not lying.
        legs_down = ankles is not None and hips is not None and             ankles[1] - hips[1] > 0.8 * torso and abs(ankles[0] - hips[0]) < 0.6 * torso
        if tilt > 55:
            posture = "bending" if legs_down and tilt < 120 else "lying"
        elif knees is not None:
            thigh_drop = knees[1] - hips[1]  # standing: about one torso; sitting: near zero
            if ankles is not None and ankles[1] - hips[1] < 0.6 * torso:
                posture = "crouching"  # feet tucked under the hips
            elif thigh_drop < 0.45 * torso:
                posture = "sitting"
            elif tilt > 35:
                posture = "bending"
            else:
                posture = "standing"

    arms = "unknown"
    head = [kp[i, 1] for i in (NOSE, L_EYE, R_EYE, L_EAR, R_EAR) if seen[i]]
    top = min(head) if head else (shoulders[1] - 0.4 * torso if shoulders is not None and torso
                                  else None)
    wrists = [kp[i, 1] for i in (L_WRI, R_WRI) if seen[i]]
    if top is not None and wrists and posture != "lying":
        up = sum(y < top - 0.1 * torso for y in wrists)  # clearly above, not level with it
        arms = "both_up" if up == 2 else "one_up" if up == 1 else "down"

    # Facing the camera, a person's left shoulder (and eye) is on the image's right; from
    # behind, on its left. Shoulders come first: models often put face points on the back of
    # a head.
    facing = "unknown"
    span = 0.15 * max(torso, 1.0)
    shoulder_w = kp[L_SHO, 0] - kp[R_SHO, 0] if seen[L_SHO] and seen[R_SHO] else None
    eyes = int(seen[L_EYE]) + int(seen[R_EYE])
    if shoulder_w is not None and shoulder_w < -span:
        facing = "away"
    elif seen[NOSE] and eyes == 2 and kp[L_EYE, 0] > kp[R_EYE, 0]:
        facing = "camera"
    elif seen[NOSE] or eyes:
        facing = "side"
    elif shoulder_w is not None:
        facing = "camera" if shoulder_w > span else "side"

    # Looking down: the nose drops well below the ears (level with them when looking ahead).
    # Not from behind, where the "nose" is often a guess on the back of the head.
    head_pose = "unknown"
    ears = [kp[i, 1] for i in (L_EAR, R_EAR) if seen[i]]
    if torso > 1 and seen[NOSE] and ears and posture != "lying" and facing != "away":
        head_pose = "down" if kp[NOSE, 1] - np.mean(ears) > 0.15 * torso else "up"

    upper = lower = None
    if image is not None and torso > 4:
        upper = _region_colour(image, _torso_mask(image.shape, kp, seen))
        lower = _region_colour(image, _legs_mask(image.shape, kp, seen, torso))

    visible = int(seen.sum())
    score = float(kp[seen, 2].mean()) if visible else 0.0
    return PoseFacts(posture, arms, facing, upper, lower, round(score, 3), visible, head_pose)


def _torso_mask(shape: tuple[int, ...], kp: np.ndarray, seen: np.ndarray) -> np.ndarray | None:
    corners = [L_SHO, R_SHO, R_HIP, L_HIP]
    if not all(seen[i] for i in corners):
        return None
    pts = kp[corners, :2]
    centre = pts.mean(0)
    pts = centre + (pts - centre) * 0.7  # stay inside the clothes: arms and background out
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, pts.astype(np.int32), 1)
    return mask


def _legs_mask(shape: tuple[int, ...], kp: np.ndarray, seen: np.ndarray,
               torso: float) -> np.ndarray | None:
    mask = np.zeros(shape[:2], np.uint8)
    width = max(2, int(torso * 0.18))
    drawn = False
    for hip, knee, ankle in ((L_HIP, L_KNEE, L_ANK), (R_HIP, R_KNEE, R_ANK)):
        for a, b in ((hip, knee), (knee, ankle)):
            if seen[a] and seen[b]:
                pa, pb = kp[a, :2], kp[b, :2]
                # shorten each segment so joints (and shoes) don't dominate
                pa, pb = pa + (pb - pa) * 0.15, pb - (pb - pa) * 0.15
                cv2.line(mask, tuple(pa.astype(int)), tuple(pb.astype(int)), 1, width)
                drawn = True
    return mask if drawn else None


def colour_names(pixels: np.ndarray) -> np.ndarray:
    """BGR pixels (N, 3) -> index into COLOURS per pixel."""
    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    h, s, v = (hsv[:, i].astype(int) for i in range(3))
    names = np.full(len(hsv), COLOURS.index("grey"))
    chroma = colours.chroma(s, v)
    lower = 0
    for upper, name in colours.HUE_BINS:
        names[chroma & (h >= lower) & (h < upper)] = COLOURS.index(name)
        lower = upper
    names[chroma & (h >= 8) & (h < 25) & (v < 150)] = COLOURS.index("brown")
    names[(~chroma) & (v >= 190) & (s < 40)] = COLOURS.index("white")
    names[v < 50] = COLOURS.index("black")
    return names


def _region_colour(image: np.ndarray, mask: np.ndarray | None,
                   min_pixels: int = 30, min_share: float = 0.35) -> str | None:
    """Most common colour name in the masked area, if it covers at least `min_share`."""
    if mask is None:
        return None
    pixels = image[mask.astype(bool)]
    if len(pixels) < min_pixels:
        return None
    counts = np.bincount(colour_names(pixels), minlength=len(COLOURS))
    best = int(counts.argmax())
    return COLOURS[best] if counts[best] >= min_share * len(pixels) else None


# ------------------------------------------------------------------ storage and drawing


def pack_keypoints(keypoints: np.ndarray, box: np.ndarray) -> bytes:
    """(17, 3) in image pixels -> float16 bytes relative to the box (0-1 inside it), so the
    stored keypoints don't depend on the stream's resolution."""
    x1, y1, x2, y2 = (float(v) for v in box)
    rel = np.asarray(keypoints, dtype=np.float32).copy()
    rel[:, 0] = (rel[:, 0] - x1) / max(x2 - x1, 1.0)
    rel[:, 1] = (rel[:, 1] - y1) / max(y2 - y1, 1.0)
    return rel.astype(np.float16).tobytes()


def unpack_keypoints(blob: bytes, box: list[float] | np.ndarray) -> np.ndarray:
    """The inverse of pack_keypoints, for a box in the same coordinates as stored."""
    x1, y1, x2, y2 = (float(v) for v in box)
    rel = np.frombuffer(blob, dtype=np.float16).astype(np.float32).reshape(-1, 3).copy()
    rel[:, 0] = rel[:, 0] * (x2 - x1) + x1
    rel[:, 1] = rel[:, 1] * (y2 - y1) + y1
    return rel


def draw_skeleton(frame: np.ndarray, keypoints: np.ndarray, min_confidence: float = 0.3,
                  colour: tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
    kp = np.asarray(keypoints)
    for a, b in SKELETON:
        if kp[a, 2] >= min_confidence and kp[b, 2] >= min_confidence:
            cv2.line(frame, (int(kp[a, 0]), int(kp[a, 1])), (int(kp[b, 0]), int(kp[b, 1])),
                     colour, 2, cv2.LINE_AA)
    for x, y, c in kp:
        if c >= min_confidence:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1, cv2.LINE_AA)
    return frame
