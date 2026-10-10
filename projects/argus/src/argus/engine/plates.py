"""Number plates: find the plate on a vehicle, then read it. Two small ONNX models, CPU-friendly.

    finding   PlateDetector: a plate detector run on each vehicle crop (not the whole frame).
              Default: open-image-models YOLOv9-t 384 "end2end" (NMS inside the graph; MIT code,
              weights trained with a fork of the GPL-3.0 YOLOv9 repo: trials until a legal
              check, or train our own YOLOX on Open Images plates and set `type: yolox`).
    reading   PlateReader: fast-plate-ocr's CCT "global" model (MIT, weights included): a
              64x128 RGB crop -> 10 character slots over 0-9 A-Z (pad `_`), plus the plate's
              country (66 regions; "Unknown" when unsure).

Models: `uv run -m argus.engine.download_model plates`. Plate text is normalised to upper-case
letters and digits only ("ab-12 3" -> "AB123"), and compared with `plate_distance`, which
forgives the characters OCR confuses most (0/O/D/Q, 1/I/L, 2/Z, 5/S, 6/G, 8/B).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

VEHICLES = ("car", "truck", "bus", "motorcycle")
# OCR look-alikes, mapped to one representative for comparisons (never for display).
_CONFUSABLE = str.maketrans({"O": "0", "D": "0", "Q": "0", "I": "1", "L": "1", "Z": "2",
                             "S": "5", "G": "6", "B": "8"})
_NOT_ALNUM = re.compile(r"[^0-9A-Z]")


def normalize_plate(text: str) -> str:
    """Upper-case letters and digits only: what is stored and compared."""
    return _NOT_ALNUM.sub("", str(text or "").upper())


def plate_key(text: str) -> str:
    """Normalised text with OCR look-alikes merged (0/O/D/Q -> 0, ...)."""
    return normalize_plate(text).translate(_CONFUSABLE)


def _levenshtein(a: str, b: str) -> int:
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (ca != cb))
    return row[-1]


def plate_distance(a: str, b: str) -> int:
    """Edits between two plates, look-alike characters counting as equal."""
    return _levenshtein(plate_key(a), plate_key(b))


@dataclass
class PlateRead:
    text: str          # normalised, pad removed
    confidence: float  # lowest character probability: one doubtful character makes it doubtful
    region: str | None = None
    region_confidence: float | None = None


class PlateReader:
    """fast-plate-ocr ONNX model + its plate config (alphabet, slots, input size)."""

    def __init__(self, model: str | Path = "models/plate_ocr_cct_s_v2.onnx",
                 config: str | Path = "models/plate_ocr_cct_s_v2.yaml",
                 device: str = "auto", num_threads: int = 0) -> None:
        from argus.engine.onnx_device import create_session

        for path in (model, config):
            if not Path(path).exists():
                raise FileNotFoundError(
                    f"{path} not found. Download with: uv run -m argus.engine.download_model plates")
        cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
        self.alphabet: str = cfg["alphabet"]
        self.pad: str = cfg["pad_char"]
        self.size = (int(cfg["img_width"]), int(cfg["img_height"]))
        self.rgb = cfg.get("image_color_mode", "rgb") == "rgb"
        self.keep_aspect = bool(cfg.get("keep_aspect_ratio", False))
        self.regions: list[str] = list(cfg.get("plate_regions") or [])
        self.session = create_session(model, device, num_threads)
        self.input = self.session.get_inputs()[0].name
        self.outputs = [o.name for o in self.session.get_outputs()]

    def preprocess(self, crop: np.ndarray) -> np.ndarray:
        """BGR plate crop -> (H, W, C) uint8 as the model was trained."""
        w, h = self.size
        image: np.ndarray
        if self.keep_aspect:  # letterbox
            scale = min(w / crop.shape[1], h / crop.shape[0])
            resized = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)),
                                        max(1, round(crop.shape[0] * scale))))
            image = np.zeros((h, w, 3), np.uint8)
            image[:resized.shape[0], :resized.shape[1]] = resized
        else:
            image = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
        if self.rgb:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[:, :, None]

    def read(self, crops: list[np.ndarray]) -> list[PlateRead]:
        if not crops:
            return []
        batch = np.stack([self.preprocess(c) for c in crops])
        outputs = dict(zip(self.outputs, self.session.run(self.outputs, {self.input: batch})))
        chars = outputs.get("plate", next(iter(outputs.values())))
        regions = outputs.get("region")
        return [self.decode(chars[i], regions[i] if regions is not None else None)
                for i in range(len(crops))]

    def decode(self, slots: np.ndarray, region: np.ndarray | None = None) -> PlateRead:
        """(slots, alphabet) probabilities -> text; trailing pad dropped."""
        idx = slots.argmax(axis=1)
        probs = slots.max(axis=1)
        text = "".join(self.alphabet[i] for i in idx)
        keep = len(text.rstrip(self.pad))
        conf = float(probs[:keep].min()) if keep else 0.0
        out = PlateRead(normalize_plate(text[:keep].replace(self.pad, "")), conf)
        if region is not None and self.regions:
            r = int(region.argmax())
            out.region = self.regions[r] if r < len(self.regions) else None
            out.region_confidence = float(region[r])
        return out


class PlateDetector:
    """Plate boxes in a (vehicle) image: [(x1, y1, x2, y2, score), ...], best first.

    type yolov9: an end2end ONNX with NMS inside (rows: batch, x1, y1, x2, y2, class, score),
                 letterboxed input, RGB / 255.
    type yolox:  a YOLOX trained on one plate class, through the everyday detector's code."""

    def __init__(self, model: str | Path = "models/plate_yolov9_t_384.onnx", type: str = "yolov9",
                 min_score: float = 0.4, device: str = "auto", num_threads: int = 0) -> None:
        if not Path(model).exists():
            raise FileNotFoundError(
                f"{model} not found. Download with: uv run -m argus.engine.download_model plates")
        self.type = type
        self.min_score = min_score
        if type == "yolov9":
            from argus.engine.onnx_device import create_session

            self.session = create_session(model, device, num_threads)
            inp = self.session.get_inputs()[0]
            self.input = inp.name
            self.input_size = int(inp.shape[2])
        elif type == "yolox":
            from argus.engine.detector import create_detector

            self.yolox = create_detector({"path": str(model), "class_names": ["plate"],
                                          "conf_threshold": min_score, "device": device})
        else:
            raise ValueError(f"plates.detector.type {type!r}: use yolov9 or yolox")

    def detect(self, image: np.ndarray) -> list[tuple[float, float, float, float, float]]:
        if self.type == "yolox":
            found = [(d.x1, d.y1, d.x2, d.y2, d.score) for d in self.yolox.detect(image)]
            return sorted(found, key=lambda b: -b[4])
        tensor, ratio, (dw, dh) = letterbox(image, self.input_size)
        rows = self.session.run(None, {self.input: tensor})[0]
        h, w = image.shape[:2]
        out = []
        for row in np.atleast_2d(rows):
            score = float(row[6])
            if score < self.min_score:
                continue
            x1, x2 = (np.array([row[1], row[3]]) - dw) / ratio
            y1, y2 = (np.array([row[2], row[4]]) - dh) / ratio
            x1, x2 = float(np.clip(x1, 0, w)), float(np.clip(x2, 0, w))
            y1, y2 = float(np.clip(y1, 0, h)), float(np.clip(y2, 0, h))
            if x2 - x1 >= 2 and y2 - y1 >= 2:
                out.append((x1, y1, x2, y2, score))
        return sorted(out, key=lambda b: -b[4])


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[float, float]]:
    """BGR image -> (1, 3, size, size) float32 RGB in 0..1, scale and padding (dw, dh)."""
    h, w = image.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w, h) \
        else image
    dw, dh = (size - nw) / 2, (size - nh) / 2
    top, left = int(round(dh - 0.1)), int(round(dw - 0.1))
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[top:top + nh, left:left + nw] = resized
    tensor = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(tensor), r, (dw, dh)


def crop_box(image: np.ndarray, box: Any, pad: float = 0.0) -> np.ndarray | None:
    """Crop (x1, y1, x2, y2), widened by `pad` of its size on each side; None if empty."""
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    h, w = image.shape[:2]
    x1, y1 = max(0, int(x1 - px)), max(0, int(y1 - py))
    x2, y2 = min(w, int(round(x2 + px))), min(h, int(round(y2 + py)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return image[y1:y2, x1:x2]


def plates_enabled(config: dict[str, Any] | None, camera: str) -> bool:
    """`plates` section on, and this camera listed (or no list: every camera)."""
    if not config or not config.get("enabled"):
        return False
    cameras = config.get("cameras") or []
    return not cameras or camera in cameras
