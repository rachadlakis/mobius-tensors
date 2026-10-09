"""Model-agnostic detection interface plus a CPU-friendly YOLOX (ONNX Runtime) backend.

The rest of the app only depends on `Detector` and `Detection`, so backends can be
swapped (another ONNX model, OpenVINO, TensorRT, ...) without touching stream/alert code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int
    label: str


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run detection on a BGR image (H, W, 3) and return boxes in image pixels."""
        ...


def letterbox(image: np.ndarray, input_size: tuple[int, int]) -> tuple[np.ndarray, float]:
    """Resize keeping aspect ratio, pad bottom/right with 114, return CHW float32 + scale."""
    h, w = input_size
    ratio = min(h / image.shape[0], w / image.shape[1])
    resized = cv2.resize(
        image,
        (int(image.shape[1] * ratio), int(image.shape[0] * ratio)),
        interpolation=cv2.INTER_LINEAR,
    )
    padded = np.full((h, w, 3), 114, dtype=np.uint8)
    padded[: resized.shape[0], : resized.shape[1]] = resized
    return np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32), ratio


def decode_yolox(outputs: np.ndarray, input_size: tuple[int, int], p6: bool = False) -> np.ndarray:
    """Turn raw YOLOX grid outputs (N, 5 + C) into absolute cx, cy, w, h in input pixels."""
    strides = [8, 16, 32, 64] if p6 else [8, 16, 32]
    grids, expanded_strides = [], []
    for stride in strides:
        hsize, wsize = input_size[0] // stride, input_size[1] // stride
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(-1, 2)
        grids.append(grid)
        expanded_strides.append(np.full((grid.shape[0], 1), stride))
    grid = np.concatenate(grids, 0)
    strides_col = np.concatenate(expanded_strides, 0)

    decoded = outputs.copy()
    decoded[:, :2] = (outputs[:, :2] + grid) * strides_col
    decoded[:, 2:4] = np.exp(outputs[:, 2:4]) * strides_col
    return decoded


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy single-class NMS on xyxy boxes."""
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        w = np.maximum(0.0, np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]))
        h = np.maximum(0.0, np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]))
        inter = w * h
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_threshold]
    return keep


class YoloxOnnxDetector:
    """YOLOX exported to ONNX (Apache-2.0), run with ONNX Runtime. CPU or GPU (`device`)."""

    def __init__(
        self,
        model_path: str | Path,
        input_size: tuple[int, int] | None = None,
        class_names: tuple[str, ...] | list[str] = COCO_CLASSES,
        conf_threshold: float = 0.35,
        nms_threshold: float = 0.45,
        class_agnostic_nms: bool = False,
        classes: list[str] | None = None,
        num_threads: int = 0,
        device: str = "auto",
    ) -> None:
        from onnx_device import create_session

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}. Run `uv run src/download_model.py` first."
            )

        self.session = create_session(model_path, device, num_threads)
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        # Exported YOLOX models have a fixed input size: trust the model over the config.
        h, w = model_input.shape[2:4]
        if isinstance(h, int) and isinstance(w, int):
            self.input_size = (h, w)
        else:
            self.input_size = input_size or (640, 640)
        self.class_names = tuple(class_names)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.class_agnostic_nms = class_agnostic_nms
        unknown = set(classes or []) - set(self.class_names)
        if unknown:
            raise ValueError(f"Unknown class name(s) in `classes`: {sorted(unknown)}")
        self.allowed_ids = (
            None if not classes else {self.class_names.index(name) for name in classes}
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        blob, ratio = letterbox(frame, self.input_size)
        raw = self.session.run(None, {self.input_name: blob[None]})[0][0]
        h, w = frame.shape[:2]
        return self.postprocess(raw, ratio, (h, w))

    def postprocess(
        self, raw: np.ndarray, ratio: float, image_shape: tuple[int, int]
    ) -> list[Detection]:
        preds = decode_yolox(raw, self.input_size)
        scores_all = preds[:, 4:5] * preds[:, 5:]
        class_ids = scores_all.argmax(1)
        scores = scores_all[np.arange(len(class_ids)), class_ids]

        mask = scores > self.conf_threshold
        if self.allowed_ids is not None:
            mask &= np.isin(class_ids, list(self.allowed_ids))
        if not mask.any():
            return []

        cx, cy, w, h = preds[mask, :4].T
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1) / ratio
        img_h, img_w = image_shape
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, img_w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, img_h)
        scores, class_ids = scores[mask], class_ids[mask]

        if self.class_agnostic_nms:
            keep = nms(boxes, scores, self.nms_threshold)
        else:
            keep = []
            for cls in np.unique(class_ids):
                idx = np.where(class_ids == cls)[0]
                keep.extend(idx[nms(boxes[idx], scores[idx], self.nms_threshold)].tolist())
            keep.sort(key=lambda i: float(scores[i]), reverse=True)

        return [
            Detection(
                x1=float(boxes[i, 0]),
                y1=float(boxes[i, 1]),
                x2=float(boxes[i, 2]),
                y2=float(boxes[i, 3]),
                score=float(scores[i]),
                class_id=int(class_ids[i]),
                label=self.class_names[int(class_ids[i])],
            )
            for i in keep
        ]


def create_detector(config: dict[str, Any]) -> Detector:
    """Build a detector from the `model` section of a config file."""
    backend = config.get("backend", "yolox_onnx")
    if backend == "yolox_onnx":
        size = config.get("input_size")
        return YoloxOnnxDetector(
            model_path=config["path"],
            input_size=(int(size[0]), int(size[1])) if size else None,
            class_names=config.get("class_names") or COCO_CLASSES,
            conf_threshold=config.get("conf_threshold", 0.35),
            nms_threshold=config.get("nms_threshold", 0.45),
            class_agnostic_nms=config.get("class_agnostic_nms", False),
            classes=config.get("classes"),
            num_threads=config.get("num_threads", 0),
            device=config.get("device", "auto"),
        )
    raise ValueError(f"Unknown detector backend: {backend}")


def color_for(key: int) -> tuple[int, int, int]:
    """Stable BGR color per class id / track id."""
    b, g, r = np.random.default_rng(key).integers(0, 255, 3)
    return int(b), int(g), int(r)


def draw_label(
    frame: np.ndarray, text: str, anchor: tuple[int, int], color: tuple[int, int, int]
) -> None:
    """Text on a filled box whose bottom-left corner sits at `anchor`."""
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    x, top = anchor[0], max(anchor[1] - th - baseline - 2, 0)
    cv2.rectangle(frame, (x, top), (x + tw + 4, top + th + baseline + 2), color, -1)
    text_color = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)
    cv2.putText(
        frame, text, (x + 2, top + th + 1),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA,
    )


def draw_box(
    frame: np.ndarray, box: tuple[float, float, float, float] | np.ndarray, text: str,
    color: tuple[int, int, int],
) -> None:
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(frame, p1, p2, color, 2)
    draw_label(frame, text, p1, color)


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    for det in detections:
        draw_box(
            frame, (det.x1, det.y1, det.x2, det.y2), f"{det.label} {det.score:.2f}",
            color_for(det.class_id),
        )
    return frame
