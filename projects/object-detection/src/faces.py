"""Face detection (YuNet) + face prints (SFace), both small ONNX models run by OpenCV.

Licences: YuNet MIT, SFace Apache-2.0 (OpenCV Zoo) - fine for commercial use.
A "face print" is a 128-number vector; two photos of the same person score a cosine
similarity above ~0.36 (SFace's published threshold), different people usually well below.

Face prints are biometric data: the recorder only computes them for cameras where faces are
explicitly enabled, and they are deleted after `face_retention_days`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

SAME_PERSON_THRESHOLD = 0.363  # SFace cosine threshold (OpenCV Zoo, LFW-calibrated)


@dataclass
class Face:
    box: tuple[float, float, float, float]  # x, y, w, h in the image passed to detect()
    score: float
    landmarks: np.ndarray  # raw YuNet row (needed for alignment)

    @property
    def size(self) -> float:
        return min(self.box[2], self.box[3])


class FaceEngine:
    def __init__(
        self,
        detector: str | Path = "models/face_detection_yunet_2023mar.onnx",
        recognizer: str | Path = "models/face_recognition_sface_2021dec.onnx",
        score_threshold: float = 0.8,
        min_face_size: int = 32,
    ) -> None:
        for path in (detector, recognizer):
            if not Path(path).exists():
                raise FileNotFoundError(
                    f"{path} not found. Download with: uv run src/download_model.py faces")
        self._detector = cv2.FaceDetectorYN.create(str(detector), "", (320, 320), score_threshold)
        self._recognizer = cv2.FaceRecognizerSF.create(str(recognizer), "")
        self.min_face_size = min_face_size

    def detect(self, image: np.ndarray) -> list[Face]:
        """All faces in a BGR image, largest first. Small images are upscaled for YuNet."""
        h, w = image.shape[:2]
        scale = 1.0
        if max(h, w) < 320:  # person crops from sub-streams are small
            scale = 320 / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_CUBIC)
        self._detector.setInputSize((image.shape[1], image.shape[0]))
        _, rows = self._detector.detect(image)
        faces = []
        for row in rows if rows is not None else []:
            row = row.copy()
            row[:14] /= scale  # box + 5 landmarks back to original coordinates
            fx, fy, fw, fh = (float(v) for v in row[:4])
            face = Face((fx, fy, fw, fh), float(row[14]), row)
            if face.size >= self.min_face_size:
                faces.append(face)
        return sorted(faces, key=lambda f: -f.size)

    def embed(self, image: np.ndarray, face: Face) -> np.ndarray:
        """Align the face using its landmarks and return a unit-length 128-d face print."""
        aligned = self._recognizer.alignCrop(image, face.landmarks)
        feature = self._recognizer.feature(aligned).ravel().astype(np.float32)
        return feature / (np.linalg.norm(feature) + 1e-12)

    def crop(self, image: np.ndarray, face: Face, margin: float = 0.3) -> np.ndarray:
        x, y, w, h = face.box
        x1, y1 = int(max(0, x - w * margin)), int(max(0, y - h * margin))
        x2 = int(min(image.shape[1], x + w * (1 + margin)))
        y2 = int(min(image.shape[0], y + h * (1 + margin)))
        return image[y1:y2, x1:x2]

    def best_face_print(self, image: np.ndarray) -> tuple[np.ndarray, Face] | None:
        """For an uploaded photo: the largest face's print, or None if no face is found."""
        faces = self.detect(image)
        if not faces:
            return None
        return self.embed(image, faces[0]), faces[0]
