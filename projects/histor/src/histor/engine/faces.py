"""Face detection (YuNet) + face prints (SFace), both small ONNX models run by OpenCV.

Licences: YuNet MIT, SFace Apache-2.0 (OpenCV Zoo) - fine for commercial use.
A "face print" is a 128-number vector; two photos of the same person score a cosine
similarity above ~0.36 (SFace's published threshold), different people usually well below.
That threshold was set on clear, frontal photos (LFW), not CCTV: matches between 0.363 and
LIKELY_SAME_PERSON are reported as "possible", above it as "likely". Calibrate both on a
pilot site's footage before trusting them.

Blurred, tiny or side-on faces match everyone a little, so every face gets a quality check
(`FaceQuality`): the recorder skips bad ones, and an uploaded photo with a bad face is refused
with the reason instead of being searched.

Face prints are biometric data: the recorder only computes them for cameras where faces are
explicitly enabled, and they are deleted after `face_retention_days`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SAME_PERSON_THRESHOLD = 0.363  # SFace cosine threshold (OpenCV Zoo, LFW-calibrated)
LIKELY_SAME_PERSON = 0.45      # stricter level for CCTV: "likely", below it only "possible"
MIN_PHOTO_FACE = 40            # uploaded photos: smaller faces carry too little detail
DETECT_MIN_SIDE, DETECT_MAX_SIDE = 320, 640  # YuNet input, longer side (see FaceEngine.detect)


@dataclass
class Face:
    box: tuple[float, float, float, float]  # x, y, w, h in the image passed to detect()
    score: float
    landmarks: np.ndarray  # raw YuNet row (needed for alignment)

    @property
    def size(self) -> float:
        return min(self.box[2], self.box[3])

    @property
    def turn(self) -> float:
        """How far the head is turned sideways: nose-tip offset from the middle of the eyes,
        relative to the eye distance. ~0-0.3 facing the camera, ~0.5 at 60 degrees."""
        right_eye, left_eye, nose = self.landmarks[4:6], self.landmarks[6:8], self.landmarks[8:10]
        eyes = abs(float(left_eye[0] - right_eye[0]))
        return abs(float(nose[0] - (right_eye[0] + left_eye[0]) / 2)) / max(eyes, 1.0)


@dataclass
class FaceQuality:
    score: float      # detector confidence
    size: float       # face size in pixels (shorter side of the box)
    turn: float       # see Face.turn
    sharpness: float  # variance of the Laplacian of the aligned 112x112 face

    @property
    def value(self) -> float:
        """0..1, for ranking a track's faces: confident, big, frontal and sharp is best."""
        return (self.score * min(1.0, self.size / 80) * float(np.clip(1 - self.turn / 0.8, 0.05, 1))
                * min(1.0, self.sharpness / 100))


@dataclass
class PhotoFace:
    """A face in an uploaded photo, ready to search with (or refused, with the reason)."""

    print: np.ndarray | None
    box: tuple[float, float, float, float]
    quality: FaceQuality
    problems: list[str]


class FaceEngine:
    def __init__(
        self,
        detector: str | Path = "models/face_detection_yunet_2023mar.onnx",
        recognizer: str | Path = "models/face_recognition_sface_2021dec.onnx",
        score_threshold: float = 0.8,
        min_face_size: int = 32,
        min_sharpness: float = 15.0,
        max_turn: float = 0.6,
    ) -> None:
        """min_sharpness / max_turn: faces below / above are not stored or searched with.
        Heuristics (sharp faces measure ~30-300, motion-blurred ones under 10): tune them on
        real footage."""
        for path in (detector, recognizer):
            if not Path(path).exists():
                raise FileNotFoundError(
                    f"{path} not found. Download with: uv run -m histor.engine.download_model faces")
        self._detector = cv2.FaceDetectorYN.create(str(detector), "", (320, 320), score_threshold)
        self._recognizer = cv2.FaceRecognizerSF.create(str(recognizer), "")
        self.min_face_size = min_face_size
        self.min_sharpness = min_sharpness
        self.max_turn = max_turn

    @classmethod
    def from_config(cls, config: dict[str, Any], for_photos: bool = False) -> FaceEngine:
        """config: the `faces` section. for_photos: searching with uploaded photos, where small
        faces are detected too so they can be refused with the reason ("too small")."""
        options = {k: v for k, v in config.items() if k in {
            "detector", "recognizer", "score_threshold", "min_face_size", "min_sharpness",
            "max_turn"}}
        if for_photos:
            options["min_face_size"] = 20
        return cls(**options)

    def detect(self, image: np.ndarray, retry: bool = False) -> list[Face]:
        """All faces in a BGR image, largest first, in the image's own coordinates.

        YuNet is unsure of faces much bigger than ~300 px (a selfie at 3088x2320 scored 0.72
        with a box on half the face; at 640 px, 0.93 and the right box), so the image is resized
        to 320-640 px on its longer side first. retry (uploaded photos): if no face is found,
        try again at full size (small faces in a big photo), then at 320 px (a face filling it).
        Face prints are still made from the full-resolution image (analyze)."""
        longest = max(image.shape[:2])
        sides = [min(max(longest, DETECT_MIN_SIDE), DETECT_MAX_SIDE)]
        if retry:
            sides += [s for s in (longest, DETECT_MIN_SIDE) if s > DETECT_MAX_SIDE or s < sides[0]]
        for side in sides:
            faces = self._detect_at(image, side / longest)
            if faces:
                return faces
        return []

    def _detect_at(self, image: np.ndarray, scale: float) -> list[Face]:
        if scale != 1.0:
            h, w = image.shape[:2]
            image = cv2.resize(image, (round(w * scale), round(h * scale)),
                               interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
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

    def analyze(self, image: np.ndarray, face: Face) -> tuple[np.ndarray, FaceQuality]:
        """Align the face using its landmarks; return its unit-length 128-d face print and
        its quality."""
        aligned = self._recognizer.alignCrop(image, face.landmarks)
        feature = self._recognizer.feature(aligned).ravel().astype(np.float32)
        gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        quality = FaceQuality(face.score, face.size, face.turn, sharpness)
        return feature / (np.linalg.norm(feature) + 1e-12), quality

    def embed(self, image: np.ndarray, face: Face) -> np.ndarray:
        return self.analyze(image, face)[0]

    def problems(self, quality: FaceQuality, min_size: float | None = None) -> list[str]:
        """Why this face is not good enough to store or search with (empty: it is)."""
        out = []
        min_size = self.min_face_size if min_size is None else min_size
        if quality.size < min_size:
            out.append(f"too small ({quality.size:.0f} px, needs {min_size:.0f})")
        if quality.sharpness < self.min_sharpness:
            out.append("blurred")
        if quality.turn > self.max_turn:
            out.append("turned away from the camera")
        return out

    def crop(self, image: np.ndarray, face: Face, margin: float = 0.3) -> np.ndarray:
        x, y, w, h = face.box
        x1, y1 = int(max(0, x - w * margin)), int(max(0, y - h * margin))
        x2 = int(min(image.shape[1], x + w * (1 + margin)))
        y2 = int(min(image.shape[0], y + h * (1 + margin)))
        return image[y1:y2, x1:x2]

    def photo_faces(self, image: np.ndarray) -> list[PhotoFace]:
        """For an uploaded photo: every face, left to right (so "face 2" means the second from
        the left), each with its print or the reasons it can't be used."""
        out = []
        for face in sorted(self.detect(image, retry=True), key=lambda f: f.box[0]):
            vector, quality = self.analyze(image, face)
            problems = self.problems(quality, min_size=MIN_PHOTO_FACE)
            out.append(PhotoFace(None if problems else vector, face.box, quality, problems))
        return out


def choose_face(faces: list[PhotoFace], choice: int | None = None) -> PhotoFace | str:
    """The face to search for in an uploaded photo, or a message saying why there is none.

    choice: 1-based, left to right. Without it, a photo with several faces is used only when
    one face is clearly the subject (at least twice as big as any other)."""
    if not faces:
        return "No face found in the photo."
    if choice is not None:
        if not 1 <= choice <= len(faces):
            return f"The photo has {len(faces)} face(s); pick 1 to {len(faces)} (left to right)."
        face = faces[choice - 1]
    elif len(faces) == 1:
        face = faces[0]
    else:
        by_size = sorted(faces, key=lambda f: -f.quality.size)
        if by_size[0].quality.size < 2 * by_size[1].quality.size:
            return (f"The photo has {len(faces)} faces; say which one to look for "
                    f"(1 to {len(faces)}, left to right).")
        face = by_size[0]
    if face.problems:
        return "The face in the photo can't be used: " + ", ".join(face.problems) + "."
    return face
