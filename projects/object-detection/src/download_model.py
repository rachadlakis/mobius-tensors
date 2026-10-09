"""Download official pretrained YOLOX ONNX models (Apache-2.0, COCO 80 classes), the face
models (`faces`), or the trial pose model (`pose_rtmpose_t`)."""

from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

RELEASE_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0"
# name -> native input size. nano/tiny are the CPU / edge picks.
MODELS = {
    "yolox_nano": 416,
    "yolox_tiny": 416,
    "yolox_s": 640,
    "yolox_m": 640,
    "yolox_l": 640,
    "yolox_x": 640,
}


# Face models (OpenCV Zoo): YuNet detector (MIT) + SFace recognizer (Apache-2.0).
ZOO_URL = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"
FACE_MODELS = {
    "face_detection_yunet_2023mar.onnx": f"{ZOO_URL}/face_detection_yunet",
    "face_recognition_sface_2021dec.onnx": f"{ZOO_URL}/face_recognition_sface",
}


# Pose, for trials only: RTMPose-t (OpenMMLab, Apache-2.0 code) was trained on seven datasets,
# some research-only (PoseTrack18...). On client sites use ViTPose-B: src/export_pose.py.
RTMPOSE_T_URL = ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
                 "rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.zip")


def fetch_rtmpose(target: Path) -> Path:
    """The zip holds an MMDeploy SDK folder; keep only its end2end.onnx."""
    if target.exists():
        print(f"Already present: {target}")
        return target
    print(f"Downloading {RTMPOSE_T_URL}")
    with urllib.request.urlopen(RTMPOSE_T_URL) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    [name] = [n for n in archive.namelist() if n.endswith("end2end.onnx")]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(archive.read(name))
    print(f"Wrote {target}")
    return target


def fetch(url: str, target: Path) -> Path:
    if target.exists():
        print(f"Already present: {target}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {target}")
    partial = target.with_suffix(target.suffix + ".part")
    urllib.request.urlretrieve(url, partial)
    partial.replace(target)  # only a complete download gets the real name
    return target


def download(name: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{name}.onnx"
    if target.exists():
        print(f"Already present: {target}")
        return target
    url = f"{RELEASE_URL}/{name}.onnx"
    print(f"Downloading {url} -> {target}")
    partial = target.with_suffix(".onnx.part")
    urllib.request.urlretrieve(url, partial)
    partial.replace(target)  # only a complete download gets the real name
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", nargs="?", default="yolox_nano", choices=[*MODELS, "faces", "pose_rtmpose_t"],
                        help="a YOLOX model, `faces` for the face detector + recognizer, or "
                             "`pose_rtmpose_t` (trial pose model)")
    parser.add_argument("--out-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    if args.name == "faces":
        for filename, base in FACE_MODELS.items():
            fetch(f"{base}/{filename}", args.out_dir / filename)
        return
    if args.name == "pose_rtmpose_t":
        fetch_rtmpose(args.out_dir / "pose_rtmpose_t.onnx")
        return
    path = download(args.name, args.out_dir)
    print(f"Done. Use input_size [{MODELS[args.name]}, {MODELS[args.name]}] for {path.name}.")


if __name__ == "__main__":
    main()
