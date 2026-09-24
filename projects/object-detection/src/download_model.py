"""Download official pretrained YOLOX ONNX models (Apache-2.0, COCO 80 classes)."""

from __future__ import annotations

import argparse
import urllib.request
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
    parser.add_argument("name", nargs="?", default="yolox_nano", choices=MODELS)
    parser.add_argument("--out-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    path = download(args.name, args.out_dir)
    print(f"Done. Use input_size [{MODELS[args.name]}, {MODELS[args.name]}] for {path.name}.")


if __name__ == "__main__":
    main()
