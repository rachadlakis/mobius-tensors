"""A test video for Histor's fire alerts, from held-out test pictures (no model saw them):
clean scenes, then indoor fire, clean, indoor smoke, clean; each picture held 3 s at 10 fps
with camera noise. Writes ../../data/test_videos/fire_loop.mp4 and fire_loop.json (what is in
view when), used by the smoke test (tests/smoke/test_smoke.py::test_fire_alerts_on_a_video).

    .venv/Scripts/python.exe -I make_test_video.py                 # from data/v2_test (hf_upload.py --get)
    .venv/Scripts/python.exe -I make_test_video.py --test data/merged
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent / "data" / "test_videos"


def classes(label: Path) -> set[int]:
    v = label.read_text().split()
    return {int(float(v[i])) for i in range(0, len(v), 5)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--test", type=Path, default=HERE / "data" / "v2_test" / "all",
                    help="dataset folder with images/test and labels/test")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    labels = args.test / "labels" / "test"
    pick = random.Random(args.seed)
    fire = pick.sample([p.stem for p in sorted(labels.glob("indoor_*.txt")) if classes(p) == {0}], 3)
    smoke = pick.sample([p.stem for p in sorted(labels.glob("indoor_*.txt")) if classes(p) == {1}], 3)
    clean = pick.sample([p.stem for p in sorted(labels.glob("dfire_*.txt"))
                         if not p.read_text().strip()], 7)
    plan = ([("clean", c) for c in clean[:3]] + [("fire", f) for f in fire]
            + [("clean", c) for c in clean[3:5]] + [("smoke", s) for s in smoke]
            + [("clean", c) for c in clean[5:7]])

    fps, hold, width, height = 10, 3, 1280, 720
    OUT.mkdir(parents=True, exist_ok=True)
    video = cv2.VideoWriter(str(OUT / "fire_loop.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                            (width, height))
    rng = np.random.default_rng(0)
    timeline = []
    for k, (kind, stem) in enumerate(plan):
        image = cv2.imread(str(args.test / "images" / "test" / f"{stem}.jpg"))
        scale = min(width / image.shape[1], height / image.shape[0])
        small = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
        canvas = np.zeros((height, width, 3), np.uint8)
        y, x = (height - small.shape[0]) // 2, (width - small.shape[1]) // 2
        canvas[y:y + small.shape[0], x:x + small.shape[1]] = small
        timeline.append({"from": k * hold, "to": (k + 1) * hold, "kind": kind, "image": stem})
        for _ in range(fps * hold):  # camera noise, so no two frames are identical
            noise = rng.normal(0, 4, canvas.shape)
            video.write(np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8))
    video.release()
    (OUT / "fire_loop.json").write_text(json.dumps(timeline, indent=1), encoding="utf-8")
    print(f"wrote {OUT / 'fire_loop.mp4'} ({len(plan) * hold} s)")


if __name__ == "__main__":
    main()
