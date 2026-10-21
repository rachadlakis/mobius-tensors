"""Test videos for Histor's fire alerts, from held-out test pictures (no model saw them), for
the scenario bench (tests/scenarios/specs/fire) and the smoke test:

    fire_loop.mp4    clean scenes, then indoor fire, clean, indoor smoke, clean; each picture
                     held 3 s at 10 fps with camera noise. With --flicker (the default) the
                     flames and the smoke move inside their labelled boxes the way they do on
                     a real camera (compose.Flicker: a fresh warp and brightness per frame for
                     flames, a slow drift for smoke); --no-flicker: still pictures, as before.
    fire_poster.mp4  one fire picture held still for 30 s: a poster or a screen showing fire

Each with a .json of what is in view when, written to ../../data/test_videos/.

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

from histor.engine.compose import Flicker, yolo_boxes

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent / "data" / "test_videos"


def classes(label: Path) -> set[int]:
    v = label.read_text().split()
    return {int(float(v[i])) for i in range(0, len(v), 5)}


def read(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return image


def place(image: np.ndarray, width: int, height: int) -> tuple[np.ndarray, float, int, int]:
    """The picture scaled into a black width x height canvas: canvas, scale, x and y offsets."""
    scale = min(width / image.shape[1], height / image.shape[0])
    small = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
    canvas = np.zeros((height, width, 3), np.uint8)
    y, x = (height - small.shape[0]) // 2, (width - small.shape[1]) // 2
    canvas[y:y + small.shape[0], x:x + small.shape[1]] = small
    return canvas, scale, x, y


def write(path: Path, pictures: list[tuple[str, np.ndarray, list[tuple[Flicker, tuple[int, int, int, int]]]]],
          fps: int, hold: float, rng: np.random.Generator) -> list[dict[str, object]]:
    """Each (kind, canvas, flickering boxes) held `hold` s with camera noise; the timeline."""
    height, width = pictures[0][1].shape[:2]
    video = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    timeline = []
    for k, (kind, canvas, moving) in enumerate(pictures):
        timeline.append({"from": k * hold, "to": (k + 1) * hold, "kind": kind})
        for _ in range(int(fps * hold)):  # camera noise, so no two frames are identical
            frame = canvas.copy()
            for flicker, (x1, y1, x2, y2) in moving:
                frame[y1:y2, x1:x2] = flicker(canvas[y1:y2, x1:x2])
            noise = rng.normal(0, 4, frame.shape)
            video.write(np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8))
    video.release()
    return timeline


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--test", type=Path, default=HERE / "data" / "v2_test" / "all",
                    help="dataset folder with images/test and labels/test")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--flicker", action=argparse.BooleanOptionalAction, default=True,
                    help="flames and smoke move inside their boxes (default)")
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
    rng = np.random.default_rng(0)
    pictures = []
    for kind, stem in plan:
        image = read(args.test / "images" / "test" / f"{stem}.jpg")
        canvas, scale, x, y = place(image, width, height)
        moving = []
        if args.flicker and kind != "clean":
            for c, (x1, y1, x2, y2) in yolo_boxes(labels / f"{stem}.txt", image.shape[1::-1]):
                box = (int(x + max(x1, 0) * scale), int(y + max(y1, 0) * scale),
                       int(x + min(x2, image.shape[1]) * scale),
                       int(y + min(y2, image.shape[0]) * scale))
                if box[2] - box[0] >= 4 and box[3] - box[1] >= 4:  # flames: fresh; smoke: drifts
                    moving.append((Flicker(rng) if c == 0 else
                                   Flicker(rng, amp=(2.0, 4.0), gain=(0.95, 1.05), smooth=0.9),
                                   box))
        pictures.append((kind, canvas, moving))
    timeline = write(OUT / "fire_loop.mp4", pictures, fps, hold, rng)
    for entry, (_, stem) in zip(timeline, plan):
        entry["image"] = stem
    (OUT / "fire_loop.json").write_text(json.dumps(timeline, indent=1), encoding="utf-8")
    print(f"wrote {OUT / 'fire_loop.mp4'} ({len(plan) * hold} s)")

    poster = fire[0]  # the first fire picture, still: no flicker
    canvas = place(read(args.test / "images" / "test" / f"{poster}.jpg"), width, height)[0]
    timeline = write(OUT / "fire_poster.mp4", [("fire", canvas, [])], fps, 30, rng)
    timeline[0]["image"] = poster
    (OUT / "fire_poster.json").write_text(json.dumps(timeline, indent=1), encoding="utf-8")
    print(f"wrote {OUT / 'fire_poster.mp4'} (30 s, still)")


if __name__ == "__main__":
    main()
