"""Run object detection on a webcam, image, video, or stream URL.

Examples:
    uv run src/main.py --source 0 --show
    uv run src/main.py --source data/street.jpg --save outputs/street.jpg
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import yaml

from detector import create_detector, draw_detections
from stream import iter_frames


def main() -> None:
    parser = argparse.ArgumentParser(description="General object detection")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--source", help="Webcam index, image/video path, or URL (overrides config)")
    parser.add_argument("--show", action="store_true", help="Display frames in a window")
    parser.add_argument("--save", type=Path, help="Save annotated output (image or .mp4)")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    detector = create_detector(config["model"])
    source = args.source if args.source is not None else str(config.get("source", 0))

    writer = None
    for frame in iter_frames(source):
        start = time.perf_counter()
        detections = detector.detect(frame)
        fps = 1.0 / (time.perf_counter() - start)

        summary = ", ".join(f"{d.label} {d.score:.2f}" for d in detections) or "none"
        print(f"[{fps:5.1f} FPS] {len(detections)} detections: {summary}")

        if args.show or args.save:
            draw_detections(frame, detections)
        if args.save:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            if args.save.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                cv2.imwrite(str(args.save), frame)
            else:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(str(args.save), cv2.VideoWriter.fourcc(*"mp4v"), 25, (w, h))
                writer.write(frame)
        if args.show:
            cv2.imshow("detections", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
