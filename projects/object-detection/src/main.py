"""Run object detection (optionally with tracking, zones and line crossing) on any source.

Examples:
    uv run src/main.py --source 0 --show
    uv run src/main.py --source gate --show        # IP camera named `gate` in the config
    uv run src/main.py --source data/street.jpg --save outputs/street.jpg
    uv run src/main.py --config configs/zones_example.yaml --source data/vtest.avi --show
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2
import yaml

from cameras import mask_url, resolve_source
from pipeline import Pipeline
from stream import iter_frames


def main() -> None:
    parser = argparse.ArgumentParser(description="General object detection")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--source",
        help="Webcam index, image/video path, URL, or camera name (overrides config)",
    )
    parser.add_argument("--show", action="store_true", help="Display frames in a window")
    parser.add_argument("--save", type=Path, help="Save annotated output (image or .mp4)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = yaml.safe_load(args.config.read_text())
    source = args.source if args.source is not None else str(config.get("source", 0))
    # Alerts get the camera name, never the resolved URL (which may hold a password).
    pipeline = Pipeline.from_config(config, source=source)
    source = resolve_source(source, config.get("cameras"))
    print(f"Source: {mask_url(source)}")

    writer = None
    for frame in iter_frames(source):
        image = frame.image
        start = time.perf_counter()
        result = pipeline.process(image, frame.timestamp)
        fps = 1.0 / (time.perf_counter() - start)

        if pipeline.monitor is None:  # with zones, the alert manager prints events instead
            summary = ", ".join(f"{d.label} {d.score:.2f}" for d in result.detections) or "none"
            print(f"[{fps:5.1f} FPS] {len(result.detections)} detections: {summary}")

        if args.show or args.save:
            pipeline.draw(image, result)
        if args.save:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            if args.save.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                cv2.imwrite(str(args.save), image)
            else:
                if writer is None:
                    h, w = image.shape[:2]
                    writer = cv2.VideoWriter(
                        str(args.save), cv2.VideoWriter.fourcc(*"mp4v"), frame.fps or 25, (w, h)
                    )
                writer.write(image)
        if args.show:
            cv2.imshow("detections", image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
