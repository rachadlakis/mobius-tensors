"""Run object detection (optionally with tracking, zones and line crossing) on any source.

Examples:
    uv run -m histor.engine.main --source 0 --show
    uv run -m histor.engine.main --source gate --show        # IP camera named `gate` in the config
    uv run -m histor.engine.main --source data/street.jpg --save outputs/street.jpg
    uv run -m histor.engine.main --config configs/engine/zones_example.yaml --source data/vtest.avi --show
    uv run -m histor.engine.main --config configs/engine/search_example.yaml   # events + search index
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import cv2
import yaml

from histor.engine.cameras import is_stream_url, main_stream_url, mask_url, resolve_source
from histor.engine.pipeline import Pipeline
from histor.engine.stream import is_live, iter_frames


def main() -> None:
    parser = argparse.ArgumentParser(description="General object detection")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--source",
        help="Webcam index, image/video path, URL, or camera name (overrides config)",
    )
    parser.add_argument("--show", action="store_true", help="Display frames in a window")
    parser.add_argument("--save", type=Path, help="Save annotated output (image or .mp4)")
    parser.add_argument("--camera", help="Name stored with events/search results (default: source)")
    parser.add_argument(
        "--start-time",
        help="For recorded files: when the video was filmed, e.g. 2026-09-25T14:00 "
             "(stored events and search results use it; default: now)",
    )
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv()  # ${VAR} secrets in the config (alert channels, cameras) come from .env
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source = args.source if args.source is not None else str(config.get("source", 0))
    # Alerts get the camera name, never the resolved URL (which may hold a password).
    camera_name = args.camera or source
    cameras = config.get("cameras") or {}
    main_url = main_stream_url(cameras[source]) if source in cameras else None
    source = resolve_source(source, cameras)
    if main_url is None and is_stream_url(source):
        main_url = str(source)  # a bare URL: record what we have
    start = datetime.fromisoformat(args.start_time).timestamp() if args.start_time else None
    # Sounds: the main stream usually carries the camera's audio; a video file its own track.
    audio = main_url or (str(source) if not str(source).isdigit() else None)
    pipeline = Pipeline.from_config(config, source=camera_name, start_time=start,
                                    live=is_live(source), main_url=main_url, audio_source=audio)
    print(f"Source: {mask_url(source)}")

    writer = None
    try:
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
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
        pipeline.close()
        if pipeline.store is not None:
            print(f"Store {pipeline.store.path}: {pipeline.store.counts()}")


if __name__ == "__main__":
    main()
