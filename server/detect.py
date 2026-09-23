"""Phase 0, Step 1: video in -> YOLO -> console + live annotated window.

Usage:
    python detect.py                          # reads data/sample.mp4
    python detect.py --source path/to/clip.mp4
    python detect.py --source 0                # webcam
    python detect.py --no-display               # console output only
"""

import argparse
from pathlib import Path
from typing import Union
import numpy as np

import cv2
from ultralytics import YOLO

BOX_COLOR = (0, 200, 0)
TEXT_COLOR = (255, 255, 255)


def parse_args():
    """Parse command-line arguments.    
    
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run YOLO detection on a video source.")
    parser.add_argument("--source", default="data/sample.mp4", help="Video file path or webcam index (e.g. 0)")
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics model weights")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--no-display", action="store_true", help="Skip the live annotated window, console output only")
    return parser.parse_args()


def resolve_source(raw_source: str) -> Union[str, int]:
    """Resolve the video source.
    
    Args:
        raw_source (str): The raw video source input (file path or webcam index).

    Returns:
        Union[str, int]: Resolved video source (file path or webcam index).

    Raises:
        SystemExit: If the video source does not exist.
    """
    if raw_source.isdigit():
        return int(raw_source)
    if not Path(raw_source).exists():
        raise SystemExit(
            f"Video source not found: {raw_source}\n"
            f"Drop a short test clip at server/data/sample.mp4, or pass --source <path>."
        )
    return raw_source


def draw_detections(frame, boxes, names) -> None:
    """Draw detected bounding boxes on the frame.

    Args:
        frame (np.ndarray): The video frame.
        boxes (list): List of detected bounding boxes.
        names (dict): Class names mapping.
    """
    for box in boxes:
        cls_name = names[int(box.cls[0])]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = (round(v) for v in box.xyxy[0].tolist())
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
        label = f"{cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), BOX_COLOR, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)


def main():
    """Main function to run YOLO detection on a video source."""
    args = parse_args()
    source = resolve_source(args.source)

    model = YOLO(args.model)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {source}")

    frame_idx = 0
    print(f"Running {args.model} on {source} (confidence >= {args.conf}).")
    print("Press 'q' in the window to stop." if not args.no_display else "Press Ctrl+C to stop.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = model.predict(frame, conf=args.conf, verbose=False)[0]
            boxes = results.boxes if results.boxes is not None else []

            for box in boxes:
                cls_name = model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = (round(v) for v in box.xyxy[0].tolist())
                print(f"frame {frame_idx:05d} | {cls_name:12s} | conf {conf:.2f} | box ({x1},{y1})-({x2},{y2})")

            if not args.no_display:
                draw_detections(frame, boxes, model.names)
                cv2.imshow("Mobius Tensors - detect.py", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        print(f"Stopped after {frame_idx} frames.")


if __name__ == "__main__":
    main()
