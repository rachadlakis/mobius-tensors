"""Test UI: pick a model, run it on an image, a video, or the webcam, compare before/after.

Run with:
    uv run --group ui src/app.py
"""

from __future__ import annotations

import copy
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np
import yaml

from cameras import LiveCapture, is_stream_url, load_cameras, mask_url, resolve_source
from detector import COCO_CLASSES, Detection, YoloxOnnxDetector, draw_detections
from download_model import MODELS, download
from live import BOUNDARY as MJPEG_BOUNDARY
from live import LiveView
from pipeline import Pipeline
from tracker import Tracker
from zones import Line, Rule, Zone, ZoneMonitor, draw_rules

MODELS_DIR = Path("models")
CONFIG_PATH = Path("configs/default.yaml")  # where IP cameras are declared
DEFAULT_MODEL = "yolox_nano"

_detectors: dict[str, YoloxOnnxDetector] = {}


def available_models() -> list[str]:
    """Official YOLOX models (downloaded on first use) plus any other .onnx in models/."""
    local = sorted(p.stem for p in MODELS_DIR.glob("*.onnx"))
    return list(MODELS) + [name for name in local if name not in MODELS]


def get_detector(
    model: str, conf: float, nms: float, classes: list[str] | None
) -> YoloxOnnxDetector:
    """Load each model once; return a cheap copy sharing the session with per-call settings."""
    if model not in _detectors:
        path = MODELS_DIR / f"{model}.onnx"
        if not path.exists() and model in MODELS:
            gr.Info(f"Downloading {model}...")
            download(model, MODELS_DIR)
        _detectors[model] = YoloxOnnxDetector(path)

    detector = copy.copy(_detectors[model])
    detector.conf_threshold = conf
    detector.nms_threshold = nms
    detector.allowed_ids = (
        {detector.class_names.index(c) for c in classes} if classes else None
    )
    return detector


def detections_table(detections: list[Detection]) -> list[list]:
    return [
        [d.label, round(d.score, 3), int(d.x1), int(d.y1), int(d.x2), int(d.y2)]
        for d in detections
    ]


def detect_image(
    image: np.ndarray | None, model: str, conf: float, nms: float, classes: list[str]
):
    if image is None:
        raise gr.Error("Upload an image first (for a live camera, use the Webcam tab).")
    detector = get_detector(model, conf, nms, classes)
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    start = time.perf_counter()
    detections = detector.detect(frame)
    ms = (time.perf_counter() - start) * 1000
    annotated = cv2.cvtColor(draw_detections(frame, detections), cv2.COLOR_BGR2RGB)
    info = f"**{model}** · {len(detections)} detections · {ms:.1f} ms"
    return annotated, detections_table(detections), info


def open_writer(fps: float, size: tuple[int, int]) -> tuple[cv2.VideoWriter, str]:
    """Prefer H.264 MP4 / VP8 WebM so the browser can play the result."""
    for fourcc, suffix in (("avc1", ".mp4"), ("VP80", ".webm"), ("mp4v", ".mp4")):
        path = tempfile.NamedTemporaryFile(suffix=suffix, delete=False).name
        writer = cv2.VideoWriter(path, cv2.VideoWriter.fourcc(*fourcc), fps, size)
        if writer.isOpened():
            return writer, path
    raise gr.Error("No usable video codec found in OpenCV.")


def detect_video(
    video: str | None,
    model: str,
    conf: float,
    nms: float,
    classes: list[str],
    progress=gr.Progress(),
):
    if video is None:
        raise gr.Error("Upload a video first.")
    detector = get_detector(model, conf, nms, classes)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer, out_path = open_writer(fps, size)

    counts: dict[str, int] = {}
    frames, infer_time = 0, 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            start = time.perf_counter()
            detections = detector.detect(frame)
            infer_time += time.perf_counter() - start
            for d in detections:
                counts[d.label] = counts.get(d.label, 0) + 1
            writer.write(draw_detections(frame, detections))
            frames += 1
            if total:
                progress(frames / total, desc=f"Frame {frames}/{total}")
    finally:
        cap.release()
        writer.release()

    if frames == 0:
        raise gr.Error("Could not read any frames from this video.")
    summary = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
    info = (
        f"**{model}** · {frames} frames · avg {infer_time / frames * 1000:.1f} ms/frame "
        f"({frames / infer_time:.0f} FPS)  \nDetections per class (summed over frames): "
        f"{summary or 'none'}"
    )
    return out_path, info


# ------------------------------------------------------------------------ live webcam

# One camera reader for the whole app; the page shows it as an MJPEG <img> (see main()).
LIVE = LiveView()
LIVE_PLACEHOLDER = (
    '<div style="padding:48px;text-align:center;opacity:.6">Press <b>Start</b> to open the '
    "camera. Frames are captured and processed here, then streamed live to this page.</div>"
)


def live_start(source: str, model: str, conf: float, nms: float, classes: list[str]):
    source = (source or "0").strip()
    LIVE.start(
        int(source) if source.isdigit() else source,
        Pipeline(get_detector(model, conf, nms, classes)),
    )
    # Cache-buster so the browser opens a fresh stream on every start.
    html = (
        f'<img src="/live/stream.mjpg?v={time.time_ns()}" alt="live view" '
        'style="width:100%;max-height:75vh;object-fit:contain;border-radius:8px">'
    )
    return html, gr.Timer(active=True)


def live_stop():
    LIVE.stop()
    return LIVE_PLACEHOLDER, gr.Timer(active=False), "Stopped."


def live_status() -> str:
    s = LIVE.stats
    if not s:
        return ""
    parts = [f"**{s.get('status', '')}**", s.get("size", "")]
    if "fps" in s:
        parts.append(f"{s['fps']} FPS")
    if "infer_ms" in s:
        parts.append(f"{s['infer_ms']} ms/frame")
    labels = ", ".join(s.get("detections", [])) or "none"
    return " · ".join(p for p in parts if p) + f"  \nDetections: {labels}"


def live_settings(model: str, conf: float, nms: float, classes: list[str]) -> None:
    """Apply model / threshold changes to the running camera without restarting it."""
    if LIVE.running:
        LIVE.set_detector(get_detector(model, conf, nms, classes))


# ------------------------------------------------------------------------ IP cameras


def configured_cameras() -> dict[str, Any]:
    return load_cameras(CONFIG_PATH) if CONFIG_PATH.exists() else {}


def detect_ip_camera(
    camera: str | None, model: str, conf: float, nms: float, classes: list[str]
):
    """Stream detections from a configured camera (by name) or a pasted URL until Stop."""
    camera = (camera or "").strip()
    if not camera:
        raise gr.Error("Pick a camera from the config or paste an rtsp:// URL.")
    try:
        url = str(resolve_source(camera, configured_cameras()))
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    if not is_stream_url(url):
        raise gr.Error(f"{camera!r} is neither a camera in {CONFIG_PATH} nor an rtsp/http URL.")

    detector = get_detector(model, conf, nms, classes)
    yield None, f"Connecting to {mask_url(url)} ..."
    with LiveCapture(url) as cap:
        while True:
            frame = cap.read(timeout=10)
            if frame is None:
                yield None, (
                    f"No video from {mask_url(url)} for 10 s, still retrying. Check the IP, "
                    "credentials and path with `uv run src/cameras.py test <camera>`."
                )
                continue
            start = time.perf_counter()
            detections = detector.detect(frame)
            ms = (time.perf_counter() - start) * 1000
            draw_detections(frame, detections)
            h, w = frame.shape[:2]
            if w > 1280:  # a 4K main stream is too heavy to push to the browser every frame
                frame = cv2.resize(frame, (1280, h * 1280 // w), interpolation=cv2.INTER_AREA)
            labels = ", ".join(f"{d.label} {d.score:.2f}" for d in detections) or "none"
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), f"{w}x{h} · {ms:.1f} ms · {labels}"


# ------------------------------------------------------------------------ zones & lines


def build_rules(shapes: list[dict[str, Any]]) -> list[Rule]:
    rules: list[Rule] = []
    for s in shapes:
        if s["type"] == "zone":
            rules.append(Zone(s["name"], s["points"], s["classes"], s["dwell_seconds"]))
        else:
            rules.append(Line(s["name"], s["points"], s["classes"], s["direction"]))
    return rules


def render_canvas(bg: np.ndarray | None, shapes: list[dict], current: list[list[int]]):
    """First frame with finished shapes plus the one being drawn (RGB)."""
    if bg is None:
        return None
    frame = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)
    rules = build_rules(shapes)
    h, w = frame.shape[:2]
    for rule in rules:
        rule.resolve((w, h))
    draw_rules(frame, rules)
    pts = np.array(current, dtype=np.int32)
    if len(pts) > 1:
        cv2.polylines(frame, [pts], False, (0, 255, 255), 2)
    for x, y in current:
        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 255), -1)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def shapes_yaml(shapes: list[dict], bg: np.ndarray | None) -> str:
    """Config snippet with normalized points (works at any camera resolution)."""
    if not shapes or bg is None:
        return "# Draw a zone or line, then click 'Add shape'."
    h, w = bg.shape[:2]
    items = []
    for s in shapes:
        item: dict[str, Any] = {
            "name": s["name"],
            "type": s["type"],
            "points": [[round(x / w, 4), round(y / h, 4)] for x, y in s["points"]],
        }
        if s["classes"]:
            item["classes"] = s["classes"]
        if s["type"] == "zone" and s["dwell_seconds"]:
            item["dwell_seconds"] = s["dwell_seconds"]
        if s["type"] == "line":
            item["direction"] = s["direction"]
        items.append(item)
    return yaml.safe_dump({"zones": items}, sort_keys=False, default_flow_style=None)


def zones_load_video(video: str | None):
    if video is None:
        return None, None, [], [], shapes_yaml([], None)
    cap = cv2.VideoCapture(video)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise gr.Error("Could not read this video.")
    bg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return bg, render_canvas(bg, [], []), [], [], shapes_yaml([], bg)


def zones_click(bg, shapes, current, shape_type, evt: gr.SelectData):
    if bg is None:
        raise gr.Error("Upload a video first.")
    if shape_type == "line" and len(current) >= 2:
        gr.Warning("A line has 2 points: click 'Add shape' or 'Undo point'.")
        return current, render_canvas(bg, shapes, current)
    current = [*current, [int(evt.index[0]), int(evt.index[1])]]
    return current, render_canvas(bg, shapes, current)


def zones_undo(bg, shapes, current):
    current = current[:-1]
    return current, render_canvas(bg, shapes, current)


def zones_clear(bg):
    return [], [], render_canvas(bg, [], []), shapes_yaml([], bg)


def zones_add(bg, shapes, current, shape_type, name, zone_classes, dwell, direction):
    name = (name or "").strip()
    if shape_type == "zone" and len(current) < 3:
        raise gr.Error("A zone needs at least 3 points.")
    if shape_type == "line" and len(current) != 2:
        raise gr.Error("A line needs exactly 2 points.")
    if not name or any(s["name"] == name for s in shapes):
        raise gr.Error("Give the shape a unique name.")
    shape = {
        "name": name, "type": shape_type, "points": current,
        "classes": list(zone_classes or []), "dwell_seconds": float(dwell or 0) or None,
        "direction": direction,
    }
    shapes = [*shapes, shape]
    next_name = f"{shape_type}_{sum(s['type'] == shape_type for s in shapes) + 1}"
    return shapes, [], render_canvas(bg, shapes, []), shapes_yaml(shapes, bg), next_name


def zones_run(
    video: str | None, shapes: list[dict], model: str, conf: float, nms: float,
    classes: list[str], progress=gr.Progress(),
):
    if video is None:
        raise gr.Error("Upload a video first.")
    if not shapes:
        raise gr.Error("Add at least one zone or line first.")
    detector = get_detector(model, conf, nms, classes)
    detector.class_agnostic_nms = True  # one box per object, or a van becomes car + truck tracks
    pipeline = Pipeline(detector, Tracker(), ZoneMonitor(build_rules(shapes)))

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer, out_path = open_writer(fps, size)
    events = []
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = pipeline.process(frame, index / fps)
            events.extend(result.events)
            writer.write(pipeline.draw(frame, result))
            index += 1
            if total:
                progress(index / total, desc=f"Frame {index}/{total}")
    finally:
        cap.release()
        writer.release()

    rows = [
        [f"{e.timestamp:.1f}", e.type, e.rule, e.label, e.track_id,
         e.direction or (f"{e.dwell_seconds:.1f}s" if e.dwell_seconds is not None else "")]
        for e in events
    ]
    lines = []
    for rule in pipeline.monitor.rules if pipeline.monitor else []:
        if isinstance(rule, Line):
            lines.append(f"**{rule.name}** (line): in {rule.counts['in']} · out {rule.counts['out']}")
        else:
            n = {t: sum(e.rule == rule.name and e.type == t for e in events)
                 for t in ("enter", "exit", "dwell")}
            lines.append(
                f"**{rule.name}** (zone): {n['enter']} entered · {n['exit']} exited · "
                f"{n['dwell']} dwell alerts"
            )
    return out_path, rows, "  \n".join(lines) or "No events."


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Object Detection Test Bench") as ui:
        gr.Markdown("# Object Detection Test Bench\nYOLOX (Apache-2.0) · ONNX Runtime · CPU")

        with gr.Row():
            model = gr.Dropdown(
                available_models(), value=DEFAULT_MODEL, label="Model",
                info="nano/tiny: fast CPU · s/m/l/x: more accurate (downloaded on first use)",
            )
            conf = gr.Slider(0.05, 0.95, value=0.35, step=0.05, label="Confidence threshold")
            nms = gr.Slider(0.1, 0.9, value=0.45, step=0.05, label="NMS IoU threshold")
            classes = gr.Dropdown(
                list(COCO_CLASSES), multiselect=True, label="Only these classes",
                info="Empty = all classes",
            )
        settings = [model, conf, nms, classes]

        with gr.Tab("Image"):
            with gr.Row():
                img_in = gr.Image(type="numpy", sources=["upload", "clipboard"], label="Original")
                img_out = gr.Image(type="numpy", label="Detections", interactive=False)
            img_btn = gr.Button("Detect", variant="primary")
            img_info = gr.Markdown()
            img_table = gr.Dataframe(
                headers=["label", "score", "x1", "y1", "x2", "y2"], label="Detections",
                interactive=False,
            )
            img_btn.click(detect_image, [img_in, *settings], [img_out, img_table, img_info])

        with gr.Tab("Video"):
            with gr.Row():
                vid_in = gr.Video(label="Original", sources=["upload"])
                vid_out = gr.Video(label="Detections", interactive=False)
            vid_btn = gr.Button("Detect", variant="primary")
            vid_info = gr.Markdown()
            vid_btn.click(detect_video, [vid_in, *settings], [vid_out, vid_info])

        with gr.Tab("Webcam"):
            with gr.Row():
                cam_source = gr.Textbox(
                    value="0", label="Camera", scale=3,
                    info="Webcam index on this machine (0, 1, ...)",
                )
                cam_start = gr.Button("Start", variant="primary", scale=1)
                cam_stop = gr.Button("Stop", scale=1)
            cam_view = gr.HTML(LIVE_PLACEHOLDER)
            cam_info = gr.Markdown()
            cam_timer = gr.Timer(0.5, active=False)
            cam_start.click(
                live_start, [cam_source, *settings], [cam_view, cam_timer], queue=False
            )
            cam_stop.click(live_stop, outputs=[cam_view, cam_timer, cam_info], queue=False)
            cam_timer.tick(live_status, outputs=[cam_info], queue=False, show_progress="hidden")

        with gr.Tab("IP camera"):
            gr.Markdown(
                f"Cameras declared under `cameras:` in `{CONFIG_PATH}`, or paste an "
                "`rtsp://user:pass@ip:554/path` URL. Frames are skipped when detection is "
                "slower than the camera, so what you see is live."
            )
            with gr.Row():
                ip_camera = gr.Dropdown(
                    list(configured_cameras()), label="Camera", allow_custom_value=True,
                    scale=3,
                )
                ip_start = gr.Button("Start", variant="primary", scale=1)
                ip_stop = gr.Button("Stop", scale=1)
            ip_out = gr.Image(type="numpy", label="Detections", interactive=False)
            ip_info = gr.Markdown()
            ip_event = ip_start.click(
                detect_ip_camera, [ip_camera, *settings], [ip_out, ip_info],
                concurrency_limit=4,
            )
            ip_stop.click(
                lambda: (None, "Stopped."), outputs=[ip_out, ip_info], cancels=[ip_event],
                queue=False,
            )

        with gr.Tab("Zones & lines"):
            gr.Markdown(
                "1. Upload a video · 2. Click points on the frame (zone: 3+ points, line: 2) · "
                "3. **Add shape** · 4. **Run**. Objects are tracked; events fire when their "
                "bottom-center (feet / wheels) enters, stays in, leaves a zone or crosses a line."
            )
            z_bg = gr.State(None)
            z_shapes = gr.State([])
            z_current = gr.State([])
            with gr.Row():
                z_video = gr.Video(label="Video", sources=["upload"])
                z_canvas = gr.Image(
                    type="numpy", label="Click to add points", interactive=False,
                )
            with gr.Row():
                z_type = gr.Radio(["zone", "line"], value="zone", label="Shape")
                z_name = gr.Textbox(value="zone_1", label="Name")
                z_classes = gr.Dropdown(
                    list(COCO_CLASSES), multiselect=True, label="Classes",
                    info="Empty = all classes",
                )
                z_dwell = gr.Number(
                    value=0, minimum=0, label="Dwell alert after (s)",
                    info="Zones: e.g. 60 = parked for a minute · 0 = off",
                )
                z_direction = gr.Radio(
                    ["both", "in", "out"], value="both", label="Line direction",
                    info="'in' = direction of the arrow",
                )
            with gr.Row():
                z_add = gr.Button("Add shape", variant="secondary")
                z_undo = gr.Button("Undo point")
                z_clear = gr.Button("Clear all")
                z_run = gr.Button("Run", variant="primary")
            with gr.Row():
                z_out = gr.Video(label="Result", interactive=False)
                with gr.Column():
                    z_summary = gr.Markdown()
                    z_yaml = gr.Code(
                        language="yaml", label="Config snippet (paste into a config file)",
                        interactive=False,
                    )
            z_events = gr.Dataframe(
                headers=["time (s)", "event", "rule", "label", "track", "detail"],
                label="Events", interactive=False,
            )

            z_video.change(
                zones_load_video, [z_video], [z_bg, z_canvas, z_shapes, z_current, z_yaml]
            )
            z_canvas.select(zones_click, [z_bg, z_shapes, z_current, z_type], [z_current, z_canvas])
            z_undo.click(zones_undo, [z_bg, z_shapes, z_current], [z_current, z_canvas])
            z_clear.click(zones_clear, [z_bg], [z_shapes, z_current, z_canvas, z_yaml])
            z_add.click(
                zones_add,
                [z_bg, z_shapes, z_current, z_type, z_name, z_classes, z_dwell, z_direction],
                [z_shapes, z_current, z_canvas, z_yaml, z_name],
            )
            z_type.change(
                lambda t, shapes: f"{t}_{sum(s['type'] == t for s in shapes) + 1}",
                [z_type, z_shapes], [z_name],
            )
            z_run.click(
                zones_run, [z_video, z_shapes, *settings], [z_out, z_events, z_summary]
            )

        # A result belongs to the settings and input that produced it: clear it as soon as
        # either changes, so a stale result is never mistaken for the new one.
        # (The webcam output refreshes on its own every frame.)
        settings_changed = [s.change for s in settings]
        # The live webcam view picks up new settings on the fly instead.
        gr.on(triggers=settings_changed, fn=live_settings, inputs=settings, queue=False)
        gr.on(
            triggers=[*settings_changed, img_in.change], fn=lambda: (None, None, ""),
            outputs=[img_out, img_table, img_info], queue=False, show_progress="hidden",
        )
        gr.on(
            triggers=[*settings_changed, vid_in.change], fn=lambda: (None, ""),
            outputs=[vid_out, vid_info], queue=False, show_progress="hidden",
        )
        gr.on(
            # gr.State has a runtime .change event that Gradio's type hints don't declare.
            triggers=[*settings_changed, z_video.change, z_shapes.change],  # type: ignore[attr-defined]
            fn=lambda: (None, None, ""), outputs=[z_out, z_events, z_summary],
            queue=False, show_progress="hidden",
        )
        gr.on(
            # The camera stream keeps the settings it started with: stop it rather than
            # keep showing detections from the old model / thresholds.
            triggers=[*settings_changed, ip_camera.change],
            fn=lambda: (None, "Settings changed: press **Start** to run with the new ones."),
            outputs=[ip_out, ip_info], cancels=[ip_event], queue=False, show_progress="hidden",
        )
    return ui


def main() -> None:
    """Serve the Gradio UI plus a raw MJPEG endpoint for the live webcam view."""
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse

    server = FastAPI()

    @server.get("/live/stream.mjpg")
    def live_stream() -> StreamingResponse:
        return StreamingResponse(
            LIVE.mjpeg(),
            media_type=f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}",
            headers={"Cache-Control": "no-store"},
        )

    app = gr.mount_gradio_app(server, build_ui(), path="/")
    print("* Running on local URL:  http://127.0.0.1:7860")
    try:
        uvicorn.run(app, host="127.0.0.1", port=7860, log_level="warning")
    finally:
        LIVE.stop()  # release the camera


if __name__ == "__main__":
    main()
