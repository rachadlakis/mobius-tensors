"""Test bench: pick a model, run it on an image, a video, or the webcam, compare before/after.
The Faces tab detects faces in uploaded photos (YuNet) and scores pairs of them (SFace).

The Argus UI shows it as its "Test bench" tab (`build_bench` + `add_live_route`). On its own:
    uv run --group ui -m argus.engine.app
"""

from __future__ import annotations

import base64
import copy
import html
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np
import yaml

from argus.engine.cameras import LiveCapture, is_stream_url, load_cameras, mask_url, resolve_source
from argus.engine.detector import COCO_CLASSES, Detection, YoloxOnnxDetector, draw_box, draw_detections
from argus.engine.download_model import FACE_MODELS, MODELS, download, fetch
from argus.engine.faces import LIKELY_SAME_PERSON, MIN_PHOTO_FACE, SAME_PERSON_THRESHOLD, FaceEngine
from argus.engine.live import BOUNDARY as MJPEG_BOUNDARY
from argus.engine.live import LiveView
from argus.engine.pipeline import Pipeline
from argus.engine.tracker import Tracker
from argus.engine.zones import Line, Rule, Zone, ZoneMonitor, draw_rules

PROJECT = Path(__file__).resolve().parents[3]  # projects/argus, absolute so any cwd works
MODELS_DIR = PROJECT / "models"
CONFIG_PATH = PROJECT / "configs" / "default.yaml"  # where IP cameras are declared
DEFAULT_MODEL = "yolox_nano"
BRAND = PROJECT.parents[1] / "Images" / "argus" / "web"  # repo's Argus kit

CSS = """
.ag-header {display: flex; align-items: center; gap: 14px; padding: 4px 0 6px}
.ag-header svg {width: 46px; height: 46px; flex: none}
.ag-title {font-size: 1.6rem; font-weight: 700; line-height: 1.1}
.ag-sub {color: var(--body-text-color-subdued); font-size: .9rem}
/* hand cursor on anything picked from a list */
input[role="combobox"], .wrap:has(> .wrap-inner input[role="combobox"]), .dropdown-arrow,
[role="option"], label:has(> input[type="radio"]), label:has(> input[type="checkbox"]),
input[type="radio"], input[type="checkbox"] {cursor: pointer}
"""

_detectors: dict[str, YoloxOnnxDetector] = {}
_face_engine: FaceEngine | None = None


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


def get_face_engine() -> FaceEngine:
    """YuNet + SFace, downloaded on first use. Small faces are kept so they show as refused."""
    global _face_engine
    if _face_engine is None:
        for filename, base in FACE_MODELS.items():
            if not (MODELS_DIR / filename).exists():
                gr.Info(f"Downloading {filename}...")
                fetch(f"{base}/{filename}", MODELS_DIR / filename)
        yunet, sface = (MODELS_DIR / name for name in FACE_MODELS)
        _face_engine = FaceEngine(yunet, sface, min_face_size=20)
    return _face_engine


def verdict(score: float) -> str:
    if score >= LIKELY_SAME_PERSON:
        return "likely same person"
    if score >= SAME_PERSON_THRESHOLD:
        return "possible"
    return "different"


def face_tile(crop: np.ndarray, face_id: str, photo: str, border: str) -> str:
    """A face crop as a fixed-size HTML tile (a Gallery stretches a few crops to full width)."""
    jpeg = base64.b64encode(cv2.imencode(".jpg", crop)[1].tobytes()).decode()
    return (f'<figure style="margin:0;width:112px;text-align:center;font-size:12px">'
            f'<img src="data:image/jpeg;base64,{jpeg}" style="width:112px;height:112px;'
            f'object-fit:contain;background:#000;border:3px solid {border};border-radius:6px">'
            f'<figcaption><b>{face_id}</b> {html.escape(photo)}</figcaption></figure>')


def faces_detect(files: list[str] | None):
    """Every face in every photo, numbered F1, F2... (photo by photo, left to right)."""
    if not files:
        raise gr.Error("Upload one or more photos first.")
    engine = get_face_engine()
    faces: list[dict[str, Any]] = []
    annotated, crops, rows = [], [], []
    for path in files:
        name = Path(path).name
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            gr.Warning(f"{name}: not an image")
            continue
        drawn = image.copy()
        found = sorted(engine.detect(image, retry=True), key=lambda f: f.box[0])
        for face in found:
            vector, quality = engine.analyze(image, face)
            problems = engine.problems(quality, min_size=MIN_PHOTO_FACE)
            face_id = f"F{len(faces) + 1}"
            faces.append({"id": face_id, "print": vector, "problems": problems})
            x, y, w, h = face.box
            color = (0, 160, 255) if problems else (60, 200, 60)
            draw_box(drawn, (x, y, x + w, y + h), f"{face_id} {face.score:.2f}", color)
            for lx, ly in face.landmarks[4:14].reshape(5, 2):
                cv2.circle(drawn, (int(lx), int(ly)), max(2, int(w / 40)), color, -1)
            crops.append(face_tile(engine.crop(image, face, margin=0.15), face_id, name,
                                   "#f59e0b" if problems else "#22c55e"))
            rows.append([face_id, name, round(face.score, 3), round(quality.size),
                         round(quality.turn, 2), round(quality.sharpness),
                         ", ".join(problems) or "ok"])
        annotated.append((cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB), f"{name} · {len(found)} face(s)"))
    ids = [f["id"] for f in faces]
    info = (f"{len(faces)} face(s) in {len(annotated)} photo(s). Orange = not good enough to "
            "search with on a real system (its scores are still shown: trust them less).")
    strip = f'<div style="display:flex;flex-wrap:wrap;gap:8px">{"".join(crops)}</div>'
    return (faces, annotated, strip, rows, info,
            gr.Dropdown(choices=ids, value=ids, multiselect=True), *faces_compare(faces, ids))


def faces_compare(faces: list[dict[str, Any]] | None, selected: list[str] | None):
    """SFace cosine similarity between the selected faces: a matrix and the pairs, best first."""
    import pandas as pd

    chosen = [f for f in faces or [] if f["id"] in (selected or [])]
    if len(chosen) < 2:
        return None, [], "Pick at least two faces to compare."
    ids = [f["id"] for f in chosen]
    prints = np.stack([f["print"] for f in chosen])
    scores = prints @ prints.T  # unit vectors: dot product = cosine similarity

    def colour(i: int, j: int) -> str:
        shade = "#e5e7eb" if i == j else {
            "likely same person": "#86efac", "possible": "#fde68a"}.get(
            verdict(scores[i, j]), "#f3f4f6")
        return f"background-color: {shade}; color: #111"

    # Text cells, coloured from the numbers: NaN on the diagonal breaks the table's updates.
    n = len(ids)
    cells = [["—" if i == j else f"{scores[i, j]:.3f}" for j in range(n)] for i in range(n)]
    matrix = pd.DataFrame(cells, index=ids, columns=ids).reset_index(names="face")
    css = pd.DataFrame([[colour(i, j) for j in range(n)] for i in range(n)], columns=ids)
    styled = matrix.style.apply(lambda _: css, axis=None, subset=ids)
    pairs = []
    for i in range(len(chosen)):
        for j in range(i + 1, len(chosen)):
            flagged = [f"{f['id']} {', '.join(f['problems'])}" for f in (chosen[i], chosen[j])
                       if f["problems"]]
            pairs.append([ids[i], ids[j], round(float(scores[i, j]), 3), verdict(scores[i, j]),
                          "; ".join(flagged)])
    pairs.sort(key=lambda row: -row[2])
    legend = (f"Cosine similarity of the SFace face prints. ≥ {LIKELY_SAME_PERSON}: likely the "
              f"same person (green) · {SAME_PERSON_THRESHOLD}–{LIKELY_SAME_PERSON}: possible "
              "(yellow) · below: different. The thresholds come from clear frontal photos (LFW).")
    return styled, pairs, legend


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
        raise gr.Error(f"{camera!r} is neither a camera in {CONFIG_PATH.name} nor an rtsp/http URL.")

    detector = get_detector(model, conf, nms, classes)
    yield None, f"Connecting to {mask_url(url)} ..."
    with LiveCapture(url) as cap:
        while True:
            frame = cap.read(timeout=10)
            if frame is None:
                yield None, (
                    f"No video from {mask_url(url)} for 10 s, still retrying. Check the IP, "
                    "credentials and path with `uv run -m argus.engine.cameras test <camera>`."
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
    with gr.Blocks(title="Argus Test Bench") as ui:
        logo = BRAND / "favicon.svg"
        gr.HTML(f'<div class="ag-header">{logo.read_text(encoding="utf-8") if logo.is_file() else ""}'
                '<div><div class="ag-title">Argus Test Bench</div>'
                '<div class="ag-sub">YOLOX (Apache-2.0) · ONNX Runtime · CPU</div></div></div>')
        build_bench()
    return ui


def build_bench() -> None:
    """The bench's settings and tabs, added to the Blocks being built (here or in Argus)."""
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
            f"Cameras declared under `cameras:` in `{CONFIG_PATH.name}`, or paste an "
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

    with gr.Tab("Faces"):
        gr.Markdown(
            "Upload photos (one or more faces each): faces are found with **YuNet** and "
            "compared with **SFace** face prints. Pick which faces to compare below."
        )
        f_state = gr.State([])
        with gr.Row():
            f_files = gr.File(file_count="multiple", file_types=["image"], type="filepath",
                              label="Photos", scale=1)
            f_gallery = gr.Gallery(label="Detections", columns=3, height=420, scale=2,
                                   object_fit="contain")
        f_btn = gr.Button("Detect faces", variant="primary")
        f_info = gr.Markdown()
        f_crops = gr.HTML()
        f_table = gr.Dataframe(
            headers=["face", "photo", "detector score", "size (px)", "turn", "sharpness",
                     "quality"],
            label="Faces found", interactive=False,
        )
        f_selected = gr.Dropdown([], multiselect=True, label="Compare these faces")
        f_legend = gr.Markdown()
        f_matrix = gr.Dataframe(label="SFace similarity", interactive=False)
        f_pairs = gr.Dataframe(
            headers=["face", "face", "SFace score", "verdict", "unreliable because"],
            label="Pairs, most similar first", interactive=False,
        )
        f_outputs = [f_state, f_gallery, f_crops, f_table, f_info, f_selected,
                     f_matrix, f_pairs, f_legend]
        f_btn.click(faces_detect, [f_files], f_outputs)
        f_selected.input(faces_compare, [f_state, f_selected], [f_matrix, f_pairs, f_legend])
        f_files.change(
            lambda: ([], None, "", None, "",
                     gr.Dropdown(choices=[], value=[], multiselect=True), None, None, ""),
            outputs=f_outputs, queue=False, show_progress="hidden",
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


def add_live_route(server: Any) -> None:
    """The raw MJPEG endpoint the Webcam tab's <img> reads (`server`: a FastAPI app)."""
    from fastapi.responses import StreamingResponse

    @server.get("/live/stream.mjpg")
    def live_stream() -> StreamingResponse:
        return StreamingResponse(
            LIVE.mjpeg(),
            media_type=f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}",
            headers={"Cache-Control": "no-store"},
        )


def main() -> None:
    """Serve the Gradio UI plus a raw MJPEG endpoint for the live webcam view."""
    import uvicorn
    from fastapi import FastAPI

    server = FastAPI()
    add_live_route(server)

    favicon = BRAND / "favicon-32.png"
    app = gr.mount_gradio_app(server, build_ui(), path="/", css=CSS,
                              favicon_path=str(favicon) if favicon.is_file() else None)
    print("* Running on local URL:  http://127.0.0.1:7860")
    try:
        uvicorn.run(app, host="127.0.0.1", port=7860, log_level="warning")
    finally:
        LIVE.stop()  # release the camera


if __name__ == "__main__":
    main()
