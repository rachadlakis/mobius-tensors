"""Test UI: record a camera (a laptop's webcam is enough), then look for a person by photo or ask.

    uv run --group ui src/ui.py                                   # configs/laptop.yaml, webcam 0
    uv run --group ui src/ui.py --config configs/default.yaml --camera gate

Tabs:
    Camera          start / stop recording, live view with tracks, the faces being stored
    Find a person   photo + time range -> visits (likely / possible), no LLM needed
    Ask             the chat (needs Ollama or Claude, see chat.models), with photo attachments

One recorder and one conversation for the whole page: a local test tool, not a multi-user app.
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np
import yaml
from dotenv import load_dotenv

import shared  # noqa: F401
from ask import build_engine
from chat import answer, model_config, system_prompt
from llm import Backend, create_backend
from pipeline import Pipeline
from search import SearchEngine
from stream import iter_frames
from tools import Toolbox

log = logging.getLogger("argus.ui")
ALL_CAMERAS = "all cameras"


class Recorder:
    """The recording loop (detect, track, index faces) on a background thread. The page polls
    `frame` for the live view."""

    def __init__(self, config: dict[str, Any], camera: str) -> None:
        self.config = config
        self.camera = camera
        self.frame: np.ndarray | None = None      # latest annotated frame, RGB
        self.raw: np.ndarray | None = None        # same frame without drawings, BGR
        self.frame_no = 0
        self.status = "Stopped."
        self.pipeline: Pipeline | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, source: str) -> None:
        if self.running:
            return
        self._stop.clear()
        self.status = f"Opening {source}..."
        self._thread = threading.Thread(target=self._run, args=(source,), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=60)  # the indexer finishes its queue first

    def _run(self, source: str) -> None:
        try:
            self.pipeline = Pipeline.from_config(self.config, source=self.camera, live=True)
            started, frames = time.monotonic(), 0
            for frame in iter_frames(source):
                if self._stop.is_set():
                    break
                result = self.pipeline.process(frame.image, frame.timestamp)
                raw = frame.image.copy()
                drawn = self.pipeline.draw(frame.image, result)
                frames += 1
                people = sum(1 for t in result.tracks if not t.missed and t.label == "person")
                fps = frames / max(1e-6, time.monotonic() - started)
                self.raw, self.frame = raw, cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB)
                self.frame_no += 1
                self.status = (f"Recording **{self.camera}** · {fps:.1f} FPS · "
                               f"{people} person(s) in view")
            else:
                self.status = "The camera stopped sending frames."
        except Exception as e:  # shown on the page instead of a silent dead thread
            log.exception("Recorder failed")
            self.status = f"Error: {e}"
        finally:
            if self.pipeline is not None:
                self.pipeline.close()
            if not self.status.startswith(("Error", "The camera")):
                self.status = "Stopped."


def read_thumb(engine: SearchEngine, rel: str | None) -> np.ndarray | None:
    path = engine.store.media_path(rel) if rel else None
    image = cv2.imread(str(path)) if path else None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image is not None else None


def hhmm(iso: str) -> str:
    return iso.replace("T", " ")[11:16]


def build_ui(config: dict[str, Any], camera: str, source: str) -> gr.Blocks:
    engine = build_engine(config)
    toolbox = Toolbox(engine, cameras=config.get("cameras"), faces_config=config.get("faces"))
    recorder = Recorder(config, camera)
    models = list((config.get("chat") or {}).get("models") or {})
    chat: dict[str, Any] = {"backend": None, "model": None, "untold": []}
    cameras = sorted(set(engine.camera_names) | {camera})

    # ------------------------------------------------------------------ camera tab

    def start(src: str) -> tuple[str, gr.Timer]:
        recorder.start(src.strip() or source)
        return recorder.status, gr.Timer(active=True)

    def stop() -> tuple[str, gr.Timer]:
        recorder.stop()
        return recorder.status, gr.Timer(active=False)

    last_shown = {"frame": -1}

    def live() -> tuple[Any, str]:
        if recorder.frame is None or recorder.frame_no == last_shown["frame"]:
            return gr.skip(), recorder.status
        last_shown["frame"] = recorder.frame_no
        return recorder.frame, recorder.status

    def stats() -> tuple[str, list[tuple[np.ndarray, str]]]:
        counts = engine.store.counts()
        lines = [f"Stored: **{counts['faces']}** face prints, {counts['sightings']} object "
                 f"sightings, {counts['events']} alerts"]
        indexer = recorder.pipeline.indexer if recorder.pipeline is not None else None
        if indexer is not None and recorder.running:
            lines.append(f"Faces skipped as blurred / small / turned away: {indexer.faces_skipped}")
            if indexer.faces is None:
                lines.append("Face recording is **off** for this camera (faces.cameras).")
            if indexer.embedder is None:
                lines.append("No CLIP model: visual search by description is off.")
        gallery = []
        for f in engine.store.latest_faces(12):
            image = read_thumb(engine, f["thumb"])
            if image is not None:
                when = datetime.fromtimestamp(f["time"]).strftime("%H:%M:%S")
                gallery.append((image, f"{f['camera']} #{f['track_id']} {when} "
                                       f"q={f['quality'] or 0:.2f}"))
        return "\n\n".join(lines), gallery

    # ------------------------------------------------------------------ find a person

    def snapshot() -> Any:
        if recorder.raw is None:
            gr.Warning("Start the camera first.")
            return gr.skip()
        return cv2.cvtColor(recorder.raw, cv2.COLOR_BGR2RGB)

    def find(photo: np.ndarray | None, face_no: float | None, when: str,
             cam: str) -> tuple[str, list[tuple[np.ndarray, str]], dict[str, Any]]:
        if engine.faces is None:
            return "Face search is off (`faces.enabled` in the config).", [], {}
        if photo is None:
            return "Upload a photo (or take one from the live camera) first.", [], {}
        result = engine.face(cv2.cvtColor(photo, cv2.COLOR_RGB2BGR), when=when.strip() or None,
                             cameras=None if cam == ALL_CAMERAS else [cam],
                             choice=int(face_no) if face_no else None)
        gallery = []
        for r in result["results"]:
            image = read_thumb(engine, r["thumb"])
            if image is not None:
                gallery.append((image, f"{r['level']} · {r['score']:.2f} · {r['camera']} "
                                       f"{hhmm(r['first_seen'])}-{hhmm(r['last_seen'])} · "
                                       f"{r['frames_matched']}/{r['frames_checked']} frames"))
        details = {"time range": result["query"]["when"], "cameras": result["query"]["cameras"],
                   "faces in photo": result.get("faces_in_photo"),
                   "faces recorded (checked)": result.get("checked"),
                   "visits": result["results"]}
        return f"### {result['answer']}", gallery, details

    # ------------------------------------------------------------------ ask

    def new_backend(model: str) -> Backend:
        chat["backend"] = create_backend(model_config(config, model), system_prompt(toolbox),
                                         toolbox.tools)
        chat["model"] = model
        chat["untold"] = list(toolbox.photos)
        return chat["backend"]

    def reset(model: str) -> tuple[list[dict[str, Any]], list[Any]]:
        chat["backend"] = None
        chat["model"] = model
        return [], []

    def attach(photo: np.ndarray | None, face_no: float | None) -> str:
        if photo is None:
            return "Choose a photo first."
        name, message = toolbox.add_photo([cv2.cvtColor(photo, cv2.COLOR_RGB2BGR)],
                                          int(face_no) if face_no else None)
        if name:
            chat["untold"].append(name)
            return f"{message}. Now ask about **{name}**, e.g. *did this person come in today?*"
        return message

    def send(message: str, history: list[dict[str, Any]],
             model: str) -> tuple[str, list[dict[str, Any]], list[Any]]:
        message = message.strip()
        if not message:
            return "", history, gr.skip()
        history = [*history, {"role": "user", "content": message}]
        calls: list[str] = []

        class Collect(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                calls.append(record.getMessage().strip().removeprefix("-> "))

        tool_log = logging.getLogger("argus.ui.tools")
        tool_log.propagate = False
        tool_log.setLevel(logging.INFO)
        handler = Collect()
        tool_log.addHandler(handler)
        question = message
        try:
            backend = chat["backend"] if chat["model"] == model and chat["backend"] else \
                new_backend(model)
            if chat["untold"]:
                question = f"[Attached for face search: {', '.join(chat['untold'])}]\n{message}"
                chat["untold"] = []
            toolbox.evidence.clear()
            reply = answer(backend, toolbox, question, tool_log)
        except (RuntimeError, SystemExit) as e:  # model server down, missing SDK or key
            reply = f"Error: {e}"
        finally:
            tool_log.removeHandler(handler)
        if calls:
            history.append({"role": "assistant", "content": "\n".join(calls),
                            "metadata": {"title": f"Looked up ({len(calls)})"}})
        history.append({"role": "assistant", "content": reply})
        evidence = [img for rel in dict.fromkeys(toolbox.evidence)
                    if (img := read_thumb(engine, rel)) is not None]
        return "", history, evidence

    # ------------------------------------------------------------------ layout

    # analytics off: Gradio would otherwise report usage to its servers
    with gr.Blocks(title="Argus test", analytics_enabled=False) as demo:
        gr.Markdown(f"# Argus test\nStore: `{engine.store.path}` · camera **{camera}** · "
                    f"face search {'on' if engine.faces is not None else '**off**'}")
        with gr.Tab("Camera"):
            with gr.Row():
                source_box = gr.Textbox(source, label="Source (webcam index, file or URL)",
                                        scale=3)
                start_btn = gr.Button("Start recording", variant="primary", scale=1)
                stop_btn = gr.Button("Stop", scale=1)
            status = gr.Markdown(recorder.status)
            with gr.Row():
                view = gr.Image(label="Live", interactive=False, height=480, scale=3)
                with gr.Column(scale=2):
                    stats_md = gr.Markdown()
                    faces_gallery = gr.Gallery(label="Faces stored (newest first)", columns=4,
                                               height=360, object_fit="contain")
            live_timer = gr.Timer(0.2, active=False)
            stats_timer = gr.Timer(2.0)
            start_btn.click(start, source_box, [status, live_timer])
            stop_btn.click(stop, None, [status, live_timer])
            live_timer.tick(live, None, [view, status], show_progress="hidden")
            stats_timer.tick(stats, None, [stats_md, faces_gallery], show_progress="hidden")
            demo.load(stats, None, [stats_md, faces_gallery])

        with gr.Tab("Find a person"):
            with gr.Row():
                with gr.Column(scale=2):
                    photo = gr.Image(label="Photo of the person", type="numpy",
                                     sources=["upload", "clipboard"], height=300)
                    snap_btn = gr.Button("Take photo from the live camera")
                    with gr.Row():
                        when = gr.Textbox("today", label="When",
                                          info="today · last 10 minutes · today 1pm to 4pm")
                        cam = gr.Dropdown([ALL_CAMERAS, *cameras], value=ALL_CAMERAS,
                                          label="Camera")
                    face_no = gr.Number(0, label="Face in the photo (0 = automatic, "
                                                 "1 = leftmost)", precision=0)
                    find_btn = gr.Button("Search", variant="primary")
                with gr.Column(scale=3):
                    answer_md = gr.Markdown()
                    results = gr.Gallery(label="Matching visits (best face of each)",
                                         columns=4, height=300, object_fit="contain")
                    details = gr.JSON(label="Details")
            snap_btn.click(snapshot, None, photo)
            find_btn.click(find, [photo, face_no, when, cam], [answer_md, results, details])

        with gr.Tab("Ask"):
            with gr.Row():
                with gr.Column(scale=1):
                    model = gr.Dropdown(models, value=(config.get("chat") or {}).get(
                        "model", models[0] if models else None), label="Model")
                    chat_photo = gr.Image(label="Photo to look for (optional)", type="numpy",
                                          sources=["upload", "clipboard"], height=220)
                    chat_face_no = gr.Number(0, label="Face in the photo (0 = automatic)",
                                             precision=0)
                    attach_btn = gr.Button("Attach photo")
                    attach_md = gr.Markdown()
                    clear_btn = gr.Button("New conversation")
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(label="Argus", height=460)
                    question = gr.Textbox(placeholder="did this person come in between 1 and 4? · "
                                                      "how many people today?",
                                          label="Question", submit_btn=True)
                    evidence = gr.Gallery(label="Evidence", columns=6, height=160,
                                          object_fit="contain")
            attach_btn.click(attach, [chat_photo, chat_face_no], attach_md)
            question.submit(send, [question, chatbot, model], [question, chatbot, evidence])
            clear_btn.click(reset, model, [chatbot, evidence])
            model.change(reset, model, [chatbot, evidence])

    demo.recorder = recorder  # type: ignore[attr-defined]  # stopped on exit by main()
    demo.toolbox = toolbox    # type: ignore[attr-defined]
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Argus test UI")
    parser.add_argument("--config", type=Path, default=Path("configs/laptop.yaml"))
    parser.add_argument("--camera", help="Camera name to record as (default: first in config)")
    parser.add_argument("--source", help="Webcam index, file or URL (default: config `source`)")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true", help="Don't open a browser tab")
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = yaml.safe_load(args.config.read_text())
    camera = args.camera or next(iter(config.get("cameras") or {}), "camera")
    source = args.source if args.source is not None else str(config.get("source", 0))
    demo = build_ui(config, camera, source)
    try:
        demo.launch(server_name="127.0.0.1", server_port=args.port,
                    inbrowser=not args.no_browser)
    finally:
        demo.recorder.stop()  # type: ignore[attr-defined]
        demo.toolbox.close()  # type: ignore[attr-defined]


if __name__ == "__main__":
    main()
