"""Test UI: record a camera (a laptop's webcam is enough), then look for a person by photo or ask.

    uv run --group ui -m histor.ui                                 # configs/laptop.yaml, webcam 0
    uv run --group ui -m histor.ui --config configs/default.yaml --camera gate

Pages (one top bar, today.py):
    Today           the home page (/): the day so far, alerts to look at, each camera's
                    activity, the journal of events, and the chat beside it (today.py)
    Cameras         record the webcam, a camera URL, or videos / photos played as a simulated
                    camera; modules on or off per recording; live view, counts, latest events
    Find a person   photo + time range -> visits (likely / possible), no LLM needed
    Ask             the chat (needs Ollama or Claude, see chat.models), with photo attachments
    Reports         the saved periodic reports (`reports:` in the config), and "make one now"
    Test bench      the engine on its own: models, thresholds, zones, faces (Developer on)

Today is plain HTML + JSON (FastAPI); the other pages are Gradio, under /app.

One recorder and one conversation for the whole page: a local test tool, not a multi-user app.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import os
import shutil
import socket
import tempfile
import threading
import time
import urllib.request
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

import cv2
import gradio as gr
import numpy as np
import uvicorn
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI
from histor.engine import app as bench  # the engine's test bench
from histor.ask import build_engine
from histor.chat import answer, model_config, system_prompt
from histor.llm import Backend, create_backend, ollama_root
from histor.engine.cameras import is_stream_url
from histor.engine.pipeline import Pipeline
from histor.report import CUSTOM, Reporter, period_text, report_text
from histor.engine.search import SearchEngine
from histor.engine.store import EventStore
from histor.engine.stream import IMAGE_EXTENSIONS, iter_frames
from histor.engine.theme import CSS, THEME
from histor.today import APP, TOPBAR_CSS, TOPBAR_JS, Today, add_today_routes, topbar_html
from histor.tools import Toolbox

log = logging.getLogger("histor.ui")
ALL_CAMERAS = "all cameras"
WHEN_CHOICES = ["last 10 minutes", "last hour", "today", "this morning", "yesterday"]
EXAMPLES = ["What happened in the last hour?", "How many people came in today?",
            "Was anyone carrying a bag this morning?", "Did any alert go off last night?"]
BRAND = Path(__file__).resolve().parents[4] / "Images" / "histor" / "web"  # repo's Histor kit

# Modules switched on or off per recording (Camera tab): label -> config section.
MODULES = {"Faces": "faces", "Pose: person down, hands up": "pose", "Number plates": "plates",
           "Sounds: alarms, glass, shots, screams": "sounds", "Camera tamper": "tamper",
           "Bags left behind": "abandoned", "Fire & smoke": "fire"}
NOT_YET = "Weapons: not offered (DOCUMENTATION.md, 46)"
# Gradio pages: the shared top bar fixed above the page, Gradio's own tab buttons hidden. In
# <head>, not Gradio's `css` (which it scopes to its container: body.hx-dev would not match).
APP_HEAD = ('<link rel="icon" href="/favicon.svg" type="image/svg+xml"><style>' + TOPBAR_CSS + """
.hx-topcell {height: 0; min-height: 0 !important; padding: 0 !important; border: 0 !important;
  overflow: visible !important}
.hx-topcell .hx-top {position: fixed; top: 0; left: 0; right: 0; z-index: 1000}
html body .gradio-container {padding-top: 76px !important}
.ag-nav > .tab-wrapper {position: absolute !important; left: -10000px; width: 1200px}
</style><script>""" + TOPBAR_JS + "</script>")
TABS = ("cameras", "find", "ask", "reports", "bench")  # ids, as in today.NAV's links
IMAGE_FPS, IMAGE_SECONDS = 5, 10   # a photo is played as a still camera for 10 s
VIDEO_TYPES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".ts"}


def modules_on(config: dict[str, Any]) -> list[str]:
    """The modules the config turns on (the Camera tab's default ticks)."""
    return [label for label, section in MODULES.items()
            if (config.get(section) or {}).get("enabled")]


def with_modules(config: dict[str, Any], chosen: list[str] | None) -> dict[str, Any]:
    """A copy of the config with only the chosen modules on, each for every camera (a module
    ticked here is meant for the camera being recorded, whatever its `cameras:` list says)."""
    if chosen is None:
        return config
    config = dict(config)
    for label, section in MODULES.items():
        cfg = dict(config.get(section) or {})
        cfg["enabled"] = label in chosen
        if cfg["enabled"]:
            cfg["cameras"] = []
        config[section] = cfg
    return config


def results_only(config: dict[str, Any], folder: Path) -> dict[str, Any]:
    """A copy of the config that keeps nothing: its own store in `folder` (thrown away later)
    and no alert channels (no email, webhook or Telegram message for a test video)."""
    config = dict(config)
    config["store"] = {**(config.get("store") or {}), "path": str(folder / "results.db"),
                       "media_dir": str(folder / "media")}
    config["alerts"] = {**(config.get("alerts") or {}), "channels": [], "webhook": None,
                        "email": None, "jsonl": None}
    return config


def is_image(path: str) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def file_frames(files: list[str], real_speed: bool, loop: bool,
                stop: threading.Event) -> Iterator[tuple[np.ndarray, float, str]]:
    """Frames of videos and photos played one after the other as one camera: (BGR image,
    seconds since the start, file name). real_speed: as fast as a live camera would send them
    (else as fast as the CPU goes); loop: start again after the last file."""
    started, offset = time.monotonic(), 0.0
    while True:
        for path in files:
            name, last, step = Path(path).name, 0.0, 1 / IMAGE_FPS
            if is_image(path):
                image = cv2.imread(path)
                if image is None:
                    log.warning("Skipped %s: not a readable image", name)
                    continue
                frames: Iterator[tuple[np.ndarray, float, float]] = (
                    (image.copy(), i / IMAGE_FPS, IMAGE_FPS)
                    for i in range(IMAGE_FPS * IMAGE_SECONDS))
            else:
                frames = ((f.image, f.timestamp, f.fps) for f in iter_frames(path))
            for image, t, fps in frames:
                if stop.is_set():
                    return
                last, step = t, 1 / (fps or 25.0)
                if real_speed and (wait := started + offset + t - time.monotonic()) > 0:
                    if stop.wait(wait):
                        return
                yield image, offset + t, name
            offset += last + step
        if not loop:
            return


class Recorder:
    """The recording loop (detect, track, index faces) on a background thread. The page polls
    `frame` for the live view. The source is a camera, or videos and photos played as one
    (a simulated camera: same pipeline, same store, under the name given)."""

    def __init__(self, config: dict[str, Any], camera: str) -> None:
        self.config = config
        self.default_camera = camera
        self.camera = camera
        self.frame: np.ndarray | None = None      # latest annotated frame, RGB
        self.raw: np.ndarray | None = None        # same frame without drawings, BGR
        self.frame_no = 0
        self.status = "Stopped."
        self.pipeline: Pipeline | None = None
        self.results: EventStore | None = None  # a results-only run's own store (else None)
        self._results_dir: Path | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, source: str, files: list[str] | None = None, camera: str | None = None,
              modules: list[str] | None = None, real_speed: bool = True,
              loop: bool = False, save: bool = True) -> None:
        """files: videos / photos to play instead of `source`; camera: the name they are
        recorded under; modules: MODULES labels to run (None: as the config says); save:
        False = files analysed only, in a store of their own (shown, then thrown away)."""
        if self.running:
            return
        self._stop.clear()
        self.discard()  # the previous results-only run
        self.camera = (camera or "").strip() or self.default_camera
        self.status = f"Opening {Path(files[0]).name if files else source}..."
        config = with_modules(self.config, modules)
        if files and not save:
            self._results_dir = Path(tempfile.mkdtemp(prefix="histor-results-"))
            config = results_only(config, self._results_dir)
            self.results = EventStore.from_config(config["store"])
        self._thread = threading.Thread(
            target=self._run, args=(source, files or [], config, real_speed, loop), daemon=True)
        self._thread.start()

    def discard(self) -> None:
        """Throw away a results-only run's store."""
        if self.results is not None:
            self.results.close()
            self.results = None
        if self._results_dir is not None:
            shutil.rmtree(self._results_dir, ignore_errors=True)
            self._results_dir = None

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=60)  # the indexer finishes its queue first

    def _run(self, source: str, files: list[str], config: dict[str, Any], real_speed: bool,
             loop: bool) -> None:
        try:
            if files:
                # Files: every indexing step runs (live=False), and sound is read from the file
                # itself (one video, played once: its audio matches the picture)
                videos = [f for f in files if not is_image(f)]
                audio = videos[0] if len(files) == 1 and videos and not loop else None
                self.pipeline = Pipeline.from_config(config, source=self.camera, live=False,
                                                     audio_source=audio)
                frames = file_frames(files, real_speed, loop, self._stop)
            else:
                self.pipeline = Pipeline.from_config(
                    config, source=self.camera, live=True,
                    audio_source=source if is_stream_url(source) else None)
                frames = ((f.image, f.timestamp, "") for f in iter_frames(source))
            started, count = time.monotonic(), 0
            for image, timestamp, name in frames:
                if self._stop.is_set():
                    break
                result = self.pipeline.process(image, timestamp)
                raw = image.copy()
                drawn = self.pipeline.draw(image, result)
                count += 1
                people = sum(1 for t in result.tracks if not t.missed and t.label == "person")
                fps = count / max(1e-6, time.monotonic() - started)
                self.raw, self.frame = raw, cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB)
                self.frame_no += 1
                what = f"{self.camera} ← {name}" if name else self.camera
                doing = "Analysing (not saved)" if self.results is not None else "Recording"
                self.status = f"{doing} {what} · {fps:.1f} fps · {people} in view"
            else:
                self.status = ("Done: all files played." if files else
                               "The camera stopped sending frames.")
        except Exception as e:  # shown on the page instead of a silent dead thread
            log.exception("Recorder failed")
            self.status = f"Error: {e}"
        finally:
            if self.pipeline is not None:
                self.pipeline.close()
                if self.pipeline.store is not None:  # flushed: its connection can go
                    self.pipeline.store.close()
            if not self.status.startswith(("Error", "The camera", "Done")):
                self.status = "Stopped."


def read_thumb(engine: SearchEngine, rel: str | None) -> np.ndarray | None:
    path = engine.store.media_path(rel) if rel else None
    image = cv2.imread(str(path)) if path else None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image is not None else None


def thumb_caption(rel: str) -> str:
    """"laptop 11:55:12" from <camera>_<ms>_<track>.jpg (thumbs) or <camera>_<ms>.jpg (scenes);
    else the file name."""
    stem = Path(rel).stem
    for parts in (stem.rsplit("_", 2), stem.rsplit("_", 1)):
        if len(parts) >= 2 and parts[1].isdigit() and len(parts[1]) >= 12:  # ms since 1970
            return f"{parts[0]} {datetime.fromtimestamp(int(parts[1]) / 1000):%H:%M:%S}"
    return Path(rel).name


def hhmm(iso: str) -> str:
    return iso.replace("T", " ")[11:16]


def card_head(title: str, sub: str = "") -> None:
    """A card's title and one line of help (inside a gr.Group with elem_classes="ag-panel")."""
    gr.HTML(f'<div class="ag-panel-title">{html.escape(title)}</div>'
            + (f'<div class="ag-panel-sub">{sub}</div>' if sub else ""), padding=False,
            elem_classes="ag-panel-head")


def box(kind: str, title: str, body: str = "") -> str:
    """A small callout; `title` is escaped, `body` is trusted HTML."""
    return (f'<div class="ag-box {kind}"><b>{html.escape(title)}</b>'
            f'{f"<p>{body}</p>" if body else ""}</div>')


def server_models(base_url: str, api_key: str | None, timeout: float = 1.5) -> list[str]:
    """Model ids an OpenAI-compatible server has (Ollama: the pulled ones). Raises if down."""
    request = urllib.request.Request(base_url.rstrip("/") + "/models")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return [m["id"] for m in json.loads(response.read()).get("data", [])]


def brand_logo() -> str:
    """The Histor eye as inline SVG (Images/histor), or nothing."""
    path = BRAND / "favicon.svg"
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def model_state(config: dict[str, Any], name: str | None) -> tuple[str, str, str]:
    """Where the chat model runs and whether it can answer right now: (ok | warn | err, title,
    body as HTML)."""
    if not name:
        return "err", "No chat model configured", "Add one under <code>chat.models</code>."
    try:
        cfg = model_config(config, name)
    except SystemExit as e:
        return "err", str(e), ""
    note = html.escape(cfg.get("note") or cfg.get("model", ""))
    if cfg.get("api") == "anthropic":
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return "err", f"{name}: SDK missing", (f"{note}<br>Run <code>uv sync --group claude"
                                                  "</code>, then restart.")
        key = "" if cfg.get("api_key") or os.environ.get("ANTHROPIC_API_KEY") else             "<br>No <code>ANTHROPIC_API_KEY</code> in <code>.env</code> (a saved login also works)."
        return "warn", f"{name}: cloud model", (f"{note}<br>Only text search results are sent "
                                                f"(times, cameras, labels), never images or "
                                                f"video.{key}")
    model = cfg.get("model", "")
    base_url = cfg.get("base_url", "http://127.0.0.1:11434/v1")
    host = urlparse(base_url).hostname or ""
    where = "runs on this machine" if host in ("127.0.0.1", "localhost", "::1") else         f"runs on {html.escape(host)}"
    try:  # Ollama lists its models on the OpenAI-style endpoint too
        available = server_models(ollama_root(base_url) + "/v1" if cfg.get("api") == "ollama"
                                  else base_url, cfg.get("api_key"))
    except Exception:
        return "err", f"{name}: model server not reachable", (
            f"{note} · {where}<br>Nothing answers at <code>{html.escape(base_url)}</code>. "
            f"For Ollama: <code>ollama serve</code>, then <code>ollama pull "
            f"{html.escape(model)}</code>.")
    if model not in available and f"{model}:latest" not in available:
        return "warn", f"{name}: model not downloaded", (
            f"{note} · {where}<br>Run <code>ollama pull {html.escape(model)}</code> "
            "(or load it in your model server).")
    return "ok", f"{name}: ready", f"{note} · {where}. Footage never leaves the site."


def model_card(config: dict[str, Any], name: str | None) -> str:
    return box(*model_state(config, name))


def build_ui(config: dict[str, Any], camera: str, source: str) -> gr.Blocks:
    engine = build_engine(config)
    toolbox = Toolbox.from_config(engine, config)
    recorder = Recorder(config, camera)
    models = list((config.get("chat") or {}).get("models") or {})
    chat: dict[str, Any] = {"backend": None, "model": None, "untold": []}
    cameras = sorted(set(engine.camera_names) | {camera})

    # ------------------------------------------------------------------ camera tab

    def status_html() -> str:
        text = recorder.status
        kind = "rec" if recorder.running and text.startswith("Recording") else \
            "busy" if recorder.running else "error" if text.startswith(("Error", "The camera")) \
            else ""
        label = {"rec": "<b>REC</b> ", "busy": "", "error": "", "": ""}[kind]
        # REC and the header say what and where: the pill keeps the figures
        shown = text.split(" · ", 1)[-1] if kind == "rec" else text
        return (f'<div class="ag-status {kind}" title="{html.escape(text)}">'
                f'<span class="ag-dot"></span><span>{label}{html.escape(shown)}</span></div>')

    def buttons() -> tuple[Any, Any]:
        return (gr.Button(interactive=not recorder.running),
                gr.Button(interactive=recorder.running))

    def start(src: str, files: list[str] | None, sim_name: str, modules: list[str],
              real_speed: bool, loop: bool, keep: str) -> tuple[Any, ...]:
        files = [f for f in files or [] if is_image(f) or Path(f).suffix.lower() in VIDEO_TYPES]
        recorder.start(src.strip() or source, files, sim_name if files else None, modules,
                       real_speed, loop, save=keep != "results")
        return status_html(), gr.Timer(active=True), gr.Button(interactive=False), \
            gr.Button(interactive=True)

    def stop() -> tuple[Any, ...]:
        recorder.stop()
        return status_html(), gr.Timer(active=False), *buttons()

    last_shown = {"frame": -1}

    def live() -> tuple[Any, ...]:
        if not recorder.running:  # stopped by itself: camera gone, error, end of file
            return gr.skip(), status_html(), gr.Timer(active=False), *buttons()
        if recorder.frame is None or recorder.frame_no == last_shown["frame"]:
            return gr.skip(), status_html(), gr.skip(), gr.skip(), gr.skip()
        last_shown["frame"] = recorder.frame_no
        return recorder.frame, status_html(), gr.skip(), gr.skip(), gr.skip()

    def running_modules() -> str:
        """Chips: which modules this recording really runs (a missing model turns one off)."""
        p = recorder.pipeline
        if p is None or not recorder.running:
            return ""
        indexer = p.indexer
        on = [("Faces", bool(indexer and indexer.faces)),
              ("Search", bool(indexer and indexer.embedder)), ("Pose", p.pose is not None),
              ("Plates", p.plates is not None), ("Sounds", p.sounds is not None),
              ("Tamper", p.tamper is not None), ("Bags left", p.abandoned is not None),
              ("Fire", p.fire is not None)]
        return '<div class="ag-chips ag-modules">' + "".join(
            f'<span class="ag-chip {"on" if ok else ""}">{n}</span>' for n, ok in on) + "</div>"

    def shown_store() -> EventStore:
        """What the counts and latest events show: a results-only run's own store, else the
        recordings."""
        return recorder.results or engine.store

    def recent_html() -> str:
        """The latest alerts and plate reads, newest first: what the modules found."""
        rows = []
        store = shown_store()
        for e in store.list_events(limit=6):
            what = " · ".join(str(e[k]) for k in ("type", "rule", "label") if e.get(k))
            rows.append(f"<li><b>{e['iso_time'][11:19]}</b> {html.escape(e['camera'])} · "
                        f"{html.escape(what)}</li>")
        plates = store.plate_rows(since=time.time() - 24 * 3600)[-4:][::-1]
        rows += [f"<li><b>{datetime.fromtimestamp(p['time']):%H:%M:%S}</b> "
                 f"{html.escape(p['camera'])} · plate <code>{html.escape(p['text'])}</code>"
                 f" ({html.escape(p.get('label') or '?')})</li>" for p in plates]
        if not rows:
            return '<div class="ag-recent"><span>Latest events</span><p>Nothing yet.</p></div>'
        return f'<div class="ag-recent"><span>Latest events</span><ul>{"".join(rows)}</ul></div>'

    def stats() -> str:
        counts = shown_store().counts()
        cards = [("Sightings", counts["sightings"]), ("Faces", counts["faces"]),
                 ("Poses", counts["poses"]), ("Plates", counts["plates"]),
                 ("Alerts", counts["events"])]
        notes = []
        indexer = recorder.pipeline.indexer if recorder.pipeline is not None else None
        if indexer is not None and recorder.running:
            cards.append(("Skipped", indexer.faces_skipped))
            notes.append("Skipped faces were blurred, too small or turned away.")
            if indexer.faces is None:
                notes.append("Face recording is <b>off</b> for this camera.")
            if indexer.embedder is None:
                notes.append("No CLIP model: search by description is off.")
        tiles = "".join(f'<div class="ag-card"><b>{n:,}</b><span>{label}</span></div>'
                        for label, n in cards)
        if recorder.results is not None:
            notes.insert(0, "<b>Results only:</b> these counts are this video's, and nothing "
                            "is saved to the recordings. They go when you press Start again.")
        page = running_modules() + f'<div class="ag-cards">{tiles}</div>'
        page += (f'<ul class="ag-notes">{"".join(f"<li>{n}</li>" for n in notes)}</ul>'
                 if notes else "")
        return page + recent_html()

    def wipe() -> tuple[Any, ...]:
        """Delete every row, JPEG and saved report (after "Yes, delete all")."""
        hide = (gr.update(visible=True), gr.update(visible=False))  # the button back, no ask
        if recorder.running:
            return (box("warn", "Stop recording first",
                        "Press <b>Stop</b>, then delete."), *hide)
        deleted = engine.store.clear()
        reports = 0
        if reporter is not None:
            for path in reporter.saved():
                path.unlink(missing_ok=True)
                path.with_suffix(".md").unlink(missing_ok=True)
                reports += 1
        what = ", ".join(f"{n:,} {name}" for name, n in (
            ("faces", deleted["faces"]), ("sightings", deleted["sightings"]),
            ("poses", deleted["poses"]), ("plates", deleted["plates"]),
            ("alerts", deleted["events"]), ("reports", reports)))
        log.info("Deleted all saved data: %s", what)
        return box("ok", "All saved data deleted", html.escape(what) + "."), *hide

    # ------------------------------------------------------------------ find a person

    def snapshot(src: str) -> Any:
        if recorder.running and recorder.raw is not None:
            return cv2.cvtColor(recorder.raw, cv2.COLOR_BGR2RGB)
        # not recording: open the camera just for this photo
        src = (src or "").strip() or source
        frames = iter_frames(src)
        try:
            # a webcam's first frames are dark while its exposure settles
            for i, frame in enumerate(frames):
                if i >= (10 if src.isdigit() else 0):
                    return cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            gr.Warning(f"Could not read the camera: {e}")
            return gr.skip()
        finally:
            frames.close()  # releases the camera
        gr.Warning("The camera sent no frame.")
        return gr.skip()

    def find(photo: np.ndarray | None, face_no: float | None, when: str,
             cam: str) -> tuple[str, list[tuple[np.ndarray, str]], dict[str, Any]]:
        if engine.faces is None:
            return box("err", "Face search is off", "<code>faces.enabled</code> in the config."), \
                [], {}
        if photo is None:
            return box("info", "Upload a photo (or take one from the live camera) first."), [], {}
        result = engine.face(cv2.cvtColor(photo, cv2.COLOR_RGB2BGR),
                             when=(when or "").strip() or None,
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
        kind = "ok" if result["results"] else "warn"
        checked = sum((result.get("checked") or {}).values())  # per camera
        body = f'<span class="ag-muted">Checked {checked} recorded face(s).</span>' \
            if result.get("face_found") else ""
        return (f'<div class="ag-box {kind}"><span class="ag-answer">'
                f'{html.escape(result["answer"])}</span>{f"<p>{body}</p>" if body else ""}</div>',
                gallery, details)

    # ------------------------------------------------------------------ reports

    try:
        reporter: Reporter | None = Reporter(config, engine)
        reporter_error = ""
    except ValueError as e:  # a bad `reports:` entry must not stop the rest of the UI
        reporter, reporter_error = None, str(e)
    report_names = [*reporter.specs, CUSTOM] if reporter else []  # custom: any range

    def report_periods(name: str | None, select: str | None = None) -> Any:
        """Saved reports of `name`, newest first; select: the one just made."""
        choices = []
        for p in (reporter.saved(name) if reporter and name else [])[:200]:
            try:
                r = reporter.load(p)
                if r.get("custom") and name != CUSTOM:  # ranges are listed under custom only
                    continue
                label = period_text(*(datetime.fromisoformat(r[k]) for k in ("since", "until")))
                choices.append((label + (" · custom" if r.get("custom") else ""), str(p)))
            except (OSError, ValueError, KeyError):  # a broken file must not hide the others
                continue
        paths = [c[1] for c in choices]
        value = select if select in paths else paths[0] if paths else None
        # nothing saved yet: no empty list, the report area says what Make will do
        return gr.update(choices=choices, value=value, visible=bool(choices))

    def no_report(name: str | None) -> str:
        """What the report area says before the first report of `name`."""
        if not reporter or not name:
            return ""
        if name == CUSTOM:
            return ("### No custom report yet\nPick *From* and *To*, then **Make it for this "
                    "range**.")
        spec = reporter.specs[name]
        start, end = spec.period(spec.period_end(datetime.now()))
        return (f"### No {name} report yet\nIt is made at the end of each period while "
                f"`histor.report` runs. **Make it now** writes the last one: "
                f"{period_text(start, end)}.")

    def pick_report(name: str | None, select: str | None = None) -> tuple[Any, ...]:
        """A report chosen: its saved periods, the newest shown (or what to do first), and
        the delete button when there is one to delete."""
        periods = report_periods(name, select)
        path = periods.get("value")
        md, gallery = show_report(path, name)
        return periods, md, gallery, gr.update(visible=bool(path)), gr.update(visible=False)

    def delete_report(path: str | None, name: str | None) -> tuple[Any, ...]:
        """The report shown, deleted (after "Yes, delete it"); the next newest is shown."""
        note = ""
        if reporter and path:
            try:
                reporter.delete(path)
                note = box("ok", "Report deleted")
            except (OSError, ValueError) as e:
                note = box("err", "Not deleted", html.escape(str(e)))
        return (*pick_report(name), note)

    def show_report(path: str | None, name: str | None = None) -> tuple[str, Any]:
        if not reporter or not path or not Path(path).is_file():
            return no_report(name), gr.update(value=[], visible=False)
        r = reporter.load(Path(path))
        thumbs = [(img, thumb_caption(rel)) for rel in r["evidence"]
                  if (img := read_thumb(engine, rel)) is not None]
        sent = "".join(f"\n- sent to {k}: {v}" for k, v in r.get("sent", {}).items())
        return report_text(r, evidence=False) + sent, gr.update(value=thumbs,
                                                                 visible=bool(thumbs))

    def send_option(name: str | None) -> Any:
        """Sending only makes sense if the report has channels."""
        channels = reporter.specs[name].channels if reporter and name in reporter.specs else []
        return gr.update(value=False, interactive=bool(channels),
                         label="Send to its channels too" if channels else
                         "Send: no channels set for this report")

    def report_mode(name: str | None) -> tuple[Any, Any]:
        """Which button: "last period" for a period report, "this range" for custom."""
        custom = name == CUSTOM
        return gr.update(visible=not custom), gr.update(visible=custom)

    def make_report(name: str | None, send: bool, since: float | None = None,
                    until: float | None = None) -> tuple[str, str | None]:
        """The last period, or the range since-until (timestamps from the date pickers)."""
        if not reporter or not name:
            return "", None
        start = end = None
        if since is not None:
            start = datetime.fromtimestamp(since)
            end = datetime.fromtimestamp(until) if until is not None else datetime.now()
            if start >= end:
                return box("err", "Check the range", "<i>From</i> must be before <i>To</i>."), None
        r = reporter.make(name, end, send=send, start=start)
        note = box("err", "No written summary", html.escape(r["error"])) if r["error"] else ""
        return note, r["path"]

    def default_range() -> tuple[datetime, datetime]:
        """Custom range preset: today so far."""
        now = datetime.now().replace(second=0, microsecond=0)
        return now.replace(hour=0, minute=0), now

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
            return box("info", "Choose a photo first.")
        name, message = toolbox.add_photo([cv2.cvtColor(photo, cv2.COLOR_RGB2BGR)],
                                          int(face_no) if face_no else None)
        if name:
            chat["untold"].append(name)
            return box("ok", f"Attached as {name}", f"{html.escape(message)}. Now ask about it, "
                       "e.g. <i>did this person come in today?</i>")
        return box("warn", "Photo not usable", html.escape(message))

    def post(message: str, history: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        """Show the question at once; `send` answers it."""
        message = message.strip()
        if not message:
            return "", history
        return "", [*history, {"role": "user", "content": message}]

    def send(history: list[dict[str, Any]], model: str) -> tuple[list[dict[str, Any]], list[Any]]:
        if not history or history[-1]["role"] != "user":  # empty question: nothing posted
            return gr.skip(), gr.skip()
        message = history[-1]["content"]
        if not isinstance(message, str):  # Gradio may hand back rich content
            message = str(message[0].get("text", "") if isinstance(message, list) and message
                          and isinstance(message[0], dict) else message)
        history = list(history)
        calls: list[str] = []

        class Collect(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                calls.append(record.getMessage().strip().removeprefix("-> "))

        tool_log = logging.getLogger("histor.ui.tools")
        tool_log.propagate = False
        tool_log.setLevel(logging.INFO)
        handler = Collect()
        tool_log.addHandler(handler)
        question = message
        started = time.monotonic()
        try:
            backend = chat["backend"] if chat["model"] == model and chat["backend"] else \
                new_backend(model)
            if chat["untold"]:
                question = f"[Attached for face search: {', '.join(chat['untold'])}]\n{message}"
                chat["untold"] = []
            toolbox.evidence.clear()
            reply = answer(backend, toolbox, question, tool_log)
        except (RuntimeError, SystemExit) as e:  # model server down, missing SDK or key
            reply = f"**Error:** {e}"
        finally:
            tool_log.removeHandler(handler)
        if calls:
            history.append({"role": "assistant", "content": "\n".join(f"`{c}`" for c in calls),
                            "metadata": {"title": f"Looked up ({len(calls)})", "status": "done",
                                         "duration": time.monotonic() - started}})
        history.append({"role": "assistant", "content": reply})
        evidence = [img for rel in dict.fromkeys(toolbox.evidence)
                    if (img := read_thumb(engine, rel)) is not None]
        return history, evidence

    # ------------------------------------------------------------------ layout

    logo = brand_logo()
    header = topbar_html("Cameras", logo)  # today.py's script marks the page shown
    avatar = BRAND / "app-icon-192.png"
    default_model = (config.get("chat") or {}).get("model", models[0] if models else None)

    def field(label: str) -> None:
        """A small caps label in front of a control on the same line."""
        gr.HTML(f'<span class="ag-field">{label}</span>', scale=0, min_width=0,
                container=False, padding=False, elem_classes="ag-fieldcell")

    # analytics off: Gradio would otherwise report usage to its servers
    with gr.Blocks(title="Histor", analytics_enabled=False) as demo:
        gr.HTML(header, padding=False, elem_classes="hx-topcell")
        # the top bar switches these tabs (TOPBAR_JS); their own buttons are hidden
        with gr.Tabs(elem_classes="ag-nav") as nav:
            with gr.Tab("Cameras", id="cameras"):
                with gr.Row(elem_classes="ag-controls", equal_height=True):
                    field("Source")
                    source_box = gr.Textbox(source, show_label=False, container=False, scale=3,
                                            max_lines=1,
                                            placeholder="0 = this webcam, or a camera URL "
                                                        "(or simulate one with files below)")
                    start_btn = gr.Button("▶ Start", variant="primary", scale=0,
                                          min_width=150)
                    stop_btn = gr.Button("■ Stop", variant="stop", scale=0, interactive=False,
                                         min_width=84)
                    status = gr.HTML(status_html(), scale=4, min_width=220, container=False,
                                     padding=False)
                with gr.Row():
                    with gr.Column(scale=3, min_width=260, elem_classes="ag-side"):
                        # A simulated camera: test real scenarios without standing in front of
                        # the webcam. Same pipeline and store as a camera, under its own name.
                        with gr.Group(elem_classes="ag-panel"):
                            card_head("Simulate a camera",
                                      "Videos or photos, played one after the other. With files "
                                      "here, <b>Start</b> plays them instead of the source.")
                            sim_files = gr.File(label="Videos or photos", file_count="multiple",
                                                file_types=["video", "image"], height=104,
                                                show_label=False, elem_classes="ag-drop")
                            sim_keep = gr.Radio(
                                [("Add to the recordings", "save"),
                                 ("Just show the results", "results")], value="save",
                                label="The files are",
                                info="Added: searchable, in Today and the reports, like a "
                                     "camera. Results only: nothing saved, no alerts sent.")
                            sim_name = gr.Textbox("sim", label="Camera name", max_lines=1)
                            with gr.Row(elem_classes="ag-row-tight"):
                                sim_real = gr.Checkbox(True, label="Real speed", min_width=0,
                                                       info="as fast as a camera sends")
                                sim_loop = gr.Checkbox(False, label="Loop", min_width=0,
                                                       info="start over after the last "
                                                            "file, until Stop")
                        with gr.Group(elem_classes="ag-panel"):
                            modules_box = gr.CheckboxGroup(
                                list(MODULES), value=modules_on(config), label="Modules",
                                info="Applied when you press Start")
                            gr.HTML(f'<div class="ag-panel-sub">{html.escape(NOT_YET)}</div>',
                                    padding=False)
                    view = gr.Image(label="Camera view", interactive=False, height=520, scale=6,
                                    buttons=["fullscreen"], elem_classes="ag-screen")
                    with gr.Column(scale=4, min_width=300, elem_classes="ag-side"):
                        stats_html = gr.HTML(padding=False)
                        with gr.Group(elem_classes="ag-panel"):
                            card_head("Saved data", "Faces, sightings, poses, plates, alerts "
                                                    "and reports, in "
                                                    f"<code>{html.escape(str(engine.store.path))}"
                                                    "</code>.")
                            wipe_btn = gr.Button("Delete all saved data", variant="stop",
                                                 size="sm", elem_classes="ag-card-btn")
                            with gr.Column(visible=False) as wipe_ask:
                                gr.HTML(box("err", "Delete everything?",
                                            "Every face, sighting, pose, plate read, alert and "
                                            "report, with their pictures. This can't be undone."),
                                        padding=False, elem_classes="ag-card-note")
                                with gr.Row(elem_classes="ag-row-tight ag-card-btns"):
                                    wipe_yes = gr.Button("Yes, delete all", variant="stop",
                                                         size="sm", min_width=0)
                                    wipe_no = gr.Button("Cancel", variant="secondary", size="sm",
                                                        min_width=0)
                            wipe_html = gr.HTML(padding=False, elem_classes="ag-card-note")
                live_timer = gr.Timer(0.2, active=False)
                stats_timer = gr.Timer(2.0)
                start_btn.click(start, [source_box, sim_files, sim_name, modules_box, sim_real,
                                        sim_loop, sim_keep],
                                [status, live_timer, start_btn, stop_btn])
                stop_btn.click(stop, None, [status, live_timer, start_btn, stop_btn])
                live_timer.tick(live, None, [view, status, live_timer, start_btn, stop_btn],
                                show_progress="hidden")
                stats_timer.tick(stats, None, stats_html, show_progress="hidden")
                wipe_btn.click(lambda: ("", gr.update(visible=False), gr.update(visible=True)),
                               None, [wipe_html, wipe_btn, wipe_ask], show_progress="hidden")
                wipe_no.click(lambda: (gr.update(visible=True), gr.update(visible=False)),
                              None, [wipe_btn, wipe_ask], show_progress="hidden")
                wipe_yes.click(wipe, None, [wipe_html, wipe_btn, wipe_ask]).then(
                    stats, None, stats_html, show_progress="hidden")
                demo.load(stats, None, stats_html)

            with gr.Tab("Find a person", id="find"):
                with gr.Row():
                    with gr.Column(scale=4, min_width=300, elem_classes="ag-side"):
                        with gr.Group(elem_classes="ag-panel"):
                            card_head("Who are you looking for?",
                                      "A clear photo of their face: upload one, paste it, or "
                                      "take a frame from the camera.")
                            photo = gr.Image(label="Photo of the person", type="numpy",
                                             sources=["upload", "clipboard"], height=230,
                                             show_label=False, elem_classes="ag-drop")
                            snap_btn = gr.Button("Use a frame from the camera", size="sm",
                                                 variant="secondary", elem_classes="ag-card-btn")
                        with gr.Group(elem_classes="ag-panel"):
                            card_head("When and where",
                                      "Pick a range or type one, e.g. <i>today 1pm to 4pm</i>.")
                            with gr.Row(elem_classes="ag-row-tight"):
                                when = gr.Dropdown(WHEN_CHOICES, value="today", label="When",
                                                   allow_custom_value=True, scale=3,
                                                   min_width=120)
                                cam = gr.Dropdown([ALL_CAMERAS, *cameras], value=ALL_CAMERAS,
                                                  label="Camera", scale=3, min_width=110)
                            face_no = gr.Number(0, label="Face in the photo",
                                                info="0 finds it by itself; in a group photo, "
                                                     "1 is the leftmost", precision=0, minimum=0)
                        find_btn = gr.Button("Search", variant="primary")
                    with gr.Column(scale=7, elem_classes="ag-side"):
                        answer_html = gr.HTML(box(
                            "info", "Visits show here",
                            "Each visit with the best face seen, likely or possible. Face "
                            "search runs on this machine: no language model needed."),
                            padding=False)
                        results = gr.Gallery(label="Matching visits · best face of each", columns=5,
                                             height=330, object_fit="cover", preview=False,
                                             fit_columns=False)
                        with gr.Accordion("Details", open=False):
                            details = gr.JSON(show_label=False)
                snap_btn.click(snapshot, source_box, photo)
                find_btn.click(find, [photo, face_no, when, cam], [answer_html, results, details])

                def camera_choices(current: str) -> Any:
                    """The configured cameras and any other that recorded (a simulation)."""
                    names = sorted(set(cameras) | {c["camera"] for c in toolbox.coverage()})
                    return gr.update(choices=[ALL_CAMERAS, *names], value=current)

                cam.focus(camera_choices, cam, cam, show_progress="hidden")
                demo.load(camera_choices, cam, cam, show_progress="hidden")

                def photo_from_link(request: gr.Request) -> Any:
                    """Today's "Find this person": /app/?tab=find&photo=<stored face or
                    snapshot>."""
                    rel = request.query_params.get("photo") if request else None
                    image = read_thumb(engine, rel) if rel else None
                    return image if image is not None else gr.skip()

                demo.load(photo_from_link, None, photo, show_progress="hidden")

            with gr.Tab("Ask", id="ask"):
                with gr.Row():
                    with gr.Column(scale=3, min_width=260, elem_classes="ag-side"):
                        with gr.Group(elem_classes="ag-panel"):
                            card_head("Model", "Who answers: footage and photos stay here "
                                               "with a local model.")
                            model = gr.Dropdown(models, value=default_model, show_label=False)
                            model_html = gr.HTML(padding=False, elem_classes="ag-card-note")
                            with gr.Row(elem_classes="ag-row-tight ag-card-btns"):
                                check_btn = gr.Button("Check again", size="sm",
                                                      variant="secondary", min_width=0)
                                clear_btn = gr.Button("New conversation", size="sm",
                                                      variant="secondary", min_width=0)
                        with gr.Group(elem_classes="ag-panel"):
                            card_head("Look for a person",
                                      "Attach a photo, then ask about it: <i>did this person "
                                      "come in today?</i>")
                            chat_photo = gr.Image(label="Photo", type="numpy",
                                                  sources=["upload", "clipboard"], height=170,
                                                  show_label=False, elem_classes="ag-drop")
                            chat_snap_btn = gr.Button("Use a frame from the camera", size="sm",
                                                      variant="secondary",
                                                      elem_classes="ag-card-btn")
                            chat_face_no = gr.Number(0, label="Face in the photo",
                                                     info="0 finds it by itself; in a group "
                                                          "photo, 1 is the leftmost",
                                                     precision=0, minimum=0)
                            attach_html = gr.HTML(padding=False, elem_classes="ag-card-note")
                        attach_btn = gr.Button("Attach photo", variant="primary")
                    with gr.Column(scale=9, elem_classes="ag-side"):
                        chatbot = gr.Chatbot(
                            label="Histor", height=460, show_label=False,
                            avatar_images=(None, str(avatar) if avatar.is_file() else None),
                            placeholder="### Ask about what the cameras saw\nAnswers come from the "
                                        "recordings: the lookups made are shown above each answer, "
                                        "the thumbnails it looked at below.")
                        question = gr.Textbox(placeholder="How many people came in today?",
                                              show_label=False, submit_btn=True, autofocus=True)
                        with gr.Column(elem_classes="ag-examples"):
                            examples = gr.Examples(EXAMPLES, question, label="Try")
                        evidence = gr.Gallery(label="Evidence", columns=8, height=130,
                                              object_fit="cover", preview=False,
                                              fit_columns=False)
                chat_snap_btn.click(snapshot, source_box, chat_photo)
                attach_btn.click(attach, [chat_photo, chat_face_no], attach_html)
                # the question shows at once, the answer follows
                question.submit(post, [question, chatbot], [question, chatbot],
                                show_progress="hidden").then(send, [chatbot, model],
                                                             [chatbot, evidence])
                # a suggestion asks at once: focus stays on its button, so Enter would not send it
                examples.load_input_event.then(post, [question, chatbot], [question, chatbot],
                                               show_progress="hidden").then(
                    send, [chatbot, model], [chatbot, evidence])
                clear_btn.click(reset, model, [chatbot, evidence])
                model.change(reset, model, [chatbot, evidence])
                demo.load(reset, model, [chatbot, evidence])  # a fresh page: the model forgets too
                for trigger in (model.change, check_btn.click, demo.load):
                    trigger(lambda m: model_card(config, m), model, model_html)

            with gr.Tab("Reports", id="reports"):
                if not report_names:
                    gr.HTML(box("err" if reporter_error else "info",
                                reporter_error or "No reports configured",
                                "" if reporter_error else
                                "Add hourly, daily, weekly or monthly reports under <code>reports:</code> in "
                                "the config (cameras, what to watch for, where to send them), then "
                                "run <code>uv run -m histor.report</code> next to the recorders."),
                            padding=False)
                else:
                    with gr.Row():
                        with gr.Column(scale=3, min_width=260, elem_classes="ag-side"):
                            # not filterable: a typed-in filter would hide all but the
                            # selected report when the list opens
                            with gr.Group(elem_classes="ag-panel"):
                                card_head("Report", "The saved ones, newest first, and a new "
                                                    "one whenever you want.")
                                report_name = gr.Dropdown(report_names, value=report_names[0],
                                                          label="Report", filterable=False)
                                report_period = gr.Dropdown([], label="Period", filterable=False,
                                                            visible=False)
                                with gr.Column(visible=False) as range_group:  # custom only
                                    range_from = gr.DateTime(label="From", include_time=True)
                                    range_to = gr.DateTime(label="To", include_time=True)
                                del_btn = gr.Button("Delete this report", variant="stop",
                                                    size="sm", visible=False,
                                                    elem_classes="ag-card-btn")
                                with gr.Column(visible=False) as del_ask:
                                    gr.HTML(box("err", "Delete this report?",
                                                "The saved summary and counts of this period. "
                                                "The recordings stay."),
                                            padding=False, elem_classes="ag-card-note")
                                    with gr.Row(elem_classes="ag-row-tight ag-card-btns"):
                                        del_yes = gr.Button("Yes, delete it", variant="stop",
                                                            size="sm", min_width=0)
                                        del_no = gr.Button("Cancel", variant="secondary",
                                                           size="sm", min_width=0)
                                send_box = gr.Checkbox(False, label="Send to its channels too")
                            make_btn = gr.Button("Make it now · last period", variant="primary")
                            range_btn = gr.Button("Make it for this range", variant="primary",
                                                  visible=False)
                            make_html = gr.HTML(padding=False)
                            made = gr.State(None)  # path of the report just made, to select it
                        with gr.Column(scale=9, elem_classes="ag-side"):
                            report_md = gr.Markdown(container=True, padding=True,
                                                    elem_classes="ag-report")
                            report_gallery = gr.Gallery(label="Evidence", columns=8, height=130,
                                                        object_fit="cover", preview=False,
                                                        fit_columns=False, visible=False)
                    picked = [report_period, report_md, report_gallery, del_btn, del_ask]
                    report_name.change(pick_report, report_name, picked)
                    del_btn.click(lambda: (gr.update(visible=False), gr.update(visible=True)),
                                  None, [del_btn, del_ask], show_progress="hidden")
                    del_no.click(lambda: (gr.update(visible=True), gr.update(visible=False)),
                                 None, [del_btn, del_ask], show_progress="hidden")
                    del_yes.click(delete_report, [report_period, report_name],
                                  [*picked, make_html])
                    report_name.change(send_option, report_name, send_box, show_progress="hidden")
                    report_name.change(lambda n: (*report_mode(n), gr.update(visible=n == CUSTOM)),
                                       report_name, [make_btn, range_btn, range_group],
                                       show_progress="hidden")
                    report_period.change(show_report, [report_period, report_name],
                                         [report_md, report_gallery])
                    # one "processing" line, under the buttons; the period list updates quietly after
                    for made_event in (
                            make_btn.click(make_report, [report_name, send_box], [make_html, made]),
                            range_btn.click(make_report, [report_name, send_box, range_from,
                                                          range_to], [make_html, made])):
                        made_event.then(pick_report, [report_name, made],  # shown even if the
                                        picked, show_progress="hidden")  # same period made again
                    demo.load(pick_report, report_name, picked)
                    demo.load(default_range, None, [range_from, range_to], show_progress="hidden")
                    demo.load(send_option, report_name, send_box, show_progress="hidden")

            with gr.Tab("Test bench", id="bench"):
                gr.HTML(box("info", "The detection engine on its own",
                            "Models and thresholds on photos, videos and cameras, zones and lines, "
                            "faces side by side. Nothing here is recorded; stop recording before "
                            "using the Webcam tab (one webcam can't feed both)."), padding=False)
                bench.build_bench()

        def tab_from_link(request: gr.Request) -> Any:
            """/app/?tab=find: the page the top bar's link names."""
            tab = request.query_params.get("tab") if request else None
            return gr.Tabs(selected=tab) if tab in TABS else gr.skip()

        demo.load(tab_from_link, None, nav, show_progress="hidden")

        demo.recorder = recorder  # type: ignore[attr-defined]  # stopped on exit by main()
        demo.toolbox = toolbox    # type: ignore[attr-defined]
        demo.engine = engine      # type: ignore[attr-defined]  # shared with the Today page
    demo.reporter = reporter  # type: ignore[attr-defined]
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Histor test UI")
    parser.add_argument("--config", type=Path, default=Path("configs/laptop.yaml"))
    parser.add_argument("--camera", help="Camera name to record as (default: first in config)")
    parser.add_argument("--source", help="Webcam index, file or URL (default: config `source`)")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true", help="Don't open a browser tab")
    args = parser.parse_args()
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", args.port)) == 0:
            raise SystemExit(f"Port {args.port} is in use (another UI still running?): stop it "
                             f"or pass --port {args.port + 1}")

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    camera = args.camera or next(iter(config.get("cameras") or {}), "camera")
    source = args.source if args.source is not None else str(config.get("source", 0))
    demo = build_ui(config, camera, source)
    recorder: Recorder = demo.recorder  # type: ignore[attr-defined]
    today = Today(config, demo.engine,  # type: ignore[attr-defined]
                  recording=lambda: {recorder.camera} if recorder.running else set(),
                  model_state=model_state)
    # Served through FastAPI rather than demo.launch(): the Today page is its own HTML + JSON,
    # and the test bench's webcam view is a raw MJPEG stream on its own route.
    server = FastAPI()
    add_today_routes(server, today, brand_logo())
    bench.add_live_route(server)
    server = gr.mount_gradio_app(server, demo, path=APP, theme=THEME, css=CSS, head=APP_HEAD,
                                 footer_links=[])
    url = f"http://127.0.0.1:{args.port}"
    print(f"* Running on local URL:  {url}")
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, [url]).start()
    try:
        uvicorn.run(server, host="127.0.0.1", port=args.port, log_level="warning")
    finally:
        bench.LIVE.stop()     # release the bench's webcam
        demo.recorder.stop()  # type: ignore[attr-defined]
        demo.recorder.discard()  # type: ignore[attr-defined]
        today.close()
        demo.toolbox.close()  # type: ignore[attr-defined]
        if demo.reporter:     # type: ignore[attr-defined]
            demo.reporter.close()  # type: ignore[attr-defined]


if __name__ == "__main__":
    main()
