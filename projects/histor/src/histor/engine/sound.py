"""Sound events from camera microphones: smoke alarms, breaking glass, gunshots, screams, sirens.

    model      YAMNet (Google, Apache-2.0; AudioSet's 521 classes), as ONNX with the mel
               frontend inside the graph: 16 kHz mono waveform in, one row of scores per 0.96 s
               frame, a frame every 0.48 s. A few milliseconds per second of audio on a CPU.
               `uv run -m histor.engine.download_model sounds` (a tf2onnx conversion of
               google/yamnet v1, pinned to one revision).
    groups     Histor event names -> AudioSet classes (DEFAULT_GROUPS, or `sounds.groups`). A group
               fires when one of its classes scores at least `min_score` in `min_frames`
               consecutive frames, then rests for `cooldown_seconds`.
    audio      ffmpeg reads the camera's audio track (the main stream usually has it) or a
               video file's; .wav files are read directly. Reconnects on its own when live.
    privacy    Audio is classified in memory and dropped: only the event (type `sound`, rule =
               the group, label = the AudioSet class) is kept. Listening is regulated in many
               places (often stricter than video): off until the client and local law allow it.
"""

from __future__ import annotations

import csv
import logging
import queue
import shutil
import subprocess
import threading
import time
import wave
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
PATCH = 15360  # samples in one 0.96 s frame
HOP = 7680     # samples between frames (0.48 s)
EVENT_TYPE = "sound"

DEFAULT_GROUPS: dict[str, list[str]] = {
    "smoke_alarm": ["Smoke detector, smoke alarm", "Fire alarm"],
    "glass_break": ["Shatter", "Glass"],
    "gunshot": ["Gunshot, gunfire", "Machine gun", "Fusillade", "Cap gun"],
    "explosion": ["Explosion"],
    "scream": ["Screaming"],
    "car_alarm": ["Car alarm"],
    "siren": ["Siren", "Civil defense siren", "Police car (siren)", "Ambulance (siren)",
              "Fire engine, fire truck (siren)"],
}


@dataclass
class SoundEvent:
    time: float   # seconds: unix time when live, offset from the start of a file otherwise
    group: str    # e.g. smoke_alarm
    label: str    # the AudioSet class that scored highest in the group
    score: float  # its best score over the frames that fired


def read_class_map(path: str | Path) -> list[str]:
    """yamnet_class_map.csv (index, mid, display_name) -> display names by index."""
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [r["display_name"] for r in sorted(rows, key=lambda r: int(r["index"]))]


class SoundClassifier:
    def __init__(self, model: str | Path = "models/sound_yamnet.onnx",
                 classes: str | Path = "models/sound_yamnet_classes.csv",
                 device: str = "cpu", num_threads: int = 1) -> None:
        """device cpu, one thread by default: the model is tiny, the cameras' detectors need
        the cores more."""
        from histor.engine.onnx_device import create_session

        for path in (model, classes):
            if not Path(path).exists():
                raise FileNotFoundError(
                    f"{path} not found. Download with: uv run -m histor.engine.download_model sounds")
        self.classes = read_class_map(classes)
        self.session = create_session(model, device, num_threads)
        self.input = self.session.get_inputs()[0].name

    def classify(self, waveform: np.ndarray) -> np.ndarray:
        """16 kHz mono float32 in [-1, 1] -> (frames, classes) scores."""
        return self.session.run(None, {self.input: np.asarray(waveform, np.float32)})[0]


class SoundRules:
    """Turns per-frame scores into events: per group, `min_frames` consecutive frames with a
    class of the group at `min_score` or more, then `cooldown_seconds` of rest."""

    def __init__(self, classes: list[str], groups: dict[str, list[str]] | None = None,
                 min_score: float = 0.3, min_frames: int = 2, cooldown_seconds: float = 30.0,
                 scores: dict[str, float] | None = None) -> None:
        """scores: a different min_score per group, e.g. {glass_break: 0.5}."""
        groups = DEFAULT_GROUPS if groups is None else groups
        index = {name: i for i, name in enumerate(classes)}
        self.groups: dict[str, list[int]] = {}
        for group, names in groups.items():
            unknown = [n for n in names if n not in index]
            if unknown:
                raise ValueError(f"sounds.groups.{group}: {unknown} are not AudioSet classes "
                                 "(see models/sound_yamnet_classes.csv, display_name)")
            if not names:
                raise ValueError(f"sounds.groups.{group}: no classes")
            self.groups[group] = [index[n] for n in names]
        self.classes = classes
        self.min_score = {g: (scores or {}).get(g, min_score) for g in self.groups}
        self.min_frames = max(1, int(min_frames))
        self.cooldown = cooldown_seconds
        self._run: dict[str, int] = {g: 0 for g in self.groups}
        self._best: dict[str, tuple[float, int]] = {}
        self._last: dict[str, float] = {}

    def update(self, scores: np.ndarray, start: float, step: float = HOP / SAMPLE_RATE
               ) -> list[SoundEvent]:
        """scores: (frames, classes) for frames starting at `start`, `step` seconds apart."""
        events = []
        for i, row in enumerate(np.atleast_2d(scores)):
            t = start + i * step
            for group, idx in self.groups.items():
                best = int(idx[int(np.argmax(row[idx]))])
                if row[best] < self.min_score[group]:
                    self._run[group] = 0
                    self._best.pop(group, None)
                    continue
                self._run[group] += 1
                prev = self._best.get(group)
                if prev is None or row[best] > prev[0]:
                    self._best[group] = (float(row[best]), best)
                if self._run[group] < self.min_frames:
                    continue
                if t - self._last.get(group, -1e18) < self.cooldown:
                    continue
                score, cls = self._best[group]
                self._last[group] = t
                events.append(SoundEvent(t, group, self.classes[cls], round(score, 3)))
        return events


# ------------------------------------------------------------------ audio sources

def read_wav(path: str | Path) -> np.ndarray:
    """A .wav file -> 16 kHz mono float32 (8/16/32-bit PCM; any rate, linearly resampled)."""
    with wave.open(str(path), "rb") as w:
        channels, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(w.getnframes())
    dtypes = {1: np.uint8, 2: np.int16, 4: np.int32}
    if width not in dtypes:
        raise ValueError(f"{path}: {8 * width}-bit samples are not supported")
    samples = np.frombuffer(raw, dtypes[width]).astype(np.float64)
    # 8-bit WAV is unsigned around 128; 16/32-bit are signed
    scaled = (samples - 128) / 128 if width == 1 else samples / float(2 ** (8 * width - 1) - 1)
    mono = scaled.reshape(-1, channels).mean(axis=1)
    if rate != SAMPLE_RATE and len(mono):
        n = int(round(len(mono) * SAMPLE_RATE / rate))
        mono = np.interp(np.linspace(0, len(mono) - 1, n), np.arange(len(mono)), mono)
    return mono.astype(np.float32)


def ffmpeg_command(source: str, ffmpeg: str = "ffmpeg") -> list[str]:
    """ffmpeg reading `source`'s audio as 16 kHz mono 16-bit PCM on stdout."""
    cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error"]
    if source.lower().startswith("rtsp://"):
        cmd += ["-rtsp_transport", "tcp"]
    return cmd + ["-i", source, "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"]


def iter_audio(source: str, live: bool, chunk_seconds: float = 0.48, ffmpeg: str = "ffmpeg",
               stop: threading.Event | None = None) -> Iterator[np.ndarray]:
    """Chunks of 16 kHz mono float32. A .wav file is read directly; anything else (camera URL,
    video file) through ffmpeg. Live sources reconnect with a growing pause."""
    stop = stop or threading.Event()
    if not live and str(source).lower().endswith(".wav"):
        audio = read_wav(source)
        size = int(chunk_seconds * SAMPLE_RATE)
        for i in range(0, len(audio), size):
            if stop.is_set():
                return
            yield audio[i:i + size]
        return
    if shutil.which(ffmpeg) is None:
        raise FileNotFoundError(f"ffmpeg not found ({ffmpeg}): sounds need it to read audio "
                                "(sudo apt install ffmpeg, or set sounds.ffmpeg)")
    size = int(chunk_seconds * SAMPLE_RATE) * 2  # bytes of int16
    pause = 1.0
    while not stop.is_set():
        proc = subprocess.Popen(ffmpeg_command(str(source), ffmpeg), stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL)
        got = False
        try:
            assert proc.stdout is not None
            while not stop.is_set():
                data = proc.stdout.read(size)
                if not data:
                    break
                got = True
                yield np.frombuffer(data[: len(data) // 2 * 2], np.int16).astype(np.float32) / 32768
        finally:
            proc.kill()
            proc.wait()
        if not live:
            return
        pause = 1.0 if got else min(pause * 2, 60.0)
        log.warning("Audio of %s ended; reconnecting in %.0f s", source, pause)
        stop.wait(pause)


# ------------------------------------------------------------------ per camera

class SoundListener:
    """Reads one source's audio on a background thread and queues SoundEvents."""

    def __init__(self, classifier: Any, rules: SoundRules, source: str, live: bool = True,
                 window_frames: int = 4, ffmpeg: str = "ffmpeg",
                 audio: Iterator[np.ndarray] | None = None) -> None:
        """window_frames: frames per model call (4 = 2.4 s of audio): bigger is cheaper per
        second of audio, smaller alerts sooner. audio: chunks to use instead of reading
        `source` (tests)."""
        self.classifier = classifier
        self.rules = rules
        self.source = source
        self.live = live
        self.window_frames = max(1, int(window_frames))
        self.ffmpeg = ffmpeg
        self.seconds = 0.0  # audio classified so far
        self.error: str | None = None
        self._stop = threading.Event()
        self._audio = audio
        self._events: queue.Queue[SoundEvent] = queue.Queue()
        self._pending: list[SoundEvent] = []
        self._thread = threading.Thread(target=self._run, daemon=True, name="sounds")
        self._thread.start()

    @classmethod
    def from_config(cls, config: dict[str, Any], source: str, live: bool) -> SoundListener:
        """config: the `sounds` section."""
        classifier = SoundClassifier(config.get("model", "models/sound_yamnet.onnx"),
                                     config.get("classes", "models/sound_yamnet_classes.csv"),
                                     config.get("device", "cpu"), config.get("num_threads", 1))
        rules = SoundRules(classifier.classes, config.get("groups"),
                           config.get("min_score", 0.3), config.get("min_frames", 2),
                           config.get("cooldown_seconds", 30.0), config.get("scores"))
        return cls(classifier, rules, source, live, config.get("window_frames", 4),
                   config.get("ffmpeg", "ffmpeg"))

    def take_events(self, until: float | None = None) -> list[SoundEvent]:
        """Events so far; with `until` (a file's video time), only those up to it, so each
        event's snapshot is taken when it happened."""
        while True:
            try:
                self._pending.append(self._events.get_nowait())
            except queue.Empty:
                break
        if until is None:
            out, self._pending = self._pending, []
        else:
            out = [e for e in self._pending if e.time <= until]
            self._pending = [e for e in self._pending if e.time > until]
        return out

    def close(self, timeout: float = 30.0) -> None:
        if self.live:
            self._stop.set()
        self._thread.join(timeout=timeout)
        self._stop.set()

    def _run(self) -> None:
        window = PATCH + HOP * (self.window_frames - 1)
        step = HOP * self.window_frames
        buffer = np.zeros(0, np.float32)
        start = time.time() if self.live else 0.0  # time of buffer[0]
        try:
            chunks = self._audio if self._audio is not None else iter_audio(
                self.source, self.live, ffmpeg=self.ffmpeg, stop=self._stop)
            for chunk in chunks:
                if self.live and not len(buffer):
                    start = time.time() - len(chunk) / SAMPLE_RATE  # after a reconnect too
                buffer = np.concatenate([buffer, chunk])
                while len(buffer) >= window:
                    self._classify(buffer[:window], start)
                    buffer, start = buffer[step:], start + step / SAMPLE_RATE
            if len(buffer) >= PATCH and not self.live:  # the end of a file
                self._classify(buffer, start)
        except FileNotFoundError as e:
            self.error = str(e)
            log.error("No sound events for %s: %s", self.source, e)
        except Exception:
            self.error = "failed"
            log.exception("Sound events failed for %s", self.source)

    def _classify(self, audio: np.ndarray, start: float) -> None:
        scores = self.classifier.classify(audio)
        for event in self.rules.update(scores, start):
            self._events.put(event)
        self.seconds += len(scores) * HOP / SAMPLE_RATE


def sounds_enabled(config: dict[str, Any] | None, camera: str) -> bool:
    if not config or not config.get("enabled"):
        return False
    cameras = config.get("cameras") or []
    return not cameras or camera in cameras
