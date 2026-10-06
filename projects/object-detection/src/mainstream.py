"""Main stream of an IP camera: saved to disk untouched, keyframes decoded for hi-res crops.

The detector reads the camera's low-resolution sub stream (cheap, every frame). This module
opens the *main* stream with one ffmpeg process per camera and does two things with it:

  * recording - packets are copied into one MP4 file per minute (`-c copy`: no decoding, no
                re-encoding, almost no CPU), so every search hit can open the real video
  * keyframes - only keyframes are decoded (`-skip_frame nokey`: about one per second with
                the camera's I-frame interval set to 1 s), so the search index crops people,
                vehicles and faces from full-resolution pixels instead of the sub stream

Needs ffmpeg >= 5.1 on the PATH (Ubuntu 24.04: `sudo apt install ffmpeg`).

    <dir>/<camera>/20261005-140100.mp4      one file per minute, named by its local start time
"""

from __future__ import annotations

import bisect
import logging
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import IO, Any
from urllib.parse import unquote, urlsplit

import cv2
import numpy as np

from cameras import mask_url

log = logging.getLogger(__name__)
SEGMENT_FORMAT = "%Y%m%d-%H%M%S"


@dataclass
class Keyframe:
    time: float  # unix seconds when it arrived
    image: np.ndarray  # BGR, full main-stream resolution


class MainStream:
    """One ffmpeg process per camera; reconnects on its own, like `LiveCapture`."""

    def __init__(
        self,
        url: str,
        camera: str,
        record_dir: str | Path | None = None,
        keyframes: bool = True,
        segment_seconds: int = 60,
        retention_days: float | None = 7,
        min_free_gb: float | None = 20,
        ffmpeg: str = "ffmpeg",
        stall_seconds: float = 20.0,
        max_reconnect_delay: float = 30.0,
    ) -> None:
        """record_dir: save the stream under <record_dir>/<camera>/ (None = don't record).
        keyframes: decode keyframes for `take_keyframe()` (for hi-res crops).
        retention_days / min_free_gb: delete this camera's oldest files past either limit."""
        if record_dir is None and not keyframes:
            raise ValueError("MainStream needs record_dir, keyframes, or both")
        self.url = url
        self.camera = camera
        self.folder = Path(record_dir) / camera if record_dir is not None else None
        self.keyframes = keyframes
        self.segment_seconds = segment_seconds
        self.retention_days = retention_days
        self.min_free_gb = min_free_gb
        self.ffmpeg = ffmpeg
        self.stall_seconds = stall_seconds
        self.max_reconnect_delay = max_reconnect_delay
        self._latest: Keyframe | None = None
        self._taken: Keyframe | None = None
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[bytes] | None = None
        self._progress = time.monotonic()
        self._stop = threading.Event()
        self._threads = [
            threading.Thread(target=self._run, daemon=True, name=f"main-{camera}"),
            threading.Thread(target=self._watch, daemon=True, name=f"main-watch-{camera}"),
        ]

    @classmethod
    def from_config(cls, config: dict[str, Any], url: str, camera: str,
                    keyframes: bool) -> MainStream | None:
        """`recording:` section of the config. None when there is nothing to do."""
        record = bool(config.get("enabled"))
        keyframes = keyframes and bool(config.get("hires_crops", True))
        if not (record or keyframes):
            return None
        options = {k: config[k] for k in ("segment_seconds", "retention_days", "min_free_gb",
                                          "ffmpeg", "stall_seconds") if k in config}
        stream = cls(url, camera, config.get("dir", "data/store/recordings") if record else None,
                     keyframes=keyframes, **options)
        return stream.start()

    def start(self) -> MainStream:
        if self.folder is not None:
            self.folder.mkdir(parents=True, exist_ok=True)
        for thread in self._threads:
            thread.start()
        return self

    def command(self) -> list[str]:
        cmd = [self.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error"]
        if self.url.lower().startswith("rtsp"):
            # tcp: no smeared frames on lossy links; timeout (µs): give up on a dead camera
            cmd += ["-rtsp_transport", "tcp", "-timeout", str(int(self.stall_seconds * 1e6))]
        if self.keyframes:
            cmd += ["-skip_frame", "nokey"]  # decoder option: the copied recording is untouched
        cmd += ["-i", self.url]
        if self.folder is not None:
            cmd += [
                "-map", "0:v:0", "-c", "copy",
                "-f", "segment", "-segment_time", str(self.segment_seconds),
                "-segment_atclocktime", "1", "-reset_timestamps", "1", "-strftime", "1",
                "-segment_format", "mp4",
                # fragmented MP4: the file being written plays, and survives a crash or power cut
                "-segment_format_options", "movflags=+frag_keyframe+empty_moov+default_base_moof",
                str(self.folder / f"{SEGMENT_FORMAT}.mp4"),
            ]
        if self.keyframes:
            # BMP is uncompressed (no encoding cost) and carries its own size in the header
            cmd += ["-map", "0:v:0", "-fps_mode", "passthrough",
                    "-f", "image2pipe", "-c:v", "bmp", "-pix_fmt", "bgr24", "pipe:1"]
        return cmd

    # ---------------------------------------------------------------- consumer side

    def latest(self) -> Keyframe | None:
        with self._lock:
            return self._latest

    def take_keyframe(self) -> Keyframe | None:
        """Newest keyframe not returned before, or None."""
        with self._lock:
            if self._latest is None or self._latest is self._taken:
                return None
            self._taken = self._latest
            return self._latest

    def close(self) -> None:
        self._stop.set()
        self._kill()
        for thread in self._threads:
            if thread.is_alive():
                thread.join(timeout=5)

    # ---------------------------------------------------------------- workers

    def _run(self) -> None:
        delay = 1.0
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                proc = subprocess.Popen(
                    self.command(), stdin=subprocess.DEVNULL, stderr=subprocess.PIPE,
                    stdout=subprocess.PIPE if self.keyframes else subprocess.DEVNULL,
                )
            except FileNotFoundError:
                log.error("ffmpeg not found (%s): main-stream recording and hi-res crops are off "
                          "for %s. Install it (sudo apt install ffmpeg) or set recording.ffmpeg.",
                          self.ffmpeg, self.camera)
                return
            self._proc = proc
            self._progress = time.monotonic()
            threading.Thread(target=self._log_errors, args=(proc.stderr,), daemon=True).start()
            log.info("Main stream of %s opened (%s)", self.camera, mask_url(self.url))
            if self.keyframes and proc.stdout is not None:
                self._read_keyframes(proc.stdout)
            else:
                while proc.poll() is None and not self._stop.is_set():
                    self._stop.wait(1)
            self._kill()
            if self.folder is not None:  # a failed start leaves empty files behind
                for _, path in _segments(self.folder):
                    if path.stat().st_size == 0:
                        path.unlink(missing_ok=True)
            if self._stop.is_set():
                return
            if time.monotonic() - started > 60:
                delay = 1.0
            log.warning("Main stream of %s ended, reconnecting in %.0f s", self.camera, delay)
            self._stop.wait(delay)
            delay = min(delay * 2, self.max_reconnect_delay)

    def _read_keyframes(self, pipe: IO[bytes]) -> None:
        while not self._stop.is_set():
            image = read_bmp(pipe)
            if image is None:
                return
            self._progress = time.monotonic()
            with self._lock:
                self._latest = Keyframe(time.time(), image)

    def _watch(self) -> None:
        """Kill a stalled ffmpeg (it is restarted by `_run`) and apply retention."""
        last_prune = 0.0
        newest_size = -1
        while not self._stop.wait(2):
            if not self.keyframes and self.folder is not None:  # progress = recording grows
                size = _newest_size(self.folder)
                if size != newest_size:
                    newest_size, self._progress = size, time.monotonic()
            if self._proc is not None and self._proc.poll() is None and \
                    time.monotonic() - self._progress > self.stall_seconds:
                log.warning("Main stream of %s stalled, restarting ffmpeg", self.camera)
                self._kill()
            if self.folder is not None and time.monotonic() - last_prune > 600:
                last_prune = time.monotonic()
                deleted = prune_recordings(self.folder, self.retention_days, self.min_free_gb)
                if deleted:
                    log.info("Deleted %d old recording(s) of %s", deleted, self.camera)

    def _kill(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()  # lets ffmpeg close the current file cleanly
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    def _log_errors(self, pipe: IO[bytes] | None) -> None:
        if pipe is None:
            return
        secret = urlsplit(self.url).password
        for raw in pipe:
            line = raw.decode(errors="replace").strip()
            if secret:  # ffmpeg echoes the URL in its errors: never log the password
                line = line.replace(secret, "***").replace(unquote(secret), "***")
            if line:
                log.warning("ffmpeg %s: %s", self.camera, line)


# ---------------------------------------------------------------------------- helpers

def read_bmp(pipe: IO[bytes]) -> np.ndarray | None:
    """Next BMP image from a stream of concatenated BMP files; None at end of stream."""
    header = _read_exact(pipe, 14)
    if header is None:
        return None
    if header[:2] != b"BM":
        raise ValueError("Unexpected data in the keyframe pipe (not a BMP)")
    size = int.from_bytes(header[2:6], "little")
    rest = _read_exact(pipe, size - 14)
    if rest is None:
        return None
    return cv2.imdecode(np.frombuffer(header + rest, np.uint8), cv2.IMREAD_COLOR)


def _read_exact(pipe: IO[bytes], n: int) -> bytes | None:
    chunks, remaining = [], n
    while remaining:
        chunk = pipe.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def segment_start(path: Path) -> float | None:
    """Unix time a recording starts, from its file name; None for other files."""
    try:
        return datetime.strptime(path.stem, SEGMENT_FORMAT).timestamp()
    except ValueError:
        return None


def _segments(folder: Path) -> list[tuple[float, Path]]:
    if not folder.is_dir():
        return []
    found = [(segment_start(Path(e.path)), Path(e.path)) for e in os.scandir(folder)
             if e.name.endswith(".mp4")]
    return sorted((t, p) for t, p in found if t is not None)


def _newest_size(folder: Path) -> int:
    segments = _segments(folder)
    return segments[-1][1].stat().st_size if segments else -1


def prune_recordings(folder: Path, retention_days: float | None, min_free_gb: float | None,
                     now: float | None = None) -> int:
    """Delete files older than `retention_days`, then the oldest ones while the disk has less
    than `min_free_gb` free. The newest file (being written) is never deleted."""
    now = time.time() if now is None else now
    segments = _segments(folder)[:-1]
    deleted = 0
    if retention_days is not None:
        cutoff = now - retention_days * 86400
        while segments and segments[0][0] < cutoff:
            segments.pop(0)[1].unlink(missing_ok=True)
            deleted += 1
    if min_free_gb is not None:
        while segments and shutil.disk_usage(folder).free < min_free_gb * 1e9:
            segments.pop(0)[1].unlink(missing_ok=True)
            deleted += 1
    return deleted


class Recordings:
    """Finds the recording (file + offset) that holds a given moment, for the API."""

    def __init__(self, root: str | Path, segment_seconds: int = 60) -> None:
        self.root = Path(root)
        self.max_gap = segment_seconds * 2  # past this, the camera was not recording
        self._cache: dict[str, tuple[float, list[tuple[float, Path]]]] = {}

    def find(self, camera: str, t: float) -> tuple[str, float] | None:
        """(path relative to root, seconds into the file) or None."""
        folder = self.root / camera
        if folder.resolve().parent != self.root.resolve():  # camera names come from requests
            return None
        segments = self._list(folder)
        i = bisect.bisect_right([s for s, _ in segments], t) - 1
        if i < 0 or t - segments[i][0] > self.max_gap:
            return None
        start, path = segments[i]
        return f"{camera}/{path.name}", t - start

    def path(self, rel: str) -> Path | None:
        """Resolve a relative path; None if it escapes the root or does not exist."""
        root = self.root.resolve()
        target = (root / rel).resolve()
        if root not in target.parents or not target.is_file() or target.suffix != ".mp4":
            return None
        return target

    def _list(self, folder: Path) -> list[tuple[float, Path]]:
        try:
            mtime = folder.stat().st_mtime  # changes when a new file is added
        except OSError:
            return []
        cached = self._cache.get(str(folder))
        if cached is None or cached[0] != mtime:
            cached = (mtime, _segments(folder))
            self._cache[str(folder)] = cached
        return cached[1]
