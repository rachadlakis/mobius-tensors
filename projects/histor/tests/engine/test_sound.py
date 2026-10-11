"""Sound events: the rules over YAMNet's scores, audio reading, the listener's timing, the
pipeline and the chat; the real model on synthetic sounds when it is downloaded."""

from __future__ import annotations

import time
import wave
from pathlib import Path

import numpy as np
import pytest

from histor.engine import download_model
from histor.engine.alerts import AlertManager, alert_message
from histor.engine.pipeline import Pipeline
from histor.engine.search import SearchEngine
from histor.engine.sound import (DEFAULT_GROUPS, HOP, PATCH, SAMPLE_RATE, SoundClassifier,
                                SoundListener, SoundRules, ffmpeg_command, iter_audio,
                                read_class_map, read_wav, sounds_enabled)
from histor.engine.store import EventStore
from histor.engine.tracker import Tracker
from histor.tools import ALERT_TYPES, Toolbox

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "sound_yamnet.onnx"
CLASSES = ROOT / "models" / "sound_yamnet_classes.csv"
NAMES = ["Speech", "Smoke detector, smoke alarm", "Fire alarm", "Shatter", "Glass", "Dog"]


@pytest.fixture
def store(tmp_path):
    s = EventStore(tmp_path / "db.sqlite")
    yield s
    s.close()


def scores(*rows: dict[str, float]) -> np.ndarray:
    out = np.zeros((len(rows), len(NAMES)), np.float32)
    for i, row in enumerate(rows):
        for name, value in row.items():
            out[i, NAMES.index(name)] = value
    return out


GROUPS = {"smoke_alarm": ["Smoke detector, smoke alarm", "Fire alarm"],
          "glass_break": ["Shatter", "Glass"]}


def write_wav(path: Path, audio: np.ndarray, rate: int = SAMPLE_RATE, channels: int = 1,
              width: int = 2) -> Path:
    data = np.repeat(audio[:, None], channels, axis=1)
    if width == 2:
        raw = (np.clip(data, -1, 1) * 32767).astype("<i2").tobytes()
    elif width == 1:
        raw = (np.clip(data, -1, 1) * 127 + 128).astype(np.uint8).tobytes()
    else:
        raw = (np.clip(data, -1, 1) * 8388607).astype("<i4").tobytes()[: len(data) * 3 * channels]
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(raw)
    return path


def smoke_alarm(seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (0.5 * np.sin(2 * np.pi * 3100 * t) * ((t % 1) < 0.5)).astype(np.float32)


# ------------------------------------------------------------------ rules

def test_rule_needs_consecutive_frames():
    rules = SoundRules(NAMES, GROUPS, min_score=0.3, min_frames=2, cooldown_seconds=0)
    hit = {"Fire alarm": 0.6}
    assert rules.update(scores(hit, {}, hit), 0.0) == []  # never two in a row
    (event,) = rules.update(scores(hit), 1.44)            # the previous frame was a hit
    assert event.group == "smoke_alarm" and event.label == "Fire alarm"
    assert event.time == pytest.approx(1.44) and event.score == pytest.approx(0.6)


def test_rule_reports_best_class_and_score_of_the_run():
    rules = SoundRules(NAMES, GROUPS, min_frames=3, cooldown_seconds=0)
    (event,) = rules.update(scores({"Fire alarm": 0.4}, {"Smoke detector, smoke alarm": 0.9},
                                   {"Fire alarm": 0.5}), 10.0)
    assert event.label == "Smoke detector, smoke alarm" and event.score == pytest.approx(0.9)
    assert event.time == pytest.approx(10.0 + 2 * HOP / SAMPLE_RATE)


def test_rule_cooldown_and_independent_groups():
    rules = SoundRules(NAMES, GROUPS, min_frames=1, cooldown_seconds=30)
    both = {"Fire alarm": 0.5, "Shatter": 0.5}
    first = rules.update(scores(both), 0.0)
    assert sorted(e.group for e in first) == ["glass_break", "smoke_alarm"]
    assert rules.update(scores(both), 10.0) == []         # resting
    assert len(rules.update(scores(both), 31.0)) == 2     # rested


def test_rule_per_group_score():
    rules = SoundRules(NAMES, GROUPS, min_score=0.3, min_frames=1, scores={"glass_break": 0.8})
    events = rules.update(scores({"Glass": 0.5, "Fire alarm": 0.5}), 0.0)
    assert [e.group for e in events] == ["smoke_alarm"]


def test_rule_config_errors():
    with pytest.raises(ValueError, match="not AudioSet classes"):
        SoundRules(NAMES, {"x": ["Explosion!"]})
    with pytest.raises(ValueError, match="no classes"):
        SoundRules(NAMES, {"x": []})


@pytest.mark.skipif(not CLASSES.exists(), reason="uv run -m histor.engine.download_model sounds")
def test_default_groups_are_real_audioset_classes():
    classes = read_class_map(CLASSES)
    assert len(classes) == 521 and classes[0] == "Speech"
    SoundRules(classes)  # raises if a default name is wrong
    assert set(DEFAULT_GROUPS) == {"smoke_alarm", "glass_break", "gunshot", "explosion",
                                   "scream", "car_alarm", "siren"}


def test_sounds_enabled():
    assert not sounds_enabled(None, "gate") and not sounds_enabled({"enabled": False}, "gate")
    assert sounds_enabled({"enabled": True}, "gate")
    assert not sounds_enabled({"enabled": True, "cameras": ["yard"]}, "gate")


def test_download_is_pinned():
    assert set(download_model.SOUND_MODELS) == {"sound_yamnet.onnx", "sound_yamnet_classes.csv"}
    assert all("/resolve/f25b741c" in u for u in download_model.SOUND_MODELS.values())


# ------------------------------------------------------------------ audio

def test_read_wav_resamples_and_mixes_down(tmp_path):
    tone = np.sin(2 * np.pi * 440 * np.arange(44100) / 44100).astype(np.float32) * 0.5
    audio = read_wav(write_wav(tmp_path / "a.wav", tone, rate=44100, channels=2))
    assert audio.dtype == np.float32 and len(audio) == SAMPLE_RATE
    assert np.abs(audio).max() == pytest.approx(0.5, abs=0.01)
    eight = read_wav(write_wav(tmp_path / "b.wav", tone[:16000], rate=16000, width=1))
    assert len(eight) == 16000 and np.abs(eight).max() == pytest.approx(0.5, abs=0.02)


def test_read_wav_rejects_24_bit(tmp_path):
    path = tmp_path / "c.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(3)
        w.setframerate(16000)
        w.writeframes(b"\x00" * 300)
    with pytest.raises(ValueError, match="24-bit"):
        read_wav(path)


def test_ffmpeg_command():
    rtsp = ffmpeg_command("rtsp://cam/main")
    assert rtsp[rtsp.index("-rtsp_transport") + 1] == "tcp"
    assert rtsp[-8:] == ["-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"]
    assert "-rtsp_transport" not in ffmpeg_command("clip.mp4")


def test_iter_audio_wav_chunks(tmp_path):
    path = write_wav(tmp_path / "a.wav", np.zeros(SAMPLE_RATE, np.float32))
    chunks = list(iter_audio(str(path), live=False, chunk_seconds=0.25))
    assert [len(c) for c in chunks] == [4000] * 4


def test_iter_audio_without_ffmpeg(monkeypatch):
    monkeypatch.setattr("histor.engine.sound.shutil.which", lambda name: None)
    with pytest.raises(FileNotFoundError, match="ffmpeg not found"):
        next(iter_audio("rtsp://cam/main", live=True))


# ------------------------------------------------------------------ listener timing

class RampClassifier:
    """Scores frame i of a window with the time (in samples) it starts at, read back from the
    audio itself: the waveform is a ramp of sample indices / 1e7."""

    classes = NAMES

    def __init__(self) -> None:
        self.starts: list[int] = []

    def classify(self, audio):
        frames = 1 + max(0, -(-(len(audio) - PATCH) // HOP))
        first = int(round(float(audio[0]) * 1e7))
        out = np.zeros((frames, len(NAMES)), np.float32)
        for i in range(frames):
            self.starts.append(first + i * HOP)
        return out


def ramp(seconds: float) -> list[np.ndarray]:
    n = int(seconds * SAMPLE_RATE)
    audio = (np.arange(n, dtype=np.float64) / 1e7).astype(np.float32)
    return [audio[i:i + 4000] for i in range(0, n, 4000)]


def test_listener_frames_follow_one_grid_without_overlap():
    clf = RampClassifier()
    listener = SoundListener(clf, SoundRules(NAMES, GROUPS), "f.wav", live=False,
                             window_frames=4, audio=iter(ramp(10)))
    listener.close()
    expected = list(range(0, 10 * SAMPLE_RATE - PATCH + 1, HOP))
    assert clf.starts[:len(expected)] == expected  # every frame once, 0.48 s apart
    assert len(set(clf.starts)) == len(clf.starts)
    assert listener.seconds == pytest.approx(len(clf.starts) * HOP / SAMPLE_RATE)


class FixedClassifier:
    """Fires the given class in frames whose start (seconds) is within [on, off)."""

    classes = NAMES

    def __init__(self, label: str, on: float, off: float) -> None:
        self.label, self.on, self.off = label, on, off
        self.t = 0.0

    def classify(self, audio):
        frames = 1 + max(0, -(-(len(audio) - PATCH) // HOP))
        out = np.zeros((frames, len(NAMES)), np.float32)
        for i in range(frames):
            if self.on <= self.t < self.off:
                out[i, NAMES.index(self.label)] = 0.9
            self.t += HOP / SAMPLE_RATE
        return out


def test_listener_events_released_when_the_video_gets_there():
    clf = FixedClassifier("Shatter", 3.0, 5.0)
    listener = SoundListener(clf, SoundRules(NAMES, GROUPS, min_frames=2), "f.wav",
                             live=False, audio=iter([np.zeros(SAMPLE_RATE * 8, np.float32)]))
    deadline = time.time() + 5
    while listener.seconds < 6 and time.time() < deadline:
        time.sleep(0.01)
    assert listener.take_events(until=2.0) == []
    (event,) = listener.take_events(until=4.0)
    assert event.group == "glass_break" and 3.0 < event.time <= 4.0
    listener.close()
    assert listener.take_events() == []


def test_listener_records_a_missing_ffmpeg(monkeypatch):
    monkeypatch.setattr("histor.engine.sound.shutil.which", lambda name: None)
    listener = SoundListener(RampClassifier(), SoundRules(NAMES, GROUPS), "rtsp://cam", live=True)
    listener.close(timeout=5)
    assert listener.error and "ffmpeg" in listener.error


def test_live_listener_stamps_wall_time():
    clf = FixedClassifier("Fire alarm", 0.0, 99.0)
    before = time.time()
    listener = SoundListener(clf, SoundRules(NAMES, GROUPS, min_frames=1), "rtsp://cam",
                             live=True, window_frames=1,
                             audio=iter([np.zeros(PATCH * 2, np.float32)]))
    listener._thread.join(5)
    (event, *_) = listener.take_events()
    assert before - 3 <= event.time <= time.time()
    listener.close()


# ------------------------------------------------------------------ pipeline and chat

class NoDetections:
    def detect(self, frame):
        return []


def test_pipeline_emits_sound_events_at_their_video_time(store):
    clf = FixedClassifier("Fire alarm", 2.0, 4.0)
    listener = SoundListener(clf, SoundRules(NAMES, GROUPS, min_frames=2), "clip.mp4",
                             live=False, audio=iter([np.zeros(SAMPLE_RATE * 6, np.float32)]))
    listener._thread.join(5)
    start = 1_800_000_000.0
    alerts = AlertManager(console=False, store=store, source="hall")
    pipe = Pipeline(NoDetections(), Tracker(), alerts=alerts, store=store, start_time=start,
                    sounds=listener)
    frame = np.zeros((120, 160, 3), np.uint8)
    seen = {}
    for i in range(12):
        for e in pipe.process(frame, i * 0.5).events:
            seen[i * 0.5] = e
    pipe.close()
    ((at, event),) = seen.items()
    assert event.type == "sound" and event.rule == "smoke_alarm" and event.label == "Fire alarm"
    assert event.track_id is None and event.score == pytest.approx(0.9)
    assert 2.0 < event.timestamp <= at  # released once the video passed it
    row = store._db.execute("SELECT type, rule, label, track_id, time, snapshot FROM events"
                            ).fetchone()
    assert tuple(row[:4]) == ("sound", "smoke_alarm", "Fire alarm", None)
    assert row[4] == pytest.approx(start + event.timestamp) and row[5]


def test_pipeline_close_flushes_sounds_after_the_last_frame(store):
    clf = FixedClassifier("Shatter", 5.0, 9.0)
    listener = SoundListener(clf, SoundRules(NAMES, GROUPS, min_frames=1), "clip.mp4",
                             live=False, audio=iter([np.zeros(SAMPLE_RATE * 10, np.float32)]))
    alerts = AlertManager(console=False, store=store, source="hall")
    pipe = Pipeline(NoDetections(), Tracker(), alerts=alerts, store=store, start_time=0.0,
                    sounds=listener)
    pipe.process(np.zeros((60, 80, 3), np.uint8), 0.0)  # the video is shorter than its audio
    pipe.close()
    assert store._db.execute("SELECT COUNT(*) FROM events WHERE type = 'sound'").fetchone()[0] == 1


def test_alert_message_for_a_sound():
    title, text = alert_message({"type": "sound", "rule": "glass_break", "label": "Shatter",
                                 "camera": "shop", "iso_time": "2026-10-09T02:13:00",
                                 "score": 0.8})
    assert title == "sound glass_break - Shatter on shop"
    assert "Heard:  Shatter" in text and "#None" not in text


def test_chat_alerts_tool_lists_sounds(store):
    store.add_event("shop", time.time() - 60, {"type": "sound", "rule": "glass_break",
                                               "label": "Shatter"}, None)
    box = Toolbox(SearchEngine(store))
    try:
        assert "sound" in ALERT_TYPES
        out = box.alerts(types=["sound"], rules=["glass_break"], when="today")
        assert [(e["rule"], e["label"]) for e in out["events"]] == [("glass_break", "Shatter")]
    finally:
        box.close()


def test_from_config_needs_an_audio_source(tmp_path, caplog):
    if not (ROOT / "models" / "yolox_nano.onnx").exists():
        pytest.skip("uv run -m histor.engine.download_model yolox_nano")
    config = {"model": {"path": str(ROOT / "models" / "yolox_nano.onnx")},
              "store": {"path": str(tmp_path / "s.db")}, "sounds": {"enabled": True}}
    pipe = Pipeline.from_config(config, source="0", live=True)
    assert pipe.sounds is None and "no audio" in caplog.text
    pipe.close()
    pipe.store.close()


# ------------------------------------------------------------------ the real model

real = pytest.mark.skipif(not (MODEL.exists() and CLASSES.exists()),
                          reason="uv run -m histor.engine.download_model sounds")


@real
def test_real_model_frames_and_silence():
    clf = SoundClassifier(MODEL, CLASSES)
    s = clf.classify(np.zeros(SAMPLE_RATE * 3, np.float32))
    assert s.shape == (6, 521)  # 1 + ceil((48000 - 15360) / 7680)
    assert clf.classes[int(s.mean(0).argmax())] == "Silence"


@real
def test_real_model_hears_a_smoke_alarm_from_a_wav(tmp_path, store):
    path = write_wav(tmp_path / "alarm.wav", np.concatenate([
        np.zeros(SAMPLE_RATE * 3, np.float32), smoke_alarm(5), np.zeros(SAMPLE_RATE * 2, np.float32)]))
    listener = SoundListener.from_config({"model": str(MODEL), "classes": str(CLASSES)},
                                         str(path), live=False)
    listener.close()
    events = listener.take_events()
    assert [e.group for e in events] == ["smoke_alarm"]
    assert 3.0 <= events[0].time <= 6.0 and events[0].score > 0.5


@real
def test_real_model_quiet_room_raises_nothing(tmp_path):
    rng = np.random.default_rng(1)
    path = write_wav(tmp_path / "quiet.wav", rng.normal(0, 0.003, SAMPLE_RATE * 8).astype(np.float32))
    listener = SoundListener.from_config({"model": str(MODEL), "classes": str(CLASSES)},
                                         str(path), live=False)
    listener.close()
    assert listener.take_events() == [] and listener.error is None


@real
def test_real_pipeline_from_config_with_a_wav(tmp_path):
    if not (ROOT / "models" / "yolox_nano.onnx").exists():
        pytest.skip("uv run -m histor.engine.download_model yolox_nano")
    path = write_wav(tmp_path / "alarm.wav", smoke_alarm(6))
    config = {"model": {"path": str(ROOT / "models" / "yolox_nano.onnx")},
              "store": {"path": str(tmp_path / "s.db")},
              "sounds": {"enabled": True, "model": str(MODEL), "classes": str(CLASSES)}}
    pipe = Pipeline.from_config(config, source="hall", live=False, audio_source=str(path),
                                start_time=1_800_000_000.0)
    assert pipe.sounds is not None and pipe.alerts is not None
    frame = np.zeros((240, 320, 3), np.uint8)
    for i in range(14):
        pipe.process(frame, i * 0.5)
    pipe.close()
    rows = pipe.store._db.execute("SELECT rule FROM events WHERE type = 'sound'").fetchall()
    pipe.store.close()
    assert [r[0] for r in rows] == ["smoke_alarm"]
