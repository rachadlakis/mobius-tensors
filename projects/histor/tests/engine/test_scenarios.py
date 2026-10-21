"""The scenario bench (engine/scenarios.py): specs, scoring, baselines and runs, without the
real models (the scored runs themselves are tests/scenarios, marker `scenario`)."""

from __future__ import annotations

import importlib.util
import json
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from histor.engine import scenarios
from histor.engine.pipeline import Pipeline
from histor.engine.scenarios import (RunResult, check_spec, compare, run_record, run_spec,
                                     score)
from kit import Scripted, fire_box

ROOT = Path(__file__).resolve().parents[2]


def spec(**fields) -> dict:
    return check_spec({"name": "s", "suite": "t", "clip": "x.mp4", **fields})


def ev(t: float, type_: str = "fire", rule: str = "fire", **more) -> dict:
    return {"t": t, "type": type_, "rule": rule, "label": rule, **more}


# ------------------------------------------------------------------ specs and scoring

def test_score_hit_miss_false_alarm_delay():
    s = spec(expect=[{"type": "fire", "rule": "fire", "from": 9, "to": 9, "max_delay": 3},
                     {"type": "fire", "rule": "smoke", "from": 24, "to": 24, "max_delay": 3}])
    result = RunResult("s", "colour", events=[ev(10.2), ev(15.0), ev(30.0, rule="smoke")],
                       seconds=39)
    out = score(result, s, ["fire"])
    assert [(h["expect"], h["delay"]) for h in out.hits] == [(0, 1.2)]
    assert out.missed == [1]
    assert [e["t"] for e in out.false_alarms] == [15.0, 30.0]  # smoke 3 s too late is one too
    assert not out.passed(s)


def test_score_matches_rule_and_note_and_window_edges():
    s = spec(expect=[{"type": "tamper", "rule": "moved", "from": 10, "to": 10, "max_delay": 5},
                     {"type": "fire", "note": "not moving", "from": 0, "max_delay": 0}])
    edge = RunResult("s", "colour", seconds=20, events=[
        ev(9.5, "tamper", "blocked"),                  # wrong rule
        ev(9.5, "tamper", "moved"),                    # from - 0.5: allowed
        ev(20.0, "fire", note="not moving")])          # the clip's end + 0: allowed
    out = score(edge, s, ["tamper", "fire"])
    assert [(h["expect"], h["t"]) for h in out.hits] == [(0, 9.5), (1, 20.0)]
    assert [e["rule"] for e in out.false_alarms] == ["blocked"]
    late = RunResult("s", "colour", seconds=20, events=[ev(15.01, "tamper", "moved"),
                                                       ev(9.49, "tamper", "moved"),
                                                       ev(5.0, "fire")])  # no note
    out = score(late, s, ["tamper", "fire"])
    assert out.missed == [0, 1] and len(out.false_alarms) == 3
    exact = score(RunResult("s", "colour", seconds=20, events=[ev(15.0, "tamper", "moved")]),
                  s, ["tamper"])
    assert exact.hits[0]["delay"] == 5.0  # to + max_delay: inclusive


def test_negative_spec_counts_every_scored_event_as_false_alarm():
    s = spec(negative=True, allow_false_alarms=1)
    result = RunResult("s", "colour", seconds=60, events=[
        ev(3, "fire"), ev(4, "tamper", "blocked"), ev(5, "enter", "door")])
    out = score(result, s, ["fire", "tamper"])
    assert [e["type"] for e in out.false_alarms] == ["fire", "tamper"]  # `enter` not scored
    assert not out.passed(s)
    assert score(RunResult("s", "colour", seconds=60, events=[ev(3)]), s, ["fire"]).passed(s)


def test_still_note_on_a_fire_positive_fails_unless_expected():
    expect = [{"type": "fire", "from": 0, "max_delay": 30}]
    result = RunResult("s", "colour", seconds=30, events=[ev(20, note="not moving")])
    out = score(result, spec(expect=expect), ["fire"])
    assert out.still == [0] and not out.passed(spec(expect=expect))
    poster = spec(expect=expect, expect_note="not moving")
    assert score(result, poster, ["fire"]).passed(poster)


def test_known_false_alarms_are_reported_apart():
    suites = {"t": {"scored_types": ["fire"], "known_false_alarms": [
        {"clip": "x.mp4", "type": "fire", "rule": "fire", "from": 22, "to": 23.5, "why": "spot"}]}}
    s = spec(negative=True)
    known = scenarios.known_false_alarms(s, suites)
    result = RunResult("s", "colour", seconds=60, events=[ev(22.6), ev(22.7, rule="smoke"),
                                                         ev(40)])
    out = score(result, s, ["fire"], known)
    assert [e["t"] for e in out.known] == [22.6]
    assert [e["t"] for e in out.false_alarms] == [22.7, 40]
    assert scenarios.known_false_alarms(spec(clip="y.mp4"), suites) == []
    later = check_spec({"name": "s", "suite": "t", "compose": {"base": "x.mp4", "start": 10}})
    assert scenarios.known_false_alarms(later, suites) == []  # other times


def test_sightings_need_kind_time_and_overlap():
    s = spec(expect_sightings=[{"kind": "motion", "from": 20, "max_delay": 1,
                                "near": [0, 330, 768, 400]}])
    rows = [{"t": 20.4, "kind": "scene", "box": [0, 0, 768, 576]},
            {"t": 20.5, "kind": "motion", "box": [10, 10, 60, 60]},    # not near the cart
            {"t": 20.6, "kind": "motion", "box": [100, 340, 160, 380]}]
    out = score(RunResult("s", "colour", seconds=30, sightings=rows), s, ["fire"])
    assert out.hits[0]["t"] == 20.6
    late = [{"t": 21.2, "kind": "motion", "box": [100, 340, 160, 380]}]
    assert score(RunResult("s", "colour", seconds=30, sightings=late), s, ["fire"]).missed == [0]


def test_spec_schema_rejects_unknown_keys_and_missing_clip_or_compose():
    with pytest.raises(ValueError, match="unknown keys"):
        spec(expected=[])
    with pytest.raises(ValueError, match="either `clip` or `compose`"):
        check_spec({"name": "s", "suite": "t"})
    with pytest.raises(ValueError, match="either `clip` or `compose`"):
        check_spec({"name": "s", "suite": "t", "clip": "a", "compose": {"base": "a"}})
    with pytest.raises(ValueError, match="needs"):
        spec(expect=[{"type": "fire", "from": 0}])            # no max_delay
    with pytest.raises(ValueError, match="needs"):
        spec(expect=[{"type": "fire", "from": 0, "max_delay": 1, "when": 3}])
    with pytest.raises(ValueError, match="expects nothing"):
        spec(negative=True, expect=[{"type": "fire", "from": 0, "max_delay": 1}])
    assert spec(visible_from=4, expect=[{"type": "fire", "from": 0, "max_delay": 1}]
                )["expect"][0]["from"] == 4


def test_every_tracked_spec_loads():
    specs = scenarios.all_specs()
    suites = scenarios.load_suites()
    assert {s["suite"] for s in specs} <= set(suites)
    assert {"fire_loop", "street_quiet", "street_cart", "pose_fall_sprite"} <= {
        s["name"] for s in specs}
    for s in specs:  # every scored type has a module in its suite's config
        config = scenarios.spec_config(s, suites)
        for t in scenarios.scored_types(s, suites):
            assert config.get(scenarios.MODULES[t]), (s["name"], t)


def test_config_merges_suite_spec_and_set_and_makes_paths_absolute():
    suites = {"t": {"scored_types": ["fire"],
                    "config": {"model": {"path": "models/a.onnx"},
                               "fire": {"enabled": True, "min_score": 0.25}}}}
    s = spec(config={"fire": {"hits": 2}})
    config = scenarios.spec_config(s, suites, ["fire.min_score=0.3", "tamper.enabled=true"],
                                   threads=3)
    assert config["fire"] == {"enabled": True, "min_score": 0.3, "hits": 2,
                              "model": {"num_threads": 3}}
    assert config["model"] == {"path": str(ROOT / "models" / "a.onnx"), "num_threads": 3}
    assert config["tamper"] == {"enabled": True}
    assert suites["t"]["config"]["fire"] == {"enabled": True, "min_score": 0.25}  # untouched


# ------------------------------------------------------------------ baselines

def record(hits, false_alarms=0, still=(), missing=None, fire_ms=10.0, change_ms=None):
    timings = {"fire": {"mean": fire_ms}} | ({"change": {"mean": change_ms}} if change_ms else {})
    return {"spec": "s", "variant": "colour", "missing": missing, "timings": timings,
            "score": {"hits": [{"expect": i, "t": d, "delay": d, "lag": d + 0.2, "fire": True}
                               for i, d in hits],
                      "false_alarms": [{}] * false_alarms, "still": list(still)}}


def test_compare_flags_lost_fire_hit_and_delay_growth_and_new_false_alarm():
    baseline = {"suite": "fire", "runs": [record([(0, 0.5), (1, 0.8)], change_ms=2.0)]}
    assert compare([record([(0, 1.4), (1, 0.8)], change_ms=2.5)], baseline) == ([], [])
    fails, _ = compare([record([(0, 1.6)], false_alarms=1)], baseline)
    assert any("#1 was hit" in f for f in fails)
    assert any("+1.1 s later" in f for f in fails)
    assert any("false alarms 0 -> 1" in f for f in fails)
    assert compare([record([(0, 0.5), (1, 0.8)], false_alarms=1)], baseline, allow_fa=True)[0] == []
    fails, _ = compare([record([(0, 0.5), (1, 0.8)], still=[0])], baseline)
    assert fails == ["s [colour]: fire caught only as a still spot ('not moving')"]
    fails, _ = compare([record([], missing="data/x.mp4")], baseline)
    assert fails == ["s [colour]: clip missing (data/x.mp4)"]
    fails, warns = compare([record([(0, 0.5), (1, 0.8)], fire_ms=14.0, change_ms=3.2)], baseline)
    assert fails == ["s [colour]: change stage 2.0 -> 3.2 ms a frame"]
    assert warns == ["s [colour]: fire 10.0 -> 14.0 ms a frame"]


def test_compare_cli_exits_1_on_a_regression(tmp_path, monkeypatch):
    s = spec(expect=[{"type": "fire", "from": 1, "to": 1, "max_delay": 1}])
    monkeypatch.setattr(scenarios, "all_specs", lambda suite, only: [s])
    suites = {"t": {"scored_types": ["fire"]}}
    monkeypatch.setattr(scenarios, "load_suites", lambda: suites)
    monkeypatch.setattr(scenarios, "_job", lambda job: run_record(
        s, RunResult("s", "colour", seconds=5, events=[ev(3.5)]), suites))
    baseline = tmp_path / "base.json"
    baseline.write_text(json.dumps({"suite": "t", "runs": [run_record(
        s, RunResult("s", "colour", seconds=5, events=[ev(1.2)]), suites)]}))
    out = tmp_path / "r.json"
    assert scenarios.main(["--compare", str(baseline), "--out", str(out)]) == 1
    assert json.loads(out.read_text())["summary"]["colour"]["missed"] == ["s#0"]


# ------------------------------------------------------------------ runs

@pytest.fixture
def still_clip(tmp_path) -> Path:
    cv2.imwrite(str(tmp_path / "base.png"), np.full((120, 160, 3), 90, np.uint8))
    return tmp_path


def fake_models(monkeypatch, detector, fire_detector) -> None:
    """The Pipeline's object detector and the fire model, without the model files."""
    monkeypatch.setattr("histor.engine.pipeline.create_detector", lambda config: detector)
    monkeypatch.setattr("histor.engine.fire.create_detector", lambda config: fire_detector)


SUITE = {"t": {"scored_types": ["fire"],
               "config": {"model": {"path": "none.onnx"}, "fire": {"enabled": True}}}}


def test_run_spec_with_scripted_detector_and_composed_clip(still_clip, monkeypatch):
    s = check_spec({"name": "s", "suite": "t", "expect": [
        {"type": "fire", "rule": "fire", "from": 1, "to": 1, "max_delay": 1}],
        "compose": {"base": "base.png", "seconds": 4, "fps": 10,
                    "layers": [{"kind": "smoke", "at": [0.5, 0.9], "from": 1}]}})
    runs = []
    for _ in range(2):
        fake_models(monkeypatch, Scripted(lambda i: []),
                    Scripted(lambda i: [fire_box()] if i >= 2 else []))
        runs.append(run_spec(s, suites=SUITE, root=still_clip, workdir=still_clip / "w"))
    a, b = runs
    assert a.frames == 40 and a.seconds == pytest.approx(4.0) and a.missing is None
    assert [(e["t"], e["type"], e["rule"], e["score"]) for e in a.events] == \
        [(1.5, "fire", "fire", 0.8)] == [(e["t"], e["type"], e["rule"], e["score"]) for e in b.events]
    assert a.fire_checks > 8 and {"detect", "fire"} <= set(a.timings)
    assert score(a, s, ["fire"]).passed(s)


def test_missing_clip_and_unbuilt_module(still_clip, monkeypatch):
    gone = check_spec({"name": "g", "suite": "t", "clip": "data/none.mp4"})
    assert run_spec(gone, suites=SUITE, root=still_clip).missing.endswith("none.mp4")
    fake_models(monkeypatch, Scripted(lambda i: []), Scripted(lambda i: []))
    tamper = check_spec({"name": "s", "suite": "t", "negative": True, "scored_types": ["tamper"],
                         "compose": {"base": "base.png", "seconds": 1}})
    with pytest.raises(RuntimeError, match=r"tamper \(tamper\)"):
        run_spec(tamper, suites=SUITE, root=still_clip)


def test_pose_event_raised_at_close_is_scored(still_clip, monkeypatch):
    """Pipeline.close() asks the pose indexer for its last events; they go to the alert log,
    which is where the bench reads events from."""
    class LatePose:
        alert_types = {"down"}

        def __init__(self):
            self.closed = False

        def update(self, frame, tracks, t, hires=None):
            pass

        def take_events(self):
            return [(2.5, 1, "down", 10.0)] if self.closed else []

        def close(self):
            self.closed = True

    monkeypatch.setattr("histor.engine.pose_index.PoseIndexer.from_config",
                        lambda *a, **k: LatePose())
    fake_models(monkeypatch, Scripted(lambda i: []), Scripted(lambda i: []))
    suites = {"p": {"scored_types": ["down"],
                    "config": {"model": {"path": "none.onnx"}, "pose": {"enabled": True}}}}
    s = check_spec({"name": "s", "suite": "p", "expect": [{"type": "down", "from": 2,
                                                           "max_delay": 1}],
                    "compose": {"base": "base.png", "seconds": 2}})
    result = run_spec(s, suites=suites, root=still_clip)
    assert [(e["t"], e["type"]) for e in result.events] == [(2.5, "down")]
    assert score(result, s, ["down"]).passed(s)


def test_realtime_drops_frames_while_busy(still_clip, monkeypatch):
    class Slow:
        def detect(self, frame):
            time.sleep(0.25)
            return []

    fake_models(monkeypatch, Slow(), Scripted(lambda i: []))
    s = check_spec({"name": "s", "suite": "t", "negative": True,
                    "compose": {"base": "base.png", "seconds": 1}})
    result = run_spec(s, suites=SUITE, root=still_clip, realtime=True)
    assert result.frames + result.skipped == 10 and 3 <= result.frames <= 5
    assert all(done >= t + 0.25 for t, done in result.done)


def test_timings_recorded_only_when_enabled():
    off = Pipeline(Scripted(lambda i: []))
    on = Pipeline(Scripted(lambda i: []), timings=True)
    frame = np.zeros((8, 8, 3), np.uint8)
    for i in range(3):
        off.process(frame, i)
        on.process(frame, i)
    assert off.timings is None
    assert list(on.timings) == ["detect"] and len(on.timings["detect"]) == 3


# ------------------------------------------------------------------ FIRESENSE

def get_firesense():
    path = ROOT / "training" / "fire_smoke" / "get_firesense.py"
    module_spec = importlib.util.spec_from_file_location("get_firesense", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["../x.avi", "/abs.avi", "C:/x.avi", "a/../../x.avi",
                                  "a\\..\\..\\x.avi"])
def test_safe_extract_rejects_traversal(tmp_path, name):
    fs = get_firesense()
    archive = tmp_path / "z.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("fine/pos/a.avi", b"ok")
        z.writestr(name, b"bad")
    with pytest.raises(ValueError, match="unsafe path"):
        fs.safe_extract(archive, tmp_path / "out")
    assert not (tmp_path / "out").exists() and not (tmp_path / "x.avi").exists()
    good = tmp_path / "g.zip"
    with zipfile.ZipFile(good, "w") as z:
        z.writestr("fire/pos/a.avi", b"ok")
    assert fs.safe_extract(good, tmp_path / "ok") == [tmp_path / "ok" / "fire" / "pos" / "a.avi"]
    with pytest.raises(FileExistsError):
        fs.safe_extract(good, tmp_path / "ok")  # only into a new, empty folder


def test_make_specs_from_layout(tmp_path):
    fs = get_firesense()
    folder = tmp_path / "data" / "test_videos" / "firesense"
    for rel in ("fire/Fire videos/pos/posVideo1.869.avi", "fire/Fire videos/neg/negVideo2.avi",
                "smoke/smoke videos/neg/negsVideo 3.avi", "smoke/smoke videos/pos/readme.txt"):
        (folder / rel).parent.mkdir(parents=True, exist_ok=True)
        (folder / rel).write_bytes(b"")
    out = tmp_path / "specs"
    out.mkdir()
    (out / "firesense_old.json").write_text("{}")
    written = fs.make_specs(folder, out, tmp_path)
    assert sorted(p.name for p in out.iterdir()) == sorted(p.name for p in written) == [
        "firesense_fire_neg_negVideo2.json", "firesense_fire_pos_posVideo1.869.json",
        "firesense_smoke_neg_negsVideo_3.json"]
    pos = check_spec(json.loads((out / "firesense_fire_pos_posVideo1.869.json").read_text()),
                     out / "firesense_fire_pos_posVideo1.869.json")
    assert pos["clip"] == "data/test_videos/firesense/fire/Fire videos/pos/posVideo1.869.avi"
    assert pos["expect"] == [{"type": "fire", "from": 0.0, "max_delay": 30.0, "early": 3.0}]
    assert pos["suite"] == "specs" and not pos["negative"]
    neg = json.loads((out / "firesense_smoke_neg_negsVideo_3.json").read_text())
    assert neg["negative"] is True and "expect" not in neg
