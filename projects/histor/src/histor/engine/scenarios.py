"""The scenario bench: real and composed clips run through the whole Pipeline, each scored
against the events it should (and should not) raise, so every change is measured before it
lands. How to run it and what the numbers mean: README, "Scenario bench".

A spec (tests/scenarios/specs/<suite>/<name>.json, tracked) is one clip and what should happen
in it:

    {"name": "street_cover", "suite": "street",
     "compose": {"base": "data/vtest.avi", "seconds": 60,
                 "layers": [{"kind": "cover", "from": 40}]},
     "config": {"tamper": {"hold_seconds": 10}},
     "expect": [{"type": "tamper", "rule": "blocked", "from": 40, "max_delay": 15}]}

    name, suite        the spec (default: the file name) and its suite in suites.yaml
    clip | compose     a project-relative video, or a recipe of layers over one (compose.py)
    config             deep-merged over the suite's config
    expect             events that must come: type (rule, label, note), from, to, max_delay and
                       `early` (seconds: a hit that soon also counts for early recall)
    expect_sightings   search rows that must be stored: kind, from, max_delay, near (a box)
    expect_motion      [low, high] per count of the run's motion search (MOTION_COUNTS):
                       a golden check that the change analysis still finds the same things
    negative           nothing of the scored types may happen (allow_false_alarms: how many may)
    scored_types       event types scored (default: the suite's); others are ignored
    expect_note        a note an alert may carry on this spec (else 'not moving' on a fire
                       positive is a failure: the fire was only caught as a still spot)
    visible_from       when the thing really comes into view (overrides each expect's `from`)
    variants           night looks to run (compose.variant; default: the suite's, else
                       colour); xfail: a known miss, and why
    licence, notes     free text

An expected event is hit by the first unused event of its type (and rule, label, note when
given) at from - 0.5 <= t <= (to or the clip's end) + max_delay; any other event of a scored type
is a false alarm, unless the suite lists it as known (`known_false_alarms`: on that clip, of
that type and rule, between from and to; reported apart, so it is seen but does not fail). An expected sighting: a row of that kind overlapping `near`, at most
max_delay after `from`. Events are read from one place: the alert log (jsonl) of the run, so events
raised when the Pipeline closes are counted too.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import statistics
import sys
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT = Path(__file__).resolve().parents[3]
BENCH = PROJECT / "tests" / "scenarios"
SUITES = BENCH / "suites.yaml"
SPECS = BENCH / "specs"
BASELINES = BENCH / "baseline"
RESULTS = PROJECT / "data" / "scenario_results"
README = PROJECT / "README.md"
TABLE_START, TABLE_END = "<!-- bench:start -->", "<!-- bench:end -->"

SPEC_KEYS = {"name", "suite", "clip", "compose", "config", "expect", "expect_sightings",
             "negative", "scored_types", "allow_false_alarms", "expect_note", "visible_from",
             "variants", "xfail", "licence", "notes", "expect_motion"}
EXPECT_KEYS = {"type", "rule", "label", "note", "from", "to", "max_delay", "early"}
SIGHTING_KEYS = {"kind", "from", "max_delay", "near"}
# expect_motion: blobs reported in all and distinct ones (motion.py), frames taken as light and
# background restarts (change.py)
MOTION_COUNTS = ("reports", "blobs", "light_changes", "restarts")
STILL_NOTE = "not moving"  # fire.py's note for a spot caught only because it stays (S3)
WINDOW_SLACK = 0.5         # an event may come this much before `from`
QUICK_SECONDS = 20.0
# which part of the Pipeline raises each event type: a scored type without it is an error
MODULES = {"fire": "fire", "tamper": "tamper", "abandoned": "abandoned", "down": "pose",
           "hands_up": "pose", "sound": "sounds", "plate": "plates", "enter": "monitor",
           "exit": "monitor", "dwell": "monitor", "cross": "monitor", "crowd": "monitor"}
FIRE_DELAY_GATE = 1.0      # s: a fire hit this much later than the baseline fails --compare
CPU_WARN, CHANGE_CPU_FAIL = 1.3, 1.5  # x the baseline's mean ms a frame, per stage


# ------------------------------------------------------------------ specs and config

def load_suites(path: Path = SUITES) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["suites"]


def load_spec(path: str | Path) -> dict[str, Any]:
    """A spec file, checked, with its defaults filled in (`path`: where it came from)."""
    path = Path(path)
    spec = json.loads(path.read_text(encoding="utf-8"))
    return check_spec(spec, path)


def check_spec(spec: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    where = path.name if path else spec.get("name", "spec")
    unknown = set(spec) - SPEC_KEYS
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}; use {sorted(SPEC_KEYS)}")
    if ("clip" in spec) == ("compose" in spec):
        raise ValueError(f"{where}: give either `clip` or `compose`")
    for e in spec.get("expect", []):
        _check_keys(where, "expect", e, EXPECT_KEYS, {"type", "from", "max_delay"})
    for e in spec.get("expect_sightings", []):
        _check_keys(where, "expect_sightings", e, SIGHTING_KEYS, {"kind", "from", "max_delay"})
    for key, bounds in spec.get("expect_motion", {}).items():
        if key not in MOTION_COUNTS or len(bounds) != 2:
            raise ValueError(f"{where}: expect_motion {key}: {bounds}; use [low, high] for "
                             f"{list(MOTION_COUNTS)}")
    if spec.get("negative") and (spec.get("expect") or spec.get("expect_sightings")):
        raise ValueError(f"{where}: a negative spec expects nothing")
    out = {"config": {}, "expect": [], "expect_sightings": [], "negative": False,
           "allow_false_alarms": 0, **copy.deepcopy(spec)}
    out.setdefault("name", path.stem if path else "spec")
    if "suite" not in out:
        if path is None:
            raise ValueError(f"{where}: no `suite`")
        out["suite"] = path.parent.name
    if "visible_from" in out:
        for e in out["expect"] + out["expect_sightings"]:
            e["from"] = out["visible_from"]
    return out


def _check_keys(where: str, part: str, item: dict[str, Any], allowed: set[str],
                needed: set[str]) -> None:
    if set(item) - allowed or needed - set(item):
        raise ValueError(f"{where}: {part} {item} needs {sorted(needed)}, may have "
                         f"{sorted(allowed)}")


def all_specs(suite: str | None = None, only: str | None = None,
              folder: Path = SPECS) -> list[dict[str, Any]]:
    paths = sorted((folder / suite).glob("*.json") if suite else folder.glob("*/*.json"))
    specs = [load_spec(p) for p in paths]
    return [s for s in specs if not only or only in s["name"]]


def clip_path(spec: dict[str, Any], root: Path = PROJECT) -> Path:
    """The video the spec reads (the base of a composed clip)."""
    return root / (spec["clip"] if "clip" in spec else spec["compose"]["base"])


def merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """`over` deep-merged into a copy of `base` (dicts merge, anything else replaces)."""
    out = copy.deepcopy(base)
    for key, value in over.items():
        out[key] = merge(out[key], value) if isinstance(value, dict) and isinstance(
            out.get(key), dict) else copy.deepcopy(value)
    return out


def set_option(config: dict[str, Any], assignment: str) -> dict[str, Any]:
    """--set fire.min_score=0.3: the value is read as YAML (numbers, lists, true)."""
    key, _, value = assignment.partition("=")
    if not key or not _:
        raise ValueError(f"--set wants key.path=value, got {assignment!r}")
    node: dict[str, Any] = {}
    out = node
    parts = key.split(".")
    for part in parts[:-1]:
        node[part] = {}
        node = node[part]
    node[parts[-1]] = yaml.safe_load(value)
    return merge(config, out)


def _project_paths(value: Any, root: Path) -> Any:
    """Suite configs name files as in configs/ ("models/x.onnx"): made absolute, so a run
    works from any folder."""
    if isinstance(value, dict):
        return {k: _project_paths(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [_project_paths(v, root) for v in value]
    if isinstance(value, str) and value.startswith(("models/", "data/")):
        return str(root / value)
    return value


def spec_config(spec: dict[str, Any], suites: dict[str, Any] | None = None,
                overrides: list[str] | None = None, threads: int = 0,
                root: Path = PROJECT) -> dict[str, Any]:
    """The suite's config, the spec's `config` and the --set overrides, in that order."""
    suites = load_suites() if suites is None else suites
    if spec["suite"] not in suites:
        raise ValueError(f"{spec['name']}: no suite {spec['suite']!r} in {sorted(suites)}")
    config = merge(suites[spec["suite"]]["config"], spec["config"])
    for assignment in overrides or []:
        config = set_option(config, assignment)
    if threads:  # --jobs: the cores shared out between the runs
        for section in ("model", "pose"):
            if isinstance(config.get(section), dict):
                config[section]["num_threads"] = threads
        for section, part in (("fire", "model"), ("search", "clip")):
            if isinstance(config.get(section), dict):
                config[section].setdefault(part, {})["num_threads"] = threads
    return _project_paths(config, root)


def scored_types(spec: dict[str, Any], suites: dict[str, Any]) -> list[str]:
    return spec.get("scored_types") or suites[spec["suite"]]["scored_types"]


def known_false_alarms(spec: dict[str, Any], suites: dict[str, Any]) -> list[dict[str, Any]]:
    """The suite's known false alarms on this spec's clip (composed from its start)."""
    if spec.get("compose", {}).get("start", 0):
        return []
    clip = spec["clip"] if "clip" in spec else spec["compose"]["base"]
    return [k for k in suites[spec["suite"]].get("known_false_alarms", []) if k["clip"] == clip]


def missing_models(config: dict[str, Any]) -> list[str]:
    """Model files the config names that are not installed."""
    found: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for v in value.values():
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)
        elif isinstance(value, str) and value.endswith(".onnx") and not Path(value).exists():
            found.append(Path(value).name)

    walk(config)
    return found


# ------------------------------------------------------------------ running

@dataclass
class RunResult:
    spec: str
    variant: str
    events: list[dict[str, Any]] = field(default_factory=list)     # t, type, rule, label, ...
    sightings: list[dict[str, Any]] = field(default_factory=list)  # t, kind, box
    frames: int = 0
    seconds: float = 0.0            # length of the clip (video time)
    fps: float = 0.0
    timings: dict[str, dict[str, float]] = field(default_factory=dict)  # stage: mean, median, p95 ms
    fire_checks: int = 0
    realtime: bool = False
    skipped: int = 0                # frames dropped by --realtime (the CPU was busy)
    done: list[tuple[float, float]] = field(default_factory=list)  # (t, when it was processed)
    missing: str | None = None      # the clip, when it is not there
    motion: dict[str, int] = field(default_factory=dict)  # MOTION_COUNTS, with motion search on


def frames(spec: dict[str, Any], root: Path = PROJECT,
           seconds: float | None = None) -> Iterator[tuple[np.ndarray, float, float]]:
    """(frame, t, fps) of the spec's clip, the first `seconds` of it if given."""
    if "compose" in spec:
        from histor.engine.compose import Composer

        composer = Composer(spec["compose"], root)
        for frame, t in composer:
            if seconds is not None and t >= seconds:
                return
            yield frame, t, composer.fps
        return
    from histor.engine.stream import iter_frames

    for f in iter_frames(str(root / spec["clip"])):
        if seconds is not None and f.timestamp >= seconds:
            return
        yield f.image, f.timestamp, f.fps


def check_modules(pipe: Any, spec: dict[str, Any], types: list[str]) -> None:
    """Every scored type needs the part of the Pipeline that raises it: a fire negative with
    fire off would pass whatever the code does."""
    absent = sorted({f"{t} ({MODULES[t]})" for t in types
                     if t in MODULES and getattr(pipe, MODULES[t]) is None})
    if (spec.get("expect_motion") or any(e["kind"] == "motion" for e in spec["expect_sightings"])
            ) and (pipe.indexer is None or pipe.indexer.motion is None):
        absent.append("motion sightings (search.motion with the CLIP image model)")
    if absent:
        raise RuntimeError(f"{spec['name']}: scored but not built: {', '.join(absent)}")


def run_spec(spec: dict[str, Any], variant: str = "colour", workdir: str | Path | None = None,
             overrides: list[str] | None = None, quick: bool = False, realtime: bool = False,
             threads: int = 0, suites: dict[str, Any] | None = None,
             root: Path = PROJECT) -> RunResult:
    """One clip through Pipeline.from_config (store and alert log in `workdir`), scored later.

    realtime: like a live camera: a frame that comes while the previous one is still being
    processed is dropped (the delays then include the CPU's lag)."""
    import tempfile

    from histor.engine.compose import variant as night
    from histor.engine.pipeline import Pipeline

    suites = load_suites() if suites is None else suites
    if not clip_path(spec, root).exists():
        return RunResult(spec["name"], variant, missing=str(clip_path(spec, root)))
    own = workdir is None
    work = Path(tempfile.mkdtemp(prefix="bench-") if workdir is None else workdir)
    work.mkdir(parents=True, exist_ok=True)
    log = work / "events.jsonl"
    log.unlink(missing_ok=True)
    config = merge(spec_config(spec, suites, overrides, threads, root),
                   {"store": {"path": str(work / "bench.db")},
                    "alerts": {"console": False, "jsonl": str(log)}})
    pipe = Pipeline.from_config(config, source=spec["name"], live=False, start_time=0.0,
                                timings=True)
    result = RunResult(spec["name"], variant, realtime=realtime)
    try:
        check_modules(pipe, spec, scored_types(spec, suites))
        look = night(variant)
        clock = 0.0  # realtime: when the processor is free again
        for frame, t, fps in frames(spec, root, QUICK_SECONDS if quick else None):
            result.fps, result.seconds = fps, t + 1 / fps
            if realtime and t < clock:
                result.skipped += 1
                continue
            start = time.perf_counter()
            pipe.process(look(frame), t)
            spent = time.perf_counter() - start
            clock = (max(clock, t) if realtime else t) + spent
            result.done.append((t, round(clock, 3)))
            result.frames += 1
            if pipe.fire is not None and pipe.fire.last_check == t:
                result.fire_checks += 1
    finally:
        pipe.close()
        if pipe.indexer is not None and pipe.indexer.motion is not None and \
                pipe.change is not None:
            result.motion = {"reports": pipe.indexer.motion.reports,
                             "blobs": pipe.indexer.motion.blobs,
                             "light_changes": pipe.change.light_changes,
                             "restarts": pipe.change.restarts}
        if pipe.store is not None:
            result.sightings = [{"t": r["time"], "kind": r["kind"],
                                 "box": json.loads(r["box"]) if r["box"] else None}
                                for r in pipe.store._db.execute(
                                    "SELECT time, kind, box FROM sightings ORDER BY time")]
            pipe.store.close()
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            e = json.loads(line)
            e["t"] = e.pop("timestamp")
            e.pop("source", None)
            result.events.append(e)
    result.timings = {stage: {"mean": round(statistics.fmean(ms), 2),
                              "median": round(statistics.median(ms), 2),
                              "p95": round(float(np.percentile(ms, 95)), 2)}
                      for stage, ms in (pipe.timings or {}).items() if ms}
    if own:
        shutil.rmtree(work, ignore_errors=True)
    return result


# ------------------------------------------------------------------ scoring

@dataclass
class Score:
    hits: list[dict[str, Any]] = field(default_factory=list)  # expect (index), t, delay, lag
    missed: list[int] = field(default_factory=list)            # expect indexes
    false_alarms: list[dict[str, Any]] = field(default_factory=list)
    known: list[dict[str, Any]] = field(default_factory=list)  # the suite's known false alarms
    early: list[int] = field(default_factory=list)  # expects with `early` that came that soon
    still: list[int] = field(default_factory=list)  # fire expects hit only by a still spot
    motion: list[str] = field(default_factory=list)  # expect_motion counts out of their range

    def passed(self, spec: dict[str, Any]) -> bool:
        return not self.missed and not self.still and not self.motion and \
            len(self.false_alarms) <= spec["allow_false_alarms"]


def expectations(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """expect + expect_sightings, in one list (scores refer to them by index)."""
    return spec["expect"] + [{**e, "sighting": True} for e in spec["expect_sightings"]]


def _lag(result: RunResult, t: float) -> float:
    """When the frame at `t` had been processed (later events: their own time)."""
    for frame_t, done in result.done:
        if frame_t >= t - 1e-6:
            return done
    return t


def _overlaps(a: list[float] | None, b: list[float] | None) -> bool:
    if a is None or b is None:
        return b is None
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def score(result: RunResult, spec: dict[str, Any], types: list[str],
          known: list[dict[str, Any]] | None = None) -> Score:
    """known: known_false_alarms(spec, suites)."""
    out = Score()
    end = result.seconds
    used: set[int] = set()
    used_sightings: set[int] = set()
    for i, e in enumerate(expectations(spec)):
        until = e["from"] if e.get("sighting") else (e.get("to") or end)
        low, high = e["from"] - WINDOW_SLACK, until + e["max_delay"]
        found = None
        if e.get("sighting"):
            for j, s in enumerate(result.sightings):
                if j not in used_sightings and s["kind"] == e["kind"] and \
                        low <= s["t"] <= high and _overlaps(s["box"], e.get("near")):
                    found = s
                    used_sightings.add(j)
                    break
        else:
            for j, ev in enumerate(result.events):
                if j in used or ev["type"] != e["type"] or not low <= ev["t"] <= high:
                    continue
                if any(k in e and ev.get(k) != e[k] for k in ("rule", "label", "note")):
                    continue
                found = ev
                used.add(j)
                break
        if found is None:
            out.missed.append(i)
            continue
        delay = round(found["t"] - e["from"], 2)
        out.hits.append({"expect": i, "t": found["t"], "delay": delay,
                         "lag": round(_lag(result, found["t"]) - e["from"], 2)})
        if "early" in e and delay <= e["early"]:
            out.early.append(i)
        if found.get("note") == STILL_NOTE and spec.get("expect_note") != STILL_NOTE and \
                e.get("type") == "fire":
            out.still.append(i)
    for key, (low, high) in spec.get("expect_motion", {}).items():
        value = result.motion.get(key)
        if value is None or not low <= value <= high:
            out.motion.append(f"{key} {value} not in [{low}, {high}]")
    for j, ev in enumerate(result.events):
        if j in used or ev["type"] not in types:
            continue
        if any(k["type"] == ev["type"] and k.get("rule", ev["rule"]) == ev["rule"]
               and k["from"] <= ev["t"] <= k["to"] for k in known or []):
            out.known.append(ev)
        else:
            out.false_alarms.append(ev)
    return out


# ------------------------------------------------------------------ summaries and baselines

def run_record(spec: dict[str, Any], result: RunResult, suites: dict[str, Any]
               ) -> dict[str, Any]:
    """One run as saved in results and baselines."""
    record = asdict(result)
    record.pop("done")
    record["sightings"] = sighting_counts(result.sightings)  # the rows stay in the run's store
    if result.missing:
        return record
    s = score(result, spec, scored_types(spec, suites), known_false_alarms(spec, suites))
    expects = expectations(spec)
    for h in s.hits:  # the delay gate applies to fire expects only
        h["fire"] = expects[h["expect"]].get("type") == "fire"
    record["score"] = {**asdict(s), "passed": s.passed(spec), "expects": len(expects),
                       "early_expects": sum(1 for e in expects if "early" in e)}
    return record


def sighting_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["kind"]] = counts.get(row["kind"], 0) + 1
    return counts


def _median(values: list[float]) -> float | None:
    return round(statistics.median(values), 2) if values else None


def summary(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per variant: recall, early recall, misses, false alarms, delays, checks and CPU."""
    out: dict[str, dict[str, Any]] = {}
    for variant in dict.fromkeys(r["variant"] for r in records):
        runs = [r for r in records if r["variant"] == variant and not r.get("missing")]
        expects = sum(r["score"]["expects"] for r in runs)
        hits = [h for r in runs for h in r["score"]["hits"]]
        fa = sum(len(r["score"]["false_alarms"]) for r in runs)
        hours = sum(r["seconds"] for r in runs) / 3600
        minutes = sum(r["seconds"] for r in runs) / 60
        stages = sorted({s for r in runs for s in r["timings"]})
        out[variant] = {
            "specs": len(runs),
            "missing": [r["spec"] for r in records if r["variant"] == variant and r.get("missing")],
            "recall": round(len(hits) / expects, 3) if expects else None,
            "early": sum(len(r["score"]["early"]) for r in runs),
            "early_expects": sum(r["score"]["early_expects"] for r in runs),
            "missed": [f"{r['spec']}#{i}" for r in runs for i in r["score"]["missed"]],
            "still_only": sum(len(r["score"]["still"]) for r in runs),
            "false_alarms": fa,
            "known_false_alarms": sum(len(r["score"]["known"]) for r in runs),
            "false_alarms_per_hour": round(fa / hours, 1) if hours else None,
            "delay_median": _median([h["delay"] for h in hits]),
            "delay_max": max((h["delay"] for h in hits), default=None),
            "lag_median": _median([h["lag"] for h in hits]),
            "fire_checks_per_minute": round(sum(r["fire_checks"] for r in runs) / minutes, 1)
            if minutes else None,
            "ms": {s: _median([r["timings"][s]["mean"] for r in runs if s in r["timings"]])
                   for s in stages},  # CPU ms a frame: the median clip's mean
        }
    return out


def compare(records: list[dict[str, Any]], baseline: dict[str, Any],
            allow_fa: bool = False, gate_lag: bool = False) -> tuple[list[str], list[str]]:
    """(failures, warnings) of these runs against a saved baseline. Fails: a hit the baseline
    had is missed, a fire alert more than FIRE_DELAY_GATE s later (the lag when `gate_lag`), a
    fire caught only as a still spot where it was not before, a new false alarm (unless
    `allow_fa`), a baseline spec whose clip is missing, the change stage CHANGE_CPU_FAIL x
    slower (mean ms a frame). Warns: any other stage CPU_WARN x slower."""
    old = {(r["spec"], r["variant"]): r for r in baseline["runs"]}
    fails: list[str] = []
    warns: list[str] = []
    for r in records:
        key = (r["spec"], r["variant"])
        name = f"{r['spec']} [{r['variant']}]"
        base = old.get(key)
        if base is None:
            warns.append(f"{name}: not in the baseline")
            continue
        if r.get("missing"):
            fails.append(f"{name}: clip missing ({r['missing']})")
            continue
        if base.get("missing"):
            continue
        now, then = r["score"], base["score"]
        hits_now = {h["expect"]: h for h in now["hits"]}
        for h in then["hits"]:
            if h["expect"] not in hits_now:
                fails.append(f"{name}: expect #{h['expect']} was hit at {h['t']} s, now missed")
                continue
            if h.get("fire"):
                key_delay = "lag" if gate_lag else "delay"
                grew = hits_now[h["expect"]][key_delay] - h[key_delay]
                if grew > FIRE_DELAY_GATE:
                    fails.append(f"{name}: fire expect #{h['expect']} {grew:+.1f} s later "
                                 f"({key_delay} {h[key_delay]} -> "
                                 f"{hits_now[h['expect']][key_delay]})")
        if now.get("motion") and not then.get("motion"):
            fails.append(f"{name}: motion counts {now['motion']}")
        if set(now["still"]) - set(then["still"]):
            fails.append(f"{name}: fire caught only as a still spot ('{STILL_NOTE}')")
        if len(now["false_alarms"]) > len(then["false_alarms"]) and not allow_fa:
            fails.append(f"{name}: false alarms {len(then['false_alarms'])} -> "
                         f"{len(now['false_alarms'])}")
        for stage, t in r["timings"].items():
            before = (base["timings"].get(stage) or {}).get("mean")
            if not before or before < 0.5:  # sub-millisecond stages: noise
                continue
            ratio = t["mean"] / before
            if stage == "change" and ratio > CHANGE_CPU_FAIL:
                fails.append(f"{name}: change stage {before} -> {t['mean']} ms a frame")
            elif ratio > CPU_WARN:
                warns.append(f"{name}: {stage} {before} -> {t['mean']} ms a frame")
    return fails, warns


def markdown(baselines: dict[str, dict[str, Any]]) -> str:
    """The README table: one row per suite and night variant."""
    rows = ["| Suite | Look | Clips | Recall | Early | Missed | False alarms | "
            "Delay median / max (s) | Fire checks/min | Detect ms | Change ms | Fire ms |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for suite, base in baselines.items():
        for variant, s in base["summary"].items():
            recall = "-" if s["recall"] is None else f"{s['recall']:.0%}"
            early = f"{s['early']}/{s['early_expects']}" if s["early_expects"] else "-"
            missed = ", ".join(s["missed"]) or "none"
            fa = f"{s['false_alarms']}" + (f" (+{s['known_false_alarms']} known)"
                                           if s["known_false_alarms"] else "")
            delay = "-" if s["delay_median"] is None else f"{s['delay_median']} / {s['delay_max']}"
            ms = {k: f"{v:.1f}" if v < 10 else f"{v:.0f}" for k, v in s["ms"].items()}
            rows.append(f"| {suite} | {variant} | {s['specs']} | {recall} | {early} | {missed} "
                        f"| {fa} | {delay} | {s['fire_checks_per_minute']:.0f} | "
                        f"{ms.get('detect', '-')} | {ms.get('change', '-')} | "
                        f"{ms.get('fire', '-')} |")
    return "\n".join(rows)


def write_readme_table(table: str, readme: Path = README) -> None:
    text = readme.read_text(encoding="utf-8")
    start, end = text.index(TABLE_START), text.index(TABLE_END)
    readme.write_text(text[:start + len(TABLE_START)] + "\n" + table + "\n" + text[end:],
                      encoding="utf-8")


# ------------------------------------------------------------------ command line

def _job(args: tuple[dict[str, Any], str, dict[str, Any]]) -> dict[str, Any]:
    spec, variant, options = args
    suites = load_suites()
    result = run_spec(spec, variant, suites=suites, **options)
    return run_record(spec, result, suites)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m histor.scenarios", description=(
        "Score the engine on the bench clips (tests/scenarios). README: 'Scenario bench'."))
    ap.add_argument("--suite", help="fire | street | pose (default: all)")
    ap.add_argument("--only", help="only specs whose name contains this")
    ap.add_argument("--variants", help="colour,grey,low,ir,dark (default: each spec's own)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY.PATH=VALUE",
                    help="config override for every run, e.g. fire.min_score=0.3")
    ap.add_argument("--jobs", type=int, default=1, help="runs in parallel (cores shared out)")
    ap.add_argument("--quick", action="store_true", help=f"first {QUICK_SECONDS:.0f} s only")
    ap.add_argument("--realtime", action=argparse.BooleanOptionalAction, default=None,
                    help="drop frames while busy, like a live camera (default: the suite's)")
    ap.add_argument("--out", type=Path, help="results json (default: data/scenario_results/)")
    ap.add_argument("--compare", type=Path, help="a baseline json: exit 1 on a regression")
    ap.add_argument("--allow-fa", action="store_true", help="new false alarms only warn")
    ap.add_argument("--save-baseline", action="store_true",
                    help="write tests/scenarios/baseline/<suite>.json")
    ap.add_argument("--markdown", action="store_true",
                    help="print the table of the saved baselines and exit")
    ap.add_argument("--readme", action="store_true",
                    help="with --markdown: write the table into README.md too")
    args = ap.parse_args(argv)

    if args.markdown:
        baselines = {p.stem: json.loads(p.read_text(encoding="utf-8"))
                     for p in sorted(BASELINES.glob("*.json"))}
        table = markdown(baselines)
        print(table)
        if args.readme:
            write_readme_table(table)
        return 0
    suites = load_suites()
    specs = all_specs(args.suite, args.only)
    if not specs:
        print("no specs match", file=sys.stderr)
        return 2
    variants = args.variants.split(",") if args.variants else None
    threads = max((os.cpu_count() or 2) // args.jobs, 1) if args.jobs > 1 else 0
    jobs = []
    for spec in specs:
        realtime = suites[spec["suite"]].get("realtime", False) if args.realtime is None \
            else args.realtime
        options = {"overrides": args.set, "quick": args.quick, "realtime": realtime,
                   "threads": threads}
        for variant in variants or spec.get("variants") or                 suites[spec["suite"]].get("variants") or ["colour"]:
            jobs.append((spec, variant, options))
    started = time.perf_counter()
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as pool:
            records = list(pool.map(_job, jobs))
    else:
        records = [_job(j) for j in jobs]
    by_name = {s["name"]: s for s in specs}
    out: dict[str, Any] = {}
    for suite in dict.fromkeys(s["suite"] for s in specs):
        mine = [r for r in records if by_name[r["spec"]]["suite"] == suite]
        out[suite] = {"suite": suite, "made": datetime.now().isoformat(timespec="seconds"),
                      "quick": args.quick, "set": args.set, "runs": mine,
                      "summary": summary(mine)}
    path = args.out or RESULTS / f"{args.suite or 'all'}-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out if len(out) > 1 else next(iter(out.values())), indent=1),
                    encoding="utf-8")
    for suite, data in out.items():
        for variant, s in data["summary"].items():
            print(f"{suite} [{variant}] {s['specs']} clips: recall {s['recall']}, early "
                  f"{s['early']}/{s['early_expects']}, missed {s['missed'] or 'none'}, "
                  f"false alarms {s['false_alarms']} ({s['false_alarms_per_hour']}/h, "
                  f"known {s['known_false_alarms']}), delay "
                  f"median {s['delay_median']} max {s['delay_max']} (lag {s['lag_median']}), "
                  f"fire checks/min {s['fire_checks_per_minute']}, ms {s['ms']}"
                  + (f", MISSING {s['missing']}" if s["missing"] else ""))
        for r in data["runs"]:
            if not r.get("missing") and not r["score"]["passed"]:
                print(f"  {r['spec']} [{r['variant']}]: missed {r['score']['missed']}, "
                      f"false alarms {[(e['t'], e['type'], e['rule']) for e in r['score']['false_alarms']]}"
                      + (f", still only {r['score']['still']}" if r["score"]["still"] else "")
                      + (f", motion {r['score']['motion']}" if r["score"]["motion"] else ""))
    print(f"{len(records)} runs in {time.perf_counter() - started:.0f} s -> {path}")
    if args.save_baseline:
        BASELINES.mkdir(parents=True, exist_ok=True)
        for suite, data in out.items():
            (BASELINES / f"{suite}.json").write_text(json.dumps(data, indent=1) + "\n",
                                                     encoding="utf-8")
            print(f"baseline saved: {BASELINES / f'{suite}.json'}")
    if args.compare:
        baseline = json.loads(args.compare.read_text(encoding="utf-8"))
        suite = baseline["suite"]
        mine = out.get(suite, {"runs": []})["runs"]
        fails, warns = compare(mine, baseline, args.allow_fa,
                               gate_lag=bool(suites[suite].get("realtime")))
        for w in warns:
            print(f"WARN {w}")
        for f in fails:
            print(f"FAIL {f}")
        if fails:
            return 1
        print(f"compare: no regression against {args.compare}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
