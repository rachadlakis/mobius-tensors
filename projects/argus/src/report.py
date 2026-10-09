"""Periodic reports: an hourly, daily or weekly summary of what the cameras saw.

    uv run src/report.py                     # run the schedules in `reports:` (keep it running)
    uv run src/report.py --now morning       # make the report of the last period now, and send it
    uv run src/report.py --list              # saved reports
    uv run src/report.py --show morning      # print the latest saved one

Each report (configs/default.yaml, `reports:`) names its period, its cameras and what the owner
cares about ("vehicles at the gate after hours"). At the end of each period Argus:

1. counts what happened, exactly, from the store: tracked objects per camera, alerts, cameras
   that recorded nothing (offline?);
2. asks the chat model to check each interest with the same tools as the chat, limited to the
   report's cameras and period, and to write a short summary;
3. saves both under data/store/reports/<name>/ (read them later: UI *Reports* tab, --show) and
   sends them to the report's channels (webhook / email, as for alerts).

If the model is down, the report is still saved and sent with the exact counts only. A period
that ended while this was not running is made on the next start (only the latest one).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, time as dtime, timedelta
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

import shared  # noqa: F401
from alerts import make_channel
from ask import build_engine
from chat import answer, model_config, system_prompt
from llm import create_backend
from search import SearchEngine
from tools import Toolbox

log = logging.getLogger("argus.report")

PERIODS = {"hourly": timedelta(hours=1), "daily": timedelta(days=1), "weekly": timedelta(days=7)}
DAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
MAX_IMAGES = 4  # thumbnails attached to an email

PROMPT = """Write the {every} report for {period}.
Cameras in this report: {cameras}.
{interests}
Exact counts for this period (from the store):
{facts}

Check the details with your tools, always with since="{since}" and until="{until}".
Write for the site's owner, in plain sentences, under {words} words:
1. What they asked to watch for: one line each, found or not, with camera and time.
2. Anything else worth knowing: alerts, activity outside usual hours, a camera that recorded nothing.
3. One line on normal activity (counts, busy times).
Only what the tools returned: no guesses presented as facts."""


@dataclass
class ReportSpec:
    name: str
    every: str = "daily"
    at: dtime = dtime(7, 0)       # daily / weekly: when a period ends
    day: int = 0                  # weekly: 0 = monday
    cameras: list[str] = field(default_factory=list)  # empty = every camera
    interests: list[str] = field(default_factory=list)
    model: str | None = None      # chat.models entry; None = chat.model
    channels: list[dict[str, Any]] = field(default_factory=list)
    delay_minutes: float = 2      # wait after the period for the recorders' last rows
    words: int = 200

    @classmethod
    def from_config(cls, d: dict[str, Any]) -> ReportSpec:
        if not d.get("name"):
            raise ValueError("Each report needs a name")
        every = str(d.get("every", "daily")).lower()
        if every not in PERIODS:
            raise ValueError(f"Report {d['name']}: every must be {', '.join(PERIODS)}")
        day = str(d.get("day", "monday")).lower()
        if day not in DAYS:
            raise ValueError(f"Report {d['name']}: day must be one of {', '.join(DAYS)}")
        interests = d.get("interests") or []
        return cls(name=str(d["name"]), every=every,
                   at=dtime.fromisoformat(str(d.get("at", "07:00"))), day=DAYS.index(day),
                   cameras=[str(c) for c in d.get("cameras") or []],
                   interests=[interests] if isinstance(interests, str) else list(interests),
                   model=d.get("model"), channels=list(d.get("channels") or []),
                   delay_minutes=float(d.get("delay_minutes", 2)),
                   words=int(d.get("words", 200)))

    def period_end(self, now: datetime) -> datetime:
        """End of the latest period that is over (and past the delay) at `now`."""
        ready = now - timedelta(minutes=self.delay_minutes)
        if self.every == "hourly":
            return ready.replace(minute=0, second=0, microsecond=0)
        end = datetime.combine(ready.date(), self.at)
        if self.every == "weekly":
            end -= timedelta(days=(end.weekday() - self.day) % 7)
        return end if end <= ready else end - PERIODS[self.every]

    def period(self, end: datetime) -> tuple[datetime, datetime]:
        return end - PERIODS[self.every], end

    def next_run(self, now: datetime) -> datetime:
        return self.period_end(now) + PERIODS[self.every] + timedelta(minutes=self.delay_minutes)


def load_specs(config: dict[str, Any]) -> list[ReportSpec]:
    specs = [ReportSpec.from_config(d) for d in config.get("reports") or []]
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"Report names must be unique: {names}")
    return specs


def _fmt(t: datetime) -> str:
    return t.strftime("%Y-%m-%d %H:%M")


def period_text(start: datetime, end: datetime) -> str:
    if end - start == timedelta(hours=1):
        return f"{start:%a %d %b %H:%M}-{end:%H:%M}"
    return f"{start:%a %d %b %H:%M} to {end:%a %d %b %H:%M}"


class Reporter:
    def __init__(self, config: dict[str, Any], engine: SearchEngine | None = None) -> None:
        self.config = config
        self.specs = {s.name: s for s in load_specs(config)}
        self.engine = engine or build_engine(config)
        # Its own toolbox: the camera limit must not leak into a chat sharing the engine.
        self.toolbox = Toolbox(self.engine, cameras=config.get("cameras"),
                               faces_config=config.get("faces"), pose_config=config.get("pose"))
        store_cfg = config.get("store") or {}
        self.dir = Path(store_cfg.get("path", "data/store/argus.db")).parent / "reports"
        try:  # fail at start, not at 07:00 tomorrow
            for spec in self.specs.values():
                spec.cameras = [self.toolbox._camera(c) for c in spec.cameras]
                for channel in spec.channels:
                    make_channel(channel)
        except Exception:
            self.toolbox.close()
            raise

    def close(self) -> None:
        self.toolbox.close()

    # ------------------------------------------------------------------ saved reports

    def path(self, name: str, end: datetime) -> Path:
        return self.dir / name / f"{end:%Y-%m-%d_%H%M}.json"

    def saved(self, name: str | None = None) -> list[Path]:
        """Saved reports, newest first."""
        pattern = f"{name}/*.json" if name else "*/*.json"
        return sorted(self.dir.glob(pattern), key=lambda p: p.stem, reverse=True)

    @staticmethod
    def load(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------ making one

    def facts(self, spec: ReportSpec, start: datetime, end: datetime) -> dict[str, Any]:
        """Exact counts from the store: never from the model."""
        tb, since, until = self.toolbox, _fmt(start), _fmt(end)
        recorded = tb.recorded(start.timestamp(), end.timestamp())
        names = spec.cameras or tb.engine.camera_names or sorted(set(tb.about) | set(recorded))
        cameras = []
        for cam in names:
            objs = tb.objects(since=since, until=until, camera=cam, limit=1)
            cameras.append({"camera": cam, "recorded": cam in recorded,
                            "total": objs["total"], "counts": objs["counts"]})
        alerts = tb.alerts(since=since, until=until, limit=10)
        return {"cameras": cameras, "alerts": {k: alerts[k] for k in ("total", "counts", "events")}}

    def make(self, name: str, end: datetime | None = None, send: bool = True) -> dict[str, Any]:
        """Make (and save, and send) one report. end: the period's end (default: latest over)."""
        spec = self.specs[name]
        end = end or spec.period_end(datetime.now())
        start, end = spec.period(end)
        tb = self.toolbox
        tb.only_cameras = list(spec.cameras)
        tb.evidence.clear()
        try:
            facts = self.facts(spec, start, end)
            snapshots = [e["snapshot"] for e in facts["alerts"]["events"] if e.get("snapshot")]
            tb.evidence.clear()  # the counts' own thumbnails are arbitrary: keep alert snapshots
            summary, error = self._summary(spec, start, end, facts)
            evidence = list(dict.fromkeys(snapshots[:3] + tb.evidence))[:8]
        finally:
            tb.only_cameras = []
        report = {"name": name, "every": spec.every, "since": _fmt(start), "until": _fmt(end),
                  "cameras": spec.cameras or "all", "interests": spec.interests,
                  "summary": summary, "error": error, "facts": facts, "evidence": evidence,
                  "model": spec.model or (self.config.get("chat") or {}).get("model"),
                  "created": _fmt(datetime.now()), "sent": {}}
        path = self.path(name, end)
        path.parent.mkdir(parents=True, exist_ok=True)
        if send and spec.channels:
            report["sent"] = self.send(spec, report)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        path.with_suffix(".md").write_text(report_text(report), encoding="utf-8")
        log.info("Report %s %s saved to %s", name, period_text(start, end), path)
        return report

    def _summary(self, spec: ReportSpec, start: datetime, end: datetime,
                 facts: dict[str, Any]) -> tuple[str | None, str | None]:
        interests = ("The owner asked to watch for:\n" + "\n".join(f"- {i}" for i in spec.interests)
                     if spec.interests else
                     "No specific interests: summarise what happened and anything unusual.")
        prompt = PROMPT.format(every=spec.every, period=period_text(start, end),
                               cameras=", ".join(spec.cameras) or "all",
                               interests=interests, facts=facts_text(facts),
                               since=_fmt(start), until=_fmt(end), words=spec.words)
        model = spec.model or (self.config.get("chat") or {}).get("model", "local")
        try:
            backend = create_backend(model_config(self.config, model),
                                     system_prompt(self.toolbox, now=end), self.toolbox.tools)
            text = answer(backend, self.toolbox, prompt, log,
                          max_rounds=min(12, 4 + 2 * len(spec.interests)))
            return text, None
        except (Exception, SystemExit) as e:  # model down, missing or slow: counts only
            log.warning("Report %s: no summary (%s)", spec.name, e)
            return None, str(e)

    def send(self, spec: ReportSpec, report: dict[str, Any]) -> dict[str, str]:
        title = f"[Argus] {report['name']}: {report['every']} report, " \
                f"{period_text(*(datetime.fromisoformat(report[k]) for k in ('since', 'until')))}"
        text = report_text(report)
        images = []
        for rel in report["evidence"][:MAX_IMAGES]:
            if path := self.engine.store.media_path(rel):
                images.append((path.name, path.read_bytes()))
        payload = {"kind": "report", **{k: report[k] for k in (
            "name", "every", "since", "until", "cameras", "interests", "summary", "facts",
            "evidence")}}
        results = {}
        for i, channel in enumerate(spec.channels):
            key = f"{i + 1}:{channel.get('type')}"
            try:
                make_channel(channel).send_report(title, text, payload, images)
                results[key] = "ok"
            except Exception as e:  # one failing channel must not stop the others
                log.error("Report %s: %s failed: %s", spec.name, key, e)
                results[key] = f"error: {e}"
        return results

    # ------------------------------------------------------------------ schedule

    def due(self, now: datetime) -> list[tuple[str, datetime]]:
        """Periods over and not made yet. A new schedule starts with the next period, so a
        fresh install doesn't report on the time before it existed."""
        out = []
        for spec in self.specs.values():
            start_file = self.dir / spec.name / ".started"
            if not start_file.exists():
                start_file.parent.mkdir(parents=True, exist_ok=True)
                start_file.write_text(_fmt(now), encoding="utf-8")
            started = datetime.fromisoformat(start_file.read_text(encoding="utf-8").strip())
            end = spec.period_end(now)
            if end > started and not self.path(spec.name, end).exists():
                out.append((spec.name, end))
        return out

    def run_forever(self) -> None:
        if not self.specs:
            raise SystemExit("No reports configured (`reports:` in the config)")
        for spec in self.specs.values():
            log.info("Report %s: %s, next at %s, cameras %s", spec.name, spec.every,
                     _fmt(spec.next_run(datetime.now())), ", ".join(spec.cameras) or "all")
        while True:
            now = datetime.now()
            for name, end in self.due(now):
                try:
                    self.make(name, end)
                except Exception:  # keep the schedule alive; try again next period
                    log.exception("Report %s failed", name)
            wake = min(s.next_run(datetime.now()) for s in self.specs.values())
            time.sleep(min(300.0, max(5.0, (wake - datetime.now()).total_seconds())))


def facts_text(facts: dict[str, Any]) -> str:
    lines = []
    for c in facts["cameras"]:
        if not c["recorded"]:
            lines.append(f"- {c['camera']}: recorded nothing in this period (offline?)")
        elif c["counts"]:
            counts = ", ".join(f"{n} {label}" for label, n in c["counts"].items())
            lines.append(f"- {c['camera']}: {counts}")
        else:
            lines.append(f"- {c['camera']}: recording, no objects tracked")
    a = facts["alerts"]
    if a["total"]:
        counts = ", ".join(f"{k}: {n}" for k, n in a["counts"].items())
        lines.append(f"- Alerts: {a['total']} ({counts})")
        for e in a["events"]:
            lines.append(f"  - {e['time'][5:16]} {e['camera']}: {e['type']} {e['rule']}"
                         f" ({e.get('label', '?')})")
        if a["total"] > len(a["events"]):
            lines.append(f"  - ... {a['total'] - len(a['events'])} more")
    else:
        lines.append("- Alerts: none")
    return "\n".join(lines)


def report_text(report: dict[str, Any]) -> str:
    """The report as plain text / Markdown: email body, .md file, UI."""
    start, end = (datetime.fromisoformat(report[k]) for k in ("since", "until"))
    cameras = report["cameras"] if isinstance(report["cameras"], str) else \
        ", ".join(report["cameras"])
    lines = [f"# {report['name']}: {report['every']} report",
             f"{period_text(start, end)} · cameras: {cameras}", ""]
    if report["interests"]:
        lines += ["Watching for: " + "; ".join(report["interests"]), ""]
    lines += [report["summary"] or f"(No written summary: {report.get('error') or 'model off'}. "
                                   "The counts below are exact.)", "",
              "## Counts", facts_text(report["facts"])]
    if report["evidence"]:
        lines += ["", "## Evidence", *(f"- {e}" for e in report["evidence"])]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Periodic reports of what the cameras saw")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--now", metavar="NAME", help="Make this report for its last period now")
    parser.add_argument("--no-send", action="store_true", help="Save only, don't send")
    parser.add_argument("--list", action="store_true", help="List saved reports")
    parser.add_argument("--show", metavar="NAME", help="Print the latest saved report")
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    reporter = Reporter(config)
    try:
        if args.list:
            for p in reporter.saved():
                r = reporter.load(p)
                print(f"{r['name']:<16} {r['since']} -> {r['until']}  {p.with_suffix('.md')}")
        elif args.show:
            paths = reporter.saved(args.show)
            print(paths[0].with_suffix(".md").read_text(encoding="utf-8") if paths
                  else f"No saved {args.show} report")
        elif args.now:
            if args.now not in reporter.specs:
                raise SystemExit(f"No report {args.now!r}. Configured: "
                                 f"{', '.join(reporter.specs) or 'none'}")
            print(report_text(reporter.make(args.now, send=not args.no_send)))
        else:
            reporter.run_forever()
    except KeyboardInterrupt:
        print()
    finally:
        reporter.close()
        reporter.engine.store.close()


if __name__ == "__main__":
    main()
