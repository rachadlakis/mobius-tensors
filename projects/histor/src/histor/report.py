"""Periodic reports: an hourly, daily or weekly summary of what the cameras saw.

    uv run -m histor.report                   # run the schedules in `reports:` (keep it running)
    uv run -m histor.report --now morning     # make the report of the last period now, and send it
    uv run -m histor.report --now morning --since "2026-10-01 00:00" --until "2026-10-08 00:00"
    uv run -m histor.report --list            # saved reports
    uv run -m histor.report --show morning    # print the latest saved one

Each report (configs/default.yaml, `reports:`) names its period, its cameras and what the owner
cares about ("vehicles at the gate after hours"). At the end of each period Histor:

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
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, time as dtime, timedelta
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from histor.engine.alerts import make_channel
from histor.ask import build_engine
from histor.chat import answer, model_config, system_prompt
from histor.llm import create_backend
from histor.engine.search import SearchEngine
from histor.tools import Toolbox

log = logging.getLogger("histor.report")

PERIODS = {"hourly": timedelta(hours=1), "daily": timedelta(days=1), "weekly": timedelta(days=7)}
EVERY = (*PERIODS, "monthly")  # a month has no fixed length: handled apart
CUSTOM = "custom"  # the report for a chosen range (Reporter.custom_spec)
DAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


def add_months(t: datetime, n: int) -> datetime:
    """Same day and time n months later (day capped at 28: every month has it)."""
    month = t.month - 1 + n
    return t.replace(year=t.year + month // 12, month=month % 12 + 1, day=min(t.day, 28))
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
Start directly with point 1: no title or heading, the report already has one.
Only the cameras listed above exist: don't mention any others.
Only what the tools returned: no guesses presented as facts."""


@dataclass
class ReportSpec:
    name: str
    every: str = "daily"
    at: dtime = dtime(7, 0)       # daily / weekly / monthly: when a period ends
    day: int = 0                  # weekly: 0 = monday; monthly: day of the month (1-28)
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
        if every not in EVERY:
            raise ValueError(f"Report {d['name']}: every must be {', '.join(EVERY)}")
        if every == "monthly":
            day = int(d.get("day", 1))
            if not 1 <= day <= 28:
                raise ValueError(f"Report {d['name']}: a monthly day must be 1 to 28")
        else:
            name = str(d.get("day", "monday")).lower()
            if name not in DAYS:
                raise ValueError(f"Report {d['name']}: day must be one of {', '.join(DAYS)}")
            day = DAYS.index(name)
        interests = d.get("interests") or []
        return cls(name=str(d["name"]), every=every,
                   at=dtime.fromisoformat(str(d.get("at", "07:00"))), day=day,
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
        if self.every == "monthly":
            end = datetime.combine(ready.date().replace(day=self.day), self.at)
            return end if end <= ready else add_months(end, -1)
        end = datetime.combine(ready.date(), self.at)
        if self.every == "weekly":
            end -= timedelta(days=(end.weekday() - self.day) % 7)
        return end if end <= ready else end - PERIODS[self.every]

    def period(self, end: datetime) -> tuple[datetime, datetime]:
        if self.every == "monthly":
            return add_months(end, -1), end
        return end - PERIODS[self.every], end

    def next_run(self, now: datetime) -> datetime:
        end = self.period_end(now)
        end = add_months(end, 1) if self.every == "monthly" else end + PERIODS[self.every]
        return end + timedelta(minutes=self.delay_minutes)


def load_specs(config: dict[str, Any]) -> list[ReportSpec]:
    specs = [ReportSpec.from_config(d) for d in config.get("reports") or []]
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"Report names must be unique: {names}")
    if CUSTOM in names:
        raise ValueError(f"{CUSTOM!r} is kept for reports on a chosen range: pick another name")
    return specs


def _fmt(t: datetime) -> str:
    return t.strftime("%Y-%m-%d %H:%M")


def period_text(start: datetime, end: datetime) -> str:
    if start.date() == end.date():
        return f"{start:%a %d %b %H:%M}-{end:%H:%M}"
    return f"{start:%a %d %b %H:%M} to {end:%a %d %b %H:%M}"


class Reporter:
    def __init__(self, config: dict[str, Any], engine: SearchEngine | None = None) -> None:
        self.config = config
        self.specs = {s.name: s for s in load_specs(config)}
        self.engine = engine or build_engine(config)
        # Its own toolbox: the camera limit must not leak into a chat sharing the engine.
        self.toolbox = Toolbox.from_config(self.engine, config)
        store_cfg = config.get("store") or {}
        self.dir = Path(store_cfg.get("path", "data/store/histor.db")).parent / "reports"
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

    def custom_spec(self) -> ReportSpec:
        """The report for any range (UI "custom", never scheduled): every camera, and what
        the configured reports watch for."""
        interests = list(dict.fromkeys(i for s in self.specs.values() for i in s.interests))
        return ReportSpec(CUSTOM, interests=interests)

    # ------------------------------------------------------------------ saved reports

    def path(self, name: str, end: datetime, start: datetime | None = None) -> Path:
        """start: a custom range, saved apart from the scheduled period ending at the same time."""
        custom = f"_from_{start:%Y-%m-%d_%H%M}" if start else ""
        return self.dir / name / f"{end:%Y-%m-%d_%H%M}{custom}.json"

    def saved(self, name: str | None = None) -> list[Path]:
        """Saved reports, newest first."""
        pattern = f"{name}/*.json" if name else "*/*.json"
        return sorted(self.dir.glob(pattern), key=lambda p: p.stem, reverse=True)

    def delete(self, path: str | Path) -> bool:
        """Delete one saved report (its .json and .md). Only a file of this reporter's folder:
        the path may come from a web page. Returns whether it was there."""
        path = Path(path).resolve()
        if path.suffix != ".json" or path.parent.parent != self.dir.resolve():
            raise ValueError(f"Not a saved report: {path}")
        found = path.is_file()
        path.unlink(missing_ok=True)
        path.with_suffix(".md").unlink(missing_ok=True)
        return found

    @staticmethod
    def load(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------ making one

    def facts(self, spec: ReportSpec, start: datetime, end: datetime) -> dict[str, Any]:
        """Exact counts from the store: never from the model."""
        tb, since, until = self.toolbox, _fmt(start), _fmt(end)
        recorded = tb.recorded(start.timestamp(), end.timestamp())
        # all cameras: the configured ones, and any other that recorded (a simulated camera)
        names = spec.cameras or (list(tb.engine.camera_names) + sorted(
            set(recorded) - set(tb.engine.camera_names)) if tb.engine.camera_names
            else sorted(set(tb.about) | set(recorded)))
        cameras = []
        for cam in names:
            objs = tb.objects(since=since, until=until, camera=cam, limit=1)
            entry = {"camera": cam, "recorded": cam in recorded,
                     "total": objs["total"], "counts": objs["counts"]}
            if cam not in recorded:  # when it last saw something: off for days, or a quiet hour?
                last = tb.last_seen(cam, start.timestamp())
                entry["last"] = _fmt(datetime.fromtimestamp(last)) if last else None
            cameras.append(entry)
        alerts = tb.alerts(since=since, until=until, limit=10)
        return {"cameras": cameras, "alerts": {k: alerts[k] for k in ("total", "counts", "events")}}

    def make(self, name: str, end: datetime | None = None, send: bool = True,
             start: datetime | None = None) -> dict[str, Any]:
        """Make (and save, and send) one report. end: the period's end (default: latest over);
        start: a custom range instead of the report's period (same cameras and interests)."""
        spec = self.custom_spec() if name == CUSTOM else self.specs[name]
        custom = start is not None
        if name == CUSTOM and not custom:
            raise ValueError("A custom report needs a range (start)")
        if custom:
            end = end or datetime.now()
            if start >= end:
                raise ValueError("The range must start before it ends")
        else:
            start, end = spec.period(end or spec.period_end(datetime.now()))
        report = {"name": name, "every": spec.every, "custom": custom,
                  "since": _fmt(start), "until": _fmt(end),
                  "cameras": spec.cameras or "all", "interests": spec.interests,
                  **self.compose(spec, start, end, custom),
                  "model": spec.model or (self.config.get("chat") or {}).get("model"),
                  "created": _fmt(datetime.now()), "sent": {}}
        path = self.path(name, end, start if custom else None)
        path.parent.mkdir(parents=True, exist_ok=True)
        if send and spec.channels:
            report["sent"] = self.send(spec, report)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        path.with_suffix(".md").write_text(report_text(report), encoding="utf-8")
        log.info("Report %s %s saved to %s", name, period_text(start, end), path)
        return {**report, "path": str(path)}  # where it was saved (not stored in the file)

    def compose(self, spec: ReportSpec, start: datetime, end: datetime,
                custom: bool = False, prompt: str = "") -> dict[str, Any]:
        """A report's content, neither saved nor sent: summary (None if the model failed),
        error, facts, evidence. prompt: instead of PROMPT, same fields (the UI's Today page
        asks for one paragraph of "today so far")."""
        tb = self.toolbox
        tb.only_cameras = list(spec.cameras)
        tb.evidence.clear()
        try:
            facts = self.facts(spec, start, end)
            snapshots = [e["snapshot"] for e in facts["alerts"]["events"] if e.get("snapshot")]
            tb.evidence.clear()  # the counts' own thumbnails are arbitrary: keep alert snapshots
            # Nothing to tell: a fixed line, no model (it only pads and guesses)
            summary, error = quiet_summary(spec, start, end, facts), None
            if summary is None:
                summary, error = self._summary(spec, start, end, facts, custom, prompt)
            evidence = list(dict.fromkeys(snapshots[:3] + tb.evidence))[:8]
        finally:
            tb.only_cameras = []
        return {"summary": summary, "error": error, "facts": facts, "evidence": evidence}

    def _summary(self, spec: ReportSpec, start: datetime, end: datetime, facts: dict[str, Any],
                 custom: bool = False, prompt: str = "") -> tuple[str | None, str | None]:
        interests = ("The owner asked to watch for:\n" + "\n".join(f"- {i}" for i in spec.interests)
                     if spec.interests else
                     "No specific interests: summarise what happened and anything unusual.")
        prompt = (prompt or PROMPT).format(every="custom" if custom else spec.every,
                               period=period_text(start, end),
                               cameras=", ".join(c["camera"] for c in facts["cameras"]),
                               interests=interests, facts=facts_text(facts),
                               since=_fmt(start), until=_fmt(end), words=spec.words)
        model = spec.model or (self.config.get("chat") or {}).get("model", "local")
        try:
            backend = create_backend(model_config(self.config, model),
                                     system_prompt(self.toolbox, now=end), self.toolbox.tools)
            text = answer(backend, self.toolbox, prompt, log,
                          max_rounds=min(12, 4 + 2 * len(spec.interests)))
            # small models add a title anyway: the report has its own
            lines = (text or "").strip().splitlines()
            while lines and (lines[0].startswith("#") or not lines[0].strip() or
                             re.fullmatch(r"\*\*[^*]*report[^*]*\*\*:?", lines[0].strip(), re.I)):
                lines.pop(0)
            return "\n".join(lines), None
        except (Exception, SystemExit) as e:  # model down, missing or slow: counts only
            log.warning("Report %s: no summary (%s)", spec.name, e)
            return None, str(e)

    def send(self, spec: ReportSpec, report: dict[str, Any]) -> dict[str, str]:
        title = f"[Histor] {report_title(report)}, " \
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


def _last_text(last: str | None) -> str:
    if not last:
        return "never recorded anything"
    t = datetime.fromisoformat(last)
    return f"last sighting {t:%H:%M}" if t.date() == datetime.now().date() else \
        f"last sighting {t:%a %d %b %H:%M}"


def quiet_summary(spec: ReportSpec, start: datetime, end: datetime,
                  facts: dict[str, Any]) -> str | None:
    """The summary of a period with no sightings and no alerts; None if there is more to say."""
    if facts["alerts"]["total"] or any(c["total"] for c in facts["cameras"]):
        return None
    period = period_text(start, end)
    if not any(c["recorded"] for c in facts["cameras"]):
        # no rows at all: the camera was off, or nobody passed (empty rooms add no rows)
        cams = facts["cameras"]
        who = "the camera" if len(cams) == 1 else "any camera"
        text = f"Nothing seen {period}: no sightings from {who}. " \
               "Either nobody passed or the recorder was off"
        text += f" ({_last_text(cams[0].get('last'))})." if len(cams) == 1 else "."
    else:
        text = f"Nothing seen {period}: recording, but no people or objects tracked."
    if spec.interests:
        text += " None of what you watch for was seen."
    return text


def facts_text(facts: dict[str, Any]) -> str:
    lines = []
    for c in facts["cameras"]:
        if not c["recorded"]:
            lines.append(f"- {c['camera']}: no sightings ({_last_text(c.get('last'))})")
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


def report_title(report: dict[str, Any]) -> str:
    """"Hourly report", or "morning: daily report" when the name says more than the period;
    "Custom report" for a chosen range."""
    kind = "custom" if report.get("custom") else report["every"]
    if report["name"].lower() in (report["every"], kind):
        return f"{kind.capitalize()} report"
    return f"{report['name']}: {kind} report"


def report_text(report: dict[str, Any], evidence: bool = True) -> str:
    """The report as plain text / Markdown: email body, .md file, UI (evidence=False: the UI
    shows the images themselves)."""
    start, end = (datetime.fromisoformat(report[k]) for k in ("since", "until"))
    names = [c["camera"] for c in report["facts"]["cameras"]]
    cameras = ", ".join(names) if names else report["cameras"] if \
        isinstance(report["cameras"], str) else ", ".join(report["cameras"])
    lines = [f"# {report_title(report)}",
             f"{period_text(start, end)} · cameras: {cameras}", ""]
    if report["interests"]:
        lines += ["Watching for: " + "; ".join(report["interests"]), ""]
    lines += [report["summary"] or f"(No written summary: {report.get('error') or 'model off'}. "
                                   "The counts below are exact.)", "",
              "## Counts", facts_text(report["facts"])]
    if evidence and report["evidence"]:
        lines += ["", "## Evidence", *(f"- {e}" for e in report["evidence"])]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Periodic reports of what the cameras saw")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--now", metavar="NAME", help="Make this report for its last period now")
    parser.add_argument("--since", help="With --now: a custom range start, e.g. 2026-10-01 00:00")
    parser.add_argument("--until", help="With --since: the range end (default: now)")
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
            since = datetime.fromisoformat(args.since) if args.since else None
            until = datetime.fromisoformat(args.until) if args.until else None
            print(report_text(reporter.make(args.now, until, send=not args.no_send,
                                            start=since)))
        else:
            reporter.run_forever()
    except KeyboardInterrupt:
        print()
    finally:
        reporter.close()
        reporter.engine.store.close()


if __name__ == "__main__":
    main()
