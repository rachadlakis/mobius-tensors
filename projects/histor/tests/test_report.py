import json
from datetime import datetime, time as dtime

import numpy as np
import pytest
from histor import report as report_mod
from histor.llm import Step, ToolCall
from histor.report import Reporter, ReportSpec, report_text, report_title
from histor.engine.search import SearchEngine
from histor.engine.store import EventStore


def at(day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 10, day, hour, minute)


def test_periods():
    daily = ReportSpec("d", "daily", at=dtime(7, 0))
    assert daily.period(daily.period_end(at(9, 7, 1))) == (at(7, 7), at(8, 7))  # in the delay
    assert daily.period_end(at(9, 7, 3)) == at(9, 7)        # past the 2 minute delay
    assert daily.next_run(at(9, 7, 3)) == at(10, 7, 2)
    hourly = ReportSpec("h", "hourly")
    assert hourly.period(hourly.period_end(at(9, 14, 30))) == (at(9, 13), at(9, 14))
    weekly = ReportSpec.from_config({"name": "w", "every": "weekly", "day": "monday", "at": "08:00"})
    assert weekly.period_end(at(9, 12)) == at(5, 8)         # 2026-10-09 is a Friday
    assert weekly.period(at(5, 8))[0] == datetime(2026, 9, 28, 8)
    with pytest.raises(ValueError):
        ReportSpec.from_config({"name": "x", "every": "yearly"})


def test_monthly_periods():
    monthly = ReportSpec.from_config({"name": "m", "every": "monthly", "day": 1, "at": "07:00"})
    assert monthly.period_end(at(14, 12)) == at(1, 7)
    assert monthly.period(at(1, 7)) == (datetime(2026, 9, 1, 7), at(1, 7))
    assert monthly.next_run(at(14, 12)) == datetime(2026, 11, 1, 7, 2)
    jan = ReportSpec.from_config({"name": "j", "every": "monthly", "day": 15})
    assert jan.period(datetime(2027, 1, 15, 7)) == (datetime(2026, 12, 15, 7),
                                                    datetime(2027, 1, 15, 7))
    with pytest.raises(ValueError, match="1 to 28"):
        ReportSpec.from_config({"name": "x", "every": "monthly", "day": 31})


class ScriptedBackend:
    name = "fake"

    def __init__(self, steps):
        self.steps, self.users, self.results = list(steps), [], []

    def add_user(self, text):
        self.users.append(text)

    def step(self, tools=True):
        return self.steps.pop(0)

    def add_results(self, results):
        self.results += results


def row(camera, t, track_id, label):
    return {"camera": camera, "time": t.timestamp(), "kind": "object", "track_id": track_id,
            "label": label, "box": [0, 0, 9, 9], "thumb": f"{label}{track_id}.jpg",
            "embedding": np.ones(512, dtype=np.float32) / np.sqrt(512)}


@pytest.fixture
def reporter(tmp_path):
    store = EventStore(tmp_path / "a.db")
    store.add_sightings([row("gate", at(8, 23), 1, "car"), row("gate", at(9, 2), 2, "car"),
                         row("yard", at(9, 3), 1, "person"), row("gate", at(9, 8), 3, "car")])
    store.add_event("yard", at(9, 3).timestamp(), {"type": "enter", "rule": "yard",
                    "label": "person", "track_id": 1}, "snapshots/y1.jpg")
    config = {"store": {"path": str(tmp_path / "a.db")},
              "cameras": {"gate": {}, "yard": {}, "lobby": {}},
              "chat": {"model": "local", "models": {"local": {"api": "openai", "model": "x"}}},
              "reports": [{"name": "morning", "every": "daily", "at": "07:00",
                           "cameras": ["gate", "lobby"], "interests": ["cars at night"]},
                          {"name": "all", "every": "hourly"}]}
    r = Reporter(config, SearchEngine(store, None, camera_names=["gate", "lobby", "yard"]))
    yield r
    r.close()
    store.close()


def test_report_counts_summary_and_camera_limit(reporter, monkeypatch):
    backend = ScriptedBackend([
        Step("", [ToolCall("1", "objects", {"labels": ["person"], "camera": "yard"})]),
        Step("", [ToolCall("2", "objects", {"labels": ["car"], "since": "2026-10-08 07:00",
                                            "until": "2026-10-09 07:00"})]),
        Step("Two cars at the gate overnight (23:00 and 02:00)."),
    ])
    monkeypatch.setattr(report_mod, "create_backend", lambda *a: backend)
    r = reporter.make("morning", at(9, 7), send=False)
    assert r["summary"].startswith("Two cars") and r["error"] is None
    gate, lobby = r["facts"]["cameras"]
    assert gate["counts"] == {"car": 2} and not lobby["recorded"]   # 08:00 car is after the period
    assert r["facts"]["alerts"]["total"] == 0                      # the yard is not in this report
    assert "cars at night" in backend.users[0] and "lobby: no sightings (never" in backend.users[0]
    assert backend.results[0].is_error and "gate, lobby" in backend.results[0].output
    assert json.loads(backend.results[1].output)["total"] == 2
    saved = json.loads(reporter.path("morning", at(9, 7)).read_text(encoding="utf-8"))
    assert saved["summary"] == r["summary"]
    assert reporter.toolbox.only_cameras == []                     # the chat is not limited after


def test_report_without_model_still_has_counts(reporter, monkeypatch):
    def down(*a):
        raise RuntimeError("Can't reach the model server")
    monkeypatch.setattr(report_mod, "create_backend", down)
    r = reporter.make("all", at(9, 4), send=False)
    assert r["summary"] is None and "model server" in r["error"]
    assert r["facts"]["alerts"]["total"] == 1 and r["evidence"] == ["snapshots/y1.jpg"]
    text = report_text(r)
    assert "counts below are exact" in text and "yard: enter yard (person)" in text


def test_quiet_period_needs_no_model(reporter, monkeypatch):
    def never(*a):
        raise AssertionError("the model must not be asked about an empty period")
    monkeypatch.setattr(report_mod, "create_backend", never)
    r = reporter.make("all", at(9, 6), send=False)                 # 05:00-06:00: nothing at all
    assert r["error"] is None and r["summary"].startswith("Nothing seen")
    gate = r["facts"]["cameras"][0]
    assert gate["last"] == "2026-10-09 02:00"                      # when it last saw something
    assert "gate: no sightings (last sighting" in report_text(r)


def test_model_title_is_dropped(reporter, monkeypatch):
    backend = ScriptedBackend([Step("## Daily Report for Thursday\n\n**Cars:** two at the gate.")])
    monkeypatch.setattr(report_mod, "create_backend", lambda *a: backend)
    r = reporter.make("morning", at(9, 7), send=False)
    assert r["summary"] == "**Cars:** two at the gate."
    assert "Only the cameras listed above exist" in backend.users[0]


def test_custom_range_is_saved_apart(reporter, monkeypatch):
    backend = ScriptedBackend([Step("Three cars.")])
    monkeypatch.setattr(report_mod, "create_backend", lambda *a: backend)
    r = reporter.make("morning", at(9, 7), send=False, start=at(8, 20))
    assert r["custom"] and r["since"] == "2026-10-08 20:00" and r["summary"] == "Three cars."
    assert "custom report" in backend.users[0]
    assert r["path"].endswith("2026-10-09_0700_from_2026-10-08_2000.json")
    assert not reporter.path("morning", at(9, 7)).exists()     # the scheduled one is untouched
    assert report_text(r).startswith("# morning: custom report")
    with pytest.raises(ValueError):
        reporter.make("morning", at(8, 7), send=False, start=at(9, 7))


def test_custom_report_any_range(reporter, monkeypatch):
    backend = ScriptedBackend([Step("Cars at the gate.")])
    monkeypatch.setattr(report_mod, "create_backend", lambda *a: backend)
    r = reporter.make("custom", at(9, 7), send=False, start=at(8, 20))
    assert r["custom"] and r["path"].replace("\\", "/").split("/reports/")[1].startswith("custom/")
    assert "cars at night" in backend.users[0]          # what the configured reports watch for
    assert report_text(r).startswith("# Custom report")
    assert "custom" not in reporter.specs               # never scheduled
    with pytest.raises(ValueError, match="range"):
        reporter.make("custom", send=False)
    with pytest.raises(ValueError, match="kept"):
        report_mod.load_specs({"reports": [{"name": "custom"}]})


def test_report_title():
    assert report_title({"name": "hourly", "every": "hourly"}) == "Hourly report"
    assert report_title({"name": "morning", "every": "daily"}) == "morning: daily report"


def test_due_starts_with_the_next_period(reporter, monkeypatch):
    monkeypatch.setattr(report_mod, "create_backend", lambda *a: ScriptedBackend([Step("ok")]))
    assert reporter.due(at(9, 7, 30)) == []                        # fresh install: nothing old
    assert reporter.due(at(10, 7, 3)) == [("morning", at(10, 7)), ("all", at(10, 7))]
    reporter.make("morning", at(10, 7), send=False)
    assert ("morning", at(10, 7)) not in reporter.due(at(10, 7, 3))


def test_send_goes_to_each_channel(reporter, monkeypatch):
    sent = []

    class FakeSink:
        def send_report(self, title, text, payload, images):
            sent.append((title, payload["kind"]))

    monkeypatch.setattr(report_mod, "make_channel",
                        lambda c: FakeSink() if c["type"] == "ok" else 1 / 0)
    reporter.specs["all"].channels = [{"type": "ok"}, {"type": "broken"}]
    monkeypatch.setattr(report_mod, "create_backend", lambda *a: ScriptedBackend([Step("ok")]))
    r = reporter.make("all", at(9, 4))
    assert sent[0][0].startswith("[Histor] all: hourly report") and sent[0][1] == "report"
    assert r["sent"]["1:ok"] == "ok" and r["sent"]["2:broken"].startswith("error")


def test_unknown_report_camera_fails_at_start(tmp_path):
    store = EventStore(tmp_path / "b.db")
    config = {"store": {"path": str(tmp_path / "b.db")},
              "reports": [{"name": "x", "cameras": ["garage"]}]}
    with pytest.raises(ValueError, match="garage"):
        Reporter(config, SearchEngine(store, None, camera_names=["gate"]))
    store.close()


def test_delete_a_saved_report(tmp_path):
    store = EventStore(tmp_path / "store" / "histor.db")
    reporter = Reporter({"store": {"path": str(store.path)}, "reports": []},
                        SearchEngine(store=store))
    saved = reporter.dir / "daily" / "2026-10-14_0700.json"
    saved.parent.mkdir(parents=True)
    saved.write_text("{}")
    saved.with_suffix(".md").write_text("x")
    assert reporter.delete(saved) and not saved.exists() and not saved.with_suffix(".md").exists()
    assert not reporter.delete(saved)                      # already gone
    with pytest.raises(ValueError):                        # never a file outside reports/
        reporter.delete(store.path)
    with pytest.raises(ValueError):
        reporter.delete(reporter.dir / ".." / "x.json")
