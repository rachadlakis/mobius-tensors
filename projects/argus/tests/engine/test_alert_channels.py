from datetime import datetime

import pytest

from argus.engine import alerts
from argus.engine.alerts import (AlertManager, ChannelFilter, EmailSink, Sink, WebhookSink, alert_message,
                    make_channel)
from argus.engine.detector import Detection
from argus.engine.tracker import Tracker
from argus.engine.zones import Zone, ZoneEvent, ZoneMonitor

FRAME = (200, 100)


def payload(rule="yard", type_="enter", label="person", hour=23):
    t = datetime(2026, 10, 6, hour, 30).timestamp()
    return {"type": type_, "rule": rule, "label": label, "camera": "cam", "track_id": 1,
            "time": t, "iso_time": datetime.fromtimestamp(t).isoformat(timespec="seconds")}


class Collect(Sink):
    def __init__(self, name: str):
        self.name = name
        self.got: list[dict] = []
        Collect.instances[name] = self

    def send(self, payload, jpeg):
        self.got.append(payload)


Collect.instances: dict[str, Collect] = {}


def test_make_channel_picks_class_and_filter():
    sink = make_channel({"type": "webhook", "url": "http://x/hook", "rules": ["gate"],
                         "hours": "22:00-06:00"})
    assert isinstance(sink, WebhookSink) and sink.filter.rules == {"gate"}
    with pytest.raises(ValueError, match="Unknown alert channel 'whatsapp'"):
        make_channel({"type": "whatsapp", "token": "t"})
    with pytest.raises(ValueError, match="hours must look like"):
        make_channel({"type": "webhook", "url": "http://x", "hours": "night"})


def test_old_webhook_and_email_keys_still_work():
    manager = AlertManager(console=False, webhook={"url": "http://x/hook"},
                           email={"host": "smtp.x", "sender": "a@x", "to": "b@x"},
                           channels=[{"type": "webhook", "url": "http://y/hook"}])
    assert [type(s) for s in manager.sinks] == [WebhookSink, WebhookSink, EmailSink]
    manager.close()


def test_channel_filter():
    assert ChannelFilter().accepts(payload())
    assert not ChannelFilter(rules=["gate"]).accepts(payload(rule="yard"))
    assert not ChannelFilter(events=["dwell"]).accepts(payload(type_="enter"))
    assert ChannelFilter(labels=["person"]).accepts(payload(label="person"))
    night = ChannelFilter(hours="22:00-06:00")  # wraps past midnight
    assert night.accepts(payload(hour=23)) and night.accepts(payload(hour=2))
    assert not night.accepts(payload(hour=12))
    day = ChannelFilter(hours="08:00-18:00")
    assert day.accepts(payload(hour=12)) and not day.accepts(payload(hour=23))


def test_channel_cooldown_is_per_rule_and_type():
    quiet = ChannelFilter(cooldown_seconds=600)
    first = payload(hour=23)
    assert quiet.accepts(first)
    assert not quiet.accepts({**first, "time": first["time"] + 60, "track_id": 2})
    assert quiet.accepts({**first, "time": first["time"] + 60, "type": "exit"})
    assert quiet.accepts({**first, "time": first["time"] + 601})


def test_each_channel_gets_only_what_its_filter_accepts(monkeypatch):
    monkeypatch.setitem(alerts.CHANNELS, "collect", Collect)
    manager = AlertManager(console=False, source="yard_cam", channels=[
        {"type": "collect", "name": "all"},
        {"type": "collect", "name": "night_people", "labels": ["person"], "hours": "22:00-06:00"},
    ])
    night = datetime(2026, 10, 6, 23, 0).timestamp()
    noon = datetime(2026, 10, 6, 12, 0).timestamp()
    manager.handle([ZoneEvent(1.0, "enter", "yard", 1, "person")], wall_time=night)
    manager.handle([ZoneEvent(2.0, "enter", "yard", 2, "car")], wall_time=night)
    manager.handle([ZoneEvent(3.0, "enter", "yard", 3, "person")], wall_time=noon)
    manager.close()
    assert [p["track_id"] for p in Collect.instances["all"].got] == [1, 2, 3]
    assert [p["track_id"] for p in Collect.instances["night_people"].got] == [1]
    assert Collect.instances["all"].got[0]["camera"] == "yard_cam"


def test_alert_message():
    title, text = alert_message({**payload(type_="dwell"), "dwell_seconds": 61.0}, "[Site]")
    assert title == "[Site] dwell yard - person on cam"
    assert "Event:  dwell (yard) 61.0" in text and "Object: person #1" in text


def det(x: float, y: float, label: str) -> Detection:
    return Detection(x - 15, y - 30, x + 15, y, 0.9, 0, label)


def test_zone_without_points_covers_the_whole_frame():
    monitor = ZoneMonitor.from_config([{"name": "anywhere", "classes": ["dog"]}])
    tracker = Tracker(min_hits=1)
    events = []
    for i, dets in enumerate([[det(30, 90, "dog")], [det(35, 90, "dog")], [det(150, 40, "car")]]):
        tracks, removed = tracker.update(dets)
        events += monitor.update(tracks, removed, i / 10, FRAME)
    assert [(e.type, e.rule, e.label) for e in events] == [("enter", "anywhere", "dog")]
    assert isinstance(monitor.rules[0], Zone)
