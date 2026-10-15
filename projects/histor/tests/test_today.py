import logging
from datetime import datetime

import numpy as np
from histor.engine.store import EventStore
from histor.today import call_text, describe, plain_summary, topbar_html

DAY = datetime(2026, 10, 14).timestamp()


def event(store: EventStore, hour: float, **e) -> dict:
    eid = store.add_event(e.pop("camera", "street"), DAY + hour * 3600,
                          {"rule": "r", "label": "person", **e}, None)
    return next(x for x in store.list_events() if x["id"] == eid)


def test_events_in_plain_words(tmp_path):
    store = EventStore(tmp_path / "s.db")
    down = describe(event(store, 11.5, type="down", rule="person_down", dwell_seconds=14.2))
    assert (down["kind"], down["group"], down["state"]) == ("alert", "People", "open")
    assert down["short"] == "Person down for 14 s" and down["time"] == "11:30:00"
    assert abs(down["hour"] - 11.5) < 1e-6
    plate = describe(event(store, 6, type="plate", rule="vans", label="truck", plate="KX482TL"))
    assert (plate["kind"], plate["group"], plate["title"]) == ("plate", "Vehicles",
                                                               "A truck, KX482TL")
    covered = describe(event(store, 14, type="tamper", rule="blocked", label="camera",
                             camera="parking"))
    assert covered["title"] == "The parking camera was covered"
    crowd = describe(event(store, 10, type="crowd", rule="lobby", count=14))
    assert crowd["short"] == "14 people in view" and crowd["kind"] == "warn"
    car = describe(event(store, 16, type="cross", rule="gate", label="car", direction="in"))
    assert (car["kind"], car["group"]) == ("info", "Vehicles")
    # a false alarm (the vision check said no) never needs a look
    eid = store.add_event("street", DAY, {"type": "sound", "rule": "glass_break",
                                          "label": "Shatter"}, None)
    store.set_verification(eid, "no", "a door slamming")
    held = describe(next(x for x in store.list_events() if x["id"] == eid))
    assert held["state"] == "held" and "a door slamming" in held["text"]


def test_mark_seen(tmp_path):
    store = EventStore(tmp_path / "s.db")
    e = event(store, 9, type="hands_up", rule="hands_up")
    store.mark_seen(e["id"])
    assert describe(store.list_events(limit=1)[0])["state"] == "seen"


def test_activity_per_half_hour(tmp_path):
    store = EventStore(tmp_path / "s.db")
    rows = [{"camera": "gate", "time": DAY + 3600 * 9 + s, "kind": "object", "track_id": t,
             "label": "person"} for s, t in ((10, 1), (20, 1), (40, 2), (1900, 3))]
    rows.append({"camera": "gate", "time": DAY + 60, "kind": "scene", "label": None})
    store.add_sightings(rows)
    bins = store.activity(DAY, DAY + 86400, 1800)["gate"]
    assert len(bins) == 48 and bins[18] == 2 and bins[19] == 1 and sum(bins) == 3
    assert store.visits(DAY, DAY + 86400) == {"gate": 3}


def test_plain_summary_and_top_bar():
    events = [{"kind": "alert", "short": "Person down", "time": "11:47:08", "camera": "street"},
              {"kind": "info", "short": "x", "time": "12:00:00", "camera": "gate"}]
    text = plain_summary(events, {"gate": 10, "street": 3})
    assert text.startswith("13 visits so far on 2 cameras, the most on gate.")
    assert "1 alert: person down at 11:47 on street." in text and "1 other event" in text
    assert plain_summary([], {}) == "Nothing recorded yet today."
    bar = topbar_html("Today", "<svg></svg>")
    today = bar[bar.index('href="/" data-tab="Today"'):]
    assert today.index("aria-current=page") < today.index("</a>")  # Today is the page shown
    assert "hx-dev-only" in bar and "hx-badge" in today


def test_lookups_read_like_calls():
    record = logging.LogRecord("t", logging.INFO, "", 0, "  -> %s %s",
                               ("people", {"camera": "street", "labels": ["person"]}), None)
    assert call_text(record) == 'people(camera="street", labels=["person"])'


def test_clear_deletes_rows_and_pictures(tmp_path):
    store = EventStore(tmp_path / "s.db")
    snap = store.save_image(np.zeros((8, 8, 3), np.uint8), "events", "a", DAY)
    store.add_event("gate", DAY, {"type": "enter", "rule": "r"}, snap)
    store.add_faces([{"camera": "gate", "time": DAY, "track_id": 1, "quality": 1.0,
                      "thumb": None, "embedding": np.ones(4, np.float32)}])
    other = tmp_path / "media" / "keep.txt"  # a file no row names stays
    other.write_text("x")
    deleted = store.clear()
    assert deleted["events"] == 1 and deleted["faces"] == 1
    assert store.counts() == {t: 0 for t in ("events", "sightings", "faces", "poses", "plates")}
    assert not (store.media_dir / snap).exists() and other.exists()
    assert (tmp_path / "s.db").exists()
