"""Camera status from the store: last activity, online, objects and alerts today, tamper.
Shared by the dashboard (`GET /status`) and the Telegram bot (`/status`)."""

from __future__ import annotations

import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from histor.engine.store import EventStore

ONLINE_SECONDS = 120     # a camera with nothing stored for this long shows as offline
TAMPER_SECONDS = 3600    # a tamper alert this recent is shown with the camera


def camera_status(store: EventStore, cameras: list[str], now: float | None = None
                  ) -> dict[str, Any]:
    """Per camera: last thing stored, online, objects tracked and alerts today, recent tamper."""
    now = time.time() if now is None else now
    midnight = datetime.fromtimestamp(now).replace(hour=0, minute=0, second=0,
                                                   microsecond=0).timestamp()
    last: dict[str, float] = {}
    tracked: dict[str, dict[str, int]] = {}
    alerts: dict[str, dict[str, int]] = {}
    tamper: dict[str, dict[str, Any]] = {}
    with store._lock:  # same package: read with the store's own connection and lock
        db = store._db
        for table in ("sightings", "poses", "events", "plates"):
            exists = db.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (table,)).fetchone()
            if not exists:
                continue
            for camera, t in db.execute(f"SELECT camera, MAX(time) FROM {table} GROUP BY camera"):
                if t is not None:
                    last[camera] = max(last.get(camera, 0.0), t)
        for camera, label, n in db.execute(
                "SELECT camera, label, COUNT(DISTINCT track_id) FROM sightings WHERE kind = "
                "'object' AND time >= ? AND track_id IS NOT NULL GROUP BY camera, label",
                (midnight,)):
            tracked.setdefault(camera, {})[label] = n
        for camera, n, held in db.execute(
                "SELECT camera, COUNT(*), SUM(verified = 'no') FROM events WHERE time >= ? "
                "GROUP BY camera", (midnight,)):
            alerts[camera] = {"total": n, "held_back": int(held or 0)}
        for camera, rule, t in db.execute(
                "SELECT camera, rule, MAX(time) FROM events WHERE type = 'tamper' AND time >= ? "
                "GROUP BY camera", (now - TAMPER_SECONDS,)):
            tamper[camera] = {"rule": rule, "time": t,
                              "iso_time": datetime.fromtimestamp(t).isoformat(timespec="seconds")}
    names = sorted(set(cameras) | set(last))
    out = []
    for name in names:
        seen = last.get(name)
        counts = dict(sorted(tracked.get(name, {}).items(), key=lambda kv: -kv[1]))
        out.append({
            "camera": name,
            "last_seen": datetime.fromtimestamp(seen).isoformat(timespec="seconds") if seen else None,
            "seconds_ago": round(now - seen) if seen else None,
            "online": bool(seen and now - seen < ONLINE_SECONDS),
            "tracked_today": counts,
            "alerts_today": alerts.get(name, {"total": 0, "held_back": 0}),
            "tamper": tamper.get(name),
        })
    return {"time": datetime.fromtimestamp(now).isoformat(timespec="seconds"),
            "online_seconds": ONLINE_SECONDS, "cameras": out}
