"""Event store: SQLite for events, search index rows and face prints, plus JPEG media on disk.

One SQLite file per site. Detection processes write, the API server reads (WAL mode lets both
run at once). Vector search is brute-force numpy over the rows in the requested time range:
fast enough for a pilot (millions of dot products per second); swap for sqlite-vec / pgvector
when a site outgrows it.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    camera TEXT NOT NULL,
    time REAL NOT NULL,            -- unix seconds (wall clock)
    type TEXT NOT NULL,            -- enter | exit | dwell | cross
    rule TEXT NOT NULL,
    label TEXT,
    track_id INTEGER,
    direction TEXT,
    dwell_seconds REAL,
    snapshot TEXT                  -- path relative to media_dir
);
CREATE INDEX IF NOT EXISTS events_time ON events(time);

-- Search index: one row per embedded image region.
--   kind = object (a tracked person/vehicle crop) | scene (whole frame) | tile (part of a frame)
CREATE TABLE IF NOT EXISTS sightings (
    id INTEGER PRIMARY KEY,
    camera TEXT NOT NULL,
    time REAL NOT NULL,
    kind TEXT NOT NULL,
    track_id INTEGER,
    label TEXT,
    box TEXT,                      -- json [x1, y1, x2, y2] in frame pixels
    thumb TEXT,
    embedding BLOB NOT NULL        -- float32, L2-normalized
);
CREATE INDEX IF NOT EXISTS sightings_time ON sightings(time);

-- Face prints (biometric data: only written for cameras where faces are enabled).
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY,
    camera TEXT NOT NULL,
    time REAL NOT NULL,
    track_id INTEGER,
    quality REAL,                  -- detector confidence x size factor
    thumb TEXT,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS faces_time ON faces(time);
"""


@dataclass
class Match:
    table: str
    id: int
    camera: str
    time: float
    score: float
    track_id: int | None
    label: str | None
    kind: str | None
    box: list[float] | None
    thumb: str | None

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["iso_time"] = datetime.fromtimestamp(self.time).isoformat(timespec="seconds")
        d["score"] = round(float(self.score), 4)
        return d


class EventStore:
    def __init__(
        self,
        path: str | Path = "data/store/mobius.db",
        media_dir: str | Path | None = None,
        retention_days: float | None = 30,
        face_retention_days: float | None = 7,
        thumb_height: int = 160,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.media_dir = Path(media_dir) if media_dir else self.path.parent / "media"
        self.media_dir.mkdir(parents=True, exist_ok=True)
        self.retention_days = retention_days
        self.face_retention_days = face_retention_days
        self.thumb_height = thumb_height
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(SCHEMA)

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> EventStore | None:
        if not config:
            return None
        return cls(**config)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ---------------------------------------------------------------- media

    def save_image(self, image: np.ndarray, folder: str, name: str, t: float,
                   max_height: int | None = None, quality: int = 85) -> str:
        """Write a JPEG under media_dir/<folder>/<YYYY-MM-DD>/ and return its relative path."""
        if max_height and image.shape[0] > max_height:
            scale = max_height / image.shape[0]
            image = cv2.resize(image, (max(1, int(image.shape[1] * scale)), max_height),
                               interpolation=cv2.INTER_AREA)
        day = datetime.fromtimestamp(t).strftime("%Y-%m-%d")
        name = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
        rel = Path(folder) / day / f"{name}.jpg"
        target = self.media_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return rel.as_posix()

    def save_thumb(self, image: np.ndarray, folder: str, name: str, t: float) -> str:
        return self.save_image(image, folder, name, t, max_height=self.thumb_height, quality=80)

    def media_path(self, rel: str) -> Path | None:
        """Resolve a stored relative path; None if it escapes media_dir or does not exist."""
        root = self.media_dir.resolve()
        target = (root / rel).resolve()
        if root not in target.parents or not target.is_file():
            return None
        return target

    # ---------------------------------------------------------------- writes

    def add_event(self, camera: str, t: float, event: dict[str, Any], snapshot: str | None) -> int:
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO events (camera, time, type, rule, label, track_id, direction,"
                " dwell_seconds, snapshot) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (camera, t, event["type"], event["rule"], event.get("label"),
                 event.get("track_id"), event.get("direction"), event.get("dwell_seconds"),
                 snapshot),
            )
            self._db.commit()
            return int(cur.lastrowid or 0)

    def add_sightings(self, rows: list[dict[str, Any]]) -> None:
        """rows: camera, time, kind, track_id, label, box, thumb, embedding (np.float32)."""
        if not rows:
            return
        with self._lock:
            self._db.executemany(
                "INSERT INTO sightings (camera, time, kind, track_id, label, box, thumb, embedding)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [(r["camera"], r["time"], r["kind"], r.get("track_id"), r.get("label"),
                  json.dumps([round(float(v), 1) for v in r["box"]]) if r.get("box") is not None
                  else None, r.get("thumb"), _blob(r["embedding"])) for r in rows],
            )
            self._db.commit()

    def add_faces(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        with self._lock:
            self._db.executemany(
                "INSERT INTO faces (camera, time, track_id, quality, thumb, embedding)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                [(r["camera"], r["time"], r.get("track_id"), r.get("quality"), r.get("thumb"),
                  _blob(r["embedding"])) for r in rows],
            )
            self._db.commit()

    # ---------------------------------------------------------------- reads

    def list_events(self, camera: str | None = None, types: list[str] | None = None,
                    since: float | None = None, until: float | None = None,
                    limit: int = 100) -> list[dict[str, Any]]:
        sql, args = "SELECT * FROM events WHERE 1=1", list[Any]()
        sql, args = _filters(sql, args, [camera] if camera else None, since, until)
        if types:
            sql += f" AND type IN ({','.join('?' * len(types))})"
            args += types
        sql += " ORDER BY time DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self._db.execute(sql, args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["iso_time"] = datetime.fromtimestamp(d["time"]).isoformat(timespec="seconds")
            out.append({k: v for k, v in d.items() if v is not None})
        return out

    def search(self, table: str, query: np.ndarray, since: float | None = None,
               until: float | None = None, cameras: list[str] | None = None,
               kinds: list[str] | None = None, min_score: float | None = None,
               top_k: int = 50) -> list[Match]:
        """Cosine similarity of `query` against stored embeddings, best first."""
        assert table in {"sightings", "faces"}
        sql, args = f"SELECT * FROM {table} WHERE 1=1", list[Any]()
        sql, args = _filters(sql, args, cameras, since, until)
        if kinds and table == "sightings":
            sql += f" AND kind IN ({','.join('?' * len(kinds))})"
            args += kinds
        with self._lock:
            rows = self._db.execute(sql, args).fetchall()
        if not rows:
            return []
        matrix = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
        q = query.astype(np.float32).ravel()
        q = q / (np.linalg.norm(q) + 1e-12)
        scores = matrix @ q
        order = np.argsort(-scores)
        out = []
        for i in order[: top_k if min_score is None else len(order)]:
            if min_score is not None and scores[i] < min_score:
                break
            r = rows[i]
            keys = r.keys()
            out.append(Match(
                table=table, id=r["id"], camera=r["camera"], time=r["time"],
                score=float(scores[i]), track_id=r["track_id"],
                label=r["label"] if "label" in keys else None,
                kind=r["kind"] if "kind" in keys else "face",
                box=json.loads(r["box"]) if "box" in keys and r["box"] else None,
                thumb=r["thumb"],
            ))
            if len(out) >= top_k:
                break
        return out

    def counts(self) -> dict[str, int]:
        with self._lock:
            return {t: self._db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                    for t in ("events", "sightings", "faces")}

    # ---------------------------------------------------------------- retention

    def purge(self, now: float | None = None) -> dict[str, int]:
        """Delete rows (and their JPEGs) older than the retention windows."""
        now = time.time() if now is None else now
        plan = {
            "events": (self.retention_days, "snapshot"),
            "sightings": (self.retention_days, "thumb"),
            "faces": (self.face_retention_days, "thumb"),
        }
        deleted = {}
        with self._lock:
            for table, (days, media_col) in plan.items():
                if days is None:
                    continue
                cutoff = now - days * 86400
                files = [r[0] for r in self._db.execute(
                    f"SELECT DISTINCT {media_col} FROM {table} WHERE time < ? AND {media_col}"
                    " IS NOT NULL", (cutoff,))]
                cur = self._db.execute(f"DELETE FROM {table} WHERE time < ?", (cutoff,))
                deleted[table] = cur.rowcount
                for rel in files:  # scene thumbs are shared by tiles: delete once, ignore misses
                    (self.media_dir / rel).unlink(missing_ok=True)
            self._db.commit()
        return deleted


def _blob(vector: np.ndarray) -> bytes:
    v = np.asarray(vector, dtype=np.float32).ravel()
    return (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32).tobytes()


def _filters(sql: str, args: list[Any], cameras: list[str] | None, since: float | None,
             until: float | None) -> tuple[str, list[Any]]:
    if cameras:
        sql += f" AND camera IN ({','.join('?' * len(cameras))})"
        args += cameras
    if since is not None:
        sql += " AND time >= ?"
        args.append(since)
    if until is not None:
        sql += " AND time < ?"
        args.append(until)
    return sql, args
