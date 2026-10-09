"""Event store: SQLite for events, search index rows, face prints and body poses, plus JPEG media
on disk.

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

from argus.engine.clip_onnx import LEGACY_MODEL

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
    embedding BLOB NOT NULL        -- float32, L2-normalized; empty when recorded without CLIP
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

-- Body poses (pose_index.py): one row when a person's pose facts change, and at least every
-- heartbeat while they stay the same. The facts are plain columns, so they are searched with SQL.
CREATE TABLE IF NOT EXISTS poses (
    id INTEGER PRIMARY KEY,
    camera TEXT NOT NULL,
    time REAL NOT NULL,
    track_id INTEGER,
    posture TEXT,                  -- standing | sitting | crouching | bending | lying | unknown
    arms TEXT,                     -- down | one_up | both_up | unknown
    facing TEXT,                   -- camera | side | away | unknown
    head TEXT,                     -- up | down | unknown
    motion TEXT,                   -- still | walking | running | unknown
    carrying TEXT,                 -- e.g. "backpack,suitcase" (pose.CARRIED); NULL when nothing
    upper_colour TEXT,             -- torso colour (pose.COLOURS); NULL when unclear
    lower_colour TEXT,             -- legs colour
    score REAL,                    -- mean confidence of the visible keypoints
    box TEXT,                      -- json [x1, y1, x2, y2] in frame pixels
    keypoints BLOB                 -- 17 x (x, y, conf) float16, relative to box; empty if off
);
CREATE INDEX IF NOT EXISTS poses_time ON poses(time);

-- Number plates (plate_index.py): one row per new text read on a vehicle (or a clearly better
-- read of it), a few per vehicle at most. A vehicle's plate is the vote over its rows.
CREATE TABLE IF NOT EXISTS plates (
    id INTEGER PRIMARY KEY,
    camera TEXT NOT NULL,
    time REAL NOT NULL,
    track_id INTEGER,
    label TEXT,                    -- vehicle class: car | truck | bus | motorcycle
    text TEXT NOT NULL,            -- upper-case letters and digits only
    confidence REAL,               -- lowest character probability of the read
    region TEXT,                   -- country guessed by the OCR model; NULL when unsure
    box TEXT,                      -- vehicle box, json [x1, y1, x2, y2] in frame pixels
    thumb TEXT                     -- the plate crop
);
CREATE INDEX IF NOT EXISTS plates_time ON plates(time);
CREATE INDEX IF NOT EXISTS plates_text ON plates(text);
"""

POSE_COLUMNS = ("posture", "arms", "facing", "head", "motion", "carrying", "upper_colour",
                "lower_colour")
# Columns added after the first release: stores made before get them on open (NULL in old rows).
ADDED_COLUMNS = {"poses": ("head", "motion", "carrying"),
                 # the plate read (watchlist `plate` events), how many (`crowd` events)
                 "events": ("plate", "count INTEGER"),
                 "sightings": ("model",)}  # CLIP model of the embedding (empty = legacy)


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
        plate_retention_days: float | None = None,
    ) -> None:
        """plate_retention_days: number plates are personal data in many places; None keeps
        them as long as the rest (retention_days)."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.media_dir = Path(media_dir) if media_dir else self.path.parent / "media"
        self.media_dir.mkdir(parents=True, exist_ok=True)
        self.retention_days = retention_days
        self.face_retention_days = face_retention_days
        self.plate_retention_days = plate_retention_days if plate_retention_days is not None \
            else retention_days
        self.thumb_height = thumb_height
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(SCHEMA)
        for table, columns in ADDED_COLUMNS.items():
            have = {r[1] for r in self._db.execute(f"PRAGMA table_info({table})")}
            for column in columns:  # "name" (TEXT) or "name TYPE"
                name, *kind = column.split()
                if name not in have:
                    self._db.execute(
                        f"ALTER TABLE {table} ADD COLUMN {name} {kind[0] if kind else 'TEXT'}")
        self._db.commit()

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
                " dwell_seconds, snapshot, plate, count) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (camera, t, event["type"], event["rule"], event.get("label"),
                 event.get("track_id"), event.get("direction"), event.get("dwell_seconds"),
                 snapshot, event.get("plate"), event.get("count")),
            )
            self._db.commit()
            return int(cur.lastrowid or 0)

    def add_sightings(self, rows: list[dict[str, Any]]) -> None:
        """rows: camera, time, kind, track_id, label, box, thumb, embedding (np.float32),
        model (the CLIP model that made the embedding)."""
        if not rows:
            return
        with self._lock:
            self._db.executemany(
                "INSERT INTO sightings (camera, time, kind, track_id, label, box, thumb, embedding,"
                " model) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(r["camera"], r["time"], r["kind"], r.get("track_id"), r.get("label"),
                  json.dumps([round(float(v), 1) for v in r["box"]]) if r.get("box") is not None
                  else None, r.get("thumb"), _blob(r.get("embedding")),
                  r.get("model") if r.get("embedding") is not None else None) for r in rows],
            )
            self._db.commit()

    def add_faces(self, rows: list[dict[str, Any]]) -> list[int]:
        """rows: camera, time, track_id, quality, thumb, embedding. Returns the new row ids."""
        ids = []
        with self._lock:
            for r in rows:
                cur = self._db.execute(
                    "INSERT INTO faces (camera, time, track_id, quality, thumb, embedding)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (r["camera"], r["time"], r.get("track_id"), r.get("quality"), r.get("thumb"),
                     _blob(r["embedding"])))
                ids.append(int(cur.lastrowid or 0))
            self._db.commit()
        return ids

    def add_poses(self, rows: list[dict[str, Any]]) -> None:
        """rows: camera, time, track_id, the POSE_COLUMNS facts, score, box, keypoints
        (bytes)."""
        if not rows:
            return
        with self._lock:
            self._db.executemany(
                f"INSERT INTO poses (camera, time, track_id, {', '.join(POSE_COLUMNS)}, score,"
                f" box, keypoints) VALUES ({', '.join('?' * (len(POSE_COLUMNS) + 6))})",
                [(r["camera"], r["time"], r.get("track_id"), *(r.get(c) for c in POSE_COLUMNS),
                  r.get("score"),
                  json.dumps([round(float(v), 1) for v in r["box"]]) if r.get("box") is not None
                  else None, r.get("keypoints") or b"") for r in rows])
            self._db.commit()

    def delete_faces(self, ids: list[int]) -> None:
        """Delete face prints and their thumbnails (e.g. a track's worse faces)."""
        if not ids:
            return
        marks = ",".join("?" * len(ids))
        with self._lock:
            thumbs = [r[0] for r in self._db.execute(
                f"SELECT thumb FROM faces WHERE id IN ({marks}) AND thumb IS NOT NULL", ids)]
            self._db.execute(f"DELETE FROM faces WHERE id IN ({marks})", ids)
            self._db.commit()
        for rel in thumbs:
            (self.media_dir / rel).unlink(missing_ok=True)

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
               top_k: int | None = 50, model: str | None = None) -> list[Match]:
        """Cosine similarity of `query` against stored embeddings, best first.

        query: one vector, or several (one per row: e.g. 3 photos of the same person); a row's
        score is then its best match among them. top_k None: every row.
        model: only sightings embedded by this CLIP model (vectors of two models can't be
        compared); None: all of them."""
        assert table in {"sightings", "faces"}
        sql, args = f"SELECT * FROM {table} WHERE length(embedding) > 0", list[Any]()
        sql, args = _filters(sql, args, cameras, since, until)
        if model and table == "sightings":
            sql += " AND COALESCE(model, ?) = ?"
            args += [LEGACY_MODEL, model]
        if kinds and table == "sightings":
            sql += f" AND kind IN ({','.join('?' * len(kinds))})"
            args += kinds
        with self._lock:
            rows = self._db.execute(sql, args).fetchall()
        if not rows:
            return []
        matrix = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
        q = np.atleast_2d(np.asarray(query, dtype=np.float32))
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
        scores = (matrix @ q.T).max(axis=1)
        order = np.argsort(-scores)
        top_k = len(order) if top_k is None else top_k
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

    def add_plates(self, rows: list[dict[str, Any]]) -> None:
        """rows: camera, time, track_id, label, text, confidence, region, box, thumb."""
        if not rows:
            return
        with self._lock:
            self._db.executemany(
                "INSERT INTO plates (camera, time, track_id, label, text, confidence, region, box,"
                " thumb) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(r["camera"], r["time"], r.get("track_id"), r.get("label"), r["text"],
                  r.get("confidence"), r.get("region"),
                  json.dumps([round(float(v), 1) for v in r["box"]]) if r.get("box") is not None
                  else None, r.get("thumb")) for r in rows])
            self._db.commit()

    def plate_rows(self, since: float | None = None, until: float | None = None,
                   cameras: list[str] | None = None) -> list[dict[str, Any]]:
        """Plate reads in a time range, oldest first."""
        sql, args = _filters("SELECT * FROM plates WHERE 1=1", [], cameras, since, until)
        with self._lock:
            rows = self._db.execute(sql + " ORDER BY time", args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["box"] = json.loads(d["box"]) if d.get("box") else None
            out.append(d)
        return out

    def embedding_models(self) -> dict[str, int]:
        """Stored sighting vectors per CLIP model, e.g. to tell how many a model switch hides."""
        with self._lock:
            rows = self._db.execute(
                "SELECT COALESCE(model, ?) AS m, COUNT(*) FROM sightings WHERE length(embedding) > 0"
                " GROUP BY m", (LEGACY_MODEL,)).fetchall()
        return {r[0]: r[1] for r in rows}

    def pose_rows(self, since: float | None = None, until: float | None = None,
                  cameras: list[str] | None = None) -> list[dict[str, Any]]:
        """Pose rows in a time range, oldest first (without the keypoints)."""
        sql, args = _filters("SELECT id, camera, time, track_id, " + ", ".join(POSE_COLUMNS)
                             + ", score, box FROM poses WHERE 1=1", [], cameras, since, until)
        with self._lock:
            rows = self._db.execute(sql + " ORDER BY time", args).fetchall()
        return [dict(r) for r in rows]

    def pose_keypoints(self, row_id: int) -> tuple[bytes, list[float]] | None:
        """One row's stored keypoints and box (for pose.unpack_keypoints)."""
        with self._lock:
            row = self._db.execute("SELECT keypoints, box FROM poses WHERE id = ?",
                                   (row_id,)).fetchone()
        if row is None or not row[0] or not row[1]:
            return None
        return row[0], json.loads(row[1])

    def object_thumb(self, camera: str, track_id: int, first: float, last: float,
                     window: float = 60.0) -> str | None:
        """A thumbnail of a tracked object from the search index (the latest in the visit)."""
        with self._lock:
            row = self._db.execute(
                "SELECT thumb FROM sightings WHERE camera = ? AND track_id = ? AND kind = 'object'"
                " AND time >= ? AND time <= ? AND thumb IS NOT NULL ORDER BY time DESC LIMIT 1",
                (camera, track_id, first - window, last + window)).fetchone()
        return row[0] if row else None

    def face_counts(self, since: float | None = None, until: float | None = None,
                    cameras: list[str] | None = None) -> dict[str, int]:
        """Face prints stored per camera in a time range: what a face search actually checked."""
        sql, args = _filters("SELECT camera, COUNT(*) FROM faces WHERE 1=1", [], cameras, since,
                             until)
        with self._lock:
            return dict(self._db.execute(sql + " GROUP BY camera", args).fetchall())

    def latest_faces(self, limit: int = 12, camera: str | None = None) -> list[dict[str, Any]]:
        """The most recently stored faces (time, camera, track, quality, thumb), newest first."""
        sql, args = _filters("SELECT id, camera, time, track_id, quality, thumb FROM faces "
                             "WHERE 1=1", [], [camera] if camera else None, None, None)
        with self._lock:
            rows = self._db.execute(sql + " ORDER BY id DESC LIMIT ?", [*args, limit]).fetchall()
        return [dict(r) for r in rows]

    def track_span(self, camera: str, track_id: int, first: float, last: float,
                   window: float = 1800.0) -> tuple[float, float]:
        """When a tracked object was in view, from its search-index rows: the person who showed
        their face at 13:44 may have been seen from behind since 13:40. `window` keeps it to the
        same visit (track ids start again from 1 when a recorder restarts)."""
        with self._lock:
            row = self._db.execute(
                "SELECT MIN(time), MAX(time) FROM sightings WHERE camera = ? AND track_id = ? "
                "AND kind = 'object' AND time >= ? AND time <= ?",
                (camera, track_id, first - window, last + window)).fetchone()
        if row is None or row[0] is None:
            return first, last
        return min(first, row[0]), max(last, row[1])

    def counts(self) -> dict[str, int]:
        with self._lock:
            return {t: self._db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                    for t in ("events", "sightings", "faces", "poses", "plates")}

    # ---------------------------------------------------------------- retention

    def purge(self, now: float | None = None) -> dict[str, int]:
        """Delete rows (and their JPEGs) older than the retention windows."""
        now = time.time() if now is None else now
        plan: dict[str, tuple[float | None, str | None]] = {
            "events": (self.retention_days, "snapshot"),
            "sightings": (self.retention_days, "thumb"),
            "faces": (self.face_retention_days, "thumb"),
            "poses": (self.retention_days, None),
            "plates": (self.plate_retention_days, "thumb"),
        }
        deleted = {}
        with self._lock:
            for table, (days, media_col) in plan.items():
                if days is None:
                    continue
                cutoff = now - days * 86400
                files = [r[0] for r in self._db.execute(
                    f"SELECT DISTINCT {media_col} FROM {table} WHERE time < ? AND {media_col}"
                    " IS NOT NULL", (cutoff,))] if media_col else []
                cur = self._db.execute(f"DELETE FROM {table} WHERE time < ?", (cutoff,))
                deleted[table] = cur.rowcount
                for rel in files:  # scene thumbs are shared by tiles: delete once, ignore misses
                    (self.media_dir / rel).unlink(missing_ok=True)
            self._db.commit()
        return deleted


def _blob(vector: np.ndarray | None) -> bytes:
    if vector is None:
        return b""
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
