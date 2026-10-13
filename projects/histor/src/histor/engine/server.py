"""HTTP API over the event store: events, snapshots, text search and face search.

Runs next to the detection processes (one `main.py` per camera) and reads the same SQLite file.

    uv run --group server -m histor.engine.server --config configs/engine/search_example.yaml
    # then open http://127.0.0.1:8000/docs for the interactive API page

Endpoints
    GET  /health                          counts of stored events / sightings / faces
    GET  /cameras                         camera names from the config
    GET  /events?camera=&type=&since=&until=&limit=
    GET  /media/{path}                    snapshot / thumbnail JPEGs
    GET  /recordings/{path}               main-stream MP4s (see `video_url` in results)
    GET  /search?q=red shirt last friday  text search
    POST /search/face (photo, when=friday, camera=shop)   "did this person come in on Friday?"
    GET  /people?posture=lying&when=last night            people by pose and clothing colour
         (posture, arms=up|both_up, facing=camera, head=down, motion=running,
          carrying=suitcase, upper=red, lower=blue; needs `pose`)
    POST /maintenance/purge               apply retention now (also runs every hour)
    GET  /status                          per camera: online, tracked and alerts today, tamper
    GET  /                                the dashboard (dashboard.py)

Login: with `server.token` in the config, every route needs it (the dashboard's session cookie,
or `Authorization: Bearer <token>`); without one, the server refuses any --host but this machine.
Over a network, serve it through HTTPS (a reverse proxy). Face search results are biometric data.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from histor.engine.cameras import camera_aliases
from histor.engine.dashboard import add_dashboard, check_host, server_token
from histor.engine.faces import SAME_PERSON_THRESHOLD
from histor.engine.mainstream import Recordings
from histor.engine.search import SearchEngine
from histor.engine.store import EventStore

log = logging.getLogger(__name__)
VIDEO_LEAD_SECONDS = 3


def create_app(config: dict[str, Any], clip: Any = None, faces: Any = None) -> Any:
    """Build the FastAPI app. `clip` / `faces` can be injected (tests); otherwise they are
    loaded from the config's `search` / `faces` sections when those are enabled."""
    store = EventStore.from_config(config.get("store"))
    if store is None:
        raise SystemExit("The config needs a `store:` section (see configs/engine/search_example.yaml)")
    search_cfg = config.get("search") or {}
    faces_cfg = config.get("faces") or {}
    if clip is None and search_cfg.get("enabled"):
        from histor.engine.clip_onnx import ClipEncoder

        clip_cfg = search_cfg.get("clip") or {}
        clip = ClipEncoder(image=None, text=clip_cfg.get("text", "models/clip_text.onnx"),
                           vocab=clip_cfg.get("vocab", "models/clip_bpe_vocab.txt.gz"),
                           device=clip_cfg.get("device", "auto"))
    if faces is None and faces_cfg.get("enabled"):
        from histor.engine.faces import FaceEngine

        faces = FaceEngine.from_config(faces_cfg, for_photos=True)
    cameras = sorted((config.get("cameras") or {}).keys())
    engine = SearchEngine(store, clip, faces, camera_names=cameras,
                          camera_aliases=camera_aliases(config.get("cameras")))
    rec_cfg = config.get("recording") or {}
    recordings = Recordings(rec_cfg.get("dir", "data/store/recordings"),
                            rec_cfg.get("segment_seconds", 60))

    @asynccontextmanager
    async def lifespan(_app: Any):  # hourly retention purge
        async def purge_loop() -> None:
            while True:
                deleted = await asyncio.to_thread(store.purge)
                if any(deleted.values()):
                    log.info("Retention purge: %s", deleted)
                await asyncio.sleep(3600)

        task = asyncio.create_task(purge_loop())
        yield
        task.cancel()

    app = FastAPI(title="Histor API", version="0.1.0", lifespan=lifespan)

    def media_url(rel: str | None) -> str | None:
        return f"/media/{rel}" if rel else None

    def with_urls(items: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
        for item in items:
            item[f"{key}_url"] = media_url(item.get(key))
            # the moment in the recording, a few seconds early to see it happen
            found = recordings.find_lead(item["camera"], item["time"], VIDEO_LEAD_SECONDS)
            item["video_url"] = (f"/recordings/{found[0]}#t={max(0, int(found[1]))}"
                                 if found else None)
        return items

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "counts": store.counts(), "text_search": clip is not None,
                "face_search": faces is not None}

    @app.get("/cameras")
    def list_cameras() -> dict[str, Any]:
        return {"cameras": cameras}

    @app.get("/events")
    def events(camera: str | None = None, type: list[str] | None = Query(None),  # noqa: A002
               since: datetime | None = None, until: datetime | None = None,
               limit: int = Query(100, le=1000)) -> dict[str, Any]:
        rows = store.list_events(camera, type, since.timestamp() if since else None,
                                 until.timestamp() if until else None, limit)
        return {"events": with_urls(rows, "snapshot")}

    @app.get("/media/{path:path}")
    def media(path: str) -> FileResponse:
        target = store.media_path(path)
        if target is None:
            raise HTTPException(404, "Not found")
        return FileResponse(target, media_type="image/jpeg")

    @app.get("/recordings/{path:path}")
    def recording(path: str) -> FileResponse:
        target = recordings.path(path)
        if target is None:
            raise HTTPException(404, "Not found")
        return FileResponse(target, media_type="video/mp4")  # supports range requests (seeking)

    @app.get("/search")
    def text_search(q: str, camera: list[str] | None = Query(None),
                    limit: int = Query(20, le=100)) -> dict[str, Any]:
        if clip is None:
            raise HTTPException(503, "Text search is not enabled (search.enabled in the config)")
        result = engine.text(q, cameras=camera, top_k=limit)
        with_urls(result["results"], "thumb")
        return result

    @app.post("/search/face")
    async def face_search(photo: UploadFile = File(...), when: str | None = Form(None),
                          camera: list[str] | None = Form(None),
                          threshold: float | None = Form(None),
                          face: int | None = Form(None),
                          limit: int = Form(20)) -> dict[str, Any]:
        """face: which face in a photo with several, 1 = leftmost. A photo whose face is too
        small, blurred or turned away comes back with face_found false and the reason."""
        if faces is None:
            raise HTTPException(503, "Face search is not enabled (faces.enabled in the config)")
        data = np.frombuffer(await photo.read(), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(400, "Could not read the photo (send a JPEG or PNG)")
        cams = [engine.camera_aliases.get(c.lower(), c) for c in camera] if camera else None
        result = engine.face(image, when=when, cameras=cams, top_k=limit, choice=face,
                             threshold=SAME_PERSON_THRESHOLD if threshold is None else threshold)
        with_urls(result["results"], "thumb")
        return result

    @app.get("/people")
    def people(when: str | None = None, camera: list[str] | None = Query(None),
               posture: list[str] | None = Query(None), arms: str | None = None,
               facing: str | None = None, upper: str | None = None, lower: str | None = None,
               head: str | None = None, motion: list[str] | None = Query(None),
               carrying: list[str] | None = Query(None),
               limit: int = Query(50, le=500)) -> dict[str, Any]:
        """People recorded with body poses: e.g. posture=lying&when=last night, or
        upper=red&lower=blue. Facts are exact per visit (see pose.py)."""
        from histor.engine.pose import CARRIED, COLOURS, FACING, HEAD, MOTIONS, POSTURES

        # arms: only the two filters match_people knows (anything else would mean "up")
        for name, value, allowed in (("arms", arms, ("up", "both_up")), ("facing", facing, FACING),
                                     ("upper", upper, COLOURS), ("lower", lower, COLOURS),
                                     ("head", head, HEAD),
                                     *(("posture", p, POSTURES) for p in posture or []),
                                     *(("motion", m, MOTIONS) for m in motion or []),
                                     *(("carrying", c, CARRIED) for c in carrying or [])):
            if value is not None and value not in allowed:
                raise HTTPException(400, f"{name}={value!r}; use one of {list(allowed)}")
        parsed = engine.parse(when or "")
        since, until = parsed.range()
        cams = [engine.camera_aliases.get(c.lower(), c) for c in camera] if camera else             parsed.cameras or None
        visits = engine.people(since, until, cams, posture, arms, facing, upper, lower,
                               head=head, motion=motion, carrying=carrying)
        shown = engine.add_thumbs(visits[-limit:][::-1])  # newest first
        return {"when": parsed.when or "any time", "total": len(visits),
                "results": with_urls([v.to_dict() for v in shown], "thumb")}

    @app.get("/plates")
    def plates(plate: str | None = None, when: str | None = None,
               camera: list[str] | None = Query(None), max_distance: int = Query(1, ge=0, le=3),
               limit: int = Query(50, le=500)) -> dict[str, Any]:
        """Number plates read on vehicles: e.g. plate=AB123CD&when=yesterday (look-alike
        characters and one wrong character forgiven), or just when=today for every plate."""
        parsed = engine.parse(when or "")
        since, until = parsed.range()
        cams = [engine.camera_aliases.get(c.lower(), c) for c in camera] if camera else \
            parsed.cameras or None
        visits = engine.plates(plate, since, until, cams, max_distance)
        shown = visits[:limit] if plate else visits[-limit:][::-1]  # best match / newest first
        return {"when": parsed.when or "any time", "plate": plate, "total": len(visits),
                "results": with_urls([v.to_dict() for v in shown], "thumb")}

    @app.post("/maintenance/purge")
    def purge() -> dict[str, Any]:
        return {"deleted": store.purge()}

    add_dashboard(app, store, cameras, server_token(config.get("server")))
    app.state.store = store
    app.state.engine = engine
    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Histor API server")
    parser.add_argument("--config", type=Path, default=Path("configs/engine/search_example.yaml"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    check_host(args.host, server_token(config.get("server")))
    app = create_app(config)
    print(f"* Dashboard on http://{args.host}:{args.port}/  (API docs: /docs)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
