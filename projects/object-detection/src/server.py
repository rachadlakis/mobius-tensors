"""HTTP API over the event store: events, snapshots, text search and face search.

Runs next to the detection processes (one `main.py` per camera) and reads the same SQLite file.

    uv run --group server --group search src/server.py --config configs/search_example.yaml
    # then open http://127.0.0.1:8000/docs for the interactive API page

Endpoints
    GET  /health                          counts of stored events / sightings / faces
    GET  /cameras                         camera names from the config
    GET  /events?camera=&type=&since=&until=&limit=
    GET  /media/{path}                    snapshot / thumbnail JPEGs
    GET  /search?q=red shirt last friday  text search
    POST /search/face (photo, when=friday, camera=shop)   "did this person come in on Friday?"
    POST /maintenance/purge               apply retention now (also runs every hour)

No authentication yet: bind to 127.0.0.1 (default) or put it behind a reverse proxy with auth
before exposing it on a network. Face search results are biometric data.
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

from faces import SAME_PERSON_THRESHOLD
from search import SearchEngine
from store import EventStore

log = logging.getLogger(__name__)


def create_app(config: dict[str, Any], clip: Any = None, faces: Any = None) -> Any:
    """Build the FastAPI app. `clip` / `faces` can be injected (tests); otherwise they are
    loaded from the config's `search` / `faces` sections when those are enabled."""
    store = EventStore.from_config(config.get("store"))
    if store is None:
        raise SystemExit("The config needs a `store:` section (see configs/search_example.yaml)")
    search_cfg = config.get("search") or {}
    faces_cfg = config.get("faces") or {}
    if clip is None and search_cfg.get("enabled"):
        from clip_onnx import ClipEncoder

        clip_cfg = search_cfg.get("clip") or {}
        clip = ClipEncoder(image=None, text=clip_cfg.get("text", "models/clip_text.onnx"),
                           vocab=clip_cfg.get("vocab", "models/clip_bpe_vocab.txt.gz"))
    if faces is None and faces_cfg.get("enabled"):
        from faces import FaceEngine

        faces = FaceEngine(**{k: v for k, v in faces_cfg.items()
                              if k in {"detector", "recognizer", "score_threshold"}},
                           min_face_size=20)  # uploaded photos: accept smaller faces
    cameras = sorted((config.get("cameras") or {}).keys())
    engine = SearchEngine(store, clip, faces, camera_names=cameras)

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

    app = FastAPI(title="Mobius Tensors API", version="0.1.0", lifespan=lifespan)

    def media_url(rel: str | None) -> str | None:
        return f"/media/{rel}" if rel else None

    def with_urls(items: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
        for item in items:
            item[f"{key}_url"] = media_url(item.get(key))
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
                          limit: int = Form(20)) -> dict[str, Any]:
        if faces is None:
            raise HTTPException(503, "Face search is not enabled (faces.enabled in the config)")
        data = np.frombuffer(await photo.read(), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(400, "Could not read the photo (send a JPEG or PNG)")
        result = engine.face(image, when=when, cameras=camera, top_k=limit,
                             threshold=SAME_PERSON_THRESHOLD if threshold is None else threshold)
        with_urls(result["results"], "thumb")
        return result

    @app.post("/maintenance/purge")
    def purge() -> dict[str, Any]:
        return {"deleted": store.purge()}

    app.state.store = store
    app.state.engine = engine
    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Mobius Tensors API server")
    parser.add_argument("--config", type=Path, default=Path("configs/search_example.yaml"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = yaml.safe_load(args.config.read_text())
    app = create_app(config)
    print(f"* API on http://{args.host}:{args.port}  (interactive docs: /docs)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
