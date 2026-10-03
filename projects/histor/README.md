# Histor

*Greek ἵστωρ: "witness, one who knows by having seen". The root of "history".*

Histor records **everything** a camera sees as searchable embeddings, so you can ask about it
later: *"was there a white van at the gate yesterday evening?"*, *"when did someone leave a bag
by the bench?"*.

Where [object-detection](../object-detection/) is about **alerts** (fire, perimeter, plates:
watch for one thing and react now), Histor is about **memory**: store all of it, ask any question
later.

## How it works

1. **Record** (one process per camera): detect and track every object (all 80 COCO classes),
   embed each tracked object with CLIP, and every few seconds embed the whole frame plus a 2x2
   grid of tiles, so things the detector has no class for are still searchable. Everything goes
   into one SQLite store per site.
2. **Ask**: a question is split into *what / when / which camera*. The *what* is embedded with
   CLIP's text model and compared with the stored vectors. Matches of the same tracked object
   are merged into one visit.
3. **Chat** (next): an LLM gets the search as a tool, asks follow-up searches itself, and
   answers in sentences with thumbnails as evidence.

Histor doesn't copy any code. It imports the detector, tracker, CLIP encoder, store and search
from `../object-detection/src` (see `src/shared.py`) and uses the models in
`../object-detection/models`.

## Quick start

Models (once, in the object-detection project):

```bash
cd ../object-detection
uv run src/download_model.py yolox_nano
uv run --group export src/export_clip.py      # CLIP -> models/clip_*.onnx (~600 MB)
```

Then here:

```bash
uv sync
uv run src/record.py --source ../object-detection/data/vtest.avi --camera street --start-time 2026-09-25T14:00
uv run src/ask.py "person with a bag on street"
uv run src/ask.py                              # interactive, empty line quits
```

Cameras are declared under `cameras:` in `configs/default.yaml`, the same way as in
[object-detection](../object-detection/README.md#ip-cameras-poe). For the HTTP API
(`/search`, `/search/face`, `/events`), run object-detection's `src/server.py` with this
project's config.

## Cost of "everything"

Embedding every class costs more CPU and storage than the people-and-vehicles default in
object-detection. A CLIP crop takes ~55 ms on a laptop CPU, and each embedding is one 512-float
row (2 KB). On live cameras the indexer drops jobs instead of slowing detection down, so a busy
scene gets sampled more sparsely rather than lagging. Tune `object_interval_seconds`,
`scene_interval_seconds` and `min_box_height` per site.

## Layout

```text
configs/default.yaml   all COCO classes embedded, store in data/store/histor.db
src/
  shared.py            puts ../object-detection/src on the import path
  record.py            recorder (object-detection's main.py with this config)
  ask.py               question -> ranked sightings (CLI / interactive)
tests/                 uv run pytest
```

## Next

- [ ] Chat layer: an LLM with tools `search(text, when, camera)`, `events(...)`, `count(...)`.
      Decide between a local model (offline, CPU servers) and an API model.
- [ ] Answers backed by evidence: thumbnails and clip links for each claim.
- [ ] Captions per scene (small VLM) stored next to the vectors, for questions CLIP handles badly
      (counting, "what was he doing").
- [ ] Multi-camera timeline: follow one visit across cameras.
