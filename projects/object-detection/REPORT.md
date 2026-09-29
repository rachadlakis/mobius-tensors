# Report: search, face search, alerts and API

2026-09-29 · object-detection project

## What was added

| Feature | What it does | Files |
| --- | --- | --- |
| **Event store** | Saves every zone/line event with a JPEG snapshot to one SQLite file per site; old data deleted automatically (30 days, face prints 7 days) | `store.py` |
| **Webhook and email alerts** | Sends each alert as JSON to a URL and/or as an email with the snapshot attached; background thread with retries | `alerts.py` |
| **Text search** | "person in a red top last friday", "white van on gate yesterday 6pm to 9pm": finds matching people, vehicles or scenes, grouped per visit | `clip_onnx.py`, `indexer.py`, `query.py`, `search.py` |
| **Face search by photo** | Upload a photo plus "friday" and get "Yes: seen 1 time, first on shop at Fri 25 Sep 10:30", or "No match". Off by default, per camera | `faces.py`, `search.py` |
| **API server** | FastAPI with `/events`, `/media`, `/search`, `/search/face`, `/health`; interactive docs at `/docs` | `server.py` |
| **CLIP export** | One-off script that turns OpenCLIP into ONNX, so servers need no PyTorch | `export_clip.py` |

Also: `--camera` and `--start-time` options in `main.py` (name a source and date a recorded
file), `download_model.py faces`, an example config `configs/search_example.yaml`, 17 new tests
(43 in total, all passing), and README sections for setup, the API and privacy.

## How it was checked

- **Text search** on the street sample video (80 s): 413 index rows. Top results were correct
  for "person with a red top" (the woman in the red jacket, 3 of the top 3), "person carrying a
  bag", "white van" (the parked van first) and "person in a black jacket".
- **Face search**: same person, flipped and resized photo → 0.89 similarity (match), a
  different person → 0.28 (no match). The full API flow answered "Yes" for the right day and
  camera, "No match" for the wrong day or a different person, and "No face found" for a dog photo.
- **The CLIP ONNX port** gives identical tokens and 1.000 cosine to the original PyTorch model.

## Decisions

- **No LLM**: dates, times and camera names are parsed by rules. An LLM can be added later for
  complex questions.
- **Not all 80 COCO classes**: only people, vehicles and bags are embedded as objects. Scene
  tiles cover everything else.
- **CLIP in float32**: int8 was 4× smaller but reordered search results too much.
- **Faces off by default**: they are biometric data. Needs client approval, notices, and a
  check of local law.

## Known limits and next steps

1. **The API has no login yet.** Keep it on 127.0.0.1 or behind a proxy with authentication.
2. **Faces need about 32+ pixels.** Detection uses the low-resolution sub-stream. Next step: cut
   face and object crops from the main stream.
3. **Search is brute-force over SQLite.** Fine for a pilot (millions of rows per second); move to
   sqlite-vec or pgvector for large sites.
4. **CLIP needs a one-time export** on a machine with internet (~600 MB of models).
5. **Next**: a dashboard page over this API, and a "deep scan" button that runs an
   open-vocabulary detector on a chosen time window for small objects.
