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
3. **Chat**: an LLM gets the store as two tools, runs as many lookups as the question needs,
   and answers in sentences, listing the thumbnails it looked at as evidence.

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

## Chat

The model is a switch in `configs/default.yaml` (`chat.model`, entries under `chat.models`).
Local is the default: the LLM runs on the site's server and nothing leaves it.

| Entry | Runs on | Setup |
| --- | --- | --- |
| `local` | Ollama, CPU (Qwen3 4B, Apache-2.0) | install [Ollama](https://ollama.com), `ollama pull qwen3:4b` |
| `local-large` | Ollama, ~20 GB RAM or GPU (Qwen3 30B-A3B) | `ollama pull qwen3:30b-a3b` |
| `claude` | Anthropic API (Claude Opus 5.5) | `uv sync --group claude`, `ANTHROPIC_API_KEY` in `.env` |

```bash
uv run src/chat.py                                        # interactive
uv run src/chat.py "how many cars came in yesterday?"
uv run src/chat.py --model claude "was there a dog on the lawn this morning?"
```

What the model can look up (both tools take the same filters: time range in words or exact
ISO `since`/`until`, camera, and `with_labels` = "on the same camera at the same time as"):

| Question | Tool call |
| --- | --- |
| What happened yesterday 6pm to 9pm? | `objects(when="yesterday 6pm to 9pm")`: everything tracked, in time order, with counts |
| How many cars came in today? | `objects(labels=["car"], when="today")`: exact, one tracked object = one visit |
| Anyone with a dog this morning? | `objects(labels=["person"], with_labels=["dog"], when="this morning")` |
| A man in a red jacket near a car? | `search(query="man in a red jacket", with_labels=["car"])`: CLIP, ranked guesses |

"Together" means the same camera and overlapping in time (`within_seconds`, default 15), not
touching in the image.

In the interactive chat, `/model claude` switches model (and starts a new conversation),
`/models` lists them. Tool calls are printed as they happen (`--quiet` hides them).

Any OpenAI-compatible server works as `api: openai`: llama.cpp's `llama-server`
(`base_url: http://127.0.0.1:8080/v1`), LM Studio, vLLM, or a hosted one with
`api_key: ${SOME_KEY}`. Add an entry, no code change. With a cloud model, the question and the
search results (times, cameras, labels, thumbnail paths) are sent, never images or video.

Small local models follow the tool instructions less reliably than large ones: if answers look
wrong, run with tool calls visible and check which lookups it made.

## Cameras

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
  ask.py               question -> ranked sightings (no LLM)
  chat.py              chat CLI: question -> LLM + tools -> answer + evidence
  tools.py             tools the LLM can call: objects (time / class / together), search
  llm.py               model backends: OpenAI-compatible (local) and Claude
tests/                 uv run pytest
```

## Next

- [x] Chat layer with switchable models (local by default).
- [ ] Chat in the browser: `/chat` endpoint + page showing the evidence thumbnails.
- [ ] Evals: a fixed set of questions on recorded footage to compare local models.
- [ ] Captions per scene (small VLM) stored next to the vectors, for questions CLIP handles badly
      (counting, "what was he doing").
- [ ] Multi-camera timeline: follow one visit across cameras.
