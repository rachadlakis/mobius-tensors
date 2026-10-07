# Argus

*After Argus Panoptes, the hundred-eyed watchman of Greek myth (πανόπτης, "all-seeing"), who never closed all his eyes at once.*

Argus records **everything** a camera sees as searchable embeddings, so you can ask about it
later: *"was there a white van at the gate yesterday evening?"*, *"when did someone leave a bag
by the bench?"*.

Argus is about **memory**: store all of it, ask any question later. It also **alerts**: the
same recorder runs [object-detection](../object-detection/)'s area and line rules live and sends
what they catch to channels such as webhook and email (see [Alerts](#alerts)).

## How it works

1. **Record** (one process per camera): detect and track every object (all 80 COCO classes),
   embed each tracked object with CLIP, and every few seconds embed the whole frame plus a 2x2
   grid of tiles, so things the detector has no class for are still searchable. Everything goes
   into one SQLite store per site.
2. **Ask**: a question is split into *what / when / which camera*. The *what* is embedded with
   CLIP's text model and compared with the stored vectors. Matches of the same tracked object
   are merged into one visit.
3. **Chat**: an LLM gets the store as three tools, runs as many lookups as the question needs,
   and answers in sentences, listing the thumbnails it looked at as evidence.
4. **Alert** (optional): areas and lines per camera fire events while recording; each is stored
   with a snapshot and sent to the channels whose filters accept it.

Argus doesn't copy any code. It imports the detector, tracker, CLIP encoder, store and search
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

What the model can look up (all tools take a time range in words or exact ISO `since`/`until`
and a camera; `objects` and `search` also take `with_labels` = "on the same camera at the same
time as"):

| Question | Tool call |
| --- | --- |
| What happened yesterday 6pm to 9pm? | `objects(when="yesterday 6pm to 9pm")`: everything tracked, in time order, with counts |
| How many cars came in today? | `objects(labels=["car"], when="today")`: exact, one tracked object = one visit |
| Anyone with a dog this morning? | `objects(labels=["person"], with_labels=["dog"], when="this morning")` |
| A man in a red jacket near a car? | `search(query="man in a red jacket", with_labels=["car"])`: CLIP, ranked guesses |
| Did any alert go off last night? | `alerts(when="last night")`: what the alert rules fired, exact |
| Who entered the yard? | `alerts(rules=["yard"], types=["enter"])` |
| *(photo attached)* Did this person come to the office between 1 and 4? | `face(photo="photo1", since="... 13:00", until="... 16:00", camera="office")`: likely / possible visits |

"Together" means the same camera and overlapping in time (`within_seconds`, default 15), not
touching in the image.

In the interactive chat, `/model claude` switches model (and starts a new conversation),
`/models` lists them. Tool calls are printed as they happen (`--quiet` hides them).

### Looking for a person from a photo

Needs `faces.enabled: true` (biometric data: client approval first, see the
[object-detection README](../object-detection/README.md#face-search-and-privacy)), and only
finds people on cameras that were recording faces at the time.

```text
you> /photo visitor.jpg
Attached as photo1
you> did this person come to the office between 1 and 4?
  -> face {'photo': 'photo1', 'since': '2026-10-07 13:00', 'until': '2026-10-07 16:00', 'camera': 'office'}
Likely yes: on the office camera from 13:40 to 13:45 (face clearly seen at 13:44, match 0.52).
```

- `/photo a.jpg b.jpg` attaches several photos of the same person (each stored face is
  compared with the best of them); `--face 2` picks a face in a group photo (1 = leftmost).
  One-shot: `uv run src/chat.py --photo visitor.jpg "did this person come in today?"`.
- The photo never leaves the machine and is not saved: its face print is kept in memory for
  the session, and the model only learns its name (`photo1`), also with a cloud model.
- A photo with a face that is too small, blurred or turned away is refused with the reason.
- Answers are "likely" or "possible" (check the thumbnail). With no match, the model says how
  many faces the checked cameras recorded and whether they can see faces: no match is not proof
  the person was not there.

Any OpenAI-compatible server works as `api: openai`: llama.cpp's `llama-server`
(`base_url: http://127.0.0.1:8080/v1`), LM Studio, vLLM, or a hosted one with
`api_key: ${SOME_KEY}`. Add an entry, no code change. With a cloud model, the question and the
search results (times, cameras, labels, thumbnail paths) are sent, never images or video.

Small local models follow the tool instructions less reliably than large ones: if answers look
wrong, run with tool calls visible and check which lookups it made.

## Alerts

Off until rules are listed. In `configs/default.yaml`:

```yaml
zones:
  - name: yard                  # area: enter / exit, and dwell after 60 s
    points: [[0.10, 0.55], [0.60, 0.50], [0.65, 0.95], [0.05, 0.95]]
    classes: [person]
    dwell_seconds: 60
  - name: gate_line             # line: cross, with direction in / out
    type: line
    points: [[0.40, 0.30], [0.40, 0.80]]
    classes: [car, truck]
  - name: anywhere_dog          # no points: anywhere in the picture
    classes: [dog]

alerts:
  cooldown_seconds: 0           # store every event; hold messages back per channel instead
  channels:
    - type: webhook
      url: ${ALERT_WEBHOOK_URL}
      cooldown_seconds: 30
    - type: email               # host, sender, to, username, password...
      labels: [person]
      hours: "22:00-06:00"
```

Draw areas and lines in object-detection's UI (**Zones & lines** tab) and paste the YAML. Each
channel filters by `rules`, `events`, `labels`, `hours` and `cooldown_seconds`. Every event goes
into the store with a snapshot, so the chat answers "did any alert go off last night?" from the
full list, whatever the channels sent. Channel details and the plan for Telegram, WhatsApp, SMS
and on-site sirens: [object-detection README](../object-detection/README.md#integrations-what-exists-and-what-comes-next).

Rules work on the detector's 80 classes today. New things to detect (fire and smoke, objects a
customer names) need a model trained for them; the rules and channels then apply unchanged.

## Cameras

Cameras are declared under `cameras:` in `configs/default.yaml`, the same way as in
[object-detection](../object-detection/README.md#ip-cameras-poe). The `recording:` section saves
each camera's main stream and crops from its full-resolution keyframes
([details](../object-detection/README.md#main-stream-recording-and-hi-res-crops)). For the HTTP API
(`/search`, `/search/face`, `/events`), run object-detection's `src/server.py` with this
project's config.

Give each camera an `about:` at install time, standing at the camera, so the chat knows what
it looks at (example in `configs/default.yaml`):

| Field | What it does |
| --- | --- |
| `description`, `location` | the model picks the right camera: "did they reach the office" means the office camera, not the corridor on the way |
| `aliases` | other names in questions ("reception" → `lobby`), for the chat and for `ask.py` |
| `covers` | entrance / exit / passage / room / outdoor / parking: "arrived" vs "passed by" |
| `faces` | good / poor / none: face search explains a "no" ("the lobby camera can't see faces well"), and tells the installer where to enable `faces.cameras` |
| `hours` | normal activity, so "after hours" questions need no extra input |
| `notes`, `zones` | what is normal ("cleaners at 19:00"), and what each alert area or line means |

It goes into the model's instructions as one short line per camera (cut at ~300 characters,
for small local models). Use roles ("manager's office"), not people's names: with a cloud
model, this text is sent too.

## Cost of "everything"

Embedding every class costs more CPU and storage than the people-and-vehicles default in
object-detection. A CLIP crop takes ~55 ms on a laptop CPU, and each embedding is one 512-float
row (2 KB). On live cameras the indexer drops jobs instead of slowing detection down, so a busy
scene gets sampled more sparsely rather than lagging. Tune `object_interval_seconds`,
`scene_interval_seconds` and `min_box_height` per site.

## Layout

```text
configs/default.yaml   all COCO classes embedded, store in data/store/argus.db
src/
  shared.py            puts ../object-detection/src on the import path
  record.py            recorder (object-detection's main.py with this config)
  ask.py               question -> ranked sightings (no LLM)
  chat.py              chat CLI: question -> LLM + tools -> answer + evidence
  tools.py             tools the LLM can call: objects (time / class / together), search, alerts
  llm.py               model backends: OpenAI-compatible (local) and Claude
tests/                 uv run pytest
```

## Next

- [x] Chat layer with switchable models (local by default).
- [x] Alerts: areas and lines, channels with filters (rules, events, labels, hours, cooldown),
      chat tool over the alert history.
- [ ] More alert channels: Telegram, WhatsApp, SMS, on-site siren / relay.
- [ ] Detectors trained per customer (fire and smoke, customer-specific objects).
- [ ] Chat in the browser: `/chat` endpoint + page showing the evidence thumbnails.
- [ ] Evals: a fixed set of questions on recorded footage to compare local models.
- [ ] Captions per scene (small VLM) stored next to the vectors, for questions CLIP handles badly
      (counting, "what was he doing").
- [ ] Multi-camera timeline: follow one visit across cameras.
