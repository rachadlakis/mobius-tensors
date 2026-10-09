# Argus

*After Argus Panoptes, the hundred-eyed watchman of Greek myth (πανόπτης, "all-seeing"), who never closed all his eyes at once.*

Argus records **everything** a camera sees as searchable embeddings, so you can ask about it
later: *"was there a white van at the gate yesterday evening?"*, *"when did someone leave a bag
by the bench?"*.

Argus is about **memory**: store all of it, ask any question later. It also **alerts**: the
same recorder runs the engine's area and line rules live and sends
what they catch to channels such as webhook and email (see [Alerts](#alerts)).

## How it works

1. **Record** (one process per camera): detect and track every object (all 80 COCO classes),
   embed each tracked object with CLIP, and every few seconds embed the whole frame plus a 2x2
   grid of tiles, so things the detector has no class for are still searchable. About once a
   second, each person's body pose (17 keypoints) becomes a few facts: posture, hands up, facing,
   head down, still / walking / running, a bag or suitcase on them, colour of top and trousers.
   Optionally, each vehicle's number plate is read and the camera's microphone listens for
   alarms and breaking glass. Everything goes into one SQLite store per site.
2. **Ask**: a question is split into *what / when / which camera*. The *what* is embedded with
   CLIP's text model and compared with the stored vectors. Matches of the same tracked object
   are merged into one visit.
3. **Chat**: an LLM gets the store as tools (objects, search, alerts, people, plates, face,
   look), runs as many lookups as the question needs, and answers in sentences, listing the
   thumbnails it looked at as evidence. `look` lets a local vision model read those images.
4. **Alert** (optional): areas, lines, crowding, abandoned bags, camera tampering, poses,
   watchlist plates and sounds fire events while recording; each is stored with a snapshot and
   sent to the channels whose filters accept it.
5. **Report** (optional): hourly, daily or weekly, a summary of what the chosen cameras saw,
   focused on what the owner asked to watch for, saved and sent (see [Reports](#reports)).

The detector, tracker, CLIP encoder, store and search are the engine in `src/argus/engine/`
([ENGINE.md](ENGINE.md)); the product modules sit on top in `src/argus/`. One uv environment,
models in `models/`.

## Quick start

```bash
uv sync
uv run -m argus.engine.download_model yolox_s      # the default detector (CPU sites)
uv run --group export -m argus.engine.export_clip      # CLIP -> models/clip_*.onnx (~600 MB), once
uv run -m argus.engine.download_model plates           # optional: number plates (plates: section)
uv run -m argus.engine.download_model sounds           # optional: sound events (sounds: section)
uv run -m argus.record --source data/vtest.avi --camera street --start-time 2026-09-25T14:00
uv run -m argus.ask "person with a bag on street"
uv run -m argus.ask                            # interactive, empty line quits
```

The models run on the GPU when there is one (`device: auto`, see [GPU](ENGINE.md#gpu)).
NVIDIA machine: `uv sync --no-group cpu --group gpu`.

## Test on a laptop

A laptop's webcam is enough to try recording, face search and the chat. Models: YOLOX and the
face models (`uv run -m argus.engine.download_model yolox_nano` and `... faces`; the laptop config
uses nano to stay fluid, sites use `yolox_s`);
CLIP is optional (without it, everything but search by description works), and so is the pose
model (`uv run --group export -m argus.engine.export_pose`): with it, the live view draws each person's
skeleton, and raising both hands for 3 s or lying down for 10 s fires an alert.

```bash
uv run --group ui -m argus.ui        # configs/laptop.yaml, opens http://127.0.0.1:7860
```

1. **Camera**: *Start recording* (source `0` = the built-in webcam). The live view shows the
   tracked objects, and *Faces stored* the face prints being kept (best 5 per visit). Leave the
   frame and come back: that's a second visit.
2. **Find a person**: upload a photo, or *Take photo from the live camera*, pick a time range
   (`today`, `last 10 minutes`) and *Search*: likely / possible visits with their best face.
   No LLM needed.
3. **Ask**: the chat, with *Attach photo* for face questions. Needs Ollama running
   (`ollama serve`, `ollama pull qwen3:4b`) or the `claude` model (`uv sync --group claude`,
   `ANTHROPIC_API_KEY` in `.env`).

The test store is `data/store/laptop.db` (delete `data/store/laptop*` to start over), separate
from a site's. Video files and camera URLs work as a source too.

## Chat

The model is a switch in `configs/default.yaml` (`chat.model`, entries under `chat.models`).
Local is the default: the LLM runs on the site's server and nothing leaves it.

| Entry | Runs on | Setup |
| --- | --- | --- |
| `local` | Ollama, CPU (Qwen3 4B, Apache-2.0) | install [Ollama](https://ollama.com), `ollama pull qwen3:4b` |
| `qwen3.5-4b` | Ollama, CPU (Qwen3.5 4B, Apache-2.0, also reads images) | `ollama pull qwen3.5:4b` |
| `granite-3b` | Ollama, CPU (IBM Granite 4.1 3B, Apache-2.0, smallest) | `ollama pull granite4.1:3b` |
| `gemma4-e4b` | Ollama, CPU (Gemma 4 E4B, Apache-2.0, also reads images) | `ollama pull gemma4:e4b` |
| `local-large` | Ollama, ~20 GB RAM or GPU (Qwen3 30B-A3B) | `ollama pull qwen3:30b-a3b` |
| `qwen3.6-large` | Ollama, ~32 GB RAM or GPU (Qwen3.6 35B-A3B, Apache-2.0) | `ollama pull qwen3.6:35b` |
| `nemotron-large` | Ollama, ~32 GB RAM or GPU (Nemotron 3.5 Lightning 30B-A3B, OpenMDW 1.1: read before shipping) | `ollama pull nemotron-3.5-lightning` |
| `claude` | Anthropic API (Claude Opus 5.5) | `uv sync --group claude`, `ANTHROPIC_API_KEY` in `.env` |

The open models after `local` were added on 2026-10-08 to compare on the same questions; the
default stays `qwen3:4b` until the planned evals pick a winner. The UI's *Ask* tab shows whether
the selected model's server is up and the model downloaded. `note:` in an entry is shown there.

```bash
uv run -m argus.chat                                      # interactive
uv run -m argus.chat "how many cars came in yesterday?"
uv run -m argus.chat --model claude "was there a dog on the lawn this morning?"
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
| Did anyone fall in the warehouse last night? | `people(postures=["lying"], when="last night")`: seconds lying per person, exact; also `alerts(types=["down"])` |
| A man in a red top and blue jeans this morning? | `people(upper_colour="red", lower_colour="blue", when="this morning")`: colours from body keypoints, exact; `search` for anything finer |
| Anyone with their hands up near the till? | `people(arms="up", camera="till")` |
| Was anyone running in the car park? | `people(motion=["running"], camera="car park")` |
| Who came in with a suitcase? | `people(carrying=["suitcase"], when="today")` |
| When did AB 123 CD come in? | `plates(plate="AB 123 CD")`: look-alikes (0/O, 8/B...) and one wrong character forgiven |
| Which plates came to the gate today? | `plates(when="today", camera="gate")` |
| How many people were at the counter at 3pm? | `objects(...)` or `search(...)` to find the moment, then `look(images=[...], question="How many people?")` |
| Did a smoke alarm go off last night? | `alerts(types=["sound"], rules=["smoke_alarm"], when="last night")` |
| Was any camera covered or turned? | `alerts(types=["tamper"])` |
| *(photo attached)* Did this person come to the office between 1 and 4? | `face(photo="photo1", since="... 13:00", until="... 16:00", camera="office")`: likely / possible visits |

"Together" means the same camera and overlapping in time (`within_seconds`, default 15), not
touching in the image.

In the interactive chat, `/model claude` switches model (and starts a new conversation),
`/models` lists them. Tool calls are printed as they happen (`--quiet` hides them).

### Looking for a person from a photo

Needs `faces.enabled: true` (biometric data: client approval first, see the
[engine docs](ENGINE.md#face-search-and-privacy)), and only
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
  One-shot: `uv run -m argus.chat --photo visitor.jpg "did this person come in today?"`.
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

Draw areas and lines in the test bench (**Zones & lines** tab) and paste the YAML. Each
channel filters by `rules`, `events`, `labels`, `hours` and `cooldown_seconds`. Every event goes
into the store with a snapshot, so the chat answers "did any alert go off last night?" from the
full list, whatever the channels sent. Channel details and the plan for Telegram, WhatsApp, SMS
and on-site sirens: [engine docs](ENGINE.md#integrations-what-exists-and-what-comes-next).

More event types, each in its own config section and through the same channels:

| Event | Rule | Section | Default |
| --- | --- | --- | --- |
| `crowd` | the area's name: `max_count` objects inside for `count_seconds` (queues, crowds) | a zone in `zones:` | — |
| `abandoned` | `abandoned_object`: a backpack, handbag or suitcase still and unattended for 60 s | `abandoned:` | off |
| `tamper` | `blocked` / `dark` / `blurred` / `moved` / `frozen`: the camera's own picture, for 20 s | `tamper:` | on |
| `plate` | the watchlist entry's name: a listed plate read twice on a vehicle | `plates:` | off (personal data) |
| `sound` | `smoke_alarm`, `glass_break`, `gunshot`, `explosion`, `scream`, `car_alarm`, `siren` | `sounds:` | off (listening is regulated) |

A one-way line (`direction: in`) is a wrong-way alarm. Sounds are classified in memory and never
stored; they need ffmpeg and a camera with a microphone. Details and thresholds:
[engine docs](ENGINE.md#more-rules-crowding-abandoned-objects-tampering).

Object rules work on the detector's 80 classes today. New things to detect (fire and smoke,
objects a customer names) need a model trained for them; the rules and channels then apply
unchanged.

Body poses add two event types: `down` (rule `person_down`: a person lying for 10 s) and
`hands_up` (both hands above the head for 3 s). Enable them in the `pose:` section
(`alerts: [down]` by default); they reach the same channels and the same `alerts` tool. Details,
models and limits: [engine docs](ENGINE.md#body-poses-what-people-were-doing).

## Reports

Set up at install time with the owner: how often, which cameras, what they care about. In
`configs/default.yaml`:

```yaml
reports:
  - name: morning
    every: daily                # hourly | daily | weekly (with day: monday)
    at: "07:00"                 # the period ends then: yesterday 07:00 to today 07:00
    cameras: [gate, yard]       # names or aliases; empty = every camera
    interests:                  # in the owner's words; empty = what happened + anything unusual
      - vehicles at the gate after hours
      - anyone in the yard at night
    channels:                   # same settings as alerts.channels; none = saved only
      - type: email
        host: smtp.example.com
        sender: argus@example.com
        to: [owner@example.com]
```

```bash
uv run -m argus.report                   # runs the schedules: keep it running next to the recorders
uv run -m argus.report --now morning     # the last period's report now (--no-send: save only)
uv run -m argus.report --list            # saved reports
uv run -m argus.report --show morning    # the latest one
```

Each report has two parts:

- **Counts**, exact, from the store: tracked objects per camera, alerts with their times, and
  any camera that recorded nothing in the period (offline?).
- **Summary**, written by the chat model (`model:`, default `chat.model`): it checks each
  interest with the chat's tools, limited to the report's cameras and period, and says what it
  found or that it found nothing. If the model is down, the report goes out with the counts only
  and says why.

Reports are saved as JSON and Markdown under `data/store/reports/<name>/` (the UI's *Reports* tab
shows them, with the evidence thumbnails, and makes one on demand), then sent: email with up to 4
thumbnails attached, webhook as JSON (`kind: report`, summary, counts, thumbnail paths). A new
schedule starts with the next period; a period that ended while `report.py` was stopped is made
when it starts again (the latest one only). With a cloud model, the counts and the interests are
sent to it, never images.

## Cameras

Cameras are declared under `cameras:` in `configs/default.yaml`, see
[IP cameras](ENGINE.md#ip-cameras-poe). The `recording:` section saves
each camera's main stream and crops from its full-resolution keyframes
([details](ENGINE.md#main-stream-recording-and-hi-res-crops)). For the HTTP API
(`/search`, `/search/face`, `/events`), run `uv run --group server -m argus.engine.server --config configs/default.yaml`.

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

Embedding every class costs more CPU and storage than the people-and-vehicles default of
`configs/engine/search_example.yaml`. A CLIP crop takes ~30 ms on a laptop CPU, and each embedding is one 512-float
row (2 KB). On live cameras the indexer drops jobs instead of slowing detection down, so a busy
scene gets sampled more sparsely rather than lagging. Tune `object_interval_seconds`,
`scene_interval_seconds` and `min_box_height` per site.

## Layout

```text
configs/default.yaml   all COCO classes embedded, store in data/store/argus.db
configs/laptop.yaml    test on a laptop: webcam = camera "laptop", faces on, own store
configs/engine/        engine-only examples (detection, zones, search)
src/argus/
  engine/              detector, tracker, zones, alerts, store, CLIP / faces / pose / plates / sounds,
                       tamper and abandoned rules, search, API (ENGINE.md)
  record.py            recorder (engine/main.py with this config)
  ask.py               question -> ranked sightings (no LLM)
  report.py            hourly / daily / weekly reports: exact counts + LLM summary, saved and sent
  chat.py              chat CLI: question -> LLM + tools -> answer + evidence
  tools.py             tools the LLM can call: objects, search, alerts, people, plates, face, look
  llm.py               model backends: OpenAI-compatible (local) and Claude
  vision.py            the local vision model behind `look` (chat.vision)
  ui.py                test UI (Gradio): record, find a person by photo, chat, reports
notebooks/             models_tour.ipynb: each model, step by step
models/                .onnx weights (git-ignored)
tests/                 uv run pytest (tests/engine/ for the engine)
```

## Next

- [x] Chat layer with switchable models (local by default).
- [x] Alerts: areas and lines, channels with filters (rules, events, labels, hours, cooldown),
      chat tool over the alert history.
- [ ] More alert channels: Telegram, WhatsApp, SMS, on-site siren / relay.
- [x] Periodic reports: hourly / daily / weekly per camera set and interests, saved and sent.
      Next: set them up from the UI; "what did last night's report say" in the chat.
- [ ] Detectors trained per customer (fire and smoke, customer-specific objects).
- [x] Body poses: posture (with bending), hands up, facing, head down, still / walking /
      running, carried bags, clothing colours (`people` tool), person-down and hands-up alerts. Next: staged-fall test at a pilot site; a smaller COCO-only pose model.
- [ ] Chat in the browser: `/chat` endpoint + page showing the evidence thumbnails.
- [ ] Evals: a fixed set of questions on recorded footage to compare local models.
- [x] Better search vectors: CLIP B/32 LAION-2B, vectors tagged with their model (2026-10-09).
- [x] `look`: a local vision model reads the evidence images for the chat (2026-10-09).
- [x] Number plates: read per vehicle, `plates` tool, `GET /plates`, watchlist alerts
      (2026-10-09).
- [ ] **Urgent: train our own YOLOX plate finder** on Open Images' "Vehicle registration plate"
      boxes (CC BY 4.0) and set `plates.detector.type: yolox`. The current finder's weights were
      trained with a fork of the GPL-3.0 YOLOv9 repo (MIT code): trials only until then. With
      our own finder the plate pipeline is fully permissive.
- [x] Sound events from camera microphones (2026-10-09). Next: check on real camera audio.
- [x] Rules without a model: camera tampering, abandoned bags, crowding (2026-10-09).
- [ ] Captions per scene (small VLM) stored next to the vectors, for questions CLIP handles badly
      (counting, "what was he doing").
- [ ] Multi-camera timeline: follow one visit across cameras.
