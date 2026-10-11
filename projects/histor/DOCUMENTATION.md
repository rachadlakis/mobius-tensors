<img src="../../Images/brand-kit/logo/logo-horizontal-light.png" alt="Mobius Tensors" width="200" align="right" />

# Histor — Documentation

*Histor is a Mobius Tensors product.*

*Greek ἵστωρ: "witness, one who knows by having seen". The root of "history".*

*October 2026. Product, technical, installation and model-research documentation of the Histor
code in this folder. The short version is the [README](README.md). Every claim was checked against
the code; keep it that way when editing: if a feature is on the roadmap, say so. Figures marked
"real" were made by running the real models on the sample video `data/vtest.avi`.*

**Contents**

- **Part 1, Product:** [1 In one line](#1-in-one-line) · [2 Short description](#2-short-description) ·
  [3 The problem](#3-the-problem) · [4 What Histor does](#4-what-histor-does) ·
  [5 Kinds of answer](#5-kinds-of-answer-and-it-says-which) · [6 Where it fits](#6-where-it-fits) ·
  [7 Private by design](#7-private-by-design) · [8 Alerts](#8-alerts-when-it-matters) ·
  [9 Where it is today](#9-where-it-is-today) · [10 Limits](#10-limits-worth-saying-out-loud) ·
  [11 How to try it](#11-how-to-try-it)
- **Part 2, How it works:** [12 Architecture](#12-architecture) · [13 Code layout](#13-code-layout) ·
  [14 Models and licences](#14-models-and-licences) · [15 Detection and tracking](#15-detection-and-tracking) ·
  [16 Areas, lines and other rules](#16-areas-lines-and-other-rules) ·
  [17 Alerts and channels](#17-alerts-and-channels) · [18 Alert checking](#18-alert-checking-a-second-look-before-sending) ·
  [19 Search: embeddings](#19-search-what-gets-embedded-and-how-a-question-finds-it) ·
  [20 Main stream](#20-main-stream-recording-and-hi-res-crops) · [21 Face search](#21-face-search-opt-in) ·
  [22 Body poses](#22-body-poses-what-people-were-doing) · [23 Number plates](#23-number-plates-opt-in) ·
  [24 Sound events](#24-sound-events-opt-in) · [25 Chat](#25-chat) · [26 Reports](#26-reports) ·
  [27 Storage and API](#27-storage-retention-and-api) · [28 Test UIs](#28-test-uis)
- **Part 3, Installing a site:** [29 What you are building](#29-what-you-are-building) ·
  [30 Bill of materials](#30-bill-of-materials) · [31 Plan before you drill](#31-plan-before-you-drill) ·
  [32 Physical installation](#32-physical-installation) · [33 First-time camera setup](#33-first-time-camera-setup) ·
  [34 Server setup](#34-server-setup) · [35 Install the software](#35-install-the-software) ·
  [36 Run it as a service](#36-run-it-as-a-service) · [37 Access the API safely](#37-access-the-api-safely) ·
  [38 Checklists](#38-checklists) · [39 Troubleshooting](#39-troubleshooting) ·
  [40 Laptop and PoE switch](#40-testing-with-a-laptop-and-a-poe-switch) · [41 Sizing](#41-sizing-rule-of-thumb)
- **Part 4, Model research:** [42 Short answer](#42-short-answer) · [43 What a model must pass](#43-what-a-model-must-pass-to-enter-histor) ·
  [44 Where new models plug in](#44-where-new-models-plug-in) · [45 Fire and smoke](#45-fire-and-smoke) ·
  [46 Weapons](#46-weapons) · [47 Other models](#47-other-models-worth-adding) ·
  [48 Threat detection, October 2026](#48-threat-detection-update-october-2026) ·
  [49 NVIDIA's Public Safety blueprint](#49-compared-with-nvidias-public-safety-blueprint)
- [Roadmap](#roadmap) · [Development log](#development-log) · [Sources](#sources)

**New on 2026-10-10**

- **Alert checking** (section 18): before an alert is sent, the local vision model looks at its
  snapshot and a close-up of the object and answers YES, NO or UNSURE. A NO holds the alert back
  (it stays in the store, marked); a model that is down or slow never stops an alert.
- **One place for the docs:** this file and the README replace `ENGINE.md`, `REPORT.md`, the old
  `docs/histor` documentation and the site installation guide.
- **MCP server** (section 25.3): the chat's tools for Claude Desktop, Claude Code or any MCP
  agent, over stdio (also through SSH from a laptop). Images stay on the site. New figures for detection, search,
  plates, sounds, tampering, alerts and the chat.
- **Research:** threat detection models and NVIDIA's Public Safety blueprint (sections 48–49).

> **To do, urgent: train our own plate finder.** The current finder (open-image-models YOLOv9-t)
> has MIT code, but its weights were trained with a fork of the GPL-3.0 YOLOv9 repo, so it is for
> trials only until a legal check. The way out is already in the code: train our own YOLOX on
> Open Images' "Vehicle registration plate" boxes (CC BY 4.0), then set
> `plates.detector.type: yolox` and its `model` path. Nothing else changes.

---

# Part 1 — Product

## 1. In one line

Histor lets you ask your cameras what happened, in plain language, and answers with the frames
as evidence.

## 2. Short description

*For the website, a brochure or LinkedIn.*

Most camera AI answers one question: "is X happening right now?". Histor answers a different one:
"what happened?". It keeps a searchable memory of what your cameras see (every tracked object,
plus regular snapshots of each scene), so you can ask later: *"Was there a white van at the gate
yesterday evening?"*, *"How many cars came in today?"*. It answers in sentences and lists the
frames it checked. It also notes what people were doing (standing, sitting, bending, lying on
the floor, hands up, walking or running, carrying a bag or a suitcase) and what colours they wore,
so *"Did anyone fall in the warehouse last night?"* gets an exact answer. It alerts you when
something happens in an area you choose, when someone stays down, when a bag is left behind or a
camera is covered, and a local vision model can look at each alert before it is sent, to hold
back false alarms. It runs on your own server, without a GPU, and by default even the language
model that answers runs locally. Your footage stays in the building.

## 3. The problem

Sites record far more video than anyone watches. Most footage is only opened after something has
gone wrong, and then a recorder lets you jump to a time, not to a moment: someone scrubs through
hours of footage. Classic camera AI answers "is X happening right now?" only for the rules set up
in advance; today's question was usually not one of them. And the rules it has raise false alarms
(a shadow, a coat on a chair), until guards stop looking.

## 4. What Histor does

1. **Record.** One recorder per camera detects and tracks every object across the 80 everyday
   object types the detector knows (people, vehicles, animals, bags, furniture…). Each tracked
   object is stored as a searchable description every couple of seconds, and every few seconds
   the whole scene is stored too, so things the detector has no name for can still be found.
   About once a second, each person's body pose (17 keypoints) is turned into a few facts:
   posture, hands raised, facing the camera, head down, still / walking / running, a bag or
   suitcase on them, and the colour of their top and trousers. Where the client has approved it,
   faces are recorded as face prints (section 21) and number plates are read (section 23).
2. **Ask.** A question is split into *what*, *when* and *which camera*. Time ranges can be plain
   words ("yesterday 6pm to 9pm", "this morning", "last 2 hours", "last friday") or exact times.
   Sightings of the same tracked object are merged into one visit.
3. **Answer.** A language model looks things up in the store (up to six rounds of lookups per
   question), answers in sentences, and lists the thumbnails it looked at as evidence.
4. **Alert.** The same recorder fires alerts on areas and lines drawn on each camera (enter,
   leave, stay too long, cross, too many at once), plus *person down*, *hands up*, bags left
   behind, a camera covered or turned, plates on a watchlist and sounds such as a smoke alarm.
   Optionally, a local vision model checks each alert before it is sent (section 18).
5. **Report.** Hourly, daily or weekly: what the chosen cameras saw, focused on what the owner
   asked to watch for (section 26).

## 5. Kinds of answer, and it says which

| Kind | How | Good for |
| --- | --- | --- |
| **Counted (exact)** | Straight from the detector and tracker: what, when, how many. One tracked object = one visit. | "How many cars came in today?", "What happened 6pm to 9pm?", "Anyone with a dog this morning?" |
| **Described (ranked)** | Visual search (CLIP) by appearance, colour, clothing, or things with no detector class. Ranked best matches, not certainties. | "A man in a red jacket near a car?", "A white van at the gate?" |
| **Pose (measured)** | Body keypoints of each person: seconds standing / sitting / crouching / bending / lying, hands above the head, facing the camera, head down, still / walking / running, a bag on them, main colour of top and trousers (11 colour names). Facts per visit, not a ranking (section 22). | "Anyone lying on the floor last night?", "Anyone running in the car park?", "Red top and blue jeans?" |
| **Plate (exact, opt-in)** | The plate text read on each vehicle; look-alike characters (0/O, 8/B…) and one wrong character forgiven (section 23). | "When did AB 123 CD come in?" |
| **Face (scored, on approval only)** | A face print from an uploaded photo compared with the faces recorded on the cameras; each visit *likely* or *possible* (section 21). | "Did the person in this photo come to the office between 1 and 4?" |
| **Looked at (described)** | A local vision model reads the evidence images: counts, actions, text (`look`, section 25). | "How many people were at the counter at 3pm?" |

"Together" ("a person with a dog") means the same camera at the same time (within 15 seconds by
default), not touching in the image.

## 6. Where it fits

| Site | Example question |
| --- | --- |
| Gates and car parks | "How many trucks came in this morning?", "When did AB 123 CD leave?" |
| Warehouses and yards | "Was anyone at the loading bay camera yesterday evening?", "Did anyone fall?" |
| Campuses and public spaces | "When did a bag first appear by the bench?" |
| Construction sites | "Was there a white van at the gate yesterday evening?" |
| Shops and offices | "How long was the queue at the till at noon?", "Was any camera covered?" |

## 7. Private by design

- Runs on the site's own server; cameras stay on the site network (Part 3). Footage never leaves
  the building.
- One memory store per site, on that server. Retention is a setting (30 days by default for
  sightings and events; recordings 7 days).
- The default chat model runs locally (Qwen3 4B through Ollama, Apache-2.0, on CPU). A larger
  local model (Qwen3 30B-A3B) needs about 20 GB of RAM or a GPU.
- A cloud model (Claude) is an opt-in switch. It receives the question and the text search
  results (times, cameras, labels, thumbnail file names), never images or video. The vision model
  behind `look` and alert checking always runs locally.
- **Biometrics off by default.** Face recognition is off unless the client has approved it in
  writing (face prints are biometric data). It can be limited to some cameras, and face prints
  are deleted after 7 days. The chat model never sees an attached photo, only its name.
- Body poses identify people far less than faces, but a way of walking can still be recognised:
  they follow the sightings' retention, can be limited to some cameras, and the raw keypoints can
  be left out (`store_keypoints: false`), keeping only the facts.
- Plates and sounds are off by default: plates are personal data in many countries, and
  listening is regulated (often more than video). Audio is classified in memory and never stored.
- **Commercially clean.** Every model that ships is under a permissive licence (Apache-2.0 or MIT)
  for code, weights and training data. AGPL/GPL and research-only models are excluded
  (section 43). Two exceptions are marked *trials only* until replaced: the plate finder's
  weights and the RTMPose-t pose model.

## 8. Alerts when it matters

Histor also alerts, from the same recorder that builds the memory: alerts for the rules set in
advance, memory for every other question.

- **Areas and lines:** draw zones and tripwires on each camera. Events: an object *enters*,
  *leaves*, *stays longer than* N seconds (a car in a fire lane, someone loitering), or *crosses*
  a line (in, out or both directions; a one-way line is a wrong-way alarm). Each rule is limited
  to chosen object types; a zone without points covers the whole picture ("a dog appears").
- **Crowding and queues:** an area with `max_count` raises `crowd` when that many objects stay
  inside it for `count_seconds` (6 people at the till for 30 s).
- **Abandoned objects:** a backpack, handbag or suitcase that stays still with nobody within
  1.5 person-heights for 60 s raises `abandoned`.
- **Camera tampering:** `tamper` when a camera's picture is covered (*blocked*), *dark*,
  *blurred*, turned (*moved*) or *frozen* for 20 s. No model, about 1 ms every 2 s; on by default.
- **Person down, hands up:** from the body poses, when someone stays lying for 10 s or keeps both
  hands above the head for 3 s. Not yet validated on real or staged falls: validate on site
  before relying on it.
- **Number plates on a watchlist** (opt-in) and **sounds** (opt-in): a smoke alarm, breaking
  glass, a gunshot, an explosion, a scream, a car alarm or a siren.
- **A second look before sending** (opt-in, section 18): a local vision model looks at the
  alert's snapshot and answers YES, NO or UNSURE ("is the person in the red box lying on the
  ground?"). A NO holds the alert back as a likely false alarm; it stays in the store, marked, so
  nothing is lost and a person can still check it. The model can veto, never raise an alert.
- **Delivery:** channels, each with its own filter: which rules, events and objects, at which
  hours ("22:00-06:00"), and a cooldown so nobody gets flooded. Webhook (into the client's own
  systems, or Slack/Teams/n8n/Zapier), email and **Telegram** work today; the snapshot (and on
  Telegram a short video clip) is optional. WhatsApp, SMS and an on-site siren are next.
- **Ask from Telegram:** the chat also runs as a Telegram bot for the chats a site allows
  (section 17.1).
- **Ask about alerts:** every event is stored with its snapshot (and verdict), whatever the
  channels sent; the chat's `alerts` tool answers "did any alert go off last night?".

**Not available yet:** object rules only work on the 80 detector types (plus the pose, plate,
sound and tamper events). Fire and smoke, or any object a customer names that is not one of the
80, need a model trained for it (the code loads custom models; the training is still to do).
Weapons are not offered: no model, and both false alarms and misses are costly (section 46).

## 9. Where it is today

| Working now | Next |
| --- | --- |
| Recording and indexing per camera, all 80 object types plus scene snapshots; CLIP LAION-2B vectors tagged per model | Re-index footage recorded with the old CLIP model |
| Plain-language questions (`ask.py`), no language model needed | A fixed set of test questions on recorded footage, to compare chat models |
| Chat with switchable models, local by default; `look` reads the evidence images | Try `qwen3.5:4b` as the chat model itself (tools + vision in one) |
| Main-stream recording with full-resolution crops | A multi-camera timeline: one visit across cameras |
| Alerts by area, line, crowding, abandoned bags, tampering, to webhook, email and **Telegram (2026-10-10)** with filters | WhatsApp, SMS and on-site siren channels |
| **Telegram bot (2026-10-10)**: ask Histor from a phone; images and clips optional | Try it with a real bot at a pilot site |
| **Alert checking by a local vision model (2026-10-10)**, tested against a stand-in server | Run on a real Ollama, then false alarms per day at a pilot site with and without it |
| Body poses: facts, `people` search, person-down and hands-up alerts | Staged-fall test at a pilot site; a smaller COCO-only pose model |
| Face search by photo (opt-in) | Calibrate the thresholds on a pilot site's footage |
| Number plates, `plates` search, watchlist alerts (trial finder) | **Our own YOLOX plate finder**; read rates on real cameras |
| Sound events from camera microphones | Check on a site's real camera audio |
| Periodic reports, saved and sent | Set them up from the UI; "what did last night's report say" |
| **MCP server: the chat's tools for any MCP client (2026-10-10)**, tested over stdio | Try it from Claude Desktop over SSH; an HTTP transport with login |
| **Dashboard with a token login (2026-10-10)**: live alerts with verdicts, camera status, search | A pilot site's feedback; logins per person; the chat in the dashboard |
| — | Fire and smoke model |

## 10. Limits worth saying out loud

- "Everything" is sampled: each tracked object every 2 s, each scene every 5 s, and objects
  smaller than 40 px tall are not described. All three are tunable per site.
- Visual search results are ranked guesses; the evidence frames are there so a person can check.
- Visit counts depend on the tracker keeping an object in view; a long occlusion can split one
  visit into two (figure 19.3 shows one woman as four visits).
- Small local models follow tool instructions less reliably than large ones.
- Poses need people at least 80 px tall. Colours are lost at night (infrared) and shift under
  coloured lighting. Posture comes from rules: someone half hidden behind a desk is "unknown", and
  sitting and crouching can be confused.
- *Running* is measured across the picture: someone far away walking straight towards the camera
  can read as still. *Carrying* only sees the four bag-like objects the detector knows.
- Face thresholds were set on clear photos, not CCTV: calibrate per pilot site. Faces need to be
  at least 32 px wide, which a sub-stream frame rarely gives.
- Plates and sounds are checked on synthetic input only, not yet on real cameras.
- Alert checking costs seconds per alert on a CPU, so a checked alert arrives that much later;
  small vision models can be wrong both ways. It is a filter for false alarms, not a guarantee.
- Retention purging runs inside the HTTP server (`server.py`): a site needs that service running.
- One shared login token per site (section 27.1); no accounts per person yet. Over a network,
  HTTPS in front of it.

## 11. How to try it

A pilot on the client's own recorded footage: we index a sample, they ask their own questions,
then they decide. For live alerts, a week on their cameras gives the number that matters: false
alarms per camera per day, with and without alert checking. A partner programme (Silver / Gold /
Platinum, 10–30 %) is drafted for introducers and CCTV installers
([docs/Partner Program — Study.md](<docs/Partner%20Program%20—%20Study.md>)).

---

# Part 2 — How it works

## 12. Architecture

![Histor: record, remember, use](docs/images/overview.svg)

```text
 IP cameras (RTSP)
   ├─ sub stream ──► detect (YOLOX-S) ─► track (IoU tracker) ─► rules ─► alerts ─► [check] ─► webhook, email
   │                                         │
   │                                         ├─► CLIP index: objects (every 2 s), scenes + 2x2 tiles (every 5 s)
   │                                         ├─► pose index: 17 keypoints → posture, arms, facing, head, colours;
   │                                         │   tracks → motion, carrying (~1/s)
   │                                         ├─► face index: YuNet + SFace, best 5 faces per visit (opt-in)
   │                                         ├─► plate index: plate finder + reader on vehicles (~1/s, opt-in)
   │                                         └─► abandoned bags, crowding; tamper check on the picture
   ├─ audio ───────► YAMNet: smoke alarm, glass, gunshot, scream, siren → events (opt-in, audio not kept)
   └─ main stream ─► ffmpeg: 1-minute MP4s (copy, no re-encode) + keyframes for full-resolution crops
                                             │
                       one SQLite store per site (events, sightings, faces, poses, plates + media)
                                             │
                                             ├─► ask.py     plain-language search, rules only (no LLM)
                                             ├─► chat.py    LLM + tools (+ a local vision model), answers with evidence
                                             ├─► report.py  hourly / daily / weekly reports
                                             └─► server.py  FastAPI: search, people, plates, events, media, recordings
```

One recorder process per camera; each heavy step (CLIP, pose, faces, plates, alert checking) runs
on its own thread and **skips samples rather than slowing detection**, so a busy scene is sampled
more sparsely instead of lagging behind. Live cameras drop stale frames: a slower model means
fewer frames per camera, not a growing delay.

### 12.1 Two pipelines

![Histor: the perception pipeline (left) and the agent pipeline (right)](docs/images/architecture.svg)

The same picture as NVIDIA's Public Safety blueprint (section 49), drawn for Histor: a perception
pipeline that runs all the time, and an agent pipeline that runs when someone asks.

**Perception pipeline (left).** Each camera gives three streams. The **sub stream** goes to
**detect + track** (YOLOX-S and an IoU tracker), whose tracks feed:

- **Rules**: areas, lines, crowding, loitering, abandoned bags, tampering → events.
- **Indexes**: CLIP vectors of objects and scenes, pose facts, face prints and plate reads (faces
  and plates opt-in), for search and the chat later. The pose and plate indexes also raise events:
  person down, hands up, a watchlist plate.
- **Alert checking**: a local vision model (Qwen3.5 4B by default) looks at each alert's snapshot
  and answers YES / NO / UNSURE before it is sent (section 18).

The **audio** goes to **YAMNet** for sound events (opt-in, audio never kept). The **main stream**
is copied by ffmpeg into 1-minute MP4s plus keyframes, with no re-encoding (section 20). There is
no message bus: `pipeline.py` passes results between threads in one process per camera, and
everything lands in **one SQLite store per site** with its media. Alerts go out to **channels**
(webhook, email, console).

**Agent pipeline (right).** The user asks in the chat (`chat.py` or the test UI) or from any MCP
client (Claude Desktop, Claude Code, a client's own agent). The **chat model** (Qwen3 4B on the
site's CPU by default, any OpenAI-compatible server, or Claude) answers by calling tools:

- **Events**: `objects`, `alerts`: what was tracked and which alerts fired, with exact counts.
- **Search**: `search`, `people`, `plates`, `face`: ranked or exact lookups in the store.
- **Look**: `look`: the local vision model reads evidence images and returns text, so images never
  leave the site.
- **Reports**: `report.py`, on a schedule: exact counts plus a summary that the chat model writes
  with the same tools.

The same tools are served over **MCP** (`mcp_server.py`, section 25.3), except `face`. Recordings
are opened through the API's `video_url` (section 27).

**Local models** do the work: `qwen3:4b` for the chat, `qwen3.5:4b` for `look` and alert checking,
CLIP B/32 to turn a question into a search vector; on the perception side YOLOX-S, ViTPose-B,
YuNet + SFace, the plate models and YAMNet. All of them run on a CPU.

| Component | Purpose |
| --- | --- |
| Perception pipeline | Detect, track and index everything; raise events from rules |
| Alert checking | A local vision model holds back false alarms before they are sent |
| Vision model | Checks alerts; reads evidence images for the chat (`look`) |
| Chat model + tools | Questions in plain language, answered with evidence; reports |
| MCP server | The same tools for any MCP agent |
| Store + recordings | One SQLite file per site with its media; MP4s from the main stream |
| API + dashboard | `server.py`: search, people, plates, events, media, recordings; `dashboard.py`: one web page with live alerts, camera cards and search, behind a token login; Gradio test UI for development |

## 13. Code layout

One Python package, `histor`, with one `uv` environment (Python ≥ 3.11). About 10,000 lines of
Python and ~310 tests (`uv run pytest`; the ones that need a model run once it is downloaded).

```text
configs/
  default.yaml         a site: every COCO class embedded, store in data/store/histor.db
  laptop.yaml          test on a laptop: webcam = camera "laptop", faces on, own store
  engine/              engine-only examples (detection, zones, search)
src/histor/
  record.py            recorder (engine/main.py with this config)
  ask.py               question -> ranked sightings (no LLM)
  chat.py              chat CLI: question -> LLM + tools -> answer + evidence
  tools.py             what the LLM may call: objects, search, alerts, people, plates, face, look
  mcp_server.py        the same tools over MCP (stdio), for Claude Desktop / Code or any agent
  telegram_bot.py      the chat as a Telegram bot, for allowed chats
  llm.py               model backends: OpenAI-compatible (local) and Claude
  vision.py            the local vision model behind `look` and alert checking
  report.py            hourly / daily / weekly reports: exact counts + LLM summary, saved and sent
  ui.py                test UI (Gradio): record, find a person by photo, chat, reports, test bench
  engine/
    detector.py        Detector interface + YOLOX ONNX backend
    onnx_device.py     ONNX Runtime sessions on the GPU if there is one, else CPU (`device:`)
    tracker.py         multi-object tracker (stable IDs across frames)
    zones.py           zone / line rules -> enter, exit, dwell, crowd, cross events
    tamper.py          camera tampering: blocked, dark, blurred, moved, frozen
    abandoned.py       bags and suitcases left still with nobody near them
    alerts.py          event sinks: console, JSON lines, store, webhook, email (filters, cooldown)
    verify.py          alert checking: the vision model's YES / NO / UNSURE before sending
    pipeline.py        detect -> track -> rules -> alerts -> indexes (shared by CLI, UI, server)
    store.py           SQLite store + media files + retention
    indexer.py         builds the search index while recording (objects, scenes, faces)
    clip_onnx.py       CLIP image/text embeddings with ONNX Runtime (no PyTorch)
    faces.py           face detection (YuNet) + face prints (SFace)
    pose.py            body keypoints (ViTPose / RTMPose) + facts
    pose_index.py      poses while recording: sampling, smoothing, rows on change, events
    plates.py          number plates: finder (YOLOv9 / YOLOX) + reader (fast-plate-ocr), matching
    plate_index.py     plates while recording: sampling, voting, rows, watchlist events
    sound.py           sound events from camera audio (YAMNet): groups, rules, ffmpeg reader
    query.py           "red shirt last friday" -> what / when / which camera
    search.py          text, face, people and plate search, results grouped into visits
    server.py          FastAPI: events, media, recordings, /search, /search/face, /people, /plates
    dashboard.py       the dashboard page (dashboard.html), /status, the token login
    status.py          camera status (online, today's counts, tamper) for the dashboard and bot
    telegram.py        Telegram Bot API client (alert channel, bot), video clips from recordings
    stream.py          frame sources (webcam, image, video, URL)
    cameras.py         IP cameras: config, capture with reconnect, discovery
    mainstream.py      main stream via ffmpeg: MP4 recording + keyframes for hi-res crops
    main.py, app.py    engine CLI; Gradio test bench
    download_model.py  fetch YOLOX (and `faces`, `plates`, `sounds`, `pose_rtmpose_t`)
    export_clip.py, export_pose.py   one-off: OpenCLIP / ViTPose-B -> ONNX
docs/                  images/ (the figures here), Histor.pptx, Project Summary (.pdf + .html
                       source: print with Chrome headless, --no-pdf-header-footer), partner programme
notebooks/             models_tour.ipynb: each model, step by step
models/                .onnx weights (git-ignored)
tests/                 tests/engine/ for the engine
```

The engine (`histor.engine`) only depends on the `Detector` interface: a custom-trained YOLOX
needs `model.path` and `model.class_names`; another backend implements
`detect(frame) -> list[Detection]` and registers in `create_detector`.

## 14. Models and licences

| Role | Model | File | Licence | Size | Cost on a laptop CPU |
| --- | --- | --- | --- | --- | --- |
| Objects | **YOLOX-S** (COCO, 80 classes); YOLOX-M for GPU servers, YOLOX-Nano for weak machines | `yolox_s.onnx` · `yolox_m.onnx` · `yolox_nano.onnx` | Apache-2.0 | 36 · 101 · 4 MB | ~76 ms a frame |
| Visual search | CLIP ViT-B/32, OpenCLIP LAION-2B `laion2b_s34b_b79k`, float32 | `clip_image.onnx`, `clip_text.onnx`, `clip_bpe_vocab.txt.gz` | MIT | 350 + 250 MB | ~30 ms per crop |
| Body pose | ViTPose-B "simple", COCO only | `pose_vitpose_b.onnx` | Apache-2.0 | 344 MB | 100–175 ms per person |
| Body pose (trials) | RTMPose-t (training data not all commercial) | `pose_rtmpose_t.onnx` | Apache-2.0 code | 13 MB | 10–17 ms per person |
| Faces (opt-in) | YuNet detector + SFace recogniser (OpenCV Zoo) | `face_detection_yunet_2023mar.onnx`, `face_recognition_sface_2021dec.onnx` | MIT / Apache-2.0 | 0.2 + 37 MB | a few ms per face |
| Plate finder (opt-in, trials) | open-image-models YOLOv9-t 384 | `plate_yolov9_t_384.onnx` | MIT code; weights via a GPL-3.0 fork | 8 MB | ~11 ms per vehicle |
| Plate reader (opt-in) | fast-plate-ocr CCT-S v2 global | `plate_ocr_cct_s_v2.onnx`, `.yaml` | MIT | 5 MB | ~10 ms per plate |
| Sounds (opt-in) | YAMNet (AudioSet, 521 classes), tf2onnx conversion, pinned revision | `sound_yamnet.onnx`, `sound_yamnet_classes.csv` | Apache-2.0 | 16 MB | ~3 ms per 3 s of audio |
| `look`, alert checking | Qwen3.5 4B via Ollama (vision); others selectable | — | Apache-2.0 | — | seconds per image |
| Chat (default) | Qwen3 4B via Ollama; others selectable (section 25) | — | Apache-2.0 | — | seconds per answer |

All vision models run as ONNX graphs with numpy/OpenCV pre- and post-processing: no PyTorch at
run time and a straightforward port to C++. PyTorch is only used in the one-off export scripts.
CLIP stays in float32: int8 was 4× smaller but reordered search results too much.

**GPU:** every ONNX model runs on the GPU when ONNX Runtime sees one, else on the CPU (`device:
auto` in `model`, `search.clip` and `pose`; also `cpu`, `cuda`, `cuda:1`, `directml`, `coreml`;
the log says which device each model got). On an NVIDIA machine swap the CPU build for the GPU
one; CUDA and cuDNN come as wheels, so only the driver is needed (CUDA 13: driver 580+):
`uv sync --no-group cpu --group gpu`, then keep those flags on `uv run`, or set `UV_NO_SYNC=1`.

## 15. Detection and tracking

![Detection and tracking pipeline](docs/images/detection-pipeline.svg)

Each frame of the camera's sub stream is letterboxed to the model's input (640 × 640 for
YOLOX-S), run through YOLOX, and its 8,400 candidate boxes are filtered by score and merged by
non-maximum suppression (`class_agnostic_nms: true` on sites, so a van seen as both *car* and
*truck* stays one box). The tracker then gives each object a stable `#id`:

- **Match:** each detection goes to the track it overlaps most (IoU ≥ 0.3). Detections at or above
  `track_threshold` (0.45) can start a track; weaker ones only extend existing tracks.
- **Confirm:** a new track needs `min_hits` (3) matches before it gets an `#id`, so one-frame
  flickers never become objects.
- **Coast:** a track survives `max_missed` (30) frames without a detection, so someone walking
  behind a pillar keeps their `#id`.
- **Label vote:** a track's label is its most confident label so far (no car/truck flicker).
- An object counts as inside a zone when its box's bottom-centre (feet, wheels) is inside.

![Real: YOLOX-S on one frame, and the tracks over the last 4 seconds](docs/images/detection-tracking.jpg)

*Real: frame 420 of the sample street video (768 × 576), YOLOX-S on CPU. Left, the raw detections
with their confidence; right, the tracker's view of the same moment, with each object's path over
the last 40 frames.*

### 15.1 Choosing a model

| Model | Input | COCO AP (val) | CPU here: ms / frame (FPS) | Use when |
| --- | --- | --- | --- | --- |
| `yolox_nano` | 416 | 25.8 | 9.5 (105) | Weak machines, laptop demos (`configs/laptop.yaml`) |
| `yolox_tiny` | 416 | 32.8 | not measured | A weak CPU that needs more than nano |
| **`yolox_s`** | 640 | 40.5 | 75.6 (13.2) | **Default for CPU sites**: much better on small and far objects |
| `yolox_m` | 640 | 46.9 | 199 (5.0) | GPU servers, or sites with few cameras |
| `yolox_l` / `x` | 640 | 49.7 / 51.1 | not measured | GPU, accuracy first |

Measured 2026-10-09 on a laptop (Intel Core 13th gen, 20 threads, CPU only, ONNX Runtime), 300
frames of `data/vtest.avi`, detection only, after a warm-up. AP from the YOLOX README. The CPU
is shared: four cameras on that laptop with `yolox_s` get about 3 detections a second each. The
tracker copes well at 3–5 a second for people, less well for fast cars. Benchmark the site's own
server before signing off. Switch with `model.path`; keep only some classes with `model.classes`.

```bash
uv run -m histor.engine.main --source 0 --show                     # webcam
uv run -m histor.engine.main --source data/img.jpg --save data/out/img.jpg
uv run -m histor.engine.main --source rtsp://camera/stream
uv run -m histor.engine.main --source gate --show                  # IP camera named in the config
```

## 16. Areas, lines and other rules

Draw **zones** (polygons) and **lines** (tripwires) on a camera's view and get events when tracked
objects interact with them:

| Rule | Events | Typical use |
| --- | --- | --- |
| zone | `enter`, `exit`, `dwell` (inside longer than `dwell_seconds`) | restricted area, loitering, a car parked > 60 s |
| zone with `max_count` | `crowd`: at least that many of its classes inside for `count_seconds`; again once the count has dropped | queues, crowding |
| line | `cross` with direction `in` / `out`; `direction: in` alone is a wrong-way alarm | counting people / cars through a gate |

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
```

Points are 0..1 of the frame (any resolution) or pixels. Draw them in the test bench's **Zones &
lines** tab and paste the YAML.

![Real: a zone and a line on the street video, and the events they produced](docs/images/zones-lines.jpg)

*Real: `configs/engine/zones_example.yaml` on the street video (YOLOX-S). The zone counts who is
inside and for how long; the line counts crossings each way. Right: the events as they are
stored, one per person and change.*

### 16.1 Abandoned objects

A backpack, handbag or suitcase whose centre stays within a quarter of its size, with no person's
centre within `owner_distance` (1.5) person-heights, for `seconds` (60), raises `abandoned` (rule
`abandoned_object`), once per object. A bag carried by someone is never still; a passer-by resets
the clock. Optional `zones`: only inside these areas. Off by default (`abandoned:`).

### 16.2 Camera tampering

Every 2 s, on a 320 px grey copy of the picture, with no model (about 1 ms):

| Check | Fires when |
| --- | --- |
| *blocked* | almost no contrast (grey level spread < 6): covered, sprayed, facing a wall |
| *dark* | very dark (mean < 12) with some contrast |
| *blurred* | sharpness (variance of the Laplacian) below a quarter of this camera's own usual, learned over its first minute |
| *moved* | the structure no longer matches the reference view: correlation of 80 × 45 edge-strength maps below 0.5 |
| *frozen* | not a single grey level changed |

Each must last `hold_seconds` (20) and fires once until it clears. On by default (`tamper:`).

![Real: the tamper checks on views of the street video](docs/images/tamper-checks.jpg)

*Real: `tamper.measure()` and `similarity()` on frames of the street video. People walking
through and 40 % less light keep the similarity at 0.78; a 40 px turn drops it to 0.45, a lens out
of focus drops sharpness to 2 % of usual, a covered lens leaves no contrast. A first version that
compared edge pixels failed on busy scenes (a turned camera still matched 60–70 %), so it was
replaced by this correlation.*

## 17. Alerts and channels

Every event (from areas and lines, crowding, abandoned bags, tampering, poses, plates and sounds)
goes through the same code (`alerts.py`): saved to the store with a JPEG snapshot of the frame
(the object boxed), then sent to each **channel** whose filter accepts it.

```yaml
alerts:
  events: null                        # event types that become alerts (null = all)
  cooldown_seconds: 0                 # global: also skips storing; prefer per-channel cooldowns
  snapshots: true                     # JPEG with the object boxed, saved + attached to emails
  channels:
    - type: webhook
      url: ${ALERT_WEBHOOK_URL}
      headers: {Authorization: "Bearer ${WEBHOOK_TOKEN}"}
      cooldown_seconds: 30            # at most one message per rule + event type per 30 s
    - type: email
      host: smtp.example.com
      port: 587
      username: ${SMTP_USER}
      password: ${SMTP_PASSWORD}      # secrets go in .env, never in the YAML
      sender: alerts@example.com
      to: [security@client.com]
      labels: [person]                # this channel: only people...
      hours: "22:00-06:00"            # ...and only at night
  verify: {enabled: false}            # alert checking, section 18
```

- Each channel filters by `rules` (zone/line names), `events`, `labels`, `hours` (wraps past
  midnight) and `cooldown_seconds`. The store still gets every event, so the chat sees everything
  even when a channel stays quiet.
- The webhook receives JSON: `id, camera, time, iso_time, type, rule, label, track_id`, plus
  `direction` / `dwell_seconds` / `count` / `plate` / `score` where they apply, `snapshot` (path
  served by the API at `/media/<path>`) and, with alert checking, `verified` and `verification`.
- Sending happens on a background thread with 2 retries; a slow mail server never stalls
  detection, and failures are logged, not fatal.

| Channel | Status | How | Needs |
| --- | --- | --- | --- |
| Webhook | works | JSON POST to any URL; also the quick route to Slack, Teams, n8n, Zapier, Make | a URL |
| Email | works | SMTP, snapshot attached | an SMTP account |
| Telegram | works | A Telegram bot posts the alert text; the snapshot and a video clip are optional (section 17.1) | a bot token (@BotFather) and the chat id |
| WhatsApp | planned | WhatsApp Business Cloud API; messages the business starts use a pre-approved template | a Meta business account, number id, token, template |
| SMS | planned | an SMS provider's HTTP API (e.g. Twilio) | an account and a sender number |
| Siren / relay | planned | HTTP call to an IP relay or a camera's alarm output | a device with an HTTP API |

Adding a channel is one class in `src/histor/engine/alerts.py` (`TelegramSink` is the latest
example); filters, cooldowns, retries and the background thread come for free:

```python
class SmsSink(Sink):
    def __init__(self, account: str, token: str, to: list[str]):
        ...

    def send(self, payload, jpeg):
        title, text = alert_message(payload)   # same wording as email and Telegram
        ...                                    # one HTTPS request to the provider

CHANNELS["sms"] = SmsSink                  # config: channels: [{type: sms, ...}]
```

### 17.1 Telegram: alerts and questions from a phone

One Telegram bot does two things, each optional:

| | What | Runs in | Config |
| --- | --- | --- | --- |
| **Alerts** | Each alert to a person, a group or a channel, with the check's verdict | the recorders (an alert channel) | `alerts.channels: [{type: telegram, ...}]` |
| **Questions** | Ask the chat from Telegram, get the answer (and the evidence) back | its own process, `histor.telegram_bot` | `telegram:` |

**What leaves the site.** Telegram is a cloud service: what the bot sends is stored on its
servers. By default only **text** goes out (the alert, the question, the answer). Images and video
are switches, to turn on only where the client agreed in writing:

| Switch | Sends | Default |
| --- | --- | --- |
| `alerts.channels[].images` | the alert's snapshot, with the text as caption | off |
| `alerts.channels[].video` | then a clip from the main-stream recording, from 3 s before the alert, `video_seconds` long (10), once it is recorded; needs `recording:` and ffmpeg | off |
| `telegram.images` | the evidence thumbnails (up to 4) after each answer; for face search, the camera's best face of each visit | off |
| `telegram.face_search` | lets a photo sent to the bot search the recorded faces (below); needs `faces.enabled` | off |

**Create the bot (5 minutes, once per site):**

1. In Telegram, open a chat with **@BotFather** (the official bot, with the blue tick) and send
   `/newbot`. Give it a name ("Histor Warehouse") and a username ending in `bot`
   (`histor_warehouse_bot`).
2. BotFather replies with a **token** like `123456789:AAH…`. It controls the bot: treat it like a
   password. On the Histor server, put it in `projects/histor/.env` as
   `TELEGRAM_BOT_TOKEN=123456789:AAH…` (`chmod 600 .env`). If it ever leaks, `/revoke` in
   BotFather issues a new one.
3. Optional, in BotFather: `/setdescription`, `/setuserpic` (the Histor icon), and
   `/setjoingroups` → *Disable* once the bot is in the groups it needs, so nobody else can add it.
4. Set `telegram.token: ${TELEGRAM_BOT_TOKEN}` in the site config and start the bot:
   `uv run -m histor.telegram_bot --config configs/site.yaml`.
5. **Get the chat ids:** each person opens the bot and sends `/id` (for a group: add the bot to the
   group, then send `/id` there; group ids are negative). The bot answers with the id and nothing
   else, because the chat is not allowed yet.
6. Put the ids in the config and restart the bot:

```yaml
telegram:
  token: ${TELEGRAM_BOT_TOKEN}
  allowed_chats: [123456789, -1001234567890]   # these chats may ask; nobody else
  images: false                # true: evidence thumbnails too (they leave the site)

alerts:
  channels:
    - type: telegram
      token: ${TELEGRAM_BOT_TOKEN}             # the same bot is fine
      chat_id: -1001234567890                  # a person, group or channel; a list for several
      images: false
      video: false
      events: [down, abandoned, tamper]        # the usual filters: rules, events, labels, hours, cooldown
```

7. Restart the recorders (`systemctl restart 'histor-camera@*'`) so the alert channel loads, and do
   a walk test: the alert arrives in Telegram. Run the bot as a service (section 36).

**How it connects.** The server only makes **outgoing** HTTPS calls to `api.telegram.org`: the bot
asks Telegram for new messages (long polling) and sends its answers. No open port, no public
address, no change to the site's router. If the site's firewall only allows listed hosts, allow
`api.telegram.org` on port 443. Only one bot process per token may poll.

**In the chat:** `/help` (examples), `/status` (cameras online, today's counts and alerts, any
tamper), `/new` (a new conversation; one also starts after 30 minutes of quiet), `/id`. Answers
come from the same model and tools as the chat (section 25), one question at a time. If the model
server is down, the bot says so.

**Face search from Telegram** (`telegram.face_search: true`, opt-in): send the bot a photo of a
person, with the question as its caption ("did he come to the office between 1 and 4?").

1. The bot downloads the photo **into memory** (it is never written to disk on the server) and
   makes a face print exactly as the chat's `/photo` does (section 21.4): one usable face, sharp
   and facing the camera, or it says why not.
2. It searches **directly**, like `POST /search/face`: the caption's time and camera words set the
   range by rules, so **no language model is needed**. The answer lists *likely* / *possible*
   visits with their times and match scores, or says how many faces were compared ("no match is
   not proof"). With `telegram.images`, the camera's best face of each visit follows.
3. The photo stays attached as `photo1`, so a follow-up ("and yesterday?") goes to the chat model
   with it.

The photo itself stays in the Telegram chat, on Telegram's servers, and a face print is biometric
data: turn this on only where the client agreed in writing (section 21), and only for the chats
that need it. The laptop demo config has it on to try with your own face.

**Checked:** `tests/test_telegram.py` (18 tests) against a stand-in Bot API server: text only by
default, the snapshot and clip only when switched on, every chat of a list, long texts split, the
token never in an error, reports, the bot answering only allowed chats and giving strangers only
their id, conversations and `/new`, evidence images, `/status`, a model error, and face search
(off by default, the largest photo size, a match without the chat model, no match, unusable
photos, the follow-up). Tried with a real bot from a phone (2026-10-10): `/id`, the allowed chat,
`/status`, `/help`, and a photo before face search existed.

## 18. Alert checking: a second look before sending

Rules fire on boxes and keypoints. A shadow, a reflection, a poster, a coat on a chair or a person
half hidden behind a desk can trip them, and every false alarm teaches the guard to ignore the
next one. **Alert checking** (`src/histor/engine/verify.py`, opt-in) puts a local vision model
between the rule and the channels: the same model the chat's `look` tool uses (Qwen3.5 4B by
default), on the site's server.

![Alert checking: stored at once, checked, then sent or held back](docs/images/alerts-flow.svg)

1. The event is **stored at once** with its snapshot, as before.
2. If its type is one to check, the model gets **two images** (the snapshot with the object in a
   red box, and a close-up of the box with some margin, enlarged to at least 96 px) and **one
   question** for that event type.
3. It answers **YES, NO or UNSURE** on the first line and one reason on the second. Anything else
   counts as UNSURE.
4. The verdict is added to the event's row (`verified`, `verification`) and to the message.
   - mode `veto` (default): a **NO is held back** from the channels. It stays in the store with
     its snapshot, so the chat ("any alerts last night?") and a person can still see it.
   - mode `tag`: everything is sent, with the verdict, for a pilot that measures the model first.
5. **Never a silent loss:** model server down, an answer later than `timeout` (60 s), or a full
   queue (20 waiting) sends the alert *unverified* (`error` / `skipped`). The model can veto an
   alert; it never raises one.

| Event | Question (default) |
| --- | --- |
| `enter`, `exit`, `dwell`, `cross` | Is there a real {label} in the red box, not a shadow, reflection, poster or other object? |
| `crowd` | Are there at least {count} {label}s in the picture? |
| `down` | Is the person in the red box lying or fallen on the ground? |
| `hands_up` | Does the person in the red box have both hands raised above their head? |
| `abandoned` | Is the object in the red box a bag or suitcase left on its own, with nobody holding or standing next to it? |
| `tamper`, `sound`, `plate` | Not checked: the picture alone can't confirm a sound; a blocked or blurred view is what the check measures; the plate reader already voted. A question can be added in the config. |

```yaml
alerts:
  verify:
    enabled: true
    events: [down, hands_up, abandoned]   # check these; empty = every type with a question
    rules: []                  # only these areas / lines; empty = all
    mode: veto                 # veto: a NO is not sent | tag: everything sent, with the verdict
    timeout: 60                # seconds; later answers count as errors (alert sent unverified)
    questions:                 # override or add, per event type
      enter: "Is there a real person (not a shadow or reflection) in the red box?"
    # model, base_url, max_side: default to chat.vision
```

**Cost and choice of events.** On a CPU each check takes seconds, so a checked alert arrives that
much later; checks run one at a time on their own thread, so detection never waits. Check the rare
alerts that cost the most when false (person down, abandoned bag, an area at night), not every
line crossing at a busy gate.

**Status (2026-10-10).** Tested end to end (`tests/engine/test_verify.py`, 17 tests): verdicts,
veto and tag modes, errors and a full queue still sending, the questions, and the real HTTP
request to a stand-in OpenAI-compatible server (two JPEGs and the question, as Ollama gets them).
Not yet run against a real vision model: no Ollama models on the build machine. Before promising
it: run it in `tag` mode for a week at a pilot site and count, per event type, the false alarms
it would have held back and the real alerts it would have wrongly held back. NVIDIA's Public Safety
blueprint uses the same pattern with an 8B model on a data-centre GPU (section 49).

## 19. Search: what gets embedded, and how a question finds it

![Search: embedding while recording, and answering a question](docs/images/search-pipeline.svg)

**While recording** (`indexer.py`, a background thread that never slows detection on live
cameras):

- each tracked object is embedded with CLIP when it appears, then at most every
  `object_interval_seconds` (2), if it is at least `min_box_height` (40 px) tall;
- every `scene_interval_seconds` (5) the whole frame and a 2 × 2 grid of tiles are embedded too,
  so things the detector has no class for (a box, an open gate, smoke) can still be found;
- with the main stream on, crops come from the full-resolution keyframe (section 20).

Each image becomes one row in `sightings`: 512 numbers (2 KB), camera, time, track, box, a small
thumbnail, and the name of the CLIP model that made the vector.

![Real: what one frame gives the index](docs/images/search-embedded.jpg)

*Real: frame 300 of the street video. The scene and its four tiles are embedded every 5 s; each
tracked object when it appears and then every 2 s.*

**At question time** (`query.py`, `search.py`, no LLM): the sentence is split into *what / when /
which camera* by rules (today, yesterday, last friday, 2026-09-25, 6pm to 9pm, after 10pm, this
morning, last 2 hours, camera names and aliases). *What* is embedded with CLIP's text model (the
words plus two phrasings, "a photo of…" and "a cctv image of…", averaged) and compared with every
stored vector of the same model in the time range (cosine similarity). Matches of the same track
are merged into one **visit**, ranked by its best frame.

![Real: the top visits for four questions on the street video](docs/images/search-results.jpg)

*Real: 253 object rows (51 people) and 80 scene rows from the 80 s street video, CLIP B/32
LAION-2B. "A person in a red top" finds the right woman four times: the tracker split her walk
into four visits (#41, #32, #34, #44), which is the "long occlusion" limit of section 10. "A
person carrying a bag" is weaker: CLIP knows the scene, not the small bag. "A white van" ranks
the tile that holds the parked van first. Scores are relative (here 0.21–0.32): ranked guesses,
never yes/no.*

- `search.classes` picks what gets embedded as objects: every class on sites
  (`configs/default.yaml`); people, vehicles and bags in `configs/engine/search_example.yaml`.
- Tiny details (earbuds, rings) are not searchable at CCTV resolution; faces need roughly 32+ px.
- **Cost of "everything":** a CLIP crop takes ~30 ms on a laptop CPU; each row is 2 KB plus a
  thumbnail. On live cameras the indexer drops jobs instead of slowing detection, so a busy scene
  is sampled more sparsely. Tune `object_interval_seconds`, `scene_interval_seconds` and
  `min_box_height` per site.
- **Vectors of two models can't be compared.** Each row is tagged with its model; search only
  compares a query with vectors of the same model, so after a switch the old rows are skipped
  (and logged) until they expire.
- Search is brute force with numpy: fine for a pilot (millions of rows a second); sqlite-vec or
  pgvector for large sites.

## 20. Main stream: recording and hi-res crops

Detection runs on the cheap sub stream, but a 640 × 360 crop of a person 15 m away is a blur to
CLIP. With a `recording:` section, each live camera process also opens the camera's **main stream**
with one ffmpeg process (`mainstream.py`):

- **Recording:** packets are copied to one MP4 per minute (`-c copy`: no decode, no re-encode,
  almost no CPU) under `recording.dir/<camera>/`. Fragmented MP4, so the file being written plays
  and survives a power cut. Old files go after `retention_days` (7), or sooner when the disk has
  less than `min_free_gb` free.
- **Hi-res crops:** only keyframes are decoded (`-skip_frame nokey`); set the camera's main-stream
  I-frame interval to 1 s (= its FPS). Each keyframe is run through the detector once and its boxes
  are paired with the live tracks (IoU), which gives exact full-resolution boxes whatever the delay
  between the two streams. Crops for CLIP, faces, poses and plates then come from the keyframe.
- **Nothing is lost between keyframes:** a new track is embedded right away from the sub stream;
  the first keyframe that catches it adds a hi-res crop. On a real-time replay of `vtest.avi`, 28
  of 38 tracked people got hi-res crops; the other 10 kept their sub-stream crop.
- Events and search results carry a `video_url` (`...mp4#t=37`) that opens the moment, 3 s early.

Cost: one extra detection per second per camera, plus disk: 1080p at 4 Mbit/s is about 43 GB
per camera per day. Needs ffmpeg ≥ 5.1; without it, recording and hi-res crops are switched off
with an error in the log, and detection carries on. Vendor cameras get their main-stream URL from
the preset; a camera given by `url:` needs `main_url:`.

## 21. Face search (opt-in)

Face search answers questions like *"Did the person in this photo come to the office between 1
and 4?"*. It is the only **biometric** feature in Histor: a face print is a measurement of the body
that can identify a person, so data protection laws such as the GDPR treat it as sensitive data.

- Face recording is **off** (`faces.enabled: false`) until the client approves it in writing; check
  local law and put up camera notices first.
- It can be limited to some cameras (`faces.cameras`), ideally those whose `about.faces` is `good`.
- Face prints are deleted after 7 days (`face_retention_days`).
- The chat model never sees an attached photo: its face print stays in Histor and the model refers to
  it by name (`photo1`), also with a cloud model.

Searches by appearance ("a man in a red jacket") are not biometric: they describe, they don't
identify. Code: [faces.py](src/histor/engine/faces.py), [indexer.py](src/histor/engine/indexer.py),
[search.py](src/histor/engine/search.py), [tools.py](src/histor/tools.py).

![Face search: recording and search by photo](docs/images/faces-pipeline.svg)

### 21.1 Two small models

| Step | Model | Licence | What it gives |
| --- | --- | --- | --- |
| Find the face | **YuNet** (`face_detection_yunet_2023mar.onnx`) | MIT | A box around each face and 5 landmarks: both eyes, the nose tip, both mouth corners |
| Describe the face | **SFace** (`face_recognition_sface_2021dec.onnx`) | Apache-2.0 | A *face print*: 128 numbers. 99.4 % on the LFW verification benchmark |

Download both with `uv run -m histor.engine.download_model faces`.

### 21.2 Recording

1. About once a second (`faces.interval_seconds`), Histor takes each tracked person's box, from the
   main-stream keyframe when there is one.
2. It crops the **top 60 %** of the person and runs **YuNet** on a copy resized to 320–640 px on its
   longer side (YuNet misses faces much bigger than ~300 px, and small crops need enlarging).
3. The 5 landmarks **align** the face: upright, eyes at fixed positions, 112 × 112 px.
4. **SFace** turns it into a **face print** of 128 numbers, scaled to length 1.
5. A **quality check** (21.3) decides whether the face is good enough to keep.
6. Each visit keeps only its **5 best faces** (`max_per_track`), so an hour at a desk is 5 rows, not
   3,600. Each kept face has a small thumbnail as evidence. Landmarks are not stored.

![Detect, align, face print](docs/images/faces-print.svg)

### 21.3 Quality check

Blurred, tiny or side-on faces "match everyone a little", which is worse than no face at all:

| Check | Measure | Default |
| --- | --- | --- |
| Size | Shorter side of the face box | ≥ 32 px on cameras (`min_face_size`), ≥ 40 px in an uploaded photo |
| Sharpness | Variance of the Laplacian of the aligned face (sharp ≈ 30–300, motion blur < 10) | ≥ 15 (`min_sharpness`) |
| Facing | *Turn*: nose-tip offset from the middle of the eyes, divided by the eye distance (≈ 0–0.3 facing, ≈ 0.5 at 60°) | ≤ 0.6 (`max_turn`) |

To rank a visit's faces, the measures are combined into one quality value (0–1): detector
confidence × size × frontal × sharp.

![Facing the camera, and match levels](docs/images/faces-quality-match.svg)

### 21.4 Searching with a photo

1. **Upload** one or more photos of the same person (test UI, chat `/photo`, or
   `POST /search/face`). The photo goes through the same steps; if no face is found at 640 px,
   detection is retried at full size, then at 320 px.
2. **Which face?** Faces are numbered left to right. A photo with several faces is used only when
   one is clearly the subject (at least twice as big as any other); otherwise Histor asks
   (`--face 2`). An unusable face is refused with the reason ("too small", "blurred", "turned away").
3. **Compare** with every stored face print in the time range and cameras, by cosine similarity
   (one numpy multiplication: 100k prints is ~50 MB and milliseconds).
4. **Group into visits** (camera + track); a visit's score is the **mean of its best 3 matches**.
   The visit covers the whole track: someone seen from behind at 13:40 who shows their face at
   13:44 arrived at 13:40 (`face_seen` says 13:44).
5. **Answer:** ≥ 0.45 is **likely** the same person; 0.363–0.45 only **possible**; below, no
   match. The answer says how many faces each camera recorded in the range: no match is not proof
   the person was not there.

```text
you> /photo visitor.jpg
Attached as photo1
you> did this person come to the office between 1 and 4?
  -> face {'photo': 'photo1', 'since': '2026-10-07 13:00', 'until': '2026-10-07 16:00', 'camera': 'office'}
Likely yes: on the office camera from 13:40 to 13:45 (face clearly seen at 13:44, match 0.52).
```

### 21.5 Accuracy on CCTV

SFace's 99.4 % and its 0.363 threshold were measured on clear, mostly frontal photos. CCTV faces
are smaller, blurred, seen from above and grey under infrared, so accuracy on site will be lower.
**Calibrate both levels on a pilot site's own footage**, and treat every match as a lead for a
person to check. Camera placement matters more than the model:

![Sub-stream frame: faces found but too small](docs/images/faces-substream.jpg)

*Real: a sub-stream frame (768 × 576) of the street video. YuNet finds three faces, but they are
8–10 px wide, far below the 32 px minimum, so none is stored. Faces need the main stream and a
camera at a door or gate, where people walk towards it at head height.*

```yaml
faces:
  enabled: false               # needs the client's written approval
  cameras: []                  # empty = every camera; better: only those with `about.faces: good`
  min_face_size: 32
  interval_seconds: 1
  max_per_track: 5
  min_sharpness: 15
  max_turn: 0.6
```

## 22. Body poses: what people were doing

Body poses answer *"Did anyone fall in the warehouse last night?"*, *"Anyone with their hands up
near the till?"*, *"Was anyone running in the car park?"*, *"Who came in with a suitcase?"* or
*"A man in a red top and blue jeans this morning?"*, and raise the *person down* and *hands up*
alerts. A pose model finds 17 points on each person's body; simple rules turn them into facts that
are stored and searched exactly. Code: [pose.py](src/histor/engine/pose.py),
[pose_index.py](src/histor/engine/pose_index.py), [tools.py](src/histor/tools.py) (`people`).

Keypoints say much less about who someone is than a face print, but a way of walking can still be
recognised: poses follow the search index's retention, can be limited to some cameras
(`pose.cameras`), and the raw keypoints can be left out (`store_keypoints: false`).

![Body poses: from a tracked person to facts, rows and events](docs/images/pose-pipeline.svg)

### 22.1 The 17 keypoints

The COCO layout: five on the head (nose, eyes, ears) and six pairs on the body (shoulders, elbows,
wrists, hips, knees, ankles). "Left" is the person's own left, so on someone facing the camera the
left keypoints are on the right of the picture: that is what makes *facing* measurable.

![The 17 COCO keypoints, and the areas used for clothing colours](docs/images/pose-keypoints.svg)

| Model | Licence | Training data | Use |
| --- | --- | --- | --- |
| **ViTPose-B "simple"** (`pose_vitpose_b.onnx`, 344 MB, `export_pose.py`) | Apache-2.0 | COCO keypoints only | **Default, shippable** |
| RTMPose-t (`pose_rtmpose_t.onnx`, 13 MB, `download_model pose_rtmpose_t`) | Apache-2.0 | 7 datasets, some research-only | Trials only |

Both are *top-down*: they take the person boxes the tracker already has, so the pose step costs one
model run per person, not per frame. Both formats load through the same code (heatmaps or SimCC,
told apart by the ONNX outputs).

### 22.2 How a keypoint is found

1. **Crop.** The person's box is enlarged by 25 % and warped to the model's 192 × 256 px input,
   from the main-stream keyframe when there is one.
2. **Heatmaps.** ViTPose returns one 48 × 64 map per keypoint: how likely that body part is at each
   spot. The peak is the keypoint, its height the **confidence** (0–1); under 0.3 counts as not
   visible.
3. **Back to the picture** with the inverse of the crop's warp.

![A person crop, its 17 keypoints, and four of the heatmaps behind them](docs/images/pose-heatmaps.png)

*Real: a crop from the street video through ViTPose-B. Left: the 17 keypoints (blue = the person's
left side, orange = right). The others: one keypoint's heatmap each. A sharp, single peak means
high confidence; a wrist against a dark coat is fainter.*

### 22.3 From keypoints to facts

No second model: the facts are rules on the keypoints, measured against the person's own **torso
length** (shoulders to hips), so they work at any distance from the camera.

| Fact | Values | Rule |
| --- | --- | --- |
| `posture` | standing, sitting, crouching, bending, lying, unknown | Torso tilted more than 55° from upright: **bending** if the legs go straight down from the hips (feet more than 0.8 torso below them), otherwise **lying**. Less tilted: **crouching** if the feet are less than 0.6 torso below the hips, **sitting** if the knees are less than 0.45 torso below the hips, **bending** if the torso leans more than 35°, else **standing**. **Unknown** when shoulders or hips are not visible |
| `arms` | down, one_up, both_up, unknown | A wrist counts as up when it is clearly (0.1 torso) above the top of the head. Not measured on someone lying |
| `facing` | camera, side, away, unknown | Shoulder order first (the person's left shoulder on the image's left means **away**); then the face: nose and both eyes in order means **camera**, part of the face means **side** |
| `head` | up, down, unknown | **Down** when the nose is more than 0.15 torso below the ears: looking at a phone, a desk, the floor |
| `upper_colour`, `lower_colour` | black, white, grey, red, orange, yellow, green, blue, purple, pink, brown | The most common colour name in the torso area (shrunk to 70 %) and along the legs; none if no colour covers 35 % |
| `motion` (from the tracks) | still, walking, running, unknown | How far the box moved since the last sample, in **body heights per second** (centre shift plus change of size): under 0.15 **still**, from 1.5 **running** (walking is about 0.8) |
| `carrying` (from the tracks) | backpack, handbag, suitcase, umbrella | One of these detector classes whose centre is inside the person's box (widened 15 % each side) and which is smaller than the person |

![Six poses and the facts the rules give them](docs/images/pose-rules.svg)

*Drawn figures, but the facts under each are the output of the real `describe()` in `pose.py`.*

### 22.4 Storing and alerting

- **Sampling:** each person about once a second (`interval_seconds`), three times less often while
  they stand still, on its own thread.
- **Smoothing:** a fact counts once it is the majority of the last 3 samples, so one bad frame
  doesn't make a standing person "lying".
- **Rows on change:** a row only when the smoothed facts change, plus a heartbeat every 10 s: an
  hour at a desk is ~360 rows, not 3,600, and the time in each posture is still exact. Each row
  (~250 bytes) keeps the 17 keypoints relative to the box (102 bytes), so a new fact can be derived
  later without running the model again.
- **Events:** `down` (rule `person_down`) when a person stays lying for 10 s, `hands_up` when both
  hands stay up for 3 s, for the types in `pose.alerts`. Same store, snapshots, channels and alert
  checking as area and line events.
- **Search:** the chat's `people` tool and `GET /people` filter by posture, arms, facing, head,
  motion, carrying, colours, camera and time, and return each visit with the seconds spent in each
  posture and motion.

### 22.5 Accuracy, limits and speed

![Real: street frame, keypoints and facts per person](docs/images/pose-street.jpg)

*Real: frame 606 of the street video (shown at 2×), YOLOX-Nano + ViTPose-B on CPU. Labels: posture,
facing, then top / trousers colour. The red-and-black jacket reads "black": the colour covering most
of the torso area wins.*

Keypoint confidence drops sharply below about 45 px tall and holds above 0.8 from about 55 px;
colours and posture need more pixels than a keypoint, so the 80 px minimum (`min_box_height`)
stays on the safe side.

![Real: mean keypoint confidence against person height](docs/images/pose-size.png)

- **Not yet validated on real falls.** Person-down is checked with unit tests and the street video
  only: before relying on it, staged falls on the site's cameras and a count of false alarms per
  day (alert checking can then hold back the obvious ones).
- Someone half hidden behind a desk is "unknown"; sitting and crouching can be confused side-on.
  Bending, head down, motion and carrying are checked with unit tests only, not real footage.
- **Head down** sees the head, not the phone. **Motion** is measured in the picture (walking
  straight at a far camera can read as still). **Carrying** needs the detector to find the bag.
- **Colours** are lost at night (infrared) and shift under coloured lighting.

![Real: pose model speed per person on a laptop CPU](docs/images/pose-speed.png)

*Measured 2026-10-09 on an Intel i7-13700H laptop, CPU only, one person per run (median of 25).*

```yaml
pose:
  enabled: true
  cameras: []                  # empty = every camera
  model: models/pose_vitpose_b.onnx   # trials: pose_rtmpose_t.onnx
  interval_seconds: 1          # per tracked person; 3x less often while they stand still
  min_box_height: 80
  heartbeat_seconds: 10
  store_keypoints: true
  alerts: [down]               # down | hands_up
  down_seconds: 10
  hands_up_seconds: 3
```

## 23. Number plates (opt-in)

With a `plates:` section, every tracked vehicle (car, truck, bus, motorcycle) gets its plate read
about once a second, on its own thread (`plate_index.py`), cut from the main-stream keyframe when
the vehicle was matched in one: plates need pixels.

![Number plates: finder, reader, vote, then store, search and watchlist](docs/images/plates-pipeline.svg)

- **Finding:** a plate detector on the vehicle's crop (default: open-image-models YOLOv9-t 384,
  NMS inside the ONNX graph, ~11 ms per vehicle). Its code is MIT but its weights were trained with
  a fork of the GPL-3.0 YOLOv9 repo: **trials only**. Urgent to-do: our own YOLOX on Open Images'
  "Vehicle registration plate" boxes (CC BY 4.0), `detector.type: yolox`.
- **Reading:** fast-plate-ocr CCT-S v2 global (MIT, ~10 ms): a 128 × 64 RGB crop in, 10 character
  slots over 0-9 A-Z (plus a pad character for unused slots), and a country guess among 66. A read
  counts when every character is at least `min_confidence` (0.6) sure.
- **Voting and storing:** a vehicle's plate is the text with the highest summed confidence over its
  reads. A row (with the plate crop) is stored for each new text, or a clearly better read, at most
  `max_per_track` (5) per vehicle. Plates follow `store.plate_retention_days` (default: the search
  index's retention).
- **Search:** `GET /plates` and the chat's `plates` tool. Look-alike characters (0/O/D/Q, 1/I/L,
  2/Z, 5/S, 6/G, 8/B) count as equal, one more wrong character is forgiven, and 3+ characters also
  match plates containing them.
- **Watchlist:** `watchlist: [{plate: "AB 123 CD", name: banned_van}]` raises a `plate` event (rule
  = the name) once per vehicle when its voted plate, read `min_reads` (2) times, matches.
- **Live view:** the voted plate is drawn under the vehicle's box.

![Real models on synthetic plates: finder and reader under blur, distance, night and angle](docs/images/plates-reads.jpg)

*Real finder and reader, synthetic car and plate (as in the tests). All five versions are found and
read exactly; the bottom chart is the reader's raw output for the clean plate: one column per slot,
one row per character. Synthetic plates are easy: angle, distance, night IR glare and dirty plates
on a real camera will lower the read rate. Not yet checked on a real camera.*

## 24. Sound events (opt-in)

With a `sounds:` section, `sound.py` listens to each camera's audio (the main stream usually
carries it; ffmpeg) or a video file's (a `.wav` is read directly). **YAMNet** (Google, Apache-2.0,
AudioSet's 521 classes) scores it: one row per 0.96 s frame, every 0.48 s, about 3 ms per 3 s of
audio on one CPU thread. Groups map Histor names to AudioSet classes:

| Group (rule) | AudioSet classes |
| --- | --- |
| `smoke_alarm` | Smoke detector, smoke alarm · Fire alarm |
| `glass_break` | Shatter · Glass |
| `gunshot` | Gunshot, gunfire · Machine gun · Fusillade · Cap gun |
| `explosion` | Explosion |
| `scream` | Screaming |
| `car_alarm` | Car alarm |
| `siren` | Siren · Civil defense siren · Police car, Ambulance, Fire engine (siren) |

A group fires when one of its classes scores `min_score` (0.3; glass and gunshot 0.4) in
`min_frames` (2) frames in a row, then rests `cooldown_seconds` (30). The event is `sound` (rule =
group, label = the class heard). From a file, each event goes out when the video reaches it, so its
snapshot shows that moment. **The audio is classified in memory and dropped**; only events are
kept. Listening is regulated in many places, so the section is off by default.

![Real: YAMNet scores on synthetic audio, and the event it raises](docs/images/sound-scores.png)

*Real model, synthetic sound: room noise, then a 3.1 kHz smoke-alarm pattern from 8 s to 16 s. The
smoke-alarm group jumps above 0.3 within a second and fires after two frames in a row; the other
groups stay near zero. Not yet checked on real camera audio (no ffmpeg on the build machine).*

## 25. Chat

![The chat: a question, tools over the store, an answer with evidence](docs/images/chat-flow.svg)

The chat model gets the store as tools and may run up to six rounds of lookups per question. Each
tool takes a time range in words or exact ISO `since` / `until`, and a camera:

| Question | Tool call | Kind |
| --- | --- | --- |
| What happened yesterday 6pm to 9pm? | `objects(when="yesterday 6pm to 9pm")`: everything tracked, in order, with counts | exact |
| How many cars came in today? | `objects(labels=["car"], when="today")`: one tracked object = one visit | exact |
| Anyone with a dog this morning? | `objects(labels=["person"], with_labels=["dog"], when="this morning")` | exact |
| A man in a red jacket near a car? | `search(query="man in a red jacket", with_labels=["car"])` | ranked |
| Did any alert go off last night? | `alerts(when="last night")`, with each alert's check verdict | exact |
| Did anyone fall in the warehouse? | `people(postures=["lying"], ...)`; also `alerts(types=["down"])` | exact |
| A red top and blue jeans this morning? | `people(upper_colour="red", lower_colour="blue", ...)` | exact |
| Was anyone running in the car park? | `people(motion=["running"], camera="car park")` | exact |
| When did AB 123 CD come in? | `plates(plate="AB 123 CD")`: look-alikes and one wrong character forgiven | exact / close |
| How many people were at the counter at 3pm? | `objects` or `search` to find the moment, then `look(images=[...], question="How many people?")` | described |
| Did a smoke alarm go off last night? | `alerts(types=["sound"], rules=["smoke_alarm"], when="last night")` | exact |
| *(photo attached)* Did this person come in between 1 and 4? | `face(photo="photo1", ...)`: likely / possible visits | scored |

"Together" means the same camera and overlapping in time (`within_seconds`, default 15).
`people`, `plates`, `face` and `look` are only offered where they are enabled or have data.

**`look`** (`vision.py`): the chat model passes 1–4 images that earlier results listed (thumbs,
scene images, snapshots, plate crops; any other path is refused). They are shrunk to `max_side`
(640 px) and sent to the local vision server; only its text answer goes back, even when the chat
model is Claude. If the vision server is down, the tool says so and the chat carries on.

### 25.1 Models

The model is a switch in `configs/default.yaml` (`chat.model`, entries under `chat.models`). Any
OpenAI-compatible server works as `api: openai` (Ollama, llama.cpp's `llama-server`, LM Studio,
vLLM, or a hosted one with `api_key: ${KEY}`): add an entry, no code change.

| Entry | Runs on | Setup |
| --- | --- | --- |
| `local` (default) | Ollama, CPU (Qwen3 4B, Apache-2.0) | `ollama pull qwen3:4b` |
| `qwen3.5-4b` | Ollama, CPU (Qwen3.5 4B, Apache-2.0, also reads images) | `ollama pull qwen3.5:4b` |
| `granite-3b` | Ollama, CPU (IBM Granite 4.1 3B, Apache-2.0, smallest) | `ollama pull granite4.1:3b` |
| `gemma4-e4b` | Ollama, CPU (Gemma 4 E4B, Apache-2.0, also reads images) | `ollama pull gemma4:e4b` |
| `local-large` | Ollama, ~20 GB RAM or GPU (Qwen3 30B-A3B) | `ollama pull qwen3:30b-a3b` |
| `qwen3.6-large` | Ollama, ~32 GB RAM or GPU (Qwen3.6 35B-A3B, Apache-2.0) | `ollama pull qwen3.6:35b` |
| `nemotron-large` | Ollama, ~32 GB RAM or GPU (Nemotron 3.5 Lightning 30B-A3B, OpenMDW 1.1: read before shipping) | `ollama pull nemotron-3.5-lightning` |
| `claude` | Anthropic API (Claude Opus 5.5) | `uv sync --group claude`, `ANTHROPIC_API_KEY` in `.env` |

The open models after `local` were added on 2026-10-08 to compare on the same questions; the
default stays `qwen3:4b` until the planned evaluation picks a winner. Small local models follow
tool instructions less reliably than large ones: if answers look wrong, run with tool calls
visible and check which lookups it made.

```bash
uv run -m histor.chat                                      # interactive; /model claude, /models, /photo
uv run -m histor.chat "how many cars came in yesterday?"
uv run -m histor.chat --model claude "was there a dog on the lawn this morning?"
```

### 25.2 What each camera is (`about:`)

Filled in at install time, standing at the camera, so the chat picks the right camera and explains
its answers. It goes into the model's instructions as one short line per camera (cut at ~300
characters for small local models). Use roles ("manager's office"), not people's names: with a
cloud model, this text is sent too.

| Field | What it does |
| --- | --- |
| `description`, `location` | "did they reach the office" means the office camera, not the corridor on the way |
| `aliases` | other names in questions ("reception" → `lobby`), for the chat and `ask.py` |
| `covers` | entrance / exit / passage / room / outdoor / parking: "arrived" vs "passed by" |
| `faces` | good / poor / none: face search explains a "no" ("the lobby camera can't see faces well") |
| `hours` | normal activity, so "after hours" needs no extra input |
| `notes`, `zones` | what is normal ("cleaners at 19:00"), and what each alert area or line means |

### 25.3 MCP server: the same tools for other agents

`histor.mcp_server` serves the chat's tools over the **Model Context Protocol**, so any MCP client
(Claude Desktop, Claude Code, a client's own agent) can ask a site's Histor what happened, with the
same descriptions, arguments and JSON results as the chat. It runs over stdin/stdout: the client
starts it as a command, and nothing new listens on the network.

| | |
| --- | --- |
| **Tools** | `objects`, `search`, `alerts`, `people`, `plates`, `look`, where enabled; all marked read-only |
| **Face search** | opt-in (`mcp.face_search`): a `face` tool that takes a photo **file** on the server (below). Off, the instructions tell the model that face search exists in the chat, UI and Telegram bot, so it never says "Histor can't match faces" |
| **Instructions** | the chat's own (cameras with their `about:`, alert rules, how to use each tool) |
| **Time** | every result carries `site_time`, the site's clock, so "last night" is resolved right even when the client's model runs elsewhere |
| **Images** | never sent: results list thumbnail paths; `look` has the site's vision model read them and returns text |
| **Privacy** | a cloud model behind the client receives the same text as the chat's `claude` model: times, cameras, labels, file names, answers |

```bash
uv sync --group mcp
uv run --group mcp -m histor.mcp_server --config configs/default.yaml   # what a client runs
claude mcp add histor -- uv run --directory /opt/histor/src/projects/histor --group mcp -m histor.mcp_server --config configs/site.yaml
```

Claude Desktop (`claude_desktop_config.json`), on the site server itself, or from a laptop over
SSH (the server's SSH login is the only access needed):

```json
{
  "mcpServers": {
    "histor": {
      "command": "ssh",
      "args": ["histor@site-server", "cd /opt/histor/src/projects/histor && ~/.local/bin/uv run --group mcp -m histor.mcp_server --config configs/site.yaml"]
    }
  }
}
```

**Face search over MCP** (`mcp.face_search: true`, and `faces.enabled`). A photo attached in the
client's chat can't reach a tool: the model sees its pixels, not its file. So the `face` tool
takes the **path of a photo file on the Histor server** ("is the person in
`C:/photos/visitor.jpg` on the cameras today?"), makes its face print as the chat's `/photo` does,
and returns likely / possible visits and what was checked, as JSON text. The photo is read, never
copied or sent anywhere.

```yaml
mcp:
  face_search: false           # biometric: the client's written approval first
  photo_dirs: []               # only photos under these folders, e.g. [C:/histor-photos]; empty = any
```

Only image files (.jpg, .jpeg, .png, .webp, .bmp, up to 20 MB) are opened; with `photo_dirs`, only
inside those folders. An unusable face comes back with the reason ("too small (31 px, needs 40)").

Checked: `tests/test_mcp.py` (tool list, read-only hints, results with `site_time` and verdicts,
errors, `look` on a listed snapshot, instructions, nothing printed on stdout; face search off by
default with the "exists elsewhere" instruction, a photo file matched, a dark photo, a text file,
a missing file, a file outside `photo_dirs`), a real stdio session against the laptop store
(list tools, `objects`, `alerts`, `face` with the real models), and Claude Desktop answering from
the laptop store (2026-10-10). Later: a streamable-HTTP transport with authentication, for clients
that can't start a command.

## 26. Reports

Set up at install time with the owner: how often, which cameras, what they care about.

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
        sender: histor@example.com
        to: [owner@example.com]
```

```bash
uv run -m histor.report                   # runs the schedules: keep it running next to the recorders
uv run -m histor.report --now morning     # the last period's report now (--no-send: save only)
uv run -m histor.report --list            # saved reports; --show morning: the latest one
```

Each report has **counts**, exact, from the store (tracked objects per camera, alerts with their
times, any camera that recorded nothing: offline?), and a **summary** written by the chat model,
which checks each interest with the chat's tools, limited to the report's cameras and period. If
the model is down, the report goes out with the counts only and says why. Reports are saved as
JSON and Markdown under `data/store/reports/<name>/` and sent: email with up to 4 thumbnails,
webhook as JSON (`kind: report`). A period that ended while `report.py` was stopped is made when it
starts again (the latest one only). With a cloud model, only counts and interests are sent.

## 27. Storage, retention and API

| What | Table | Used for |
| --- | --- | --- |
| Events + snapshot (+ check verdict) | `events` | alerts, history, `alerts` tool |
| CLIP vectors of objects, frames and tiles (+ model name) | `sightings` | text search |
| Face prints (only where enabled) | `faces` | search by photo |
| Body pose facts per person (+ keypoints) | `poses` | people search, person-down |
| Plate reads with a crop | `plates` | plate search, watchlist |

One SQLite file per site (WAL mode), shared by every recorder and the API; media (snapshots,
thumbnails) next to it. A store made by an older version gets new columns when it is opened.
**Retention:** sightings and events 30 days, face prints 7, plates `plate_retention_days`
(default: same as sightings), recordings 7 days (or sooner when disk is low). Purging runs inside
the API server.

```bash
uv sync --group server
uv run --group server -m histor.engine.server --config configs/default.yaml   # http://127.0.0.1:8000/docs
```

| Endpoint | Example |
| --- | --- |
| `GET /search?q=` | `/search?q=person in a red top last friday` · `/search?q=white van on gate yesterday 6pm to 9pm` |
| `POST /search/face` | form fields `photo`, `when`, `camera`, `face` (1 = leftmost) → *"Yes, likely: 2 visit(s), first on shop at Fri 25 Sep 10:30 (match 0.58)."* |
| `GET /people` | `?posture=lying&when=last night` · `?upper=red&lower=blue&camera=shop` · `?arms=up` |
| `GET /plates` | `?plate=AB 123 CD&when=yesterday` · `?when=today&camera=gate` |
| `GET /events` | `?camera=gate&type=cross&since=2026-09-25T00:00&limit=50` (rows include `verified`, `verification`) |
| `GET /media/{path}` | snapshots and thumbnails |
| `GET /recordings/{path}` | main-stream MP4s; results carry a `video_url` that opens the moment, 3 s early |
| `GET /health`, `GET /cameras`, `POST /maintenance/purge` | status, camera list, apply retention now |

### 27.1 Dashboard and login

The API server also serves the **dashboard** at `/` (`engine/dashboard.py`, `dashboard.html`):
one plain web page for the people who watch a site, on a desktop or a phone, light or dark. No
framework and no build step: it reads the endpoints above and refreshes every 5 s.

![The dashboard on the street video's demo store](docs/images/dashboard.jpg)

*The dashboard on a demo store: the street video recorded with the zones of section 16 and alert
checking on. With no vision model running, real checks came back "Not checked (model error)" and
the alerts went out unverified, as designed; the two verdicts shown ("Checked: real", "Held back")
were set by hand for this picture.*

- **Cameras:** online (something stored in the last 2 minutes) or offline, last activity, objects
  tracked today per class, alerts today and how many the check held back, and a tamper alert from
  the last hour. From `GET /status`.
- **Alerts:** the feed, newest first, with the snapshot (click for full size and the recording),
  what happened in words, the check's verdict and reason. Filters: today / 24 h / 7 days, camera,
  type, and sent vs held back. New alerts flash as they arrive.
- **Find by description** (`/search`) and **number plate** (`/plates`), with thumbnails.

**Login.** One token per site, in the config:

```yaml
server:
  token: ${HISTOR_TOKEN}       # in .env: python -c "import secrets; print(secrets.token_urlsafe(24))"
```

- With a token, **every route needs it**: the page shows a sign-in form and sets a session cookie
  (HttpOnly, SameSite=Strict, derived from the token, not the token itself); scripts send
  `Authorization: Bearer <token>`. A wrong token waits a second before answering.
- Without a token (`token: ""`, the default), the server **refuses any `--host` but this machine**.
  A `${VAR}` that isn't set is an error, never "no login".
- Over a network, put HTTPS in front (a reverse proxy, section 37) so the cookie can't be read on
  the way. One shared token per site is a start; logins per person are an open question.

Checked: `tests/engine/test_dashboard.py` (every route closed without the token, Bearer and cookie
sign-in, logout, the cookie isn't the token, `/status` counts, the host check, slowed wrong
logins), and in the browser on the demo store: sign-in, feed, filters, viewer, search, phone width,
light and dark, no console errors.

## 28. Test UIs

```bash
uv run --group ui -m histor.ui                                   # configs/laptop.yaml, http://127.0.0.1:7860
uv run --group ui -m histor.ui --config configs/default.yaml     # a site's config
```

Tabs: **Camera** (start recording, live view with tracks, skeletons and plates), **Find a person**
(photo → likely / possible visits, no LLM), **Ask** (the chat, with *Attach photo*; shows whether
the model's server is up), **Reports**, and the engine's **Test bench**: pick a model and compare
original vs. annotated for an image, a video, the webcam or an IP camera; draw **Zones & lines**
and copy the YAML; the **Faces** tab shows the faces found in several photos, their quality and
the SFace similarity between them. Gradio is in the optional `ui` group, not installed on servers.

---

# Part 3 — Installing a site

The step-by-step guide for an Histor site: cabling the cameras to a PoE switch, connecting the
switch to the server, putting everything on the right network, and starting Histor. Follow it in
order: every step ends with a check, so a problem is caught where it happens. Cameras live on an
**isolated camera network**, the server is the only machine that talks to them, and nothing about
the cameras is exposed to the internet.

## 29. What you are building

```text
              Internet / client office LAN
                          |
                 +--------+--------+
                 | Router / office |
                 |     switch      |
                 +--------+--------+
                          |  Cat6 - NIC 1 "LAN" (DHCP, has a gateway)
                 +--------+--------+
                 |  Linux server   |<-- UPS
                 |  Histor + API    |
                 +--------+--------+
                          |  Cat6 - NIC 2 "cameras" (static 10.50.0.1, NO gateway)
                          |         into the switch's uplink port
                 +--------+--------+
                 |   PoE switch    |<-- UPS
                 +--+-----+-----+--+
                    |     |     |      Cat6, max 100 m each, carries data + power
                 +--+-+ +-+--+ +-+--+
                 |Cam1| |Cam2| |Cam3|  ...
                 +----+ +----+ +----+
```

- **PoE switch:** one cable per camera carries both video and power.
- **Server, two network ports:** one faces the cameras, the other the office LAN / internet. The
  server does **not** route between them, so cameras can never reach the internet and the internet
  can never reach the cameras.
- **Software:** one Histor recorder per camera, one API process, and (optionally) Ollama for the
  chat and the vision model, and the report scheduler.

> **Only one network port on the server?** Add a USB 3.0 gigabit Ethernet adapter and use it as
> the camera port. A managed switch with VLANs works too, but takes more configuration and is
> easier to get wrong.

## 30. Bill of materials

| Item | What to buy | Notes |
| --- | --- | --- |
| IP cameras | ONVIF + RTSP, H.264 (or H.265), PoE (802.3af) | Hikvision, Dahua, Reolink, Uniview and Axis have presets. Avoid cloud-only / Wi-Fi consumer cameras: often no RTSP. A microphone only if sound events are wanted and allowed. |
| PoE switch | Ports ≥ cameras + 1 (uplink) + 1 spare. **Gigabit uplink.** 802.3af/at | Check the **total PoE budget** (31.2), not just the port count. |
| Server | x86-64, 8+ cores, 16–32 GB RAM, SSD for the system, **a large HDD for recordings**, **2 Ethernet ports** | Ubuntu Server 24.04 LTS. CPU-only is fine (section 41). More RAM if the chat and vision models run here. |
| Cable | Cat6 (Cat5e acceptable), **solid pure copper**, outdoor/UV-rated outside | Never CCA (copper-clad aluminium): it overheats and drops voltage under PoE. |
| Connectors | RJ45 plugs (shielded if the cable is), boots | Plus outdoor junction boxes / waterproof caps. |
| Recording disk | Surveillance-grade HDD (WD Purple, Seagate SkyHawk) | 1080p at 4 Mbit/s = **~43 GB per camera per day**. 8 cameras × 7 days ≈ 2.4 TB. |
| Tools | Crimper, cable tester (wire map), ladder, drill | A PoE tester is a cheap, big time saver. |
| Power | UPS sized for server + switch | If the switch loses power, every camera goes dark. |
| Commissioning laptop | Any laptop with an Ethernet port | To activate cameras with the vendor's tool (Hikvision SADP, Dahua ConfigTool). |

## 31. Plan before you drill

### 31.1 IP address plan

Write this down before you touch a camera, and leave a copy at the site.

| Device | IP address | Notes |
| --- | --- | --- |
| Server, camera port (NIC 2) | `10.50.0.1/24` | No gateway |
| Camera 1 (e.g. `gate`) | `10.50.0.11` | Gateway `10.50.0.1` or empty |
| Camera 2 (e.g. `yard`) | `10.50.0.12` | |
| Camera 3 (e.g. `shop_entrance`) | `10.50.0.13` | |
| Managed switch (if any) | `10.50.0.2` | |
| Server, LAN port (NIC 1) | From the client's DHCP | Ask IT to reserve it |

Why not the factory default `192.168.1.x`: many offices already use `192.168.1.0/24`, and if the
camera network uses the same range as the office LAN, the server can't tell which port to use.
If `10.50.0.0/24` is in use at the client, pick any other private range that isn't.

### 31.2 PoE power budget

Add up the **maximum** power of every camera (datasheet, "max power consumption", IR on).

| Standard | Per port at the switch | Typical device |
| --- | --- | --- |
| 802.3af (PoE) | 15.4 W | Fixed bullet / dome camera |
| 802.3at (PoE+) | 30 W | Cameras with heater, motorised zoom, strong IR |
| 802.3bt (PoE++) | 60–90 W | PTZ cameras with heater |

Example: 6 cameras × 7 W = 42 W. A 65 W switch is fine; a 30 W one looks fine during the day and
starts rebooting cameras at night when the IR LEDs switch on. Keep **at least 20 % headroom**.

### 31.3 Cable runs

- **100 m maximum** per cable, switch to camera, including patch cords; longer needs a PoE extender
  or a second switch.
- At least 30 cm from mains power cables when running in parallel; cross them at right angles.
- Outdoor: outdoor-rated cable in conduit, a drip loop before each entry, a surge protector on long
  exposed runs.

### 31.4 Bandwidth

| Stream | Typical bitrate (H.264) |
| --- | --- |
| Sub stream (640×360 – 704×576) | 0.5 – 1 Mbit/s |
| Main stream 1080p | 4 – 6 Mbit/s |
| Main stream 4 MP / 4K | 6 – 16 Mbit/s |

The detector reads the **sub stream**; the server also opens every camera's **main stream**
(recording + hi-res crops), so use a **gigabit** uplink: 20 cameras × (6 + 1) Mbit/s is ~140 Mbit/s.

### 31.5 Where to put each camera

- 2.5–4 m high, angled down; not into the sun, headlights or bright windows (backlight ruins
  detection).
- **Faces** need ~32+ px on the image: a camera close to a door, at head height, not a wide shot
  of the car park (section 21.5).
- **Plates** need a camera aimed at the lane where vehicles slow down, plates at most ~30° off-axis.
- **Poses** need people at least 80 px tall; **person down** needs the floor in view.
- Fill in each camera's `about:` standing at the camera (section 25.2).

## 32. Physical installation

**Step 1 — Terminate and test every cable.** Crimp both ends T568B; test each cable **before**
mounting anything (all 8 wires must pass); label both ends with the camera name and switch port.
*Check:* every cable passes, every end is labelled.

**Step 2 — Mount the cameras** (31.5); seal the outdoor connector.

**Step 3 — Install the switch and server.** Both in a ventilated cabinet, both on the UPS. Each
camera cable into a PoE port (write down which). The **switch's uplink** into the server's **NIC 2**;
the server's **NIC 1** into the client's LAN. Power on the switch, then the server.
*Check:* each camera port shows a link LED and a PoE LED; IR LEDs glow faintly red in the dark after
~1 minute. Link but no PoE: check the budget and the cable.

> **Do not connect the PoE switch to the office LAN as well.** The camera network stays connected
> to the server only.

## 33. First-time camera setup

New cameras come with the **same factory IP** (Hikvision `192.168.1.64`, Dahua `192.168.1.108`,
Reolink: DHCP) and, for most vendors, **no password** until activated. Two cameras with the same IP
conflict, so configure them **one at a time**, with the commissioning laptop plugged into a spare
switch port.

**Step 4 — Put the laptop on the factory subnet** (admin PowerShell; `Get-NetAdapter` shows the
adapter name). No gateway, so the laptop's internet stays on Wi-Fi:

```powershell
New-NetIPAddress -InterfaceAlias "Ethernet" -IPAddress 192.168.1.10 -PrefixLength 24
New-NetIPAddress -InterfaceAlias "Ethernet" -IPAddress 10.50.0.250 -PrefixLength 24
```

**Step 5 — Activate and re-address each camera.** Unplug all cameras but one, then for each:

1. Run the vendor tool (**SADP**, **ConfigTool**) or open `http://192.168.1.64` (or `.108`).
2. **Activate** it with a strong, unique password, saved in the company password manager under the
   site and camera name. Never a default or shared password.
3. Change its IP to the plan's (e.g. `10.50.0.11`, mask `255.255.255.0`, gateway `10.50.0.1`).
4. On the camera's web page at its new address, set:

| Setting | Value | Why |
| --- | --- | --- |
| ONVIF | **Enabled**, with an ONVIF user | Lets `cameras discover` find it |
| RTSP | Enabled, port **554**, **digest** authentication | How the server reads video |
| Sub stream | H.264, 640×360 to 704×576, **10–15 FPS**, I-frame interval = 2× FPS | What the detector reads: cheap to decode |
| Main stream | H.264 (H.265 only if the server decodes it fine), 10–15 FPS, **I-frame interval = FPS**, constant or capped bitrate | Recorded as-is; one keyframe a second gives the hi-res crops |
| Audio | On, AAC or G.711, only if sound events are used and allowed | Carried in the main stream |
| Smart codec (H.264+ / H.265+) | **Off** | Long I-frame gaps cause grey/smeared frames after reconnects |
| Time | NTP server `10.50.0.1`, correct time zone | "yesterday 6pm" depends on it |
| UPnP, P2P / cloud (Hik-Connect, DMSS…) | **Off** | The camera must not try to reach the internet |
| Firmware | Latest from the vendor | Older firmware has known vulnerabilities |
| On-screen camera name | Same as in the config (`gate`, `yard`…) | Self-explanatory snapshots |

*Check:* all cameras plugged in again, each opens in the browser at its planned `10.50.0.x` address.

> **Client already has an NVR?** You can read the cameras through it (the NVR's IP and the
> `channel:` setting). Direct-to-camera is still preferred: one less box that can fail, no re-encode.

## 34. Server setup

Ubuntu Server 24.04 LTS, freshly installed, with a non-root admin user and SSH.

**Step 6 — Configure the two network ports.** `ip -br link` shows their names (unplug one cable and
run it again if unsure: the one now `DOWN` is it). Create `/etc/netplan/60-histor.yaml`:

```yaml
network:
  version: 2
  ethernets:
    enp1s0:                     # NIC 1: client LAN / internet
      dhcp4: true
    enp2s0:                     # NIC 2: PoE switch / cameras
      dhcp4: false
      addresses: [10.50.0.1/24]
      # no gateway, no DNS: this network goes nowhere
```

```bash
sudo chmod 600 /etc/netplan/60-histor.yaml
sudo netplan try        # press Enter within 120 s to keep it, otherwise it rolls back
ip -br addr show enp2s0       # shows 10.50.0.1/24
ping -c 3 10.50.0.11          # every camera answers
ip route                      # exactly one "default via" line, on enp1s0
```

**Step 7 — Keep the networks separate.** `sysctl net.ipv4.ip_forward` must print 0. Firewall:
SSH from the LAN, time for the cameras, nothing else in. Never port-forward RTSP (554) or the
camera pages (80/443) on the client's router.

```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow in on enp1s0 to any port 22 proto tcp
sudo ufw allow in on enp2s0 from 10.50.0.0/24 to any port 123 proto udp
sudo ufw enable
```

**Step 8 — Serve the time to the cameras.**

```bash
sudo apt install -y chrony
echo "allow 10.50.0.0/24" | sudo tee /etc/chrony/conf.d/cameras.conf
sudo systemctl restart chrony
timedatectl                    # "System clock synchronized: yes", correct time zone
```

Set the zone if needed (`sudo timedatectl set-timezone Asia/Beirut`). *Check:* after a few minutes
each camera's clock matches the server.

## 35. Install the software

**Step 9 — uv, the code and the models.**

```bash
sudo apt install -y git ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | sh     # installs uv to ~/.local/bin
sudo mkdir -p /opt/histor && sudo chown $USER /opt/histor
git clone <repo-url> /opt/histor/src
cd /opt/histor/src/projects/histor
uv sync --group server
uv run -m histor.engine.download_model yolox_s
uv run --group export -m histor.engine.export_clip   # text search (~600 MB)
uv run --group export -m histor.engine.export_pose   # body poses (344 MB)
uv run -m histor.engine.download_model faces         # only if face search is used AND approved
uv run -m histor.engine.download_model plates        # only if plates are used (trial finder)
uv run -m histor.engine.download_model sounds        # only if sound events are used
```

For the chat, the `look` tool and alert checking, install [Ollama](https://ollama.com) on the
server and pull the models: `ollama pull qwen3:4b` and `ollama pull qwen3.5:4b`.

**Step 10 — Find the cameras from the server.**

```bash
uv run -m histor.engine.cameras discover            # ONVIF: IP + model of each camera
uv run -m histor.engine.cameras scan 10.50.0.0/24   # fallback: every host with RTSP (554) open
```

Every camera in the plan must appear. `discover` empty but `scan` finds them: ONVIF is off (fine).

**Step 11 — Write the site config.** `cp configs/default.yaml configs/site.yaml`, then declare the
cameras (and their `about:`, section 25.2):

```yaml
cameras:
  gate:
    vendor: hikvision          # hikvision | dahua | reolink | uniview | axis
    host: 10.50.0.11
    username: admin
    password: ${GATE_CAM_PASSWORD}
    stream: sub                # low-res sub stream for detection; main stream opened for recording
  other_brand:                 # unknown vendor: give the full RTSP URLs
    url: rtsp://${OTHER_USER}:${OTHER_PASS}@10.50.0.13:554/stream2
    main_url: rtsp://${OTHER_USER}:${OTHER_PASS}@10.50.0.13:554/stream1
```

Passwords go in `.env` next to the config (git-ignored, `chmod 600`), never in the YAML. Point the
store and recordings at the big disk and agree the retention with the client (`store.path`,
`recording.dir: /srv/recordings`, `recording.retention_days`, `min_free_gb`). Then add the site's
zones, lines, alert channels and alert checking (sections 16–18; draw zones in the test bench).

**Step 12 — Test each camera.**

```bash
uv run -m histor.engine.cameras --config configs/site.yaml list              # URLs, passwords masked
uv run -m histor.engine.cameras --config configs/site.yaml test gate --save data/gate.jpg
uv run -m histor.record --config configs/site.yaml --source gate             # Ctrl+C once it works
```

Expected: `OK: 640x360, ~15.0 FPS received` and a snapshot: right camera, right angle, in focus.

## 36. Run it as a service

One systemd service per camera, one for the API (which also applies retention), one for reports.
`/etc/systemd/system/histor-camera@.service` (the `@` makes it a template, one instance per camera):

```ini
[Unit]
Description=Histor recorder, camera %i
After=network-online.target
Wants=network-online.target

[Service]
User=histor
WorkingDirectory=/opt/histor/src/projects/histor
ExecStart=/home/histor/.local/bin/uv run -m histor.record --config configs/site.yaml --source %i
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

`histor-api.service` is the same with `Description=Histor API` and
`ExecStart=... uv run --group server -m histor.engine.server --config configs/site.yaml`;
`histor-reports.service` with `ExecStart=... uv run -m histor.report --config configs/site.yaml`;
`histor-telegram.service` (only with the Telegram bot, section 17.1) with
`ExecStart=... uv run -m histor.telegram_bot --config configs/site.yaml`.
Replace `histor` with the server's user name, then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now histor-camera@gate histor-camera@yard histor-api histor-reports
sudo systemctl enable --now histor-telegram                       # only with the Telegram bot
systemctl status 'histor-*'
journalctl -u histor-camera@gate -f               # live logs of one camera
```

*Check:* reboot the server; without logging in, every service comes back. Unplug a camera for 30 s:
its log shows the stream reconnecting by itself.

## 37. Access the API safely

The API and dashboard listen on `127.0.0.1:8000` by default. Set `server.token` (section 27.1)
before anything else: without it the server refuses any other `--host`. From a laptop, the
simplest safe route is an SSH tunnel, using it as if local:

```bash
ssh -L 8000:127.0.0.1:8000 histor@<server-LAN-IP>
```

Then <http://127.0.0.1:8000/> (dashboard) or `/docs` (API). For permanent access from the office
or a phone: a VPN (WireGuard / Tailscale), or a reverse proxy with HTTPS (Caddy, nginx) in front of
`127.0.0.1:8000`, with the token set; never a raw port forward.
*Check:* the dashboard asks for the token, then shows every camera online and alerts arriving
during the walk test.

## 38. Checklists

**Security and privacy**

- [ ] Every camera has its own strong password; no factory defaults remain.
- [ ] Camera P2P/cloud and UPnP are off; firmware is up to date.
- [ ] The PoE switch is connected to the server only, never to the office LAN.
- [ ] `net.ipv4.ip_forward = 0`; the camera port has no gateway; firewall on; no port forwards.
- [ ] `server.token` set (in `.env`); API on `127.0.0.1`, reached through SSH tunnel / VPN / HTTPS proxy.
- [ ] `.env` is `chmod 600` and not committed anywhere.
- [ ] Faces only with the client's **written approval**, local law checked, notices posted,
      `face_retention_days` set. Plates and sounds likewise only where agreed.
- [ ] `retention_days` and `recording.retention_days` agreed with the client and set.

**Commissioning (hand-over)**

- [ ] IP plan and switch port map printed and left in the cabinet; cables tested and labelled.
- [ ] PoE budget checked at night with all IR on (no camera reboots).
- [ ] All cameras show the correct time (NTP from the server).
- [ ] `cameras test` passes for every camera; snapshots reviewed; `about:` filled in.
- [ ] Services survive a reboot and a camera unplug/replug.
- [ ] Walk test: someone walks each zone / crosses each line; events appear in `/events` with a
      correct snapshot; alerts (webhook / email) arrive; with alert checking, each has a verdict.
- [ ] Server and switch on UPS; UPS tested by pulling the mains plug.
- [ ] Passwords in the company password manager under the site name.

## 39. Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Switch port: link LED but no PoE LED | Budget exceeded, or camera needs PoE+ | Check total budget; PoE+ port |
| Camera reboots at night | IR LEDs push power over the budget | Bigger switch or fewer cameras per switch |
| No link at all | Bad crimp, cable > 100 m, CCA cable | Re-test; re-crimp; pure copper |
| Server's camera port shows `169.254.x.x` | Netplan not applied to the right port | Fix the port name in `60-histor.yaml`, `sudo netplan apply` |
| `ping` works, `cameras test` fails | Wrong RTSP path, password, or RTSP off | Try the URL in VLC; check `vendor:` / `path:`; re-type the password in `.env` |
| `401 Unauthorized` | Wrong password, or camera locked after failed logins | Wait 30 min or unlock from the camera page; fix `.env` |
| `discover` finds nothing, `scan` does | ONVIF off | Enable ONVIF (optional) |
| Two cameras flicker in and out | Duplicate IP (two still on factory default) | Redo Step 5 one camera at a time |
| Grey / smeared frames | Smart codec on, or packet loss | Turn off H.264+/H.265+; check the cable |
| Events show the wrong time | Camera or server clock wrong | `timedatectl`, chrony, camera NTP |
| `ffmpeg not found` in the log | ffmpeg missing | `sudo apt install ffmpeg`, restart the services |
| No recordings, `Could not find tag for codec` | Main stream is MJPEG or an unusual codec | Set it to H.264 |
| `video_url: null` in results | No recording at that time | `recording.enabled`, free space, the camera's log |
| Alerts all `verified: error` | Ollama not running, model not pulled, or too slow | `ollama serve`, `ollama pull qwen3.5:4b`; raise `timeout` or check fewer event types |
| CPU at 100 %, detection lags | Too many cameras or the main stream used for detection | `stream: sub`, lower camera FPS, smaller model; section 41 |
| Office internet breaks when the server is plugged in | A gateway on the camera port, or the PoE switch also cabled to the LAN | Remove the gateway; disconnect the switch from the LAN |

**RTSP paths by vendor** (what `vendor:` expands to; `channel` = 1 for a camera, N on an NVR):

| Vendor | Main stream | Sub stream |
| --- | --- | --- |
| Hikvision | `/Streaming/Channels/101` | `/Streaming/Channels/102` |
| Dahua | `/cam/realmonitor?channel=1&subtype=0` | `/cam/realmonitor?channel=1&subtype=1` |
| Reolink | `/h264Preview_01_main` | `/h264Preview_01_sub` |
| Uniview | `/unicast/c1/s0/live` | `/unicast/c1/s1/live` |
| Axis | `/axis-media/media.amp?camera=1` | (same, set resolution in the camera) |

Full form: `rtsp://user:password@10.50.0.11:554/<path>`. Test from a laptop with VLC or from the
server with `ffprobe -rtsp_transport tcp "rtsp://..."`.

## 40. Testing with a laptop and a PoE switch

For a demo or a bench test before a site: PoE cameras into a PoE switch, the switch into the
laptop's Ethernet port. A plain PoE switch has no DHCP server, so the adapter gets a `169.254.x.x`
address: give it a static IP in the cameras' subnet (admin PowerShell), without a gateway:

```powershell
New-NetIPAddress -InterfaceAlias "Ethernet" -IPAddress 192.168.1.10 -PrefixLength 24
# undo later: Remove-NetIPAddress -InterfaceAlias "Ethernet" -IPAddress 192.168.1.10
```

Then `uv run -m histor.engine.cameras discover`, declare them under `cameras:` in
`configs/laptop.yaml` (passwords in `.env`), `cameras test <name>`, and record from the test UI or
`uv run -m histor.engine.main --source <name> --show`. Streams are read over TCP on a background
thread, stale frames are dropped, a rebooted camera reconnects by itself, and passwords are masked
in all logs and alerts.

## 41. Sizing rule of thumb

| Cameras | Server | Model | Recording disk (1080p, 7 days) |
| --- | --- | --- | --- |
| 1 – 4 | 8-core CPU, 16 GB RAM (32 GB with the chat and vision models) | `yolox_s`, sub stream: ~3 detections a second per camera on a 20-thread laptop CPU | 2 TB |
| 5 – 12 | 16-core CPU, 32 GB RAM | `yolox_s` if the benchmark gives 3+ a second per camera, else `yolox_tiny` | 4 – 6 TB |
| 12+ / heavy use | Benchmark first; a GPU server with `yolox_m` (and the vision model on the GPU) if the CPU can't keep up | `yolox_m` | ~0.3 TB per camera |

Starting points, not guarantees: run the site's cameras for an hour and watch the CPU (`htop`)
before signing off. Alert checking adds seconds of CPU per checked alert, not per frame.

---

# Part 4 — Model research

State as of 6 October 2026 (sections 42–47) and 10 October 2026 (sections 48–49). Licences were
read from each project's GitHub or Hugging Face page on those dates. They change, so re-check
before a model ships to a client. Numbers marked *estimate* are ours and need measuring on the
target server. Sources are at the bottom.

## 42. Short answer

| Need                        | Recommendation                                                                                                                                                                                                                                                             | Why                                                                                                                                                                                                                                             |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Fire & smoke**      | Train our own YOLOX on open data (D-Fire, CC0; FASDD, CC BY 4.0). Run it about once a second next to the everyday detector, alert only when the detection persists, and send the alert to a person. Add a free smoke/fire score on the scene vectors Histor already stores. | Every ready-made fire model we found is AGPL (trained with Ultralytics) or research-only. The open datasets are fine. Single-frame detectors raise too many false alarms; a 2026 study cut them from 52 % to 4 % with multi-frame confirmation. |
| **Weapons**           | Keep "not offered". For a client pilot only: person crop from the main stream → weapon detector → persistence check → local vision model as a veto →**a guard decides**. Data from Open Images (CC BY 4.0) plus staged footage on the client's own cameras.      | 0.1 % false positives at 25 fps is 90 false alarms an hour. Commercial systems still fail in public (a bag of chips led to a student being handcuffed, October 2025). Most CCTV gun datasets are non-commercial.                                |
| **Cheapest win**      | **Done 2026-10-09:** replaced the CLIP weights (B/32, LAION-400M) with OpenCLIP B/32 LAION-2B (MIT).                                                                                                                                                                                             | Same architecture and speed, better on every benchmark we checked: COCO retrieval +3.9, Flickr +5.2, ImageNet +3.6 points.                                                                                                                      |
| **Next most useful**  | A`look` tool: a small local vision-language model (Qwen3.5 2B/4B or Gemma 4 E2B/E4B, both Apache-2.0, vision supported in Ollama) reads the evidence frames when a question needs it.                                                                                    | Covers what CLIP can't (counting, "what was he doing"), double-checks alerts, and costs nothing until asked.                                                                                                                                    |
| **Also worth adding** | Plate reading (fast-plate-ocr, MIT), person-down from pose (RTMPose, Apache-2.0, 3–5 ms per person on CPU), open-vocabulary auto-labelling for objects a customer names (OWLv2 / OmDet-Turbo, Apache-2.0), sound events from camera microphones (YAMNet, Apache-2.0).     | All run on CPU through ONNX and have permissive licences.                                                                                                                                                                                       |

## 43. What a model must pass to enter Histor

1. **Commercial-use licence for code, weights *and* training data**: Apache-2.0, MIT, BSD, CC0, CC BY.
   Not AGPL/GPL, not CC BY-NC, not "research only".
2. **Runs on CPU** through ONNX Runtime at a cost a site server can carry for all its cameras.
3. **Portable to C++**: an ONNX graph plus pre/post-processing in plain numpy/OpenCV.
4. **Fits an existing slot** (section 44): the `Detector` protocol, the CLIP vectors, or a chat tool.

The trap: most free fire and gun models on Hugging Face were trained with Ultralytics, whose position
is that models trained and exported with its AGPL code are AGPL too (unless you buy its enterprise
licence). That rules out nearly every ready-made fire and gun model, including Pyronear's YOLO11
wildfire models.

Web-scraped datasets (even CC0 or CC BY collections) don't clear the copyright of each photo. Training
on them is common practice; keep a short model card per trained model listing every dataset and its
licence, so a client's lawyer can check.

### Excluded (checked 2026-10-06)

| Model / data                                                                                                                             | Licence                                                                                 | Note                                                                  |
| ---------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| Ultralytics YOLOv5/v8/11/26, YOLOv10, YOLOv12, YOLOE, and weights trained with them (most Hugging Face fire/gun models, Pyronear YOLO11) | AGPL-3.0                                                                                | Copyleft reaches the shipped model                                    |
| YOLOv6, YOLOv7, YOLOv9 (original repo), YOLO-World, DOSOD                                                                                | GPL-3.0                                                                                 | Same                                                                  |
| YOLO-NAS weights                                                                                                                         | Non-commercial                                                                          |                                                                       |
| MetaCLIP / MetaCLIP 2                                                                                                                    | CC BY-NC 4.0                                                                            |                                                                       |
| MobileCLIP2, DFN CLIP                                                                                                                    | Apple research licence                                                                  |                                                                       |
| Moondream 3                                                                                                                              | BSL 1.1, no third-party service                                                         | We would be the service                                               |
| Rex-Omni                                                                                                                                 | IDEA licence + Qwen research licence                                                    |                                                                       |
| SAM 3                                                                                                                                    | SAM Licence: forbids use related to "the development or use of guns or illegal weapons" | Can't touch the weapons work; also GPU-sized                          |
| SH17 (PPE), OmniFall (falls, as a whole), RWF-2000 (fights), SAFIRE (fire benchmark)                                                     | Non-commercial                                                                          | Evaluation at most                                                    |
| DETR family (RT-DETR, D-FINE, DEIM, RF-DETR)                                                                                             | Apache-2.0                                                                              | Licence fine, but already ruled out as too heavy for GPU-less servers |

## 44. Where new models plug in

| Slot                                                                                                                      | Today                                                                                                 | What a new model needs                                                                                                                                                                                                                                                                                                                                          |
| ------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Detector** ([detector.py](src/histor/engine/detector.py))                                      | One YOLOX ONNX per pipeline; a custom-trained YOLOX only needs`class_names` in the config           | `Pipeline` holds a single detector ([pipeline.py](src/histor/engine/pipeline.py)). A second model (fire/smoke, weapons, plates) needs a small composite `Detector` that runs each model at its own interval and merges the boxes. One combined COCO + fire model is not worth it: COCO would have to be re-labelled for the new classes. |
| **CLIP vectors**                                                                                                    | Every scene (frame + 2×2 tiles) every 5 s and every tracked object every 2 s are embedded and stored | A linear classifier on stored vectors costs one dot product.                                                                                                                                                                                                                                                                                                    |
| **Hi-res crops**                                                                                                    | Main-stream keyframes (~1/s) give full-resolution crops of tracked objects                            | The right input for small things: guns, plates, faces.                                                                                                                                                                                                                                                                                                          |
| **Chat tools** ([tools.py](src/histor/tools.py), [llm.py](src/histor/llm.py)) | Text only: the model sees JSON results, never pixels                                                  | A`look` tool that runs a *local* vision model and returns text. The cloud model (Claude) then still never receives images.                                                                                                                                                                                                                                  |

Rare, costly events (fire, weapons) should all go through the same chain: cheap signals first,
persistence second, a second opinion third, and a person last.

```mermaid
flowchart LR
  F[Sub-stream frames] --> D[Everyday detector<br/>YOLOX, 80 classes]
  F --> S[Specialist detector<br/>about once a second]
  C[Scene vectors<br/>every 5 s, already stored] --> P[CLIP score]
  S --> T{Persists?<br/>N of M checks}
  P --> T
  T -- no --> X[Stored and searchable,<br/>no alert]
  T -- yes --> V[Local vision model<br/>looks at hi-res keyframes]
  V -- clearly not --> X
  V -- plausible --> H[Alert to a person<br/>snapshot + clip]
  H --> E[Person escalates]
```

## 45. Fire and smoke

### 45.1 What to expect

- **False alarms are the problem, not misses.** A 2026 review of fire detection on surveillance video
  puts precision at about 75 % in the best cases and 60 % on average: detectors find fire but also
  find many things that aren't.
- **Persistence fixes most of it.** Dumitru & Spînu (2026) cut the fire/smoke false-alarm rate on
  video clips from 52 % to 4 % (fire 69 % → 6 %, smoke 22 % → 0 %) with multi-frame confirmation and
  confidence-weighted voting. MULTICAST did the same for guns (section 46).
- **Confusers**: sunsets, clouds, fog, steam (firefighting water vapour is often called smoke), warm
  lamps and streetlights, headlights, emergency lights, reflections.
- **Night**: cameras switch to infrared grayscale and flames saturate, so colour cues vanish. Training
  data must include grayscale/IR images.
- **Data beats architecture**: models trained on one dataset transfer poorly to another (20-year
  dataset review). On the FireAndSmoke dataset, YOLOv5 through YOLOv10 all reach about 0.91 mAP@50 for
  fire. Footage from the client's own cameras matters more than which detector we pick.
- **Vision-language models are weak at fire scenes too**: on SAFIRE (September 2026), ten open models
  of 8–38B parameters averaged 61.9 %. Use them to veto obvious confusers, never to detect alone.
- **Standards**: NFPA 72 accepts video smoke/flame detection only when the whole system is *listed*
  for it (UL 268B covers video image smoke detectors). Histor's fire alerts are an extra pair of
  eyes, not the fire alarm, and must be described that way.

### 45.2 Datasets we can use

| Dataset                              | Size                                                                            | Classes                                                                      | View                              | Licence                                                          | Use                                               |
| ------------------------------------ | ------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | --------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------- |
| **D-Fire**                     | 21,527 images; 14,692 fire + 11,865 smoke boxes; 9,838 images with neither      | fire, smoke (YOLO format)                                                    | Ground level, mostly web images   | CC0 (authors note they hold no copyright over the source images) | **Train**                                   |
| **FASDD**                      | 122,634 images (FASDD_CV ≈ 95k ground level, plus drone and satellite subsets) | flame, smoke                                                                 | Mixed                             | CC BY 4.0                                                        | **Train** (CV subset)                       |
| **Pyro-SDIS** (Pyronear)       | 12,378 images, 12,137 smoke boxes, 1,634 negatives                              | smoke                                                                        | Fixed towers, long-range wildfire | Apache-2.0                                                       | Train, only for outdoor sites with a horizon view |
| Pyronear OpenFire                    | 1k–10k images                                                                  | Wildfire                                                                     | Outdoor                           | Apache-2.0                                                       | Optional                                          |
| FireAndSmoke (Catargiu et al., 2024) | 22k+ images, 93k instances                                                      | fire, smoke,**other** (sunsets, lamps, headlights, emergency lights…) | Ground level                      | CC BY per the paper; the repo has no licence file                | Hard negatives, once the licence is confirmed     |
| FireSense                            | 49 videos                                                                       | fire/smoke vs none                                                           | Ground level                      | CC BY 4.0                                                        | Test (video)                                      |
| MIVIA fire / smoke                   | 31 fire + 149 smoke videos                                                      | fire / smoke vs none                                                         | Ground level                      | Check                                                            | Test (video)                                      |
| MmodalFire (2026)                    | 65 indoor videos with smoke-density, temperature and IR/UV sensor traces        | fire / non-fire                                                              | Indoor                            | Check                                                            | Test, indoor                                      |
| SAFIRE (2026)                        | 83k captioned images, 20 scenario types                                         | Classification, Q&A                                                          | Mixed                             | CC BY 4.0, but the paper limits it to non-commercial research    | Benchmark only                                    |

### 45.3 Models and training

- **No ready-made model we can ship.** Public weights are Ultralytics (AGPL), Darknet/YOLOv5 weights
  from the D-Fire authors' repo (not our format; YOLOv5 is AGPL), or research code.
- **Train YOLOX ourselves** with the official repo (Apache-2.0), starting from COCO weights. The export
  uses the same ONNX layout and decoder Histor already runs, so the only config change is
  `path` plus `class_names: [fire, smoke]`.
  - Start with **YOLOX-Tiny** (416 px, 6.45 GFLOPs). Move to **YOLOX-S** (640 px, 26.8 GFLOPs) if small
    or distant smoke is missed.
  - Data: all of D-Fire + FASDD_CV, Pyro-SDIS only for outdoor sites. Remove near-duplicates across
    datasets (perceptual hash) *before* splitting, because web-scraped sets overlap.
  - Augment for CCTV: grayscale (night IR), H.264/JPEG artefacts, downscaling to sub-stream
    resolution, blur.
  - Expect roughly 78–81 % mAP@0.5 on D-Fire, which is what published lightweight YOLO variants reach
    (YOLOv11-CHBG 78.4 %, YOLOGX 80.9 %; heavier ones up to 89.7 %). mAP is not the acceptance metric
    though (section 45.5).
  - Cost: hours on one rented GPU, not days (*estimate*).
- **Alternative trainer**: LibreYOLO (MIT code, trains YOLOX and YOLOv9, exports ONNX/OpenVINO).
  Friendlier API, but a young project with one main maintainer; the official YOLOX repo is the safer
  default.

### 45.4 Recommended design

1. **Free score from CLIP.** Histor already embeds every scene every 5 s, and `search` can already rank
   scenes for "smoke" (the scene tiles exist precisely for things the detector has no class for).
   - Zero-shot: compare scene vectors with "smoke", "flames", "a fire" against negatives ("fog",
     "steam", "sunset", "car headlights", "a lamp").
   - Then a linear probe: logistic regression on the 512-number vectors of D-Fire/FASDD images,
     embedded with the same CLIP model. Store a smoke/fire score per scene.
   - Use: questions like "any smoke in the warehouse last night?", and one vote in the persistence
     check. Not alarm-grade on its own: on SAFIRE, a CLIP + SVM classifier reached F1 ≈ 0.79 with 30 %
     of the labels.
2. **Specialist detector** (YOLOX-Tiny fire/smoke) every 0.5–1 s instead of every frame, since fire
   develops over seconds. Boxes allow zones ("smoke near the generator") and growth checks. At two runs
   a second it costs about as much as YOLOX-Nano at 10 fps (*estimate* from FLOPs).
3. **Persistence check**, no model: alert only if the detector or the CLIP score fires in N of the
   last M checks (e.g. 5 of 8 over ~8 s), in about the same place. Flames flicker and smoke spreads;
   a static lamp does neither.
4. **Second opinion, then a person.** For a candidate that persists, send 2–3 hi-res keyframes to the
   local vision model: "Is there real fire or smoke? yes/no and one reason". It may veto (a kettle's
   steam, a sunset), never confirm alone. Then alert a person with the snapshot and clip.
5. **Cheaper signals where a site has them.**
   - Cameras with microphones: YAMNet recognises "Smoke detector, smoke alarm" and "Fire alarm". An
     existing alarm going off is a very specific signal and costs almost nothing to hear.
   - Cameras with built-in fire analytics (thermal / bi-spectrum, Hikvision HEOP, Dahua WizMind): take
     their events over ONVIF instead of detecting again.

### 45.5 Acceptance test (per site, before alerts go live)

- **False alarms per camera per day** on at least 7 days of the client's normal footage. Agree the
  target with the client before the pilot.
- **Detection rate and seconds to alert** on positive clips: MIVIA, FireSense and MmodalFire videos,
  plus a controlled smoke test on site if the client allows it.

## 46. Weapons

### 46.1 Why "not offered" is still right

- **Base rates.** Guns are rare, frames are not: a 0.1 % false-positive rate at 25 fps gives 90 false
  alarms an hour on one camera (University of Granada). Guards switch such systems off.
- **Small objects on the sub stream.** A handgun is a handful of pixels in a 640×360 frame, often
  partly hidden by a hand or clothing.
- **Confusers**: phones, wallets, cards, umbrellas, drills, black bags, shadows.
- **Pose is a weak cue**: a February 2026 study notes that people carrying guns tend to walk normally,
  so "arm extended" rules add little.
- **Real deployments still fail in public:**
  - Omnilert, Baltimore County, October 2025: an empty Doritos bag was flagged as a gun; police
    handcuffed and searched a student.
  - ZeroEyes routes every detection through a 24/7 human verification centre before alerting, and
    still had false alarms: a shadow of a student's arm triggered a school lockdown in Texas
    (October 2023); theatre prop guns triggered a lockdown and police response at a university
    (June 2024).
  - The FTC acted against Evolv (November 2024) for overstating what its AI weapon screening detects.
    The order bans unsupported claims and let some school customers cancel their contracts.

Lesson for us: never let a model call the police, and only claim what our own site tests show.

### 46.2 Datasets

| Dataset                                                    | Content                                                                                                                                        | Licence                                               | Use                              |
| ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- | -------------------------------- |
| **Open Images V7**                                   | Boxes for Handgun, Rifle, Shotgun, Weapon, Knife, Kitchen knife, Dagger, Sword, and confusers (phones, umbrellas…). Everyday photos, not CCTV | Annotations CC BY 4.0; images listed as CC BY 2.0     | **Main training source**   |
| OD-WeaponDetection (University of Granada)                 | Pistol and knife detection, plus "Sohas": pistol, knife, smartphone, bill, purse, card (the classic confusers)                                 | Repo: CC BY 4.0; project page says "research purpose" | Train after written confirmation |
| YouTube-GDD                                                | Frames from YouTube videos, with context                                                                                                       | Repo Apache-2.0; the frames belong to the uploaders   | Caution                          |
| US Real-time gun detection in CCTV (University of Seville) | Real university CCTV + Unity synthetic images                                                                                                  | CC BY-NC 4.0; commercial use by agreement             | Test, or ask                     |
| UCLM firearm action dataset (2024)                         | 398 videos: handgun, machine gun, none; PTZ camera + phones, actors                                                                            | CC BY-NC 3.0; commercial use with permission          | Test, or ask                     |
| CCTV-Gun benchmark                                         | Combines Monash Guns, Seville and UCF-Crime                                                                                                    | Each source's terms (mostly research)                 | Test only                        |

### 46.3 If a client asks for a pilot

1. **Where**: only chosen cameras and zones (entrances, cash desks), chosen hours.
2. **Input**: tracked people → hi-res crop from the main-stream keyframe (Histor already builds these
   for search) → detection on the crop, where the gun has many more pixels than in the sub stream.
3. **Detector**: YOLOX-S on person crops; classes handgun, long gun, knife, *plus confuser classes*
   (phone, wallet, umbrella, tool). Confuser classes reduce false positives in the Granada studies.
4. **Persistence** on the same track over several crops. MULTICAST (spatial + temporal confirmation)
   removed 80 % of false alarms compared with a single-frame Faster R-CNN; a two-stage detector with a
   temporal window removed 71 % at 34 fps.
5. **Local vision model as a veto** on the crops ("is this person holding a gun?").
6. **A guard decides**, on a phone, with the crop and the clip. Wording: "possible weapon, please
   check". Histor never escalates by itself.
7. **Data we own**: staged walk-throughs with replica/airsoft guns on the client's own cameras (real
   angles, lenses, IR) + Open Images + synthetic renders if needed.
8. **Acceptance**: false alarms per camera per day on a normal week, and detection rate on staged
   walk-throughs. Marketing quotes only these numbers.

Knives: the everyday COCO detector already has a `knife` class, but it learned kitchens, not CCTV.
Treat knives exactly like guns.

## 47. Other models worth adding

### 47.1 Better search vectors (CLIP) — **built (2026-10-09)**

Now in the code: `export_clip.py` exports ViT-B-32 LAION-2B by default (`--preset laion400m`
keeps the old weights) and writes the model's name into both ONNX files. Every sighting row stores
that name; search compares a query only with vectors of the same model, so after the switch the
old rows are skipped (and logged) until they expire. Checked on the real export: "a dog", "people
playing basketball" and "a footballer kicking a ball" each rank the right sample image first;
~30 ms per crop on this laptop. Tests: `tests/engine/test_clip.py`.


| Model (OpenCLIP name)                                 | Licence    | GFLOPs | ImageNet zero-shot | COCO retrieval\* | Flickr retrieval\* | Work to switch                                                                                                         |
| ----------------------------------------------------- | ---------- | ------ | ------------------ | ---------------- | ------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| ViT-B-32-quickgelu`laion400m_e32` (before 2026-10-09) | MIT        | 14.8   | 62.9               | 44.0             | 70.3               | —                                                                                                                     |
| **ViT-B-32 `laion2b_s34b_b79k`** (**today**)              | MIT        | 14.8   | 66.6               | 47.8             | 75.4               | `ARCH = "ViT-B-32"` + weights URL in [export_clip.py](src/histor/engine/export_clip.py); re-embed |
| ViT-B-32`datacomp_xl_s13b_b90k`                     | MIT        | 14.8   | 69.2               | 45.3             | 70.0               | Same                                                                                                                   |
| ViT-B-32-256`datacomp_s34b_b86k`                    | MIT        | 17.5   | 72.8               | 48.9             | 74.9               | Also 256 px input in`clip_onnx.preprocess` (fixed at 224 today)                                                      |
| ViT-B-16-SigLIP`webli`                              | Apache-2.0 | 46.4   | 76.0               | 56.8             | 81.9               | ~3× CPU per crop; different tokenizer and normalisation                                                               |

\* Retrieval scores from the OpenCLIP benchmark table, comparable within the table.

- **LAION-2B B/32 is a drop-in**: best retrieval at the same cost (~30 ms per crop on this laptop, measured 2026-10-09).
- Bigger jumps (SigLIP / SigLIP 2 B/16, PE-Core B/16, all Apache-2.0) cost about 3× per crop. Worth it
  on GPU sites, or for objects only. SigLIP 2 is multilingual, which matters if clients will ask in
  languages other than English.
- Vectors from two models can't be compared. Tag each stored row with its model, or start a new store
  and let the old rows expire (30 days), or re-embed from the recordings (7 days).
- Keep float32: int8 already reordered results too much (see the development log).

### 47.2 A `look` tool with a local vision-language model — **built (2026-10-09)**

Now in the code: `vision.py` and the `look` tool (`chat.vision` in the config, `qwen3.5:4b` by
default). The chat model passes 1–4 images that earlier results listed (thumbs, scene images, alert
snapshots, plate crops). Any other path is refused, so it can't open arbitrary files. The images
are shrunk to `max_side` (640 px) and sent to the local server; only the answer goes back. If the
vision server is down, the tool says so and the chat carries on. Tests: `tests/test_look.py`,
against a stand-in OpenAI-compatible server. Not yet run against a real Ollama: no Ollama on the
build machine.


**Design**: a new chat tool, `look(evidence, question)`. The chat model passes thumbnails or hi-res
crops it already found; a local vision model answers in text. It runs locally even when the chat model
is Claude, so images never leave the site; only the text answer reaches the cloud model (section 7).

**Why at question time**: captioning every scene every 5 s on every camera is too heavy for a CPU.
At question time only the top 4–8 frames need reading. Per-scene captions (on the roadmap) can come
later, sparsely, with the smallest models.

**Bonus**: Qwen3.5's small models are natively multimodal *and* call tools. `qwen3.5:4b` could replace
today's `qwen3:4b` and look at frames itself. Decide with the planned evaluation questions.

| Model                        | Sizes                             | Licence    | Runs in                                                  | Notes                                                              |
| ---------------------------- | --------------------------------- | ---------- | -------------------------------------------------------- | ------------------------------------------------------------------ |
| **Qwen3.5** (Feb 2026) | 0.8B, 2B, 4B, 9B                  | Apache-2.0 | Ollama (vision, tools, thinking; 1.2–7.6 GB), llama.cpp | Same family as today's chat model                                  |
| **Gemma 4** (2026)     | E2B, E4B (also 12B, 26B-A4B, 31B) | Apache-2.0 | Ollama (vision, tools, audio)                            | Visual token budget 70–1120: a direct speed-vs-detail knob on CPU |
| Qwen3-VL                     | 2B, 4B, 8B                        | Apache-2.0 | Ollama, llama.cpp                                        | Previous generation, good at boxes and grounding                   |
| SmolVLM2                     | 256M, 500M, 2.2B                  | Apache-2.0 | llama.cpp, ONNX                                          | Small enough for sparse per-scene captions                         |
| Florence-2                   | 0.23B, 0.77B                      | MIT        | ONNX / transformers                                      | Captions, phrase grounding and OCR in one small model              |
| Granite Vision               | 2B                                | Apache-2.0 | Ollama (3.2)                                             | Tuned for documents and charts                                     |
| InternVL3.5                  | 1B, 2B                            | Apache-2.0 | transformers                                             |                                                                    |
| Moondream 2                  | ~2B                               | Apache-2.0 | Ollama                                                   | Caption, detect, point (Moondream 3 is excluded)                   |

Speed: on CPU, expect seconds per image for 1–4B models, growing with image size; keep images around
384–640 px (*estimate, measure*). Unsloth's Qwen3.5 GGUF files don't do vision in Ollama (separate
projector file); use Ollama's own `qwen3.5` tags, or llama.cpp's `llama-server` with `--mmproj`.

### 47.3 Number plates — **built (2026-10-09)**

Now in the code: `plates.py` (finder + reader), `plate_index.py` (per camera, on its own thread:
each vehicle about once a second, from the main-stream keyframe when it was matched; reads voted per
vehicle; at most 5 rows per vehicle with a plate crop each), the `plates` table, the `plates`
chat tool and `GET /plates`. Look-alike characters (0/O/D/Q, 1/I/L, 8/B, 5/S, 2/Z, 6/G) and one
wrong character are forgiven, and 3+ characters also find plates containing them. A watchlist
raises `plate` alerts. Models: `download_model plates` (finder = open-image-models YOLOv9-t 384,
trials until the legal check below; reader = fast-plate-ocr CCT-S v2 global, MIT). A YOLOX finder
of our own plugs in with `plates.detector.type: yolox` (**urgent to-do**: train it, see the note
at the top). Checked with the real models on synthetic
plates (finder box within 8 px, reader exact, watchlist event, search). Not yet checked on a real
camera. Tests: `tests/engine/test_plates.py`.


| Part                        | Option                                                                            | Licence                                                                              | Note                                                                                          |
| --------------------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------- |
| Reading                     | **fast-plate-ocr** (small CCT models, ONNX)                                 | MIT, weights included (author's statement, 2026)                                     | Global model with optional region prediction; sub-millisecond per plate on a GPU, tiny on CPU |
| Finding plates              | **Train YOLOX** on Open Images "Vehicle registration plate" + regional data | CC BY 4.0 annotations                                                                | Fits the pipeline; run on vehicle crops                                                       |
| Finding plates (ready-made) | open-image-models YOLOv9-t/s                                                      | MIT code; weights trained with a fork of the GPL-3.0 YOLOv9 repo, on ~11k web images | Legal check first                                                                             |
| Finding plates (ready-made) | OpenCV Zoo LPD-YuNet                                                              | Apache-2.0                                                                           | Trained on Chinese plates only                                                                |
| Other text                  | PaddleOCR PP-OCR                                                                  | Apache-2.0                                                                           | Truck logos, signs, container numbers                                                         |

Gives exact answers to "when did plate 123-ABC come in?", and matches the plates service line.

### 47.4 Pose: person down, hands up — **built (2026-10-08)**

How it works, with figures: section 22.

Now in the code: [pose.py](src/histor/engine/pose.py) (model + facts),
[pose_index.py](src/histor/engine/pose_index.py) (sampling, storage, events), the
`poses` table, `GET /people` and the chat's `people` tool. Facts stored per person: posture
(standing / sitting / crouching / lying), arms (hands above the head), facing, and the colour of
the top and trousers from the torso and leg areas the keypoints outline (11 colour names, no extra
model). Events: `down` (lying 10 s) and `hands_up` (both hands up 3 s).

Licence check of the ready-made weights (2026-10-08):

| Model                                                                                                 | Weights licence       | Training data                                                                                                       | Speed here (laptop CPU, ONNX Runtime) | Use                                                      |
| ----------------------------------------------------------------------------------------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------- | -------------------------------------------------------- |
| **ViTPose-B "simple"** (`usyd-community/vitpose-base-simple`, exported by `export_pose.py`) | Apache-2.0            | COCO keypoints only (backbone pre-trained with MAE on ImageNet-1k, as most backbones are)                           | 100–175 ms per person, 344 MB        | **Default, shippable**                             |
| RTMPose-t (rtmlib / OpenMMLab ONNX,`download_model.py pose_rtmpose_t`)                              | Apache-2.0            | "body7": COCO + AI Challenger, CrowdPose, MPII, sub-JHMDB, Halpe, PoseTrack18; several are research-only or unclear | 10–17 ms per person, 13 MB           | Trials only                                              |
| RTMO / RTMPose COCO-only                                                                              | Apache-2.0            | COCO (+ AI Challenger pre-training for RTMPose)                                                                     | not measured                          | Only as PyTorch checkpoints: would need MMPose to export |
| YOLO-pose (Ultralytics), OpenPose                                                                     | AGPL / non-commercial |                                                                                                                     |                                       | Excluded                                                 |

Both formats load through the same code (heatmaps or SimCC, told apart by the ONNX outputs).
The cost gap is 10×: a COCO-only small model would be the next improvement, by exporting
RTMPose-s COCO from MMPose or fine-tuning one on COCO ourselves.

- **RTMPose** via **rtmlib** (Apache-2.0, ONNX Runtime/OpenVINO). It is top-down, so it runs on the
  person tracks Histor already has. On an Intel i7-11700 CPU: RTMPose-t 3.2 ms and RTMPose-s 4.5 ms per
  person (COCO AP 65.9 / 69.7); RTMPose-m 75.8 AP at 90+ fps. Prefer COCO-only checkpoints: the
  multi-dataset ones mix in sets with their own terms.
- **Person down** = keypoints + box shape + time on the ground, as a rule. Tune on CAUCAFall and EDF
  (CC BY 4.0); OmniFall as a whole is non-commercial, so evaluation only.
- MediaPipe Pose (Apache-2.0, in OpenCV Zoo, with C++ demos) is the alternative.
- PaddleDetection's PP-Human (Apache-2.0) has fall, fight, smoking and phone-call pipelines; useful as
  a reference design, but it needs Paddle → ONNX conversion.

### 47.5 Open-vocabulary detection: deep scan and auto-labelling

| Model               | Licence    | Speed                                                                               |
| ------------------- | ---------- | ----------------------------------------------------------------------------------- |
| OWLv2 base          | Apache-2.0 | ~3 s per 640×480 image on CPU (third-party measurement)                            |
| OmDet-Turbo-T       | Apache-2.0 | ~20 fps in PyTorch fp32, 140 fps with TensorRT on an A100; no CPU numbers published |
| Grounding DINO tiny | Apache-2.0 | Heavy; repo idle since 2024                                                         |
| Florence-2          | MIT        | "Several seconds" per image on CPU                                                  |

Excluded: YOLO-World (GPL-3.0), YOLOE (AGPL-3.0), DOSOD (GPL-3.0).

Two uses, both fine at seconds per image:

1. **Deep scan** (on the roadmap): run on a
   chosen time window of recorded keyframes, as a background job.
2. **Auto-labelling**: when a customer names an object that is not one of the 80 ("forklift",
   "pallet", "ladder"), the open-vocabulary model labels their footage, a person fixes the labels, and
   we train a YOLOX specialist that runs live. This closes the gap named in section 8 ("any object
   a customer names… need a model trained for it").

### 47.6 Sound events — **built (2026-10-09)**

Now in the code: `sound.py`. YAMNet (an Apache-2.0 tf2onnx conversion, pinned to one revision;
`download_model sounds`) runs on the camera's audio (ffmpeg) or a video file's. Seven groups by
default (smoke_alarm, glass_break, gunshot, explosion, scream, car_alarm, siren), each a list of
AudioSet classes and editable. A group fires after 2 frames in a row at ≥ 0.3 (glass and gunshot
0.4), then rests 30 s. From a file, each event is released when the video reaches it, so the
snapshot shows that moment. The audio is never stored. Checked with the real model: a synthetic
3.1 kHz smoke-alarm pattern fires `smoke_alarm` at the right time, a quiet room fires nothing.
Not yet checked on real camera audio (no ffmpeg on the build machine). Tests:
`tests/engine/test_sound.py`.


**YAMNet** (Apache-2.0, MobileNet, 521 AudioSet classes) recognises "Smoke detector, smoke alarm",
"Fire alarm", "Gunshot, gunfire", "Explosion", "Glass" / "Shatter", "Screaming", "Shout", "Car alarm"
and sirens, at negligible CPU cost. Only for cameras with microphones, and only where local law allows
listening: classify in memory and keep only the event label, never the audio.

### 47.7 Following a person across cameras

For the multi-camera timeline on the roadmap: start with the CLIP object vectors Histor already stores,
plus which cameras are next to each other and how long the walk takes. If that's not enough: OpenCV Zoo's
Youtu ReID (Apache-2.0, ONNX, C++ demo), or OSNet (torchreid, MIT code). The usual re-ID training sets
(Market-1501, MSMT17) are released for research, and DukeMTMC was withdrawn by its authors in 2019, so
check the weights' training data. Treat re-ID like faces: client approval first.

### 47.8 Safety gear (PPE), for industrial clients

Train YOLOX on **SHWD** (MIT; 7,581 images, helmeted vs bare heads) and **Hard Hat Workers** (public
domain; some Roboflow variants CC BY 4.0). SH17 is CC BY-NC-SA, so excluded.

### 47.9 Fights and violence

Postpone. The datasets are mostly YouTube or movie clips under research terms (RWF-2000 needs its
lab's approval for commercial use), and video models are heavy on CPU. Meanwhile: pose rules (several
people, fast limb motion, someone down) as candidates, checked by the vision model (now:
alert checking, section 18). Newer options and a plan: section 48.

### 47.10 Rules that need no new model — **built (2026-10-09)**

Now in the code: `tamper.py`, `abandoned.py` and zones with `max_count` (section 8). On *moved*,
the first design (share of edge pixels still in place) failed its test: on busy scenes edges are
everywhere, so a turned camera still looked 60–70 % the same. It now compares 80×45 maps of edge
strength by correlation. The same street 35 s later scores 0.78, darker light 0.79, a 40 px turn
0.40, a different view 0.18; the threshold is 0.5. Tests: `tests/engine/test_rules.py`.


- **Camera tampering**: picture blurred (Laplacian variance), blocked (flat histogram), too dark, or
  moved (compare with a reference frame). Operationally very valuable.
- **Abandoned object**: a bag or suitcase track that stays still for N minutes with no person near it.
- **Crowding / queue**: person count in a zone above N.
- **Wrong way**: already possible with a one-direction line rule.

### 47.11 The everyday detector

Keep YOLOX. Data, not architecture, limits fire and weapons. Permissive alternatives if we ever need
them: NanoDet-Plus (Apache-2.0, also in OpenCV Zoo, very fast on CPU), PP-PicoDet / PP-YOLOE+
(Apache-2.0), DAMO-YOLO (Apache-2.0), and MultimediaTechLab/YOLO (MIT training code for the YOLOv9 and
YOLOv7 architectures).


## 48. Threat detection: update, October 2026

What is new since section 46–47, and whether it fits Histor. Histor already covers most of a
vendor's "threat" list with rules: intrusion and line crossing, loitering (`dwell` on a person),
crowding, abandoned objects, tampering, person down, hands up, and gunshot / glass / scream sounds.

| Candidate | New? | Fits Histor? | Verdict |
| --- | --- | --- | --- |
| **Threat scores on the CLIP vectors already stored** ("a fight", "a person pointing a gun"; the VadCLIP idea of prompting CLIP for anomalies) | The approach is known; using our stored vectors is free | Yes: one dot product per stored vector, no new model | **First.** Generalise the planned smoke score (roadmap #2) into a threat score: searchable ("any fights last night?") and one vote towards an alert, never an alert alone. S |
| **Fights from poses**: our keypoints → a small skeleton model (ST-GCN / PoseC3D in MMAction2, Apache-2.0) or joint-angle rules. 2025–26 papers report ~93–95 % on RWF-2000 | Yes, several 2025–26 papers | Yes: pose already runs on person tracks; skeleton models are tiny on CPU | **Prototype.** The blocker is data: RWF-2000 is non-commercial. Plan: label short clips with the local vision model, a person checks, train; scream / shout sound as a second signal; alert checking as the third. M |
| **Cosmos-Reason2-2B** (NVIDIA) as the vision model for alert checking | Yes: NVIDIA's Public Safety blueprint uses the 8B to verify alerts | Partly: NVIDIA Open Model License (commercial use allowed, but revocable, and ends if its guardrails are bypassed); heavy on CPU | Benchmark against `qwen3.5:4b` (Apache-2.0) in `tag` mode; keep Qwen the default |
| Open-world anomaly models (LaGoVAD, ICLR 2026; Holmes-VAU; LAVAD) | Yes | No: research code, GPU, licences unclear | Skip for now |
| Ready-made fight detectors (YOLOv8 fight models on GitHub, 2026 lightweight CNN papers) | Yes | No: AGPL, or no code / unclear licence | Excluded |
| Weapons | Only a synthetic dataset (Simuletic, CC BY 4.0, 141 images; its "rifles vs umbrellas" set lists two conflicting licences) | Unchanged | Keep "not offered" and the pilot-only plan (46.3) |

Two rules that vendors sell and Histor lacks, both small: **running / sudden motion** (an alert on
the `motion` fact, which is already computed) and **tailgating** (two people through a door line
within N seconds; with the access-control system's badge events it becomes "two people on one
badge"). Order: CLIP threat score, then running and fight candidates from pose plus sound, checked
by the vision model; a skeleton model once labelled clips exist.

## 49. Compared with NVIDIA's Public Safety blueprint

NVIDIA's Public Safety AI Blueprint (VSS 3.0, July 2026) is a reference design with the same shape
as Histor: perception → rules → a vision model that verifies alerts → storage → an agent with tools.
Section 12.1 draws Histor in the blueprint's layout.

| NVIDIA blueprint | Histor |
| --- | --- |
| Cameras → VST / VMS (video storage) | `mainstream.py` records the main stream itself |
| RT-CV: RT-DETR detector + tracking | YOLOX-S + IoU tracker, on CPU |
| Behaviour analytics: **tailgating** only, for now | Areas, lines, crowding, loitering, abandoned bags, tampering, poses, plates, sounds |
| **Alert verification**: Cosmos Reason2 8B | **Alert checking** (section 18): Qwen3.5 4B by default, any local vision model |
| Message bus + Logstash + Elastic ("video analytics storage") | One SQLite file per site |
| Agent: Nemotron Nano 9B v2, tools over **MCP** (video IO, analytics, report generation) | `chat.py` + `tools.py`; the same tools over **MCP** (`mcp_server.py`, section 25.3); `report.py` |
| Embeddings: Cosmos-Embed1 | CLIP B/32 LAION-2B |
| Validated on **H100, L40S or RTX PRO 6000**, 128 GB RAM, 18–32 cores; NGC access by request; Metropolis licences | **CPU only**, ~16–32 GB RAM; permissive licences, no account needed |

Positioning: NVIDIA's is a GPU data-centre reference design for one scenario; Histor covers many
scenarios on a CPU server already on site. Taken from it: alert verification (built, section 18),
tailgating (section 48), and the tools over MCP (built, section 25.3).

---

# Roadmap

| # | Step | Size | Gives |
| - | --- | --- | --- |
| 1 | **Urgent: train our own YOLOX plate finder** on Open Images "Vehicle registration plate" (CC BY 4.0), then `plates.detector.type: yolox` | M | A fully permissive plate pipeline |
| 2 | Alert checking on a real Ollama, then a week in `tag` mode at a pilot site: false alarms held back vs real alerts wrongly held back, per event type | S | The number to quote |
| 3 | Smoke/fire and threat CLIP scores on the scene vectors: zero-shot, then a probe trained on D-Fire/FASDD (45.4, 48) | S | "Any smoke last night?" + a cheap vote for alerts |
| 4 | Composite detector + YOLOX fire/smoke + persistence check + alert checking (45) | M–L | Fire/smoke alerts, ready for a pilot |
| 5 | Person-down on staged falls at a pilot site; a COCO-only small pose model (47.4) | M | Person-down alerts we can promise |
| 6 | Running and tailgating rules; fight candidates from pose + sound (48) | S–M | More of the vendors' threat list |
| 7 | WhatsApp, SMS, siren channels; the chat in the dashboard; logins per person | M | Delivery and access |
| 8 | Evaluation questions on recorded footage to pick the chat model (`qwen3:4b` vs `qwen3.5:4b`…) | S | A measured default |
| 9 | Open-vocabulary auto-labelling → per-customer YOLOX (47.5) | L | "Any object the customer names" |
| 10 | Multi-camera timeline: one visit across cameras (47.7) | M | "Where did they go next?" |
| 11 | Sound events and plates checked on a site's real cameras | S | Real read and false-alarm rates |
| 12 | Weapons pilot with one client: staged data, a guard in the loop (46.3) | L | Only if the false-alarm test passes |
| later | MCP over streamable HTTP with authentication, for MCP clients that can't start a command (stdio over SSH works today, section 25.3) | S | Remote agents without SSH |

S = days, M = 1–3 weeks, L = longer (rough).

**Done:** CLIP LAION-2B (2026-10-09) · `look` tool (2026-10-09) · plates with a trial finder
(2026-10-09) · pose, person-down, hands-up (2026-10-08) · sound events (2026-10-09) · tampering,
abandoned bags, crowding (2026-10-09) · periodic reports · **alert checking (2026-10-10)** ·
**MCP server (2026-10-10)** ·
**dashboard with login (2026-10-10)** ·
**Telegram alerts and bot (2026-10-10)**.

# Development log

**2026-10-10 — Telegram.** `engine/telegram.py` (Bot API client), a `telegram` alert channel
(text by default; snapshot and a clip from the recording optional) and `telegram_bot.py` (ask
Histor from Telegram, allowed chats only, evidence images optional; face search from a photo,
opt-in, without the chat model). 18 tests against a stand-in Bot API server, and a real bot. `engine/status.py`: camera status shared by the dashboard and the bot.

**2026-10-10 — Dashboard and login.** `engine/dashboard.py` + `dashboard.html`: the page at `/`
(cameras, live alerts with verdicts, search, plates), `GET /status`, and a token login in front of
the whole API; the server refuses to listen beyond the machine without one. 6 tests and a browser
check on a demo store.

**2026-10-10 — Alert checking; one documentation.** `engine/verify.py`: the local vision model
checks each alert's snapshot and close-up before sending (YES / NO / UNSURE, veto or tag mode,
never a silent loss); verdicts in the `events` table, the webhook / email message, `/events` and
the chat's `alerts` tool; 17 tests. `mcp_server.py`: the chat's tools for any MCP client over
stdio (official MCP SDK 2.3, `mcp` dependency group); 4 tests plus a real stdio session. Documentation merged into the README and this file, with new
figures made from the real models.

**2026-10-08 / 09 — Part 4 items built.** Body poses (ViTPose-B, facts, `people`, person-down and
hands-up), then CLIP LAION-2B with per-model vectors, the `look` tool, number plates, sound events,
tampering / abandoned / crowding rules, YOLOX-S as the default detector (measured nano / s / m on
CPU), and the merge of the object-detection engine into this package (one package, one `uv`
environment, one test UI).

**2026-09-29 — Search, face search, alerts and API** (then in the separate object-detection
project):

| Feature | What it does | Files |
| --- | --- | --- |
| Event store | Every zone/line event with a JPEG snapshot in one SQLite file per site; old data deleted automatically | `store.py` |
| Webhook and email alerts | JSON to a URL and/or email with the snapshot; background thread with retries | `alerts.py` |
| Text search | "person in a red top last friday": people, vehicles or scenes, grouped per visit | `clip_onnx.py`, `indexer.py`, `query.py`, `search.py` |
| Face search by photo | "Yes: seen 1 time, first on shop at Fri 25 Sep 10:30", or "No match"; off by default | `faces.py`, `search.py` |
| API server | `/events`, `/media`, `/search`, `/search/face`, `/health`; docs at `/docs` | `server.py` |
| CLIP export | OpenCLIP → ONNX once, so servers need no PyTorch | `export_clip.py` |

How it was checked: text search on the 80 s street video (413 index rows) put the right sightings
first for "person with a red top", "person carrying a bag", "white van" and "person in a black
jacket"; the same face flipped and resized scored 0.89, a different person 0.28; the CLIP ONNX port
gives identical tokens and 1.000 cosine to the PyTorch model. Decisions: dates and cameras parsed
by rules, no LLM; CLIP in float32 (int8 was 4× smaller but reordered results too much); faces off
by default.

# Sources

**Threat detection and NVIDIA (section 48–49)**

- NVIDIA Public Safety Blueprint: https://docs.nvidia.com/vss/3.0.0/publicsafety-docs/latest/Introduction.html and prerequisites: https://docs.nvidia.com/vss/3.0.0/publicsafety-docs/latest/Prerequisites.html
- Cosmos-Reason2-2B and its licence: https://huggingface.co/nvidia/Cosmos-Reason2-2B
- Lightweight CNN for violence detection (Sci. Rep. 2026): https://www.nature.com/articles/s41598-026-37743-0
- Embedded real-time violence detection (Sci. Rep. 2026): https://www.nature.com/articles/s41598-026-44939-x
- Short-Window Sliding Learning, LLM auto-labelling: https://arxiv.org/abs/2511.10866
- Human skeletons and change detection for violence detection: https://cs.paperswithcode.com/paper/human-skeletons-and-change-detection-for
- MMAction2 skeleton models (ST-GCN, PoseC3D): https://mmaction2.readthedocs.io/en/0.x/skeleton_models.html
- LaGoVAD (ICLR 2026): https://arxiv.org/abs/2503.13160
- Fine-grained prompting of VLMs for anomaly detection (WACV 2026): https://arxiv.org/abs/2510.02155
- VAD survey, DNN to MLLM: https://arxiv.org/abs/2507.21649
- Simuletic synthetic weapon dataset: https://huggingface.co/datasets/Simuletic/cctv-weapon-dataset
- Milestone AI security plugin bundle: https://www.milestonesys.com/globalassets/marketplace/uploaded-assets/0012000000gnzijaae/aisecurity_en_compressed.pdf


**Fire and smoke**

- D-Fire dataset (licence file: CC0): https://github.com/gaia-solutions-on-demand/DFireDataset
- FASDD: https://essd.copernicus.org/preprints/essd-2023-73/
- Pyro-SDIS: https://huggingface.co/datasets/pyronear/pyro-sdis
- Pyronear OpenFire: https://huggingface.co/datasets/pyronear/openfire
- FireAndSmoke dataset and YOLOv5–v10 comparison: https://pmc.ncbi.nlm.nih.gov/articles/PMC11398105/
- Fire and smoke datasets in 20 years (review): https://arxiv.org/abs/2503.14552
- MIVIA fire detection dataset: https://mivia.unisa.it/datasets/video-analysis-datasets/fire-detection-dataset/
- MmodalFire (Scientific Data, 2026): https://www.nature.com/articles/s41597-026-06810-6
- SAFIRE benchmark (2026): https://arxiv.org/abs/2609.07823
- Temporal event validation, 52 % → 4 % false alarms (Dumitru & Spînu, 2026): https://arxiv.org/abs/2607.03131
- Track-level verification of fire/smoke alarms (STV-FSANet, 2026): https://doi.org/10.3390/s26154970
- Real-time fire detection from video surveillance cameras, where are we now?: https://www.researchgate.net/publication/400740974_Real-time_fire_detection_from_videosurveillance_cameras_where_are_we_now
- YOLOv11-CHBG on D-Fire: https://www.mdpi.com/2571-6255/8/9/338
- SFFA-YOLO: https://doi.org/10.3390/app16031599
- UL 268B (video image smoke detectors): https://www.shopulstandards.com/ProductDetail.aspx?UniqueKey=19730
- NFPA 72 and video image detection: https://www.jensenhughes.com/insights/whats-the-deal-with-video-image-detection
- Hikvision HEOP fire and smoke detection: https://partner.hikvision.com/tpp/Solution/SolutionDetail?Id=3c3a9953-f653-45ad-917e-9fb395e317c0
- YOLOX: https://github.com/Megvii-BaseDetection/YOLOX
- LibreYOLO: https://github.com/LibreYOLO/libreyolo

**Weapons**

- Omnilert incident (CNN): https://www.cnn.com/2025/10/25/us/baltimore-student-chips-ai-gun-detection-hnk
- AI Incident Database #1250: https://incidentdatabase.ai/cite/1250/
- ZeroEyes operations center: https://zeroeyes.com/operations-center
- ZeroEyes false alarms and expert criticism (StateScoop, 2024): https://statescoop.com/zeroeyes-school-safety-ai-firearm-detection-2024/
- FTC action against Evolv: https://www.ftc.gov/news-events/news/press-releases/2024/11/ftc-takes-action-against-evolv-technologies-deceiving-users-about-its-ai-powered-security-screening
- False-positive reduction in handgun detection (90 false alarms/hour): https://link.springer.com/article/10.1007/s00521-020-05365-w
- MULTICAST: https://arxiv.org/abs/2104.11653
- Effective strategies for real-time weapons detection (−71 % false positives): https://doi.org/10.3390/app14188198
- Are handgun detection models ready for camera surveillance? (2026): https://link.springer.com/chapter/10.1007/978-3-032-15993-9_8
- Handgun detection with pose + appearance: https://arxiv.org/abs/2010.13753
- Open Images boxable classes: https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv
- Open Images licence: https://storage.googleapis.com/openimages/web/index.html
- OD-WeaponDetection: https://github.com/ari-dasci/OD-WeaponDetection and https://sci2s.ugr.es/weapons-detection
- YouTube-GDD: https://github.com/UCAS-GYX/YouTube-GDD
- US Real-time gun detection in CCTV dataset: https://github.com/Deepknowledge-US/US-Real-time-gun-detection-in-CCTV-An-open-problem-dataset
- UCLM firearm action dataset: https://pmc.ncbi.nlm.nih.gov/articles/PMC10827673/
- CCTV-Gun benchmark: https://github.com/srikarym/CCTV-Gun
- SAM 3 licence: https://github.com/facebookresearch/sam3/blob/main/LICENSE

**Other models**

- OpenCLIP benchmark results: https://github.com/mlfoundations/open_clip/blob/main/docs/openclip_results.csv
- OpenCLIP B/32 LAION-2B: https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K
- SigLIP 2: https://huggingface.co/google/siglip2-base-patch16-224
- Perception Encoder: https://huggingface.co/facebook/PE-Core-B16-224
- Qwen3.5: https://huggingface.co/Qwen/Qwen3.5-4B and https://ollama.com/library/qwen3.5
- Gemma 4: https://huggingface.co/google/gemma-4-E4B-it and https://ollama.com/library/gemma4
- Qwen3-VL: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
- SmolVLM2: https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct
- Florence-2: https://huggingface.co/microsoft/Florence-2-base
- Moondream 3 licence: https://huggingface.co/moondream/moondream3-preview
- Running Qwen3.5 locally (Unsloth): https://unsloth.ai/docs/models/qwen3.5
- fast-plate-ocr: https://github.com/ankandrew/fast-plate-ocr
- Plate model weight licences (author's answers): https://github.com/ankandrew/fast-alpr/issues/49, https://github.com/ankandrew/open-image-models/issues/20, https://github.com/ankandrew/open-image-models/issues/22
- OpenCV Zoo: https://github.com/opencv/opencv_zoo
- RTMPose: https://arxiv.org/abs/2303.07399
- rtmlib: https://github.com/Tau-J/rtmlib
- ViTPose-B simple (Apache-2.0, COCO): https://huggingface.co/usyd-community/vitpose-base-simple
- ViTPose paper: https://arxiv.org/abs/2204.12484
- OmniFall: https://arxiv.org/abs/2505.19889
- PP-Human: https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/deploy/pipeline/README_en.md
- RWF-2000: https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection
- OWLv2: https://huggingface.co/google/owlv2-base-patch16-ensemble
- OWLv2 CPU timing: https://github.com/kurtvalcorza/owlv2-detection-pipeline
- OmDet-Turbo: https://arxiv.org/abs/2403.06892
- Grounding DINO: https://github.com/IDEA-Research/GroundingDINO
- YOLO-World: https://github.com/AILab-CVC/YOLO-World
- YOLOE: https://github.com/THU-MIG/yoloe
- YAMNet: https://github.com/tensorflow/models/tree/master/research/audioset/yamnet
- SHWD: https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset
- Hard Hat Workers: https://public.roboflow.com/object-detection/hard-hat-workers
- SH17: https://github.com/ahmadmughees/SH17dataset
- NanoDet: https://github.com/RangiLyu/nanodet
- MultimediaTechLab/YOLO: https://github.com/MultimediaTechLab/YOLO
