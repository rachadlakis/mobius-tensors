# Object Detection

General-purpose object detection that runs on CPU-only devices, and on a GPU when there is one.

- **Model:** [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (Apache-2.0, safe for commercial use), exported to ONNX
- **Runtime:** ONNX Runtime (no PyTorch needed at inference)
- **Default:** YOLOX-Nano, pretrained on COCO (80 classes)

## Quick start

```bash
uv sync
uv run src/download_model.py yolox_nano          # -> models/yolox_nano.onnx
uv run src/main.py --source 0 --show             # webcam
uv run src/main.py --source data/img.jpg --save data/out/img.jpg
uv run src/main.py --source video.mp4 --save data/out/video.mp4
uv run src/main.py --source rtsp://camera/stream
uv run src/main.py --source gate --show          # IP camera named in the config
```

## GPU

Every model (detector, CLIP, pose) runs on the GPU when ONNX Runtime can see one, else on the CPU:
`device: auto` in the `model`, `search.clip` and `pose` sections (the default when left out).
Other values: `cpu`, `cuda`, `cuda:1` (second GPU), `directml` (any GPU on Windows), `coreml`.
The log says which device each model got.

On an NVIDIA machine, swap the CPU build of ONNX Runtime for the GPU one. CUDA and cuDNN come as
wheels, so only the NVIDIA driver is needed (CUDA 13: driver 580 or newer):

```bash
uv sync --no-group cpu --group gpu                  # add --group server etc. as needed
```

Then keep the same flags on `uv run` (`uv run --no-group cpu --group gpu src/main.py ...`), or
set `UV_NO_SYNC=1` on that machine: a plain `uv run` would reinstall the CPU build on top.

## Test UI

```bash
uv run --group ui src/app.py      # http://127.0.0.1:7860
```

Pick a model, then compare original vs. annotated for an uploaded image, a video, the live webcam,
or an IP camera (**IP camera** tab). The **Faces** tab takes several photos, shows the faces found
(YuNet, with their quality) and the SFace similarity between the faces you pick.
Gradio is in the optional `ui` dependency group, so it is not installed on servers by default.

## Choosing a model

| Model | Input | Use when |
| --- | --- | --- |
| `yolox_nano` | 416 | Weak CPUs, Raspberry Pi-class devices |
| `yolox_tiny` | 416 | CPU, better accuracy than nano |
| `yolox_s` | 640 | Strong CPU or GPU |
| `yolox_m` / `l` / `x` | 640 | GPU, accuracy first |

Change `model.path` in `configs/default.yaml` to switch (input size is read from the model).
Set `model.classes` (e.g. `[person, car]`) to keep only some classes.

## IP cameras (PoE)

PoE cameras plugged into a PoE switch, with the switch wired to the laptop's Ethernet port.

**1. Put the laptop on the cameras' subnet.** A plain PoE switch has no DHCP server, so
the Ethernet adapter gets a `169.254.x.x` address and can't reach the cameras. Give it a
static IP in the cameras' subnet. Factory defaults are often `192.168.1.64` (Hikvision) or
`192.168.1.108` (Dahua). In an admin PowerShell (`Get-NetAdapter` lists the adapter names):

```powershell
New-NetIPAddress -InterfaceAlias "Ethernet" -IPAddress 192.168.1.10 -PrefixLength 24
# undo later: Remove-NetIPAddress -InterfaceAlias "Ethernet" -IPAddress 192.168.1.10
```

Don't set a gateway on this adapter, so internet traffic keeps going over Wi-Fi.
If the switch is on a network with a router (DHCP), skip this step.

**2. Find the cameras.**

```bash
uv run src/cameras.py discover            # ONVIF, probes every network adapter
uv run src/cameras.py scan 192.168.1.0/24 # fallback: hosts with RTSP (port 554) open
```

**3. Declare them** under `cameras:` in the config (see `configs/default.yaml`). Put
passwords in `.env` (git-ignored) and reference them as `${VAR}`:

```yaml
cameras:
  gate:
    vendor: hikvision     # hikvision | dahua | reolink | uniview | axis, or `path:` / `url:`
    host: 192.168.1.64
    username: admin
    password: ${GATE_CAM_PASSWORD}
    stream: sub           # low-res sub stream: enough for detection, far cheaper on CPU
```

**4. Check and run.**

```bash
uv run src/cameras.py test gate --save data/gate.jpg   # resolution, FPS, snapshot
uv run src/main.py --source gate --show
```

Streams are read over TCP on a background thread. Frames are dropped when detection is
slower than the camera, so results stay live instead of lagging behind. If a camera
reboots or the switch loses power, the stream reconnects on its own. Passwords are
masked in all logs and alerts. For several cameras, run one process per camera.

## Zones, lines and alerts

Draw **zones** (polygons) and **lines** (tripwires) on the camera view and get events when
tracked objects interact with them:

| Rule | Events | Typical use |
| --- | --- | --- |
| zone | `enter`, `exit`, `dwell` (inside > `dwell_seconds`) | parking spot occupied > 60 s, restricted area |
| line | `cross` with direction `in` / `out` | counting people / cars through a gate |

```bash
uv run src/main.py --config configs/zones_example.yaml --show   # demo on data/vtest.avi
```

- Draw shapes in the UI's **Zones & lines** tab and paste the generated YAML into a config.
- Points are normalized 0..1 (any resolution) or pixels. A zone without `points` is the whole
  picture: `{name: anywhere, classes: [dog]}` alerts when a dog appears anywhere.
- An object counts as inside when its box's bottom-center (feet / wheels) is inside.
- Events go to the console and `alerts.jsonl` (one JSON object per line), with filters and a cooldown.
  With a `store:` section they are also saved to SQLite with a JPEG snapshot, and can be sent to
  **channels** such as webhook and email (see below).
- Objects are tracked across frames (ByteTrack-style IoU tracker), so each gets a stable `#id`,
  survives short occlusions (`tracking.max_missed`), and `car`/`truck` label flicker is smoothed.
- Use `class_agnostic_nms: true` with zones so one vehicle never yields two boxes.

## Alerts: snapshot and channels

Add to the `alerts:` section of a config (full example: `configs/search_example.yaml`):

```yaml
alerts:
  events: [dwell, cross]              # event types that become alerts (null = all)
  cooldown_seconds: 0                 # global: also skips storing; prefer per-channel cooldowns
  snapshots: true                     # JPEG with the object boxed, saved + attached to emails
  channels:
    - type: webhook
      url: https://example.com/hooks/mobius
      headers: {Authorization: "Bearer ${WEBHOOK_TOKEN}"}
      cooldown_seconds: 30            # at most one message per rule + event type per 30 s
    - type: email
      host: smtp.gmail.com
      port: 587
      username: ${SMTP_USER}
      password: ${SMTP_PASSWORD}       # secrets go in .env, never in the YAML
      sender: alerts@your-company.com
      to: [security@client.com]
      labels: [person]                 # this channel: only people...
      hours: "22:00-06:00"             # ...and only at night
```

Each channel can filter what it receives: `rules` (zone/line names), `events`, `labels`, `hours`
(wraps past midnight) and `cooldown_seconds`. The event store still gets every event, so the
history and Argus's chat see everything even when a channel stays quiet. The older shorthand
`webhook:` / `email:` keys still work (one channel each).

- The webhook receives JSON: `camera, time, iso_time, type, rule, label, track_id, direction /
  dwell_seconds, snapshot` (snapshot = path served by the API at `/media/<path>`).
- Sending happens on a background thread with 2 retries, so a slow mail server never stalls
  detection; failures are logged, not fatal.

### Integrations: what exists and what comes next

| Channel | Status | How | Needs |
| --- | --- | --- | --- |
| Webhook | works | JSON POST to any URL. Also the quick route to Slack, Teams, n8n, Zapier, Make or a client's own system | a URL |
| Email | works | SMTP, snapshot attached | an SMTP account |
| Telegram | planned | Bot API `sendPhoto` to a chat or group: snapshot + alert text as caption | a bot token (from @BotFather) and the chat id |
| WhatsApp | planned | WhatsApp Business Cloud API (Meta). Messages the business starts must use a pre-approved template | a Meta business account, phone number id, access token, an approved template |
| SMS | planned | an SMS provider's HTTP API (e.g. Twilio) | an account and a sender number |
| Siren / relay | planned | HTTP call to an IP relay or a camera's alarm output, to sound an alarm on site | a device with an HTTP API |

Adding a channel is one class in `src/alerts.py`, no other change:

```python
class TelegramSink(Sink):
    def __init__(self, token: str, chat_id: str):
        self.token, self.chat_id = expand_env(token), chat_id

    def send(self, payload, jpeg):
        title, text = alert_message(payload)   # same wording as email
        ...                                    # one HTTPS request to the service

CHANNELS["telegram"] = TelegramSink        # config: channels: [{type: telegram, token: ..., chat_id: ...}]
```

Filters, cooldowns, retries and the background thread come for free. Detectors for new things
(fire and smoke, or any object a customer names) are a model change, not an alert change: once
the model outputs the label, the same zones and channels apply.

## Event store, search and the API

With a `store:` section, every camera process saves to one SQLite file per site:

| What | Table | Used for |
| --- | --- | --- |
| Zone/line events + snapshot | `events` | event history, alerts |
| CLIP vectors of tracked people/vehicles, whole frames and tiles | `sightings` | **text search** |
| Face prints (only where enabled) | `faces` | **search by photo** |
| Body pose facts per person: posture, arms, facing, colours (+ keypoints) | `poses` | **people search**, person-down alerts |

**Setup (once per machine):**

```bash
uv sync --group search --group server
uv run src/download_model.py yolox_nano
uv run --group export src/export_clip.py      # CLIP -> models/clip_*.onnx (~600 MB, needs internet;
                                              # installs PyTorch in the export group only)
uv run src/download_model.py faces            # only if face search will be used
uv run --group export src/export_pose.py      # body poses -> models/pose_vitpose_b.onnx (344 MB)
```

**Run** (one recorder per camera + one API):

```bash
uv run --group search src/main.py --config configs/search_example.yaml --source gate
uv run --group search src/main.py --config configs/search_example.yaml --source data/vtest.avi \
    --camera street --start-time 2026-09-25T14:00          # recorded file: say when it was filmed
uv run --group server --group search src/server.py --config configs/search_example.yaml
```

Open <http://127.0.0.1:8000/docs> to try the API in the browser.

| Endpoint | Example |
| --- | --- |
| `GET /search?q=` | `/search?q=person in a red top last friday` · `/search?q=white van on gate yesterday 6pm to 9pm` |
| `POST /search/face` | form fields `photo` (JPEG/PNG), `when` (e.g. `friday`), `camera` and `face` (which face if the photo has several, 1 = leftmost; optional) → *"Yes, likely: 2 visit(s), first on shop at Fri 25 Sep 10:30 (match 0.58)."* |
| `GET /people` | `?posture=lying&when=last night` · `?upper=red&lower=blue&camera=shop` · `?arms=up`: people by body pose and clothing colour (needs `pose`) |
| `GET /events` | `?camera=gate&type=cross&since=2026-09-25T00:00&limit=50` (also `type=down`, `type=hands_up` from poses) |
| `GET /media/{path}` | snapshots and thumbnails |
| `GET /recordings/{path}` | main-stream MP4s; events and search results carry a `video_url` (`...mp4#t=37`) that opens the moment, 3 s early |
| `GET /health`, `GET /cameras`, `POST /maintenance/purge` | status, camera list, apply retention now |

How search works:

- **While recording** (background thread, never slows detection on live cameras): each tracked
  person/vehicle is embedded with CLIP when it appears and then at most every
  `object_interval_seconds`; every `scene_interval_seconds` the whole frame plus a 2x2 grid of
  tiles is embedded too, so objects the detector does not know (a box, an open gate) can
  still be found.
- **At query time**: the sentence is split into *what / when / which camera* by simple rules
  (`src/query.py`: today, yesterday, last friday, 2026-09-25, 6pm to 9pm, after 10pm, this
  morning, last 2 hours, camera names). No LLM needed. *What* is embedded with CLIP's text
  model and compared with the stored vectors; matches of the same tracked object are merged
  into one result ("visit").
- `search.classes` picks what gets embedded as objects. Default: people, vehicles, bags. Not all
  80 COCO classes: most never matter on CCTV and add false hits and CPU cost; the scene tiles
  cover the rest.
- Scores are relative (CLIP similarity ~0.2-0.4): results are ranked, not yes/no.
- Tiny details (earbuds, rings) are not searchable at CCTV resolution; faces need roughly 32+
  pixels, which usually means the camera's main stream (below) or a camera placed at the door.

### Main stream: recording and hi-res crops

Detection runs on the cheap sub stream, but a 640x360 crop of a person 15 m away is a blur to
CLIP. With a `recording:` section (see `configs/search_example.yaml`), each live camera process
also opens the camera's **main stream** with one ffmpeg process (`src/mainstream.py`):

- **Recording**: packets are copied to one MP4 per minute (`-c copy`: no decode, no re-encode,
  almost no CPU) under `recording.dir/<camera>/`. Fragmented MP4, so the file being written plays
  and survives a power cut. Old files go after `retention_days`, or sooner when the disk has
  less than `min_free_gb` free.
- **Hi-res crops**: only keyframes are decoded (`-skip_frame nokey`). Set the camera's
  main-stream I-frame interval to 1 s (= its FPS) for one per second. Each keyframe is run
  through the detector once and its boxes are paired with the live tracks (IoU), which gives
  exact full-resolution boxes whatever the delay between the two streams. Crops for CLIP, faces
  and scene tiles then come from the keyframe.
- **Nothing is lost between keyframes**: a new track is still embedded right away from the sub
  stream; the first keyframe that catches it adds a hi-res crop, and later crops come from
  keyframes. A far object too small in the sub stream (`min_box_height`) can now be indexed
  from the main stream alone. On a real-time replay of `vtest.avi`, 28 of 38 tracked people got
  hi-res crops; the other 10 kept their sub-stream crop.

Cost: one extra detection per second per camera, plus disk: 1080p at 4 Mbit/s is about 43 GB
per camera per day. Needs ffmpeg >= 5.1 (`sudo apt install ffmpeg`); without it, recording and
hi-res crops are switched off with an error in the log, and detection carries on. Vendor
cameras get their main-stream URL from the preset; a camera given by `url:` needs `main_url:`.

### Face search and privacy

Face prints are **biometric personal data**. They are only computed when `faces.enabled: true`,
optionally limited to some cameras (`faces.cameras: [shop_entrance]`), and deleted after
`face_retention_days` (default 7). Before enabling it at a client site: check local law, get
the client's written approval, and put up camera notices. The API has **no authentication
yet**: keep it on `127.0.0.1` or behind a reverse proxy with login.

Models: YuNet face detector (MIT) + SFace recognizer (Apache-2.0), both from OpenCV Zoo.

How faces are stored (`src/indexer.py`): about once a second per tracked person, the top of
the person crop (from the main-stream keyframe when there is one) goes through YuNet, which
gives a box and 5 landmarks (eyes, nose, mouth corners). The landmarks straighten the face and
SFace turns it into a 128-number face print. The landmarks themselves are not stored.

- **Quality gate**: a face that is too small (`min_face_size`), blurred (`min_sharpness`,
  variance of the Laplacian) or turned away (`max_turn`, from the landmarks) is skipped: those
  match everyone a little and cause most false hits.
- **Best faces only**: each track keeps its `max_per_track` (5) clearest faces; a better one
  replaces the worst. An hour at a desk is 5 rows, not 3600.

How a photo is searched (`src/search.py`):

- The photo goes through the same steps. A photo with several faces needs `face=N` unless one
  face is clearly the subject; a face under 40 px, blurred or turned away is refused with the
  reason instead of giving a misleading answer.
- It is compared with every stored face print in the time range (one numpy multiplication;
  100k prints is ~50 MB and milliseconds, an index only pays off at millions).
- Matches are grouped per visit (camera + track). A visit's score is the mean of its best 3
  frames; **likely** at 0.45 and above, **possible** from SFace's 0.363 threshold. Both were
  set on clear frontal photos, not CCTV: calibrate them on a pilot site's footage.
- The visit is stretched to the whole person track from the search index: someone seen from
  behind at 13:40 who shows their face at 13:44 arrived at 13:40 (`face_seen` says 13:44).
- `checked` gives the face prints stored per camera in the range. No match is not proof the
  person was not there; no prints at all means the question can't be answered.

### Body poses: what people were doing

With a `pose:` section (example in `configs/search_example.yaml`), each tracked person gets
body keypoints about once a second ([src/pose.py](src/pose.py), [src/pose_index.py](src/pose_index.py)).
Top-down: the person boxes the tracker already has are cropped (from the main-stream keyframe
when there is one) and a pose model returns 17 keypoints. Simple rules on the keypoints give the
facts that are stored and searched:

| Fact | Values | How |
| --- | --- | --- |
| `posture` | standing, sitting, crouching, bending, lying, unknown | torso angle; thigh and feet height against torso length (a tilted torso over straight legs is bending, not lying) |
| `arms` | down, one_up, both_up | wrists clearly above the head |
| `facing` | camera, side, away | shoulder order first (a person's left shoulder is on the image's right when facing the camera), then nose and eyes |
| `head` | up, down | nose well below the ears: looking at a phone, a desk |
| `motion` | still, walking, running | from the tracks: box movement since the last sample, in body heights per second |
| `carrying` | backpack, handbag, suitcase, umbrella | from the tracks: one of these detections on the person |
| `upper_colour`, `lower_colour` | black, white, grey, red, orange, yellow, green, blue, purple, pink, brown | most common colour inside the torso (shoulders–hips) and along the legs |

Stored smartly:

- **Smoothed:** a fact counts once it is the majority of the last 3 samples (one bad frame
  doesn't make someone "lying").
- **On change:** a row only when the facts change, plus a heartbeat every 10 s. One hour at a
  desk is ~360 rows instead of 3600, and `/people` still gives the seconds spent in each
  posture. Each row also keeps the 17 keypoints (102 bytes, relative to the box), so a new fact
  can be derived later from the stored keypoints without running the model again
  (`store_keypoints: false` keeps only the facts).
- **Sampled sparsely when nothing moves:** someone standing still is sampled 3× less often.
- **Own thread:** pose never slows detection or the CLIP index; on live cameras a busy worker
  skips samples instead of falling behind.

Events: `down` (rule `person_down`) when a person stays lying for `down_seconds` (10), and
`hands_up` when both hands stay up for `hands_up_seconds` (3), only for the types listed in
`pose.alerts`. They go through the same channels, store and snapshots as area and line events.
Checked with unit tests and the sample street video; validate person-down on staged falls at a
site before relying on it.

| Pose model | How to get it | Speed (laptop CPU) | Use |
| --- | --- | --- | --- |
| ViTPose-B "simple": Apache-2.0, fine-tuned on COCO only | `uv run --group export src/export_pose.py` | 100–175 ms per person | default, shippable |
| RTMPose-t: Apache-2.0, but trained on 7 datasets, some research-only | `uv run src/download_model.py pose_rtmpose_t` | 10–17 ms per person | trials only |

Limits: people under `min_box_height` (80 px) get no pose; colours are lost at night (infrared)
and shift under coloured light; someone half hidden behind a desk is "unknown"; sitting and
crouching can be confused. Keypoints are not a face print, but a walk can be recognised: they
follow the search index's retention.

### Models and licences

| Model | File | Licence | Size |
| --- | --- | --- | --- |
| YOLOX-Nano | `yolox_nano.onnx` | Apache-2.0 | 4 MB |
| CLIP ViT-B/32 (OpenCLIP, LAION-400M) | `clip_image.onnx`, `clip_text.onnx`, `clip_bpe_vocab.txt.gz` | MIT | 350 + 250 MB |
| YuNet | `face_detection_yunet_2023mar.onnx` | MIT | 0.2 MB |
| SFace | `face_recognition_sface_2021dec.onnx` | Apache-2.0 | 37 MB |
| ViTPose-B simple (COCO) | `pose_vitpose_b.onnx` | Apache-2.0 | 344 MB |
| RTMPose-t (trials only: training data not all commercial) | `pose_rtmpose_t.onnx` | Apache-2.0 code | 13 MB |

CLIP is kept in float32: int8 quantization was tested and changed search rankings too much.
Speed on a laptop CPU: ~55 ms per embedded crop.

## Layout

```text
configs/    YAML configs (one per deployment/client)
src/
  detector.py        Detector interface + YOLOX ONNX backend
  onnx_device.py     ONNX Runtime sessions on the GPU if there is one, else CPU (`device:`)
  tracker.py         Multi-object tracker (stable IDs across frames)
  zones.py           Zone / line rules -> enter, exit, dwell, cross events
  alerts.py          Event sinks: console, JSON lines, store, webhook, email (filters, cooldown)
  pipeline.py        detect -> track -> rules -> alerts -> search index (shared by CLI, UI, server)
  store.py           SQLite event store + media files + retention
  indexer.py         Builds the search index while recording (objects, scenes, faces)
  clip_onnx.py       CLIP image/text embeddings with ONNX Runtime (no PyTorch)
  faces.py           Face detection (YuNet) + face prints (SFace)
  pose.py            Body keypoints (ViTPose / RTMPose) + facts: posture, arms, facing, colours
  pose_index.py      Records poses while recording: sampling, smoothing, rows on change, events
  query.py           "red shirt last friday" -> what / when / which camera
  search.py          Text and face search, results grouped into visits
  server.py          FastAPI: events, media, /search, /search/face, /people
  export_clip.py     One-off: OpenCLIP -> ONNX
  export_pose.py     One-off: ViTPose-B -> ONNX
  stream.py          Frame sources (webcam, image, video, URL)
  cameras.py         IP cameras: config, live capture with reconnect, discovery
  mainstream.py      Main stream via ffmpeg: recording to MP4 + keyframes for hi-res crops
  main.py            CLI entry point
  app.py             Gradio test UI
  download_model.py  Fetch pretrained YOLOX ONNX models (and `faces`, `pose_rtmpose_t`)
notebooks/  models_tour.ipynb: see and learn each model (YOLOX, tracker, YuNet, SFace, pose, CLIP)
models/     .onnx weights (git-ignored)
data/       sample inputs/outputs (git-ignored)
training/   (todo) fine-tuning on custom datasets
tests/      uv run pytest
```

## Custom models

The app only depends on the `Detector` interface in `src/detector.py`. To use a custom-trained
YOLOX, export it to ONNX, set `model.path` and list your classes in `model.class_names`.
To add another backend, implement `detect(frame) -> list[Detection]` and register it in `create_detector`.
