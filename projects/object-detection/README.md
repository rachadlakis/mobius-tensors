# Object Detection

General-purpose object detection that runs on CPU-only devices.

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

## Test UI

```bash
uv run --group ui src/app.py      # http://127.0.0.1:7860
```

Pick a model, then compare original vs. annotated for an uploaded image, a video, the live webcam,
or an IP camera (**IP camera** tab).
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
- Points are normalized 0..1 (any resolution) or pixels.
- An object counts as inside when its box's bottom-center (feet / wheels) is inside.
- Events go to the console and `alerts.jsonl` (one JSON object per line), with filters and a cooldown.
  With a `store:` section they are also saved to SQLite with a JPEG snapshot, and can be sent by
  **webhook** and **email** (see below).
- Objects are tracked across frames (ByteTrack-style IoU tracker), so each gets a stable `#id`,
  survives short occlusions (`tracking.max_missed`), and `car`/`truck` label flicker is smoothed.
- Use `class_agnostic_nms: true` with zones so one vehicle never yields two boxes.

## Alerts: snapshot, webhook, email

Add to the `alerts:` section of a config (full example: `configs/search_example.yaml`):

```yaml
alerts:
  events: [dwell, cross]
  cooldown_seconds: 10
  snapshots: true                     # JPEG with the object boxed, saved + attached to emails
  webhook:
    url: https://example.com/hooks/mobius
    headers: {Authorization: "Bearer ${WEBHOOK_TOKEN}"}
  email:
    host: smtp.gmail.com
    port: 587
    username: ${SMTP_USER}
    password: ${SMTP_PASSWORD}         # secrets go in .env, never in the YAML
    sender: alerts@your-company.com
    to: [security@client.com]
```

- The webhook receives JSON: `camera, time, iso_time, type, rule, label, track_id, direction /
  dwell_seconds, snapshot` (snapshot = path served by the API at `/media/<path>`).
- Sending happens on a background thread with 2 retries, so a slow mail server never stalls
  detection; failures are logged, not fatal.

## Event store, search and the API

With a `store:` section, every camera process saves to one SQLite file per site:

| What | Table | Used for |
| --- | --- | --- |
| Zone/line events + snapshot | `events` | event history, alerts |
| CLIP vectors of tracked people/vehicles, whole frames and tiles | `sightings` | **text search** |
| Face prints (only where enabled) | `faces` | **search by photo** |

**Setup (once per machine):**

```bash
uv sync --group search --group server
uv run src/download_model.py yolox_nano
uv run --group export src/export_clip.py      # CLIP -> models/clip_*.onnx (~600 MB, needs internet;
                                              # installs PyTorch in the export group only)
uv run src/download_model.py faces            # only if face search will be used
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
| `POST /search/face` | form fields `photo` (JPEG/PNG), `when` (e.g. `friday`), `camera` (optional) → *"Yes: seen 2 time(s), first on shop at Fri 25 Sep 10:30."* |
| `GET /events` | `?camera=gate&type=cross&since=2026-09-25T00:00&limit=50` |
| `GET /media/{path}` | snapshots and thumbnails |
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
  pixels, which usually means the camera's main stream or a camera placed at the door.

### Face search and privacy

Face prints are **biometric personal data**. They are only computed when `faces.enabled: true`,
optionally limited to some cameras (`faces.cameras: [shop_entrance]`), and deleted after
`face_retention_days` (default 7). Before enabling it at a client site: check local law, get
the client's written approval, and put up camera notices. The API has **no authentication
yet**: keep it on `127.0.0.1` or behind a reverse proxy with login.

Models: YuNet face detector (MIT) + SFace recognizer (Apache-2.0), both from OpenCV Zoo;
two photos of the same person score above ~0.36 cosine similarity (`threshold` form field).

### Models and licences

| Model | File | Licence | Size |
| --- | --- | --- | --- |
| YOLOX-Nano | `yolox_nano.onnx` | Apache-2.0 | 4 MB |
| CLIP ViT-B/32 (OpenCLIP, LAION-400M) | `clip_image.onnx`, `clip_text.onnx`, `clip_bpe_vocab.txt.gz` | MIT | 350 + 250 MB |
| YuNet | `face_detection_yunet_2023mar.onnx` | MIT | 0.2 MB |
| SFace | `face_recognition_sface_2021dec.onnx` | Apache-2.0 | 37 MB |

CLIP is kept in float32: int8 quantization was tested and changed search rankings too much.
Speed on a laptop CPU: ~55 ms per embedded crop.

## Layout

```text
configs/    YAML configs (one per deployment/client)
src/
  detector.py        Detector interface + YOLOX ONNX backend
  tracker.py         Multi-object tracker (stable IDs across frames)
  zones.py           Zone / line rules -> enter, exit, dwell, cross events
  alerts.py          Event sinks: console, JSON lines, store, webhook, email (filters, cooldown)
  pipeline.py        detect -> track -> rules -> alerts -> search index (shared by CLI, UI, server)
  store.py           SQLite event store + media files + retention
  indexer.py         Builds the search index while recording (objects, scenes, faces)
  clip_onnx.py       CLIP image/text embeddings with ONNX Runtime (no PyTorch)
  faces.py           Face detection (YuNet) + face prints (SFace)
  query.py           "red shirt last friday" -> what / when / which camera
  search.py          Text and face search, results grouped into visits
  server.py          FastAPI: events, media, /search, /search/face
  export_clip.py     One-off: OpenCLIP -> ONNX
  stream.py          Frame sources (webcam, image, video, URL)
  cameras.py         IP cameras: config, live capture with reconnect, discovery
  main.py            CLI entry point
  app.py             Gradio test UI
  download_model.py  Fetch pretrained YOLOX ONNX models (and `faces`)
models/     .onnx weights (git-ignored)
data/       sample inputs/outputs (git-ignored)
training/   (todo) fine-tuning on custom datasets
tests/      uv run pytest
```

## Custom models

The app only depends on the `Detector` interface in `src/detector.py`. To use a custom-trained
YOLOX, export it to ONNX, set `model.path` and list your classes in `model.class_names`.
To add another backend, implement `detect(frame) -> list[Detection]` and register it in `create_detector`.
