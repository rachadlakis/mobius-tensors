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
- Objects are tracked across frames (ByteTrack-style IoU tracker), so each gets a stable `#id`,
  survives short occlusions (`tracking.max_missed`), and `car`/`truck` label flicker is smoothed.
- Use `class_agnostic_nms: true` with zones so one vehicle never yields two boxes.

## Layout

```text
configs/    YAML configs (one per deployment/client)
src/
  detector.py        Detector interface + YOLOX ONNX backend
  tracker.py         Multi-object tracker (stable IDs across frames)
  zones.py           Zone / line rules -> enter, exit, dwell, cross events
  alerts.py          Event sinks: console, JSON lines (filters, cooldown)
  pipeline.py        detect -> track -> rules -> alerts (shared by CLI, UI, server)
  stream.py          Frame sources (webcam, image, video, URL)
  cameras.py         IP cameras: config, live capture with reconnect, discovery
  main.py            CLI entry point
  app.py             Gradio test UI
  download_model.py  Fetch pretrained YOLOX ONNX models
models/     .onnx weights (git-ignored)
data/       sample inputs/outputs (git-ignored)
training/   (todo) fine-tuning on custom datasets
tests/      uv run pytest
```

## Custom models

The app only depends on the `Detector` interface in `src/detector.py`. To use a custom-trained
YOLOX, export it to ONNX, set `model.path` and list your classes in `model.class_names`.
To add another backend, implement `detect(frame) -> list[Detection]` and register it in `create_detector`.
