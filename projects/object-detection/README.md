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
```

## Choosing a model

| Model | Input | Use when |
| --- | --- | --- |
| `yolox_nano` | 416 | Weak CPUs, Raspberry Pi-class devices |
| `yolox_tiny` | 416 | CPU, better accuracy than nano |
| `yolox_s` | 640 | Strong CPU or GPU |
| `yolox_m` / `l` / `x` | 640 | GPU, accuracy first |

Change `model.path` and `model.input_size` in `configs/default.yaml` to switch.
Set `model.classes` (e.g. `[person, car]`) to keep only some classes.

## Layout

```text
configs/    YAML configs (one per deployment/client)
src/
  detector.py        Detector interface + YOLOX ONNX backend
  stream.py          Frame sources (webcam, image, video, URL)
  main.py            CLI entry point
  download_model.py  Fetch pretrained YOLOX ONNX models
  alerts.py          (todo) alert rules on detections
models/     .onnx weights (git-ignored)
data/       sample inputs/outputs (git-ignored)
training/   (todo) fine-tuning on custom datasets
tests/      uv run pytest
```

## Custom models

The app only depends on the `Detector` interface in `src/detector.py`. To use a custom-trained
YOLOX, export it to ONNX, set `model.path` and list your classes in `model.class_names`.
To add another backend, implement `detect(frame) -> list[Detection]` and register it in `create_detector`.
