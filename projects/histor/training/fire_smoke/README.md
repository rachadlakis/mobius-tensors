# Fire and smoke detector: training

Trains Histor's fire/smoke specialist detector (DOCUMENTATION.md section 45). Every run, with its
data, settings, cost and results, is recorded in [../README.md](../README.md). Everything for it
lives here: datasets, runs, exported models and the `.venv`. Only the scripts and this README are
in git.

## Model

[LibreYOLO](https://github.com/LibreYOLO/libreyolo) (MIT code) training **YOLOv9**, starting from
COCO weights `LibreYOLO9t` / `LibreYOLO9s`. Those weights are the YOLOv9 authors' own MIT release
([MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO), v1.0-alpha), only
re-keyed. Not Ultralytics (AGPL) and not the original YOLOv9 repo (GPL).

Detection, not segmentation: the open fire/smoke datasets only have boxes, smoke has no clean
edge to draw a mask around, and Histor's rules work on boxes.

## Layout

```
fire_smoke/
  sources.yaml        datasets to merge and their class ids
  prepare_data.py     data/raw/* -> data/merged/ (dedupe, clean, split 80/10/10)
  .venv/              PyTorch (CUDA 12.8) + LibreYOLO
  data/downloads/     archives as downloaded
  data/raw/<name>/    each dataset unpacked, untouched
  data/merged/        the training set + dataset.yaml + report.json
  runs/               training runs
  models/             exported ONNX for Histor
```

## Datasets

| Name     | Images | Licence   | Source                                                                                                            |
| -------- | ------ | --------- | ----------------------------------------------------------------------------------------------------------------- |
| `dfire`  | 21,527 | CC0 1.0   | [gaiasd/DFireDataset](https://github.com/gaiasd/DFireDataset)                                                     |
| `cqu`    | 11,027 | CC BY 4.0 | [CQU, Figshare 10.25946/28747046](https://acquire.cqu.edu.au/articles/dataset/Annotated_Fire_-Smoke_Image_Dataset_for_fire_detection_Using_YOLO_/28747046) |
| `indoor` | 5,000  | CC BY 4.0 | [Zenodo 15826133](https://zenodo.org/records/15826133)                                                            |
| `pyro`   | 12,378 (6,000 used) | Apache-2.0 | [pyronear/pyro-sdis](https://huggingface.co/datasets/pyronear/pyro-sdis) v2.0, added 2026-10-15                    |

`pyro` is long-range wildfire smoke seen from fixed towers, plus 1,634 frames the Pyronear system
wrongly flagged (clouds, fog, glare), which make good hard negatives. It fills the weak spot of
the first model (small, distant smoke). Options in `sources.yaml`: `fixed_cameras` (the same view
for months, so the hash would call every frame a copy: split by camera instead), `cctv: false`
(already camera frames; downscaling would erase the smoke) and `max_images` (Histor is indoor
first). `unpack_pyro.py` turns its parquet files into images and labels, one folder per camera.
The other sources keep exactly the same split as before.

CC BY means the model card must credit the dataset authors. Add a dataset only if its licence
allows commercial use (DOCUMENTATION.md section 43) and record it here and in `sources.yaml`.

## Steps

```bash
# 1. environment (done once)
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/Scripts/python.exe "libreyolo[onnx]" onnxruntime

# 2. unpack each dataset into data/raw/<name>/, then build the merged set
.venv/Scripts/python.exe -I prepare_data.py --check
.venv/Scripts/python.exe -I prepare_data.py

# 3. train (best checkpoint is exported to models/fire_smoke_<run>.onnx)
.venv/Scripts/python.exe -I train.py
```

Score a finished run on the held-out test split with
`.venv/Scripts/python.exe -I evaluate.py runs/<name>/weights/best.pt` (writes `runs/<name>/test_metrics.json`).

Smoke test an exported ONNX the way Histor runs it (onnxruntime on CPU, no torch):
`.venv/Scripts/python.exe -I smoke_test.py models/<name>.onnx` scores each test image (does a fire
picture raise fire, does a clean one stay quiet) at several thresholds, times CPU inference and
writes `runs/smoke_<name>.json` plus a contact sheet. Its decoder matches LibreYOLO's own to 0.05 px,
so it is the reference for the YOLOv9 decoder in `engine/detector.py`. `--images <dir>` runs it
on any folder of pictures.

On RunPod (Linux): `bash fetch_data.sh` downloads and unpacks the three datasets on the pod (faster
than uploading), then `python -m venv --system-site-packages /root/venv` (keeps the image's
PyTorch) and install LibreYOLO there. YOLOv9-S, batch 32, 12 workers on an RTX PRO 4500 Blackwell:
~3 min an epoch, 60 epochs in 3 h for about $2.20 (October 2026).

On the pod, work happens on the container disk (`/root/fire_smoke`, fast, wiped when the pod
stops). `/workspace` is the network volume and survives stopping and deleting the pod. During a
run, a small loop (`sync.sh` on the pod) copies the run folder, the logs, `report.json` and
`models/` to `/workspace/fire_smoke/` every 10 minutes and once more when training ends. A second
one (`finish.sh`) waits for the end, runs `evaluate.py` and `smoke_test.py`, copies the results to
`/workspace`, writes `/workspace/fire_smoke/FINISHED` and stops the pod (`runpodctl stop pod
$RUNPOD_POD_ID`), so no GPU hours are paid after training. **Lesson from 2026-10-15:** that stop
failed. A script started with `nohup` from SSH does not get RunPod's `RUNPOD_API_KEY` /
`RUNPOD_POD_ID` (they live in `/etc/rp_environment`, which only interactive shells load), so
start such scripts with `source /etc/rp_environment`, and still check the pod's status yourself.

Live view: `monitor.py` serves a page on http://localhost:8765 with every run's curves (laptop runs
from `runs/`, pod runs over the pod's direct SSH, refreshed each minute) and the pod's cost so far.
In the Claude app it is the `training-monitor` entry in `.claude/launch.json`; update `--pod` and
`--pod-start` there for a new pod.

Models: `hf_models.py` uploads `models/` (every `.onnx`, `.pt`, test metrics, `MODELS.md`, and
`hf_model_card.md` as the card) to the private
[Rachad/histor-fire-smoke-models](https://huggingface.co/Rachad/histor-fire-smoke-models). After
a new upload, update `FIRE_REVISION` and `FIRE_MODELS` in `src/histor/engine/download_model.py`.

Dataset backup: `hf_upload.py <user>/<repo>` packs `data/merged/` into one tar per split and
pushes it to a private Hugging Face dataset; `--get` brings it back on a new machine. The pod's
`huggingface_hub` looks for its token under `$XDG_CACHE_HOME` (`/workspace/.cache`), so pass
`HF_TOKEN` explicitly.

Scripts must run from a file (not `python -c`): on Windows the data-loader workers re-import the
main module.

Speed on the laptop (RTX 4050 6 GB, YOLOv9-T, 640 px, batch 16, 2 workers): about 35 images/s,
so ~12 minutes an epoch for ~22k training images. Measured with LibreYOLO's `profile=True`.
- The YOLOv9 PGI auxiliary branch is off (`aux_weight=0`): with it on, training ran at 11 img/s.
- "CUDA out of memory" or "paging file is too small" with free VRAM means Windows ran out of
  *virtual* memory: each worker process holds ~2 GB, and GPU memory needs backing too. Close big
  programs (an Ollama model, a crashed earlier run still in Task Manager) or use fewer workers.

The exported ONNX is not the YOLOX layout Histor decodes today (one `output` tensor, top-left
letterbox, class names in the ONNX metadata), so `engine/detector.py` needs a YOLOv9 decoder
before the model plugs in.

Test video for Histor's fire alerts: `.venv/Scripts/python.exe -I make_test_video.py` builds
`../../data/test_videos/fire_loop.mp4` (39 s, held-out test pictures). Check it with
`uv run pytest tests/smoke -m smoke -k fire`.
