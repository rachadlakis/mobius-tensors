# Histor: model training

The record of every model Histor trains itself and every training run: data, settings, hardware,
time, cost and results. How-to steps live in each model's folder (for example
[fire_smoke/README.md](fire_smoke/README.md)); this file says what was done and what came out.

Add a run here when it starts and fill in the results when it ends. Failed and stopped runs stay
listed too.

## Rules for every model

- **Licences:** code, starting weights and training data must all allow commercial use (MIT,
  Apache-2.0, BSD, CC0, CC BY). No Ultralytics (AGPL), no original YOLOv7/v9 repos (GPL), no
  "non-commercial" data. CC BY datasets must be credited in the model card. See DOCUMENTATION.md
  section 43.
- **Target:** CPU servers first. Every model is exported to ONNX and smoke-tested with onnxruntime on
  CPU, the way Histor runs it.
- **Test split:** scored once per finished model, never used to pick checkpoints or settings.
- **Names:** `<task>_<arch>_<imgsz>_e<epochs>_<where>_<date>` (date = when training started).
- **In git:** scripts and READMEs only. Data, runs, `.pt` and `.onnx` stay on disk (`.gitignore`).

## Models

| Task | Trained by us? | Model in use | Status |
| --- | --- | --- | --- |
| Fire and smoke | Yes ([below](#fire-and-smoke)) | `fire_smoke_yolov9s_640_e60_runpod_2026-10-14` | Tested, not wired into Histor yet (needs a YOLOv9 decoder in `engine/detector.py`) |
| Objects (80 COCO classes) | No: official YOLOX weights, Apache-2.0 | YOLOX-S / M / Nano | Shipping |
| Pose | No: ViTPose-B (COCO only) | ViTPose-B | Shipping |
| Number plates | Not yet: own YOLOX finder planned (DOCUMENTATION.md, plates section) | open-image-models YOLOv9-t, trials only | Legal check pending |

The vetted list of ready-made models is in DOCUMENTATION.md Part 4.

---

## Fire and smoke

Folder: [fire_smoke/](fire_smoke/). Classes: `0 = fire`, `1 = smoke`. Detection (boxes), not
segmentation.

- **Trainer:** [LibreYOLO](https://github.com/LibreYOLO/libreyolo) 1.6.0, MIT.
- **Architecture:** YOLOv9 (T and S sizes).
- **Starting weights:** `LibreYOLO9t` / `LibreYOLO9s`. These are COCO weights from the YOLOv9
  authors' MIT rewrite ([MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO)),
  re-keyed. Not the GPL original.
- **Why YOLOv9 and not YOLOX like the object detector:** DOCUMENTATION.md section 45 planned YOLOX.
  LibreYOLO was much easier to run on Windows and RunPod, and YOLOv9-S beats YOLOX-S on COCO
  (about 46.8 vs 40.5 AP, published figures). The cost is a second decoder in Histor.
- **Export:** ONNX, 640×640, one `output` tensor `(1, 6, 8400)`: xyxy boxes in input pixels plus two
  sigmoid scores, top-left letterbox, RGB scaled 0–1, class names in the ONNX metadata.
  `smoke_test.py` holds the reference decoder; it matches LibreYOLO to 0.05 px.

### Data versions

All built by `prepare_data.py` from `sources.yaml`:
- Near-duplicates are grouped into scenes with a dHash.
- At most 10 frames are kept per scene.
- Each scene goes whole to one split (80/10/10).
- 20 % of training images are replaced by a CCTV-like copy: grayscale for infrared night, low
  resolution, blur, heavy JPEG.

| Version | Sources | Train | Val | Test | Used by |
| --- | --- | --- | --- | --- | --- |
| v0 (trial) | `indoor` only | 2,159 | 280 | – | run 0 |
| v1 | `dfire` (CC0), `cqu` (CC BY 4.0), `indoor` (CC BY 4.0) | 17,751 | 2,417 | 2,300 | runs 1, 2 |
| v2 | v1 + `pyro` (Pyro-SDIS v2.0, Apache-2.0, capped at 6,000) | *filled in after the merge* | | | run 3 |

**v1 details:**
- After merging: 19,678 scenes. Dropped 11,002 copies, 3,632 frames over the per-scene cap, and
  82 frames of unknown orientation.
- 3,523 CCTV-like copies.
- 4,115 negatives (images with no fire or smoke) in train.
- On RunPod, v1 was rebuilt from a D-Fire mirror and came out slightly different: val 2,442 and
  test 2,332 images (vs 2,417 and 2,300 locally).

**v2:** adds long-range smoke from fixed towers, the first model's weak spot. It also adds 1,634
Pyronear false alarms (clouds, fog, glare) as hard negatives. Pyro is split by camera, skips the
hash and gets no CCTV copies. The v1 sources keep exactly the same split.

### Settings shared by all runs

LibreYOLO defaults unless noted:

| Setting | Value |
| --- | --- |
| Image size | 640 |
| Optimizer | SGD, lr0 0.01, momentum 0.937, weight decay 0.0005, Nesterov |
| Schedule | linear to 1 % of lr0; 3 warm-up epochs (from lr 0.0001) |
| Augmentation | mosaic 1.0, mixup 0.1, HSV, horizontal flip 0.5, translate 0.1, scale 0.5–1.0 |
| Last 15 epochs | no augmentation (mosaic off). This gives the jump in the curves at epoch N−15. |
| Other | EMA 0.9999, AMP fp16, patience 30, seed 0 |
| YOLOv9 PGI auxiliary branch | off (`aux_weight 0`); with it on, the laptop trained at 11 img/s instead of 35 |

### Runs

#### Run 0: `trial_indoor` (pipeline check)

| | |
| --- | --- |
| Date | 2026-10-13 |
| Hardware | Laptop, RTX 4050 6 GB |
| Model | YOLOv9-T, batch 16, 2 workers |
| Data | v0 (indoor only) |
| Epochs | 3 planned, stopped after 1 (~4 min) |
| Result | val mAP50 0.085, as expected after one epoch |
| Outcome | Pipeline works end to end. Discarded. |

#### Run 1: `yolo9t_640` (laptop)

| | |
| --- | --- |
| Date | started 2026-10-13, resumed 2026-10-14 15:27 UTC |
| Hardware | Laptop, RTX 4050 6 GB, 120 s cooling pause after each epoch |
| Model | YOLOv9-T (~8 MB ONNX), batch 16, 2 workers |
| Data | v1 |
| Epochs | 60 planned; at epoch 53 on 2026-10-15 |
| Speed | ~35 img/s; about 21 min an epoch including the pause |
| Best so far (val) | epoch 53: mAP50 0.638, mAP50-95 0.354, recall 0.583 |
| Snapshot | `fire_smoke_yolov9t_640_e15_laptop_2026-10-13` (epoch 15: val mAP50 0.575, mAP50-95 0.308) |
| Outcome | Running. To do when it ends: score on the test split, smoke test, export. |

#### Run 2: `yolo9s_640` (RunPod), the current best

| | |
| --- | --- |
| Model file | `fire_smoke_yolov9s_640_e60_runpod_2026-10-14` (`.onnx` 29 MB, `.pt` 88 MB) |
| Date | 2026-10-14, 12:05 UTC |
| Hardware | RunPod, RTX PRO 4500 Blackwell |
| Model | YOLOv9-S, batch 32, 12 workers |
| Data | v1 (pod build) |
| Epochs | 60 of 60, ~2 min 50 s each, 2 h 50 min in total |
| Cost | about $2.20 |
| Best (val) | epoch 60: mAP50 0.677, mAP50-95 0.382, recall 0.584 |

Test split (2,332 images):

| mAP50 | mAP50-95 | mAP75 | Recall | mAP small / medium / large | Best F1 |
| --- | --- | --- | --- | --- | --- |
| 0.686 | 0.384 | 0.363 | 0.580 | 0.159 / 0.270 / 0.464 | 0.653 at conf 0.13 |

Smoke test (`smoke_test.py`, onnxruntime CPU with 4 threads, 2,300 local test images):
- **Speed:** 94 ms a frame for the model (p95 105 ms), 105 ms including decoding.
- **Per image:** does a picture with fire raise fire, and does a clean picture stay quiet?

| Threshold | Fire caught | Smoke caught | Clean, no alarm |
| --- | --- | --- | --- |
| 0.15 | 94.6 % | 87.1 % | 94.0 % |
| 0.25 | 87.8 % | 75.8 % | 96.7 % |
| 0.35 | 76.8 % | 65.5 % | 98.4 % |

**What we learned:**
- mAP and loss were still improving at epoch 60, partly because the schedule ends there.
- Recall levels off at about 0.58 from epoch 20.
- Small, distant smoke is the weak spot (mAP small 0.16).
- **False alarms:** the sun, sunsets, lit clouds and street lamps.
- **Misses:** large hazy forest fires where smoke covers the flames.

#### Run 3: `yolo9s_640_e100_pyro` (RunPod), in progress

| | |
| --- | --- |
| Date | 2026-10-15 |
| Hardware | RunPod, RTX PRO 4500 Blackwell Server Edition, $0.72/h |
| Model | YOLOv9-S, batch 32, 12 workers |
| Data | v2 (adds Pyro-SDIS) |
| Epochs | 100 |
| Why | more epochs (run 2 was still improving) and long-range smoke data |
| Estimate | ~6–7 h, ~$5 |
| Result | *to fill in* |

Run 3 changes two things at once, the epochs and the data. A gain can't be split between them
without another run. That was accepted to save a run.
