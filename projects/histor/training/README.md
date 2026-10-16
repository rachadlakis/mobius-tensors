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
- **Models:** in the private Hugging Face model repo
  [Rachad/histor-fire-smoke-models](https://huggingface.co/Rachad/histor-fire-smoke-models),
  with a model card. Upload with `fire_smoke/hf_models.py`. Histor installs the shipped ones with
  `uv run -m histor.engine.download_model fire_smoke`, which needs `HF_TOKEN` or
  `HUGGINGFACE_API_KEY` in `.env`.
- **Training data:** in the private dataset
  [Rachad/histor-fire-smoke](https://huggingface.co/datasets/Rachad/histor-fire-smoke)
  (`fire_smoke/hf_upload.py`).

## Models

| Task | Trained by us? | Model in use | Status |
| --- | --- | --- | --- |
| Fire and smoke | Yes ([below](#fire-and-smoke)) | `fire_smoke_yolov9s_640_e100_runpod_2026-10-15` (Tiny e100 for weak CPUs) | **In Histor** since 2026-10-16: `engine/fire.py`, `fire:` config section, `download_model fire_smoke` |
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
| v2 | v1 + `pyro` (Pyro-SDIS v2.0, Apache-2.0, capped at 6,000) | 22,826 | 3,380 | 2,627 | run 3 |

**v1 details:**
- After merging: 19,678 scenes. Dropped 11,002 copies, 3,632 frames over the per-scene cap, and
  82 frames of unknown orientation.
- 3,523 CCTV-like copies.
- 4,115 negatives (images with no fire or smoke) in train.
- On RunPod, v1 was rebuilt from a D-Fire mirror and came out slightly different: val 2,442 and
  test 2,332 images (vs 2,417 and 2,300 locally).

**Backup:** v2 as trained (one tar per split, plus `dataset.yaml`, `report.json`, `sources.yaml`
and a card crediting every source) is in the private Hugging Face dataset
[Rachad/histor-fire-smoke](https://huggingface.co/datasets/Rachad/histor-fire-smoke). Get it on a
new pod with `python hf_upload.py Rachad/histor-fire-smoke --get` instead of `fetch_data.sh` +
`prepare_data.py`, which takes minutes instead of ~25. The token is `HUGGINGFACE_API_KEY` in
`projects/histor/.env`; on a pod, put it in `HF_TOKEN`. Keep the repo private.

**v2:** adds long-range smoke from fixed towers, the first model's weak spot. It also adds
Pyronear false alarms (clouds, fog, glare) as hard negatives. Pyro is split by camera, skips the
hash and gets no CCTV copies. The v1 sources keep exactly the same split.

v2 details (pod build, 2026-10-15):
- **Pyro:** 6,000 of 12,378 images, random sample, seed 0. That gives 5,896 smoke boxes and
  810 false-alarm frames.
- **Train:** 18,070 positives and 4,756 negatives; 19,049 smoke and 18,201 fire boxes.
- **CCTV-like copies:** 3,647.
- **Pyro boxes are tiny:** plumes on the horizon, often under 10 px at 640 (checked with
  `preview.py`, `runs/preview_pyro.jpg`).
- **Pyro's share of each split is uneven:** whole cameras go to one split, so Pyro takes about 28 %
  of val and 11 % of test.

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
| Epochs | 60 of 60, finished 2026-10-15 08:23 UTC (16.9 h of training, plus the pauses) |
| Speed | ~35 img/s; about 21 min an epoch including the pause |
| Best (val) | epoch 60: mAP50 0.655, mAP50-95 0.366 |
| Model file | `fire_smoke_yolov9t_640_e60_laptop_2026-10-15` (`.onnx` 8 MB) |
| Test | mAP50 0.668, mAP50-95 0.369, recall 0.569, mAP small 0.152, best F1 0.64 at conf 0.21 |
| Snapshot | `fire_smoke_yolov9t_640_e15_laptop_2026-10-13` (epoch 15: val mAP50 0.575, mAP50-95 0.308), superseded |
| Outcome | Only 0.015–0.018 below run 2 (S) on the test split, at about a third of the size. Smoke test still to do. |

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
| Date | 2026-10-15, 06:34 UTC |
| Hardware | RunPod, RTX PRO 4500 Blackwell Server Edition (32 GB), $0.72/h |
| Model | YOLOv9-S, batch 32, 12 workers, 713 iterations an epoch |
| Data | v2 (adds Pyro-SDIS) |
| Epochs | 100 |
| Command | `python train.py --size s --epochs 100 --batch 32 --workers 12 --name yolo9s_640_e100_pyro` |
| Why | more epochs (run 2 was still improving) and long-range smoke data |
| Estimate | ~6.5 h, ~$5 |
| What happened | **Stopped early at epoch 63 of 100** (patience 30: best val mAP50-95 0.355 at epoch 33, then flat). So it never reached the last 15 no-mosaic epochs, which gave runs 1 and 2 their final +0.014 to +0.032. 3.7 h. Fixed: `train.py` now defaults to `--patience 0` and passes it on `--resume`. |
| Test (v2 split, best = epoch 33) | mAP50 0.634, mAP50-95 0.347. Not comparable with runs 1–2: the v2 test includes Pyro's tiny distant smoke. |
| Resume | approved 2026-10-15: after run 4, `train.py --resume .../last.pt --patience 0` continues it from epoch 63 to 100 (~2.2 h, ~$1.70). The epoch-63 results are kept as `*_e63stop_*`. |
| Result | 100 of 100 (resume 15:15–17:27 UTC, 2.2 h). Test (v2): mAP50 0.693, mAP50-95 0.385, mAP small 0.187, best F1 0.67 at conf 0.23. Model `fire_smoke_yolov9s_640_e100_runpod_2026-10-15`. The resume added +0.038 mAP50-95 over the epoch-63 stop. |

#### Run 4: `yolo9t_640_e100_pyro` (RunPod)

| | |
| --- | --- |
| Hardware | the same pod, started by `finish.sh` as soon as run 3 is scored |
| Model | YOLOv9-T, batch 32, 12 workers, `--patience 0` |
| Data | v2 |
| Epochs | 100, hard cap 5.5 h (`timeout`) to stay inside the $11 RunPod balance. A first start with patience 30 was restarted at epoch 3 (2026-10-15 10:30 UTC) so it can't stop early like run 3. |
| Why | Small sites and weak CPUs: T is about a third of S's compute, so it fits several cameras at 1–2 checks a second |
| Time | 100 of 100 epochs, 4.72 h (10:30–15:13 UTC), ~2.8 min an epoch; ~$3.40 |
| Model file | `fire_smoke_yolov9t_640_e100_runpod_2026-10-15` (`.onnx` 8 MB), best = epoch 100 |
| Test (v2 split) | mAP50 0.671, mAP50-95 0.369, recall 0.558, mAP small 0.161, best F1 0.64 at conf 0.23 |
| Outcome | Best model so far for its size; see [Comparison](#comparison-on-one-test-set). |

Run 3 changes two things at once, the epochs and the data. A gain can't be split between them
without another run. That was accepted to save a run.

### Comparison on one test set

All models scored on the **v2 test split** (2,627 images, none seen in training by any model), with
`compare.py` (mAP via LibreYOLO on the laptop GPU) and `smoke_test.py` (onnxruntime, CPU,
per-image alarms at confidence 0.25). Split by source:
- **all:** the whole split.
- **old:** D-Fire, CQU and Indoor (2,332 images). This is exactly run 2's own test set, so it shows
  whether the new data cost anything on ordinary fire and smoke.
- **pyro:** 295 images of distant smoke from towers, including frames of clouds, fog and glare.

Results in `fire_smoke/runs/compare_v2_test.json`.

mAP50-95 (mAP50 in brackets):

| Model | Data | all | old | pyro |
| --- | --- | --- | --- | --- |
| Tiny 60 ep, laptop (run 1) | v1 | 0.347 (0.636) | 0.369 (0.669) | 0.043 (0.126) |
| Small 60 ep (run 2) | v1 | 0.361 (0.653) | **0.384** (0.686) | 0.045 (0.147) |
| Small stopped at 63 (run 3) | v2 | 0.347 (0.634) | 0.355 (0.643) | **0.261** (0.526) |
| **Tiny 100 ep (run 4)** | v2 | **0.369** (0.671) | 0.381 (0.685) | 0.229 (0.484) |
| **Small resumed to 100 (run 3)** | v2 | **0.385** (0.693) | **0.395** (0.702) | **0.268** (0.550) |

Per image at confidence 0.25 (share of pictures that raise the right alarm):

| Model | Old: fire caught | Old: smoke caught | Old: clean quiet | Indoor: smoke caught | Pyro: smoke caught | Pyro: smoke false alarm |
| --- | --- | --- | --- | --- | --- | --- |
| Tiny 60 (run 1) | 90.4 % | 81.2 % | 94.6 % | 71.3 % | 14.0 % | 26.7 % |
| Small 60 (run 2) | 87.6 % | 75.6 % | 96.7 % | 66.7 % | 13.2 % | 17.8 % |
| Small 63 (run 3) | 83.2 % | 70.3 % | 94.0 % | 54.6 % | 66.4 % | 20.0 % |
| Tiny 100 (run 4) | 91.4 % | 83.2 % | 95.1 % | 82.4 % | 73.6 % | 33.3 % |
| **Small 100 (run 3)** | **92.2 %** | **84.9 %** | **96.1 %** | **83.3 %** | 69.6 % | 20.0 % |

Fires caught at the **same false-alarm rate** (a fairer comparison than one fixed threshold,
because models score differently). Per image, old sources (2,332) and Indoor alone (243, so
1 image ≈ 1 point):

| Model | Fire, 2 % false | Fire, 5 % false | Smoke, 2 % false | Smoke, 5 % false | Indoor fire, 5 % | Indoor smoke, 5 % |
| --- | --- | --- | --- | --- | --- | --- |
| Tiny 60 v1 | 80.1 % | 94.1 % | 66.9 % | 81.2 % | 93.3 % | 81.5 % |
| Small 60 v1 | 79.8 % | **96.1 %** | 67.0 % | 83.2 % | 94.6 % | **88.0 %** |
| Tiny 100 v2 | 81.3 % | 94.1 % | **70.3 %** | 83.2 % | 91.3 % | 87.0 % |
| Small 100 v2 | **84.2 %** | 94.3 % | 68.2 % | **85.3 %** | **95.3 %** | 86.1 % |

At equal false alarms the models differ by only a few points. The operating point matters more:
letting false alarms go from 2 % to 5 % per image raises fires caught by about 10 points. Histor's
persistence check is what keeps those extra false alarms from reaching people.

Takeaways:
- **Best model: Small 100 (run 3, resumed).** It is best on every slice, and it raises fire alarms
  on 92 % of fire pictures and smoke alarms on 85 % of smoke pictures. Tiny 100 is the choice for
  weak CPUs, about 1 point behind.
- **Pyro-SDIS works:** distant smoke goes from ~14 % to ~74 % caught.
- **It cost nothing on the old sources:** Tiny 100 matches Small 60 there (0.381 vs 0.384) and
  catches more indoor smoke (82 % vs 67 %).
- **The price is more false alarms on the Pyronear hard negatives** (clouds, fog), 33 % vs 18 %.
  Histor's persistence check (alarm only if it holds over several checks) is meant for exactly that.
- **CPU timings from the scoring pass are not used:** other jobs shared the CPU (the same Tiny network
  measured 29 and 64 ms). Clean benchmark, nothing else running, laptop i7-13700H, onnxruntime,
  640×640, 300 test images:

| Network | 4 threads | 2 threads | Checks a second on 4 threads |
| --- | --- | --- | --- |
| Tiny (8 MB) | 26 ms (p95 30) | 40 ms | ~38 |
| Small (29 MB) | 79 ms (p95 90) | 122 ms | ~13 |

  At 2 checks a second per camera, 4 threads carry ~19 cameras on Tiny or ~6 on Small
  (estimate; a server CPU will differ).

### In Histor (2026-10-16)

`engine/detector.py` has a `yolov9_onnx` backend (the same decoding as `smoke_test.py`, checked
on 25 test images: identical detections, boxes within 0.05 px). `engine/fire.py` runs it about
twice a second and raises a `fire` alert (rule `fire` or `smoke`) when it is seen on 3 of the
last 5 checks. The vision model's second opinion never holds a fire alert back.

**End-to-end test:** `fire_smoke/make_test_video.py` builds a 39 s video from held-out test
pictures: clean, indoor fire, clean, indoor smoke, clean, each picture held 3 s.
`tests/smoke/test_smoke.py::test_fire_alerts_on_a_video` runs it through a real pipeline with
Small and with Tiny:
- **Fire:** alarm at 10 s, 1 s after it came into view.
- **Smoke:** alarm at 25 s, also 1 s after.
- **Clean stretches:** no false alarm.
- **Store:** both events stored with a boxed snapshot.

These are still dataset pictures, not real camera footage; the next test is real fire video
(FireSense, a site's own footage).
