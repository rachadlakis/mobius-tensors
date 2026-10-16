---
license: other
license_name: mobius-tensors-internal
tags: [fire, smoke, object-detection, yolov9, onnx]
---

# Histor fire and smoke detectors (private)

Mobius Tensors' own fire/smoke detectors for Histor. YOLOv9 trained with
[LibreYOLO](https://github.com/LibreYOLO/libreyolo) (MIT) from the YOLOv9 authors' MIT COCO
weights ([MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO)). Classes:
`0 = fire`, `1 = smoke`. Keep this repository private.

## Which file to use

| File | Use | Test mAP50 / mAP50-95 (v2 split) | CPU, 4 threads |
| --- | --- | --- | --- |
| `fire_smoke_yolov9s_640_e100_runpod_2026-10-15.onnx` | **Default.** Best on every test slice | 0.693 / 0.385 | ~79 ms |
| `fire_smoke_yolov9t_640_e100_runpod_2026-10-15.onnx` | Weak CPUs, many cameras | 0.671 / 0.369 | ~26 ms |

Older models (`e60`, `e63stop`, `e15`) are kept for comparison only. Each `.onnx` (640×640, for
Histor) has a `.pt` checkpoint next to it (for evaluation or fine-tuning) and a
`.test_metrics.json`. `MODELS.md` lists them all.

**Per image on the old-source test images** (D-Fire, CQU, Indoor; 2,332 images), threshold 0.25,
default model:
- **Pictures with fire:** 92 % raise a fire alarm.
- **Pictures with smoke:** 85 % raise a smoke alarm.
- **Clean pictures:** 96 % stay quiet.

Full comparison, method and caveats: `projects/histor/training/README.md` in the Mobius Tensors
repository.

## ONNX format

- **Input:** `images`, (1, 3, 640, 640), RGB scaled to 0–1, letterboxed to the top left with
  grey 114.
- **Output:** `output`, (1, 6, 8400): x1, y1, x2, y2 in input pixels, then the fire and smoke scores
  (sigmoid already applied).
- **After the model:** non-maximum suppression per class.
- **Metadata:** class names in `names`.

## Limits

- **Missed on purpose-built tests:**
  - fires that fill the whole frame (no object to box);
  - very small flames (a lighter);
  - distant smoke on the horizon (about 70 % caught).
- **False alarms:**
  - fog;
  - red lamps;
  - contrails;
  - sunlit clouds.

  Histor alerts only on detections that persist over several checks.
- **Not a certified fire detector.** It adds early warning to certified smoke and heat detectors
  (EN 54 / UL 268); it does not replace them.

## Training data and credits

| Dataset | Licence |
| --- | --- |
| [D-Fire](https://github.com/gaiasd/DFireDataset) (Venâncio et al.) | CC0 1.0 |
| [Annotated Fire-Smoke Image Dataset](https://doi.org/10.25946/28747046) (CQU) | CC BY 4.0 |
| [Indoor Fire Smoke](https://zenodo.org/records/15826133) | CC BY 4.0 |
| [Pyro-SDIS v2.0](https://huggingface.co/datasets/pyronear/pyro-sdis) (Pyronear and SDIS partners), v2 models only | Apache-2.0 |

The training set as used is the private dataset `Rachad/histor-fire-smoke`.
