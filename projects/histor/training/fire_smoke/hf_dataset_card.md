---
license: other
license_name: mixed-cc0-cc-by-4.0-apache-2.0
tags: [fire, smoke, object-detection, yolo]
---

# Histor fire and smoke, v2 (private)

The exact training set of Histor's fire/smoke detector (Gradient Labs), run 3 onwards.
Built by `prepare_data.py` in `projects/histor/training/fire_smoke/` from the sources below:
near-duplicates grouped into scenes and split 80/10/10 by scene, 20 % of train images replaced by
a CCTV-like copy, Pyro-SDIS split by camera.

YOLO format, classes `0 = fire`, `1 = smoke`. One tar per split (`train.tar`, `val.tar`,
`test.tar`), each holding `images/<split>/` and `labels/<split>/`. Unpack all three next to
`dataset.yaml` and fix its `path`. `report.json` has the counts.

| Split | Images |
| --- | --- |
| train | 22,826 |
| val | 3,380 |
| test | 2,627 |

## Sources and credits

| Source | Licence | Link |
| --- | --- | --- |
| D-Fire (Venâncio et al.) | CC0 1.0 | https://github.com/gaiasd/DFireDataset |
| Annotated Fire-Smoke Image Dataset (CQU) | CC BY 4.0 | https://doi.org/10.25946/28747046 |
| Indoor Fire Smoke | CC BY 4.0 | https://zenodo.org/records/15826133 |
| Pyro-SDIS v2.0 (Pyronear and SDIS partners), 6,000 of 12,378 images | Apache-2.0 | https://huggingface.co/datasets/pyronear/pyro-sdis |

Images were resized, recompressed or converted to grayscale for the CCTV-like copies, and labels
were remapped to the class ids above. Keep this repository private; the credits above must travel
with any model trained on it.
