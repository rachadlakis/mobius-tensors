"""Unpack Pyro-SDIS (Hugging Face parquet) into data/raw/pyro/ as YOLO images + labels.

    python unpack_pyro.py data/downloads/pyro/*.parquet

One folder per camera (images/<partner>__<camera>/), so prepare_data.py can keep each camera
whole in one split: the towers never move, so frames of one camera look alike.
Needs pyarrow.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "raw" / "pyro"


def safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(s or "unknown")).strip("-") or "unknown"


def main() -> None:
    n = neg = 0
    for f in sys.argv[1:]:
        table = pq.read_table(f, columns=["image", "annotations", "image_name", "partner", "camera"])
        for row in table.to_pylist():
            cam = f"{safe(row['partner'])}__{safe(row['camera'])}"
            stem = Path(safe(row["image_name"])).stem
            (OUT / "images" / cam).mkdir(parents=True, exist_ok=True)
            (OUT / "labels" / cam).mkdir(parents=True, exist_ok=True)
            (OUT / "images" / cam / f"{stem}.jpg").write_bytes(row["image"]["bytes"])
            ann = (row["annotations"] or "").strip()
            (OUT / "labels" / cam / f"{stem}.txt").write_text(ann + ("\n" if ann else ""), encoding="utf-8")
            n += 1
            neg += not ann
        print(f"{f}: {n} images so far ({neg} without smoke)")


if __name__ == "__main__":
    main()
