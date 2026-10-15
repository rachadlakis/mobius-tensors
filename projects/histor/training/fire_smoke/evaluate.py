"""Score a trained checkpoint on the held-out test split (never seen in training or model choice).

    .venv/Scripts/python.exe -I evaluate.py runs/<name>/weights/best.pt
    .venv/Scripts/python.exe -I evaluate.py runs/<name>/weights/best.pt --split val

Writes runs/<name>/test_metrics.json next to the run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--data", type=Path, default=HERE / "data" / "merged" / "dataset.yaml")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    from libreyolo import LibreYOLO

    run = args.checkpoint.parent.parent
    metrics = LibreYOLO(str(args.checkpoint)).val(
        data=str(args.data),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=str(run),
        name=f"eval_{args.split}",
        exist_ok=True,
    )
    out = run / f"{args.split}_metrics.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
