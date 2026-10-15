"""Train the fire/smoke detector on data/merged/ and export it to ONNX for Histor.

    .venv/Scripts/python.exe -I train.py                       # YOLOv9-T, 640 px
    .venv/Scripts/python.exe -I train.py --size s --batch 8    # YOLOv9-S if small smoke is missed
    .venv/Scripts/python.exe -I train.py --resume runs/<name>/weights/last.pt
    .venv/Scripts/python.exe -I train.py --export runs/<name>/weights/best.pt   # after a Ctrl+C
    .venv/Scripts/python.exe -I train.py --resume runs/<name>/weights/last.pt --pause 120   # cool down

Runs land in runs/<name>/; the best checkpoint is exported to models/fire_smoke_<name>.onnx.
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, default=HERE / "data" / "merged" / "dataset.yaml")
    ap.add_argument("--size", default="t", choices=["t", "s", "m", "c"], help="YOLOv9 size")
    ap.add_argument("--epochs", type=int, default=60, help="~12 min each on the laptop GPU")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16, help="-1 = largest that fits the GPU")
    # Each Windows worker process loads the CUDA libraries (~2 GB of virtual memory each).
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lr0", type=float, default=0.01, help="SGD start learning rate")
    # Off by default: the last 15 epochs (no mosaic, lowest learning rate) give the best model, and
    # validation can sit flat for 30+ epochs before them (run 3 stopped at 63 of 100 that way).
    ap.add_argument("--patience", type=int, default=0, help="stop after N epochs without gain (0 = never)")
    ap.add_argument("--name", default=None, help="run name (default yolo9<size>_<imgsz>)")
    ap.add_argument("--project", type=Path, default=HERE / "runs")
    ap.add_argument("--resume", type=Path, default=None, help="checkpoint to continue from")
    ap.add_argument("--pause", type=int, default=0, help="seconds to rest after each epoch (laptop heat)")
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--export", type=Path, default=None, help="only export this checkpoint")
    args = ap.parse_args()

    from libreyolo import LibreYOLO

    name = args.name or f"yolo9{args.size}_{args.imgsz}"
    if args.export:
        export(LibreYOLO, args.export, args.imgsz, args.export.parent.parent.name)
        return
    callbacks = [Pause(args.pause)] if args.pause else None
    if args.resume:
        model = LibreYOLO(str(args.resume))
        # Saved settings come back with the checkpoint; patience is overridden so a run that
        # stopped early can be continued to its planned last epoch.
        results = model.train(data=str(args.data), resume=True, patience=args.patience, callbacks=callbacks)
    else:
        # COCO weights from the YOLOv9 authors' MIT release; the 80-class head is rebuilt for 2.
        model = LibreYOLO(f"LibreYOLO9{args.size}.pt")
        results = model.train(
            data=str(args.data),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            lr0=args.lr0,
            workers=args.workers,
            patience=args.patience,
            eval_interval=1,  # validate every epoch (default 10), so best.pt and patience are exact
            # PGI auxiliary branch off: training-only, helps deep models most (YOLOv9 paper), and
            # on the 6 GB RTX 4050 it made training 3x slower (11 vs 35 img/s at batch 16).
            aux_weight=0.0,
            save_period=0,  # only best.pt and last.pt (last is written every epoch); disk is tight
            project=str(args.project),
            name=name,
            exist_ok=True,
            # Fire and smoke can be anywhere in the frame and any way round except upside down.
            flip_prob=0.5,
            mixup_prob=0.1,
            callbacks=callbacks,
        )
    print({k: v for k, v in results.items() if isinstance(v, (int, float, str))})

    if args.no_export:
        return
    best = Path(results.get("best_checkpoint") or args.project / name / "weights" / "best.pt")
    export(LibreYOLO, best, args.imgsz, best.parent.parent.name)


class Pause:
    """Rest the GPU after every epoch (checkpoint already saved), so a laptop can cool down."""

    def __init__(self, seconds: int) -> None:
        self.seconds = seconds

    def on_train_epoch_end(self, event) -> None:
        print(f"epoch {event.epoch}/{event.total_epochs} done, pausing {self.seconds} s", flush=True)
        time.sleep(self.seconds)


def export(LibreYOLO, checkpoint: Path, imgsz: int, name: str) -> None:
    onnx = LibreYOLO(str(checkpoint)).export(format="onnx", imgsz=imgsz)
    out = HERE / "models" / f"fire_smoke_{name}.onnx"
    out.parent.mkdir(exist_ok=True)
    shutil.copy2(onnx, out)
    print(f"exported {out}")


if __name__ == "__main__":
    main()
