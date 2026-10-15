"""Smoke test an exported fire/smoke ONNX the way Histor will run it: onnxruntime on CPU, no torch.

    .venv/Scripts/python.exe -I smoke_test.py models/<name>.onnx               # whole test split
    .venv/Scripts/python.exe -I smoke_test.py models/<name>.onnx --limit 300
    .venv/Scripts/python.exe -I smoke_test.py models/<name>.onnx --images dir  # any pictures, no labels
    .venv/Scripts/python.exe -I smoke_test.py models/<name>.onnx --root data/v2_test/all --tag v2test

Checks the ONNX loads, decodes its output (YOLOv9 layout: one `output` tensor (1, 4+nc, anchors),
xyxy in letterboxed input pixels, sigmoid scores, top-left letterbox), and scores it per image,
which is what an alarm cares about: does a fire picture raise fire, does a clean picture stay quiet.
Writes runs/smoke_<model>.json and a contact sheet runs/smoke_<model>.jpg (green = label,
red = fire, blue = smoke).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COLOURS = {0: (40, 40, 255), 1: (255, 120, 40)}  # BGR: fire red, smoke blue
THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.35, 0.50]


def letterbox(rgb: np.ndarray, size: int) -> tuple[np.ndarray, float]:
    h, w = rgb.shape[:2]
    ratio = min(size / h, size / w)
    nh, nw = max(int(h * ratio), 1), max(int(w * ratio), 1)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[:nh, :nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    blob = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    return np.ascontiguousarray(blob), ratio


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> list[int]:
    order = scores.argsort()[::-1]
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        order = order[1:][inter / (area[i] + area[order[1:]] - inter + 1e-9) <= iou]
    return keep


def decode(out: np.ndarray, ratio: float, w: int, h: int, conf: float, iou: float) -> list[dict]:
    pred = out[0].T  # (anchors, 4+nc)
    boxes, scores = pred[:, :4], pred[:, 4:]
    dets = []
    for cls in range(scores.shape[1]):  # multi-label, NMS per class, like LibreYOLO
        m = scores[:, cls] > conf
        if not m.any():
            continue
        b, s = boxes[m] / ratio, scores[m, cls]
        b[:, [0, 2]] = b[:, [0, 2]].clip(0, w)
        b[:, [1, 3]] = b[:, [1, 3]].clip(0, h)
        for i in nms(b, s, iou):
            dets.append({"cls": cls, "score": float(s[i]), "box": [round(float(v), 1) for v in b[i]]})
    return sorted(dets, key=lambda d: -d["score"])


def read_gt(img: Path, w: int, h: int) -> list[tuple[int, list[float]]]:
    lbl = img.parent.parent.parent / "labels" / img.parent.name / f"{img.stem}.txt"
    if not lbl.exists():
        return []
    gt = []
    for line in lbl.read_text().split("\n"):
        v = line.split()
        if len(v) == 5:
            c, cx, cy, bw, bh = int(v[0]), *map(float, v[1:])
            gt.append((c, [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]))
    return gt


def draw(bgr: np.ndarray, gt, dets, tile: int) -> np.ndarray:
    for _, b in gt:
        cv2.rectangle(bgr, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (60, 220, 60), 2)
    for d in dets:
        b = [int(v) for v in d["box"]]
        cv2.rectangle(bgr, (b[0], b[1]), (b[2], b[3]), COLOURS[d["cls"]], 3)
        cv2.putText(bgr, f"{d['score']:.2f}", (b[0] + 2, max(b[1] - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6 * bgr.shape[1] / tile, COLOURS[d["cls"]], 2)
    s = tile / max(bgr.shape[:2])
    small = cv2.resize(bgr, (int(bgr.shape[1] * s), int(bgr.shape[0] * s)))
    canvas = np.full((tile, tile, 3), 30, dtype=np.uint8)
    canvas[: small.shape[0], : small.shape[1]] = small
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("model", type=Path)
    ap.add_argument("--split", default="test")
    ap.add_argument("--root", type=Path, default=HERE / "data" / "merged", help="dataset holding images/<split>")
    ap.add_argument("--tag", default="", help="suffix for the output names (e.g. the test set)")
    ap.add_argument("--images", type=Path, default=None, help="folder of pictures instead of a split")
    ap.add_argument("--limit", type=int, default=0, help="score only this many images (0 = all)")
    ap.add_argument("--conf", type=float, default=0.25, help="threshold for the boxes drawn")
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--threads", type=int, default=4, help="CPU threads (a modest server)")
    args = ap.parse_args()

    import onnxruntime as ort

    model = args.model
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = args.threads
    sess = ort.InferenceSession(str(model), opts, providers=["CPUExecutionProvider"])
    meta = sess.get_modelmeta().custom_metadata_map
    names = {int(k): v for k, v in json.loads(meta.get("names", '{"0":"fire","1":"smoke"}')).items()}
    size = int(meta.get("imgsz", 640))
    assert meta.get("letterbox_pad", "topleft") == "topleft", meta.get("letterbox_pad")
    inp = sess.get_inputs()[0].name
    print(f"{model.name}: input {sess.get_inputs()[0].shape}, output {sess.get_outputs()[0].shape}, {names}")

    root = args.images or args.root / "images" / args.split
    images = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    random.Random(0).shuffle(images)
    if args.limit:
        images = images[: args.limit]

    sess.run(None, {inp: np.zeros((1, 3, size, size), np.float32)})  # warm-up
    rows, ms_model, ms_total = [], [], []
    for n, img in enumerate(images, 1):
        t0 = time.perf_counter()
        bgr = cv2.imread(str(img))
        if bgr is None:
            print(f"unreadable: {img}")
            continue
        h, w = bgr.shape[:2]
        blob, ratio = letterbox(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), size)
        t1 = time.perf_counter()
        out = sess.run(None, {inp: blob})[0]
        t2 = time.perf_counter()
        dets = decode(out, ratio, w, h, min(THRESHOLDS), args.iou)
        ms_model.append((t2 - t1) * 1000)
        ms_total.append((time.perf_counter() - t0) * 1000)
        gt = read_gt(img, w, h)
        rows.append({
            "image": img.name,
            "gt": sorted({c for c, _ in gt}),
            "max": {c: max((d["score"] for d in dets if d["cls"] == c), default=0.0) for c in names},
            "dets": [d for d in dets if d["score"] > args.conf],
            "_gt_boxes": gt,
        })
        if n % 250 == 0:
            print(f"  {n}/{len(images)}  {np.mean(ms_model):.0f} ms/img model")

    # Image-level: a picture "raises" a class if any box of that class clears the threshold.
    labelled = args.images is None

    def image_level(subset: list[dict], t: float) -> dict:
        r = {}
        for c, name in names.items():
            pos = [x for x in subset if c in x["gt"]]
            neg = [x for x in subset if c not in x["gt"]]
            r[f"{name}_hit"] = sum(x["max"][c] > t for x in pos) / max(len(pos), 1)
            r[f"{name}_false"] = sum(x["max"][c] > t for x in neg) / max(len(neg), 1)
        bg = [x for x in subset if not x["gt"]]
        r["clean_quiet"] = sum(all(v <= t for v in x["max"].values()) for x in bg) / max(len(bg), 1)
        return {k: round(v, 3) for k, v in r.items()}

    by_t = {f"{t:.2f}": image_level(rows, t) for t in THRESHOLDS}
    # Merged images are named <source>_<hash>: score each source on its own, and the v1 sources
    # together ("old"), so a gain on one kind of picture can't hide a loss on another.
    sources: dict[str, list[dict]] = {}
    for x in rows:
        sources.setdefault(x["image"].split("_", 1)[0], []).append(x)
    if len(sources) > 1 and "pyro" in sources:
        sources["old"] = [x for x in rows if not x["image"].startswith("pyro_")]
    by_source = {s: {"images": len(xs), **{f"{t:.2f}": image_level(xs, t) for t in THRESHOLDS}}
                 for s, xs in sorted(sources.items())}

    counts = {
        "images": len(rows),
        **{f"with_{names[c]}": sum(c in x["gt"] for x in rows) for c in names},
        "clean": sum(not x["gt"] for x in rows),
    }
    speed = {
        "threads": args.threads,
        "model_ms_mean": round(float(np.mean(ms_model)), 1),
        "model_ms_p95": round(float(np.percentile(ms_model, 95)), 1),
        "total_ms_mean": round(float(np.mean(ms_total)), 1),
    }
    report = {"model": model.name, "source": str(root), "counts": counts, "speed_cpu": speed}
    if labelled:
        report["image_level"] = by_t
        report["by_source"] = by_source

    stem = f"smoke_{model.stem}{'_' + args.tag if args.tag else ''}"
    out = HERE / "runs" / f"{stem}.json"
    out.parent.mkdir(exist_ok=True)
    report["images"] = [{k: v for k, v in x.items() if not k.startswith("_")} for x in rows]
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")

    print(json.dumps({k: v for k, v in report.items() if k != "images"}, indent=1))
    if labelled:
        print("\nthreshold  " + "  ".join(f"{k:>11}" for k in next(iter(by_t.values()))))
        for t, r in by_t.items():
            print(f"{t:>9}  " + "  ".join(f"{v:>11.1%}" for v in r.values()))

    # Contact sheet: 6 per row, fire / smoke / clean, plus the worst misses and false alarms.
    tile, per = 320, 6
    groups = [("fire", [x for x in rows if 0 in x["gt"]]), ("smoke", [x for x in rows if 1 in x["gt"]]),
              ("clean", [x for x in rows if not x["gt"]])] if labelled else [("images", rows)]
    if labelled:
        groups.append(("missed", sorted([x for x in rows if x["gt"] and not x["dets"]],
                                        key=lambda x: x["image"])))
        groups.append(("false alarm", sorted([x for x in rows if not x["gt"] and x["dets"]],
                                             key=lambda x: -max(x["max"].values()))))
    by_name = {p.name: p for p in images}
    sheet_rows = []
    for label, items in groups:
        tiles = []
        for x in items[:per]:
            bgr = cv2.imread(str(by_name[x["image"]]))
            tiles.append(draw(bgr, x["_gt_boxes"], x["dets"], tile))
        tiles += [np.full((tile, tile, 3), 30, np.uint8)] * (per - len(tiles))
        head = np.full((26, tile * per, 3), 30, np.uint8)
        cv2.putText(head, f"{label} ({len(items)})", (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1)
        sheet_rows += [head, np.hstack(tiles)]
    sheet = HERE / "runs" / f"{stem}.jpg"
    cv2.imwrite(str(sheet), np.vstack(sheet_rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"\nwrote {out}\nwrote {sheet}")


if __name__ == "__main__":
    main()
