"""Contact sheet of labelled images, one colour per class id, to check what each id means.

    .venv/Scripts/python.exe -I preview.py data/raw/indoor --out runs/preview_indoor.jpg
    .venv/Scripts/python.exe -I preview.py data/merged/images/train --names fire smoke
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw

from collections import Counter

from prepare_data import IMAGE_EXTS, label_path, read_labels

COLOURS = [(255, 40, 40), (40, 120, 255), (40, 220, 40), (255, 200, 0), (220, 0, 220)]
TILE = 320


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, default=Path("runs/preview.jpg"))
    ap.add_argument("--n", type=int, default=6, help="images per class id")
    ap.add_argument("--names", nargs="*", default=None, help="label each id with a name")
    args = ap.parse_args()

    by_class: dict[int, list[tuple[Path, list[list[float]]]]] = {}
    images = sorted(p for p in args.root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    random.Random(0).shuffle(images)
    identity = {i: i for i in range(20)}
    for img in images:
        lines = read_labels(label_path(img), identity, Counter())  # polygons come back as boxes
        if not lines:
            continue
        boxes = [[float(v) for v in ln.split()] for ln in lines]
        ids = {int(b[0]) for b in boxes}
        if len(ids) == 1:  # one class only, so the picture shows what that id means
            (cid,) = ids
            if len(by_class.setdefault(cid, [])) < args.n:
                by_class[cid].append((img, boxes))
        if by_class and all(len(v) >= args.n for v in by_class.values()) and len(by_class) >= 2:
            break

    rows = sorted(by_class)
    sheet = Image.new("RGB", (TILE * args.n, (TILE + 24) * len(rows)), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)
    for r, cid in enumerate(rows):
        name = args.names[cid] if args.names and cid < len(args.names) else ""
        draw.text((6, r * (TILE + 24) + 4), f"class {cid} {name}", fill=COLOURS[cid % 5])
        for c, (img, boxes) in enumerate(by_class[cid]):
            with Image.open(img) as im:
                im = im.convert("RGB")
            im.thumbnail((TILE, TILE))
            d = ImageDraw.Draw(im)
            for b in boxes:
                cx, cy, w, h = b[1] * im.width, b[2] * im.height, b[3] * im.width, b[4] * im.height
                d.rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], outline=COLOURS[cid % 5], width=3)
            sheet.paste(im, (c * TILE, r * (TILE + 24) + 24))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out, quality=85)
    print(f"{args.out}: class ids {rows}")


if __name__ == "__main__":
    main()
