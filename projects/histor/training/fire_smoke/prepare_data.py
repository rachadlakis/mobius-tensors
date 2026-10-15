"""Merge the datasets in data/raw/ into one YOLO dataset in data/merged/.

For each source in sources.yaml: find image/label pairs, remap classes, fix or drop bad boxes,
then group near-identical images across all sources into scenes (frames of one video, the same
photo re-uploaded or flipped). Copies are dropped, at most --max-per-scene frames are kept, and
each scene goes to one split (80/10/10) so validation never sees a frame of a training video.
Images are hard-linked, not copied (disk is tight).

A share of the training images (--cctv) is replaced by a CCTV-like version: grayscale like a
camera at night on infrared, low resolution like a sub stream, blur, heavy JPEG. The trainer has
no such augmentation and web photos are much cleaner than camera frames.

    .venv/Scripts/python.exe -I prepare_data.py            # build data/merged/
    .venv/Scripts/python.exe -I prepare_data.py --check    # only report, write nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image, ImageFilter

HERE = Path(__file__).resolve().parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TRAINER_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}  # what LibreYOLO reads; others become .jpg
SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}
SCENE_BITS = 4  # dHash bits that may differ for two images to count as one scene (video frames)
COPY_BITS = 2  # ... and to count as the same picture (re-uploads, flipped augmentation copies)


FLIPS = [
    None,
    Image.Transpose.FLIP_LEFT_RIGHT,
    Image.Transpose.FLIP_TOP_BOTTOM,
    Image.Transpose.ROTATE_90,
    Image.Transpose.ROTATE_180,
    Image.Transpose.ROTATE_270,
]


def _dhash(gray: Image.Image) -> int:
    px = gray.resize((9, 8), Image.Resampling.LANCZOS).tobytes()
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (px[row * 9 + col] > px[row * 9 + col + 1])
    return bits


def dhashes(path: Path) -> list[int] | None:
    """64-bit difference hash of the image and of its flips and rotations. Survives resizing,
    re-compression and brightness changes, not crops. The flips catch the mirrored and rotated
    copies that Roboflow exports add as augmentation."""
    try:
        with Image.open(path) as im:
            gray = im.convert("L")
    except Exception:
        return None
    return [_dhash(gray if t is None else gray.transpose(t)) for t in FLIPS]


def label_path(image: Path) -> Path:
    """YOLO convention: .../images/x.jpg -> .../labels/x.txt (falls back to x.txt beside it)."""
    parts = list(image.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image.with_suffix(".txt")


def read_labels(path: Path, class_map: dict[int, int], stats: Counter) -> list[str] | None:
    """Remapped YOLO lines, or None if the label file is missing. Polygons become boxes."""
    if not path.exists():
        return None
    lines = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        vals = raw.split()
        if not vals:
            continue
        try:
            cls, nums = int(float(vals[0])), [float(v) for v in vals[1:]]
        except ValueError:
            stats["bad line"] += 1
            continue
        if cls not in class_map:
            stats[f"dropped class {cls}"] += 1
            continue
        if len(nums) == 4:
            cx, cy, w, h = nums
        elif len(nums) >= 6 and len(nums) % 2 == 0:  # segmentation polygon
            xs, ys = nums[0::2], nums[1::2]
            cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            w, h = max(xs) - min(xs), max(ys) - min(ys)
            stats["polygon -> box"] += 1
        else:
            stats["bad line"] += 1
            continue
        x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
        x2, y2 = min(1.0, cx + w / 2), min(1.0, cy + h / 2)
        if x2 - x1 < 1e-3 or y2 - y1 < 1e-3:
            stats["empty box"] += 1
            continue
        if (x1, y1, x2, y2) != (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2):
            stats["clipped box"] += 1
        lines.append(
            f"{class_map[cls]} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}"
        )
    return lines


def find_duplicates(variants: list[list[int]]) -> list[int]:
    """Scene id per item; items within SCENE_BITS share one (transitively). An item's original
    hash is compared with every variant (flip/rotation) of the others. Multi-index: split the 64
    bits into SCENE_BITS + 1 chunks, so two hashes within SCENE_BITS agree on at least one chunk."""
    owner = [i for i, vs in enumerate(variants) for _ in vs]
    original = [k == 0 for vs in variants for k in range(len(vs))]
    hashes = [h for vs in variants for h in vs]
    parent = list(range(len(variants)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    n_chunks = SCENE_BITS + 1
    width = 64 // n_chunks
    for c in range(n_chunks):
        shift = c * width
        mask = (1 << (width if c < n_chunks - 1 else 64 - shift)) - 1
        buckets: dict[int, list[int]] = defaultdict(list)
        for i, h in enumerate(hashes):
            buckets[(h >> shift) & mask].append(i)
        for members in buckets.values():
            if len(members) > 200:  # flat/blank images all collide; exact matches only
                exact: dict[int, int] = {}
                for e in members:
                    f = exact.setdefault(hashes[e], e)
                    if owner[f] != owner[e] and (original[e] or original[f]):
                        parent[find(owner[e])] = find(owner[f])
                continue
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    e, f = members[a], members[b]
                    if owner[e] == owner[f] or not (original[e] or original[f]):
                        continue
                    if bin(hashes[e] ^ hashes[f]).count("1") <= SCENE_BITS:
                        parent[find(owner[e])] = find(owner[f])
    return [find(i) for i in range(len(variants))]


def split_of(key: str) -> str:
    """Deterministic split from a stable key, so re-running gives the same split."""
    x = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    acc = 0.0
    for name, frac in SPLITS.items():
        acc += frac
        if x < acc:
            return name
    return "test"


def link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def cctv_version(src: Path, dst: Path, rng: random.Random) -> None:
    """Same picture as a cheap camera would give it. Size is kept, so YOLO boxes still fit."""
    with Image.open(src) as im:
        im = im.convert("RGB")
    w, h = im.size
    if rng.random() < 0.5:  # night: IR cameras are grayscale
        im = im.convert("L").convert("RGB")
    if rng.random() < 0.6:  # sub stream: 640x360 or less, upscaled back
        s = rng.uniform(0.25, 0.6) * min(1.0, 1280 / max(w, h))
        small = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.Resampling.BILINEAR)
        im = small.resize((w, h), Image.Resampling.BILINEAR)
    if rng.random() < 0.3:  # out of focus or motion
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 1.5)))
    im.save(dst, "JPEG", quality=rng.randint(25, 60))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--raw", type=Path, default=HERE / "data" / "raw")
    ap.add_argument("--out", type=Path, default=HERE / "data" / "merged")
    ap.add_argument("--config", type=Path, default=HERE / "sources.yaml")
    ap.add_argument("--check", action="store_true", help="report only, write nothing")
    ap.add_argument("--cctv", type=float, default=0.2, help="share of train images made CCTV-like")
    ap.add_argument("--max-per-scene", type=int, default=10, help="frames kept per video/scene")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    names: list[str] = cfg["names"]
    items = []  # (source, image path, label lines)
    report: dict[str, dict] = {}

    for source, spec in cfg["sources"].items():
        root = args.raw / source
        if not root.is_dir():
            print(f"skip {source}: {root} not found")
            continue
        class_map = {int(k): names.index(v) for k, v in spec["classes"].items()}
        stats: Counter = Counter()
        images = (p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
        images = sorted(p for p in images if not p.name.startswith("._"))  # macOS junk
        if spec.get("max_images") and len(images) > spec["max_images"]:
            stats["over max_images (skipped)"] = len(images) - spec["max_images"]
            images = sorted(random.Random(0).sample(images, spec["max_images"]))
        for img in images:
            lines = read_labels(label_path(img), class_map, stats)
            if lines is None:
                stats["no label file (skipped)"] += 1
                continue
            stats["images"] += 1
            stats["negatives" if not lines else "positives"] += 1
            for line in lines:
                stats[f"box {names[int(line.split()[0])]}"] += 1
            items.append((source, img, lines))
        report[source] = dict(stats)
        print(f"{source}: {dict(stats)}")

    if not items:
        raise SystemExit("no images found; put datasets in data/raw/<name>/ (see sources.yaml)")

    # Fixed cameras (one view, frames hours apart): every frame looks like a copy of the others to
    # the hash, so skip it and keep each camera whole in one split instead.
    fixed = [it for it in items if cfg["sources"][it[0]].get("fixed_cameras")]
    items = [it for it in items if not cfg["sources"][it[0]].get("fixed_cameras")]

    print(f"hashing {len(items)} images ...")
    variants, kept = [], []
    for item in items:
        hs = dhashes(item[1])
        if hs is None:
            report[item[0]]["unreadable"] = report[item[0]].get("unreadable", 0) + 1
            continue
        variants.append(hs)
        kept.append(item)
    groups = find_duplicates(variants)

    # Per scene: originals first, then most boxes; skip copies of a frame already chosen, cap
    # the count. Roboflow exports bake augmentation into train/ (upside-down and rotated copies)
    # but never into valid/ or test/, so a copy found there is the original. Fire and smoke rise:
    # when a frame has an upside-down/rotated copy and neither is known original, drop both.
    def original(i: int) -> bool:
        return any(p.lower() in ("valid", "val", "test") for p in kept[i][1].parts)

    scenes: dict[int, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        scenes[g].append(i)
    chosen: list[tuple[int, int]] = []  # (scene, item)
    n_copies = n_capped = n_orientation = 0
    for g, members in scenes.items():
        members.sort(key=lambda i: (not original(i), -len(kept[i][2])))
        picked: list[int] = []
        unsure: set[int] = set()
        for i in members:
            copy_of = []  # (picked item, which variant matched: 0 same, 1 mirrored, 2+ turned)
            for j in picked:
                bits, k = min((bin(variants[i][0] ^ v).count("1"), k) for k, v in enumerate(variants[j]))
                if bits <= COPY_BITS:
                    copy_of.append((j, k))
            if copy_of:
                n_copies += 1
                unsure.update(j for j, k in copy_of if k >= 2 and not original(j))
            elif len(picked) >= args.max_per_scene:
                n_capped += 1
            else:
                picked.append(i)
        n_orientation += len(unsure)
        chosen += [(g, i) for i in picked if i not in unsure]
    print(
        f"{len(scenes)} scenes; dropped {n_copies} copies, {n_orientation} frames of unknown "
        f"orientation and {n_capped} frames over {args.max_per_scene} per scene; kept {len(chosen)}"
    )

    counts: dict[str, Counter] = {s: Counter() for s in SPLITS}
    plan = []
    keyed = [(f"{variants[g][0]:016x}", kept[i]) for g, i in chosen]
    keyed += [(f"{source}/{img.parent.name}", (source, img, lines)) for source, img, lines in fixed]
    for key, (source, img, lines) in keyed:
        split = split_of(key)
        stem = f"{source}_{hashlib.sha1(str(img.relative_to(args.raw)).encode()).hexdigest()[:12]}"
        plan.append((split, stem, img, lines, cfg["sources"][source].get("cctv", True)))
        counts[split]["images"] += 1
        counts[split]["negatives" if not lines else "positives"] += 1
        for line in lines:
            counts[split][names[int(line.split()[0])]] += 1
    for split, c in counts.items():
        print(f"{split}: {dict(c)}")

    if args.check:
        return
    if args.out.exists():
        shutil.rmtree(args.out)
    for split in SPLITS:
        (args.out / "images" / split).mkdir(parents=True)
        (args.out / "labels" / split).mkdir(parents=True)
    n_cctv = 0
    for split, stem, img, lines, cctv in plan:
        rng = random.Random(stem)
        ext = img.suffix.lower()
        if split == "train" and rng.random() < args.cctv and cctv:
            cctv_version(img, args.out / "images" / split / f"{stem}.jpg", rng)
            n_cctv += 1
        elif ext in TRAINER_EXTS:
            link_or_copy(img, args.out / "images" / split / f"{stem}{ext}")
        else:
            with Image.open(img) as im:
                im.convert("RGB").save(args.out / "images" / split / f"{stem}.jpg", quality=95)
        (args.out / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
    (args.out / "dataset.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(args.out),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "nc": len(names),
                "names": names,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (args.out / "report.json").write_text(
        json.dumps(
            {
                "sources": {s: {**cfg["sources"][s], "stats": report[s]} for s in report},
                "scenes": len(scenes),
                "copies_dropped": n_copies,
                "unknown_orientation_dropped": n_orientation,
                "frames_over_cap_dropped": n_capped,
                "cctv_versions": n_cctv,
                "splits": {s: dict(c) for s, c in counts.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.out} ({n_cctv} train images made CCTV-like)")


if __name__ == "__main__":
    main()
