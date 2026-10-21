"""FIRESENSE fire and smoke videos (Dimitropoulos et al., Zenodo record 836749, CC BY 4.0) for
the scenario bench: check the two zips, unpack each into its own new folder, and write one
bench spec per video (tests/scenarios/specs/fire/firesense_*.json).

Put fire_videos.1406.zip and smoke_videos.1407.zip (https://zenodo.org/records/836749) in
../../data/test_videos/firesense/zips/ first. Nothing in the zips is ever run.

    .venv/Scripts/python.exe -I get_firesense.py           # check, unpack, write the specs
    .venv/Scripts/python.exe -I get_firesense.py --specs   # the specs again (after an edit)

Positives expect a `fire` alert (flames or smoke) within 30 s of the clip's start, and count
for early recall within 3 s; negatives expect none. Labels are per video, not per frame: a
spec's `visible_from` can be set by hand where the fire shows up late.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import sys
import zipfile
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
FOLDER = PROJECT / "data" / "test_videos" / "firesense"
SPECS = PROJECT / "tests" / "scenarios" / "specs" / "fire"
# name: (size, md5) from https://zenodo.org/api/records/836749; unpacked into FOLDER / kind
ZIPS = {"fire_videos.1406.zip": (622676845, "7cfe8286cc4b45193149186bf67ea00d", "fire"),
        "smoke_videos.1407.zip": (197589846, "5ee6bc867c4a9332b4e75f308c26fc23", "smoke")}
VIDEOS = {".avi", ".mp4", ".mov", ".mkv", ".mpg", ".mpeg", ".wmv", ".flv"}
MAX_DELAY, EARLY = 30.0, 3.0


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(path: Path, size: int, digest: str) -> str | None:
    """What is wrong with a downloaded zip, or None."""
    if not path.exists():
        part = path.with_name(path.name + ".part")
        have = f" ({part.stat().st_size:,} of {size:,} bytes so far)" if part.exists() else ""
        return f"{path.name}: not downloaded{have}"
    if path.stat().st_size != size:
        return f"{path.name}: {path.stat().st_size:,} bytes, expected {size:,}"
    if md5(path) != digest:
        return f"{path.name}: md5 differs from Zenodo's {digest}"
    with zipfile.ZipFile(path) as z:
        bad = z.testzip()
    return f"{path.name}: damaged member {bad}" if bad else None


def safe_extract(archive: Path, dest: Path) -> list[Path]:
    """Unpack into `dest`, which must be new or empty. Refuses the whole zip if any member is
    absolute, climbs out with '..' or is a link; writes plain files only."""
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"{dest} is not empty")
    with zipfile.ZipFile(archive) as z:
        members = z.infolist()
        for m in members:
            name = m.filename.replace("\\", "/")
            parts = PurePosixPath(name).parts
            if name.startswith("/") or (parts and ":" in parts[0]) or ".." in parts:
                raise ValueError(f"{archive.name}: unsafe path {m.filename!r}")
            if stat.S_ISLNK(m.external_attr >> 16):
                raise ValueError(f"{archive.name}: link {m.filename!r}")
        dest.mkdir(parents=True, exist_ok=True)
        written = []
        for m in members:
            if m.is_dir():
                continue
            target = dest.joinpath(*PurePosixPath(m.filename.replace("\\", "/")).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(m) as src, target.open("wb") as out:
                while chunk := src.read(1 << 20):
                    out.write(chunk)
            written.append(target)
    return written


def label(path: Path, root: Path) -> str | None:
    """pos | neg from the nearest folder whose name says so (FIRESENSE: .../pos/, .../neg/)."""
    for part in reversed(path.relative_to(root).parts[:-1]):
        low = part.lower()
        if "neg" in low:
            return "neg"
        if "pos" in low:
            return "pos"
    return None


def make_specs(folder: Path = FOLDER, out: Path = SPECS, project: Path = PROJECT) -> list[Path]:
    """One spec per video under folder/fire and folder/smoke; the old firesense_* removed."""
    for old in out.glob("firesense_*.json"):
        old.unlink()
    written = []
    unknown = []
    for kind in ("fire", "smoke"):
        root = folder / kind
        for video in sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEOS):
            side = label(video, root)
            if side is None:
                unknown.append(video)
                continue
            spec: dict[str, object] = {"clip": video.relative_to(project).as_posix()}
            if side == "pos":
                spec["expect"] = [{"type": "fire", "from": 0.0, "max_delay": MAX_DELAY,
                                   "early": EARLY}]
            else:
                spec["negative"] = True
            spec["notes"] = f"FIRESENSE {kind} video, {'positive' if side == 'pos' else 'negative'}."
            path = out / f"firesense_{kind}_{side}_{video.stem.replace(' ', '_')}.json"
            path.write_text(json.dumps(spec, indent=1) + "\n", encoding="utf-8")
            written.append(path)
    if unknown:
        raise ValueError(f"no pos/neg folder for {len(unknown)} videos, e.g. {unknown[0]}: "
                         f"check the layout and the `label` rule")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--specs", action="store_true", help="only write the specs")
    args = ap.parse_args()
    if not args.specs:
        problems = [p for name, (size, digest, _) in ZIPS.items()
                    if (p := check(FOLDER / "zips" / name, size, digest))]
        if problems:
            sys.exit("\n".join(problems))
        for name, (_, _, kind) in ZIPS.items():
            files = safe_extract(FOLDER / "zips" / name, FOLDER / kind)
            print(f"{name}: {len(files)} files -> {FOLDER / kind}")
    specs = make_specs()
    print(f"{len(specs)} specs -> {SPECS}")


if __name__ == "__main__":
    main()
