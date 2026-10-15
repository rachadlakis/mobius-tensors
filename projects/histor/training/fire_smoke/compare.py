"""Score several fire/smoke models on one test set, split by where the pictures come from.

    .venv/Scripts/python.exe -I compare.py data/v2_test/all <model stem> <model stem> ...

Each stem names models/<stem>.pt (for mAP, on the GPU if there is one) and models/<stem>.onnx
(for smoke_test.py on CPU). The test set is scored whole ("all"), without Pyro-SDIS ("old": the
v1 sources, i.e. run 2's own test images) and Pyro only ("pyro": tiny distant smoke).
Writes runs/compare_<test set>.json and prints a Markdown table.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KEYS = {"mAP50": "metrics/mAP50", "mAP50-95": "metrics/mAP50-95", "recall": "metrics/recall",
        "mAP_small": "metrics/mAP_small"}


def subset(all_root: Path, name: str, keep) -> Path:
    """data/<test>/<name>/images/test + labels/test, hard-linked from all/."""
    root = all_root.parent / name
    for kind in ("images", "labels"):
        (root / kind / "test").mkdir(parents=True, exist_ok=True)
    for img in (all_root / "images" / "test").iterdir():
        if not keep(img.name):
            continue
        lbl = all_root / "labels" / "test" / f"{img.stem}.txt"
        for src, dst in ((img, root / "images" / "test" / img.name), (lbl, root / "labels" / "test" / lbl.name)):
            if src.exists() and not dst.exists():
                os.link(src, dst)
    return root


def yaml_for(root: Path) -> Path:
    y = root / "dataset.yaml"
    y.write_text(
        f"path: {root}\ntrain: images/test\nval: images/test\ntest: images/test\nnc: 2\nnames:\n- fire\n- smoke\n",
        encoding="utf-8",
    )
    return y


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("test", type=Path, help="dataset folder with images/test and labels/test")
    ap.add_argument("models", nargs="+")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--no-smoke", action="store_true")
    args = ap.parse_args()

    from libreyolo import LibreYOLO

    all_root = args.test.resolve()
    sets = {
        "all": all_root,
        "old": subset(all_root, "old", lambda n: not n.startswith("pyro_")),
        "pyro": subset(all_root, "pyro", lambda n: n.startswith("pyro_")),
    }
    tag = all_root.parent.name
    out = HERE / "runs" / f"compare_{tag}.json"
    results = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}

    for stem in args.models:
        res = results.setdefault(stem, {})
        for set_name, root in sets.items():
            if f"map_{set_name}" in res:
                continue
            n = len(list((root / "images" / "test").iterdir()))
            m = LibreYOLO(str(HERE / "models" / f"{stem}.pt")).val(
                data=str(yaml_for(root)), split="test", imgsz=640, batch=16, workers=2,
                project=str(HERE / "runs" / "compare"), name=f"{stem}_{set_name}", exist_ok=True,
            )
            res[f"map_{set_name}"] = {"images": n, **{k: round(float(m[v]), 4) for k, v in KEYS.items()}}
            out.write_text(json.dumps(results, indent=1), encoding="utf-8")
            print(stem, set_name, res[f"map_{set_name}"], flush=True)
        if not args.no_smoke and "smoke" not in res:
            subprocess.run([sys.executable, "-I", str(HERE / "smoke_test.py"), str(HERE / "models" / f"{stem}.onnx"),
                            "--root", str(all_root), "--tag", tag, "--threads", str(args.threads)],
                           check=True, stdout=subprocess.DEVNULL)
            s = json.loads((HERE / "runs" / f"smoke_{stem}_{tag}.json").read_text(encoding="utf-8"))
            res["smoke"] = {"speed_cpu": s["speed_cpu"],
                            "by_source": {k: {"images": v["images"], "0.25": v["0.25"]} for k, v in s["by_source"].items()},
                            "image_level_all": s["image_level"]}
            out.write_text(json.dumps(results, indent=1), encoding="utf-8")
            print(stem, "smoke", s["speed_cpu"], flush=True)

    print(f"\n| Model | mAP50-95 all | old | pyro | mAP50 all | old | pyro | CPU ms |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for stem in args.models:
        r = results[stem]
        ms = r.get("smoke", {}).get("speed_cpu", {}).get("model_ms_mean", "–")
        print(f"| `{stem}` | " + " | ".join(f"{r[f'map_{s}'][k]:.3f}" for k in ("mAP50-95", "mAP50") for s in sets) + f" | {ms} |")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
