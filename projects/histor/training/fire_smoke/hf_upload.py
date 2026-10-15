"""Back up the merged training set to a private Hugging Face dataset repo, one tar per split.

    python hf_upload.py Rachad/histor-fire-smoke          # token from HF_TOKEN or `hf auth login`
    python hf_upload.py Rachad/histor-fire-smoke --get    # download and unpack into data/merged/

The card (hf_dataset_card.md) credits every source; keep the repo private.
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

HERE = Path(__file__).resolve().parent
MERGED = HERE / "data" / "merged"
SPLITS = ["train", "val", "test"]


def upload(repo: str, staging: Path) -> None:
    staging.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        out = staging / f"{split}.tar"
        if not out.exists():
            with tarfile.open(out.with_suffix(".part"), "w") as tar:  # jpgs: no gain from gzip
                for kind in ("images", "labels"):
                    tar.add(MERGED / kind / split, arcname=f"{kind}/{split}")
            out.with_suffix(".part").rename(out)
        print(f"{out.name}: {out.stat().st_size / 1e9:.2f} GB")
    shutil.copy2(MERGED / "dataset.yaml", staging / "dataset.yaml")
    shutil.copy2(MERGED / "report.json", staging / "report.json")
    shutil.copy2(HERE / "sources.yaml", staging / "sources.yaml")
    shutil.copy2(HERE / "hf_dataset_card.md", staging / "README.md")

    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=staging,
                      ignore_patterns=["*.part", ".cache/*"], commit_message="v2 training set")
    print(f"uploaded to https://huggingface.co/datasets/{repo}")


def get(repo: str) -> None:
    local = Path(snapshot_download(repo, repo_type="dataset", local_dir=HERE / "data" / "hf"))
    MERGED.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        with tarfile.open(local / f"{split}.tar") as tar:
            tar.extractall(MERGED, filter="data")
    yaml_text = (local / "dataset.yaml").read_text(encoding="utf-8")
    first, rest = yaml_text.split("\n", 1)
    (MERGED / "dataset.yaml").write_text(f"path: {MERGED}\n{rest}", encoding="utf-8")
    shutil.copy2(local / "report.json", MERGED / "report.json")
    print(f"unpacked into {MERGED} (old path line was: {first})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("repo")
    ap.add_argument("--get", action="store_true", help="download instead of upload")
    ap.add_argument("--staging", type=Path, default=HERE / "data" / "hf_upload")
    args = ap.parse_args()
    get(args.repo) if args.get else upload(args.repo, args.staging)


if __name__ == "__main__":
    main()
