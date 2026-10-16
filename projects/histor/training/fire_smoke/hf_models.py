"""Publish the trained fire/smoke models to a private Hugging Face model repo (not git: the .pt
files are 88 MB each).

    .venv/Scripts/python.exe -I hf_models.py                  # upload models/ (new or changed)
    .venv/Scripts/python.exe -I hf_models.py --repo user/name

Uploads every .onnx, .pt and .test_metrics.json in models/, plus MODELS.md and the card
(hf_model_card.md, as README.md). Prints the commit to pin in histor/engine/download_model.py.
Token: HF_TOKEN, or HUGGINGFACE_API_KEY in projects/histor/.env.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = "Rachad/histor-fire-smoke-models"


def token() -> str | None:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    env = HERE.parent.parent / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "HUGGINGFACE_API_KEY":
                return value.strip().strip("'\"")
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--repo", default=REPO)
    args = ap.parse_args()

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=token())
    api.create_repo(args.repo, repo_type="model", private=True, exist_ok=True)
    models = HERE / "models"
    files = sorted(p for p in models.iterdir()
                   if p.suffix in (".onnx", ".pt") or p.name.endswith(".test_metrics.json")
                   or p.name == "MODELS.md")
    ops = [CommitOperationAdd(path_in_repo=p.name, path_or_fileobj=str(p)) for p in files]
    ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(HERE / "hf_model_card.md")))
    print(f"uploading {len(files)} files ({sum(p.stat().st_size for p in files) / 1e6:.0f} MB) "
          f"to {args.repo} ...", flush=True)
    info = api.create_commit(args.repo, ops, repo_type="model", commit_message="Fire/smoke models")
    print(f"commit {info.oid}\nhttps://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
