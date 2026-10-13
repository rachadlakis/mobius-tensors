"""Ask what the cameras saw.

    uv run -m histor.ask "white van at the gate yesterday 6pm to 9pm"
    uv run -m histor.ask                # interactive: one question per line, empty line quits

v0: a question -> ranked sightings (CLIP text search; time and camera parsed by rules, no LLM).
Next: a chat layer where an LLM calls this search as a tool and answers in sentences.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml
from histor.engine.cameras import camera_aliases
from histor.engine.clip_onnx import ClipEncoder
from histor.engine.search import SearchEngine
from histor.engine.store import EventStore


def build_engine(config: dict[str, Any]) -> SearchEngine:
    store = EventStore.from_config(config.get("store"))
    if store is None:
        raise SystemExit("The config needs a `store:` section (see configs/default.yaml)")
    clip_cfg = (config.get("search") or {}).get("clip") or {}
    text = clip_cfg.get("text", "models/clip_text.onnx")  # same defaults as the API server
    vocab = clip_cfg.get("vocab", "models/clip_bpe_vocab.txt.gz")
    clip = None
    if Path(text).exists() and Path(vocab).exists():
        clip = ClipEncoder(image=None, text=text, vocab=vocab,
                           device=clip_cfg.get("device", "auto"))
    else:  # objects, alerts and face search still work
        print(f"{text} not found: visual search by description is off")
    faces_cfg = config.get("faces") or {}
    faces = None
    if faces_cfg.get("enabled"):  # face search with attached photos (chat.py /photo)
        from histor.engine.faces import FaceEngine

        faces = FaceEngine.from_config(faces_cfg, for_photos=True)
    cameras = config.get("cameras") or {}
    return SearchEngine(store, clip, faces, camera_names=sorted(cameras),
                        camera_aliases=camera_aliases(cameras))


def format_answer(answer: dict[str, Any], media_dir: Path) -> str:
    q = answer["query"]
    lines = [f"  looking for '{q['text']}' on {q['cameras']}, {q['when']}"]
    if not answer["results"]:
        lines.append("  nothing found")
    for i, r in enumerate(answer["results"], 1):
        when = r["first_seen"].replace("T", " ")
        if r["last_seen"] != r["first_seen"]:
            when += f" -> {r['last_seen'].split('T')[1]}"
        label = r["label"] or r["kind"]
        lines.append(f"  {i:2}. {r['score']:.3f}  {r['camera']:<12} {when}  {label}"
                     f"  {media_dir / r['thumb'] if r['thumb'] else ''}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask what the cameras saw")
    parser.add_argument("question", nargs="*", help="e.g. red car on gate yesterday evening")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--top", type=int, default=10, help="Number of results")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    engine = build_engine(config)

    def ask(question: str) -> None:
        try:
            print(format_answer(engine.text(question, top_k=args.top), engine.store.media_dir))
        except RuntimeError as e:  # no CLIP text model: say so instead of a traceback
            print(e)

    try:
        if args.question:
            ask(" ".join(args.question))
            return
        while question := input("histor> ").strip():
            ask(question)
    except (EOFError, KeyboardInterrupt):
        print()
    finally:
        engine.store.close()


if __name__ == "__main__":
    main()
