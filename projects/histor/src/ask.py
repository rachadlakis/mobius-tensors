"""Ask what the cameras saw.

    uv run src/ask.py "white van at the gate yesterday 6pm to 9pm"
    uv run src/ask.py                  # interactive: one question per line, empty line quits

v0: a question -> ranked sightings (CLIP text search; time and camera parsed by rules, no LLM).
Next: a chat layer where an LLM calls this search as a tool and answers in sentences.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

import shared  # noqa: F401
from clip_onnx import ClipEncoder
from search import SearchEngine
from store import EventStore


def build_engine(config: dict[str, Any]) -> SearchEngine:
    store = EventStore.from_config(config.get("store"))
    if store is None:
        raise SystemExit("The config needs a `store:` section (see configs/default.yaml)")
    clip_cfg = (config.get("search") or {}).get("clip") or {}
    clip = ClipEncoder(image=None, text=clip_cfg["text"], vocab=clip_cfg["vocab"])
    return SearchEngine(store, clip, camera_names=sorted((config.get("cameras") or {}).keys()))


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

    config = yaml.safe_load(args.config.read_text())
    engine = build_engine(config)
    ask = lambda q: print(format_answer(engine.text(q, top_k=args.top),  # noqa: E731
                                        engine.store.media_dir))
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
