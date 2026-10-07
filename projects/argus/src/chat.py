"""Chat with the cameras: an LLM answers from the Argus store using the search tools.

    uv run src/chat.py                                   # interactive, model = chat.model
    uv run src/chat.py "how many cars came in yesterday?"
    uv run src/chat.py --model claude "was there a dog on the lawn this morning?"
    uv run src/chat.py --photo visitor.jpg "did this person come to the office between 1 and 4?"

Interactive commands: /photo <file> [<file> ...] [--face N] attaches photos of one person for
face search (N picks the face when a photo has several, 1 = leftmost), /photos lists them,
/model <name> switches model (starts a new conversation), /models lists them, empty line quits.
Models are declared under `chat.models` in the config.

Photos stay on this machine: only their face prints are kept, and the model only learns their
names ("photo1"), never the image, also with a cloud model.
"""

from __future__ import annotations

import argparse
import logging
import shlex
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from dotenv import load_dotenv

import shared  # noqa: F401
from ask import build_engine
from cameras import expand_env
from llm import Backend, Step, create_backend
from tools import Toolbox

MAX_TOOL_ROUNDS = 6

SYSTEM = """You answer questions about what security cameras recorded.
Everything you know about the footage comes from your tools: never invent sightings, times or counts.

Now: {now}
Cameras (what each one looks at, and what it recorded):
{cameras}
Alert rules that have fired so far:
{rules}

How to work:
- Pick cameras from their descriptions: "did anyone reach the office" -> the camera covering the
  office, not the corridor on the way. Cameras can be named by their aliases.
- `objects` lists what the detector tracked, in time order, with counts. Exact.
  Time only: "what happened yesterday 6pm to 9pm" -> objects(when="yesterday 6pm to 9pm").
  Count a class: "how many cars today" -> objects(labels=["car"], when="today").
  Together: "people with a dog" -> objects(labels=["person"], with_labels=["dog"]).
- `search` finds things by description (colors, clothing, actions, things with no class).
  Ranked guesses. Takes the same when / since / until / camera / with_labels filters.
- `alerts` lists what the site's alert rules fired (enter / exit / dwell / cross on named areas
  and lines). Exact. "did any alert go off last night" -> alerts(when="last night");
  "who entered the yard" -> alerts(rules=["yard"], types=["enter"]).{face_help}
- For exact times use since/until ("2026-10-02 18:00"), resolved against Now above. Hours
  without am/pm ("between 1 and 4") mean daytime (13:00 to 16:00) unless the question or the
  camera's usual hours say otherwise; say what you assumed.
- Run several lookups when a question has several parts; rephrase once if nothing matches.

How to answer:
- Short and concrete: camera, date and time, what was seen.
- Say how sure you are. Search scores are relative: a top hit is a candidate, not proof.
- If the tools found nothing, say so, and say what time range and cameras you checked."""

FACE_HELP = """
- `face` looks for the person in a photo the user attached (photo1, photo2, ...: you get only
  the name). "did this person come to the office between 1 and 4" ->
  face(photo="photo1", since="<today> 13:00", until="<today> 16:00", camera="office").
  "likely" visits are strong matches; for "possible" ones tell the user to check the thumbnail.
  No match is not proof: say how many faces the checked cameras recorded and whether they can
  see faces (face_recording_on, face_view). Without an attached photo, ask for one."""


def model_config(config: dict[str, Any], name: str) -> dict[str, Any]:
    models = (config.get("chat") or {}).get("models") or {}
    if name not in models:
        raise SystemExit(f"No chat model {name!r}. Configured: {', '.join(models) or 'none'}")
    return {k: expand_env(v) if isinstance(v, str) else v for k, v in models[name].items()}


def system_prompt(toolbox: Toolbox, now: datetime | None = None) -> str:
    cameras = [f"- {line}" for line in toolbox.cameras()] or ["- nothing recorded yet"]
    rules = [f"- {r['camera']}: {r['rule']} ({r['events']} events)" for r in toolbox.alert_rules()]
    return SYSTEM.format(now=(now or datetime.now()).strftime("%A %Y-%m-%d %H:%M"),
                         cameras="\n".join(cameras), rules="\n".join(rules) or "- none yet",
                         face_help=FACE_HELP if toolbox.engine.faces is not None else "")


def answer(backend: Backend, toolbox: Toolbox, question: str,
           log: logging.Logger | None = None) -> str:
    """One question: let the model call tools until it answers (at most MAX_TOOL_ROUNDS)."""
    backend.add_user(question)
    step = Step("")
    for _ in range(MAX_TOOL_ROUNDS):
        step = backend.step()
        if not step.calls:
            return step.text
        if log:
            for c in step.calls:
                log.info("  -> %s %s", c.name, c.args)
        backend.add_results([toolbox.run(c) for c in step.calls])
    return step.text or "(Stopped: too many tool calls without an answer. Try a narrower question.)"


def read_image(path: str | Path) -> np.ndarray:
    """cv2.imread can't open non-ASCII paths on Windows; decoding the bytes can."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError as e:
        raise ValueError(f"Can't read {path}: {e.strerror}") from None
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"{path} is not an image (use a JPEG or PNG)")
    return image


def attach(toolbox: Toolbox, paths: list[str], choice: int | None = None) -> str | None:
    """Attach photos of one person; prints what happened, returns the photo's name."""
    try:
        images = [read_image(p) for p in paths]
    except ValueError as e:
        print(e)
        return None
    name, message = toolbox.add_photo(images, choice)
    print(message)
    return name


def parse_photo_command(rest: str) -> tuple[list[str], int | None]:
    """'a.jpg "my photos/b.jpg" --face 2' -> (['a.jpg', 'my photos/b.jpg'], 2)."""
    words = [w.strip('"\'') for w in shlex.split(rest, posix=False)]
    choice = None
    if "--face" in words:
        i = words.index("--face")
        if i + 1 >= len(words) or not words[i + 1].isdigit():
            raise ValueError("--face needs a number (1 = leftmost face)")
        choice = int(words[i + 1])
        del words[i: i + 2]
    if not words:
        raise ValueError("Usage: /photo <file> [<file> ...] [--face N]")
    return words, choice


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with what the cameras saw")
    parser.add_argument("question", nargs="*")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--model", help="Entry of chat.models (default: chat.model)")
    parser.add_argument("--photo", action="append", default=[],
                        help="Photo of a person to look for (repeat for more photos of them)")
    parser.add_argument("--face", type=int, help="Which face in the photo (1 = leftmost)")
    parser.add_argument("--quiet", action="store_true", help="Don't print tool calls")
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(message)s")
    log = logging.getLogger("argus")
    config = yaml.safe_load(args.config.read_text())
    engine = build_engine(config)
    toolbox = Toolbox(engine, cameras=config.get("cameras"), faces_config=config.get("faces"))
    media_dir = engine.store.media_dir
    untold: list[str] = []  # photos attached since the model last heard about them

    def start(name: str) -> Backend:
        backend = create_backend(model_config(config, name), system_prompt(toolbox),
                                 toolbox.tools)
        untold[:] = list(toolbox.photos)  # a new conversation: tell it about every photo
        log.info("Model: %s (%s)", name, backend.name)
        return backend

    def ask(question: str) -> None:
        toolbox.evidence.clear()
        if untold:
            question = f"[Attached for face search: {', '.join(untold)}]\n{question}"
            untold.clear()
        try:
            print(answer(backend, toolbox, question, log))
        except RuntimeError as e:
            print(f"Error: {e}")
            return
        if toolbox.evidence:
            print("Evidence:")
            for thumb in dict.fromkeys(toolbox.evidence):  # dedupe, keep order
                print(f"  {media_dir / thumb}")

    backend = start(args.model or (config.get("chat") or {}).get("model", "local"))
    try:
        if args.photo and (name := attach(toolbox, args.photo, args.face)):
            untold.append(name)
        elif args.photo and args.question:
            return  # the photo was refused: the question was about it
        if args.question:
            ask(" ".join(args.question))
            return
        while line := input("\nyou> ").strip():
            if line == "/models":
                print(", ".join((config.get("chat") or {}).get("models") or {}))
            elif line.startswith("/model "):
                try:
                    backend = start(line.split(maxsplit=1)[1])  # new conversation
                except SystemExit as e:  # unknown name or missing SDK: keep the current model
                    print(e)
            elif line == "/photos":
                print(", ".join(toolbox.photos) or "No photos attached.")
            elif line.startswith("/photo"):
                try:
                    paths, choice = parse_photo_command(line[len("/photo"):])
                except ValueError as e:
                    print(e)
                    continue
                if name := attach(toolbox, paths, choice):
                    untold.append(name)
            else:
                ask(line)
    except (EOFError, KeyboardInterrupt):
        print()
    finally:
        toolbox.close()
        engine.store.close()


if __name__ == "__main__":
    main()
