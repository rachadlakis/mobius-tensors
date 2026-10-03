"""Chat with the cameras: an LLM answers from the Histor store using the search tools.

    uv run src/chat.py                                   # interactive, model = chat.model
    uv run src/chat.py "how many cars came in yesterday?"
    uv run src/chat.py --model claude "was there a dog on the lawn this morning?"

Interactive commands: /model <name> switches model (starts a new conversation), /models lists
them, empty line quits. Models are declared under `chat.models` in the config.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

import shared  # noqa: F401
from ask import build_engine
from cameras import expand_env
from llm import Backend, Step, create_backend
from tools import TOOLS, Toolbox

MAX_TOOL_ROUNDS = 6

SYSTEM = """You answer questions about what security cameras recorded.
Everything you know about the footage comes from your tools: never invent sightings, times or counts.

Now: {now}
Recorded so far:
{coverage}

How to work:
- `objects` lists what the detector tracked, in time order, with counts. Exact.
  Time only: "what happened yesterday 6pm to 9pm" -> objects(when="yesterday 6pm to 9pm").
  Count a class: "how many cars today" -> objects(labels=["car"], when="today").
  Together: "people with a dog" -> objects(labels=["person"], with_labels=["dog"]).
- `search` finds things by description (colors, clothing, actions, things with no class).
  Ranked guesses. Takes the same when / since / until / camera / with_labels filters.
- For exact times use since/until ("2026-10-02 18:00"), resolved against Now above.
- Run several lookups when a question has several parts; rephrase once if nothing matches.

How to answer:
- Short and concrete: camera, date and time, what was seen.
- Say how sure you are. Search scores are relative: a top hit is a candidate, not proof.
- If the tools found nothing, say so, and say what time range and cameras you checked."""


def model_config(config: dict[str, Any], name: str) -> dict[str, Any]:
    models = (config.get("chat") or {}).get("models") or {}
    if name not in models:
        raise SystemExit(f"No chat model {name!r}. Configured: {', '.join(models) or 'none'}")
    return {k: expand_env(v) if isinstance(v, str) else v for k, v in models[name].items()}


def system_prompt(toolbox: Toolbox) -> str:
    coverage = toolbox.coverage()
    lines = [f"- {c['camera']}: {c['from']} to {c['to']}" for c in coverage] or ["- nothing yet"]
    return SYSTEM.format(now=datetime.now().strftime("%A %Y-%m-%d %H:%M"),
                         coverage="\n".join(lines))


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with what the cameras saw")
    parser.add_argument("question", nargs="*")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--model", help="Entry of chat.models (default: chat.model)")
    parser.add_argument("--quiet", action="store_true", help="Don't print tool calls")
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(message)s")
    log = logging.getLogger("histor")
    config = yaml.safe_load(args.config.read_text())
    engine = build_engine(config)
    toolbox = Toolbox(engine)
    media_dir = engine.store.media_dir

    def start(name: str) -> Backend:
        backend = create_backend(model_config(config, name), system_prompt(toolbox), TOOLS)
        log.info("Model: %s (%s)", name, backend.name)
        return backend

    def ask(question: str) -> None:
        toolbox.evidence.clear()
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
            else:
                ask(line)
    except (EOFError, KeyboardInterrupt):
        print()
    finally:
        toolbox.close()
        engine.store.close()


if __name__ == "__main__":
    main()
