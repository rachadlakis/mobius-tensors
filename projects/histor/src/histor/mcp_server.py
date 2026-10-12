"""The chat's tools as an MCP server, so any MCP client (Claude, ChatGPT, a client's
own agent) can ask a site's Histor what happened.

    uv run --group mcp -m histor.mcp_server --config configs/default.yaml

It speaks MCP over stdin/stdout: the client starts it as a command. From a laptop, the command
can be `ssh histor@site-server "cd /opt/histor/src/projects/histor && uv run --group mcp -m
histor.mcp_server --config configs/site.yaml"`: nothing new is exposed on the network.

The tools are the chat's (tools.py), with the same descriptions, arguments and JSON results:
objects, search, alerts, people, plates and look, where they are enabled. All are read-only.
Images never leave the site: results carry thumbnail paths, and `look` answers about them in text
with the local vision model. With a cloud model behind the client, what it receives is that text,
as with the chat's `claude` model.

Face search (opt-in: `mcp.face_search: true`, and `faces.enabled`): a `face` tool that takes the
path of a photo file on the Histor server (a photo attached in the client's chat can't reach a
tool), makes its face print as the chat's /photo does, and returns likely / possible visits. Off,
the instructions tell the model that face search exists elsewhere (chat, UI, Telegram), so it
never says Histor can't match faces.

    mcp:
      face_search: false        # biometric: the client's written approval first
      photo_dirs: []            # only photos under these folders; empty = any path
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Any

import anyio
import mcp_types as types
import yaml
from dotenv import load_dotenv
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from histor.ask import build_engine
from histor.chat import FACE_HELP, read_image, system_prompt
from histor.llm import Tool, ToolCall
from histor.tools import _WHEN, Toolbox

log = logging.getLogger(__name__)

NOTE = """
You reach this site's Histor through MCP tools. "Now" above is when the server started: every
tool result carries `site_time`, the site's current time; resolve "today", "last night"... against
it. Results list thumbnail and snapshot paths as evidence: pass them to `look` to have the site's
own vision model read them; the images themselves are never sent."""


FACE_OFF = """
- Face search by photo exists in Histor (its chat, test UI, API and Telegram bot) but is not
  available through this connection. If asked to find a person from a photo, say so, and offer to
  search by appearance with `search` and `people` instead. Never say Histor can't match faces."""

MCP_FACE_HELP = """
- `face` finds a person from a photo **file on the Histor server**: face(photo_path="C:/photos/
  visitor.jpg", when="today"). A photo attached in this chat can't be passed to a tool: ask the
  user for the file's path on the server. "likely" visits are strong matches; for "possible" ones
  tell the user to check the thumbnail. No match is not proof: say how many faces the checked
  cameras recorded (`checked`). A match is a lead for a person to check, never an identification."""

MCP_FACE_TOOL = Tool("face", (
    "Find a person from a photo file on the Histor server: compares the photo's face with the "
    "faces recorded on the cameras. Each visit has a level: likely (strong match over several "
    "frames) or possible (weak: check the thumbnail). `checked` says how many faces each camera "
    "recorded in the range: no match is not proof the person was not there. The photo must show "
    "one face, sharp and facing the camera, or the reason is returned."),
    {"type": "object", "properties": {
        "photo_path": {"type": "string",
                       "description": "Path of a JPEG / PNG on the Histor server, e.g. "
                                      "C:/photos/visitor.jpg"},
        "face": {"type": "integer",
                 "description": "Which face if the photo has several (1 = leftmost)"},
        **_WHEN,
        "limit": {"type": "integer", "description": "Max visits listed, default 10"},
    }, "required": ["photo_path"]})

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_PHOTO_BYTES = 20_000_000


def build_toolbox(config: dict[str, Any], face_search: bool = False) -> Toolbox:
    """The chat's toolbox; face search only with `mcp.face_search` (and `faces.enabled`)."""
    if not face_search:  # don't even load the face models
        config = {**config, "faces": {**(config.get("faces") or {}), "enabled": False}}
    with redirect_stdout(sys.stderr):  # stdout is the MCP channel: keep stray prints off it
        engine = build_engine(config)
    return Toolbox.from_config(engine, config)


def photo_file(value: Any, photo_dirs: list[Path]) -> Path:
    """A readable image file, inside `photo_dirs` when those are set."""
    path = Path(str(value or "")).expanduser()
    if not str(value or "").strip():
        raise ValueError("photo_path: give the path of a JPEG or PNG on the Histor server")
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"photo_path: {path.name} is not a photo (use .jpg, .jpeg or .png)")
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        raise ValueError(f"photo_path: {path} does not exist on the Histor server") from None
    if photo_dirs and not any(d == resolved.parent or d in resolved.parents for d in photo_dirs):
        raise ValueError("photo_path: only photos in these folders can be used: "
                         + ", ".join(str(d) for d in photo_dirs))
    if resolved.stat().st_size > MAX_PHOTO_BYTES:
        raise ValueError("photo_path: the file is too big for a photo")
    return resolved


def create_server(toolbox: Toolbox, face_search: bool = False,
                  photo_dirs: list[str] | None = None) -> Server:
    lock = threading.Lock()  # one call at a time: the toolbox keeps per-session state
    face_on = face_search and toolbox.engine.faces is not None
    dirs = [Path(d).expanduser().resolve() for d in photo_dirs or []]
    tools = [t for t in toolbox.tools if t.name != "face"] + ([MCP_FACE_TOOL] if face_on else [])

    def face_call(arguments: dict[str, Any]) -> ToolCall | str:
        """The photo file -> an attached face print -> the chat's `face` call (or an error)."""
        args = dict(arguments)
        try:
            image = read_image(photo_file(args.pop("photo_path", None), dirs))
        except ValueError as e:
            return f"Error: {e}"
        name, note = toolbox.add_photo([image], args.pop("face", None))
        if name is None:
            return f"Error: {note}"
        return ToolCall(id="mcp", name="face", args={"photo": name, **args})

    def run(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        with lock:
            call: ToolCall | str = ToolCall(id="mcp", name=name, args=arguments)
            if name == "face":
                call = face_call(arguments) if face_on else \
                    "Error: face search is not available through this connection"
            if isinstance(call, str):
                return types.CallToolResult(content=[types.TextContent(text=call)],
                                            is_error=True)
            result = toolbox.run(call)
        if result.is_error:
            text = result.output
        else:
            out = json.loads(result.output)
            out = {"site_time": datetime.now().isoformat(timespec="seconds"), **out} \
                if isinstance(out, dict) else out
            text = json.dumps(out, ensure_ascii=False)
        return types.CallToolResult(content=[types.TextContent(text=text)],
                                    is_error=result.is_error)

    async def list_tools(ctx: Any, params: Any) -> types.ListToolsResult:
        return types.ListToolsResult(tools=[
            types.Tool(name=t.name, description=t.description, input_schema=t.parameters,
                       annotations=types.ToolAnnotations(read_only_hint=True,
                                                         open_world_hint=False))
            for t in tools])

    async def call_tool(ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        log.info("-> %s %s", params.name, params.arguments)
        return await anyio.to_thread.run_sync(run, params.name, dict(params.arguments or {}))

    instructions = system_prompt(toolbox).replace(FACE_HELP, "") + \
        (MCP_FACE_HELP if face_on else FACE_OFF) + NOTE
    return Server("histor", version="0.1.0", title="Histor", instructions=instructions,
                  on_list_tools=list_tools, on_call_tool=call_tool)


def main() -> None:
    parser = argparse.ArgumentParser(description="Histor's chat tools as an MCP server (stdio)")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args()
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    settings = config.get("mcp") or {}
    face_search = bool(settings.get("face_search"))
    toolbox = build_toolbox(config, face_search)
    server = create_server(toolbox, face_search, settings.get("photo_dirs"))
    log.info("Histor MCP server: %s; face search %s", ", ".join(t.name for t in toolbox.tools),
             "on" if face_search and toolbox.engine.faces is not None else "off")

    async def serve() -> None:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    try:
        anyio.run(serve)
    finally:
        toolbox.close()


if __name__ == "__main__":
    main()
