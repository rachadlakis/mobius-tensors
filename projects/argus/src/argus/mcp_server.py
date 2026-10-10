"""The chat's tools as an MCP server, so any MCP client (Claude Desktop, Claude Code, a client's
own agent) can ask a site's Argus what happened.

    uv run --group mcp -m argus.mcp_server --config configs/default.yaml

It speaks MCP over stdin/stdout: the client starts it as a command. From a laptop, the command
can be `ssh argus@site-server "cd /opt/argus/src/projects/argus && uv run --group mcp -m
argus.mcp_server --config configs/site.yaml"`: nothing new is exposed on the network.

The tools are the chat's (tools.py), with the same descriptions, arguments and JSON results:
objects, search, alerts, people, plates and look, where they are enabled. All are read-only.
Left out: `face`, which needs a photo attached in the chat and is biometric. Images never leave
the site: results carry thumbnail paths, and `look` answers about them in text with the local
vision model. With a cloud model behind the client, what it receives is that text, as with the
chat's `claude` model.
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

from argus.ask import build_engine
from argus.chat import system_prompt
from argus.llm import ToolCall
from argus.tools import Toolbox

log = logging.getLogger(__name__)

NOTE = """
You reach this site's Argus through MCP tools. "Now" above is when the server started: every
tool result carries `site_time`, the site's current time; resolve "today", "last night"... against
it. Results list thumbnail and snapshot paths as evidence: pass them to `look` to have the site's
own vision model read them; the images themselves are never sent."""


def build_toolbox(config: dict[str, Any]) -> Toolbox:
    """The chat's toolbox without face search (no photo attachments over MCP)."""
    config = {**config, "faces": {**(config.get("faces") or {}), "enabled": False}}
    with redirect_stdout(sys.stderr):  # stdout is the MCP channel: keep stray prints off it
        engine = build_engine(config)
    return Toolbox.from_config(engine, config)


def create_server(toolbox: Toolbox) -> Server:
    lock = threading.Lock()  # one call at a time: the toolbox keeps per-session state

    def run(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        with lock:
            result = toolbox.run(ToolCall(id="mcp", name=name, args=arguments))
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
            for t in toolbox.tools])

    async def call_tool(ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        log.info("-> %s %s", params.name, params.arguments)
        return await anyio.to_thread.run_sync(run, params.name, dict(params.arguments or {}))

    return Server("argus", version="0.1.0", title="Argus",
                  instructions=system_prompt(toolbox) + NOTE,
                  on_list_tools=list_tools, on_call_tool=call_tool)


def main() -> None:
    parser = argparse.ArgumentParser(description="Argus's chat tools as an MCP server (stdio)")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args()
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    toolbox = build_toolbox(config)
    server = create_server(toolbox)
    log.info("Argus MCP server: %s", ", ".join(t.name for t in toolbox.tools))

    async def serve() -> None:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    try:
        anyio.run(serve)
    finally:
        toolbox.close()


if __name__ == "__main__":
    main()
