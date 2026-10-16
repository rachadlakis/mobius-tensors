"""
Lightweight ADK callbacks shared by every employee.

Two concerns, nothing more - we deliberately do NOT add content-filtering or
response-rewriting callbacks:

1. **Observability.** A structured log line for every tool call (which agent,
   which tool, how long, ok / failed) and every model turn (token usage). This
   is what turns a slow delegation - e.g. the Chief of Staff waiting on the
   Researcher - from a black box into something you can read in the logs.

2. **A hard fetch budget for the Researcher.** Its instruction already asks
   it to stay near ~4 page fetches, but a prompt is not a guarantee: a looping
   agent would keep piling up robots.txt lookups + politeness delays until the
   Chief of Staff's A2A client times out. ``enforce_scrape_budget`` caps it in
   code (``WEB_INTEL_MAX_FETCHES_PER_TASK``).

Callbacks here never raise - a logging bug must not take down a request - and
they return ``None`` (don't touch the model/tool result) except the budget
guard, which returns a dict to skip an over-budget fetch.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from google.adk.tools.base_tool import BaseTool
from google.genai import types

from shared.config import WEB_INTEL_MAX_FETCHES_PER_TASK

logger = logging.getLogger("ergon.employees")

# function_call_id -> monotonic start time. Module-level (one OS process per
# agent server), so tool timing never has to be written into session state / the
# database. after_tool pops its entry; a tool that dies before after_tool runs
# leaks one float, which is negligible.
_tool_starts: dict[str, float] = {}

# invocation_id -> web fetches used so far, for the Researcher's fetch budget. Cleared by
# clear_scrape_budget (after_agent_callback) at the end of each request.
_fetch_counts: dict[str, int] = {}

# Tools that fetch web pages. An employee with any of them gets the fetch budget.
FETCH_TOOLS = frozenset({"fetch_page", "web_search"})

# Mirrors the read=True clamp in web_tools.web_search - keep the two in step or
# the budget bills for pages that were never fetched.
_WEB_SEARCH_READ_CAP = 5


def _short_args(args: dict[str, Any], limit: int = 160) -> str:
    text = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _call_key(ctx: Any) -> str:
    return getattr(ctx, "function_call_id", None) or getattr(ctx, "invocation_id", "?")


# --------------------------------------------------------------------------- #
# Observability - used by every agent
# --------------------------------------------------------------------------- #
def log_before_tool(
    tool: BaseTool, args: dict[str, Any], tool_context: Any
) -> Optional[dict]:
    """Record the start time and log the call. Never blocks the tool."""
    try:
        _tool_starts[_call_key(tool_context)] = time.monotonic()
        agent = getattr(tool_context, "agent_name", "?")
        logger.info("[tool] %s -> %s(%s)", agent, tool.name, _short_args(args))
    except Exception:  # pragma: no cover - logging must not break a request
        logger.debug("log_before_tool failed", exc_info=True)
    return None


def log_after_tool(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: Any,
    tool_response: Any,
) -> Optional[dict]:
    """Log the call's duration and whether it succeeded. Never rewrites the
    response (returns None)."""
    try:
        start = _tool_starts.pop(_call_key(tool_context), None)
        secs = f"{time.monotonic() - start:.1f}s" if start is not None else "?"
        agent = getattr(tool_context, "agent_name", "?")
        failed = (
            isinstance(tool_response, dict) and tool_response.get("success") is False
        )
        if failed:
            logger.warning(
                "[tool] %s <- %s %s FAILED: %s",
                agent,
                tool.name,
                secs,
                tool_response.get("error", ""),
            )
        else:
            logger.info("[tool] %s <- %s %s ok", agent, tool.name, secs)
    except Exception:  # pragma: no cover
        logger.debug("log_after_tool failed", exc_info=True)
    return None


def log_after_model(callback_context: Any, llm_response: Any) -> None:
    """Log token usage for the turn - cheap cost visibility, and the hook a
    future multi-tenant build would use for per-tenant metering."""
    try:
        usage = getattr(llm_response, "usage_metadata", None)
        if usage is not None:
            logger.info(
                "[model] %s tokens prompt=%s output=%s total=%s",
                getattr(callback_context, "agent_name", "?"),
                getattr(usage, "prompt_token_count", "?"),
                getattr(usage, "candidates_token_count", "?"),
                getattr(usage, "total_token_count", "?"),
            )
    except Exception:  # pragma: no cover
        logger.debug("log_after_model failed", exc_info=True)
    return None


# --------------------------------------------------------------------------- #
# Every agent - keep un-ingestible file blobs out of the LLM request
#
# adk web puts an upload on the user turn as an inline blob, and `send_message`
# forwards files to a worker as inline blobs too. The model can't read a binary
# file's bytes anyway, and for an unsupported MIME type (a `.xlsx`, say) the
# LiteLlm client RAISES while building the request - before the first tool call,
# so the whole turn dies. This callback swaps each such part for a short text
# note. The bytes are not lost: the Chief of Staff keeps every upload in
# `session_files`, and an executor can save forwarded files as named
# artifacts for file-reading tools (BaseA2AExecutor._save_incoming_files).
# --------------------------------------------------------------------------- #
# What an inline part can safely carry into the model.
_LLM_SAFE_INLINE_PREFIXES = ("image/", "text/")
_LLM_SAFE_INLINE_MIMES = {"application/pdf", "application/json"}


def strip_unreadable_file_parts(callback_context: Any, llm_request: Any) -> None:
    """before_model_callback: replace inline file blobs the model can't ingest
    with a text note (filename + which tool opens it). Never raises."""
    try:
        contents = getattr(llm_request, "contents", None) or []
        for idx, content in enumerate(contents):
            parts = getattr(content, "parts", None) or []
            if not any(
                isinstance(getattr(getattr(p, "inline_data", None), "data", None),
                           (bytes, bytearray))
                for p in parts
            ):
                continue
            new_parts = []
            for part in parts:
                blob = getattr(part, "inline_data", None)
                data = getattr(blob, "data", None) if blob else None
                mime = (getattr(blob, "mime_type", "") or "").lower()
                if not isinstance(data, (bytes, bytearray)) or (
                    mime in _LLM_SAFE_INLINE_MIMES
                    or mime.startswith(_LLM_SAFE_INLINE_PREFIXES)
                ):
                    new_parts.append(part)
                    continue
                name = getattr(blob, "display_name", None) or "attached_file"
                new_parts.append(types.Part(text=(
                    f"[attached file: {name} ({mime or 'unknown type'})] The file's "
                    f"contents are not shown here. Open it with a file-aware tool "
                    f"by that name if you have one."
                )))
            # Replace the list entry rather than mutating `content` in place -
            # the same Content object is referenced elsewhere (the orchestrator's
            # user_content, which _remember_uploads still needs intact).
            contents[idx] = types.Content(
                role=getattr(content, "role", None) or "user", parts=new_parts
            )
    except Exception:  # pragma: no cover - a callback must not break a request
        logger.debug("strip_unreadable_file_parts failed", exc_info=True)
    return None


# --------------------------------------------------------------------------- #
# Researcher only - code-enforced fetch budget
# --------------------------------------------------------------------------- #
def enforce_scrape_budget(
    tool: BaseTool, args: dict[str, Any], tool_context: Any
) -> Optional[dict]:
    """Skip a web fetch once this request has already fetched too many pages.

    Returning a dict here makes ADK skip the tool and hand this dict back to the
    model as the tool result - so the agent is told, in-band, to stop fetching
    and answer with what it has.
    """
    if tool.name not in FETCH_TOOLS:
        return None

    # What will this call actually fetch?
    #   fetch_page                 -> 1 page
    #   web_search(read=True)      -> up to max_results pages (default 2)
    #   web_search(read=False)     -> 0 pages (just snippets) - not charged
    if tool.name == "web_search":
        if not args.get("read"):
            return None
        # web_search clamps max_results to 5 when read=True, so charging the
        # raw argument would bill 99 pages for a call that fetches 5.
        try:
            requested = int(args.get("max_results") or 2)
        except (TypeError, ValueError):
            requested = 2
        cost = min(max(1, requested), _WEB_SEARCH_READ_CAP)
    else:
        cost = 1

    inv = getattr(tool_context, "invocation_id", "?")
    used = _fetch_counts.get(inv, 0)
    if used >= WEB_INTEL_MAX_FETCHES_PER_TASK:
        logger.warning(
            "[researcher] fetch budget exhausted (%s/%s) - skipping %s",
            used,
            WEB_INTEL_MAX_FETCHES_PER_TASK,
            tool.name,
        )
        return {
            "success": False,
            "error": (
                f"Fetch budget for this request is exhausted "
                f"({used}/{WEB_INTEL_MAX_FETCHES_PER_TASK} pages). Do not fetch "
                f"more - answer from what you have already retrieved, and say "
                f"which parts you could not cover."
            ),
        }
    _fetch_counts[inv] = used + cost
    return None


def clear_scrape_budget(callback_context: Any) -> None:
    """after_agent_callback: drop this request's fetch counter."""
    _fetch_counts.pop(getattr(callback_context, "invocation_id", "?"), None)
    return None


# --------------------------------------------------------------------------- #
# Ready-to-spread kwargs for LlmAgent(...)
# --------------------------------------------------------------------------- #
# before_* / after_* accept a list; list order is execution order.
WORKER_CALLBACKS: dict[str, Any] = {
    "before_model_callback": [strip_unreadable_file_parts],
    "before_tool_callback": [log_before_tool],
    "after_tool_callback": [log_after_tool],
    "after_model_callback": [log_after_model],
}

WEB_INTEL_CALLBACKS: dict[str, Any] = {
    "before_model_callback": [strip_unreadable_file_parts],
    "before_tool_callback": [enforce_scrape_budget, log_before_tool],
    "after_tool_callback": [log_after_tool],
    "after_model_callback": [log_after_model],
    "after_agent_callback": clear_scrape_budget,
}
