"""Chat model backends behind one interface, so the model is a config switch.

    api: ollama     Ollama's own API, with the context size set per request (`num_ctx`). Its
                    OpenAI-compatible endpoint keeps Ollama's default (4096 tokens) and silently
                    cuts longer prompts: the chat's instructions and tool results get lost.
                    `think: false` turns a thinking model's reasoning off (faster replies).
    api: openai     Any OpenAI-compatible server. Local: llama.cpp server, LM Studio, vLLM.
                    Hosted ones work too (base_url + api_key). Plain HTTP, no SDK.
    api: anthropic  Claude through the Anthropic SDK (`uv sync --group claude`).

Each backend keeps the conversation in its own wire format; the agent loop in chat.py only sees
`step()` -> text + tool calls, and hands tool outputs back with `add_results()`.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema of the arguments


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class ToolResult:
    call: ToolCall
    output: str
    is_error: bool = False


@dataclass
class Step:
    text: str
    calls: list[ToolCall] = field(default_factory=list)


class Backend(Protocol):
    name: str

    def add_user(self, text: str) -> None: ...
    def step(self, tools: bool = True) -> Step: ...  # tools=False: an answer, no tool calls
    def add_results(self, results: list[ToolResult]) -> None: ...


def create_backend(config: dict[str, Any], system: str, tools: list[Tool]) -> Backend:
    """config: one entry of `chat.models` (see configs/default.yaml)."""
    options = dict(config)
    api = options.pop("api", "openai")
    options.pop("note", None)  # shown in the UI only
    if api == "ollama":
        return Ollama(system, tools, **options)
    if api == "openai":
        return OpenAICompatible(system, tools, **options)
    if api == "anthropic":
        return Claude(system, tools, **options)
    raise ValueError(f"Unknown chat api {api!r} (use ollama, openai or anthropic)")


# ------------------------------------------------------------------ OpenAI-compatible (local)

THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)  # reasoning some local models print


def post_json(url: str, body: dict[str, Any], api_key: str | None, timeout: float,
              model: str) -> dict[str, Any]:
    """POST to an OpenAI-compatible endpoint; errors become RuntimeErrors a user can act on."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, json.dumps(body).encode(), headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{url} returned {e.code}: {e.read().decode(errors='replace')}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Can't reach the model server at {url} ({e.reason}). "
                           "Is it running? For Ollama: `ollama serve`, then "
                           f"`ollama pull {model}`") from None


class OpenAICompatible:
    def __init__(self, system: str, tools: list[Tool], model: str,
                 base_url: str = "http://127.0.0.1:11434/v1", api_key: str | None = None,
                 temperature: float = 0.2, timeout: float = 300) -> None:
        self.name = model
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        self.tools = [{"type": "function", "function": {
            "name": t.name, "description": t.description, "parameters": t.parameters}}
            for t in tools]
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": system}]

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def step(self, tools: bool = True) -> Step:
        body = {"model": self.model, "messages": self.messages, "tools": self.tools,
                "temperature": self.temperature, "stream": False}
        if not tools:  # the tools stay listed: some servers want them while calls are in history
            body["tool_choice"] = "none"
        message = self._post(body)["choices"][0]["message"]
        calls = [ToolCall(c.get("id") or f"call_{len(self.messages)}_{i}",
                          c["function"]["name"], _arguments(c["function"].get("arguments")))
                 for i, c in enumerate(message.get("tool_calls") or [])]
        self.messages.append({k: v for k, v in message.items()
                              if k in {"role", "content", "tool_calls"}})
        return Step(THINK.sub("", message.get("content") or "").strip(), calls)

    def add_results(self, results: list[ToolResult]) -> None:
        for r in results:
            self.messages.append({"role": "tool", "tool_call_id": r.call.id, "content": r.output})

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        return post_json(self.url, body, self.api_key, self.timeout, self.model)


# ------------------------------------------------------------------ Ollama (local)

OLLAMA_CONTEXT = 16384  # tokens: instructions + tool schemas + a few results; Ollama's default is 4096


def ollama_root(base_url: str) -> str:
    """http://host:11434 from a base URL given with or without the OpenAI-style /v1."""
    return base_url.rstrip("/").removesuffix("/v1")


class Ollama(OpenAICompatible):
    """Ollama's /api/chat: the OpenAI-compatible conversation, plus `num_ctx` per request."""

    def __init__(self, system: str, tools: list[Tool], model: str,
                 base_url: str = "http://127.0.0.1:11434", api_key: str | None = None,
                 temperature: float = 0.2, timeout: float = 300,
                 num_ctx: int = OLLAMA_CONTEXT, think: bool | None = None) -> None:
        """think: None = the model's default; false skips a thinking model's reasoning."""
        super().__init__(system, tools, model, base_url, api_key, temperature, timeout)
        self.url = ollama_root(base_url) + "/api/chat"
        self.num_ctx = int(num_ctx)
        self.think = think

    def step(self, tools: bool = True) -> Step:
        body = {"model": self.model, "messages": self.messages, "stream": False,
                "options": {"temperature": self.temperature, "num_ctx": self.num_ctx}}
        if tools:  # Ollama has no tool_choice: no tools listed, no calls
            body["tools"] = self.tools
        if self.think is not None:
            body["think"] = bool(self.think)
        message = self._post(body)["message"]
        calls = [ToolCall(c.get("id") or f"call_{len(self.messages)}_{i}",
                          c["function"]["name"], _arguments(c["function"].get("arguments")))
                 for i, c in enumerate(message.get("tool_calls") or [])]
        self.messages.append({k: v for k, v in message.items()
                              if k in {"role", "content", "tool_calls"}})
        return Step(THINK.sub("", message.get("content") or "").strip(), calls)

    def add_results(self, results: list[ToolResult]) -> None:
        for r in results:
            self.messages.append({"role": "tool", "content": r.output, "tool_name": r.call.name,
                                  "tool_call_id": r.call.id})


def _arguments(args: Any) -> dict[str, Any]:
    """Tool call arguments: the JSON string the OpenAI spec uses, or an object (Ollama, some
    other servers)."""
    if isinstance(args, dict):
        return args
    try:
        parsed = json.loads(args) if isinstance(args, str) and args.strip() else {}
    except json.JSONDecodeError:
        return {"_invalid_json": args}
    return parsed if isinstance(parsed, dict) else {"_invalid_json": args}


# ------------------------------------------------------------------ Claude

# Claude's safety classifiers can decline a request; this re-runs it on Anthropic's
# recommended fallback model instead of returning a refusal.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class Claude:
    def __init__(self, system: str, tools: list[Tool], model: str = "claude-opus-5-5",
                 effort: str = "low", max_tokens: int = 16000, api_key: str | None = None) -> None:
        try:
            import anthropic
        except ImportError:
            raise SystemExit("Claude needs the SDK: uv sync --group claude") from None
        # No api_key: the SDK reads ANTHROPIC_API_KEY or an `ant auth login` profile.
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.name = model
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.system = system
        self.tools = [{"name": t.name, "description": t.description, "input_schema": t.parameters}
                      for t in tools]
        self.messages: list[dict[str, Any]] = []

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def step(self, tools: bool = True) -> Step:
        response = self.client.beta.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=self.system, tools=self.tools,
            tool_choice={"type": "auto" if tools else "none"},
            messages=self.messages, output_config={"effort": self.effort},
            cache_control={"type": "ephemeral"}, betas=[FALLBACK_BETA], fallbacks="default",
        )
        # Append the full content (thinking blocks included), never a rewritten copy.
        self.messages.append({"role": "assistant", "content": response.content})
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        if response.stop_reason == "refusal":
            return Step(text or "(The model declined to answer this question.)")
        if response.stop_reason == "max_tokens":
            raise RuntimeError(f"Answer cut off at max_tokens={self.max_tokens}")
        calls = [ToolCall(b.id, b.name, dict(b.input))
                 for b in response.content if b.type == "tool_use"]
        return Step(text, calls)

    def add_results(self, results: list[ToolResult]) -> None:
        self.messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": r.call.id, "content": r.output,
             "is_error": r.is_error} for r in results]})
