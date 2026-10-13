"""Smoke tests with a real local model (Ollama) instead of the fake one: the chat's tool calls,
`look`, alert checking, a report's summary and the Telegram bot (against the fake Telegram
server), on the recorded smoke site.

    ollama serve; ollama pull qwen3.5:2b-q4_K_M   # tools + images: chat, look and checking
    uv run pytest -m smoke tests/smoke/test_ollama.py

HISTOR_SMOKE_MODEL picks another model. Answers vary from run to run: the tests check that
each part works (tools called, answers given, verdicts parsed), not what the model says.
"""

from __future__ import annotations

import json
import os
import urllib.request

import cv2
import pytest
import yaml
from smoke_kit import MISSING, TOKEN, FakeTelegram, run

OLLAMA = "http://127.0.0.1:11434"
MODEL = os.environ.get("HISTOR_SMOKE_MODEL", "qwen3.5:2b-q4_K_M")


def pulled() -> list[str]:
    try:
        with urllib.request.urlopen(OLLAMA + "/v1/models", timeout=2) as response:
            return [m["id"] for m in json.loads(response.read())["data"]]
    except OSError:
        return []


pytestmark = [pytest.mark.smoke,
              pytest.mark.skipif(bool(MISSING), reason=f"needs models/ and data/: {MISSING}"),
              pytest.mark.skipif(MODEL not in pulled(),
                                 reason=f"needs `ollama serve` and `ollama pull {MODEL}`")]


@pytest.fixture(scope="module")
def local_site(site, tmp_path_factory):
    """The smoke site's config with the real model for the chat and the vision checks."""
    config = {**site.config, "chat": {
        "model": "local",
        "models": {"local": {"api": "ollama", "base_url": OLLAMA, "model": MODEL,
                             "temperature": 0.2}},
        "vision": {"api": "ollama", "model": MODEL, "base_url": OLLAMA, "max_side": 512}}}
    path = tmp_path_factory.mktemp("ollama") / "site.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config, path


def test_chat_calls_tools_and_answers(local_site):
    _, path = local_site
    r = run("-m", "histor.chat", "--config", path,
            "How many people did the street camera see? Look at one of them and say what "
            "they are wearing.", timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
    calls = [line.split()[1] for line in r.stderr.splitlines() if line.lstrip().startswith("->")]
    assert calls and set(calls) <= {"objects", "search", "people", "alerts", "look"}, calls
    answer = r.stdout.split("Evidence:")[0].strip()
    assert answer and not answer.startswith(("Error", "(Stopped")), r.stdout


def test_look_describes_an_evidence_image(local_site):
    from histor.ask import build_engine
    from histor.llm import ToolCall
    from histor.tools import Toolbox

    config, _ = local_site
    toolbox = Toolbox.from_config(build_engine(config), config)
    try:  # through run(), as the chat does: look only opens images a result listed
        found = json.loads(toolbox.run(ToolCall("1", "objects", {"labels": ["person"],
                                                                  "limit": 3})).output)
        thumb = next(v["thumb"] for v in found["visits"] if v["thumb"])
        result = toolbox.run(ToolCall("2", "look", {
            "images": [thumb],
            "question": "Is there a person in this picture? Answer yes or no first."}))
        assert not result.is_error, result.output
        out = json.loads(result.output)
        assert out["model"] == MODEL and out["answer"] and out["answer"] != "(no answer)"
        assert "yes" in out["answer"].lower()  # an object thumb of a tracked person
    finally:
        toolbox.close()
        toolbox.engine.store.close()


def test_alert_checking_gives_a_verdict(local_site, site):
    from histor.engine.store import EventStore
    from histor.engine.verify import AlertVerifier

    config, _ = local_site
    store = EventStore(site.store)
    try:
        event = next(e for e in store.list_events(types=["enter"], limit=50) if e.get("snapshot"))
        snapshot = cv2.imread(str(store.media_path(event["snapshot"])))
    finally:
        store.close()
    verifier = AlertVerifier.from_config({"enabled": True, "timeout": 300},
                                         config["chat"]["vision"])
    assert verifier is not None
    verdicts: list[tuple[str, str]] = []
    verifier.submit(event, [snapshot], lambda payload, v, reason: verdicts.append((v, reason)))
    verifier.close(timeout=600)
    # a parsed verdict, not "error"; small models sometimes skip the reason line (allowed)
    assert verdicts and verdicts[0][0] in ("yes", "no", "unsure"), verdicts


def test_report_has_a_written_summary(local_site, site):
    _, path = local_site
    r = run("-m", "histor.report", "--config", path, "--now", "smoke", "--no-send",
            timeout=1200)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "# smoke: hourly report" in r.stdout and "No written summary" not in r.stdout


def test_telegram_bot_answers_with_the_model(local_site, site):
    from histor.ask import build_engine
    from histor.chat import model_config, system_prompt
    from histor.engine.telegram import TelegramAPI
    from histor.llm import create_backend
    from histor.telegram_bot import MODEL_DOWN, TelegramBot
    from histor.tools import Toolbox

    config, _ = local_site
    engine = build_engine(config)
    toolbox = Toolbox.from_config(engine, config)
    model = model_config(config, "local")
    tg = FakeTelegram()  # its own: the session's fake keeps the recording's alerts for test_smoke
    bot = TelegramBot(TelegramAPI(TOKEN, tg.url), toolbox,
                      lambda: create_backend(model, system_prompt(toolbox), toolbox.tools),
                      {"42"}, images=True, cameras=["street"], face_search=True)
    tools: list[str] = []
    run_tool = toolbox.run
    toolbox.run = lambda call: tools.append(call.name) or run_tool(call)  # type: ignore[method-assign]
    try:
        def say(update: dict) -> str:
            tg.requests.clear()
            bot.handle({"update_id": 1, "message": {"chat": {"id": 42}, **update}})
            (reply,) = [r["fields"]["text"] for r in tg.sent("sendMessage")]
            assert reply and reply not in (MODEL_DOWN, "(no answer)"), reply
            assert not reply.startswith("(Stopped"), reply
            return reply

        say({"text": "How many people did the street camera see?"})
        assert tools, "the model answered without looking anything up"
        say({"text": "And when was the first one?"})  # same conversation: tool results kept
        tg.file = site.face.read_bytes()
        say({"photo": [{"file_id": "big"}], "caption": "today"})  # face search, no model
        tools.clear()
        # explicit, so small models call the tool too: this checks the plumbing (the photo's
        # name told to the model, the face tool run for it), not the model's judgement
        say({"text": "Search by face: was the person in my photo on the street camera?"})
        assert "face" in tools, tools
    finally:
        tg.close()
        toolbox.close()
        engine.store.close()
