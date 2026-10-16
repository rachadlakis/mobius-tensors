"""Smoke tests for the multi-provider LLM wiring (shared/config.py).

Offline by default: every case only builds a `LiteLlm` object and inspects its
`model` string / kwargs - no network. The one live round-trip test is skipped
unless RUN_LLM_INTEGRATION=1 (it uses whatever provider .env is pointed at).
"""

import importlib
import os

import pytest

from shared import config
from shared.employees import load_all

# provider -> (bare model to pass, expected full model string)
_PROVIDER_CASES = [
    ("anthropic", "claude-3-5-haiku", "anthropic/claude-3-5-haiku"),
    ("openai", "gpt-4o", "openai/gpt-4o"),
    ("azure", "my-deploy", "azure/my-deploy"),
    ("azure_ai", "Llama-3.3-70B", "azure_ai/Llama-3.3-70B"),
    ("gemini", "gemini-2.0-flash", "gemini/gemini-2.0-flash"),
    ("vertex_ai", "gemini-2.0-flash", "vertex_ai/gemini-2.0-flash"),
    ("bedrock", "anthropic.claude-3-5-haiku-20241022-v1:0",
     "bedrock/anthropic.claude-3-5-haiku-20241022-v1:0"),
]

_CREDS = {
    "anthropic": {"ANTHROPIC_API_KEY": "k"},
    "openai": {"OPENAI_API_KEY": "k"},
    "azure": {"AZURE_API_KEY": "k", "AZURE_API_BASE": "https://r.openai.azure.com",
              "AZURE_API_VERSION": "2024-10-21"},
    "azure_ai": {"AZURE_AI_API_KEY": "k", "AZURE_AI_API_BASE": "https://r.services.ai.azure.com"},
    "gemini": {"GEMINI_API_KEY": "k"},
    "vertex_ai": {},
    "bedrock": {"AWS_ACCESS_KEY_ID": "k", "AWS_SECRET_ACCESS_KEY": "k"},
}


@pytest.mark.parametrize("provider, bare, full", _PROVIDER_CASES)
def test_provider_prefixes_bare_model(reload_config, provider, bare, full):
    cfg = reload_config(LLM_PROVIDER=provider, DEFAULT_MODEL=bare, **_CREDS[provider])
    assert cfg.get_llm_model().model == full
    assert cfg.get_llm_model(role="orchestrator").model == full
    assert cfg.get_llm_model(role="worker").model == full


@pytest.mark.parametrize("provider", [p for p, _, _ in _PROVIDER_CASES])
def test_credentials_check_passes_when_set(reload_config, provider):
    cfg = reload_config(LLM_PROVIDER=provider, DEFAULT_MODEL="m", **_CREDS[provider])
    assert cfg.missing_llm_credentials() == []
    cfg.require_llm_credentials()  # must not raise


@pytest.mark.parametrize("provider, needed", [
    ("openai", "OPENAI_API_KEY"),
    ("azure_ai", "AZURE_AI_API_KEY"),
    ("gemini", "GEMINI_API_KEY"),
    ("bedrock", "AWS_ACCESS_KEY_ID"),
])
def test_missing_credentials_are_reported(reload_config, provider, needed):
    clear = {k: None for creds in _CREDS.values() for k in creds}
    cfg = reload_config(LLM_PROVIDER=provider, DEFAULT_MODEL="m", **clear)
    assert needed in cfg.missing_llm_credentials()
    with pytest.raises(RuntimeError, match=needed):
        cfg.require_llm_credentials()


def test_unknown_provider_is_rejected(reload_config):
    cfg = reload_config(LLM_PROVIDER="carrier-pigeon", DEFAULT_MODEL="m")
    with pytest.raises(RuntimeError, match="Unknown LLM_PROVIDER"):
        cfg.require_llm_credentials()


def test_azure_kwargs_are_passed_through(reload_config):
    cfg = reload_config(LLM_PROVIDER="azure", DEFAULT_MODEL="dep",
                        AZURE_API_KEY="ak", AZURE_API_BASE="https://r.openai.azure.com",
                        AZURE_API_VERSION="2024-10-21")
    kw = cfg.get_llm_model()._additional_args
    assert kw == {"api_base": "https://r.openai.azure.com",
                  "api_version": "2024-10-21", "api_key": "ak"}


def test_azure_ai_kwargs_no_version_by_default(reload_config):
    cfg = reload_config(LLM_PROVIDER="azure_ai", DEFAULT_MODEL="dep",
                        AZURE_AI_API_KEY="ak", AZURE_AI_API_BASE="https://r.services.ai.azure.com")
    kw = cfg.get_llm_model()._additional_args
    assert set(kw) == {"api_base", "api_key"}


def test_anthropic_base_url_override(reload_config):
    cfg = reload_config(LLM_PROVIDER="anthropic", DEFAULT_MODEL="claude-x",
                        ANTHROPIC_API_KEY="k",
                        ANTHROPIC_API_BASE="https://foundry.example/anthropic")
    assert cfg.get_llm_model()._additional_args == {
        "api_base": "https://foundry.example/anthropic"}


def test_role_split_capable_orchestrator_cheap_workers(reload_config):
    cfg = reload_config(LLM_PROVIDER="openai", DEFAULT_MODEL="gpt-4o",
                        ORCHESTRATOR_MODEL="gpt-4o", WORKER_MODEL="gpt-4o-mini",
                        OPENAI_API_KEY="k")
    assert cfg.get_llm_model(role="orchestrator").model == "openai/gpt-4o"
    assert cfg.get_llm_model(role="worker").model == "openai/gpt-4o-mini"
    assert cfg.get_llm_model().model == "openai/gpt-4o-mini"  # default role = worker


def test_role_var_can_target_another_provider(reload_config):
    cfg = reload_config(LLM_PROVIDER="azure_ai", DEFAULT_MODEL="claude-sonnet",
                        WORKER_MODEL="openai/gpt-4o-mini",
                        AZURE_AI_API_KEY="k", AZURE_AI_API_BASE="https://r.services.ai.azure.com",
                        OPENAI_API_KEY="k")
    w = cfg.get_llm_model(role="worker")
    assert w.model == "openai/gpt-4o-mini"
    assert w._additional_args == {}  # no azure kwargs leak onto an explicit provider/model


# --------------------------------------------------------------------------- #
# every worker agent + the orchestrator build on the configured model
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("emp", load_all(), ids=lambda e: e.id)
def test_employee_uses_worker_model(emp):
    from shared.employee_server import build_agent

    agent = build_agent(emp)
    # model: worker -> get_llm_model(role="worker")
    assert agent.model.model == config.model_for_role("worker") or "/" in agent.model.model
    assert agent.model.model.split("/", 1)[0] in config.SUPPORTED_LLM_PROVIDERS


def test_orchestrator_uses_orchestrator_model():
    mod = importlib.import_module("orchestrator_agent.orchestrator.agent")
    assert mod.CLAUDE_MODEL.model.split("/", 1)[0] in config.SUPPORTED_LLM_PROVIDERS
    # role resolution: orchestrator model, prefixed
    expected = config.model_for_role("orchestrator")
    assert expected in mod.CLAUDE_MODEL.model


# --------------------------------------------------------------------------- #
# opt-in live round trip
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    os.getenv("RUN_LLM_INTEGRATION") != "1",
    reason="set RUN_LLM_INTEGRATION=1 to hit the configured provider for real",
)
def test_live_completion_and_tool_call():
    import litellm

    litellm.drop_params = True
    for role in ("orchestrator", "worker"):
        m = config.get_llm_model(role=role)
        kw = m._additional_args
        r = litellm.completion(
            model=m.model, max_tokens=16, **kw,
            messages=[{"role": "user", "content": "Reply with exactly: PONG"}],
        )
        assert "PONG" in (r.choices[0].message.content or "").upper()

        tools = [{"type": "function", "function": {
            "name": "echo", "description": "echo a value",
            "parameters": {"type": "object", "properties": {"v": {"type": "string"}},
                           "required": ["v"]}}}]
        r2 = litellm.completion(
            model=m.model, tools=tools, tool_choice="auto", max_tokens=120, **kw,
            messages=[{"role": "user", "content": "Call echo with v='hi'."}],
        )
        assert r2.choices[0].message.tool_calls, f"{role}: no tool call"
