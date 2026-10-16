"""Tests for shared/config.py — env parsing, URL builders, MCP tool filter.

The `reload_config` fixture lives in conftest.py (shared with test_llm_smoke.py).
"""

import pytest

from shared import config


# --------------------------------------------------------------------------- #
# Model helper
# --------------------------------------------------------------------------- #
class TestGetLlmModel:
    def test_adds_provider_prefix_when_missing(self, reload_config):
        cfg = reload_config(LLM_PROVIDER="anthropic")
        model = cfg.get_llm_model("claude-haiku-4-5-20251001")
        assert model.model == "anthropic/claude-haiku-4-5-20251001"

    def test_keeps_existing_provider_prefix(self):
        model = config.get_llm_model("openai/gpt-4o")
        assert model.model == "openai/gpt-4o"

    def test_defaults_to_configured_model(self, reload_config):
        cfg = reload_config(LLM_PROVIDER="anthropic", DEFAULT_MODEL="claude-x")
        assert cfg.get_llm_model().model == "anthropic/claude-x"

    def test_role_split_orchestrator_capable_workers_cheap(self, reload_config):
        cfg = reload_config(
            LLM_PROVIDER="azure_ai",
            DEFAULT_MODEL="claude-sonnet-4-6",
            ORCHESTRATOR_MODEL="claude-opus-4-7",
            WORKER_MODEL="claude-sonnet-4-6",
            AZURE_AI_API_KEY="k", AZURE_AI_API_BASE="https://x.services.ai.azure.com",
        )
        assert cfg.get_llm_model(role="orchestrator").model == "azure_ai/claude-opus-4-7"
        assert cfg.get_llm_model(role="worker").model == "azure_ai/claude-sonnet-4-6"
        assert cfg.get_llm_model().model == "azure_ai/claude-sonnet-4-6"  # default role = worker

    def test_role_falls_back_to_default_model(self, reload_config):
        cfg = reload_config(LLM_PROVIDER="openai", DEFAULT_MODEL="gpt-4o",
                            ORCHESTRATOR_MODEL=None, WORKER_MODEL=None)
        assert cfg.get_llm_model(role="orchestrator").model == "openai/gpt-4o"
        assert cfg.get_llm_model(role="worker").model == "openai/gpt-4o"

    def test_role_model_can_cross_providers(self, reload_config):
        cfg = reload_config(LLM_PROVIDER="azure_ai", DEFAULT_MODEL="claude-opus-4-7",
                            WORKER_MODEL="openai/gpt-4o-mini",
                            AZURE_AI_API_KEY="k", AZURE_AI_API_BASE="https://x.services.ai.azure.com")
        assert cfg.get_llm_model(role="worker").model == "openai/gpt-4o-mini"

    def test_openai_provider_prefixes_bare_name(self, reload_config):
        cfg = reload_config(LLM_PROVIDER="openai", DEFAULT_MODEL="gpt-4o")
        assert cfg.get_llm_model().model == "openai/gpt-4o"

    def test_gemini_provider_default_model(self, reload_config):
        cfg = reload_config(LLM_PROVIDER="gemini", DEFAULT_MODEL=None)
        assert cfg.DEFAULT_MODEL == "gemini-2.0-flash"
        assert cfg.get_llm_model().model == "gemini/gemini-2.0-flash"

    def test_azure_provider_passes_base_and_version(self, reload_config):
        cfg = reload_config(
            LLM_PROVIDER="azure",
            DEFAULT_MODEL="my-gpt4o-deploy",
            AZURE_API_KEY="az-key",
            AZURE_API_BASE="https://res.openai.azure.com",
            AZURE_API_VERSION="2024-10-21",
        )
        m = cfg.get_llm_model()
        assert m.model == "azure/my-gpt4o-deploy"
        assert m._additional_args["api_base"] == "https://res.openai.azure.com"
        assert m._additional_args["api_version"] == "2024-10-21"
        assert m._additional_args["api_key"] == "az-key"

    def test_azure_ai_foundry_provider(self, reload_config):
        cfg = reload_config(
            LLM_PROVIDER="azure_ai",
            DEFAULT_MODEL="Llama-3.3-70B-Instruct",
            AZURE_AI_API_KEY="ai-key",
            AZURE_AI_API_BASE="https://res.services.ai.azure.com/models",
        )
        m = cfg.get_llm_model()
        assert m.model == "azure_ai/Llama-3.3-70B-Instruct"
        assert m._additional_args["api_base"].endswith("/models")
        assert "api_version" not in m._additional_args

    def test_explicit_slash_model_overrides_provider(self, reload_config):
        cfg = reload_config(LLM_PROVIDER="azure", DEFAULT_MODEL="anthropic/claude-haiku-4-5-20251001")
        assert cfg.get_llm_model().model == "anthropic/claude-haiku-4-5-20251001"


class TestLlmCredentials:
    def test_missing_credentials_lists_absent_vars(self, reload_config):
        cfg = reload_config(LLM_PROVIDER="openai", OPENAI_API_KEY=None)
        assert cfg.missing_llm_credentials() == ["OPENAI_API_KEY"]

    def test_no_missing_when_set(self, reload_config):
        cfg = reload_config(LLM_PROVIDER="openai", OPENAI_API_KEY="sk-x")
        assert cfg.missing_llm_credentials() == []

    def test_vertex_needs_no_env_vars(self, reload_config):
        cfg = reload_config(LLM_PROVIDER="vertex_ai")
        assert cfg.missing_llm_credentials() == []

    def test_require_raises_on_unknown_provider(self, reload_config):
        cfg = reload_config(LLM_PROVIDER="pigeon-post")
        with pytest.raises(RuntimeError, match="Unknown LLM_PROVIDER"):
            cfg.require_llm_credentials()

    def test_require_raises_when_credentials_missing(self, reload_config):
        cfg = reload_config(LLM_PROVIDER="azure_ai", AZURE_AI_API_KEY=None,
                            AZURE_AI_API_BASE=None, DEFAULT_MODEL="x")
        with pytest.raises(RuntimeError, match="AZURE_AI_API_KEY"):
            cfg.require_llm_credentials()

    def test_require_raises_when_no_model_for_azure(self, reload_config):
        cfg = reload_config(LLM_PROVIDER="azure", AZURE_API_KEY="k",
                            AZURE_API_BASE="https://x", DEFAULT_MODEL=None)
        with pytest.raises(RuntimeError, match="No model"):
            cfg.require_llm_credentials()

    def test_require_passes_for_anthropic_with_key(self, reload_config):
        cfg = reload_config(LLM_PROVIDER="anthropic", ANTHROPIC_API_KEY="sk-ant-x")
        cfg.require_llm_credentials()  # no raise


# --------------------------------------------------------------------------- #
# Host / port helpers
# --------------------------------------------------------------------------- #
class TestAgentUrls:
    def test_default_ports(self):
        assert config.AGENT_PORTS == {
            "data": 10003,
            "web_intel": 10005,
        }

    def test_agent_url_builds_from_host_and_port(self):
        assert config.agent_url("data") == f"http://{config.AGENT_HOST}:10003"
        assert config.agent_url("web_intel") == f"http://{config.AGENT_HOST}:10005"

    def test_agent_url_unknown_name_raises(self):
        with pytest.raises(KeyError):
            config.agent_url("nope")

    def test_ports_are_overridable_via_env(self, reload_config):
        cfg = reload_config(WEB_INTEL_PORT="20005", DATA_PORT="20003")
        assert cfg.WEB_INTEL_PORT == 20005
        assert cfg.AGENT_PORTS["data"] == 20003


# --------------------------------------------------------------------------- #
# Session backend config
# --------------------------------------------------------------------------- #
class TestSessionConfig:
    def test_defaults_to_memory(self, reload_config):
        cfg = reload_config(SESSION_BACKEND=None, SESSION_DB_URL=None)
        assert cfg.SESSION_BACKEND == "memory"
        assert cfg.SESSION_DB_URL.startswith("sqlite+aiosqlite:///")
        assert cfg.SESSION_DB_URL.endswith("ergon.db")

    def test_backend_is_lowercased_and_stripped(self, reload_config):
        cfg = reload_config(SESSION_BACKEND="  DataBase  ")
        assert cfg.SESSION_BACKEND == "database"

    def test_db_url_env_override(self, reload_config):
        cfg = reload_config(SESSION_DB_URL="postgresql://u:p@h:5432/db")
        assert cfg.SESSION_DB_URL == "postgresql://u:p@h:5432/db"

    def test_default_db_url_has_no_backslashes(self, reload_config):
        cfg = reload_config(SESSION_DB_URL=None)
        assert "\\" not in cfg.SESSION_DB_URL


# --------------------------------------------------------------------------- #
# Web search / scrape config
# --------------------------------------------------------------------------- #
class TestSearchConfig:
    def test_defaults(self, reload_config):
        cfg = reload_config(
            SEARCH_PROVIDER=None, SEARCH_MAX_RESULTS=None, SCRAPE_MAX_CHARS=None,
            SCRAPE_RESPECT_ROBOTS=None,
        )
        assert cfg.SEARCH_PROVIDER == "duckduckgo"
        assert cfg.SEARCH_MAX_RESULTS == 5
        assert cfg.SCRAPE_MAX_CHARS == 8000
        assert cfg.SCRAPE_RESPECT_ROBOTS is True

    @pytest.mark.parametrize("value, expected", [
        ("false", False), ("0", False), ("no", False), ("off", False),
        ("true", True), ("1", True), ("yes", True),
    ])
    def test_respect_robots_parsing(self, reload_config, value, expected):
        assert reload_config(SCRAPE_RESPECT_ROBOTS=value).SCRAPE_RESPECT_ROBOTS is expected

    def test_provider_lowercased(self, reload_config):
        assert reload_config(SEARCH_PROVIDER="  DuckDuckGo ").SEARCH_PROVIDER == "duckduckgo"


# --------------------------------------------------------------------------- #
# MCP tool filter
# --------------------------------------------------------------------------- #
class _FakeTool:
    def __init__(self, name):
        self.name = name


class TestCreateToolFilter:
    def test_returns_none_for_empty_list(self):
        assert config.create_tool_filter([]) is None

    def test_returns_none_when_wildcard_present(self):
        assert config.create_tool_filter(["*"]) is None

    def test_filter_accepts_two_arg_signature(self):
        """ADK calls tool_filter(tool, readonly_context) — must not raise."""
        f = config.create_tool_filter(["send_email"])
        assert f(_FakeTool("send_email"), None) is True
        assert f(_FakeTool("send_email")) is True  # readonly_context defaulted

    def test_filter_allows_only_listed_tools(self):
        f = config.create_tool_filter(["list_events"])
        assert f(_FakeTool("list_events"), None) is True
        assert f(_FakeTool("delete_everything"), None) is False
