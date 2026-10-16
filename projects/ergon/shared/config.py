import os

from dotenv import load_dotenv

load_dotenv()

# User context - fixed for now, change later for multi-tenant
USER_ID = os.getenv("USER_ID", "demo_user_001")
TENANT_ID = os.getenv("TENANT_ID", "demo_tenant_001")

## =============================================================================
# LLM provider
#
# Every agent talks to the model through LiteLLM (google.adk ... LiteLlm), which
# routes on a "<provider>/<model>" string and reads that provider's standard
# env vars. Pick the provider with LLM_PROVIDER and the model with DEFAULT_MODEL
# (a bare name gets the provider's prefix; a name that already contains "/" is
# used verbatim, so you can still point one agent at another provider by hand).
#
# Split by role to save money. The orchestrator makes the routing/delegation
# decisions and merges the workers' replies, so it runs on the MOST CAPABLE
# model; the workers run bounded, well-scoped tasks, so they can run on a
# CHEAPER model:
#   ORCHESTRATOR_MODEL - routing / delegation / merging  (capable)
#   WORKER_MODEL       - the workers' execution          (cheaper)
# Each falls back to DEFAULT_MODEL when unset, so a single DEFAULT_MODEL still
# works. All roles use the SAME LLM_PROVIDER; put a "provider/model" string in a
# role var only if you deliberately want that role on a different provider.
#
#   LLM_PROVIDER   DEFAULT_MODEL (example)              credentials in .env
#   ------------   ---------------------------------    ----------------------------------------
#   anthropic      claude-haiku-4-5-20251001            ANTHROPIC_API_KEY
#   openai         gpt-4o                               OPENAI_API_KEY
#   azure          <your deployment name>               AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION
#   azure_ai       <your Foundry model/deployment>      AZURE_AI_API_KEY, AZURE_AI_API_BASE
#   gemini         gemini-2.0-flash                     GEMINI_API_KEY
#   vertex_ai      gemini-2.0-flash                     gcloud ADC (+ VERTEXAI_PROJECT / VERTEXAI_LOCATION)
#   bedrock        anthropic.claude-3-5-haiku-...-v1:0  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION_NAME
#
#   azure    = Azure OpenAI (GPT models you deployed in Azure OpenAI)
#   azure_ai = Azure AI Foundry serverless models (Llama, Mistral, Phi, DeepSeek,
#              Cohere, and OpenAI/Anthropic models served from an AI Foundry project)
## =============================================================================

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()

# A sensible model per provider, used when DEFAULT_MODEL is unset. Empty means
# "you must set DEFAULT_MODEL yourself" (Azure needs the deployment name).
_PROVIDER_DEFAULT_MODEL = {
    "anthropic": "claude-haiku-4-5-20251001",
    "openai": "gpt-4o",
    "azure": "",
    "azure_ai": "",
    "gemini": "gemini-2.0-flash",
    "vertex_ai": "gemini-2.0-flash",
    "bedrock": "anthropic.claude-3-5-haiku-20241022-v1:0",
}

# Env vars each provider needs. LiteLLM reads them itself at call time; this map
# is only for a fast, clear failure at server startup (require_llm_credentials).
_PROVIDER_REQUIRED_ENV = {
    "anthropic": ["ANTHROPIC_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "azure": ["AZURE_API_KEY", "AZURE_API_BASE"],
    "azure_ai": ["AZURE_AI_API_KEY", "AZURE_AI_API_BASE"],
    "gemini": ["GEMINI_API_KEY"],
    "vertex_ai": [],  # gcloud Application Default Credentials, checked by the SDK
    "bedrock": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
}

SUPPORTED_LLM_PROVIDERS = tuple(_PROVIDER_REQUIRED_ENV)

DEFAULT_MODEL = (
    os.getenv("DEFAULT_MODEL", "").strip()
    or _PROVIDER_DEFAULT_MODEL.get(LLM_PROVIDER, "")
)
# Per-role overrides; empty -> fall back to DEFAULT_MODEL.
ORCHESTRATOR_MODEL = os.getenv("ORCHESTRATOR_MODEL", "").strip()
WORKER_MODEL = os.getenv("WORKER_MODEL", "").strip()

_ROLE_MODEL = {"orchestrator": ORCHESTRATOR_MODEL, "worker": WORKER_MODEL}


def model_for_role(role: str = "worker") -> str:
    """The model name for a role ('orchestrator' | 'worker'), before prefixing."""
    return _ROLE_MODEL.get(role, "") or DEFAULT_MODEL


def _litellm_extra_kwargs() -> dict:
    """Provider kwargs LiteLLM needs passed explicitly rather than via env.

    Azure routing (deployment endpoint + api-version) is fiddly enough that we
    hand it to LiteLlm directly when the vars are present.
    """
    kw: dict = {}
    if LLM_PROVIDER == "azure":
        pairs = (("AZURE_API_BASE", "api_base"),
                 ("AZURE_API_VERSION", "api_version"),
                 ("AZURE_API_KEY", "api_key"))
    elif LLM_PROVIDER == "azure_ai":
        pairs = (("AZURE_AI_API_BASE", "api_base"),
                 ("AZURE_AI_API_VERSION", "api_version"),  # optional, some Foundry endpoints want it
                 ("AZURE_AI_API_KEY", "api_key"))
    elif LLM_PROVIDER == "anthropic":
        # lets you point the anthropic provider at an Anthropic-compatible
        # gateway (e.g. Azure AI Foundry's /anthropic endpoint) via .env
        pairs = (("ANTHROPIC_API_BASE", "api_base"),)
    else:
        pairs = ()
    for env_name, arg in pairs:
        val = os.getenv(env_name)
        if val:
            kw[arg] = val
    return kw


def get_llm_model(model: str | None = None, *, role: str = "worker"):
    """A LiteLlm for the active provider.

    Model name resolution: explicit `model` arg, else the role's model
    (ORCHESTRATOR_MODEL / WORKER_MODEL), else DEFAULT_MODEL. A name that already
    carries a "<provider>/" prefix is used verbatim; otherwise LLM_PROVIDER's
    prefix is prepended.
    """
    from google.adk.models.lite_llm import LiteLlm

    name = (model or model_for_role(role) or "").strip()
    if not name:
        raise RuntimeError(
            f"No model configured for the {role} (LLM_PROVIDER={LLM_PROVIDER!r}). "
            "Set DEFAULT_MODEL in .env (for Azure, that's the deployment name)."
        )
    if "/" in name:
        # explicit "<provider>/<model>" - the caller owns routing, so don't
        # attach LLM_PROVIDER's base/key kwargs (they'd be for the wrong service)
        return LiteLlm(model=name)
    return LiteLlm(model=f"{LLM_PROVIDER}/{name}", **_litellm_extra_kwargs())


def missing_llm_credentials() -> list[str]:
    """Required env vars for the active LLM_PROVIDER that are not set."""
    return [v for v in _PROVIDER_REQUIRED_ENV.get(LLM_PROVIDER, []) if not os.getenv(v)]


def require_llm_credentials() -> None:
    """Raise a clear error if the active provider can't be used. Call at startup."""
    if LLM_PROVIDER not in _PROVIDER_REQUIRED_ENV:
        raise RuntimeError(
            f"Unknown LLM_PROVIDER={LLM_PROVIDER!r}. "
            f"Supported: {', '.join(SUPPORTED_LLM_PROVIDERS)}."
        )
    missing = missing_llm_credentials()
    if missing:
        raise RuntimeError(
            f"LLM_PROVIDER={LLM_PROVIDER!r} needs {', '.join(missing)} set in .env "
            "(or switch LLM_PROVIDER)."
        )
    for role in ("orchestrator", "worker"):
        if not model_for_role(role):
            raise RuntimeError(
                f"No model for the {role} (LLM_PROVIDER={LLM_PROVIDER!r}). Set "
                f"DEFAULT_MODEL (or {role.upper()}_MODEL) in .env."
            )


## =============================================================================
# Session storage backend
#
# Every agent's ADK Runner needs a session service. Two options:
#   "memory" -> InMemorySessionService   (fast, zero-setup, lost on restart)
#   "database" -> DatabaseSessionService  (persists sessions + state to a DB)
#
# Select with SESSION_BACKEND in .env. When "database", SESSION_DB_URL points at
# the store (SQLite by default; any SQLAlchemy URL works, e.g. PostgreSQL).
# The actual service objects are built in shared/persistence.py so importing
# this module never requires SQLAlchemy unless the DB backend is chosen.
## =============================================================================

SESSION_BACKEND = os.getenv("SESSION_BACKEND", "memory").strip().lower()

# ADK's DatabaseSessionService uses SQLAlchemy's async engine, so the URL needs
# an async driver: sqlite+aiosqlite for SQLite, postgresql+asyncpg for Postgres.
# shared/persistence.py upgrades a plain sqlite:// URL automatically, so a bare
# "sqlite:///..." in .env still works.
#
# Default SQLite file lives next to this repo's root. Override for PostgreSQL:
#   SESSION_DB_URL=postgresql+asyncpg://user:pass@localhost:5432/ergon
_DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "ergon.db"
).replace("\\", "/")
SESSION_DB_URL = os.getenv("SESSION_DB_URL", f"sqlite+aiosqlite:///{_DEFAULT_DB_PATH}")


## =============================================================================
# Agent server settings (host/ports)
## =============================================================================

AGENT_HOST = os.getenv("AGENT_HOST", "localhost")
DATA_PORT = int(os.getenv("DATA_PORT", "10003"))
WEB_INTEL_PORT = int(os.getenv("WEB_INTEL_PORT", "10005"))

AGENT_PORTS = {
    "data": DATA_PORT,
    "web_intel": WEB_INTEL_PORT,
}


def agent_url(agent_name: str) -> str:
    """Builds the base URL for a worker agent by name ('data', 'web_intel')."""
    return f"http://{AGENT_HOST}:{AGENT_PORTS[agent_name]}"


# --- Orchestrator -> worker call resilience (see orchestrator_agent/.../agent.py) ---
# The orchestrator delegates over a blocking A2A call. A slow, wedged, crashed or
# restarting worker must NOT take down the whole orchestrator turn - the failed
# delegation is turned into a normal tool result the orchestrator LLM can read
# and route around ("that agent is down, proceed with what we have / retry").
#
#   A2A_CLIENT_TIMEOUT_SECONDS   - read in remote_agent_connection.py; per-call
#                                  read timeout (default 300s).
#   A2A_HEALTHCHECK_TIMEOUT_SECONDS - quick agent-card GET before delegating; a
#                                  dead/wedged worker then fails in seconds
#                                  instead of blocking for the full timeout.
#                                  0 disables the pre-flight check.
#   A2A_MAX_RETRIES              - retries on a *transient* failure (connection
#                                  refused/reset, 502/503/504). Timeouts are not
#                                  retried (they would just time out again).
#   A2A_RETRY_BACKOFF_SECONDS   - base sleep between retries (doubles each try).
A2A_HEALTHCHECK_TIMEOUT_SECONDS = float(os.getenv("A2A_HEALTHCHECK_TIMEOUT_SECONDS", "5"))
A2A_MAX_RETRIES = int(os.getenv("A2A_MAX_RETRIES", "1"))
A2A_RETRY_BACKOFF_SECONDS = float(os.getenv("A2A_RETRY_BACKOFF_SECONDS", "1.5"))


## =============================================================================
# Web search / scraping (Researcher — see shared/web_tools.py)
#
# Deliberately conservative, public-web-only settings:
#   - Search via DuckDuckGo through the keyless `ddgs` package (free, no account;
#     HTML-scraped and rate-limited, so treat transient failures as normal).
#   - Page extraction via httpx + trafilatura, GET only, no cookies, no auth.
#   - robots.txt (including Crawl-delay) is honoured; a politeness gap is kept
#     between hits on the same host.
#   - Requests to non-public / internal addresses (localhost, private ranges,
#     link-local, the cloud metadata IP) are refused (SSRF guard).
#   - Response bodies are size-capped; redirects are limited and must stay http(s).
#
# SCRAPE_RESPECT_ROBOTS can be turned off, but doing so is the operator's own
# legal/ethical call - the agent still won't log in, pay, or bypass access
# controls. Leaving it on is strongly recommended.
## =============================================================================

SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "duckduckgo").strip().lower()
SEARCH_MAX_RESULTS = int(os.getenv("SEARCH_MAX_RESULTS", "5"))
SEARCH_REGION = os.getenv("SEARCH_REGION", "us-en")

SCRAPE_MAX_CHARS = int(os.getenv("SCRAPE_MAX_CHARS", "8000"))
SCRAPE_TIMEOUT_SECONDS = float(os.getenv("SCRAPE_TIMEOUT_SECONDS", "15"))
SCRAPE_MAX_BYTES = int(os.getenv("SCRAPE_MAX_BYTES", str(5 * 1024 * 1024)))  # 5 MiB
SCRAPE_MAX_REDIRECTS = int(os.getenv("SCRAPE_MAX_REDIRECTS", "5"))
SCRAPE_POLITENESS_DELAY_SECONDS = float(
    os.getenv("SCRAPE_POLITENESS_DELAY_SECONDS", "1.0")
)
SCRAPE_RESPECT_ROBOTS = os.getenv("SCRAPE_RESPECT_ROBOTS", "true").strip().lower() not in (
    "0", "false", "no", "off", ""
)
# Honest, identifiable bot UA - not a spoofed browser string. Add a contact URL
# in .env once the company has a public domain: "ErgonResearcher/1.0 (+https://.../bot)".
WEB_INTEL_USER_AGENT = os.getenv("WEB_INTEL_USER_AGENT", "ErgonResearcher/1.0")
# Hard ceiling on web fetches per single request, enforced in a before_tool
# callback (shared/callbacks.py). The prompt asks the agent to stay near ~4;
# this stops a runaway loop from piling up robots.txt lookups + politeness
# delays until the orchestrator times out.
WEB_INTEL_MAX_FETCHES_PER_TASK = int(os.getenv("WEB_INTEL_MAX_FETCHES_PER_TASK", "6"))

# Weather (Researcher - get_weather tool). Open-Meteo: free, keyless, no
# account. These are documented public JSON API endpoints, so get_weather is an
# API call, not a scrape (no robots.txt dance, but still SSRF-guarded).
WEATHER_GEOCODE_URL = os.getenv(
    "WEATHER_GEOCODE_URL", "https://geocoding-api.open-meteo.com/v1/search"
)
WEATHER_FORECAST_URL = os.getenv(
    "WEATHER_FORECAST_URL", "https://api.open-meteo.com/v1/forecast"
)
WEATHER_FORECAST_DAYS = int(os.getenv("WEATHER_FORECAST_DAYS", "7"))

# Default recency window for web_search(recency=...) - the DuckDuckGo "news"
# vertical. Values: d (day), w (week), m (month), y (year).
NEWS_DEFAULT_TIMELIMIT = os.getenv("NEWS_DEFAULT_TIMELIMIT", "w").strip().lower()


## =============================================================================
# MCP connectors
#
# An employee can reach an outside system (mail, calendar, CRM, tracker...)
# through an MCP toolset. Connectors are added per employee in a later phase;
# this helper is kept because its signature is easy to get wrong.
## =============================================================================


def create_tool_filter(allowed_tools: list[str]):
    """
    Build an MCP tool filter that only lets `allowed_tools` through.

    Returns None (no filter: every tool) when the list is empty or contains "*".
    """
    if not allowed_tools or "*" in allowed_tools:
        return None

    def tool_filter(tool, readonly_context=None) -> bool:
        """True if the tool should be included.

        ADK's BaseToolset._is_tool_selected() calls this as
        tool_filter(tool, readonly_context) - matching google.adk.tools.base_toolset.ToolPredicate's
        __call__ signature is required, or McpToolset.get_tools() throws
        "tool_filter() takes N positional arguments but 2 were given" on every
        tool and silently returns zero tools (the agent then has no real MCP
        tools at all and may hallucinate tool calls instead of erroring).
        """
        return tool.name in allowed_tools

    return tool_filter
