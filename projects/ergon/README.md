# Ergon

**Ergon by Mobius Tensors**: a team of AI employees. You talk to a **Chief of Staff**; it splits
your request into tasks, hands each one to the right employee, and gives you back one result.

Built on Google ADK and the A2A protocol. Every employee is its own small HTTP server, so any
A2A client can also call one directly.

The roadmap is in [PLAN.md](PLAN.md).

## The team today

| Employee | Defined in | Port | Does | Tools |
|---|---|---|---|---|
| **Chief of Staff** | `orchestrator_agent/` | 8000 (`adk web`) | Takes your request, delegates, merges the answers, passes files between employees | `send_message` |
| **Researcher** | `employees/researcher.yaml` | 10005 | Web research with sources: markets, competitors, leads, technology, news | `web_search`, `fetch_page`, `get_weather` |
| **Data Analyst** | `employees/data_analyst.yaml` | 10003 | Analyses uploaded spreadsheets/CSVs, KPIs, trends, PDF reports | `write_pdf_report`, `list_workspace_files` |

More employees (Writer, Sales Rep, Support Agent, ...) are planned. See [PLAN.md](PLAN.md).

## Adding an employee

1. Copy `employees/researcher.yaml` to `employees/<id>.yaml` (lowercase, underscores).
2. Give it a unique `name` and `port`, its card text (`description`, `skill`), its `tools` and
   its `instruction`.
3. Tools are referenced by name from `shared/tool_registry.py`. A new tool is a function in a
   `shared/` module plus one line in that registry.
4. Restart `run_all.py` and the Chief of Staff. Both find the new employee on their own.

## Setup

```bash
uv venv
.venv\Scripts\activate          # Windows; source .venv/bin/activate on Mac/Linux
uv pip install -e ".[dev]"
```

Copy `.env.example` to `.env` and fill in the block for your `LLM_PROVIDER` (anthropic, openai,
azure, azure_ai, gemini, vertex_ai or bedrock, all through LiteLLM).

## Run

```bash
# Terminal 1: every employee in employees/
python run_all.py
# or one at a time:  python -m shared.employee_server researcher

# Terminal 2: the Chief of Staff with the chat UI, then open http://localhost:8000
cd orchestrator_agent
adk web
```

Files you upload in the chat are forwarded to whichever employee gets the task. Spreadsheets
and CSVs arrive as CSV text, other files as file parts. Files an employee produces come back as
downloads and are forwarded to the next employee.

## Test

```bash
python -m pytest                  # offline: no web, no LLM
uvx ruff@latest check .
RUN_LLM_INTEGRATION=1 python -m pytest tests/test_llm_smoke.py   # hits the configured provider
```

## Layout

```
employees/            one YAML file per employee
orchestrator_agent/   Chief of Staff (A2A client + routing + file hand-off)
shared/
  employees.py        loads and validates employees/*.yaml
  employee_server.py  runs any employee as an A2A server
  tool_registry.py    every tool an employee can be given, by name
  config.py           LLM provider, host, search settings, MCP tool filter
  executor.py         A2A <-> ADK bridge every employee runs on
  a2a_files.py        file parts in and out of A2A messages
  attachments.py      Excel/CSV -> CSV text
  callbacks.py        tool/token logging, Researcher fetch budget
  persistence.py      in-memory or database sessions
  web_tools.py        search, page reading, weather (SSRF guard, robots.txt)
  documents.py        PDF reports into workspace/
tests/                offline pytest suite
run_all.py            starts every employee in employees/
```
