# Ergon

**Ergon by Mobius Tensors**: a team of AI employees for any company. You talk to a **Chief of
Staff**; it splits your request into tasks, hands each one to the right employee, and gives you
back one result. Every employee knows the company it works for: its profile and its documents.

Built on Google ADK and the A2A protocol. Every employee is its own small HTTP server, so any
A2A client can also call one directly.

The roadmap is in [PLAN.md](PLAN.md).

## The team today

| Employee | Defined in | Port | Does | Tools |
|---|---|---|---|---|
| **Chief of Staff** | `orchestrator_agent/` | 8000 (`adk web`) | Takes your request, delegates, merges the answers, passes files between employees | `send_message` |
| **Researcher** | `employees/researcher.yaml` | 10005 | Web research with sources: markets, competitors, leads, technology, news | `web_search`, `fetch_page`, `get_weather`, knowledge |
| **Data Analyst** | `employees/data_analyst.yaml` | 10003 | Runs Python on uploaded spreadsheets: KPIs, trends, charts, cleaned tables; prices from the price list | `run_python`, `calculate`, Excel, PDF, knowledge |
| **Writer** | `employees/writer.yaml` | 10006 | Posts, articles, website and brochure copy, newsletters, emails in the company's voice | Word, text files, PDF, knowledge |
| **Sales Rep** | `employees/sales_rep.yaml` | 10007 | Outreach emails, proposals and quotes from the price list | `calculate`, Word, Excel, PDF, text files, knowledge |
| **Support Agent** | `employees/support_agent.yaml` | 10008 | Answers customers from the company documents, drafts replies, flags gaps | `calculate`, text files, knowledge |
| **Engineer** | `employees/engineer.yaml` | 10009 | Code changes in the company's repositories via Claude Code, each on its own branch, for review | `run_coding_task`, `list_repositories`, knowledge |

Knowledge = `search_company_knowledge` and `read_company_document`. Employees **draft**; none of
them sends, posts or pays. More employees (Project Manager, Executive Assistant, Finance, ...)
come with connectors. See [PLAN.md](PLAN.md).

**The Engineer** hands each task to Claude Code (the Claude Agent SDK, bundled with its own CLI)
in a separate git worktree on a new `ergon/<task>-<time>` branch. Your checkout and branches are
never touched; `git push` is denied to Claude Code. The result is a local branch for you to
review, or a draft pull request when `ENGINEER_OPEN_PR=true`. Repositories are listed in `.env`
(`ERGON_REPOS=name=path;...`); cost per task is capped (`ENGINEER_MAX_BUDGET_USD`, default $5).

`run_python` runs the Data Analyst's scripts in a separate Python process with a time limit and
no API keys, in its own folder under `workspace/runs/`. It is **not a security sandbox**: keep it
for internal use until it runs in a container.

## The company

Ergon works for one company at a time, chosen with `ERGON_COMPANY` in `.env`:

| Folder | What |
|---|---|
| `companies/example/` | Example Co., a made-up sample (the default). Copy it to start your own |
| `companies/mobius_tensors/` | Mobius Tensors: Histor, services, internal pricing |
| `companies/<id>/profile.yaml` | name, summary, about, offer, voice, rules: put at the top of every employee's instructions |
| `companies/<id>/knowledge/` | Markdown or text documents employees search (keyword ranking, no database) |

`ERGON_COMPANY` can also be a path to a company folder anywhere. Profile and document edits apply
on the next turn, without a restart.

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
companies/            one folder per company (profile.yaml + knowledge/)
employees/            one YAML file per employee
orchestrator_agent/   Chief of Staff (A2A client + routing + file hand-off)
shared/
  company.py          the selected company: profile, knowledge search
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
  documents.py        PDF, Word, Excel and text files into workspace/
  calculator.py       safe arithmetic (calculate)
  sandbox.py          run_python for the Data Analyst
  coding.py           run_coding_task for the Engineer (Claude Code in a git worktree)
tests/                offline pytest suite
run_all.py            starts every employee in employees/
```
