# Ergon — plan

**Ergon by Mobius Tensors** is a team of AI employees that any company can run. Each employee is a role with a job
description, tools and permissions. A **Chief of Staff** takes your request, hands the work to the
right people and gives you back one result. Everything specific to a company lives in its own
`companies/<id>/` folder, never in code.

**Rule:** nothing from the client project this started as may come back: no client names, their
domain concepts, keys, data or docs.

**Companies configured:** a made-up sample (`example`) and Mobius Tensors, so we use the team
ourselves first. A client is just another `companies/<id>/` folder.

---

## 1 · Where we are (2026-10-16)

**Phase 0 (done):** the client project is gone; the generic engine stays. Deleted: the
client-specific employees and tools, their notebook, scripts, docs, the studio UI prototype (its
whole data model was client-specific), every saved session and artifact, generated reports and
caches, and the client's keys in `.env`.

**Phase 1 (done):** employees are YAML files. One generic server runs any of them, a tool
registry maps tool names to functions, and `run_all.py` and the Chief of Staff both read
`employees/`. The Researcher and the Data Analyst were moved over unchanged.

**Phase 2 (done):** the company brain, and Ergon stays general. Each company is a folder
`companies/<id>/` with a `profile.yaml` (put at the top of every instruction, live) and a
`knowledge/` folder that employees search (`search_company_knowledge`, `read_company_document`).
`ERGON_COMPANY` picks one: `example` (a made-up sample, the default and the template) or
`mobius_tensors` (Histor, services, internal pricing).

**Phase 3 (done):** the core team. New employees **Writer**, **Sales Rep** and **Support
Agent**; the **Data Analyst** now runs Python on the task's files (`run_python`, a local
subprocess with a time limit and no secrets) and every employee that handles prices has
`calculate`. Document output: PDF, Word, Excel, Markdown/text/CSV/HTML. The Chief of Staff
forwards spreadsheets as real files to employees whose card says they take files.

**Engineer (added early, from the second wave):** code changes through Claude Code (Claude Agent
SDK) in a git worktree on its own `ergon/...` branch; delivers the branch, or a draft pull
request when `ENGINEER_OPEN_PR=true`. Employees can now set their own `timeout_seconds`.

| Piece | What it is |
|---|---|
| `employees/*.yaml` | one file per employee: card, tools, model, instructions |
| `shared/employees.py` | loads and validates the YAML (unique names and ports, clear errors) |
| `shared/tool_registry.py` | every tool an employee can be given, by name |
| `shared/employee_server.py` | `python -m shared.employee_server <id>` runs any employee |
| `shared/documents.py` | `write_pdf_report`, `list_workspace_files`; files land in `workspace/` |
| `shared/company.py` | the selected company: profile, knowledge search (BM25, rebuilt when a file changes) |
| `shared/sandbox.py`, `shared/calculator.py` | `run_python` for the Data Analyst, `calculate` for prices and totals |
| `companies/<id>/` | one folder per company: `profile.yaml` + `knowledge/` |
| `orchestrator_agent/` | the **Chief of Staff**; finds the team in `employees/` |
| `tests/` | offline suite, all green |

---

## 2 · Shape

```
ergon/
├── companies/               # the ONLY thing you edit per company; ERGON_COMPANY picks one
│   ├── example/             # made-up sample company: the default and the template
│   └── mobius_tensors/      # profile.yaml (who we are, offer, voice, rules) + knowledge/
├── employees/               # one YAML file per AI employee
│   ├── researcher.yaml, data_analyst.yaml
│   └── writer.yaml, sales_rep.yaml, support_agent.yaml
├── orchestrator_agent/      # the Chief of Staff (adk web serves it)
├── shared/                  # the engine: employee server, tool registry, tools, A2A plumbing
├── workspace/               # files employees produce (gitignored)
├── tests/
└── run_all.py               # starts every employee found in employees/
```

The earlier idea of renaming `shared/` to `core/` and `orchestrator_agent/` to `chief_of_staff/`
was dropped: it would touch every import and test for no change in behaviour.

**Hiring an employee is a YAML file, not Python:**

```yaml
name: Sales Rep
port: 10006
description: Finds and wins customers; drafts outreach, proposals and quotes.
model: worker              # worker (cheap) | orchestrator (capable) | an explicit model name
tools: [write_pdf_report]  # names from shared/tool_registry.py
skill:
  id: sales
  name: Sales
  description: Prospecting, outreach emails, proposals and quotes from the price list.
  tags: [sales, proposals]
  examples: ["Draft a proposal for Acme for 12 cameras."]
instruction: |
  **Role:** You are the Sales Rep. ...
```

Still to add as the phases need it: `permission: read | draft | act` (Phase 4).

## 3 · Shared foundations

Every employee depends on these. Build them once, before adding roles.

1. **Company brain (done).** `profile.yaml` is put at the top of every instruction;
   `search_company_knowledge` searches `knowledge/` with keyword ranking (BM25, in memory, no
   database). Add PDF/Word documents and embeddings later if keyword search falls short.
2. **Document output (done).** PDF, Word, Excel and Markdown/text/CSV/HTML. Files land in
   `workspace/` and flow back through A2A.
3. **Permission levels.** `read` looks things up. `draft` produces text and files for a human to
   use. `act` changes the outside world (send, post, book, pay) and **always pauses for your
   approval** using A2A's `input-required` state. In v1 no employee has `act`: they draft, you
   send.
4. **Work log.** Who did what, which tools, tokens and cost per request. The numbers are already
   measured in `callbacks.py`; they just need to be stored.
5. **Connectors (later).** Each employee's YAML can list MCP servers (Gmail, Calendar, Drive,
   Slack, CRM, GitHub, Linear).

---

## 4 · The employees

### Core team (built in Phases 0-3)

These cover the daily work of a small company and reuse most of the existing code.

| Employee | What they do | Tools | Permission | Built from |
|---|---|---|---|---|
| **Chief of Staff** | Your single point of contact. Splits a request, assigns it, chains employees, merges the result, asks you before any `act` step | `send_message`, team roster, work log | — | Orchestrator |
| **Researcher** | Market, competitor, lead and technology research, always with sources | web search, news, page reading, knowledge | read | Web Intelligence |
| **Data Analyst** | Analyses uploaded Excel/CSV, KPIs, charts, Excel/PDF reports | Python sandbox (pandas, matplotlib), file tools | read + files | Data agent |
| **Writer** (marketing and content) | Posts, blog articles, website copy, brochure text, newsletters in the company's voice | knowledge, document output | draft | built |
| **Sales Rep** | Prospect lists (via Researcher), outreach emails, proposals, quotes from the price list | knowledge, price calculator, document output | draft | built |
| **Support Agent** | Answers customer questions from the knowledge base, drafts replies, flags what the docs don't cover | knowledge | draft | built |

### Next wave (v2): one at a time, each needs a connector

| Employee | What they do | Needs |
|---|---|---|
| **Project Manager** | Goals into tasks and timelines, weekly status reports, tracks deadlines | Linear / Jira / GitHub Issues MCP · `act` with approval |
| **Executive Assistant** | Inbox triage, meeting prep, scheduling, meeting minutes | Gmail / Outlook + Calendar MCP · `act` with approval |
| **Finance & Admin** | Quotes into invoices, expense sheets, monthly cash summary. Never pays anything | spreadsheets, accounting export |
| **Recruiter** | Job posts, CV screening against a rubric, interview kits | document input |
| **Legal Reviewer** | Checks contracts and NDAs against a checklist and flags risks. A human always signs | document input, checklist in `companies/<id>/knowledge/` |
| ~~**Engineer**~~ **built** | Code changes on its own branch via Claude Code; review, scripts, docs | Claude Agent SDK, git worktrees; GitHub MCP later |

### Example: one request, several employees

> "Find 20 security integrators in the UAE and draft an intro email for Histor to each."

Chief of Staff → **Researcher** (list with sources) → **Data Analyst** (cleaned into Excel) →
**Sales Rep** (one personalised email each, from the Histor profile and price list) → you get
`leads.xlsx` and 20 drafts to send yourself.

---

## 5 · Phases

| # | Phase | Effort | Done when |
|---|---|---|---|
| 0 | ~~**Remove the client project:** delete its code, data and keys, strip config and web tools, rename the package to `ergon`~~ **done** | 1 day | no client trace left; engine tests pass |
| 1 | ~~**Employee engine:** YAML employees, generic `employee_server.py`, tool registry, `run_all.py` reads `employees/`. Port Researcher and Data Analyst~~ **done** | 2–3 days | two employees running from YAML, Chief of Staff routes between them |
| 2 | ~~**Company brain:** `profile.yaml`, `knowledge/`, knowledge search, profile in every prompt. Fill in Mobius Tensors~~ **done** (general: `companies/<id>/`) | 2 days | "What does Histor cost for 12 cameras?" is answered correctly from the knowledge |
| 3 | ~~**Core team:** Writer, Sales Rep, Support Agent, document output, Python sandbox for the Analyst~~ **done** (live run of the UAE example still to do) | 3–4 days | the UAE example above runs end to end |
| 4 | **Approvals and work log:** draft/act permissions with pause-and-approve, run records, cost per request | 3–4 days | an `act` tool cannot run without your yes; every request has a cost line |
| 5 | **Connectors and v2 employees:** one employee at a time via MCP | 2–3 days each | — |
| 6 | **Team UI:** a new "your team" screen to hire, configure and watch employees | 1–2 weeks | — |

Phases 0–3 (about two weeks) give a working six-person team.

## 6 · Open decisions

- **Internal tool or product for clients?** Today one company per running Ergon
  (`ERGON_COMPANY`); a client is a new `companies/<id>/` folder. Multi-company login and billing only if we sell it.
- **Default LLM.** Keep LiteLLM's provider switch; pick one default (capable model for the Chief
  of Staff, cheaper for the employees, as today).
- **Python sandbox for the Analyst.** Local subprocess with limits (simple) or a container (safer).
  Default: a subprocess for internal use, a container before any client uses it.
