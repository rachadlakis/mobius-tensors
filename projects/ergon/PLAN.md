# Ergon — plan

**Ergon by Mobius Tensors** is a team of AI employees that any company can run. Each employee is a role with a job
description, tools and permissions. A **Chief of Staff** takes your request, hands the work to the
right people and gives you back one result. Everything specific to a company lives in one
`company/` folder, never in code.

**Rule:** nothing from the client project this started as may come back: no client names, their
domain concepts, keys, data or docs.

**First company configured:** Mobius Tensors, so we use the team ourselves first. A client is
then just another `company/` folder.

---

## 1 · Phase 0 is done (2026-10-16)

The client project is gone; the generic engine stays.

| Kept and cleaned | Now |
|---|---|
| `shared/` | executor, file hand-off, callbacks, persistence, config (LLM switching), web tools (search, page reading, weather) |
| `shared/documents.py` | new: `write_pdf_report`, `list_workspace_files`; files land in `workspace/` |
| `orchestrator_agent/` | the **Chief of Staff** |
| `web_intel_agent_adk/` | the **Researcher** |
| `data_agent_adk/` | the **Data Analyst**: works from uploaded files, writes PDF reports |
| `tests/` | offline suite, all green |

Deleted: the client-specific employees and tools, their notebook, scripts, docs, the studio UI
prototype (its whole data model was client-specific), every saved session and artifact, generated
reports and caches, and the client's keys in `.env`.

---

## 2 · New shape

```
ergon/
├── company/                 # the ONLY thing you edit per company
│   ├── profile.yaml         # who we are, what we sell, prices, tone of voice, policies, who approves what
│   └── knowledge/           # documents employees read: product docs, FAQs, brochures, price list
├── employees/               # one YAML file per AI employee
│   ├── researcher.yaml
│   ├── data_analyst.yaml
│   └── ...
├── chief_of_staff/          # the orchestrator
├── core/                    # the engine (was shared/)
│   └── employee_server.py   # ONE generic launcher instead of a copy-pasted __main__.py per agent
├── tools/                   # tool library, by family: web, knowledge, documents, sandbox, connectors
├── workspace/               # files employees produce (gitignored)
├── ui/
├── tests/
└── run_all.py               # starts every employee found in employees/
```

**Hiring an employee is a YAML file, not Python:**

```yaml
name: Sales Rep
port: 10006
role: >
  You find and win customers for {company.name}. You write outreach, proposals and quotes
  using only the products and prices in the company profile.
skills: [prospecting, outreach, proposals, quotes]
tools: [search_company_knowledge, price_quote, write_docx, write_pdf]
permission: draft          # read | draft | act
model: worker              # worker (cheap) | lead (capable)
```

## 3 · Shared foundations

Every employee depends on these. Build them once, before adding roles.

1. **Company brain.** `profile.yaml` is injected into every employee's instructions;
   `search_company_knowledge` searches `company/knowledge/`. Start with keyword search on SQLite
   (no new infrastructure) and add embeddings later.
2. **Document output.** Write Markdown, PDF, Word and Excel (`fpdf2` and `openpyxl` are already
   in; add `python-docx`). Files land in `workspace/` and flow back through A2A as today.
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

### Core team: build first (6)

These cover the daily work of a small company and reuse most of the existing code.

| Employee | What they do | Tools | Permission | Built from |
|---|---|---|---|---|
| **Chief of Staff** | Your single point of contact. Splits a request, assigns it, chains employees, merges the result, asks you before any `act` step | `send_message`, team roster, work log | — | Orchestrator |
| **Researcher** | Market, competitor, lead and technology research, always with sources | web search, news, page reading, knowledge | read | Web Intelligence |
| **Data Analyst** | Analyses uploaded Excel/CSV, KPIs, charts, Excel/PDF reports | Python sandbox (pandas, matplotlib), file tools | read + files | Data agent |
| **Writer** (marketing and content) | Posts, blog articles, website copy, brochure text, newsletters in the company's voice | knowledge, document output | draft | new |
| **Sales Rep** | Prospect lists (via Researcher), outreach emails, proposals, quotes from the price list | knowledge, price calculator, document output | draft | new |
| **Support Agent** | Answers customer questions from the knowledge base, drafts replies, flags what the docs don't cover | knowledge | draft | new |

### Next wave (v2): one at a time, each needs a connector

| Employee | What they do | Needs |
|---|---|---|
| **Project Manager** | Goals into tasks and timelines, weekly status reports, tracks deadlines | Linear / Jira / GitHub Issues MCP · `act` with approval |
| **Executive Assistant** | Inbox triage, meeting prep, scheduling, meeting minutes | Gmail / Outlook + Calendar MCP · `act` with approval |
| **Finance & Admin** | Quotes into invoices, expense sheets, monthly cash summary. Never pays anything | spreadsheets, accounting export |
| **Recruiter** | Job posts, CV screening against a rubric, interview kits | document input |
| **Legal Reviewer** | Checks contracts and NDAs against a checklist and flags risks. A human always signs | document input, checklist in `company/` |
| **Engineer** | Code review, scripts, technical docs | GitHub MCP, sandbox |

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
| 1 | **Employee engine:** YAML employees, generic `employee_server.py`, tool registry, `run_all.py` reads `employees/`. Port Researcher and Data Analyst | 2–3 days | two employees running from YAML, Chief of Staff routes between them |
| 2 | **Company brain:** `profile.yaml`, `knowledge/`, knowledge search, profile in every prompt. Fill in Mobius Tensors | 2 days | "What does Histor cost for 12 cameras?" is answered correctly from the profile |
| 3 | **Core team:** Writer, Sales Rep, Support Agent, document output, Python sandbox for the Analyst | 3–4 days | the UAE example above runs end to end |
| 4 | **Approvals and work log:** draft/act permissions with pause-and-approve, run records, cost per request | 3–4 days | an `act` tool cannot run without your yes; every request has a cost line |
| 5 | **Connectors and v2 employees:** one employee at a time via MCP | 2–3 days each | — |
| 6 | **Team UI:** a new "your team" screen to hire, configure and watch employees | 1–2 weeks | — |

Phases 0–3 (about two weeks) give a working six-person team.

## 6 · Open decisions

- **Internal tool or product for clients?** Default: build for one company, keep `company/`
  separate so a client is a new folder. Multi-company login and billing only if we sell it.
- **Default LLM.** Keep LiteLLM's provider switch; pick one default (capable model for the Chief
  of Staff, cheaper for the employees, as today).
- **Python sandbox for the Analyst.** Local subprocess with limits (simple) or a container (safer).
  Default: a subprocess for internal use, a container before any client uses it.
