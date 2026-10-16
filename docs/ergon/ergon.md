<img src="../../Images/brand-kit/logo/logo-horizontal-light.png" alt="Mobius Tensors" width="200" align="right" />

# Ergon

*A Mobius Tensors project. Ergon (Greek ἔργον) means "work".*

A team of AI employees for any company. You talk to a **Chief of Staff**; it splits your request
into tasks, hands each one to the right employee, and gives you back one result. Every employee
knows the company it works for: its profile and its documents.

| | |
| --- | --- |
| **Status** | In development: phases 0–2 of the plan done (2026-10-16). The Chief of Staff, a Researcher and a Data Analyst work today; more employees are planned |
| **Code** | [projects/ergon](../../projects/ergon/) (Python, Google ADK + A2A protocol) |
| **Docs** | [README](../../projects/ergon/README.md) · [PLAN.md](../../projects/ergon/PLAN.md) (roadmap) |

## What it is

- **Employees are YAML files.** Each one is a role with a job description, tools and a model.
  Hiring a new employee means adding a file, not writing code.
- **One company per folder.** Everything specific to a company (who it is, what it sells, its
  voice, its rules and its documents) lives in `companies/<id>/`, never in code.
- **Every employee is its own small A2A server**, so any A2A client can also call one directly.
- **Any LLM provider** through LiteLLM: Anthropic, OpenAI, Azure, Gemini, Vertex AI or Bedrock.

## The team

| Employee | Does |
| --- | --- |
| Chief of Staff | Takes the request, delegates, merges the answers |
| Researcher | Web research with sources: markets, competitors, leads, technology, news |
| Data Analyst | Spreadsheets and CSVs, KPIs, trends, PDF reports |
| *Planned* | Writer, Sales Rep, Support Agent; later PM, Executive Assistant, Finance, Recruiter, Legal, Engineer |

## First user: Mobius Tensors

Ergon runs for Mobius Tensors first (`companies/mobius_tensors/`: Histor, services, internal
pricing), so we use the team ourselves before offering it to clients. A client is just another
`companies/<id>/` folder.

Overlaps with [agents](../agents/agents.md) (custom agents and chatbots for customers) and
[agent-deployment](../agent-deployment/agent-deployment.md): Ergon is the ready-made team; the
agents project builds one agent around one customer workflow.
