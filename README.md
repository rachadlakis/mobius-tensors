<div align="center">

<img src="Images/brand-kit/logo/logo-horizontal-light.png" alt="Mobius Tensors logo" width="420" />

# Mobius Tensors

A computer vision and AI solutions company. One tested core, configured per use case.

</div>

## Projects

One folder per project in [docs/](docs/), each with a short page:

| Project | Description | Status |
| --- | --- | --- |
| [histor](docs/histor/histor.md) | Ask your cameras what happened; alerts with a local check, dashboard, MCP server; includes the detection engine (YOLOX → ONNX Runtime) | Working code |
| [ergon](docs/ergon/ergon.md) | Ergon: a team of AI employees (Chief of Staff, Researcher, Data Analyst, more planned) for any company; Google ADK + A2A | In development |
| [agent-deployment](docs/agent-deployment/agent-deployment.md) | Installing and managing AI agents (Claude Code, Cowork, Codex, …) on the customer side | Research and templates |
| [agents](docs/agents/agents.md) | Agentic AI agents and chatbots for customers' workflows | Not started |
| [automations](docs/automations/automations.md) | Business workflows automated with AI | Not started |
| [distributed-training](docs/distributed-training/distributed-training.md) | Training and fine-tuning LLMs, object detection models and VLMs, across GPUs when needed | Not started |
| [forecasting](docs/forecasting/forecasting.md) | Forecasts from customers' time series, and statistical studies | Not started |

The website lives in its own repo, next to this one: `../mobius-website`.

## Docs

Company-wide, in [docs/mobius-tensors](docs/mobius-tensors/):

- [Mobius Tensors — AI Solution Platform](docs/mobius-tensors/Mobius%20Tensors%20—%20AI%20Solution%20Platform.md): company, use cases and ideas, architecture, roadmap and build plan, risks, glossary

Per project, next to its code:

- [Histor](projects/histor/README.md): [README](projects/histor/README.md) and [DOCUMENTATION.md](projects/histor/DOCUMENTATION.md) (product, how it works, site installation, model research, roadmap); presentation, project summary PDF and partner programme in [projects/histor/docs](projects/histor/docs/)

## Layout

```
mobius-tensors/
├── docs/
│   ├── mobius-tensors/   company, architecture and roadmap docs
│   └── <project>/        one folder per project: <project>.md (what it is, status, links), and room
│                         for decks and PDFs
├── Images/      logo, banner and brand kit (Mobius Tensors in brand-kit/, Histor in histor/)
└── projects/    one folder per product, each with its own uv environment
```

## Development

Open `mobius-tensors.code-workspace` in VS Code so each project uses its own `.venv`.
To add a project, create it under `projects/`, run `uv sync` in it, and add it to the workspace file.
