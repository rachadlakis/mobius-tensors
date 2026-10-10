<div align="center">

<img src="Images/logo.jpeg" alt="Mobius Tensors logo" width="140" />

# Mobius Tensors

One reusable AI solution platform, configured per use case.

</div>

## Projects

One folder per project in [docs/](docs/), each with a short page:

| Project | Description | Status |
| --- | --- | --- |
| [argus](docs/argus/argus.md) | Ask your cameras what happened; alerts with a local check, dashboard, MCP server; includes the detection engine (YOLOX → ONNX Runtime) | Working code |
| [agent-deployment](docs/agent-deployment/agent-deployment.md) | Installing and managing AI agents (Claude Code, Cowork, Codex, …) on the customer side | Research and templates |
| [agents](docs/agents/agents.md) | AI agents for customers' workflows | Not started |
| [automations](docs/automations/automations.md) | Business workflows automated with AI | Not started |
| [distributed-training](docs/distributed-training/distributed-training.md) | Training and fine-tuning models across GPUs | Not started |
| [forecasting](docs/forecasting/forecasting.md) | Forecasts from customers' time series | Not started |
| [object-detection](docs/object-detection/object-detection.md) | Merged into Argus as `argus.engine` (2026-10-09) | Merged |

The website lives in its own repo, next to this one: `../mobius-website`.

## Docs

Company-wide, in [docs/mobius-tensors](docs/mobius-tensors/):

- [Mobius Tensors — AI Solution Platform](docs/mobius-tensors/Mobius%20Tensors%20—%20AI%20Solution%20Platform.md): company, use cases and ideas, architecture, roadmap and build plan, risks, glossary

Per project, next to its code:

- [Argus](projects/argus/README.md): [README](projects/argus/README.md) and [DOCUMENTATION.md](projects/argus/DOCUMENTATION.md) (product, how it works, site installation, model research, roadmap); presentation, project summary PDF and partner programme in [projects/argus/docs](projects/argus/docs/)

## Layout

```
mobius-tensors/
├── docs/
│   ├── mobius-tensors/   company, architecture and roadmap docs
│   └── <project>/        one folder per project: <project>.md (what it is, status, links), and room
│                         for decks and PDFs
├── Images/      logo, banner and brand kit (Mobius Tensors in brand-kit/, Argus in argus/)
└── projects/    one folder per product, each with its own uv environment
```

## Development

Open `mobius-tensors.code-workspace` in VS Code so each project uses its own `.venv`.
To add a project, create it under `projects/`, run `uv sync` in it, and add it to the workspace file.
