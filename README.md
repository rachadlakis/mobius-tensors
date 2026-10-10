<div align="center">

<img src="Images/logo.jpeg" alt="Mobius Tensors logo" width="140" />

# Mobius Tensors

One reusable AI solution platform, configured per use case.

</div>

## Projects

| Project | Description |
| --- | --- |
| [argus](projects/argus/) | Record everything cameras see as embeddings, then ask about it; includes the detection engine (YOLOX → ONNX Runtime) |
| [agents](projects/agents/) | Not started |
| [agent-deployment](projects/agent-deployment/) | Installing and managing AI agents (Claude Code, Cowork, Codex, …) on the customer side |

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
│   ├── mobius-tensors/   company, architecture, roadmap and installation docs
│   └── <project>/        one folder per project (descriptions, presentations)
├── Images/      logo, banner and brand kit (Mobius Tensors in brand-kit/, Argus in argus/)
└── projects/    one folder per product, each with its own uv environment
```

## Development

Open `mobius-tensors.code-workspace` in VS Code so each project uses its own `.venv`.
To add a project, create it under `projects/`, run `uv sync` in it, and add it to the workspace file.
