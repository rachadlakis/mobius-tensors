<div align="center">

<img src="Images/logo.jpeg" alt="Mobius Tensors logo" width="140" />

# Mobius Tensors

One reusable AI solution platform, configured per use case.

</div>

## Projects

| Project | Description |
| --- | --- |
| [object-detection](projects/object-detection/) | CPU-only object detection (YOLOX → ONNX Runtime) |
| [agents](projects/agents/) | Not started |

The website lives in its own repo, next to this one: `../mobius-website`.

## Docs

- [Mobius Tensors — AI Solution Platform](docs/Mobius%20Tensors%20—%20AI%20Solution%20Platform.md)
- [Architecture](docs/architecture.md)
- [Roadmap](docs/Roadmap.md)
- [Implementation Plan — From Trained Model to App](docs/Implementation%20Plan%20—%20From%20Trained%20Model%20to%20App.md)
- [Ideas](docs/Ideas.md)

## Layout

```
mobius-tensors/
├── docs/        company, architecture and roadmap docs
├── Images/      logo, banner and brand kit
└── projects/    one folder per product, each with its own uv environment
```

## Development

Open `mobius-tensors.code-workspace` in VS Code so each project uses its own `.venv`.
To add a project, create it under `projects/`, run `uv sync` in it, and add it to the workspace file.
