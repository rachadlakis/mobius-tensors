<img src="../../Images/brand-kit/logo/logo-horizontal-light.png" alt="Mobius Tensors" width="200" align="right" />

# Histor

*Histor is a Mobius Tensors product.*

Ask your cameras what happened, in plain language, and get the frames as evidence.

| | |
| --- | --- |
| **Status** | Working code; pilot-ready on recorded footage, live alerts to be validated at a pilot site |
| **Code** | [projects/histor](../../projects/histor/) (Python, one `uv` environment) |
| **Docs** | [README](../../projects/histor/README.md) · [DOCUMENTATION.md](../../projects/histor/DOCUMENTATION.md) (product, how it works, site installation, model research, roadmap) · [OPEN-QUESTIONS.md](../../projects/histor/OPEN-QUESTIONS.md) · deck, project summary PDF and partner programme in [projects/histor/docs](../../projects/histor/docs/) · brochures: [Solutions](<brochures/Histor — Solutions.pdf>) and [Technical](<brochures/Histor — Technical.pdf>) |

## What it is

Histor keeps a searchable memory of everything a site's cameras see: every tracked object and
regular snapshots of each scene are stored as CLIP vectors, and each person's body pose becomes
facts (posture, hands up, walking or running, carried bags, clothing colours). Questions like
*"was there a white van at the gate yesterday evening?"* or *"did anyone fall in the warehouse
last night?"* are answered by a chat model with tools over that memory, with the thumbnails as
evidence.

It also alerts as things happen: areas and lines, crowding, abandoned bags, camera tampering,
person down, watchlist plates and sounds. A local vision model can look at each alert before it
is sent and hold back false alarms. Alerts go to webhook, email and Telegram and appear on a
dashboard with a token login. The chat also runs as a Telegram bot, and its tools are an MCP
server for other agents.

## Key choices

- Runs on the site's own server, **without a GPU** (ONNX Runtime); footage stays in the building.
- The chat and vision models run locally by default (Ollama); a cloud model is opt-in and only
  ever receives text.
- **Commercially clean:** only permissive licences (Apache-2.0, MIT) for code, weights and
  training data. Two trial-only exceptions are marked in the docs (the plate finder, RTMPose-t).
- Biometrics (faces), plates and sound are off until a client approves them.
- Histor carries no Mobius Tensors branding: it is planned to go to another company.

## Next

Our own YOLOX plate finder (urgent, licence), alert checking and person-down measured at a
pilot site, a fire and smoke model, more alert channels. Full order:
[DOCUMENTATION.md, Roadmap](../../projects/histor/DOCUMENTATION.md#roadmap).
