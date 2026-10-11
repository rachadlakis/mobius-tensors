<img src="../../Images/brand-kit/logo/logo-horizontal-light.png" alt="Mobius Tensors" width="200" align="right" />

# Agent Deployment

*A Mobius Tensors project.*

Install and manage AI agents (Claude Code, Claude Desktop / Cowork, Codex, Gemini CLI,
open-source agents) on the customer's side: their machines, their network, their cloud account,
under their rules.

| | |
| --- | --- |
| **Status** | Research and templates only; nothing tested on a customer machine yet |
| **Code** | [projects/agent-deployment](../../projects/agent-deployment/) (documents and policy templates, no code) |
| **Docs** | [README](../../projects/agent-deployment/README.md) · [research.md](../../projects/agent-deployment/research.md) (agents, where the model runs, admin controls, risks) · [playbook.md](../../projects/agent-deployment/playbook.md) (discovery → pilot → rollout → handover) · [templates](../../projects/agent-deployment/templates/) |

## What it is

A service, not a product: instead of every employee installing AI tools on their own with
personal accounts, the customer gets approved agents and models with security policy, single
sign-on, logging and cost control from day one.

| Package | What the customer gets |
| --- | --- |
| Assessment | Which agent, where the model runs, which policies, a cost estimate (1–2 weeks) |
| Pilot | 5–20 users in one team with managed policy and telemetry (2–4 weeks) |
| Rollout | Fleet install through their MDM / Group Policy, network allowlist, SSO, training |
| Managed service | Updates, usage and cost reports, new MCP integrations, support |
| Custom agents | Agents for their workflows with the Claude Agent SDK and MCP servers |

The templates hold starting policy files per agent (Claude Code managed settings, also on Amazon
Bedrock; Codex `requirements.toml`; Gemini CLI on Vertex AI).

## Next

Run the playbook on our own machines first (Windows and Linux, Claude Code and Claude Desktop),
write install scripts that pin a version and place the policy file, then a one-page offer.
