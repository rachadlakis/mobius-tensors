<img src="../../Images/brand-kit/logo/logo-horizontal-light.png" alt="Mobius Tensors" width="200" align="right" />

# Agent Deployment

*A Mobius Tensors project.*

Install and manage AI agents (Claude Code, Claude Desktop / Cowork, Codex, Gemini CLI, open-source agents)
on the customer's side: their machines, their network, their cloud account, under their rules.

The customer gets working agents with security policy, logging and cost control set up from day one,
instead of every employee installing tools on their own.

```mermaid
flowchart LR
  subgraph Before["Without us"]
    direction TB
    E1[Employee A] --> T1[Installs tool X<br/>personal account]
    E2[Employee B] --> T2[Installs tool Y<br/>personal API key]
    E3[Employee C] --> T3[Pastes code into<br/>a web chat]
  end
  subgraph After["With Mobius Tensors"]
    direction TB
    P[(Company policy<br/>SSO + rules + logs)]
    F1[Employee A] --> P
    F2[Employee B] --> P
    F3[Employee C] --> P
    P --> M[Approved agents<br/>approved models]
  end
  Before -. "we set up" .-> After
```

## How it fits together

```mermaid
flowchart LR
  subgraph Customer["Customer site"]
    direction TB
    Admin[IT admin<br/>MDM / Group Policy / admin console] -- pushes policy --> Dev
    subgraph Dev["Employee machine"]
      direction TB
      Agent[Agent<br/>Claude Code / Cowork / Codex / Gemini CLI]
      Policy[Managed policy<br/>cannot be overridden]
      Policy --> Agent
    end
    Agent --> Proxy[Corporate proxy<br/>+ firewall allowlist]
    Agent -- MCP --> Internal[Internal systems<br/>Git, Jira, DBs, docs]
    Agent -- OpenTelemetry --> SIEM[(Collector / SIEM<br/>usage, cost, audit)]
  end
  Proxy --> GW{LLM gateway?<br/>optional}
  GW --> Vendor[Vendor plan<br/>Claude / ChatGPT]
  GW --> Cloud[Customer cloud<br/>Bedrock / Vertex / Foundry]
  Proxy -.-> Local[Local models<br/>Ollama / vLLM<br/>offline sites]
```

## What we sell

```mermaid
flowchart LR
  A[Assessment<br/>1-2 weeks] --> B[Pilot<br/>2-4 weeks, 5-20 users]
  B -- go --> C[Rollout<br/>team by team]
  B -- no-go --> X[Report + recommendations]
  C --> D[Managed service<br/>monthly]
  C --> E[Custom agents<br/>Agent SDK + MCP]
  D --> E
```

| Package | What the customer gets |
| --- | --- |
| **Assessment** | Discovery call + questionnaire, a recommendation (which agent, where the model runs, which policies), cost estimate |
| **Pilot** | 5-20 users, one team, managed policy, telemetry, 2-4 weeks, short report with usage and issues |
| **Rollout** | Fleet install through their MDM / Group Policy / scripts, network allowlist, SSO, training session |
| **Managed service** | Version and policy updates, usage and cost reports, new MCP integrations, support |
| **Custom agents** | Agents built for their workflows with the Claude Agent SDK + MCP servers to their internal systems |

## Folder map

```text
agent-deployment/
├── README.md        this file
├── research.md      landscape: agents, where the model runs, admin controls per agent, risks, sources
├── playbook.md      step-by-step customer engagement: discovery → pilot → rollout → handover
└── templates/       starting policy files per agent (copy, then edit per customer)
```

## Status

Research and templates only. Nothing tested on a customer machine yet.

Next steps:

1. Run the playbook on our own machines first (Windows + Linux, Claude Code + Claude Desktop).
2. Write install scripts (PowerShell + bash) that install a pinned version and place the policy file.
3. Write a one-page offer / solution sheet for the brochures.
