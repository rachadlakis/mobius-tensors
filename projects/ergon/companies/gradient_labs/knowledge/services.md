# Gradient Labs services

Gradient Labs is a company that sells computer vision and AI solutions. Histor is the flagship
product (see histor.md). The services below are listed with their status: only Histor and the
computer vision behind it have delivered work; "scoping" means no project has been delivered yet,
so describe what we can do, never past results.

## Computer vision (in use inside Histor)

The detection engine under Histor: detection, tracking, zones and rules on standard servers.

- Number-plate reading and watchlists.
- Perimeter and zone watch.
- People and vehicle counting.
- Pose: falls, hands up, running.
- Runs headless on ONNX Runtime, CPU or GPU.

## AI agent rollout (service)

Claude Code, Codex, Gemini CLI and open-source agents installed on the customer's side, under
their rules: policy, single sign-on, logging and cost control from day one. Research and policy
templates exist; nothing has been tested on a customer machine yet.

| Package | What the customer gets |
| --- | --- |
| Assessment | Which agent, where the model runs, which policies, a cost estimate (1-2 weeks) |
| Pilot | 5-20 users in one team with managed policy and telemetry (2-4 weeks) |
| Rollout | Fleet install through their device management, network allowlist, SSO, training |
| Managed service | Updates, usage and cost reports, new integrations, support |
| Custom agents | Agents for their workflows with the Claude Agent SDK and MCP servers |

## AI agents and chatbots (scoping)

Agents and chatbots we build for a customer's workflows, with tools, memory, guardrails and a
person to escalate to: agents that use the customer's tools for internal tasks, chatbots and
assistants over their own documents, customer-facing chatbots, voice agents.

## Automations (scoping)

Business workflows automated with AI: data moves between the customer's tools and a model steps
in where judgement is needed (classify, extract, summarise, route). Camera alerts from Histor can
start a workflow.

## Training and fine-tuning (scoping)

Training and fine-tuning models on the customer's data: LLMs, object detection models for their
own objects, vision-language models, distributed training when one GPU is not enough.

## Forecasting and statistical studies (scoping)

Forecasts from the customer's time series (demand, sales, load, traffic) with uncertainty ranges,
refreshed on a schedule; statistical studies and reports on their data. Counts from Histor
cameras can be an input.
