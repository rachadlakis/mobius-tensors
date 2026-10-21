# CLAUDE.md

Guidance for Claude Code in this project.

## What this is

**Ergon** (by Gradient Labs): a team of AI employees on Google ADK + A2A. A Chief of Staff
(`orchestrator_agent/`) delegates to employees, each a standalone A2A server defined by one YAML
file in `employees/`. `README.md` is the overview; `PLAN.md` is the roadmap and the roster.

This folder lives inside the `mobius-tensors` git repo; run git commands from here as usual.

## Rules

- **Never call it "ai-company"**: the name is Ergon.
- The project started as a client project. **No trace of that client may come back**: no client
  names, their domain concepts, keys or data. Use generic business examples.
- Ergon is general: it works for any company. Company-specific facts (products, prices, tone,
  rules) go in `companies/<id>/` (profile.yaml + knowledge/), never in code or in an employee's
  YAML. Employees and code must not mention a specific company. `ERGON_COMPANY` picks the
  company; the tests always use `companies/example` (a made-up sample).
- Employees draft; they do not send, post, pay or delete. Outward actions will need an approval
  step (Phase 4).
- Employees never do arithmetic in their heads: prices and totals go through `calculate`, data
  through `run_python`.
- `run_python` (`shared/sandbox.py`) is a local subprocess with limits, not a security sandbox. It
  must never inherit `.env` secrets (`_child_env`); keep it internal until it runs in a container.
- The Engineer (`shared/coding.py`) must never change the user's checkout or existing branches:
  work happens in a worktree on a new `ergon/...` branch, `git push`/branch switching are in
  `_DISALLOWED_TOOLS`, and pushing a draft PR happens only with `ENGINEER_OPEN_PR=true`. Its
  offline tests fake Claude Code; `RUN_ENGINEER_LIVE=1` runs the real one (costs money).

## Environment

Uses **`uv`**, not raw `pip`.

```bash
uv venv
.venv\Scripts\activate
uv pip install -e ".[dev]"
```

Copy `.env.example` to `.env` and fill in the block for your `LLM_PROVIDER`.

## Commands

```bash
python run_all.py                 # every employee in employees/
python -m shared.employee_server researcher   # one employee
cd orchestrator_agent && adk web  # Chief of Staff + chat UI on :8000

python -m pytest                  # offline suite, a few seconds
uvx ruff@latest check .           # pyflakes + import sort only
```

`ERGON_ORCHESTRATOR_AUTOINIT=0` (set by the tests) stops the Chief of Staff from contacting
employees at import time.

## Architecture notes

- Every employee has the same shape, built by `shared/employee_server.py` from its YAML:
  Agent Card → A2A server → `shared/executor.py` (`BaseA2AExecutor`) → ADK Runner → `LlmAgent`
  → tools. Add an employee with a YAML file, not Python; add a tool by writing it in `shared/`
  and registering it in `shared/tool_registry.py`.
- An employee with `web_search` or `fetch_page` automatically gets the fetch budget
  (`callbacks.FETCH_TOOLS`).
- An employee with a file tool (`tool_registry.FILE_TOOLS`, today `run_python`) advertises
  `FILE_INPUT_MODE` on its card and saves incoming files as artifacts; the Chief of Staff then
  sends it spreadsheets as real files plus a preview instead of pasted CSV text.
- Every instruction (employees and the Chief of Staff) starts with the company profile
  (`shared/company.py`, `company_block`). It is an instruction *provider*, recomposed every
  turn, so profile edits apply live and ADK's `{placeholder}` substitution is skipped.
- The Chief of Staff discovers employees from their cards on the first turn, sends work with
  `send_message`, and turns any failure into a readable tool result instead of an exception.
- Files: the Chief of Staff keeps a per-session file store (`tool_context.state["session_files"]`).
  Spreadsheets are forwarded as CSV text, other files as A2A file parts. A file an employee saves
  with `tool_context.save_artifact` is harvested by the executor and returned to the Chief of Staff.
- Tools that do blocking I/O must be `async` and push the work to a thread
  (`asyncio.to_thread`); a sync tool freezes the employee's server.
- An employee can set `timeout_seconds` in its YAML when its work outlasts the Chief of Staff's
  default 300 s wait (the Engineer: 1800).
- `config.create_tool_filter` must keep the `(tool, readonly_context)` signature or
  `McpToolset.get_tools()` silently returns zero tools.
