"""
run_coding_task: the Engineer's tool. Claude Code (through the Claude Agent SDK)
does the work; this module keeps it on its own branch.

For each task:

1. A git worktree is created from the base branch on a new branch
   ``ergon/<task>-<time>``, in ``workspace/worktrees/`` - the repository's own
   checkout and branches are never touched.
2. Claude Code runs in that worktree: it can read, edit and run commands (tests),
   but is told not to push or switch branches, and ``git push`` is denied.
3. Whatever it changed is committed on the branch, the worktree is removed, and the
   branch stays in the repository for review. A branch with no changes is deleted.
4. Only when ``ENGINEER_OPEN_PR=true``: the branch is pushed and a *draft* pull
   request is opened with ``gh``. Off by default: employees draft, people ship.

Repositories the Engineer may work on are listed in ``.env``:
``ERGON_REPOS=website=C:/code/site;app=C:/code/app`` (name=path, ``;``-separated).

Like run_python, this runs on the machine Ergon runs on, for internal use. Claude
Code's commands are not sandboxed; a container comes before any outside use.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

from shared import documents

logger = logging.getLogger(__name__)

ENGINEER_MODEL = os.getenv("ENGINEER_MODEL", "claude-opus-5-5").strip()
ENGINEER_MAX_TURNS = int(os.getenv("ENGINEER_MAX_TURNS", "80"))
ENGINEER_MAX_BUDGET_USD = float(os.getenv("ENGINEER_MAX_BUDGET_USD", "5"))
ENGINEER_TIMEOUT_SECONDS = float(os.getenv("ENGINEER_TIMEOUT_SECONDS", "1500"))
_TRUE = ("1", "true", "yes", "on")

# Auto-approved: Claude Code may read, edit and run commands in its worktree.
_ALLOWED_TOOLS = ["Read", "Edit", "Write", "Glob", "Grep", "Bash"]
# Denied even if asked: publishing, rewriting history, leaving the worktree's branch.
_DISALLOWED_TOOLS = [
    "Bash(git push:*)",
    "Bash(git remote:*)",
    "Bash(git checkout:*)",
    "Bash(git switch:*)",
    "Bash(git worktree:*)",
    "Bash(git reset --hard:*)",
    "Bash(gh:*)",
]

_RULES = """You are working for Ergon's Engineer on one task, in a git worktree that is
already on its own branch. Rules:
- Work only inside the current folder. Do not push, do not switch or create branches,
  do not rewrite history. Commit your work on the current branch with clear messages.
- Follow the repository's own conventions (CLAUDE.md, README, existing style).
- Run the existing tests or checks for what you change, and fix what you break.
- If the task is unclear or would need secrets, a deploy or a destructive step, stop
  and explain instead of guessing.
- Finish with a short summary: what you changed (files), how you checked it, and what
  is left or uncertain."""


class CodingError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Repositories
# --------------------------------------------------------------------------- #
def repositories() -> dict[str, Path]:
    """Name -> path for every repository in ERGON_REPOS (read on every call)."""
    repos: dict[str, Path] = {}
    for item in (os.getenv("ERGON_REPOS") or "").split(";"):
        if "=" in item:
            name, path = item.split("=", 1)
            if name.strip() and path.strip():
                repos[name.strip()] = Path(path.strip()).expanduser()
    return repos


async def list_repositories() -> dict:
    """
    List the code repositories the Engineer may work on (from ERGON_REPOS in .env).

    Returns:
        dict: success and repositories - a list of {name, path, branch, exists}.
    """
    return await asyncio.to_thread(_list_repositories_blocking)


def _list_repositories_blocking() -> dict:
    out = []
    for name, path in repositories().items():
        branch = None
        exists = (path / ".git").exists()
        if exists:
            try:
                branch = _git(path, "rev-parse", "--abbrev-ref", "HEAD")
            except CodingError:
                pass
        out.append({"name": name, "path": str(path), "branch": branch, "exists": exists})
    return {
        "success": True,
        "repositories": out,
        "note": "" if out else "No repositories configured: set ERGON_REPOS in .env.",
    }


# --------------------------------------------------------------------------- #
# Git
# --------------------------------------------------------------------------- #
def _git(cwd: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    if check and proc.returncode != 0:
        raise CodingError(f"git {' '.join(args)} failed: {proc.stderr.strip()[:500]}")
    return proc.stdout.strip()


def _slug(text: str) -> str:
    words = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return words[:40].rstrip("-") or "task"


def _start_worktree(repo: Path, base: str, task: str) -> tuple[str, Path]:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    branch = f"ergon/{_slug(task)}-{stamp}"
    worktree = documents.WORKSPACE_DIR / "worktrees" / f"{repo.name}-{stamp}"
    worktree.parent.mkdir(parents=True, exist_ok=True)
    _git(repo, "worktree", "add", "-b", branch, str(worktree), base)
    return branch, worktree


def _finish_worktree(repo: Path, worktree: Path, branch: str, base: str, task: str) -> dict:
    """Commit leftovers, describe the branch, remove the worktree. A branch with
    no commits beyond the base is deleted."""
    if _git(worktree, "status", "--porcelain"):
        _git(worktree, "add", "-A")
        first_line = task.strip().splitlines()[0][:72] if task.strip() else "task"
        _git(worktree, "commit", "-m", f"Ergon Engineer: {first_line}")
    commits = _git(repo, "log", "--oneline", f"{base}..{branch}").splitlines()
    diff_stat = _git(repo, "diff", "--stat", f"{base}...{branch}") if commits else ""
    _git(repo, "worktree", "remove", "--force", str(worktree), check=False)
    if not commits:
        _git(repo, "branch", "-D", branch, check=False)
    return {"commits": commits, "diff_stat": diff_stat}


def _open_draft_pr(repo: Path, branch: str, base: str, title: str, body: str) -> str:
    _git(repo, "push", "-u", "origin", branch)
    proc = subprocess.run(
        ["gh", "pr", "create", "--draft", "--base", base, "--head", branch,
         "--title", title, "--body", body],
        cwd=repo, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if proc.returncode != 0:
        raise CodingError(f"gh pr create failed: {proc.stderr.strip()[:500]}")
    return proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""


# --------------------------------------------------------------------------- #
# Claude Code
# --------------------------------------------------------------------------- #
async def _run_claude_code(worktree: Path, task: str) -> dict:
    """Run one Claude Agent SDK query in the worktree. Never raises: every
    failure comes back as a dict the Engineer can report."""
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ClaudeSDKError,
        CLINotFoundError,
        ResultMessage,
        TextBlock,
        query,
    )

    options = ClaudeAgentOptions(
        cwd=str(worktree),
        model=ENGINEER_MODEL or None,
        permission_mode="acceptEdits",
        allowed_tools=_ALLOWED_TOOLS,
        disallowed_tools=_DISALLOWED_TOOLS,
        max_turns=ENGINEER_MAX_TURNS,
        max_budget_usd=ENGINEER_MAX_BUDGET_USD,
        system_prompt={"type": "preset", "preset": "claude_code", "append": _RULES},
        setting_sources=["project"],  # the repository's CLAUDE.md and project settings
    )
    last_text = ""
    result: ResultMessage | None = None

    async def consume():
        nonlocal last_text, result
        async for message in query(prompt=task, options=options):
            if isinstance(message, AssistantMessage):
                texts = [b.text for b in message.content if isinstance(b, TextBlock)]
                if texts:
                    last_text = "\n".join(texts)
            elif isinstance(message, ResultMessage):
                result = message

    try:
        await asyncio.wait_for(consume(), timeout=ENGINEER_TIMEOUT_SECONDS)
        error = None
    except asyncio.TimeoutError:
        error = f"Stopped after {ENGINEER_TIMEOUT_SECONDS:.0f} s (ENGINEER_TIMEOUT_SECONDS)."
    except CLINotFoundError:
        error = "Claude Code was not found. Reinstall claude-agent-sdk or set it up."
    except ClaudeSDKError as e:
        error = f"{type(e).__name__}: {str(e)[:500]}"

    if result is not None and result.is_error and not error:
        error = f"Claude Code stopped: {result.subtype}" + (
            f" ({'; '.join(result.errors)[:300]})" if result.errors else "")
    return {
        "summary": (result.result if result and result.result else last_text)[:6000],
        "cost_usd": round(result.total_cost_usd, 4) if result and result.total_cost_usd else None,
        "turns": result.num_turns if result else None,
        "stopped_because": result.subtype if result else None,
        "error": error,
    }


# --------------------------------------------------------------------------- #
# The tool
# --------------------------------------------------------------------------- #
async def run_coding_task(repository: str, task: str, base_branch: str | None = None) -> dict:
    """
    Have Claude Code do a coding task in one of the company's repositories, on a
    new branch of its own. Nothing is pushed or merged: the result is a branch
    (or, when the company enabled it, a draft pull request) for a person to review.

    Give a complete task: what to change, where, and how to check it ("run the
    tests"). One task per call. A task can take several minutes.

    Args:
        repository: the repository's name, as listed by list_repositories().
        task: the full task description for the coding agent.
        base_branch: branch to start from; defaults to the repository's current branch.

    Returns:
        dict: success, repository, branch, base, commits, diff_stat, summary,
            cost_usd, turns, pull_request; or success=False and an error.
    """
    repos = repositories()
    if repository not in repos:
        return {"success": False, "error": f"Unknown repository {repository!r}.",
                "repositories": sorted(repos)}
    if not (task or "").strip():
        return {"success": False, "error": "Empty task."}
    repo = repos[repository]
    try:
        base = base_branch or await asyncio.to_thread(
            _git, repo, "rev-parse", "--abbrev-ref", "HEAD")
        await asyncio.to_thread(_git, repo, "rev-parse", "--verify", base)
        branch, worktree = await asyncio.to_thread(_start_worktree, repo, base, task)
    except (CodingError, OSError) as e:
        return {"success": False, "repository": repository, "error": str(e)}

    run = await _run_claude_code(worktree, task)
    try:
        finished = await asyncio.to_thread(_finish_worktree, repo, worktree, branch, base, task)
    except CodingError as e:
        return {"success": False, "repository": repository, "branch": branch,
                "error": f"Could not commit the work: {e}", **run}

    out = {
        "success": run["error"] is None and bool(finished["commits"]),
        "repository": repository,
        "base": base,
        "branch": branch if finished["commits"] else None,
        **finished,
        **run,
        "pull_request": None,
    }
    if not finished["commits"] and not run["error"]:
        out["error"] = "No changes were made; the branch was deleted."
    if out["branch"] and (os.getenv("ENGINEER_OPEN_PR") or "").strip().lower() in _TRUE:
        title = f"Ergon Engineer: {task.strip().splitlines()[0][:72]}"
        try:
            out["pull_request"] = await asyncio.to_thread(
                _open_draft_pr, repo, branch, base, title, run["summary"] or task)
        except CodingError as e:
            out["pull_request_error"] = str(e)
    return out
