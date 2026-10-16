"""Tests for the Engineer's tool (shared/coding.py).

A real git repository is made in tmp_path; Claude Code itself is replaced by a
fake that edits files in the worktree, so these run offline and free. One opt-in
test runs the real Claude Code (RUN_ENGINEER_LIVE=1, costs a few cents).
"""

import os
import subprocess

import pytest

from shared import coding, documents


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                          check=True).stdout.strip()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    path = tmp_path / "app"
    path.mkdir()
    _git(path, "init", "-q", "-b", "main")
    _git(path, "config", "user.email", "test@example.com")
    _git(path, "config", "user.name", "Test")
    (path / "README.md").write_text("# App\n", encoding="utf-8")
    _git(path, "add", "-A")
    _git(path, "commit", "-q", "-m", "initial")
    monkeypatch.setenv("ERGON_REPOS", f"app={path};missing={tmp_path / 'nope'}")
    monkeypatch.setattr(documents, "WORKSPACE_DIR", tmp_path / "workspace")
    monkeypatch.delenv("ENGINEER_OPEN_PR", raising=False)
    return path


def _fake_claude(edit=None, commit=False, error=None):
    """Replace the Claude Code run: optionally edit/commit in the worktree."""
    seen = {}

    async def run(worktree, task):
        seen["worktree"], seen["task"] = worktree, task
        seen["branch_during_run"] = _git(worktree, "rev-parse", "--abbrev-ref", "HEAD")
        if edit:
            for name, text in edit.items():
                (worktree / name).write_text(text, encoding="utf-8")
            if commit:
                _git(worktree, "add", "-A")
                _git(worktree, "commit", "-q", "-m", "Claude's own commit")
        return {"summary": "Changed things.", "cost_usd": 0.12, "turns": 3,
                "stopped_because": "success", "error": error}

    return run, seen


class TestRepositories:
    async def test_lists_configured_repositories(self, repo):
        out = await coding.list_repositories()
        by_name = {r["name"]: r for r in out["repositories"]}
        assert by_name["app"]["branch"] == "main"
        assert by_name["app"]["exists"] is True
        assert by_name["missing"]["exists"] is False

    async def test_none_configured(self, monkeypatch):
        monkeypatch.delenv("ERGON_REPOS", raising=False)
        out = await coding.list_repositories()
        assert out["repositories"] == []
        assert "ERGON_REPOS" in out["note"]

    def test_parsing_ignores_junk(self, monkeypatch):
        monkeypatch.setenv("ERGON_REPOS", "a=/x; junk ;=nope;b= /y ;")
        assert set(coding.repositories()) == {"a", "b"}


class TestRunCodingTask:
    async def test_work_lands_on_its_own_branch_and_main_is_untouched(self, repo, monkeypatch):
        run, seen = _fake_claude(edit={"app.py": "print('hi')\n"})
        monkeypatch.setattr(coding, "_run_claude_code", run)

        out = await coding.run_coding_task("app", "Add a hello script")

        assert out["success"] is True
        assert out["branch"].startswith("ergon/add-a-hello-script-")
        assert seen["branch_during_run"] == out["branch"]
        assert out["base"] == "main"
        assert len(out["commits"]) == 1 and "Ergon Engineer: Add a hello script" in out["commits"][0]
        assert "app.py" in out["diff_stat"]
        assert out["cost_usd"] == 0.12
        # the repository's own checkout and branch are untouched
        assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "main"
        assert not (repo / "app.py").exists()
        # the branch has the change; the worktree is gone
        assert _git(repo, "show", f"{out['branch']}:app.py") == "print('hi')"
        assert not seen["worktree"].exists()
        assert out["pull_request"] is None

    async def test_claude_commits_are_kept(self, repo, monkeypatch):
        run, _ = _fake_claude(edit={"a.txt": "1"}, commit=True)
        monkeypatch.setattr(coding, "_run_claude_code", run)
        out = await coding.run_coding_task("app", "Do it")
        assert [c.split(" ", 1)[1] for c in out["commits"]] == ["Claude's own commit"]

    async def test_no_changes_deletes_the_branch(self, repo, monkeypatch):
        run, _ = _fake_claude()
        monkeypatch.setattr(coding, "_run_claude_code", run)
        out = await coding.run_coding_task("app", "Look but do not touch")
        assert out["success"] is False
        assert out["branch"] is None
        assert "No changes" in out["error"]
        assert _git(repo, "branch", "--list", "ergon/*") == ""

    async def test_claude_error_keeps_partial_work_for_review(self, repo, monkeypatch):
        run, _ = _fake_claude(edit={"half.py": "x = 1\n"}, error="Claude Code stopped: error_max_turns")
        monkeypatch.setattr(coding, "_run_claude_code", run)
        out = await coding.run_coding_task("app", "Big task")
        assert out["success"] is False
        assert out["error"].endswith("error_max_turns")
        assert out["branch"]  # partial work kept on its branch

    async def test_base_branch(self, repo, monkeypatch):
        _git(repo, "branch", "develop")
        run, _ = _fake_claude(edit={"d.txt": "d"})
        monkeypatch.setattr(coding, "_run_claude_code", run)
        out = await coding.run_coding_task("app", "On develop", base_branch="develop")
        assert out["base"] == "develop"

    async def test_unknown_base_branch(self, repo):
        out = await coding.run_coding_task("app", "x", base_branch="nope")
        assert out["success"] is False
        assert "nope" in out["error"]

    async def test_unknown_repository_lists_known(self, repo):
        out = await coding.run_coding_task("other", "x")
        assert out["success"] is False
        assert out["repositories"] == ["app", "missing"]

    async def test_empty_task(self, repo):
        assert (await coding.run_coding_task("app", "  "))["success"] is False

    async def test_draft_pr_only_when_enabled(self, repo, monkeypatch):
        run, _ = _fake_claude(edit={"p.txt": "p"})
        monkeypatch.setattr(coding, "_run_claude_code", run)
        calls = []
        monkeypatch.setattr(coding, "_open_draft_pr",
                            lambda *a: calls.append(a) or "https://example.test/pr/1")

        out = await coding.run_coding_task("app", "Change p")
        assert out["pull_request"] is None and calls == []

        monkeypatch.setenv("ENGINEER_OPEN_PR", "true")
        out = await coding.run_coding_task("app", "Change p again")
        assert out["pull_request"] == "https://example.test/pr/1"
        _, branch, base, title, _ = calls[0]
        assert (branch, base) == (out["branch"], "main")
        assert title.startswith("Ergon Engineer:")


def test_claude_code_may_not_push_or_leave_its_branch():
    denied = " ".join(coding._DISALLOWED_TOOLS)
    for command in ("git push", "git checkout", "git switch", "git worktree", "gh"):
        assert f"Bash({command}:*)" in denied


@pytest.mark.skipif(os.getenv("RUN_ENGINEER_LIVE") != "1",
                    reason="set RUN_ENGINEER_LIVE=1 to run real Claude Code (costs a few cents)")
async def test_live_claude_code_makes_a_small_change(repo, monkeypatch):
    monkeypatch.setattr(coding, "ENGINEER_MAX_BUDGET_USD", 1.0)
    out = await coding.run_coding_task(
        "app", "Add a file hello.py that prints 'hello'. Run it once to check. Keep it to one line.")
    assert out["success"] is True, out
    assert "hello" in _git(repo, "show", f"{out['branch']}:hello.py")
    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "main"
