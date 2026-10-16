"""Tests for run_python (shared/sandbox.py). These really start a Python process,
so they take a second or two; the timeout test uses a 2 s limit."""

import os

import pytest
from google.genai import types

from shared import documents, sandbox


class FakeCtx:
    """The artifact slice of an ADK ToolContext, in memory (async, like ADK's)."""

    def __init__(self, files=None):
        self.store = {
            name: types.Part(inline_data=types.Blob(data=data, mime_type="text/csv"))
            for name, data in (files or {}).items()
        }

    async def list_artifacts(self):
        return list(self.store)

    async def load_artifact(self, name):
        return self.store[name]

    async def save_artifact(self, name, part):
        self.store[name] = part
        return 0


@pytest.fixture(autouse=True)
def workspace(monkeypatch, tmp_path):
    monkeypatch.setattr(documents, "WORKSPACE_DIR", tmp_path)
    return tmp_path


async def test_reads_inputs_prints_and_returns_new_files():
    ctx = FakeCtx({"sales.csv": b"month,sales\nJan,100\nFeb,140\n"})
    code = (
        "import pandas as pd\n"
        "df = pd.read_csv('sales.csv')\n"
        "print('total', df['sales'].sum())\n"
        "df.to_csv('out.csv', index=False)\n"
    )
    out = await sandbox.run_python(code, ctx)
    assert out["success"] is True, out
    assert "total 240" in out["stdout"]
    assert out["input_files"] == ["sales.csv"]
    assert out["files_created"] == ["out.csv"]
    assert "out.csv" in ctx.store  # saved as an artifact -> travels back
    assert ctx.store["out.csv"].inline_data.mime_type == "text/csv"


async def test_input_files_are_not_sent_back_as_new():
    ctx = FakeCtx({"a.csv": b"x\n1\n"})
    out = await sandbox.run_python("print(open('a.csv').read())", ctx)
    assert out["files_created"] == []


async def test_chart_is_saved_without_a_display():
    ctx = FakeCtx()
    code = "import matplotlib.pyplot as plt\nplt.plot([1, 2, 3])\nplt.savefig('c.png')\n"
    out = await sandbox.run_python(code, ctx)
    assert out["success"] is True, out
    assert ctx.store["c.png"].inline_data.mime_type == "image/png"


async def test_secrets_are_not_passed_to_the_child(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-secret")
    out = await sandbox.run_python("import os; print(sorted(os.environ))", FakeCtx())
    assert out["success"] is True
    assert "ANTHROPIC_API_KEY" not in out["stdout"]
    assert "sk-test-secret" not in out["stdout"]


async def test_error_is_summarised():
    out = await sandbox.run_python("1 / 0", FakeCtx())
    assert out["success"] is False
    assert out["error"].startswith("ZeroDivisionError")
    assert "Traceback" in out["stderr"]


async def test_timeout_kills_the_script(monkeypatch):
    monkeypatch.setattr(sandbox, "TIMEOUT_SECONDS", 2)
    out = await sandbox.run_python("import time\nwhile True: time.sleep(0.1)", FakeCtx())
    assert out["success"] is False
    assert out["timed_out"] is True
    assert "Stopped after" in out["error"]


async def test_output_is_capped():
    out = await sandbox.run_python("print('x' * 50000)", FakeCtx())
    assert out["success"] is True
    assert len(out["stdout"]) < 11_000
    assert "more characters cut" in out["stdout"]


async def test_large_files_are_skipped(monkeypatch):
    monkeypatch.setattr(sandbox, "_MAX_FILE_BYTES", 10)
    ctx = FakeCtx()
    out = await sandbox.run_python("open('big.txt', 'w').write('y' * 100)", ctx)
    assert out["files_created"] == []
    assert out["files_skipped"][0].startswith("big.txt")
    assert "big.txt" not in ctx.store


async def test_each_run_has_its_own_folder(workspace):
    await sandbox.run_python("open('a.txt', 'w').write('1')", FakeCtx())
    await sandbox.run_python("import os; print(os.listdir('.'))", FakeCtx())
    runs = sorted((workspace / "runs").iterdir())
    assert len(runs) == 2
    assert not (runs[1] / "a.txt").exists()


async def test_works_without_a_tool_context():
    out = await sandbox.run_python("print('hi')")
    assert out["success"] is True
    assert out["input_files"] == []


async def test_empty_code():
    assert (await sandbox.run_python("   "))["success"] is False


async def test_artifact_names_cannot_escape_the_run_folder(workspace):
    ctx = FakeCtx({"../evil.csv": b"x"})
    out = await sandbox.run_python("import os; print(sorted(os.listdir('.')))", ctx)
    assert out["input_files"] == ["evil.csv"]
    assert not (workspace / "runs" / "evil.csv").exists()
    assert os.path.exists(workspace)  # nothing deleted outside
