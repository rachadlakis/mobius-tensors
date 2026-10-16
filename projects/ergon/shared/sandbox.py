"""
run_python: let an employee (the Data Analyst) run Python on the files it was given.

Each call gets a fresh folder under ``workspace/runs/``. The session's files (uploads
the Chief of Staff forwarded, files earlier calls produced) are copied in by name, the
code runs there in a separate Python process, and every new file it writes (a chart, a
cleaned CSV, an Excel sheet) is saved as an artifact so it travels back to the user.

This is a *local subprocess with limits*, for internal use - not a security sandbox:

* the child gets a minimal environment (no API keys from .env), isolated mode (-I),
  a working folder of its own, and a time limit (SANDBOX_TIMEOUT_SECONDS, default 60);
* output and file sizes are capped;
* it can still read the disk and reach the network as the user running Ergon.

Run it in a container before letting anyone outside the company send it code
(PLAN.md, open decisions).
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import mimetypes
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from google.adk.tools.tool_context import ToolContext
from google.genai import types as genai_types

from shared import documents

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = float(os.getenv("SANDBOX_TIMEOUT_SECONDS", "60"))
_MAX_OUTPUT_CHARS = 10_000
_MAX_FILE_BYTES = 5 * 1024 * 1024
_MAX_NEW_FILES = 10
_MAX_CODE_CHARS = 50_000

# The only environment variables the child inherits: enough for Python, numpy and
# matplotlib to start on Windows, macOS and Linux. Nothing from .env.
# Fixed types for common outputs: Windows' registry maps .csv to an Excel type.
_MIME = {
    ".csv": "text/csv", ".txt": "text/plain", ".md": "text/markdown", ".json": "application/json",
    ".html": "text/html", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml", ".pdf": "application/pdf",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

_ENV_KEEP = ("PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP",
             "HOME", "USERPROFILE", "LANG", "LC_ALL", "NUMBER_OF_PROCESSORS")


def _child_env(run_dir: Path) -> dict[str, str]:
    env = {k: os.environ[k] for k in _ENV_KEEP if k in os.environ}
    env.update({
        "MPLBACKEND": "Agg",              # charts render to files, never a window
        "MPLCONFIGDIR": str(run_dir / ".mpl"),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    return env


def _cap(text: str) -> str:
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text
    return text[:_MAX_OUTPUT_CHARS] + f"\n... [{len(text) - _MAX_OUTPUT_CHARS} more characters cut]"


def _kill_tree(proc: subprocess.Popen) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
    else:
        proc.kill()


def _run_blocking(run_dir: Path, timeout: float) -> dict:
    """Run main.py in run_dir and collect what it printed."""
    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    proc = subprocess.Popen(
        [sys.executable, "-I", "main.py"],
        cwd=run_dir, env=_child_env(run_dir),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
        creationflags=flags,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        out, err = proc.communicate()
        timed_out = True
    return {
        "exit_code": proc.returncode,
        "timed_out": timed_out,
        "stdout": _cap(out.decode("utf-8", "replace")),
        "stderr": _cap(err.decode("utf-8", "replace")),
    }


async def _maybe_await(value):
    return await value if inspect.isawaitable(value) else value


async def _copy_session_files(tool_context: ToolContext | None, run_dir: Path) -> list[str]:
    """Write every session artifact into run_dir under its own (safe) name."""
    if tool_context is None:
        return []
    copied = []
    try:
        names = await _maybe_await(tool_context.list_artifacts())
    except Exception as e:  # no artifact service
        logger.warning("run_python: could not list artifacts: %s", e)
        return []
    for name in names or []:
        safe = Path(str(name)).name
        if not safe or safe.startswith("."):
            continue
        try:
            part = await _maybe_await(tool_context.load_artifact(name))
        except Exception as e:
            logger.warning("run_python: could not load %s: %s", name, e)
            continue
        data = getattr(getattr(part, "inline_data", None), "data", None)
        if isinstance(data, (bytes, bytearray)):
            (run_dir / safe).write_bytes(bytes(data))
            copied.append(safe)
    return copied


async def _save_new_files(
    tool_context: ToolContext | None, run_dir: Path, before: set[str]
) -> tuple[list[str], list[str]]:
    """Save files the code created as artifacts. Returns (saved, skipped)."""
    saved, skipped = [], []
    new = sorted(
        p for p in run_dir.iterdir()
        if p.is_file() and p.name not in before and p.name != "main.py"
    )
    for path in new:
        if len(saved) >= _MAX_NEW_FILES:
            skipped.append(f"{path.name} (more than {_MAX_NEW_FILES} files)")
            continue
        size = path.stat().st_size
        if size > _MAX_FILE_BYTES:
            skipped.append(f"{path.name} (larger than 5 MB)")
            continue
        if tool_context is not None:
            mime = (_MIME.get(path.suffix.lower())
                    or mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            try:
                await _maybe_await(tool_context.save_artifact(
                    path.name,
                    genai_types.Part(inline_data=genai_types.Blob(
                        data=path.read_bytes(), mime_type=mime)),
                ))
            except Exception as e:
                skipped.append(f"{path.name} (could not save: {e})")
                continue
        saved.append(path.name)
    return saved, skipped


async def run_python(code: str, tool_context: ToolContext | None = None) -> dict:
    """
    Run a Python script on the files you were given, and return what it printed
    plus any files it wrote.

    The script runs in its own folder. Every file attached to the task is in
    that folder under its own name (e.g. pd.read_excel("sales.xlsx")). pandas,
    numpy, matplotlib and openpyxl are available. Save charts with
    plt.savefig("chart.png") and tables with df.to_csv("result.csv") or
    df.to_excel("result.xlsx"): every new file is sent back to the user. print()
    what you want to read. Time limit 60 s by default; no input().

    Args:
        code: a complete Python script.

    Returns:
        dict: success, exit_code, timed_out, stdout, stderr, input_files,
            files_created, files_skipped.
    """
    code = code or ""
    if not code.strip():
        return {"success": False, "error": "Empty code."}
    if len(code) > _MAX_CODE_CHARS:
        return {"success": False, "error": f"Code longer than {_MAX_CODE_CHARS} characters."}

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = documents.WORKSPACE_DIR / "runs" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    inputs = await _copy_session_files(tool_context, run_dir)
    (run_dir / "main.py").write_text(code, encoding="utf-8")
    before = {p.name for p in run_dir.iterdir()}

    result = await asyncio.to_thread(_run_blocking, run_dir, TIMEOUT_SECONDS)
    saved, skipped = await _save_new_files(tool_context, run_dir, before)

    ok = result["exit_code"] == 0 and not result["timed_out"]
    out = {"success": ok, **result, "input_files": inputs,
           "files_created": saved, "files_skipped": skipped}
    if result["timed_out"]:
        out["error"] = f"Stopped after {TIMEOUT_SECONDS:.0f} s. Make the script faster or smaller."
    elif not ok:
        last = [ln for ln in result["stderr"].splitlines() if ln.strip()][-1:] or ["unknown error"]
        out["error"] = re.sub(r"\s+", " ", last[0])[:300]
    return out
