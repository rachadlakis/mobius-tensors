"""
Document output tools: turn an employee's finished work into a file.

Every file is written to WORKSPACE_DIR and, when called as an agent tool, also
saved as a session artifact so the Chief of Staff (and the user) receive the
actual file, not just a server-local path.
"""

import asyncio
import inspect
import logging
import re
from datetime import datetime
from pathlib import Path

from fpdf import FPDF
from google.adk.agents.llm_agent import ToolUnion
from google.adk.tools.tool_context import ToolContext
from google.genai import types as genai_types

logger = logging.getLogger(__name__)

WORKSPACE_DIR = Path(__file__).parent.parent / "workspace"

_MAX_TABLE_ROWS = 200


def _latin1(text: object) -> str:
    """fpdf's core fonts are Latin-1 only; replace anything else with '?'
    rather than failing the whole document."""
    return str(text).encode("latin-1", "replace").decode("latin-1")


def _safe_stem(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_").lower()
    return stem[:60] or "report"


class ReportPDF(FPDF):
    """A plain A4 report: title on every page, page number in the footer."""

    def __init__(self, title: str):
        super().__init__()
        self.report_title = _latin1(title)
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, self.report_title, align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(
            0, 10,
            f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')} | Page {self.page_no()}",
            align="C",
        )

    def add_heading(self, text: str):
        self.set_font("Helvetica", "B", 12)
        self.set_fill_color(240, 240, 240)
        self.cell(0, 9, _latin1(text), fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def add_text(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 6, _latin1(text), new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def add_table(self, headers: list, rows: list):
        if not headers:
            return
        col_width = (self.w - 20) / len(headers)
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(210, 210, 210)
        for h in headers:
            self.cell(col_width, 7, _latin1(h)[:30], border=1, fill=True, align="C")
        self.ln()
        self.set_font("Helvetica", "", 8)
        for row in rows[:_MAX_TABLE_ROWS]:
            for i in range(len(headers)):
                cell = row[i] if i < len(row) else ""
                self.cell(col_width, 6, _latin1(cell)[:35], border=1)
            self.ln()
        self.ln(3)


async def _save_file(
    data: bytes, filename: str, mime: str, tool_context: ToolContext | None
) -> str:
    """Write `data` to WORKSPACE_DIR and, when called as a tool, save it as an
    artifact too. Returns the on-disk path."""
    WORKSPACE_DIR.mkdir(exist_ok=True)
    filepath = WORKSPACE_DIR / filename
    # Blocking disk write off the event loop: these tools are coroutines and
    # must not stall the employee's server (a stalled loop fails health checks).
    await asyncio.to_thread(filepath.write_bytes, data)
    if tool_context is not None:
        try:
            part = genai_types.Part(inline_data=genai_types.Blob(data=data, mime_type=mime))
            result = tool_context.save_artifact(filename, part)
            if inspect.isawaitable(result):
                await result
        except Exception as e:  # artifact service missing / backend error
            logger.warning("could not save artifact %s: %s", filename, e)
    return str(filepath)


async def write_pdf_report(
    title: str,
    sections: list[dict],
    file_name: str | None = None,
    tool_context: ToolContext | None = None,
) -> dict:
    """
    Write a PDF report and return it to the user as a downloadable file.

    Args:
        title: the report title, shown at the top of every page.
        sections: the report body, in order. Each section is a dict with any of:
            "heading" (str), "text" (str, paragraphs separated by blank lines),
            "table" ({"headers": [str, ...], "rows": [[value, ...], ...]}).
            Tables show at most 200 rows.
        file_name: optional base name (no extension). Defaults to the title.

    Returns:
        dict: success, file_name, file_path, sections; on failure success=False
            and an error.
    """
    try:
        if not isinstance(sections, list) or not sections:
            return {"success": False, "error": "sections must be a non-empty list."}
        pdf = ReportPDF(title)
        pdf.add_page()
        for section in sections:
            if not isinstance(section, dict):
                continue
            if section.get("heading"):
                pdf.add_heading(section["heading"])
            if section.get("text"):
                pdf.add_text(section["text"])
            table = section.get("table")
            if isinstance(table, dict):
                pdf.add_table(table.get("headers") or [], table.get("rows") or [])

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{_safe_stem(file_name or title)}_{stamp}.pdf"
        filepath = await _save_file(
            bytes(pdf.output()), filename, "application/pdf", tool_context
        )
        return {
            "success": True,
            "file_name": filename,
            "file_path": filepath,
            "sections": len(sections),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def list_workspace_files() -> dict:
    """
    List the files employees have produced in the workspace, newest first.

    Returns:
        dict: success, total, and files - a list of {file_name, size_kb, created_at}.
    """
    try:
        files = []
        if WORKSPACE_DIR.exists():
            for f in WORKSPACE_DIR.iterdir():
                if f.is_file():
                    st = f.stat()
                    files.append({
                        "file_name": f.name,
                        "size_kb": round(st.st_size / 1024, 2),
                        "created_at": datetime.fromtimestamp(st.st_mtime).isoformat(),
                    })
        files.sort(key=lambda x: x["created_at"], reverse=True)
        return {"success": True, "files": files, "total": len(files)}
    except Exception as e:
        return {"success": False, "error": str(e)}


DOCUMENT_TOOLS: list[ToolUnion] = [write_pdf_report, list_workspace_files]
