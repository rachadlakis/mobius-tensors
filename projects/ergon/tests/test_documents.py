"""Tests for the document output tools in shared/documents.py.

PDFs are really written (to a tmp dir) so the fpdf layout code is exercised
end to end.
"""

import os
import time

import pytest

from shared import documents

SECTIONS = [
    {"heading": "Summary", "text": "Revenue grew 12%.\n\nCosts were flat."},
    {"heading": "By region", "table": {
        "headers": ["Region", "Revenue"],
        "rows": [["North", 120], ["South", 80]],
    }},
]


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    monkeypatch.setattr(documents, "WORKSPACE_DIR", tmp_path)
    return tmp_path


def _assert_pdf_written(result, workspace):
    assert result["success"] is True
    path = workspace / result["file_name"]
    assert path.exists()
    assert path.read_bytes().startswith(b"%PDF")


class TestWritePdfReport:
    async def test_writes_pdf(self, workspace):
        result = await documents.write_pdf_report("Q3 Sales", SECTIONS)
        _assert_pdf_written(result, workspace)
        assert result["file_name"].startswith("q3_sales_")
        assert result["sections"] == 2

    async def test_custom_file_name_is_sanitised(self, workspace):
        result = await documents.write_pdf_report("T", SECTIONS, file_name="../Board pack!")
        _assert_pdf_written(result, workspace)
        assert result["file_name"].startswith("board_pack_")

    async def test_non_latin_text_does_not_fail(self, workspace):
        result = await documents.write_pdf_report(
            "تقرير", [{"heading": "ملخص", "text": "€ 100 — ok"}]
        )
        _assert_pdf_written(result, workspace)

    async def test_ragged_table_rows(self, workspace):
        result = await documents.write_pdf_report("T", [{"table": {
            "headers": ["a", "b", "c"], "rows": [[1], [1, 2, 3, 4]],
        }}])
        _assert_pdf_written(result, workspace)

    async def test_empty_sections_is_an_error(self, workspace):
        result = await documents.write_pdf_report("T", [])
        assert result["success"] is False


class TestArtifactSaving:
    async def test_saves_pdf_as_artifact_when_tool_context_present(self, workspace):
        saved = {}

        class FakeCtx:
            def save_artifact(self, filename, part):
                saved["filename"] = filename
                saved["bytes"] = part.inline_data.data
                saved["mime"] = part.inline_data.mime_type
                return 1  # sync, like a stub artifact service

        result = await documents.write_pdf_report("T", SECTIONS, tool_context=FakeCtx())
        assert result["success"] is True
        assert saved["filename"] == result["file_name"]
        assert saved["bytes"].startswith(b"%PDF")
        assert saved["mime"] == "application/pdf"

    async def test_artifact_save_failure_does_not_break_report(self, workspace):
        class BoomCtx:
            def save_artifact(self, filename, part):
                raise RuntimeError("no artifact service")

        result = await documents.write_pdf_report("T", SECTIONS, tool_context=BoomCtx())
        assert result["success"] is True  # PDF still written to disk

    async def test_async_save_artifact_is_awaited(self, workspace):
        seen = {}

        class AsyncCtx:
            async def save_artifact(self, filename, part):  # like the real ADK ToolContext
                seen["filename"] = filename
                return 1

        result = await documents.write_pdf_report("T", SECTIONS, tool_context=AsyncCtx())
        assert result["success"] is True
        assert seen["filename"] == result["file_name"]


class TestListWorkspaceFiles:
    def test_lists_files_newest_first(self, workspace):
        old = workspace / "old.pdf"
        new = workspace / "new.csv"
        old.write_bytes(b"%PDF-old")
        new.write_bytes(b"a,b")
        os.utime(old, (time.time() - 100, time.time() - 100))

        result = documents.list_workspace_files()
        assert result["success"] is True
        assert result["total"] == 2
        assert result["files"][0]["file_name"] == "new.csv"
        assert "size_kb" in result["files"][0]

    def test_missing_dir(self, monkeypatch, tmp_path):
        monkeypatch.setattr(documents, "WORKSPACE_DIR", tmp_path / "nope")
        assert documents.list_workspace_files() == {"success": True, "files": [], "total": 0}
