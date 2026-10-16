"""Tests for shared/attachments.py - uploaded data file -> LLM-readable text."""

import io

import pytest

from shared import attachments


def _xlsx_bytes(rows, sheet_title="Sheet1", extra_sheets=None):
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = sheet_title
    for row in rows:
        ws.append(row)
    for name, srows in (extra_sheets or {}).items():
        ws2 = wb.create_sheet(name)
        for row in srows:
            ws2.append(row)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


class TestLooksTabular:
    @pytest.mark.parametrize("mime, name", [
        (XLSX_MIME, "x.xlsx"),
        ("text/csv", "x.csv"),
        (None, "data.CSV"),
        ("application/octet-stream", "report.xlsx"),
        ("text/tab-separated-values", "x.tsv"),
    ])
    def test_true_for_tabular(self, mime, name):
        assert attachments.looks_tabular(mime, name) is True

    @pytest.mark.parametrize("mime, name", [
        ("application/pdf", "x.pdf"),
        ("image/png", "chart.png"),
        (None, "notes.docx"),
    ])
    def test_false_for_non_tabular(self, mime, name):
        assert attachments.looks_tabular(mime, name) is False


class TestXlsxToText:
    def test_basic_grid_becomes_csv(self):
        raw = _xlsx_bytes([["name", "city"], ["Branch A", "Dubai"]])
        out = attachments.to_text(raw, XLSX_MIME, "sales.xlsx")
        assert "name,city" in out
        assert "Branch A,Dubai" in out

    def test_arabic_survives(self):
        raw = _xlsx_bytes([["الاسم", "المدينة"], ["المستودع الرئيسي", "دبي"]])
        out = attachments.to_text(raw, XLSX_MIME, "sales.xlsx")
        assert "المستودع الرئيسي,دبي" in out
        assert "الاسم,المدينة" in out

    def test_detected_by_extension_when_mime_is_generic(self):
        raw = _xlsx_bytes([["a", "b"]])
        out = attachments.to_text(raw, "application/octet-stream", "upload.xlsx")
        assert "a,b" in out

    def test_multiple_sheets_are_labelled(self):
        raw = _xlsx_bytes([["x"]], sheet_title="first",
                          extra_sheets={"second": [["y"]]})
        out = attachments.to_text(raw, XLSX_MIME, "multi.xlsx")
        assert "# sheet: first" in out
        assert "# sheet: second" in out

    def test_row_cap_is_enforced(self):
        rows = [[f"r{i}"] for i in range(attachments.MAX_ROWS + 50)]
        out = attachments.to_text(_xlsx_bytes(rows), XLSX_MIME, "big.xlsx")
        assert "truncated at" in out

    def test_corrupt_xlsx_returns_none(self):
        assert attachments.to_text(b"not a zip file", XLSX_MIME, "bad.xlsx") is None


class TestCsvToText:
    def test_utf8_passes_through(self):
        raw = "name,city\nBranch A,Dubai\n".encode("utf-8")
        out = attachments.to_text(raw, "text/csv", "f.csv")
        assert out.strip() == "name,city\nBranch A,Dubai"

    def test_utf8_bom_is_stripped(self):
        raw = "name,city\nA,Dubai\n".encode("utf-8-sig")  # Excel-style BOM prefix
        assert raw.startswith(b"\xef\xbb\xbf")
        out = attachments.to_text(raw, "text/csv", "f.csv")
        assert out.startswith("name,city")

    def test_cp1256_arabic_is_recovered(self):
        raw = "الاسم,المدينة\nالمستودع,دبي\n".encode("cp1256")
        out = attachments.to_text(raw, "text/csv", "arabic.csv")
        assert "الاسم,المدينة" in out
        assert "المستودع,دبي" in out

    def test_non_tabular_returns_none(self):
        assert attachments.to_text(b"%PDF-1.4", "application/pdf", "x.pdf") is None