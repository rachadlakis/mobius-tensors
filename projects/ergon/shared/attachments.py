"""Turn a user-uploaded data file into text an LLM can actually read.

`adk web` attaches an upload to the user turn as an inline `Blob`. A binary
spreadsheet (`.xlsx`) is useless to the model as raw bytes, and a CSV forwarded
as an A2A file part arrives base64-encoded - which the model reads as noise.
This module converts the common tabular formats to a plain UTF-8 CSV string
(Arabic and other non-Latin text preserved) that the Orchestrator pastes into
the delegated task instead of forwarding the bytes.

Non-tabular uploads (PDF, images, ...) are left alone - `to_text` returns None
and the caller forwards them as a proper base64 file part.
"""

from __future__ import annotations

import csv
import io

# Caps so one big upload can't blow the model's context window. Applied per
# file; a truncation marker is appended when they bite.
MAX_ROWS = 500
MAX_CHARS = 60_000

# Extensions / MIME types we convert to CSV text.
_XLSX_EXTS = (".xlsx", ".xlsm")
_XLSX_MIMES = {
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel.sheet.macroenabled.12",
}
_TEXT_EXTS = (".csv", ".tsv", ".txt", ".tab")
_TEXT_MIMES = {
    "text/csv", "application/csv",
    "text/tab-separated-values",
    "text/plain",
}

# Decoding order for text files: BOM'd UTF-8, plain UTF-8, then Windows Arabic
# (cp1256 - what Excel's "Save As CSV" writes on an Arabic Windows) and Western
# (cp1252), then latin-1 which never raises and keeps the bytes recoverable.
_TEXT_ENCODINGS = ("utf-8-sig", "utf-8", "cp1256", "cp1252", "latin-1")


def looks_tabular(mime_type: str | None, filename: str | None) -> bool:
    """True if `to_text` would try to convert this upload (xlsx or delimited text)."""
    name = (filename or "").lower()
    mime = (mime_type or "").lower()
    if name.endswith(_XLSX_EXTS) or mime in _XLSX_MIMES:
        return True
    return name.endswith(_TEXT_EXTS) or mime in _TEXT_MIMES


def to_text(raw: bytes, mime_type: str | None, filename: str | None) -> str | None:
    """Best-effort conversion of an uploaded data file to CSV / plain text.

    Returns None when the file isn't a spreadsheet or delimited-text file, or
    when conversion fails - the caller should then forward the raw bytes.
    """
    name = (filename or "").lower()
    mime = (mime_type or "").lower()

    if name.endswith(_XLSX_EXTS) or mime in _XLSX_MIMES:
        return _xlsx_to_csv(raw)
    if name.endswith(_TEXT_EXTS) or mime in _TEXT_MIMES:
        return _cap(_decode_text(raw))
    return None


def _decode_text(raw: bytes) -> str:
    for enc in _TEXT_ENCODINGS:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")  # unreachable in practice; latin-1 maps all bytes


def _xlsx_to_csv(raw: bytes) -> str | None:
    try:
        from openpyxl import load_workbook
    except ImportError:  # openpyxl not installed - let the caller forward bytes
        return None
    try:
        wb = load_workbook(io.BytesIO(raw), read_only=True, data_only=True)
    except Exception:
        return None

    try:
        multi = len(wb.sheetnames) > 1
        chunks: list[str] = []
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            buf = io.StringIO()
            writer = csv.writer(buf)
            truncated = False
            for i, row in enumerate(ws.iter_rows(values_only=True)):
                if i >= MAX_ROWS:
                    truncated = True
                    break
                writer.writerow(["" if cell is None else cell for cell in row])
            body = buf.getvalue()
            if truncated:
                body += f"... (truncated at {MAX_ROWS} rows)\n"
            if multi:
                body = f"# sheet: {sheet_name}\n{body}"
            chunks.append(body)
            if sum(len(c) for c in chunks) >= MAX_CHARS:
                break
        return _cap("\n".join(chunks)) or None
    finally:
        wb.close()


def _cap(text: str) -> str:
    if len(text) > MAX_CHARS:
        return text[:MAX_CHARS] + "\n... (truncated)"
    return text