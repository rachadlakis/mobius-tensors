"""
Tests for the A2A <-> Google GenAI part converters in shared/a2a_files.py.

The worker `agent_executor.py` modules all import these from that one module
(they used to ship byte-identical copies), so there is nothing to drift.
"""

import base64

import pytest
from a2a.types import FilePart, FileWithBytes, FileWithUri, Part, TextPart
from google.genai import types

from shared import a2a_files as de


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


class TestA2AToGenAI:
    def test_text_part(self):
        out = de.convert_a2a_part_to_genai(Part(root=TextPart(text="hello")))
        assert out.text == "hello"

    def test_file_with_uri(self):
        part = Part(root=FilePart(file=FileWithUri(uri="gs://b/x.pdf", mime_type="application/pdf")))
        out = de.convert_a2a_part_to_genai(part)
        assert out.file_data.file_uri == "gs://b/x.pdf"
        assert out.file_data.mime_type == "application/pdf"

    def test_file_with_bytes_is_base64_decoded(self):
        part = Part(root=FilePart(
            file=FileWithBytes(bytes=_b64(b"payload"), mime_type="text/plain")
        ))
        out = de.convert_a2a_part_to_genai(part)
        assert out.inline_data.data == b"payload"
        assert out.inline_data.mime_type == "text/plain"

    def test_file_with_bytes_preserves_arabic(self):
        raw = "المستودع الرئيسي,دبي\n".encode("utf-8")
        part = Part(root=FilePart(
            file=FileWithBytes(bytes=_b64(raw), mime_type="text/csv")
        ))
        out = de.convert_a2a_part_to_genai(part)
        assert out.inline_data.data == raw
        assert out.inline_data.data.decode("utf-8").startswith("المستودع")

    def test_file_with_bytes_non_base64_falls_back_to_raw_text(self):
        part = Part(root=FilePart(file=FileWithBytes(bytes="payload!", mime_type="text/plain")))
        out = de.convert_a2a_part_to_genai(part)
        assert out.inline_data.data == b"payload!"

    def test_file_with_bytes_defaults_mime_type(self):
        part = Part(root=FilePart(file=FileWithBytes(bytes=_b64(b"x"), mime_type=None)))
        out = de.convert_a2a_part_to_genai(part)
        assert out.inline_data.mime_type == "application/octet-stream"

    def test_list_conversion(self):
        parts = [Part(root=TextPart(text="a")), Part(root=TextPart(text="b"))]
        out = de.convert_a2a_parts_to_genai(parts)
        assert [p.text for p in out] == ["a", "b"]


class TestGenAIToA2A:
    def test_text_part(self):
        out = de.convert_genai_part_to_a2a(types.Part(text="hi"))
        assert isinstance(out.root, TextPart)
        assert out.root.text == "hi"

    def test_file_data_part(self):
        part = types.Part(file_data=types.FileData(file_uri="gs://b/x", mime_type="image/png"))
        out = de.convert_genai_part_to_a2a(part)
        assert isinstance(out.root, FilePart)
        assert out.root.file.uri == "gs://b/x"

    def test_file_data_without_uri_raises(self):
        part = types.Part(file_data=types.FileData(file_uri=None, mime_type="image/png"))
        with pytest.raises(ValueError):
            de.convert_genai_part_to_a2a(part)

    def test_inline_data_part_is_base64_encoded(self):
        part = types.Part(inline_data=types.Blob(data=b"bytes", mime_type="text/plain"))
        out = de.convert_genai_part_to_a2a(part)
        assert isinstance(out.root, FilePart)
        assert out.root.file.bytes == _b64(b"bytes")
        assert base64.b64decode(out.root.file.bytes) == b"bytes"

    def test_empty_parts_are_filtered_out(self):
        parts = [types.Part(text="keep"), types.Part(text=""), types.Part(text=None)]
        out = de.convert_genai_parts_to_a2a(parts)
        assert len(out) == 1
        assert out[0].root.text == "keep"

    def test_unsupported_part_raises(self):
        with pytest.raises(ValueError):
            de.convert_genai_part_to_a2a(types.Part())


class TestRoundTrip:
    def test_text_survives_round_trip(self):
        original = [Part(root=TextPart(text="round trip"))]
        genai = de.convert_a2a_parts_to_genai(original)
        back = de.convert_genai_parts_to_a2a(genai)
        assert back[0].root.text == "round trip"

    def test_binary_file_survives_round_trip(self):
        raw = bytes(range(256)) + "مرحبا".encode("utf-8")
        original = [Part(root=FilePart(file=FileWithBytes(bytes=_b64(raw), mime_type="application/octet-stream")))]
        genai = de.convert_a2a_parts_to_genai(original)
        assert genai[0].inline_data.data == raw
        back = de.convert_genai_parts_to_a2a(genai)
        assert base64.b64decode(back[0].root.file.bytes) == raw


class TestFilePartBuilders:
    def test_file_part_dict_shape(self):
        d = de.file_part_dict("r.pdf", b"%PDF-1.4", "application/pdf")
        assert d["kind"] == "file"
        assert d["file"]["name"] == "r.pdf"
        assert d["file"]["mimeType"] == "application/pdf"
        assert base64.b64decode(d["file"]["bytes"]) == b"%PDF-1.4"
        assert Part.model_validate(d).root.file.name == "r.pdf"

    def test_file_part_typed(self):
        p = de.file_part("chart.png", b"\x89PNG", "image/png")
        assert isinstance(p.root, FilePart)
        assert p.root.file.name == "chart.png"
        assert base64.b64decode(p.root.file.bytes) == b"\x89PNG"

    def test_file_part_defaults_mime(self):
        p = de.file_part("x", b"data", None)
        assert p.root.file.mime_type == "application/octet-stream"

    @pytest.mark.parametrize("n, expected", [
        (12, "12 B"), (2048, "2.0 KB"), (5 * 1024 * 1024, "5.0 MB"),
    ])
    def test_human_size(self, n, expected):
        assert de.human_size(n) == expected
