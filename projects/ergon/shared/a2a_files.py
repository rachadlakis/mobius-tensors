"""A2A <-> Google GenAI part conversion, plus file/artifact plumbing.

All four worker `agent_executor.py` modules import the converters from here (one
source of truth - they used to ship byte-identical copies). The extra helpers
support passing real files between agents:

* `convert_a2a_part_to_genai` base64-decodes an inbound `FileWithBytes` (per the
  A2A spec `bytes` is base64) so a forwarded upload reaches the model as its
  actual bytes.
* `convert_genai_part_to_a2a` base64-encodes an outbound inline blob, so a file
  the agent produced leaves as a valid A2A `FilePart`.
* `file_part_dict` / `file_part` build an A2A file part from raw bytes - used
  when harvesting artifacts an agent saved during a turn.
"""

from __future__ import annotations

import base64
import binascii

from a2a.types import FilePart, FileWithBytes, FileWithUri, Part, TextPart
from google.genai import types

DEFAULT_MIME = "application/octet-stream"


# --------------------------------------------------------------------------- #
# A2A  ->  Google GenAI
# --------------------------------------------------------------------------- #
def convert_a2a_parts_to_genai(parts: list[Part]) -> list[types.Part]:
    """Convert a list of A2A Part types into a list of Google Gen AI Part types."""
    return [convert_a2a_part_to_genai(part) for part in parts]


def convert_a2a_part_to_genai(part: Part) -> types.Part:
    """Convert a single A2A Part type into a Google Gen AI Part type."""
    root = part.root
    if isinstance(root, TextPart):
        return types.Part(text=root.text)
    if isinstance(root, FilePart):
        if isinstance(root.file, FileWithUri):
            return types.Part(
                file_data=types.FileData(
                    file_uri=root.file.uri, mime_type=root.file.mime_type
                )
            )
        if isinstance(root.file, FileWithBytes):
            # A2A FileWithBytes.bytes is a base64-encoded string (per the spec),
            # so decode it back to real bytes. Only fall back to treating it as
            # raw UTF-8 text if it isn't valid base64 (a loose/legacy sender).
            try:
                data = base64.b64decode(root.file.bytes, validate=True)
            except (binascii.Error, ValueError):
                data = root.file.bytes.encode("utf-8")
            return types.Part(
                inline_data=types.Blob(
                    data=data,
                    mime_type=root.file.mime_type or DEFAULT_MIME,
                    # keep the filename so a worker can save it as a named
                    # artifact and a tool can then load it by that name
                    display_name=getattr(root.file, "name", None),
                )
            )
        raise ValueError(f"Unsupported file type: {type(root.file)}")
    raise ValueError(f"Unsupported part type: {type(part)}")


# --------------------------------------------------------------------------- #
# Google GenAI  ->  A2A
# --------------------------------------------------------------------------- #
def convert_genai_parts_to_a2a(parts: list[types.Part]) -> list[Part]:
    """Convert a list of Google Gen AI Part types into a list of A2A Part types."""
    return [
        convert_genai_part_to_a2a(part)
        for part in parts
        if (part.text or part.file_data or part.inline_data)
    ]


def convert_genai_part_to_a2a(part: types.Part) -> Part:
    """Convert a single Google Gen AI Part type into an A2A Part type."""
    if part.text:
        return Part(root=TextPart(text=part.text))
    if part.file_data:
        if not part.file_data.file_uri:
            raise ValueError("File URI is missing")
        return Part(
            root=FilePart(
                file=FileWithUri(
                    uri=part.file_data.file_uri,
                    mime_type=part.file_data.mime_type,
                )
            )
        )
    if part.inline_data:
        if not part.inline_data.data:
            raise ValueError("Inline data is missing")
        return Part(
            root=FilePart(
                file=FileWithBytes(
                    bytes=base64.b64encode(part.inline_data.data).decode("ascii"),
                    mime_type=part.inline_data.mime_type,
                )
            )
        )
    raise ValueError(f"Unsupported part type: {part}")


# --------------------------------------------------------------------------- #
# Building file parts from raw bytes
# --------------------------------------------------------------------------- #
def file_part_dict(name: str, data: bytes, mime_type: str | None) -> dict:
    """A2A file message part (dict form, for a raw send_message payload)."""
    return {
        "kind": "file",
        "file": {
            "bytes": base64.b64encode(data).decode("ascii"),
            "mimeType": mime_type or DEFAULT_MIME,
            "name": name,
        },
    }


def file_part(name: str, data: bytes, mime_type: str | None) -> Part:
    """A2A file Part (typed, for TaskUpdater.add_artifact)."""
    return Part(
        root=FilePart(
            file=FileWithBytes(
                bytes=base64.b64encode(data).decode("ascii"),
                mime_type=mime_type or DEFAULT_MIME,
                name=name,
            )
        )
    )


def human_size(n: float) -> str:
    if n < 1024:
        return f"{n:.0f} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"
