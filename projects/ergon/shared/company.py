"""
The company Ergon works for: its profile and its knowledge documents.

Ergon is general: each company is a folder ``companies/<id>/`` with

    profile.yaml     name, summary, about, offer, voice, rules
    knowledge/       Markdown or plain-text documents employees can search

and ``ERGON_COMPANY`` in .env picks one (default ``example``, a made-up sample).
``ERGON_COMPANY`` can also be a path to a company folder anywhere on disk.

The profile goes into every employee's instructions (and the Chief of Staff's) on
every turn, so edits apply without a restart. The knowledge folder is searched
with ``search_company_knowledge``: keyword ranking (BM25) over sections of each
document, rebuilt automatically when a file changes. No database, no embeddings.
"""

from __future__ import annotations

import asyncio
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

COMPANIES_DIR = Path(__file__).parent.parent / "companies"
DEFAULT_COMPANY = "example"

KNOWLEDGE_EXTS = (".md", ".txt")
_MAX_SECTION_CHARS = 1500
_MAX_DOCUMENT_CHARS = 20_000


class CompanyConfigError(ValueError):
    """The selected company folder or its profile is missing or invalid."""


# --------------------------------------------------------------------------- #
# Which company
# --------------------------------------------------------------------------- #
def company_dir() -> Path:
    """The selected company's folder (read from the environment on every call)."""
    choice = (os.getenv("ERGON_COMPANY") or DEFAULT_COMPANY).strip()
    path = Path(choice)
    if not path.is_absolute() and len(path.parts) == 1:  # a bare id, not a path
        path = COMPANIES_DIR / choice
    if not (path / "profile.yaml").is_file():
        known = sorted(p.name for p in COMPANIES_DIR.glob("*") if (p / "profile.yaml").is_file())
        raise CompanyConfigError(
            f"No company profile at {path / 'profile.yaml'}. "
            f"Set ERGON_COMPANY to one of: {', '.join(known) or 'none'} (or a folder path)."
        )
    return path


# --------------------------------------------------------------------------- #
# Profile
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Profile:
    name: str
    summary: str
    about: str = ""
    offer: list[str] = field(default_factory=list)
    voice: str = ""
    rules: list[str] = field(default_factory=list)


_profile_cache: dict[Path, tuple[float, Profile]] = {}


def _str_list(data: dict, key: str) -> list[str]:
    value = data.get(key) or []
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise CompanyConfigError(f"profile.yaml: '{key}' must be a list of strings.")
    return [v.strip() for v in value if v.strip()]


def load_profile(directory: Path | None = None) -> Profile:
    """The company profile, re-read only when profile.yaml changes."""
    path = (directory or company_dir()) / "profile.yaml"
    mtime = path.stat().st_mtime
    cached = _profile_cache.get(path)
    if cached and cached[0] == mtime:
        return cached[1]
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise CompanyConfigError(f"profile.yaml is not valid YAML: {e}") from e
    if not isinstance(data, dict):
        raise CompanyConfigError("profile.yaml: expected a mapping at the top level.")
    for key in ("name", "summary"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise CompanyConfigError(f"profile.yaml: '{key}' is required.")
    profile = Profile(
        name=data["name"].strip(),
        summary=data["summary"].strip(),
        about=str(data.get("about") or "").strip(),
        offer=_str_list(data, "offer"),
        voice=str(data.get("voice") or "").strip(),
        rules=_str_list(data, "rules"),
    )
    _profile_cache[path] = (mtime, profile)
    return profile


def render_profile(profile: Profile) -> str:
    """The profile as a Markdown block for an instruction."""
    out = [f"## Your company: {profile.name}", "", profile.summary]
    if profile.about:
        out += ["", profile.about]
    if profile.offer:
        out += ["", "**What we offer:**"] + [f"- {item}" for item in profile.offer]
    if profile.voice:
        out += ["", f"**How we write:** {profile.voice}"]
    if profile.rules:
        out += ["", "**Rules you always follow:**"] + [f"- {rule}" for rule in profile.rules]
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Knowledge: sections of each document, ranked with BM25
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Section:
    document: str  # file name, e.g. "pricing.md"
    heading: str   # "Title > Section"
    text: str


_STOPWORDS = frozenset(
    "a an and are as at be by for from has have how i in is it its of on or our the this to "
    "was we what when where which who why will with you your do does can".split()
)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


_SUFFIXES = ("ing", "ed", "es", "s")


def _stem(token: str) -> str:
    """A light English stemmer: "returns", "returned" and "returning" all
    become "return"; "price" and "pricing" both become "pric". Words in other
    scripts are left alone."""
    if not token.isascii():
        return token
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            token = token[: -len(suffix)]
            break
    if len(token) > 4 and token.endswith("e"):
        token = token[:-1]
    return token


def _tokens(text: str) -> list[str]:
    return [_stem(t) for t in re.findall(r"\w+", text.lower()) if t not in _STOPWORDS]


def _split_long(text: str) -> list[str]:
    """Split text longer than _MAX_SECTION_CHARS on paragraph breaks."""
    if len(text) <= _MAX_SECTION_CHARS:
        return [text]
    parts, current = [], ""
    for para in re.split(r"\n\s*\n", text):
        if current and len(current) + len(para) > _MAX_SECTION_CHARS:
            parts.append(current.strip())
            current = ""
        current += para + "\n\n"
    if current.strip():
        parts.append(current.strip())
    return parts


def split_sections(name: str, text: str) -> list[Section]:
    """Markdown headings start sections; each heading path is kept with its text."""
    sections: list[Section] = []
    path: list[str] = []
    body: list[str] = []

    def flush():
        content = "\n".join(body).strip()
        if content:
            heading = " > ".join(path) or name
            sections.extend(Section(name, heading, part) for part in _split_long(content))

    for line in text.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            flush()
            body = []
            level = len(m.group(1))
            path = path[: level - 1] + [m.group(2).strip()]
        else:
            body.append(line)
    flush()
    return sections


class KnowledgeIndex:
    """BM25 over every section of every knowledge document."""

    def __init__(self, sections: list[Section]):
        self.sections = sections
        # The heading counts twice: a section titled "Returns" is about returns.
        self._docs = [
            Counter(_tokens(f"{s.heading} {s.heading} {s.document} {s.text}")) for s in sections
        ]
        self._lengths = [sum(d.values()) for d in self._docs]
        self._avg = (sum(self._lengths) / len(self._lengths)) if sections else 0.0
        df: Counter = Counter()
        for d in self._docs:
            df.update(d.keys())
        n = len(sections)
        self._idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def search(self, query: str, limit: int = 5) -> list[tuple[float, Section]]:
        terms = _tokens(query)
        k1, b = 1.5, 0.75
        scored = []
        for doc, length, section in zip(self._docs, self._lengths, self.sections):
            score = 0.0
            for t in terms:
                tf = doc.get(t)
                if tf:
                    norm = k1 * (1 - b + b * length / (self._avg or 1))
                    score += self._idf[t] * tf * (k1 + 1) / (tf + norm)
            if score > 0:
                scored.append((score, section))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:limit]


_index_cache: dict[Path, tuple[tuple, KnowledgeIndex]] = {}


def knowledge_files(directory: Path | None = None) -> list[Path]:
    folder = (directory or company_dir()) / "knowledge"
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in KNOWLEDGE_EXTS)


def knowledge_index(directory: Path | None = None) -> KnowledgeIndex:
    """The index for the company, rebuilt only when a knowledge file changes."""
    root = directory or company_dir()
    files = knowledge_files(root)
    signature = tuple((str(p), p.stat().st_mtime, p.stat().st_size) for p in files)
    cached = _index_cache.get(root)
    if cached and cached[0] == signature:
        return cached[1]
    folder = root / "knowledge"
    sections: list[Section] = []
    for p in files:
        name = p.relative_to(folder).as_posix()
        sections += split_sections(name, p.read_text(encoding="utf-8", errors="replace"))
    index = KnowledgeIndex(sections)
    _index_cache[root] = (signature, index)
    return index


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
def _search_blocking(query: str, max_results: int) -> dict:
    query = (query or "").strip()
    if not query:
        return {"success": False, "error": "Empty query."}
    try:
        root = company_dir()
        index = knowledge_index(root)
    except CompanyConfigError as e:
        return {"success": False, "error": str(e)}
    limit = max(1, min(int(max_results or 5), 10))
    hits = index.search(query, limit)
    return {
        "success": True,
        "company": load_profile(root).name,
        "query": query,
        "count": len(hits),
        "results": [
            {"document": s.document, "section": s.heading, "text": s.text, "score": round(score, 2)}
            for score, s in hits
        ],
        "note": "" if hits else (
            "Nothing matched. Try other words, or read_company_document() to list the documents."
        ),
    }


def _read_blocking(name: str | None) -> dict:
    try:
        root = company_dir()
    except CompanyConfigError as e:
        return {"success": False, "error": str(e)}
    folder = root / "knowledge"
    files = {p.relative_to(folder).as_posix(): p for p in knowledge_files(root)}
    if not name:
        return {"success": True, "documents": sorted(files)}
    if name not in files:
        return {"success": False, "error": f"No document named {name!r}.", "documents": sorted(files)}
    text = files[name].read_text(encoding="utf-8", errors="replace")
    return {
        "success": True,
        "document": name,
        "text": text[:_MAX_DOCUMENT_CHARS],
        "truncated": len(text) > _MAX_DOCUMENT_CHARS,
    }


async def search_company_knowledge(query: str, max_results: int = 5) -> dict:
    """
    Search the company's own documents (products, prices, policies, FAQs) and
    return the best-matching sections. Use it before answering anything about
    the company's offer, prices or rules - never answer those from memory.

    Args:
        query: keywords or a question, e.g. "price per extra camera".
        max_results: how many sections to return (default 5, at most 10).

    Returns:
        dict: success, company, query, count, and results - a list of
            {document, section, text, score}, best first.
    """
    return await asyncio.to_thread(_search_blocking, query, max_results)


async def read_company_document(name: str | None = None) -> dict:
    """
    Read one of the company's documents in full. Call it with no name to list
    the documents.

    Args:
        name: the document's file name as listed (e.g. "pricing.md").

    Returns:
        dict: success, document, text, truncated; or the list of documents.
    """
    return await asyncio.to_thread(_read_blocking, name)


KNOWLEDGE_TOOLS = (search_company_knowledge, read_company_document)
KNOWLEDGE_TOOL_NAMES = frozenset(t.__name__ for t in KNOWLEDGE_TOOLS)


# --------------------------------------------------------------------------- #
# Instructions
# --------------------------------------------------------------------------- #
def company_block(tool_names: set[str] | frozenset[str] = frozenset(), *, for_chief: bool = False) -> str:
    """What every instruction starts with: the profile, plus how to reach the
    company documents."""
    root = company_dir()
    text = render_profile(load_profile(root))
    docs = [p.relative_to(root / "knowledge").as_posix() for p in knowledge_files(root)]
    if docs:
        if KNOWLEDGE_TOOL_NAMES & set(tool_names):
            text += (
                f"\n\n**Company documents:** {', '.join(docs)}. Look things up with "
                "`search_company_knowledge` (or read one with `read_company_document`) before "
                "answering about our products, prices or policies. Never guess them."
            )
        elif for_chief:
            text += (
                f"\n\n**Company documents:** {', '.join(docs)}. Your employees can search them; "
                "delegate questions about our products, prices or policies."
            )
    return text
