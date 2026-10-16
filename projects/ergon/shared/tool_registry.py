"""
Every tool an employee can be given, by name.

An employee's YAML lists tool names; this registry turns them into the actual
functions. Adding a tool to Ergon means writing it in a shared module and
registering it here - nothing else.
"""

from collections.abc import Callable

from shared.calculator import calculate
from shared.company import read_company_document, search_company_knowledge
from shared.documents import (
    list_workspace_files,
    write_pdf_report,
    write_spreadsheet,
    write_text_file,
    write_word_document,
)
from shared.sandbox import run_python
from shared.web_tools import fetch_page, get_weather, web_search

TOOLS: dict[str, Callable] = {
    tool.__name__: tool
    for tool in (
        # web (shared/web_tools.py)
        web_search,
        fetch_page,
        get_weather,
        # documents (shared/documents.py)
        write_pdf_report,
        write_word_document,
        write_spreadsheet,
        write_text_file,
        list_workspace_files,
        # numbers (shared/calculator.py) and code (shared/sandbox.py)
        calculate,
        run_python,
        # the company's own knowledge (shared/company.py)
        search_company_knowledge,
        read_company_document,
    )
}


def resolve_tools(names: list[str]) -> list[Callable]:
    """The tool functions for `names`, in order. Raises KeyError naming every
    unknown tool, so a typo in a YAML file fails at startup, not mid-task."""
    unknown = [n for n in names if n not in TOOLS]
    if unknown:
        raise KeyError(
            f"Unknown tool(s): {', '.join(unknown)}. Available: {', '.join(sorted(TOOLS))}."
        )
    return [TOOLS[n] for n in names]


# Tools that open the task's files by name. An employee with one of them gets
# forwarded files as real files (not pasted text) and saves them on arrival.
FILE_TOOLS = frozenset({"run_python"})
