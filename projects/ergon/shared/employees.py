"""
Load AI employees from ``employees/*.yaml``.

One YAML file is one employee: who they are, what they advertise on their A2A
card, which tools they get, and their instructions. This module only reads and
validates the files - it builds nothing - so the Chief of Staff and run_all.py
can use it without importing ADK.

    name: Researcher                 # card name; the Chief of Staff delegates by it
    port: 10005                      # override with <ID>_PORT in .env (e.g. RESEARCHER_PORT)
    description: One line for the card.
    model: worker                    # worker | orchestrator | an explicit model name
    tools: [web_search, fetch_page]  # names from shared/tool_registry.py
    skill:
      id: web_research
      name: Web Research
      description: What the Chief of Staff should send here.
      tags: [web, research]
      examples: ["Who are our main competitors?"]
    instruction: |
      **Role:** ...

The employee's id is the file name (``researcher.yaml`` -> ``researcher``).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from shared.config import AGENT_HOST

EMPLOYEES_DIR = Path(__file__).parent.parent / "employees"

_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class EmployeeConfigError(ValueError):
    """An employee YAML file is missing something or has a wrong value."""


@dataclass(frozen=True)
class Skill:
    id: str
    name: str
    description: str
    tags: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Employee:
    id: str
    name: str
    port: int
    description: str
    instruction: str
    skill: Skill
    tools: list[str] = field(default_factory=list)
    model: str = "worker"
    source: Path | None = None

    @property
    def agent_name(self) -> str:
        """The ADK agent name: the display name as an identifier."""
        return re.sub(r"\W+", "_", self.name).strip("_")

    @property
    def url(self) -> str:
        return f"http://{AGENT_HOST}:{self.port}"


def _require(data: dict, key: str, kind: type, where: str):
    value = data.get(key)
    if not isinstance(value, kind) or (isinstance(value, str) and not value.strip()):
        raise EmployeeConfigError(f"{where}: '{key}' must be a non-empty {kind.__name__}.")
    return value


def _str_list(data: dict, key: str, where: str) -> list[str]:
    value = data.get(key) or []
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        raise EmployeeConfigError(f"{where}: '{key}' must be a list of strings.")
    return list(value)


def load_employee(path: str | Path) -> Employee:
    """Read and validate one employee file."""
    path = Path(path)
    where = path.name
    emp_id = path.stem
    if not _ID_RE.match(emp_id):
        raise EmployeeConfigError(
            f"{where}: file name must be lowercase letters, digits and underscores."
        )
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise EmployeeConfigError(f"{where}: not valid YAML: {e}") from e
    if not isinstance(data, dict):
        raise EmployeeConfigError(f"{where}: expected a mapping at the top level.")

    port = data.get("port")
    env_port = os.getenv(f"{emp_id.upper()}_PORT")
    if env_port:
        port = env_port
    try:
        port = int(port)
    except (TypeError, ValueError):
        raise EmployeeConfigError(f"{where}: 'port' must be a number.") from None

    skill_data = _require(data, "skill", dict, where)
    skill = Skill(
        id=_require(skill_data, "id", str, f"{where} skill"),
        name=_require(skill_data, "name", str, f"{where} skill"),
        description=_require(skill_data, "description", str, f"{where} skill").strip(),
        tags=_str_list(skill_data, "tags", f"{where} skill"),
        examples=_str_list(skill_data, "examples", f"{where} skill"),
    )

    return Employee(
        id=emp_id,
        name=_require(data, "name", str, where).strip(),
        port=port,
        description=_require(data, "description", str, where).strip(),
        instruction=_require(data, "instruction", str, where),
        skill=skill,
        tools=_str_list(data, "tools", where),
        model=str(data.get("model") or "worker").strip(),
        source=path,
    )


def load_all(directory: str | Path = EMPLOYEES_DIR) -> list[Employee]:
    """Every employee in `directory`, sorted by id. Names and ports must be unique."""
    employees = [load_employee(p) for p in sorted(Path(directory).glob("*.yaml"))]
    for attr in ("name", "port"):
        seen: dict = {}
        for emp in employees:
            value = getattr(emp, attr)
            if value in seen:
                raise EmployeeConfigError(
                    f"{emp.id} and {seen[value]} have the same {attr} ({value})."
                )
            seen[value] = emp.id
    return employees


def find(id_or_path: str, directory: str | Path = EMPLOYEES_DIR) -> Employee:
    """An employee by id (``researcher``) or by path to its YAML file."""
    path = Path(id_or_path)
    if path.suffix in (".yaml", ".yml") and path.exists():
        return load_employee(path)
    candidate = Path(directory) / f"{id_or_path}.yaml"
    if not candidate.exists():
        known = ", ".join(e.id for e in load_all(directory)) or "none"
        raise EmployeeConfigError(f"No employee '{id_or_path}'. Known: {known}.")
    return load_employee(candidate)
