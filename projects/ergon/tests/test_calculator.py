"""Tests for the safe calculator (shared/calculator.py)."""

import pytest

from shared.calculator import calculate


@pytest.mark.parametrize("expression, result", [
    ("6000 + 400 * max(0, 12 - 8)", 7600),
    ("max(600, 180 * 2)", 600),
    ("(420 + 25) * 3", 1335),
    ("1200 * 0.15", 180),
    ("0.1 + 0.2", 0.3),
    ("ceil(13 / 8)", 2),
    ("floor(7.9)", 7),
    ("round(2.345, 2)", 2.35),
    ("-5 + abs(-3)", -2),
    ("2 ** 10", 1024),
    ("17 // 5 + 17 % 5", 5),
    ("sqrt(16)", 4),
])
def test_arithmetic(expression, result):
    out = calculate(expression)
    assert out["success"] is True
    assert out["result"] == result


@pytest.mark.parametrize("expression", [
    "__import__('os').system('echo hi')",
    "open('x.txt')",
    "x + 1",
    "(1).__class__",
    "[1, 2]",
    "'a' * 3",
    "True + 1",
    "max(1, key=abs)",
    "lambda: 1",
])
def test_anything_but_arithmetic_is_refused(expression):
    out = calculate(expression)
    assert out["success"] is False
    assert "error" in out


def test_huge_powers_are_refused():
    assert calculate("2 ** 1000")["success"] is False
    assert calculate("((10 ** 100) ** 100) ** 100")["success"] is False


def test_division_by_zero():
    assert calculate("1 / 0") == {
        "success": False, "expression": "1 / 0", "error": "Division by zero.",
    }


def test_empty_and_too_long():
    assert calculate("  ")["success"] is False
    assert calculate("1+" * 300 + "1")["success"] is False


def test_syntax_error_is_reported():
    out = calculate("2 +* 3")
    assert out["success"] is False
    assert out["error"].startswith("Cannot evaluate")
