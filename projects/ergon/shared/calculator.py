"""
A safe calculator tool, so employees never do arithmetic in their heads.

Language models are unreliable at multi-step arithmetic (quotes, totals,
percentages). ``calculate`` evaluates one expression with Python's own number
rules, but only numbers, arithmetic operators and a few named functions: no
names, attributes, calls to anything else, or imports.
"""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable

Number = int | float


def _min(*values: Number) -> Number:
    return min(values)


def _max(*values: Number) -> Number:
    return max(values)


def _abs(value: Number) -> Number:
    return abs(value)


def _round(value: Number, ndigits: Number | None = None) -> Number:
    if ndigits is None:
        return round(value)
    return round(value, int(ndigits))


def _ceil(value: Number) -> Number:
    return math.ceil(value)


def _floor(value: Number) -> Number:
    return math.floor(value)


def _sqrt(value: Number) -> Number:
    return math.sqrt(value)

_BINARY: dict[type[ast.operator], Callable[[Number, Number], Number]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY: dict[type[ast.unaryop], Callable[[Number], Number]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_FUNCTIONS: dict[str, Callable[..., Number]] = {
    "min": _min,
    "max": _max,
    "abs": _abs,
    "round": _round,
    "ceil": _ceil,
    "floor": _floor,
    "sqrt": _sqrt,
}
_MAX_EXPONENT = 100
_MAX_LENGTH = 500


def _eval(node: ast.AST) -> Number:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow):
            if abs(right) > _MAX_EXPONENT:
                raise ValueError(f"exponent larger than {_MAX_EXPONENT}")
            # stop ((10**100)**100)**100 before Python builds a huge integer
            if isinstance(left, int) and left.bit_length() * abs(right) > 100_000:
                raise ValueError("result too large")
        return _BINARY[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id in _FUNCTIONS and not node.keywords:
        return _FUNCTIONS[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError(f"not allowed: {ast.dump(node)[:60]}")


def calculate(expression: str) -> dict:
    """
    Evaluate an arithmetic expression exactly. Use it for every price, total,
    discount or percentage instead of doing the maths yourself.

    Allowed: numbers, + - * / // % **, parentheses, and min, max, abs, round,
    ceil, floor, sqrt. Write 1,200 as 1200 and 15% as 0.15.

    Args:
        expression: e.g. "6000 + 400 * max(0, 12 - 8)" or "max(600, 180 * 12)".

    Returns:
        dict: success, expression, result; on failure success=False and an error.
    """
    expression = (expression or "").strip()
    if not expression:
        return {"success": False, "error": "Empty expression."}
    if len(expression) > _MAX_LENGTH:
        return {"success": False, "error": f"Expression longer than {_MAX_LENGTH} characters."}
    try:
        result = _eval(ast.parse(expression, mode="eval"))
    except ZeroDivisionError:
        return {"success": False, "expression": expression, "error": "Division by zero."}
    except (SyntaxError, ValueError, TypeError, OverflowError) as e:
        return {"success": False, "expression": expression, "error": f"Cannot evaluate: {e}"}
    if isinstance(result, float):
        result = round(result, 10)  # 0.1 + 0.2 -> 0.3, not 0.30000000000000004
        if result.is_integer() and abs(result) < 1e15:
            result = int(result)
    return {"success": True, "expression": expression, "result": result}
