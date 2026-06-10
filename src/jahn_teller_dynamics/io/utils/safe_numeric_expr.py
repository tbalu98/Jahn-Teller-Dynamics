"""Safe evaluation of small integer arithmetic in config files (no ``eval()``)."""

from __future__ import annotations

import ast
from typing import Union


def parse_config_int_expression(s: Union[str, int]) -> int:
    """
    Parse a non-negative integer from a literal or arithmetic expression string.

    Allowed: ``+, -, *, `` (integer division not included), parentheses, unary ``-``.
    Expressions are evaluated as Python integer arithmetic with :mod:`ast` (no builtins).
    """
    if isinstance(s, bool):
        raise TypeError("Boolean is not accepted as integer config expression")
    if isinstance(s, int):
        return _check_nonneg_int(int(s))

    stripped = str(s).strip().lower()
    if stripped in ("", "none", "auto"):
        raise ValueError(f"Cannot parse integer from empty or placeholder: {s!r}")
    node = ast.parse(stripped, mode="eval")
    val = _eval_int_ast(node.body)
    return _check_nonneg_int(val)


def _check_nonneg_int(n: int) -> int:
    if n < 0:
        raise ValueError(f"Phonon cutoffs must be non-negative; got {n}")
    return n


def _eval_int_ast(n: ast.AST) -> int:
    if isinstance(n, ast.Expression):
        return _eval_int_ast(n.body)
    if isinstance(n, getattr(ast, "Constant", ())):
        v = getattr(n, "value", None)
        if isinstance(v, int):
            return v
        raise ValueError(f"Expected integer literal, got {v!r}")
    if getattr(ast, "Num", None) is not None and isinstance(n, ast.Num):
        if isinstance(n.n, int):
            return n.n
        raise ValueError(f"Expected integer literal, got {n.n!r}")
    if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult)):
        left = _eval_int_ast(n.left)
        right = _eval_int_ast(n.right)
        if isinstance(n.op, ast.Add):
            return left + right
        if isinstance(n.op, ast.Sub):
            return left - right
        return left * right
    if isinstance(n, ast.UnaryOp):
        operand = _eval_int_ast(n.operand)
        if isinstance(n.op, ast.UAdd):
            return +operand
        if isinstance(n.op, ast.USub):
            return -operand
    if isinstance(n, ast.Tuple):
        raise ValueError("Tuple not allowed in integer expression")
    if isinstance(n, ast.Call):
        raise ValueError("Function calls not allowed in integer expression")
    raise ValueError(f"Unsupported expression AST: {type(n).__name__}")

