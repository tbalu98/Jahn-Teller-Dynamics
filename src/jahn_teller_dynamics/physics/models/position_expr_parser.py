"""
Phonon position expression parser.

Parses and evaluates string expressions like ``2*qx^2 + qx*qy^2``,
``2 + q1``, ``(q1 + q2)^2``, or ``exp(q1)`` into :class:`MatrixOperator` instances
for a single- or multi-mode phonon subsystem.

``exp(A)`` is evaluated either **exactly** (dense ``scipy.linalg.expm``, default),
or approximately as ``\\sum_{k=0}^{N} A^k/k!`` when ``exp_approximation_order`` is set
(an integer ``N``, e.g. from ``[PVC]`` / ``[LVC]`` .cfg).

The Taylor path keeps sparse/dense type through matrix-operator arithmetic instead
of allocating a dense ``dim×dim`` buffer for ``expm``.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional, Protocol, Tuple

import numpy as np
from scipy.linalg import expm as scipy_expm

import jahn_teller_dynamics.math.maths as maths
import jahn_teller_dynamics.math.matrix_mechanics as mm


class _ModeProtocol(Protocol):
    def get_position_operator(self, coord: str): ...
    def create_id_op(self): ...


# --------------------------------------------------------------------------- #
# Matrix utilities (numeric power; exp = exact scipy or truncated Taylor series)
# --------------------------------------------------------------------------- #


def _dim_of(op: mm.MatrixOperator) -> int:
    return len(op.matrix)


def _id_matrix_like(op: mm.MatrixOperator) -> mm.MatrixOperator:
    dim = _dim_of(op)
    mtype = type(op.matrix)
    if mtype is maths.SparseMatrix:
        return mm.MatrixOperator.create_id_matrix_op(dim, matrix_type=maths.SparseMatrix)
    return mm.MatrixOperator.create_id_matrix_op(dim, matrix_type=maths.Matrix)


def matrix_int_power(op: mm.MatrixOperator, n: int) -> mm.MatrixOperator:
    """Return ``op**n`` for non-negative integer ``n`` (``n==0`` → identity)."""
    if n < 0:
        raise ValueError("Exponent in position expression must be a non-negative integer")
    if n == 0:
        return _id_matrix_like(op)
    acc = op
    for _ in range(n - 1):
        acc = acc * op
    return acc


def matrix_operator_expm(
    op: mm.MatrixOperator,
    *,
    exp_approximation_order: Optional[int] = None,
) -> mm.MatrixOperator:
    """
    Matrix exponential ``exp(A)``.

    * If ``exp_approximation_order`` is ``None``: exact evaluation via dense
      :func:`scipy.linalg.expm` on ``A``.
    * If set to non-negative integer ``N``: truncated Taylor expansion
      ``sum_{k=0}^N A^k / k!`` using recurrence ``P_k = P_{k-1}(A/k)``. Preserves sparse
      structure when ``A`` is sparse-backed.

    Args:
        op: Argument matrix ``A``.
        exp_approximation_order: Highest power ``N`` in Taylor sum inclusive; ``0`` yields ``I``.
    """
    if exp_approximation_order is None:
        m = op.matrix
        if isinstance(m, maths.SparseMatrix):
            arr = np.asarray(m.to_dense_matrix().matrix, dtype=np.complex128)
        else:
            arr = np.asarray(m.matrix, dtype=np.complex128)
        e = scipy_expm(arr)
        out = mm.MatrixOperator(maths.Matrix(np.asmatrix(e)))
        if op.subsys_name:
            out.subsys_name = op.subsys_name
        return out

    nmax = int(exp_approximation_order)
    if nmax < 0:
        raise ValueError("exp_approximation_order must be non-negative")

    acc = _id_matrix_like(op)
    term = acc
    for k in range(1, nmax + 1):
        term = term * (op / float(k))
        acc = acc + term

    if op.subsys_name:
        acc.subsys_name = op.subsys_name
    return acc


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #

_Token = Tuple[str, Any]


def tokenize(expr: str) -> List[_Token]:
    """
    Tokenize a phonon position expression.

    - Coordinates: ``q`` + label (e.g. ``q1``, ``q3x``, ``qx``).
    - Numbers: integer or decimal (optional scientific notation); ``-`` is a separate
      token (unary minus is applied in the parser).
    - ``**`` and ``^`` are exponentiation; ``^`` after ``q`` is still split so
      ``(a+b)^2`` works.
    - ``exp( ... )``: matrix exponential; exact ``expm`` unless a phonon subsystem sets
      ``exp_approximation_order = N`` or the parser is given a truncated order.
    """
    tokens: List[_Token] = []
    s = expr.replace(" ", "").replace("\t", "")
    n = len(s)
    i = 0
    # float: digits . digits [eE...]  or  . digits  or  digits eE
    num_re = re.compile(
        r"(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?"
    )

    def read_number() -> float:
        nonlocal i
        m = num_re.match(s, i)
        if not m:
            raise ValueError(f"Invalid number at position {i} in {expr!r}")
        val = float(m.group(0))
        i = m.end()
        return val

    while i < n:
        c = s[i]
        if c in "+-*/^()":
            if s.startswith("**", i):
                tokens.append(("POW2",))
                i += 2
            elif c == "*":
                tokens.append(("MUL",))
                i += 1
            elif c == "^":
                tokens.append(("CARET",))
                i += 1
            elif c == "+":
                tokens.append(("PLUS",))
                i += 1
            elif c == "-":
                tokens.append(("MINUS",))
                i += 1
            elif c == "(":
                tokens.append(("LPAREN",))
                i += 1
            elif c == ")":
                tokens.append(("RPAREN",))
                i += 1
            continue
        if c in "0123456789" or (c == "." and i + 1 < n and s[i + 1].isdigit()):
            val = read_number()
            tokens.append(("NUMBER", val))
            continue
        if s.startswith("exp(", i):
            tokens.append(("EXP",))
            i += 3
            continue
        # Q coordinate: q + identifier body
        if c == "q" and i + 1 < n and (s[i + 1].isalnum() or s[i + 1] == "_"):
            start = i
            i += 1
            while i < n:
                if s[i] == "q" and i > start + 1:
                    break
                if s[i].isalnum() or s[i] == "_":
                    i += 1
                else:
                    break
            label = s[start:i]
            tokens.append(("Q", label))
            continue
        raise ValueError(f"Unexpected character at position {i} in {expr!r}: {c!r}")

    tokens.append(("EOF",))
    return tokens


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


class PositionExprParser:
    """Recursive descent parser for phonon position / matrix expressions."""

    def __init__(
        self,
        tokens: List[_Token],
        mode_node: _ModeProtocol,
        allowed_coords: Optional[List[str]] = None,
        *,
        exp_approximation_order: Optional[int] = None,
    ) -> None:
        self.tokens = tokens
        self.pos = 0
        self.mode_node = mode_node
        self.allowed_coords = set(allowed_coords) if allowed_coords else None
        self._exp_approximation_order = exp_approximation_order
        # Cache one identity in case NUMBER tokens need a scalar; dimension from first use.
        self._id_op_cache: Optional[mm.MatrixOperator] = None

    def _id_op(self) -> mm.MatrixOperator:
        if self._id_op_cache is None:
            self._id_op_cache = self.mode_node.create_id_op()
        return self._id_op_cache

    def _peek(self) -> _Token:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else ("EOF",)

    def _consume(self) -> _Token:
        if self.pos >= len(self.tokens):
            raise ValueError("Unexpected end of expression")
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def _resolve_coord(self, label: str) -> str:
        if self.allowed_coords is None:
            return label
        if label in self.allowed_coords:
            return label
        if label.startswith("q") and len(label) > 1 and label[1:] in self.allowed_coords:
            return label[1:]
        return label

    def _validate_coord(self, label: str) -> None:
        if self.allowed_coords is not None:
            resolved = self._resolve_coord(label)
            if resolved not in self.allowed_coords:
                raise ValueError(
                    f"Coordinate '{label}' not in mode. Allowed: {sorted(self.allowed_coords)}"
                )

    def _parse_exponent_value(self) -> int:
        """Non-negative integer exponent (NUMBER token with integral value)."""
        if self._peek()[0] != "NUMBER":
            raise ValueError("Expected a non-negative number as exponent")
        _, val = self._consume()
        if val < 0 or abs(val - round(val)) > 1e-12 * (1.0 + abs(val)):
            raise ValueError("Exponent must be a non-negative integer")
        return int(round(val))

    def parse(self) -> mm.MatrixOperator:
        result = self._parse_add()
        if self._peek()[0] != "EOF":
            raise ValueError(f"Unexpected token after expression: {self._peek()}")
        return result

    def _parse_add(self) -> mm.MatrixOperator:
        left = self._parse_term()
        while True:
            t = self._peek()[0]
            if t == "PLUS":
                self._consume()
                right = self._parse_term()
                left = left + right
            elif t == "MINUS":
                self._consume()
                right = self._parse_term()
                left = left - right
            else:
                break
        return left

    def _parse_term(self) -> mm.MatrixOperator:
        left = self._parse_unary()
        while self._peek()[0] == "MUL":
            self._consume()
            right = self._parse_unary()
            left = left * right
        return left

    def _parse_unary(self) -> mm.MatrixOperator:
        if self._peek()[0] == "MINUS":
            self._consume()
            u = self._parse_unary()
            return (-1) * u
        return self._parse_postfix()

    def _parse_postfix(self) -> mm.MatrixOperator:
        base = self._parse_molecule()
        while self._peek()[0] in ("CARET", "POW2"):
            self._consume()
            n = self._parse_exponent_value()
            base = matrix_int_power(base, n)
        return base

    def _parse_molecule(self) -> mm.MatrixOperator:
        t = self._peek()[0]
        if t == "EXP":
            self._consume()
            if self._consume()[0] != "LPAREN":
                raise ValueError("exp must be followed by '('")
            inner = self._parse_add()
            if self._consume()[0] != "RPAREN":
                raise ValueError("Missing ')' after exp(...)")
            return matrix_operator_expm(
                inner, exp_approximation_order=self._exp_approximation_order
            )
        if t == "NUMBER":
            _, val = self._consume()
            return val * self._id_op()
        if t == "Q":
            return self._parse_q_product()
        if t == "LPAREN":
            self._consume()
            inner = self._parse_add()
            if self._consume()[0] != "RPAREN":
                raise ValueError("Missing closing parenthesis")
            return inner
        raise ValueError(f"Unexpected token: {self._peek()}")

    def _parse_q_product(self) -> mm.MatrixOperator:
        """One or more coordinate atoms, optionally implicit multiply (e.g. ``q1q2``)."""

        def q_atom() -> mm.MatrixOperator:
            _, label = self._consume()
            self._validate_coord(label)
            coord = self._resolve_coord(label) if self.allowed_coords else label
            op = self.mode_node.get_position_operator(coord)
            if self._peek()[0] in ("CARET", "POW2"):
                self._consume()
                n = self._parse_exponent_value()
                if n < 0:
                    raise ValueError("Exponent in q^n must be a non-negative integer")
                if n == 0:
                    return self._id_op()
                return matrix_int_power(op, n)
            return op

        acc = q_atom()
        while self._peek()[0] == "Q":
            acc = acc * q_atom()
        return acc


def evaluate_position_expression(
    mode_node: _ModeProtocol,
    expr: str,
    allowed_coords: Optional[List[str]] = None,
    *,
    exp_approximation_order: Optional[int] = None,
) -> mm.MatrixOperator:
    """
    Parse and evaluate a phonon / matrix operator expression in one subsystem.

    Features:

    - Scalars: ``c`` and ``c * ...`` (scalar times identity on the full space).
    - Addition and subtraction: ``A + B``, ``1.5 + q1`` (interpreted as ``1.5*I + q1``).
    - Multiplication: ``*`` and implicit product between coordinates (e.g. ``q1q2``).
    - Powers: ``^n`` and ``**n`` (non-negative integer) on a sub-expression, e.g.
      ``(q1 + q2)^2``, ``q1^2``, ``q1**2``.
    - Matrix exponential: ``exp(A)`` — exact ``scipy.linalg.expm`` by default; if
      ``exp_approximation_order`` is an integer ``N``, use ``sum_{k=0}^N A^k/k!`` instead.

    Args:
        mode_node: Object with ``get_position_operator(coord)`` and ``create_id_op()``.
        expr: Expression string.
        allowed_coords: Optional list of valid coordinate names for validation.
        exp_approximation_order: If ``None``, use exact ``expm`` and also check
            ``getattr(mode_node, "exp_approximation_order", None)`` for a phonon-subsystem value.

    Returns:
        The resulting :class:`~jahn_teller_dynamics.math.matrix_mechanics.MatrixOperator`.
    """
    order_eff = exp_approximation_order
    if order_eff is None:
        order_eff = getattr(mode_node, "exp_approximation_order", None)

    tokens = tokenize(expr)
    parser = PositionExprParser(
        tokens,
        mode_node,
        allowed_coords,
        exp_approximation_order=order_eff,
    )
    return parser.parse()
