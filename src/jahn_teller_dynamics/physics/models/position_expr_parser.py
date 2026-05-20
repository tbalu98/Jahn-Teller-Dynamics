"""
Phonon position expression parser.

Parses and evaluates string expressions like ``2*qx^2 + qx*qy^2``,
``2 + q1``, ``(q1 + q2)^2``, ``exp(q1)``, ``i*q1`` (multiply by the imaginary unit),
or ``q3+`` / ``q3-`` (shorthand for ``(q3x ± i*q3y)/sqrt(2)`` when both coordinates exist) into
:class:`MatrixOperator` instances for a single- or multi-mode phonon subsystem.

``exp(A)`` is evaluated either **exactly** (dense ``scipy.linalg.expm``, default),
or approximately as ``\\sum_{k=0}^{N} A^k/k!`` when ``exp_approximation_order`` is set
(an integer ``N``, e.g. from ``[PVC]`` / ``[LVC]`` .cfg).

The Taylor path keeps sparse/dense type through matrix-operator arithmetic instead
of allocating a dense ``dim×dim`` buffer for ``expm``.
"""

from __future__ import annotations

import math
import re
from typing import Any, List, Optional, Protocol, Tuple

# ``q{stem}±`` = (q{stem}x ± i*q{stem}y) / sqrt(2)
_INV_SQRT2 = 1.0 # 1.0 #/ math.sqrt(2.0)

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
# q{stem}+ / q{stem}- shorthand: (q{stem}x ± i*q{stem}y) / sqrt(2)
# --------------------------------------------------------------------------- #


def q_cartesian_labels(stem: str) -> Tuple[str, str]:
    """
    Coordinate labels for ``q{stem}+`` / ``q{stem}-`` shorthand: ``(q{stem}x, q{stem}y)``.

    Example: stem ``"3"`` → ``("q3x", "q3y")``; stem ``"j"`` → ``("qjx", "qjy")``.
    """
    s = str(stem).strip()
    if not s:
        raise ValueError("q± shorthand requires a non-empty mode stem before '+' or '-'")
    return f"q{s}x", f"q{s}y"


def q_plus_cartesian_labels(stem: str) -> Tuple[str, str]:
    """Alias for :func:`q_cartesian_labels` (``q{stem}+`` / ``q{stem}-`` share the same pair)."""
    return q_cartesian_labels(stem)


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #

_Token = Tuple[str, Any]


def tokenize(expr: str) -> List[_Token]:
    """
    Tokenize a phonon position expression.

    - Coordinates: ``q`` + label (e.g. ``q1``, ``q3x``, ``qx``).
    - ``q{stem}+`` / ``q{stem}-`` (suffix ``+`` or ``-`` with no identifier after it):
      shorthand for ``(q{stem}x ± i*q{stem}y) / sqrt(2)`` (both coordinates required).
    - Imaginary unit: standalone ``i`` (not part of a longer name), e.g. ``i*q1`` or
      ``iq1`` (implicit multiply).
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
        if c == "i":
            nxt = s[i + 1] if i + 1 < n else ""
            if nxt == "" or nxt in "+-*/^)":
                tokens.append(("IMAG",))
                i += 1
                continue
            if nxt == "q":
                tokens.append(("IMAG",))
                i += 1
                continue
        # Q coordinate: q + identifier body; optional q{stem}+ shorthand
        if c == "q" and i + 1 < n and (s[i + 1].isalnum() or s[i + 1] == "_"):
            start = i
            i += 1
            body_start = i
            while i < n:
                if s[i] == "q" and i > start + 1:
                    break
                if s[i].isalnum() or s[i] == "_":
                    i += 1
                else:
                    break
            body = s[body_start:i]
            if body and i < n and s[i] in "+-":
                if i + 1 < n and (s[i + 1].isalnum() or s[i + 1] == "_"):
                    pass
                elif s[i] == "+":
                    i += 1
                    tokens.append(("Q_PLUS", body))
                    continue
                else:
                    i += 1
                    tokens.append(("Q_MINUS", body))
                    continue
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
        if self._peek()[0] == "IMAG":
            self._consume()
            if self._peek()[0] == "MUL":
                self._consume()
            return (1j) * self._parse_unary()
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
        if t == "Q_PLUS":
            _, stem = self._consume()
            return self._parse_q_cartesian(stem, subtract_imag=False)
        if t == "Q_MINUS":
            _, stem = self._consume()
            return self._parse_q_cartesian(stem, subtract_imag=True)
        if t == "LPAREN":
            self._consume()
            inner = self._parse_add()
            if self._consume()[0] != "RPAREN":
                raise ValueError("Missing closing parenthesis")
            return inner
        raise ValueError(f"Unexpected token: {self._peek()}")

    def _parse_q_cartesian(self, stem: str, *, subtract_imag: bool) -> mm.MatrixOperator:
        """
        ``q{stem}+`` → ``(q{stem}x + i*q{stem}y) / sqrt(2)``;
        ``q{stem}-`` → ``(q{stem}x - i*q{stem}y) / sqrt(2)``.
        """
        lx, ly = q_cartesian_labels(stem)
        self._validate_coord(lx)
        self._validate_coord(ly)
        cx = self._resolve_coord(lx) if self.allowed_coords else lx
        cy = self._resolve_coord(ly) if self.allowed_coords else ly
        op_x = self.mode_node.get_position_operator(cx)
        op_y = self.mode_node.get_position_operator(cy)
        if subtract_imag:
            combo = op_x - (1j * op_y)
        else:
            combo = op_x + (1j * op_y)
        return _INV_SQRT2 * combo

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


class _PolinomHermitianConjugator:
    """
    Emit the Hermitian-conjugate phonon polynomial as a string (position operators
    are Hermitian; ``i`` changes sign; products reverse order; ``q{stem}+`` ↔ ``q{stem}-``).
    """

    def __init__(self, tokens: List[_Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    def _peek(self) -> _Token:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else ("EOF",)

    def _consume(self) -> _Token:
        if self.pos >= len(self.tokens):
            raise ValueError("Unexpected end of expression")
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def emit(self) -> str:
        result = self._emit_add()
        if self._peek()[0] != "EOF":
            raise ValueError(f"Unexpected token after expression: {self._peek()}")
        return result

    def _emit_add(self) -> str:
        out = self._emit_term()
        while self._peek()[0] in ("PLUS", "MINUS"):
            op = "+" if self._consume()[0] == "PLUS" else "-"
            nxt = self._emit_term()
            if op == "+" and nxt.startswith("-"):
                out = f"{out}{nxt}"
            elif op == "-" and nxt.startswith("-"):
                out = f"{out}+{nxt[1:]}"
            else:
                out = f"{out}{op}{nxt}"
        return out

    def _emit_term(self) -> str:
        factors = [self._emit_unary()]
        while self._peek()[0] == "MUL":
            self._consume()
            factors.append(self._emit_unary())
        if len(factors) == 1:
            return factors[0]
        return "*".join(reversed(factors))

    def _emit_unary(self) -> str:
        if self._peek()[0] == "MINUS":
            self._consume()
            return f"-({self._emit_unary()})"
        if self._peek()[0] == "IMAG":
            self._consume()
            if self._peek()[0] == "MUL":
                self._consume()
            if self._peek()[0] == "EOF":
                return "-i"
            inner = self._emit_unary()
            if inner.startswith("(") and inner.endswith(")"):
                return f"-i*{inner}"
            return f"-i*({inner})"
        return self._emit_postfix()

    def _emit_postfix(self) -> str:
        base = self._emit_molecule()
        while self._peek()[0] in ("CARET", "POW2"):
            self._consume()
            _, val = self._consume()
            n = int(round(val))
            base = f"({base})^{n}" if "+" in base or "-" in base or "*" in base else f"{base}^{n}"
        return base

    def _emit_molecule(self) -> str:
        t = self._peek()[0]
        if t == "EXP":
            self._consume()
            if self._consume()[0] != "LPAREN":
                raise ValueError("exp must be followed by '('")
            inner = self._emit_add()
            if self._consume()[0] != "RPAREN":
                raise ValueError("Missing ')' after exp(...)")
            return f"exp({inner})"
        if t == "NUMBER":
            _, val = self._consume()
            if val == int(val):
                return str(int(val))
            return str(val)
        if t == "Q":
            return self._emit_q_product()
        if t == "Q_PLUS":
            _, stem = self._consume()
            return f"q{stem}-"
        if t == "Q_MINUS":
            _, stem = self._consume()
            return f"q{stem}+"
        if t == "LPAREN":
            self._consume()
            inner = self._emit_add()
            if self._consume()[0] != "RPAREN":
                raise ValueError("Missing closing parenthesis")
            return f"({inner})"
        raise ValueError(f"Unexpected token: {self._peek()}")

    def _emit_q_product(self) -> str:
        def q_atom() -> str:
            _, label = self._consume()
            if self._peek()[0] in ("CARET", "POW2"):
                self._consume()
                _, val = self._consume()
                n = int(round(val))
                return f"{label}^{n}"
            return label

        atoms = [q_atom()]
        while self._peek()[0] == "Q":
            atoms.append(q_atom())
        if len(atoms) == 1:
            return atoms[0]
        return "*".join(reversed(atoms))


def hermitian_conjugate_polinom(expr: str) -> str:
    """
    Hermitian-conjugate a phonon position-operator polynomial string.

    Assumes each coordinate ``qk`` (and ``qkx``, ``qky``, …) is Hermitian. Then:

    - ``q{stem}+`` ↔ ``q{stem}-`` (same ``1/sqrt(2)`` normalization as in evaluation).
    - ``i*A`` → ``-i*A`` (with product order reversed when ``A`` is a product).
    - ``A*B`` → ``B*A`` for operator products.

    Whitespace in ``expr`` is stripped before parsing.
    """
    tokens = tokenize(expr)
    return _PolinomHermitianConjugator(tokens).emit()


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
    - Imaginary unit: ``i`` multiplies by ``1j`` (e.g. ``i*q1``, ``2*i*q1``).
    - ``q{stem}+`` / ``q{stem}-``: shorthand for ``(q{stem}x ± i*q{stem}y) / sqrt(2)``
      (both Cartesian coordinates required).

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
