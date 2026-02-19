"""
Phonon position expression parser.

Parses and evaluates string expressions like '2*qx^2 + qx*qy^2' or '2*(qx^2 + qy^2)'
into MatrixOperators for a single phonon mode. Expressions operate on spatial
coordinates within one mode (e.g. qx, qy for a 2D mode with labels ['x','y']).

Used by one_mode_phonon_sys and MultiModePhononSystem for position calculator
functionality.
"""

import re
from typing import List, Tuple, Any, Literal, Protocol, Optional


class _ModeProtocol(Protocol):
    """Protocol for a single phonon mode (position operators + identity)."""

    def get_position_operator(self, coord: str): ...

    def create_id_op(self): ...


def tokenize(expr: str) -> List[Tuple[str, Any]]:
    """
    Tokenize a phonon position expression string into (type, value) tokens.

    Coordinate format: q<label>^<exp> where label is an identifier (e.g. x, y, xy).
    Examples: qx, qy^2, qx*qy
    """
    tokens: List[Tuple[str, Any]] = []
    expr = expr.replace(" ", "").replace("\t", "")
    i = 0
    n = len(expr)
    while i < n:
        if expr[i].isdigit():
            start = i
            while i < n and expr[i].isdigit():
                i += 1
            tokens.append(("INT", int(expr[start:i])))
            continue
        if expr[i] == "+":
            tokens.append(("PLUS",))
            i += 1
            continue
        if expr[i] == "-":
            tokens.append(("MINUS",))
            i += 1
            continue
        if expr[i] == "*":
            tokens.append(("MUL",))
            i += 1
            continue
        if expr[i] == "(":
            tokens.append(("LPAREN",))
            i += 1
            continue
        if expr[i] == ")":
            tokens.append(("RPAREN",))
            i += 1
            continue
        # q<label> or q<label>^<exp> where label is [a-zA-Z][a-zA-Z0-9_]*
        if expr[i] == "q" and i + 1 < n and expr[i + 1].isalpha():
            i += 1
            start = i
            while i < n and (expr[i].isalnum() or expr[i] == "_"):
                i += 1
            label = expr[start:i]
            exp = 1
            if i < n and expr[i] == "^":
                i += 1
                start = i
                while i < n and expr[i].isdigit():
                    i += 1
                exp = int(expr[start:i])
            tokens.append(("Q", (label, exp)))
            continue
        raise ValueError(f"Unexpected character at position {i} in '{expr}': '{expr[i]}'")
    tokens.append(("EOF",))
    return tokens


class PositionExprParser:
    """Recursive descent parser for phonon position expressions within a single mode."""

    def __init__(
        self,
        tokens: List[Tuple[str, Any]],
        mode_node: _ModeProtocol,
        allowed_coords: Optional[List[str]] = None,
    ) -> None:
        self.tokens = tokens
        self.pos = 0
        self.mode_node = mode_node
        self.allowed_coords = set(allowed_coords) if allowed_coords else None

    def _peek(self) -> Tuple[str, Any]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else ("EOF",)

    def _consume(self) -> Tuple[str, Any]:
        if self.pos >= len(self.tokens):
            raise ValueError("Unexpected end of expression")
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def _validate_coord(self, label: str) -> None:
        if self.allowed_coords is not None and label not in self.allowed_coords:
            raise ValueError(
                f"Coordinate '{label}' not in mode. Allowed: {sorted(self.allowed_coords)}"
            )

    def _parse_expr(self):
        """Parse additive expression: term (+|-) term ..."""
        left = self._parse_term()
        while True:
            peek = self._peek()[0]
            if peek == "PLUS":
                self._consume()
                right = self._parse_term()
                left = left + right
            elif peek == "MINUS":
                self._consume()
                right = self._parse_term()
                left = left - right
            else:
                break
        return left

    def _parse_term(self):
        """Parse multiplicative expression: factor * factor ..."""
        left = self._parse_factor()
        while self._peek()[0] == "MUL":
            self._consume()
            right = self._parse_factor()
            left = left * right
        return left

    def _parse_factor(self):
        """Parse unary: INT | ( expr ) | position_product"""
        peek = self._peek()[0]
        if peek == "INT":
            _, val = self._consume()
            return val * self.mode_node.create_id_op()
        if peek == "LPAREN":
            self._consume()
            expr = self._parse_expr()
            if self._consume()[0] != "RPAREN":
                raise ValueError("Missing closing parenthesis")
            return expr
        if peek == "Q":
            return self._parse_position_product()
        if peek == "MINUS":
            self._consume()
            return -1 * self._parse_factor()
        raise ValueError(f"Unexpected token: {self._peek()}")

    def _parse_position_product(self):
        """Parse one or more adjacent Q terms: qx^a qy^b ... (coordinates of same mode)"""
        result = None
        while self._peek()[0] == "Q":
            _, (label, exp) = self._consume()
            self._validate_coord(label)
            if exp == 0:
                continue
            op = self.mode_node.get_position_operator(label)
            mode_op = op
            for _ in range(exp - 1):
                mode_op = mode_op * op
            if result is None:
                result = mode_op
            else:
                result = result * mode_op
        if result is None:
            return self.mode_node.create_id_op()
        return result

    def parse(self):
        result = self._parse_expr()
        if self._peek()[0] != "EOF":
            raise ValueError(f"Unexpected token after expression: {self._peek()}")
        return result


def evaluate_position_expression(
    mode_node: _ModeProtocol,
    expr: str,
    allowed_coords: Optional[List[str]] = None,
):
    """
    Parse and evaluate a phonon position expression for a single mode.

    Args:
        mode_node: Single phonon mode with get_position_operator(coord) and create_id_op().
        expr: Expression string (e.g. '2*qx^2 + qx*qy^2', '2*(qx^2 + qy^2)').
        allowed_coords: Optional list of valid coordinate labels for validation.

    Returns:
        MatrixOperator: The evaluated expression (in subsystem/single-mode space).
    """
    tokens = tokenize(expr)
    parser = PositionExprParser(tokens, mode_node, allowed_coords)
    return parser.parse()
