"""
Multi-mode quantum harmonic oscillator with constrained total phonon number.

This module provides a phonon system where the Hilbert space is restricted to
states with sum(n_i) <= order across all modes, yielding a smaller basis than
the product space (order+1)^N. Each mode has a distinct energy quantum (hbar*omega)
and position operator prefactor.

This is a **single node** with many operators. The Hilbert space cannot be
expressed as a tensor product of single-mode spaces, so there are no child nodes.
"""

from __future__ import annotations

from typing import List, Optional, Union, Literal, Dict, TYPE_CHECKING

import jahn_teller_dynamics.math.braket_formalism as bf
import jahn_teller_dynamics.math.matrix_mechanics as mm
import jahn_teller_dynamics.math.maths as maths
import jahn_teller_dynamics.physics.quantum_system as qs
import numpy as np

if TYPE_CHECKING:
    from jahn_teller_dynamics.math.matrix_mechanics import MatrixOperator
    from jahn_teller_dynamics.math.braket_formalism import ket_state


class _ModeView:
    """
    Lightweight view for a single mode's operators (for interface compatibility).
    Not a tree node—just a facade providing .operators and .qm_nums_names so
    code expecting a mode node (e.g. position expr parser) can use it.
    """

    def __init__(
        self,
        mode_id: str,
        operators: Dict[str, "MatrixOperator"],
        qm_nums_names: List[str],
    ) -> None:
        self.id = mode_id
        self.operators = operators
        self.qm_nums_names = qm_nums_names

    def get_position_operator(self, coord: str) -> "MatrixOperator":
        op = self.operators.get(coord) or self.operators.get("X")
        if op is None:
            raise ValueError(
                f"Coordinate '{coord}' not found. Available: {list(self.qm_nums_names)}"
            )
        return op


class MultiModeConstrainedPhononSystem(qs.quantum_system_node):
    """
    Single-node multi-mode quantum harmonic oscillator with total-phonon-number constraint.

    Hilbert space: |n1, n2, ..., nN⟩ with n1 + n2 + ... + nN <= order.
    Each mode i has energy hbar*omega_i and position prefactor ∝ 1/sqrt(omega_i)
    when dimensionless_coordinates=False.

    This is a **single quantum_system_node** with many operators (K, K_1..K_N,
    q1..qN, X, Y). The space cannot be factored as a tensor product of mode
    subspaces, so there are no child nodes.

    Compatible with LVC_model and create_lvc_hamiltonian via is_constrained_multimode.
    """

    is_constrained_multimode: bool = True

    def __init__(
        self,
        modes: List[float],
        order: int,
        use_sparse: bool = True,
        phonon_system_id: str = "phonon_system",
        dimensionless_coordinates: bool = True,
        null_point_vib: bool = True,
    ) -> None:
        """
        Args:
            modes: List of mode energies (hbar*omega) for each mode.
            order: Maximum total phonon number (sum of quantum numbers <= order).
            use_sparse: If True, use sparse matrices.
            phonon_system_id: ID for this node.
            dimensionless_coordinates: If True, q = (a+a†)/√2; else q = (a+a†)/(√2 ω^0.5).
            null_point_vib: If True, include 0.5*hbar*omega zero-point energy in H.
        """
        self.order = order
        self.modes = list(modes)
        self.use_sparse = use_sparse
        self.dimensionless_coordinates = dimensionless_coordinates
        self.null_point_vib = null_point_vib
        self.id = phonon_system_id
        self.children: List[qs.quantum_system_node] = []  # Single node: no children

        num_modes = len(self.modes)
        qm_nums_names = [f"q{i}" for i in range(1, num_modes + 1)]

        # Build Hilbert space: states |n1,...,nN⟩ with sum <= order
        self.calculation_bases = mm.hilber_space_bases().harm_osc_sys(
            num_modes, order, qm_nums_names
        )
        self.base_states = self.calculation_bases
        self.dim = self.calculation_bases.dim

        self.names_dict = {
            name: idx for idx, name in enumerate(qm_nums_names)
        }
        self.qm_nums_names = qm_nums_names

        self.mx_op_builder = mm.braket_to_matrix_formalism(
            self.calculation_bases, use_sparse=use_sparse
        )

        self._build_creator_annihilator_ops()

        self.operators: Dict[str, mm.MatrixOperator] = {}
        self._mode_views: Dict[int, _ModeView] = {}  # mode index (1-based) -> view
        self._build_hamiltonian_and_position_ops()

        super().__init__(
            phonon_system_id,
            base_states=self.base_states,
            operators=self.operators,
            children=[],  # Single node, no children
        )

    def create_hilbert_space(self) -> None:
        """Override: keep our constrained basis (no children to kron)."""
        pass

    def _build_creator_annihilator_ops(self) -> None:
        """Build creator and annihilator matrix operators for each mode."""
        self.create_mx_ops: Dict[str, mm.MatrixOperator] = {}
        self.annil_mx_ops: Dict[str, mm.MatrixOperator] = {}
        for name in self.qm_nums_names:
            idx = self.names_dict[name]
            creator_op = bf.creator_operator(idx, name)
            annil_op = bf.annihilator_operator(idx, name)
            self.create_mx_ops[name] = self.mx_op_builder.create_MatrixOperator(
                creator_op, subsys_name=self.id
            )
            self.annil_mx_ops[name] = self.mx_op_builder.create_MatrixOperator(
                annil_op, subsys_name=self.id
            )

    def _build_hamiltonian_and_position_ops(self) -> None:
        """Build K, per-mode K_i, and position operators. All on this single node."""
        H_parts: List[mm.MatrixOperator] = []
        for i, (omega, name) in enumerate(zip(self.modes, self.qm_nums_names)):
            n_op = self.create_mx_ops[name] * self.annil_mx_ops[name]
            H_i = omega * n_op.round(0).change_type(np.int16)
            if self.null_point_vib:
                id_op = self._create_id_op()
                H_i = H_i + 0.5 * omega * id_op
            H_parts.append(H_i)

        self.operators["K"] = sum(H_parts)

        # Position operators: q_i = (a_i + a†_i) / denom_i (per-mode prefactor)
        for i, (omega, name) in enumerate(zip(self.modes, self.qm_nums_names)):
            if self.dimensionless_coordinates:
                denom = mm.SQRT_2
            else:
                denom = mm.SQRT_2 * (omega ** 0.5)
            pos_op = (
                self.annil_mx_ops[name] + self.create_mx_ops[name]
            ) / denom
            pos_op.subsys_name = self.id
            self.operators[name] = pos_op

        # Legacy aliases
        if len(self.qm_nums_names) >= 1:
            self.operators["X"] = self.operators[self.qm_nums_names[0]]
        if len(self.qm_nums_names) >= 2:
            self.operators["Y"] = self.operators[self.qm_nums_names[1]]

        # Per-mode operator views (for get_mode_node interface, not tree children)
        for i in range(1, len(self.modes) + 1):
            omega = self.modes[i - 1]
            coord = self.qm_nums_names[i - 1]

            n_op = (
                self.create_mx_ops[coord] * self.annil_mx_ops[coord]
            ).round(0).change_type(np.int16)
            K_i = omega * n_op
            if self.null_point_vib:
                K_i = K_i + 0.5 * omega * self._create_id_op()

            mode_ops = {
                "K": K_i,
                coord: self.operators[coord],
                "X": self.operators[coord],
            }
            self._mode_views[i] = _ModeView(
                mode_id=f"mode_{i}",
                operators=mode_ops,
                qm_nums_names=[coord],
            )

    def _create_id_op(self) -> mm.MatrixOperator:
        """Create identity operator."""
        matrix_type = maths.SparseMatrix if self.use_sparse else maths.Matrix
        return mm.MatrixOperator.create_id_matrix_op(
            dim=self.dim, matrix_type=matrix_type
        )

    def _resolve_mode_index(self, mode: Union[int, str]) -> int:
        """Resolve mode to 1-based index."""
        if isinstance(mode, int):
            if mode < 1:
                raise ValueError("mode index must be 1-based (mode_1, mode_2, ...)")
            return mode
        if isinstance(mode, str) and mode.startswith("mode_"):
            return int(mode.split("_")[1])
        raise ValueError(f"Invalid mode: {mode}")

    def get_mode_node(self, mode: Union[int, str]) -> _ModeView:
        """
        Return a view for the given mode (for interface compatibility).
        This is NOT a tree node—the system is a single node with many operators.
        """
        idx = self._resolve_mode_index(mode)
        if idx not in self._mode_views:
            raise ValueError(
                f"Mode {idx} not found in phonon system '{self.id}' "
                f"(modes 1..{len(self.modes)})"
            )
        return self._mode_views[idx]

    def get_operator_subsystem(self, operator_id: str, mode: Union[int, str] = 1):
        """Get operator for a mode (subsystem view = full view for single node)."""
        view = self.get_mode_node(mode)
        op = view.operators.get(operator_id)
        if op is None and operator_id in view.qm_nums_names:
            op = view.operators.get(operator_id)
        if op is None and operator_id == "X":
            op = view.operators.get("X")
        if op is None:
            raise ValueError(
                f"Operator '{operator_id}' not found for mode {mode}. "
                f"Available: {list(view.operators.keys())}"
            )
        return op

    def get_operator_full(self, operator_id: str, mode: Union[int, str] = 1):
        """Get operator in full phonon space (same as subsystem for single node)."""
        return self.get_operator_subsystem(operator_id, mode)

    def get_position_operator(
        self,
        mode: Union[int, str],
        coord: str,
        view: Literal["subsystem", "full"] = "full",
    ):
        """Get position operator for a coordinate in a mode."""
        view_obj = self.get_mode_node(mode)
        op = view_obj.operators.get(coord) or view_obj.operators.get("X")
        if op is None:
            raise ValueError(
                f"Coordinate '{coord}' not found. Available: {view_obj.qm_nums_names}"
            )
        return op

    def get_mode_coordinate_labels(self, mode: Union[int, str]) -> List[str]:
        """Return coordinate labels for a mode (e.g. ['q1'])."""
        view = self.get_mode_node(mode)
        return list(view.qm_nums_names)

    def get_mode_energy_operator(self, mode: Union[int, str]) -> "MatrixOperator":
        """Return phonon energy operator K_i for mode (for LVC Hamiltonian)."""
        return self.get_operator_subsystem("K", mode)

    def get_mode_position_operator(
        self, mode: Union[int, str], coord: Optional[str] = None
    ) -> "MatrixOperator":
        """Return position operator for mode (for LVC Hamiltonian)."""
        if coord is None:
            labels = self.get_mode_coordinate_labels(mode)
            coord = labels[0]
        return self.get_position_operator(mode, coord, view="full")

    def get_position_squared_operator(
        self,
        mode: Union[int, str] = 1,
        view: Literal["subsystem", "full"] = "full",
    ) -> "MatrixOperator":
        """Get squared-position operator for the first coordinate (legacy XX)."""
        coord = self.get_mode_coordinate_labels(mode)[0]
        q_op = self.get_position_operator(mode, coord, view=view)
        return q_op * q_op

    def evaluate_position_expression(
        self,
        expr: str,
        mode: Union[int, str] = 1,
        view: Literal["subsystem", "full"] = "full",
    ) -> "MatrixOperator":
        """Parse and evaluate a position expression for a mode."""
        from .position_expr_parser import evaluate_position_expression as _eval

        mode_view = self.get_mode_node(mode)
        labels = list(mode_view.qm_nums_names)
        return _eval(mode_view, expr, allowed_coords=labels)
