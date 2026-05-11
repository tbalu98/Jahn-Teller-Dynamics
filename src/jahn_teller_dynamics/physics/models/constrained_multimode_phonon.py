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

import copy
import math
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union, Literal, Dict, TYPE_CHECKING, Callable, Tuple

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
        id_op: Optional["MatrixOperator"] = None,
    ) -> None:
        self.id = mode_id
        self.operators = operators
        self.qm_nums_names = qm_nums_names
        self._id_op = id_op

    def create_id_op(self) -> "MatrixOperator":
        """Identity on phonon space (for expression parser integer coefficients)."""
        if self._id_op is not None:
            return self._id_op
        raise ValueError(
            "create_id_op() requires id_op=... when constructing _ModeView for expression parsing"
        )

    def get_position_operator(self, coord: str) -> "MatrixOperator":
        op = self.operators.get(coord) or self.operators.get("X")
        # Support qx -> x when mode uses short labels (e.g. one_mode with ["x","y"])
        if op is None and coord.startswith("q") and len(coord) > 1:
            op = self.operators.get(coord[1:])
        if op is None:
            raise ValueError(
                f"Coordinate '{coord}' not found. Available: {list(self.qm_nums_names)}"
            )
        return op


@dataclass(frozen=True)
class ConstrainedFockLadderBuild:
    """Hilbert basis + ladder operators from :func:`build_constrained_fock_basis_and_ladder_operators`."""

    calculation_bases: mm.hilber_space_bases
    create_mx_ops: Dict[str, mm.MatrixOperator]
    annil_mx_ops: Dict[str, mm.MatrixOperator]


def build_constrained_fock_basis_and_ladder_operators(
    num_modes: int,
    order: int,
    qm_nums_names: Sequence[str],
    *,
    subsystem_name: str = "phonon_system",
    use_sparse: bool = True,
) -> ConstrainedFockLadderBuild:
    """
    Build the constrained Fock basis (``sum_i n_i <= order``) and bosonic creator matrices in one pass.

    For each transition ``|\\text{parent}\\rangle \\to |\\text{child}\\rangle`` that adds one
    quantum in mode ``m``, the matrix element is filled as ``M[i,j] = bra_j(\\cdots)ket_i``
    expects in :mod:`~jahn_teller_dynamics.math.matrix_mechanics` adjacent-order ladder builds:
    ``i`` is the ket index for the **child** (raised occupation) and ``j`` the ket/bra partner
    index for the **parent**, with value :math:`\\sqrt{n_m^{(\\text{child})}}`.


    Basis order and indexing follow the same DFS as
    :meth:`~jahn_teller_dynamics.math.matrix_mechanics.hilber_space_bases.dynamic_create_hosc_eigen_states_list`.
    """
    names = [str(n).strip() for n in qm_nums_names]
    if len(names) != num_modes:
        raise ValueError(
            f"qm_nums_names has length {len(names)} but num_modes is {num_modes}"
        )
    if num_modes < 1:
        raise ValueError("num_modes must be >= 1")
    if order < 0:
        raise ValueError("order must be non-negative")

    dim_total = math.comb(num_modes + order, order)
    bras: List[bf.bra_state] = []
    kets: List[bf.ket_state] = []
    indexer = mm.BasisStateIndexer()
    seen: set[Tuple[int, ...]] = set()
    occ_to_index: Dict[Tuple[int, ...], int] = {}

    # Stored as (child_ket_idx, parent_ket_idx, amp); matrix fill M[child, parent] (see adjacent-order builders).
    edges_per_mode: List[List[Tuple[int, int, float]]] = [[] for _ in range(num_modes)]

    ground = tuple([0] * num_modes)
    seen.add(ground)
    root_idx = indexer.allocate(None)
    occ_to_index[ground] = root_idx

    bras.append(bf.bra_state(qm_nums=list(ground)))
    kets.append(bf.ket_state(qm_nums=list(ground)))

    def dfs(parent_occ: List[int], parent_basis_index: int) -> None:
        for mi in range(num_modes):
            occ = copy.copy(parent_occ)
            occ[mi] += 1
            if sum(occ) > order:
                continue
            tup = tuple(occ)
            amp = math.sqrt(float(tup[mi]))

            if tup not in seen:
                seen.add(tup)
                child_idx = indexer.allocate(parent_basis_index)
                occ_to_index[tup] = child_idx
                edges_per_mode[mi].append((child_idx, parent_basis_index, amp))

                bras.append(bf.bra_state(qm_nums=list(tup)))
                kets.append(bf.ket_state(qm_nums=list(tup)))
                dfs(list(tup), child_idx)
            else:
                child_idx = occ_to_index[tup]
                edges_per_mode[mi].append((child_idx, parent_basis_index, amp))

    dfs(list(ground), root_idx)

    if len(kets) != dim_total:
        raise RuntimeError(
            f"Internal error: expected dim {dim_total} from combinatorics, got {len(kets)}"
        )

    bases = mm.hilber_space_bases(bra_states=bras, ket_states=kets, names=list(names))
    bases.basis_state_indexer = indexer
    bases.basis_parent_of = indexer.parent_of

    create_mx_ops: Dict[str, mm.MatrixOperator] = {}
    annil_mx_ops: Dict[str, mm.MatrixOperator] = {}

    complex_dtype = maths.complex_number_typ

    if use_sparse:
        from scipy.sparse import dok_matrix

        for mi, nm in enumerate(names):
            dok = dok_matrix((dim_total, dim_total), dtype=complex_dtype)
            for ci, pi, val in edges_per_mode[mi]:
                dok[ci, pi] = val + 0j
            csr = dok.tocsr()
            c_op = mm.MatrixOperator(
                maths.SparseMatrix(csr),
                name="",
                subsys_name=subsystem_name,
            )
            create_mx_ops[nm] = c_op
            annil_mx_ops[nm] = c_op.adjoint()
    else:
        for mi, nm in enumerate(names):
            mat = np.zeros((dim_total, dim_total), dtype=complex_dtype)
            for ci, pi, val in edges_per_mode[mi]:
                mat[ci, pi] = val + 0j
            c_op = mm.MatrixOperator(
                maths.Matrix(mat),
                name="",
                subsys_name=subsystem_name,
            )
            create_mx_ops[nm] = c_op
            annil_mx_ops[nm] = c_op.adjoint()

    return ConstrainedFockLadderBuild(
        calculation_bases=bases,
        create_mx_ops=create_mx_ops,
        annil_mx_ops=annil_mx_ops,
    )


class MultiModeConstrainedPhononSystem(qs.quantum_system_node):
    """
    Single-node multi-mode quantum harmonic oscillator with total-phonon-number constraint.

    Hilbert space: |n1, n2, ..., nN⟩ with n1 + n2 + ... + nN <= order.
    Basis states and bosonic ladder matrices are built together by
    :func:`build_constrained_fock_basis_and_ladder_operators` (single DFS over occupations).
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
        mode_names: Optional[Sequence[str]] = None,
        build_log: Optional[Callable[[str], None]] = None,
        *,
        exp_approximation_order: Optional[int] = None,
    ) -> None:
        """
        Args:
            modes: List of mode energies (hbar*omega) for each mode.
            order: Maximum total phonon number (sum of quantum numbers <= order).
            use_sparse: If True, creator/annihilator matrices are CSR
                :class:`~jahn_teller_dynamics.math.maths.SparseMatrix` (built in one pass with
                :func:`build_constrained_fock_basis_and_ladder_operators`; no full dense
                ``dim×dim`` ladder fill). If False, dense NumPy matrices are used for the same graph.
            phonon_system_id: ID for this node.
            dimensionless_coordinates: If True, q = (a+a†)/√2; else q = (a+a†)/(√2 ω^0.5).
            null_point_vib: If True, include 0.5*hbar*omega zero-point energy in H.
            mode_names: One label per mode (Fock/position names, e.g. ``q1``, ``q3x``). If
                omitted, names are ``q1``..``qN`` in list order. Used for
                :attr:`qm_nums_names` and position-operator keys.
            build_log: If set, called with short human-readable build progress and timings
                (Hilbert basis, ladder operators, Hamiltonian/position build).
            exp_approximation_order: If a non-negative int ``N``, ``exp(...)`` in coupling
                expressions uses ``sum_{k=0}^N A^k/k!``; ``None`` uses exact ``expm``.
        """
        self.order = order
        self.modes = list(modes)
        self.use_sparse = use_sparse
        self.dimensionless_coordinates = dimensionless_coordinates
        self.null_point_vib = null_point_vib
        self.exp_approximation_order = exp_approximation_order
        self.id = phonon_system_id
        self.children: List[qs.quantum_system_node] = []  # Single node: no children

        num_modes = len(self.modes)
        if mode_names is not None:
            qm_nums_names = [str(n).strip() for n in mode_names]
            if len(qm_nums_names) != num_modes:
                raise ValueError(
                    f"mode_names has length {len(qm_nums_names)} but modes has {num_modes} entries"
                )
            if any(n == "" for n in qm_nums_names):
                raise ValueError("mode_names must not contain empty strings")
            if len(set(qm_nums_names)) != len(qm_nums_names):
                raise ValueError("mode_names must be unique")
        else:
            qm_nums_names = [f"q{i}" for i in range(1, num_modes + 1)]

        # Build Hilbert space: states |n1,...,nN⟩ with sum <= order
        t_basis = time.perf_counter()
        if build_log is not None:
            build_log(
                f"Phonon [{phonon_system_id}]: building quantum states "
                f"(constrained Fock basis: {num_modes} mode(s), sum n_i <= {order})"
            )

        # Basis + a† / a (adj) in one DFS: see build_constrained_fock_basis_and_ladder_operators.
        built = build_constrained_fock_basis_and_ladder_operators(
            num_modes,
            order,
            qm_nums_names,
            subsystem_name=phonon_system_id,
            use_sparse=use_sparse,
        )
        self.calculation_bases = built.calculation_bases
        self.create_mx_ops = built.create_mx_ops
        self.annil_mx_ops = built.annil_mx_ops

        self.base_states = self.calculation_bases
        self.dim = self.calculation_bases.dim
        if build_log is not None:
            build_log(
                f"Phonon [{phonon_system_id}]: quantum states + ladder operators ready — "
                f"dim = {self.dim} (wall time {time.perf_counter() - t_basis:.4f} s)"
            )

        self.names_dict = {
            name: idx for idx, name in enumerate(qm_nums_names)
        }
        self.qm_nums_names = qm_nums_names

        self.mx_op_builder = mm.braket_to_matrix_formalism(
            self.calculation_bases, use_sparse=use_sparse
        )  # For other matrix builds; ladder ops already from build_constrained_fock_basis_and_ladder_operators.

        t_h = time.perf_counter()
        if build_log is not None:
            build_log(
                f"Phonon [{phonon_system_id}]: building K and position operators (per-mode H, q_i)"
            )
        self.operators: Dict[str, mm.MatrixOperator] = {}
        self._mode_views: Dict[int, _ModeView] = {}  # mode index (1-based) -> view
        self._build_hamiltonian_and_position_ops()
        if build_log is not None:
            build_log(
                f"Phonon [{phonon_system_id}]: K and position operators done "
                f"({time.perf_counter() - t_h:.4f} s)"
            )

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
        """
        Legacy path: build creators via :meth:`braket_to_matrix_formalism.create_MatrixOperators_adjacent_orders`.

        :class:`MultiModeConstrainedPhononSystem` uses :func:`build_constrained_fock_basis_and_ladder_operators`
        in ``__init__`` instead; keep this only for debugging parity with the braket implementation.
        """
        self.create_mx_ops: Dict[str, mm.MatrixOperator] = {}
        self.annil_mx_ops: Dict[str, mm.MatrixOperator] = {}
        creators: list[bf.operator] = []
        for name in self.qm_nums_names:
            idx = self.names_dict[name]
            creators.append(bf.creator_operator(idx, name))
        mats = self.mx_op_builder.create_MatrixOperators_adjacent_orders(
            creators,
            subsys_name=self.id,
        )
        for i, name in enumerate(self.qm_nums_names):
            create_op = mats[i]
            self.create_mx_ops[name] = create_op
            self.annil_mx_ops[name] = create_op.adjoint()

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
        """Resolve mode to 1-based index (also accepts a coordinate name from :attr:`qm_nums_names`)."""
        if isinstance(mode, int):
            if mode < 1:
                raise ValueError("mode index must be 1-based (mode_1, mode_2, ...)")
            return mode
        if isinstance(mode, str):
            if mode in self.qm_nums_names:
                return int(self.names_dict[mode] + 1)
            if mode.startswith("mode_"):
                return int(mode.split("_")[1])
        raise ValueError(f"Invalid mode: {mode!r} (use 1..{len(self.modes)} or a name in {self.qm_nums_names})")

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
        """Parse and evaluate a position expression. Uses full system (all coords) for single-node constrained space."""
        from .position_expr_parser import evaluate_position_expression as _eval

        # Single node with all coords (q1, q2, ...); use full system so expressions like q1*q2 work
        full_view = _ModeView(
            mode_id="full",
            operators=self.operators,
            qm_nums_names=self.qm_nums_names,
            id_op=self._create_id_op(),
        )
        return _eval(
            full_view,
            expr,
            allowed_coords=list(self.qm_nums_names),
            exp_approximation_order=self.exp_approximation_order,
        )

    def evaluate_coupling_expression(self, expr: str) -> "MatrixOperator":
        """PVC / DJT: same as :meth:`evaluate_position_expression` for the constrained multimode basis."""
        return self.evaluate_position_expression(expr)
