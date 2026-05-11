"""
Quantum system tree construction.

This module provides factory functions to build quantum system trees
for different Jahn-Teller models.

System structures:
- Electron-phonon: point_defect -> [nuclei (mode_1), electron_system (orbital_system)]
- Spin-electron-phonon: point_defect -> [nuclei (mode_1), electron_system (orbital_system, spin_system)]
- Minimal model: point_defect -> [electron_system (orbital_system, spin_system)]
"""

from typing import TYPE_CHECKING, Optional, Sequence, List, Union, Literal, Dict, Tuple, Callable
if TYPE_CHECKING:
    import jahn_teller_dynamics.physics.quantum_system as qs
    import jahn_teller_dynamics.physics.jahn_teller_theory as jt
else:
    import jahn_teller_dynamics.physics.quantum_system as qs

from .position_expr_parser import evaluate_position_expression as _evaluate_position_expression


def _parse_mode_config(
    item: Union[float, Tuple[float, Sequence[str]]],
) -> Tuple[float, List[str], bool]:
    """Convert mode config to (omega, labels, is_bare_float)."""
    if isinstance(item, (int, float)):
        return float(item), ["x"], True  # 1D placeholder label; caller assigns q1, q2, ...
    omega, labels = item[0], list(item[1])
    return float(omega), labels, False


def _expand_multimode_to_one_dim_modes(
    modes: Sequence[Union[float, Tuple[float, Sequence[str]]]],
) -> List[Tuple[float, str]]:
    """
    Flatten user mode entries into ``(frequency, coordinate_label)`` pairs.

    A bare float counts as **one** 1D mode (label ``q{entry_index}`` in the user's list).
    A tuple ``(omega, ['x','y',...])`` becomes **multiple** modes sharing ``omega``, one label
    each—tensor product of independent oscillators, not a coupled multi-coordinate subspace.
    """
    out: List[Tuple[float, str]] = []
    for entry_idx, m in enumerate(modes, start=1):
        omega, labels, is_bare_float = _parse_mode_config(m)
        if is_bare_float:
            out.append((omega, f"q{entry_idx}"))
            continue
        if not labels:
            raise ValueError(
                "Each (omega, labels) phonon mode entry must declare at least one coordinate label."
            )
        for lab in labels:
            out.append((omega, str(lab)))
    return out


class _MultiModeTensorCouplingExprView:
    """
    Adapter for :func:`position_expr_parser.evaluate_position_expression` on a tensor
    product of single-mode phonon subspaces (PVC / DJT coupling rows with ``q1*q2*...``).
    """

    def __init__(self, phonon: "MultiModePhononSystem") -> None:
        self._mm = phonon

    def create_id_op(self):
        import jahn_teller_dynamics.math.matrix_mechanics as mm
        import jahn_teller_dynamics.math.maths as maths

        use_sparse = getattr(self._mm, "use_sparse", False)
        matrix_type = maths.SparseMatrix if use_sparse else maths.Matrix
        return mm.MatrixOperator.create_id_matrix_op(
            dim=self._mm.dim, matrix_type=matrix_type
        )

    def get_position_operator(self, coord: str):
        for i in range(1, len(self._mm.children) + 1):
            mode_node = self._mm.get_mode_node(i)
            if coord in mode_node.qm_nums_names:
                return self._mm.get_position_operator(mode=i, coord=coord, view="full")
        raise ValueError(
            f"Coordinate {coord!r} not found in any mode. "
            f"Modes: {[self._mm.get_mode_coordinate_labels(i) for i in range(1, len(self._mm.children) + 1)]}"
        )


class MultiModePhononSystem(qs.quantum_system_node):
    """
    Wrapper node for a tensor product of independent 1D harmonic modes.

    Every child ``mode_k`` is a :class:`~jahn_teller_dynamics.physics.quantum_physics.OneDimPhononSys`
    with a single coordinate. A user entry ``(omega, ['x','y'])`` becomes **two** children (same
    ``omega``, labels ``x`` and ``y``); intra-row coupling such as ``q_x * q_y`` is expressed via
    :meth:`evaluate_coupling_expression`, not single-mode polynomials.

    Args:
        modes: See factory :func:`build_phonon_system`.
        order: Per-mode truncation (same semantics as legacy ``one_mode_phonon_sys`` for one axis):
            active dimension ``order + 1`` per 1D mode.
        use_sparse: Matrix backend for ladder and related operators.
        phonon_system_id: Root node id.
        dimensionless_coordinates: If True, :math:`q=(a+a^\\dagger)/\\sqrt{2}`.
        null_point_vib: If True, add zero-point energy :math:`\\frac12\\hbar\\omega` per oscillator axis.
        exp_approximation_order: If set to a non-negative int ``N``, each ``exp(...)`` in coupling
            expressions uses ``\\sum_{k=0}^N A^k/k!``; ``None`` uses exact ``scipy.linalg.expm``.
    """

    def __init__(
        self,
        modes: Sequence[Union[float, Tuple[float, Sequence[str]]]],
        order: int,
        use_sparse: bool = True,
        phonon_system_id: str = "phonon_system",
        dimensionless_coordinates: bool = True,
        null_point_vib: bool = True,
        *,
        exp_approximation_order: Optional[int] = None,
    ) -> None:
        from jahn_teller_dynamics.physics.quantum_physics import OneDimPhononSys

        self.order = order
        self.use_sparse = use_sparse
        self.exp_approximation_order = exp_approximation_order
        flat_modes = _expand_multimode_to_one_dim_modes(modes)
        self._mode_configs = [(omega, [label]) for omega, label in flat_modes]
        self.modes = [omega for omega, _ in flat_modes]

        eff_dim = order + 1
        mode_nodes: List["qs.quantum_system_node"] = []
        for i, (omega, coord_name) in enumerate(flat_modes, start=1):
            mode_id = f"mode_{i}"
            mode_nodes.append(
                OneDimPhononSys(
                    omega,
                    eff_dim,
                    coord_name=coord_name,
                    phonon_sys_name=mode_id,
                    id=mode_id,
                    use_sparse=use_sparse,
                    dimensionless_coordinates=dimensionless_coordinates,
                    null_point_vib=null_point_vib,
                    matching_phonon_order=order,
                )
            )

        super().__init__(phonon_system_id, children=mode_nodes)

        # Total phonon harmonic operator on the tensor-product space (for PVC / diagnostics).
        k_parts = [self.get_operator_full("K", mode=i) for i in range(1, len(mode_nodes) + 1)]
        self.operators["K"] = sum(k_parts)

    def _mode_id(self, mode: Union[int, str]) -> str:
        if isinstance(mode, int):
            if mode < 1:
                raise ValueError("mode index must be 1-based (mode_1, mode_2, ...)")
            return f"mode_{mode}"
        return mode

    def get_mode_node(self, mode: Union[int, str]) -> "qs.quantum_system_node":
        mode_id = self._mode_id(mode)
        node = self.find_node(mode_id)
        if node is None:
            raise ValueError(f"Mode '{mode_id}' not found in phonon system '{self.id}'")
        return node  # type: ignore[return-value]

    def get_operator_subsystem(self, operator_id: str, mode: Union[int, str] = 1):
        """
        Get an operator in **subsystem view** (single mode only, not embedded).
        operator_id can be a coordinate label (e.g. 'x', 'y') or legacy 'X', 'Y', 'XX'.
        """
        mode_node = self.get_mode_node(mode)
        if operator_id not in mode_node.operators:
            raise ValueError(f"Operator '{operator_id}' not found in {mode_node.id}")
        return mode_node.operators[operator_id]

    def get_operator_full(self, operator_id: str, mode: Union[int, str] = 1):
        """
        Get an operator in **multi-mode view** (embedded into the full phonon system).
        """
        mode_id = self._mode_id(mode)
        return self.create_operator(operator_id=operator_id, operator_system_id=mode_id)

    def get_position_operator(
        self,
        mode: Union[int, str],
        coord: str,
        view: Literal["subsystem", "full"] = "full",
    ):
        """
        Get the position operator for a coordinate in a mode.

        Args:
            mode: Mode index (1-based) or mode id.
            coord: Coordinate label (e.g. 'x', 'y') from the mode's qm_nums_names.
            view: 'subsystem' (single-mode) or 'full' (embedded).
        """
        mode_node = self.get_mode_node(mode)
        op = mode_node.get_position_operator(coord)
        if view == "subsystem":
            return op
        mode_id = self._mode_id(mode)
        return self.create_operator(operator_id=coord, operator_system_id=mode_id)

    def get_position_operators(
        self,
        mode: Union[int, str],
        view: Literal["subsystem", "full"] = "full",
    ) -> Dict[str, object]:
        """
        Get position operators for all coordinates in a mode.

        Returns a dict keyed by coordinate label (e.g. 'x', 'y').
        """
        mode_node = self.get_mode_node(mode)
        ops: Dict[str, object] = {}
        for label in mode_node.qm_nums_names:
            ops[label] = self.get_position_operator(mode=mode, coord=label, view=view)
        return ops

    def get_mode_coordinate_labels(self, mode: Union[int, str]) -> List[str]:
        """Return the coordinate labels for a mode (e.g. ['x','y'])."""
        mode_node = self.get_mode_node(mode)
        return list(mode_node.qm_nums_names)

    def get_position_squared_operator(
        self,
        mode: Union[int, str] = 1,
        view: Literal["subsystem", "full"] = "full",
    ):
        """
        Get :math:`q^2` for the single coordinate of this 1D mode (stored as ``XX``).

        Each child mode has exactly one ladder; ``XX`` is the coupled-truncation legacy convention
        from :class:`~jahn_teller_dynamics.physics.quantum_physics.OneDimPhononSys`.
        """
        if view == "subsystem":
            return self.get_operator_subsystem("XX", mode=mode)
        return self.get_operator_full("XX", mode=mode)

    def evaluate_position_expression(
        self,
        expr: str,
        mode: Union[int, str] = 1,
        view: Literal["subsystem", "full"] = "full",
    ):
        """
        Parse and evaluate a position expression on **one** 1D child mode.

        Allowed coordinates are that mode's single label only (typically ``qx`` syntax in the parser).
        Powers and polynomials in that coordinate are supported. Products of coordinates that live on
        different child modes belong in :meth:`evaluate_coupling_expression`.

        Example (mode label ``q1``):
            ``'q1^2 + 2*q1'``

        Args:
            expr: Expression string. Whitespace is ignored.
            mode: Mode index (1-based) or mode id.
            view: ``'full'`` (embedded in full phonon tensor space) or ``'subsystem'``.

        Returns:
            MatrixOperator: The evaluated expression.
        """
        mode_node = self.get_mode_node(mode)
        labels = list(mode_node.qm_nums_names)
        op = _evaluate_position_expression(
            mode_node,
            expr,
            allowed_coords=labels,
            exp_approximation_order=getattr(self, "exp_approximation_order", None),
        )
        if view == "subsystem":
            return op
        mode_id = self._mode_id(mode)
        return self._embed_subsystem_operator(op, mode_id)

    def evaluate_coupling_expression(self, expr: str):
        """
        Evaluate a polynomial coupling expression on the **full** multimode tensor-product space.

        Coordinate names are the per-mode labels (e.g. ``q1``, ``q2`` from CSV), consistent
        with :class:`~jahn_teller_dynamics.physics.models.constrained_multimode_phonon.MultiModeConstrainedPhononSystem`.
        """
        helper = _MultiModeTensorCouplingExprView(self)
        labels: List[str] = []
        for i in range(1, len(self.children) + 1):
            labels.extend(self.get_mode_coordinate_labels(i))
        return _evaluate_position_expression(
            helper,
            expr,
            allowed_coords=labels,
            exp_approximation_order=getattr(self, "exp_approximation_order", None),
        )

    def _embed_subsystem_operator(self, op, mode_id: str):
        """Embed a subsystem operator into the full phonon space."""
        import itertools
        import operator as op_mod
        import jahn_teller_dynamics.math.matrix_mechanics as mm
        import jahn_teller_dynamics.math.maths as maths

        left_systems, system, right_systems = self.find_leaves_avoid(mode_id)
        if system is None:
            raise ValueError(f"Mode '{mode_id}' not found")
        left_dims = [x.dim for x in left_systems]
        left_dim = list(itertools.accumulate(left_dims, op_mod.mul))[-1] if left_dims else 1
        right_dims = [x.dim for x in right_systems]
        right_dim = list(itertools.accumulate(right_dims, op_mod.mul))[-1] if right_dims else 1
        use_sparse = getattr(self, "use_sparse", False)
        matrix_type = maths.SparseMatrix if use_sparse else maths.Matrix
        I_left = mm.MatrixOperator.create_id_matrix_op(dim=left_dim, matrix_type=matrix_type)
        I_right = mm.MatrixOperator.create_id_matrix_op(dim=right_dim, matrix_type=matrix_type)
        return I_left ** op ** I_right


def build_phonon_system(
    modes: Sequence[Union[float, Tuple[float, Sequence[str]]]],
    order: int,
    use_sparse: bool = False,
    phonon_system_id: str = "phonon_system",
    dimensionless_coordinates: bool = True,
    null_point_vib: bool = True,
    *,
    exp_approximation_order: Optional[int] = None,
) -> "qs.quantum_system_node":
    """
    Build a tensor-product phonon system: each child is a 1D oscillator.

    Args:
        modes: List of user-defined mode entries. Each item is either:
            - float/int: one 1D oscillator; label ``qk`` where ``k`` is the entry index in this list.
            - ``(omega, labels)``: one row per label—``(1.0, ['x','y'])`` yields **two**
              :class:`OneDimPhononSys` children (both with ``omega=1.0``), tensor-multiplied together.
        order: Per-axis truncation order (active Fock dimension ``order + 1`` each).
        use_sparse: If True, prefer sparse operators.
        phonon_system_id: Parent node id.
        dimensionless_coordinates: If True (default), use dimensionless q=(a+a†)/√2.
        null_point_vib: If True (default), include 0.5*ħω zero-point energy in H.

    Returns:
        :class:`MultiModePhononSystem` with ``mode_1``, ``mode_2``, ….
    """
    return MultiModePhononSystem(
        modes=modes,
        order=order,
        use_sparse=use_sparse,
        phonon_system_id=phonon_system_id,
        dimensionless_coordinates=dimensionless_coordinates,
        null_point_vib=null_point_vib,
        exp_approximation_order=exp_approximation_order,
    )


def build_phonon_system_constrained(
    modes: Sequence[float],
    order: int,
    use_sparse: bool = False,
    phonon_system_id: str = "phonon_system",
    dimensionless_coordinates: bool = True,
    null_point_vib: bool = True,
    mode_names: Optional[Sequence[str]] = None,
    build_log: Optional[Callable[[str], None]] = None,
    *,
    exp_approximation_order: Optional[int] = None,
) -> "qs.quantum_system_node":
    """
    Build a multi-mode phonon system with total-phonon-number constraint.

    Hilbert space: states |n1,...,nN⟩ with sum(n_i) <= order (smaller than product).

    Args:
        modes: List of mode energies (hbar*omega).
        order: Maximum total phonon number.
        use_sparse: If True, use sparse matrices.
        phonon_system_id: Node id.
        dimensionless_coordinates: If True, q = (a+a†)/√2.
        null_point_vib: If True, include zero-point energy.
        mode_names: Optional one label per mode (Fock/position); default ``q1``..``qN``.
        build_log: Optional callback for build progress and timings (Hilbert basis, ladders, etc.).

    Returns:
        MultiModeConstrainedPhononSystem.
    """
    from .constrained_multimode_phonon import MultiModeConstrainedPhononSystem

    return MultiModeConstrainedPhononSystem(
        modes=list(modes),
        order=order,
        use_sparse=use_sparse,
        phonon_system_id=phonon_system_id,
        dimensionless_coordinates=dimensionless_coordinates,
        null_point_vib=null_point_vib,
        mode_names=mode_names,
        build_log=build_log,
        exp_approximation_order=exp_approximation_order,
    )


def build_phonon_system_constrained_from_csv(
    modes_csv_path: str,
    *,
    order: int,
    use_sparse: bool = False,
    phonon_system_id: str = "phonon_system",
    separator: str = ";",
    dimensionless_coordinates: bool = True,
    null_point_vib: bool = True,
    mode_numbers: Optional[Sequence[int]] = None,
    exp_approximation_order: Optional[int] = None,
) -> "qs.quantum_system_node":
    """
    Build a constrained multi-mode phonon system from CSV.
    Hilbert space: sum(n_i) <= order (smaller than product).
    """
    from jahn_teller_dynamics.io.file_io.csv_reader import CSVReader

    reader = CSVReader(separator=separator)
    table = reader.read_modes_flexible(
        modes_csv_path,
        mode_numbers=list(mode_numbers) if mode_numbers is not None else None,
    )
    return build_phonon_system_constrained(
        modes=table.omegas,
        order=order,
        use_sparse=use_sparse,
        phonon_system_id=phonon_system_id,
        dimensionless_coordinates=dimensionless_coordinates,
        null_point_vib=null_point_vib,
        mode_names=table.labels,
        exp_approximation_order=exp_approximation_order,
    )


def build_phonon_system_from_csv(
    modes_csv_path: str,
    *,
    order: int,
    use_sparse: bool = False,
    phonon_system_id: str = "phonon_system",
    separator: str = ";",
    dimensionless_coordinates: bool = True,
    null_point_vib: bool = True,
    mode_numbers: Optional[Sequence[int]] = None,
    exp_approximation_order: Optional[int] = None,
) -> "qs.quantum_system_node":
    """
    Build a multi-mode phonon system from a CSV like:

        mode;omega
        1;0.0041
        2;0.0035
        ...

    Args:
        modes_csv_path: Path to modes CSV.
        order: Truncation order for each 1D phonon mode.
        use_sparse: If True, prefer sparse operators.
        phonon_system_id: Parent node id (default: 'phonon_system').
        separator: CSV delimiter (default: ';').
        dimensionless_coordinates: If True (default), use dimensionless coordinates.
        null_point_vib: If True (default), include zero-point energy in H.

    Returns:
        MultiModePhononSystem (as a quantum_system_node subclass).
    """
    from jahn_teller_dynamics.io.file_io.csv_reader import CSVReader

    reader = CSVReader(separator=separator)
    omegas = reader.read_modes(modes_csv_path, mode_numbers=mode_numbers)
    return build_phonon_system(
        modes=omegas,
        order=order,
        use_sparse=use_sparse,
        phonon_system_id=phonon_system_id,
        dimensionless_coordinates=dimensionless_coordinates,
        null_point_vib=null_point_vib,
        exp_approximation_order=exp_approximation_order,
    )


def build_electron_phonon_system(
    jt_theory: 'jt.Jahn_Teller_Theory',
    order: int,
    spatial_dim: int = 2,
    use_sparse: bool = False
) -> 'qs.quantum_system_tree':
    """
    Build quantum system tree for electron-phonon JT model.
    
    Structure:
        point_defect
        ├── nuclei
        │   └── mode_1 (phonon system)
        └── electron_system
            └── orbital_system
    
    Args:
        jt_theory: JT theory parameters (needs hw_meV attribute)
        order: Phonon truncation order
        spatial_dim: Spatial dimension (default: 2 for E⊗e JT)
        use_sparse: If True, use SparseMatrix instead of Matrix for operators
        
    Returns:
        quantum_system_tree: Constructed system tree
    """
    from jahn_teller_dynamics.physics.quantum_physics import one_mode_phonon_sys
    
    # Build orbital system
    orbital_system = qs.quantum_system_node.create_2D_orbital_system_node(use_sparse=use_sparse)
    electron_system = qs.quantum_system_node('electron_system', children=[orbital_system])
    
    # Build phonon system (defaults: dimensionless coords, null-point vib)
    mode_1 = one_mode_phonon_sys(
        jt_theory.hw_meV,
        spatial_dim,
        order,
        ["x", "y"],
        "mode_1",
        "mode_1",
        use_sparse=use_sparse,
        dimensionless_coordinates=True,
        null_point_vib=True,
    )
    
    # Build nuclei
    nuclei = qs.quantum_system_node('nuclei')
    
    # Build point defect
    point_defect_node = qs.quantum_system_node(
        'point_defect', 
        children=[nuclei, electron_system]
    )
    
    # Create tree and insert phonon mode
    point_defect_tree = qs.quantum_system_tree(point_defect_node)
    point_defect_tree.insert_node('nuclei', mode_1)
    
    return point_defect_tree


def build_spin_electron_phonon_system(
    jt_theory: 'jt.Jahn_Teller_Theory',
    order: int,
    spatial_dim: int = 2,
    use_sparse: bool = False
) -> 'qs.quantum_system_tree':
    """
    Build quantum system tree for spin-electron-phonon JT model.
    
    Structure:
        point_defect
        ├── nuclei
        │   └── mode_1 (phonon system)
        └── electron_system
            ├── orbital_system
            └── spin_system
    
    Args:
        jt_theory: JT theory parameters (needs hw_meV attribute)
        order: Phonon truncation order
        spatial_dim: Spatial dimension (default: 2 for E⊗e JT)
        use_sparse: If True, use SparseMatrix instead of Matrix for operators
        
    Returns:
        quantum_system_tree: Constructed system tree
    """
    from jahn_teller_dynamics.physics.quantum_physics import one_mode_phonon_sys
    
    # Build orbital system
    orbital_system = qs.quantum_system_node.create_2D_orbital_system_node(use_sparse=use_sparse)
    electron_system = qs.quantum_system_node('electron_system', children=[orbital_system])
    
    # Build spin system
    spin_sys = qs.quantum_system_node.create_spin_system_node(use_sparse=use_sparse)
    
    # Build phonon system (defaults: dimensionless coords, null-point vib)
    mode_1 = one_mode_phonon_sys(
        jt_theory.hw_meV,
        spatial_dim,
        order,
        ["x", "y"],
        "mode_1",
        "mode_1",
        use_sparse=use_sparse,
        dimensionless_coordinates=True,
        null_point_vib=True,
    )
    
    # Build nuclei
    nuclei = qs.quantum_system_node('nuclei')
    
    # Build point defect
    point_defect_node = qs.quantum_system_node(
        'point_defect', 
        children=[nuclei, electron_system]
    )
    
    # Create tree and insert nodes
    point_defect_tree = qs.quantum_system_tree(point_defect_node)
    point_defect_tree.insert_node('nuclei', mode_1)
    point_defect_tree.insert_node('electron_system', spin_sys)
    
    return point_defect_tree


def build_minimal_model_system(use_sparse: bool = False) -> 'qs.quantum_system_tree':
    """
    Build quantum system tree for minimal model (no phonons).
    
    Structure:
        point_defect
        └── electron_system
            ├── orbital_system
            └── spin_system
    
    This is used for the four-state model Hamiltonian where phonons
    are integrated out and only the electronic degrees of freedom remain.
    
    Args:
        use_sparse: If True, create operators as SparseMatrix instead of Matrix
    
    Returns:
        quantum_system_tree: Constructed system tree
    """
    # Build orbital and spin systems
    orbital_system = qs.quantum_system_node.create_2D_orbital_system_node(use_sparse=use_sparse)
    spin_sys = qs.quantum_system_node.create_spin_system_node(use_sparse=use_sparse)
    
    # Build electron system
    electron_system = qs.quantum_system_node(
        'electron_system', 
        children=[orbital_system, spin_sys]
    )
    
    # Build point defect
    point_defect_system = qs.quantum_system_node(
        'point_defect', 
        children=[electron_system]
    )
    
    # Create tree
    return qs.quantum_system_tree(point_defect_system)

