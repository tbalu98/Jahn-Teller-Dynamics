"""
Quantum system tree construction.

This module provides factory functions to build quantum system trees
for different Jahn-Teller models.

System structures:
- Electron-phonon: point_defect -> [nuclei (mode_1), electron_system (orbital_system)]
- Spin-electron-phonon: point_defect -> [nuclei (mode_1), electron_system (orbital_system, spin_system)]
- Minimal model: point_defect -> [electron_system (orbital_system, spin_system)]
"""

from typing import TYPE_CHECKING, Sequence, List, Union, Literal, Dict, Tuple
if TYPE_CHECKING:
    import jahn_teller_dynamics.physics.quantum_system as qs
    import jahn_teller_dynamics.physics.jahn_teller_theory as jt
else:
    import jahn_teller_dynamics.physics.quantum_system as qs

from .position_expr_parser import evaluate_position_expression as _evaluate_position_expression


def _parse_mode_config(
    item: Union[float, Tuple[float, Sequence[str]]],
) -> Tuple[float, List[str]]:
    """Convert mode config to (omega, labels)."""
    if isinstance(item, (int, float)):
        return float(item), ["x"]  # 1D default
    omega, labels = item[0], list(item[1])
    return float(omega), labels


class MultiModePhononSystem(qs.quantum_system_node):
    """
    Wrapper node for a multi-mode phonon system.

    Each mode can have N spatial dimensions with user-defined coordinate labels
    (e.g. N=2 with ['x','y'] or N=5 with ['x','y','z','u','w']). Position expressions
    operate within a single mode on its coordinates (e.g. qx*qy^2, 2*(qx^2 + qy^2)).
    """

    def __init__(
        self,
        modes: Sequence[Union[float, Tuple[float, Sequence[str]]]],
        order: int,
        use_sparse: bool = True,
        phonon_system_id: str = "phonon_system",
        dimensionless_coordinates: bool = True,
        null_point_vib: bool = True,
    ) -> None:
        from jahn_teller_dynamics.physics.quantum_physics import one_mode_phonon_sys

        self.order = order
        self.use_sparse = use_sparse
        self._mode_configs: List[Tuple[float, List[str]]] = [
            _parse_mode_config(m) for m in modes
        ]
        self.modes = [mc[0] for mc in self._mode_configs]

        mode_nodes: List["qs.quantum_system_node"] = []
        for i, (omega, labels) in enumerate(self._mode_configs, start=1):
            mode_id = f"mode_{i}"
            mode_nodes.append(
                one_mode_phonon_sys(
                    omega,
                    len(labels),
                    order,
                    labels,
                    mode_id,
                    mode_id,
                    use_sparse=use_sparse,
                    dimensionless_coordinates=dimensionless_coordinates,
                    null_point_vib=null_point_vib,
                )
            )

        super().__init__(phonon_system_id, children=mode_nodes)

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
        Get the squared-position operator for the first coordinate (legacy XX).

        For backward compatibility with 1D/2D modes.
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
        Parse and evaluate a position expression for a single mode.

        Expressions operate on coordinates within that mode (e.g. qx, qy).
        Supports: products (qx*qy^2), addition/subtraction, scalar mult, parentheses.

        Examples (for mode with labels ['x','y']):
            'qx*qy^2'       -> qx · qy²
            '2*(qx^2 + qy^2)' -> 2·(qx² + qy²)
            'qx + qy'       -> qx + qy

        Args:
            expr: Expression string. Whitespace is ignored.
            mode: Mode index (1-based) or mode id.
            view: 'full' (embedded in multi-mode space) or 'subsystem' (single-mode).

        Returns:
            MatrixOperator: The evaluated expression.
        """
        mode_node = self.get_mode_node(mode)
        labels = list(mode_node.qm_nums_names)
        op = _evaluate_position_expression(mode_node, expr, allowed_coords=labels)
        if view == "subsystem":
            return op
        mode_id = self._mode_id(mode)
        return self._embed_subsystem_operator(op, mode_id)

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
) -> "qs.quantum_system_node":
    """
    Build a phonon system node with multiple modes.

    Args:
        modes: List of mode configs. Each item is either:
            - float: 1D mode with default coordinate 'x'
            - (omega, labels): e.g. (1.0, ['x','y']) or (2.0, ['x','y','z','u','w'])
        order: Truncation order per mode.
        use_sparse: If True, prefer sparse operators.
        phonon_system_id: Parent node id.
        dimensionless_coordinates: If True (default), use dimensionless q=(a+a†)/√2.
        null_point_vib: If True (default), include 0.5*ħω zero-point energy in H.

    Returns:
        MultiModePhononSystem with mode_1, mode_2, ... (each one_mode_phonon_sys).
    """
    return MultiModePhononSystem(
        modes=modes,
        order=order,
        use_sparse=use_sparse,
        phonon_system_id=phonon_system_id,
        dimensionless_coordinates=dimensionless_coordinates,
        null_point_vib=null_point_vib,
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
    omegas = reader.read_modes(modes_csv_path)
    return build_phonon_system(
        modes=omegas,
        order=order,
        use_sparse=use_sparse,
        phonon_system_id=phonon_system_id,
        dimensionless_coordinates=dimensionless_coordinates,
        null_point_vib=null_point_vib,
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

