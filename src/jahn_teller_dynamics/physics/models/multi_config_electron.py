"""
Multi-configuration electron subsystem configured from a CSV file.

The CSV is expected to contain two columns:
  - state: name/label of the electronic basis state
  - value: diagonal electronic state energy associated with that state

Example (semicolon separated):
    state;value
    11;0
    12;7.902e-05
    13;0.07331616
    31;-0.03173566
    32;0.07632567
    33;0.07636797
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

import jahn_teller_dynamics.math.maths as maths
import jahn_teller_dynamics.math.matrix_mechanics as mm
import jahn_teller_dynamics.physics.quantum_system as qs
from jahn_teller_dynamics.io.file_io.csv_reader import CSVReader


@dataclass
class multi_config_electron:
    """
    Wrapper around a ``quantum_system_node`` for a multi-configuration electronic basis.

    After construction, ``node.operators['state_energy']`` is a diagonal sparse operator
    (MatrixOperator wrapping a SparseMatrix) with entries taken from the CSV ``value`` column.

    Backward-compatibility:
      - ``node.operators['epsilon']`` is also set as an alias to ``state_energy``.
      - Coupling operators are stored as ``coupling_mode_{i}`` (alias: ``V_mode_{i}``)
      - Tuning operators are stored as ``tuning_mode_{i}`` (alias: ``kappa_mode_{i}``)
    """

    node: qs.quantum_system_node
    state_to_index: Dict[str, int]

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        *,
        subsystem_id: str = "electron_system",
        qm_nums_name: str = "state",
        separator: str = ";",
        use_sparse: bool = True,
    ) -> "multi_config_electron":
        """
        Build the electron subsystem from a CSV config.

        Args:
            csv_path: Path to the config CSV.
            subsystem_id: Node id to assign to the created quantum_system_node.
            qm_nums_name: Quantum number name used in the basis (default: 'state').
            separator: CSV separator (default: ';').
            use_sparse: If True, epsilon is built as a SparseMatrix operator.
        """
        reader = CSVReader(separator=separator)
        states, values = reader.read_epsilon(csv_path)

        # Hilbert space basis: each state label is a single quantum number entry
        bases = mm.hilber_space_bases().from_qm_nums_list(
            [[s] for s in states],
            qm_nums_names=[qm_nums_name],
        )

        dim = bases.dim
        if len(values) != dim:
            raise ValueError(f"CSV row count ({len(values)}) does not match basis dim ({dim})")

        # Build diagonal state-energy operator (legacy name: epsilon)
        state_energy_op = reader.build_epsilon_operator(values=values, dim=dim, use_sparse=use_sparse)
        state_energy_op.subsys_name = subsystem_id

        node = qs.quantum_system_node(
            subsystem_id,
            base_states=bases,
            operators={
                "state_energy": state_energy_op,
                "epsilon": state_energy_op,  # legacy alias
            },
            dim=dim,
        )
        # Keep track of sparse preference on the node (consistent with other builders)
        node.use_sparse = use_sparse

        state_to_index = {s: i for i, s in enumerate(states)}
        return cls(node=node, state_to_index=state_to_index)

    @classmethod
    def from_state_energy_list(
        cls,
        states: List[str],
        values: Union[np.ndarray, Sequence[float]],
        *,
        subsystem_id: str = "electron_system",
        qm_nums_name: str = "state",
        use_sparse: bool = True,
    ) -> "multi_config_electron":
        """
        Build electron subsystem from parallel lists of state labels and energies (no file I/O).

        Same layout as :meth:`from_csv` after reading ``state`` / ``value`` columns.
        """
        values = np.asarray(values, dtype=float)
        reader = CSVReader(separator=";")
        bases = mm.hilber_space_bases().from_qm_nums_list(
            [[s] for s in states],
            qm_nums_names=[qm_nums_name],
        )
        dim = bases.dim
        if len(values) != dim:
            raise ValueError(f"len(values) ({len(values)}) != number of states ({dim})")
        state_energy_op = reader.build_epsilon_operator(
            values=values, dim=dim, use_sparse=use_sparse
        )
        state_energy_op.subsys_name = subsystem_id
        node = qs.quantum_system_node(
            subsystem_id,
            base_states=bases,
            operators={
                "state_energy": state_energy_op,
                "epsilon": state_energy_op,
            },
            dim=dim,
        )
        node.use_sparse = use_sparse
        state_to_index = {s: i for i, s in enumerate(states)}
        return cls(node=node, state_to_index=state_to_index)

    def get_state_energy_operator(self) -> mm.MatrixOperator:
        return self.node.operators["state_energy"]

    def get_epsilon_operator(self) -> mm.MatrixOperator:
        # legacy alias
        return self.get_state_energy_operator()

    def get_off_diag_coupling_operator(self, mode_num: int, *, operator_name: Optional[str] = None) -> mm.MatrixOperator:
        """
        Get an already-created electron-phonon coupling operator in the **electron subsystem view**.

        By default, coupling operators created by ``create_phonon_coupling_interactions`` are stored as:
            coupling_mode_{mode_num}
        (legacy alias: V_mode_{mode_num})
        """
        if operator_name is None:
            operator_name = f"coupling_mode_{mode_num}"
        if operator_name in self.node.operators:
            return self.node.operators[operator_name]

        # Fallback to legacy name
        legacy_name = f"V_mode_{mode_num}"
        if legacy_name in self.node.operators:
            return self.node.operators[legacy_name]

        if operator_name not in self.node.operators:
            raise ValueError(
                f"Coupling operator '{operator_name}' not found on electron node '{self.node.id}'. "
                f"Create it first via create_phonon_coupling_interactions(..., mode_num={mode_num})."
            )
        return self.node.operators[operator_name]

    def get_diag_coupling_operator(self, mode_num: int, *, operator_name: Optional[str] = None) -> mm.MatrixOperator:
        """
        Get an already-created tuning (diagonal) operator in the **electron subsystem view**.

        By default, operators created by ``create_non_coupling_kappa_matrix`` are stored as:
            tuning_mode_{mode_num}
        (legacy alias: kappa_mode_{mode_num})
        """
        if operator_name is None:
            operator_name = f"tuning_mode_{mode_num}"
        if operator_name in self.node.operators:
            return self.node.operators[operator_name]

        # Fallback to legacy name
        legacy_name = f"kappa_mode_{mode_num}"
        if legacy_name in self.node.operators:
            return self.node.operators[legacy_name]

        if operator_name not in self.node.operators:
            raise ValueError(
                f"Non-coupling operator '{operator_name}' not found on electron node '{self.node.id}'. "
                f"Create it first via create_non_coupling_kappa_matrix(..., mode_num={mode_num})."
            )
        return self.node.operators[operator_name]

    def get_mode_coupling_operator(
        self,
        mode_num: int,
        *,
        off_diag_operator_name: Optional[str] = None,
        diag_operator_name: Optional[str] = None,
    ) -> mm.MatrixOperator:
        """
        Return the **total coupling operator** for a given mode:

            V_total(mode) = V_off_diag(mode) + V_diag(mode)

        Both are returned in the **electron subsystem view** (not embedded).
        """
        V = self.get_off_diag_coupling_operator(mode_num, operator_name=off_diag_operator_name)
        K = self.get_diag_coupling_operator(mode_num, operator_name=diag_operator_name)
        return V + K

    def _state_labels_in_order(self) -> List[str]:
        """
        Return basis state labels in the row/column order used by the electron matrices.
        """
        # state_to_index is built in CSV order; preserve that order
        return list(self.state_to_index.keys())

    def write_operator_matrix_to_csv(
        self,
        op: mm.MatrixOperator,
        out_csv_path: str,
        *,
        separator: str = ";",
        float_format: str = "%.16g",
    ) -> None:
        """
        Write an operator matrix to CSV with labeled rows/columns (state labels).

        The output is a square table with header row/col being the state labels.
        """
        # Use the shared CSVWriter utility for consistent formatting across the project
        from jahn_teller_dynamics.io.file_io.csv_writer import CSVWriter

        if self.node.base_states is None:
            raise ValueError("multi_config_electron.node.base_states is required to write MatrixOperators")

        writer = CSVWriter(separator=separator, index=True)
        writer.write_matrix_operator(op, out_csv_path, basis=self.node.base_states)

    def write_epsilon_to_csv(
        self,
        out_csv_path: str,
        *,
        separator: str = ";",
        float_format: str = "%.16g",
    ) -> None:
        """Write the diagonal state-energy operator matrix to CSV (legacy name: epsilon)."""
        self.write_operator_matrix_to_csv(
            self.get_epsilon_operator(),
            out_csv_path,
            separator=separator,
            float_format=float_format,
        )

    def write_state_energy_to_csv(
        self,
        out_csv_path: str,
        *,
        separator: str = ";",
        float_format: str = "%.16g",
    ) -> None:
        """Write the diagonal state-energy operator matrix to CSV."""
        self.write_operator_matrix_to_csv(
            self.get_state_energy_operator(),
            out_csv_path,
            separator=separator,
            float_format=float_format,
        )

    def write_coupling_to_csv(
        self,
        coupling_csv_path: str,
        mode_num: int,
        out_csv_path: str,
        *,
        separator: str = ";",
        float_format: str = "%.16g",
        symmetric: bool = True,
    ) -> None:
        """
        Build and write the coupling matrix for a given vibrational mode.
        """
        op = self.create_phonon_coupling_interactions(
            coupling_csv_path,
            mode_num=mode_num,
            operator_name=None,
            separator=separator,
            symmetric=symmetric,
            add_to_node=False,
            use_sparse=True,
        )
        self.write_operator_matrix_to_csv(op, out_csv_path, separator=separator, float_format=float_format)

    def create_non_coupling_kappa_matrix(
        self,
        csv_path: str,
        mode_num: int,
        *,
        operator_name: Optional[str] = None,
        separator: str = ";",
        add_to_node: bool = True,
        use_sparse: Optional[bool] = None,
    ) -> mm.MatrixOperator:
        """
        Create the non-coupling (diagonal) kappa matrix for a selected vibrational mode.

        CSV format (semicolon separated by default):
            state;vibrational_mode;value
            11;q1;-0.00399
            ...

        For each state, the value is placed on the diagonal at the index determined
        by the state-to-index mapping created by ``from_csv``.
        """
        legacy_name: Optional[str] = None
        if operator_name is None:
            operator_name = f"tuning_mode_{mode_num}"
            legacy_name = f"kappa_mode_{mode_num}"

        if use_sparse is None:
            use_sparse = getattr(self.node, "use_sparse", True)

        reader = CSVReader(separator=separator)
        df = reader.read_kappa_rows(csv_path)
        dim = self.node.dim
        op = reader.build_kappa_operator(
            df=df,
            state_to_index=self.state_to_index,
            dim=dim,
            mode_num=mode_num,
            use_sparse=use_sparse,
        )

        op.subsys_name = self.node.id
        if add_to_node:
            self.node.operators[operator_name] = op
            if legacy_name is not None and legacy_name not in self.node.operators:
                self.node.operators[legacy_name] = op
        return op

    def write_kappa_to_csv(
        self,
        kappa_csv_path: str,
        mode_num: int,
        out_csv_path: str,
        *,
        separator: str = ";",
        float_format: str = "%.16g",
    ) -> None:
        """
        Build and write the diagonal kappa matrix for a given vibrational mode.
        """
        op = self.create_non_coupling_kappa_matrix(
            kappa_csv_path,
            mode_num=mode_num,
            operator_name=None,
            separator=separator,
            add_to_node=False,
            use_sparse=True,
        )
        self.write_operator_matrix_to_csv(op, out_csv_path, separator=separator, float_format=float_format)

    def create_phonon_coupling_interactions(
        self,
        csv_path: str,
        mode_num: int,
        *,
        operator_name: Optional[str] = None,
        separator: str = ";",
        symmetric: bool = True,
        add_to_node: bool = True,
        use_sparse: Optional[bool] = None,
    ) -> mm.MatrixOperator:
        """
        Create electron-phonon coupling interaction matrix for a selected vibrational mode.

        CSV format (semicolon separated by default):
            state_i;state_j;vibrational_mode;value
            11;12;q1;-2.6579e-05
            ...

        The ``state_i`` / ``state_j`` labels are mapped to row/column indices using
        the basis loaded by ``from_csv``.

        Args:
            csv_path: Path to coupling-coefficients CSV.
            mode_num: Vibrational mode index to construct (matches CSV ``vibrational_mode``).
            operator_name: Name to store the operator under. Defaults to ``f"coupling_mode_{mode_num}"``.
            separator: CSV delimiter (default: ';').
            symmetric: If True, also sets the transposed entry (j,i). Recommended for Hermitian couplings.
            add_to_node: If True, stores the created operator in ``self.node.operators``.
            use_sparse: If True, build as SparseMatrix; if None, uses ``self.node.use_sparse`` if present else True.

        Returns:
            MatrixOperator for the coupling matrix (sparse by default).
        """
        legacy_name: Optional[str] = None
        if operator_name is None:
            operator_name = f"coupling_mode_{mode_num}"
            legacy_name = f"V_mode_{mode_num}"

        if use_sparse is None:
            use_sparse = getattr(self.node, "use_sparse", True)

        reader = CSVReader(separator=separator)
        df = reader.read_coupling_rows(csv_path)
        dim = self.node.dim
        op = reader.build_coupling_operator(
            df=df,
            state_to_index=self.state_to_index,
            dim=dim,
            mode_num=mode_num,
            symmetric=symmetric,
            use_sparse=use_sparse,
        )

        op.subsys_name = self.node.id
        if add_to_node:
            self.node.operators[operator_name] = op
            if legacy_name is not None and legacy_name not in self.node.operators:
                self.node.operators[legacy_name] = op
        return op
