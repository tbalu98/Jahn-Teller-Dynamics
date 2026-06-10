"""
PVC (polynomial vibronic coupling) quantum system tree.

Like :class:`LVC_model`, this is a :class:`quantum_system_tree` with:

- ``orbital_system`` (alias ``electron_system``): :class:`multi_config_electron` or
  :class:`multi_config_orbital_spin_electron` (CSV with ``spin`` column)
- ``phonon_system``: constrained or tensor-product multimode phonons

Polynomial coupling data live in a separate table (CSV). Build the composite Hamiltonian with:

- Row-by-row: :func:`~jahn_teller_dynamics.physics.hamiltonians.djt_polynomial_hamiltonian.create_pvc_hamiltonian`
- Grouped by expression: :func:`~jahn_teller_dynamics.physics.hamiltonians.djt_polynomial_hamiltonian_grouped.create_pvc_hamiltonian_grouped`
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Sequence, Union

import jahn_teller_dynamics.math_utils.matrix_mechanics as mm

import numpy as np

import jahn_teller_dynamics.physics.quantum_system as qs
from jahn_teller_dynamics.io.electron_state_resolution import (
    build_electron_state_to_id,
    resolve_electron_state_label,
)
from jahn_teller_dynamics.io.file_io.csv_reader import CSVReader
from jahn_teller_dynamics.physics.hamiltonians.djt_polynomial_hamiltonian import DJTCouplingRow
from jahn_teller_dynamics.physics.models.constrained_multimode_phonon import (
    MultiModeConstrainedPhononSystem,
)
from jahn_teller_dynamics.physics.models.multi_config_electron import multi_config_electron
from jahn_teller_dynamics.physics.models.orbital_spin_electron import (
    OrbitalSpinLayout,
    csv_has_spin_column,
    load_spin_orbital_operator_from_csv,
    multi_config_orbital_spin_electron,
)
from jahn_teller_dynamics.physics.models.system_builder import (
    MultiModePhononSystem,
    build_phonon_system_constrained,
    build_phonon_system_energy_cutoff,
)


class PVC_model(qs.quantum_system_tree):
    """
    Composite PVC system tree:

        PVC_model(root)
        ├── orbital_system   (multi_config_electron.node; alias electron_system)
        └── phonon_system    (constrained or tensor-product multimode phonons)

    Grouped coupling builders attach one :math:`N \\times N` operator per CSV expression
    on the orbital node (``coupling_<expression>`` keys) for inspection and spin extensions.
    """

    def __init__(
        self,
        root_node: qs.quantum_system_node,
        *,
        electron: Union[multi_config_electron, multi_config_orbital_spin_electron],
        phonons: Union[MultiModeConstrainedPhononSystem, MultiModePhononSystem],
        coupling_rows: Sequence[DJTCouplingRow],
        electron_energies: np.ndarray,
        orbital_spin_layout: Optional[OrbitalSpinLayout] = None,
        spin_orbital_operator: Optional[mm.MatrixOperator] = None,
    ) -> None:
        super().__init__(root_node)
        self.electron = electron
        self.phonons = phonons
        self.coupling_rows = list(coupling_rows)
        self.electron_energies = electron_energies
        self.orbital_spin_layout = orbital_spin_layout
        self.spin_orbital_operator = spin_orbital_operator
        self.orbital_coupling_operators: Dict[str, mm.MatrixOperator] = {}
        self.orbital_coupling_operator_names: Dict[str, str] = {}

    @property
    def orbital(self) -> Union[multi_config_electron, multi_config_orbital_spin_electron]:
        """Orbital subsystem (same node as ``electron``)."""
        return self.electron

    @property
    def node(self) -> qs.quantum_system_node:  # pragma: no cover
        return self.root_node

    @staticmethod
    def _set_use_sparse_recursive(node: qs.quantum_system_node, value: bool) -> None:
        node.use_sparse = value
        for child in node.children:
            PVC_model._set_use_sparse_recursive(child, value)  # type: ignore[arg-type]

    @classmethod
    def build(
        cls,
        electron: Union[multi_config_electron, multi_config_orbital_spin_electron],
        phonons: Union[MultiModeConstrainedPhononSystem, MultiModePhononSystem],
        coupling_rows: Sequence[DJTCouplingRow],
        electron_energies: Union[np.ndarray, Sequence[float]],
        *,
        system_id: str = "PVC_model",
        use_sparse: Optional[bool] = None,
        orbital_spin_layout: Optional[OrbitalSpinLayout] = None,
        spin_orbital_operator: Optional[mm.MatrixOperator] = None,
    ) -> "PVC_model":
        if electron.node is None:
            raise ValueError("electron.node is required")
        if not isinstance(
            phonons, (MultiModeConstrainedPhononSystem, MultiModePhononSystem)
        ):
            raise TypeError(
                "PVC_model requires MultiModeConstrainedPhononSystem or MultiModePhononSystem for phonons"
            )

        root = qs.quantum_system_node(system_id, children=[electron.node, phonons])

        if use_sparse is None:
            use_sparse = bool(
                getattr(electron.node, "use_sparse", False) or getattr(phonons, "use_sparse", False)
            )
        cls._set_use_sparse_recursive(root, use_sparse)

        e_arr = np.asarray(electron_energies, dtype=float)
        if orbital_spin_layout is None and isinstance(
            electron, multi_config_orbital_spin_electron
        ):
            orbital_spin_layout = electron.layout
        return cls(
            root,
            electron=electron,
            phonons=phonons,
            coupling_rows=coupling_rows,
            electron_energies=e_arr,
            orbital_spin_layout=orbital_spin_layout,
            spin_orbital_operator=spin_orbital_operator,
        )

    @classmethod
    def from_csvs(
        cls,
        *,
        electron_energies_csv_path: str,
        modes_csv_path: str,
        coupling_csv_path: str,
        order: int = 2,
        maximum_quanta_per_mode: Optional[int] = None,
        maximum_total_phonon_quanta: Optional[int] = None,
        phonon_encut: Optional[float] = None,
        use_maximum_quanta_from_modes_csv: bool = False,
        system_id: str = "PVC_model",
        electron_system_id: str = "orbital_system",
        phonon_system_id: str = "phonon_system",
        use_sparse: bool = True,
        dimensionless_coordinates: bool = True,
        null_point_vib: bool = True,
        mode_numbers: Optional[Sequence[int]] = None,
        build_log: Optional[Callable[[str], None]] = None,
        exp_approximation_order: Optional[int] = None,
        tune_tuning: float = 1.0,
        tune_coupling: float = 1.0,
        spin_orbital_operator_csv_path: str = "",
    ) -> "PVC_model":
        reader = CSVReader()
        orbital_spin_layout: Optional[OrbitalSpinLayout] = None
        t0 = time.perf_counter()
        if csv_has_spin_column(electron_energies_csv_path):
            if build_log is not None:
                build_log(
                    "PVC: building orbital–spin electronic basis (el_state, energy, spin CSV)"
                )
            table = reader.read_orbital_spin_electron_table(electron_energies_csv_path)
            electron = multi_config_orbital_spin_electron.from_table(
                table,
                subsystem_id=electron_system_id,
                use_sparse=use_sparse,
            )
            orbital_spin_layout = electron.layout
            values = table.energies
            states = list(table.orbital_labels)
            state_to_el_id = electron.orbital_label_to_id
            if build_log is not None:
                build_log(
                    f"PVC: orbital–spin basis ready — n_orbitals = {orbital_spin_layout.n_orbitals}, "
                    f"dim = {orbital_spin_layout.dim} "
                    f"(wall time {time.perf_counter() - t0:.4f} s)"
                )
        else:
            states, values = reader.read_diagonal_state_energies(electron_energies_csv_path)
            state_to_el_id = build_electron_state_to_id(states)
            if build_log is not None:
                build_log("PVC: building electronic state basis (from CSV diagonal energies)")
            electron = multi_config_electron.from_state_energy_list(
                list(states),
                values,
                subsystem_id=electron_system_id,
                use_sparse=use_sparse,
            )
            if build_log is not None:
                build_log(
                    f"PVC: electronic basis ready — n_states = {electron.node.dim} "
                    f"(wall time {time.perf_counter() - t0:.4f} s)"
                )
        t_modes = time.perf_counter()
        if build_log is not None:
            build_log("PVC: reading mode frequencies and selecting modes from CSV")
        modes_table = reader.read_modes_flexible(
            modes_csv_path,
            mode_numbers=list(mode_numbers) if mode_numbers is not None else None,
        )
        if build_log is not None:
            build_log(
                f"PVC: using {len(modes_table.omegas)} phonon mode(s) "
                f"(wall time {time.perf_counter() - t_modes:.4f} s)"
            )
        mp = maximum_quanta_per_mode if maximum_quanta_per_mode is not None else 0
        mt = maximum_total_phonon_quanta if maximum_total_phonon_quanta is not None else 0
        use_per_mode_csv = bool(use_maximum_quanta_from_modes_csv)

        truncation_flags = sum(
            1
            for cond in (
                phonon_encut is not None,
                mp > 0,
                mt > 0,
                use_per_mode_csv,
            )
            if cond
        )
        if truncation_flags > 1:
            raise ValueError(
                "PVC_model.from_csvs: set at most one of phonon_encut, "
                "maximum_quanta_per_mode, maximum_total_phonon_quanta, or "
                "use_maximum_quanta_from_modes_csv."
            )

        t_ph = time.perf_counter()
        if phonon_encut is not None:
            if float(phonon_encut) < 0:
                raise ValueError("PVC_model.from_csvs: phonon_encut must be non-negative")
            if build_log is not None:
                build_log(
                    "PVC: building constrained phonon subsystem "
                    f"(sum_i n_i e_i <= {float(phonon_encut):g})"
                )
            phonons = build_phonon_system_energy_cutoff(
                modes=modes_table.omegas,
                phonon_encut=float(phonon_encut),
                use_sparse=use_sparse,
                phonon_system_id=phonon_system_id,
                dimensionless_coordinates=dimensionless_coordinates,
                null_point_vib=null_point_vib,
                mode_names=modes_table.labels,
                build_log=build_log,
                exp_approximation_order=exp_approximation_order,
            )
        elif use_per_mode_csv:
            if modes_table.maximum_quanta is None:
                raise ValueError(
                    "PVC_model.from_csvs: use_maximum_quanta_from_modes_csv=True "
                    "requires a 'maximum_quanta' column in the modes CSV."
                )
            per_mode_orders = list(modes_table.maximum_quanta)
            if any(n < 0 for n in per_mode_orders):
                raise ValueError(
                    "PVC_model.from_csvs: 'maximum_quanta' entries must be non-negative."
                )
            mode_cfgs = [(w, [lab]) for w, lab in zip(modes_table.omegas, modes_table.labels)]
            if build_log is not None:
                pretty = ", ".join(
                    f"{lab}={n}" for lab, n in zip(modes_table.labels, per_mode_orders)
                )
                build_log(
                    "PVC: building tensor-product multimode phonon subsystem "
                    f"(per-mode cutoffs from modes.csv: {pretty})"
                )
            phonons = MultiModePhononSystem(
                modes=mode_cfgs,
                order=per_mode_orders,
                use_sparse=use_sparse,
                phonon_system_id=phonon_system_id,
                dimensionless_coordinates=dimensionless_coordinates,
                null_point_vib=null_point_vib,
                exp_approximation_order=exp_approximation_order,
            )
        elif mp > 0:
            mode_cfgs = [(w, [lab]) for w, lab in zip(modes_table.omegas, modes_table.labels)]
            if build_log is not None:
                build_log(
                    "PVC: building tensor-product multimode phonon subsystem "
                    f"(per-mode cutoff n_i <= {mp})"
                )
            phonons = MultiModePhononSystem(
                modes=mode_cfgs,
                order=mp,
                use_sparse=use_sparse,
                phonon_system_id=phonon_system_id,
                dimensionless_coordinates=dimensionless_coordinates,
                null_point_vib=null_point_vib,
                exp_approximation_order=exp_approximation_order,
            )
        else:
            constrained_n = mt if mt > 0 else order
            if constrained_n <= 0:
                raise ValueError(
                    "PVC_model.from_csvs: maximum_total_phonon_quanta / order must be positive "
                    "when not using maximum_quanta_per_mode."
                )
            if build_log is not None:
                build_log(
                    "PVC: building constrained phonon subsystem "
                    f"(sum_i n_i <= {constrained_n})"
                )
            phonons = build_phonon_system_constrained(
                modes=modes_table.omegas,
                order=constrained_n,
                use_sparse=use_sparse,
                phonon_system_id=phonon_system_id,
                dimensionless_coordinates=dimensionless_coordinates,
                null_point_vib=null_point_vib,
                mode_names=modes_table.labels,
                build_log=build_log,
                exp_approximation_order=exp_approximation_order,
            )
        if build_log is not None:
            build_log(
                f"PVC: phonon subsystem build finished "
                f"(wall time {time.perf_counter() - t_ph:.4f} s)"
            )
        spin_orbital_op: Optional[mm.MatrixOperator] = None
        if spin_orbital_operator_csv_path:
            dim_spin = int(electron.node.dim)
            if build_log is not None:
                build_log(
                    f"PVC: loading spin–orbital operator ({dim_spin}×{dim_spin}) from "
                    f"{spin_orbital_operator_csv_path}"
                )
            spin_orbital_op = load_spin_orbital_operator_from_csv(
                spin_orbital_operator_csv_path,
                dim_spin,
                use_sparse=use_sparse,
            )
            electron.node.operators["spin_orbital"] = spin_orbital_op

        t_cpl = time.perf_counter()
        if build_log is not None:
            build_log("PVC: reading coupling expression table from CSV")
        df = reader.read_pvc_coupling_rows(coupling_csv_path)
        coupling_rows: List[DJTCouplingRow] = []

        n_orb = (
            orbital_spin_layout.n_orbitals
            if orbital_spin_layout is not None
            else len(states)
        )

        for _, row in df.iterrows():
            el1 = resolve_electron_state_label(
                row["el_state_1"], state_to_el_id, n_orbitals=n_orb
            )
            el2 = resolve_electron_state_label(
                row["el_state_2"], state_to_el_id, n_orbitals=n_orb
            )
            coeff = complex(row["coeff"])
            if el1 == el2:
                coeff *= complex(tune_tuning)
            else:
                coeff *= complex(tune_coupling)
            coupling_rows.append(
                DJTCouplingRow(
                    el1,
                    el2,
                    str(row["polinom"]).strip(),
                    coeff,
                )
            )
        if build_log is not None:
            build_log(
                f"PVC: loaded {len(coupling_rows)} coupling row(s) "
                f"({time.perf_counter() - t_cpl:.4f} s); assembling PVC model tree"
            )
        return cls.build(
            electron=electron,
            phonons=phonons,
            coupling_rows=coupling_rows,
            electron_energies=values,
            system_id=system_id,
            use_sparse=use_sparse,
            orbital_spin_layout=orbital_spin_layout,
            spin_orbital_operator=spin_orbital_op,
        )
