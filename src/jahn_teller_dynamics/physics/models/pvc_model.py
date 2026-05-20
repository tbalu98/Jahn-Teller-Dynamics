"""
PVC (polynomial vibronic coupling) quantum system tree.

Like :class:`LVC_model`, this is a :class:`quantum_system_tree` with:

- ``orbital_system`` (alias ``electron_system``): :class:`multi_config_electron`
- ``phonon_system``: constrained or tensor-product multimode phonons

Polynomial coupling data live in a separate table (CSV). Build the composite Hamiltonian with:

- Row-by-row: :func:`~jahn_teller_dynamics.physics.hamiltonians.djt_polynomial_hamiltonian.create_pvc_hamiltonian`
- Grouped by expression: :func:`~jahn_teller_dynamics.physics.hamiltonians.djt_polynomial_hamiltonian_grouped.create_pvc_hamiltonian_grouped`
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Sequence, Union

import jahn_teller_dynamics.math.matrix_mechanics as mm

import numpy as np

import jahn_teller_dynamics.physics.quantum_system as qs
from jahn_teller_dynamics.io.file_io.csv_reader import CSVReader
from jahn_teller_dynamics.physics.hamiltonians.djt_polynomial_hamiltonian import DJTCouplingRow
from jahn_teller_dynamics.physics.models.constrained_multimode_phonon import (
    MultiModeConstrainedPhononSystem,
)
from jahn_teller_dynamics.physics.models.multi_config_electron import multi_config_electron
from jahn_teller_dynamics.physics.models.system_builder import (
    MultiModePhononSystem,
    build_phonon_system_constrained,
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
        electron: multi_config_electron,
        phonons: Union[MultiModeConstrainedPhononSystem, MultiModePhononSystem],
        coupling_rows: Sequence[DJTCouplingRow],
        electron_energies: np.ndarray,
    ) -> None:
        super().__init__(root_node)
        self.electron = electron
        self.phonons = phonons
        self.coupling_rows = list(coupling_rows)
        self.electron_energies = electron_energies
        self.orbital_coupling_operators: Dict[str, mm.MatrixOperator] = {}
        self.orbital_coupling_operator_names: Dict[str, str] = {}

    @property
    def orbital(self) -> multi_config_electron:
        """Orbital subsystem (same node as ``electron``; extension point for spin)."""
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
        electron: multi_config_electron,
        phonons: Union[MultiModeConstrainedPhononSystem, MultiModePhononSystem],
        coupling_rows: Sequence[DJTCouplingRow],
        electron_energies: Union[np.ndarray, Sequence[float]],
        *,
        system_id: str = "PVC_model",
        use_sparse: Optional[bool] = None,
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
        return cls(
            root,
            electron=electron,
            phonons=phonons,
            coupling_rows=coupling_rows,
            electron_energies=e_arr,
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
    ) -> "PVC_model":
        reader = CSVReader()
        states, values = reader.read_diagonal_state_energies(electron_energies_csv_path)
        # Row order in electron_energies.csv defines internal 1-based electronic IDs.
        state_to_el_id = {str(s).strip(): i + 1 for i, s in enumerate(states)}
        t0 = time.perf_counter()
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

        t_ph = time.perf_counter()
        if mp > 0:
            if mt > 0:
                raise ValueError(
                    "PVC_model.from_csvs: set only one of maximum_quanta_per_mode > 0 "
                    "(MultiModePhononSystem) or maximum_total_phonon_quanta > 0 "
                    "(MultiModeConstrainedPhononSystem)."
                )
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
        t_cpl = time.perf_counter()
        if build_log is not None:
            build_log("PVC: reading coupling expression table from CSV")
        df = reader.read_pvc_coupling_rows(coupling_csv_path)
        coupling_rows: List[DJTCouplingRow] = []

        def _resolve_el_state_token(raw: object) -> int:
            token = str(raw).strip()
            if token == "":
                raise ValueError("Empty electron-state token in coupling CSV")
            # Prefer explicit label mapping from electron_energies.csv row order.
            if token in state_to_el_id:
                return int(state_to_el_id[token])
            # Backward compatibility: numeric IDs in coupling CSV.
            try:
                idx = int(token)
            except ValueError as e:
                raise ValueError(
                    f"Unknown electronic state label {token!r} in coupling CSV. "
                    "Use labels defined in electron_energies.csv or numeric IDs."
                ) from e
            if idx < 1 or idx > len(states):
                raise ValueError(
                    f"Electron state index {idx} out of range for {len(states)} states."
                )
            return idx

        for _, row in df.iterrows():
            el1 = _resolve_el_state_token(row["el_state_1"])
            el2 = _resolve_el_state_token(row["el_state_2"])
            coeff = float(row["coeff"])
            if el1 == el2:
                coeff *= float(tune_tuning)
            else:
                coeff *= float(tune_coupling)
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
        )
