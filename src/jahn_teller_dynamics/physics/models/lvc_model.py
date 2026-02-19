"""
LVC model system composition.

IMPORTANT DESIGN NOTE
---------------------
`LVC_model` is implemented as a `quantum_system_tree` (Exe_tree-style), not as a
single node wrapper. Its root node has two children:
  - `electron_system` (configured by `multi_config_electron`)
  - `phonon_system` (a `MultiModePhononSystem`)
"""

from __future__ import annotations

from typing import Optional, Union, Literal, Sequence

import jahn_teller_dynamics.physics.quantum_system as qs
from jahn_teller_dynamics.physics.models.multi_config_electron import multi_config_electron
from jahn_teller_dynamics.physics.models.system_builder import (
    MultiModePhononSystem,
    build_phonon_system,
    build_phonon_system_from_csv,
)


class LVC_model(qs.quantum_system_tree):
    """
    Composite LVC system tree:

        LVC_model(root)
        ├── electron_system  (multi_config_electron.node)
        └── phonon_system    (MultiModePhononSystem with mode_1, mode_2, ...)
    """

    def __init__(
        self,
        root_node: qs.quantum_system_node,
        *,
        electron: multi_config_electron,
        phonons: MultiModePhononSystem,
    ) -> None:
        super().__init__(root_node)
        self.electron = electron
        self.phonons = phonons

    # Backwards-compatible alias used by earlier scaffolding/tests
    @property
    def node(self) -> qs.quantum_system_node:  # pragma: no cover
        return self.root_node

    @staticmethod
    def _set_use_sparse_recursive(node: qs.quantum_system_node, value: bool) -> None:
        node.use_sparse = value
        for child in node.children:
            LVC_model._set_use_sparse_recursive(child, value)  # type: ignore[arg-type]

    @classmethod
    def build(
        cls,
        electron: multi_config_electron,
        phonons: MultiModePhononSystem,
        *,
        system_id: str = "LVC_model",
        use_sparse: Optional[bool] = None,
    ) -> "LVC_model":
        if electron.node is None:
            raise ValueError("electron.node is required")

        root = qs.quantum_system_node(system_id, children=[electron.node, phonons])

        if use_sparse is None:
            use_sparse = bool(getattr(electron.node, "use_sparse", False) or getattr(phonons, "use_sparse", False))
        cls._set_use_sparse_recursive(root, use_sparse)

        return cls(root, electron=electron, phonons=phonons)

    @classmethod
    def from_csvs(
        cls,
        *,
        # New physical naming (preferred)
        state_energy_csv_path: Optional[str] = None,
        coupling_csv_path: Optional[str] = None,
        tuning_csv_path: Optional[str] = None,
        # Legacy naming (kept for backwards compatibility)
        epsilon_csv_path: Optional[str] = None,
        lambda_csv_path: Optional[str] = None,
        kappa_csv_path: Optional[str] = None,
        order: int,
        modes: Optional[Sequence[float]] = None,
        modes_csv_path: Optional[str] = None,
        separator: str = ";",
        system_id: str = "LVC_model",
        electron_system_id: str = "electron_system",
        phonon_system_id: str = "phonon_system",
        build_all_mode_couplings: bool = True,
        build_all_mode_kappas: bool = True,
        symmetric_couplings: bool = True,
        use_sparse: bool = True,
        dimensionless_coordinates: bool = True,
        null_point_vib: bool = True,
    ) -> "LVC_model":
        # Normalize names: prefer physical naming, fallback to legacy
        if state_energy_csv_path is None:
            state_energy_csv_path = epsilon_csv_path
        if state_energy_csv_path is None:
            raise ValueError("state_energy_csv_path (or legacy epsilon_csv_path) is required")

        if coupling_csv_path is None:
            coupling_csv_path = lambda_csv_path
        if tuning_csv_path is None:
            tuning_csv_path = kappa_csv_path

        electron = multi_config_electron.from_csv(
            state_energy_csv_path,
            subsystem_id=electron_system_id,
            separator=separator,
            use_sparse=use_sparse,
        )

        if modes_csv_path is not None:
            phonons = build_phonon_system_from_csv(
                modes_csv_path,
                order=order,
                use_sparse=use_sparse,
                phonon_system_id=phonon_system_id,
                separator=separator,
                dimensionless_coordinates=dimensionless_coordinates,
                null_point_vib=null_point_vib,
            )
            num_modes = len(phonons.modes)
        else:
            if modes is None:
                raise ValueError("Either modes_csv_path or modes must be provided")
            phonons = build_phonon_system(
                modes=modes,
                order=order,
                use_sparse=use_sparse,
                phonon_system_id=phonon_system_id,
                dimensionless_coordinates=dimensionless_coordinates,
                null_point_vib=null_point_vib,
            )
            num_modes = len(modes)

        model = cls.build(electron=electron, phonons=phonons, system_id=system_id, use_sparse=use_sparse)

        if coupling_csv_path is not None and build_all_mode_couplings:
            for i in range(1, num_modes + 1):
                electron.create_phonon_coupling_interactions(
                    coupling_csv_path,
                    mode_num=i,
                    separator=separator,
                    symmetric=symmetric_couplings,
                    add_to_node=True,
                    use_sparse=use_sparse,
                )

        if tuning_csv_path is not None and build_all_mode_kappas:
            for i in range(1, num_modes + 1):
                electron.create_non_coupling_kappa_matrix(
                    tuning_csv_path,
                    mode_num=i,
                    separator=separator,
                    add_to_node=True,
                    use_sparse=use_sparse,
                )

        return model

    # ---------------------------------------------------------------------
    # Convenience wrappers (subsystem views)
    # ---------------------------------------------------------------------

    def get_state_energy_operator(self):
        """Electron-subsystem view state_energy (not embedded)."""
        return self.electron.get_state_energy_operator()

    def get_epsilon_operator(self):
        """Electron-subsystem view epsilon (legacy alias for state_energy)."""
        return self.electron.get_epsilon_operator()

    def get_off_diag_coupling_operator(self, mode_num: int, *, operator_name: Optional[str] = None):
        """Electron-subsystem view off-diagonal coupling (not embedded)."""
        return self.electron.get_off_diag_coupling_operator(mode_num, operator_name=operator_name)

    def get_diag_coupling_operator(self, mode_num: int, *, operator_name: Optional[str] = None):
        """Electron-subsystem view diagonal tuning (legacy: kappa) coupling (not embedded)."""
        return self.electron.get_diag_coupling_operator(mode_num, operator_name=operator_name)

    def get_mode_coupling_operator(
        self,
        mode_num: int,
        *,
        view: Literal["tree", "electron"] = "tree",
        off_diag_operator_name: Optional[str] = None,
        diag_operator_name: Optional[str] = None,
    ):
        """
        Total coupling operator for a mode:

            V_total(mode) = V_off_diag(mode) + V_diag(mode)

        - view='electron': electron-subsystem view (dim = dim(electron))
        - view='tree': embedded into full LVC tree view (dim = dim(electron ⊗ phonons))
        """
        if view == "electron":
            return self.electron.get_mode_coupling_operator(
                mode_num,
                off_diag_operator_name=off_diag_operator_name,
                diag_operator_name=diag_operator_name,
            )

        off_name = off_diag_operator_name or f"coupling_mode_{mode_num}"
        diag_name = diag_operator_name or f"tuning_mode_{mode_num}"
        V_off = self.create_operator(off_name, operator_sys=self.electron.node.id)
        V_diag = self.create_operator(diag_name, operator_sys=self.electron.node.id)
        return V_off + V_diag

    def get_pos_i_operator(
        self,
        mode: Union[int, str] = 1,
        *,
        coord: Optional[str] = None,
        view: Literal["subsystem", "full"] = "full",
    ):
        """
        Phonon position operator for a coordinate in subsystem/full view.

        If coord is None, uses the first coordinate of the mode (backward compat).
        """
        if coord is None:
            labels = self.phonons.get_mode_coordinate_labels(mode)
            coord = labels[0]
        return self.phonons.get_position_operator(mode=mode, coord=coord, view=view)

    def get_energy_i_operator(
        self,
        mode: Union[int, str] = 1,
        *,
        view: Literal["subsystem", "full"] = "full",
    ):
        """Phonon energy operator K_i in subsystem/full phonon-system view."""
        if view == "subsystem":
            return self.phonons.get_operator_subsystem("K", mode=mode)
        return self.phonons.get_operator_full("K", mode=mode)

    def get_pos_ii_operator(
        self,
        mode: Union[int, str] = 1,
        *,
        view: Literal["subsystem", "full"] = "full",
    ):
        """Phonon squared-position operator XX_i in subsystem/full phonon-system view."""
        return self.phonons.get_position_squared_operator(mode=mode, view=view)

    # ---------------------------------------------------------------------
    # Convenience wrappers (full system tree view)
    # ---------------------------------------------------------------------

    def mode_energy_i_operator(self, mode: Union[int, str] = 1):
        """K_i embedded in the full LVC tree view."""
        mode_id = self.phonons._mode_id(mode)  # type: ignore[attr-defined]
        return self.create_operator("K", operator_sys=mode_id)

    def get_electron_energy_operator(self):
        """state_energy embedded in the full LVC tree view (legacy method name)."""
        return self.get_electron_state_energy_operator()

    def get_electron_state_energy_operator(self):
        """state_energy embedded in the full LVC tree view."""
        return self.create_operator("state_energy", operator_sys=self.electron.node.id)

