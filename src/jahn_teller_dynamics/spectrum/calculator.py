"""
SpectrumCalculator - Computes LVC absorption spectrum from eigenvectors and dipole matrices.

Receives resolved paths and settings; has no knowledge of .cfg files.
Configuration is set by SpectrumBuilder.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jahn_teller_dynamics.io.file_io.npz_reader import load_lvc_npz
from jahn_teller_dynamics.io.utils.file_utils import create_directory
from jahn_teller_dynamics.math.maths import Lorentzian


# 1 Hartree = 27.211386245988 eV (CODATA 2018)
HARTREE_TO_EV = 27.211386245988


def _read_dipole_matrix_csv(path: str | Path, sep: str = ";") -> np.ndarray:
    """
    Read dipole matrix from CSV. Expects d×d matrix.
    Diagonal elements are set to zero.
    """
    path = Path(path)
    df = pd.read_csv(path, sep=sep, index_col=0)
    arr = np.zeros((len(df), len(df.columns)), dtype=complex)
    for i in range(len(df)):
        for j in range(len(df.columns)):
            val = df.iloc[i, j]
            try:
                arr[i, j] = complex(val)
            except (TypeError, ValueError):
                arr[i, j] = np.nan
    if arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Dipole matrix must be square, got shape {arr.shape}")
    #np.fill_diagonal(arr, 0.0)
    return arr


def _energy_to_eV(
    delta_energies_raw: List[float],
    energy_unit: str,
) -> np.ndarray:
    """Convert energy differences to eV."""
    _unit = (energy_unit or "ev").lower()
    arr = np.array(delta_energies_raw, dtype=float)
    if _unit in ("ev",):
        return arr
    if _unit in ("mev",):
        return arr / 1000.0
    if _unit in ("cm-1", "inv_cm"):
        return arr * 0.00012398426
    if _unit in ("hartree", "ha", "eh"):
        return arr * HARTREE_TO_EV
    return arr


class SpectrumCalculator:
    """
    Calculator for LVC absorption spectrum.

    Receives paths and settings set by SpectrumBuilder. Performs the spectrum
    calculation: loads NPZ and dipole CSVs, computes intensities, plots and saves.
    """

    def __init__(
        self,
        npz_path: str = "",
        dipole_path: str = "",
        dipole_x_path: str = "",
        dipole_y_path: str = "",
        dipole_z_path: str = "",
        output_folder: str = "",
        output_prefix: str = "spectrum",
        out_path: str = "",
        separator: str = ";",
        energy_unit: str = "ev",
        use_dipole_xyz: bool = False,
        spectrum_range_min: float = 1.0,
        spectrum_range_max: float = 2.0,
        smearing: Optional[Lorentzian] = None,
        run_dir: Optional[Path] = None,
    ):
        """
        Args:
            npz_path: Absolute path to eigenvectors.npz.
            dipole_path: Path to single dipole matrix CSV (when use_dipole_xyz is False).
            dipole_x_path, dipole_y_path, dipole_z_path: Paths to dipole X/Y/Z CSVs.
            output_folder: Base folder for output (absolute or relative to repo).
            output_prefix: Prefix for output files.
            out_path: Output filename override.
            separator: CSV separator for dipole files.
            energy_unit: Energy unit of NPZ eigenvalues (ev, mev, cm-1, hartree).
            use_dipole_xyz: If True, use dipole_x/y/z; otherwise use dipole_path.
            spectrum_range_min: Minimum energy (eV) for spectrum x-axis.
            spectrum_range_max: Maximum energy (eV) for spectrum x-axis.
            smearing: Optional Lorentzian smearing instance; None if no smearing.
        """
        self.npz_path = npz_path
        self.dipole_path = dipole_path
        self.dipole_x_path = dipole_x_path
        self.dipole_y_path = dipole_y_path
        self.dipole_z_path = dipole_z_path
        self.output_folder = output_folder
        self.output_prefix = output_prefix
        self.out_path = out_path
        self.separator = separator or ";"
        self.energy_unit = (energy_unit or "ev").lower()
        self.use_dipole_xyz = use_dipole_xyz
        self.spectrum_range_min = spectrum_range_min
        self.spectrum_range_max = spectrum_range_max
        self.smearing = smearing
        self._run_dir = Path(run_dir) if run_dir is not None else Path.cwd()

    def _resolve_output_folder(self) -> Path:
        """Resolved output folder (all outputs go here). Paths relative to run_dir (cwd)."""
        rf = self.output_folder
        if rf:
            p = Path(rf).expanduser()
            if not p.is_absolute():
                p = self._run_dir / p
            return p.resolve()
        return (self._run_dir / "results").resolve()

    def _resolve_output_path(self) -> Path:
        """Resolved output PNG path."""
        res_folder = self._resolve_output_folder()
        if self.out_path:
            out = Path(self.out_path)
            rel = out.name if out.is_absolute() else self.out_path
            return res_folder / rel
        return res_folder / f"{self.output_prefix}.png"

    def _load_dipoles(self) -> List[np.ndarray]:
        """Load dipole matrix/matrices from CSV(s)."""
        sep = self.separator or ";"
        if self.use_dipole_xyz:
            paths = [
                Path(self.dipole_x_path),
                Path(self.dipole_y_path),
                Path(self.dipole_z_path),
            ]
            dipoles = [_read_dipole_matrix_csv(p, sep=sep) for p in paths]
            d_el = dipoles[0].shape[0]
            for j, d in enumerate(dipoles):
                if d.shape[0] != d_el or d.shape[1] != d_el:
                    raise ValueError(
                        f"Dipole matrix {j} has shape {d.shape}, expected ({d_el}, {d_el})"
                    )
            return dipoles
        path = Path(self.dipole_path)
        return [_read_dipole_matrix_csv(path, sep=sep)]

    def calculate(self, show_plot: bool = True) -> int:
        """
        Perform the spectrum calculation: load data, compute intensities, plot, save.

        Returns:
            0 on success, 1 on error (e.g. file not found).
        """
        if not self.npz_path:
            print("Error: npz path not configured")
            return 1
        if not self.use_dipole_xyz and not self.dipole_path:
            print("Error: dipole or dipole_x/y/z paths not configured")
            return 1
        if self.use_dipole_xyz and not all(
            [self.dipole_x_path, self.dipole_y_path, self.dipole_z_path]
        ):
            print("Error: all of dipole_x, dipole_y, dipole_z must be set")
            return 1

        npz_path = Path(self.npz_path)
        if not npz_path.exists():
            print(f"NPZ not found: {npz_path}")
            return 1

        if self.use_dipole_xyz:
            for p in [self.dipole_x_path, self.dipole_y_path, self.dipole_z_path]:
                if not Path(p).exists():
                    print(f"Dipole CSV not found: {p}")
                    return 1
        elif not Path(self.dipole_path).exists():
            print(f"Dipole CSV not found: {self.dipole_path}")
            return 1

        dipoles = self._load_dipoles()

        d_el = dipoles[0].shape[0]
        data = load_lvc_npz(npz_path)
        eig_vecs = data["eigenvectors"]
        eig_vals = data["eigenvalues"]
        dim = data["dim"]

        if dim % d_el != 0:
            raise ValueError(
                f"Full dim ({dim}) must be divisible by electron dim ({d_el})."
            )
        n_ph = dim // d_el

        M_fulls = [
            np.kron(d, np.eye(n_ph, dtype=complex)) for d in dipoles
        ]

        e0 = eig_vecs[:, 0]
        n_states = eig_vecs.shape[1]
        E0 = complex(eig_vals[0]).real

        intensities = []
        delta_energies_raw = []
        for i in range(n_states):
            ei = eig_vecs[:, i]
            Ei = complex(eig_vals[i]).real
            total = 0.0
            for M_full in M_fulls:
                mat_el = np.vdot(e0, M_full @ ei)
                total += np.abs(mat_el) ** 2
            intensities.append(total)
            delta_energies_raw.append(Ei - E0)

        delta_energies = _energy_to_eV(delta_energies_raw, self.energy_unit)

        out_path = self._resolve_output_path()
        create_directory(out_path.parent)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.stem(
            delta_energies, intensities,
            linefmt="C0-", basefmt=" ", markerfmt="C0o"
        )
        ax.set_xlabel(r"$\Delta E$ (eV)")
        ax.set_ylabel(
            r"$\sum_\alpha |\langle e_0 | D_\alpha | e_i \rangle|^2$"
            if self.use_dipole_xyz
            else r"$|\langle e_0 | M | e_i \rangle|^2$"
        )
        ax.set_title("Absorption spectrum")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        if show_plot:
            plt.show()
        plt.close()
        print(f"Spectrum plot saved to {out_path}")

        spec_csv = out_path.with_suffix(".csv")
        pd.DataFrame({
            "delta_E_eV": delta_energies,
            "intensity": intensities,
        }).to_csv(spec_csv, index=False, sep=";")
        print(f"Spectrum data saved to {spec_csv}")

        return 0

    def calculate_smearing(self, range_eV: List[float], show_plot: bool = True) -> int:
        """
        Perform the spectrum calculation: load data, compute intensities, plot, save.

        Returns:
            0 on success, 1 on error (e.g. file not found).
        """
        
        min_energy = min(range_eV)
        max_energy = max(range_eV)
        
        if not self.npz_path:
            print("Error: npz path not configured")
            return 1
        if not self.use_dipole_xyz and not self.dipole_path:
            print("Error: dipole or dipole_x/y/z paths not configured")
            return 1
        if self.use_dipole_xyz and not all(
            [self.dipole_x_path, self.dipole_y_path, self.dipole_z_path]
        ):
            print("Error: all of dipole_x, dipole_y, dipole_z must be set")
            return 1

        npz_path = Path(self.npz_path)
        if not npz_path.exists():
            print(f"NPZ not found: {npz_path}")
            return 1

        if self.use_dipole_xyz:
            for p in [self.dipole_x_path, self.dipole_y_path, self.dipole_z_path]:
                if not Path(p).exists():
                    print(f"Dipole CSV not found: {p}")
                    return 1
        elif not Path(self.dipole_path).exists():
            print(f"Dipole CSV not found: {self.dipole_path}")
            return 1

        dipoles = self._load_dipoles()

        d_el = dipoles[0].shape[0]
        data = load_lvc_npz(npz_path)
        eig_vecs = data["eigenvectors"]
        eig_vals = data["eigenvalues"]
        dim = data["dim"]

        if dim % d_el != 0:
            raise ValueError(
                f"Full dim ({dim}) must be divisible by electron dim ({d_el})."
            )
        n_ph = dim // d_el

        M_fulls = [
            np.kron(d, np.eye(n_ph, dtype=complex)) for d in dipoles
        ]

        e0 = eig_vecs[:, 0]
        n_states = eig_vecs.shape[1]
        E0 = complex(eig_vals[0]).real

        intensity_smears = [0.0] * len(range_eV)
        intensity_peaks = []
        delta_energies = []
        for i in range(n_states):
            ei = eig_vecs[:, i]
            Ei = complex(eig_vals[i]).real
            delta_energy = _energy_to_eV([Ei - E0], self.energy_unit)[0] 
            if delta_energy  > max_energy or delta_energy < min_energy:
                continue
            total = np.float64(0.0)
            for M_full in M_fulls:
                mat_el = np.vdot(e0, M_full @ ei)
                total += np.abs(mat_el) ** 2
            self.smearing.center = delta_energy
            if total > 0.0:
                delta_energies.append(delta_energy)
                intensity_peaks.append(total)

                for j, energy in enumerate(range_eV):
                    intensity_smears[j] += total * self.smearing.evaluate(energy )

        out_path = self._resolve_output_path()
        create_directory(out_path.parent)





        fig, ax = plt.subplots(figsize=(8, 5))
        ax.stem(
            delta_energies, intensity_peaks,
            linefmt="C0-", basefmt=" ", markerfmt="C0o"
        )
        ax.plot(
            range_eV, intensity_smears,
            color="C0"
        )
        ax.set_xlabel(r"$\Delta E$ (eV)")
        ax.set_ylabel(
            r"$\sum_\alpha |\langle e_0 | D_\alpha | e_i \rangle|^2$"
            if self.use_dipole_xyz
            else r"$|\langle e_0 | M | e_i \rangle|^2$"
        )
        ax.set_title("Absorption spectrum")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        if show_plot:
            plt.show()
        plt.close()
        print(f"Spectrum plot saved to {out_path}")

        spec_csv = out_path.with_suffix(".csv")
        pd.DataFrame({
            "delta_E_eV": range_eV,
            "intensity_smearing": intensity_smears,
        }).to_csv(spec_csv, index=False, sep=";")
        print(f"Spectrum data saved to {spec_csv}")

        return 0

