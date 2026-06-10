"""
SpectrumCalculator - Computes LVC absorption spectrum from eigenvectors and dipole matrices.

The full-space dipole ``kron(D_el, I_ph)`` is kept in sparse CSR form so spectrum runs do not
allocate a dense ``dim(H) × dim(H)`` matrix.

Receives resolved paths and settings; has no knowledge of .cfg files.
Configuration is set by SpectrumBuilder.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, eye, kron as sp_kron, lil_matrix

from jahn_teller_dynamics.io.file_io.npz_reader import load_lvc_npz
from jahn_teller_dynamics.io.utils.file_utils import create_directory
from jahn_teller_dynamics.math_utils.maths import Lorentzian


# 1 Hartree = 27.211386245988 eV (CODATA 2018)
HARTREE_TO_EV = 27.211386245988


_DIPOLE_CSV_COLUMNS = ("state_i", "state_j", "value")


def _is_dipole_csv(path: str | Path, sep: str = ";") -> bool:
    """Return True if ``path`` is a ``state_i;state_j;value`` dipole CSV."""
    try:
        header = pd.read_csv(path, sep=sep, nrows=0).columns
    except Exception:
        return False
    cols_norm = {str(c).strip().lower() for c in header}
    return set(_DIPOLE_CSV_COLUMNS).issubset(cols_norm)


def _read_dipole_csv(
    path: str | Path,
    sep: str = ";",
) -> csr_matrix:
    """
    Read a dipole matrix from a ``state_i;state_j;value`` CSV and return
    a sparse Hermitian CSR matrix.

    Expected columns: ``state_i;state_j;value`` (case-insensitive). The
    matrix dimension is the number of *distinct* labels in the
    ``state_i`` column; each label is mapped to a matrix index by its
    order of first appearance (repeated labels do not increase the
    dimension). ``state_j`` values must all be among those labels.

    Each row contributes one matrix entry ``M[i, j] = value``; the
    conjugate transposed entry ``M[j, i] = conj(value)`` is filled
    automatically, so only one triangle of the matrix needs to be listed
    in the file. The diagonal is taken to be real (any imaginary
    component is discarded with a warning).

    Args:
        path: CSV file path.
        sep: CSV separator (default ``;``).

    Returns:
        Hermitian complex :class:`scipy.sparse.csr_matrix` of shape
        ``(dim, dim)`` where ``dim`` is the number of distinct ``state_i``
        labels.
    """
    path = Path(path)
    df = pd.read_csv(path, sep=sep)
    cols = {str(c).strip().lower(): c for c in df.columns}
    missing = [c for c in _DIPOLE_CSV_COLUMNS if c not in cols]
    if missing:
        raise ValueError(
            f"Dipole CSV {path} is missing column(s) {missing}; "
            f"found {list(df.columns)}"
        )

    i_labels_raw = df[cols["state_i"]].tolist()
    j_labels_raw = df[cols["state_j"]].tolist()
    v_arr = [complex(v) for v in df[cols["value"]].tolist()]

    if not i_labels_raw:
        raise ValueError(f"Dipole CSV {path} contains no rows")

    label_to_idx: dict[object, int] = {}
    for lbl in i_labels_raw:
        if lbl not in label_to_idx:
            label_to_idx[lbl] = len(label_to_idx)
    dim = len(label_to_idx)

    unknown_j = sorted({lbl for lbl in j_labels_raw if lbl not in label_to_idx},
                       key=str)
    if unknown_j:
        raise ValueError(
            f"Dipole CSV {path}: state_j contains label(s) {unknown_j} "
            f"that do not appear in the state_i column; the basis is defined "
            f"by the distinct values of state_i."
        )

    # Electronic dimension is typically small; use lil so assignments match
    # dense semantics if the file lists both (i,j) and (j,i).
    M = lil_matrix((dim, dim), dtype=np.complex128)
    seen: dict[tuple[int, int], complex] = {}
    for lbl_i, lbl_j, v in zip(i_labels_raw, j_labels_raw, v_arr):
        i = label_to_idx[lbl_i]
        j = label_to_idx[lbl_j]
        prior = seen.get((i, j))
        if prior is not None:
            if not np.isclose(prior, v):
                raise ValueError(
                    f"Dipole CSV {path} has conflicting entries for "
                    f"(state_i={lbl_i}, state_j={lbl_j}): {prior} vs {v}"
                )
            continue
        if i == j:
            if v.imag != 0.0:
                print(
                    f"Warning: discarding imaginary part of diagonal dipole entry "
                    f"(state_i=state_j={lbl_i}) = {v} in {path}"
                )
            M[i, i] = v.real
        else:
            mirror = seen.get((j, i))
            if mirror is not None and not np.isclose(mirror, np.conjugate(v)):
                raise ValueError(
                    f"Dipole CSV {path} is not Hermitian at "
                    f"(state_i={lbl_i}, state_j={lbl_j}): "
                    f"M[{lbl_i},{lbl_j}]={v}, M[{lbl_j},{lbl_i}]={mirror} "
                    f"(expected the latter to equal conj of the former)"
                )
            M[i, j] = v
            M[j, i] = np.conjugate(v)
        seen[(i, j)] = v

    return M.tocsr()


def _read_dipole_matrix_csv(path: str | Path, sep: str = ";") -> csr_matrix:
    """
    Read a dipole matrix from CSV. Two formats are auto-detected:

    1. ``state_i;state_j;value`` dipole CSV — see :func:`_read_dipole_csv`.
    2. Legacy dense ``d × d`` matrix — first column is the row index,
       the header row contains column indices. Returned as CSR (small
       ``d`` only; prefer triplet CSV for large operators).
    """
    path = Path(path)
    if _is_dipole_csv(path, sep=sep):
        return _read_dipole_csv(path, sep=sep)

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
    return csr_matrix(arr)


def _kron_d_el_identity_nph(
    d_el: csr_matrix,
    n_ph: int,
) -> csr_matrix:
    """
    Sparse ``kron(D_el, I_{n_ph})`` matching the legacy dense layout.

    The previous implementation used ``np.kron(d, np.eye(n_ph))``, which
    allocated a full ``(dim, dim)`` dense matrix with ``dim = d_el * n_ph``
    and was the dominant memory cost. Here ``nnz`` is at most
    ``nnz(D_el) * n_ph``.
    """
    dtype = np.promote_types(d_el.dtype, np.complex128)
    d_csr = d_el.astype(dtype, copy=False).tocsr()
    Iph = eye(n_ph, dtype=dtype, format="csr")
    return sp_kron(d_csr, Iph, format="csr")


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
        npz_paths: Optional[List[str]] = None,
    ):
        """
        Args:
            npz_path: Absolute path to eigenvectors.npz. Equivalent to passing
                a single-element ``npz_paths`` list; kept for backwards
                compatibility with callers that have only one NPZ.
            npz_paths: Ordered list of eigenvectors NPZs. The first file is
                assumed to contain the ground state (used to define ``e0``
                / ``E0``); the remaining files contribute additional
                final-state eigenkets. Useful when each NPZ comes from a
                separate SLEPc shift–invert run targeting a different
                energy window so you don't have to diagonalize the full
                spectrum.
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
        paths: List[str] = []
        if npz_paths:
            paths = [str(p) for p in npz_paths if str(p).strip()]
        elif npz_path:
            paths = [str(npz_path)]
        self.npz_paths: List[str] = paths
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

    @property
    def npz_path(self) -> str:
        """First NPZ path (ground-state file). Empty if none configured."""
        return self.npz_paths[0] if self.npz_paths else ""

    def _check_npz_paths_exist(self) -> Optional[str]:
        """Return an error message if any configured NPZ path is missing."""
        if not self.npz_paths:
            return "Error: npz path not configured"
        for p in self.npz_paths:
            if not Path(p).exists():
                return f"NPZ not found: {p}"
        return None

    def _load_eigen_data(self) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Load eigenvectors/eigenvalues from all configured NPZs and concatenate.

        The first file is treated as the ground-state file: its column 0
        contributes ``e0`` and its eigenvalue 0 contributes ``E0`` in the
        callers. The full concatenated stack is returned so callers may
        use every eigenpair as a candidate final state.

        All NPZs must share the same Hilbert-space dimension ``dim`` and
        eigenvector length; mismatches raise ``ValueError``.
        """
        if not self.npz_paths:
            raise ValueError("SpectrumCalculator: no NPZ paths configured")
        first = load_lvc_npz(Path(self.npz_paths[0]))
        vecs_list = [np.asarray(first["eigenvectors"])]
        vals_list = [np.asarray(first["eigenvalues"])]
        dim = int(first["dim"])
        vec_len = vecs_list[0].shape[0]
        for p in self.npz_paths[1:]:
            ni = load_lvc_npz(Path(p))
            di = int(ni["dim"])
            if di != dim:
                raise ValueError(
                    f"NPZ {p}: dim={di} differs from {self.npz_paths[0]} "
                    f"(dim={dim}); cannot combine eigenpairs across "
                    "incompatible Hilbert spaces."
                )
            vi = np.asarray(ni["eigenvectors"])
            if vi.shape[0] != vec_len:
                raise ValueError(
                    f"NPZ {p}: eigenvectors have length {vi.shape[0]}, "
                    f"expected {vec_len} from {self.npz_paths[0]}"
                )
            vecs_list.append(vi)
            vals_list.append(np.asarray(ni["eigenvalues"]))
        eig_vecs = np.concatenate(vecs_list, axis=1)
        eig_vals = np.concatenate(vals_list)
        return eig_vecs, eig_vals, dim

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

    def _load_dipoles(self) -> List[csr_matrix]:
        """Load dipole matrix/matrices from CSV(s) as sparse CSR."""
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
        err = self._check_npz_paths_exist()
        if err is not None:
            print(err)
            return 1
        if not self.use_dipole_xyz and not self.dipole_path:
            print("Error: dipole or dipole_x/y/z paths not configured")
            return 1
        if self.use_dipole_xyz and not all(
            [self.dipole_x_path, self.dipole_y_path, self.dipole_z_path]
        ):
            print("Error: all of dipole_x, dipole_y, dipole_z must be set")
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
        eig_vecs, eig_vals, dim = self._load_eigen_data()

        if dim % d_el != 0:
            raise ValueError(
                f"Full dim ({dim}) must be divisible by electron dim ({d_el})."
            )
        n_ph = dim // d_el

        M_fulls = [_kron_d_el_identity_nph(d, n_ph) for d in dipoles]

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

        err = self._check_npz_paths_exist()
        if err is not None:
            print(err)
            return 1
        if not self.use_dipole_xyz and not self.dipole_path:
            print("Error: dipole or dipole_x/y/z paths not configured")
            return 1
        if self.use_dipole_xyz and not all(
            [self.dipole_x_path, self.dipole_y_path, self.dipole_z_path]
        ):
            print("Error: all of dipole_x, dipole_y, dipole_z must be set")
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
        eig_vecs, eig_vals, dim = self._load_eigen_data()
        if len(self.npz_paths) > 1:
            print(
                f"Spectrum: loaded {eig_vecs.shape[1]} eigenpair(s) from "
                f"{len(self.npz_paths)} NPZ file(s) "
                f"(ground state from {self.npz_paths[0]})"
            )

        if dim % d_el != 0:
            raise ValueError(
                f"Full dim ({dim}) must be divisible by electron dim ({d_el})."
            )
        n_ph = dim // d_el

        M_fulls = [_kron_d_el_identity_nph(d, n_ph) for d in dipoles]

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

