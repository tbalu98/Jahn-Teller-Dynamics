"""
PVC (polynomial vibronic coupling) configuration parsing.

PVCConfigParser reads .cfg files and builds PVCCalculation objects for
:mod:`jahn_teller_dynamics.PVC` runs.
"""

from __future__ import annotations

from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from jahn_teller_dynamics.io.config.reader import ConfigReader
from jahn_teller_dynamics.io.utils.safe_numeric_expr import parse_config_int_expression
from jahn_teller_dynamics.io.utils.path_manager import PathManager
from jahn_teller_dynamics.io.utils.run_context import run_dir as get_run_dir

_HAMILTONIAN_BUILDER_GROUPED = frozenset({"grouped", "expression", "expressions"})
_HAMILTONIAN_BUILDER_LINED = frozenset({"lined", "line", "row", "rows", "legacy"})


def normalize_hamiltonian_builder(raw: str) -> str:
    """
    Normalize ``hamiltonian_builder`` from .cfg / CLI to ``grouped`` or ``lined``.

    - ``grouped`` (aliases ``expression``, ``expressions``) — one orbital matrix per CSV expression.
    - ``lined`` (aliases ``row``, ``line``, ``legacy``) — legacy row-by-row ``|i⟩⟨j| ⊗ V(p)``.
    """
    value = (raw or "lined").strip().lower()
    if value in _HAMILTONIAN_BUILDER_GROUPED:
        return "grouped"
    if value in _HAMILTONIAN_BUILDER_LINED:
        return "lined"
    allowed = ", ".join(sorted(_HAMILTONIAN_BUILDER_GROUPED | _HAMILTONIAN_BUILDER_LINED))
    raise ValueError(f"hamiltonian_builder must be one of: {allowed}; got {raw!r}")


@dataclass
class PVCCalculation:
    """
    Parameters for a PVC Hamiltonian diagonalization (polynomial coupling + phonons).

    Phonon Hilbert space (set one of):

    - ``maximum_number_of_vibrational_quanta`` > 0 →
      :class:`~jahn_teller_dynamics.physics.models.constrained_multimode_phonon.MultiModeConstrainedPhononSystem`
      (total phonon number ``sum_i n_i <= N``).
    - ``maximum_number_of_quanta_per_dim`` > 0 →
      :class:`~jahn_teller_dynamics.physics.models.system_builder.MultiModePhononSystem`
      (per-mode cutoff ``n_i <= N`` for each mode).

    If neither is set, ``order`` is used as the constrained total-phonon cutoff (legacy default).

    Paths resolve relative to the run directory (cwd), same as :class:`LVCCalculation`.

    MPI / PETSc diagonalization:

    - Set ``eigensolver`` to ``slepc`` (or aliases ``petsc`` / ``slepc_eps``), **or**
    - Set ``running_environment`` to ``multiprocessor``, ``mpi``, ``parallel``, or ``hpc`` when
      ``eigensolver`` is left empty, to route the PVC driver to ``SLEPc`` (requires PETSc/SLEPc).

    Spectral selection (all PVC backends: dense, scipy sparse ``eigsh``, SLEPc):

    - ``num_eigs`` — number of eigenpairs (see also ``--num-eigs``).
    - ``eigensolver_sigma`` (aliases ``eigensolver_shift``, ``eigensolver_target``, ``spectral_sigma``) —
      shift / target for shift–invert (interior region). With SLEPc, sets ST shift–invert and
      ``which`` to largest magnitude on the transformed operator.
    - ``eigensolver_spectral_which`` (aliases ``eigensolver_which``, ``spectral_which``) —
      e.g. ``smallest_real``, ``largest_real``, ``smallest_magnitude``, ``nearest`` (``nearest``
      requires ``eigensolver_sigma``), or SciPy tokens ``SA`` / ``LA`` / ``SM`` / ``LM``.
    - ``eigensolver_tol``, ``eigensolver_max_it`` (``eigensolver_maxiter``), ``eigensolver_ncv`` —
      scipy ``eigsh`` convergence knobs (sparse backend only; defaults ``1e-10``, ``10000``, auto).

    Coupling CSV coefficient scaling (after ``el_state_1`` / ``el_state_2`` are resolved to the
    same 1-based indices as ``electron_energies.csv`` row order):

    - ``tune_tuning`` — multiply ``coeff`` when both indices are equal (default ``1.0``).
    - ``tune_coupling`` — multiply ``coeff`` when the indices differ (default ``1.0``).

    Coupling-table completion before building ``H`` (see
    :func:`~jahn_teller_dynamics.physics.hamiltonians.djt_polynomial_hamiltonian.build_polynomial_coupling_hamiltonian`):

    - ``hermitian_completion`` — if ``true``, add missing partner rows ``(j, i, p†, c)`` for
      off-diagonal ``(i, j, p, c)`` (default ``true``).
    - ``diagonal_completion`` — if ``true``, replicate each diagonal row ``(i, i, p, c)`` to all
      other electronic states (default ``false``).
    - ``symmetrize_hamiltonian`` — if ``true``, replace assembled ``H`` by ``(H + H†) / 2`` when row
      completion did not already yield a Hermitian matrix (default ``true``).

    Hamiltonian assembly (``hamiltonian_builder`` in ``[PVC]`` or ``[essentials]``):

    - ``lined`` (default; alias ``row``) — legacy row-by-row ``|i⟩⟨j| ⊗ V(p)`` expansion.
    - ``grouped`` (alias ``expression``) — one :math:`N \\times N` orbital operator per CSV
      expression on the orbital subsystem, then tensored with phonon operators.
    """

    data_dir: str = ""
    out_dir: str = ""
    electron_energies_path: str = ""
    modes_path: str = ""
    coupling_path: str = ""

    electron_energies: str = "electron_energies.csv"
    modes: str = "modes.csv"
    coupling_csv: str = "coupling.csv"

    maximum_number_of_vibrational_quanta: int = 0
    maximum_number_of_quanta_per_dim: int = 0
    order: int = 2
    num_eigs: Optional[int] = None
    eigensolver_sigma: Optional[float] = None
    eigensolver_spectral_which: str = ""
    eigensolver_tol: Optional[float] = None
    eigensolver_max_it: Optional[int] = None
    eigensolver_ncv: Optional[int] = None
    use_sparse: bool = True
    eigensolver: str = ""
    running_environment: str = ""
    modes_to_use: Optional[list[int]] = None

    separator: str = ";"
    dimensionless_coordinates: bool = True
    null_point_vib: bool = True
    exp_approximation_order: Optional[int] = None
    save_npz: bool = False
    save_csv: bool = True
    # Filename (or path) used when saving eigenvectors as NPZ. A bare filename
    # is resolved under ``out_dir``; an absolute / explicit relative path is
    # used as given. ``.npz`` is appended if missing.
    npz_filename: str = "eigenvectors.npz"

    tune_tuning: float = 1.0
    tune_coupling: float = 1.0
    hermitian_completion: bool = True
    diagonal_completion: bool = False
    symmetrize_hamiltonian: bool = True
    hamiltonian_builder: str = "lined"

    def resolve_input_paths(self, run_dir: Optional[Path] = None) -> tuple[Path, Path, Path]:
        """
        Resolve electron energies, modes, and polynomial-coupling CSV paths.

        Returns:
            (electron_energies_path, modes_path, coupling_path)
        """
        rd = run_dir or get_run_dir()
        data = Path(self.data_dir).expanduser()
        if not data.is_absolute():
            data = (rd / data).resolve()

        def _path(explicit: str, filename: str) -> Path:
            if explicit:
                p = Path(explicit).expanduser()
                if not p.is_absolute():
                    p = (rd / p).resolve()
            else:
                p = data / filename
            return p.resolve()

        return (
            _path(self.electron_energies_path, self.electron_energies),
            _path(self.modes_path, self.modes),
            _path(self.coupling_path, self.coupling_csv),
        )

    def resolve_out_dir(self, run_dir: Optional[Path] = None) -> Path:
        """Resolve output directory to absolute Path."""
        rd = run_dir or get_run_dir()
        if self.out_dir:
            p = Path(self.out_dir).expanduser()
            if not p.is_absolute():
                p = (rd / p).resolve()
            return p.resolve()
        return (rd / "results" / "PVC").resolve()

    def resolve_npz_path(self, run_dir: Optional[Path] = None) -> Path:
        """
        Resolve the absolute path of the eigenvectors NPZ output file.

        A bare filename (e.g. ``"eigenvectors_excited_1eV.npz"``) is placed
        under :meth:`resolve_out_dir`; a relative path is joined to
        ``run_dir``; an absolute path is used as-is. A ``.npz`` suffix is
        appended if missing.
        """
        rd = run_dir or get_run_dir()
        name = (self.npz_filename or "eigenvectors.npz").strip()
        if not name:
            name = "eigenvectors.npz"
        if not name.lower().endswith(".npz"):
            name = name + ".npz"
        p = Path(name).expanduser()
        if p.is_absolute():
            return p.resolve()
        # No directory component → place under out_dir; otherwise resolve
        # against the run directory.
        if p.parent == Path(""):
            return (self.resolve_out_dir(rd) / p).resolve()
        return (rd / p).resolve()


class PVCConfigParser:
    """
    Parses .cfg files and builds :class:`PVCCalculation` objects.

    Reads ``[PVC]`` and ``[essentials]`` sections.
    """

    def __init__(self, config_file_path: str, run_dir: Optional[Path] = None):
        self.run_dir = run_dir or get_run_dir()
        cfg_path = Path(config_file_path).expanduser()
        if not cfg_path.is_absolute():
            cfg_path = (self.run_dir / cfg_path).resolve()
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")

        self.config_file_path = str(cfg_path)
        cp = ConfigParser()
        cp.read(cfg_path)

        self.config = cp
        self.reader = ConfigReader(cp)
        self.paths = PathManager(self.reader, self.config_file_path)

        pvc_section = None
        for candidate in ("PVC", "pvc"):
            if cp.has_section(candidate):
                pvc_section = candidate
                break
        essentials_section = "essentials" if cp.has_section("essentials") else None
        if pvc_section is None and essentials_section is None:
            raise ValueError(f"Missing [PVC] (or [essentials]) section in config: {cfg_path}")

        self._pvc_section = pvc_section
        self._essentials_section = essentials_section
        self._base = self.run_dir

    def _get_path(self, opt: str, section: Optional[str] = None) -> Optional[str]:
        sec = section or self._pvc_section or self._essentials_section
        if sec is None or not self.config.has_option(sec, opt):
            return None
        raw = self.config.get(sec, opt, fallback="").strip()
        if not raw:
            return ""
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = (self._base / p).resolve()
        return str(p)

    def _get_str(self, opt: str, default: str = "") -> str:
        for sec in (self._pvc_section, self._essentials_section):
            if sec is not None and self.config.has_option(sec, opt):
                return self.config.get(sec, opt, fallback=default).strip()
        return default

    def _get_int(self, opt: str, default: Optional[int] = None) -> Optional[int]:
        for sec in (self._pvc_section, self._essentials_section):
            if sec is not None and self.config.has_option(sec, opt):
                return self.config.getint(sec, opt, fallback=default)
        return default

    def _get_bool(self, opt: str, default: Optional[bool] = None) -> Optional[bool]:
        for sec in (self._pvc_section, self._essentials_section):
            if sec is not None and self.config.has_option(sec, opt):
                return self.config.getboolean(sec, opt, fallback=default)
        return default

    def _get_nonneg_int_maybe_expr(self, opt: str) -> Optional[int]:
        """
        Read a non-negative integer from [PVC] or [essentials] (PVC wins if both define it).

        Plain integers and small arithmetic expressions (e.g. ``2+3``) are accepted.
        """
        for sec in (self._pvc_section, self._essentials_section):
            if sec is not None and self.config.has_option(sec, opt):
                raw = self.config.get(sec, opt).strip()
                if not raw:
                    return None
                try:
                    v = int(raw)
                except ValueError:
                    v = parse_config_int_expression(raw)
                if v < 0:
                    raise ValueError(f"Option {opt!r} must be non-negative; got {v}")
                return v
        return None

    def build_calculation(self) -> PVCCalculation:
        """Build :class:`PVCCalculation` from the parsed config."""
        calc = PVCCalculation()

        if self._essentials_section is not None:
            in_folder = self.paths.get_data_folder_name().rstrip("/")
            if in_folder:
                p = Path(in_folder).expanduser()
                if not p.is_absolute():
                    p = (self._base / p).resolve()
                calc.data_dir = str(p)

            out_folder = self.paths.get_res_folder_name().rstrip("/")
            if out_folder:
                p = Path(out_folder).expanduser()
                if not p.is_absolute():
                    p = (self._base / p).resolve()
                calc.out_dir = str(p)

            if self.config.has_option(self._essentials_section, "use_sparse"):
                calc.use_sparse = self.config.getboolean(
                    self._essentials_section, "use_sparse", fallback=True
                )

        if self._pvc_section is not None:
            if (d := self._get_path("data_dir", self._pvc_section)) is not None:
                calc.data_dir = d
            if (o := self._get_path("out_dir", self._pvc_section) or self._get_path("out-dir", self._pvc_section)) is not None and o:
                calc.out_dir = o

            for opt, attr in [
                ("electron_energies", "electron_energies"),
                ("modes", "modes"),
                ("coupling", "coupling_csv"),
            ]:
                val = self._get_str(opt, getattr(calc, attr))
                if val:
                    setattr(calc, attr, val)

            for opt, attr, alts in [
                ("electron_energies_path", "electron_energies_path", ["electron_energies_csv_path"]),
                ("modes_path", "modes_path", ["modes_csv_path"]),
                ("coupling_path", "coupling_path", ["polynomial_coupling_path", "pvc_coupling_path"]),
            ]:
                val = self._get_path(opt, self._pvc_section)
                for alt in alts:
                    if val is None:
                        val = self._get_path(alt, self._pvc_section)
                if val is not None:
                    setattr(calc, attr, val)

            raw = self._get_str("modes_to_use")
            if raw:
                raw_lower = raw.strip().lower()
                if raw_lower in ("all", "none", ""):
                    calc.modes_to_use = None
                else:
                    try:
                        calc.modes_to_use = [int(x.strip()) for x in raw.split(",") if x.strip()]
                    except ValueError:
                        calc.modes_to_use = None

            raw = self._get_str("num_eigs") or self._get_str("num-eigs")
            if raw is not None:
                raw = raw.strip().lower()
                if raw in ("", "all", "none"):
                    calc.num_eigs = None
                else:
                    try:
                        calc.num_eigs = int(raw)
                    except ValueError:
                        calc.num_eigs = parse_config_int_expression(raw)

            if (u := self._get_bool("use_sparse")) is not None:
                calc.use_sparse = u

            if self.config.has_option(self._pvc_section, "separator"):
                sep_val = self._get_str("separator")
                calc.separator = sep_val if sep_val else ";"

        es_use = (
            self._get_str("eigensolver").strip()
            or self._get_str("eigen_solver").strip()
        )
        if es_use:
            calc.eigensolver = es_use
        renv_use = self._get_str("running_environment").strip()
        if renv_use:
            calc.running_environment = renv_use

        for key in (
            "eigensolver_sigma",
            "eigensolver_shift",
            "eigensolver_target",
            "spectral_sigma",
        ):
            v = self._get_str(key, "").strip()
            if v:
                calc.eigensolver_sigma = float(v)
                break
        for key in ("eigensolver_spectral_which", "eigensolver_which", "spectral_which"):
            v = self._get_str(key, "").strip()
            if v:
                calc.eigensolver_spectral_which = v
                break

        for key in ("eigensolver_tol", "eigsh_tol", "eps_tol"):
            v = self._get_str(key, "").strip()
            if v:
                calc.eigensolver_tol = float(v)
                break
        for key in ("eigensolver_max_it", "eigensolver_maxiter", "eigsh_maxiter"):
            v = self._get_str(key, "").strip()
            if v:
                calc.eigensolver_max_it = int(float(v))
                break
        for key in ("eigensolver_ncv", "eigsh_ncv"):
            v = self._get_str(key, "").strip()
            if v:
                calc.eigensolver_ncv = int(float(v))
                break

        # Phonon / output flags may appear in [PVC] or [essentials]; [PVC] wins if both set.
        for opt, attr in [
            ("dimensionless_coordinates", "dimensionless_coordinates"),
            ("null_point_vib", "null_point_vib"),
            ("save_npz", "save_npz"),
            ("save_csv", "save_csv"),
        ]:
            if (v := self._get_bool(opt)) is not None:
                setattr(calc, attr, v)

        for opt, attr in [
            ("order", "order"),
            ("maximum_number_of_vibrational_quanta", "maximum_number_of_vibrational_quanta"),
            ("maximum_number_of_quanta_per_dim", "maximum_number_of_quanta_per_dim"),
        ]:
            if (v := self._get_nonneg_int_maybe_expr(opt)) is not None:
                setattr(calc, attr, v)

        if (v := self._get_nonneg_int_maybe_expr("exp_approximation_order")) is not None:
            calc.exp_approximation_order = v

        for key in ("npz_filename", "eigenvectors_npz_filename", "output_npz_filename"):
            v = self._get_str(key, "").strip()
            if v:
                calc.npz_filename = v
                break

        for key, attr in (
            ("tune_tuning", "tune_tuning"),
            ("tune_coupling", "tune_coupling"),
        ):
            v = self._get_str(key, "").strip()
            if v:
                setattr(calc, attr, float(v))

        for opt, attr in (
            ("hermitian_completion", "hermitian_completion"),
            ("diagonal_completion", "diagonal_completion"),
            ("symmetrize_hamiltonian", "symmetrize_hamiltonian"),
        ):
            if (v := self._get_bool(opt)) is not None:
                setattr(calc, attr, v)

        for key in ("hamiltonian_builder", "hamiltonian_build", "pvc_hamiltonian_builder"):
            v = self._get_str(key, "").strip()
            if v:
                calc.hamiltonian_builder = normalize_hamiltonian_builder(v)
                break

        return calc
