"""
PVC (polynomial vibronic coupling) configuration parsing.

PVCConfigParser reads .cfg files and builds PVCCalculation objects for
:mod:`jahn_teller_dynamics.jtd_run` runs.
"""

from __future__ import annotations

import warnings
from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

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
    - ``phonon_encut`` > 0 → constrained phonons by excitation energy
      ``sum_i n_i e_i <= phonon_encut`` (``e_i`` from modes CSV).
    - ``maximum_number_of_quanta_per_dim`` > 0 →
      :class:`~jahn_teller_dynamics.physics.models.system_builder.MultiModePhononSystem`
      (per-mode cutoff ``n_i <= N`` with the **same** ``N`` for every mode).
    - ``use_maximum_quanta_from_modes_csv = true`` →
      :class:`~jahn_teller_dynamics.physics.models.system_builder.MultiModePhononSystem`
      with per-mode ``N_i`` read from the ``maximum_quanta`` column of ``modes.csv``
      (each mode can have a different cap). Hilbert dimension is ``prod_i (N_i + 1)``.

    Set only one of the four truncation keys above. If none are set, ``order`` is used as the
    constrained total-phonon cutoff (legacy default).

    Paths resolve relative to the run directory (cwd), same as :class:`LVCCalculation`.

    MPI / PETSc diagonalization:

    - Set ``eigensolver`` to ``slepc`` (or aliases ``petsc`` / ``slepc_eps``), **or**
    - Set ``running_environment`` to ``multiprocessor``, ``mpi``, ``parallel``, or ``hpc`` when
      ``eigensolver`` is left empty, to route the jtd_run driver to ``SLEPc`` (requires PETSc/SLEPc).

    INI layout (``PVCConfigParser``):

    - ``[essentials]`` — ``input_folder``, ``output_folder``, CSV ``separator``, I/O flags
      (``save_npz``, ``save_csv``, ``save_hamiltonian``, ``hamiltonian_filename``,
      ``load_hamiltonians``, ``eigensolve``), ``use_sparse``, ``npz_filename``, and optionally
      phonon truncation keys (same four as below; ``[model]`` overrides on conflict).
    - ``[model]`` — phonon basis truncation, CSV inputs, coupling completion, coordinates,
      spin–orbital operator, ``modes_to_use``, ``tune_*``, ``exp_approximation_order``.
    - ``[builder]`` (optional) — parallel lined coupling assembly. If this section exists,
      parallel build is **on** by default; set ``parallel_lined_coupling = false`` to disable.
      Keys: ``lined_coupling_pool``, ``lined_coupling_batches``, ``lined_coupling_workers``,
      plus split-job options (``load_coupling_rows``, chunk indices, ``save_bare_hamiltonian``, …).
    - ``[eigensolver]`` — diagonalization backend and spectral options.

    Legacy ``[PVC]`` is treated as an alias for ``[model]`` (deprecated). Eigensolver keys in
    ``[eigensolver]`` override the same keys in legacy ``[PVC]`` / ``[essentials]``.

    ``[eigensolver]`` keys (legacy names in ``[PVC]`` still accepted with a warning):

    - ``solver`` / ``eigensolver`` / ``backend`` — ``sparse`` (aliases: ``scipy``, ``scipy_sparse``,
      ``eigsh``, ``arpack`` — SciPy Lanczos / ``eigsh``), ``dense``, or ``slepc``.
    - ``num_eigs`` / ``k`` — number of eigenpairs (``all`` = full spectrum on dense).
    - ``sigma`` / ``shift`` / ``target`` / ``spectral_sigma`` — shift–invert target (sparse/SLEPc).
    - ``spectral_which`` / ``which`` — ``smallest_real``, ``largest_real``, ``nearest``, ``SA``, …
    - ``tol``, ``max_iter`` / ``maxiter``, ``ncv`` — convergence knobs for both the sparse
      ``eigsh`` ARPACK backend and the SLEPc backend (``tol`` → ``EPSSetTolerances`` tolerance,
      ``max_iter`` → its max iterations, ``ncv`` → ``EPSSetDimensions`` subspace size).
    - ``mode`` — shift–invert mode: ``normal``, ``buckling``, ``cayley`` (with ``sigma`` set).
    - ``v0_file`` / ``v0`` — path to ``.npy`` initial Lanczos vector (length = dim(H)).
    - ``return_eigenvectors`` — ``true``/``false`` (default ``true``).
    - ``rng_seed`` — seed for ARPACK starting vector when ``v0_file`` is not set.
    - ``use_block_diagonalization`` — sparse solver block structure (default ``true``).
    - ``require_hermitian`` — if ``false``, diagonalize non-Hermitian ``H`` with ``eigs`` / SLEPc
      NHEP instead of aborting (default ``true``). Alias: ``allow_non_hermitian``.

    Electron-state labels: ``electron_energies.csv`` ``el_state`` values may be arbitrary
    non-empty strings (e.g. ``ground``, ``T2``). Coupling CSV ``el_state_1`` / ``el_state_2``
    (aliases ``el_state_n`` / ``el_state_m``)
    use those same labels; legacy numeric 1-based indices are still accepted.

    Coupling CSV coefficient scaling (after ``el_state_1`` / ``el_state_2`` are resolved to the
    same 1-based indices as ``electron_energies.csv`` row order):

    - ``tune_tuning`` — multiply ``coeff`` when both indices are equal (default ``1.0``).
    - ``tune_coupling`` — multiply ``coeff`` when the indices differ (default ``1.0``).
      ``coeff`` itself may be complex (e.g. ``1.0+2.0i``, ``3.0i``).

    Coupling-table completion before building ``H`` (see
    :func:`~jahn_teller_dynamics.physics.hamiltonians.djt_polynomial_hamiltonian.build_polynomial_coupling_hamiltonian`):

    - ``hermitian_completion`` — if ``true``, add missing partner rows ``(j, i, p†, c)`` for
      off-diagonal ``(i, j, p, c)`` (default ``true``).
    - ``diagonal_completion`` — if ``true``, for each diagonal row ``(i, i, p, c)`` add
      ``(s, s, p†, c)`` on other orbitals ``s`` (``p†`` = Hermitian conjugate of the phonon
      expression, e.g. ``qi+`` → ``qi-``; default ``false``).
    - ``permutation_completion`` — if ``true``, after diagonal completion complete each
      product for coordinate-factor permutations. On constrained phonon spaces this adds
      rows (one per distinct ``polinom`` string, same ``coeff``). With per-mode
      ``maximum_quanta`` (tensor-product phonons), position operators commute and
      ``coeff`` is multiplied by the distinct permutation count (repeated slots are
      not double-counted) instead (default ``false``).
    - ``symmetrize_hamiltonian`` — if ``true``, replace assembled ``H`` by ``(H + H†) / 2`` when row
      completion did not already yield a Hermitian matrix (default ``true``).

    Hamiltonian assembly always uses the **lined** row-by-row builder. Parallel assembly is
    controlled only by the presence of ``[builder]`` (see above). ``hamiltonian_builder = grouped``
    in a .cfg is ignored with a warning.

    Spin–orbital operator (optional, orbital–spin basis only):

    - ``spin_orbital_operator`` — CSV filename (under ``input_folder`` / ``data_dir``) for a
      dense ``N \\times N`` matrix (:math:`N` = number of orbital–spin states), added as
      ``kron(H_{spin}, I_{phonon})`` in the Hamiltonian builder.
    - ``spin_orbital_operator_path`` — explicit path override.
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
    phonon_encut: Optional[float] = None
    use_maximum_quanta_from_modes_csv: bool = False
    order: int = 2
    num_eigs: Optional[int] = None
    eigensolver_sigma: Optional[float] = None
    eigensolver_spectral_which: str = ""
    eigensolver_tol: Optional[float] = None
    eigensolver_max_it: Optional[int] = None
    eigensolver_ncv: Optional[int] = None
    eigensolver_mode: str = ""
    eigensolver_v0_path: str = ""
    eigensolver_return_eigenvectors: Optional[bool] = None
    eigensolver_rng_seed: Optional[int] = None
    eigensolver_use_block_diagonalization: Optional[bool] = None
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
    save_hamiltonian: bool = False
    eigensolve: bool = True
    # Comma-separated Hamiltonian NPZ stems or paths (see :meth:`resolve_hamiltonian_load_paths`).
    load_hamiltonians: str = ""
    # Output file for :meth:`resolve_hamiltonian_save_path` (bare name → under ``out_dir``).
    hamiltonian_filename: str = "hamiltonian.npz"
    # Filename (or path) used when saving eigenvectors as NPZ. A bare filename
    # is resolved under ``out_dir``; an absolute / explicit relative path is
    # used as given. ``.npz`` is appended if missing.
    npz_filename: str = "eigenvectors.npz"

    tune_tuning: float = 1.0
    tune_coupling: float = 1.0
    hermitian_completion: bool = True
    diagonal_completion: bool = False
    permutation_completion: bool = False
    symmetrize_hamiltonian: bool = True
    require_hermitian: bool = True
    hamiltonian_builder: str = "lined"
    parallel_lined_coupling: bool = False
    lined_coupling_workers: Optional[int] = None
    lined_coupling_batches: int = 4
    lined_coupling_pool: str = "spawn"
    coupling_batch_index: Optional[int] = None
    coupling_batch_count: Optional[int] = None
    coupling_batch_stem: str = "coupling_part"
    save_bare_hamiltonian: bool = False
    spin_orbital_operator: str = ""
    spin_orbital_operator_path: str = ""

    # ------------------------------------------------------------------
    # Options for the split planner → builder → eigensolver pipeline
    # (:mod:`jahn_teller_dynamics.hamiltonian_planner`,
    # :mod:`jahn_teller_dynamics.hamiltonian_builder`,
    # :mod:`jahn_teller_dynamics.eigensolver`).
    # These do not affect monolithic :mod:`jahn_teller_dynamics.jtd_run` runs.
    # ------------------------------------------------------------------
    # Planner output / builder default input. Bare filename → resolved
    # under :meth:`resolve_out_dir`.
    coupling_rows_filename: str = "coupling_rows.npz"
    # Explicit override (path or bare name) for the builder's planner input.
    load_coupling_rows: str = ""
    # Coupling-row slice consumed by the builder; half-open [from, to).
    # ``None`` defaults: from=0, to=n_coupling_rows (no slicing).
    coupling_from_index: Optional[int] = None
    coupling_to_index: Optional[int] = None
    # Alternative slice spec: chunk index ``i`` out of ``N`` equal-sized chunks.
    # ``coupling_row_chunk_index`` may also be supplied via ``SLURM_ARRAY_TASK_ID``.
    coupling_row_chunk_index: Optional[int] = None
    coupling_row_chunk_count: Optional[int] = None
    # Chunk *range* spec (half-open): build chunks
    # ``[coupling_row_chunk_from, coupling_row_chunk_to)`` of ``coupling_row_chunk_count``
    # in a single builder process, producing one combined partial NPZ. Overrides
    # ``coupling_row_chunk_index`` / ``SLURM_ARRAY_TASK_ID`` when set.
    coupling_row_chunk_from: Optional[int] = None
    coupling_row_chunk_to: Optional[int] = None
    # Builder flags: which bare terms to include in *this* partial Hamiltonian.
    include_electron_energy: bool = False
    include_phonon_energy: bool = False
    # Builder output file naming. ``partial_hamiltonian_filename`` wins when set
    # (full filename); otherwise we emit ``{stem}_{from:05d}_{to:05d}.npz`` or
    # ``{stem}_bare.npz`` (bare term jobs).
    partial_hamiltonian_stem: str = "hamiltonian_part"
    partial_hamiltonian_filename: str = ""

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

    def resolve_spin_orbital_operator_path(
        self, run_dir: Optional[Path] = None
    ) -> Optional[Path]:
        """Resolve optional spin–orbital operator matrix CSV (``N×N``, orbital–spin basis)."""
        explicit = (self.spin_orbital_operator_path or "").strip()
        name = (self.spin_orbital_operator or "").strip()
        if not explicit and not name:
            return None
        rd = run_dir or get_run_dir()
        if explicit:
            p = Path(explicit).expanduser()
            if not p.is_absolute():
                p = (rd / p).resolve()
            return p.resolve()
        data = Path(self.data_dir).expanduser()
        if not data.is_absolute():
            data = (rd / data).resolve()
        return (data / name).resolve()

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

    def parse_load_hamiltonian_names(self) -> list[str]:
        """Split ``load_hamiltonians`` into non-empty name/path tokens."""
        return [
            part.strip()
            for part in (self.load_hamiltonians or "").split(",")
            if part.strip()
        ]

    def resolve_hamiltonian_save_path(self, run_dir: Optional[Path] = None) -> Path:
        """Absolute path for saving the built Hamiltonian NPZ."""
        rd = run_dir or get_run_dir()
        name = (self.hamiltonian_filename or "hamiltonian.npz").strip()
        if not name:
            name = "hamiltonian.npz"
        if not name.lower().endswith(".npz"):
            name = name + ".npz"
        p = Path(name).expanduser()
        if p.is_absolute():
            return p.resolve()
        if p.parent == Path(""):
            return (self.resolve_out_dir(rd) / p).resolve()
        return (rd / p).resolve()

    def resolve_hamiltonian_load_paths(self, run_dir: Optional[Path] = None) -> list[Path]:
        """Resolve each entry in ``load_hamiltonians`` to an existing or expected NPZ path."""
        from jahn_teller_dynamics.io.file_io.hamiltonian_npz import resolve_hamiltonian_file_path

        rd = run_dir or get_run_dir()
        search = self.resolve_out_dir(rd)
        return [
            resolve_hamiltonian_file_path(name, search_dir=search, run_dir=rd)
            for name in self.parse_load_hamiltonian_names()
        ]


_LEGACY_PVC_SECTIONS = ("PVC", "pvc")
_MODEL_SECTIONS = ("model", "Model", "MODEL")
_BUILDER_SECTIONS = ("builder", "Builder", "BUILDER")
_EIGENSOLVER_SECTIONS = ("eigensolver", "Eigensolver", "EIGENSOLVER")

# Keys that enable parallel build when found in legacy [PVC] (deprecated location).
_LEGACY_BUILDER_OPTION_NAMES = frozenset(
    {
        "parallel_lined_coupling",
        "parallel_coupling_build",
        "build_coupling_parallel",
        "lined_coupling_workers",
        "coupling_build_workers",
        "parallel_lined_workers",
        "build_coupling_workers",
        "lined_coupling_batches",
        "lined_coupling_memory_batches",
        "coupling_memory_batches",
        "memory_batches",
        "lined_coupling_pool",
    }
)


def _first_section_name(cp: ConfigParser, candidates: Sequence[str]) -> Optional[str]:
    for name in candidates:
        if cp.has_section(name):
            return name
    return None


class PVCConfigParser:
    """
    Parses .cfg files and builds :class:`PVCCalculation` objects.

    Sections: ``[essentials]``, ``[model]``, optional ``[builder]``, ``[eigensolver]``.
    Legacy ``[PVC]`` is an alias for ``[model]``.
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

        self._essentials_section = "essentials" if cp.has_section("essentials") else None
        self._model_section = _first_section_name(cp, _MODEL_SECTIONS)
        self._legacy_pvc_section = _first_section_name(cp, _LEGACY_PVC_SECTIONS)
        self._builder_section = _first_section_name(cp, _BUILDER_SECTIONS)
        self._eigensolver_section = _first_section_name(cp, _EIGENSOLVER_SECTIONS)

        if (
            self._model_section is None
            and self._legacy_pvc_section is None
            and self._essentials_section is None
            and self._eigensolver_section is None
            and self._builder_section is None
        ):
            raise ValueError(
                f"Config must contain [essentials], [model], [builder], [eigensolver], "
                f"or legacy [PVC]: {cfg_path}"
            )

        if self._legacy_pvc_section is not None:
            warnings.warn(
                f"[{self._legacy_pvc_section}] is deprecated; use [model] and [builder] "
                f"in {cfg_path}",
                DeprecationWarning,
                stacklevel=2,
            )

        self._base = self.run_dir

    def _model_sections(self) -> tuple[str, ...]:
        """[model] first, then legacy [PVC] for keys not in [model]."""
        return tuple(
            s
            for s in (self._model_section, self._legacy_pvc_section)
            if s is not None
        )

    def _essentials_sections(self) -> tuple[str, ...]:
        return tuple(
            s
            for s in (self._essentials_section, self._legacy_pvc_section)
            if s is not None
        )

    def _builder_sections(self) -> tuple[str, ...]:
        """[builder] only; legacy parallel keys may be read from [PVC] with a warning."""
        if self._builder_section is not None:
            return (self._builder_section,)
        if self._legacy_has_parallel_build_keys():
            return (self._legacy_pvc_section,)  # type: ignore[return-value]
        return ()

    def _legacy_has_parallel_build_keys(self) -> bool:
        sec = self._legacy_pvc_section
        if sec is None:
            return False
        return any(self.config.has_option(sec, k) for k in _LEGACY_BUILDER_OPTION_NAMES)

    def _eigensolver_sections_priority(self) -> tuple[str, ...]:
        return tuple(
            s
            for s in (
                self._eigensolver_section,
                self._legacy_pvc_section,
                self._essentials_section,
            )
            if s is not None
        )

    def _warn_legacy_section(self, key: str, found_in: str, target: str) -> None:
        warnings.warn(
            f"Option {key!r} in [{found_in}] is deprecated; use [{target}] in "
            f"{self.config_file_path}",
            DeprecationWarning,
            stacklevel=3,
        )

    def _get_str_in(
        self, sections: Sequence[str], opt: str, default: str = ""
    ) -> str:
        for sec in sections:
            if self.config.has_option(sec, opt):
                return self.config.get(sec, opt, fallback=default).strip()
        return default

    def _get_bool_in(self, sections: Sequence[str], opt: str) -> Optional[bool]:
        for sec in sections:
            if self.config.has_option(sec, opt):
                return self.config.getboolean(sec, opt)
        return None

    def _get_int_in(
        self, sections: Sequence[str], opt: str, default: Optional[int] = None
    ) -> Optional[int]:
        for sec in sections:
            if self.config.has_option(sec, opt):
                return self.config.getint(sec, opt, fallback=default)
        return default

    def _get_nonneg_int_maybe_expr_in(
        self, sections: Sequence[str], opt: str
    ) -> Optional[int]:
        for sec in sections:
            if self.config.has_option(sec, opt):
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

    def _get_path_in(self, sections: Sequence[str], opt: str) -> Optional[str]:
        for sec in sections:
            if not self.config.has_option(sec, opt):
                continue
            raw = self.config.get(sec, opt, fallback="").strip()
            if not raw:
                return ""
            p = Path(raw).expanduser()
            if not p.is_absolute():
                p = (self._base / p).resolve()
            return str(p)
        return None

    def _first_nonempty_str(self, *keys: str) -> str:
        for key in keys:
            for sec in self._eigensolver_sections_priority():
                if self.config.has_option(sec, key):
                    raw = self.config.get(sec, key, fallback="").strip()
                    if raw:
                        if sec != self._eigensolver_section and self._eigensolver_section:
                            self._warn_legacy_section(key, sec, "eigensolver")
                        return raw
        return ""

    def _first_bool(self, *keys: str) -> Optional[bool]:
        for key in keys:
            for sec in self._eigensolver_sections_priority():
                if self.config.has_option(sec, key):
                    if sec != self._eigensolver_section and self._eigensolver_section:
                        self._warn_legacy_section(key, sec, "eigensolver")
                    return self.config.getboolean(sec, key)
        return None

    def _first_float(self, *keys: str) -> Optional[float]:
        for key in keys:
            for sec in self._eigensolver_sections_priority():
                if self.config.has_option(sec, key):
                    if sec != self._eigensolver_section and self._eigensolver_section:
                        self._warn_legacy_section(key, sec, "eigensolver")
                    return float(self.config.get(sec, key).strip())
        return None

    def _first_int(self, *keys: str) -> Optional[int]:
        for key in keys:
            for sec in self._eigensolver_sections_priority():
                if self.config.has_option(sec, key):
                    if sec != self._eigensolver_section and self._eigensolver_section:
                        self._warn_legacy_section(key, sec, "eigensolver")
                    return int(float(self.config.get(sec, key).strip()))
        return None

    def _apply_eigensolver_settings(self, calc: PVCCalculation) -> None:
        """Load eigensolver backend and scipy ``eigsh`` / spectral options from .cfg sections."""
        es_use = self._first_nonempty_str(
            "solver", "eigensolver", "eigen_solver", "backend"
        )
        if es_use:
            calc.eigensolver = es_use

        raw = self._first_nonempty_str("num_eigs", "num-eigs", "k")
        if raw:
            raw_lower = raw.lower()
            if raw_lower in ("all", "none"):
                calc.num_eigs = None
            else:
                try:
                    calc.num_eigs = int(raw)
                except ValueError:
                    calc.num_eigs = parse_config_int_expression(raw)

        sigma = self._first_nonempty_str(
            "sigma",
            "shift",
            "target",
            "spectral_sigma",
            "eigensolver_sigma",
            "eigensolver_shift",
            "eigensolver_target",
        )
        if sigma:
            calc.eigensolver_sigma = float(sigma)

        which = self._first_nonempty_str(
            "spectral_which",
            "which",
            "eigensolver_spectral_which",
            "eigensolver_which",
        )
        if which:
            calc.eigensolver_spectral_which = which

        tol = self._first_float("tol", "eigensolver_tol", "eigsh_tol", "eps_tol")
        if tol is not None:
            calc.eigensolver_tol = tol

        max_it = self._first_int(
            "max_iter", "maxiter", "eigensolver_max_it", "eigensolver_maxiter", "eigsh_maxiter"
        )
        if max_it is not None:
            calc.eigensolver_max_it = max_it

        ncv = self._first_int("ncv", "eigensolver_ncv", "eigsh_ncv")
        if ncv is not None:
            calc.eigensolver_ncv = ncv

        mode = self._first_nonempty_str("mode", "eigensolver_mode", "eigsh_mode")
        if mode:
            calc.eigensolver_mode = mode

        v0_path = self._first_nonempty_str("v0_file", "v0", "eigensolver_v0", "eigsh_v0")
        if v0_path:
            p = Path(v0_path).expanduser()
            if not p.is_absolute():
                p = (self._base / p).resolve()
            calc.eigensolver_v0_path = str(p)

        ret_ev = self._first_bool(
            "return_eigenvectors", "eigensolver_return_eigenvectors", "eigsh_return_eigenvectors"
        )
        if ret_ev is not None:
            calc.eigensolver_return_eigenvectors = ret_ev

        rng_seed = self._first_int("rng_seed", "eigensolver_rng_seed", "eigsh_rng_seed", "seed")
        if rng_seed is not None:
            calc.eigensolver_rng_seed = rng_seed

        use_block = self._first_bool(
            "use_block_diagonalization",
            "eigensolver_use_block_diagonalization",
            "block_diagonalization",
        )
        if use_block is not None:
            calc.eigensolver_use_block_diagonalization = use_block

        if (v := self._first_bool("require_hermitian")) is not None:
            calc.require_hermitian = v
        if (v := self._first_bool("allow_non_hermitian")) is not None:
            calc.require_hermitian = not v

    def _apply_essentials(self, calc: PVCCalculation) -> None:
        ess = self._essentials_sections()
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

        if (d := self._get_path_in(ess, "data_dir")) is not None:
            calc.data_dir = d
        out = self._get_path_in(ess, "out_dir") or self._get_path_in(ess, "out-dir")
        if out is not None and out:
            calc.out_dir = out

        if (u := self._get_bool_in(ess, "use_sparse")) is not None:
            calc.use_sparse = u

        sep_val = self._get_str_in(ess, "separator")
        if sep_val:
            calc.separator = sep_val if sep_val else ";"

        for opt, attr in (
            ("save_npz", "save_npz"),
            ("save_csv", "save_csv"),
            ("save_hamiltonian", "save_hamiltonian"),
            ("save_hamiltonian_npz", "save_hamiltonian"),
            ("eigensolve", "eigensolve"),
        ):
            if (v := self._get_bool_in(ess, opt)) is not None:
                setattr(calc, attr, v)

        for key in ("load_hamiltonians", "load_hamiltonian", "hamiltonians_to_load"):
            v = self._get_str_in(ess, key)
            if v:
                calc.load_hamiltonians = v
                break

        for key in ("hamiltonian_filename", "hamiltonian_npz", "hamiltonian_output"):
            v = self._get_str_in(ess, key)
            if v:
                calc.hamiltonian_filename = v
                break

        for key in ("npz_filename", "eigenvectors_npz_filename", "output_npz_filename"):
            v = self._get_str_in(ess, key)
            if v:
                calc.npz_filename = v
                break

        self._apply_phonon_truncation(calc, ess)

    def _apply_phonon_truncation(
        self, calc: PVCCalculation, sections: Sequence[str]
    ) -> None:
        """
        Phonon Hilbert-space truncation keys.

        May appear in ``[essentials]`` and/or ``[model]`` (``[model]`` is applied second and
        overrides ``[essentials]`` when both set the same option).
        """
        if not sections:
            return
        for opt, attr in [
            ("order", "order"),
            ("maximum_number_of_vibrational_quanta", "maximum_number_of_vibrational_quanta"),
            ("maximum_number_of_quanta_per_dim", "maximum_number_of_quanta_per_dim"),
        ]:
            if (v := self._get_nonneg_int_maybe_expr_in(sections, opt)) is not None:
                setattr(calc, attr, v)

        for key in ("phonon_encut", "phonon_energy_cutoff", "phonon_energy_cut"):
            v = self._get_str_in(sections, key)
            if v:
                encut = float(v)
                if encut < 0:
                    raise ValueError(f"{key} must be non-negative; got {encut!r}")
                calc.phonon_encut = encut
                break

        if (v := self._get_bool_in(sections, "use_maximum_quanta_from_modes_csv")) is not None:
            calc.use_maximum_quanta_from_modes_csv = v

    def _apply_model(self, calc: PVCCalculation) -> None:
        model = self._model_sections()
        if not model:
            return

        for opt, attr in [
            ("electron_energies", "electron_energies"),
            ("modes", "modes"),
            ("coupling", "coupling_csv"),
        ]:
            val = self._get_str_in(model, opt)
            if val:
                setattr(calc, attr, val)

        for opt, attr, alts in [
            ("electron_energies_path", "electron_energies_path", ["electron_energies_csv_path"]),
            ("modes_path", "modes_path", ["modes_csv_path"]),
            ("coupling_path", "coupling_path", ["polynomial_coupling_path", "pvc_coupling_path"]),
        ]:
            val = self._get_path_in(model, opt)
            for alt in alts:
                if val is None:
                    val = self._get_path_in(model, alt)
            if val is not None:
                setattr(calc, attr, val)

        raw = self._get_str_in(model, "modes_to_use")
        if raw:
            raw_lower = raw.strip().lower()
            if raw_lower in ("all", "none", ""):
                calc.modes_to_use = None
            else:
                try:
                    calc.modes_to_use = [int(x.strip()) for x in raw.split(",") if x.strip()]
                except ValueError:
                    calc.modes_to_use = None

        self._apply_phonon_truncation(calc, model)

        if (v := self._get_nonneg_int_maybe_expr_in(model, "exp_approximation_order")) is not None:
            calc.exp_approximation_order = v

        for key, attr in (("tune_tuning", "tune_tuning"), ("tune_coupling", "tune_coupling")):
            v = self._get_str_in(model, key)
            if v:
                setattr(calc, attr, float(v))

        for opt, attr in (
            ("hermitian_completion", "hermitian_completion"),
            ("diagonal_completion", "diagonal_completion"),
            ("permutation_completion", "permutation_completion"),
            ("symmetrize_hamiltonian", "symmetrize_hamiltonian"),
            ("dimensionless_coordinates", "dimensionless_coordinates"),
            ("null_point_vib", "null_point_vib"),
        ):
            if (v := self._get_bool_in(model, opt)) is not None:
                setattr(calc, attr, v)

        for opt, attr in (
            ("spin_orbital_operator", "spin_orbital_operator"),
            ("orbital_spin_operator", "spin_orbital_operator"),
        ):
            v = self._get_str_in(model, opt)
            if v:
                calc.spin_orbital_operator = v
                break

        for key in (
            "spin_orbital_operator_path",
            "orbital_spin_operator_path",
            "spin_orbital_operator_csv_path",
        ):
            v = self._get_str_in(model, key)
            if v:
                calc.spin_orbital_operator_path = v
                break

        for key in ("hamiltonian_builder", "hamiltonian_build", "pvc_hamiltonian_builder"):
            v = self._get_str_in(model, key)
            if v:
                chosen = normalize_hamiltonian_builder(v)
                if chosen != "lined":
                    warnings.warn(
                        f"hamiltonian_builder={v!r} is ignored; jtd_run always uses the lined "
                        f"builder ({self.config_file_path})",
                        UserWarning,
                        stacklevel=2,
                    )
                break

    def _apply_builder(self, calc: PVCCalculation) -> None:
        has_builder_section = self._builder_section is not None
        legacy_parallel = self._legacy_has_parallel_build_keys()
        builder_secs = self._builder_sections()

        if has_builder_section:
            calc.parallel_lined_coupling = True
        elif legacy_parallel:
            if self._legacy_pvc_section:
                self._warn_legacy_section(
                    "parallel_lined_coupling (or related)",
                    self._legacy_pvc_section,
                    "builder",
                )
            calc.parallel_lined_coupling = True
        else:
            calc.parallel_lined_coupling = False
            return

        if not builder_secs:
            return

        for opt, attr in (
            ("parallel_lined_coupling", "parallel_lined_coupling"),
            ("parallel_coupling_build", "parallel_lined_coupling"),
            ("build_coupling_parallel", "parallel_lined_coupling"),
        ):
            if (v := self._get_bool_in(builder_secs, opt)) is not None:
                calc.parallel_lined_coupling = v
                break

        for key in (
            "lined_coupling_workers",
            "coupling_build_workers",
            "parallel_lined_workers",
            "build_coupling_workers",
        ):
            v = self._get_str_in(builder_secs, key)
            if v:
                calc.lined_coupling_workers = int(float(v))
                break

        for key in (
            "lined_coupling_batches",
            "lined_coupling_memory_batches",
            "coupling_memory_batches",
            "memory_batches",
        ):
            v = self._get_str_in(builder_secs, key)
            if v:
                calc.lined_coupling_batches = max(1, int(float(v)))
                break

        pool = self._get_str_in(builder_secs, "lined_coupling_pool")
        if pool:
            calc.lined_coupling_pool = pool

        for key in ("coupling_batch_index", "coupling_batch_id"):
            v = self._get_str_in(builder_secs, key)
            if v:
                calc.coupling_batch_index = int(float(v))
                break
        for key in ("coupling_batch_count", "coupling_batch_total"):
            v = self._get_str_in(builder_secs, key)
            if v:
                calc.coupling_batch_count = max(1, int(float(v)))
                break
        stem = self._get_str_in(builder_secs, "coupling_batch_stem")
        if stem:
            calc.coupling_batch_stem = stem
        if (v := self._get_bool_in(builder_secs, "save_bare_hamiltonian")) is not None:
            calc.save_bare_hamiltonian = v

        for key in ("coupling_rows_filename", "coupling_rows_npz", "coupling_rows_output"):
            v = self._get_str_in(builder_secs, key)
            if v:
                calc.coupling_rows_filename = v
                break
        for key in ("load_coupling_rows", "coupling_rows_input", "coupling_rows_path"):
            v = self._get_str_in(builder_secs, key)
            if v:
                calc.load_coupling_rows = v
                break
        for key in ("coupling_from_index", "from_index", "coupling_row_from"):
            v = self._get_str_in(builder_secs, key)
            if v:
                calc.coupling_from_index = int(float(v))
                break
        for key in ("coupling_to_index", "to_index", "coupling_row_to"):
            v = self._get_str_in(builder_secs, key)
            if v:
                calc.coupling_to_index = int(float(v))
                break
        for key in ("coupling_row_chunk_index", "coupling_chunk_index", "row_chunk_index"):
            v = self._get_str_in(builder_secs, key)
            if v:
                calc.coupling_row_chunk_index = int(float(v))
                break
        for key in ("coupling_row_chunk_count", "coupling_chunk_count", "row_chunk_count"):
            v = self._get_str_in(builder_secs, key)
            if v:
                calc.coupling_row_chunk_count = max(1, int(float(v)))
                break
        for key in ("coupling_row_chunk_from", "coupling_chunk_from", "row_chunk_from"):
            v = self._get_str_in(builder_secs, key)
            if v:
                calc.coupling_row_chunk_from = int(float(v))
                break
        for key in ("coupling_row_chunk_to", "coupling_chunk_to", "row_chunk_to"):
            v = self._get_str_in(builder_secs, key)
            if v:
                calc.coupling_row_chunk_to = int(float(v))
                break
        for opt, attr in (
            ("include_electron_energy", "include_electron_energy"),
            ("include_phonon_energy", "include_phonon_energy"),
        ):
            if (v := self._get_bool_in(builder_secs, opt)) is not None:
                setattr(calc, attr, v)
        for key in ("partial_hamiltonian_stem", "partial_h_stem", "partial_hamiltonian_prefix"):
            v = self._get_str_in(builder_secs, key)
            if v:
                calc.partial_hamiltonian_stem = v
                break
        for key in ("partial_hamiltonian_filename", "partial_hamiltonian_npz", "partial_h_filename"):
            v = self._get_str_in(builder_secs, key)
            if v:
                calc.partial_hamiltonian_filename = v
                break

    def build_calculation(self) -> PVCCalculation:
        """Build :class:`PVCCalculation` from the parsed config."""
        calc = PVCCalculation()
        calc.hamiltonian_builder = "lined"

        self._apply_essentials(calc)
        self._apply_model(calc)
        self._apply_builder(calc)
        self._apply_eigensolver_settings(calc)

        renv_use = self._first_nonempty_str("running_environment")
        if renv_use:
            calc.running_environment = renv_use

        return calc
