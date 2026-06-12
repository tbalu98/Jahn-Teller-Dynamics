"""
Jahn-Teller dynamics (JTD) driver: build and diagonalize a vibronic coupling
Hamiltonian from CSV inputs (polynomial / dJT PVC model).

Expected inputs (under ``--data-dir`` by default):

- ``electron_energies.csv`` — diagonal electronic energies (``el_state``, ``energy`` or ``state``, ``value``).
  ``el_state`` may be any non-empty string label (e.g. ``ground``, ``T2``, ``A``); row order defines
  the internal basis index used by coupling rows.
- ``modes.csv`` — mode frequencies (``mode``, ``energy`` or ``omega``)
- ``coupling.csv`` — coupling rows (``el_state_1``, ``el_state_2``, ``expression``, ``coeff``).
  ``el_state_1`` / ``el_state_2`` may use the same string labels as ``electron_energies.csv``
  (legacy numeric 1-based indices still accepted).
  Legacy column names ``polinom`` / ``polynomial`` are still accepted for the coupling expression.
  ``coeff`` may be complex (e.g. ``1.0+2.0i``, ``3.0i``) for spin–orbital-style couplings without
  a separate spin subsystem.

Phonon space: **constrained** multimode Hilbert space (``maximum_number_of_vibrational_quanta``
or legacy ``order``: ``sum_i n_i <= N``), **energy-cutoff** truncation (``phonon_encut``:
``sum_i n_i e_i <= E``), tensor-product truncation with ``maximum_number_of_quanta_per_dim``
(same ``N`` for every mode), or **per-mode** tensor-product truncation with
``use_maximum_quanta_from_modes_csv = true`` (each mode's ``N_i`` is read from the
``maximum_quanta`` column of ``modes.csv``). Set only one truncation key.

Spectral / eigensolver knobs: dedicated ``[eigensolver]`` section (recommended) or legacy keys in
``[PVC]`` / ``[essentials]`` — backend, ``num_eigs``, ``sigma``, ``spectral_which``, ``tol``,
``max_iter``, ``ncv``, ``mode``, ``v0_file``, ``return_eigenvectors``, ``rng_seed``. CLI mirrors
the main sparse options (``--num-eigs``, ``--eigensolver-sigma``, …).

Coupling exponentials (``exp(...)`` in coupling CSV expressions): ``exp_approximation_order`` (INI,
non-negative integer) selects a truncated Taylor sum for the matrix exponential; omit for exact ``expm``. CLI: ``--exp-approximation-order``.

Coupling CSV ``coeff`` scaling: ``tune_tuning`` (INI / ``--tune-tuning``) multiplies ``coeff`` when
``el_state_1`` and ``el_state_2`` resolve to the same 1-based index; ``tune_coupling`` when they differ.
Defaults are ``1.0``.

Coupling completion: ``hermitian_completion``, ``diagonal_completion``, ``permutation_completion``,
and ``symmetrize_hamiltonian``
in ``[PVC]`` / ``[essentials]`` (partner rows, diagonal replication, and ``(H+H†)/2`` when needed).

Hamiltonian assembly (``[PVC]`` / ``[essentials]``): ``hamiltonian_builder = lined`` (default,
row-by-row CSV) or ``grouped`` (one orbital :math:`N \\times N` operator per expression).
Parallel lined build (``parallel_lined_coupling``): rows sorted by ``polinom``, processed in
``lined_coupling_batches`` memory waves with cache cleared between waves; optional
``lined_coupling_workers`` parallel **processes** per wave (Unix fork; ``--ntasks=1`` on Slurm).

Hamiltonian checkpointing (``[PVC]`` / ``[essentials]``):

- ``save_hamiltonian`` — write the built ``H`` to NPZ (``hamiltonian_filename``, default
  ``hamiltonian.npz`` under ``output_folder``).
- ``load_hamiltonians`` — comma-separated NPZ stems/paths; skip CSV build, sum loaded
  matrices, then diagonalize (unless ``eigensolve = false``).
- ``eigensolve`` — if ``false``, only build and/or save ``H`` (for partial coupling jobs).

Optional ``spin_orbital_operator`` / ``spin_orbital_operator_path``: dense ``N \\times N`` CSV
(``N`` = orbital–spin states) added as ``kron(H_{spin}, I_{phonon})``.

Run (from repo root)::

    python3 -m jahn_teller_dynamics.jtd_run

MPI clusters (PETSc/SLEPc): launch with ``srun`` / ``mpirun`` using an interpreter linked to PETSc,
set ``eigensolver = slepc`` or ``running_environment = multiprocessor`` in ``[PVC]``. Only rank 0 writes
CSV/NPZ output when running with multiple MPI ranks.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from jahn_teller_dynamics.io.config.pvc_config import (
    PVCConfigParser,
    PVCCalculation,
    normalize_hamiltonian_builder,
)
from jahn_teller_dynamics.io.utils.run_context import RunContext
from jahn_teller_dynamics.io.utils.timestamp_print import print_ts


def _default_data_dir() -> Path:
    return RunContext.from_cwd().data_dir("dJT_data") / "trial_inputs"


def _pvc_mpi_secondary_rank_skip_file_io(backend_norm: str) -> bool:
    """When SLEPc diagonalizes under MPI, avoid duplicate writes from non-root ranks."""
    if backend_norm != "slepc":
        return False
    try:
        from petsc4py import PETSc

        if not PETSc.Sys.isInitialized():
            return False
        w = PETSc.COMM_WORLD
        return bool(w.getSize() > 1 and w.getRank() != 0)
    except Exception:
        return False


def _human_bytes(n: float) -> str:
    """Format a byte count as a short human-readable string."""
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    size = float(n)
    for u in units:
        if size < 1024.0 or u == units[-1]:
            return f"{size:.0f} {u}" if u == "B" else f"{size:.2f} {u}"
        size /= 1024.0
    return f"{size:.2f} PiB"


def describe_hamiltonian_matrix(H: Any) -> str:
    """
    Return a one-line description of the Hamiltonian matrix backing the
    eigensolver: wrapper class, underlying object (e.g. ``csr_matrix``),
    shape, sparse format / nnz / density (sparse case) or dense memory,
    dtype, and rough storage size.
    """
    wrapper = getattr(H, "matrix", H)
    inner = getattr(wrapper, "matrix", wrapper)

    parts: list[str] = [
        f"wrapper={type(wrapper).__name__}",
        f"backend={type(inner).__module__}.{type(inner).__name__}",
    ]
    shape = getattr(inner, "shape", None)
    if shape is not None:
        parts.append(f"shape={tuple(int(x) for x in shape)}")

    nnz = getattr(inner, "nnz", None)
    fmt = getattr(inner, "format", None)
    is_sparse = nnz is not None
    if is_sparse:
        if fmt:
            parts.append(f"format={fmt}")
        parts.append(f"nnz={int(nnz)}")
        if shape is not None:
            n_total = int(shape[0]) * int(shape[1])
            if n_total > 0:
                density = float(nnz) / float(n_total)
                parts.append(f"density={density:.3e}")
        try:
            mem = int(inner.data.nbytes) + int(inner.indices.nbytes) + int(inner.indptr.nbytes)
            parts.append(f"mem≈{_human_bytes(mem)}")
        except AttributeError:
            pass
    else:
        nbytes = getattr(inner, "nbytes", None)
        if nbytes is not None:
            parts.append(f"mem≈{_human_bytes(int(nbytes))}")

    dtype = getattr(inner, "dtype", None)
    if dtype is not None:
        parts.append(f"dtype={dtype}")

    return ", ".join(parts)


def _calc_to_argparse_defaults(calc: PVCCalculation) -> dict[str, Any]:
    return {
        "data_dir": calc.data_dir,
        "out_dir": calc.out_dir,
        "electron_energies": calc.electron_energies,
        "modes": calc.modes,
        "coupling_csv": calc.coupling_csv,
        "electron_energies_path": calc.electron_energies_path,
        "modes_path": calc.modes_path,
        "coupling_path": calc.coupling_path,
        "modes_to_use": calc.modes_to_use,
        "order": calc.order,
        "maximum_number_of_vibrational_quanta": calc.maximum_number_of_vibrational_quanta,
        "maximum_number_of_quanta_per_dim": calc.maximum_number_of_quanta_per_dim,
        "phonon_encut": calc.phonon_encut,
        "use_maximum_quanta_from_modes_csv": calc.use_maximum_quanta_from_modes_csv,
        "num_eigs": calc.num_eigs,
        "use_sparse": calc.use_sparse,
        "eigensolver": calc.eigensolver,
        "running_environment": calc.running_environment,
        "separator": calc.separator,
        "dimensionless_coordinates": calc.dimensionless_coordinates,
        "null_point_vib": calc.null_point_vib,
        "save_npz": calc.save_npz,
        "save_csv": calc.save_csv,
        "eigensolver_sigma": calc.eigensolver_sigma,
        "eigensolver_spectral_which": calc.eigensolver_spectral_which,
        "eigensolver_tol": calc.eigensolver_tol,
        "eigensolver_max_it": calc.eigensolver_max_it,
        "eigensolver_ncv": calc.eigensolver_ncv,
        "exp_approximation_order": calc.exp_approximation_order,
        "npz_filename": calc.npz_filename,
        "tune_tuning": calc.tune_tuning,
        "tune_coupling": calc.tune_coupling,
        "hermitian_completion": calc.hermitian_completion,
        "diagonal_completion": calc.diagonal_completion,
        "permutation_completion": calc.permutation_completion,
        "symmetrize_hamiltonian": calc.symmetrize_hamiltonian,
        "require_hermitian": calc.require_hermitian,
        "hamiltonian_builder": calc.hamiltonian_builder,
        "parallel_lined_coupling": calc.parallel_lined_coupling,
        "lined_coupling_workers": calc.lined_coupling_workers,
        "lined_coupling_batches": calc.lined_coupling_batches,
        "lined_coupling_pool": calc.lined_coupling_pool,
        "coupling_batch_index": calc.coupling_batch_index,
        "coupling_batch_count": calc.coupling_batch_count,
        "coupling_batch_stem": calc.coupling_batch_stem,
        "save_bare_hamiltonian": calc.save_bare_hamiltonian,
        "spin_orbital_operator": calc.spin_orbital_operator,
        "spin_orbital_operator_path": calc.spin_orbital_operator_path,
        "save_hamiltonian": calc.save_hamiltonian,
        "eigensolve": calc.eigensolve,
        "load_hamiltonians": calc.load_hamiltonians,
        "hamiltonian_filename": calc.hamiltonian_filename,
        "coupling_rows_filename": calc.coupling_rows_filename,
        "load_coupling_rows": calc.load_coupling_rows,
        "coupling_from_index": calc.coupling_from_index,
        "coupling_to_index": calc.coupling_to_index,
        "coupling_row_chunk_index": calc.coupling_row_chunk_index,
        "coupling_row_chunk_count": calc.coupling_row_chunk_count,
        "coupling_row_chunk_from": calc.coupling_row_chunk_from,
        "coupling_row_chunk_to": calc.coupling_row_chunk_to,
        "include_electron_energy": calc.include_electron_energy,
        "include_phonon_energy": calc.include_phonon_energy,
        "partial_hamiltonian_stem": calc.partial_hamiltonian_stem,
        "partial_hamiltonian_filename": calc.partial_hamiltonian_filename,
    }


def _args_to_calculation(args: argparse.Namespace, run_dir: Path) -> PVCCalculation:
    modes_to_use = getattr(args, "modes_to_use", None)
    if isinstance(modes_to_use, str):
        modes_to_use = [int(x.strip()) for x in modes_to_use.split(",") if x.strip()] or None
    elif modes_to_use is not None and not isinstance(modes_to_use, list):
        modes_to_use = None

    return PVCCalculation(
        data_dir=args.data_dir,
        out_dir=getattr(args, "out_dir", "") or "",
        electron_energies=args.electron_energies,
        modes=args.modes,
        coupling_csv=args.coupling_csv,
        electron_energies_path=getattr(args, "electron_energies_path", "") or "",
        modes_path=getattr(args, "modes_path", "") or "",
        coupling_path=getattr(args, "coupling_path", "") or "",
        modes_to_use=modes_to_use,
        maximum_number_of_vibrational_quanta=int(
            getattr(args, "maximum_number_of_vibrational_quanta", 0)
        ),
        maximum_number_of_quanta_per_dim=int(getattr(args, "maximum_number_of_quanta_per_dim", 0)),
        phonon_encut=getattr(args, "phonon_encut", None),
        use_maximum_quanta_from_modes_csv=bool(
            getattr(args, "use_maximum_quanta_from_modes_csv", False)
        ),
        order=int(args.order),
        num_eigs=args.num_eigs,
        use_sparse=bool(args.use_sparse),
        eigensolver=str(getattr(args, "eigensolver", "") or ""),
        running_environment=str(getattr(args, "running_environment", "") or ""),
        separator=str(getattr(args, "separator", ";")),
        dimensionless_coordinates=bool(getattr(args, "dimensionless_coordinates", True)),
        null_point_vib=bool(getattr(args, "null_point_vib", True)),
        save_npz=bool(getattr(args, "save_npz", False)),
        save_csv=bool(getattr(args, "save_csv", True)),
        eigensolver_sigma=getattr(args, "eigensolver_sigma", None),
        eigensolver_spectral_which=str(
            getattr(args, "eigensolver_spectral_which", "") or ""
        ),
        eigensolver_tol=getattr(args, "eigensolver_tol", None),
        eigensolver_max_it=getattr(args, "eigensolver_max_it", None),
        eigensolver_ncv=getattr(args, "eigensolver_ncv", None),
        exp_approximation_order=getattr(args, "exp_approximation_order", None),
        npz_filename=str(getattr(args, "npz_filename", "") or "eigenstates.npz"),
        tune_tuning=float(getattr(args, "tune_tuning", 1.0)),
        tune_coupling=float(getattr(args, "tune_coupling", 1.0)),
        hermitian_completion=bool(getattr(args, "hermitian_completion", True)),
        diagonal_completion=bool(getattr(args, "diagonal_completion", False)),
        permutation_completion=bool(getattr(args, "permutation_completion", False)),
        symmetrize_hamiltonian=bool(getattr(args, "symmetrize_hamiltonian", True)),
        require_hermitian=bool(getattr(args, "require_hermitian", True)),
        hamiltonian_builder=normalize_hamiltonian_builder(
            str(getattr(args, "hamiltonian_builder", "lined") or "lined")
        ),
        parallel_lined_coupling=bool(getattr(args, "parallel_lined_coupling", False)),
        lined_coupling_workers=getattr(args, "lined_coupling_workers", None),
        lined_coupling_batches=int(getattr(args, "lined_coupling_batches", 4)),
        lined_coupling_pool=str(getattr(args, "lined_coupling_pool", "") or "spawn"),
        coupling_batch_index=getattr(args, "coupling_batch_index", None),
        coupling_batch_count=getattr(args, "coupling_batch_count", None),
        coupling_batch_stem=str(getattr(args, "coupling_batch_stem", "") or "coupling_part"),
        save_bare_hamiltonian=bool(getattr(args, "save_bare_hamiltonian", False)),
        spin_orbital_operator=str(getattr(args, "spin_orbital_operator", "") or ""),
        spin_orbital_operator_path=str(
            getattr(args, "spin_orbital_operator_path", "") or ""
        ),
        save_hamiltonian=bool(getattr(args, "save_hamiltonian", False)),
        eigensolve=bool(getattr(args, "eigensolve", True)),
        load_hamiltonians=str(getattr(args, "load_hamiltonians", "") or ""),
        hamiltonian_filename=str(
            getattr(args, "hamiltonian_filename", "") or "hamiltonian.npz"
        ),
        coupling_rows_filename=str(
            getattr(args, "coupling_rows_filename", "") or "coupling_rows.npz"
        ),
        load_coupling_rows=str(getattr(args, "load_coupling_rows", "") or ""),
        coupling_from_index=getattr(args, "coupling_from_index", None),
        coupling_to_index=getattr(args, "coupling_to_index", None),
        coupling_row_chunk_index=getattr(args, "coupling_row_chunk_index", None),
        coupling_row_chunk_count=getattr(args, "coupling_row_chunk_count", None),
        coupling_row_chunk_from=getattr(args, "coupling_row_chunk_from", None),
        coupling_row_chunk_to=getattr(args, "coupling_row_chunk_to", None),
        include_electron_energy=bool(getattr(args, "include_electron_energy", False)),
        include_phonon_energy=bool(getattr(args, "include_phonon_energy", False)),
        partial_hamiltonian_stem=str(
            getattr(args, "partial_hamiltonian_stem", "") or "hamiltonian_part"
        ),
        partial_hamiltonian_filename=str(
            getattr(args, "partial_hamiltonian_filename", "") or ""
        ),
    )


def read_pvc_cfg(cfg_path: str) -> dict[str, Any]:
    """
    Read an INI-style .cfg file and return argparse-defaults for :mod:`jahn_teller_dynamics.jtd_run`.

    CLI args always override config values.
    """
    run_ctx = RunContext.from_cwd()
    parser = PVCConfigParser(cfg_path, run_dir=run_ctx.run_dir)
    calc = parser.build_calculation()
    return _calc_to_argparse_defaults(calc)


def _build_parser_with_cfg_defaults(argv: list[str] | None) -> argparse.ArgumentParser:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to INI-style .cfg ([essentials], [model], optional [builder], [eigensolver]).",
    )
    ns, _ = pre.parse_known_args(argv)

    p = build_arg_parser()
    # Merge .cfg into argparse defaults when --config is passed, and also when it is omitted
    # but build_arg_parser() supplies a default path (same behavior users expect as LVC.py).
    pre_cfg = (getattr(ns, "config", None) or "").strip()
    cfg_path = pre_cfg or (p.get_default("config") or "").strip()
    if cfg_path:
        cfg_defaults = read_pvc_cfg(cfg_path)
        p.set_defaults(**cfg_defaults)
    return p


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build and diagonalize a Jahn-Teller vibronic Hamiltonian (polynomial coupling) from CSV inputs."
    )
    p.add_argument(
        "--config",
        type=str,
        default="config_files/PVC_trial_inputs.cfg",
        help="Path to INI-style .cfg ([essentials], [model], optional [builder], [eigensolver]).",
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default=str(_default_data_dir()),
        help="Directory containing PVC CSV files (electron energies, modes, polynomial coupling).",
    )
    p.add_argument(
        "--electron-energies",
        type=str,
        default="electron_energies.csv",
        help="Diagonal electronic energies CSV (el_state, energy) or (state, value).",
    )
    p.add_argument(
        "--modes",
        type=str,
        default="modes.csv",
        help="Modes CSV (mode, energy or omega).",
    )
    p.add_argument(
        "--coupling",
        dest="coupling_csv",
        type=str,
        default="coupling.csv",
        help="Coupling CSV (el_state_1, el_state_2, expression, coeff); polinom/polynomial alias for expression column.",
    )
    p.add_argument(
        "--modes-to-use",
        type=str,
        default="",
        help="Comma-separated mode indices to include: for numeric 'mode' column, the "
        "mode *values* in the CSV (e.g. 1,2,3); for string mode names, 1-based row index "
        "in file order. Default: all.",
    )
    p.add_argument(
        "--electron-energies-path",
        type=str,
        default="",
        help="Explicit path to electron energies CSV (overrides --data-dir/--electron-energies).",
    )
    p.add_argument(
        "--modes-path",
        type=str,
        default="",
        help="Explicit path to modes CSV (overrides --data-dir/--modes).",
    )
    p.add_argument(
        "--coupling-path",
        type=str,
        default="",
        help="Explicit path to coupling CSV (overrides --data-dir/--coupling).",
    )
    p.add_argument(
        "--maximum-number-of-vibrational-quanta",
        dest="maximum_number_of_vibrational_quanta",
        type=int,
        default=0,
        help="Constrained multimode phonons: sum_i n_i <= N. Use 0 to fall back to --order.",
    )
    p.add_argument(
        "--maximum-number-of-quanta-per-dim",
        dest="maximum_number_of_quanta_per_dim",
        type=int,
        default=0,
        help="Tensor-product multimode phonons: per-mode cutoff n_i <= N per mode.",
    )
    p.add_argument(
        "--phonon-encut",
        dest="phonon_encut",
        type=float,
        default=None,
        metavar="E",
        help="Constrained phonons by excitation energy: sum_i n_i e_i <= E (cm^-1, from modes CSV).",
    )
    p.add_argument(
        "--use-maximum-quanta-from-modes-csv",
        dest="use_maximum_quanta_from_modes_csv",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Tensor-product phonons with per-mode cutoff read from the 'maximum_quanta' "
            "column of modes.csv (one integer N_i per mode). Mutually exclusive with "
            "--phonon-encut, --maximum-number-of-vibrational-quanta and "
            "--maximum-number-of-quanta-per-dim."
        ),
    )
    p.add_argument(
        "--order",
        type=int,
        default=2,
        help="Legacy: maximum total phonon number when constrained and maximum_number_of_vibrational_quanta is 0.",
    )
    p.add_argument(
        "--exp-approximation-order",
        dest="exp_approximation_order",
        type=int,
        default=None,
        metavar="N",
        help="Taylor order N for coupling exp(...) (sum A^k/k! through k=N); omit for exact scipy expm.",
    )
    p.add_argument(
        "--tune-tuning",
        dest="tune_tuning",
        type=float,
        default=1.0,
        metavar="X",
        help="Multiply coupling CSV coeff when el_state_1 and el_state_2 resolve to the same index (default 1).",
    )
    p.add_argument(
        "--tune-coupling",
        dest="tune_coupling",
        type=float,
        default=1.0,
        metavar="X",
        help="Multiply coupling CSV coeff when el_state_1 and el_state_2 resolve to different indices (default 1).",
    )
    p.add_argument(
        "--hermitian-completion",
        dest="hermitian_completion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add missing (j,i,p†,c) partner rows for off-diagonal coupling (default: on).",
    )
    p.add_argument(
        "--diagonal-completion",
        dest="diagonal_completion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Replicate each diagonal (i,i,p,c) row to all other electronic states (default: off).",
    )
    p.add_argument(
        "--permutation-completion",
        dest="permutation_completion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add rows with permuted coordinate factors in each multiplicative term (default: off).",
    )
    p.add_argument(
        "--symmetrize-hamiltonian",
        dest="symmetrize_hamiltonian",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply (H + H†) / 2 when row completion did not yield a Hermitian H (default: on).",
    )
    p.add_argument(
        "--require-hermitian",
        dest="require_hermitian",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Abort if H fails the Hermiticity check (default: on). Use --no-require-hermitian "
        "to diagonalize with scipy eigs / SLEPc NHEP.",
    )
    p.add_argument(
        "--hamiltonian-builder",
        dest="hamiltonian_builder",
        type=str,
        default="lined",
        metavar="MODE",
        help="Hamiltonian assembly: lined (row-by-row, default) or grouped (one N×N orbital op per expression).",
    )
    p.add_argument(
        "--parallel-lined-coupling",
        dest="parallel_lined_coupling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Parallel threaded build of coupling rows (lined builder only; default: off).",
    )
    p.add_argument(
        "--lined-coupling-workers",
        dest="lined_coupling_workers",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Parallel worker processes for memory batches in parallel lined build "
            "(default: CPU count; requires fork, set 1 for serial batches)."
        ),
    )
    p.add_argument(
        "--lined-coupling-batches",
        dest="lined_coupling_batches",
        type=int,
        default=4,
        metavar="N",
        help="Memory batches: sort rows by polinom, cache phonon ops per batch, clear cache between batches (default: 4).",
    )
    p.add_argument(
        "--spin-orbital-operator",
        dest="spin_orbital_operator",
        type=str,
        default="",
        help="CSV filename for N×N spin–orbital operator (orbital–spin basis; under data-dir).",
    )
    p.add_argument(
        "--spin-orbital-operator-path",
        dest="spin_orbital_operator_path",
        type=str,
        default="",
        help="Explicit path to spin–orbital operator CSV (overrides --spin-orbital-operator).",
    )
    p.add_argument(
        "--num-eigs",
        type=int,
        default=None,
        help="Number of lowest eigenvalues/eigenvectors (default: all).",
    )
    p.add_argument("--use-sparse", action="store_true", default=True, help="Use sparse operators/solver (default).")
    p.add_argument("--no-sparse", dest="use_sparse", action="store_false", help="Force dense operators/solver.")
    p.add_argument(
        "--eigensolver",
        type=str,
        default="",
        metavar="MODE",
        help="Eigen solver: sparse/scipy/eigsh (SciPy Lanczos), dense, slepc/PETSc (MPI-capable); overrides default from use-sparse.",
    )
    p.add_argument(
        "--running-environment",
        dest="running_environment",
        type=str,
        default="",
        metavar="ENV",
        help="If set to multiprocessor/mpi/hpc parallel (without an explicit --eigensolver), select SLEPc.",
    )
    p.add_argument(
        "--eigensolver-sigma",
        dest="eigensolver_sigma",
        type=float,
        default=None,
        metavar="SHIFT",
        help="Shift/target eigenvalue for shift–invert (scipy eigsh sigma; SLEPc ST); optional.",
    )
    p.add_argument(
        "--eigensolver-spectral-which",
        dest="eigensolver_spectral_which",
        type=str,
        default="",
        metavar="MODE",
        help="Spectral selection: smallest_real, largest_real, nearest (needs --eigensolver-sigma), SA/LM, …",
    )
    p.add_argument(
        "--eigensolver-tol",
        dest="eigensolver_tol",
        type=float,
        default=None,
        metavar="TOL",
        help="scipy eigsh ARPACK tolerance (sparse backend; default 1e-10).",
    )
    p.add_argument(
        "--eigensolver-max-it",
        dest="eigensolver_max_it",
        type=int,
        default=None,
        metavar="N",
        help="scipy eigsh max Lanczos iterations (sparse backend; default 10000).",
    )
    p.add_argument(
        "--eigensolver-ncv",
        dest="eigensolver_ncv",
        type=int,
        default=None,
        metavar="N",
        help="scipy eigsh Krylov subspace size (sparse; default max(30, 2*num_eigs+1)).",
    )
    p.add_argument(
        "--out-dir",
        dest="out_dir",
        type=str,
        default="",
        help="Output directory (default: <run_dir>/results/PVC). Relative to cwd.",
    )
    p.add_argument(
        "--separator",
        type=str,
        default=";",
        help="CSV separator for written eigenvalue/eigenvector files (default: ';').",
    )
    p.add_argument(
        "--dimensionless-coordinates",
        dest="dimensionless_coordinates",
        action="store_true",
        default=True,
        help="Use dimensionless position coordinates for phonons (default).",
    )
    p.add_argument(
        "--no-dimensionless-coordinates",
        dest="dimensionless_coordinates",
        action="store_false",
        help="Use dimensional position coordinates.",
    )
    p.add_argument(
        "--null-point-vib",
        dest="null_point_vib",
        action="store_true",
        default=True,
        help="Include zero-point energy 0.5*ħω in phonon operators where applicable (default).",
    )
    p.add_argument(
        "--no-null-point-vib",
        dest="null_point_vib",
        action="store_false",
        help="Omit zero-point energy from phonon definitions.",
    )
    p.add_argument(
        "--save-npz",
        dest="save_npz",
        action="store_true",
        default=False,
        help="Save eigenstates (eigenvectors and eigenvalues) in compressed NPZ format.",
    )
    p.add_argument(
        "--npz-filename",
        dest="npz_filename",
        type=str,
        default="eigenstates.npz",
        metavar="NAME",
        help=(
            "Filename (or path) for the saved eigenstates NPZ. A bare "
            "filename is placed under out_dir; absolute or relative paths "
            "are used as-is. '.npz' is appended if missing. "
            "Default: eigenstates.npz."
        ),
    )
    p.add_argument(
        "--save-csv",
        dest="save_csv",
        action="store_true",
        default=True,
        help="Save eigenvectors and eigenvalues to CSV files (default).",
    )
    p.add_argument(
        "--no-save-csv",
        dest="save_csv",
        action="store_false",
        help="Do not save eigenvectors and eigenvalues to CSV files.",
    )
    p.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        default=False,
        help="Suppress detailed build log (Hilbert basis, ladder operators, and per-step wall times).",
    )
    p.add_argument(
        "--save-hamiltonian",
        dest="save_hamiltonian",
        action="store_true",
        default=False,
        help="Save the built Hamiltonian matrix to NPZ (see --hamiltonian-filename).",
    )
    p.add_argument(
        "--hamiltonian-filename",
        dest="hamiltonian_filename",
        type=str,
        default="hamiltonian.npz",
        metavar="NAME",
        help="NPZ file for save_hamiltonian (bare name → under out_dir). Default: hamiltonian.npz.",
    )
    p.add_argument(
        "--load-hamiltonians",
        dest="load_hamiltonians",
        type=str,
        default="",
        metavar="NAMES",
        help=(
            "Comma-separated Hamiltonian NPZ stems or paths to load and sum "
            "(skips CSV Hamiltonian build). Resolved under out_dir when bare names."
        ),
    )
    p.add_argument(
        "--eigensolve",
        dest="eigensolve",
        action="store_true",
        default=True,
        help="Diagonalize after build/load (default).",
    )
    p.add_argument(
        "--no-eigensolve",
        dest="eigensolve",
        action="store_false",
        help="Only build and/or save Hamiltonian; skip [eigensolver] diagonalization.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    # Flush so batch schedulers (SLURM file redirect) show progress immediately.
    # Also set: PYTHONUNBUFFERED=1 or python -u on the job script.
    from jahn_teller_dynamics.physics.hamiltonians.djt_polynomial_hamiltonian import (
        PhononRebuildSpec,
        build_and_save_coupling_batch,
        build_bare_vibronic_hamiltonian,
        create_pvc_hamiltonian,
        create_pvc_hamiltonian_parallel,
        prepare_completed_coupling_rows,
    )
    from jahn_teller_dynamics.physics.hamiltonians.djt_polynomial_hamiltonian_grouped import (
        create_pvc_hamiltonian_grouped,
    )
    from jahn_teller_dynamics.physics.models.pvc_model import PVC_model
    from jahn_teller_dynamics.io.file_io.csv_writer import CSVWriter
    from jahn_teller_dynamics.io.file_io.hamiltonian_npz import (
        aggregate_hamiltonians_from_paths,
        basis_labels_to_hilbert_bases,
        save_hamiltonian_npz,
    )
    from jahn_teller_dynamics.math_utils.eigen_solver import (
        assert_matrix_operator_hermitian,
        check_matrix_operator_hermiticity,
        create_pvc_eigen_solver,
        resolve_pvc_eigensolver_backend,
    )
    from jahn_teller_dynamics.math_utils.matrix_mechanics import MatrixOperator
    import jahn_teller_dynamics.math_utils.maths as maths

    def _out(msg: str) -> None:
        print_ts(msg, flush=True)

    print_ts("jtd_run: parsing CLI / config...", flush=True)
    args = _build_parser_with_cfg_defaults(argv).parse_args(argv)
    quiet: bool = bool(getattr(args, "quiet", False))
    build_log: Optional[Callable[[str], None]] = None if quiet else _out

    run_ctx = RunContext.from_cwd()
    run_dir = run_ctx.run_dir
    calc = _args_to_calculation(args, run_dir)
    backend = resolve_pvc_eigensolver_backend(
        calc.eigensolver,
        calc.running_environment,
        use_sparse=calc.use_sparse,
    )

    load_paths = calc.resolve_hamiltonian_load_paths(run_dir)
    skip_hamiltonian_build = bool(load_paths)

    se_path = modes_path = coupling_path = None
    spin_op_path = None
    if not skip_hamiltonian_build:
        _out("jtd_run: resolving input paths (electron / modes / coupling CSV)...")
        se_path, modes_path, coupling_path = calc.resolve_input_paths(run_dir)
        spin_op_path = calc.resolve_spin_orbital_operator_path(run_dir)
        for pth, label in [
            (se_path, "electron energies"),
            (modes_path, "modes"),
            (coupling_path, "coupling"),
        ]:
            if not pth.exists():
                raise FileNotFoundError(f"Missing required {label} file: {pth}")
        if spin_op_path is not None and not spin_op_path.exists():
            raise FileNotFoundError(f"Missing spin–orbital operator file: {spin_op_path}")
    else:
        _out(
            f"jtd_run: load_hamiltonians set ({len(load_paths)} file(s)) — skipping CSV Hamiltonian build."
        )
        for pth in load_paths:
            if not pth.exists():
                raise FileNotFoundError(f"Missing Hamiltonian NPZ: {pth}")

    mq = getattr(calc, "maximum_number_of_quanta_per_dim", 0) or 0
    mv = getattr(calc, "maximum_number_of_vibrational_quanta", 0) or 0
    encut = getattr(calc, "phonon_encut", None)
    use_per_mode_csv = bool(
        getattr(calc, "use_maximum_quanta_from_modes_csv", False)
    )
    active_truncations = sum(
        1
        for cond in (mq > 0, mv > 0, encut is not None, use_per_mode_csv)
        if cond
    )
    if active_truncations > 1:
        raise ValueError(
            "jtd_run: set at most one phonon truncation: phonon_encut, "
            "maximum_number_of_vibrational_quanta, maximum_number_of_quanta_per_dim, "
            "or use_maximum_quanta_from_modes_csv."
        )
    if encut is not None and float(encut) < 0:
        raise ValueError(f"jtd_run: phonon_encut must be non-negative; got {encut!r}")

    phonon_descr = (
        f"tensor-product phonons (max quanta/mode <= {mq})"
        if mq > 0
        else (
            f"energy-cutoff phonons (sum n_i e_i <= {float(encut):g})"
            if encut is not None
            else (
                "tensor-product phonons (per-mode N_i from modes.csv 'maximum_quanta')"
                if use_per_mode_csv
                else (
                    "constrained phonons (sum quanta "
                    + (f"<={mv}" if mv > 0 else f"<={calc.order}")
                    + ")"
                )
            )
        )
    )

    model: Optional[PVC_model] = None
    quantum_bases = None
    H: Optional[MatrixOperator] = None

    mpi_rank = 0
    mpi_size = 1
    mpi_comm = None
    if backend == "slepc":
        try:
            from petsc4py import PETSc

            if not PETSc.Sys.isInitialized():
                PETSc.Sys.initialize([])
            mpi_comm = PETSc.COMM_WORLD.tompi4py()
            mpi_rank = int(mpi_comm.Get_rank())
            mpi_size = int(mpi_comm.Get_size())
        except Exception:
            mpi_rank = 0
            mpi_size = 1
            mpi_comm = None

    builder_rank = 0
    should_build_locally = not (backend == "slepc" and mpi_size > 1 and mpi_rank != builder_rank)

    if should_build_locally and not skip_hamiltonian_build:
        _out(f"[1/4] Building PVC model from CSV inputs (electron + {phonon_descr})...")
        model = PVC_model.from_csvs(
            electron_energies_csv_path=str(se_path),
            modes_csv_path=str(modes_path),
            coupling_csv_path=str(coupling_path),
            order=calc.order,
            maximum_quanta_per_mode=int(mq) if mq > 0 else None,
            maximum_total_phonon_quanta=int(mv) if mv > 0 else None,
            phonon_encut=float(encut) if encut is not None else None,
            use_maximum_quanta_from_modes_csv=use_per_mode_csv,
            use_sparse=calc.use_sparse,
            dimensionless_coordinates=calc.dimensionless_coordinates,
            null_point_vib=calc.null_point_vib,
            mode_numbers=calc.modes_to_use,
            build_log=build_log,
            exp_approximation_order=calc.exp_approximation_order,
            tune_tuning=calc.tune_tuning,
            tune_coupling=calc.tune_coupling,
            spin_orbital_operator_csv_path=str(spin_op_path) if spin_op_path else "",
        )
        quantum_bases = model.root_node.base_states
        _out(
            f"  → PVC model built: dim(electron)={model.electron.node.dim}, "
            f"dim(phonon)={model.phonons.dim}, dim(full)={model.root_node.dim}"
        )

        mq = int(mq)
        mv = int(mv)
        phonon_rebuild = PhononRebuildSpec(
            modes_csv_path=str(modes_path),
            order=calc.order,
            maximum_quanta_per_mode=mq if mq > 0 else 0,
            maximum_total_phonon_quanta=mv if mv > 0 else 0,
            phonon_encut=float(encut) if encut is not None else None,
            use_maximum_quanta_from_modes_csv=use_per_mode_csv,
            use_sparse=calc.use_sparse,
            dimensionless_coordinates=calc.dimensionless_coordinates,
            null_point_vib=calc.null_point_vib,
            exp_approximation_order=calc.exp_approximation_order,
            mode_numbers=tuple(calc.modes_to_use) if calc.modes_to_use else None,
        )
        n_orbitals = (
            model.orbital_spin_layout.n_orbitals
            if model.orbital_spin_layout is not None
            else model.electron.node.dim
        )
        lined_kw = dict(
            hermitian_completion=calc.hermitian_completion,
            diagonal_completion=calc.diagonal_completion,
            permutation_completion=calc.permutation_completion,
            symmetrize_hamiltonian=calc.symmetrize_hamiltonian,
        )

        if calc.save_bare_hamiltonian and mpi_rank == 0:
            out_dir = calc.resolve_out_dir(run_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            bare_path = out_dir / "hamiltonian_bare.npz"
            h_bare = build_bare_vibronic_hamiltonian(
                model.phonons,
                model.electron_energies,
                spin_orbital_operator=getattr(model, "spin_orbital_operator", None),
            )
            basis_save = [str(s) for s in model.root_node.base_states._ket_states]
            save_hamiltonian_npz(
                bare_path,
                h_bare,
                label="bare",
                basis_labels=basis_save,
                extra_metadata={"component": "bare_vibronic"},
            )
            _out(f"  → Saved bare Hamiltonian (no coupling) to: {bare_path}")

        if calc.coupling_batch_index is not None:
            batch_count = calc.coupling_batch_count or calc.lined_coupling_batches
            completed = prepare_completed_coupling_rows(
                model.coupling_rows,
                n_orbitals,
                diagonal_completion=calc.diagonal_completion,
                permutation_completion=calc.permutation_completion,
                hermitian_completion=calc.hermitian_completion,
                phonon=model.phonons,
            )
            out_dir = calc.resolve_out_dir(run_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            batch_path = out_dir / (
                f"{calc.coupling_batch_stem}_{int(calc.coupling_batch_index):05d}.npz"
            )
            _out(
                f"[2/4] Building coupling batch {calc.coupling_batch_index} / "
                f"{batch_count - 1} (distributed job)..."
            )
            build_and_save_coupling_batch(
                phonon_rebuild=phonon_rebuild,
                completed_rows=completed,
                batch_index=int(calc.coupling_batch_index),
                batch_count=int(batch_count),
                n_orbitals=n_orbitals,
                use_sparse=calc.use_sparse,
                output_path=batch_path,
            )
            _out(f"  → Saved coupling batch to: {batch_path}")
            quantum_bases = model.root_node.base_states
            H = None
        else:
            builder = normalize_hamiltonian_builder(getattr(calc, "hamiltonian_builder", "lined"))
            _out(f"  → hamiltonian_builder = {builder}")
            if builder == "grouped":
                if calc.parallel_lined_coupling:
                    _out(
                        "  → Warning: parallel_lined_coupling is ignored when hamiltonian_builder=grouped"
                    )
                _out(
                    "[2/4] Constructing PVC Hamiltonian (grouped orbital operators per expression)..."
                )
                H = create_pvc_hamiltonian_grouped(model, **lined_kw)
                n_orbital_ops = len(getattr(model, "orbital_coupling_operators", {}))
                _out(f"  → Registered {n_orbital_ops} orbital coupling operator(s) on the tree")
            elif calc.parallel_lined_coupling:
                _out(
                    "[2/4] Constructing PVC Hamiltonian (lined row-by-row, parallel coupling build)..."
                )
                H = create_pvc_hamiltonian_parallel(
                    model,
                    n_workers=calc.lined_coupling_workers,
                    n_memory_batches=calc.lined_coupling_batches,
                    phonon_rebuild=phonon_rebuild,
                    lined_coupling_pool=calc.lined_coupling_pool,
                    **lined_kw,
                )
            else:
                _out(
                    "[2/4] Constructing PVC Hamiltonian (lined row-by-row coupling)..."
                )
                H = create_pvc_hamiltonian(model, **lined_kw)
            _out(f"  → Hamiltonian matrix: dim(H) = {H.matrix.dim}")

        if H is not None and calc.save_hamiltonian and mpi_rank == 0:
            h_path = calc.resolve_hamiltonian_save_path(run_dir)
            basis_save = [str(s) for s in model.root_node.base_states._ket_states]
            save_hamiltonian_npz(
                h_path,
                H,
                label=h_path.stem,
                basis_labels=basis_save,
                extra_metadata={
                    "maximum_number_of_vibrational_quanta": calc.maximum_number_of_vibrational_quanta,
                    "maximum_number_of_quanta_per_dim": calc.maximum_number_of_quanta_per_dim,
                    "phonon_encut": calc.phonon_encut,
                    "use_maximum_quanta_from_modes_csv": calc.use_maximum_quanta_from_modes_csv,
                    "order": calc.order,
                },
            )
            _out(f"  → Saved Hamiltonian to: {h_path}")

    if skip_hamiltonian_build and should_build_locally:
        _out("[2/4] Loading and aggregating Hamiltonian NPZ file(s)...")
        H, load_meta = aggregate_hamiltonians_from_paths(load_paths)
        _out(
            f"  → Aggregated {len(load_paths)} Hamiltonian(s), dim(H) = {H.matrix.dim}, "
            f"nnz ≈ {load_meta.get('nnz', '?')}"
        )
        labels = load_meta.get("basis_labels")
        if labels:
            quantum_bases = basis_labels_to_hilbert_bases(labels)
        else:
            dim_loaded = int(H.matrix.dim)
            quantum_bases = basis_labels_to_hilbert_bases(
                [str(i) for i in range(dim_loaded)]
            )
            _out(
                "  → Warning: no basis_labels in Hamiltonian NPZ; using numeric indices for output.",
                flush=True,
            )

    if backend == "slepc" and mpi_size > 1 and mpi_comm is not None:
        from scipy.sparse import csr_matrix

        payload = None
        if mpi_rank == builder_rank:
            if H is None or quantum_bases is None:
                raise RuntimeError("Builder rank failed to construct Hamiltonian/basis for SLEPc.")
            if isinstance(H.matrix, maths.SparseMatrix):
                H_csr = H.matrix.matrix.tocsr()
            else:
                H_csr = csr_matrix(H.matrix.matrix)
            payload = {
                "shape": H_csr.shape,
                "indptr": H_csr.indptr,
                "indices": H_csr.indices,
                "data": H_csr.data,
                "basis": quantum_bases,
            }

        payload = mpi_comm.bcast(payload, root=builder_rank)
        h_shape = payload["shape"]
        h_indptr = payload["indptr"]
        h_indices = payload["indices"]
        h_data = payload["data"]
        quantum_bases = payload["basis"]
        H_csr_all = csr_matrix((h_data, h_indices, h_indptr), shape=h_shape)
        H = MatrixOperator(maths.SparseMatrix(H_csr_all))

    if H is None or quantum_bases is None:
        if calc.coupling_batch_index is not None and not calc.eigensolve:
            _out("[3/4] Coupling batch job complete (no full H assembled).")
            _out("=" * 70)
            _out("PVC distributed coupling batch saved (no eigensolve).")
            _out("=" * 70)
            return 0
        raise RuntimeError("jtd_run: failed to prepare Hamiltonian and basis for diagonalization.")

    if not calc.eigensolve:
        _out("[3/4] Skipping diagonalization (eigensolve = false).")
        _out("=" * 70)
        _out("PVC Hamiltonian build/load complete (no eigensolve).")
        _out(f"dim(H) = {H.matrix.dim}")
        _out("=" * 70)
        return 0

    _out("[3/4] Diagonalizing Hamiltonian...")
    dim_h = int(H.matrix.dim)
    num_of_vals_req = calc.num_eigs
    # num_eigs=all / none → full spectrum (pass None; dense solver uses all dim states).
    if num_of_vals_req is not None and num_of_vals_req >= dim_h:
        num_of_vals_req = None

    # scipy.sparse.linalg.eigsh requires k < dim - 1; full or near-full spectrum needs dense.
    if backend == "sparse":
        k_sparse = dim_h if num_of_vals_req is None else int(num_of_vals_req)
        if k_sparse >= dim_h - 1:
            _out(
                f"  → num_eigs requests {k_sparse} of {dim_h} eigenpairs; "
                "sparse eigsh cannot supply a full spectrum — using dense eigensolver."
            )
            backend = "dense"

    hc = check_matrix_operator_hermiticity(H)
    if calc.require_hermitian and not hc.is_hermitian:
        assert_matrix_operator_hermitian(H, context="PVC Hamiltonian")
    if mpi_rank == 0:
        _out(f"  → Matrix being diagonalized: {describe_hamiltonian_matrix(H)}")
        _out(f"  → Hermiticity check: {hc.summary_line()}")
        if not hc.is_hermitian and not calc.require_hermitian:
            _out(
                "  → Warning: H is not Hermitian; using general eigensolver (eigs / SLEPc NHEP). "
                "Eigenvalues may be complex; interpret splittings with care."
            )

    allow_non_hermitian = bool(not calc.require_hermitian and not hc.is_hermitian)

    eigen_solver = create_pvc_eigen_solver(backend, calc=calc)
    _out(
        f"  → Eigen backend: {backend}"
        + (" (PETSc/SLEPc — pass -eps_* options before script args if needed)" if backend == "slepc" else "")
    )

    sw = (getattr(calc, "eigensolver_spectral_which", "") or "").strip()

    solve_kw: dict = {
        "num_of_vals": num_of_vals_req,
        "quantum_states_bases": quantum_bases,
        "spectral_sigma": getattr(calc, "eigensolver_sigma", None),
        "spectral_which": sw if sw else None,
        "allow_non_hermitian": allow_non_hermitian,
    }
    if backend == "sparse":
        from jahn_teller_dynamics.math_utils.eigen_solver import sparse_eigsh_kwargs_from_calc

        solve_kw.update(sparse_eigsh_kwargs_from_calc(calc, dim=dim_h))
    elif backend == "slepc":
        from jahn_teller_dynamics.math_utils.eigen_solver import slepc_kwargs_from_calc

        solve_kw.update(slepc_kwargs_from_calc(calc))

    eig_space = eigen_solver.solve(H, **solve_kw)

    eig_vals = [k.eigen_val for k in eig_space.eigen_kets]
    _out(f"  → Computed {len(eig_vals)} eigenvalues/eigenvectors")

    if _pvc_mpi_secondary_rank_skip_file_io(backend):
        return 0

    _out("[4/4] Saving results...")
    out_dir = calc.resolve_out_dir(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if calc.save_csv:
        writer = CSVWriter(separator=calc.separator, index=True)
        eig_vec_path = str(out_dir / "eigenvectors.csv")
        eig_val_path = str(out_dir / "eigenvalues.csv")
        writer.write_eigen_vectors_and_values(eig_space, eig_vec_path, eig_val_path)
        _out(f"Saved eigenvectors to: {eig_vec_path}")
        _out(f"Saved eigenvalues  to: {eig_val_path}")

    if calc.save_npz:
        eig_vecs = np.column_stack(
            [np.array(ket.coeffs.coeffs).flatten() for ket in eig_space.eigen_kets]
        )
        eig_vals_arr = np.array([complex(k.eigen_val) for k in eig_space.eigen_kets])
        if model is not None:
            basis_labels = np.array(
                [str(s) for s in model.root_node.base_states._ket_states]
            )
        elif quantum_bases is not None:
            basis_labels = np.array(
                [str(s) for s in quantum_bases._ket_states], dtype=object
            )
        else:
            basis_labels = np.array(
                [str(i) for i in range(H.matrix.dim)], dtype=object
            )
        npz_path = calc.resolve_npz_path(run_dir)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            npz_path,
            eigenvectors=eig_vecs,
            eigenvalues=eig_vals_arr,
            basis_labels=basis_labels,
            order=calc.order,
            maximum_number_of_vibrational_quanta=calc.maximum_number_of_vibrational_quanta,
            maximum_number_of_quanta_per_dim=calc.maximum_number_of_quanta_per_dim,
            phonon_encut=calc.phonon_encut,
            use_maximum_quanta_from_modes_csv=calc.use_maximum_quanta_from_modes_csv,
            dim=H.matrix.dim,
        )
        _out(f"Saved NPZ to: {npz_path}")

    _out("=" * 70)
    _out("PVC diagonalization complete")
    _out(f"dim(H) = {H.matrix.dim}")
    _out(f"computed eigenvalues = {len(eig_vals)}")
    _out("-" * 70)
    for i, ev in enumerate(eig_vals):
        try:
            ev_real = float(ev.real)  # type: ignore[union-attr]
            _out(f"E[{i:03d}] = {ev_real:.12g}")
        except Exception:
            _out(f"E[{i:03d}] = {ev}")
    _out("=" * 70)

    return 0


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except (OSError, ValueError):
            pass
    raise SystemExit(main())
