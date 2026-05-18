"""
Main entry point for building and diagonalizing a PVC (polynomial vibronic coupling)
Hamiltonian from CSV inputs.

Expected inputs (under ``--data-dir`` by default):

- ``electron_energies.csv`` — diagonal electronic energies (``el_state``, ``energy`` or ``state``, ``value``)
- ``modes.csv`` — mode frequencies (``mode``, ``energy`` or ``omega``)
- ``coupling.csv`` — coupling rows (``el_state_1``, ``el_state_2``, ``expression``, ``coeff``).
  Legacy column names ``polinom`` / ``polynomial`` are still accepted for the coupling expression.

Phonon space: **constrained** multimode Hilbert space (``maximum_number_of_vibrational_quanta``
or legacy ``order``: ``sum_i n_i <= N``), or tensor-product truncation with ``maximum_number_of_quanta_per_dim``.

Spectral knobs in ``[PVC]`` / ``[essentials]``: ``num_eigs``, ``eigensolver_sigma`` (or
``eigensolver_target`` / ``spectral_sigma``), and ``eigensolver_spectral_which`` (or
``spectral_which``). CLI: ``--num-eigs``, ``--eigensolver-sigma``, ``--eigensolver-spectral-which``.

Coupling exponentials (``exp(...)`` in coupling CSV expressions): ``exp_approximation_order`` (INI,
non-negative integer) selects a truncated Taylor sum for the matrix exponential; omit for exact ``expm``. CLI: ``--exp-approximation-order``.

Coupling CSV ``coeff`` scaling: ``tune_tuning`` (INI / ``--tune-tuning``) multiplies ``coeff`` when
``el_state_1`` and ``el_state_2`` resolve to the same 1-based index; ``tune_coupling`` when they differ.
Defaults are ``1.0``.

Run (from repo root)::

    python3 -m jahn_teller_dynamics.PVC

MPI clusters (PETSc/SLEPc): launch with ``srun`` / ``mpirun`` using an interpreter linked to PETSc,
set ``eigensolver = slepc`` or ``running_environment = multiprocessor`` in ``[PVC]``. Only rank 0 writes
CSV/NPZ output when running with multiple MPI ranks.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from jahn_teller_dynamics.io.config.pvc_config import PVCConfigParser, PVCCalculation
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
        "exp_approximation_order": calc.exp_approximation_order,
        "npz_filename": calc.npz_filename,
        "tune_tuning": calc.tune_tuning,
        "tune_coupling": calc.tune_coupling,
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
        exp_approximation_order=getattr(args, "exp_approximation_order", None),
        npz_filename=str(getattr(args, "npz_filename", "") or "eigenvectors.npz"),
        tune_tuning=float(getattr(args, "tune_tuning", 1.0)),
        tune_coupling=float(getattr(args, "tune_coupling", 1.0)),
    )


def read_pvc_cfg(cfg_path: str) -> dict[str, Any]:
    """
    Read an INI-style .cfg file and return argparse-defaults for :mod:`jahn_teller_dynamics.PVC`.

    CLI args always override config values.
    """
    run_ctx = RunContext.from_cwd()
    parser = PVCConfigParser(cfg_path, run_dir=run_ctx.run_dir)
    calc = parser.build_calculation()
    return _calc_to_argparse_defaults(calc)


def _build_parser_with_cfg_defaults(argv: list[str] | None) -> argparse.ArgumentParser:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default="", help="Path to INI-style .cfg file with [PVC] section.")
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
        description="Build and diagonalize a PVC (polynomial vibronic coupling) Hamiltonian from CSV inputs."
    )
    p.add_argument(
        "--config",
        type=str,
        default="config_files/PVC_trial_inputs.cfg",
        help="Path to INI-style .cfg file with [PVC] section.",
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
        help="Eigen solver: sparse (scipy), dense, slepc/PETSc (MPI-capable); overrides default from use-sparse.",
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
        help="Save eigenvectors and eigenvalues in compressed NPZ format.",
    )
    p.add_argument(
        "--npz-filename",
        dest="npz_filename",
        type=str,
        default="eigenvectors.npz",
        metavar="NAME",
        help=(
            "Filename (or path) for the saved eigenvectors NPZ. A bare "
            "filename is placed under out_dir; absolute or relative paths "
            "are used as-is. '.npz' is appended if missing. "
            "Default: eigenvectors.npz."
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
    return p


def main(argv: list[str] | None = None) -> int:
    # Flush so batch schedulers (SLURM file redirect) show progress immediately.
    # Also set: PYTHONUNBUFFERED=1 or python -u on the job script.
    from jahn_teller_dynamics.physics.hamiltonians.djt_polynomial_hamiltonian import create_pvc_hamiltonian
    from jahn_teller_dynamics.physics.models.pvc_model import PVC_model
    from jahn_teller_dynamics.io.file_io.csv_writer import CSVWriter
    from jahn_teller_dynamics.math.eigen_solver import (
        create_pvc_eigen_solver,
        resolve_pvc_eigensolver_backend,
    )
    from jahn_teller_dynamics.math.matrix_mechanics import MatrixOperator
    import jahn_teller_dynamics.math.maths as maths

    def _out(msg: str) -> None:
        print_ts(msg, flush=True)

    print_ts("PVC: parsing CLI / config...", flush=True)
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

    _out("PVC: resolving input paths (electron / modes / coupling CSV)...")
    se_path, modes_path, coupling_path = calc.resolve_input_paths(run_dir)
    for pth, label in [
        (se_path, "electron energies"),
        (modes_path, "modes"),
        (coupling_path, "coupling"),
    ]:
        if not pth.exists():
            raise FileNotFoundError(f"Missing required {label} file: {pth}")

    mq = getattr(calc, "maximum_number_of_quanta_per_dim", 0) or 0
    mv = getattr(calc, "maximum_number_of_vibrational_quanta", 0) or 0
    if mq > 0 and mv > 0:
        raise ValueError(
            "PVC: set only one of maximum_number_of_quanta_per_dim > 0 (tensor-product phonons) "
            "or maximum_number_of_vibrational_quanta > 0 (constrained total phonon number)."
        )

    phonon_descr = (
        f"tensor-product phonons (max quanta/mode <= {mq})"
        if mq > 0
        else (
            "constrained phonons (sum quanta "
            + (f"<={mv}" if mv > 0 else f"<={calc.order}")
            + ")"
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

    if should_build_locally:
        _out(f"[1/4] Building PVC model from CSV inputs (electron + {phonon_descr})...")
        model = PVC_model.from_csvs(
            electron_energies_csv_path=str(se_path),
            modes_csv_path=str(modes_path),
            coupling_csv_path=str(coupling_path),
            order=calc.order,
            maximum_quanta_per_mode=int(mq) if mq > 0 else None,
            maximum_total_phonon_quanta=int(mv) if mv > 0 else None,
            use_sparse=calc.use_sparse,
            dimensionless_coordinates=calc.dimensionless_coordinates,
            null_point_vib=calc.null_point_vib,
            mode_numbers=calc.modes_to_use,
            build_log=build_log,
            exp_approximation_order=calc.exp_approximation_order,
            tune_tuning=calc.tune_tuning,
            tune_coupling=calc.tune_coupling,
        )
        quantum_bases = model.root_node.base_states
        _out(
            f"  → PVC model built: dim(electron)={model.electron.node.dim}, "
            f"dim(phonon)={model.phonons.dim}, dim(full)={model.root_node.dim}"
        )

        _out(
            "[2/4] Constructing PVC Hamiltonian (diag electronic + polynomial coupling rows)..."
        )
        H = create_pvc_hamiltonian(model)
        _out(f"  → Hamiltonian matrix: dim(H) = {H.matrix.dim}")

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
        raise RuntimeError("PVC: failed to prepare Hamiltonian and basis for diagonalization.")

    _out("[3/4] Diagonalizing Hamiltonian...")
    num_of_vals_req = calc.num_eigs
    if num_of_vals_req is None:
        num_of_vals_req = H.matrix.dim - 1

    eigen_solver = create_pvc_eigen_solver(backend)
    _out(
        f"  → Eigen backend: {backend}"
        + (" (PETSc/SLEPc — pass -eps_* options before script args if needed)" if backend == "slepc" else "")
    )
    _out(f"  → Matrix being diagonalized: {describe_hamiltonian_matrix(H)}")

    sw = (getattr(calc, "eigensolver_spectral_which", "") or "").strip()

    eig_space = eigen_solver.solve(
        H,
        num_of_vals=num_of_vals_req,
        quantum_states_bases=quantum_bases,
        spectral_sigma=getattr(calc, "eigensolver_sigma", None),
        spectral_which=sw if sw else None,
    )

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
        if model is None:
            raise RuntimeError("PVC: model must exist on writer rank to save NPZ basis labels.")
        eig_vecs = np.column_stack(
            [np.array(ket.coeffs.coeffs).flatten() for ket in eig_space.eigen_kets]
        )
        eig_vals_arr = np.array([complex(k.eigen_val) for k in eig_space.eigen_kets])
        basis_labels = np.array([str(s) for s in model.root_node.base_states._ket_states])
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
