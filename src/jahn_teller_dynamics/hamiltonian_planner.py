"""
Stage 1 of the split PVC build pipeline: **plan** a polynomial vibronic coupling
Hamiltonian.

This is a memory-efficient alternative to running the full :mod:`jahn_teller_dynamics.jtd_run`
end-to-end. The pipeline splits into three independent jobs::

    jtd_run.py  ≡  hamiltonian_planner.py  →  hamiltonian_builder.py  →  eigensolver.py

The planner:

1. Reads the same INI ``.cfg`` file as :mod:`jahn_teller_dynamics.jtd_run` (``[PVC]`` /
   ``[essentials]`` sections).
2. Loads the input CSVs (electron energies, modes, polynomial coupling rows),
   applies ``tune_tuning`` / ``tune_coupling`` and orbital-label resolution exactly
   like :class:`~jahn_teller_dynamics.physics.models.pvc_model.PVC_model`.
3. Applies ``diagonal_completion`` and ``permutation_completion`` so the saved
   row list is the **same** sequence the assembly loop iterates over (Hermitian
   partners are still added at assembly time via the ``hermitian_completion``
   flag stored in the NPZ).
4. Writes everything required to rebuild the phonon space and assemble partial
   coupling Hamiltonians into a single ``.npz`` file (default
   ``coupling_rows.npz`` under ``output_folder``).
5. Prints ``n_coupling_rows=<N>`` to ``stdout`` so a Slurm submit script can
   pick the array size for stage 2.

Run::

    python3 -m jahn_teller_dynamics.hamiltonian_planner \\
        --config data/dJT_data/methoxy/builder_method/planner.cfg

Output payload (NPZ keys)::

    rows_el_state_1, rows_el_state_2, rows_polinom,
    rows_coeff_real, rows_coeff_imag       # completed coupling rows
    electron_energies                      # length = dim_el (orbital-spin if applicable)
    mode_labels, mode_omegas               # phonon modes (already filtered by modes_to_use)
    mode_maximum_quanta                    # per-mode N_i (empty when not used)
    basis_labels                           # full Hilbert basis labels
    spin_orbital_operator                  # dense complex matrix (empty when absent)
    has_spin_layout                        # bool — orbital–spin basis active
    orbital_spin_labels, orbital_spin_S    # orbital–spin layout (when applicable)
    phonon_truncation_kind                 # constrained|tensor_uniform|tensor_per_mode|encut
    order, maximum_quanta_per_mode,
    maximum_total_phonon_quanta,
    phonon_encut, use_maximum_quanta_from_modes_csv,
    use_sparse, dimensionless_coordinates,
    null_point_vib, exp_approximation_order,
    hermitian_completion                   # used at assembly time
    diagonal_completion, permutation_completion
    n_orbitals, dim_el, dim_ph, dim_full
    modes_csv_path                         # absolute path to modes CSV (for
                                           # spawn-pool builder workers to
                                           # rebuild phonon)

Only the rows are large; everything else is metadata.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from jahn_teller_dynamics.jtd_run import _args_to_calculation, _build_parser_with_cfg_defaults
from jahn_teller_dynamics.io.config.pvc_config import PVCCalculation
from jahn_teller_dynamics.io.utils.run_context import RunContext
from jahn_teller_dynamics.io.utils.timestamp_print import print_ts


_PLANNER_NPZ_VERSION = 1


def _phonon_truncation_kind(calc: PVCCalculation) -> str:
    mq = int(getattr(calc, "maximum_number_of_quanta_per_dim", 0) or 0)
    mv = int(getattr(calc, "maximum_number_of_vibrational_quanta", 0) or 0)
    encut = getattr(calc, "phonon_encut", None)
    use_per_mode = bool(getattr(calc, "use_maximum_quanta_from_modes_csv", False))
    if encut is not None:
        return "encut"
    if use_per_mode:
        return "tensor_per_mode"
    if mq > 0:
        return "tensor_uniform"
    if mv > 0 or int(calc.order) > 0:
        return "constrained"
    raise ValueError("Cannot resolve phonon truncation; set one of the truncation keys in .cfg")


def resolve_coupling_rows_path(calc: PVCCalculation, run_dir: Path) -> Path:
    """
    Resolve where the planner writes the coupling-rows NPZ.

    Bare filename → under ``calc.resolve_out_dir(run_dir)``; relative path → under
    ``run_dir``; absolute path → as-is. ``.npz`` is appended if missing.
    """
    name = (getattr(calc, "coupling_rows_filename", "") or "coupling_rows.npz").strip()
    if not name.lower().endswith(".npz"):
        name = name + ".npz"
    p = Path(name).expanduser()
    if p.is_absolute():
        return p.resolve()
    if p.parent == Path("."):
        return (calc.resolve_out_dir(run_dir) / p).resolve()
    return (run_dir / p).resolve()


def main(argv: Optional[list[str]] = None) -> int:
    from jahn_teller_dynamics.physics.hamiltonians.djt_polynomial_hamiltonian import (
        prepare_completed_coupling_rows,
    )
    from jahn_teller_dynamics.physics.models.pvc_model import PVC_model

    import argparse

    print_ts("hamiltonian_planner: parsing CLI / config...", flush=True)
    parser = _build_parser_with_cfg_defaults(argv)
    # Planner-specific flag. Use SUPPRESS so omitting it on CLI does not
    # clobber a value coming from set_defaults(cfg).
    parser.add_argument(
        "--coupling-rows-filename",
        dest="coupling_rows_filename",
        type=str,
        default=argparse.SUPPRESS,
        metavar="NAME",
        help=(
            "Output NPZ filename or path for completed coupling rows "
            "(bare name → under out_dir; default: coupling_rows.npz)."
        ),
    )
    args = parser.parse_args(argv)
    quiet = bool(getattr(args, "quiet", False))

    run_ctx = RunContext.from_cwd()
    run_dir = run_ctx.run_dir
    calc = _args_to_calculation(args, run_dir)

    if not (calc.coupling_rows_filename or "").strip():
        calc.coupling_rows_filename = "coupling_rows.npz"

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

    mq = int(getattr(calc, "maximum_number_of_quanta_per_dim", 0) or 0)
    mv = int(getattr(calc, "maximum_number_of_vibrational_quanta", 0) or 0)
    encut = getattr(calc, "phonon_encut", None)
    use_per_mode = bool(getattr(calc, "use_maximum_quanta_from_modes_csv", False))
    active = sum(1 for c in (mq > 0, mv > 0, encut is not None, use_per_mode) if c)
    if active > 1:
        raise ValueError(
            "hamiltonian_planner: set at most one phonon truncation: phonon_encut, "
            "maximum_number_of_vibrational_quanta, maximum_number_of_quanta_per_dim, "
            "or use_maximum_quanta_from_modes_csv."
        )

    build_log = None if quiet else (lambda m: print_ts(m, flush=True))

    print_ts(
        "hamiltonian_planner: building PVC model from CSV (electron + phonons + raw coupling)...",
        flush=True,
    )
    t0 = time.perf_counter()
    model = PVC_model.from_csvs(
        electron_energies_csv_path=str(se_path),
        modes_csv_path=str(modes_path),
        coupling_csv_path=str(coupling_path),
        order=calc.order,
        maximum_quanta_per_mode=mq if mq > 0 else None,
        maximum_total_phonon_quanta=mv if mv > 0 else None,
        phonon_encut=float(encut) if encut is not None else None,
        use_maximum_quanta_from_modes_csv=use_per_mode,
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
    print_ts(
        f"  → PVC model built: dim(electron)={model.electron.node.dim}, "
        f"dim(phonon)={model.phonons.dim}, dim(full)={model.root_node.dim} "
        f"(wall {time.perf_counter() - t0:.2f}s)",
        flush=True,
    )

    n_orbitals = (
        model.orbital_spin_layout.n_orbitals
        if model.orbital_spin_layout is not None
        else model.electron.node.dim
    )
    print_ts(
        f"hamiltonian_planner: applying row completions "
        f"(diagonal={calc.diagonal_completion}, permutation={calc.permutation_completion})...",
        flush=True,
    )
    completed = prepare_completed_coupling_rows(
        model.coupling_rows,
        n_orbitals,
        diagonal_completion=calc.diagonal_completion,
        permutation_completion=calc.permutation_completion,
        hermitian_completion=calc.hermitian_completion,
        phonon=model.phonons,
    )
    n_rows = len(completed)
    print_ts(f"  → completed rows: {n_rows}", flush=True)

    payload = _build_planner_payload(
        calc=calc,
        model=model,
        completed_rows=completed,
        n_orbitals=n_orbitals,
        spin_op_path=spin_op_path,
        modes_csv_path=modes_path,
    )

    out_path = resolve_coupling_rows_path(calc, run_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    print_ts(f"hamiltonian_planner: saved coupling rows + metadata to: {out_path}", flush=True)

    # Machine-readable line for Slurm submit scripts (kept simple: key=value).
    print(f"n_coupling_rows={n_rows}", flush=True)
    return 0


def _build_planner_payload(
    *,
    calc: PVCCalculation,
    model: Any,
    completed_rows: list,
    n_orbitals: int,
    spin_op_path: Optional[Path],
    modes_csv_path: Path,
) -> dict[str, np.ndarray]:
    from jahn_teller_dynamics.io.file_io.csv_reader import CSVReader

    el1 = np.asarray([r.el_state_1 for r in completed_rows], dtype=np.int32)
    el2 = np.asarray([r.el_state_2 for r in completed_rows], dtype=np.int32)
    polinom = np.asarray([r.polinom for r in completed_rows], dtype=object)
    coeff = np.asarray([complex(r.coeff) for r in completed_rows], dtype=np.complex128)

    basis_labels = np.asarray(
        [str(s) for s in model.root_node.base_states._ket_states], dtype=object
    )

    # Re-read the modes CSV with the same modes_to_use filter — same data the
    # phonon system was built from, but stored independently from internal attrs.
    modes_table = CSVReader().read_modes_flexible(
        str(modes_csv_path),
        mode_numbers=list(calc.modes_to_use) if calc.modes_to_use else None,
    )
    mode_labels = list(modes_table.labels)
    mode_omegas = np.asarray(list(modes_table.omegas), dtype=np.float64)

    truncation_kind = _phonon_truncation_kind(calc)
    payload: dict[str, np.ndarray] = {
        "planner_npz_version": np.int32(_PLANNER_NPZ_VERSION),
        "rows_el_state_1": el1,
        "rows_el_state_2": el2,
        "rows_polinom": polinom,
        "rows_coeff_real": coeff.real.astype(np.float64),
        "rows_coeff_imag": coeff.imag.astype(np.float64),
        "n_coupling_rows": np.int64(len(completed_rows)),
        "electron_energies": np.asarray(model.electron_energies, dtype=np.float64),
        "mode_labels": np.asarray(mode_labels, dtype=object),
        "mode_omegas": mode_omegas,
        "basis_labels": basis_labels,
        "phonon_truncation_kind": np.array(truncation_kind),
        "order": np.int32(calc.order),
        "maximum_quanta_per_mode": np.int32(
            int(getattr(calc, "maximum_number_of_quanta_per_dim", 0) or 0)
        ),
        "maximum_total_phonon_quanta": np.int32(
            int(getattr(calc, "maximum_number_of_vibrational_quanta", 0) or 0)
        ),
        "phonon_encut": np.float64(
            float(calc.phonon_encut) if calc.phonon_encut is not None else float("nan")
        ),
        "use_maximum_quanta_from_modes_csv": np.bool_(
            bool(calc.use_maximum_quanta_from_modes_csv)
        ),
        "use_sparse": np.bool_(bool(calc.use_sparse)),
        "dimensionless_coordinates": np.bool_(bool(calc.dimensionless_coordinates)),
        "null_point_vib": np.bool_(bool(calc.null_point_vib)),
        "exp_approximation_order": np.int32(
            int(calc.exp_approximation_order) if calc.exp_approximation_order is not None else -1
        ),
        "hermitian_completion": np.bool_(bool(calc.hermitian_completion)),
        "diagonal_completion": np.bool_(bool(calc.diagonal_completion)),
        "permutation_completion": np.bool_(bool(calc.permutation_completion)),
        "n_orbitals": np.int32(int(n_orbitals)),
        "dim_el": np.int32(int(model.electron.node.dim)),
        "dim_ph": np.int32(int(model.phonons.dim)),
        "dim_full": np.int32(int(model.root_node.dim)),
    }

    # Modes CSV maximum_quanta column (only meaningful for
    # use_maximum_quanta_from_modes_csv=True; preserved for traceability either way).
    if modes_table.maximum_quanta is not None:
        payload["mode_maximum_quanta"] = np.asarray(
            list(modes_table.maximum_quanta), dtype=np.int32
        )
    else:
        payload["mode_maximum_quanta"] = np.asarray([], dtype=np.int32)

    # Modes_to_use filter (passed to phonon system at planning time; needed only for traceability).
    if calc.modes_to_use:
        payload["modes_to_use"] = np.asarray(
            [int(m) for m in calc.modes_to_use], dtype=np.int32
        )
    else:
        payload["modes_to_use"] = np.asarray([], dtype=np.int32)

    # Orbital–spin layout.
    layout = model.orbital_spin_layout
    if layout is not None:
        payload["has_spin_layout"] = np.bool_(True)
        payload["orbital_spin_labels"] = np.asarray(
            list(layout.orbital_labels), dtype=object
        )
        payload["orbital_spin_S"] = np.asarray(list(layout.spin_S), dtype=np.float64)
    else:
        payload["has_spin_layout"] = np.bool_(False)
        from jahn_teller_dynamics.io.electron_state_resolution import (
            electron_labels_in_basis_order,
        )

        electron = getattr(model, "electron", None)
        state_to_index = getattr(electron, "state_to_index", None)
        if state_to_index:
            payload["electron_state_labels"] = np.asarray(
                electron_labels_in_basis_order(state_to_index), dtype=object
            )

    # Spin–orbital operator matrix (dense complex, NxN).
    spin_op = getattr(model, "spin_orbital_operator", None)
    if spin_op is not None:
        import jahn_teller_dynamics.math_utils.maths as maths

        m = spin_op.matrix
        if isinstance(m, maths.SparseMatrix):
            arr = m.matrix.toarray().astype(np.complex128)
        else:
            arr = np.asarray(m.matrix, dtype=np.complex128)
        payload["spin_orbital_operator"] = arr
    else:
        payload["spin_orbital_operator"] = np.empty((0, 0), dtype=np.complex128)
    payload["spin_orbital_operator_source"] = np.array(
        str(spin_op_path) if spin_op_path is not None else ""
    )

    # Absolute path to the modes CSV so spawn-pool builder workers can rebuild
    # phonon from a pickle-friendly PhononRebuildSpec without re-reading any cfg.
    try:
        payload["modes_csv_path"] = np.array(str(Path(modes_csv_path).resolve()))
    except Exception:
        payload["modes_csv_path"] = np.array(str(modes_csv_path))

    return payload


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except (OSError, ValueError):
            pass
    raise SystemExit(main())
