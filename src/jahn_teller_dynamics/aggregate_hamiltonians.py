"""
Aggregate-only stage of the split PVC build pipeline.

This is a trimmed-down sibling of :mod:`jahn_teller_dynamics.eigensolver`: it
resolves the ``load_hamiltonians`` entries, sums every matching partial
Hamiltonian NPZ file, optionally symmetrizes the result, and writes the
aggregated Hamiltonian back to a single NPZ. It performs **no** diagonalization.

Unlike :mod:`jahn_teller_dynamics.eigensolver`, missing inputs never abort the
run: explicit (non-glob) entries that do not exist on disk are skipped with a
warning, exactly like glob patterns that match nothing. This means you do *not*
need ``hamiltonian_part_bare.npz`` (or any other specific partial) to be present
— whatever is found is summed. The run only fails if *no* input matches at all.

Output goes to ``hamiltonian_filename`` (default ``hamiltonian.npz``) under the
configured output folder, via :meth:`PVCCalculation.resolve_hamiltonian_save_path`.

Run::

    python3 -m jahn_teller_dynamics.aggregate_hamiltonians \\
        --config data/dJT_data/methoxy/builder_method/eigensolver.cfg
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

from jahn_teller_dynamics.jtd_run import (
    _args_to_calculation,
    _build_parser_with_cfg_defaults,
    describe_hamiltonian_matrix,
)
from jahn_teller_dynamics.eigensolver import resolve_partial_hamiltonian_paths
from jahn_teller_dynamics.io.utils.run_context import RunContext
from jahn_teller_dynamics.io.utils.timestamp_print import print_ts


def _resolve_input_dir(calc, run_dir: Path) -> Path:
    """
    Directory the partial chunks are read from.

    Uses ``input_folder`` (stored as ``calc.data_dir``); if that is unset, falls
    back to ``output_folder`` (:meth:`PVCCalculation.resolve_out_dir`) so a
    single-folder layout keeps working.
    """
    data_dir = (getattr(calc, "data_dir", "") or "").strip()
    if not data_dir:
        return calc.resolve_out_dir(run_dir)
    p = Path(data_dir).expanduser()
    if not p.is_absolute():
        p = (run_dir / p).resolve()
    return p.resolve()


def main(argv: Optional[list[str]] = None) -> int:
    from jahn_teller_dynamics.io.file_io.hamiltonian_npz import (
        aggregate_hamiltonians_from_paths,
        basis_labels_to_hilbert_bases,
        save_hamiltonian_npz,
    )
    from jahn_teller_dynamics.math_utils.eigen_solver import (
        check_matrix_operator_hermiticity,
    )

    print_ts("aggregate: parsing CLI / config...", flush=True)
    parser = _build_parser_with_cfg_defaults(argv)
    args = parser.parse_args(argv)

    run_ctx = RunContext.from_cwd()
    run_dir = run_ctx.run_dir
    calc = _args_to_calculation(args, run_dir)

    # Read the partial chunks from input_folder (calc.data_dir) and save the
    # aggregated Hamiltonian to output_folder (calc.out_dir). When input_folder
    # is unset, fall back to output_folder so a single-folder setup still works.
    input_dir = _resolve_input_dir(calc, run_dir)
    out_dir = calc.resolve_out_dir(run_dir)

    print_ts(f"aggregate: resolving partial Hamiltonian inputs from {input_dir} ...", flush=True)
    candidates = resolve_partial_hamiltonian_paths(calc, run_dir, search_dir=input_dir)

    # Tolerant existence filter: unlike the eigensolver, a missing explicit
    # entry (e.g. the bare partial) is skipped with a warning instead of raising.
    paths: list[Path] = []
    for p in candidates:
        if p.exists():
            paths.append(p)
        else:
            print_ts(
                f"  → warning: input not found, skipping: {p}",
                flush=True,
            )
    if not paths:
        raise FileNotFoundError(
            "aggregate: none of the resolved load_hamiltonians entries exist on disk "
            f"(searched relative to input_folder={input_dir} and run_dir={run_dir})."
        )

    print_ts(f"  → {len(paths)} partial Hamiltonian file(s):", flush=True)
    for p in paths:
        print_ts(f"      • {p}", flush=True)

    print_ts(f"aggregate: summing {len(paths)} partial Hamiltonian(s)...", flush=True)
    agg_t0 = time.perf_counter()

    def _agg_progress(idx: int, n_total: int, path: Path, piece_meta: dict) -> None:
        elapsed = time.perf_counter() - agg_t0
        nnz = piece_meta.get("nnz", "?")
        print_ts(
            f"  → [{idx}/{n_total}] summed {path.name} "
            f"(dim={piece_meta.get('dim', '?')}, nnz={nnz}) "
            f"[{elapsed:.1f}s elapsed]",
            flush=True,
        )

    H, meta = aggregate_hamiltonians_from_paths(paths, progress=_agg_progress)

    labels = meta.get("basis_labels")
    if labels:
        quantum_bases = basis_labels_to_hilbert_bases(labels)
    else:
        dim_loaded = int(H.matrix.dim)
        quantum_bases = basis_labels_to_hilbert_bases(
            [str(i) for i in range(dim_loaded)]
        )
        print_ts(
            "  → Warning: no basis_labels in partial NPZ files; using numeric indices.",
            flush=True,
        )
    print_ts(
        f"  → aggregated dim(H) = {H.matrix.dim}, nnz ≈ {meta.get('nnz', '?')}",
        flush=True,
    )

    # Optional symmetrization (same logic / flag as eigensolver and jtd_run.py).
    if calc.symmetrize_hamiltonian:
        pre = check_matrix_operator_hermiticity(H)
        if not pre.is_hermitian:
            print_ts(
                "aggregate: applying Hermitian symmetrization (H + H†) / 2 to aggregated H...",
                flush=True,
            )
            H = H.symmetrize_hermitian()
            post = check_matrix_operator_hermiticity(H)
            print_ts(f"  → after symmetrization: {post.summary_line()}", flush=True)

    print_ts(f"aggregate: saving aggregated Hamiltonian to output_folder {out_dir} ...", flush=True)
    h_path = calc.resolve_hamiltonian_save_path(run_dir)
    basis_save = [str(s) for s in quantum_bases._ket_states]
    extra_metadata: dict[str, Any] = {
        "maximum_number_of_vibrational_quanta": calc.maximum_number_of_vibrational_quanta,
        "maximum_number_of_quanta_per_dim": calc.maximum_number_of_quanta_per_dim,
        "phonon_encut": calc.phonon_encut,
        "use_maximum_quanta_from_modes_csv": calc.use_maximum_quanta_from_modes_csv,
        "order": calc.order,
        "n_partials_aggregated": len(paths),
    }
    save_hamiltonian_npz(
        h_path,
        H,
        label=h_path.stem,
        basis_labels=basis_save,
        extra_metadata=extra_metadata,
    )

    print_ts("=" * 70, flush=True)
    print_ts("aggregation complete", flush=True)
    print_ts(f"  matrix: {describe_hamiltonian_matrix(H)}", flush=True)
    print_ts(f"  partials summed: {len(paths)}", flush=True)
    print_ts(f"  saved aggregated Hamiltonian to: {h_path}", flush=True)
    print_ts("=" * 70, flush=True)
    return 0


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except (OSError, ValueError):
            pass
    raise SystemExit(main())
