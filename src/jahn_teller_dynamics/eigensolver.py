"""
Stage 3 of the split PVC build pipeline: **aggregate and diagonalize**.

This script reads the partial Hamiltonian NPZ files produced by
:mod:`jahn_teller_dynamics.hamiltonian_builder`, sums them, optionally
symmetrizes the result via :math:`(H + H^\\dagger)/2`, and diagonalizes using
the same backend selection logic as :mod:`jahn_teller_dynamics.jtd_run`
(scipy dense / scipy sparse / PETSc-SLEPc).

Inputs are resolved through the ``load_hamiltonians`` cfg / CLI option from
:mod:`jahn_teller_dynamics.jtd_run`. Comma-separated tokens may be:

- glob patterns relative to the builder output directory (e.g.
  ``hamiltonian_part_*.npz``); the script expands them and sorts the matches,
- bare filenames under ``out_dir``,
- relative paths under the run directory,
- absolute paths.

Eigenvalues and eigenvectors are saved to ``out_dir`` in the same formats
as :mod:`jahn_teller_dynamics.jtd_run` (``save_csv`` / ``save_npz``).

Run::

    python3 -m jahn_teller_dynamics.eigensolver \\
        --config data/dJT_data/methoxy/builder_method/eigensolver.cfg
"""

from __future__ import annotations

import glob as _glob
from pathlib import Path
from typing import Any, Optional

import numpy as np

from jahn_teller_dynamics.jtd_run import (
    _args_to_calculation,
    _build_parser_with_cfg_defaults,
    _pvc_mpi_secondary_rank_skip_file_io,
    describe_hamiltonian_matrix,
)
from jahn_teller_dynamics.io.config.pvc_config import PVCCalculation
from jahn_teller_dynamics.io.utils.run_context import RunContext
from jahn_teller_dynamics.io.utils.timestamp_print import print_ts


def resolve_partial_hamiltonian_paths(
    calc: PVCCalculation, run_dir: Path, search_dir: Optional[Path] = None
) -> list[Path]:
    """
    Resolve ``load_hamiltonians`` to a flat sorted list of NPZ paths.

    Tokens with glob characters (``*``, ``?``, ``[``) are expanded against
    ``search_dir`` (when bare) or the run directory (when path-like). When
    ``search_dir`` is ``None`` it defaults to the configured output directory
    (:meth:`PVCCalculation.resolve_out_dir`); callers that store partials
    elsewhere (e.g. the aggregator reading from ``input_folder``) can pass an
    explicit directory.

    If a glob token matches no files, it is skipped with a warning. This lets a
    single ``load_hamiltonians`` list support multiple builder naming schemes
    (e.g. ``*_chunk_*.npz`` vs ``*_chunks_*.npz``). If *no* token matches any
    file overall, an error is raised.
    """
    tokens = [t.strip() for t in (calc.load_hamiltonians or "").split(",") if t.strip()]
    if not tokens:
        raise ValueError(
            "eigensolver: load_hamiltonians is empty — set it to a comma-separated "
            "list of partial Hamiltonian NPZ files (globs are allowed)."
        )
    base_dir = search_dir if search_dir is not None else calc.resolve_out_dir(run_dir)
    matched: list[Path] = []
    glob_chars = set("*?[")
    for token in tokens:
        p_token = Path(token).expanduser()
        if any(c in token for c in glob_chars):
            search_root: Path
            if p_token.is_absolute():
                search_root = p_token.parent
                pattern = p_token.name
            elif p_token.parent != Path("."):
                search_root = (run_dir / p_token.parent).resolve()
                pattern = p_token.name
            else:
                search_root = base_dir
                pattern = token
            hits = sorted(Path(h).resolve() for h in _glob.glob(str(search_root / pattern)))
            if not hits:
                print_ts(
                    f"eigensolver: warning — pattern {token!r} matched no files "
                    f"(searched in {search_root}); skipping.",
                    flush=True,
                )
                continue
            matched.extend(hits)
            continue
        # Non-glob token: delegate to the standard PVC resolver.
        from jahn_teller_dynamics.io.file_io.hamiltonian_npz import resolve_hamiltonian_file_path

        matched.append(resolve_hamiltonian_file_path(token, search_dir=base_dir, run_dir=run_dir))
    # Stable de-dup preserving first occurrence (sum each file once even when
    # globs overlap with explicit names).
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in matched:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        unique.append(rp)
    if not unique:
        raise FileNotFoundError(
            "eigensolver: none of the load_hamiltonians entries matched any files "
            f"(searched relative to search_dir={base_dir} and run_dir={run_dir})."
        )
    return unique


def main(argv: Optional[list[str]] = None) -> int:
    from jahn_teller_dynamics.io.file_io.csv_writer import CSVWriter
    from jahn_teller_dynamics.io.file_io.hamiltonian_npz import (
        aggregate_hamiltonians_from_paths,
        basis_labels_to_hilbert_bases,
    )
    from jahn_teller_dynamics.math_utils.eigen_solver import (
        assert_matrix_operator_hermitian,
        check_matrix_operator_hermiticity,
        create_pvc_eigen_solver,
        resolve_pvc_eigensolver_backend,
        slepc_kwargs_from_calc,
        sparse_eigsh_kwargs_from_calc,
    )

    print_ts("eigensolver: parsing CLI / config...", flush=True)
    parser = _build_parser_with_cfg_defaults(argv)
    args = parser.parse_args(argv)

    run_ctx = RunContext.from_cwd()
    run_dir = run_ctx.run_dir
    calc = _args_to_calculation(args, run_dir)

    backend = resolve_pvc_eigensolver_backend(
        calc.eigensolver,
        calc.running_environment,
        use_sparse=calc.use_sparse,
    )

    print_ts(f"eigensolver: resolving partial Hamiltonian inputs...", flush=True)
    paths = resolve_partial_hamiltonian_paths(calc, run_dir)
    print_ts(f"  → {len(paths)} partial Hamiltonian file(s):", flush=True)
    for p in paths:
        print_ts(f"      • {p}", flush=True)
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"eigensolver: Hamiltonian NPZ not found: {p}")

    # MPI bookkeeping: under SLEPc with multiple ranks, build the aggregated
    # CSR on rank 0 and broadcast (mirrors jtd_run.py behavior).
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
    should_aggregate_locally = not (
        backend == "slepc" and mpi_size > 1 and mpi_rank != builder_rank
    )

    H = None
    quantum_bases = None
    if should_aggregate_locally:
        print_ts(
            f"eigensolver: aggregating {len(paths)} partial Hamiltonian(s)...",
            flush=True,
        )
        import time as _time

        _agg_t0 = _time.perf_counter()

        def _agg_progress(idx: int, n_total: int, path: Path, piece_meta: dict) -> None:
            elapsed = _time.perf_counter() - _agg_t0
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

    if backend == "slepc" and mpi_size > 1 and mpi_comm is not None:
        from scipy.sparse import csr_matrix
        import jahn_teller_dynamics.math_utils.maths as maths
        import jahn_teller_dynamics.math_utils.matrix_mechanics as mm

        payload = None
        if mpi_rank == builder_rank:
            if H is None or quantum_bases is None:
                raise RuntimeError("Builder rank failed to aggregate H/basis for SLEPc.")
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
        H_csr_all = csr_matrix(
            (payload["data"], payload["indices"], payload["indptr"]),
            shape=payload["shape"],
        )
        H = mm.MatrixOperator(maths.SparseMatrix(H_csr_all))
        quantum_bases = payload["basis"]

    if H is None or quantum_bases is None:
        raise RuntimeError("eigensolver: failed to prepare aggregated H for diagonalization.")

    # Optional symmetrization (same logic as jtd_run.py).
    if calc.symmetrize_hamiltonian:
        pre = check_matrix_operator_hermiticity(H)
        if not pre.is_hermitian:
            print_ts(
                "eigensolver: applying Hermitian symmetrization (H + H†) / 2 to aggregated H...",
                flush=True,
            )
            H = H.symmetrize_hermitian()
            post = check_matrix_operator_hermiticity(H)
            print_ts(f"  → after symmetrization: {post.summary_line()}", flush=True)

    # Optionally persist the aggregated (and symmetrized) Hamiltonian. Set
    # save_hamiltonian_npz = true (alias of save_hamiltonian) in the cfg.
    if calc.save_hamiltonian and mpi_rank == 0:
        from jahn_teller_dynamics.io.file_io.hamiltonian_npz import save_hamiltonian_npz

        h_path = calc.resolve_hamiltonian_save_path(run_dir)
        basis_save = [str(s) for s in quantum_bases._ket_states]
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
        print_ts(f"eigensolver: saved aggregated Hamiltonian to: {h_path}", flush=True)

    if not calc.eigensolve:
        print_ts("eigensolver: eigensolve=false — skipping diagonalization.", flush=True)
        print_ts("=" * 70, flush=True)
        print_ts(f"dim(H) = {H.matrix.dim}", flush=True)
        print_ts("=" * 70, flush=True)
        return 0

    print_ts("eigensolver: diagonalizing aggregated Hamiltonian...", flush=True)
    dim_h = int(H.matrix.dim)
    num_of_vals_req = calc.num_eigs
    if num_of_vals_req is not None and num_of_vals_req >= dim_h:
        num_of_vals_req = None

    if backend == "sparse":
        k_sparse = dim_h if num_of_vals_req is None else int(num_of_vals_req)
        if k_sparse >= dim_h - 1:
            print_ts(
                f"  → num_eigs requests {k_sparse} of {dim_h} eigenpairs; "
                "sparse eigsh cannot supply a full spectrum — using dense eigensolver.",
                flush=True,
            )
            backend = "dense"

    hc = check_matrix_operator_hermiticity(H)
    if calc.require_hermitian and not hc.is_hermitian:
        assert_matrix_operator_hermitian(H, context="aggregated PVC Hamiltonian")
    if mpi_rank == 0:
        print_ts(f"  → Matrix being diagonalized: {describe_hamiltonian_matrix(H)}", flush=True)
        print_ts(f"  → Hermiticity check: {hc.summary_line()}", flush=True)
        if not hc.is_hermitian and not calc.require_hermitian:
            print_ts(
                "  → Warning: H is not Hermitian; using general eigensolver (eigs / SLEPc NHEP).",
                flush=True,
            )
    allow_non_hermitian = bool(not calc.require_hermitian and not hc.is_hermitian)

    eigen_solver = create_pvc_eigen_solver(backend, calc=calc)
    print_ts(
        f"  → Eigen backend: {backend}"
        + (
            " (PETSc/SLEPc — pass -eps_* options before script args if needed)"
            if backend == "slepc"
            else ""
        ),
        flush=True,
    )

    sw = (getattr(calc, "eigensolver_spectral_which", "") or "").strip()
    solve_kw: dict[str, Any] = {
        "num_of_vals": num_of_vals_req,
        "quantum_states_bases": quantum_bases,
        "spectral_sigma": getattr(calc, "eigensolver_sigma", None),
        "spectral_which": sw if sw else None,
        "allow_non_hermitian": allow_non_hermitian,
    }
    if backend == "sparse":
        solve_kw.update(sparse_eigsh_kwargs_from_calc(calc, dim=dim_h))
    elif backend == "slepc":
        solve_kw.update(slepc_kwargs_from_calc(calc))

    eig_space = eigen_solver.solve(H, **solve_kw)
    eig_vals = [k.eigen_val for k in eig_space.eigen_kets]
    print_ts(f"  → Computed {len(eig_vals)} eigenvalues/eigenvectors", flush=True)

    if _pvc_mpi_secondary_rank_skip_file_io(backend):
        return 0

    print_ts("eigensolver: saving results...", flush=True)
    out_dir = calc.resolve_out_dir(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if calc.save_csv:
        writer = CSVWriter(separator=calc.separator, index=True)
        eig_vec_path = str(out_dir / "eigenvectors.csv")
        eig_val_path = str(out_dir / "eigenvalues.csv")
        writer.write_eigen_vectors_and_values(eig_space, eig_vec_path, eig_val_path)
        print_ts(f"Saved eigenvectors to: {eig_vec_path}", flush=True)
        print_ts(f"Saved eigenvalues  to: {eig_val_path}", flush=True)

    if calc.save_npz:
        eig_vecs = np.column_stack(
            [np.array(ket.coeffs.coeffs).flatten() for ket in eig_space.eigen_kets]
        )
        eig_vals_arr = np.array([complex(k.eigen_val) for k in eig_space.eigen_kets])
        basis_labels = np.array(
            [str(s) for s in quantum_bases._ket_states], dtype=object
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
        print_ts(f"Saved NPZ to: {npz_path}", flush=True)

    print_ts("=" * 70, flush=True)
    print_ts("eigensolver diagonalization complete", flush=True)
    print_ts(f"dim(H) = {H.matrix.dim}", flush=True)
    print_ts(f"computed eigenvalues = {len(eig_vals)}", flush=True)
    print_ts("-" * 70, flush=True)
    for i, ev in enumerate(eig_vals):
        try:
            ev_real = float(ev.real)  # type: ignore[union-attr]
            print_ts(f"E[{i:03d}] = {ev_real:.12g}", flush=True)
        except Exception:
            print_ts(f"E[{i:03d}] = {ev}", flush=True)
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
