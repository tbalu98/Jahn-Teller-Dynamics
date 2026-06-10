"""
Stage 2 of the split PVC build pipeline: **build** one partial Hamiltonian NPZ.

This script reads the coupling-row NPZ produced by
:mod:`jahn_teller_dynamics.hamiltonian_planner` and writes **one** partial
:math:`H` to disk. Each job assembles a subset of the total Hamiltonian — either:

- a slice of the coupling rows ``[coupling_from_index, coupling_to_index)``,
- and/or the bare electron term :math:`H_{el} \\otimes I_{ph}` (+ optional
  :math:`H_{spin} \\otimes I_{ph}`) when ``include_electron_energy=true``,
- and/or the bare phonon term :math:`I_{el} \\otimes K_{ph}` when
  ``include_phonon_energy=true``.

Run the builder as many times as needed (Slurm array, separate launches, …);
each invocation produces a separate ``hamiltonian_part_*.npz`` file. The final
:mod:`jahn_teller_dynamics.eigensolver` script sums them and diagonalizes.

CLI / cfg knobs (``[essentials]``, ``[model]``, ``[builder]``, ``[eigensolver]``; legacy ``[PVC]`` alias):

- ``load_coupling_rows`` — path to planner NPZ. Defaults to
  ``{out_dir}/{coupling_rows_filename}``.
- ``coupling_from_index`` / ``coupling_to_index`` — half-open row range.
- ``coupling_row_chunk_index`` / ``coupling_row_chunk_count`` — split the row
  set into ``count`` equal chunks and build chunk ``index``. The chunk index
  can also be supplied via the environment variable ``SLURM_ARRAY_TASK_ID``.
- ``coupling_row_chunk_from`` / ``coupling_row_chunk_to`` (with
  ``coupling_row_chunk_count``) — build a half-open *range* of chunks
  ``[from, to)`` of the total ``count`` in a single builder invocation,
  emitting one combined partial NPZ. Lets one process cover several chunks.
- ``[builder]`` section (optional) — if present, parallel lined coupling is on
  by default. Keys: ``lined_coupling_batches``, ``lined_coupling_workers``,
  ``lined_coupling_pool`` (default ``spawn``). Set ``parallel_lined_coupling = false``
  to disable. Split-job keys (``load_coupling_rows``, chunk indices, …) also live here.
- ``include_electron_energy`` / ``include_phonon_energy`` — add bare terms.
- ``partial_hamiltonian_stem`` (default ``hamiltonian_part``) — output prefix.
- ``partial_hamiltonian_filename`` — explicit output filename (overrides stem).

Run::

    python3 -m jahn_teller_dynamics.hamiltonian_builder \\
        --config data/dJT_data/methoxy/builder_method/builder_coupling_array.cfg
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from jahn_teller_dynamics.jtd_run import _args_to_calculation, _build_parser_with_cfg_defaults
from jahn_teller_dynamics.io.config.pvc_config import PVCCalculation
from jahn_teller_dynamics.io.utils.run_context import RunContext
from jahn_teller_dynamics.io.utils.timestamp_print import print_ts


def resolve_coupling_rows_input_path(calc: PVCCalculation, run_dir: Path) -> Path:
    """
    Resolve the planner NPZ to load.

    Priority: ``load_coupling_rows`` (cfg / CLI) → ``coupling_rows_filename``.
    Bare filename → under :meth:`PVCCalculation.resolve_out_dir`.
    Relative path → under ``run_dir``. Absolute path → as-is. ``.npz`` is
    appended if missing.
    """
    name = (calc.load_coupling_rows or "").strip() or (calc.coupling_rows_filename or "coupling_rows.npz").strip()
    if not name.lower().endswith(".npz"):
        name = name + ".npz"
    p = Path(name).expanduser()
    if p.is_absolute():
        return p.resolve()
    if p.parent == Path("."):
        return (calc.resolve_out_dir(run_dir) / p).resolve()
    return (run_dir / p).resolve()


def load_planner_npz(path: Path) -> dict[str, Any]:
    """Load a planner NPZ into a plain dict (rebuilds coupling rows + metadata)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Planner NPZ not found: {path}")
    z = np.load(path, allow_pickle=True)

    def _scalar(key: str, default: Any = None) -> Any:
        if key not in z.files:
            return default
        arr = np.asarray(z[key])
        return arr.item() if arr.shape == () else arr.ravel()[0]

    def _array(key: str, default: Optional[np.ndarray] = None) -> np.ndarray:
        if key not in z.files:
            if default is None:
                raise KeyError(f"Required key missing in planner NPZ: {key}")
            return default
        return np.asarray(z[key])

    coeff = _array("rows_coeff_real").astype(np.float64) + 1j * _array(
        "rows_coeff_imag"
    ).astype(np.float64)
    out: dict[str, Any] = {
        "rows_el_state_1": _array("rows_el_state_1").astype(np.int32),
        "rows_el_state_2": _array("rows_el_state_2").astype(np.int32),
        "rows_polinom": [str(s) for s in _array("rows_polinom").tolist()],
        "rows_coeff": coeff.astype(np.complex128),
        "n_coupling_rows": int(_scalar("n_coupling_rows", 0)),
        "electron_energies": _array("electron_energies").astype(np.float64),
        "mode_labels": [str(s) for s in _array("mode_labels").tolist()],
        "mode_omegas": _array("mode_omegas").astype(np.float64),
        "mode_maximum_quanta": _array(
            "mode_maximum_quanta", np.asarray([], dtype=np.int32)
        ).astype(np.int32),
        "modes_to_use": _array("modes_to_use", np.asarray([], dtype=np.int32)).astype(
            np.int32
        ),
        "basis_labels": [str(s) for s in _array("basis_labels").tolist()],
        "phonon_truncation_kind": str(_scalar("phonon_truncation_kind", "constrained")),
        "order": int(_scalar("order", 0)),
        "maximum_quanta_per_mode": int(_scalar("maximum_quanta_per_mode", 0)),
        "maximum_total_phonon_quanta": int(_scalar("maximum_total_phonon_quanta", 0)),
        "phonon_encut": float(_scalar("phonon_encut", float("nan"))),
        "use_maximum_quanta_from_modes_csv": bool(
            _scalar("use_maximum_quanta_from_modes_csv", False)
        ),
        "use_sparse": bool(_scalar("use_sparse", True)),
        "dimensionless_coordinates": bool(_scalar("dimensionless_coordinates", True)),
        "null_point_vib": bool(_scalar("null_point_vib", True)),
        "exp_approximation_order": int(_scalar("exp_approximation_order", -1)),
        "hermitian_completion": bool(_scalar("hermitian_completion", True)),
        "diagonal_completion": bool(_scalar("diagonal_completion", False)),
        "permutation_completion": bool(_scalar("permutation_completion", False)),
        "n_orbitals": int(_scalar("n_orbitals", 0)),
        "dim_el": int(_scalar("dim_el", 0)),
        "dim_ph": int(_scalar("dim_ph", 0)),
        "dim_full": int(_scalar("dim_full", 0)),
        "has_spin_layout": bool(_scalar("has_spin_layout", False)),
        "spin_orbital_operator": _array(
            "spin_orbital_operator", np.empty((0, 0), dtype=np.complex128)
        ).astype(np.complex128),
        "modes_csv_path": str(_scalar("modes_csv_path", "")),
    }
    if out["has_spin_layout"]:
        out["orbital_spin_labels"] = [
            str(s) for s in _array("orbital_spin_labels").tolist()
        ]
        out["orbital_spin_S"] = _array("orbital_spin_S").astype(np.float64).tolist()
    return out


def rebuild_phonon_system(planner: dict[str, Any], *, build_log=None) -> Any:
    """Rebuild the phonon subsystem exactly as the planner did."""
    from jahn_teller_dynamics.physics.models.system_builder import (
        MultiModePhononSystem,
        build_phonon_system_constrained,
        build_phonon_system_energy_cutoff,
    )

    kind = planner["phonon_truncation_kind"]
    # Cast to plain Python floats — passing numpy.float64 into the phonon
    # operator construction causes numpy's __mul__ dispatch to return ndarrays
    # of MatrixOperator pieces instead of keeping the result as a MatrixOperator.
    modes = [float(w) for w in planner["mode_omegas"]]
    mode_names = [str(s) for s in planner["mode_labels"]]
    use_sparse = planner["use_sparse"]
    dimensionless = planner["dimensionless_coordinates"]
    nullpv = planner["null_point_vib"]
    exp_order = (
        None if planner["exp_approximation_order"] < 0 else planner["exp_approximation_order"]
    )

    if kind == "encut":
        return build_phonon_system_energy_cutoff(
            modes=modes,
            phonon_encut=float(planner["phonon_encut"]),
            use_sparse=use_sparse,
            dimensionless_coordinates=dimensionless,
            null_point_vib=nullpv,
            mode_names=mode_names,
            build_log=build_log,
            exp_approximation_order=exp_order,
        )
    if kind == "tensor_per_mode":
        per_mode = list(planner["mode_maximum_quanta"])
        if not per_mode:
            raise ValueError(
                "Planner NPZ marks tensor_per_mode truncation but stored no "
                "mode_maximum_quanta — re-run the planner."
            )
        mode_cfgs = [(w, [lab]) for w, lab in zip(modes, mode_names)]
        return MultiModePhononSystem(
            modes=mode_cfgs,
            order=[int(n) for n in per_mode],
            use_sparse=use_sparse,
            dimensionless_coordinates=dimensionless,
            null_point_vib=nullpv,
            exp_approximation_order=exp_order,
        )
    if kind == "tensor_uniform":
        mode_cfgs = [(w, [lab]) for w, lab in zip(modes, mode_names)]
        return MultiModePhononSystem(
            modes=mode_cfgs,
            order=int(planner["maximum_quanta_per_mode"]),
            use_sparse=use_sparse,
            dimensionless_coordinates=dimensionless,
            null_point_vib=nullpv,
            exp_approximation_order=exp_order,
        )
    # Default: total-quanta constrained.
    n_total = int(planner["maximum_total_phonon_quanta"]) or int(planner["order"])
    return build_phonon_system_constrained(
        modes=modes,
        order=n_total,
        use_sparse=use_sparse,
        dimensionless_coordinates=dimensionless,
        null_point_vib=nullpv,
        mode_names=mode_names,
        build_log=build_log,
        exp_approximation_order=exp_order,
    )


def build_orbital_spin_layout_from_planner(planner: dict[str, Any]):
    """Rebuild :class:`OrbitalSpinLayout` from planner metadata (or ``None``)."""
    if not planner["has_spin_layout"]:
        return None
    from jahn_teller_dynamics.physics.models.orbital_spin_electron import (
        build_orbital_spin_layout,
    )

    return build_orbital_spin_layout(
        planner["orbital_spin_labels"], planner["orbital_spin_S"]
    )


def build_spin_orbital_operator_from_planner(
    planner: dict[str, Any], *, use_sparse: bool
):
    """Wrap stored dense spin–orbital operator as a :class:`MatrixOperator`."""
    arr = planner["spin_orbital_operator"]
    if arr.size == 0:
        return None
    import jahn_teller_dynamics.math_utils.maths as maths
    import jahn_teller_dynamics.math_utils.matrix_mechanics as mm

    if use_sparse:
        from scipy.sparse import csr_matrix

        return mm.MatrixOperator(maths.SparseMatrix(csr_matrix(arr)))
    return mm.MatrixOperator(maths.Matrix(np.asmatrix(arr)))


def _resolve_modes_csv_path(
    calc: PVCCalculation, planner: dict[str, Any], run_dir: Path
) -> Optional[Path]:
    """
    Pick the modes CSV path used to construct a :class:`PhononRebuildSpec`.

    Priority: cfg-resolved ``modes`` (via :meth:`PVCCalculation.resolve_input_paths`,
    using ``input_folder`` / ``data_dir`` / explicit override) → planner NPZ's
    embedded ``modes_csv_path``. Returns ``None`` if neither path exists on disk.
    """
    cfg_modes_name = (calc.modes or "").strip()
    cfg_modes_path = (calc.modes_path or "").strip()
    if cfg_modes_name or cfg_modes_path:
        try:
            _, modes_path, _ = calc.resolve_input_paths(run_dir)
            if modes_path.exists():
                return modes_path
        except Exception:
            pass
    embedded = (planner.get("modes_csv_path") or "").strip()
    if embedded:
        p = Path(embedded).expanduser()
        if p.exists():
            return p.resolve()
    return None


def build_phonon_rebuild_spec_from_planner(
    planner: dict[str, Any], modes_csv_path: Path
):
    """
    Translate planner NPZ metadata + ``modes_csv_path`` into a
    :class:`PhononRebuildSpec` suitable for spawn-pool builder workers.
    """
    from jahn_teller_dynamics.physics.hamiltonians.djt_polynomial_hamiltonian import (
        PhononRebuildSpec,
    )

    mode_numbers: Optional[tuple] = None
    if planner.get("modes_to_use") is not None and len(planner["modes_to_use"]) > 0:
        mode_numbers = tuple(int(m) for m in planner["modes_to_use"])

    exp_order = (
        None
        if int(planner.get("exp_approximation_order", -1)) < 0
        else int(planner["exp_approximation_order"])
    )
    encut_val: Optional[float] = None
    raw_encut = float(planner.get("phonon_encut", float("nan")))
    if raw_encut == raw_encut:  # not NaN
        encut_val = raw_encut

    return PhononRebuildSpec(
        modes_csv_path=str(modes_csv_path),
        order=int(planner.get("order", 0)),
        maximum_quanta_per_mode=int(planner.get("maximum_quanta_per_mode", 0)),
        maximum_total_phonon_quanta=int(planner.get("maximum_total_phonon_quanta", 0)),
        use_sparse=bool(planner.get("use_sparse", True)),
        dimensionless_coordinates=bool(planner.get("dimensionless_coordinates", True)),
        null_point_vib=bool(planner.get("null_point_vib", True)),
        exp_approximation_order=exp_order,
        mode_numbers=mode_numbers,
        phonon_encut=encut_val,
        use_maximum_quanta_from_modes_csv=bool(
            planner.get("use_maximum_quanta_from_modes_csv", False)
        ),
    )


def resolve_row_slice(
    calc: PVCCalculation, n_rows: int
) -> tuple[int, int, Optional[int], Optional[tuple[int, int]]]:
    """
    Resolve ``[from_index, to_index)`` given cfg / CLI / Slurm env.

    Returns ``(from_index, to_index, chunk_index, chunk_range)`` where:

    - ``chunk_index`` is the resolved single chunk index (or ``None`` when
      explicit row indices or a chunk range were used);
    - ``chunk_range`` is ``(chunk_from, chunk_to)`` half-open when the builder
      should cover that span of chunks (otherwise ``None``).

    Empty slices ``from == to`` are valid and mean "no coupling rows in this job".

    Priority of slice specs (first present wins):

    1. ``coupling_from_index`` / ``coupling_to_index`` — explicit row range.
    2. ``coupling_row_chunk_from`` / ``coupling_row_chunk_to`` — half-open
       chunk range over ``coupling_row_chunk_count`` total chunks.
    3. ``coupling_row_chunk_index`` (or ``SLURM_ARRAY_TASK_ID``) +
       ``coupling_row_chunk_count`` — single chunk index.
    4. No slicing → build all coupling rows in this job.
    """
    # 1) Explicit row indices.
    if calc.coupling_from_index is not None or calc.coupling_to_index is not None:
        f = int(calc.coupling_from_index) if calc.coupling_from_index is not None else 0
        t = int(calc.coupling_to_index) if calc.coupling_to_index is not None else n_rows
        if f < 0:
            f = max(0, n_rows + f)
        if t < 0:
            t = max(0, n_rows + t)
        f = max(0, min(n_rows, f))
        t = max(f, min(n_rows, t))
        return f, t, None, None

    chunk_count = calc.coupling_row_chunk_count

    # 2) Half-open chunk range.
    if calc.coupling_row_chunk_from is not None or calc.coupling_row_chunk_to is not None:
        if chunk_count is None or chunk_count <= 0:
            raise ValueError(
                "coupling_row_chunk_from / coupling_row_chunk_to require "
                "coupling_row_chunk_count > 0."
            )
        chunk_count = int(chunk_count)
        cf = (
            int(calc.coupling_row_chunk_from)
            if calc.coupling_row_chunk_from is not None
            else 0
        )
        ct = (
            int(calc.coupling_row_chunk_to)
            if calc.coupling_row_chunk_to is not None
            else chunk_count
        )
        if cf < 0 or ct > chunk_count or cf > ct:
            raise ValueError(
                f"coupling_row_chunk_from={cf}, coupling_row_chunk_to={ct} "
                f"out of range for coupling_row_chunk_count={chunk_count}."
            )
        if n_rows == 0:
            return 0, 0, None, (cf, ct)
        chunk_size = (n_rows + chunk_count - 1) // chunk_count
        f = cf * chunk_size
        t = min(n_rows, ct * chunk_size)
        return f, t, None, (cf, ct)

    # 3) Single chunk index (CLI / cfg / SLURM_ARRAY_TASK_ID).
    chunk_idx = calc.coupling_row_chunk_index
    if chunk_idx is None:
        env = os.environ.get("SLURM_ARRAY_TASK_ID", "").strip()
        if env:
            try:
                chunk_idx = int(env)
            except ValueError:
                chunk_idx = None
    if chunk_idx is None and chunk_count is None:
        # 4) No slicing requested: build the whole coupling block in this job.
        return 0, n_rows, None, None
    if chunk_count is None or chunk_count <= 0:
        raise ValueError(
            "coupling_row_chunk_index requires coupling_row_chunk_count > 0."
        )
    if chunk_idx is None:
        raise ValueError(
            "coupling_row_chunk_count was set but no chunk index "
            "(coupling_row_chunk_index or SLURM_ARRAY_TASK_ID) was provided."
        )
    chunk_idx = int(chunk_idx)
    chunk_count = int(chunk_count)
    if chunk_idx < 0 or chunk_idx >= chunk_count:
        raise ValueError(
            f"coupling_row_chunk_index={chunk_idx} out of range "
            f"[0, {chunk_count})."
        )
    if n_rows == 0:
        return 0, 0, chunk_idx, None
    chunk_size = (n_rows + chunk_count - 1) // chunk_count
    f = chunk_idx * chunk_size
    t = min(n_rows, f + chunk_size)
    return f, t, chunk_idx, None


def resolve_partial_output_path(
    calc: PVCCalculation,
    run_dir: Path,
    *,
    from_index: int,
    to_index: int,
    include_bare: bool,
    has_coupling: bool,
    chunk_index: Optional[int],
    chunk_range: Optional[tuple[int, int]] = None,
) -> Path:
    """Pick the output path for the partial Hamiltonian NPZ."""
    explicit = (calc.partial_hamiltonian_filename or "").strip()
    if explicit:
        name = explicit
    else:
        stem = (calc.partial_hamiltonian_stem or "hamiltonian_part").strip() or "hamiltonian_part"
        if not has_coupling and include_bare:
            name = f"{stem}_bare.npz"
        elif chunk_range is not None and not include_bare:
            cf, ct = chunk_range
            name = f"{stem}_chunks_{int(cf):05d}_{int(ct):05d}.npz"
        elif chunk_index is not None and not include_bare:
            name = f"{stem}_chunk_{int(chunk_index):05d}.npz"
        else:
            name = f"{stem}_{int(from_index):05d}_{int(to_index):05d}.npz"
            if include_bare:
                name = name.replace(".npz", "_with_bare.npz")
    if not name.lower().endswith(".npz"):
        name = name + ".npz"
    p = Path(name).expanduser()
    if p.is_absolute():
        return p.resolve()
    if p.parent == Path("."):
        return (calc.resolve_out_dir(run_dir) / p).resolve()
    return (run_dir / p).resolve()


def _assemble_coupling_slice(
    *,
    phonon,
    sliced_assembly: list,
    n_orbitals: int,
    use_sparse: bool,
    orbital_spin_layout,
    hermitian_completion: bool,
    log_expressions: bool,
    parallel: bool,
    n_memory_batches: int,
    n_workers: Optional[int],
    pool_method: str,
    phonon_rebuild,
    has_spin_layout: bool,
):
    """
    Build the coupling-row contribution for an already-sliced assembly plan.

    Mirrors :func:`_add_lined_coupling_batched` from the lined builder but
    operates on the half-open chunk/row slice picked by the builder. Falls back
    to serial assembly when ``parallel`` is disabled, when the multiprocessing
    context is unavailable, or when a spin–orbital layout would otherwise be
    silently dropped by the spawn worker.
    """
    from jahn_teller_dynamics.physics.hamiltonians.djt_polynomial_hamiltonian import (
        CouplingBatchBuildSpec,
        _accumulate_coupling_rows_streaming,
        _assembly_to_tuples,
        _chunk_rows,
        _load_partial_coupling_npz,
        _merge_coupling_into_hamiltonian,
        _pick_batch_process_workers,
        _resolve_mp_pool_context,
        _warn_parallel_batch_memory,
        build_coupling_batch_to_npz,
    )
    import gc
    import os as _os
    import shutil
    import tempfile

    stream_kw = dict(
        n_orbitals=int(n_orbitals),
        use_sparse=bool(use_sparse),
        orbital_spin_layout=orbital_spin_layout,
        hermitian_completion=bool(hermitian_completion),
    )

    n_batches = max(1, int(n_memory_batches))
    batches = _chunk_rows(sliced_assembly, n_batches)
    n_proc = _pick_batch_process_workers(n_workers, len(batches)) if parallel else 1

    if not parallel or n_proc <= 1:
        print_ts(
            f"  → {len(sliced_assembly)} coupling assembly entry(ies), "
            f"{len(batches)} memory batch(es) (serial)",
            flush=True,
        )
        partial = None
        for batch_idx, batch in enumerate(batches, start=1):
            if not batch:
                continue
            print_ts(
                f"  → memory batch {batch_idx}/{len(batches)} (serial)", flush=True
            )
            block = _accumulate_coupling_rows_streaming(
                phonon, batch, log_expressions=log_expressions, **stream_kw
            )
            partial = block if partial is None else _merge_coupling_into_hamiltonian(
                partial, block
            )
        return partial

    # Parallel path: spawn pool only. Fork pool inside the builder would need
    # the parent process to share its phonon with workers via module globals
    # (mirroring jtd_run.py's _mp_fork_build_batch); we keep things simple here
    # and require a phonon_rebuild + spawn for safe HPC behaviour. Without it,
    # we run serially below.
    can_spawn = phonon_rebuild is not None and not has_spin_layout
    if not can_spawn:
        if has_spin_layout:
            print_ts(
                "  → warning: spin–orbital layout is present and the spawn "
                "worker would drop it; falling back to serial memory batches.",
                flush=True,
            )
        else:
            print_ts(
                "  → warning: no PhononRebuildSpec is available (cfg.modes "
                "empty and planner NPZ has no modes_csv_path); falling back "
                "to serial memory batches.",
                flush=True,
            )
        partial = None
        for batch_idx, batch in enumerate(batches, start=1):
            if not batch:
                continue
            print_ts(
                f"  → memory batch {batch_idx}/{len(batches)} (serial)", flush=True
            )
            block = _accumulate_coupling_rows_streaming(
                phonon, batch, log_expressions=log_expressions, **stream_kw
            )
            partial = block if partial is None else _merge_coupling_into_hamiltonian(
                partial, block
            )
        return partial

    ctx, pool_kind = _resolve_mp_pool_context("spawn")
    if ctx is None:
        print_ts(
            "  → spawn multiprocessing context unavailable; building memory "
            "batches serially.",
            flush=True,
        )
        partial = None
        for batch_idx, batch in enumerate(batches, start=1):
            if not batch:
                continue
            block = _accumulate_coupling_rows_streaming(
                phonon, batch, log_expressions=log_expressions, **stream_kw
            )
            partial = block if partial is None else _merge_coupling_into_hamiltonian(
                partial, block
            )
        return partial

    tmpdir = tempfile.mkdtemp(prefix="jt_pvc_builder_batch_")
    print_ts(
        f"  → {len(sliced_assembly)} coupling assembly entry(ies), "
        f"{len(batches)} memory batch(es), {n_proc} worker process(es) "
        f"({pool_kind} pool, NPZ spill)",
        flush=True,
    )
    _warn_parallel_batch_memory(n_proc, len(batches), pool_kind=pool_kind)
    partial = None
    try:
        assert phonon_rebuild is not None
        specs = [
            CouplingBatchBuildSpec(
                phonon=phonon_rebuild,
                batch_rows=_assembly_to_tuples(batch),
                n_orbitals=int(n_orbitals),
                use_sparse=bool(use_sparse),
                hermitian_completion=bool(hermitian_completion),
                output_path=_os.path.join(tmpdir, f"batch_{i:05d}.npz"),
            )
            for i, batch in enumerate(batches)
            if batch
        ]
        with ctx.Pool(processes=n_proc, maxtasksperchild=1) as pool:
            for batch_idx, batch_path in enumerate(
                pool.imap(build_coupling_batch_to_npz, specs, chunksize=1),
                start=1,
            ):
                print_ts(
                    f"  → merged memory batch {batch_idx}/{len(specs)}",
                    flush=True,
                )
                if not batch_path:
                    continue
                block = _load_partial_coupling_npz(batch_path)
                try:
                    _os.remove(batch_path)
                except OSError:
                    pass
                partial = block if partial is None else _merge_coupling_into_hamiltonian(
                    partial, block
                )
                gc.collect()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return partial


def main(argv: Optional[list[str]] = None) -> int:
    from jahn_teller_dynamics.physics.hamiltonians.djt_polynomial_hamiltonian import (
        DJTCouplingRow,
        _coupling_assembly_plan,
        _coupling_polinom_group_key,
        build_bare_vibronic_hamiltonian,
        phonon_position_operators_commute,
    )
    from jahn_teller_dynamics.io.file_io.hamiltonian_npz import save_hamiltonian_npz
    import jahn_teller_dynamics.math_utils.matrix_mechanics as mm

    import argparse

    print_ts("hamiltonian_builder: parsing CLI / config...", flush=True)
    parser = _build_parser_with_cfg_defaults(argv)
    # All builder-specific CLI flags use argparse.SUPPRESS so omitting them
    # does not clobber values that came from the .cfg via set_defaults().
    parser.add_argument(
        "--load-coupling-rows",
        dest="load_coupling_rows",
        type=str,
        default=argparse.SUPPRESS,
        metavar="PATH",
        help="Path (or bare name under out_dir) of the planner NPZ.",
    )
    parser.add_argument(
        "--coupling-from-index",
        dest="coupling_from_index",
        type=int,
        default=argparse.SUPPRESS,
        metavar="I",
        help="Half-open coupling-row range start index.",
    )
    parser.add_argument(
        "--coupling-to-index",
        dest="coupling_to_index",
        type=int,
        default=argparse.SUPPRESS,
        metavar="J",
        help="Half-open coupling-row range stop index (exclusive).",
    )
    parser.add_argument(
        "--coupling-row-chunk-index",
        dest="coupling_row_chunk_index",
        type=int,
        default=argparse.SUPPRESS,
        metavar="I",
        help="Chunk index (overrides SLURM_ARRAY_TASK_ID; needs --coupling-row-chunk-count).",
    )
    parser.add_argument(
        "--coupling-row-chunk-count",
        dest="coupling_row_chunk_count",
        type=int,
        default=argparse.SUPPRESS,
        metavar="N",
        help="Number of equal-sized chunks to split the row set into.",
    )
    parser.add_argument(
        "--coupling-row-chunk-from",
        dest="coupling_row_chunk_from",
        type=int,
        default=argparse.SUPPRESS,
        metavar="I",
        help=(
            "Half-open chunk range start index. With --coupling-row-chunk-to "
            "and --coupling-row-chunk-count, a single builder process covers "
            "chunks [I, J) and writes one combined partial NPZ."
        ),
    )
    parser.add_argument(
        "--coupling-row-chunk-to",
        dest="coupling_row_chunk_to",
        type=int,
        default=argparse.SUPPRESS,
        metavar="J",
        help=(
            "Half-open chunk range stop index (exclusive). Used with "
            "--coupling-row-chunk-from and --coupling-row-chunk-count."
        ),
    )
    parser.add_argument(
        "--include-electron-energy",
        dest="include_electron_energy",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Add diag(E_el) ⊗ I_ph (+ optional H_spin ⊗ I_ph) to the partial.",
    )
    parser.add_argument(
        "--include-phonon-energy",
        dest="include_phonon_energy",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Add I_el ⊗ K_ph to the partial.",
    )
    parser.add_argument(
        "--partial-hamiltonian-stem",
        dest="partial_hamiltonian_stem",
        type=str,
        default=argparse.SUPPRESS,
        metavar="STEM",
        help="Output filename prefix (default: hamiltonian_part).",
    )
    parser.add_argument(
        "--partial-hamiltonian-filename",
        dest="partial_hamiltonian_filename",
        type=str,
        default=argparse.SUPPRESS,
        metavar="NAME",
        help="Explicit output filename (overrides stem-based naming).",
    )
    args = parser.parse_args(argv)
    quiet = bool(getattr(args, "quiet", False))

    run_ctx = RunContext.from_cwd()
    run_dir = run_ctx.run_dir
    calc = _args_to_calculation(args, run_dir)

    coupling_npz = resolve_coupling_rows_input_path(calc, run_dir)
    print_ts(f"hamiltonian_builder: loading coupling rows from {coupling_npz}", flush=True)
    planner = load_planner_npz(coupling_npz)
    n_rows = planner["n_coupling_rows"]
    print_ts(
        f"  → n_coupling_rows={n_rows}, dim(electron)={planner['dim_el']}, "
        f"dim(phonon)={planner['dim_ph']}, dim(full)={planner['dim_full']}, "
        f"truncation={planner['phonon_truncation_kind']}",
        flush=True,
    )

    from_idx, to_idx, chunk_idx, chunk_range = resolve_row_slice(calc, n_rows)
    include_el = bool(calc.include_electron_energy)
    include_ph = bool(calc.include_phonon_energy)
    has_coupling = to_idx > from_idx
    include_bare = include_el or include_ph
    if not has_coupling and not include_bare:
        raise ValueError(
            "hamiltonian_builder: nothing to build — empty coupling slice and no bare terms "
            "(set include_electron_energy / include_phonon_energy, "
            "or widen [coupling_from_index, coupling_to_index))."
        )
    if has_coupling:
        if chunk_range is not None:
            cf, ct = chunk_range
            slice_desc = (
                f"chunks [{cf}, {ct}) of {int(calc.coupling_row_chunk_count)}"
            )
        elif chunk_idx is not None:
            slice_desc = f"chunk_index={chunk_idx}"
        else:
            slice_desc = "chunk_index=None"
        print_ts(
            f"  → coupling slice [{from_idx}, {to_idx}) "
            f"({to_idx - from_idx} row(s) of {n_rows}; {slice_desc})",
            flush=True,
        )
    if include_el:
        print_ts("  → including electron diagonal term (H_el ⊗ I_ph)", flush=True)
    if include_ph:
        print_ts("  → including phonon kinetic term (I_el ⊗ K_ph)", flush=True)

    build_log = None if quiet else (lambda m: print_ts(m, flush=True))

    print_ts("hamiltonian_builder: rebuilding phonon subsystem...", flush=True)
    t_ph = time.perf_counter()
    phonon = rebuild_phonon_system(planner, build_log=build_log)
    print_ts(
        f"  → phonon ready (dim={phonon.dim}, wall {time.perf_counter() - t_ph:.2f}s)",
        flush=True,
    )

    orbital_spin_layout = build_orbital_spin_layout_from_planner(planner)
    spin_orbital_op = build_spin_orbital_operator_from_planner(
        planner, use_sparse=planner["use_sparse"]
    )

    h: Optional[mm.MatrixOperator] = None

    if include_bare:
        electron_energies = planner["electron_energies"]
        if include_el and include_ph:
            print_ts(
                "  → building bare vibronic Hamiltonian (electron + phonon[ + spin-orbital])...",
                flush=True,
            )
            h_bare = build_bare_vibronic_hamiltonian(
                phonon,
                electron_energies,
                spin_orbital_operator=spin_orbital_op,
            )
            h = _accumulate(h, h_bare)
        else:
            if include_el:
                print_ts(
                    "  → building electron diagonal term (H_el ⊗ I_ph"
                    + (" + H_spin ⊗ I_ph" if spin_orbital_op is not None else "")
                    + ")...",
                    flush=True,
                )
                h_el = _build_electron_term(
                    phonon=phonon,
                    electron_energies=electron_energies,
                    spin_orbital_operator=spin_orbital_op,
                )
                h = _accumulate(h, h_el)
            if include_ph:
                print_ts(
                    "  → building phonon kinetic term (I_el ⊗ K_ph)...", flush=True
                )
                h_ph = _build_phonon_term(phonon=phonon, dim_el=int(planner["dim_el"]))
                h = _accumulate(h, h_ph)

    if has_coupling:
        rows = [
            DJTCouplingRow(
                int(planner["rows_el_state_1"][i]),
                int(planner["rows_el_state_2"][i]),
                str(planner["rows_polinom"][i]),
                complex(planner["rows_coeff"][i]),
            )
            for i in range(n_rows)
        ]
        assembly = _coupling_assembly_plan(
            rows, hermitian_completion=planner["hermitian_completion"]
        )
        commute = phonon_position_operators_commute(phonon)
        sorted_assembly = sorted(
            assembly,
            key=lambda e: _coupling_polinom_group_key(
                e[0].polinom, position_operators_commute=commute
            ),
        )
        sliced = sorted_assembly[from_idx:to_idx]

        parallel = bool(calc.parallel_lined_coupling)
        n_workers = calc.lined_coupling_workers
        n_mem_batches = int(calc.lined_coupling_batches or 1)
        pool_method = (calc.lined_coupling_pool or "spawn").strip().lower()
        if pool_method != "spawn":
            print_ts(
                f"  → note: lined_coupling_pool={pool_method!r} is not "
                "supported by the builder; using spawn pool (the only safe "
                "option here — the builder cannot share its phonon system "
                "with fork workers).",
                flush=True,
            )
            pool_method = "spawn"
        phonon_rebuild = None
        if parallel:
            modes_csv_path = _resolve_modes_csv_path(calc, planner, run_dir)
            if modes_csv_path is not None:
                phonon_rebuild = build_phonon_rebuild_spec_from_planner(
                    planner, modes_csv_path
                )

        print_ts(
            f"hamiltonian_builder: assembling coupling slice "
            f"({len(sliced)} assembly entries after sort+slice; "
            f"parallel={parallel}, batches={n_mem_batches}, "
            f"workers={n_workers}, pool={pool_method})...",
            flush=True,
        )
        t_cpl = time.perf_counter()
        partial = _assemble_coupling_slice(
            phonon=phonon,
            sliced_assembly=sliced,
            n_orbitals=int(planner["n_orbitals"]),
            use_sparse=bool(planner["use_sparse"]),
            orbital_spin_layout=orbital_spin_layout,
            hermitian_completion=planner["hermitian_completion"],
            log_expressions=not quiet,
            parallel=parallel,
            n_memory_batches=n_mem_batches,
            n_workers=n_workers,
            pool_method=pool_method,
            phonon_rebuild=phonon_rebuild,
            has_spin_layout=bool(planner.get("has_spin_layout", False)),
        )
        print_ts(
            f"  → coupling slice ready (wall {time.perf_counter() - t_cpl:.2f}s)",
            flush=True,
        )
        if partial is not None:
            h = _accumulate(h, partial)

    if h is None:
        # Should have been caught above; defensive.
        raise RuntimeError("hamiltonian_builder: no terms were assembled.")

    out_path = resolve_partial_output_path(
        calc,
        run_dir,
        from_index=from_idx,
        to_index=to_idx,
        include_bare=include_bare,
        has_coupling=has_coupling,
        chunk_index=chunk_idx,
        chunk_range=chunk_range,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "partial": True,
        "from_index": int(from_idx),
        "to_index": int(to_idx),
        "n_rows_total": int(n_rows),
        "include_electron_energy": include_el,
        "include_phonon_energy": include_ph,
        "phonon_truncation_kind": str(planner["phonon_truncation_kind"]),
        "order": int(planner["order"]),
        "maximum_number_of_vibrational_quanta": int(
            planner["maximum_total_phonon_quanta"]
        ),
        "maximum_number_of_quanta_per_dim": int(planner["maximum_quanta_per_mode"]),
    }
    if chunk_idx is not None:
        metadata["chunk_index"] = int(chunk_idx)
    if chunk_range is not None:
        metadata["chunk_from"] = int(chunk_range[0])
        metadata["chunk_to"] = int(chunk_range[1])
    if calc.coupling_row_chunk_count is not None:
        metadata["chunk_count"] = int(calc.coupling_row_chunk_count)
    save_hamiltonian_npz(
        out_path,
        h,
        label=out_path.stem,
        basis_labels=planner["basis_labels"],
        extra_metadata=metadata,
    )
    print_ts(f"hamiltonian_builder: saved partial Hamiltonian to: {out_path}", flush=True)
    print_ts(
        f"  → dim(H)={h.matrix.dim}, rows assembled={to_idx - from_idx}, "
        f"bare={'on' if include_bare else 'off'}",
        flush=True,
    )
    return 0


def _accumulate(acc: Optional[Any], term: Optional[Any]) -> Optional[Any]:
    if term is None:
        return acc
    if acc is None:
        return term
    return acc + term


def _build_electron_term(*, phonon, electron_energies, spin_orbital_operator):
    """Construct ``H_el ⊗ I_ph`` (+ optional ``H_spin ⊗ I_ph``) alone."""
    import jahn_teller_dynamics.math_utils.maths as maths
    import jahn_teller_dynamics.math_utils.matrix_mechanics as mm
    from scipy.sparse import diags

    dim_el = len(electron_energies)
    matrix_type = maths.SparseMatrix if phonon.use_sparse else maths.Matrix
    id_ph = mm.MatrixOperator.create_id_matrix_op(dim=phonon.dim, matrix_type=matrix_type)
    diag_vals = np.asarray(electron_energies, dtype=np.complex128)
    if phonon.use_sparse:
        op_el = mm.MatrixOperator(
            maths.SparseMatrix(diags(diag_vals, offsets=0, shape=(dim_el, dim_el)).tocsr())
        )
    else:
        op_el = mm.MatrixOperator(maths.Matrix(np.asmatrix(np.diag(diag_vals))))
    term = op_el ** id_ph
    if spin_orbital_operator is not None:
        term = term + spin_orbital_operator ** id_ph
    return term


def _build_phonon_term(*, phonon, dim_el):
    """Construct ``I_el ⊗ K_ph`` alone."""
    import jahn_teller_dynamics.math_utils.maths as maths
    import jahn_teller_dynamics.math_utils.matrix_mechanics as mm

    matrix_type = maths.SparseMatrix if phonon.use_sparse else maths.Matrix
    id_el = mm.MatrixOperator.create_id_matrix_op(dim=int(dim_el), matrix_type=matrix_type)
    return id_el ** phonon.operators["K"]


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except (OSError, ValueError):
            pass
    raise SystemExit(main())
