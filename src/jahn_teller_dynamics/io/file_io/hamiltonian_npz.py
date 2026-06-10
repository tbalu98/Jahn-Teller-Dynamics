"""
Save and load PVC Hamiltonian matrices as compressed NPZ (CSR or dense).

Used for split build / aggregate / diagonalize workflows on memory-limited nodes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
from scipy.sparse import csr_matrix, load_npz

import jahn_teller_dynamics.math_utils.maths as maths
import jahn_teller_dynamics.math_utils.matrix_mechanics as mm
from jahn_teller_dynamics.math_utils.matrix_mechanics import hilber_space_bases


def parse_hamiltonian_name_list(raw: str) -> list[str]:
    """Comma-separated Hamiltonian stems or paths (whitespace trimmed)."""
    return [part.strip() for part in (raw or "").split(",") if part.strip()]


def resolve_hamiltonian_file_path(
    name: str,
    *,
    search_dir: Path,
    run_dir: Path,
) -> Path:
    """
    Resolve a Hamiltonian file path from a cfg entry.

    - Absolute paths are used as-is (``.npz`` appended if missing).
    - Relative paths with a directory component are resolved under ``run_dir``.
    - Bare names are looked up under ``search_dir`` as ``{name}.npz``.
    """
    token = (name or "").strip()
    if not token:
        raise ValueError("Empty Hamiltonian name in load_hamiltonians.")
    p = Path(token).expanduser()
    if not p.name.lower().endswith(".npz"):
        p = p.with_suffix(".npz")
    if p.is_absolute():
        return p.resolve()
    if p.parent != Path("."):
        return (run_dir / p).resolve()
    return (search_dir / p.name).resolve()


def _matrix_operator_to_csr(op: mm.MatrixOperator) -> csr_matrix:
    matrix = op.matrix
    if isinstance(matrix, maths.SparseMatrix):
        return matrix.matrix.tocsr()
    if isinstance(matrix, maths.Matrix):
        return csr_matrix(np.asarray(matrix.matrix, dtype=np.complex128))
    raise TypeError(f"Unsupported matrix backing type: {type(matrix)}")


def save_hamiltonian_npz(
    path: Union[str, Path],
    hamiltonian: mm.MatrixOperator,
    *,
    label: str = "",
    basis_labels: Optional[Sequence[str]] = None,
    extra_metadata: Optional[dict[str, Any]] = None,
) -> Path:
    """
    Write ``hamiltonian`` to ``path`` (``.npz`` suffix added if missing).

    Stores CSR ``indptr``, ``indices``, ``data``, ``shape``, optional ``basis_labels``,
    and scalar metadata fields (strings / numbers only).
    """
    out = Path(path).expanduser()
    if not out.name.lower().endswith(".npz"):
        out = out.with_suffix(".npz")
    out.parent.mkdir(parents=True, exist_ok=True)

    csr = _matrix_operator_to_csr(hamiltonian)
    payload: dict[str, Any] = {
        "format": np.array("csr"),
        "shape": np.asarray(csr.shape, dtype=np.int64),
        "indptr": np.ascontiguousarray(csr.indptr),
        "indices": np.ascontiguousarray(csr.indices),
        "data": np.ascontiguousarray(csr.data, dtype=np.complex128),
        "dim": np.int64(csr.shape[0]),
    }
    if label:
        payload["label"] = np.array(label)
    if basis_labels is not None:
        payload["basis_labels"] = np.asarray([str(x) for x in basis_labels], dtype=object)
    if extra_metadata:
        for key, val in extra_metadata.items():
            if isinstance(val, (int, float, bool, np.integer, np.floating)):
                payload[str(key)] = np.asarray(val)
            elif isinstance(val, str):
                payload[str(key)] = np.array(val)

    np.savez_compressed(out, **payload)
    return out.resolve()


def load_hamiltonian_npz(path: Union[str, Path]) -> tuple[mm.MatrixOperator, dict[str, Any]]:
    """Load a Hamiltonian written by :func:`save_hamiltonian_npz`."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Hamiltonian NPZ not found: {p}")

    z = np.load(p, allow_pickle=True)
    meta: dict[str, Any] = {"path": p}
    if "label" in z.files:
        meta["label"] = str(np.asarray(z["label"]).ravel()[0])
    if "basis_labels" in z.files:
        meta["basis_labels"] = [str(x) for x in np.asarray(z["basis_labels"]).ravel()]

    fmt = str(np.asarray(z["format"]).ravel()[0]) if "format" in z.files else "csr"
    if fmt == "csr":
        if all(k in z.files for k in ("indptr", "indices", "data", "shape")):
            shape = tuple(int(x) for x in np.asarray(z["shape"]).ravel()[:2])
            # Preserve the stored integer index dtype. Forcing int32 here would
            # overflow for matrices with >= 2**31 stored nonzeros: indptr[-1]
            # (= nnz) wraps to a negative value, and scipy then raises
            # "negative dimensions are not allowed" on the next binary op.
            indices_arr = np.asarray(z["indices"])
            indptr_arr = np.asarray(z["indptr"])
            if indices_arr.dtype.kind not in ("i", "u"):
                indices_arr = indices_arr.astype(np.int64)
            if indptr_arr.dtype.kind not in ("i", "u"):
                indptr_arr = indptr_arr.astype(np.int64)
            csr = csr_matrix(
                (
                    np.asarray(z["data"], dtype=np.complex128),
                    indices_arr,
                    indptr_arr,
                ),
                shape=shape,
            )
        else:
            csr = load_npz(p).tocsr().astype(np.complex128)
        op = mm.MatrixOperator(maths.SparseMatrix(csr))
        meta["dim"] = int(csr.shape[0])
        meta["nnz"] = int(csr.nnz)
        return op, meta

    raise ValueError(f"Unsupported Hamiltonian NPZ format {fmt!r} in {p}")


def aggregate_hamiltonians_from_paths(
    paths: Sequence[Union[str, Path]],
    *,
    progress: Optional[Callable[[int, int, Path, dict[str, Any]], None]] = None,
) -> tuple[mm.MatrixOperator, dict[str, Any]]:
    """
    Sum Hamiltonians from NPZ files; returns combined operator and merged metadata.

    If ``progress`` is given, it is called once per file as
    ``progress(index, total, path, piece_meta)`` (``index`` is 1-based) after
    that file has been loaded and added to the accumulator. Useful for tracking
    long-running aggregations.
    """
    if not paths:
        raise ValueError("aggregate_hamiltonians_from_paths: no paths given.")

    total: Optional[mm.MatrixOperator] = None
    meta: dict[str, Any] = {"sources": []}
    basis_labels: Optional[list[str]] = None
    n_paths = len(paths)

    for idx, path in enumerate(paths, start=1):
        op, piece_meta = load_hamiltonian_npz(path)
        meta["sources"].append(str(path))
        if basis_labels is None and piece_meta.get("basis_labels"):
            basis_labels = list(piece_meta["basis_labels"])
        if total is None:
            total = op
            dim = piece_meta.get("dim", op.matrix.dim)
        else:
            if op.matrix.dim != total.matrix.dim:
                raise ValueError(
                    f"Hamiltonian dimension mismatch: {path} has dim={op.matrix.dim}, "
                    f"accumulator has dim={total.matrix.dim}"
                )
            total = total + op
        if progress is not None:
            progress(idx, n_paths, Path(path), piece_meta)
        # Drop the per-file operator (and its metadata) before the next
        # load_hamiltonian_npz allocates, so the previous partial can be freed
        # immediately instead of lingering through the next load. For the first
        # file `total` aliases this object, so it stays alive via `total`.
        del op, piece_meta

    if total is None:
        raise RuntimeError("aggregate_hamiltonians_from_paths: no Hamiltonian loaded.")
    if basis_labels is not None:
        meta["basis_labels"] = basis_labels
    meta["dim"] = total.matrix.dim
    return total, meta


def basis_labels_to_hilbert_bases(labels: Sequence[str]) -> hilber_space_bases:
    """Minimal basis wrapper for eigenvector CSV/NPZ export after load-only runs."""
    lbl = [str(x) for x in labels]
    bases = hilber_space_bases(names=["state"])
    bases._ket_states = lbl
    bases.dim = len(lbl)
    return bases
