"""
Reader for LVC eigenvectors.npz files.

Loads the compressed NPZ format saved by LVC.py (with --save-npz) and reconstructs
ket_vector objects and eigen_vector_space for use in the rest of the codebase.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from jahn_teller_dynamics.math_utils.matrix_mechanics import eigen_vector_space, ket_vector


def load_lvc_npz(path: str | Path) -> dict:
    """
    Load raw arrays from an LVC eigenvectors.npz file.

    Args:
        path: Path to the .npz file.

    Returns:
        dict with keys: 'eigenvectors', 'eigenvalues', 'basis_labels', 'order', 'dim'
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NPZ file not found: {path}")
    data = np.load(path, allow_pickle=True)
    return {
        "eigenvectors": data["eigenvectors"],
        "eigenvalues": data["eigenvalues"],
        "basis_labels": data["basis_labels"],
        "order": int(data["order"]),
        "dim": int(data["dim"]),
    }


def load_ket_vectors_from_npz(path: str | Path) -> list["ket_vector"]:
    """
    Load ket_vector objects from an LVC eigenvectors.npz file.

    Args:
        path: Path to the .npz file.

    Returns:
        List of ket_vector objects (one per eigenstate) with coeffs and eigen_val set.
    """
    from jahn_teller_dynamics.math_utils.matrix_mechanics import ket_vector

    raw = load_lvc_npz(path)
    eig_vecs = raw["eigenvectors"]  # (dim, num_eigs)
    eig_vals = raw["eigenvalues"]

    kets = []
    for j in range(eig_vecs.shape[1]):
        coeffs = [complex(c) for c in eig_vecs[:, j]]
        ev = complex(eig_vals[j])
        kets.append(ket_vector(coeffs, eigen_val=ev))
    return kets


def load_eigen_vector_space_from_npz(path: str | Path) -> "eigen_vector_space":
    """
    Load eigen_vector_space from an LVC eigenvectors.npz file.

    Reconstructs ket_vectors and a minimal hilber_space_bases (index-based)
    so the result is compatible with eigen_vector_space consumers.

    Args:
        path: Path to the .npz file.

    Returns:
        eigen_vector_space with eigen_kets and quantum_states_basis.
    """
    from jahn_teller_dynamics.math_utils.matrix_mechanics import (
        hilber_space_bases,
        eigen_vector_space,
    )

    raw = load_lvc_npz(path)
    kets = load_ket_vectors_from_npz(path)
    dim = raw["dim"]

    # Create index-based basis (labels 0, 1, ..., dim-1) so eigenvector
    # coefficients match the same ordering as when saved.
    qm_nums_list = [[i] for i in range(dim)]
    basis = hilber_space_bases().from_qm_nums_list(
        qm_nums_list, qm_nums_names=["basis_idx"]
    )
    return eigen_vector_space(basis, kets)
