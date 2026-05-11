"""
Eigenvalue and eigenvector solver module.

This module provides a flexible interface for solving eigenvalue problems,
allowing for different implementations (dense, sparse, iterative, etc.)
without coupling to the MatrixOperator class.

Example usage:
    # Default dense solver
    op = MatrixOperator(matrix)
    result = op.calc_eigen_vals_vects()
    
    # Custom solver (e.g., for sparse matrices)
    from jahn_teller_dynamics.math.eigen_solver import DenseEigenSolver
    solver = DenseEigenSolver()
    op.set_eigen_solver(solver)
    result = op.calc_eigen_vals_vects()
    
    # Sparse solver
    # from jahn_teller_dynamics.math.eigen_solver import SparseEigenSolver
    # sparse_solver = SparseEigenSolver()
    # op.set_eigen_solver(sparse_solver)
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Any, Union
import warnings

import numpy as np
from scipy.sparse import block_diag
from scipy.linalg import eig as eigs
import jahn_teller_dynamics.math.maths as maths
from jahn_teller_dynamics.math.matrix_mechanics import (
    MatrixOperator,
    ket_vector,
    eigen_vector_space,
    hilber_space_bases,
    DEFAULT_ROUNDING_PRECISION,
)


def spectral_which_to_scipy_eigsh(
    spectral_which: Optional[str], spectral_sigma: Optional[float]
) -> str:
    """
    Map [.cfg-style] spectral selection to scipy ``eigsh(..., which=...)``.

    When ``spectral_sigma`` is set (shift–invert / interior search), scipy expects ``which='LM'``
    unless the caller explicitly chooses another eigenvalue grouping compatible with sigma.
    """
    s = (spectral_which or "").strip().lower().replace("-", "_")
    if spectral_sigma is not None:
        if not s or s in ("nearest", "closest", "lm",):
            return "LM"
        if s in ("sm", "smallest_mag", "smallest_magnitude"):
            return "LM"
        if s == "lm":
            return "LM"
        if s in ("sa", "smallest_algebraic", "smallest_real"):
            raise ValueError(
                "SciPy eigsh shift-invert: use eigensolver_spectral_which nearest (or LM), "
                "not smallest_algebraic/smallest_real, when eigensolver_sigma is set."
            )

    scipy_which_aliases = frozenset(
        {"sa", "la", "lm", "sm", "be", "sam", "largest_algebraic"}
    )
    raw_u = (spectral_which or "").strip().upper()
    if raw_u in ("SA", "LA", "LM", "SM", "BE"):
        return raw_u
    friendly = {
        "smallest_real": "SA",
        "smallest_algebraic": "SA",
        "smallest_mag": "SM",
        "smallest_magnitude": "SM",
        "largest_real": "LA",
        "largest_algebraic": "LA",
        "nearest": "LM",
        "closest": "LM",
        "either_end": "BE",
        "bucket": "BE",
        "ordering_sm": "SM",
        "ordering_sa": "SA",
        "ordering_la": "LA",
    }
    if s in friendly:
        return friendly[s]
    if s in scipy_which_aliases:
        return s.upper()
    return "SA"


def dense_pick_indices(
    eigenvalues: np.ndarray, nev: int, spectral_sigma: Optional[float], spectral_which: Optional[str]
) -> np.ndarray:
    """Indices into ascending ``eigenvalues`` (Hermitian convention from ``numpy.linalg.eigh``)."""
    dim = eigenvalues.shape[0]
    n = max(1, min(int(nev), dim))
    w = eigenvalues.astype(float, copy=False).real if np.iscomplexobj(eigenvalues) else eigenvalues.astype(float)

    if spectral_sigma is not None:
        tgt = float(spectral_sigma)
        return np.argsort(np.abs(w - tgt))[:n]

    s = (spectral_which or "").strip().lower().replace("-", "_")
    order_desc = (-w).argsort()
    asc = slice(None)

    if not s or s in ("sa", "smallest_real", "smallest_algebraic"):
        idx = np.arange(n)
        return idx
    if s in ("la", "largest_real", "largest_algebraic"):
        return np.arange(dim - n, dim)
    if s in ("sm", "smallest_mag", "smallest_magnitude"):
        return np.argsort(np.abs(w))[:n]
    if s in ("lm", "nearest", "closest"):
        raise ValueError("dense_pick_indices for nearest eigenvalues requires eigensolver_sigma")
    raise ValueError(f"Unknown spectral_which for dense picker: {spectral_which!r}")


class EigenSolver(ABC):
    """
    Abstract base class for eigenvalue/eigenvector solvers.
    
    This allows for different implementations:
    - Dense matrix solvers (numpy.linalg.eig)
    - Sparse matrix solvers (scipy.sparse.linalg)
    - Iterative solvers for large systems
    - GPU-accelerated solvers
    """
    
    @abstractmethod
    def solve(
        self,
        matrix_operator: MatrixOperator,
        num_of_vals: Optional[int] = None,
        ordering_type: Optional[str] = None,
        quantum_states_bases: Optional[hilber_space_bases] = None,
        spectral_sigma: Optional[float] = None,
        spectral_which: Optional[str] = None,
    ) -> eigen_vector_space:
        """
        Solve eigenvalue problem for a matrix operator.
        
        Args:
            matrix_operator: Matrix operator to solve
            num_of_vals: Number of eigenvalues to calculate (None = all)
            ordering_type: Type of ordering (legacy dense hook; overlaps ``spectral_which``).
            quantum_states_bases: Hilbert space basis for quantum states (optional)
            spectral_sigma: Shift / target eigenvalue for interior or shift–invert solves.
            spectral_which: Selection mode (friendly names like ``smallest_real`` or scipy letters).
            
        Returns:
            eigen_vector_space: Object containing eigen kets and basis
        """
        pass


class DenseEigenSolver(EigenSolver):
    """
    Dense matrix eigenvalue solver using numpy.
    
    This is the default solver for small to medium-sized matrices.
    Uses numpy's dense matrix eigenvalue routines.
    
    Can also accept sparse matrices (SparseMatrix) and convert them to dense before solving.
    """
    
    def solve_block(
        self,
        matrix: Union[maths.Matrix, maths.SparseMatrix],
        num_of_vals: Optional[int] = None,
        ordering_type: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve eigenvalue problem for a single block (matrix).
        
        Accepts both dense and sparse matrices. Converts sparse to dense before solving.
        
        Args:
            matrix: Matrix or SparseMatrix to diagonalize
            num_of_vals: Number of eigenvalues to compute (default: all, not used but kept for compatibility)
            ordering_type: Ordering type (not used, kept for compatibility)
            
        Returns:
            Tuple of (eigenvalues, eigenvectors) as numpy arrays
        """
        # Convert to dense if sparse
        if isinstance(matrix, maths.SparseMatrix):
            dense_matrix = np.array(matrix.matrix.todense(), dtype=maths.complex_number_typ)
        elif isinstance(matrix, maths.Matrix):
            dense_matrix = np.array(matrix.matrix, dtype=maths.complex_number_typ)
        else:
            raise TypeError(f"Expected Matrix or SparseMatrix, got {type(matrix)}")
        
        # Use dense solver
        eigen_vals, eigen_vects = eigs(dense_matrix)
        
        # Ensure complex128 for consistency
        eigen_vals = eigen_vals.astype(maths.complex_number_typ)
        eigen_vects = eigen_vects.astype(maths.complex_number_typ)
        
        # Sort by real part (for Hermitian matrices, eigenvalues are real)
        idx = np.argsort(eigen_vals.real)
        eigen_vals = eigen_vals[idx]
        eigen_vects = eigen_vects[:, idx]
        
        return eigen_vals, eigen_vects
    
    def solve(
        self,
        matrix_operator: MatrixOperator,
        num_of_vals: Optional[int] = None,
        ordering_type: Optional[str] = None,
        quantum_states_bases: Optional[hilber_space_bases] = None,
        spectral_sigma: Optional[float] = None,
        spectral_which: Optional[str] = None,
    ) -> eigen_vector_space:
        """
        Hermitian diagonalization via ``numpy.linalg.eigh`` with optional spectral selection.

        Uses ``spectral_sigma`` / ``spectral_which`` from [.cfg]; ``ordering_type`` is a legacy
        alias when ``spectral_which`` is unset (e.g. ``SM``, ``LA`` SciPy-like tokens).
        
        Args:
            matrix_operator: Matrix operator to solve
            num_of_vals: Number of eigenvalues to calculate (None = all)
            ordering_type: Type of ordering to apply (optional)
            quantum_states_bases: Hilbert space basis for quantum states (optional)
            spectral_sigma: Center eigenvalue — pick ``num_of_vals`` states closest (real axis).
            spectral_which: ``smallest_real``, ``largest_real``, ``smallest_magnitude``, etc.
            
        Returns:
            eigen_vector_space: Object containing eigen kets and basis
        """
        matrix = matrix_operator.matrix
        if isinstance(matrix, maths.SparseMatrix):
            H = np.array(matrix.matrix.todense(), dtype=maths.complex_number_typ)
        elif isinstance(matrix, maths.Matrix):
            H = np.array(matrix.matrix, dtype=maths.complex_number_typ)
        else:
            raise TypeError(f"DenseEigenSolver expects Matrix or SparseMatrix backend, got {type(matrix)}")
        Herm = 0.5 * (H + np.conjugate(H.T))
        dim = Herm.shape[0]
        nev = dim if num_of_vals is None else min(max(1, int(num_of_vals)), dim)
        sw_eff = (
            spectral_which
            if (spectral_which is not None and str(spectral_which).strip())
            else ordering_type
        )
        # Map legacy SciPy-ish ordering tokens
        if sw_eff:
            otl = str(sw_eff).strip().upper().replace("-", "_")
            if otl in {"SM"}:
                sw_eff = "smallest_mag"
            elif otl in {"SA"}:
                sw_eff = "smallest_real"
            elif otl in {"LA"}:
                sw_eff = "largest_real"
        eig_w, eig_V = np.linalg.eigh(Herm)
        picks = dense_pick_indices(eig_w, nev, spectral_sigma, sw_eff)
        eigen_vals = eig_w[picks]
        eigen_vects = eig_V[:, picks]
        
        # Convert to ket vectors
        eigen_kets = []
        for i in range(len(eigen_vals)):
            eigen_ket = ket_vector(
                maths.col_vector(
                    np.transpose(
                        np.round(
                            np.matrix([eigen_vects[:, i]]), 
                            DEFAULT_ROUNDING_PRECISION
                        )
                    )
                ),
                round(eigen_vals[i].real, DEFAULT_ROUNDING_PRECISION)
            )
            eigen_kets.append(eigen_ket)
        
        # Sort by eigenvalue
        eigen_kets = sorted(eigen_kets, key=lambda x: x.eigen_val)
        
        # Use provided basis or existing one
        basis = quantum_states_bases
        if basis is None:
            basis = matrix_operator.quantum_state_bases
        
        return eigen_vector_space(basis, eigen_kets)


class SparseEigenSolver(EigenSolver):
    """
    Sparse matrix eigenvalue solver using scipy.sparse.linalg.
    
    This solver is optimized for large, sparse matrices. It uses iterative
    methods to compute a subset of eigenvalues/eigenvectors efficiently.
    
    Can optionally use block diagonalization for matrices with block structure,
    which can significantly improve performance.
    """
    
    def __init__(
        self,
        use_block_diagonalization: bool = False,
        eig_state_per_block: Optional[int] = None,
        default_num_of_vals: int = 100,
    ):
        """
        Initialize sparse eigenvalue solver.
        
        Args:
            use_block_diagonalization: If True, decompose matrix into blocks using graph analysis
                                     and solve each block separately (more efficient for block-diagonal matrices)
            eig_state_per_block: Number of eigenstates to compute per block (only used if
                               use_block_diagonalization=True)
            default_num_of_vals: Default number of eigenpairs to compute when num_of_vals is not provided.
        """
        self.use_block_diagonalization = use_block_diagonalization
        self.eig_state_per_block = eig_state_per_block
        self.default_num_of_vals = default_num_of_vals
    
    def solve(
        self,
        matrix_operator: MatrixOperator,
        num_of_vals: Optional[int] = None,
        ordering_type: Optional[str] = None,
        quantum_states_bases: Optional[hilber_space_bases] = None,
        spectral_sigma: Optional[float] = None,
        spectral_which: Optional[str] = None,
    ) -> eigen_vector_space:
        """
        Sparse Hermitian eigenproblem via ``scipy.sparse.linalg.eigsh`` (optionally shift–invert).
        
        Args:
            matrix_operator: Operator to diagonalize (CSR-backed).
            num_of_vals: Number of eigenpairs (defaults to ``default_num_of_vals``).
            ordering_type: Legacy alias for ``spectral_which`` when the latter is empty.
            quantum_states_bases: Output basis.
            spectral_sigma: Shift for ``eigsh(..., sigma=...)`` (interior / target region).
            spectral_which: ``smallest_real``, ``nearest``, or SciPy tokens ``SA``/``LM`` etc.
        """
        from scipy.sparse.linalg import eigsh as sparse_eigsh
        from scipy.sparse import csr_matrix
        import numpy as np
        
        matrix = matrix_operator.matrix
        
        if isinstance(matrix, maths.SparseMatrix):
            sparse_matrix_obj = matrix
            sparse_matrix = matrix.matrix.tocsr()
        elif isinstance(matrix, maths.Matrix):
            sparse_matrix = csr_matrix(matrix.matrix)
            sparse_matrix_obj = maths.SparseMatrix(sparse_matrix)
        else:
            try:
                sparse_matrix = csr_matrix(matrix.matrix)
                sparse_matrix_obj = maths.SparseMatrix(sparse_matrix)
            except AttributeError:
                raise TypeError(f"Cannot convert {type(matrix)} to SparseMatrix/CSR for eigsh")

        dim = sparse_matrix.shape[0]

        if num_of_vals is None:
            num_of_vals = self.default_num_of_vals

        sw_eff = (
            spectral_which
            if (spectral_which is not None and str(spectral_which).strip())
            else ordering_type
        )
        k = min(int(num_of_vals), dim)
        if dim <= 1:
            dense_solver = DenseEigenSolver()
            return dense_solver.solve(
                matrix_operator,
                num_of_vals=num_of_vals,
                ordering_type=ordering_type,
                quantum_states_bases=quantum_states_bases,
                spectral_sigma=spectral_sigma,
                spectral_which=sw_eff,
            )
        # scipy eigsh requires k < dim; avoid falling back to a full dense H.
        if k >= dim:
            warnings.warn(
                f"SparseEigenSolver: requested num_of_vals capped from {k} to {dim - 1} "
                f"(matrix dim={dim}) to avoid allocating a dense Hamiltonian.",
                RuntimeWarning,
                stacklevel=2,
            )
            k = dim - 1

        which_scipy = spectral_which_to_scipy_eigsh(sw_eff, spectral_sigma)

        eigsh_kw: dict = {"k": k, "which": which_scipy}
        if spectral_sigma is not None:
            eigsh_kw["sigma"] = spectral_sigma

        eigen_vals, eigen_vects = sparse_eigsh(sparse_matrix, **eigsh_kw)

        eigen_vals = np.array(eigen_vals, dtype=maths.complex_number_typ)
        eigen_vects = np.array(eigen_vects, dtype=maths.complex_number_typ)

        idx = np.argsort(eigen_vals.real)
        eigen_vals = eigen_vals[idx]
        eigen_vects = eigen_vects[:, idx]
        
        # Wrap eigenpairs into ket vectors
        eigen_kets = []
        for i in range(len(eigen_vals)):
            vec = eigen_vects[:, i]
            col_vec = maths.col_vector(
                np.transpose(
                    np.round(
                        np.matrix([vec]),
                        DEFAULT_ROUNDING_PRECISION
                    )
                )
            )
            eigen_ket = ket_vector(
                col_vec,
                round(eigen_vals[i].real, DEFAULT_ROUNDING_PRECISION)
            )
            eigen_kets.append(eigen_ket)
        
        # Final sort by eigenvalue (defensive)
        eigen_kets = sorted(eigen_kets, key=lambda x: x.eigen_val)
        
        # Use provided basis or the operator's basis
        basis = quantum_states_bases
        if basis is None:
            basis = matrix_operator.quantum_state_bases
        
        return eigen_vector_space(basis, eigen_kets)


def resolve_pvc_eigensolver_backend(
    eigensolver: str = "",
    running_environment: str = "",
    *,
    use_sparse: bool,
) -> str:
    """
    Map config/CLI knobs to normalized PVC backends: ``"sparse"``, ``"dense"``, ``"slepc"``.

    * ``eigensolver`` beats ``running_environment``.
    * If neither selects SLEPc, ``use_sparse`` chooses scipy sparse vs dense.
    """
    name = (eigensolver or "").strip().lower().replace("-", "_")
    slepc_names = frozenset({"slepc", "petsc", "petsc_eps", "slepc_eps", "mpi_slepc"})
    dense_names = frozenset({"dense", "scipy_dense", "numpy", "full"})
    sparse_names = frozenset({"sparse", "scipy_sparse", "scipy", "eigsh", "arpack"})
    if name in slepc_names:
        return "slepc"
    if name in dense_names:
        return "dense"
    if name in sparse_names:
        return "sparse"

    renv = (running_environment or "").strip().lower().replace("-", "_")
    if renv in frozenset(
        {"multiprocessor", "multi_processor", "mpi", "parallel", "hpc", "cluster"}
    ):
        return "slepc"

    return "sparse" if use_sparse else "dense"


def create_pvc_eigen_solver(backend: str) -> EigenSolver:
    """Instantiate the solver used by :mod:`jahn_teller_dynamics.PVC`."""
    b = (backend or "").strip().lower()
    if b == "slepc":
        from jahn_teller_dynamics.math.slepc_eigen_solver import SLEPcEigenSolver

        return SLEPcEigenSolver()
    if b == "sparse":
        return SparseEigenSolver(use_block_diagonalization=True)
    if b == "dense":
        return DenseEigenSolver()
    raise ValueError(f"Unknown PVC eigensolver backend {backend!r}")


# Default solver instance
_default_solver = DenseEigenSolver()
def solve_eigenvalue_problem(
    matrix_operator: MatrixOperator,
    num_of_vals: Optional[int] = None,
    ordering_type: Optional[str] = None,
    quantum_states_bases: Optional[hilber_space_bases] = None,
    solver: Optional[EigenSolver] = None,
    spectral_sigma: Optional[float] = None,
    spectral_which: Optional[str] = None,
) -> eigen_vector_space:
    """
    Convenience function to solve eigenvalue problems.
    
    Args:
        matrix_operator: Matrix operator to solve
        num_of_vals: Number of eigenvalues to calculate (None = all)
        ordering_type: Type of ordering to apply (optional)
        quantum_states_bases: Hilbert space basis for quantum states (optional)
        solver: Optional custom solver (defaults to DenseEigenSolver)
        
    Returns:
        eigen_vector_space: Object containing eigen kets and basis
    """
    if solver is None:
        solver = _default_solver
    
    return solver.solve(
        matrix_operator,
        num_of_vals=num_of_vals,
        ordering_type=ordering_type,
        quantum_states_bases=quantum_states_bases,
        spectral_sigma=spectral_sigma,
        spectral_which=spectral_which,
    )

