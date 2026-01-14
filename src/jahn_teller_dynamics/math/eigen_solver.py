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
import numpy as np
from scipy.sparse import block_diag
from scipy.linalg import eig as eigs
import jahn_teller_dynamics.math.maths as maths
from jahn_teller_dynamics.math.matrix_mechanics import (
    MatrixOperator, 
    ket_vector, 
    eigen_vector_space,
    hilber_space_bases,
    DEFAULT_ROUNDING_PRECISION
)


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
        quantum_states_bases: Optional[hilber_space_bases] = None
    ) -> eigen_vector_space:
        """
        Solve eigenvalue problem for a matrix operator.
        
        Args:
            matrix_operator: Matrix operator to solve
            num_of_vals: Number of eigenvalues to calculate (None = all)
            ordering_type: Type of ordering to apply (optional)
            quantum_states_bases: Hilbert space basis for quantum states (optional)
            
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
        quantum_states_bases: Optional[hilber_space_bases] = None
    ) -> eigen_vector_space:
        """
        Solve eigenvalue problem using dense matrix methods.
        
        Args:
            matrix_operator: Matrix operator to solve
            num_of_vals: Number of eigenvalues to calculate (None = all)
            ordering_type: Type of ordering to apply (optional)
            quantum_states_bases: Hilbert space basis for quantum states (optional)
            
        Returns:
            eigen_vector_space: Object containing eigen kets and basis
        """
        # Get eigenvalues and eigenvectors from underlying matrix
        eigen_vals, eigen_vects = matrix_operator.matrix.get_eig_vals(num_of_vals, ordering_type)
        
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
        quantum_states_bases: Optional[hilber_space_bases] = None
    ) -> eigen_vector_space:
        """
        Solve eigenvalue problem using sparse matrix methods (without block search).
        
        This implementation:
        - Always converts the input matrix to a CSR sparse matrix.
        - Uses scipy.sparse.linalg.eigsh for Hermitian matrices when a subset of
          eigenvalues is requested (num_of_vals < dim).
        - Falls back to dense diagonalization (DenseEigenSolver.solve_block)
          when all eigenvalues are needed or when eigsh cannot be used.
        
        Args:
            matrix_operator: Matrix operator to solve
            num_of_vals: Number of eigenvalues to calculate. If None, defaults to self.default_num_of_vals.
            ordering_type: Type of ordering (currently ignored; eigsh uses 'SA')
            quantum_states_bases: Hilbert space basis for quantum states (optional)
            
        Returns:
            eigen_vector_space: Object containing eigen kets and basis
        """
        from scipy.sparse.linalg import eigsh as sparse_eigsh
        from scipy.sparse import csr_matrix
        import numpy as np
        
        # Underlying matrix (dense or sparse wrapper)
        matrix = matrix_operator.matrix
        
        # Convert to our SparseMatrix wrapper and CSR for eigsh
        if isinstance(matrix, maths.SparseMatrix):
            sparse_matrix_obj = matrix
            sparse_matrix = matrix.matrix.tocsr()
        elif isinstance(matrix, maths.Matrix):
            sparse_matrix = csr_matrix(matrix.matrix)
            sparse_matrix_obj = maths.SparseMatrix(sparse_matrix)
        else:
            # Try to interpret as having a .matrix attribute
            try:
                sparse_matrix = csr_matrix(matrix.matrix)
                sparse_matrix_obj = maths.SparseMatrix(sparse_matrix)
            except AttributeError:
                raise TypeError(f"Cannot convert {type(matrix)} to SparseMatrix/CSR for eigsh")
        
        dim = sparse_matrix.shape[0]

        # Default behavior for sparse solver: compute a subset unless explicitly asked otherwise.
        if num_of_vals is None:
            num_of_vals = self.default_num_of_vals
        
        # Decide how many eigenvalues to compute
        k = num_of_vals
        if k >= dim:
            # eigsh requires k < N; for all states, use dense
            dense_solver = DenseEigenSolver()
            eigen_vals, eigen_vects = dense_solver.solve_block(sparse_matrix_obj)
        else:
            # Use Hermitian sparse solver: smallest algebraic eigenvalues
            eigen_vals, eigen_vects = sparse_eigsh(
                sparse_matrix,
                k=k,
                which="SA"
            )
            
            # Cast to complex_number_typ for consistency
            eigen_vals = np.array(eigen_vals, dtype=maths.complex_number_typ)
            eigen_vects = np.array(eigen_vects, dtype=maths.complex_number_typ)
            
            # Sort by eigenvalue (real part; should be real for Hermitian)
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

# Default solver instance
_default_solver = DenseEigenSolver()
def solve_eigenvalue_problem(
    matrix_operator: MatrixOperator,
    num_of_vals: Optional[int] = None,
    ordering_type: Optional[str] = None,
    quantum_states_bases: Optional[hilber_space_bases] = None,
    solver: Optional[EigenSolver] = None
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
        quantum_states_bases=quantum_states_bases
    )

