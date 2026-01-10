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
from typing import Optional, List, Tuple
import numpy as np
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
    """
    
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
    
    def __init__(self, use_block_diagonalization: bool = False, eig_state_per_block: Optional[int] = None):
        """
        Initialize sparse eigenvalue solver.
        
        Args:
            use_block_diagonalization: If True, decompose matrix into blocks using graph analysis
                                     and solve each block separately (more efficient for block-diagonal matrices)
            eig_state_per_block: Number of eigenstates to compute per block (only used if
                               use_block_diagonalization=True)
        """
        self.use_block_diagonalization = use_block_diagonalization
        self.eig_state_per_block = eig_state_per_block
    
    def solve(
        self,
        matrix_operator: MatrixOperator,
        num_of_vals: Optional[int] = None,
        ordering_type: Optional[str] = None,
        quantum_states_bases: Optional[hilber_space_bases] = None
    ) -> eigen_vector_space:
        """
        Solve eigenvalue problem using sparse matrix methods.
        
        Args:
            matrix_operator: Matrix operator to solve
            num_of_vals: Number of eigenvalues to calculate (None = all for small matrices,
                        or a subset for large sparse matrices)
            ordering_type: Type of ordering ('SM' for smallest magnitude, 'SA' for smallest
                         algebraic, 'LM' for largest magnitude, 'LA' for largest algebraic)
            quantum_states_bases: Hilbert space basis for quantum states (optional)
            
        Returns:
            eigen_vector_space: Object containing eigen kets and basis
        """
        from scipy.sparse.linalg import eigs as sparse_eigs
        from scipy.sparse import csr_matrix
        import numpy as np
        
        # Check if matrix is already sparse or needs conversion
        matrix = matrix_operator.matrix
        
        # Convert to sparse if needed
        if isinstance(matrix, maths.SparseMatrix):
            sparse_matrix_obj = matrix
            sparse_matrix = matrix.matrix
        elif isinstance(matrix, maths.Matrix):
            # Convert dense matrix to sparse
            sparse_matrix_obj = maths.SparseMatrix(matrix.matrix)
            sparse_matrix = csr_matrix(matrix.matrix)
        else:
            # Try to convert to sparse
            try:
                sparse_matrix_obj = maths.SparseMatrix(matrix.matrix)
                sparse_matrix = csr_matrix(matrix.matrix)
            except AttributeError:
                raise TypeError(f"Cannot convert {type(matrix)} to sparse matrix")
        
        # Use block diagonalization if requested and matrix is sparse
        if self.use_block_diagonalization and isinstance(matrix, (maths.SparseMatrix, maths.Matrix)):
            try:
                # Determine number of eigenstates per block
                eig_state_per_block = self.eig_state_per_block
                if eig_state_per_block is None:
                    # Default: use num_of_vals if specified, otherwise compute all for small blocks
                    if num_of_vals is not None:
                        eig_state_per_block = num_of_vals
                    else:
                        eig_state_per_block = 10  # Default for large matrices
                
                # Use block diagonalization
                eigen_kets, new_basis_order, eigen_vects_op = sparse_matrix_obj.calc_eigen_all_sparse_blocks(
                    eig_state_per_block
                )
                
                # Limit to requested number if specified
                if num_of_vals is not None and len(eigen_kets) > num_of_vals:
                    eigen_kets = eigen_kets[:num_of_vals]
                
                # Use provided basis or existing one
                basis = quantum_states_bases
                if basis is None:
                    basis = matrix_operator.quantum_state_bases
                
                return eigen_vector_space(basis, eigen_kets)
            except ImportError:
                # networkx not available, fall through to regular sparse solver
                import warnings
                warnings.warn("Block diagonalization requested but networkx not available. Using regular sparse solver.")
            except Exception as e:
                # Block diagonalization failed, fall through to regular sparse solver
                import warnings
                warnings.warn(f"Block diagonalization failed: {e}. Using regular sparse solver.")
        
        # Determine number of eigenvalues to compute
        dim = sparse_matrix.shape[0]
        if num_of_vals is None:
            # For sparse matrices, typically compute a subset
            # Default to computing all for small matrices, subset for large ones
            num_of_vals = min(dim, 10) if dim > 100 else dim
        
        # Limit to matrix dimension
        num_of_vals = min(num_of_vals, dim)
        
        # Set default ordering
        if ordering_type is None:
            ordering_type = 'SM'  # Smallest magnitude
        
        # For small matrices or when k >= N-1, use dense solver
        # Sparse iterative solvers require k < N-1
        if num_of_vals >= dim - 1:
            # Convert to dense and use dense solver
            import warnings
            warnings.warn(f"Requested {num_of_vals} eigenvalues for {dim}x{dim} matrix. "
                         f"Using dense solver (sparse solver requires k < N-1).")
            dense_solver = DenseEigenSolver()
            return dense_solver.solve(
                matrix_operator,
                num_of_vals=num_of_vals,
                ordering_type=ordering_type,
                quantum_states_bases=quantum_states_bases
            )
        
        # Solve sparse eigenvalue problem
        try:
            eigen_vals, eigen_vects = sparse_eigs(
                sparse_matrix, 
                k=num_of_vals, 
                which=ordering_type
            )
        except Exception as e:
            # Fallback to dense solver if sparse solver fails
            import warnings
            warnings.warn(f"Sparse eigenvalue solver failed: {e}. Falling back to dense solver.")
            dense_solver = DenseEigenSolver()
            return dense_solver.solve(
                matrix_operator,
                num_of_vals=num_of_vals,
                ordering_type=ordering_type,
                quantum_states_bases=quantum_states_bases
            )
        
        # Convert to ket vectors
        eigen_kets = []
        for i in range(len(eigen_vals)):
            # Extract eigenvector column
            eigen_vect_col = eigen_vects[:, i]
            
            # Convert to dense if needed
            if hasattr(eigen_vect_col, 'todense'):
                eigen_vect_col = eigen_vect_col.todense()
            
            # Ensure it's a column vector
            if eigen_vect_col.shape[0] == 1:
                eigen_vect_col = eigen_vect_col.T
            
            # Normalize eigenvector (ensure consistent normalization with dense solver)
            eigen_vect_col = np.array(eigen_vect_col).flatten()
            norm = np.linalg.norm(eigen_vect_col)
            if norm > 0:
                eigen_vect_col = eigen_vect_col / norm
            
            # Phase convention: ensure first non-zero element has positive real part
            # If real part is zero or negative, flip the phase
            # This matches the convention used by dense solvers (numpy.linalg.eig)
            for j in range(len(eigen_vect_col)):
                if abs(eigen_vect_col[j]) > 1e-10:
                    # Check if we need to flip phase
                    if eigen_vect_col[j].real < -1e-10:
                        # Negative real part - flip
                        eigen_vect_col = -eigen_vect_col
                    elif abs(eigen_vect_col[j].real) < 1e-10:
                        # Real part is essentially zero - check imaginary part
                        if eigen_vect_col[j].imag < -1e-10:
                            eigen_vect_col = -eigen_vect_col
                    break
            
            # Convert to column vector format matching dense solver convention
            # Dense solver uses: np.matrix([eigen_vects[:, i]]) which is (1, n)
            # Then transposes to get (n, 1) for col_vector
            eigen_vect_col = np.matrix([eigen_vect_col])  # Shape: (1, n)
            
            eigen_ket = ket_vector(
                maths.col_vector(
                    np.transpose(
                        np.round(
                            eigen_vect_col, 
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

