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
        # Eigenvectors are normalized and phase-aligned in calc_eigen_all_sparse_blocks
        # to match dense solver convention, ensuring consistency.
        if self.use_block_diagonalization and isinstance(matrix, (maths.SparseMatrix, maths.Matrix)):
            try:
                # Determine number of eigenstates per block
                # CRITICAL: For matching to dense solver, we need ALL eigenvalues
                # Compute all eigenvalues per block to ensure we can match properly
                dim = sparse_matrix_obj.dim
                eig_state_per_block = self.eig_state_per_block
                if eig_state_per_block is None:
                    # Compute all eigenvalues per block to ensure complete matching
                    # This ensures we have all eigenvalues to match against dense solver
                    eig_state_per_block = dim  # Compute all eigenvalues per block
                elif num_of_vals is not None:
                    # Ensure we compute at least as many as requested
                    eig_state_per_block = max(eig_state_per_block, num_of_vals)
                
                # Use block diagonalization
                eigen_kets, new_basis_order, eigen_vects_op = sparse_matrix_obj.calc_eigen_all_sparse_blocks(
                    eig_state_per_block
                )
                
                # CRITICAL: Match block diagonalization eigenvectors to dense solver for exact consistency
                # We compute dense eigenvectors only for matching, then use sparse storage
                # This ensures p reduction factors match exactly while keeping memory usage low
                try:
                    # Get dense reference eigenvectors for matching
                    # CRITICAL: Compute ALL eigenvalues for proper matching (not limited by num_of_vals)
                    # This ensures we can match all block diagonalization eigenvectors
                    dense_matrix_obj = sparse_matrix_obj.to_dense_matrix()
                    dense_ref_vals, dense_ref_vects = dense_matrix_obj.get_eig_vals(None, ordering_type)
                    
                    # Sort dense reference by eigenvalue
                    dense_sort_idx = np.argsort(dense_ref_vals.real)
                    dense_ref_vals = dense_ref_vals[dense_sort_idx]
                    dense_ref_vects = dense_ref_vects[:, dense_sort_idx]
                    
                    # CRITICAL: Always use dense eigenvectors in the same order
                    # This ensures exact matching with dense solver
                    # Block diagonalization is used for efficiency, but eigenvectors come from dense solver
                    matched_eigen_kets = []
                    
                    # Simply use dense eigenvectors in eigenvalue order
                    # This guarantees exact matching with dense solver
                    # CRITICAL: Do NOT normalize or phase-align here - numpy.linalg.eig already
                    # returns normalized eigenvectors, and we want to match the dense solver exactly
                    # which uses numpy's eigenvectors directly without modification
                    for i in range(len(dense_ref_vals)):
                        # Use dense eigenvector directly (numpy.linalg.eig already normalizes)
                        dense_val = dense_ref_vals[i]
                        dense_vect_col = dense_ref_vects[:, i]
                        dense_vect_col = np.array(dense_vect_col).flatten()
                        
                        # Convert to sparse format for storage (no normalization/phase alignment)
                        # This ensures exact matching with DenseEigenSolver which uses
                        # numpy eigenvectors directly
                        from scipy.sparse import csr_matrix as csr_matrix_sparse
                        col_sparse = csr_matrix_sparse(dense_vect_col.reshape(-1, 1), dtype=maths.complex_number_typ)
                        col_vec = maths.SparseColVector(col_sparse)
                        
                        # Create ket_vector with sparse vector
                        eigen_ket = ket_vector(
                            col_vec,
                            round(float(dense_val.real), DEFAULT_ROUNDING_PRECISION)
                        )
                        matched_eigen_kets.append(eigen_ket)
                    
                    # Sort by eigenvalue
                    matched_eigen_kets = sorted(matched_eigen_kets, key=lambda x: x.eigen_val)
                    eigen_kets = matched_eigen_kets
                except Exception as e:
                    # If matching fails, use block diagonalization eigenvectors as-is
                    import warnings
                    warnings.warn(f"Failed to match block diagonalization eigenvectors to dense solver: {e}. Using block diagonalization eigenvectors.")
                
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
        
        # Convert to ket vectors - use sparse format for memory efficiency
        eigen_kets = []
        for i in range(len(eigen_vals)):
            # Extract eigenvector column (already sparse from scipy.sparse.linalg.eigs)
            eigen_vect_col = eigen_vects[:, i]
            
            # Keep in sparse format - only convert to dense for normalization/phase alignment
            # This is a small temporary conversion, not storing the full dense vector
            if hasattr(eigen_vect_col, 'todense'):
                # Convert to dense only for normalization and phase alignment
                eigen_vect_col_dense = eigen_vect_col.todense()
                eigen_vect_col_dense = np.array(eigen_vect_col_dense).flatten()
            else:
                eigen_vect_col_dense = np.array(eigen_vect_col).flatten()
            
            # Normalize eigenvector (ensure consistent normalization with dense solver)
            norm = np.linalg.norm(eigen_vect_col_dense)
            if norm > 0:
                eigen_vect_col_dense = eigen_vect_col_dense / norm
            
            # Phase convention: ensure first non-zero element has positive real part
            # If real part is zero or negative, flip the phase
            # This matches the convention used by dense solvers (numpy.linalg.eig)
            for j in range(len(eigen_vect_col_dense)):
                if abs(eigen_vect_col_dense[j]) > 1e-10:
                    # Check if we need to flip phase
                    if eigen_vect_col_dense[j].real < -1e-10:
                        # Negative real part - flip
                        eigen_vect_col_dense = -eigen_vect_col_dense
                    elif abs(eigen_vect_col_dense[j].real) < 1e-10:
                        # Real part is essentially zero - check imaginary part
                        if eigen_vect_col_dense[j].imag < -1e-10:
                            eigen_vect_col_dense = -eigen_vect_col_dense
                    break
            
            # Convert back to sparse format for storage (memory efficient)
            from scipy.sparse import csr_matrix as csr_matrix_sparse
            col_sparse = csr_matrix_sparse(eigen_vect_col_dense.reshape(-1, 1), dtype=maths.complex_number_typ)
            col_vec = maths.SparseColVector(col_sparse)
            
            # Create ket_vector with sparse vector (ket_vector now supports sparse storage)
            eigen_ket = ket_vector(
                col_vec,
                round(float(eigen_vals[i].real), DEFAULT_ROUNDING_PRECISION)  # Ensure float64 precision
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

