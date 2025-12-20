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
    
    # Future: Sparse solver
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

