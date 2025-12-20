"""
Math module for quantum mechanical calculations.

This module provides core mathematical utilities including:
- Matrix mechanics (operators, vectors, bases)
- Eigenvalue/eigenvector solvers
- Mathematical operations
- Bra-ket formalism
"""

# Export main classes for convenience
from jahn_teller_dynamics.math.matrix_mechanics import (
    MatrixOperator,
    ket_vector,
    bra_vector,
    hilber_space_bases,
    eigen_vector_space,
)

from jahn_teller_dynamics.math.eigen_solver import (
    EigenSolver,
    DenseEigenSolver,
    solve_eigenvalue_problem,
)

__all__ = [
    'MatrixOperator',
    'ket_vector',
    'bra_vector',
    'hilber_space_bases',
    'eigen_vector_space',
    'EigenSolver',
    'DenseEigenSolver',
    'solve_eigenvalue_problem',
]
