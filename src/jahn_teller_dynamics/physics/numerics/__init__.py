"""
Numerical computation modules for Jahn-Teller systems.

This package provides functions for numerical calculations and observables:
- Transition intensities
- Magnetic interaction eigenstates
"""

from .observables import (
    compute_transition_intensities,
    compute_magnetic_interaction_eigen_kets,
)

__all__ = [
    'compute_transition_intensities',
    'compute_magnetic_interaction_eigen_kets',
]

