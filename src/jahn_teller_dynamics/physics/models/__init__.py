"""
Model building modules for Jahn-Teller physics.

This package provides functions to construct quantum system trees for different
Jahn-Teller models:
- Electron-phonon systems
- Spin-electron-phonon systems  
- Minimal model systems (no phonons)

It also provides functions to compute reduction factors and related quantities.
"""

from .system_builder import (
    build_electron_phonon_system,
    build_spin_electron_phonon_system,
    build_minimal_model_system,
)

from .reduction_factors import (
    compute_reduction_factors,
    compute_K_JT_factor,
    ReductionFactorsResult,
)

from .operator_manager import (
    store_and_get_root_operator,
    add_operator_to_hamiltonian,
)

__all__ = [
    'build_electron_phonon_system',
    'build_spin_electron_phonon_system',
    'build_minimal_model_system',
    'compute_reduction_factors',
    'compute_K_JT_factor',
    'ReductionFactorsResult',
    'store_and_get_root_operator',
    'add_operator_to_hamiltonian',
]

