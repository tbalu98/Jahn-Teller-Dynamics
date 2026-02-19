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
    build_phonon_system,
    MultiModePhononSystem,
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

from .multi_config_electron import (
    multi_config_electron,
)

from .lvc_model import (
    LVC_model,
)

from .position_expr_parser import (
    tokenize,
    PositionExprParser,
    evaluate_position_expression,
)

__all__ = [
    'build_electron_phonon_system',
    'build_spin_electron_phonon_system',
    'build_minimal_model_system',
    'build_phonon_system',
    'MultiModePhononSystem',
    'compute_reduction_factors',
    'compute_K_JT_factor',
    'ReductionFactorsResult',
    'store_and_get_root_operator',
    'add_operator_to_hamiltonian',
    'multi_config_electron',
    'LVC_model',
    'tokenize',
    'PositionExprParser',
    'evaluate_position_expression',
]

