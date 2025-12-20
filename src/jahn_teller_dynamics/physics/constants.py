"""
Physical constants for Jahn-Teller calculations.

This module provides physical constants used throughout the Jahn-Teller dynamics
calculations, including fundamental constants, conversion factors, and numerical
precision settings.
"""

# Fundamental physical constants
# ==============================

# Reduced Planck constant in meV·s
# hbar = h / (2π) ≈ 6.5821195 × 10^-13 meV·s
hbar_meVs: float = 6.5821195e-13

# Bohr magneton in meV/T
# μ_B = e·hbar / (2·m_e) ≈ 0.057883671 meV/T
Bohn_magneton_meV_T: float = 0.057883671

# Electron g-factor (g-factor for free electron)
# g_e ≈ 2.0023
g_factor: float = 2.0023

# Numerical precision settings
# ============================

# Default rounding precision (number of decimal places)
round_precision_dig: int = 7

__all__ = [
    'hbar_meVs',
    'Bohn_magneton_meV_T',
    'g_factor',
    'round_precision_dig',
]

