"""
JT Theory building and calculation components.

This package provides functionality for building Jahn-Teller theory objects
from various sources and performing calculations on them.
"""

# Main public API - use lazy imports to avoid circular dependencies
__all__ = [
    'JTTheoryBuilder',
    'JT_Calculator',
    'calc_transition_energies',
]

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    """Lazy import function to avoid circular dependencies."""
    if name == 'JTTheoryBuilder':
        from .builder import JTTheoryBuilder
        return JTTheoryBuilder
    elif name == 'JT_Calculator':
        from .calculator import JT_Calculator
        return JT_Calculator
    elif name == 'calc_transition_energies':
        from .calculator import calc_transition_energies
        return calc_transition_energies
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

