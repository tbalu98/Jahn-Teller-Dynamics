"""
Visualization and plotting for Jahn-Teller dynamics.

This package provides plotting and visualization functionality for
Jahn-Teller calculations, including energy states, APES, ZPL transitions,
and contour plots.
"""

# Main public API - use lazy imports to avoid circular dependencies
__all__ = [
    'Plotter',
]

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    """Lazy import function to avoid circular dependencies."""
    if name == 'Plotter':
        from .plotter import Plotter
        return Plotter
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

