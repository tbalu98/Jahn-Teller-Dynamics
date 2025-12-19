"""
Workflow orchestration for Jahn-Teller calculations.

This package provides high-level workflow coordination for running
Jahn-Teller calculations, including ZPL and single-case workflows.
"""

# Main public API - use lazy imports to avoid circular dependencies
__all__ = [
    'JTOrchestrator',
]

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    """Lazy import function to avoid circular dependencies."""
    if name == 'JTOrchestrator':
        from .orchestrator import JTOrchestrator
        return JTOrchestrator
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

