"""
Utility functions for the io package.

This package provides various utility functions for calculations,
operations, file handling, and path management.
"""

# Main public API - use lazy imports to avoid circular dependencies
__all__ = [
    'create_directory',
    'PathManager',
]

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    """Lazy import function to avoid circular dependencies."""
    if name == 'create_directory':
        from .file_utils import create_directory
        return create_directory
    elif name == 'PathManager':
        from .path_manager import PathManager
        return PathManager
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

