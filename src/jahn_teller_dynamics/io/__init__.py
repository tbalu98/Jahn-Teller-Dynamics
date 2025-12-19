"""
Jahn-Teller Dynamics IO Package

Main entry point for configuration parsing, file I/O, workflow orchestration,
and visualization.
"""

# Main public API - use lazy imports to avoid circular dependencies
__all__ = [
    'JTConfigParser',
    'JTOrchestrator',
    'Plotter',
    'PathManager',
    'CSVWriter',
]

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    """Lazy import function to avoid circular dependencies."""
    if name == 'JTConfigParser':
        from .config.parser import JTConfigParser
        return JTConfigParser
    elif name == 'JTOrchestrator':
        from .workflow.orchestrator import JTOrchestrator
        return JTOrchestrator
    elif name == 'Plotter':
        from .visualization.plotter import Plotter
        return Plotter
    elif name == 'PathManager':
        from .utils.path_manager import PathManager
        return PathManager
    elif name == 'CSVWriter':
        from .file_io.csv_writer import CSVWriter
        return CSVWriter
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
