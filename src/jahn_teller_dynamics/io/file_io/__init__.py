"""
File I/O operations for various file formats.

This package provides functionality for reading and writing different file formats
used in Jahn-Teller calculations, including CSV, VASP, and XML formats.
"""

# Lazy imports to avoid circular dependencies
# Import these directly from their modules when needed:
#   from jahn_teller_dynamics.io.file_io.csv_writer import CSVWriter
#   import jahn_teller_dynamics.io.file_io.vasp as V
#   import jahn_teller_dynamics.io.file_io.xml_parser as xml_parser

__all__ = [
    'CSVWriter',
    'vasp',
    'xml_parser',
]

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    """Lazy import function to avoid circular dependencies."""
    if name == 'CSVWriter':
        from .csv_writer import CSVWriter
        return CSVWriter
    elif name == 'vasp':
        from . import vasp
        return vasp
    elif name == 'xml_parser':
        from . import xml_parser
        return xml_parser
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

