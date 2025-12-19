"""
Configuration parsing and writing components.

This package provides all functionality for reading, parsing, and writing
Jahn-Teller configuration files.
"""

# Constants (re-export for convenience) - these are safe to import immediately
from .constants import *

# Main public API - use lazy imports to avoid circular dependencies
__all__ = [
    'JTConfigParser',
    'ConfigReader',
    'ConfigWriter',
    'ParameterExtractor',
    'SectionTypeDetector',
    'FieldVectorParser',
]

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    """Lazy import function to avoid circular dependencies."""
    if name == 'JTConfigParser':
        from .parser import JTConfigParser
        return JTConfigParser
    elif name == 'ConfigReader':
        from .reader import ConfigReader
        return ConfigReader
    elif name == 'ConfigWriter':
        from .writer import ConfigWriter
        return ConfigWriter
    elif name == 'ParameterExtractor':
        from .parameter_extractor import ParameterExtractor
        return ParameterExtractor
    elif name == 'SectionTypeDetector':
        from .section_detector import SectionTypeDetector
        return SectionTypeDetector
    elif name == 'FieldVectorParser':
        from .field_parser import FieldVectorParser
        return FieldVectorParser
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

