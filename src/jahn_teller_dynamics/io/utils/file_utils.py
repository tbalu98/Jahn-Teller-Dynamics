"""
File utility functions.

This module provides utility functions for file and directory operations.
"""

import os


def create_directory(directory_path: str) -> None:
    """
    Create a directory if it doesn't exist.
    
    Args:
        directory_path: Path to the directory to create
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

