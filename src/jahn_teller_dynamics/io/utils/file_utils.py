"""
File utility functions.

This module provides utility functions for file and directory operations.
"""

import os
from pathlib import Path
from typing import Union


def get_repo_root() -> Path:
    """
    Get the repository root (parent of src/).

    Assumes this module lives under src/jahn_teller_dynamics/io/utils/.
    """
    # file_utils.py -> utils -> io -> jahn_teller_dynamics -> src -> repo
    return Path(__file__).resolve().parents[4]


def create_directory(directory_path: Union[str, Path]) -> None:
    """
    Create a directory if it doesn't exist.

    Args:
        directory_path: Path to the directory to create
    """
    path = Path(directory_path) if not isinstance(directory_path, Path) else directory_path
    path.mkdir(parents=True, exist_ok=True)


def resolve_path_relative(
    path_str: str,
    base: Union[str, Path],
    repo_root: Union[str, Path],
) -> Path:
    """
    Resolve a path; if relative, interpret relative to base (itself relative to repo_root).

    Prefer RunContext.resolve() for new code; this is kept for backward compatibility.

    Args:
        path_str: Path string (can be absolute or relative)
        base: Base path for relative resolution (relative to repo_root if not absolute)
        repo_root: Repository root for resolving relative bases

    Returns:
        Resolved absolute Path
    """
    pp = Path(path_str.strip())
    base_p = Path(base)
    repo_p = Path(repo_root)
    if not pp.is_absolute():
        if not base_p.is_absolute():
            base_p = repo_p / base_p
        pp = base_p / pp
    return pp.resolve()

