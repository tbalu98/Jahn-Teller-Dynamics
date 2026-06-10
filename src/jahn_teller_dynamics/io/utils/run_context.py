"""
Centralized path resolution for Jahn-Teller Dynamics.

All entry points (Exe, LVC, LVC_spectrum) use the run directory (cwd) as the base
for resolving relative paths. RunContext provides a single place for this logic.

Usage:
    ctx = RunContext.from_cwd()
    cfg_path = ctx.resolve(config_path_str)
    data_dir = ctx.resolve(data_dir_str, base=ctx.data_dir())
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


@dataclass
class RunContext:
    """
    Centralized path resolution.

    All paths are resolved relative to run_dir (typically Path.cwd()).
    Provides standard defaults for data/ and results/ directories.
    """

    run_dir: Path

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir).resolve()

    @classmethod
    def from_cwd(cls) -> "RunContext":
        """Create RunContext with current working directory as run_dir."""
        return cls(run_dir=Path.cwd())

    def resolve(
        self,
        path_str: str,
        *,
        base: Optional[Union[str, Path]] = None,
    ) -> Path:
        """
        Resolve a path string to an absolute Path.

        If path_str is absolute, return it resolved.
        If path_str is relative, resolve it against base (default: run_dir).

        Args:
            path_str: Path string (e.g. from config or CLI).
            base: Base for relative resolution. Defaults to run_dir.

        Returns:
            Resolved absolute Path.
        """
        if not path_str or not str(path_str).strip():
            raise ValueError("path_str cannot be empty")
        p = Path(str(path_str).strip()).expanduser()
        if p.is_absolute():
            return p.resolve()
        base_path = Path(base) if base is not None else self.run_dir
        if not base_path.is_absolute():
            base_path = (self.run_dir / base_path).resolve()
        return (base_path / p).resolve()

    def resolve_optional(
        self,
        path_str: str,
        *,
        base: Optional[Union[str, Path]] = None,
    ) -> Optional[Path]:
        """
        Resolve a path string, or return None if empty.

        Convenience for optional config paths.
        """
        if not path_str or not str(path_str).strip():
            return None
        return self.resolve(path_str, base=base)

    def data_dir(self, subpath: str = "") -> Path:
        """Default data directory: run_dir / 'data'."""
        p = self.run_dir / "data"
        if subpath:
            p = p / subpath
        return p.resolve()

    def results_dir(self, subpath: str = "") -> Path:
        """Default results directory: run_dir / 'results'."""
        p = self.run_dir / "results"
        if subpath:
            p = p / subpath
        return p.resolve()

    def config_dir(self, subpath: str = "") -> Path:
        """Default config directory: run_dir / 'config_files'."""
        p = self.run_dir / "config_files"
        if subpath:
            p = p / subpath
        return p.resolve()


def run_dir() -> Path:
    """
    Get the run directory (cwd).

    Convenience for code that only needs Path.cwd() without full RunContext.
    Use RunContext when you need resolve() or data/results dirs.
    """
    return Path.cwd()
