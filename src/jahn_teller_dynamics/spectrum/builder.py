"""
SpectrumBuilder - Builds SpectrumCalculator from .cfg configuration files.

Separates the logic of constructing the spectrum calculator (config loading,
path resolution) from the actual spectrum calculation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from jahn_teller_dynamics.math.maths import Lorentzian

from .config import SpectrumConfig
from .calculator import SpectrumCalculator


class SpectrumBuilder:
    """
    Builds SpectrumCalculator from .cfg files.

    Handles loading the [spectrum] section, resolving paths, and constructing
    a ready-to-use SpectrumCalculator.
    """

    @classmethod
    def build_from_config(
        cls,
        config_path: str | Path,
        *,
        run_dir: Optional[Path] = None,
        overrides: Optional[dict] = None,
    ) -> SpectrumCalculator:
        """
        Build SpectrumCalculator from a .cfg file.

        Args:
            config_path: Path to the .cfg file with [spectrum] section (relative to run_dir or absolute).
            run_dir: Base for resolving relative paths (cwd). Defaults to Path.cwd(). Same as LVC.py/Exe.py.
            overrides: Optional dict to override config values (e.g. from CLI).

        Returns:
            SpectrumCalculator configured and ready for calculation.
        """
        config = SpectrumConfig.from_config_file(
            config_path, run_dir=run_dir, overrides=overrides or {}
        )
        smearing = Lorentzian.from_config(
            config.smearing_function,
            config.smearing_HWHM,
            config.smearing_amplitude,
        )
        run_dir_resolved = run_dir if run_dir is not None else Path.cwd()
        return SpectrumCalculator(
            npz_paths=config.npz_paths,
            dipole_path=config.dipole_path,
            dipole_x_path=config.dipole_x_path,
            dipole_y_path=config.dipole_y_path,
            dipole_z_path=config.dipole_z_path,
            output_folder=config.output_folder,
            output_prefix=config.output_prefix,
            out_path=config.out_path,
            separator=config.separator,
            energy_unit=config.energy_unit,
            use_dipole_xyz=config.use_dipole_xyz,
            spectrum_range_min=config.spectrum_range_min,
            spectrum_range_max=config.spectrum_range_max,
            smearing=smearing,
            run_dir=run_dir_resolved,
        )
