"""
Spectrum configuration loading via ConfigReader.

Reads the [spectrum] section from .cfg files and resolves paths
relative to data_folder and output_folder. Uses PathManager and file_utils.
"""

from __future__ import annotations

from configparser import ConfigParser
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from jahn_teller_dynamics.io.config.reader import ConfigReader
from jahn_teller_dynamics.io.config.constants import (
    spectrum_section,
    smearing_function_section,
    smearing_function_opt,
    smearing_HWHM_opt,
    smearing_amplitude_opt,
)
from jahn_teller_dynamics.io.utils.path_manager import PathManager
from jahn_teller_dynamics.io.utils.run_context import RunContext


def _spectrum_output_folder_raw_from_overrides(
    overrides: dict, path_manager: PathManager
) -> str:
    """Prefer output_folder override, then legacy results_folder, then .cfg."""
    for key in ("output_folder", "results_folder"):
        val = overrides.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return path_manager.get_spectrum_output_folder()


def _parse_smearing_float(
    val: Optional[str | float], fallback: Optional[float] = None
) -> Optional[float]:
    """Parse a smearing float from override; return fallback if invalid."""
    if val is None:
        return fallback
    try:
        return float(val)
    except (TypeError, ValueError):
        return fallback


def _parse_npz_value(val: object) -> List[str]:
    """
    Normalise the ``npz`` option to an ordered list of (unresolved) paths.

    Accepts:
      * ``None`` / empty → ``[]``
      * ``list`` / ``tuple`` (from CLI ``--npz a.npz b.npz``)
      * single string (from .cfg or single-value override)
      * comma-separated or newline-separated string
        (e.g. ``npz = ground.npz, excited_1eV.npz`` in .cfg)

    The first entry is conventionally the file containing the ground state.
    """
    if val is None:
        return []
    if isinstance(val, (list, tuple)):
        return [str(x).strip() for x in val if str(x).strip()]
    if not isinstance(val, str):
        val = str(val)
    out: List[str] = []
    for line in val.splitlines():
        for chunk in line.split(","):
            chunk = chunk.strip()
            if chunk:
                out.append(chunk)
    return out


@dataclass
class SpectrumConfig:
    """
    Configuration for spectrum calculation, loaded from [spectrum] section.

    All paths are resolved to absolute paths. Paths in config are relative to
    data_folder (for dipole) or output_folder (for npz, out).

    The ``npz`` option may name a single eigenvectors NPZ or a list of them
    (comma- or newline-separated in .cfg, or multiple values on the CLI). The
    first NPZ is assumed to contain the ground state; the remaining files
    contribute additional final-state eigenkets to the absorption spectrum
    (useful when each NPZ is produced by a separate SLEPc shift–invert run
    targeting a specific energy window).
    """

    npz_paths: List[str] = field(default_factory=list)
    dipole_path: str = ""
    dipole_x_path: str = ""
    dipole_y_path: str = ""
    dipole_z_path: str = ""
    output_folder: str = ""
    output_prefix: str = "spectrum"
    out_path: str = ""
    separator: str = ";"
    energy_unit: str = "ev"
    use_dipole_xyz: bool = False
    spectrum_range_min: float = 1.0
    spectrum_range_max: float = 2.0
    # Optional smearing (from [smearing_function] section)
    smearing_function: str = ""
    smearing_HWHM: Optional[float] = None
    smearing_amplitude: float = 1.0

    @property
    def npz_path(self) -> str:
        """First NPZ path (ground state); empty when no NPZ is configured.

        Kept as a property for backwards compatibility with single-file
        callers (``vertical_excitation.py``, the existing CLI argument
        parser, etc.). Multi-file consumers should read ``npz_paths``.
        """
        return self.npz_paths[0] if self.npz_paths else ""

    @classmethod
    def from_config_reader(
        cls,
        reader: ConfigReader,
        run_dir: Optional[Path] = None,
        *,
        overrides: Optional[dict] = None,
        config_path: Optional[str] = None,
    ) -> "SpectrumConfig":
        """
        Build SpectrumConfig from a ConfigReader that has read a config with [spectrum].

        Args:
            reader: ConfigReader wrapping a ConfigParser with [spectrum] section.
            run_dir: Base for resolving relative paths (cwd). Defaults to Path.cwd().
            overrides: Optional dict of option -> value to override config (e.g. from CLI).
            config_path: Optional path to config file (for PathManager).
        """
        overrides = overrides or {}
        ctx = RunContext(run_dir=run_dir if run_dir is not None else Path.cwd())

        if not reader.has_section(spectrum_section):
            return cls()

        path_manager = PathManager(reader, config_path)

        def get(opt: str, default: str = "") -> str:
            val = overrides.get(opt)
            if val is not None:
                return str(val).strip()
            return path_manager.get_option_from_section(spectrum_section, opt, default)

        data_folder_raw = overrides.get("data_folder") or path_manager.get_spectrum_data_folder()
        output_folder_raw = _spectrum_output_folder_raw_from_overrides(
            overrides, path_manager
        )

        data_folder = ctx.run_dir
        output_folder_path = ctx.run_dir
        if data_folder_raw:
            data_folder = ctx.resolve(data_folder_raw)
        if output_folder_raw:
            output_folder_path = ctx.resolve(output_folder_raw)

        # `npz` may be a single path, a comma-separated list, or — via CLI
        # override — a true Python list. Special-case it because `get()`
        # str()-ifies everything (which would mangle a list).
        npz_override = overrides.get("npz")
        if npz_override is not None:
            npz_list_raw = _parse_npz_value(npz_override)
        else:
            npz_list_raw = _parse_npz_value(
                path_manager.get_option_from_section(spectrum_section, "npz", "")
            )

        dipole_raw = get("dipole")
        dipole_x_raw = get("dipole_x")
        dipole_y_raw = get("dipole_y")
        dipole_z_raw = get("dipole_z")

        npz_paths = [
            str(ctx.resolve(p, base=output_folder_path)) for p in npz_list_raw
        ]
        dipole_path = str(ctx.resolve(dipole_raw, base=data_folder)) if dipole_raw else ""
        dipole_x_path = str(ctx.resolve(dipole_x_raw, base=data_folder)) if dipole_x_raw else ""
        dipole_y_path = str(ctx.resolve(dipole_y_raw, base=data_folder)) if dipole_y_raw else ""
        dipole_z_path = str(ctx.resolve(dipole_z_raw, base=data_folder)) if dipole_z_raw else ""

        output_folder_str = (
            str(output_folder_path) if output_folder_path != ctx.run_dir else ""
        )
        output_prefix = get("output_prefix", "spectrum")
        out_path = get("out")
        separator = get("separator", ";") or ";"
        energy_unit = (get("energy_unit", "ev") or "ev").lower()

        use_dipole_xyz = bool(dipole_x_path and dipole_y_path and dipole_z_path)

        # Spectrum energy range (eV)
        def get_float(opt: str, default: float) -> float:
            val = overrides.get(opt)
            if val is not None:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    pass
            raw = path_manager.get_option_from_section(spectrum_section, opt, str(default))
            if raw:
                try:
                    return float(raw)
                except ValueError:
                    pass
            return default

        spectrum_range_min = get_float("spectrum_range_min", 1.0)
        spectrum_range_max = get_float("spectrum_range_max", 2.0)

        # Optional [smearing_function] section
        smearing_function = ""
        smearing_HWHM: Optional[float] = None
        smearing_amplitude = 1.0
        if reader.has_section(smearing_function_section):
            smearing_function = path_manager.get_option_from_section(
                smearing_function_section, smearing_function_opt, ""
            )
            hwhm_str = path_manager.get_option_from_section(
                smearing_function_section, smearing_HWHM_opt, ""
            )
            if hwhm_str:
                try:
                    smearing_HWHM = float(hwhm_str)
                except ValueError:
                    pass
            amp_str = path_manager.get_option_from_section(
                smearing_function_section, smearing_amplitude_opt, "1.0"
            )
            if amp_str:
                try:
                    smearing_amplitude = float(amp_str)
                except ValueError:
                    smearing_amplitude = 1.0

        return cls(
            npz_paths=npz_paths,
            dipole_path=dipole_path,
            dipole_x_path=dipole_x_path,
            dipole_y_path=dipole_y_path,
            dipole_z_path=dipole_z_path,
            output_folder=output_folder_str,
            output_prefix=output_prefix,
            out_path=out_path,
            separator=separator,
            energy_unit=energy_unit,
            use_dipole_xyz=use_dipole_xyz,
            spectrum_range_min=spectrum_range_min,
            spectrum_range_max=spectrum_range_max,
            smearing_function=(
                str(overrides["smearing_function"]).strip()
                if overrides.get("smearing_function") else smearing_function
            ),
            smearing_HWHM=(
                _parse_smearing_float(overrides.get("smearing_HWHM"), smearing_HWHM)
                if "smearing_HWHM" in overrides else smearing_HWHM
            ),
            smearing_amplitude=(
                _parse_smearing_float(overrides.get("smearing_amplitude"), 1.0) or 1.0
                if "smearing_amplitude" in overrides else smearing_amplitude
            ),
        )

    @classmethod
    def from_config_file(
        cls,
        config_path: str | Path,
        run_dir: Optional[Path] = None,
        *,
        overrides: Optional[dict] = None,
    ) -> "SpectrumConfig":
        """
        Load SpectrumConfig from a .cfg file using ConfigReader.

        Args:
            config_path: Path to the .cfg file (relative to run_dir or absolute).
            run_dir: Base for resolving relative paths (cwd). Defaults to Path.cwd().
            overrides: Optional dict to override config values (e.g. from CLI).
        """
        ctx = RunContext(run_dir=run_dir if run_dir is not None else Path.cwd())
        config_path = Path(config_path).expanduser()
        if not config_path.is_absolute():
            config_path = ctx.resolve(str(config_path))
        if not config_path.exists():
            # No config file: build from overrides only
            overrides = overrides or {}
            of_raw = overrides.get("output_folder") or overrides.get(
                "results_folder", ""
            )
            data_folder_raw = overrides.get("data_folder", "")
            output_folder_path = ctx.resolve(of_raw or ".")
            data_folder = ctx.resolve(data_folder_raw or ".")

            npz_raw = overrides.get("npz")
            dipole_raw = overrides.get("dipole", "")
            dx = overrides.get("dipole_x", "")
            dy = overrides.get("dipole_y", "")
            dz = overrides.get("dipole_z", "")

            npz_paths = [
                str(ctx.resolve(p, base=output_folder_path))
                for p in _parse_npz_value(npz_raw)
            ]
            dipole_path = str(ctx.resolve(dipole_raw, base=data_folder)) if dipole_raw else ""
            dipole_x_path = str(ctx.resolve(dx, base=data_folder)) if dx else ""
            dipole_y_path = str(ctx.resolve(dy, base=data_folder)) if dy else ""
            dipole_z_path = str(ctx.resolve(dz, base=data_folder)) if dz else ""

            output_folder_str = (
                str(output_folder_path) if output_folder_path != ctx.run_dir else ""
            )
            sf = (overrides.get("smearing_function") or "").strip()
            sh = _parse_smearing_float(overrides.get("smearing_HWHM"), None)
            sa = _parse_smearing_float(overrides.get("smearing_amplitude"), 1.0) or 1.0
            return cls(
                npz_paths=npz_paths,
                dipole_path=dipole_path,
                dipole_x_path=dipole_x_path,
                dipole_y_path=dipole_y_path,
                dipole_z_path=dipole_z_path,
                output_folder=output_folder_str,
                output_prefix=overrides.get("output_prefix", "spectrum"),
                out_path=overrides.get("out", ""),
                separator=overrides.get("separator", overrides.get("sep", ";")) or ";",
                energy_unit=(overrides.get("energy_unit", "ev") or "ev").lower(),
                use_dipole_xyz=bool(dipole_x_path and dipole_y_path and dipole_z_path),
                spectrum_range_min=_parse_smearing_float(overrides.get("spectrum_range_min"), 1.0) or 1.0,
                spectrum_range_max=_parse_smearing_float(overrides.get("spectrum_range_max"), 2.0) or 2.0,
                smearing_function=sf,
                smearing_HWHM=sh,
                smearing_amplitude=sa,
            )

        cp = ConfigParser()
        cp.read(config_path)
        reader = ConfigReader(cp)
        return cls.from_config_reader(
            reader, run_dir=run_dir, overrides=overrides, config_path=str(config_path)
        )
