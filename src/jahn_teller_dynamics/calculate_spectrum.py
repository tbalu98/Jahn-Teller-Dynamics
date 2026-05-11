#!/usr/bin/env python3
"""
Calculate absorption spectrum from LVC eigenvectors and dipole matrix.

Uses SpectrumBuilder to construct SpectrumCalculator from .cfg, then runs
the spectrum calculation.

Run:
    LVC_spectrum --config config_files/NV_spectrum.cfg
    LVC_spectrum --npz ... --dipole ...
"""

from __future__ import annotations
import numpy as np
import argparse
from pathlib import Path

from jahn_teller_dynamics.spectrum import SpectrumBuilder
from jahn_teller_dynamics.io.utils.run_context import RunContext

DEFAULT_CONFIG = RunContext.from_cwd().config_dir() / "NV_spectrum.cfg"


def _cli_overrides(args: argparse.Namespace) -> dict:
    """Build overrides dict from CLI args (non-empty values override config)."""
    overrides = {}
    if args.npz:
        overrides["npz"] = args.npz
    if args.dipole:
        overrides["dipole"] = args.dipole
    if getattr(args, "dipole_x", ""):
        overrides["dipole_x"] = args.dipole_x
    if getattr(args, "dipole_y", ""):
        overrides["dipole_y"] = args.dipole_y
    if getattr(args, "dipole_z", ""):
        overrides["dipole_z"] = args.dipole_z
    if args.output_folder:
        overrides["output_folder"] = args.output_folder
    if args.output_prefix:
        overrides["output_prefix"] = args.output_prefix
    if args.out:
        overrides["out"] = args.out
    if args.sep:
        overrides["separator"] = args.sep
    if args.energy_unit:
        overrides["energy_unit"] = args.energy_unit
    if getattr(args, "spectrum_range_min", None) is not None:
        overrides["spectrum_range_min"] = args.spectrum_range_min
    if getattr(args, "spectrum_range_max", None) is not None:
        overrides["spectrum_range_max"] = args.spectrum_range_max
    return overrides


def main() -> int:
    parser = argparse.ArgumentParser(description="Calculate LVC absorption spectrum")
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to .cfg file with [spectrum] section (overrides other args)",
    )
    parser.add_argument("--npz", type=str, default="", help="Path to eigenvectors.npz")
    parser.add_argument("--dipole", type=str, default="", help="Path to dipole matrix CSV (d×d), or use dipole_x/y/z for total")
    parser.add_argument("--dipole-x", type=str, default="", dest="dipole_x", help="Path to dipole X matrix CSV")
    parser.add_argument("--dipole-y", type=str, default="", dest="dipole_y", help="Path to dipole Y matrix CSV")
    parser.add_argument("--dipole-z", type=str, default="", dest="dipole_z", help="Path to dipole Z matrix CSV")
    parser.add_argument(
        "--output-folder",
        "--results-folder",
        type=str,
        default="",
        dest="output_folder",
        help="Base folder for all output files ([spectrum] output_folder)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output filename/path under output_folder (overrides output_prefix)",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="",
        dest="output_prefix",
        help="Prefix for output files: creates {prefix}.png and {prefix}.csv in output_folder",
    )
    parser.add_argument("--sep", type=str, default="", help="CSV separator for dipole file")
    parser.add_argument(
        "--spectrum-range-min",
        type=float,
        default=None,
        dest="spectrum_range_min",
        help="Minimum energy (eV) for spectrum range (default: 1.0)",
    )
    parser.add_argument(
        "--spectrum-range-max",
        type=float,
        default=None,
        dest="spectrum_range_max",
        help="Maximum energy (eV) for spectrum range (default: 2.0)",
    )
    parser.add_argument(
        "--energy-unit",
        type=str,
        default="",
        dest="energy_unit",
        choices=["", "eV", "ev", "meV", "mev", "cm-1", "inv_cm", "hartree", "ha"],
        help="Energy unit of NPZ eigenvalues (output always eV)",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not display the plot (save only)")
    args = parser.parse_args()

    run_ctx = RunContext.from_cwd()
    cfg_path = Path(args.config).expanduser() if args.config else DEFAULT_CONFIG
    if args.config and not cfg_path.is_absolute():
        cfg_path = run_ctx.resolve(str(cfg_path))
    elif not args.config:
        cfg_path = cfg_path.resolve()
    overrides = _cli_overrides(args)

    calculator = SpectrumBuilder.build_from_config(cfg_path, run_dir=run_ctx.run_dir, overrides=overrides)

    if not calculator.npz_path:
        parser.error("--npz is required, or provide --config with npz in [spectrum]")
    if not calculator.use_dipole_xyz and not calculator.dipole_path:
        parser.error(
            "Provide --dipole or all of --dipole-x, --dipole-y, --dipole-z "
            "(or dipole_x, dipole_y, dipole_z in config)"
        )

    show_plot = not getattr(args, "no_show", False)
    range_eV = np.linspace(
        calculator.spectrum_range_min,
        calculator.spectrum_range_max,
        1000,
    )
    #return calculator.calculate( show_plot=show_plot)

    return calculator.calculate_smearing(range_eV=range_eV, show_plot=show_plot)


if __name__ == "__main__":
    raise SystemExit(main())
