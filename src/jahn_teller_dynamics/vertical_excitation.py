#!/usr/bin/env python3
"""
Vertical excitation stick spectrum from eigenvectors in NPZ.

Builds an initial state |ψ₀⟩ in the same basis as the saved eigenvectors: all zeros,
then components 0 and dim//2 set to 1, normalized by 1/√2. For each eigenstate |n⟩,
intensity I_n = |⟨ψ₀|n⟩|². Plots sticks at the eigenenergies (converted to eV per config).

Uses :func:`~jahn_teller_dynamics.io.file_io.npz_reader.load_lvc_npz` and the same
``[spectrum]`` config layout as :mod:`calculate_spectrum` (``npz``, ``output_folder``,
``output_prefix``, ``energy_unit``, ``spectrum_range_*``). Dipole entries are ignored.

Run::

    python -m jahn_teller_dynamics.vertical_excitation --config work_configfiles/NO3N/NO3N_PVC_spectrum.cfg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jahn_teller_dynamics.io.file_io.npz_reader import load_lvc_npz
from jahn_teller_dynamics.io.utils.file_utils import create_directory
from jahn_teller_dynamics.io.utils.run_context import RunContext
from jahn_teller_dynamics.io.utils.timestamp_print import print_ts
from jahn_teller_dynamics.spectrum.calculator import _energy_to_eV
from jahn_teller_dynamics.spectrum.config import SpectrumConfig

DEFAULT_CONFIG = RunContext.from_cwd().config_dir() / "NV_spectrum.cfg"


def vertical_initial_state(dim: int) -> np.ndarray:
    """
    |ψ₀⟩ with entries 0 and dim//2 equal to 1/√2 (others zero).

    If dim is 1, only index 0 exists; setting dim//2 == 0 gives amplitude √2 on that
    component after the final normalization, as both assignments hit the same index.
    """
    v = np.zeros(dim, dtype=np.complex128)
    v[0] = 1.0
    v[dim // 2] = 1.0
    v /= np.sqrt(2.0)
    return v


def vertical_excitation_intensities(
    initial: np.ndarray,
    eigenvectors: np.ndarray,
) -> np.ndarray:
    """
    I_j = |⟨initial|column_j⟩|² with eigenvectors of shape (dim, n_states).
    """
    if eigenvectors.shape[0] != initial.shape[0]:
        raise ValueError(
            f"initial dim {initial.shape[0]} != eigenvector rows {eigenvectors.shape[0]}"
        )
    overlaps = initial.conj() @ eigenvectors
    return np.abs(overlaps) ** 2


def _cli_overrides(args: argparse.Namespace) -> dict:
    overrides: dict = {}
    if args.npz:
        overrides["npz"] = args.npz
    if args.output_folder:
        overrides["output_folder"] = args.output_folder
    if args.output_prefix:
        overrides["output_prefix"] = args.output_prefix
    if args.out:
        overrides["out"] = args.out
    if args.energy_unit:
        overrides["energy_unit"] = args.energy_unit
    if getattr(args, "spectrum_range_min", None) is not None:
        overrides["spectrum_range_min"] = args.spectrum_range_min
    if getattr(args, "spectrum_range_max", None) is not None:
        overrides["spectrum_range_max"] = args.spectrum_range_max
    return overrides


def _output_folder_path(output_folder: str, run_dir: Path) -> Path:
    if output_folder:
        p = Path(output_folder).expanduser()
        if not p.is_absolute():
            p = run_dir / p
        return p.resolve()
    return (run_dir / "results").resolve()


def _vertical_output_png(config: SpectrumConfig, run_dir: Path) -> Path:
    """PNG under output_folder: ``{output_prefix}_vertical_excitation.png`` or ``{out_stem}_vertical_excitation.png``."""
    res_folder = _output_folder_path(config.output_folder, run_dir)
    if config.out_path:
        stem = Path(config.out_path).stem
        return res_folder / f"{stem}_vertical_excitation.png"
    prefix = config.output_prefix or "spectrum"
    return res_folder / f"{prefix}_vertical_excitation.png"


def run_vertical_excitation(
    config: SpectrumConfig,
    *,
    run_dir: Path,
    show_plot: bool = True,
) -> int:
    if not config.npz_path:
        print_ts(
            "Error: NPZ path not set. "
            "Use a .cfg with a [spectrum] section, e.g. npz = eigenvectors.npz "
            "(path is relative to results_folder / output_folder in that section), "
            "or override with:  --npz /absolute/or/repo-relative/path/to/eigenvectors.npz\n"
            "Note: PVC writes eigenvectors.npz under output_folder when save_npz is true; "
            "a PVC-only .cfg (no [spectrum]) will not define npz for this script.",
            flush=True,
        )
        return 1
    npz_path = Path(config.npz_path)
    if not npz_path.exists():
        print_ts(f"NPZ not found: {npz_path}", flush=True)
        return 1

    raw = load_lvc_npz(npz_path)
    dim_file = int(raw["dim"])
    eig_vecs = np.asarray(raw["eigenvectors"], dtype=np.complex128)
    eig_vals = np.asarray(raw["eigenvalues"])

    if eig_vecs.shape[0] != dim_file:
        raise ValueError(
            f"NPZ dim ({dim_file}) does not match eigenvectors rows ({eig_vecs.shape[0]})."
        )

    psi0 = vertical_initial_state(dim_file)
    intensities = vertical_excitation_intensities(psi0, eig_vecs)

    energies_raw = [float(np.real(complex(e))) for e in eig_vals]
    energies_eV = _energy_to_eV(energies_raw, config.energy_unit)

    png_path = _vertical_output_png(config, run_dir)
    create_directory(png_path.parent)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.stem(
        energies_eV,
        intensities,
        linefmt="C1-",
        basefmt=" ",
        markerfmt="C1o",
    )
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel(r"$|\langle \psi_0 | n \rangle|^2$")
    ax.set_title("Vertical excitation (stick spectrum)")
    ax.set_xlim(config.spectrum_range_min, config.spectrum_range_max)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close()
    print_ts(f"Vertical excitation plot saved to {png_path}", flush=True)

    csv_path = png_path.with_suffix(".csv")
    pd.DataFrame(
        {
            "energy_eV": energies_eV,
            "intensity": intensities,
        }
    ).to_csv(csv_path, index=False, sep=config.separator or ";")
    print_ts(f"Vertical excitation data saved to {csv_path}", flush=True)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Vertical excitation stick spectrum from eigenvectors.npz (uses [spectrum] config)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to .cfg with [spectrum] (npz, output_folder, output_prefix, energy_unit, …).",
    )
    parser.add_argument("--npz", type=str, default="", help="Override path to eigenvectors.npz")
    parser.add_argument(
        "--output-folder",
        "--results-folder",
        type=str,
        default="",
        dest="output_folder",
        help="Override output folder from config ([spectrum] output_folder)",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="",
        dest="output_prefix",
        help="Override spectrum output_prefix (vertical files use {prefix}_vertical_excitation)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Override output base name under output_folder",
    )
    parser.add_argument(
        "--energy-unit",
        type=str,
        default="",
        dest="energy_unit",
        choices=["", "eV", "ev", "meV", "mev", "cm-1", "inv_cm", "hartree", "ha"],
        help="Unit of eigenvalues in NPZ (same as spectrum)",
    )
    parser.add_argument(
        "--spectrum-range-min",
        type=float,
        default=None,
        dest="spectrum_range_min",
        help="X-axis minimum (eV)",
    )
    parser.add_argument(
        "--spectrum-range-max",
        type=float,
        default=None,
        dest="spectrum_range_max",
        help="X-axis maximum (eV)",
    )
    parser.add_argument("--no-show", action="store_true", help="Save plot only, do not display")
    args = parser.parse_args()

    run_ctx = RunContext.from_cwd()
    cfg_path = Path(args.config).expanduser() if args.config else DEFAULT_CONFIG
    if args.config and not cfg_path.is_absolute():
        cfg_path = run_ctx.resolve(str(cfg_path))
    elif not args.config:
        cfg_path = cfg_path.resolve()

    overrides = _cli_overrides(args)
    config = SpectrumConfig.from_config_file(
        cfg_path, run_dir=run_ctx.run_dir, overrides=overrides
    )

    show_plot = not args.no_show
    return run_vertical_excitation(config, run_dir=run_ctx.run_dir, show_plot=show_plot)


if __name__ == "__main__":
    raise SystemExit(main())
