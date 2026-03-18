"""
Main entry point for building and diagonalizing an LVC Hamiltonian from CSV inputs.

Default inputs are expected under:
    data/LVC_model/
        - LVC_epsilon.csv        (state_energy; legacy name: epsilon)
        - LVC_lambda.csv         (coupling; legacy name: lambda)
        - LVC_kappa.csv          (tuning; legacy name: kappa)
        - modes.csv

Run (from repo root):
    python3 -m jahn_teller_dynamics.LVC
or:
    python3 src/jahn_teller_dynamics/LVC.py
"""

from __future__ import annotations

from pathlib import Path
import argparse
from typing import Any

import numpy as np

from jahn_teller_dynamics.io.config.lvc_config import LVCConfigParser, LVCCalculation


def _run_dir() -> Path:
    """Directory where the script is run from (cwd). All paths are relative to this, same as Exe.py."""
    return Path.cwd()


def _default_data_dir() -> Path:
    """Default data dir relative to run directory (cwd)."""
    return _run_dir() / "data" / "LVC_model" / "butatrien_molecule"


def _calc_to_argparse_defaults(calc: LVCCalculation) -> dict[str, Any]:
    """Convert LVCCalculation to dict for argparse set_defaults."""
    return {
        "data_dir": calc.data_dir,
        "out_dir": calc.out_dir,
        "state_energy": calc.state_energy,
        "coupling_csv": calc.coupling_csv,
        "tuning_csv": calc.tuning_csv,
        "modes": calc.modes,
        "state_energy_path": calc.state_energy_path,
        "coupling_path": calc.coupling_path,
        "tuning_path": calc.tuning_path,
        "modes_path": calc.modes_path,
        "modes_to_use": calc.modes_to_use,
        "order": calc.order,
        "max_phonon_quanta": calc.max_phonon_quanta,
        "num_eigs": calc.num_eigs,
        "use_sparse": calc.use_sparse,
        "separator": calc.separator,
        "dimensionless_coordinates": calc.dimensionless_coordinates,
        "null_point_vib": calc.null_point_vib,
        "save_npz": calc.save_npz,
        "save_csv": calc.save_csv,
    }


def _args_to_calculation(args: argparse.Namespace, run_dir: Path) -> LVCCalculation:
    """Build LVCCalculation from parsed args (config + CLI merged)."""
    modes_to_use = getattr(args, "modes_to_use", None)
    if isinstance(modes_to_use, str):
        modes_to_use = [int(x.strip()) for x in modes_to_use.split(",") if x.strip()] or None
    elif modes_to_use is not None and not isinstance(modes_to_use, list):
        modes_to_use = None

    return LVCCalculation(
        data_dir=args.data_dir,
        out_dir=getattr(args, "out_dir", "") or "",
        state_energy=args.state_energy,
        coupling_csv=args.coupling_csv,
        tuning_csv=args.tuning_csv,
        modes=args.modes,
        state_energy_path=getattr(args, "state_energy_path", "") or "",
        coupling_path=getattr(args, "coupling_path", "") or "",
        tuning_path=getattr(args, "tuning_path", "") or "",
        modes_path=getattr(args, "modes_path", "") or "",
        modes_to_use=modes_to_use,
        order=int(args.order),
        max_phonon_quanta=getattr(args, "max_phonon_quanta", None),
        num_eigs=args.num_eigs,
        use_sparse=bool(args.use_sparse),
        separator=str(getattr(args, "separator", ";")),
        dimensionless_coordinates=bool(getattr(args, "dimensionless_coordinates", True)),
        null_point_vib=bool(getattr(args, "null_point_vib", True)),
        save_npz=bool(getattr(args, "save_npz", False)),
        save_csv=bool(getattr(args, "save_csv", True)),
    )


def read_lvc_cfg(cfg_path: str) -> dict[str, Any]:
    """
    Read an INI-style .cfg file and return argparse-defaults for `LVC.py`.

    Uses LVCConfigParser internally. Config file path is relative to run directory (cwd).
    CLI args always override config values.
    """
    run_dir = _run_dir()
    parser = LVCConfigParser(cfg_path, run_dir=run_dir)
    calc = parser.build_calculation()
    return _calc_to_argparse_defaults(calc)


def _build_parser_with_cfg_defaults(argv: list[str] | None) -> argparse.ArgumentParser:
    """
    Build the main argparse parser, optionally applying defaults from --config.

    CLI args always override config values.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default="", help="Path to INI-style .cfg file with [LVC] section.")
    ns, _ = pre.parse_known_args(argv)

    p = build_arg_parser()
    if getattr(ns, "config", ""):
        cfg_defaults = read_lvc_cfg(ns.config)
        # Only set defaults for keys present in cfg
        p.set_defaults(**cfg_defaults)
    return p


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build and diagonalize an LVC Hamiltonian from CSV inputs.")
    p.add_argument("--config", type=str, default="config_files/LVC_butatrien_molecule.cfg", help="Path to INI-style .cfg file with [LVC] section.")
    p.add_argument("--data-dir", type=str, default=str(_default_data_dir()), help="Directory containing LVC CSV files.")
    p.add_argument(
        "--state-energy",
        "--epsilon",
        dest="state_energy",
        type=str,
        default="orb_energies.csv",
        help="State energy CSV filename (state;value). Legacy alias: --epsilon.",
    )
    p.add_argument(
        "--coupling",
        "--lambda",
        dest="coupling_csv",
        type=str,
        default="coupling.csv",
        help="Off-diagonal coupling CSV filename (state_i;state_j;vibrational_mode;value). Legacy alias: --lambda.",
    )
    p.add_argument(
        "--tuning",
        "--kappa",
        dest="tuning_csv",
        type=str,
        default="tuning.csv",
        help="Diagonal tuning CSV filename (state;vibrational_mode;value). Legacy alias: --kappa.",
    )
    p.add_argument("--modes", type=str, default="modes.csv", help="Modes CSV filename (mode;omega).")
    p.add_argument(
        "--modes-to-use",
        type=str,
        default="",
        help="Comma-separated mode numbers to include from modes.csv (e.g. 1,2,3). Default: all modes.",
    )
    p.add_argument(
        "--state-energy-path",
        "--epsilon-path",
        dest="state_energy_path",
        type=str,
        default="",
        help="Optional explicit path to state_energy CSV (overrides --data-dir/--state-energy). Legacy alias: --epsilon-path.",
    )
    p.add_argument(
        "--coupling-path",
        "--lambda-path",
        dest="coupling_path",
        type=str,
        default="",
        help="Optional explicit path to coupling CSV (overrides --data-dir/--coupling). Legacy alias: --lambda-path.",
    )
    p.add_argument(
        "--tuning-path",
        "--kappa-path",
        dest="tuning_path",
        type=str,
        default="",
        help="Optional explicit path to tuning CSV (overrides --data-dir/--tuning). Legacy alias: --kappa-path.",
    )
    p.add_argument("--modes-path", type=str, default="", help="Optional explicit path to modes CSV (overrides --data-dir/--modes).")

    p.add_argument("--order", type=int, default=2, help="Phonon truncation order (default: 2).")
    p.add_argument(
        "--max-phonon-quanta",
        dest="max_phonon_quanta",
        type=int,
        default=None,
        help="When set, use constrained phonon space (sum of quantum numbers <= this) instead of per-mode order.",
    )
    p.add_argument("--num-eigs", type=int, default=None, help="Number of lowest eigenvalues/eigenvectors (default: all).")
    p.add_argument("--use-sparse", action="store_true", default=True, help="Use sparse operators/solver (default).")
    p.add_argument("--no-sparse", dest="use_sparse", action="store_false", help="Force dense operators/solver.")
    p.add_argument(
        "--out-dir",
        dest="out_dir",
        type=str,
        default="",
        help="Output directory (default: <run_dir>/results/LVC). Relative to cwd.",
    )
    p.add_argument("--separator", type=str, default=";", help="CSV separator (default: ';').")
    p.add_argument(
        "--dimensionless-coordinates",
        dest="dimensionless_coordinates",
        action="store_true",
        default=True,
        help="Use dimensionless position coordinates (default).",
    )
    p.add_argument(
        "--no-dimensionless-coordinates",
        dest="dimensionless_coordinates",
        action="store_false",
        help="Use dimensional position coordinates.",
    )
    p.add_argument(
        "--null-point-vib",
        dest="null_point_vib",
        action="store_true",
        default=True,
        help="Include zero-point energy 0.5*ħω in phonon Hamiltonian (default).",
    )
    p.add_argument(
        "--no-null-point-vib",
        dest="null_point_vib",
        action="store_false",
        help="Omit zero-point energy from phonon Hamiltonian.",
    )
    p.add_argument(
        "--save-npz",
        dest="save_npz",
        action="store_true",
        default=False,
        help="Save eigenvectors and eigenvalues in compressed NPZ format.",
    )
    p.add_argument(
        "--save-csv",
        dest="save_csv",
        action="store_true",
        default=True,
        help="Save eigenvectors and eigenvalues to CSV files (default).",
    )
    p.add_argument(
        "--no-save-csv",
        dest="save_csv",
        action="store_false",
        help="Do not save eigenvectors and eigenvalues to CSV files.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    from jahn_teller_dynamics.physics.models.lvc_model import LVC_model
    from jahn_teller_dynamics.physics.hamiltonians.lvc_hamiltonian import create_lvc_hamiltonian
    from jahn_teller_dynamics.io.file_io.csv_writer import CSVWriter

    args = _build_parser_with_cfg_defaults(argv).parse_args(argv)
    run_dir = _run_dir()
    calc = _args_to_calculation(args, run_dir)

    se_path, coupling_path, tuning_path, modes_path = calc.resolve_input_paths(run_dir)
    for pth in [se_path, coupling_path, tuning_path, modes_path]:
        if not pth.exists():
            raise FileNotFoundError(f"Missing required input file: {pth}")

    print("[1/4] Building LVC model from CSV inputs (electron + phonon subsystems)...")
    model = LVC_model.from_csvs(
        state_energy_csv_path=str(se_path),
        coupling_csv_path=str(coupling_path),
        tuning_csv_path=str(tuning_path),
        modes_csv_path=str(modes_path),
        mode_numbers=calc.modes_to_use,
        order=calc.order,
        max_phonon_quanta=calc.max_phonon_quanta,
        use_sparse=calc.use_sparse,
        dimensionless_coordinates=calc.dimensionless_coordinates,
        null_point_vib=calc.null_point_vib,
    )
    print(f"  → LVC model built: dim(electron)={model.electron.node.dim}, dim(phonon)={model.phonons.dim}, dim(full)={model.root_node.dim}")

    print("[2/4] Constructing LVC Hamiltonian (state_energy + Σ K_i + Σ X_i·V_i)...")
    H = create_lvc_hamiltonian(model)
    print(f"  → Hamiltonian matrix: dim(H) = {H.matrix.dim}")

    print("[3/4] Diagonalizing Hamiltonian...")
    num_of_vals = None if calc.num_eigs is None else calc.num_eigs
    if num_of_vals is None:
        num_of_vals = H.matrix.dim
    eig_space = H.calc_eigen_vals_vects(
        num_of_vals=num_of_vals,
        quantum_states_bases=model.root_node.base_states,
    )

    eig_vals = [k.eigen_val for k in eig_space.eigen_kets]
    print(f"  → Computed {len(eig_vals)} eigenvalues/eigenvectors")

    print("[4/4] Saving results...")
    out_dir = calc.resolve_out_dir(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if calc.save_csv:
        writer = CSVWriter(separator=calc.separator, index=True)
        eig_vec_path = str(out_dir / "eigenvectors.csv")
        eig_val_path = str(out_dir / "eigenvalues.csv")
        writer.write_eigen_vectors_and_values(eig_space, eig_vec_path, eig_val_path)
        print(f"Saved eigenvectors to: {eig_vec_path}")
        print(f"Saved eigenvalues  to: {eig_val_path}")

    if calc.save_npz:
        eig_vecs = np.column_stack([
            np.array(ket.coeffs.coeffs).flatten()
            for ket in eig_space.eigen_kets
        ])
        eig_vals_arr = np.array([complex(k.eigen_val) for k in eig_space.eigen_kets])
        basis_labels = np.array([str(s) for s in model.root_node.base_states._ket_states])
        npz_path = out_dir / "eigenvectors.npz"
        np.savez_compressed(
            npz_path,
            eigenvectors=eig_vecs,
            eigenvalues=eig_vals_arr,
            basis_labels=basis_labels,
            order=calc.order,
            dim=H.matrix.dim,
        )
        print(f"Saved NPZ to: {npz_path}")

    print("=" * 70)
    print("LVC diagonalization complete")
    print(f"dim(H) = {H.matrix.dim}")
    print(f"computed eigenvalues = {len(eig_vals)}")
    print("-" * 70)
    for i, ev in enumerate(eig_vals):
        try:
            ev_real = float(ev.real)  # type: ignore[union-attr]
            print(f"E[{i:03d}] = {ev_real:.12g}")
        except Exception:
            print(f"E[{i:03d}] = {ev}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

