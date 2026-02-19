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
from configparser import ConfigParser
from typing import Any, Mapping, Optional


def _default_data_dir() -> Path:
    # .../src/jahn_teller_dynamics/LVC.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2] / "data" / "LVC_model" / "butatrien_molecule"

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_lvc_cfg(cfg_path: str) -> dict[str, Any]:
    """
    Read an INI-style .cfg file and return argparse-defaults for `LVC.py`.

    Notes
    -----
    - The config section is `[LVC]` (case-insensitive).
    - Relative paths in the config are interpreted relative to the **repo root**.
      (So `data/LVC_model` and `results/LVC` work regardless of current working dir.)
    - CLI args always override config values.
    """
    cfg_file = Path(cfg_path).expanduser()
    if not cfg_file.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_file}")

    cp = ConfigParser()
    cp.read(cfg_file)

    # ConfigParser is case-insensitive for option names; section names are case-sensitive.
    # We support:
    # - [LVC] for LVC-specific inputs
    # - [essentials] (existing project convention) for common IO + order + sparse
    lvc_section = None
    for candidate in ("LVC", "lvc"):
        if cp.has_section(candidate):
            lvc_section = candidate
            break
    essentials_section = "essentials" if cp.has_section("essentials") else None

    if lvc_section is None and essentials_section is None:
        raise ValueError(f"Missing [LVC] (or [essentials]) section in config: {cfg_file}")

    get = cp.get  # (section, option)
    base = _repo_root()

    def _get_path_opt(opt_name: str, fallback: Optional[str] = None) -> Optional[str]:
        section = lvc_section if lvc_section is not None else essentials_section  # type: ignore[assignment]
        if section is None or not cp.has_option(section, opt_name):
            return fallback
        raw = get(section, opt_name, fallback=fallback)
        if raw is None:
            return None
        raw = raw.strip()
        if raw == "":
            return ""
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = (base / p)
        return str(p.resolve())

    def _get_str(opt_name: str, fallback: Optional[str] = None) -> Optional[str]:
        section = lvc_section if lvc_section is not None else essentials_section  # type: ignore[assignment]
        if section is None or not cp.has_option(section, opt_name):
            return fallback
        val = get(section, opt_name, fallback=fallback)
        return val.strip() if val is not None else None

    def _get_int(opt_name: str, fallback: Optional[int] = None) -> Optional[int]:
        section = lvc_section if lvc_section is not None else essentials_section  # type: ignore[assignment]
        if section is None or not cp.has_option(section, opt_name):
            return fallback
        return cp.getint(section, opt_name, fallback=fallback)  # type: ignore[arg-type]

    def _get_bool(opt_name: str, fallback: Optional[bool] = None) -> Optional[bool]:
        section = lvc_section if lvc_section is not None else essentials_section  # type: ignore[assignment]
        if section is None or not cp.has_option(section, opt_name):
            return fallback
        return cp.getboolean(section, opt_name, fallback=fallback)  # type: ignore[arg-type]

    defaults: dict[str, Any] = {}

    # -------------------------
    # Essentials-style defaults
    # -------------------------
    # Map existing project keys -> LVC CLI keys.
    if essentials_section is not None:
        # input_folder / output_folder / maximum_number_of_vibrational_quanta / use_sparse
        in_folder = cp.get(essentials_section, "input_folder", fallback="").strip()
        if in_folder:
            p = Path(in_folder).expanduser()
            if not p.is_absolute():
                p = base / p
            defaults["data_dir"] = str(p.resolve())

        out_folder = cp.get(essentials_section, "output_folder", fallback="").strip()
        if out_folder:
            p = Path(out_folder).expanduser()
            if not p.is_absolute():
                p = base / p
            defaults["out_dir"] = str(p.resolve())

        max_q = cp.getint(essentials_section, "maximum_number_of_vibrational_quanta", fallback=0)
        if max_q:
            defaults["order"] = int(max_q)

        if cp.has_option(essentials_section, "use_sparse"):
            defaults["use_sparse"] = cp.getboolean(essentials_section, "use_sparse", fallback=True)

    # -------------------------
    # LVC-specific overrides
    # -------------------------
    # Paths: either provide explicit *_path options, or data-dir + filenames.
    data_dir = _get_path_opt("data_dir", fallback=None) if lvc_section is not None else None
    if data_dir is not None:
        defaults["data_dir"] = data_dir

    # Filenames (used together with data-dir)
    state_energy = _get_str("state_energy", fallback=None) if lvc_section is not None else None
    if state_energy is None and lvc_section is not None:
        state_energy = _get_str("epsilon", fallback=None)  # legacy
    if state_energy is not None:
        defaults["state_energy"] = state_energy

    coupling = _get_str("coupling", fallback=None) if lvc_section is not None else None
    if coupling is None and lvc_section is not None:
        coupling = _get_str("lambda", fallback=None)  # legacy
    if coupling is not None:
        defaults["coupling_csv"] = coupling

    tuning = _get_str("tuning", fallback=None) if lvc_section is not None else None
    if tuning is None and lvc_section is not None:
        tuning = _get_str("kappa", fallback=None)  # legacy
    if tuning is not None:
        defaults["tuning_csv"] = tuning

    modes = _get_str("modes", fallback=None) if lvc_section is not None else None
    if modes is not None:
        defaults["modes"] = modes

    # Explicit CSV paths (override data-dir + filenames)
    eps_p = None
    lam_p = None
    kap_p = None
    mod_p = None
    if lvc_section is not None:
        eps_p = (
            _get_path_opt("state_energy_path", fallback=None)
            or _get_path_opt("state_energy_csv_path", fallback=None)
            or _get_path_opt("epsilon_path", fallback=None)  # legacy
            or _get_path_opt("epsilon_csv_path", fallback=None)  # legacy
        )
        lam_p = (
            _get_path_opt("coupling_path", fallback=None)
            or _get_path_opt("coupling_csv_path", fallback=None)
            or _get_path_opt("lambda_path", fallback=None)  # legacy
            or _get_path_opt("lambda_csv_path", fallback=None)  # legacy
        )
        kap_p = (
            _get_path_opt("tuning_path", fallback=None)
            or _get_path_opt("tuning_csv_path", fallback=None)
            or _get_path_opt("kappa_path", fallback=None)  # legacy
            or _get_path_opt("kappa_csv_path", fallback=None)  # legacy
        )
        mod_p = _get_path_opt("modes_path", fallback=None) or _get_path_opt("modes_csv_path", fallback=None)
    if eps_p is not None:
        defaults["state_energy_path"] = eps_p
    if lam_p is not None:
        defaults["coupling_path"] = lam_p
    if kap_p is not None:
        defaults["tuning_path"] = kap_p
    if mod_p is not None:
        defaults["modes_path"] = mod_p

    # Run parameters
    order = _get_int("order", fallback=None) if lvc_section is not None else None
    if order is not None:
        defaults["order"] = int(order)

    num_eigs = None
    if lvc_section is not None:
        num_eigs = _get_int("num_eigs", fallback=None) or _get_int("num-eigs", fallback=None)
    if num_eigs is not None:
        defaults["num_eigs"] = int(num_eigs)

    use_sparse = _get_bool("use_sparse", fallback=None) if lvc_section is not None else None
    if use_sparse is not None:
        defaults["use_sparse"] = bool(use_sparse)

    out_dir = None
    if lvc_section is not None:
        out_dir = _get_path_opt("out_dir", fallback=None) or _get_path_opt("out-dir", fallback=None)
    if out_dir is not None and out_dir != "":
        defaults["out_dir"] = out_dir

    separator = _get_str("separator", fallback=None) if lvc_section is not None else None
    if separator is not None:
        defaults["separator"] = separator

    # Phonon options (defaults: dimensionless_coordinates=True, null_point_vib=True)
    def _get_bool_any(opt: str) -> Optional[bool]:
        for sec in (lvc_section, essentials_section):
            if sec is not None and cp.has_option(sec, opt):
                return cp.getboolean(sec, opt, fallback=True)
        return None

    dimless = _get_bool_any("dimensionless_coordinates")
    if dimless is not None:
        defaults["dimensionless_coordinates"] = dimless
    npv = _get_bool_any("null_point_vib")
    if npv is not None:
        defaults["null_point_vib"] = npv

    return defaults


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
    p.add_argument("--num-eigs", type=int, default=20, help="Number of lowest eigenvalues/eigenvectors to compute.")
    p.add_argument("--use-sparse", action="store_true", default=True, help="Use sparse operators/solver (default).")
    p.add_argument("--no-sparse", dest="use_sparse", action="store_false", help="Force dense operators/solver.")
    p.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output directory for eigenvalues/eigenvectors CSVs (default: <repo_root>/results/LVC).",
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
    return p


def main(argv: list[str] | None = None) -> int:
    from jahn_teller_dynamics.physics.models.lvc_model import LVC_model
    from jahn_teller_dynamics.physics.hamiltonians.lvc_hamiltonian import create_lvc_hamiltonian
    from jahn_teller_dynamics.io.file_io.csv_writer import CSVWriter

    args = _build_parser_with_cfg_defaults(argv).parse_args(argv)

    repo_root = _repo_root()
    data_dir = Path(args.data_dir).expanduser()
    if not data_dir.is_absolute():
        data_dir = (repo_root / data_dir)
    data_dir = data_dir.resolve()

    se_path = Path(args.state_energy_path).expanduser() if getattr(args, "state_energy_path", "") else (data_dir / args.state_energy)
    coupling_path = Path(args.coupling_path).expanduser() if getattr(args, "coupling_path", "") else (data_dir / args.coupling_csv)
    tuning_path = Path(args.tuning_path).expanduser() if getattr(args, "tuning_path", "") else (data_dir / args.tuning_csv)
    modes_path = Path(args.modes_path).expanduser() if getattr(args, "modes_path", "") else (data_dir / args.modes)

    se_path = se_path.resolve()
    coupling_path = coupling_path.resolve()
    tuning_path = tuning_path.resolve()
    modes_path = modes_path.resolve()

    for pth in [se_path, coupling_path, tuning_path, modes_path]:
        if not pth.exists():
            raise FileNotFoundError(f"Missing required input file: {pth}")

    # Build model from CSVs (phonon order is explicitly set here)
    model = LVC_model.from_csvs(
        state_energy_csv_path=str(se_path),
        coupling_csv_path=str(coupling_path),
        tuning_csv_path=str(tuning_path),
        modes_csv_path=str(modes_path),
        order=int(args.order),
        use_sparse=bool(args.use_sparse),
        dimensionless_coordinates=bool(getattr(args, "dimensionless_coordinates", True)),
        null_point_vib=bool(getattr(args, "null_point_vib", True)),
    )

    # Build Hamiltonian once
    H = create_lvc_hamiltonian(model)

    # Diagonalize
    eig_space = H.calc_eigen_vals_vects(
        num_of_vals=int(args.num_eigs),
        quantum_states_bases=model.root_node.base_states,
    )

    eig_vals = [k.eigen_val for k in eig_space.eigen_kets]

    # Save eigenvectors and eigenvalues
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (repo_root / "results" / "LVC")
    out_dir.mkdir(parents=True, exist_ok=True)

    writer = CSVWriter(separator=str(args.separator), index=True)
    eig_vec_path = str(out_dir / "eigenvectors.csv")
    eig_val_path = str(out_dir / "eigenvalues.csv")
    writer.write_eigen_vectors_and_values(eig_space, eig_vec_path, eig_val_path)
    print(f"Saved eigenvectors to: {eig_vec_path}")
    print(f"Saved eigenvalues  to: {eig_val_path}")

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

