"""
LVC configuration parsing.

LVCConfigParser reads .cfg files and builds LVCCalculation objects that contain
all the details needed for an LVC Hamiltonian diagonalization run.
"""

from __future__ import annotations

from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from jahn_teller_dynamics.io.config.reader import ConfigReader
from jahn_teller_dynamics.io.utils.path_manager import PathManager


def _run_dir() -> Path:
    """Directory where the script is run from (cwd). Paths are relative to this."""
    return Path.cwd()


@dataclass
class LVCCalculation:
    """
    Holds all parameters for an LVC Hamiltonian diagonalization.

    Built from a .cfg file by LVCConfigParser. Paths are resolved relative to
    the run directory (cwd).
    """

    # Paths
    data_dir: str = ""
    out_dir: str = ""
    state_energy_path: str = ""
    coupling_path: str = ""
    tuning_path: str = ""
    modes_path: str = ""

    # Filenames (used with data_dir when explicit paths not set)
    state_energy: str = "orb_energies.csv"
    coupling_csv: str = "coupling.csv"
    tuning_csv: str = "tuning.csv"
    modes: str = "modes.csv"

    # Run parameters
    order: int = 2
    max_phonon_quanta: Optional[int] = None  # When set, use MultiModeConstrainedPhononSystem (sum <= this)
    num_eigs: Optional[int] = None
    use_sparse: bool = True
    modes_to_use: Optional[list[int]] = None

    # Options
    separator: str = ";"
    dimensionless_coordinates: bool = True
    null_point_vib: bool = True
    save_npz: bool = False
    save_csv: bool = True

    def resolve_input_paths(self, run_dir: Optional[Path] = None) -> tuple[Path, Path, Path, Path]:
        """
        Resolve state_energy, coupling, tuning, modes paths to absolute Paths.

        Returns:
            (se_path, coupling_path, tuning_path, modes_path)
        """
        rd = run_dir or _run_dir()
        data = Path(self.data_dir).expanduser()
        if not data.is_absolute():
            data = (rd / data).resolve()

        def _path(explicit: str, filename: str) -> Path:
            if explicit:
                p = Path(explicit).expanduser()
                if not p.is_absolute():
                    p = (rd / p).resolve()
            else:
                p = data / filename
            return p.resolve()

        return (
            _path(self.state_energy_path, self.state_energy),
            _path(self.coupling_path, self.coupling_csv),
            _path(self.tuning_path, self.tuning_csv),
            _path(self.modes_path, self.modes),
        )

    def resolve_out_dir(self, run_dir: Optional[Path] = None) -> Path:
        """Resolve output directory to absolute Path."""
        rd = run_dir or _run_dir()
        if self.out_dir:
            p = Path(self.out_dir).expanduser()
            if not p.is_absolute():
                p = (rd / p).resolve()
            return p.resolve()
        return (rd / "results" / "LVC").resolve()


class LVCConfigParser:
    """
    Parses .cfg files and builds LVCCalculation objects.

    Reads [LVC] and [essentials] sections. Paths are resolved relative to the
    run directory (cwd), same as Exe.py.
    """

    def __init__(self, config_file_path: str, run_dir: Optional[Path] = None):
        """
        Args:
            config_file_path: Path to .cfg file (relative to run_dir or absolute).
            run_dir: Base for resolving relative paths. Defaults to cwd.
        """
        self.run_dir = run_dir or _run_dir()
        cfg_path = Path(config_file_path).expanduser()
        if not cfg_path.is_absolute():
            cfg_path = (self.run_dir / cfg_path).resolve()
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")

        self.config_file_path = str(cfg_path)
        cp = ConfigParser()
        cp.read(cfg_path)

        self.config = cp
        self.reader = ConfigReader(cp)
        self.paths = PathManager(self.reader, self.config_file_path)

        lvc_section = None
        for candidate in ("LVC", "lvc"):
            if cp.has_section(candidate):
                lvc_section = candidate
                break
        essentials_section = "essentials" if cp.has_section("essentials") else None
        if lvc_section is None and essentials_section is None:
            raise ValueError(f"Missing [LVC] (or [essentials]) section in config: {cfg_path}")

        self._lvc_section = lvc_section
        self._essentials_section = essentials_section
        self._base = self.run_dir

    def _get_path(self, opt: str, section: Optional[str] = None) -> Optional[str]:
        sec = section or self._lvc_section or self._essentials_section
        if sec is None or not self.config.has_option(sec, opt):
            return None
        raw = self.config.get(sec, opt, fallback="").strip()
        if not raw:
            return ""
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = (self._base / p).resolve()
        return str(p)

    def _get_str(self, opt: str, default: str = "") -> str:
        for sec in (self._lvc_section, self._essentials_section):
            if sec is not None and self.config.has_option(sec, opt):
                return self.config.get(sec, opt, fallback=default).strip()
        return default

    def _get_int(self, opt: str, default: Optional[int] = None) -> Optional[int]:
        for sec in (self._lvc_section, self._essentials_section):
            if sec is not None and self.config.has_option(sec, opt):
                return self.config.getint(sec, opt, fallback=default)
        return default

    def _get_bool(self, opt: str, default: Optional[bool] = None) -> Optional[bool]:
        for sec in (self._lvc_section, self._essentials_section):
            if sec is not None and self.config.has_option(sec, opt):
                return self.config.getboolean(sec, opt, fallback=default)
        return default

    def build_calculation(self) -> LVCCalculation:
        """Build LVCCalculation from the parsed config."""
        calc = LVCCalculation()

        # Paths from [essentials] via PathManager
        if self._essentials_section is not None:
            in_folder = self.paths.get_data_folder_name().rstrip("/")
            if in_folder:
                p = Path(in_folder).expanduser()
                if not p.is_absolute():
                    p = (self._base / p).resolve()
                calc.data_dir = str(p)

            out_folder = self.paths.get_res_folder_name().rstrip("/")
            if out_folder:
                p = Path(out_folder).expanduser()
                if not p.is_absolute():
                    p = (self._base / p).resolve()
                calc.out_dir = str(p)

            max_q = self._get_int("maximum_number_of_vibrational_quanta", 0)
            if max_q:
                calc.order = max_q

            if self.config.has_option(self._essentials_section, "use_sparse"):
                calc.use_sparse = self.config.getboolean(
                    self._essentials_section, "use_sparse", fallback=True
                )

        # [LVC] overrides
        if self._lvc_section is not None:
            if (d := self._get_path("data_dir", self._lvc_section)) is not None:
                calc.data_dir = d
            if (o := self._get_path("out_dir", self._lvc_section) or self._get_path("out-dir", self._lvc_section)) is not None and o:
                calc.out_dir = o

            for opt, attr, legacy in [
                ("state_energy", "state_energy", "epsilon"),
                ("coupling", "coupling_csv", "lambda"),
                ("tuning", "tuning_csv", "kappa"),
                ("modes", "modes", None),
            ]:
                val = self._get_str(opt, getattr(calc, attr)) or (self._get_str(legacy) if legacy else "")
                if val:
                    setattr(calc, attr, val)

            for opt, attr, alts in [
                ("state_energy_path", "state_energy_path", ["state_energy_csv_path", "epsilon_path", "epsilon_csv_path"]),
                ("coupling_path", "coupling_path", ["coupling_csv_path", "lambda_path", "lambda_csv_path"]),
                ("tuning_path", "tuning_path", ["tuning_csv_path", "kappa_path", "kappa_csv_path"]),
                ("modes_path", "modes_path", ["modes_csv_path"]),
            ]:
                val = self._get_path(opt, self._lvc_section)
                for alt in alts:
                    if val is None:
                        val = self._get_path(alt, self._lvc_section)
                if val is not None:
                    setattr(calc, attr, val)

            raw = self._get_str("modes_to_use")
            if raw:
                raw_lower = raw.strip().lower()
                if raw_lower in ("all", "none", ""):
                    calc.modes_to_use = None
                else:
                    try:
                        calc.modes_to_use = [int(x.strip()) for x in raw.split(",") if x.strip()]
                    except ValueError:
                        calc.modes_to_use = None

            if (o := self._get_int("order")) is not None:
                calc.order = o

            if (mq := self._get_int("max_phonon_quanta")) is not None:
                calc.max_phonon_quanta = mq

            raw = self._get_str("num_eigs") or self._get_str("num-eigs")
            if raw is not None:
                raw = raw.strip().lower()
                calc.num_eigs = None if raw in ("", "all", "none") else int(raw) if raw.isdigit() else None

            if (u := self._get_bool("use_sparse")) is not None:
                calc.use_sparse = u

            if self.config.has_option(self._lvc_section, "separator"):
                sep_val = self._get_str("separator")
                calc.separator = sep_val if sep_val else ";"

            for opt, attr in [
                ("dimensionless_coordinates", "dimensionless_coordinates"),
                ("null_point_vib", "null_point_vib"),
                ("save_npz", "save_npz"),
                ("save_csv", "save_csv"),
            ]:
                if (v := self._get_bool(opt)) is not None:
                    setattr(calc, attr, v)

        return calc
