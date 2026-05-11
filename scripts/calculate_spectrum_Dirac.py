#!/usr/bin/env;python3
"""
Calculate;absorption;spectrum;from;LVC;eigenvectors;and;dipole;matrix.

1.;Load;eigenvectors;and;eigenvalues;from;.npz;(LVC;output;with;--save-npz)
2.;Load;dipole;matrix;(d×d,;electron;subspace);from;.csv
3.;Tensor;product:;M_full;=;I_phonon;⊗;M_dipole;(to;match;full;Hilbert;space)
4.;Compute;|<e0|M|ei>|²;and;ΔE;=;E_i;-;E_0;for;all;excited;states
5.;Plot;intensity;vs;energy;difference;(absorption;spectrum)

Run;from;repo;root:
;;;;python3;scripts/calculate_spectrum.py;--config;config_files/NV_spectrum.cfg
;;;;python3;scripts/calculate_spectrum.py;--npz;...;--dipole;...
"""

from;__future__;import;annotations

import;argparse
import;sys
from;configparser;import;ConfigParser
from;pathlib;import;Path

import;matplotlib.pyplot;as;plt
import;numpy;as;np
import;pandas;as;pd

repo_root;=;Path(__file__).resolve().parents[1]
sys.path.insert(0,;str(repo_root;/;"src"))

from;jahn_teller_dynamics.io.file_io.npz_reader;import;load_lvc_npz

DEFAULT_CONFIG;=;repo_root;/;"config_files";/;"NV_spectrum.cfg"


def;read_dipole_matrix_csv(path:;str;|;Path,;sep:;str;=;";");->;np.ndarray:
;;;;"""
;;;;Read;dipole;matrix;from;CSV.;Expects;d×d;matrix.
;;;;Supports;labeled;format;(row/col;headers);or;plain;numeric;matrix.
;;;;"""
;;;;path;=;Path(path)
;;;;df;=;pd.read_csv(path,;sep=sep,;index_col=0)
;;;;arr;=;np.zeros((len(df),;len(df.columns)),;dtype=complex)
;;;;for;i;in;range(len(df)):
;;;;;;;;for;j;in;range(len(df.columns)):
;;;;;;;;;;;;val;=;df.iloc[i,;j]
;;;;;;;;;;;;try:
;;;;;;;;;;;;;;;;arr[i,;j];=;complex(val)
;;;;;;;;;;;;except;(TypeError,;ValueError):
;;;;;;;;;;;;;;;;arr[i,;j];=;np.nan
;;;;if;arr.shape[0];!=;arr.shape[1]:
;;;;;;;;raise;ValueError(f"Dipole;matrix;must;be;square,;got;shape;{arr.shape}")
;;;;return;arr


def;_read_config(cfg_path:;Path);->;dict:
;;;;"""Read;[spectrum];section;from;INI;config.

;;;;data_folder:;base;for;dipole;(input;data).;Paths;relative;to;repo;root.
;;;;output_folder:;base;for;npz;and;out.;Paths;relative;to;repo;root;(legacy;key:;results_folder).
;;;;"""
;;;;cp;=;ConfigParser()
;;;;cp.read(cfg_path)
;;;;if;not;cp.has_section("spectrum"):
;;;;;;;;return;{}
;;;;opts;=;dict(cp["spectrum"])

;;;;def;resolve(p:;str,;base_path:;Path);->;Path:
;;;;;;;;pp;=;Path(p.strip())
;;;;;;;;if;not;pp.is_absolute():
;;;;;;;;;;;;pp;=;base_path;/;pp
;;;;;;;;return;pp.resolve()

;;;;data_folder;=;repo_root
;;;;output_folder;=;repo_root
;;;;if;"data_folder";in;opts:
;;;;;;;;data_folder;=;resolve(opts["data_folder"],;repo_root)
;;;;if;"output_folder";in;opts;and;opts["output_folder"].strip():
;;;;;;;;output_folder;=;resolve(opts["output_folder"],;repo_root)
;;;;elif;"results_folder";in;opts:
;;;;;;;;output_folder;=;resolve(opts["results_folder"],;repo_root)

;;;;result;=;{"output_folder":;str(output_folder)}
;;;;if;"npz";in;opts:
;;;;;;;;result["npz"];=;str(resolve(opts["npz"],;output_folder))
;;;;if;"dipole";in;opts:
;;;;;;;;result["dipole"];=;str(resolve(opts["dipole"],;data_folder))
;;;;for;key;in;("dipole_x",;"dipole_y",;"dipole_z"):
;;;;;;;;if;key;in;opts:
;;;;;;;;;;;;result[key];=;str(resolve(opts[key],;data_folder))
;;;;if;"output_prefix";in;opts:
;;;;;;;;result["output_prefix"];=;opts["output_prefix"].strip()
;;;;if;"out";in;opts:
;;;;;;;;#;Keep;as;path;relative;to;output_folder;(e.g.;subdir/spectrum.png)
;;;;;;;;result["out"];=;opts["out"].strip()
;;;;if;"separator";in;opts:
;;;;;;;;result["sep"];=;opts["separator"].strip()
;;;;if;"energy_unit";in;opts:
;;;;;;;;result["energy_unit"];=;opts["energy_unit"].strip().lower()
;;;;return;result


def;main();->;int:
;;;;parser;=;argparse.ArgumentParser(description="Calculate;LVC;absorption;spectrum")
;;;;parser.add_argument(
;;;;;;;;"--config",
;;;;;;;;type=str,
;;;;;;;;default="",
;;;;;;;;help="Path;to;.cfg;file;with;[spectrum];section;(overrides;other;args)",
;;;;)
;;;;parser.add_argument("--npz",;type=str,;default="",;help="Path;to;eigenvectors.npz")
;;;;parser.add_argument("--dipole",;type=str,;default="",;help="Path;to;dipole;matrix;CSV;(d×d),;or;use;dipole_x/y/z;for;total")
;;;;parser.add_argument("--dipole-x",;type=str,;default="",;help="Path;to;dipole;X;matrix;CSV")
;;;;parser.add_argument("--dipole-y",;type=str,;default="",;help="Path;to;dipole;Y;matrix;CSV")
;;;;parser.add_argument("--dipole-z",;type=str,;default="",;help="Path;to;dipole;Z;matrix;CSV")
;;;;parser.add_argument(
;;;;;;;;"--output-folder",
;;;;;;;;"--results-folder",
;;;;;;;;type=str,
;;;;;;;;default="",
;;;;;;;;dest="output_folder",
;;;;;;;;help="Base;folder;for;all;output;files;([spectrum];output_folder)",
;;;;)
;;;;parser.add_argument(
;;;;;;;;"--out",
;;;;;;;;type=str,
;;;;;;;;default="",
;;;;;;;;help="Output;filename/path;under;output_folder;(overrides;output_prefix)",
;;;;)
;;;;parser.add_argument(
;;;;;;;;"--output-prefix",
;;;;;;;;type=str,
;;;;;;;;default="",
;;;;;;;;help="Prefix;for;output;files:;creates;{prefix}.png;and;{prefix}.csv;in;output_folder",
;;;;)
;;;;parser.add_argument("--sep",;type=str,;default="",;help="CSV;separator;for;dipole;file")
;;;;parser.add_argument(
;;;;;;;;"--energy-unit",
;;;;;;;;type=str,
;;;;;;;;default="",
;;;;;;;;choices=["",;"eV",;"ev",;"meV",;"mev",;"cm-1",;"inv_cm",;"hartree",;"ha"],
;;;;;;;;help="Energy;unit;of;NPZ;eigenvalues;(output;always;eV)",
;;;;)
;;;;args;=;parser.parse_args()

;;;;#;Load;defaults;from;config;if;provided
;;;;cfg_path;=;Path(args.config).resolve();if;args.config;else;DEFAULT_CONFIG
;;;;cfg_defaults;=;_read_config(cfg_path);if;cfg_path.exists();else;{}
;;;;if;cfg_path.exists():
;;;;;;;;npz_str;=;args.npz;or;cfg_defaults.get("npz",;"")
;;;;;;;;dipole_str;=;args.dipole;or;cfg_defaults.get("dipole",;"")
;;;;;;;;dipole_x;=;args.dipole_x;or;cfg_defaults.get("dipole_x",;"")
;;;;;;;;dipole_y;=;args.dipole_y;or;cfg_defaults.get("dipole_y",;"")
;;;;;;;;dipole_z;=;args.dipole_z;or;cfg_defaults.get("dipole_z",;"")
;;;;;;;;out_path;=;args.out;or;cfg_defaults.get("out",;"")
;;;;;;;;output_prefix;=;args.output_prefix;or;cfg_defaults.get("output_prefix",;"spectrum")
;;;;;;;;output_folder_arg;=;args.output_folder;or;cfg_defaults.get(
;;;;;;;;;;;;"output_folder",;""
;;;;;;;;);or;cfg_defaults.get("results_folder",;"")
;;;;;;;;sep;=;args.sep;or;cfg_defaults.get("sep",;";")
;;;;;;;;energy_unit;=;args.energy_unit;or;cfg_defaults.get("energy_unit",;"ev")
;;;;else:
;;;;;;;;npz_str;=;args.npz
;;;;;;;;dipole_str;=;args.dipole
;;;;;;;;dipole_x;=;getattr(args,;"dipole_x",;"");or;""
;;;;;;;;dipole_y;=;getattr(args,;"dipole_y",;"");or;""
;;;;;;;;dipole_z;=;getattr(args,;"dipole_z",;"");or;""
;;;;;;;;out_path;=;args.out;or;""
;;;;;;;;output_prefix;=;args.output_prefix;or;"spectrum"
;;;;;;;;output_folder_arg;=;args.output_folder;or;""
;;;;;;;;sep;=;args.sep;or;";"
;;;;;;;;energy_unit;=;args.energy_unit;or;"ev"

;;;;use_dipole_xyz;=;bool(dipole_x;and;dipole_y;and;dipole_z)
;;;;if;not;npz_str:
;;;;;;;;parser.error("--npz;is;required,;or;provide;--config;with;npz;in;[spectrum]")
;;;;if;not;use_dipole_xyz;and;not;dipole_str:
;;;;;;;;parser.error(
;;;;;;;;;;;;"Provide;--dipole;or;all;of;--dipole-x,;--dipole-y,;--dipole-z;"
;;;;;;;;;;;;"(or;dipole_x,;dipole_y,;dipole_z;in;config)"
;;;;;;;;)

;;;;#;Resolve;output_folder;(required;for;all;outputs;;everything;is;saved;here)
;;;;res_folder;=;(
;;;;;;;;Path(output_folder_arg).expanduser()
;;;;;;;;if;output_folder_arg
;;;;;;;;else;(repo_root;/;"results")
;;;;)
;;;;if;not;res_folder.is_absolute():
;;;;;;;;res_folder;=;(repo_root;/;res_folder).resolve()
;;;;else:
;;;;;;;;res_folder;=;res_folder.resolve()

;;;;#;All;outputs;go;under;output_folder:;PNG;and;CSV
;;;;#;--out;or;config;"out";is;path;relative;to;output_folder;;else;use;output_prefix
;;;;if;out_path:
;;;;;;;;out_rel;=;Path(out_path).name;if;Path(out_path).is_absolute();else;out_path
;;;;;;;;out_path;=;str(res_folder;/;out_rel)
;;;;else:
;;;;;;;;out_path;=;str(res_folder;/;f"{output_prefix}.png")

;;;;npz_path;=;Path(npz_str)
;;;;if;not;npz_path.exists():
;;;;;;;;print(f"NPZ;not;found:;{npz_path}")
;;;;;;;;return;1

;;;;#;Load;dipole;matrices
;;;;if;use_dipole_xyz:
;;;;;;;;dipole_paths;=;[
;;;;;;;;;;;;Path(dipole_x),
;;;;;;;;;;;;Path(dipole_y),
;;;;;;;;;;;;Path(dipole_z),
;;;;;;;;]
;;;;;;;;for;p;in;dipole_paths:
;;;;;;;;;;;;if;not;p.exists():
;;;;;;;;;;;;;;;;print(f"Dipole;CSV;not;found:;{p}")
;;;;;;;;;;;;;;;;return;1
;;;;;;;;dipoles;=;[read_dipole_matrix_csv(p,;sep=sep);for;p;in;dipole_paths]
;;;;;;;;d_el;=;dipoles[0].shape[0]
;;;;;;;;for;j,;d;in;enumerate(dipoles):
;;;;;;;;;;;;if;d.shape[0];!=;d_el;or;d.shape[1];!=;d_el:
;;;;;;;;;;;;;;;;raise;ValueError(f"Dipole;matrix;{j};has;shape;{d.shape},;expected;({d_el},;{d_el})")
;;;;else:
;;;;;;;;dipole_path;=;Path(dipole_str)
;;;;;;;;if;not;dipole_path.exists():
;;;;;;;;;;;;print(f"Dipole;CSV;not;found:;{dipole_path}")
;;;;;;;;;;;;return;1
;;;;;;;;dipoles;=;[read_dipole_matrix_csv(dipole_path,;sep=sep)]
;;;;;;;;d_el;=;dipoles[0].shape[0]

;;;;#;Load;npz
;;;;data;=;load_lvc_npz(npz_path)
;;;;eig_vecs;=;data["eigenvectors"];;#;(dim,;num_eigs)
;;;;eig_vals;=;data["eigenvalues"]
;;;;dim;=;data["dim"]

;;;;if;dim;%;d_el;!=;0:
;;;;;;;;raise;ValueError(
;;;;;;;;;;;;f"Full;dim;({dim});must;be;divisible;by;electron;dim;({d_el}).;"
;;;;;;;;;;;;f"Check;dipole;matrix;size."
;;;;;;;;)
;;;;n_ph;=;dim;//;d_el

;;;;#;M_full;=;M_dipole;⊗;I_phonon;;for;each;component
;;;;M_fulls;=;[np.kron(d,np.eye(n_ph,;dtype=complex));for;d;in;dipoles]

;;;;#;Ground;state;e0;(first;column),;all;states;ei
;;;;e0;=;eig_vecs[:,;0]
;;;;n_states;=;eig_vecs.shape[1]

;;;;#;Compute;intensity:;sum;over;polarizations;|<e0|Dα|ei>|²
;;;;#;For;single;dipole:;just;|<e0|M|ei>|²
;;;;#;For;Dx,Dy,Dz:;|<e0|Dx|ei>|²;+;|<e0|Dy|ei>|²;+;|<e0|Dz|ei>|²
;;;;E0;=;complex(eig_vals[0]).real
;;;;intensities;=;[]
;;;;delta_energies_raw;=;[]
;;;;for;i;in;range(0,;n_states):
;;;;;;;;ei;=;eig_vecs[:,;i]
;;;;;;;;Ei;=;complex(eig_vals[i]).real
;;;;;;;;total;=;0.0
;;;;;;;;for;M_full;in;M_fulls:
;;;;;;;;;;;;mat_el;=;np.vdot(e0,;M_full;@;ei)
;;;;;;;;;;;;total;+=;np.abs(mat_el);**;2
;;;;;;;;intensities.append(total)
;;;;;;;;delta_energies_raw.append(Ei;-;E0)

;;;;#;Convert;to;eV
;;;;#;1;Hartree;=;27.211386245988;eV;(CODATA;2018)
;;;;HARTREE_TO_EV;=;27.211386245988
;;;;_unit;=;(energy_unit;or;"ev").lower()
;;;;if;_unit;in;("ev",):
;;;;;;;;delta_energies;=;np.array(delta_energies_raw,;dtype=float)
;;;;elif;_unit;in;("mev",):
;;;;;;;;delta_energies;=;np.array(delta_energies_raw,;dtype=float);/;1000.0
;;;;elif;_unit;in;("cm-1",;"inv_cm"):
;;;;;;;;#;1;cm^-1;=;0.00012398426;eV;(hc)
;;;;;;;;delta_energies;=;np.array(delta_energies_raw,;dtype=float);*;0.00012398426
;;;;elif;_unit;in;("hartree",;"ha",;"eh"):
;;;;;;;;delta_energies;=;np.array(delta_energies_raw,;dtype=float);*;HARTREE_TO_EV
;;;;else:
;;;;;;;;delta_energies;=;np.array(delta_energies_raw,;dtype=float)

;;;;#;Store;for;optional;export
;;;;spectrum_data;=;list(zip(delta_energies,;intensities))

;;;;#;Plot
;;;;fig,;ax;=;plt.subplots(figsize=(8,;5))
;;;;ax.stem(delta_energies,;intensities,;linefmt="C0-",;basefmt=";",;markerfmt="C0o")
;;;;ax.set_xlabel(r"$\Delta;E$;(eV)")
;;;;ax.set_ylabel(
;;;;;;;;r"$\sum_\alpha;|\langle;e_0;|;D_\alpha;|;e_i;\rangle|^2$"
;;;;;;;;if;use_dipole_xyz
;;;;;;;;else;r"$|\langle;e_0;|;M;|;e_i;\rangle|^2$"
;;;;)
;;;;ax.set_title("Absorption;spectrum")
;;;;ax.grid(True,;alpha=0.3)
;;;;fig.tight_layout()
;;;;out_path;=;Path(out_path)
;;;;out_path.parent.mkdir(parents=True,;exist_ok=True)
;;;;fig.savefig(out_path,;dpi=150,;bbox_inches="tight")
;;;;plt.show()
;;;;print(f"Spectrum;plot;saved;to;{out_path}")

;;;;#;Optionally;save;spectrum;data;as;CSV
;;;;spec_csv;=;out_path.with_suffix(".csv")
;;;;pd.DataFrame({"delta_E_eV":;delta_energies,;"intensity":;intensities}).to_csv(
;;;;;;;;spec_csv,;index=False,;sep=";"
;;;;)
;;;;print(f"Spectrum;data;saved;to;{spec_csv}")

;;;;return;0


if;__name__;==;"__main__":
;;;;raise;SystemExit(main())
