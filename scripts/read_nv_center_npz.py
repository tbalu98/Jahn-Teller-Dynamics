#!/usr/bin/env python3
"""
Read LVC NPZ data from results/LVC/NV_center/eigenvectors.npz.

Uses load_lvc_npz to load the compressed eigenvectors and eigenvalues,
then prints a summary of the data.

Run from repo root:
    python3 scripts/read_nv_center_npz.py [npz_path]

If npz_path is omitted, uses results/LVC/NV_center/eigenvectors.npz
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from jahn_teller_dynamics.io.file_io.npz_reader import load_lvc_npz

DEFAULT_NPZ = repo_root / "results" / "LVC" / "NV_center" / "eigenvectors.npz"


def main() -> int:
    npz_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_NPZ
    if not npz_path.exists():
        print(f"NPZ file not found: {npz_path}")
        return 1

    data = load_lvc_npz(npz_path)
    print(f"Loaded: {npz_path}")
    print(f"  dimension: {data['dim']}")
    print(f"  order: {data['order']}")
    print(f"  eigenvectors shape: {data['eigenvectors'].shape}")
    print(f"  eigenvalues shape: {data['eigenvalues'].shape}")
    print(f"  basis_labels: {len(data['basis_labels'])} states")
    print()
    print("Lowest 10 eigenvalues:")
    for i, ev in enumerate(data["eigenvalues"][:10]):
        print(f"  E[{i}] = {complex(ev):.10g}")
    if len(data["eigenvalues"]) > 10:
        print(f"  ... and {len(data['eigenvalues']) - 10} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
