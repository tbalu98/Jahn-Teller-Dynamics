#!/usr/bin/env python3
"""
Convergence test of LVC eigenvalues vs maximum_number_of_vibrational_quanta for NV center.

Uses NV_center.cfg as base config, varies the phonon truncation order from 1 to 8,
saves outputs in separate folders, and collects the lowest 4 eigenvalues to CSV for plotting.

Run from repo root:
    python3 scripts/nv_lvc_convergence_test.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

# Ensure src is on path when run from repo root
repo_root = Path(__file__).resolve().parents[1]
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

BASE_CONFIG = repo_root / "work_configfiles" / "3C_diamond_defects" / "NV_defect" / "NV_center.cfg"
OUT_BASE = repo_root / "results" / "LVC_model" / "NV_center_convergence"
SUMMARY_CSV = OUT_BASE / "eigenvalue_convergence.csv"
NUM_EIGS_TO_COLLECT = 4


def run_lvc_for_order(order: int) -> Path | None:
    """
    Run LVC for a given order. Output goes to OUT_BASE/order_N/.
    Returns the output directory path, or None on failure.
    """
    out_dir = OUT_BASE / f"order_{order}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build argv to override order and out_dir
    argv = [
        "--config", str(BASE_CONFIG),
        "--order", str(order),
        "--out-dir", str(out_dir),
        "--num-eigs", str(max(10, NUM_EIGS_TO_COLLECT)),
    ]
    try:
        from jahn_teller_dynamics.LVC import main as lvc_main
        exit_code = lvc_main(argv)
        return out_dir if exit_code == 0 else None
    except Exception as e:
        print(f"  Order {order}: Error - {e}")
        return None


def read_dimension(out_dir: Path) -> int | None:
    """Read Hilbert space dimension from eigenvectors.csv (number of basis states = rows)."""
    vec_csv = out_dir / "eigenvectors.csv"
    if not vec_csv.exists():
        return None
    with open(vec_csv) as f:
        return sum(1 for _ in f) - 1  # minus header


def read_lowest_eigenvalues(eig_csv: Path, n: int = NUM_EIGS_TO_COLLECT) -> list[float] | None:
    """Read the lowest n eigenvalues from LVC eigenvalues.csv (separator ';')."""
    if not eig_csv.exists():
        return None
    vals = []
    with open(eig_csv, newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            try:
                vals.append(float(row["eigenenergy"]))
            except (KeyError, ValueError):
                continue
            if len(vals) >= n:
                break
    return vals[:n] if len(vals) >= n else None


def main() -> int:
    if not BASE_CONFIG.exists():
        print(f"Config not found: {BASE_CONFIG}")
        return 1

    orders = list(range(1, 9))  # 1 to 8 inclusive
    OUT_BASE.mkdir(parents=True, exist_ok=True)

    print("Running NV center LVC convergence test (order 1 to 8)...")
    for order in orders:
        print(f"  Order {order}...", end=" ", flush=True)
        out_dir = run_lvc_for_order(order)
        if out_dir is not None:
            print(f"-> {out_dir}")
        else:
            print("FAILED")

    # Collect dimension and lowest 4 eigenvalues from each order
    rows: list[dict[str, str | float | int]] = []
    for order in orders:
        out_dir = OUT_BASE / f"order_{order}"
        eig_csv = out_dir / "eigenvalues.csv"
        dim = read_dimension(out_dir)
        vals = read_lowest_eigenvalues(eig_csv)
        if vals is None:
            print(f"Warning: Could not read eigenvalues for order {order}")
            rows.append({"order": order, "dimension": dim or "", "E0": "", "E1": "", "E2": "", "E3": ""})
        else:
            rows.append({
                "order": order,
                "dimension": dim if dim is not None else "",
                "E0": vals[0],
                "E1": vals[1],
                "E2": vals[2],
                "E3": vals[3],
            })

    # Write summary CSV
    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["order", "dimension", "E0", "E1", "E2", "E3"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSummary saved to {SUMMARY_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
