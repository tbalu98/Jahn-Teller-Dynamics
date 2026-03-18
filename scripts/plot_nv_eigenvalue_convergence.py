#!/usr/bin/env python3
"""
Plot eigenvalue convergence from NV center LVC results.

Reads eigenvalues from each order_1, order_2, ... folder and plots all 10
lowest eigenvalues vs Hilbert space dimension.

Run from repo root:
    python3 scripts/plot_nv_eigenvalue_convergence.py [results_dir]

If results_dir is omitted, uses results/LVC_model/NV_center_convergence
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

repo_root = Path(__file__).resolve().parents[1]
DEFAULT_DIR = repo_root / "results" / "LVC_model" / "NV_center_convergence"
NUM_EIGS = 10


def read_dimension(out_dir: Path) -> int | None:
    """Read Hilbert space dimension from eigenvectors.csv (number of basis states = rows)."""
    vec_csv = out_dir / "eigenvectors.csv"
    if not vec_csv.exists():
        return None
    with open(vec_csv) as f:
        return sum(1 for _ in f) - 1


def read_eigenvalues(eig_csv: Path, n: int = NUM_EIGS) -> list[float] | None:
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
    parser = argparse.ArgumentParser(description="Plot NV LVC eigenvalue convergence vs dimension")
    parser.add_argument(
        "results_dir",
        nargs="?",
        default=str(DEFAULT_DIR),
        help="Path to NV_center_convergence folder containing order_1, order_2, ...",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Directory not found: {results_dir}")
        return 1

    # Find order_N folders and sort by order
    order_dirs = []
    for d in results_dir.iterdir():
        if d.is_dir() and d.name.startswith("order_"):
            try:
                order = int(d.name.split("_")[1])
                order_dirs.append((order, d))
            except (IndexError, ValueError):
                continue
    order_dirs.sort(key=lambda x: x[0])

    if not order_dirs:
        print(f"No order_N folders found in {results_dir}")
        return 1

    # Collect dimension and eigenvalues for each order
    dimensions = []
    eig_by_state: list[list[float]] = [[] for _ in range(NUM_EIGS)]
    orders_used = []

    for order, out_dir in order_dirs:
        dim = read_dimension(out_dir)
        vals = read_eigenvalues(out_dir / "eigenvalues.csv")
        if dim is None or vals is None:
            print(f"Warning: Skipping order {order} (missing data)")
            continue
        dimensions.append(dim)
        orders_used.append(order)
        for i in range(min(len(vals), NUM_EIGS)):
            eig_by_state[i].append(vals[i])
        # Pad with NaN if fewer than NUM_EIGS
        for i in range(len(vals), NUM_EIGS):
            eig_by_state[i].append(float("nan"))

    dimensions = np.array(dimensions)
    if len(dimensions) == 0:
        print("No valid data found.")
        return 1

    plt.rcParams["font.size"] = 14
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = plt.cm.tab10.colors
    for i in range(NUM_EIGS):
        vals = np.array(eig_by_state[i])
        valid = np.isfinite(vals)
        if np.any(valid):
            ax.plot(
                dimensions[valid], vals[valid], "o-",
                label=rf"$E_{i}$", color=colors[i % len(colors)], markersize=5
            )

    ax.set_xlabel("Hilbert space dimension")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("NV center LVC: eigenvalue convergence vs dimensionality")
    ax.legend(ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)

    if dimensions.max() / max(dimensions.min(), 1) > 100:
        ax.set_xscale("log")

    fig.tight_layout(pad=1.2)
    fig.subplots_adjust(left=0.15, bottom=0.12)

    out_path = results_dir / "eigenvalue_convergence.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close()
    print(f"Plot saved to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
