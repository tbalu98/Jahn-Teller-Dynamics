"""Resolve electron-state labels from PVC CSV inputs to 1-based coupling indices."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Union

from jahn_teller_dynamics.io.file_io.csv_reader import CSVReader


def build_electron_state_to_id(
    labels: Sequence[str], *, one_based: bool = True
) -> Dict[str, int]:
    """
    Map electron-state labels to internal indices.

    Row order in ``electron_energies.csv`` defines the basis order. PVC coupling rows
    use **1-based** indices (``1 … n_orbitals``) by default.
    """
    base = 1 if one_based else 0
    out: Dict[str, int] = {}
    for i, raw in enumerate(labels):
        label = str(raw).strip()
        if label == "":
            raise ValueError("Empty electron-state label in energies CSV")
        if label in out:
            raise ValueError(f"Duplicate electron-state label {label!r} in energies CSV")
        out[label] = i + base
    return out


def resolve_electron_state_label(
    token: object,
    state_to_id: Mapping[str, int],
    *,
    n_orbitals: int,
    allow_numeric_fallback: bool = True,
) -> int:
    """
    Resolve a coupling CSV ``el_state_1`` / ``el_state_2`` token to a 1-based index.

    Lookup order:

    1. Exact label match in ``state_to_id`` (from ``electron_energies.csv`` row order).
    2. If ``allow_numeric_fallback``, interpret as a legacy 1-based integer index.
    """
    text = str(token).strip()
    if text == "":
        raise ValueError("Empty electron-state token in coupling CSV")
    if text in state_to_id:
        return int(state_to_id[text])
    if not allow_numeric_fallback:
        raise ValueError(
            f"Unknown electronic state label {text!r} in coupling CSV. "
            "Use labels defined in electron_energies.csv."
        )
    try:
        idx = int(text)
    except ValueError as exc:
        raise ValueError(
            f"Unknown electronic state label {text!r} in coupling CSV. "
            "Use labels defined in electron_energies.csv or numeric 1-based IDs."
        ) from exc
    if idx < 1 or idx > n_orbitals:
        raise ValueError(
            f"Electron state index {idx} out of range for {n_orbitals} orbital(s)."
        )
    return idx


def electron_labels_in_basis_order(state_to_index: Mapping[str, int]) -> list[str]:
    """Return labels sorted by 0-based basis index (``multi_config_electron.state_to_index``)."""
    return [label for label, _ in sorted(state_to_index.items(), key=lambda kv: kv[1])]


def state_to_id_from_electron_energies_csv(
    path: Union[str, Path],
    *,
    reader: Optional[CSVReader] = None,
) -> Dict[str, int]:
    """Build a 1-based label map from a diagonal or orbital–spin electron energies CSV."""
    from jahn_teller_dynamics.physics.models.orbital_spin_electron import csv_has_spin_column

    csv_reader = reader or CSVReader()
    path = Path(path)
    if csv_has_spin_column(path):
        table = csv_reader.read_orbital_spin_electron_table(path)
        return build_electron_state_to_id(table.orbital_labels)
    labels, _ = csv_reader.read_diagonal_state_energies(path)
    return build_electron_state_to_id(labels)
