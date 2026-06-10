"""
Orbital–spin electronic subsystem for PVC / DJT polynomial coupling.

CSV columns: ``el_state``, ``energy``, ``spin`` (total spin quantum number *S*, e.g. ``1/2``).

Each orbital row defines :math:`2S+1` basis kets with :math:`m_s = -S, -S+1, \\ldots, S`.
The full Hilbert dimension is :math:`N = \\sum_k (2S_k+1)`.

Coupling CSV ``el_state_1`` / ``el_state_2`` still refer to **orbital** indices (1-based);
:func:`~jahn_teller_dynamics.physics.hamiltonians.djt_polynomial_hamiltonian.electron_coupling_matrix_operator`
embeds :math:`|i\\rangle\\langle j| \\otimes I_{\\mathrm{spin}}` when all orbitals share the
same *S* (so ``n_{s,i} = n_{s,j}`); only ``(m_s, m_s)`` pairs are set in that block.
Different *S* per orbital (e.g. ``1/2`` coupled to ``1``) is rejected — that needs a full
spin-coupling matrix, not ``I_{\\mathrm{spin}}``.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix, diags as sp_diags

import jahn_teller_dynamics.math_utils.maths as maths
import jahn_teller_dynamics.math_utils.matrix_mechanics as mm
import jahn_teller_dynamics.physics.quantum_system as qs
from jahn_teller_dynamics.io.file_io.csv_reader import CSVReader


def parse_spin_quantum_number(raw: object) -> float:
    """
    Parse a spin column value (``1/2``, ``0.5``, ``1``, …) to total spin *S*.
    """
    text = str(raw).strip()
    if not text:
        raise ValueError("Empty spin quantum number")
    if "/" in text:
        num, den = text.split("/", 1)
        return float(Fraction(int(num.strip()), int(den.strip())))
    return float(text)


def spin_state_count(S: float) -> int:
    """Number of :math:`m_s` levels for total spin *S* (integer or half-integer): ``2S+1``."""
    twice = round(2 * S)
    if abs(2 * S - twice) > 1e-9:
        raise ValueError(f"Spin S={S!r} must be integer or half-integer")
    return int(twice + 1)


def ms_values_for_spin(S: float) -> List[float]:
    """Projection quantum numbers from :math:`m_s=-S` to :math:`m_s=+S` in steps of 1."""
    n = spin_state_count(S)
    return [-S + k for k in range(n)]


def format_ms_label(ms: float) -> str:
    """Format :math:`m_s` for basis labels (e.g. ``+1/2``, ``-1``)."""
    if abs(ms - round(ms)) < 1e-12:
        v = int(round(ms))
        return f"+{v}" if v > 0 else str(v)
    frac = Fraction(ms).limit_denominator(100)
    num, den = frac.numerator, frac.denominator
    if num > 0:
        return f"+{num}/{den}" if den != 1 else f"+{num}"
    return f"{num}/{den}" if den != 1 else str(num)


@dataclass(frozen=True)
class OrbitalSpinLayout:
    """
    Indexing of the orbital–spin product basis.

    Attributes:
        n_orbitals: Number of orbital rows in the CSV.
        dim: Total Hilbert dimension ``N``.
        offsets: Start index in the full basis for each orbital (0-based orbital index).
        n_spin: List of spin-state counts per orbital.
        spin_S: Total spin *S* per orbital.
    """

    n_orbitals: int
    dim: int
    offsets: Tuple[int, ...]
    n_spin: Tuple[int, ...]
    spin_S: Tuple[float, ...]
    orbital_labels: Tuple[str, ...]

    def orbital_index_valid(self, orbital_index_0: int) -> bool:
        return 0 <= orbital_index_0 < self.n_orbitals

    def require_uniform_spin(self) -> None:
        """Raise if orbitals do not all share the same total spin *S* (same ``n_spin``)."""
        if len(self.spin_S) < 2:
            return
        s0 = self.spin_S[0]
        if any(abs(s - s0) > 1e-9 for s in self.spin_S[1:]):
            detail = ", ".join(
                f"{lab}: S={s}" for lab, s in zip(self.orbital_labels, self.spin_S)
            )
            raise ValueError(
                "Orbital–spin PVC expects the same spin quantum number S on every "
                f"el_state row (so coupling is |i><j| ⊗ I_spin). Got: {detail}"
            )


def build_orbital_spin_layout(
    orbital_labels: Sequence[str],
    spin_S: Sequence[float],
) -> OrbitalSpinLayout:
    """Build cumulative offsets from per-orbital spin multiplicities."""
    labels = tuple(str(s).strip() for s in orbital_labels)
    spins = tuple(float(s) for s in spin_S)
    if len(labels) != len(spins):
        raise ValueError("orbital_labels and spin_S must have the same length")
    n_spin = tuple(spin_state_count(s) for s in spins)
    offsets: List[int] = []
    total = 0
    for ns in n_spin:
        offsets.append(total)
        total += ns
    return OrbitalSpinLayout(
        n_orbitals=len(labels),
        dim=total,
        offsets=tuple(offsets),
        n_spin=n_spin,
        spin_S=spins,
        orbital_labels=labels,
    )


def orbital_spin_basis_qm_nums(
    orbital_labels: Sequence[str],
    spin_S: Sequence[float],
) -> Tuple[List[List[Union[str, float]]], OrbitalSpinLayout]:
    """
    Full list of ``[el_state, m_s]`` quantum numbers and the layout metadata.
    """
    layout = build_orbital_spin_layout(orbital_labels, spin_S)
    qm_list: List[List[Union[str, float]]] = []
    for o_idx, label in enumerate(layout.orbital_labels):
        for ms in ms_values_for_spin(layout.spin_S[o_idx]):
            qm_list.append([label, format_ms_label(ms)])
    if len(qm_list) != layout.dim:
        raise RuntimeError("orbital–spin basis size mismatch")
    return qm_list, layout


def orbital_spin_electron_matrix_operator(
    layout: OrbitalSpinLayout,
    orbital_i: int,
    orbital_j: int,
    *,
    use_sparse: bool,
) -> mm.MatrixOperator:
    """
    Coupling on orbital indices ``(i, j)`` (0-based): spin-conserving block
    ``|orb_i, m_s\\rangle\\langle orb_j, m_s|`` (diagonal in the local ``m_s`` block),
    i.e. the same complex-orbital matrix element on each spin component — equivalent to
    ``|i\\rangle\\langle j| \\otimes I_{\\mathrm{spin}}`` embedded in the ``N×N`` space.

    Requires ``n_spin[i] == n_spin[j]`` (same *S* on both orbitals). Mismatched spin
    (e.g. ``S=1/2`` on one row and ``S=1`` on another) has no meaning for this embedding.
    """
    if not layout.orbital_index_valid(orbital_i) or not layout.orbital_index_valid(orbital_j):
        raise ValueError(
            f"Orbital indices ({orbital_i},{orbital_j}) out of range for "
            f"n_orbitals={layout.n_orbitals}"
        )
    n_i = layout.n_spin[orbital_i]
    n_j = layout.n_spin[orbital_j]
    if n_i != n_j:
        li = layout.orbital_labels[orbital_i]
        lj = layout.orbital_labels[orbital_j]
        si = layout.spin_S[orbital_i]
        sj = layout.spin_S[orbital_j]
        raise ValueError(
            f"Orbital coupling ({li}, {lj}) needs equal spin dimension (|i><j| ⊗ I_spin), "
            f"but S={si} (n={n_i}) vs S={sj} (n={n_j}). Use the same spin column on all "
            "el_state rows, or build a custom spin–orbital matrix via spin_orbital_operator."
        )
    row0 = layout.offsets[orbital_i]
    col0 = layout.offsets[orbital_j]
    n_tot = layout.dim
    n_s = n_i

    if use_sparse:
        rows_i, cols, data = [], [], []
        val = 1.0 + 0.0j
        for di in range(n_s):
            rows_i.append(row0 + di)
            cols.append(col0 + di)
            data.append(val)
        if not data:
            sp = csr_matrix((n_tot, n_tot), dtype=maths.complex_number_typ)
        else:
            sp = csr_matrix(
                (data, (rows_i, cols)),
                shape=(n_tot, n_tot),
                dtype=maths.complex_number_typ,
            )
        return mm.MatrixOperator(maths.SparseMatrix(sp))

    arr = np.zeros((n_tot, n_tot), dtype=np.complex128)
    for di in range(n_s):
        arr[row0 + di, col0 + di] = 1.0
    return mm.MatrixOperator(maths.Matrix(np.asmatrix(arr)))


def orbital_spin_coupling_block_with_coeff(
    layout: OrbitalSpinLayout,
    orbital_i: int,
    orbital_j: int,
    coeff: complex,
    *,
    use_sparse: bool,
) -> mm.MatrixOperator:
    """Scaled orbital–spin coupling block ``coeff * O_{ij}``."""
    op = orbital_spin_electron_matrix_operator(
        layout, orbital_i, orbital_j, use_sparse=use_sparse
    )
    if coeff == 1.0:
        return op
    if coeff == 0.0:
        matrix_type = maths.SparseMatrix if use_sparse else maths.Matrix
        return mm.MatrixOperator.create_null_matrix_op(layout.dim, matrix_type=matrix_type)
    return coeff * op


@dataclass
class multi_config_orbital_spin_electron:
    """
    Electronic subsystem with orbital × spin-resolved basis from CSV.

    ``node.base_states`` uses quantum numbers ``el_state`` and ``m_s``.
    ``layout`` describes embedding of orbital-indexed coupling blocks.
    """

    node: qs.quantum_system_node
    layout: OrbitalSpinLayout
    orbital_label_to_id: Dict[str, int]

    @property
    def state_to_index(self) -> Dict[str, int]:
        """1-based orbital id by label (for coupling CSV resolution)."""
        return self.orbital_label_to_id

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        *,
        subsystem_id: str = "orbital_spin_system",
        separator: str = ";",
        use_sparse: bool = True,
    ) -> "multi_config_orbital_spin_electron":
        reader = CSVReader(separator=separator)
        table = reader.read_orbital_spin_electron_table(csv_path)
        return cls.from_table(table, subsystem_id=subsystem_id, use_sparse=use_sparse)

    @classmethod
    def from_table(
        cls,
        table: "OrbitalSpinElectronTable",
        *,
        subsystem_id: str = "orbital_spin_system",
        use_sparse: bool = True,
    ) -> "multi_config_orbital_spin_electron":
        qm_list, layout = orbital_spin_basis_qm_nums(table.orbital_labels, table.spin_S)
        layout.require_uniform_spin()
        bases = mm.hilber_space_bases().from_qm_nums_list(
            qm_list,
            qm_nums_names=["el_state", "m_s"],
        )
        if len(table.energies) != layout.dim:
            raise ValueError(
                f"Energy vector length ({len(table.energies)}) != basis dim ({layout.dim})"
            )
        diag = table.energies.astype(maths.complex_number_typ)
        if use_sparse:
            sp = sp_diags(diag, offsets=0, shape=(layout.dim, layout.dim), format="csr")
            state_energy_op = mm.MatrixOperator(maths.SparseMatrix(sp))
        else:
            state_energy_op = mm.MatrixOperator(
                maths.Matrix(np.asmatrix(np.diag(diag)))
            )
        state_energy_op.subsys_name = subsystem_id

        node = qs.quantum_system_node(
            subsystem_id,
            base_states=bases,
            operators={
                "state_energy": state_energy_op,
                "epsilon": state_energy_op,
            },
            dim=layout.dim,
        )
        node.use_sparse = use_sparse

        orbital_label_to_id = {
            label: i + 1 for i, label in enumerate(layout.orbital_labels)
        }
        return cls(
            node=node,
            layout=layout,
            orbital_label_to_id=orbital_label_to_id,
        )


@dataclass
class OrbitalSpinElectronTable:
    """Parsed orbital–spin electron CSV (one row per orbital)."""

    orbital_labels: List[str]
    spin_S: List[float]
    energies: np.ndarray


def load_spin_orbital_operator_from_csv(
    path: Union[str, Path],
    dim: int,
    *,
    separator: str = ";",
    use_sparse: bool = True,
    operator_name: str = "spin_orbital",
) -> mm.MatrixOperator:
    """
    Load an ``dim × dim`` spin–orbital operator matrix from a dense square CSV.

    Row/column order must match the orbital–spin basis from
    :func:`orbital_spin_basis_qm_nums` (orbital CSV row order, then :math:`m_s` ascending).
    """
    reader = CSVReader(separator=separator)
    arr = reader.read_square_matrix_csv(path, expected_dim=dim)
    if use_sparse:
        sp = csr_matrix(arr, dtype=maths.complex_number_typ)
        op = mm.MatrixOperator(maths.SparseMatrix(sp))
    else:
        op = mm.MatrixOperator(maths.Matrix(np.asmatrix(arr)))
    op.subsys_name = operator_name
    return op


def csv_has_spin_column(path: Union[str, Path]) -> bool:
    """True if the electron energies CSV includes a ``spin`` column."""
    reader = CSVReader()
    df = reader.read_df_auto(str(path))
    return "spin" in df.columns
