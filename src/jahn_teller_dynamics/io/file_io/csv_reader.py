"""
CSVReader: dedicated CSV parsing utilities (paired with CSVWriter).

This class centralizes all CSV parsing logic so model classes (e.g.
``multi_config_electron``) can stay focused on representing subsystems and
storing operators, not on reading file formats.

Supported CSV formats (semicolon separated by default):

1) Epsilon (diagonal energies)
    state;value

2) Lambda / coupling (generally off-diagonal)
    state_i;state_j;vibrational_mode;value

3) Kappa / non-coupling (diagonal, mode dependent)
    state;vibrational_mode;value

4) Phonon modes
    mode;omega
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import diags as sp_diags
from scipy.sparse import dok_matrix

import jahn_teller_dynamics.math.maths as maths
import jahn_teller_dynamics.math.matrix_mechanics as mm


@dataclass
class CSVReader:
    separator: str = ";"

    def _read_df(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path, sep=self.separator)
        df.columns = [c.strip().lower() for c in df.columns]
        return df

    # ------------------------------------------------------------------
    # Epsilon
    # ------------------------------------------------------------------
    def read_epsilon(self, path: str) -> Tuple[List[str], np.ndarray]:
        df = self._read_df(path)
        if "state" not in df.columns or "value" not in df.columns:
            raise ValueError("Epsilon CSV must contain columns: 'state' and 'value'")
        df = df[["state", "value"]].copy()
        df["state"] = df["state"].astype(str).str.strip()
        df["value"] = pd.to_numeric(df["value"], errors="raise")
        states = df["state"].tolist()
        values = df["value"].to_numpy(dtype=float)
        return states, values

    def build_epsilon_operator(
        self,
        *,
        values: np.ndarray,
        dim: int,
        use_sparse: bool = True,
    ) -> mm.MatrixOperator:
        if len(values) != dim:
            raise ValueError(f"epsilon values length ({len(values)}) does not match dim ({dim})")
        diag = values.astype(maths.complex_number_typ)
        if use_sparse:
            sp = sp_diags(diag, offsets=0, shape=(dim, dim), format="csr")
            return mm.MatrixOperator(maths.SparseMatrix(sp))
        mx = np.matrix(np.diag(diag))
        return mm.MatrixOperator(maths.Matrix(mx))

    # ------------------------------------------------------------------
    # Coupling (lambda)
    # ------------------------------------------------------------------
    def read_coupling_rows(self, path: str) -> pd.DataFrame:
        df = self._read_df(path)
        required = {"state_i", "state_j", "vibrational_mode", "value"}
        if not required.issubset(set(df.columns)):
            raise ValueError(f"Coupling CSV must contain columns: {sorted(required)}")
        df = df[list(required)].copy()
        df["state_i"] = df["state_i"].astype(str).str.strip()
        df["state_j"] = df["state_j"].astype(str).str.strip()
        df["vibrational_mode"] = pd.to_numeric(df["vibrational_mode"], errors="raise").astype(int)
        df["value"] = pd.to_numeric(df["value"], errors="raise")
        return df

    def build_coupling_operator(
        self,
        *,
        df: pd.DataFrame,
        state_to_index: Dict[str, int],
        dim: int,
        mode_num: int,
        symmetric: bool = True,
        use_sparse: bool = True,
    ) -> mm.MatrixOperator:
        dfm = df[df["vibrational_mode"] == int(mode_num)]
        # If the input CSV already contains both (i,j) and (j,i) entries, and we also
        # symmetrize by adding the mirrored element, we'd double-count. To make
        # `symmetric=True` idempotent, only mirror when the reverse directed pair is
        # NOT explicitly present in the input for this mode.
        provided_pairs: set[tuple[int, int]] = set()
        for row in dfm.itertuples(index=False):
            si = getattr(row, "state_i")
            sj = getattr(row, "state_j")
            if si not in state_to_index or sj not in state_to_index:
                raise ValueError(f"Unknown state in coupling CSV: {si} or {sj}")
            provided_pairs.add((state_to_index[si], state_to_index[sj]))

        if use_sparse:
            mat = dok_matrix((dim, dim), dtype=maths.complex_number_typ)
            for row in dfm.itertuples(index=False):
                si = getattr(row, "state_i")
                sj = getattr(row, "state_j")
                val = getattr(row, "value")
                if si not in state_to_index or sj not in state_to_index:
                    raise ValueError(f"Unknown state in coupling CSV: {si} or {sj}")
                i = state_to_index[si]
                j = state_to_index[sj]
                mat[i, j] = mat.get((i, j), 0) + maths.complex_number_typ(val)
                if symmetric and i != j and (j, i) not in provided_pairs:
                    mat[j, i] = mat.get((j, i), 0) + maths.complex_number_typ(val)
            return mm.MatrixOperator(maths.SparseMatrix(mat.tocsr()))

        mx = np.zeros((dim, dim), dtype=maths.complex_number_typ)
        for row in dfm.itertuples(index=False):
            si = getattr(row, "state_i")
            sj = getattr(row, "state_j")
            val = getattr(row, "value")
            if si not in state_to_index or sj not in state_to_index:
                raise ValueError(f"Unknown state in coupling CSV: {si} or {sj}")
            i = state_to_index[si]
            j = state_to_index[sj]
            mx[i, j] += maths.complex_number_typ(val)
            if symmetric and i != j and (j, i) not in provided_pairs:
                mx[j, i] += maths.complex_number_typ(val)
        return mm.MatrixOperator(maths.Matrix(np.matrix(mx)))

    # ------------------------------------------------------------------
    # Non-coupling (kappa)
    # ------------------------------------------------------------------
    def read_kappa_rows(self, path: str) -> pd.DataFrame:
        df = self._read_df(path)
        required = {"state", "vibrational_mode", "value"}
        if not required.issubset(set(df.columns)):
            raise ValueError(f"Kappa CSV must contain columns: {sorted(required)}")
        df = df[["state", "vibrational_mode", "value"]].copy()
        df["state"] = df["state"].astype(str).str.strip()
        df["vibrational_mode"] = pd.to_numeric(df["vibrational_mode"], errors="raise").astype(int)
        df["value"] = pd.to_numeric(df["value"], errors="raise")
        return df

    def build_kappa_operator(
        self,
        *,
        df: pd.DataFrame,
        state_to_index: Dict[str, int],
        dim: int,
        mode_num: int,
        use_sparse: bool = True,
    ) -> mm.MatrixOperator:
        dfm = df[df["vibrational_mode"] == int(mode_num)]
        diag = np.zeros(dim, dtype=maths.complex_number_typ)
        for row in dfm.itertuples(index=False):
            s = getattr(row, "state")
            val = getattr(row, "value")
            if s not in state_to_index:
                raise ValueError(f"Unknown state in kappa CSV: {s}")
            i = state_to_index[s]
            diag[i] = diag[i] + maths.complex_number_typ(val)

        if use_sparse:
            sp = sp_diags(diag, offsets=0, shape=(dim, dim), format="csr")
            return mm.MatrixOperator(maths.SparseMatrix(sp))
        return mm.MatrixOperator(maths.Matrix(np.matrix(np.diag(diag))))

    # ------------------------------------------------------------------
    # Phonon modes
    # ------------------------------------------------------------------
    def read_modes(self, path: str) -> List[float]:
        """
        Read phonon mode frequencies/energies from a CSV like:

            mode;omega
            1;0.0041
            2;0.0035
            ...

        Returns:
            A list of omegas ordered by increasing `mode`.
        """
        df = self._read_df(path)
        required = {"mode", "omega"}
        if not required.issubset(set(df.columns)):
            raise ValueError(f"Modes CSV must contain columns: {sorted(required)}")
        df = df[["mode", "omega"]].copy()
        df["mode"] = pd.to_numeric(df["mode"], errors="raise").astype(int)
        df["omega"] = pd.to_numeric(df["omega"], errors="raise").astype(float)

        df = df.sort_values("mode", ascending=True)
        modes = df["mode"].to_numpy(dtype=int)
        if len(modes) == 0:
            raise ValueError("Modes CSV is empty")
        if np.any(modes < 1):
            raise ValueError("Modes CSV contains mode < 1 (expected 1-based indices)")

        return df["omega"].to_list()

