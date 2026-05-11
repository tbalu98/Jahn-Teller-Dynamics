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
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.sparse import diags as sp_diags
from scipy.sparse import dok_matrix

import jahn_teller_dynamics.math.maths as maths
import jahn_teller_dynamics.math.matrix_mechanics as mm


@dataclass(frozen=True)
class ModesFromCsv:
    """
    Phonon mode identifiers and frequencies read from a modes table.

    ``labels[i]`` and ``omegas[i]`` refer to the same mode. For legacy CSV with
    a purely numeric ``mode`` column, ``labels`` are ``q1``, ``q2``, ... in
    **sorted** mode-number order. For a string ``mode`` column (e.g. ``q1``,
    ``q3x``), ``labels`` are those strings in **file order**.
    """

    labels: List[str]
    omegas: List[float]


@dataclass
class CSVReader:
    separator: str = ";"

    @staticmethod
    def detect_separator(path: Union[str, Path]) -> str:
        """Use `;` if it appears more often than `,` in the header line; else `,`."""
        path = Path(path)
        line = path.read_text(encoding="utf-8", errors="replace").split("\n", 1)[0]
        return ";" if line.count(";") > line.count(",") else ","

    def _read_df(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path, sep=self.separator)
        df.columns = [c.strip().lower() for c in df.columns]
        return df

    def read_df_auto(self, path: Union[str, Path]) -> pd.DataFrame:
        """Read CSV using :meth:`detect_separator` (overrides instance ``separator``)."""
        path = str(path)
        sep = self.detect_separator(path)
        df = pd.read_csv(path, sep=sep)
        df.columns = [str(c).strip().lower() for c in df.columns]
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

    def read_diagonal_state_energies(self, path: Union[str, Path]) -> Tuple[List[str], np.ndarray]:
        """
        Diagonal electronic energies: flexible column names.

        Accepts either:

        - ``state``, ``value`` (LVC / legacy epsilon), or
        - ``el_state``, ``energy`` (PVC / dJT trial style; ``el_state`` can be any non-empty label).
        """
        df = self.read_df_auto(path)
        if "el_state" in df.columns and "energy" in df.columns:
            df = df[["el_state", "energy"]].copy()
            df["el_state"] = df["el_state"].astype(str).str.strip()
            if (df["el_state"] == "").any():
                raise ValueError("Diagonal energies CSV contains an empty el_state label")
            if df["el_state"].duplicated().any():
                raise ValueError("Diagonal energies CSV contains duplicate el_state labels")
            df["energy"] = pd.to_numeric(df["energy"], errors="raise")
            # Preserve file order: row number implies internal electronic index.
            states = df["el_state"].tolist()
            values = df["energy"].to_numpy(dtype=float)
            return states, values
        if "state" in df.columns and "value" in df.columns:
            df = df[["state", "value"]].copy()
            df["state"] = df["state"].astype(str).str.strip()
            df["value"] = pd.to_numeric(df["value"], errors="raise")
            states = df["state"].tolist()
            values = df["value"].to_numpy(dtype=float)
            return states, values
        raise ValueError(
            "Expected columns (state, value) or (el_state, energy); got: "
            f"{list(df.columns)}"
        )

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
    def read_modes(
        self, path: str, mode_numbers: Optional[List[int]] = None
    ) -> List[float]:
        """
        Read phonon mode frequencies/energies from a CSV like:

            mode;omega
            1;0.0041
            2;0.0035
            ...

        Args:
            path: Path to modes CSV.
            mode_numbers: Optional list of mode numbers to include (e.g. [1, 3, 5]).
                If None, all modes from the CSV are used.

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

        if mode_numbers is not None:
            allowed = set(int(m) for m in mode_numbers)
            missing = allowed - set(df["mode"])
            if missing:
                raise ValueError(
                    f"Modes {sorted(missing)} not found in CSV (available: {sorted(df['mode'].tolist())})"
                )
            df = df[df["mode"].isin(allowed)].copy()

        df = df.sort_values("mode", ascending=True)
        modes = df["mode"].to_numpy(dtype=int)
        if len(modes) == 0:
            raise ValueError("Modes CSV is empty (or no modes match mode_numbers)")
        if np.any(modes < 1):
            raise ValueError("Modes CSV contains mode < 1 (expected 1-based indices)")

        return df["omega"].to_list()

    def read_modes_flexible(
        self,
        path: Union[str, Path],
        mode_numbers: Optional[Sequence[int]] = None,
    ) -> ModesFromCsv:
        """
        Read ``mode`` and ``omega`` or ``energy`` columns.

        If every ``mode`` cell parses as a (finite) number, the behaviour matches
        legacy: rows are ordered by increasing numeric ``mode``, and quantum
        labels default to ``q1``, ``q2``, ... in that order. The optional
        ``mode_numbers`` filter selects by that numeric **mode** value (e.g. 1, 2, 3).

        If any ``mode`` value is non-numeric (e.g. ``q1``, ``q3x``), the column is
        treated as explicit mode **names**; row **file order** is preserved, and
        ``mode_numbers`` if set filters by 1-based **row index** (1 = first data row
        after the header).

        Returns:
            :class:`ModesFromCsv` with ``labels`` and ``omegas`` of equal length.
        """
        df = self.read_df_auto(path)
        omega_col = "omega" if "omega" in df.columns else "energy"
        if "mode" not in df.columns or omega_col not in df.columns:
            raise ValueError(
                f"Modes CSV must contain 'mode' and 'omega' or 'energy'; got {list(df.columns)}"
            )
        work = df[["mode", omega_col]].copy()
        work.columns = ["mode", "omega"]
        work["omega"] = pd.to_numeric(work["omega"], errors="raise").astype(float)

        mode_num = pd.to_numeric(work["mode"], errors="coerce")
        use_int_mode = bool(mode_num.notna().all())

        if not use_int_mode:
            work["label"] = work["mode"].map(lambda x: str(x).strip())
            if (work["label"] == "").any():
                raise ValueError("Modes CSV has an empty mode name")
            if work["label"].nunique() != len(work):
                raise ValueError("Modes CSV has duplicate mode names")
            work = work.reset_index(drop=True)
            if mode_numbers is not None:
                n = len(work)
                allowed = {int(m) for m in mode_numbers}
                bad = allowed - set(range(1, n + 1))
                if bad:
                    raise ValueError(
                        f"Invalid mode index {sorted(bad)} for named modes (valid 1..{n})"
                    )
                mask = [(i + 1) in allowed for i in range(len(work))]
                work = work[mask]
            if len(work) == 0:
                raise ValueError("Modes CSV is empty (or no modes match mode_numbers)")
            labels = work["label"].tolist()
            omegas = work["omega"].tolist()
            return ModesFromCsv(labels=labels, omegas=omegas)

        work["mode"] = mode_num.astype(int)
        if (mode_num != work["mode"]).any():
            raise ValueError("Modes CSV 'mode' column must contain integer values")

        if mode_numbers is not None:
            allowed = {int(m) for m in mode_numbers}
            missing = allowed - set(work["mode"])
            if missing:
                raise ValueError(
                    f"Modes {sorted(missing)} not found in CSV (available: {sorted(work['mode'].tolist())})"
                )
            work = work[work["mode"].isin(allowed)].copy()

        work = work.sort_values("mode", ascending=True)
        modes = work["mode"].to_numpy(dtype=int)
        if len(modes) == 0:
            raise ValueError("Modes CSV is empty (or no modes match mode_numbers)")
        if np.any(modes < 1):
            raise ValueError("Modes CSV contains mode < 1 (expected 1-based indices)")

        n = len(modes)
        labels = [f"q{i}" for i in range(1, n + 1)]
        omegas = work["omega"].to_list()
        return ModesFromCsv(labels=labels, omegas=omegas)

    def read_pvc_coupling_rows(self, path: Union[str, Path]) -> "pd.DataFrame":
        """
        PVC coupling table: ``el_state_1``, ``el_state_2``, coupling expression, ``coeff``.

        The expression column may be named ``expression`` (preferred), ``polinom``, or
        ``polynomial`` (exactly one of these must be present).
        """
        df = self.read_df_auto(path)
        expr_col = next(
            (c for c in ("expression", "polinom", "polynomial") if c in df.columns),
            "",
        )
        if not expr_col:
            raise ValueError(
                "PVC coupling CSV must contain el_state_1, el_state_2, "
                f"expression or polinom or polynomial, coeff; got {list(df.columns)}"
            )
        required = {"el_state_1", "el_state_2", expr_col, "coeff"}
        if not required.issubset(df.columns):
            raise ValueError(
                "PVC coupling CSV must contain el_state_1, el_state_2, "
                f"a coupling column ({expr_col}), and coeff; got {list(df.columns)}"
            )
        out = df[["el_state_1", "el_state_2", expr_col, "coeff"]].copy()
        out.columns = ["el_state_1", "el_state_2", "polinom", "coeff"]
        out["el_state_1"] = out["el_state_1"].astype(str).str.strip()
        out["el_state_2"] = out["el_state_2"].astype(str).str.strip()
        if (out["el_state_1"] == "").any() or (out["el_state_2"] == "").any():
            raise ValueError("PVC coupling CSV contains empty el_state_1/el_state_2 labels")
        out["coeff"] = pd.to_numeric(out["coeff"], errors="raise").astype(float)
        out["polinom"] = out["polinom"].astype(str).str.strip()
        return out

