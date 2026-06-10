"""
Optional SLEPc / PETSc eigen solver for Hermitian sparse problems on MPI clusters (e.g. ``srun``).

Requires ``petsc4py`` + ``slepc4py`` linked to PETSc/SLEPc (typically built with MPI and a
matching scalar type for the Hamiltonian). PETSc/SLEPc options such as ``-eps_type krylovschur``
may be appended to the launch command ahead of PVC arguments.

Distributed layout: each MPI rank contributes a contiguous block of CSR rows to a PETSc ``Mat``.
Eigenvectors gathered on rank zero via ``Scatter.toZero``, then broadcast to all ranks so
:class:`~jahn_teller_dynamics.math_utils.matrix_mechanics.eigen_vector_space` matches non-MPI callers.

``nev`` is clamped to ``dim - 1`` (when ``dim > 1``) if the request is ``>= dim``, with a
``RuntimeWarning``, so the Hamiltonian is **not** copied to a dense array (cf.
:class:`~jahn_teller_dynamics.math_utils.eigen_solver.DenseEigenSolver` for explicit full dense diagonalization).
Only the trivial ``dim <= 1`` case uses that dense helper.
"""

from __future__ import annotations

import sys
import warnings
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix

import jahn_teller_dynamics.math_utils.maths as maths
from jahn_teller_dynamics.io.utils.timestamp_print import print_ts
from jahn_teller_dynamics.math_utils.eigen_solver import DenseEigenSolver, EigenSolver

_SLEPC_DEFAULT_TOL = 1e-10
_SLEPC_DEFAULT_MAX_IT = 10000
_SLEPC_MIN_NCV = 30
_SLEPC_NCV_FACTOR = 2
from jahn_teller_dynamics.math_utils.matrix_mechanics import (
    DEFAULT_ROUNDING_PRECISION,
    MatrixOperator,
    eigen_vector_space,
    hilber_space_bases,
    ket_vector,
)


def _ensure_petsc_initialized(PETSc) -> None:
    if not PETSc.Sys.isInitialized():
        PETSc.Sys.initialize(sys.argv)


def _scatter_dist_vec_and_bcast_to_all(v_dist, comm):
    """Full-length real numpy vector replicated on every MPI rank from a parallel ``Vec``."""
    from petsc4py import PETSc

    scatter, v_seq = PETSc.Scatter.toZero(v_dist)
    scatter.scatter(v_dist, v_seq, False, PETSc.Scatter.Mode.FORWARD)

    rank = comm.getRank()
    dim = v_dist.getSize()
    chunk = np.empty(dim, dtype=np.float64)
    if rank == 0:
        chunk[:] = np.asarray(v_seq.getArray(readonly=True), dtype=np.float64).flatten()
    size = comm.getSize()
    if size > 1:
        mpi = comm.tompi4py()
        mpi.Bcast(chunk, root=0)
    return chunk


def _matrix_operator_to_csr(matrix_operator: MatrixOperator) -> csr_matrix:
    matrix = matrix_operator.matrix
    if isinstance(matrix, maths.SparseMatrix):
        return matrix.matrix.tocsr()
    if isinstance(matrix, maths.Matrix):
        return csr_matrix(matrix.matrix)
    try:
        return csr_matrix(matrix.matrix)
    except AttributeError as e:
        raise TypeError(
            f"Expected MatrixOperator backing Matrix or SparseMatrix; got {type(matrix)}"
        ) from e


def _csr_row_partition(n: int, rank: int, size: int) -> tuple[int, int]:
    chunk = max(1, (n + max(size, 1) - 1) // max(size, 1))
    row_start = min(rank * chunk, n)
    row_end = min((rank + 1) * chunk, n)
    return row_start, row_end


def _slepc_eigenvalue_real_imag(eps, i: int) -> tuple[float, float]:
    """Handle slepc4py variants: ``getEigenvalue`` may be a scalar, complex, ndarray, or (re, im) pair."""
    v = eps.getEigenvalue(i)
    if isinstance(v, (tuple, list)) and len(v) >= 2:
        return float(v[0]), float(v[1])
    arr = np.asarray(v, dtype=np.complex128).ravel()
    if arr.size == 0:
        return 0.0, 0.0
    z = complex(arr[0])
    return z.real, z.imag


def _slepc_st_shift_invert(st, sigma: float, SLEPc) -> None:
    """Enable shift-and-invert around ``sigma`` on SLEPc spectral transform (PETSc version quirks).

    Prefer ``SINVERT`` (true shift–and–invert); plain ``SHIFT`` is a different ST and pairs poorly
    with ``EPSWhich.TARGET_MAGNITUDE`` on recent SLEPc.
    """
    ST = SLEPc.ST.Type
    candidates: list[object] = []
    for name in ("SINVERT", "SINV"):
        obj = getattr(ST, name, None)
        if obj is not None:
            candidates.append(obj)
    set_ok = False
    for tn in candidates:
        try:
            st.setType(tn)
            set_ok = True
            break
        except Exception:
            continue
    if not set_ok:
        for name in ("sinvert", "shift"):
            try:
                st.setType(name)
                set_ok = True
                break
            except Exception:
                continue
    if not set_ok:
        raise RuntimeError(
            "SLEPc: could not set spectral-transform type for shift–invert (need SINVERT/SHIFT)."
        )
    sig = float(sigma)
    if hasattr(st, "setShift"):
        st.setShift(sig)
        return
    if hasattr(st, "setSigma"):
        st.setSigma(sig)
        return
    raise RuntimeError("SLEPc spectral transform: ST object has no setShift/setSigma")


def _slepc_clear_shift_invert(st, SLEPc) -> None:
    """
    On the existing ``ST``, leave shift-and-invert (e.g. from ``-st_type sinvert`` in PETSc options).
    Does not replace ``eps.setST`` (that breaks ``EPSSetOperators`` state).
    """
    ST = SLEPc.ST.Type
    for name in ("SHIFT", "CAYLEY", "SHELL"):
        tn = getattr(ST, name, None)
        if tn is None:
            continue
        try:
            st.setType(tn)
            if name == "SHIFT" and hasattr(st, "setShift"):
                st.setShift(0.0)
            return
        except Exception:
            continue
    for name in ("shift", "cayley", "shell"):
        try:
            st.setType(name)
            return
        except Exception:
            continue


def _slepc_pick_ncv(nev: int, dim: int) -> int:
    return min(dim, max(_SLEPC_MIN_NCV, _SLEPC_NCV_FACTOR * int(nev) + 1))


def _slepc_configure_spectrum(
    eps,
    spectral_sigma: Optional[float],
    spectral_which: Optional[str],
    ordering_type: Optional[str],
    SLEPc,
) -> None:
    """Map ``spectral_*`` / legacy ``ordering_type`` to ``EPS`` targets and ``ST`` shift–invert."""
    sw_eff = (
        spectral_which
        if (spectral_which is not None and str(spectral_which).strip())
        else ordering_type
    )
    if sw_eff:
        otl = str(sw_eff).strip().upper().replace("-", "_")
        if otl == "SM":
            sw_eff = "smallest_mag"
        elif otl == "SA":
            sw_eff = "smallest_real"
        elif otl == "LA":
            sw_eff = "largest_real"
    s = (str(sw_eff or "")).strip().lower().replace("-", "_")

    if spectral_sigma is not None:
        sigma = float(spectral_sigma)
        # SLEPc 3.23+ rejects shift–and–invert unless ``which`` is target-based (e.g.
        # ``EPS_TARGET_MAGNITUDE``, not ``EPS_LARGEST_MAGNITUDE``); see EPSCheckCompatibleST().
        eps.setTarget(sigma)
        st = eps.getST()
        _slepc_st_shift_invert(st, sigma, SLEPc)
        Which = SLEPc.EPS.Which
        tw_target_mag = getattr(Which, "TARGET_MAGNITUDE", None)
        tw_target_real = getattr(Which, "TARGET_REAL", None)
        s_which = (str(spectral_which or "")).strip().lower().replace("-", "_")
        if tw_target_mag is None and tw_target_real is None:
            eps.setWhichEigenpairs(Which.LARGEST_MAGNITUDE)
        elif s_which in ("target_real", "smallest_distance_real"):
            eps.setWhichEigenpairs(
                tw_target_real if tw_target_real is not None else tw_target_mag
            )
        else:
            # nearest / closest / LM / unset → smallest |lambda - tau|
            eps.setWhichEigenpairs(
                tw_target_mag if tw_target_mag is not None else Which.LARGEST_MAGNITUDE
            )
        return

    if s in ("nearest", "closest", "target"):
        raise ValueError(
            "SLEPc: spectral selection 'nearest' requires eigensolver_sigma (shift–invert target)."
        )

    _slepc_clear_shift_invert(eps.getST(), SLEPc)

    Which = SLEPc.EPS.Which
    if not s or s in ("smallest_real", "smallest_algebraic"):
        eps.setWhichEigenpairs(Which.SMALLEST_REAL)
    elif s in ("largest_real", "largest_algebraic"):
        eps.setWhichEigenpairs(Which.LARGEST_REAL)
    elif s in ("smallest_mag", "sm", "smallest_magnitude"):
        eps.setWhichEigenpairs(Which.SMALLEST_MAGNITUDE)
    elif s in ("largest_mag", "lm", "largest_magnitude"):
        eps.setWhichEigenpairs(Which.LARGEST_MAGNITUDE)
    else:
        ru = (str(sw_eff or "")).strip().upper()
        if ru == "SA":
            eps.setWhichEigenpairs(Which.SMALLEST_REAL)
        elif ru == "LA":
            eps.setWhichEigenpairs(Which.LARGEST_REAL)
        elif ru == "SM":
            eps.setWhichEigenpairs(Which.SMALLEST_MAGNITUDE)
        elif ru == "LM":
            eps.setWhichEigenpairs(Which.LARGEST_MAGNITUDE)
        else:
            eps.setWhichEigenpairs(Which.SMALLEST_REAL)


def _assert_petsc_supports_complex_hamiltonian(A_csr: csr_matrix) -> None:
    """PETSc built with real scalars silently keeps only Re(H) when casting CSR data."""
    from petsc4py import PETSc

    if not np.iscomplexobj(A_csr.data):
        return
    if np.issubdtype(PETSc.ScalarType, np.complexfloating):
        return
    imag = np.abs(A_csr.data.imag)
    scale = np.maximum(1.0, np.abs(A_csr.data.real))
    n_imag = int(np.count_nonzero(imag > 1e-12 * scale))
    if n_imag == 0:
        return
    raise TypeError(
        "SLEPc/PETSc was built with real scalars (PETSc.ScalarType is real), but the "
        f"Hamiltonian has {n_imag}/{A_csr.data.size} CSR entries with significant imaginary "
        "parts. petsc4py will cast those to float64 (real part only), so SLEPc diagonalizes "
        "Re(H), not H — eigenvalues then disagree with scipy.sparse.linalg.eigsh on the full "
        "complex matrix. Rebuild PETSc and SLEPc with --with-scalar-type=complex (or use "
        "eigensolver solver = sparse / dense for this model)."
    )


def _csr_to_mpi_aij(A_csr: csr_matrix, comm):
    from petsc4py import PETSc

    _assert_petsc_supports_complex_hamiltonian(A_csr)

    nrows, ncols = A_csr.shape
    rank = comm.getRank()
    size = comm.getSize()
    r0, r1 = _csr_row_partition(nrows, rank, size)
    dtype = PETSc.ScalarType
    itype = PETSc.IntType
    if r0 >= r1:
        ia = np.asarray([0], dtype=itype)
        ja = np.zeros(0, dtype=itype)
        data = np.zeros(0, dtype=dtype)
    else:
        local = A_csr[r0:r1]
        ia = np.ascontiguousarray(local.indptr.astype(itype))
        ja = np.ascontiguousarray(local.indices.astype(itype))
        data = np.ascontiguousarray(local.data.astype(dtype, copy=False))
    mat = PETSc.Mat().createAIJ(size=(nrows, ncols), csr=(ia, ja, data), comm=comm)
    mat.assemble()
    return mat


class SLEPcEigenSolver(EigenSolver):
    """Hermitian sparse eigenproblem solved with ``slepc4py.SLEPc.EPS`` + PETSc ``Mat``."""

    def __init__(self, default_num_of_vals: int = 100):
        self.default_num_of_vals = default_num_of_vals

    def solve(
        self,
        matrix_operator: MatrixOperator,
        num_of_vals: Optional[int] = None,
        ordering_type: Optional[str] = None,
        quantum_states_bases: Optional[hilber_space_bases] = None,
        spectral_sigma: Optional[float] = None,
        spectral_which: Optional[str] = None,
        *,
        allow_non_hermitian: bool = False,
        tol: Optional[float] = None,
        max_iter: Optional[int] = None,
        ncv: Optional[int] = None,
    ) -> eigen_vector_space:
        try:
            from petsc4py import PETSc
            from slepc4py import SLEPc
        except ImportError as e:
            raise ImportError(
                "eigensolver SLEPc requires petsc4py and slepc4py (MPI-linked PETSc/SLEPc)."
            ) from e

        _ensure_petsc_initialized(PETSc)
        comm = PETSc.COMM_WORLD

        A_csr = _matrix_operator_to_csr(matrix_operator)
        dim = A_csr.shape[0]
        if num_of_vals is None:
            num_of_vals = self.default_num_of_vals
        nev = min(int(num_of_vals), dim)
        if nev <= 0:
            nev = min(dim, max(1, self.default_num_of_vals))

        # Avoid DenseEigenSolver(H).todense(): O(dim²) RAM. Cap iterative nev like scipy eigsh (k < dim).
        if dim <= 1:
            dense_solver = DenseEigenSolver()
            return dense_solver.solve(
                matrix_operator,
                num_of_vals=num_of_vals,
                ordering_type=ordering_type,
                quantum_states_bases=quantum_states_bases,
                spectral_sigma=spectral_sigma,
                spectral_which=spectral_which,
                allow_non_hermitian=allow_non_hermitian,
            )
        if nev >= dim:
            warnings.warn(
                f"SLEPc: requested num_of_vals capped from {nev} to {dim - 1} "
                f"(matrix dim={dim}) to avoid allocating a dense Hamiltonian.",
                RuntimeWarning,
                stacklevel=2,
            )
            nev = dim - 1

        A_mat = _csr_to_mpi_aij(A_csr, comm)

        eps = SLEPc.EPS().create(comm=comm)
        eps.setOperators(A_mat)
        if allow_non_hermitian:
            eps.setProblemType(SLEPc.EPS.ProblemType.NHEP)
        else:
            eps.setProblemType(SLEPc.EPS.ProblemType.HEP)
        try:
            eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
        except Exception:
            try:
                eps.setType("krylovschur")
            except Exception:
                pass

        # ncv: honor an explicit cfg value (clamped to nev < ncv <= dim); else auto-pick.
        if ncv is not None:
            ncv = min(dim, max(int(ncv), nev + 1))
        else:
            ncv = _slepc_pick_ncv(nev, dim)
        try:
            eps.setDimensions(nev=nev, ncv=ncv)
        except TypeError:
            eps.setDimensions(nev=nev)

        # tol / max_iter: honor explicit cfg values, else fall back to defaults.
        eff_tol = float(tol) if tol is not None else _SLEPC_DEFAULT_TOL
        eff_max_it = int(max_iter) if max_iter is not None else _SLEPC_DEFAULT_MAX_IT
        eps.setTolerances(eff_tol, eff_max_it)
        # Do not use eps.setMonitor(True): slepc4py expects a callable, not a bool.

        eps.setFromOptions()
        # Apply spectrum *after* setFromOptions so defaults / CLI do not wipe target shift–invert
        # settings (EPSWhich.TARGET_MAGNITUDE + EPSSetTarget), required since SLEPc ~3.23.
        _slepc_configure_spectrum(
            eps, spectral_sigma, spectral_which, ordering_type, SLEPc
        )
        if comm.getRank() == 0:
            sw = (spectral_which or "").strip() or "smallest_real"
            problem = "NHEP" if allow_non_hermitian else "HEP"
            print_ts(
                f"  → SLEPc: dim={dim}, nev={nev}, ncv={ncv}, "
                f"tol={eff_tol}, max_it={eff_max_it}, "
                f"problem={problem}, spectral_which={sw!r}",
                flush=True,
            )
        eps.solve()

        nconv = eps.getConverged()
        if nconv <= 0:
            raise RuntimeError(
                "SLEPc EPS: no eigenpairs converged (check Hermiticity, nev, and -eps_* options)."
            )
        if nconv < nev:
            warnings.warn(
                f"SLEPc: only {nconv}/{nev} eigenpairs converged.",
                RuntimeWarning,
                stacklevel=2,
            )
        if comm.getRank() == 0:
            print_ts(f"  → SLEPc: nconv={nconv}/{nev}", flush=True)

        take = min(int(nconv), int(nev))

        eigen_vals: list[complex] = []
        cols: list[np.ndarray] = []
        for i in range(take):
            lam_r, lam_i = _slepc_eigenvalue_real_imag(eps, i)
            lam = complex(lam_r, lam_i)
            if not allow_non_hermitian and abs(lam_i) > 1e-8 * max(1.0, abs(lam_r)):
                raise NotImplementedError(
                    "Non-real eigenvalues from SLEPc HEP — matrix may not be Hermitian in PETSc. "
                    "Set require_hermitian=false in the PVC config to use the NHEP solver."
                )
            eigen_vals.append(lam)
            vr, vi = A_mat.createVecs()
            eps.getEigenpair(i, vr, vi)
            pr = _scatter_dist_vec_and_bcast_to_all(vr, comm)
            pi = _scatter_dist_vec_and_bcast_to_all(vi, comm)
            full = (pr + 1j * pi).astype(np.complex128, copy=False)
            cols.append(full)

        eig_mat = np.column_stack(cols)
        eig_arr = np.array(eigen_vals, dtype=np.complex128)

        order = np.lexsort((eig_arr.imag, eig_arr.real))
        eig_arr = eig_arr[order]
        eig_mat = eig_mat[:, order]

        eigen_kets = []
        for j in range(eig_mat.shape[1]):
            col = eig_mat[:, j]
            col_vec = maths.col_vector(
                np.transpose(
                    np.round(
                        np.matrix([col]),
                        DEFAULT_ROUNDING_PRECISION,
                    )
                )
            )
            ev = complex(eig_arr[j]) if allow_non_hermitian else round(float(eig_arr[j].real), DEFAULT_ROUNDING_PRECISION)
            eigen_kets.append(ket_vector(col_vec, ev))

        basis = quantum_states_bases
        if basis is None:
            basis = matrix_operator.quantum_state_bases

        return eigen_vector_space(basis, eigen_kets)
