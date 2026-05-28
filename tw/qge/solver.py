"""Outer fsolve wrapper. Port of ``solve_model.m`` + Table N main loop.

The original MATLAB code packs the unknowns into a single vector
``V = [what(1..N), dL(1..N-1)]`` (since dL[N]=1 is fixed) and asks
``fsolve`` to drive these residuals to zero:

1. ``what1(1..N-1) - what(1..N-1)``  (excluding ROW)
2. ``sum(Inew) - 1``                   (total absorption normalized)
3. ``lambda2 / lambda - dL(1..N-1)``   (migration consistency)

In Python we use ``scipy.optimize.root`` with the ``hybr`` (Powell hybrid)
method, the closest equivalent to MATLAB's default ``trust-region-dogleg``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import root

from .equations import ModelOutputs, model_equations
from .initialize import BaselineEquilibrium


@dataclass
class SolveResult:
    """Final converged solution + diagnostics."""

    what: np.ndarray         # (1, N)
    dL: np.ndarray           # (1, N) — dL[-1] = 1 enforced
    outputs: ModelOutputs    # equations() evaluated at the solution
    converged: bool
    message: str


def _pack(what: np.ndarray, dL_provs: np.ndarray,
           frozen_idx: tuple[int, ...] = ()) -> np.ndarray:
    """Pack (what, dL[free_provs]) into a flat vector for the solver.

    Frozen provinces have dL fixed at 1 and are not unknowns.
    """
    if not frozen_idx:
        return np.concatenate([what.ravel(), dL_provs.ravel()])
    N_provs = dL_provs.size
    free_mask = np.ones(N_provs, dtype=bool)
    for k in frozen_idx:
        free_mask[k] = False
    return np.concatenate([what.ravel(), dL_provs.ravel()[free_mask]])


def _unpack(V: np.ndarray, N: int,
             frozen_idx: tuple[int, ...] = ()) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of _pack. dL[N-1] (ROW) and any frozen provinces fixed at 1."""
    what = V[:N].reshape(1, N)
    n_provs = N - 1
    if not frozen_idx:
        dL_provs = V[N: 2 * N - 1].reshape(1, n_provs)
    else:
        n_free = n_provs - len(frozen_idx)
        dL_free = V[N: N + n_free]
        dL_provs_flat = np.ones(n_provs)
        free_mask = np.ones(n_provs, dtype=bool)
        for k in frozen_idx:
            free_mask[k] = False
        dL_provs_flat[free_mask] = dL_free
        dL_provs = dL_provs_flat.reshape(1, n_provs)
    dL = np.concatenate([dL_provs, np.array([[1.0]])], axis=1)
    return what, dL


def residual(
    V: np.ndarray,
    baseline: BaselineEquilibrium,
    *,
    tauhat: np.ndarray | None = None,
    Ahat: np.ndarray | None = None,
    Sn_override: np.ndarray | None = None,
    frozen_idx: tuple[int, ...] = (),
    kappa_override: np.ndarray | float | None = None,
) -> np.ndarray:
    """Stacked residual matching solve_model.m's objective."""
    N = baseline.N
    what, dL = _unpack(V, N, frozen_idx)
    out = model_equations(baseline, what, dL,
                          tauhat=tauhat, Ahat=Ahat, Sn_override=Sn_override,
                          frozen_idx=frozen_idx, kappa_override=kappa_override)
    res1 = (out.what1[0, : N - 1] - what[0, : N - 1])          # (N-1,)
    res2 = np.array([out.Inew.sum() - 1.0])                    # (1,)
    res3_full = (out.lambda2.ravel() / baseline.lambda_.ravel()
                 - dL[0, : N - 1])                              # (N-1,)
    if frozen_idx:
        free_mask = np.ones(N - 1, dtype=bool)
        for k in frozen_idx:
            free_mask[k] = False
        res3 = res3_full[free_mask]
    else:
        res3 = res3_full
    return np.concatenate([res1, res2, res3])


def solve(
    baseline: BaselineEquilibrium,
    *,
    tauhat: np.ndarray | None = None,
    Ahat: np.ndarray | None = None,
    Sn_override: np.ndarray | None = None,
    frozen_idx: tuple[int, ...] = (),
    kappa_override: np.ndarray | float | None = None,
    method: str = "hybr",
    tol: float = 1e-10,
    initial_guess: np.ndarray | None = None,
    verbose: bool = False,
) -> SolveResult:
    """Find the counterfactual (what, dL) by driving the residual to zero."""
    N = baseline.N
    n_unknowns = 2 * N - 1 - len(frozen_idx)
    V0 = initial_guess if initial_guess is not None else np.ones(n_unknowns)

    def f(V: np.ndarray) -> np.ndarray:
        return residual(V, baseline,
                        tauhat=tauhat, Ahat=Ahat, Sn_override=Sn_override,
                        frozen_idx=frozen_idx, kappa_override=kappa_override)
    res = root(f, V0, method=method, tol=tol)

    what, dL = _unpack(res.x, N, frozen_idx)
    outputs = model_equations(baseline, what, dL,
                              tauhat=tauhat, Ahat=Ahat, Sn_override=Sn_override,
                              frozen_idx=frozen_idx,
                              kappa_override=kappa_override)
    if verbose:
        print(f"  solver: {'OK' if res.success else 'FAIL'}, "
              f"|residual|∞={np.abs(res.fun).max():.2e}, nfev={res.nfev}")
    return SolveResult(what=what, dL=dL, outputs=outputs,
                        converged=bool(res.success), message=str(res.message))
