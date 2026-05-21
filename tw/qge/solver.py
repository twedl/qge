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


def _pack(what: np.ndarray, dL_provs: np.ndarray) -> np.ndarray:
    """Pack (what, dL[provs]) into a flat (2N-1,) vector for the solver."""
    return np.concatenate([what.ravel(), dL_provs.ravel()])


def _unpack(V: np.ndarray, N: int) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of _pack. dL[N-1] (ROW) is fixed at 1."""
    what = V[:N].reshape(1, N)
    dL_provs = V[N: 2 * N - 1].reshape(1, N - 1)
    dL = np.concatenate([dL_provs, np.array([[1.0]])], axis=1)
    return what, dL


def residual(
    V: np.ndarray,
    baseline: BaselineEquilibrium,
    *,
    tauhat: np.ndarray | None = None,
    Ahat: np.ndarray | None = None,
    Sn_override: np.ndarray | None = None,
) -> np.ndarray:
    """Stacked residual matching solve_model.m's objective."""
    N = baseline.N
    what, dL = _unpack(V, N)
    out = model_equations(baseline, what, dL,
                          tauhat=tauhat, Ahat=Ahat, Sn_override=Sn_override)
    res1 = (out.what1[0, : N - 1] - what[0, : N - 1])          # (N-1,)
    res2 = np.array([out.Inew.sum() - 1.0])                    # (1,)
    res3 = (out.lambda2.ravel() / baseline.lambda_.ravel()
            - dL[0, : N - 1])                                   # (N-1,)
    return np.concatenate([res1, res2, res3])


def solve(
    baseline: BaselineEquilibrium,
    *,
    tauhat: np.ndarray | None = None,
    Ahat: np.ndarray | None = None,
    Sn_override: np.ndarray | None = None,
    method: str = "hybr",
    tol: float = 1e-10,
    initial_guess: np.ndarray | None = None,
    verbose: bool = False,
) -> SolveResult:
    """Find the counterfactual (what, dL) by driving the residual to zero."""
    N = baseline.N
    V0 = initial_guess if initial_guess is not None else np.ones(2 * N - 1)

    def f(V: np.ndarray) -> np.ndarray:
        return residual(V, baseline,
                        tauhat=tauhat, Ahat=Ahat, Sn_override=Sn_override)
    res = root(f, V0, method=method, tol=tol)

    what, dL = _unpack(res.x, N)
    outputs = model_equations(baseline, what, dL,
                              tauhat=tauhat, Ahat=Ahat, Sn_override=Sn_override)
    if verbose:
        print(f"  solver: {'OK' if res.success else 'FAIL'}, "
              f"|residual|∞={np.abs(res.fun).max():.2e}, nfev={res.nfev}")
    return SolveResult(what=what, dL=dL, outputs=outputs,
                        converged=bool(res.success), message=str(res.message))
