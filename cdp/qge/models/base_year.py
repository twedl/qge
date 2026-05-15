"""CDP year-2000 base-year equilibrium solver.

Direct port of ``Base_year.m`` and ``solvewnew.m`` from the Caliendo-
Dvorkin-Parro (2019) replication kit. Computes the initial static
equilibrium ω (factor prices), Dinp (trade shares), Xp (expenditure),
VARjnp/VALjnp (structures/labor value), Phat (price index) consistent
with the calibrated 2000 data.

The outputs of this static solve seed the dynamic baseline (Phase 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from qge.helpers import Dinprime, GMCnew, P_h_om, expenditurenew
from qge.io import RawInputs, load_inputs


@dataclass(frozen=True)
class EquilibriumResult:
    """One converged temporary-equilibrium snapshot — output of both the
    static Base_Year solve and each quarter of the dynamic baseline."""

    om: np.ndarray
    wf0: np.ndarray
    rf0: np.ndarray
    VARjnp: np.ndarray
    VALjnp: np.ndarray
    Phat: np.ndarray
    phat: np.ndarray
    Dinp: np.ndarray
    Xp: np.ndarray
    Snp: np.ndarray
    xbilatp: np.ndarray
    iterations: int


# Backwards-compatible alias for the Phase 1 name.
BaseYearResult = EquilibriumResult


def _recover_outputs(
    om: np.ndarray, Ljn_hat: np.ndarray,
    VARjn0: np.ndarray, VALjn0: np.ndarray,
    Snp: np.ndarray, Dinp: np.ndarray, Xp: np.ndarray, phat: np.ndarray,
    raw: RawInputs, iterations: int,
) -> EquilibriumResult:
    """Shared post-iteration block. Recovers wages, rentals, value-added
    pieces, price index, and bilateral trade flows from the converged
    factor prices ``om``.
    """
    J, N, R = raw.J, raw.N, raw.R
    wf0 = np.empty((J, N))
    wf0[:, :R] = om[:, :R] * (Ljn_hat[:, :R] ** (-raw.B[:, :R]))
    wf0[:, R:] = om[:, R:]
    rf0 = np.empty((J, N))
    rf0[:, :R] = wf0[:, :R] * Ljn_hat[:, :R]
    rf0[:, R:] = om[:, R:]
    VARjnp = VARjn0 * om * (Ljn_hat ** (1 - raw.B))
    VALjnp = wf0 * Ljn_hat * VALjn0
    Phat = np.prod(phat ** raw.alphas, axis=0)
    xbilatp = Xp.flatten()[:, None] * Dinp
    return EquilibriumResult(
        om=om, wf0=wf0, rf0=rf0,
        VARjnp=VARjnp, VALjnp=VALjnp,
        Phat=Phat, phat=phat,
        Dinp=Dinp, Xp=Xp, Snp=Snp,
        xbilatp=xbilatp, iterations=iterations,
    )


def solvewnew(
    om: np.ndarray,
    Ljn_hat: np.ndarray,
    raw: RawInputs,
    kappa_hat: np.ndarray,
    lambda_hat: np.ndarray,
    Snp: np.ndarray,
    *,
    tol: float = 1e-7,
    vfactor: float = -0.05,
    maxit: int = int(1e6),
    verbose: bool = False,
) -> EquilibriumResult:
    """Iterate on factor prices until labor-market clearing converges.

    Direct port of ``solvewnew.m``. Returns a populated ``EquilibriumResult``.
    """
    J, N, R = raw.J, raw.N, raw.R
    ommax = 1.0
    itw = 1
    om = om.copy()

    while itw <= maxit and ommax > tol:
        phat, c = P_h_om(
            om, kappa_hat, lambda_hat, raw.T,
            raw.G, raw.gamma, raw.Din,
            J, N, maxit=int(1e10), tol=1e-10,
        )
        Dinp = Dinprime(raw.Din, kappa_hat, lambda_hat, c, phat, raw.T, J, N, raw.gamma)
        Xp = expenditurenew(
            J, N, raw.alphas, raw.B, raw.G, Dinp, om, Ljn_hat,
            Snp, raw.VARjn0, raw.VALjn0, raw.io,
        )
        omef0 = GMCnew(
            Xp, Dinp, J, N, raw.B, raw.gamma, Ljn_hat,
            raw.VARjn0, raw.VALjn0, R,
        )

        ZW = om - omef0
        om1 = om * (1 + vfactor * ZW / om)
        # Foreign countries have a single national wage (row 0 of each
        # column is the full information); US states have sectoral wages.
        om_world = np.concatenate([
            (om1[:, :R] - om[:, :R]).flatten("F"),
            (om1[0, R:] - om[0, R:]),
        ])
        ommax = float(np.sum(om_world ** 2))
        om = om1
        if verbose and itw % 100 == 0:
            print(f"  itw={itw:6d}  ommax={ommax:.6e}")
        itw += 1

    return _recover_outputs(
        om, Ljn_hat, raw.VARjn0, raw.VALjn0,
        Snp, Dinp, Xp, phat, raw, itw,
    )


def compute_baseline(
    raw: Optional[RawInputs] = None,
    *,
    tol: float = 1e-7,
    vfactor: float = -0.05,
    maxit: int = int(1e6),
    verbose: bool = False,
) -> EquilibriumResult:
    """End-to-end CDP base-year solve. Mirrors ``Base_year.m``."""
    if raw is None:
        raw = load_inputs()
    J, N = raw.J, raw.N
    kappa_hat = np.ones((J * N, N))
    lambda_hat = np.ones((J, N))
    Snp = np.zeros(N)

    om = np.ones((J, N))
    Ljn_hat = np.ones((J, N))
    return solvewnew(
        om, Ljn_hat, raw, kappa_hat, lambda_hat, Snp,
        tol=tol, vfactor=vfactor, maxit=maxit, verbose=verbose,
    )
