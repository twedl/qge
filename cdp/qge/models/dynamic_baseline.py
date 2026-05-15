"""Dynamic baseline 2000-2007 solver — Phase 2b (Step 2 of CDP §3.1).

Direct port of solve_tvf.m and Step_2_Baseline_00_07.m. For each of 28
quarter-to-quarter transitions, solves a temporary equilibrium that
pins factor prices to match the constructed Phase 2a data targets
(``pi_tilde0``, ``pi_tilde1``, ``om0``). The state ``(pi, VARjn0,
VALjn0)`` carries forward from one quarter's converged equilibrium into
the next quarter's initial conditions.

``expenditurenew`` and ``GMCnew`` from ``qge.helpers`` are reused as-is —
the MATLAB ``expenditure_tvf`` / ``GMC_tvf`` are mathematically
identical to the static-Phase-1 versions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qge.dynamic import N_QUARTERS, N_TRANS, QuarterlySeries, build_quarterly_series
from qge.dynamic_helpers import Dinprime_tvf, P_h_om_tvf, shifted_pi_k
from qge.helpers import GMCnew, expenditurenew
from qge.io import RawInputs, load_inputs
from qge.models.base_year import (
    EquilibriumResult,
    _recover_outputs,
    compute_baseline,
)


@dataclass(frozen=True)
class DynamicBaseline2000_2007:
    """Output of Step 2 — 29-quarter dynamic baseline (2000Q1 anchor + 28 transitions)."""

    New_Din_baseline: np.ndarray         # (J*N, N, N_QUARTERS)
    New_series_xbilat: np.ndarray         # (J*N, N, N_QUARTERS)
    New_series_wageshat: np.ndarray       # (J, N, N_QUARTERS)
    final_equilibrium: EquilibriumResult  # 2007Q4 state — seed for Phase 2c


def solve_tvf(
    om: np.ndarray,
    Ljn_hat: np.ndarray,
    VARjn0: np.ndarray,
    VALjn0: np.ndarray,
    pi: np.ndarray,
    Snp: np.ndarray,
    kappa_hat: np.ndarray,
    A_hat: np.ndarray,
    raw: RawInputs,
    pi_tilde1: np.ndarray,
    pi_tilde0: np.ndarray,
    om0: np.ndarray,
    *,
    tol: float = 1e-7,
    vfactor: float = -0.05,
    maxit: int = int(1e6),
) -> EquilibriumResult:
    """One quarter's temporary-equilibrium solve."""
    J, N, R = raw.J, raw.N, raw.R
    om = om.copy()
    pi_k = shifted_pi_k(pi, kappa_hat, pi_tilde0, pi_tilde1, raw.T, N)
    ommax = 1.0
    itw = 1

    while itw <= maxit and ommax > tol:
        phat, x_hat = P_h_om_tvf(
            om, om0, A_hat, raw.T, raw.G, raw.gamma, pi_k,
            J, N, int(1e10), 1e-10,
        )
        Dinp = Dinprime_tvf(A_hat, x_hat, phat, raw.T, J, N, raw.gamma, pi_k)
        Xp = expenditurenew(
            J, N, raw.alphas, raw.B, raw.G, Dinp, om, Ljn_hat,
            Snp, VARjn0, VALjn0, raw.io,
        )
        omef0 = GMCnew(
            Xp, Dinp, J, N, raw.B, raw.gamma, Ljn_hat,
            VARjn0, VALjn0, R,
        )
        ZW = om - omef0
        om1 = om * (1 + vfactor * ZW / om)
        # Foreign countries have a single national wage (row 0 only);
        # US states have sectoral wages.
        om_world = np.concatenate([
            (om1[:, :R] - om[:, :R]).flatten("F"),
            (om1[0, R:] - om[0, R:]),
        ])
        ommax = float(np.sum(om_world ** 2))
        om = om1
        itw += 1

    return _recover_outputs(
        om, Ljn_hat, VARjn0, VALjn0, Snp, Dinp, Xp, phat, raw, itw,
    )


def compute_dynamic_baseline_2000_2007(
    raw: RawInputs | None = None,
    baseline: EquilibriumResult | None = None,
    quarterly: QuarterlySeries | None = None,
    rep_dir: Path | None = None,
    *,
    tol: float = 1e-7,
    vfactor: float = -0.05,
    maxit: int = int(1e6),
    verbose: bool = False,
) -> DynamicBaseline2000_2007:
    """Run the 28-quarter dynamic-baseline temporary-equilibrium sequence."""
    if raw is None:
        raw = load_inputs()
    if baseline is None:
        baseline = compute_baseline(raw=raw, tol=1e-7, vfactor=-0.05)
    if quarterly is None:
        assert rep_dir is not None, "rep_dir required when quarterly is not provided"
        quarterly = build_quarterly_series(rep_dir, baseline, raw.gamma, raw.B)

    J, N, R = raw.J, raw.N, raw.R

    # Drop the non-employment row; pad foreign columns with ones.
    Ljn_hat0 = np.ones((J, N, N_QUARTERS))
    Ljn_hat0[:, :R, :] = quarterly.series_Ljn0hat[1:, :, :]

    VARjn0 = baseline.VARjnp.copy()
    VALjn0 = baseline.VALjnp.copy()
    pi = baseline.Dinp.copy()

    New_Din_baseline = np.empty((J * N, N, N_QUARTERS))
    New_series_xbilat = np.empty((J * N, N, N_QUARTERS))
    New_series_wageshat = np.ones((J, N, N_QUARTERS))

    New_Din_baseline[..., 0] = baseline.Dinp
    New_series_xbilat[..., 0] = baseline.xbilatp

    kappa_hat = np.ones((J * N, N))
    A_hat = np.ones((J, N))     # CDP baseline has no TFP shocks
    Snp = np.zeros(N)

    result: EquilibriumResult | None = None
    for t in range(N_TRANS):
        Ljn_hat = Ljn_hat0[:, :, t + 1]
        om0 = quarterly.series_wageshat[:, :, t + 1] * (Ljn_hat ** raw.B)

        result = solve_tvf(
            om0, Ljn_hat, VARjn0, VALjn0, pi, Snp, kappa_hat, A_hat,
            raw, quarterly.Din_baseline[..., t + 1], quarterly.Din_baseline[..., t], om0,
            tol=tol, vfactor=vfactor, maxit=maxit,
        )

        if verbose:
            print(f"  quarter {t + 1:2d}/{N_TRANS}  iters={result.iterations}")

        VARjn0 = result.VARjnp
        VALjn0 = result.VALjnp
        pi = result.Dinp

        New_Din_baseline[..., t + 1] = result.Dinp
        New_series_xbilat[..., t + 1] = result.xbilatp
        New_series_wageshat[..., t + 1] = result.wf0

    assert result is not None
    return DynamicBaseline2000_2007(
        New_Din_baseline=New_Din_baseline,
        New_series_xbilat=New_series_xbilat,
        New_series_wageshat=New_series_wageshat,
        final_equilibrium=result,
    )
