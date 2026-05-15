"""Counterfactual economy — Phase 3 of CDP §3.2.

Direct port of ``Counterfactual_economy.m``. The algorithm follows
Appendix 3 Part II of the paper: an outer Bellman fixed point on the
value-function path ``V`` (= 1/ν times the exponential of the
counterfactual–baseline value differences). Each outer iteration:

1. Build a path of migration matrices μ_cf from baseline μ and the
   candidate V (via ``compute_mu_path_cf``).
2. Evolve labor forward under μ_cf.
3. For each of ``time`` quarters, solve a temporary equilibrium with
   ``A_hat = 1 / china_TFP`` (the inverse China shock) and the
   baseline trade-share/wage targets.
4. Update V via the counterfactual Bellman recurrence
   (``bellman_update_V_cf``).

Iterate until ``max |V_new − V| < toldyn``. The saved
``Counterfactual_economy.mat`` is a converged seed — when used as the
starting V, the algorithm converges in one outer iteration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qge.counterfactual_dynamics import (
    BETA, NU, bellman_update_V_cf, china_tfp_shock_path,
    compute_mu_path_cf, pack_rwage_us,
)
from qge.io import RawInputs, load_inputs
from qge.models.base_year import EquilibriumResult, compute_baseline
from qge.models.baseline_economy import BaselineEconomy, TOTAL_QUARTERS
from qge.models.dynamic_baseline import solve_tvf


@dataclass(frozen=True)
class CounterfactualEconomy:
    """Output of Phase 3 — the China-shock counterfactual.

    Mirrors ``Counterfactual_economy.mat``:
    * ``V`` — (RJ1, time) converged counterfactual value function
    * ``mu`` — (RJ1, RJ1, time) counterfactual migration path
    * ``Ldyn`` — (J+1, R, time) counterfactual US labor allocations
    * ``realwages`` — (J, N, time) counterfactual real wage changes
      relative to baseline
    * ``rwage`` — (R, J+1, time) realwages padded with non-employment
      and permuted to (states, markets, time)
    """

    V: np.ndarray
    mu: np.ndarray
    Ldyn: np.ndarray
    realwages: np.ndarray
    rwage: np.ndarray
    outer_iters: int


def _evolve_labor_forward(
    mu_path: np.ndarray, L0: np.ndarray, *, J: int, R: int,
) -> np.ndarray:
    """Apply mu_path forward to L0, returning (J+1, R, time) US labor."""
    RJ1, _, time = mu_path.shape
    Ldyn = np.empty((RJ1, time))
    Ldyn[:, 0] = L0
    for t in range(time - 2):
        Ldyn[:, t + 1] = mu_path[..., t].T @ Ldyn[:, t]
    Ldyn[:, time - 1] = 0.0
    return Ldyn.reshape(J + 1, R, time, order="F")


def _inner_equilibrium_path_cf(
    Ldyn: np.ndarray, base_year: EquilibriumResult, baseline_econ: BaselineEconomy,
    A_hat_path: np.ndarray, raw: RawInputs,
    *, time: int, tol: float, vfactor: float, maxit: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the per-period counterfactual temporary equilibria.

    Returns ``(realwages, Ljn_hat0)``. The realwages are
    ``(wf_cf / wf_baseline) / Phat_cf`` — the counterfactual change in
    consumption per labor market relative to baseline. ``base_year``
    is the **Phase 1 static 2000 baseline** (not 2007Q4): the
    counterfactual replays the full 200-quarter path starting from the
    2000 anchor that produced the baseline economy.
    """
    J, N, R = raw.J, raw.N, raw.R
    Ltemp = Ldyn[1:, :, :]                                # drop non-employment row
    pi_baseline = baseline_econ.series_pi
    series_wages = baseline_econ.series_wages
    Ljn_hat0_baseline = np.ones((J, N, time))
    Ljn_hat0_baseline[:, :R, :] = baseline_econ.series_Ljnhat

    pi = base_year.Dinp.copy()
    VARjn0 = base_year.VARjnp.copy()
    VALjn0 = base_year.VALjnp.copy()

    realwages = np.ones((J, N, time))
    kappa_hat = np.ones((J * N, N))
    Snp = np.zeros(N)

    for t in range(time - 2):
        Ljn_hat = np.ones((J, N))
        Ljn_hat[:, :R] = Ltemp[:, :, t + 1] / Ltemp[:, :, t]
        w0 = series_wages[:, :, t + 1]
        Ljn_hat00 = Ljn_hat0_baseline[:, :, t + 1]
        om0 = w0 * (Ljn_hat00 ** raw.B)

        result = solve_tvf(
            om0, Ljn_hat, VARjn0, VALjn0, pi, Snp, kappa_hat, A_hat_path[..., t],
            raw, pi_baseline[..., t + 1], pi_baseline[..., t], om0,
            tol=tol, vfactor=vfactor, maxit=maxit,
        )

        VARjn0 = result.VARjnp
        VALjn0 = result.VALjnp
        pi = result.Dinp
        realwages[:, :, t + 1] = (result.wf0 / w0) / result.Phat[None, :]

    return realwages, Ljn_hat0_baseline


def compute_counterfactual_economy(
    V_seed: np.ndarray,
    baseline_econ: BaselineEconomy,
    base_year: EquilibriumResult,
    raw: RawInputs | None = None,
    *,
    time: int = TOTAL_QUARTERS,
    outer_tol: float = 1e-3,
    max_outer_iter: int = 50,
    tol: float = 1e-7,
    vfactor: float = -0.05,
    maxit: int = int(1e6),
    verbose: bool = False,
) -> CounterfactualEconomy:
    """Run the counterfactual economy with the China shock removed."""
    if raw is None:
        raw = load_inputs()
    J, N, R = raw.J, raw.N, raw.R
    A_hat_path = china_tfp_shock_path(J, N, time)

    L0 = baseline_econ.L0_initial
    mu_baseline = baseline_econ.series_mu[..., :time]    # (RJ1, RJ1, time)

    V = V_seed.copy()
    if V.shape != (R * (J + 1), time):
        raise ValueError(f"V_seed shape {V.shape} ≠ ({R * (J + 1)}, {time})")

    mu_cf = Ldyn = realwages = rwage_us = None
    for outer in range(1, max_outer_iter + 1):
        mu_cf = compute_mu_path_cf(mu_baseline, V)
        Ldyn = _evolve_labor_forward(mu_cf, L0, J=J, R=R)
        realwages, _ = _inner_equilibrium_path_cf(
            Ldyn, base_year, baseline_econ, A_hat_path, raw,
            time=time, tol=tol, vfactor=vfactor, maxit=maxit,
        )
        rwage_us = pack_rwage_us(realwages, R=R)
        V_new = bellman_update_V_cf(V, mu_baseline, mu_cf, rwage_us, R=R, J=J)

        # MATLAB max(|V_new[:, 1:] - V[:, 1:]|): excludes the t=0 boundary.
        excess = float(np.abs(V_new[:, 1:] - V[:, 1:]).max())
        if verbose:
            print(f"  outer={outer:2d}  Ymax={excess:.6e}")
        V = 0.5 * (V_new + V)
        if excess < outer_tol:
            break

    # rwage = realwages padded with non-employment, permuted to (R, J+1, time).
    realwages_padded = np.ones((J + 1, N, time))
    realwages_padded[1:, :, :] = realwages
    rwage = realwages_padded[:, :R, :].transpose(1, 0, 2)

    return CounterfactualEconomy(
        V=V, mu=mu_cf, Ldyn=Ldyn,
        realwages=realwages, rwage=rwage,
        outer_iters=outer,
    )
