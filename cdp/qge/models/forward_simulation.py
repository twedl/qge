"""Forward simulation 2007Q4 → 2007Q4 + T quarters — Phase 2c (Step 3).

Direct port of Step_3_Baseline_2007_forward.m. With constant
fundamentals (no shocks), the economy still evolves because workers
reallocate based on forward-looking value functions. The algorithm is
an outer Bellman fixed point on the value-function path ``Yt``:

* For a guess ``Yt``, build the migration-probability path via
  ``compute_mu_path`` (workers go where lifetime value is high).
* Evolve labor forward via ``evolve_labor_forward``.
* For each of ``time`` quarters, solve a temporary equilibrium (the
  static factor-price fixed point with no shocks).
* Update ``Yt`` via ``bellman_update_Y``.

Iterate until ``max |Y_new − Yt| < toldyn``. The saved
``Hvectnoshock`` from ``Baseline_2007.mat`` is a converged seed — when
used as the starting ``Yt``, the algorithm converges in one outer iter.

The ``solve_cf`` of the MATLAB code is solve_tvf with the data-target
factors disabled (``pi_tilde0 = pi_tilde1 = pi`` so the shift ratio is
1, and ``om0 = 1`` so ``log(om/om0) = log(om)``). We reuse solve_tvf
under those conditions instead of porting solve_cf separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qge.dynamic import QuarterlySeries, build_quarterly_series
from qge.forward_dynamics import (
    bellman_update_Y, compute_mu_path, evolve_labor_forward,
)
from qge.io import RawInputs, load_inputs
from qge.models.base_year import EquilibriumResult, compute_baseline
from qge.models.dynamic_baseline import (
    DynamicBaseline2000_2007, compute_dynamic_baseline_2000_2007, solve_tvf,
)


DEFAULT_FORWARD_TIME = 200            # quarters in the forward simulation
DEFAULT_OUTER_TOL = 1e-3              # toldyn in the MATLAB code
DEFAULT_MAX_OUTER_ITER = 50           # generous; empirical convergence is ~1-10 iters


@dataclass(frozen=True)
class ForwardSimulation:
    """Output of Step 3 — 200-period forward simulation from 2007Q4."""

    Hvectnoshock: np.ndarray         # (RJ1, time) converged value function
    pi_baseline: np.ndarray           # (J*N, N, time) trade shares
    xbilat_out: np.ndarray            # (J*N, N, time) trade flows
    wages0: np.ndarray                # (J, N, time) wage changes
    Ljn_hat0: np.ndarray              # (J, N, time) labor changes
    mu: np.ndarray                    # (RJ1, RJ1, time) migration flows
    outer_iters: int


def _inner_equilibrium_path(
    Ldyn: np.ndarray, baseline_2007: EquilibriumResult, raw: RawInputs,
    *, time: int, tol: float, vfactor: float, maxit: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the per-period temporary-equilibrium sequence.

    Solves 200 sequential static equilibria (one per quarter) with
    constant fundamentals. Each quarter inherits ``(pi, VARjn0,
    VALjn0)`` from the previous quarter's converged state and computes
    the new ``(Dinp, wf0, xbilatp)``. Returns the four time-series
    tensors plus real wages used downstream by the Bellman update.
    """
    J, N, R = raw.J, raw.N, raw.R

    pi_baseline = np.zeros((J * N, N, time))
    xbilat_out = np.zeros((J * N, N, time))
    wages0 = np.zeros((J, N, time))
    Ljn_hat0 = np.zeros((J, N, time))
    realwages = np.ones((J, N, time))

    pi_baseline[..., 0] = baseline_2007.Dinp
    xbilat_out[..., 0] = baseline_2007.xbilatp

    pi = baseline_2007.Dinp.copy()
    VARjn0 = baseline_2007.VARjnp.copy()
    VALjn0 = baseline_2007.VALjnp.copy()
    Ltemp = Ldyn[1:, :, :]                     # drop non-employment row
    kappa_hat = np.ones((J * N, N))
    A_hat = np.ones((J, N))
    Snp = np.zeros(N)
    om_seed = np.ones((J, N))                  # MATLAB resets om = ones each period
    Ljn_hat = np.ones((J, N))                  # reuse across periods; only US block changes

    for t in range(time - 2):
        Ljn_hat[:, :R] = Ltemp[:, :, t + 1] / Ltemp[:, :, t]
        Ljn_hat0[:, :, t + 1] = Ljn_hat

        # solve_cf == solve_tvf with pi_tilde0 = pi_tilde1 = pi, om0 = 1.
        result = solve_tvf(
            om_seed, Ljn_hat, VARjn0, VALjn0, pi, Snp, kappa_hat, A_hat,
            raw, pi, pi, om_seed,
            tol=tol, vfactor=vfactor, maxit=maxit,
        )

        VARjn0 = result.VARjnp
        VALjn0 = result.VALjnp
        pi = result.Dinp

        pi_baseline[..., t + 1] = result.Dinp
        xbilat_out[..., t + 1] = result.xbilatp
        wages0[..., t + 1] = result.wf0
        realwages[..., t + 1] = result.wf0 / result.Phat[None, :]

    return pi_baseline, xbilat_out, wages0, Ljn_hat0, realwages


def compute_baseline_forward_2007(
    Yt_seed: np.ndarray,
    raw: RawInputs | None = None,
    baseline: EquilibriumResult | None = None,
    dynamic_2000_2007: DynamicBaseline2000_2007 | None = None,
    quarterly: QuarterlySeries | None = None,
    rep_dir: Path | None = None,
    *,
    time: int = DEFAULT_FORWARD_TIME,
    outer_tol: float = DEFAULT_OUTER_TOL,
    max_outer_iter: int = DEFAULT_MAX_OUTER_ITER,
    tol: float = 1e-7,
    vfactor: float = -0.05,
    maxit: int = int(1e6),
    verbose: bool = False,
) -> ForwardSimulation:
    """Run the 2007-forward dynamic baseline.

    Seeds the outer Bellman fixed-point on ``Yt`` from ``Yt_seed``. If
    ``Yt_seed`` is already the converged solution, exits after one
    outer iteration (Ymax < tol).
    """
    if raw is None:
        raw = load_inputs()
    if baseline is None:
        baseline = compute_baseline(raw=raw, tol=1e-7, vfactor=-0.05)
    if quarterly is None:
        assert rep_dir is not None, "rep_dir required when quarterly is not provided"
        quarterly = build_quarterly_series(rep_dir, baseline, raw.gamma, raw.B)
    if dynamic_2000_2007 is None:
        dynamic_2000_2007 = compute_dynamic_baseline_2000_2007(
            raw=raw, baseline=baseline, quarterly=quarterly,
        )

    J, R = raw.J, raw.R
    mu_init = quarterly.series_mu[..., -1]              # last Phase 2a flow
    L0 = quarterly.series_L0[:, -1]                      # labor at end of 2007Q4
    baseline_2007 = dynamic_2000_2007.final_equilibrium

    Yt = Yt_seed.copy()
    if Yt.shape[1] != time:
        raise ValueError(
            f"Yt_seed has {Yt.shape[1]} quarters, expected {time}"
        )

    mu_path = pi_baseline = xbilat_out = wages0 = Ljn_hat0 = realwages = None
    for outer in range(1, max_outer_iter + 1):
        mu_path = compute_mu_path(mu_init, Yt)
        Ldyn = evolve_labor_forward(mu_path, L0, J=J, R=R)
        pi_baseline, xbilat_out, wages0, Ljn_hat0, realwages = _inner_equilibrium_path(
            Ldyn, baseline_2007, raw,
            time=time, tol=tol, vfactor=vfactor, maxit=maxit,
        )
        Y_new = bellman_update_Y(Yt, mu_path, realwages, R=R, J=J)
        excess = float(np.abs(Y_new[:, :time] - Yt[:, :time]).max())
        if verbose:
            print(f"  outer={outer:2d}  Ymax={excess:.6e}")
        # MATLAB always averages then exits — replicate so a seeded run
        # matches the saved-fixture half-step convention.
        Yt = 0.5 * (Y_new + Yt)
        if excess < outer_tol:
            break

    assert mu_path is not None
    return ForwardSimulation(
        Hvectnoshock=Yt,
        pi_baseline=pi_baseline,
        xbilat_out=xbilat_out,
        wages0=wages0,
        Ljn_hat0=Ljn_hat0,
        mu=mu_path,
        outer_iters=outer,
    )
