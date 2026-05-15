"""Dynamic baseline data construction — Phase 2a (Step 1 of CDP §3.1).

Builds the quarterly 2000-2007 time series of bilateral trade, value
added, wages, labor allocations, and trade shares that feed Step 2's
temporary-equilibrium solver. Direct port of ``Step_1_data.m`` plus its
``LMC.m`` helper.

Inputs (read from the CDP MATLAB replication kit):

* ``xbilat{2000..2007}.csv`` — yearly bilateral trade flows (J*N × N).
* ``mu.mat`` — gross migration flows ``series_mu_adj`` of shape
  ``(RJ1, RJ1, 28)`` plus ``L0_initial`` of shape ``(RJ1, 1)`` for the
  initial labor allocation (RJ1 = R · (J + 1), the US labor-market
  count including the non-employment sector).
* ``L0_zero_nonem.csv`` — alternative initial labor (CSV with the
  non-employment row zeroed; the MATLAB code uses the mu.mat L0_initial,
  not this CSV — kept for completeness).

The CDP Base_Year results (xbilat00, VARjn00, VALjn00) and γ from the
static calibration come in via a ``BaseYearResult`` argument.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from qge.models.base_year import BaseYearResult

N_QUARTERS = 29     # 2000Q1 through 2007Q1 — 29 anchor points, 28 transitions.
N_TRANS = N_QUARTERS - 1


@dataclass(frozen=True)
class QuarterlySeries:
    """Output of Step 1 — quarterly time series 2000Q1-2007Q1.

    Shape conventions:
    * ``J``  productive sectors (22)
    * ``N``  regions (87 = 50 US states + 37 countries)
    * ``R``  US states (50); foreign block is ``N - R = 37``
    * ``RJ1 = R · (J + 1)``  US labor markets including non-employment
    """

    Din_baseline: np.ndarray         # (J*N, N, N_QUARTERS)
    series_xbilat: np.ndarray         # (J*N, N, N_QUARTERS)
    series_wageshat: np.ndarray       # (J, N, N_QUARTERS)
    series_Ljn0hat: np.ndarray        # (J+1, R, N_QUARTERS)
    series_mu: np.ndarray             # (RJ1, RJ1, N_TRANS)
    L0_initial: np.ndarray            # (RJ1,)


# ---------------------------------------------------------------- inputs


def _load_yearly_xbilat(rep_dir: Path, year: int) -> np.ndarray:
    """Read one ``xbilat{year}.csv`` — (1914, 87) comma-separated."""
    return np.loadtxt(rep_dir / f"xbilat{year}.csv", delimiter=",")


def _interpolate_to_quarterly(rep_dir: Path) -> np.ndarray:
    """Build the (1914, 87, 29) quarterly xbilat by geometric interpolation.

    The MATLAB Step_1_data.m takes each yearly pair (year_t, year_t+1)
    and produces 4 quarterly slices by applying ``(x_t+1 / x_t)^(1/4)``
    four times to ``x_t``. After all years, NaNs are zeroed, then a
    backward sweep propagates zeros earlier in time (a cell that ever
    reaches zero is treated as zero from the start).
    """
    yearly = [_load_yearly_xbilat(rep_dir, y) for y in range(2000, 2008)]
    slices = [yearly[0]]
    for x_t, x_t1 in zip(yearly[:-1], yearly[1:]):
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = (x_t1 / x_t) ** 0.25
        for _ in range(4):
            with np.errstate(invalid="ignore"):
                slices.append(ratio * slices[-1])

    arr = np.stack(slices, axis=-1)
    arr = np.where(np.isnan(arr) | np.isinf(arr), 0.0, arr)

    # Backward-propagate zeros: if cell at t is 0, force t-1 = 0. Iterate
    # back through all transitions so a zero at the end propagates all
    # the way to t = 0.
    for k in range(N_TRANS):
        later = N_TRANS - k
        earlier = later - 1
        zero_mask = arr[:, :, later] == 0
        arr[:, :, earlier][zero_mask] = 0.0
    return arr


def _load_mu(rep_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load μ migration flows and initial labor from ``mu.mat``."""
    m = loadmat(str(rep_dir / "mu.mat"))
    return m["series_mu_adj"], m["L0_initial"].ravel()


# ---------------------------------------------------------------- helpers


def labor_market_clearing(X: np.ndarray, gamma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Port of ``LMC.m`` — value added and per-quarter change.

    ``X`` shape ``(J*N, N, time)``; ``gamma`` shape ``(J, N)``.
    Returns ``(VA, VAhat)`` both shape ``(J, N, time)``. ``VAhat[..., -1]``
    is left at 1 (the closing change for a series is undefined).
    """
    J, N = gamma.shape
    time = X.shape[2]
    X_4d = X.reshape(J, N, N, time)
    Exjnp = X_4d.sum(axis=1)                # sum over destinations → (J, N, time)
    VA = gamma[:, :, None] * Exjnp
    VAhat = np.ones_like(VA)
    with np.errstate(invalid="ignore", divide="ignore"):
        VAhat[..., :-1] = VA[..., 1:] / VA[..., :-1]
    VAhat = np.where(np.isnan(VAhat) | np.isinf(VAhat), 1.0, VAhat)
    return VA, VAhat


# ---------------------------------------------------------------- main builder


def build_quarterly_series(
    rep_dir: Path, baseline: BaseYearResult, gamma: np.ndarray, B: np.ndarray
) -> QuarterlySeries:
    """Build the full Step 1 quarterly time series.

    ``baseline`` carries the post-solve Base_Year quantities (VARjn00,
    VALjn00, xbilat00); ``gamma`` (J, N) and ``B`` (J, N) come from the
    Base_Year RawInputs. Returns a ``QuarterlySeries`` of all arrays
    needed downstream by Step 2.
    """
    rep_dir = Path(rep_dir)
    J, N = gamma.shape
    R = 50                              # CDP fixed
    JNT1 = J + 1                        # 23: productive sectors plus non-employment
    RJ1 = R * JNT1                       # 1150 US labor markets

    # Step 1a — yearly to quarterly xbilat.
    xbilat = _interpolate_to_quarterly(rep_dir)
    # MATLAB renormalizes change-ratios to 1 wherever divisions blow up.
    with np.errstate(invalid="ignore", divide="ignore"):
        xbilathat = np.where(
            xbilat[..., :-1] > 0,
            xbilat[..., 1:] / np.maximum(xbilat[..., :-1], 1e-30),
            1.0,
        )
    xbilathat = np.where(np.isnan(xbilathat) | np.isinf(xbilathat), 1.0, xbilathat)

    # Step 1b — migration flows.
    series_mu, L0_initial = _load_mu(rep_dir)
    assert series_mu.shape == (RJ1, RJ1, N_TRANS)
    assert L0_initial.shape == (RJ1,)

    # Step 1c — value added trajectory from market clearing.
    _, VAhat = labor_market_clearing(xbilat, gamma)
    VA00 = baseline.VARjnp + baseline.VALjnp

    series_VA = np.empty((J, N, N_QUARTERS))
    series_VA[..., 0] = VA00
    for t in range(1, N_QUARTERS):
        series_VA[..., t] = series_VA[..., t - 1] * VAhat[..., t - 1]

    # Step 1d — labor allocations through μ.
    series_L0 = np.empty((RJ1, N_QUARTERS))
    series_L0[:, 0] = L0_initial
    for t in range(1, N_QUARTERS):
        series_L0[:, t] = series_mu[..., t - 1].T @ series_L0[:, t - 1]
    series_Ljn0 = series_L0.reshape(JNT1, R, N_QUARTERS, order="F")

    # Step 1e — wages = VAL / labor for US sectors, sector-shared for foreign.
    series_VAL = (1 - B[..., None]) * series_VA
    series_wages = np.empty((J, N, N_QUARTERS))
    # US sectoral wages: VAL[j, state, t] / Ljn0[j+1, state, t]. Non-employment
    # is row 0 of series_Ljn0; productive sectors are rows 1..J.
    series_wages[:, :R, :] = series_VAL[:, :R, :] / series_Ljn0[1:, :, :]
    # Foreign block — MATLAB Step_1_data.m has a real indexing bug here:
    # it reads from US-state indices 0..N-R-1 instead of foreign indices
    # R..N-1 when filling series_wages[:, R:N, t]. We replicate the bug
    # so the saved series_wageshat matches the MATLAB fixture (the bug is
    # benign for Step 2 since these wages enter only as the initial guess
    # for the foreign-country factor prices, which the solver iterates).
    us_block_totals = series_VAL[:, : N - R, :].sum(axis=0, keepdims=True)
    series_wages[:, R:, :] = np.broadcast_to(
        us_block_totals, (J, N - R, N_QUARTERS)
    ).copy()

    # Step 1f — bilateral trade flows and shares.
    series_xbilat = np.empty_like(xbilat)
    series_xbilat[..., 0] = baseline.xbilatp
    for t in range(1, N_QUARTERS):
        series_xbilat[..., t] = series_xbilat[..., t - 1] * xbilathat[..., t - 1]

    Xjn = series_xbilat.sum(axis=1, keepdims=True)   # (J*N, 1, T) destination totals
    Din_baseline = np.where(Xjn > 0, series_xbilat / np.maximum(Xjn, 1e-30), 0.0)

    # Step 1g — change-ratios of wages and labor.
    series_wageshat = np.ones((J, N, N_QUARTERS))
    series_wageshat[..., 1:] = series_wages[..., 1:] / series_wages[..., :-1]
    series_Ljn0hat = np.ones((JNT1, R, N_QUARTERS))
    series_Ljn0hat[..., 1:] = series_Ljn0[..., 1:] / series_Ljn0[..., :-1]

    return QuarterlySeries(
        Din_baseline=Din_baseline,
        series_xbilat=series_xbilat,
        series_wageshat=series_wageshat,
        series_Ljn0hat=series_Ljn0hat,
        series_mu=series_mu,
        L0_initial=L0_initial,
    )
