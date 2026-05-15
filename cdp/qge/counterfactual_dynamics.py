"""Counterfactual-economy primitives — Phase 3 of CDP §3.2.

The China-shock counterfactual perturbs the baseline economy by removing
the estimated 2000-2007 productivity gains in 12 Chinese sectors. The
algorithm is structurally similar to Phase 2c's forward simulation but
uses the saved BaselineEconomy as a reference: trade-share targets,
migration flows, and wages come from the baseline; the counterfactual
solves for the deviation.

This module exports the math primitives:

* ``china_tfp_shock_path`` — the (J, N, time) A_hat tensor with the
  inverse-China-shock applied to sectors 0..11 of the China region
  (index 56) for the first 28 quarters.
* ``compute_mu_path_cf`` — the counterfactual migration-flow path,
  built from the baseline ``series_mu`` and the candidate value
  function V.
* ``bellman_update_V_cf`` — one Bellman update on V given the
  candidate mu path and realized real wages.
"""

from __future__ import annotations

import numpy as np

from qge.helpers import BETA, NU, scrub

# China is the 7th country after 50 US states: ordering per CDP Readme is
# Australia, Austria, Belgium, Bulgaria, Brazil, Canada, China, ... so
# 0-indexed region index 56.
CHINA_REGION_IDX = 56

# Estimated relative annual China TFP changes 2000→2007 for the 12
# CDP tradable sectors (Section 5 of the paper, Table 4). Each entry is
# the cumulative 7-year productivity rise; quarterly shocks are the
# 28th root.
CHINA_ANNUAL_TFP = (
    25.6097, 7.6093, 5.2773, 9.2464, 7.6897, 45.7781,
    5.9298, 5.6533, 64.5661, 38.9064, 196.6509, 3.5324,
)
N_CHINA_SHOCK_SECTORS = len(CHINA_ANNUAL_TFP)         # 12
N_CHINA_SHOCK_QUARTERS = 28                            # 2000Q1 .. 2007Q4


def china_tfp_shock_path(J: int, N: int, time: int) -> np.ndarray:
    """Build the (J, N, time) inverse-China-shock A_hat tensor.

    The MATLAB counterfactual sets ``A_hat = 1 ./ As`` where ``As`` has
    the China quarterly TFP gains in the first 12 sectors of region 57
    (1-indexed) for the first 28 quarters and 1 elsewhere. Returns the
    completed A_hat ready to feed into ``solve_tvf``.
    """
    if N_CHINA_SHOCK_SECTORS > J:
        raise ValueError("China shock spans 12 sectors; model has J < 12")
    As = np.ones((J, N, time))
    quarterly_growth = np.asarray(CHINA_ANNUAL_TFP) ** (1.0 / N_CHINA_SHOCK_QUARTERS)
    As[:N_CHINA_SHOCK_SECTORS, CHINA_REGION_IDX, :N_CHINA_SHOCK_QUARTERS] = (
        quarterly_growth[:, None]
    )
    return 1.0 / As


def compute_mu_path_cf(
    mu_baseline: np.ndarray, V: np.ndarray, *, beta: float = BETA
) -> np.ndarray:
    """Counterfactual migration-flow path from baseline mu and candidate V.

    The MATLAB Step 2 (Appendix 3 Part II) constructs ``mu`` as:

    * ``mu[..., 0] = mu_baseline[..., 1] · V[k, 1]^β``  (unnormalized "jump")
    * ``mu[..., 1] = mu[..., 0] · V[k, 2]^β`` / row-sums
    * for ``t ≥ 1``: ``mu[..., t+1] = (mu_b[..., t+1] / mu_b[..., t]) ·
      mu[..., t] · V[k, t+2]^β`` / row-sums

    All NaNs (from 0/0 ratios at zero baseline flows) are zeroed.
    """
    RJ1, time = V.shape
    mu = np.empty((RJ1, RJ1, time))

    mu[..., 0] = mu_baseline[..., 1] * (V[:, 1] ** beta)[None, :]

    num = mu[..., 0] * (V[:, 2] ** beta)[None, :]
    np.nan_to_num(num, copy=False, nan=0.0)
    mu[..., 1] = num / num.sum(axis=1, keepdims=True)

    for t in range(1, time - 2):
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = mu_baseline[..., t + 1] / mu_baseline[..., t]
        np.nan_to_num(ratio, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        num = ratio * mu[..., t] * (V[:, t + 2] ** beta)[None, :]
        np.nan_to_num(num, copy=False, nan=0.0)
        mu[..., t + 1] = num / num.sum(axis=1, keepdims=True)

    # Last slice carried forward (MATLAB leaves at zeros; both are unused
    # downstream — the integration test indexes [..., :-1]).
    mu[..., time - 1] = mu[..., time - 2]
    return mu


def bellman_update_V_cf(
    V: np.ndarray,
    mu_baseline: np.ndarray,
    mu_cf: np.ndarray,
    rwage_us: np.ndarray,
    *,
    R: int,
    J: int,
    beta: float = BETA,
    nu: float = NU,
) -> np.ndarray:
    """Counterfactual Bellman update on V.

    ``rwage_us`` is the US-only real-wage hat series flattened to
    ``(RJ1, time)`` with the non-employment row first (row 0 of each
    state's block) and US state ordering preserved.

    Three regions of the output:

    * ``V_new[:, 0] = 0`` (boundary; unused downstream by the outer loop)
    * ``V_new[:, 1]`` uses the special "jump" formula with ``mu_baseline[..., 1]``
    * ``V_new[:, 2..time-2]`` uses the lambda recurrence: ratio ·
      ``mu_cf`` · ``rwage_us^(1/ν)`` · ``V[k, t+1]^β``
    * ``V_new[:, time-1] = 1`` (steady-state terminal)
    """
    RJ1, time = V.shape
    rwnu = rwage_us ** (1.0 / nu)                                # (RJ1, time)

    V_new = np.zeros((RJ1, time))
    # Special "jump" at t = 1.
    V_new[:, 1] = rwnu[:, 1] * (
        mu_baseline[..., 1] * (V[:, 1] ** beta)[None, :] * (V[:, 2] ** beta)[None, :]
    ).sum(axis=1)

    # General case for t = 2..time-2.
    for t in range(2, time - 1):
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = mu_baseline[..., t] / mu_baseline[..., t - 1]
        np.nan_to_num(ratio, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        lam = ratio * mu_cf[..., t - 1] * rwnu[:, t:t + 1]
        V_new[:, t] = (lam * (V[:, t + 1] ** beta)[None, :]).sum(axis=1)

    V_new[:, -1] = 1.0
    return V_new


def pack_rwage_us(realwages: np.ndarray, R: int) -> np.ndarray:
    """Pad ``(J, N, time)`` realwages with non-employment row, flatten to (RJ1, time).

    The padded array is ``(J+1, R, time)`` with the non-employment row
    (index 0) set to 1 and rows 1..J copied from US states of
    ``realwages``. Flattens C-order so ``rwage_us[state*(J+1) + market, t]``
    is the consumption shock for that labor market.
    """
    J = realwages.shape[0]
    time = realwages.shape[2]
    padded = np.ones((J + 1, R, time))
    padded[1:, :, :] = realwages[:, :R, :]
    # State-outer, market-inner: shape (R, J+1, time) → reshape C-order.
    return padded.transpose(1, 0, 2).reshape(R * (J + 1), time)
