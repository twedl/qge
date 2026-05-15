"""Employment and welfare effects of the China shock — Phase 4.

Pure analysis over baseline + counterfactual outputs. Ports the
numerical kernels of:

* ``Employment_effects.m`` — share-decomposition of US labor across
  sectors and states with and without the China shock; aggregate
  manufacturing / non-manufacturing / non-employment effects.
* ``Welfare_effects.m`` — consumption-equivalent welfare changes per
  US labor market (paper eq. 28).
* ``Adjustment_costs.m`` — welfare decomposed into transitional-dynamics
  vs steady-state pieces.

All MATLAB figure-generation code is dropped — users plot the returned
DataFrames / arrays directly with matplotlib if they want.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qge.counterfactual_dynamics import BETA, NU
from qge.models.baseline_economy import BaselineEconomy
from qge.models.counterfactual import CounterfactualEconomy


# Sector ranges within the 22-sector productive list:
N_MANUF = 12                                     # sectors 0..11 — tradable manufacturing
TRADE_SECTOR = 12                                # sector 12 — wholesale/retail (tradable but treated separately)
CONSTRUCTION_SECTOR = 13                         # sector 13 — construction
SERVICES_RANGE = slice(14, 22)                   # sectors 14..21 — services


@dataclass(frozen=True)
class EmploymentEffects:
    """China-shock employment effects at the long-run horizon (t = T-1)."""

    manuf_share_change: float                    # share-of-total-employment % point change
    nonmanuf_share_change: float
    construction_share_change: float
    trade_share_change: float
    services_share_change: float
    nonemployment_share_change: float            # share-of-total-population % point change

    sectoral_manuf_contrib: np.ndarray            # (N_MANUF,) % contribution of each manuf sector
    sectoral_nonmanuf_contrib: np.ndarray         # (10,) % contribution of each non-manuf sector
    regional_manuf_contrib: np.ndarray            # (R,) % contribution of each US state
    regional_nonmanuf_contrib: np.ndarray         # (R,) % contribution of each US state


@dataclass(frozen=True)
class WelfareEffects:
    """China-shock welfare changes per labor market."""

    logdelta: np.ndarray                          # (R, J+1, T) — log consumption-equivalent change at each t
    aggregate_welfare_pct: float                  # L0-weighted aggregate (% × 100)


# ---------------------------------------------------------------- employment


def _evolve_baseline_labor(
    L0_initial: np.ndarray, series_mu: np.ndarray, *, J: int, R: int, time: int
) -> np.ndarray:
    """Evolve the baseline labor allocation 200 quarters forward.

    Mirrors the MATLAB Employment_effects.m loop that uses
    ``series_mu`` from the baseline workspace. Returns (J+1, R, time).
    """
    RJ1 = R * (J + 1)
    L = np.zeros((RJ1, time))
    L[:, 0] = L0_initial
    for t in range(time - 2):
        L[:, t + 1] = series_mu[..., t].T @ L[:, t]
    L[:, time - 1] = 0.0
    return L.reshape(J + 1, R, time, order="F")


def compute_employment_effects(
    baseline_econ: BaselineEconomy,
    counterfactual: CounterfactualEconomy,
    *, J: int = 22, R: int = 50,
) -> EmploymentEffects:
    """Long-run share-of-employment effects of the China shock."""
    time = counterfactual.Ldyn.shape[2]
    L_baseline = _evolve_baseline_labor(
        baseline_econ.L0_initial, baseline_econ.series_mu,
        J=J, R=R, time=time,
    )
    L_cf = counterfactual.Ldyn

    horizon = time - 1                          # MATLAB Time = 199, last filled slot
    # Drop the non-employment row to get employment-only labor.
    emp_baseline = L_baseline[1:, :, horizon]
    emp_cf = L_cf[1:, :, horizon]
    total_emp_baseline = emp_baseline.sum()
    total_emp_cf = emp_cf.sum()
    share_baseline = emp_baseline / total_emp_baseline
    share_cf = emp_cf / total_emp_cf

    def _agg(arr_2d: np.ndarray, sector_slice) -> float:
        return float(arr_2d[sector_slice, :].sum())

    share_manuf_base = _agg(share_baseline, slice(0, N_MANUF))
    share_manuf_cf = _agg(share_cf, slice(0, N_MANUF))
    share_construction_base = _agg(share_baseline, slice(CONSTRUCTION_SECTOR, CONSTRUCTION_SECTOR + 1))
    share_construction_cf = _agg(share_cf, slice(CONSTRUCTION_SECTOR, CONSTRUCTION_SECTOR + 1))
    share_trade_base = _agg(share_baseline, slice(TRADE_SECTOR, TRADE_SECTOR + 1))
    share_trade_cf = _agg(share_cf, slice(TRADE_SECTOR, TRADE_SECTOR + 1))
    share_services_base = _agg(share_baseline, SERVICES_RANGE)
    share_services_cf = _agg(share_cf, SERVICES_RANGE)
    share_nonmanuf_base = _agg(share_baseline, slice(TRADE_SECTOR, 22))
    share_nonmanuf_cf = _agg(share_cf, slice(TRADE_SECTOR, 22))

    nonemp_share_base = L_baseline[0, :, horizon].sum() / L_baseline[..., horizon].sum()
    nonemp_share_cf = L_cf[0, :, horizon].sum() / L_cf[..., horizon].sum()

    # Sectoral contributions to manufacturing decline (Figure 2 in paper).
    # Use t=1 (second slot) employment as the "initial" reference per MATLAB.
    emp_init_base = L_baseline[1:, :, 1] / L_baseline[1:, :, 1].sum()
    manuf_init = emp_init_base[:N_MANUF, :].sum(axis=1)
    manuf_ss_base = share_baseline[:N_MANUF, :].sum(axis=1)
    manuf_ss_cf = share_cf[:N_MANUF, :].sum(axis=1)
    chg_base = manuf_ss_base - manuf_init
    chg_cf = manuf_ss_cf - manuf_init
    china_effect_j = -(chg_cf - chg_base)
    sectoral_manuf_contrib = 100.0 * china_effect_j / china_effect_j.sum()

    nonmanuf_init = emp_init_base[TRADE_SECTOR:, :].sum(axis=1)
    nonmanuf_ss_base = share_baseline[TRADE_SECTOR:, :].sum(axis=1)
    nonmanuf_ss_cf = share_cf[TRADE_SECTOR:, :].sum(axis=1)
    chg_base = nonmanuf_ss_base - nonmanuf_init
    chg_cf = nonmanuf_ss_cf - nonmanuf_init
    china_effect_j = -(chg_cf - chg_base)
    sectoral_nonmanuf_contrib = 100.0 * china_effect_j / china_effect_j.sum()

    # Regional contributions to aggregate manufacturing employment decline.
    manuf_n_init = emp_init_base[:N_MANUF, :].sum(axis=0)
    manuf_n_ss_base = share_baseline[:N_MANUF, :].sum(axis=0)
    manuf_n_ss_cf = share_cf[:N_MANUF, :].sum(axis=0)
    chg_n_base = manuf_n_ss_base - manuf_n_init
    chg_n_cf = manuf_n_ss_cf - manuf_n_init
    china_effect_n = -(chg_n_cf - chg_n_base)
    regional_manuf_contrib = 100.0 * china_effect_n / china_effect_n.sum()

    nonmanuf_n_init = emp_init_base[TRADE_SECTOR:, :].sum(axis=0)
    nonmanuf_n_ss_base = share_baseline[TRADE_SECTOR:, :].sum(axis=0)
    nonmanuf_n_ss_cf = share_cf[TRADE_SECTOR:, :].sum(axis=0)
    chg_n_base = nonmanuf_n_ss_base - nonmanuf_n_init
    chg_n_cf = nonmanuf_n_ss_cf - nonmanuf_n_init
    china_effect_n = -(chg_n_cf - chg_n_base)
    regional_nonmanuf_contrib = 100.0 * china_effect_n / china_effect_n.sum()

    return EmploymentEffects(
        manuf_share_change=100.0 * (share_manuf_base - share_manuf_cf),
        nonmanuf_share_change=100.0 * (share_nonmanuf_base - share_nonmanuf_cf),
        construction_share_change=100.0 * (share_construction_base - share_construction_cf),
        trade_share_change=100.0 * (share_trade_base - share_trade_cf),
        services_share_change=100.0 * (share_services_base - share_services_cf),
        nonemployment_share_change=-100.0 * (nonemp_share_cf - nonemp_share_base),
        sectoral_manuf_contrib=sectoral_manuf_contrib,
        sectoral_nonmanuf_contrib=sectoral_nonmanuf_contrib,
        regional_manuf_contrib=regional_manuf_contrib,
        regional_nonmanuf_contrib=regional_nonmanuf_contrib,
    )


# ---------------------------------------------------------------- welfare


def compute_welfare_effects(
    baseline_econ: BaselineEconomy,
    counterfactual: CounterfactualEconomy,
    L0_initial: np.ndarray,
    *,
    J: int = 22, R: int = 50,
    beta: float = BETA, nu: float = NU,
) -> WelfareEffects:
    """Consumption-equivalent welfare change per labor market.

    Paper eq. 28:

        log δ_{n,j,t} = (1-β) · Σ_{s≥t} β^(s-t) · log(rwage^(-1) / (μ_cf/μ_b)^(-ν))

    Aggregate welfare = L0-weighted mean of log δ at t = 1 (second
    quarter; the first quarter has values pinned to 1 by the boundary).
    """
    T = counterfactual.rwage.shape[2]
    mu_baseline = baseline_econ.series_mu[..., :T]
    mu_cf = counterfactual.mu
    rwage_cf = counterfactual.rwage             # (R, J+1, T)

    # Extract diagonal of μ per labor market (the staying probability).
    diag_b = np.zeros((R, J + 1, T))
    diag_cf = np.zeros((R, J + 1, T))
    for t in range(T - 1):
        diag_b[..., t] = np.diag(mu_baseline[..., t]).reshape(J + 1, R, order="F").T
        diag_cf[..., t] = np.diag(mu_cf[..., t]).reshape(J + 1, R, order="F").T
    diag_b[..., T - 1] = diag_b[..., T - 2]
    diag_cf[..., T - 1] = diag_cf[..., T - 2]

    # Cumulative product of rwage_cf up to t (= levrwage_kappa in MATLAB).
    levrwage_cf = np.cumprod(rwage_cf, axis=2)

    # Discount factors with a steady-state tail.
    betavec = np.empty((R, J + 1, T))
    betavec[..., 0] = 1.0
    for t in range(T - 1):
        betavec[..., t + 1] = betavec[..., t] * beta
    betavec[..., T - 1] = betavec[..., T - 1] / (1 - beta)

    # hatlevrwage = 1 / levrwage_cf, hatdiag_mu = 1 / (diag_cf / diag_b)
    hat_lev_rwage = 1.0 / levrwage_cf
    with np.errstate(divide="ignore", invalid="ignore"):
        hat_diag_mu = diag_b / diag_cf
    hat_diag_mu = np.where(np.isnan(hat_diag_mu) | np.isinf(hat_diag_mu), 1.0, hat_diag_mu)

    logdelta = np.full((R, J + 1, T), np.nan)
    for t in range(1, T):
        with np.errstate(invalid="ignore", divide="ignore"):
            integrand = np.log(hat_lev_rwage[..., t:] / (hat_diag_mu[..., t:] ** nu))
        weighted = betavec[..., t:] * np.where(np.isnan(integrand) | np.isinf(integrand), 0.0, integrand)
        logdelta[..., t] = (1 - beta) * weighted.sum(axis=2) / beta ** (t - 1)

    # Aggregate using initial labor distribution: L0 reshaped (R, J+1).
    L0 = L0_initial.reshape(J + 1, R, order="F").T   # (R, J+1)
    aggregate = float((L0 * logdelta[..., 1]).sum() * 100.0)

    return WelfareEffects(logdelta=logdelta, aggregate_welfare_pct=aggregate)
