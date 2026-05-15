"""Pandas DataFrame views over CDP solver outputs.

Each solver dataclass (``EquilibriumResult``, ``BaselineEconomy``,
``CounterfactualEconomy``, ``EmploymentEffects``, ``WelfareEffects``)
holds raw numpy tensors. The free functions here wrap those tensors
into pandas DataFrames indexed by sector / region / quarter names,
mirroring the pattern used by the ``cprhs/`` package's
``.regional_summary()`` / ``.sectoral_summary()`` methods.

Conventions:
* Single-quarter slices return a wide ``(sector, region)`` DataFrame.
* Multi-quarter series can return either a wide ``(sector × region,
  quarter)`` DataFrame or a long ``(sector, region, quarter, value)``
  DataFrame depending on what's most readable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qge.labels import (
    COUNTRIES, LABOR_MARKETS, REGIONS, SECTORS, US_STATES,
    quarter_labels,
)


# ---------------------------------------------------------------- single-time slices


def trade_flows_at(xbilat: np.ndarray, t: int) -> pd.DataFrame:
    """Bilateral trade flows at quarter ``t`` as a long-form DataFrame.

    ``xbilat`` has shape ``(J*N, N, time)`` or ``(J*N, N)`` if already
    sliced. Returns columns ``[sector, destination, source, value]``.
    """
    if xbilat.ndim == 3:
        arr = xbilat[..., t]
    else:
        arr = xbilat
    J, N = len(SECTORS), len(REGIONS)
    if arr.shape != (J * N, N):
        raise ValueError(f"xbilat slice shape {arr.shape} ≠ ({J*N}, {N})")
    arr_3d = arr.reshape(J, N, N)
    return _long_form_3d(
        arr_3d,
        ("sector", SECTORS), ("destination", REGIONS), ("source", REGIONS),
    )


def trade_shares_at(pi: np.ndarray, t: int) -> pd.DataFrame:
    """Same as trade_flows_at but for trade shares (``Din`` / ``pi``)."""
    return trade_flows_at(pi, t)


def wages_at(wages: np.ndarray, t: int) -> pd.DataFrame:
    """Wage change at quarter ``t`` as a wide ``(sector × region)`` DataFrame."""
    if wages.ndim == 3:
        arr = wages[..., t]
    else:
        arr = wages
    return pd.DataFrame(arr, index=list(SECTORS), columns=list(REGIONS))


def labor_at(L: np.ndarray, t: int) -> pd.DataFrame:
    """US labor allocation at quarter ``t``: ``(labor market × state)`` wide.

    Accepts shape ``(J+1, R, time)`` or ``(J+1, R)``. Row 0 is
    non-employment.
    """
    if L.ndim == 3:
        arr = L[..., t]
    else:
        arr = L
    return pd.DataFrame(arr, index=list(LABOR_MARKETS), columns=list(US_STATES))


# ---------------------------------------------------------------- time series


def wage_series(wages: np.ndarray, region: str | None = None) -> pd.DataFrame:
    """Wage time series. ``(J, N, time)`` → wide ``(sector, quarter)`` for one
    region, or ``(sector × region, quarter)`` if ``region`` is None."""
    J, N, T = wages.shape
    quarters = quarter_labels(n_quarters=T)
    if region is not None:
        n = REGIONS.index(region)
        return pd.DataFrame(wages[:, n, :], index=list(SECTORS), columns=list(quarters))
    idx = pd.MultiIndex.from_product([SECTORS, REGIONS], names=["sector", "region"])
    return pd.DataFrame(wages.reshape(J * N, T), index=idx, columns=list(quarters))


def trade_share_series(pi: np.ndarray, sector: str, destination: str) -> pd.DataFrame:
    """For one (sector, destination) pair, return ``(source, quarter)`` shares."""
    j = SECTORS.index(sector)
    d = REGIONS.index(destination)
    N = len(REGIONS)
    T = pi.shape[2]
    quarters = quarter_labels(n_quarters=T)
    # pi has shape (J*N, N, T). Row j*N + d is the destination block in sector j.
    return pd.DataFrame(pi[j * N + d, :, :], index=list(REGIONS), columns=list(quarters))


def employment_series(L: np.ndarray) -> pd.DataFrame:
    """Labor evolution as a long-form DataFrame.

    ``L`` has shape ``(J+1, R, time)``. Returns columns
    ``[labor_market, state, quarter, value]``.
    """
    J1, R, T = L.shape
    return _long_form_3d(
        L,
        ("labor_market", LABOR_MARKETS),
        ("state", US_STATES),
        ("quarter", quarter_labels(n_quarters=T)),
    )


# ---------------------------------------------------------------- aggregations


def manufacturing_share_series(L: np.ndarray) -> pd.Series:
    """Aggregate share of manufacturing employment over time.

    Manufacturing = sectors 0..11 (labor-market rows 1..12).
    """
    J1, R, T = L.shape
    quarters = quarter_labels(n_quarters=T)
    employed = L[1:, :, :].sum(axis=(0, 1))               # (T,)
    manuf = L[1:13, :, :].sum(axis=(0, 1))                  # (T,)
    share = np.where(employed > 0, manuf / employed, np.nan)
    return pd.Series(share, index=list(quarters), name="manuf_share")


def nonemployment_share_series(L: np.ndarray) -> pd.Series:
    """Aggregate non-employment share of total US population over time."""
    J1, R, T = L.shape
    quarters = quarter_labels(n_quarters=T)
    total = L.sum(axis=(0, 1))
    nonemp = L[0, :, :].sum(axis=0)
    share = np.where(total > 0, nonemp / total, np.nan)
    return pd.Series(share, index=list(quarters), name="nonemp_share")


# ---------------------------------------------------------------- effects DataFrames


def employment_effects_dataframes(effects) -> dict[str, pd.DataFrame | pd.Series]:
    """Wrap EmploymentEffects into labeled pandas containers.

    Returns a dict with:
    * ``aggregate`` — pd.Series of the four big share-change scalars
    * ``sectoral_manuf`` — pd.Series (sector → % contribution to mfg decline)
    * ``sectoral_nonmanuf`` — pd.Series (sector → % contribution to non-mfg rise)
    * ``regional_manuf`` — pd.Series (state → % contribution to mfg decline)
    * ``regional_nonmanuf`` — pd.Series (state → % contribution to non-mfg rise)
    """
    return {
        "aggregate": pd.Series({
            "manufacturing": effects.manuf_share_change,
            "non-manufacturing": effects.nonmanuf_share_change,
            "construction": effects.construction_share_change,
            "trade": effects.trade_share_change,
            "services": effects.services_share_change,
            "non-employment": effects.nonemployment_share_change,
        }, name="share change (pp)"),
        "sectoral_manuf": pd.Series(
            effects.sectoral_manuf_contrib,
            index=list(SECTORS[:12]), name="mfg sector contribution (%)",
        ),
        "sectoral_nonmanuf": pd.Series(
            effects.sectoral_nonmanuf_contrib,
            index=list(SECTORS[12:22]), name="non-mfg sector contribution (%)",
        ),
        "regional_manuf": pd.Series(
            effects.regional_manuf_contrib,
            index=list(US_STATES), name="state contribution to mfg decline (%)",
        ),
        "regional_nonmanuf": pd.Series(
            effects.regional_nonmanuf_contrib,
            index=list(US_STATES), name="state contribution to non-mfg rise (%)",
        ),
    }


def welfare_logdelta_at(welfare, t: int) -> pd.DataFrame:
    """Per-labor-market log-welfare change at quarter ``t`` as ``(state × labor market)``."""
    return pd.DataFrame(
        welfare.logdelta[..., t] * 100.0,
        index=list(US_STATES),
        columns=list(LABOR_MARKETS),
    )


def welfare_summary(welfare) -> pd.Series:
    """Aggregate welfare summary."""
    return pd.Series({
        "aggregate (%)": welfare.aggregate_welfare_pct,
        "max gain (%)": float(np.nanmax(welfare.logdelta[..., 1]) * 100.0),
        "max loss (%)": float(np.nanmin(welfare.logdelta[..., 1]) * 100.0),
    }, name="welfare")


# ---------------------------------------------------------------- internals


def _long_form_3d(
    arr: np.ndarray, *axes: tuple[str, tuple[str, ...]]
) -> pd.DataFrame:
    """3-D array → long-form (one row per cell). Vectorized."""
    if arr.shape != tuple(len(labels) for _, labels in axes):
        raise ValueError(f"shape {arr.shape} ≠ {tuple(len(l) for _, l in axes)}")
    idx = np.indices(arr.shape).reshape(arr.ndim, -1)
    cols = {name: np.asarray(labels)[idx[i]] for i, (name, labels) in enumerate(axes)}
    cols["value"] = arr.ravel()
    return pd.DataFrame(cols)
