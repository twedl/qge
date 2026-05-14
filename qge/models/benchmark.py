"""Benchmark CPRHS model — port of CPRHS_Benchmark.m + the shock drivers.

Public API:
    compute_baseline()              CPRHS_Benchmark.m
    compute_regional_shock(region)  Regional_shocks_Benchmark.m, one region
    compute_sectoral_shock(sector)  Sectoral_shocks_Benchmark.m, one sector
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from qge.elasticities import ElasticityRow, regional_elasticities, sectoral_elasticities
from qge.helpers import (
    Dinprime,
    GDP,
    GMC,
    GOTFP,
    Lchange,
    P_h_om,
    _exports_by_source,
    expenditure,
    neweq,
)
from qge.io import RawInputs, load_base_year, load_raw_inputs


# ---------------------------------------------------------------- result types


@dataclass
class BenchmarkResult:
    """Output of compute_baseline — mirrors *_RS variables saved by MATLAB."""

    Ln: np.ndarray
    xbilat: np.ndarray
    VAR: np.ndarray
    VAL: np.ndarray
    Ljn: np.ndarray
    Chi: float
    Chin: np.ndarray
    LnIn: np.ndarray
    Sn: np.ndarray
    Bn: np.ndarray  # = TD_out of neweq (MATLAB driver renames it on save)


@dataclass(frozen=True)
class RegionalSweepResult:
    """Output of regional_sweep — 50 shocks plus 50 elasticity rows."""

    shocks: list  # list[BenchmarkShockResult]
    elasticities: list  # list[ElasticityRow]


@dataclass(frozen=True)
class SectoralSweepResult:
    """Output of sectoral_sweep — 26 shocks plus 26 elasticity rows."""

    shocks: list  # list[BenchmarkShockResult]
    elasticities: list  # list[ElasticityRow]


@dataclass
class BenchmarkShockResult:
    """Output of compute_*_shock — mirrors the workspace dumped by MATLAB."""

    TFP_hat: float
    GDP_hat: float
    V_hat: float
    L_hat: np.ndarray  # (N,)
    Ljn_hat: np.ndarray  # (J, N)
    TFPj: np.ndarray  # (J,)
    TFPn: np.ndarray  # (N,)
    GDPj: np.ndarray  # (J,)
    GDPn: np.ndarray  # (N,)
    P_index_hat: np.ndarray  # (N,)
    # Elasticity normalizers
    Yn: np.ndarray
    Y: float
    Yj: np.ndarray
    VAn0: np.ndarray
    VA0: float
    VAj0: np.ndarray
    # Equilibrium state
    om: np.ndarray
    Dinp: np.ndarray
    Xp: np.ndarray
    iterations: int


# ---------------------------------------------------------------- private


@dataclass(frozen=True)
class _Calibration:
    T: np.ndarray
    B: np.ndarray
    G: np.ndarray
    G_3d: np.ndarray
    gamma: np.ndarray
    alphas: np.ndarray
    io: np.ndarray
    J: int
    N: int


@dataclass(frozen=True)
class _XbilatState:
    Bn: np.ndarray
    X0: np.ndarray
    Din: np.ndarray
    VAL: np.ndarray
    VAR: np.ndarray
    Chin: np.ndarray
    Sn: np.ndarray
    LnIn: np.ndarray
    VALjn0: np.ndarray
    Exjn0: np.ndarray
    Ln: np.ndarray  # state-level Ln in effect for this run


@dataclass
class _LoopState:
    om: np.ndarray
    L_hat: np.ndarray
    phat: np.ndarray
    c: np.ndarray
    Dinp: np.ndarray
    Xp: np.ndarray
    V_hat: float
    P_index_hat: np.ndarray
    iterations: int


def _build_calibration(raw: RawInputs) -> _Calibration:
    J, N = raw.J, raw.N
    IO_norm = raw.IO / raw.IO.sum(axis=0, keepdims=True)
    # Block n[j, k] = (1 - gamma[k, n]) * IO_norm[j, k]
    G_3d = (1 - raw.gamma.T)[:, None, :] * IO_norm[None, :, :]
    G = G_3d.reshape(N * J, J)
    return _Calibration(
        T=raw.T, B=raw.B, G=G, G_3d=G_3d,
        gamma=raw.gamma, alphas=raw.alphas, io=raw.io, J=J, N=N,
    )


def _derive_from_xbilat(xbilat: np.ndarray, Ln: np.ndarray, cal: _Calibration) -> _XbilatState:
    J, N = cal.J, cal.N
    xbilat_3d = xbilat.reshape(J, N, N)
    M = xbilat_3d.sum(axis=2)
    E = xbilat_3d.sum(axis=1)
    Bn = E.sum(axis=0) - M.sum(axis=0)

    X0 = xbilat.sum(axis=1).reshape(J, N)
    Din = xbilat / xbilat.sum(axis=1, keepdims=True)

    Exjn0 = _exports_by_source(Din, X0, J, N)
    VALjn0 = cal.gamma * (1 - cal.B) * Exjn0

    aux2 = X0 - np.einsum("njk,kn->jn", cal.G_3d, E)
    b1 = cal.B[0, :]
    VAL = (1 - b1) * (Bn + aux2.sum(axis=0))
    VAR = (b1 / (1 - b1)) * VAL

    Chin = float(np.sum(cal.io * VAR)) * Ln
    Sn = Bn - cal.io * VAR + Chin
    LnIn = VAL + VAR + Chin - cal.io * VAR - Sn

    return _XbilatState(
        Bn=Bn, X0=X0, Din=Din, VAL=VAL, VAR=VAR,
        Chin=Chin, Sn=Sn, LnIn=LnIn,
        VALjn0=VALjn0, Exjn0=Exjn0, Ln=Ln,
    )


def _run_outer_loop(
    state: _XbilatState,
    cal: _Calibration,
    kappa_hat: np.ndarray,
    lambda_hat: np.ndarray,
    Snp,
    *,
    tol: float,
    vfactor: float = -0.1,
    maxit: int = 1_000_000,
    verbose: bool = False,
) -> _LoopState:
    J, N = cal.J, cal.N
    om = np.ones(N)
    L_hat = np.ones(N)
    phat = c = Dinp = Xp = None
    V_hat = 0.0
    P_index_hat = np.ones(N)
    ommax = np.inf
    it = 0
    while it < maxit and ommax > tol:
        phat, c = P_h_om(om, kappa_hat, lambda_hat, cal.T, cal.B, cal.G,
                         cal.gamma, state.Din, J, N)
        Dinp = Dinprime(state.Din, kappa_hat, lambda_hat, c, phat,
                        cal.T, J, N, cal.gamma)
        L_hat, V_hat, P_index_hat = Lchange(
            om, phat, cal.alphas, cal.B, L_hat, state.Ln, state.LnIn,
            state.Bn, Snp, state.VAR, cal.io,
        )
        Xp = expenditure(J, N, cal.alphas, cal.B, cal.G, Dinp, om, L_hat,
                         state.Ln, Snp, state.VAR, state.VAL, cal.io)
        omef0 = GMC(Xp, Dinp, J, N, cal.B, cal.gamma, state.LnIn, L_hat,
                    state.VAR, state.VAL)
        om1 = om * (1 + vfactor * (om - omef0) / om)
        ommax = float(np.sum(np.abs(om1 - om)))
        om = om1
        it += 1
        if verbose and (it % 50 == 0 or ommax <= tol):
            print(f"  outer iter {it:5d}  ||Δom||_1 = {ommax:.3e}")
    if ommax > tol:
        raise RuntimeError(
            f"Outer loop did not converge in {it} iters; ||Δom||_1={ommax:.3e}"
        )
    return _LoopState(
        om=om, L_hat=L_hat, phat=phat, c=c, Dinp=Dinp, Xp=Xp,
        V_hat=V_hat, P_index_hat=P_index_hat, iterations=it,
    )


def _post_shock_accounting(
    loop: _LoopState, state: _XbilatState, cal: _Calibration,
) -> BenchmarkShockResult:
    J, N = cal.J, cal.N
    b1 = cal.B[0, :]
    wf0 = loop.om * (loop.L_hat ** (-b1))

    Exjn = _exports_by_source(loop.Dinp, loop.Xp, J, N)
    Exjn0 = state.Exjn0

    VALjn = cal.gamma * (1 - cal.B) * Exjn0
    VARjn = (cal.B / (1 - cal.B)) * VALjn
    Ljn_hat = (1.0 / (VALjn * wf0[None, :])) * cal.gamma * (1 - cal.B) * Exjn

    TFP_hat, TFPj, TFPn, _, _, Yj, Yn, Y = GOTFP(loop.c, loop.phat, Exjn0, J, N)
    GDP_hat, GDPj, GDPn, _, VAj0, VAn0, VA0 = GDP(
        loop.phat, wf0, Ljn_hat, VALjn, VARjn, J, N
    )

    return BenchmarkShockResult(
        TFP_hat=TFP_hat, GDP_hat=GDP_hat, V_hat=loop.V_hat,
        L_hat=loop.L_hat, Ljn_hat=Ljn_hat,
        TFPj=TFPj, TFPn=TFPn, GDPj=GDPj, GDPn=GDPn,
        P_index_hat=loop.P_index_hat,
        Yn=Yn, Y=Y, Yj=Yj,
        VAn0=VAn0, VA0=VA0, VAj0=VAj0,
        om=loop.om, Dinp=loop.Dinp, Xp=loop.Xp,
        iterations=loop.iterations,
    )


def _baseline_quantities(
    baseline: Optional[BenchmarkResult],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (xbilat, Ln, Ljn) from a user-provided baseline or the MATLAB golden state."""
    if baseline is not None:
        return (
            baseline.xbilat,
            np.asarray(baseline.Ln).ravel(),
            np.asarray(baseline.Ljn),
        )
    gold = load_base_year("Benchmark")
    return gold["xbilat_RS"], gold["Ln_RS"].ravel(), gold["Ljn_RS"]


# ---------------------------------------------------------------- public API


def compute_baseline(
    raw: Optional[RawInputs] = None,
    *,
    tol: float = 1e-12,
    vfactor: float = -0.1,
    maxit: int = 1_000_000,
    verbose: bool = False,
) -> BenchmarkResult:
    """Compute the Benchmark baseline equilibrium from raw inputs.

    Port of CPRHS_Benchmark.m end-to-end.
    """
    if raw is None:
        raw = load_raw_inputs()
    cal = _build_calibration(raw)
    Ln = raw.L_j_n.sum(axis=0) / raw.L_j_n.sum()
    state = _derive_from_xbilat(raw.xbilat, Ln, cal)

    Snp = np.zeros(cal.N)
    kappa_hat = np.ones((cal.J * cal.N, cal.N))
    lambda_hat = np.ones((cal.J, cal.N))
    loop = _run_outer_loop(
        state, cal, kappa_hat, lambda_hat, Snp,
        tol=tol, vfactor=vfactor, maxit=maxit, verbose=verbose,
    )

    # Implied initial sector-state employment for neweq.
    w0 = state.VAL / Ln
    Ljn0 = state.VALjn0 / w0[None, :]
    wf0 = loop.om * (loop.L_hat ** (-cal.B[0, :]))

    out = neweq(
        cal.J, cal.N, loop.Xp, loop.Dinp, cal.G, cal.B, cal.gamma,
        Ljn0, wf0, state.VALjn0, cal.io, loop.L_hat, Ln,
    )
    return BenchmarkResult(
        Ln=out["Ln"], xbilat=out["xbilat"], VAR=out["VAR"], VAL=out["VAL"],
        Ljn=out["Ljn"], Chi=out["Chi"], Chin=out["Chin"], LnIn=out["LnIn"],
        Sn=out["Sn"], Bn=out["TD"],
    )


def _run_shock(
    lambda_hat: np.ndarray,
    *,
    baseline: Optional[BenchmarkResult],
    raw: Optional[RawInputs],
    tol: float,
    vfactor: float,
    maxit: int,
    verbose: bool,
) -> BenchmarkShockResult:
    if raw is None:
        raw = load_raw_inputs()
    cal = _build_calibration(raw)
    xbilat, Ln, _ = _baseline_quantities(baseline)
    state = _derive_from_xbilat(xbilat, Ln, cal)
    kappa_hat = np.ones((cal.J * cal.N, cal.N))
    loop = _run_outer_loop(
        state, cal, kappa_hat, lambda_hat, Snp=0.0,  # MATLAB sets Sn = 0; Snp = Sn
        tol=tol, vfactor=vfactor, maxit=maxit, verbose=verbose,
    )
    return _post_shock_accounting(loop, state, cal)


def compute_regional_shock(
    region: int,
    *,
    baseline: Optional[BenchmarkResult] = None,
    raw: Optional[RawInputs] = None,
    shock: float = 1.1,
    tol: float = 1e-12,
    vfactor: float = -0.1,
    maxit: int = 1_000_000,
    verbose: bool = False,
) -> BenchmarkShockResult:
    """TFP shock of factor `shock` applied to every sector in `region`.

    Port of Regional_shocks_Benchmark.m for one region.

    Parameters
    ----------
    region : int (0-indexed)
        State index, 0..49. MATLAB region 1 (Alabama) is Python region 0.
    baseline : BenchmarkResult or None
        If None, loads the MATLAB golden Base_Year_Benchmark.mat.
    shock : float
        Productivity factor (1.1 = +10%).
    """
    if raw is None:
        raw = load_raw_inputs()
    lambda_hat = np.ones((raw.J, raw.N))
    lambda_hat[:, region] = shock
    return _run_shock(
        lambda_hat, baseline=baseline, raw=raw,
        tol=tol, vfactor=vfactor, maxit=maxit, verbose=verbose,
    )


def compute_sectoral_shock(
    sector: int,
    *,
    baseline: Optional[BenchmarkResult] = None,
    raw: Optional[RawInputs] = None,
    shock: float = 1.1,
    tol: float = 1e-8,
    vfactor: float = -0.1,
    maxit: int = 1_000_000,
    verbose: bool = False,
) -> BenchmarkShockResult:
    """TFP shock of factor `shock` applied to `sector` across every state.

    Port of Sectoral_shocks_Benchmark.m for one sector. MATLAB default tol is 1e-8.

    Parameters
    ----------
    sector : int (0-indexed)
        Sector index, 0..25. MATLAB sector 11 (Computers and Electronics)
        is Python sector 10.
    """
    if raw is None:
        raw = load_raw_inputs()
    lambda_hat = np.ones((raw.J, raw.N))
    lambda_hat[sector, :] = shock
    return _run_shock(
        lambda_hat, baseline=baseline, raw=raw,
        tol=tol, vfactor=vfactor, maxit=maxit, verbose=verbose,
    )


def regional_sweep(
    *,
    baseline: Optional[BenchmarkResult] = None,
    raw: Optional[RawInputs] = None,
    shock: float = 1.1,
    tol: float = 1e-12,
    vfactor: float = -0.1,
    maxit: int = 1_000_000,
    verbose: bool = False,
) -> RegionalSweepResult:
    """Sweep a 10% TFP shock over all 50 US states and return aggregate elasticities.

    Mirrors a full pass of Regional_shocks_Benchmark.m followed by
    Aggregate_elasticities_regional_shocks.m. Calibration and the xbilat-derived
    state are built once and reused across the 50 shocks.

    Wall-clock: ~10 minutes on a laptop (each shock ~10s at tol=1e-12).
    """
    if raw is None:
        raw = load_raw_inputs()
    cal = _build_calibration(raw)
    xbilat, Ln, _ = _baseline_quantities(baseline)
    state = _derive_from_xbilat(xbilat, Ln, cal)
    kappa_hat = np.ones((cal.J * cal.N, cal.N))

    shocks: list[BenchmarkShockResult] = []
    elast_rows: list[ElasticityRow] = []
    for region in range(cal.N):
        lambda_hat = np.ones((cal.J, cal.N))
        lambda_hat[:, region] = shock
        loop = _run_outer_loop(
            state, cal, kappa_hat, lambda_hat, Snp=0.0,
            tol=tol, vfactor=vfactor, maxit=maxit,
        )
        sres = _post_shock_accounting(loop, state, cal)
        shocks.append(sres)
        row = regional_elasticities(sres, region=region, Ln=Ln)
        elast_rows.append(row)
        if verbose:
            print(
                f"  region {region + 1:2d}/{cal.N}  iters={sres.iterations:4d}  "
                f"TFP={row.TFP:+.4f}  GDP={row.GDP:+.4f}  welfare={row.welfare:+.4f}"
            )
    return RegionalSweepResult(shocks=shocks, elasticities=elast_rows)


def sectoral_sweep(
    *,
    baseline: Optional[BenchmarkResult] = None,
    raw: Optional[RawInputs] = None,
    shock: float = 1.1,
    tol: float = 1e-8,
    vfactor: float = -0.1,
    maxit: int = 1_000_000,
    verbose: bool = False,
) -> SectoralSweepResult:
    """Sweep a 10% TFP shock over all 26 sectors and return aggregate elasticities.

    Mirrors a full pass of Sectoral_shocks_Benchmark.m followed by
    Aggregate_elasticities_sectoral_shocks.m. Welfare uses baseline `Ljn` for
    the denominator (sum over states gives sector employment share).

    Wall-clock: ~2-3 minutes on a laptop (each shock ~5s at MATLAB tol=1e-8).
    """
    if raw is None:
        raw = load_raw_inputs()
    cal = _build_calibration(raw)
    xbilat, Ln, Ljn = _baseline_quantities(baseline)
    state = _derive_from_xbilat(xbilat, Ln, cal)
    kappa_hat = np.ones((cal.J * cal.N, cal.N))

    shocks: list[BenchmarkShockResult] = []
    elast_rows: list[ElasticityRow] = []
    for sector in range(cal.J):
        lambda_hat = np.ones((cal.J, cal.N))
        lambda_hat[sector, :] = shock
        loop = _run_outer_loop(
            state, cal, kappa_hat, lambda_hat, Snp=0.0,
            tol=tol, vfactor=vfactor, maxit=maxit,
        )
        sres = _post_shock_accounting(loop, state, cal)
        shocks.append(sres)
        row = sectoral_elasticities(sres, sector=sector, Ljn=Ljn)
        elast_rows.append(row)
        if verbose:
            print(
                f"  sector {sector + 1:2d}/{cal.J}  iters={sres.iterations:4d}  "
                f"TFP={row.TFP:+.4f}  GDP={row.GDP:+.4f}  welfare={row.welfare:+.4f}"
            )
    return SectoralSweepResult(shocks=shocks, elasticities=elast_rows)
