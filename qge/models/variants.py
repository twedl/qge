"""NS / NR / NRNS baseline equilibria.

Each variant is a structural restriction of the Benchmark model:

* **NS**  (No Sectoral linkages) — γ → 1, G → 0 in the outer loop.
  Initial state is computed with the original IO structure; the loop runs
  with no input-output linkages.

* **NR**  (No Regional trade) — `kappa_hat` shocked to +∞ off-diagonal for
  the 15 tradable sectors, so interstate trade in those sectors is shut
  off. Trade deficits, portfolio income (`io`), and `Sn` are zeroed.

* **NRNS** — starts from NR's post-equilibrium economy, then additionally
  sets γ → 1, G → 0 like NS.

All three reuse the Benchmark outer loop (`P_h_om`-based) and `neweq`
postprocessing. The CPRHS shock-script variants `P_h_omNI` is only needed
for the counterfactuals on these baselines, not the baselines themselves.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np

from qge.helpers import neweq
from qge.io import RawInputs, load_base_year, load_raw_inputs
from qge.models.benchmark import (
    BenchmarkResult,
    _build_calibration,
    _Calibration,
    _derive_from_xbilat,
    _run_outer_loop,
    _XbilatState,
)


def _no_sectoral_linkages(cal: _Calibration) -> _Calibration:
    """Variant cal with γ → 1 and G → 0 (used by NS and NRNS).

    Note this only removes the sectoral input-output linkages; it does not
    touch the global-portfolio scalar `cal.io` — those are independent
    structural restrictions.
    """
    gamma = np.ones_like(cal.gamma)
    G_3d = np.zeros_like(cal.G_3d)
    G = np.zeros_like(cal.G)
    return replace(cal, gamma=gamma, G=G, G_3d=G_3d)


def _zero_bn_val_var(state: _XbilatState, cal: _Calibration) -> tuple[np.ndarray, np.ndarray]:
    """Recompute (VAL, VAR) under Bn=0 by subtracting the Bn contribution.

    The MATLAB NR/NRNS scripts zero out the trade-deficit term before
    computing VAL — equivalently, VAL_new = VAL - (1-b)*Bn.
    """
    b = cal.B
    VAL = state.VAL - (1 - b) * state.Bn
    VAR = (b / (1 - b)) * VAL
    return VAL, VAR


def _no_trade_kappa(tradable_idx, J: int, N: int) -> np.ndarray:
    """`kappa_hat` that shuts off interstate trade for the given tradable sectors.

    Each tradable sector gets an identity-with-∞-off-diagonal block (only
    intra-state trade allowed); all other sectors keep kappa = 1 (no shock).
    The set of tradable sectors is supplied explicitly by index, dropping the
    CPRHS "first-15" convention.
    """
    kappa = np.ones((J, N, N))
    if len(tradable_idx) > 0:
        diag_inf = np.where(np.eye(N) == 1, 1.0, np.inf)
        kappa[list(tradable_idx)] = diag_inf
    return kappa.reshape(J * N, N)


def _build_benchmark_result(loop, state, cal, Ljn_for_neweq, Ln) -> BenchmarkResult:
    """Common neweq postprocessing → BenchmarkResult."""
    wf0 = loop.om * (loop.L_hat ** (-cal.B))
    out = neweq(
        cal.J, cal.N, loop.Xp, loop.Dinp, cal.G, cal.B, cal.gamma,
        Ljn_for_neweq, wf0, state.VALjn0, cal.io, loop.L_hat, Ln,
    )
    return BenchmarkResult(
        Ln=out["Ln"], xbilat=out["xbilat"], VAR=out["VAR"], VAL=out["VAL"],
        Ljn=out["Ljn"], Chi=out["Chi"], Chin=out["Chin"], LnIn=out["LnIn"],
        Sn=out["Sn"], Bn=out["TD"],
        sectors=cal.sectors, regions=cal.regions,
    )


def compute_baseline_ns(
    *,
    raw: Optional[RawInputs] = None,
    benchmark: Optional[BenchmarkResult] = None,
    tol: float = 1e-12,
    vfactor: float = -0.1,
    maxit: int = 1_000_000,
    verbose: bool = False,
) -> BenchmarkResult:
    """No-sectoral-linkages baseline (CPRHS_NS.m).

    Starts from the Benchmark baseline, then runs the outer loop with γ=1
    and G=0. Initial state (Bn, VAL, VAR, …) is computed using the original
    IO structure; only the iteration is no-IO.
    """
    if raw is None:
        raw = load_raw_inputs()
    cal = _build_calibration(raw)

    if benchmark is not None:
        xbilat = benchmark.xbilat
        Ln = np.asarray(benchmark.Ln).ravel()
        Ljn_for_neweq = np.asarray(benchmark.Ljn)
    else:
        gold = load_base_year("Benchmark")
        xbilat = gold["xbilat_RS"]
        Ln = gold["Ln_RS"].ravel()
        Ljn_for_neweq = gold["Ljn_RS"]

    state = _derive_from_xbilat(xbilat, Ln, cal)
    cal_ns = _no_sectoral_linkages(cal)

    Snp = np.zeros(cal.N)
    kappa_hat = np.ones((cal.J * cal.N, cal.N))
    lambda_hat = np.ones((cal.J, cal.N))
    loop = _run_outer_loop(
        state, cal_ns, kappa_hat, lambda_hat, Snp,
        tol=tol, vfactor=vfactor, maxit=maxit, verbose=verbose,
    )
    return _build_benchmark_result(loop, state, cal_ns, Ljn_for_neweq, Ln)


def compute_baseline_nr(
    *,
    tradable: list[str],
    raw: Optional[RawInputs] = None,
    benchmark: Optional[BenchmarkResult] = None,
    tol: float = 1e-12,
    vfactor: float = -0.1,
    maxit: int = 1_000_000,
    verbose: bool = False,
) -> BenchmarkResult:
    """No-regional-trade baseline (CPRHS_NR.m).

    Interstate trade is shut off for the sectors named in ``tradable``
    (kappa_hat → ∞ off-diagonal); non-tradable sectors are unchanged.
    Trade deficits, portfolio income, and `Sn` are zeroed; `LnIn = VAL + VAR`.

    Parameters
    ----------
    tradable : list of sector names
        Sectors whose interstate trade is shut off. Required (no default) so
        the tradable/non-tradable split is always an explicit choice. For
        CPRHS: pass ``raw.sectors[:15]`` (paper's first-15 convention).
    """
    if raw is None:
        raw = load_raw_inputs()
    cal = _build_calibration(raw)

    if benchmark is not None:
        xbilat = benchmark.xbilat
        Ln = np.asarray(benchmark.Ln).ravel()
    else:
        gold = load_base_year("Benchmark")
        xbilat = gold["xbilat_RS"]
        Ln = gold["Ln_RS"].ravel()
    Ljn_for_neweq = raw.L_j_n  # NR uses the raw employment matrix

    state = _derive_from_xbilat(xbilat, Ln, cal)
    VAL_nr, VAR_nr = _zero_bn_val_var(state, cal)
    state = replace(
        state,
        Bn=np.zeros(cal.N),
        VAL=VAL_nr,
        VAR=VAR_nr,
        LnIn=VAL_nr + VAR_nr,
    )
    cal_nr = replace(cal, io=np.zeros_like(cal.io))

    unknown = [s for s in tradable if s not in raw.sectors]
    if unknown:
        raise ValueError(f"unknown tradable sector(s): {unknown}")
    tradable_idx = [raw.sectors.index(s) for s in tradable]
    kappa_hat = _no_trade_kappa(tradable_idx, cal.J, cal.N)
    lambda_hat = np.ones((cal.J, cal.N))
    loop = _run_outer_loop(
        state, cal_nr, kappa_hat, lambda_hat, Snp=0.0,
        tol=tol, vfactor=vfactor, maxit=maxit, verbose=verbose,
    )
    return _build_benchmark_result(loop, state, cal_nr, Ljn_for_neweq, Ln)


def compute_baseline_nrns(
    *,
    raw: Optional[RawInputs] = None,
    nr_baseline: Optional[BenchmarkResult] = None,
    tol: float = 1e-12,
    vfactor: float = -0.1,
    maxit: int = 1_000_000,
    verbose: bool = False,
) -> BenchmarkResult:
    """No-regional-trade + no-sectoral-linkages baseline (CPRHS_NRNS.m).

    Loads the NR post-equilibrium state, then runs the outer loop with
    γ=1, G=0, io=0, and trade-balance closures. Despite being layered on
    NR, kappa_hat is back to all-ones — the block-diagonal trade pattern
    is already baked into `xbilat`.
    """
    if raw is None:
        raw = load_raw_inputs()
    cal = _build_calibration(raw)

    if nr_baseline is not None:
        xbilat = nr_baseline.xbilat
        Ln = np.asarray(nr_baseline.Ln).ravel()
    else:
        gold = load_base_year("NR")
        xbilat = gold["xbilat_NRS"]
        Ln = gold["Ln_NRS"].ravel()
    Ljn_for_neweq = raw.L_j_n  # NRNS chains through NR which uses raw L_j_n

    state = _derive_from_xbilat(xbilat, Ln, cal)
    VAL, VAR = _zero_bn_val_var(state, cal)
    state = replace(
        state,
        Bn=np.zeros(cal.N),
        VAL=VAL,
        VAR=VAR,
        LnIn=VAL + VAR,
    )
    cal_nrns = _no_sectoral_linkages(replace(cal, io=np.zeros_like(cal.io)))

    Snp = 0.0
    kappa_hat = np.ones((cal.J * cal.N, cal.N))
    lambda_hat = np.ones((cal.J, cal.N))
    loop = _run_outer_loop(
        state, cal_nrns, kappa_hat, lambda_hat, Snp,
        tol=tol, vfactor=vfactor, maxit=maxit, verbose=verbose,
    )
    return _build_benchmark_result(loop, state, cal_nrns, Ljn_for_neweq, Ln)
