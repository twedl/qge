"""Compare AT-2016 vs AKT-2019 trade-cost regression specs on canada_2022.

Both papers invert the same Head-Ries τ̄ from bilateral trade shares. They
diverge in the regression that follows:

  AT-2016:    ln τ̄_{ni}^j = δ^j · ln(d̄_{ni}) + ι_n + η_i + ε
                where d̄ = d_{ni} / √(d_{nn} d_{ii}) (Mayer-Zignago normalized)

  AT+nbr:     ln τ̄_{ni}^j = δ^j · ln(d̄_{ni}) + α_N^j · N_{ni} + ι_n + η_i + ε
                hybrid — AT distance functional form + AKT neighbor dummy.
                Isolates the neighbor effect under log-distance.

  AKT-2019:   ln τ̄_{ni,t}^j = α_D^j · D_{ni} + α_N^j · N_{ni}
                              + γ_{n,t} + η_{i,t} + ε
                where D in 1000 km (linear), N is 0/1 shared-land-border dummy

canada_2022 is a single year, so AKT's importer-year/exporter-year FEs collapse
to importer/exporter FEs. The three regressions thus use the same FE structure
and the same Head-Ries τ̄ on the LHS; they differ only in distance functional
form and the presence of a neighbor dummy.

  AT → AT+nbr decomposition: contribution of the neighbor dummy under log d.
  AT+nbr → AKT decomposition: contribution of switching log d → linear D.

Outputs:
  * Per-sector α₁, α₂, δ, plus R² for the geographic-piece only fit
    (drop FEs to compare each spec's *structural* explanatory power).
  * Trade-weighted aggregate AKT non-geographic τ̄ vs AT symmetric non-distance
    τ̄. Both definitions: τ̄_total / τ̄_geo, weighted by raw bilateral trade.

Run from tw/:
    TW_CALIBRATION=canada_2022 uv run python scripts/compare_at_akt_regressions.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from qge.io import load_inputs  # noqa: E402
from scripts.build_trade_costs import (  # noqa: E402
    build_distance_matrix,
    head_ries_inversion,
    PROVINCE_LATLON,
)


# Land-border adjacency among the 11 regions. WLD inherits the USA's adjacency
# (consistent with the US-population-weighted centroid TW uses for WLD).
ADJACENCY: dict[str, set[str]] = {
    "AB":  {"BC", "SK", "WLD"},
    "BC":  {"AB", "WLD"},
    "MB":  {"ON", "SK", "WLD"},
    "NB":  {"NS", "QC", "WLD"},
    "NL":  {"QC"},
    "NS":  {"NB"},
    "ON":  {"MB", "QC", "WLD"},
    "PE":  set(),                     # island
    "QC":  {"NB", "NL", "ON", "WLD"},
    "SK":  {"AB", "MB", "WLD"},
    "WLD": {"AB", "BC", "MB", "NB", "ON", "QC", "SK"},
}


def build_neighbor_matrix(regions: list[str]) -> np.ndarray:
    N = len(regions)
    nbr = np.zeros((N, N), dtype=int)
    for n, rn in enumerate(regions):
        for i, ri in enumerate(regions):
            nbr[n, i] = int(ri in ADJACENCY.get(rn, set()))
    return nbr


def _ols_with_fe(y, X_struct, n_idx, i_idx, N):
    """OLS with structural regressors X_struct + importer FE + exporter FE.

    Returns (coefs_struct, ss_res, ss_tot, ss_struct_only).
    `ss_struct_only` = SSR from regressing y on X_struct + const only
    (no FEs), to gauge the structural piece's own explanatory power.
    """
    n_obs = len(y)
    k = X_struct.shape[1]
    # Full design: [X_struct, importer_dummies, exporter_dummies, const]
    fe_n = np.zeros((n_obs, N - 1))
    fe_i = np.zeros((n_obs, N - 1))
    for r in range(1, N):
        fe_n[n_idx == r, r - 1] = 1.0
        fe_i[i_idx == r, r - 1] = 1.0
    X_full = np.hstack([X_struct, fe_n, fe_i, np.ones((n_obs, 1))])
    beta, _, _, _ = np.linalg.lstsq(X_full, y, rcond=None)
    yhat = X_full @ beta
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    # Structural-only fit
    X_struct_only = np.hstack([X_struct, np.ones((n_obs, 1))])
    b_s, _, _, _ = np.linalg.lstsq(X_struct_only, y, rcond=None)
    yhat_s = X_struct_only @ b_s
    ss_struct_only = float(((y - yhat_s) ** 2).sum())
    return beta[:k], ss_res, ss_tot, ss_struct_only


def run_regressions(tau_bar, d_raw, d_bar, nbr, theta, regions, sectors):
    """Run AT, AT+nbr, and AKT regressions per sector. Returns coef + obs frames."""
    N, _, J = tau_bar.shape
    rows_coef = []
    rows_obs = []
    for j in range(J):
        obs = []
        for n in range(N):
            for i in range(N):
                if n == i:
                    continue
                tb = tau_bar[n, i, j]
                if np.isnan(tb) or tb <= 0:
                    continue
                obs.append((n, i, math.log(tb), math.log(d_bar[n, i]),
                            d_raw[n, i] / 1000.0, float(nbr[n, i])))
        if len(obs) < 2 * N:
            rows_coef.append((sectors[j], *([np.nan] * 11), len(obs)))
            continue
        a = np.array(obs)
        n_idx = a[:, 0].astype(int)
        i_idx = a[:, 1].astype(int)
        y     = a[:, 2]
        log_d_bar = a[:, 3]
        D_km1000  = a[:, 4]
        N_dummy   = a[:, 5]

        # AT: ln τ̄ ~ ln(d̄) + FEs
        c_at, ss_at, ss_tot, ss_struct_at = _ols_with_fe(
            y, log_d_bar.reshape(-1, 1), n_idx, i_idx, N)
        delta_at = float(c_at[0])

        # AT+nbr: ln τ̄ ~ ln(d̄) + N + FEs
        X_atn = np.column_stack([log_d_bar, N_dummy])
        c_atn, ss_atn, _, ss_struct_atn = _ols_with_fe(
            y, X_atn, n_idx, i_idx, N)
        delta_atn, an_atn = float(c_atn[0]), float(c_atn[1])

        # AKT: ln τ̄ ~ D + N + FEs
        X_akt = np.column_stack([D_km1000, N_dummy])
        c_akt, ss_akt, _, ss_struct_akt = _ols_with_fe(
            y, X_akt, n_idx, i_idx, N)
        ad_akt, an_akt = float(c_akt[0]), float(c_akt[1])

        rows_coef.append((
            sectors[j],
            delta_at, 1 - ss_at/ss_tot, 1 - ss_struct_at/ss_tot,
            delta_atn, an_atn, 1 - ss_atn/ss_tot, 1 - ss_struct_atn/ss_tot,
            ad_akt, an_akt, 1 - ss_akt/ss_tot, 1 - ss_struct_akt/ss_tot,
            theta[j], len(obs),
        ))

        for k in range(len(obs)):
            log_geo_at  = delta_at * log_d_bar[k]
            log_geo_atn = delta_atn * log_d_bar[k] + an_atn * N_dummy[k]
            log_geo_akt = ad_akt * D_km1000[k] + an_akt * N_dummy[k]
            log_tau     = y[k]
            rows_obs.append({
                "sector":      sectors[j],
                "n":           regions[n_idx[k]],
                "i":           regions[i_idx[k]],
                "tau_bar":     math.exp(log_tau),
                "tau_geo_at":  math.exp(log_geo_at),
                "tau_geo_atn": math.exp(log_geo_atn),
                "tau_geo_akt": math.exp(log_geo_akt),
                "tau_nd_at":   math.exp(log_tau - log_geo_at),
                "tau_ng_atn":  math.exp(log_tau - log_geo_atn),
                "tau_ng_akt":  math.exp(log_tau - log_geo_akt),
                "neighbor":    int(N_dummy[k]),
            })

    coef_tbl = pd.DataFrame(rows_coef, columns=[
        "sector",
        "delta_at",   "r2_at_full",    "r2_at_struct",
        "delta_atn",  "alpha_nbr_atn", "r2_atn_full",   "r2_atn_struct",
        "alpha_dist", "alpha_nbr_akt", "r2_akt_full",   "r2_akt_struct",
        "theta", "N",
    ])
    obs_tbl = pd.DataFrame(rows_obs)
    return coef_tbl, obs_tbl


def main() -> None:
    raw = load_inputs()
    sectors = list(raw.sectors)
    regions = list(raw.regions)
    N, J = len(regions), len(sectors)

    # Reshape PI back to (N, N, J) — destination n, source i, sector j.
    pi_3d = np.zeros((N, N, J))
    for i in range(N):
        for j in range(J):
            for n in range(N):
                pi_3d[n, i, j] = raw.PI[i * J + j, n]
    theta = raw.theta.ravel()

    # Distance and neighbor matrices
    d_raw, d_bar = build_distance_matrix(regions)
    nbr = build_neighbor_matrix(regions)

    print(f"Regions: {regions}")
    print(f"Neighbor pairs: {int(nbr.sum() // 2)} unordered "
          f"(including WLD-{{AB,BC,MB,NB,ON,QC,SK}})")

    tau_bar = head_ries_inversion(pi_3d, theta)

    coef, obs = run_regressions(tau_bar, d_raw, d_bar, nbr, theta,
                                 regions, sectors)

    # ── Per-sector coefficient table ─────────────────────────────────────
    print("\n" + "=" * 120)
    print("Per-sector coefficients (canada_2022, single year)")
    print("=" * 120)
    print(f"{'Sector':<23} {'θ':>5} | "
          f"{'AT δ':>7} | "
          f"{'AT+N δ':>7} {'AT+N α_N%':>10} | "
          f"{'AKT α_D':>9} {'α_D%/1k':>9} {'AKT α_N%':>9}")
    print("-" * 120)
    for _, r in coef.iterrows():
        if np.isnan(r.delta_at):
            print(f"{r.sector[:23]:<23} {r.theta:>5.1f} | (no off-diag trade)")
            continue
        a_n_atn_pct = (math.exp(r.alpha_nbr_atn) - 1.0) * 100
        a_d_pct = (math.exp(r.alpha_dist) - 1.0) * 100
        a_n_akt_pct = (math.exp(r.alpha_nbr_akt) - 1.0) * 100
        print(f"{r.sector[:23]:<23} {r.theta:>5.1f} | "
              f"{r.delta_at:>+7.3f} | "
              f"{r.delta_atn:>+7.3f} {a_n_atn_pct:>+9.1f}% | "
              f"{r.alpha_dist:>+9.3f} {a_d_pct:>+8.1f}% {a_n_akt_pct:>+8.1f}%")

    # ── Trade-weighted aggregates ────────────────────────────────────────
    # Build a weight from raw bilateral flows: w_{n,i,j} = π_{n,i,j} × I_n
    # (proxy for value of n's spending on i in sector j). We don't have I^j
    # so use π × I_n as a sector-uniform weight per pair.
    weights = []
    I_n = raw.I_baseline[0]
    sec_to_j = {s: k for k, s in enumerate(sectors)}
    reg_to_k = {r: k for k, r in enumerate(regions)}
    for _, r in obs.iterrows():
        n_k = reg_to_k[r["n"]]
        i_k = reg_to_k[r["i"]]
        j_k = sec_to_j[r["sector"]]
        weights.append(pi_3d[n_k, i_k, j_k] * I_n[n_k])
    obs["w"] = weights

    def _wmean_pct(col, mask=None):
        d = obs if mask is None else obs[mask]
        w = d["w"].to_numpy()
        v = d[col].to_numpy() - 1.0
        return float(np.average(v, weights=w)) * 100 if w.sum() > 0 else float("nan")

    interprov = obs["n"].ne("WLD") & obs["i"].ne("WLD")
    all_pairs = pd.Series(True, index=obs.index)

    print("\n" + "=" * 100)
    print("Trade-weighted aggregate τ̄ - 1 (% markup) by definition")
    print("=" * 100)
    print(f"{'Slice':<26} {'τ̄ obs':>9} {'AT geo':>9} {'AT+N geo':>10} {'AKT geo':>9} "
          f"{'AT NDist':>10} {'AT+N NG':>10} {'AKT NG':>9}")
    print("-" * 100)
    for label, mask in [("All pairs incl. WLD", all_pairs),
                         ("Interprovincial only", interprov)]:
        v_obs   = _wmean_pct("tau_bar",     mask)
        v_at_g  = _wmean_pct("tau_geo_at",  mask)
        v_atn_g = _wmean_pct("tau_geo_atn", mask)
        v_akt_g = _wmean_pct("tau_geo_akt", mask)
        v_at_n  = _wmean_pct("tau_nd_at",   mask)
        v_atn_n = _wmean_pct("tau_ng_atn",  mask)
        v_akt_n = _wmean_pct("tau_ng_akt",  mask)
        print(f"{label:<26} {v_obs:>+8.1f}% {v_at_g:>+8.1f}% {v_atn_g:>+9.1f}% "
              f"{v_akt_g:>+8.1f}% {v_at_n:>+9.1f}% {v_atn_n:>+9.1f}% "
              f"{v_akt_n:>+8.1f}%")

    # Decomposition of total AT→AKT divergence into neighbor + functional-form
    print("\n  Interprovincial NG decomposition (% markup):")
    m = interprov
    v_at_n  = _wmean_pct("tau_nd_at",  m)
    v_atn_n = _wmean_pct("tau_ng_atn", m)
    v_akt_n = _wmean_pct("tau_ng_akt", m)
    print(f"    AT non-distance                                    {v_at_n:>+7.1f}%")
    print(f"    + neighbor dummy (under log d̄):    Δ={v_atn_n - v_at_n:>+5.1f}pp  {v_atn_n:>+7.1f}%  (AT+N)")
    print(f"    + linear D (replacing log d̄):       Δ={v_akt_n - v_atn_n:>+5.1f}pp  {v_akt_n:>+7.1f}%  (AKT)")
    print(f"    Total AT → AKT divergence:                          "
          f"{v_akt_n - v_at_n:>+5.1f}pp")

    print("\n" + "=" * 100)
    print("Per-sector trade-weighted aggregates (interprovincial only)")
    print("=" * 100)
    print(f"{'Sector':<23} {'τ̄ obs':>9} {'AT NDist':>9} {'AT+N NG':>9} {'AKT NG':>9} "
          f"{'+N effect':>11} {'+linD eff':>11}")
    print("-" * 100)
    for sec in sectors:
        m = (obs["sector"] == sec) & interprov
        if m.sum() < 4:
            continue
        v_obs   = _wmean_pct("tau_bar",   m)
        v_at_n  = _wmean_pct("tau_nd_at", m)
        v_atn_n = _wmean_pct("tau_ng_atn",m)
        v_akt_n = _wmean_pct("tau_ng_akt",m)
        nbr_eff  = v_atn_n - v_at_n        # AT → AT+nbr
        lin_eff  = v_akt_n - v_atn_n       # AT+nbr → AKT
        print(f"{sec[:23]:<23} {v_obs:>+8.1f}% {v_at_n:>+8.1f}% {v_atn_n:>+8.1f}% "
              f"{v_akt_n:>+8.1f}% {nbr_eff:>+10.1f}pp {lin_eff:>+10.1f}pp")

    # ── Where do AT and AKT diverge most? ────────────────────────────────
    print("\n" + "=" * 100)
    print("Largest pair-level (AKT NG − AT NDist) divergences "
          "(interprovincial; sector-pair obs)")
    print("=" * 100)
    obs_ip = obs[interprov].copy()
    obs_ip["diff_ng_minus_nd"] = obs_ip["tau_ng_akt"] - obs_ip["tau_nd_at"]
    top = obs_ip.nlargest(8, "diff_ng_minus_nd")
    print(f"{'sector':<22} {'n→i':<12} {'nbr':>4} {'τ̄ obs':>8} "
          f"{'AT NDist':>10} {'AT+N NG':>10} {'AKT NG':>9} {'Δ':>8}")
    for _, r in top.iterrows():
        pair = f"{r['n']}→{r['i']}"
        print(f"{r.sector[:22]:<22} {pair:<12} {r.neighbor:>4} "
              f"{r.tau_bar-1:>+7.1%} {r.tau_nd_at-1:>+9.1%} "
              f"{r.tau_ng_atn-1:>+9.1%} {r.tau_ng_akt-1:>+8.1%} "
              f"{r.diff_ng_minus_nd:>+7.2f}")


if __name__ == "__main__":
    main()
