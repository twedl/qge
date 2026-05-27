"""AT-2016 vs AKT-2019 trade-cost regressions at the 29-sector + ATL schema.

Mirrors `compare_at_akt_regressions.py` (which runs on the 22-sector × 11-region
canada_2022 calibration) but builds the trade-share matrix directly from
provincial SUTs + ICIO 2022 at the finer 29-sector × 8-region schema. Outputs
the same three regression specs:

  AT-2016:  ln τ̄ ~ δ ln(d̄) + ι_n + η_i
  AT+N:     ln τ̄ ~ δ ln(d̄) + α_N N + ι_n + η_i
  AKT-2019: ln τ̄ ~ α_D D + α_N N + ι_n + η_i

Run from tw/:
    uv run python scripts/compare_at_akt_atlm.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from scripts.diagnose_positivity_atlm import (  # noqa: E402
    add_icio_wld_intra, build_flows, compute_shares,
)
from scripts.sector_schema_m import (  # noqa: E402
    REGIONS, SECTORS, THETA, WLD,
)
from scripts.build_trade_costs import (  # noqa: E402
    head_ries_inversion, haversine_km,
)


# 8-region centroid + area for distance matrix.
# ATL = population-weighted centroid of NB+NS+PE+NL (2022 pops ~770/970/155/520k).
REGION_LATLON: dict[str, tuple[float, float]] = {
    "AB":  (51.700, -114.000),
    "BC":  (49.250, -123.100),
    "MB":  (49.895,  -97.138),
    "ON":  (43.700,  -79.400),
    "QC":  (45.700,  -73.300),
    "SK":  (50.450, -104.617),
    "ATL": (45.900,  -62.060),
    "WLD": (39.500,  -98.350),
}

REGION_AREA_KM2: dict[str, float] = {
    "AB":    661_848,
    "BC":    944_735,
    "MB":    647_797,
    "ON":  1_076_395,
    "QC":  1_542_056,
    "SK":    651_036,
    # ATL = NB + NS + PE + NL = 72,908 + 55,284 + 5,660 + 405,212
    "ATL":   539_064,
    "WLD": 50_000_000,
}

# Land-border adjacency: ATL borders QC (NB-QC) and WLD (NB-Maine).
ADJACENCY: dict[str, set[str]] = {
    "AB":  {"BC", "SK", "WLD"},
    "BC":  {"AB", "WLD"},
    "MB":  {"ON", "SK", "WLD"},
    "ON":  {"MB", "QC", "WLD"},
    "QC":  {"ATL", "ON", "WLD"},
    "SK":  {"AB", "MB", "WLD"},
    "ATL": {"QC", "WLD"},
    "WLD": {"AB", "BC", "MB", "ON", "QC", "SK", "ATL"},
}


def build_distance_matrix(regions: list[str]) -> tuple[np.ndarray, np.ndarray]:
    N = len(regions)
    d = np.zeros((N, N))
    for n, rn in enumerate(regions):
        lat_n, lon_n = REGION_LATLON[rn]
        for i, ri in enumerate(regions):
            if n == i:
                d[n, i] = (2.0 / 3.0) * math.sqrt(REGION_AREA_KM2[rn] / math.pi)
            else:
                lat_i, lon_i = REGION_LATLON[ri]
                d[n, i] = haversine_km(lat_n, lon_n, lat_i, lon_i)
    d_bar = d / np.sqrt(np.outer(np.diag(d), np.diag(d)))
    return d, d_bar


def build_neighbor_matrix(regions: list[str]) -> np.ndarray:
    N = len(regions)
    nbr = np.zeros((N, N), dtype=int)
    for n, rn in enumerate(regions):
        for i, ri in enumerate(regions):
            nbr[n, i] = int(ri in ADJACENCY.get(rn, set()))
    return nbr


def _ols_with_fe(y, X_struct, n_idx, i_idx, N):
    """OLS with structural regressors + importer FE + exporter FE."""
    n_obs = len(y)
    k = X_struct.shape[1]
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
    return beta[:k], ss_res, ss_tot


def run_regressions(tau_bar, d_raw, d_bar, nbr, theta_vec, regions, sectors):
    """AT, AT+N, AKT regressions per sector → coef + obs frames."""
    N, _, J = tau_bar.shape
    rows_coef, rows_obs = [], []
    for j in range(J):
        obs = []
        for n in range(N):
            for i in range(N):
                if n == i: continue
                tb = tau_bar[n, i, j]
                if np.isnan(tb) or tb <= 0: continue
                obs.append((n, i, math.log(tb), math.log(d_bar[n, i]),
                            d_raw[n, i] / 1000.0, float(nbr[n, i])))
        if len(obs) < 2 * N:
            rows_coef.append((sectors[j], *([np.nan] * 6), theta_vec[j], len(obs)))
            continue
        a = np.array(obs)
        n_idx = a[:, 0].astype(int); i_idx = a[:, 1].astype(int)
        y = a[:, 2]
        log_d_bar = a[:, 3]; D_km1000 = a[:, 4]; N_dummy = a[:, 5]

        c_at, _, _ = _ols_with_fe(y, log_d_bar.reshape(-1, 1), n_idx, i_idx, N)
        delta_at = float(c_at[0])

        c_atn, _, _ = _ols_with_fe(y, np.column_stack([log_d_bar, N_dummy]),
                                    n_idx, i_idx, N)
        delta_atn, an_atn = float(c_atn[0]), float(c_atn[1])

        c_akt, _, _ = _ols_with_fe(y, np.column_stack([D_km1000, N_dummy]),
                                    n_idx, i_idx, N)
        ad_akt, an_akt = float(c_akt[0]), float(c_akt[1])

        rows_coef.append((
            sectors[j], delta_at, delta_atn, an_atn, ad_akt, an_akt,
            theta_vec[j], len(obs),
        ))

        for k in range(len(obs)):
            log_geo_at  = delta_at * log_d_bar[k]
            log_geo_atn = delta_atn * log_d_bar[k] + an_atn * N_dummy[k]
            log_geo_akt = ad_akt * D_km1000[k] + an_akt * N_dummy[k]
            rows_obs.append({
                "sector":     sectors[j],
                "n":          regions[n_idx[k]],
                "i":          regions[i_idx[k]],
                "tau_bar":    math.exp(y[k]),
                "tau_nd_at":  math.exp(y[k] - log_geo_at),
                "tau_ng_atn": math.exp(y[k] - log_geo_atn),
                "tau_ng_akt": math.exp(y[k] - log_geo_akt),
                "neighbor":   int(N_dummy[k]),
            })

    coef_tbl = pd.DataFrame(rows_coef, columns=[
        "sector", "delta_at", "delta_atn", "alpha_nbr_atn",
        "alpha_dist", "alpha_nbr_akt", "theta", "n_obs",
    ])
    obs_tbl = pd.DataFrame(rows_obs)
    return coef_tbl, obs_tbl


def main() -> None:
    print(f"Regions ({len(REGIONS)}): {REGIONS}")
    print(f"Sectors ({len(SECTORS)})")
    print()

    flows = build_flows()
    print()
    print("Filling WLD-WLD via ICIO 2022 ...")
    add_icio_wld_intra(flows)

    pi = compute_shares(flows)
    theta_vec = np.array([THETA[s] for s in SECTORS])

    print("Inverting Head-Ries τ̄ ...")
    tau_bar = head_ries_inversion(pi, theta_vec)

    d_raw, d_bar = build_distance_matrix(list(REGIONS))
    nbr = build_neighbor_matrix(list(REGIONS))
    print(f"Neighbor pairs: {int(nbr.sum() // 2)} unordered "
          f"(incl. WLD-{{AB,BC,MB,ON,QC,SK,ATL}})")

    coef, obs = run_regressions(tau_bar, d_raw, d_bar, nbr, theta_vec,
                                 list(REGIONS), list(SECTORS))

    # ── Per-sector coefficient table ─────────────────────────────────────
    print("\n" + "=" * 110)
    print(f"Per-sector coefficients (canada_2022_atlm, {len(SECTORS)} sectors × "
          f"{len(REGIONS)} regions)")
    print("=" * 110)
    print(f"{'Sector':<28} {'θ':>5} | {'AT δ':>7} | {'AT+N δ':>7} {'+N α%':>7} |"
          f" {'AKT α_D':>8} {'D%/1k':>7} {'+N α%':>7} | {'obs':>4}")
    print("-" * 110)
    for _, r in coef.iterrows():
        if np.isnan(r.delta_at):
            print(f"{r.sector[:28]:<28} {r.theta:>5.1f} | (insufficient obs: "
                  f"{int(r.n_obs)})")
            continue
        a_n_atn_pct = (math.exp(r.alpha_nbr_atn) - 1.0) * 100
        a_d_pct = (math.exp(r.alpha_dist) - 1.0) * 100
        a_n_akt_pct = (math.exp(r.alpha_nbr_akt) - 1.0) * 100
        print(f"{r.sector[:28]:<28} {r.theta:>5.1f} | {r.delta_at:>+7.3f} | "
              f"{r.delta_atn:>+7.3f} {a_n_atn_pct:>+6.1f}% | "
              f"{r.alpha_dist:>+8.3f} {a_d_pct:>+6.1f}% {a_n_akt_pct:>+6.1f}% |"
              f" {int(r.n_obs):>4}")

    # ── Trade-weighted aggregates ────────────────────────────────────────
    # Weight: w_{n,i,j} = flows[n,i,j] (the raw dollar trade volume, since
    # we're working with shares whose denominator is per-(n,j) absorption).
    weights = []
    sec_to_j = {s: k for k, s in enumerate(SECTORS)}
    reg_to_k = {r: k for k, r in enumerate(REGIONS)}
    for _, r in obs.iterrows():
        n_k, i_k = reg_to_k[r["n"]], reg_to_k[r["i"]]
        j_k = sec_to_j[r["sector"]]
        weights.append(flows[n_k, i_k, j_k])
    obs["w"] = weights

    def _wmean_pct(col, mask=None):
        d = obs if mask is None else obs[mask]
        w = d["w"].to_numpy()
        v = d[col].to_numpy() - 1.0
        return float(np.average(v, weights=w)) * 100 if w.sum() > 0 else float("nan")

    interprov = obs["n"].ne("WLD") & obs["i"].ne("WLD")
    all_pairs = pd.Series(True, index=obs.index)

    print("\n" + "=" * 90)
    print(f"Trade-weighted aggregate τ̄ − 1 (% markup) — canada_2022_atlm "
          f"({len(SECTORS)}×{len(REGIONS)})")
    print("=" * 90)
    print(f"{'Slice':<26} {'τ̄ obs':>9} {'AT NDist':>10} {'AT+N NG':>10} {'AKT NG':>9}")
    print("-" * 90)
    for label, mask in [("All pairs incl. WLD", all_pairs),
                        ("Intra-Canada only", interprov)]:
        v_obs   = _wmean_pct("tau_bar",     mask)
        v_at_n  = _wmean_pct("tau_nd_at",   mask)
        v_atn_n = _wmean_pct("tau_ng_atn",  mask)
        v_akt_n = _wmean_pct("tau_ng_akt",  mask)
        print(f"{label:<26} {v_obs:>+8.1f}% {v_at_n:>+9.1f}% {v_atn_n:>+9.1f}% "
              f"{v_akt_n:>+8.1f}%")

    # Decomposition
    m = interprov
    v_at  = _wmean_pct("tau_nd_at",  m)
    v_atn = _wmean_pct("tau_ng_atn", m)
    v_akt = _wmean_pct("tau_ng_akt", m)
    print(f"\n  Intra-Canada NG decomposition:")
    print(f"    AT non-distance                                    {v_at:>+7.1f}%")
    print(f"    + neighbor dummy (under log d̄):    Δ={v_atn - v_at:>+5.1f}pp  "
          f"{v_atn:>+7.1f}%  (AT+N)")
    print(f"    + linear D (replacing log d̄):       Δ={v_akt - v_atn:>+5.1f}pp  "
          f"{v_akt:>+7.1f}%  (AKT)")
    print(f"    Total AT → AKT divergence:                          "
          f"{v_akt - v_at:>+5.1f}pp")

    # ── Per-sector intra-Canada aggregates ───────────────────────────────
    print("\n" + "=" * 100)
    print("Per-sector intra-Canada τ̄ − 1 (% markup)")
    print("=" * 100)
    print(f"{'Sector':<28} {'τ̄ obs':>9} {'AT NDist':>9} {'AT+N NG':>9} "
          f"{'AKT NG':>9} {'pairs':>6}")
    print("-" * 100)
    for sec in SECTORS:
        m = (obs["sector"] == sec) & interprov
        if m.sum() < 4: continue
        v_obs   = _wmean_pct("tau_bar",   m)
        v_at_n  = _wmean_pct("tau_nd_at", m)
        v_atn_n = _wmean_pct("tau_ng_atn",m)
        v_akt_n = _wmean_pct("tau_ng_akt",m)
        print(f"{sec[:28]:<28} {v_obs:>+8.1f}% {v_at_n:>+8.1f}% {v_atn_n:>+8.1f}% "
              f"{v_akt_n:>+8.1f}% {int(m.sum()):>6}")


if __name__ == "__main__":
    main()
