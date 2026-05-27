"""Multi-year-pooled Waugh + distance regressions for tauhat extraction.

AT-2016 use a panel of provincial trade flows for their FE regressions, not
a single year. Re-extracting t_i^j and δ^j from a single year (2010) gives
noisier estimates, lowering the cell-level match with maindata.mat's tauhats.

This script pools StatCan 12-10-0088 for a chosen year range (default
2007-2014, the panel AT-2016 had at publication), runs:

  ln(π_{n,i,t}^j / π_{n,n,t}^j) = γ_j ln(d̄_{n,i}) + ι_n^j + η_i^j + δ_t^j + ε

with time-invariant importer & exporter FEs plus year fixed effects, and
extracts time-invariant t_i^j = exp(-η_i^j / θ_j) (Convention D — exporter
FE only, validated as the closest match to MAT's backed-out values).

WLD is included via a per-year scale: each year's interprov-only π are
renormalized so destination-side row sums = 1 across provinces (excluding
WLD). WLD-involving pairs are still handled by the single-year Waugh in
build_trade_costs.py — this script only updates t_i^j for the 10 provinces.

Run from tw/:
    uv run python scripts/build_tauhat_pooled.py \\
        --year-lo 2007 --year-hi 2014 \\
        --calibration canada_2010_rebuild
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_canada_2022 import (  # noqa: E402
    MCODE_TO_TW, SECTORS, PROVINCES_FULL, PROVINCES_SHORT,
    PROV_FULL_TO_SHORT, STATCAN_TRADE_CSV, _PRODUCT_RE,
)
from scripts.build_trade_costs import (  # noqa: E402
    build_distance_matrix, head_ries_inversion, _gravity_regression,
    build_shocks, reshape_to_long,
)


def load_pi_one_year(year: int, all_rows: pd.DataFrame) -> dict[str, np.ndarray]:
    """Build interprov PI for one year from a preloaded 12-10-0088 frame.

    Returns dict: 'pi' shape (10, 10, J), 'flows_dollars' shape (10, 10, J).
    Diagonal cells = within-province absorption inferred residually as
        total absorption - sum across other-province imports - international.
    Following AT-2016 we use the *interprov-share-of-interprov-absorption*:
        π_{n,i}^j = flow[n,i,j] / sum_k flow[n,k,j]   where k ranges over
        all sources INCLUDING the within-province diagonal.
    """
    df = all_rows[all_rows["REF_DATE"] == year].copy()
    df = df[df["GEO"].isin(PROVINCES_FULL)]
    code = df["Product"].astype(str).str.extract(_PRODUCT_RE)
    df["sector"] = code[0].map(MCODE_TO_TW)
    df = df[df["sector"].notna()]

    sec_idx = {s: i for i, s in enumerate(SECTORS)}
    prov_idx = {p: i for i, p in enumerate(PROVINCES_SHORT)}
    N, J = len(PROVINCES_SHORT), len(SECTORS)

    flows = np.zeros((N, N, J))   # flows[dest, source, sector]

    # Interprov bilateral
    inter = df[df["Trade flow detail"].astype(str).str.startswith("To ")].copy()
    inter["destination_full"] = inter["Trade flow detail"].str.removeprefix("To ")
    inter = inter[inter["destination_full"].isin(PROVINCES_FULL)]
    inter["destination"] = inter["destination_full"].map(PROV_FULL_TO_SHORT)
    inter["source"] = inter["GEO"].map(PROV_FULL_TO_SHORT)
    agg = inter.groupby(["sector", "destination", "source"], as_index=False)["VALUE"].sum()
    for r in agg.itertuples(index=False):
        flows[prov_idx[r.destination], prov_idx[r.source], sec_idx[r.sector]] = r.VALUE

    # Domestic absorption (diagonal): use TOTAL trade flow column to infer
    # within-province absorption. Approximation: from the same table, the
    # rows where Trade flow detail = "Domestic" give within-province use.
    dom = df[df["Trade flow detail"].astype(str) == "Domestic"].copy()
    dom["destination"] = dom["GEO"].map(PROV_FULL_TO_SHORT)
    dom["source"] = dom["destination"]
    dom_agg = dom.groupby(["sector", "destination"], as_index=False)["VALUE"].sum()
    for r in dom_agg.itertuples(index=False):
        flows[prov_idx[r.destination], prov_idx[r.destination], sec_idx[r.sector]] = r.VALUE

    # Per (dest, sector), normalize to shares
    totals = flows.sum(axis=1, keepdims=True)
    pi = np.where(totals > 0, flows / np.maximum(totals, 1e-12), 0.0)
    return {"pi": pi, "flows": flows}


def run_pooled_waugh(
    pi_by_year: dict[int, np.ndarray], d_bar: np.ndarray, theta_provs: np.ndarray,
    weight: str = "none",
) -> tuple[np.ndarray, pd.DataFrame]:
    """Run pooled Waugh per sector with importer + exporter + year FEs.

    Args:
        weight: 'none' (OLS, equal weights), 'pi' (weight obs by π_{n,i,j,t}),
            or 'pi_log' (weight by π × baseline absorption proxy).

    Returns:
        t : shape (N_provs, J) — exp(-η_i^j / θ_j)
        coefs : per-sector diagnostic table
    """
    years = sorted(pi_by_year)
    T = len(years)
    N = pi_by_year[years[0]].shape[0]   # 10 provs
    J = pi_by_year[years[0]].shape[2]

    t = np.ones((N, J))
    rows = []
    for j in range(J):
        theta_j = theta_provs[j]
        obs = []
        for t_idx, yr in enumerate(years):
            pi = pi_by_year[yr]
            for n in range(N):
                for i in range(N):
                    if n == i: continue
                    if pi[n, i, j] <= 0 or pi[n, n, j] <= 0: continue
                    obs.append((n, i, t_idx,
                                math.log(pi[n, i, j] / pi[n, n, j]),
                                math.log(d_bar[n, i]),
                                pi[n, i, j]))
        if len(obs) < 2 * N + T:
            rows.append((SECTORS[j], np.nan, np.nan, len(obs)))
            continue
        a = np.array(obs)
        n_idx = a[:, 0].astype(int)
        i_idx = a[:, 1].astype(int)
        t_idx_arr = a[:, 2].astype(int)
        y = a[:, 3]
        logd = a[:, 4]
        pi_obs = a[:, 5]

        # Design matrix: [const, log d̄, imp_1..N-1, exp_1..N-1, year_1..T-1]
        n_obs = len(y)
        X = np.zeros((n_obs, 2 + 2*(N-1) + (T-1)))
        X[:, 0] = 1.0
        X[:, 1] = logd
        for r in range(1, N):
            X[n_idx == r, 1 + r] = 1.0
            X[i_idx == r, 1 + (N-1) + r] = 1.0
        for tt in range(1, T):
            X[t_idx_arr == tt, 1 + 2*(N-1) + tt] = 1.0

        # Apply WLS transformation: scale rows by √w, then OLS on transformed.
        if weight == "pi":
            w = pi_obs
        elif weight == "pi_log":
            # Log-weight: less aggressive than direct π weighting.
            w = np.log1p(pi_obs * 1000.0)
        else:
            w = np.ones(n_obs)
        sw = np.sqrt(w)
        Xw = X * sw[:, None]
        yw = y * sw
        beta, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
        gamma = float(beta[1])
        # Exporter FE for r=1..N-1; r=0 reference (η_0 = 0).
        eta = np.concatenate(([0.0], beta[1 + (N-1) + 1 : 1 + 2*(N-1) + 1]))
        # Convention D: log(t_r) = -η_r / θ
        log_t = -eta / theta_j
        t[:, j] = np.exp(log_t)
        yhat = X @ beta
        r2 = 1.0 - float(((y - yhat)**2).sum()) / float(((y - y.mean())**2).sum())
        rows.append((SECTORS[j], gamma, r2, len(obs)))
    coefs = pd.DataFrame(rows, columns=["sector", "gamma", "r2", "n_obs"])
    return t, coefs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year-lo", type=int, default=2007)
    parser.add_argument("--year-hi", type=int, default=2014)
    parser.add_argument("--calibration", default="canada_2010_rebuild")
    parser.add_argument("--weight", choices=("none", "pi", "pi_log"),
                        default="none",
                        help="Regression weights: none = equal-weight OLS, "
                             "pi = WLS weighted by trade share, pi_log = "
                             "log1p(π × 1000)")
    parser.add_argument("--averaging", choices=("pooled", "per_year"),
                        default="pooled",
                        help="pooled = single regression across all years with "
                             "year FEs (default); per_year = run separate "
                             "regression each year, take geometric mean of t_r")
    args = parser.parse_args()

    root = (Path(__file__).resolve().parent.parent / "data" / "inputs"
            / args.calibration)
    print(f"Building pooled tauhats for {args.calibration} from "
          f"{args.year_lo}-{args.year_hi} 12-10-0088 panel\n")

    # Load 12-10-0088 once
    print(f"  Loading {STATCAN_TRADE_CSV} ...")
    all_rows = pd.read_csv(STATCAN_TRADE_CSV, low_memory=False)

    # Build interprov PI per year
    pi_by_year = {}
    for yr in range(args.year_lo, args.year_hi + 1):
        d = load_pi_one_year(yr, all_rows)
        pi_by_year[yr] = d["pi"]
        # quick sanity print
        pi_diag = np.diagonal(d["pi"]).T  # (J, N)
        print(f"  {yr}: π diagonal mean over (n,j) = {pi_diag.mean():.3f}")

    # Sector dispersion (θ) from canada_2010_rebuild
    sd = pd.read_parquet(root / "sectoral_dispersion.parquet")
    theta = 1.0 / sd.set_index("sector").loc[list(SECTORS), "value"].to_numpy()

    # Distance matrix on 10 provinces only (drop WLD)
    provs10 = list(PROVINCES_SHORT)
    d_raw_full, d_bar_full = build_distance_matrix(provs10)

    # Run pooled (or per-year averaged) Waugh
    print(f"\n  Running regression ({args.year_hi-args.year_lo+1} years × "
          f"{len(SECTORS)} sectors), weight={args.weight}, averaging={args.averaging}...")
    if args.averaging == "pooled":
        t_provs, coefs = run_pooled_waugh(pi_by_year, d_bar_full, theta, weight=args.weight)
    else:
        # Per-year: run pooled on each year separately, geom-mean across years
        N_p = len(PROVINCES_SHORT); J = len(SECTORS)
        t_per_year = []
        for yr in sorted(pi_by_year):
            t_yr, _ = run_pooled_waugh({yr: pi_by_year[yr]}, d_bar_full, theta,
                                         weight=args.weight)
            t_per_year.append(t_yr)
        t_stack = np.log(np.clip(np.stack(t_per_year), 1e-10, None))
        t_provs = np.exp(t_stack.mean(axis=0))
        coefs = None

    # Print per-sector t_r summary (geom mean across sectors per region)
    print("\n  Pooled t_r (geometric mean across sectors):")
    for r_idx, r in enumerate(provs10):
        vals = t_provs[r_idx, :]
        valid = vals[~np.isnan(vals) & (vals > 0)]
        if len(valid) > 0:
            print(f"    {r}: t̄={np.exp(np.log(valid).mean()):.3f}")

    # ── Reuse the 2010 τ̄ (Head-Ries) and single-year WLD t from existing
    #    canada_2010_rebuild parquets, replace only the 10-province t with
    #    the pooled estimates. Then rebuild shocks.
    print("\n  Reusing 2010 τ̄ and single-year WLD t; rebuilding shocks ...")

    # Reconstruct the 2010 pi (11 regions) and compute τ̄
    raw_bt = pd.read_parquet(root / "bilateral_trade.parquet")
    regions = sorted(raw_bt.destination.unique())
    provinces = sorted([r for r in regions if r != "WLD"])
    regions = provinces + ["WLD"]
    N_full = len(regions)
    J = len(SECTORS)
    reg_idx = {r: k for k, r in enumerate(regions)}
    sec_idx = {s: k for k, s in enumerate(SECTORS)}
    pi_full = np.zeros((N_full, N_full, J))
    for r in raw_bt.itertuples(index=False):
        pi_full[reg_idx[r.destination], reg_idx[r.source], sec_idx[r.sector]] = r.value
    tau_bar = head_ries_inversion(pi_full, theta)
    d_raw_full, d_bar_full = build_distance_matrix(regions)

    # Build a (N_full, J) t array: provs from pooled, WLD from single-year
    # rerun. Easier: just take pooled for first 10 rows and set t_WLD per
    # the current canada_2010_rebuild Waugh.
    # Reload tau_bar + d_bar + theta and run single-year Waugh to get t_WLD.
    from scripts.build_trade_costs import waugh_regression
    t_single, _ = waugh_regression(pi_full, d_bar_full, theta, regions, list(SECTORS))

    t_combined = np.ones((N_full, J))
    # Pooled t for the 10 provinces (regions list is alphabetic → provs first)
    for idx_pool, p in enumerate(provs10):
        idx_full = reg_idx[p]
        t_combined[idx_full, :] = t_provs[idx_pool, :]
    # WLD from single-year
    t_combined[reg_idx["WLD"], :] = t_single[reg_idx["WLD"], :]

    # Distance regression: reuse single-year for now (already interprov-only fixed)
    from scripts.build_trade_costs import distance_regression
    delta, _ = distance_regression(tau_bar, d_bar_full, regions, list(SECTORS))

    shocks = build_shocks(tau_bar, t_combined, delta, d_bar_full, regions=regions)
    for name, arr in shocks.items():
        df = reshape_to_long(arr, list(SECTORS), regions)
        df.to_parquet(root / f"{name}.parquet", index=False)
        off = df[df.source != df.destination]
        print(f"  wrote {name}.parquet: off-diag mean={off['value'].mean():.4f}, "
              f"min={off['value'].min():.4f}")


if __name__ == "__main__":
    main()
