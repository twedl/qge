"""Part 2b — score candidate θ vectors against AKT-2019 Head-Ries outputs.

AKT-2019 Table 1 reports per-sector trade-barrier components (2015 goods):
    τ̄ obs : total Head-Ries iceberg trade barrier   = ((π_nn π_ii)/(π_ni π_in))^(1/2θ) − 1
    τ̄ geo : geographic component                    = projection on dist + nbr
    τ̄ NG  : non-geographic component                = τ̄ obs minus τ̄ geo (log-additive)

τ̄ obs is the *summary Head-Ries index* — pure function of θ and trade-share
data, no regression. We match it first, then check the gravity-decomposed
τ̄ NG.

Approach:
1. Run scripts/akt_replication.py's pipeline once at its baseline θ to extract
   per-pair log_τ̂ and log_τ̂_NG.
2. For any other θ vector, scale per-pair logs by k_j = θ_baseline_j / θ_new_j.
3. Trade-weighted-mean per sector → derived τ̄ obs and τ̄ NG.
4. Compare to AKT Table 1; report RMSE per candidate.
5. Per-sector inversion: find θ_j that exactly matches AKT Table 1 col 1
   (τ̄ obs) and col 3 (τ̄ NG) — separately.

Candidate θ vectors at AKT 9-goods grain:
    benchmark    : CP-2015 99% sample + corrections, simple-mean to AKT-9
    fullsample   : CP-2015 Full sample, simple-mean to AKT-9 (AT-2016 convention)
    akt_repl     : the THETA dict currently in scripts/akt_replication.py
    uniform_X    : single-θ ablations (4, 5, 6, 8.28)

Run from cprhs/:
    uv run python scripts/infer_akt_theta.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).parent))
from akt_replication import (  # noqa: E402
    AKT_GOODS_SECTORS,
    REGIONS_CA,
    THETA as BASELINE_THETA,
    add_usa_to_panel,
    akt_regression,
    build_distance_neighbor,
    head_ries,
    load_canada_us_trade,
    load_interprov,
)

# ── AKT-2019 Table 1 published values (2015 goods, %) ─────────────────────
# Per sector: (τ̄ obs, τ̄ geo, τ̄ NG)
AKT_TABLE1: dict[str, tuple[float, float, float]] = {
    "Agriculture":             (37.4, 16.6, 20.8),
    "Food":                    (60.5, 33.1, 27.4),
    "Mining":                  (15.8,  3.2, 12.6),
    "Textile":                 (20.4, 11.1,  9.3),
    "Wood and Paper":          (21.5,  8.3, 13.2),
    "Petroleum and Chemicals": (12.2,  4.9,  7.3),
    "Metals":                  (50.5, 19.3, 31.2),
    "Machinery and Equipment": (37.0, 14.6, 22.4),
    "Other Manufacturing":     (46.1, 15.8, 30.3),
}
# All-goods pooled (AKT text, year 2015): τ̄ obs ≈ 36, τ̄ geo ≈ 17, τ̄ NG ≈ 19.
AKT_ALL_GOODS = (36.0, 17.0, 19.0)


# ── Candidate θ vectors at AKT-9 grain ─────────────────────────────────────
# CP-2015 Table 1 values (kept locally so this script is self-contained).
_CP_FULL: dict[str, float] = {
    "Agriculture": 8.11, "Mining": 15.72, "Food": 2.55, "Textile": 5.56,
    "Wood": 10.83, "Paper": 9.07, "Petroleum": 51.08, "Chemicals": 4.75,
    "Plastics": 1.66, "Minerals": 2.76, "Basic metals": 7.99,
    "Metal products": 4.30, "Machinery": 1.52, "Office": 12.79,
    "Electrical": 10.60, "Communication": 7.07, "Medical": 9.98,
    "Auto": 1.01, "Other transport": 0.37, "Other": 5.00,
}
_CP_99_RAW: dict[str, float] = {
    "Agriculture": 9.11, "Mining": 13.53, "Food": 2.62, "Textile": 8.10,
    "Wood": 11.50, "Paper": 16.52, "Petroleum": 64.85, "Chemicals": 3.13,
    "Plastics": 1.67, "Minerals": 2.41, "Basic metals": 3.28,
    "Metal products": 6.99, "Machinery": 1.45, "Office": 12.95,
    "Electrical": 12.91, "Communication": 3.95, "Medical": 8.71,
    "Auto": 1.84, "Other transport": 0.39, "Other": 3.98,
}
# Benchmark = 99% sample with CP's published substitutions (4.49 mfg-aggregate).
_CP_99_SUB: dict[str, float] = {
    **_CP_99_RAW, "Basic metals": 4.49, "Machinery": 4.49, "Auto": 4.49,
}


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def cp_to_akt9(cp: dict[str, float]) -> dict[str, float]:
    """Simple-mean aggregation from CP-2015's 20 sub-sectors to AKT's 9 goods
    sectors. Mirrors the AKT S&U → ISIC concordance:
      AKT Petroleum and Chemicals = CP {Petroleum, Chemicals, Plastics, Minerals}
      AKT Metals                  = CP {Basic metals, Metal products}
      AKT Machinery and Equipment = CP {Machinery, Office, Electrical,
                                        Communication, Medical, Auto, Other transport}
      AKT Wood and Paper          = CP {Wood, Paper}
    The other 5 AKT sectors are 1-to-1 with a CP sub-sector.
    """
    return {
        "Agriculture":             cp["Agriculture"],
        "Food":                    cp["Food"],
        "Mining":                  cp["Mining"],
        "Textile":                 cp["Textile"],
        "Wood and Paper":          _mean([cp["Wood"], cp["Paper"]]),
        "Petroleum and Chemicals": _mean([cp["Petroleum"], cp["Chemicals"],
                                           cp["Plastics"], cp["Minerals"]]),
        "Metals":                  _mean([cp["Basic metals"], cp["Metal products"]]),
        "Machinery and Equipment": _mean([cp["Machinery"], cp["Office"],
                                           cp["Electrical"], cp["Communication"],
                                           cp["Medical"], cp["Auto"],
                                           cp["Other transport"]]),
        "Other Manufacturing":     cp["Other"],
    }


CANDIDATES: dict[str, dict[str, float]] = {
    "benchmark":    cp_to_akt9(_CP_99_SUB),
    "fullsample":   cp_to_akt9(_CP_FULL),
    "akt_repl":     dict(BASELINE_THETA),
    "uniform_4":    {s: 4.0  for s in AKT_GOODS_SECTORS},
    "uniform_5":    {s: 5.0  for s in AKT_GOODS_SECTORS},
    "uniform_6":    {s: 6.0  for s in AKT_GOODS_SECTORS},
    "uniform_8.28": {s: 8.28 for s in AKT_GOODS_SECTORS},
}


# ── Helpers ───────────────────────────────────────────────────────────────
def tw_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Trade-weighted mean, ignoring non-finite entries and zero weights."""
    mask = np.isfinite(values) & (weights > 0)
    if not mask.any():
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


def _logcol(which: str) -> str:
    """Map 'obs' → 'log_tau', 'ng' → 'log_tau_ng'."""
    if which == "obs":
        return "log_tau"
    if which == "ng":
        return "log_tau_ng"
    raise ValueError(f"unknown 'which': {which}")


def per_sector(df: pd.DataFrame, theta: dict[str, float], which: str
                ) -> dict[str, float]:
    """Per-sector trade-weighted τ̄ (in %) for the given θ vector and quantity.

    Uses the identity log_τ̂_new = log_τ̂_baseline · (θ_baseline / θ_new),
    so τ̄_new = exp(log_τ̂_baseline · k_j) − 1 where k_j = θ_baseline_j / θ_new_j.
    """
    col = _logcol(which)
    out: dict[str, float] = {}
    for s in AKT_GOODS_SECTORS:
        sub = df[df.sector == s]
        if sub.empty:
            out[s] = float("nan")
            continue
        k = BASELINE_THETA[s] / theta[s]
        tau = np.exp(sub[col].to_numpy() * k) - 1.0
        out[s] = tw_mean(tau, sub["X_ni"].to_numpy()) * 100
    return out


def pooled_all_goods(df: pd.DataFrame, theta: dict[str, float], which: str) -> float:
    """All-goods trade-weighted τ̄ (in %) under the given θ vector."""
    col = _logcol(which)
    df = df.copy()
    df["k"] = df["sector"].map({s: BASELINE_THETA[s] / theta[s]
                                 for s in AKT_GOODS_SECTORS})
    df = df.dropna(subset=["k"])
    tau = np.exp(df[col].to_numpy() * df["k"].to_numpy()) - 1.0
    return tw_mean(tau, df["X_ni"].to_numpy()) * 100


def invert_per_sector(df: pd.DataFrame, target: dict[str, float], which: str
                       ) -> dict[str, float]:
    """Per-sector θ that hits the given AKT target (in %) under `which`."""
    col = _logcol(which)
    out: dict[str, float] = {}
    for s in AKT_GOODS_SECTORS:
        sub = df[df.sector == s]
        if sub.empty:
            out[s] = float("nan")
            continue
        log_t = sub[col].to_numpy()
        X = sub["X_ni"].to_numpy()
        target_frac = target[s] / 100.0
        def f(k: float) -> float:
            tau = np.exp(log_t * k) - 1.0
            return tw_mean(tau, X) - target_frac
        try:
            k = float(brentq(f, 1e-2, 1e2))
            out[s] = BASELINE_THETA[s] / k
        except ValueError:
            out[s] = float("nan")
    return out


def fit_score(derived: dict[str, float], target: dict[str, float]
              ) -> tuple[float, float]:
    """RMSE, MAE in % points across 9 sectors."""
    diffs = np.array([derived[s] - target[s] for s in AKT_GOODS_SECTORS])
    return float(np.sqrt(np.nanmean(diffs ** 2))), float(np.nanmean(np.abs(diffs)))


# ── Main ───────────────────────────────────────────────────────────────────
def main() -> None:
    csv = Path("/tmp/sc12100088/12100088.csv")
    if not csv.exists():
        raise FileNotFoundError(f"missing {csv}; run StatCan 12-10-0088 fetcher first")
    canus = Path("/tmp/canada_us_trade_2007_2015.parquet")

    print("Loading panel + running AKT pipeline at baseline θ …")
    panel = load_interprov(csv, range(2007, 2016))
    if canus.exists():
        panel = add_usa_to_panel(panel, load_canada_us_trade(canus))
    hr = head_ries(panel)
    dist, nbr = build_distance_neighbor()
    df, _coef = akt_regression(hr, dist, nbr)

    # 2015 interprov pairs only.
    can_set = set(REGIONS_CA)
    y = df[(df.year == 2015) & df.n.isin(can_set) & df.i.isin(can_set)].copy()

    # ── Step 1 — Summary Head-Ries: τ̄ obs ──
    print("\n" + "=" * 78)
    print(" STEP 1.  Summary Head-Ries index — τ̄ obs (%) — 2015 interprov")
    print("=" * 78 + "\n")
    rows = [{"sector": s, "AKT τ̄ obs": AKT_TABLE1[s][0]} for s in AKT_GOODS_SECTORS]
    for name, theta in CANDIDATES.items():
        sector_vals = per_sector(y, theta, which="obs")
        for r in rows:
            r[name] = sector_vals[r["sector"]]
    obs_df = pd.DataFrame(rows)

    # Append all-goods row
    all_goods_row = {"sector": "ALL GOODS", "AKT τ̄ obs": AKT_ALL_GOODS[0]}
    for name, theta in CANDIDATES.items():
        all_goods_row[name] = pooled_all_goods(y, theta, which="obs")
    obs_df = pd.concat([obs_df, pd.DataFrame([all_goods_row])], ignore_index=True)

    print(obs_df.to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    print("\nFit on τ̄ obs across 9 sectors (RMSE / MAE in pp):\n")
    target_obs = {s: AKT_TABLE1[s][0] for s in AKT_GOODS_SECTORS}
    score_rows = []
    for name, theta in CANDIDATES.items():
        rmse, mae = fit_score(per_sector(y, theta, which="obs"), target_obs)
        score_rows.append({"candidate": name, "RMSE": rmse, "MAE": mae})
    score_df = pd.DataFrame(score_rows).sort_values("RMSE")
    print(score_df.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    # ── Step 2 — Non-geographic decomposition: τ̄ NG ──
    print("\n" + "=" * 78)
    print(" STEP 2.  Non-geographic Head-Ries — τ̄ NG (%) — 2015 interprov")
    print("=" * 78 + "\n")
    rows = [{"sector": s, "AKT τ̄ NG": AKT_TABLE1[s][2]} for s in AKT_GOODS_SECTORS]
    for name, theta in CANDIDATES.items():
        sector_vals = per_sector(y, theta, which="ng")
        for r in rows:
            r[name] = sector_vals[r["sector"]]
    ng_df = pd.DataFrame(rows)

    all_goods_row_ng = {"sector": "ALL GOODS", "AKT τ̄ NG": AKT_ALL_GOODS[2]}
    for name, theta in CANDIDATES.items():
        all_goods_row_ng[name] = pooled_all_goods(y, theta, which="ng")
    ng_df = pd.concat([ng_df, pd.DataFrame([all_goods_row_ng])], ignore_index=True)

    print(ng_df.to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    print("\nFit on τ̄ NG across 9 sectors (RMSE / MAE in pp):\n")
    target_ng = {s: AKT_TABLE1[s][2] for s in AKT_GOODS_SECTORS}
    score_rows = []
    for name, theta in CANDIDATES.items():
        rmse, mae = fit_score(per_sector(y, theta, which="ng"), target_ng)
        score_rows.append({"candidate": name, "RMSE": rmse, "MAE": mae})
    score_df = pd.DataFrame(score_rows).sort_values("RMSE")
    print(score_df.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    # ── Step 3 — Per-sector inversion ──
    print("\n" + "=" * 78)
    print(" STEP 3.  Inferred θ to hit AKT Table 1 exactly")
    print("=" * 78 + "\n")
    inf_obs = invert_per_sector(y, target_obs, which="obs")
    inf_ng = invert_per_sector(y, target_ng, which="ng")
    inv_rows = []
    for s in AKT_GOODS_SECTORS:
        inv_rows.append({
            "sector":      s,
            "benchmark":   CANDIDATES["benchmark"][s],
            "fullsample":  CANDIDATES["fullsample"][s],
            "θ inv (obs)": inf_obs[s],
            "θ inv (NG)":  inf_ng[s],
        })
    inv_df = pd.DataFrame(inv_rows)
    print(inv_df.to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
