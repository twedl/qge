"""Part 2b — infer AKT-2019 effective θ via Head-Ries matching (cprhs/THETA.md).

AKT-2019 (Alvarez, Krznar & Tombe, IMF WP 19/158) do not publish their θ
vector. The Head-Ries τ̂_{ni}^j and the resulting non-geographic component
τ̂_NG^j depend on θ_j as:

    log(τ̂_{ni}^j)  = z_{ni}^j / (2 θ_j)
    log(τ̂_NG^j)    = z_resid^j / (2 θ_j)

where z is the pure data quantity log((π_nn π_ii) / (π_ni π_in)) and
z_resid is its residual after the gravity regression (dist + neighbor +
exporter-year FE + importer-year FE). Both are independent of θ.

So if we run the AKT pipeline once with any baseline θ vector and then scale
θ_j by k_j, log(τ̂_NG^j) scales by 1/k_j and τ̂_NG^j_{ni} = exp(log_τ̂_NG / k) − 1.
For each sector j we solve a 1-D root-finding problem:

    AKT_target_NG^j  ≡  trade-weighted-mean over (n,i) of (exp(log_τ̂_NG_{ni} / k_j) − 1)

to find k_j; the inferred θ_j = θ_baseline_j · k_j.

Reference targets are AKT 2019 Table 1 (2015 goods, per-sector NG) and the
"all goods NG ≈ 19%" headline. Table 2 (per-province NG) is goods+services
in AKT's paper, so we don't fit it directly.

Run from cprhs/:
    uv run python scripts/infer_akt_theta.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

# Reuse the AKT pipeline from scripts/akt_replication.py.
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

# ── AKT-2019 published reference values ───────────────────────────────────
# Table 1, year 2015 goods: per-sector non-geographic τ̂_NG (%).
AKT_TABLE1_NG: dict[str, float] = {
    "Agriculture":             20.8,
    "Food":                    27.4,
    "Mining":                  12.6,
    "Textile":                  9.3,
    "Wood and Paper":          13.2,
    "Petroleum and Chemicals":  7.3,
    "Metals":                  31.2,
    "Machinery and Equipment": 22.4,
    "Other Manufacturing":     30.3,
}
# All-goods 2015 NG ≈ 19% (Table 1 footnote / Table 2 goods row).
AKT_ALL_GOODS_NG = 19.0


# ── helpers ────────────────────────────────────────────────────────────────
def trade_weighted_NG(log_tau_ng: np.ndarray, X: np.ndarray, k: float) -> float:
    """Trade-weighted mean of τ̂_NG = exp(log_τ̂_NG / k) − 1.

    `log_tau_ng` and `X` are per-(n,i,year) arrays under a baseline θ. Scaling
    θ_j by k_j scales log_τ̂_NG by 1/k_j (see module docstring)."""
    tau_ng = np.exp(log_tau_ng / k) - 1.0
    mask = np.isfinite(tau_ng) & (X > 0)
    if not mask.any():
        return float("nan")
    return float(np.average(tau_ng[mask], weights=X[mask]))


def infer_k(log_tau_ng: np.ndarray, X: np.ndarray, target_pct: float) -> float:
    """Find k > 0 such that trade_weighted_NG(...; k) · 100 == target_pct.

    Bisects on [0.01, 100] (covers θ in [θ_baseline·0.01, θ_baseline·100],
    enough to span the literature 0.5 ≤ θ ≤ 65)."""
    target = target_pct / 100.0
    def f(k: float) -> float:
        return trade_weighted_NG(log_tau_ng, X, k) - target
    # f is monotonically decreasing in k where log_τ̂ > 0 dominates the
    # trade-weighted average; brentq verifies sign change at the endpoints.
    try:
        return float(brentq(f, 1e-2, 1e2))
    except ValueError:
        return float("nan")


# ── main ───────────────────────────────────────────────────────────────────
def main() -> None:
    csv = Path("/tmp/sc12100088/12100088.csv")
    if not csv.exists():
        raise FileNotFoundError(f"missing {csv}; run StatCan 12-10-0088 fetcher first")
    canus_parquet = Path("/tmp/canada_us_trade_2007_2015.parquet")

    print("Loading interprovincial panel + Canada-US bilateral …")
    panel = load_interprov(csv, range(2007, 2016))
    if canus_parquet.exists():
        panel = add_usa_to_panel(panel, load_canada_us_trade(canus_parquet))
    print(f"  panel: {len(panel):,} rows, regions={panel.origin.nunique()}")

    print("Running Head-Ries + AKT regression at baseline θ …")
    hr = head_ries(panel)
    dist, nbr = build_distance_neighbor()
    df, _coef = akt_regression(hr, dist, nbr)

    # AKT Tables 1-2 cover 2015 interprovincial only.
    can_set = set(REGIONS_CA)
    y = df[df.year == 2015].copy()
    y = y[y.n.isin(can_set) & y.i.isin(can_set)]

    # Per-sector inference.
    print("\nInferring per-sector θ to match AKT Table 1 (2015 goods, NG %):\n")
    rows = []
    for sector in AKT_GOODS_SECTORS:
        sub = y[y.sector == sector]
        if sub.empty:
            continue
        log_tau_ng = sub["log_tau_ng"].to_numpy()
        X = sub["X_ni"].to_numpy()
        baseline_ng_pct = trade_weighted_NG(log_tau_ng, X, k=1.0) * 100
        k = infer_k(log_tau_ng, X, AKT_TABLE1_NG[sector])
        rows.append({
            "sector":           sector,
            "θ_baseline":       BASELINE_THETA[sector],
            "NG_baseline (%)":  baseline_ng_pct,
            "AKT target (%)":   AKT_TABLE1_NG[sector],
            "k_scale":          k,
            "θ_inferred":       BASELINE_THETA[sector] * k if np.isfinite(k) else float("nan"),
        })
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # All-goods pooled aggregate, baseline and rescaled.
    log_all = y["log_tau_ng"].to_numpy()
    X_all = y["X_ni"].to_numpy()
    pooled_baseline = trade_weighted_NG(log_all, X_all, k=1.0) * 100
    k_pooled = infer_k(log_all, X_all, AKT_ALL_GOODS_NG)
    print(f"\nAll-goods pooled NG (interprovincial, 2015):")
    print(f"  baseline θ:       {pooled_baseline:6.2f}%   (AKT target ≈ {AKT_ALL_GOODS_NG}%)")
    print(f"  uniform k_pooled: {k_pooled:.4f}  →  uniform θ_inferred would scale "
          f"all sectors by this factor (no single sector-specific θ)")


if __name__ == "__main__":
    main()
