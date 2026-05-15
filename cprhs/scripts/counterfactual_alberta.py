"""Example counterfactual scaffolding for Alberta.

Two steps:

1. Measure Alberta's average bilateral trade costs with (a) the United
   States and (b) the other Canadian provinces, via the Head-Ries
   ad-valorem-equivalent wedge implied by observed trade shares:

       τ_{AB,i}^j = ((π_{AB,i}^j / π_{AB,AB}^j) · (π_{i,AB}^j / π_{i,i}^j))^(-T_j/2) - 1

   where π_{n,k}^j is region n's expenditure share on source k in sector j
   and T_j = 1/θ_j. The wedge is symmetric.

2. Counterfactual: lift AB↔within-Canada iceberg costs by a single uniform
   factor (1 + τ̄_{AB,USA}) / (1 + τ̄_{AB,Can}) in every tradable sector,
   then re-solve the model. Report Alberta's outcomes and the (VA-weighted)
   rest-of-Canada outcomes.

Run:
    uv run python scripts/counterfactual_alberta.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qge.io import load_inputs
from qge.models.benchmark import _run_shock

CALIBRATION = "data/inputs/canada_2021_partners/"

CANADIAN_PROVINCES = (
    "Alberta",
    "British Columbia",
    "Manitoba",
    "New Brunswick",
    "Newfoundland and Labrador",
    "Nova Scotia",
    "Ontario",
    "Prince Edward Island",
    "Quebec",
    "Saskatchewan",
)

# Tradable goods sectors in the canada_2021 taxonomy — primary + manufacturing,
# matching CPRHS's first-15-of-26 designation. The remaining 12 sectors are
# services / construction / utilities and don't trade in the bilateral sense.
TRADABLE_SECTORS = (
    "Agriculture, Forestry, Fishing",
    "Computers, Electronics, Electrical",
    "Food, Beverage, Tobacco",
    "Furniture and Other Manufacturing",
    "Metals and Machinery",
    "Mining and Extraction",
    "Non-metallic Mineral Products",
    "Petroleum and Chemicals",
    "Textile, Apparel, Leather",
    "Transportation Equipment",
    "Wood, Paper, Printing",
)


def head_ries_wedge(Din: np.ndarray, T: np.ndarray, n: int, i: int) -> np.ndarray:
    """Per-sector iceberg trade-cost wedge τ between regions n and i.

    Din has shape (J, N_dest, N_src); T has shape (J,). Returns τ of shape
    (J,). NaN where any of the four trade shares is zero (no implied cost).
    """
    pi_ni = Din[:, n, i]
    pi_in = Din[:, i, n]
    pi_nn = Din[:, n, n]
    pi_ii = Din[:, i, i]
    with np.errstate(divide="ignore", invalid="ignore"):
        product = (pi_ni / pi_nn) * (pi_in / pi_ii)
        d = np.where(product > 0, product ** (-T / 2.0), np.nan)
    return d - 1.0


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Trade-flow-weighted mean of τ, ignoring NaN entries."""
    mask = np.isfinite(values) & (weights > 0)
    return float(np.sum(values[mask] * weights[mask]) / np.sum(weights[mask]))


def main() -> None:
    raw = load_inputs(CALIBRATION)
    xbilat_3d = raw.xbilat.reshape(raw.J, raw.N, raw.N)  # (sector, dest, source)
    Din = xbilat_3d / xbilat_3d.sum(axis=2, keepdims=True)

    ab = raw.regions.index("Alberta")
    us = raw.regions.index("United States")
    other_provinces = [p for p in CANADIAN_PROVINCES if p != "Alberta"]
    tradable_idx = [raw.sectors.index(s) for s in TRADABLE_SECTORS]
    tradable_T = raw.T[tradable_idx]
    tradable_Din = Din[tradable_idx]
    tradable_x = xbilat_3d[tradable_idx]

    # Bilateral two-way trade flow between Alberta and partner i in sector j.
    def flow(i: int) -> np.ndarray:
        return tradable_x[:, ab, i] + tradable_x[:, i, ab]

    tau_us = head_ries_wedge(tradable_Din, tradable_T, ab, us)
    w_us = flow(us)

    tau_prov = np.column_stack([
        head_ries_wedge(tradable_Din, tradable_T, ab, raw.regions.index(p))
        for p in other_provinces
    ])
    w_prov = np.column_stack([flow(raw.regions.index(p)) for p in other_provinces])

    # Per-sector trade-weighted mean across the 9 other provinces.
    tau_prov_by_sector = np.array([
        _weighted_mean(tau_prov[j], w_prov[j]) for j in range(len(TRADABLE_SECTORS))
    ])

    by_sector = pd.DataFrame(
        {"AB–USA":                  tau_us,
         "AB–avg(other provinces)": tau_prov_by_sector},
        index=TRADABLE_SECTORS,
    )

    print(f"Head-Ries trade-cost wedges (ad-valorem equivalent), {len(TRADABLE_SECTORS)} tradable sectors:")
    print("(within-Canada column is trade-weighted across 9 provinces)\n")
    print(by_sector.map(lambda v: f"{v:>7.1%}" if pd.notna(v) else "    n/a").to_string())
    print()

    avg_us = _weighted_mean(tau_us, w_us)
    avg_can = _weighted_mean(tau_prov.ravel(), w_prov.ravel())

    print(f"Alberta–USA              trade-weighted average trade cost: {avg_us:.1%}")
    print(f"Alberta–within-Canada    trade-weighted average trade cost: {avg_can:.1%}")

    # ---- Step 2: counterfactual ----------------------------------------

    factor = (1 + avg_us) / (1 + avg_can)
    print(f"\nCounterfactual: lift AB↔province iceberg costs by {factor:.3f}× "
          f"in every tradable sector\n(brings AB-within-Canada average up to AB-USA level).")

    kappa_hat = np.ones((raw.J, raw.N, raw.N))
    for j in tradable_idx:
        for p in other_provinces:
            p_idx = raw.regions.index(p)
            kappa_hat[j, ab, p_idx] = factor
            kappa_hat[j, p_idx, ab] = factor
    kappa_hat_2d = kappa_hat.reshape(raw.J * raw.N, raw.N)

    result = _run_shock(
        lambda_hat=np.ones((raw.J, raw.N)),
        baseline=None, raw=raw, kappa_hat=kappa_hat_2d,
        tol=1e-8, vfactor=-0.1, maxit=1_000_000, verbose=False,
    )
    print(f"Solved in {result.iterations} iterations.\n")

    summary = result.regional_summary()
    summary["real_GDPn_hat"] = summary["GDPn_hat"] / summary["P_index_hat"]

    cols = ["L_hat", "P_index_hat", "TFPn_hat", "GDPn_hat", "real_GDPn_hat"]
    rows_to_print = ["Alberta"] + [p for p in CANADIAN_PROVINCES if p != "Alberta"]

    # ROC aggregate: VA-weighted mean over the 9 other provinces.
    roc_va = summary.loc[rows_to_print[1:], "VAn0"].to_numpy()
    roc_w = roc_va / roc_va.sum()
    roc_row = pd.Series(
        {c: float(np.sum(summary.loc[rows_to_print[1:], c].to_numpy() * roc_w))
         for c in cols},
        name="Rest of Canada (VA-weighted)",
    )

    out = pd.concat([summary.loc[rows_to_print, cols], roc_row.to_frame().T])

    def pct(v: float) -> str:
        return f"{(v - 1) * 100:+6.2f}%"

    print("Counterfactual outcomes (% change from baseline):\n")
    print(out.map(pct).to_string())
    print(f"\nAggregate world welfare V_hat:    {pct(result.V_hat)}")
    print(f"Aggregate world TFP_hat:          {pct(result.TFP_hat)}")
    print(f"Aggregate world GDP_hat:          {pct(result.GDP_hat)}")


if __name__ == "__main__":
    main()
