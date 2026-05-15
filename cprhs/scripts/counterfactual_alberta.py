"""Example counterfactual scaffolding for Alberta.

Step 1: measure Alberta's average bilateral trade costs with (a) the United
States and (b) the other Canadian provinces, using the Head-Ries
ad-valorem-equivalent wedge implied by observed trade shares.

For sector j and partner i:

    τ_{AB,i}^j = ((π_{AB,i}^j / π_{AB,AB}^j) · (π_{i,AB}^j / π_{i,i}^j))^(-T_j/2) - 1

where π_{n,k}^j is region n's expenditure share on source k in sector j,
T_j = 1/θ_j is the sectoral trade elasticity reciprocal, and τ > 0 is the
iceberg trade cost (ad-valorem-equivalent).

The wedge is symmetric — d_{n,i} = d_{i,n} — so τ_{AB,US} reads as
"the average iceberg cost on a representative shipment between Alberta and
the US, regardless of direction."

Run:
    uv run python scripts/counterfactual_alberta.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qge.io import load_inputs

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


def main() -> None:
    raw = load_inputs(CALIBRATION)
    Din = (raw.xbilat.reshape(raw.J, raw.N, raw.N)
           / raw.xbilat.reshape(raw.J, raw.N, raw.N).sum(axis=2, keepdims=True))

    ab = raw.regions.index("Alberta")
    us = raw.regions.index("United States")
    other_provinces = [p for p in CANADIAN_PROVINCES if p != "Alberta"]
    tradable_idx = [raw.sectors.index(s) for s in TRADABLE_SECTORS]
    tradable_T = raw.T[tradable_idx]
    tradable_Din = Din[tradable_idx]

    tau_us = head_ries_wedge(tradable_Din, tradable_T, ab, us)
    tau_provinces = pd.DataFrame(
        {p: head_ries_wedge(tradable_Din, tradable_T, ab, raw.regions.index(p))
         for p in other_provinces},
        index=TRADABLE_SECTORS,
    )

    by_sector = pd.DataFrame(
        {"AB–USA": tau_us,
         "AB–avg(other provinces)": tau_provinces.mean(axis=1, skipna=True)},
        index=TRADABLE_SECTORS,
    )

    print(f"Head-Ries trade-cost wedges (ad-valorem equivalent), {len(TRADABLE_SECTORS)} tradable sectors:\n")
    print(by_sector.map(lambda v: f"{v:>7.1%}" if pd.notna(v) else "    n/a").to_string())
    print()

    avg_us = float(np.nanmean(tau_us))
    avg_can = float(np.nanmean(tau_provinces.to_numpy()))

    print(f"Alberta–USA              average trade cost: {avg_us:.1%}")
    print(f"Alberta–within-Canada    average trade cost: {avg_can:.1%}")

    dropped = tau_provinces.isna().stack()
    dropped = dropped[dropped].index.tolist()
    if dropped:
        print(f"\nDropped from within-Canada mean (no bilateral trade in this sector):")
        for sector, province in dropped:
            print(f"  {sector} × {province}")


if __name__ == "__main__":
    main()
