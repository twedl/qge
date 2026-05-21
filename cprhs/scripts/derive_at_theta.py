"""Derive AT-2016 Table 9 from CP-2015 (Part 2a of cprhs/THETA.md).

AT-2016 publish a θ vector at 22-sector ISIC Rev3 grain (Table 9, paper p.26)
and tell us they aggregate CP-2015: "trade elasticity is from the Caliendo
and Parro (2015) estimates, averaged up to a slightly higher level of
aggregation." This script identifies *how* by testing every combination of:

    column ∈ {Full sample, 99% sample, 99% sample + corrections, 97.5% sample}
    scheme ∈ {arith mean of θ, harmonic mean of θ (arith mean of 1/θ),
              N-weighted mean of θ, N-weighted mean of 1/θ}

For each combination we compute the derived AT θ vector and score it against
AT-2016 Table 9 by RMSE/MAE on the 8 manufacturing rows. Service sectors
(ISIC 40+) are imposed at θ=5 in AT per their footnote citing
Costinot-Rodríguez-Clare 2014; excluded from scoring.

Run from cprhs/:
    uv run python scripts/derive_at_theta.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── CP-2015 Table 1 (paper p.18, year 1993) ────────────────────────────────
# Per-sector θ across the three sample-restriction columns. N is the 99%
# sample column N from CP Table 1; used as a weight in N-weighted schemes
# (CP report separate N for full and 97.5% but we don't carry them — a
# refinement if 99%-N weights turn out to win).
CP2015: dict[int, tuple[str, float, float, float, int]] = {
    1:  ("Agriculture",         8.11,  9.11,  16.88, 430),
    2:  ("Mining",              15.72, 13.53, 17.39, 178),
    3:  ("Food",                2.55,  2.62,  2.46,  429),
    4:  ("Textile",             5.56,  8.10,  1.74,  314),
    5:  ("Wood",                10.83, 11.50, 11.22, 191),
    6:  ("Paper",               9.07,  16.52, 2.57,  352),
    7:  ("Petroleum",           51.08, 64.85, 61.25, 86),
    8:  ("Chemicals",           4.75,  3.13,  2.94,  341),
    9:  ("Plastics",            1.66,  1.67,  0.60,  272),
    10: ("Minerals",            2.76,  2.41,  2.99,  263),
    11: ("Basic metals",        7.99,  3.28,  -0.05, 288),
    12: ("Fabricated metals",   4.30,  6.99,  0.52,  314),
    13: ("Machinery n.e.c.",    1.52,  1.45,  -2.82, 290),
    14: ("Office, computing",   12.79, 12.95, 11.47, 126),
    15: ("Electrical",          10.60, 12.91, 3.37,  269),
    16: ("Communication",       7.07,  3.95,  4.82,  143),
    17: ("Medical",             9.98,  8.71,  1.97,  237),
    18: ("Auto",                1.01,  1.84,  -3.06, 126),
    19: ("Other transport",     0.37,  0.39,  0.53,  226),
    20: ("Other manufacturing", 5.00,  3.98,  3.06,  227),
}

# Pooled mfg-aggregate per CP-2015 sample column (Table 1 last row). Used to
# implement CP's published substitution rule: for Basic metals, Machinery
# n.e.c., and Auto, replace the per-sector estimate with this aggregate.
CP2015_MFG_AGG: dict[str, float] = {"full": 4.45, "99": 4.49, "975": 3.29}

# Sectors CP-2015 themselves override (sign-flipped under restriction).
BENCHMARK_SUBS: tuple[int, ...] = (11, 13, 18)

# Number of ISIC Rev3 2-digit codes each CP sub-sector covers. Used by the
# ISIC-count weighted aggregations: if AT aggregates by giving each underlying
# 2-digit ISIC code equal weight (rather than each CP sub-sector equal
# weight), this is the weight. Only differs from simple-mean for the
# Food+Textile and Ag+Mining buckets (within an AT row, ISIC-count weighting
# is identical to simple mean wherever the constituent CP sub-sectors all
# cover the same number of 2-digit codes).
CP_ISIC2_COUNT: dict[int, int] = {
    1: 5,   # Agriculture covers ISIC 01-05
    2: 5,   # Mining 10-14
    3: 2,   # Food 15-16
    4: 3,   # Textile 17-19
    5: 1,   # Wood 20
    6: 2,   # Paper 21-22
    7: 1,   # Petroleum 23
    8: 1,   # Chemicals 24
    9: 1,   # Plastics 25
    10: 1,  # Minerals 26
    11: 1,  # Basic metals 27
    12: 1,  # Fabricated metals 28
    13: 1,  # Machinery n.e.c. 29
    14: 1,  # Office 30
    15: 1,  # Electrical 31
    16: 1,  # Communication 32
    17: 1,  # Medical 33
    18: 1,  # Auto 34
    19: 1,  # Other transport 35
    20: 2,  # Other manufacturing 36-37
}


def cp_theta(cp_id: int, column: str) -> float:
    """CP-2015 θ for a sub-sector under a column choice.

    Columns:
        "full"  — Full sample θ
        "99"    — 99% sample θ, raw (no substitutions)
        "99sub" — 99% sample with CP's published substitutions (the benchmark)
        "975"   — 97.5% sample θ
    """
    _, full, t99, t975, _ = CP2015[cp_id]
    if column == "full":
        return full
    if column == "99":
        return t99
    if column == "99sub":
        return CP2015_MFG_AGG["99"] if cp_id in BENCHMARK_SUBS else t99
    if column == "975":
        return t975
    raise ValueError(f"unknown column: {column}")


def aggregate(scheme: str, column: str, cp_ids: list[int]) -> float:
    """Aggregate a list of CP sub-sectors into one AT-row θ value."""
    thetas = np.array([cp_theta(i, column) for i in cp_ids])
    n_w = np.array([CP2015[i][4] for i in cp_ids], dtype=float)  # N_99
    isic_w = np.array([CP_ISIC2_COUNT[i] for i in cp_ids], dtype=float)

    if scheme == "arith_theta":
        return float(np.mean(thetas))
    if scheme == "arith_invtheta":   # harmonic mean of θ
        return float(1.0 / np.mean(1.0 / thetas))
    if scheme == "N_theta":
        return float(np.average(thetas, weights=n_w))
    if scheme == "N_invtheta":
        return float(1.0 / np.average(1.0 / thetas, weights=n_w))
    if scheme == "ISIC_theta":   # weight each constituent by its ISIC-2-digit code count
        return float(np.average(thetas, weights=isic_w))
    if scheme == "ISIC_invtheta":
        return float(1.0 / np.average(1.0 / thetas, weights=isic_w))
    raise ValueError(f"unknown scheme: {scheme}")


# ── AT-2016 Table 9 (paper p.26) ───────────────────────────────────────────
# Format: AT sector → (ISIC range, published θ, [CP sub-sector IDs])
# Services and other non-manufacturing (ISIC 40+) have empty CP lists — AT
# impose θ=5 per their footnote citing Costinot-Rodríguez-Clare (2014).
AT2016: dict[str, tuple[str, float, list[int]]] = {
    "Agriculture, Mining":     ("01-14", 11.92, [1, 2]),
    "Food, Textiles":          ("15-19", 4.56,  [3, 4]),
    "Wood":                    ("20",    10.83, [5]),
    "Paper":                   ("21-22", 9.07,  [6]),
    "Chemicals, Rubber":       ("23-25", 19.16, [7, 8, 9]),
    "Metals":                  ("26-28", 5.02,  [10, 11, 12]),
    "Equipment, Vehicles":     ("29-35", 6.19,  [13, 14, 15, 16, 17, 18, 19]),
    "Manufacturing, n.e.c.":   ("36-37", 5.00,  [20]),
    "Utilities":               ("40-41", 5.00,  []),
    "Construction":            ("45",    5.00,  []),
    "Wholesale and Retail":    ("50-52", 5.00,  []),
    "Hotels and Restaurants":  ("55",    5.00,  []),
    "Transport":               ("60-63", 5.00,  []),
    "Communication":           ("64",    5.00,  []),
    "Finance":                 ("65-67", 5.00,  []),
    "Real Estate":             ("70-71", 5.00,  []),
    "Software":                ("72",    5.00,  []),
    "Other Business Services": ("73-74", 5.00,  []),
    "Public Admin.":           ("75",    5.00,  []),
    "Education":               ("80",    5.00,  []),
    "Health and Social":       ("85",    5.00,  []),
    "Other Services":          ("90-93", 5.00,  []),
}

COLUMNS = ("full", "99", "99sub", "975")
SCHEMES = ("arith_theta", "arith_invtheta",
           "N_theta",     "N_invtheta",
           "ISIC_theta",  "ISIC_invtheta")


def derive_at_vector(column: str, scheme: str) -> pd.DataFrame:
    """Return per-AT-sector (ISIC, AT θ, derived θ) for one (column, scheme)."""
    rows = []
    for at_name, (isic, at_theta, cp_ids) in AT2016.items():
        derived = aggregate(scheme, column, cp_ids) if cp_ids else float("nan")
        rows.append({"AT sector": at_name, "ISIC": isic,
                     "AT θ": at_theta, "derived θ": derived})
    return pd.DataFrame(rows)


def score(df: pd.DataFrame) -> tuple[float, float, int]:
    """RMSE, MAE, and number of exact matches (|resid| < 0.01) on
    manufacturing rows (those with a non-NaN derived θ)."""
    mfg = df.dropna(subset=["derived θ"])
    diff = mfg["derived θ"] - mfg["AT θ"]
    rmse = float(np.sqrt((diff ** 2).mean()))
    mae = float(diff.abs().mean())
    exact = int((diff.abs() < 0.01).sum())
    return rmse, mae, exact


def main() -> None:
    print("Part 2a — derive AT-2016 Table 9 from CP-2015.\n")

    score_rows = []
    for column in COLUMNS:
        for scheme in SCHEMES:
            df = derive_at_vector(column, scheme)
            rmse, mae, exact = score(df)
            score_rows.append({"column": column, "scheme": scheme,
                                "RMSE": rmse, "MAE": mae, "exact/8": exact})

    score_df = pd.DataFrame(score_rows).sort_values("RMSE").reset_index(drop=True)
    print("Score against AT-2016 Table 9 manufacturing rows "
          f"(8 of 22 sectors), {len(COLUMNS)*len(SCHEMES)} combinations:\n")
    print(score_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    best = score_df.iloc[0]
    print(f"\nBest fit:  column = {best['column']!r}, scheme = {best['scheme']!r}"
          f"   →   RMSE = {best['RMSE']:.4f}, exact matches = {best['exact/8']}/8\n")

    # Per-sector side-by-side: AT θ vs every scheme under the winning column.
    best_col = best["column"]
    out = pd.DataFrame(
        {"AT sector": [n for n in AT2016],
         "ISIC":      [v[0] for v in AT2016.values()],
         "AT θ":      [v[1] for v in AT2016.values()]}
    )
    for scheme in SCHEMES:
        df = derive_at_vector(best_col, scheme)
        out[scheme] = df["derived θ"]
    print(f"Per-sector AT θ vs each scheme under column={best_col!r}:\n")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # Spotlight Food, Textiles — the only manufacturing row that simple-mean
    # under Full sample does not reproduce exactly.
    print("\nFood, Textiles diagnostic (AT publishes 4.56):\n")
    food_textile = []
    for column in COLUMNS:
        for scheme in SCHEMES:
            v = aggregate(scheme, column, [3, 4])
            food_textile.append({"column": column, "scheme": scheme,
                                  "derived θ": v, "residual": v - 4.56})
    ft_df = pd.DataFrame(food_textile).sort_values("residual",
                                                    key=lambda s: s.abs())
    print(ft_df.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))


if __name__ == "__main__":
    main()
