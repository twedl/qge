"""Table 2 — Decomposing the sources of fiscal transfers (2007–2016).

Port of Table2.R + Table2.m combined. Two stages:

1. Build the per-province × per-component "TotalGap" matrix in Python
   (replacement for the R pipeline that produces ``ForMATLAB_Table2``).
2. Compute average marginal contributions per component via Shapley value
   (replacement for the MATLAB permutation-based code; analytically
   equivalent to averaging across all 10! = 3.6 million orderings).

Reports three columns:
  - Share of GDP (%)
  - Share of transfers (%)
  - Correlation with GDP per capita

Components, in the order MATLAB and the paper use:
  1. Personal Income Taxes        (From households)
  2. Equalization/Stabilization   (-Taxation agreements)
  3. Corporate Income Taxes       (From corporations, gbe liabilities)
  4. CPP Net Contributions        (CPP surplus or deficit)
  5. Non-Defence Purchases        (-Non-defence)
  6. EI Payments less Receipts    (-EI benefits + EI contributions)
  7. OAS Benefits                 (-Old Age Security)
  8. GST and Excise Taxes         (Taxes on products)
  9. Defence Purchases            (-Defence)
 10. Other Items                  (residual)

Run from tw/:
    uv run python scripts/table2.py
"""

from __future__ import annotations

from itertools import combinations
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd

INPUTS = Path(__file__).resolve().parent.parent / "data" / "inputs" / "canada_2010"

# Total fiscal transfers as a share of Canadian GDP, 2007-2016 (the multiplier
# Table2.m uses for column 1). The same figure is computed inline below from
# the data (the "transfer/GDP" line at the bottom of Table2.R) but we keep
# the literal constant as the published reference.
TOTAL_TRANSFER_PCT_OF_GDP = 1.92

PROV_ORDER = ("BC", "AB", "SK", "MB", "ON", "QC", "NB", "NS", "PE", "NL")

# Component labels in the order they will be reported in
COMPONENT_LABELS = (
    "Personal Income Taxes",
    "Equalization / Stabilization",
    "Corporate Income Taxes",
    "CPP Net Contributions",
    "Non-Defence Purchases",
    "EI Payments less Receipts",
    "OAS Benefits",
    "GST and Excise Taxes",
    "Defence Purchases",
    "Other Items",
)


def _build_components_matrix() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (transfers_matrix, pop_by_GEO, gdp_by_GEO).

    transfers_matrix: rows = provinces (in PROV_ORDER), columns = components
    (in COMPONENT_LABELS), values = annual average TotalGap (province's
    deviation in dollars from the Canadian per-capita average for that
    component, summed across 2007-2016, divided by 10 years).
    """
    currentdata = pd.read_parquet(INPUTS / "fiscal_currentdata.parquet")
    pop = pd.read_parquet(INPUTS / "fiscal_pop.parquet")
    gdp = pd.read_parquet(INPUTS / "fiscal_GDP_prov.parquet")
    provnames = pd.read_parquet(INPUTS / "fiscal_provnames.parquet")

    # ── 1. Pull CPP surplus/deficit out and relabel as a federal category ──
    cpp = currentdata[
        (currentdata["Levels.of.government"] == "Canada Pension Plan")
        & (currentdata["Estimates"] == "General governments surplus or deficit")
    ].copy()
    cpp["Levels.of.government"] = "Federal general government"
    cpp["Estimates"] = "CPP Net Contributions"

    data = pd.concat(
        [currentdata[currentdata["Levels.of.government"] == "Federal general government"],
         cpp],
        ignore_index=True,
    )

    # ── 2. Combine EI categories (benefits w/ inverted sign + contributions) ──
    data["Value"] = np.where(
        data["Estimates"] == "Employment Insurance benefits",
        -data["Value"], data["Value"],
    )
    data["Estimates"] = data["Estimates"].replace({
        "Employment Insurance benefits": "EI Payments less Receipts",
        "Contributions to social insurance plans": "EI Payments less Receipts",
    })

    # ── 3. Keep only the relevant categories + exclude Canada + territories ──
    keep_categories = [
        "From households",
        "From corporations and government business enterprises, liabilities",
        "CPP Net Contributions",
        "Defence",
        "Non-defence",
        "EI Payments less Receipts",
        "Taxes on products",
        "Old Age Security",
        "Taxation agreements",
        "General governments surplus or deficit",
    ]
    data = data[data["Estimates"].isin(keep_categories)]
    data = data[~data["GEO"].isin(
        ("Canada", "Outside Canada", "Yukon", "Nunavut", "Northwest Territories"))]

    # ── 4. Sign flips on spending-from-federal categories ────────────────
    flip_signs = {"Taxation agreements", "Defence", "Non-defence", "Old Age Security"}
    data["Value"] = np.where(data["Estimates"].isin(flip_signs),
                              -data["Value"], data["Value"])

    # ── 5. Aggregate to (Ref_Date, GEO, EST) ─────────────────────────────
    agg = (data.groupby(["Ref_Date", "GEO", "Estimates"], as_index=False)["Value"]
                .sum())

    # ── 6. Derive "Other Factors" residual from federal surplus/deficit ──
    # other = sum(federal_sd + CPP) − sum(everything except federal_sd)
    # And replace the federal_sd row with the "Other Factors" row.
    def _residualize(g: pd.DataFrame) -> pd.DataFrame:
        is_fed_sd = g["Estimates"] == "General governments surplus or deficit"
        is_cpp = g["Estimates"] == "CPP Net Contributions"
        total = g.loc[is_fed_sd | is_cpp, "Value"].sum()
        temp = g.loc[~is_fed_sd, "Value"].sum()
        other = total - temp
        g = g.copy()
        g.loc[is_fed_sd, "Value"] = other
        g.loc[is_fed_sd, "Estimates"] = "Other Factors"
        return g

    agg = (agg.groupby(["Ref_Date", "GEO"], group_keys=False)
                .apply(_residualize, include_groups=False))
    # Re-attach the group keys (apply with include_groups=False drops them)
    grouped = (data.groupby(["Ref_Date", "GEO", "Estimates"], as_index=False)["Value"]
                    .sum())
    agg_full = []
    for (yr, geo), g in grouped.groupby(["Ref_Date", "GEO"]):
        agg_full.append(_residualize(g).assign(Ref_Date=yr, GEO=geo))
    agg = pd.concat(agg_full, ignore_index=True)

    # ── 7. Join pop/GDP, compute PerCapita and TotalGap ──────────────────
    agg = agg.merge(provnames, on="GEO", how="left")
    agg = agg.merge(pop, on=["Ref_Date", "short"], how="left")
    agg = agg.merge(gdp[["Ref_Date", "GEO", "GDP"]], on=["Ref_Date", "GEO"], how="left")

    # PerCapita per (year, GEO, EST)
    agg["PerCapita"] = 1_000_000 * agg["Value"] / agg["pop"]

    # AvgPerCapita per (year, EST) across provinces
    avg = (agg.groupby(["Ref_Date", "Estimates"], as_index=False)
               .apply(lambda g: pd.Series({
                   "AvgPerCapita": 1_000_000 * g["Value"].sum() / g["pop"].sum()
               }), include_groups=False))
    agg = agg.merge(avg, on=["Ref_Date", "Estimates"], how="left")

    agg["GapPerCapita"] = agg["PerCapita"] - agg["AvgPerCapita"]
    agg["TotalGap"] = agg["GapPerCapita"] * agg["pop"] / 1_000_000

    # Relabel categories to display labels
    label_map = {
        "From households":                                                 "Personal Income Taxes",
        "From corporations and government business enterprises, liabilities":
                                                                            "Corporate Income Taxes",
        "CPP Net Contributions":                                            "CPP Net Contributions",
        "Defence":                                                          "Defence Purchases",
        "Non-defence":                                                      "Non-Defence Purchases",
        "EI Payments less Receipts":                                        "EI Payments less Receipts",
        "Taxes on products":                                                "GST and Excise Taxes",
        "Old Age Security":                                                 "OAS Benefits",
        "Taxation agreements":                                              "Equalization / Stabilization",
        "Other Factors":                                                    "Other Items",
    }
    agg["component"] = agg["Estimates"].map(label_map)

    # ── 8. Sum TotalGap across 2007-2016, divide by 10 years ─────────────
    # No further sign-flip — the R pipeline only flips Taxation agreements /
    # Defence / Non-defence / OAS once (step 4 above), and the resulting
    # ForMATLAB_Table2 matrix has those categories positive for net
    # recipient provinces (= "this much fiscal transfer flowed in").
    avg_gap = (agg.groupby(["short", "component"], as_index=False)["TotalGap"]
                    .sum())
    avg_gap["TotalGap"] = avg_gap["TotalGap"] / 10  # annual average

    transfers = avg_gap.pivot(index="short", columns="component", values="TotalGap")
    transfers = transfers.reindex(index=list(PROV_ORDER),
                                    columns=list(COMPONENT_LABELS))

    pop_by_geo = pop.groupby("short", as_index=False)["pop"].sum().query("short in @PROV_ORDER")
    gdp_by_geo = gdp.groupby("short", as_index=False)["GDP"].sum().query("short in @PROV_ORDER")
    return transfers, pop_by_geo, gdp_by_geo


def _shapley_shares(transfers: np.ndarray) -> np.ndarray:
    """Average marginal contribution of each component (Shapley value).

    For a coalition-game value function v(S) = ½·Σ_n |Σ_{j∈S} T_{n,j}|,
    Shapley(j) = Σ_{S not containing j} [|S|! (n−1−|S|)! / n!] · (v(S∪{j}) − v(S)).

    The MATLAB code enumerates all 10! = 3,628,800 column orderings; the
    Shapley formula computes the same answer in O(n · 2^n) work.
    """
    n_provinces, n_comp = transfers.shape

    # Cache v(S) for every subset
    v_cache = np.zeros(1 << n_comp)
    for mask in range(1, 1 << n_comp):
        cols = [j for j in range(n_comp) if mask & (1 << j)]
        v_cache[mask] = 0.5 * np.abs(transfers[:, cols].sum(axis=1)).sum()

    shapley = np.zeros(n_comp)
    for j in range(n_comp):
        for k in range(n_comp):
            for S in combinations([i for i in range(n_comp) if i != j], k):
                mask = sum(1 << i for i in S)
                w = factorial(k) * factorial(n_comp - 1 - k) / factorial(n_comp)
                shapley[j] += w * (v_cache[mask | (1 << j)] - v_cache[mask])
    return shapley


def main() -> None:
    transfers_df, pop_df, gdp_df = _build_components_matrix()
    transfers = transfers_df.to_numpy()

    print("Per-province annual-average TotalGap matrix (in dollars):\n")
    print(transfers_df.round(0).to_string(float_format=lambda v: f"{v:>10,.0f}"))

    # Total transfers as % of GDP (cross-check the 1.92 constant)
    total_transfers = 0.5 * np.abs(transfers.sum(axis=1)).sum()
    total_gdp_annual = gdp_df["GDP"].sum() / 10
    total_pct_gdp = 100 * total_transfers / total_gdp_annual
    print(f"\nTotal annual transfers: ${total_transfers:,.1f}M")
    print(f"Avg annual GDP:          ${total_gdp_annual:,.1f}M")
    print(f"Transfers / GDP:         {total_pct_gdp:.2f}%   (paper: 1.92%)\n")

    # Shapley shares
    shapley = _shapley_shares(transfers)
    column2 = shapley / shapley.sum() * 100               # share of transfers (%)
    column1 = column2 * TOTAL_TRANSFER_PCT_OF_GDP / 100   # share of GDP (%)

    # Correlation with GDP per capita. The R script in Table2.R flips the
    # spending-on-province categories BACK to their natural sign convention
    # before computing this correlation (so e.g. equalization shows the
    # expected -0.77 with GDP/capita rather than the +0.78 it has under the
    # internal Shapley accounting convention). See Table2.R lines 78-79.
    gdp_df = gdp_df.set_index("short").loc[list(PROV_ORDER)]
    pop_df = pop_df.set_index("short").loc[list(PROV_ORDER)]
    gdp_per_capita = (gdp_df["GDP"] / pop_df["pop"] * 1_000_000).to_numpy()
    flip_back = {"Equalization / Stabilization", "Defence Purchases",
                  "OAS Benefits", "Non-Defence Purchases"}
    sign = np.array([-1 if c in flip_back else 1 for c in COMPONENT_LABELS])
    per_capita = sign * transfers / pop_df["pop"].to_numpy()[:, None] * 1_000_000
    correl = np.array([np.corrcoef(gdp_per_capita, per_capita[:, j])[0, 1]
                        for j in range(transfers.shape[1])])

    # ── Display sorted by share-of-transfers, with Other Items last ──────
    rows = list(zip(COMPONENT_LABELS, column1, column2, correl))
    other_row = [r for r in rows if r[0] == "Other Items"][0]
    rest = [r for r in rows if r[0] != "Other Items"]
    rest_sorted = sorted(rest, key=lambda r: -r[1])

    print("=== Table 2: Decomposing the sources of fiscal transfers (2007-2016) ===\n")
    print(f"  {'Component':<32} {'GDP %':>8} {'Trnsfr %':>10} {'Corr/GDPpc':>12}")
    print(f"  {'-' * 64}")
    for name, c1, c2, cor in rest_sorted + [other_row]:
        print(f"  {name:<32} {c1:>8.2f} {c2:>10.1f} {cor:>+12.2f}")


if __name__ == "__main__":
    main()
