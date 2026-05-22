"""Table 1 — Interprovincial fiscal transfers in Canada (2007–2016).

Direct port of Table1.R. Reads the four FiscalData parquets (written by
either ``scripts/convert_fiscaldata.py`` from the shipped RData or
``scripts/fetch_fiscal_data.py`` from the StatCan WDS API) and reports
per-capita expenditure / revenue gaps vs the national average, the
implied net transfer, GDP per capita, and population / GDP shares.

Run from tw/:
    uv run python scripts/table1.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

INPUTS = Path(__file__).resolve().parent.parent / "data" / "inputs" / "canada_2010"


def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_parquet(INPUTS / "fiscal_currentdata.parquet"),
        pd.read_parquet(INPUTS / "fiscal_pop.parquet"),
        pd.read_parquet(INPUTS / "fiscal_GDP_prov.parquet"),
        pd.read_parquet(INPUTS / "fiscal_provnames.parquet"),
    )


def main() -> None:
    currentdata, pop, gdp, provnames = _load()

    # ── 1. Filter to federal + CPP revenue and expenditure ──────────────
    fed = currentdata[
        currentdata["Levels.of.government"].isin(
            ["Canada Pension Plan", "Federal general government"])
        & currentdata["Estimates"].isin(
            ["General governments revenue", "General governments expenditure"])
        & (currentdata["GEO"] != "Outside Canada")
    ].copy()
    fed["category"] = np.where(
        fed["Estimates"].str.contains("revenue", case=False), "revenue", "expenditure"
    )

    rev_exp = (fed.groupby(["Ref_Date", "GEO", "category"], as_index=False)["Value"].sum()
                   .pivot(index=["Ref_Date", "GEO"], columns="category", values="Value")
                   .reset_index())

    # ── 2. Join with provnames, pop, GDP ─────────────────────────────────
    df = (rev_exp.merge(provnames, on="GEO", how="left")
                  .merge(pop, on=["Ref_Date", "short"], how="left")
                  .merge(gdp[["Ref_Date", "short", "GDP"]],
                         on=["Ref_Date", "short"], how="left"))

    # ── 3. Aggregate over 2007-2016 ──────────────────────────────────────
    agg = (df.groupby("short", as_index=False)[["expenditure", "revenue", "pop", "GDP"]]
              .sum())
    agg = agg[agg["pop"].notna() & (agg["pop"] > 0)]
    # Drop Canada total since per-capita reporting is by-province
    agg = agg[agg["short"] != "CAN"]

    # ── 4. Per-capita gaps relative to the national average ──────────────
    exppc_CDN = agg["expenditure"].sum() / agg["pop"].sum()
    revpc_CDN = agg["revenue"].sum() / agg["pop"].sum()
    agg["exppc"] = agg["expenditure"] / agg["pop"]
    agg["revpc"] = agg["revenue"] / agg["pop"]
    agg["expgap"] = ((agg["exppc"] - exppc_CDN) * 1_000_000).round().astype(int)
    agg["revgap"] = ((agg["revpc"] - revpc_CDN) * 1_000_000).round().astype(int)
    agg["transfer"] = agg["expgap"] - agg["revgap"]
    agg["GDPpc"] = (agg["GDP"] / agg["pop"] * 1_000_000).round().astype(int)
    agg["popshare"] = (100 * (agg["pop"] / agg["pop"].sum())).round(1)
    agg["gdpshare"] = (100 * (agg["GDP"] / agg["GDP"].sum())).round(1)

    # ── 5. Display matching the paper's province order ──────────────────
    display_order = ("BC", "AB", "SK", "MB", "ON", "QC",
                      "NB", "NS", "PE", "NL")
    out = agg.set_index("short").loc[list(display_order)].reset_index()

    print("\n=== Table 1: Interprovincial fiscal transfers in Canada (2007-2016) ===\n")
    print(f"  {'Prov':<5} {'F. spend gap':>14} {'F. rev gap':>12} {'Net trnsfr':>12} "
          f"{'GDP/cap':>10} {'Pop %':>7} {'Econ %':>7}")
    print(f"  {'-' * 71}")
    for _, r in out.iterrows():
        print(f"  {r['short']:<5} "
              f"{r['expgap']:>14,d} {r['revgap']:>12,d} {r['transfer']:>12,d} "
              f"{r['GDPpc']:>10,d} {r['popshare']:>7.1f} {r['gdpshare']:>7.1f}")


if __name__ == "__main__":
    main()
