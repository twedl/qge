"""Fetch StatCan fiscal-data tables via the WDS API.

Replaces the manual FiscalData.RData dependency. Pulls three tables and
writes long-form parquets under ``data/inputs/canada_2010/``:

  36-10-0450 → fiscal_currentdata.parquet  (federal/CPP rev & expenditure)
  17-10-0005 → fiscal_pop.parquet          (population by province)
  36-10-0222 → fiscal_GDP_prov.parquet     (nominal GDP by province)

The replication package shipped a 2017 vintage of these tables in
FiscalData.RData. This fetcher pulls the *current* vintage from StatCan,
which may differ slightly due to data revisions. For exact reproducibility
against the paper, prefer ``convert_fiscaldata.py`` (which re-shapes the
shipped RData file).

Run from tw/:
    uv run python scripts/fetch_fiscal_data.py
"""

from __future__ import annotations

import json
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

WDS_BASE = "https://www150.statcan.gc.ca/t1/wds/rest"
CACHE_DIR = Path("/tmp/statcan_fiscal")
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "inputs" / "canada_2010"

TABLES = {
    "36100450": "fiscal_currentdata",  # federal rev & expenditure
    "17100005": "fiscal_pop",          # population
    "36100222": "fiscal_GDP_prov",     # GDP at current prices, expenditure-based
}

# Province / Canada labels we keep (match the provnames in the RData)
KEEP_GEO = (
    "Canada", "Newfoundland and Labrador", "Prince Edward Island",
    "Nova Scotia", "New Brunswick", "Quebec", "Ontario", "Manitoba",
    "Saskatchewan", "Alberta", "British Columbia",
    "Yukon", "Northwest Territories", "Nunavut",
)
PROVNAMES_SHORT = {
    "Canada": "CAN",
    "Newfoundland and Labrador": "NL", "Prince Edward Island": "PE",
    "Nova Scotia": "NS", "New Brunswick": "NB", "Quebec": "QC",
    "Ontario": "ON", "Manitoba": "MB", "Saskatchewan": "SK",
    "Alberta": "AB", "British Columbia": "BC",
    "Yukon": "YT", "Northwest Territories": "NT", "Nunavut": "NU",
}


def _fetch_table(pid: str) -> Path:
    """Download a StatCan full-table CSV via the WDS API; cache locally."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = CACHE_DIR / f"{pid}.csv"
    if csv_path.exists():
        return csv_path

    zip_path = CACHE_DIR / f"{pid}-eng.zip"
    if not zip_path.exists():
        api_url = f"{WDS_BASE}/getFullTableDownloadCSV/{pid}/en"
        print(f"  GET {api_url}")
        with urllib.request.urlopen(api_url) as r:
            resp = json.load(r)
        if resp.get("status") != "SUCCESS":
            raise RuntimeError(f"WDS API failed for {pid}: {resp}")
        download_url = resp["object"]
        print(f"  → {download_url}")
        with urllib.request.urlopen(download_url) as src, open(zip_path, "wb") as dst:
            dst.write(src.read())

    with zipfile.ZipFile(zip_path) as zf:
        target = next(name for name in zf.namelist() if name.endswith(f"{pid}.csv"))
        with zf.open(target) as src, open(csv_path, "wb") as dst:
            dst.write(src.read())
    return csv_path


def _to_year(s: pd.Series) -> pd.Series:
    """Extract numeric year from a REF_DATE column that may be 'YYYY' or 'YYYY-MM'."""
    s = s.astype(str)
    return pd.to_numeric(s.str[:4], errors="coerce").astype("Int64")


def _build_currentdata(csv: Path) -> pd.DataFrame:
    """36-10-0450: federal & CPP revenue/expenditure by Estimate × GEO × year."""
    df = pd.read_csv(csv, low_memory=False, dtype={"VALUE": float})
    df = df.rename(columns={"REF_DATE": "Ref_Date", "VALUE": "Value"})
    df["Ref_Date"] = _to_year(df["Ref_Date"])
    df = df[df["GEO"].isin(KEEP_GEO)]
    df = df[df["Levels of government"].isin(
        ["Canada Pension Plan", "Federal general government"])]
    df["Levels.of.government"] = df["Levels of government"]
    return df[["Ref_Date", "GEO", "DGUID", "Levels.of.government", "Estimates",
                "UOM", "SCALAR_FACTOR", "VECTOR", "COORDINATE", "Value",
                "STATUS", "SYMBOL", "TERMINATED", "DECIMALS"]]


def _build_pop(csv: Path) -> pd.DataFrame:
    """17-10-0005: provincial population by year. Output (Ref_Date, short, pop)."""
    df = pd.read_csv(csv, low_memory=False, dtype={"VALUE": float})
    df = df.rename(columns={"REF_DATE": "Ref_Date", "VALUE": "Value"})
    df["Ref_Date"] = _to_year(df["Ref_Date"])
    df = df[df["GEO"].isin(KEEP_GEO)]
    # Keep age "All ages" + sex "Both sexes" rows (population total)
    df = df[df["Sex"] == "Both sexes"]
    df = df[df["Age group"] == "All ages"]
    df["short"] = df["GEO"].map(PROVNAMES_SHORT)
    df = df.dropna(subset=["short", "Value"])
    df = df.rename(columns={"Value": "pop"})
    return df[["Ref_Date", "short", "pop"]].drop_duplicates(["Ref_Date", "short"])


def _build_gdp(csv: Path) -> pd.DataFrame:
    """36-10-0222 expenditure-based provincial GDP at current prices. Output
    (Ref_Date, GEO, GDP, short, pop, gdp_pc) — pop/gdp_pc filled later by merging."""
    df = pd.read_csv(csv, low_memory=False, dtype={"VALUE": float})
    df = df.rename(columns={"REF_DATE": "Ref_Date", "VALUE": "Value"})
    df["Ref_Date"] = _to_year(df["Ref_Date"])
    df = df[df["GEO"].isin(KEEP_GEO)]
    df = df[df["Prices"] == "Current prices"]
    df = df[df["Estimates"] == "Gross domestic product at market prices"]
    df["short"] = df["GEO"].map(PROVNAMES_SHORT)
    df = df.dropna(subset=["short", "Value"])
    df = df.rename(columns={"Value": "GDP"})
    return df[["Ref_Date", "GEO", "GDP", "short"]].drop_duplicates(["Ref_Date", "GEO"])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {OUTPUT_DIR}\n")

    print("Fetching StatCan tables …")
    csv36100450 = _fetch_table("36100450")
    csv17100005 = _fetch_table("17100005")
    csv36100222 = _fetch_table("36100222")

    print("\nBuilding parquets …")
    currentdata = _build_currentdata(csv36100450)
    currentdata.to_parquet(OUTPUT_DIR / "fiscal_currentdata.parquet", index=False)
    print(f"  fiscal_currentdata.parquet         "
          f"{len(currentdata):,} rows, {currentdata['Ref_Date'].min()}-{currentdata['Ref_Date'].max()}")

    pop = _build_pop(csv17100005)
    pop.to_parquet(OUTPUT_DIR / "fiscal_pop.parquet", index=False)
    print(f"  fiscal_pop.parquet                 "
          f"{len(pop):,} rows, {pop['Ref_Date'].min()}-{pop['Ref_Date'].max()}")

    gdp = _build_gdp(csv36100222)
    # Join in pop so the parquet matches the RData schema exactly
    gdp = gdp.merge(pop, on=["Ref_Date", "short"], how="left")
    gdp["gdp_pc"] = gdp["GDP"] / gdp["pop"]
    gdp.to_parquet(OUTPUT_DIR / "fiscal_GDP_prov.parquet", index=False)
    print(f"  fiscal_GDP_prov.parquet            "
          f"{len(gdp):,} rows, {gdp['Ref_Date'].min()}-{gdp['Ref_Date'].max()}")

    provnames = pd.DataFrame(
        [{"GEO": geo, "short": short}
         for geo, short in PROVNAMES_SHORT.items()]
    )
    provnames.to_parquet(OUTPUT_DIR / "fiscal_provnames.parquet", index=False)
    print(f"  fiscal_provnames.parquet           {len(provnames)} rows")

    print("\nDone.")


if __name__ == "__main__":
    main()
