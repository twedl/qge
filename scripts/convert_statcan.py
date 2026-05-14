"""Build a Canadian calibration of the qge model from Statistics Canada data.

So far this covers only `bilateral_trade.parquet`, sourced from Table
12-10-0088-01 (Interprovincial and international trade flows, summary level).
Subsequent passes will add the other seven inputs (employment, IO matrix,
value-added share, etc.) from related StatCan Supply-Use-Tables.

Usage::

    uv run python scripts/convert_statcan.py            # default year 2019
    uv run python scripts/convert_statcan.py --year 2017
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from qge.io import INPUTS_ROOT, _array_to_long

# ---------------------------------------------------------------- constants

CACHE_DIR = Path("/tmp/statcan")
WDS_BASE = "https://www150.statcan.gc.ca/t1/wds/rest"

DEFAULT_YEAR = 2019

# Ten provinces. The territories (Yukon, Northwest Territories, Nunavut) are
# dropped — they are very small and produce zero output in many sectors, which
# would violate the interior-equilibrium requirement.
PROVINCES: tuple[str, ...] = (
    "Newfoundland and Labrador",
    "Prince Edward Island",
    "Nova Scotia",
    "New Brunswick",
    "Quebec",
    "Ontario",
    "Manitoba",
    "Saskatchewan",
    "Alberta",
    "British Columbia",
)

# Statistics Canada table identifiers.
TABLE_TRADE_SUMMARY = "12100088"  # Interprovincial and international trade flows, summary level


# StatCan summary-level products aggregate into this 22-sector target taxonomy.
# Choice notes:
#  - Roughly NAICS 2-digit on the services side; manufacturing split into
#    ~8 sub-sectors (matching CPRHS-style granularity).
#  - Pure intermediate aggregates (Wholesale margins, Retail margins) live with
#    "Trade".
#  - Government services (G-codes) are folded into Education / Health / Public
#    Administration alongside their commercial counterparts.
#  - "Other Services" absorbs M81 + M9 codes plus non-profit (N) services.
#
# Encoded as (target_sector, [list_of_StatCan_product_codes_assigned_to_it]).
# Edit this list to retune the taxonomy.
SECTOR_AGGREGATION: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Agriculture, Forestry, Fishing",
        ("M111B", "M112A", "M11D0", "M11E0", "M1140", "M1150")),
    ("Mining and Extraction",
        ("M21B0", "M2122", "M2123", "M2130", "M21A0")),
    ("Utilities", ("M2200",)),
    ("Construction",
        ("M23A0", "M23B0", "M23C0", "M23D0")),
    ("Food, Beverage, Tobacco",
        ("M31C0", "M312A")),
    ("Textile, Apparel, Leather", ("M31D0",)),
    ("Wood, Paper, Printing",
        ("M3210", "M3220", "M3230")),
    ("Petroleum and Chemicals",
        ("M3240", "M3250", "M3260")),
    ("Non-metallic Mineral Products", ("M3270",)),
    ("Metals and Machinery",
        ("M3310", "M3320", "M3330")),
    ("Computers, Electronics, Electrical",
        ("M334C", "M3350")),
    ("Transportation Equipment",
        ("M3363", "M336A")),
    ("Furniture and Other Manufacturing",
        ("M3370", "M3B00")),
    ("Wholesale and Retail Trade",
        ("M4100", "M4A00")),
    ("Transportation Services", ("M4B00",)),
    ("Information and Communication",
        ("M5170", "M51D0", "M51E0", "M5E00")),
    ("Finance and Insurance",
        ("M52C0", "M5F00")),
    ("Real Estate, Rental, Leasing",
        ("M53C0", "M53D0")),
    ("Professional and Administrative Services",
        ("M5417", "M541E", "M5G00")),
    ("Education",
        ("M6100", "G6100")),
    ("Health",
        ("M6200", "G6200")),
    ("Arts, Recreation, Accommodation, Food",
        ("M7100", "M7200")),
    ("Public Administration and Other Services",
        ("M8100", "M9A00", "M9B00", "N0000",
         "G9110", "G9120", "G9130", "G9140")),
)


def _aggregation_map() -> dict[str, str]:
    """{ StatCan product code → target sector name }."""
    out: dict[str, str] = {}
    for target, codes in SECTOR_AGGREGATION:
        for code in codes:
            if code in out:
                raise ValueError(f"product code {code} assigned twice")
            out[code] = target
    return out


# ---------------------------------------------------------------- helpers


def _fetch_table(pid: str) -> Path:
    """Download a StatCan full-table CSV via the WDS API; cache locally."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = CACHE_DIR / f"{pid}.csv"
    if csv_path.exists():
        return csv_path

    zip_path = CACHE_DIR / f"{pid}-eng.zip"
    if not zip_path.exists():
        api_url = f"{WDS_BASE}/getFullTableDownloadCSV/{pid}/en"
        with urllib.request.urlopen(api_url) as r:
            resp = json.load(r)
        if resp.get("status") != "SUCCESS":
            raise RuntimeError(f"StatCan WDS API failed for {pid}: {resp}")
        download_url = resp["object"]
        with urllib.request.urlopen(download_url) as src, open(zip_path, "wb") as dst:
            dst.write(src.read())

    with zipfile.ZipFile(zip_path) as zf:
        target = next(name for name in zf.namelist() if name.endswith(f"{pid}.csv"))
        with zf.open(target) as src, open(csv_path, "wb") as dst:
            dst.write(src.read())
    return csv_path


_PRODUCT_RE = re.compile(r"^(?P<name>.+?) \[(?P<code>[A-Z][0-9A-Z]+)\]$")


def _parse_product(label: str) -> tuple[str | None, str | None]:
    """Split 'Forestry products and services [M11E0]' into name + code.

    Returns (None, None) for aggregate rows without bracketed codes
    ('Total products', 'Total goods', 'Total services').
    """
    m = _PRODUCT_RE.match(label.strip())
    if m is None:
        return (None, None)
    return m.group("name"), m.group("code")


# ---------------------------------------------------------------- bilateral trade


def build_bilateral_trade(year: int = DEFAULT_YEAR) -> pd.DataFrame:
    """Return long-form (sector, destination, source, value) trade flows.

    Source: Statistics Canada Table 12-10-0088-01.
        - province-to-province flows only (drops international, aggregates,
          territorial enclaves);
        - 10 provinces (territories dropped);
        - real product codes (M-/G-/N-coded); drops Fictive (F-codes) and
          Taxes on products (P-code) aggregates;
        - VALUE is in millions of dollars (preserved as-is).
    """
    csv_path = _fetch_table(TABLE_TRADE_SUMMARY)
    df = pd.read_csv(csv_path, dtype={"VALUE": float}, low_memory=False)

    df = df[df["REF_DATE"] == year].copy()
    if df.empty:
        raise ValueError(f"no rows for year {year} in {csv_path}")

    df = df[df["GEO"].isin(PROVINCES)]
    df["destination"] = df["Trade flow detail"].str.removeprefix("To ")
    df = df[df["destination"].isin(PROVINCES)]

    parsed = df["Product"].apply(_parse_product)
    df["product_name"] = parsed.str[0]
    df["product_code"] = parsed.str[1]
    df = df[df["product_code"].notna()]

    # Map StatCan product codes to the target sector aggregation. Drop product
    # codes not covered by SECTOR_AGGREGATION (Fictive F-codes and tax P-codes).
    agg_map = _aggregation_map()
    df["sector"] = df["product_code"].map(agg_map)
    unknown = sorted(df.loc[df["sector"].isna(), "product_code"].unique())
    if unknown and not all(c.startswith(("F", "P")) for c in unknown):
        raise ValueError(
            f"product codes not in SECTOR_AGGREGATION: {unknown}"
        )
    df = df[df["sector"].notna()]

    df = df.rename(columns={"VALUE": "value", "GEO": "source"}).loc[
        :, ["sector", "destination", "source", "value"]
    ]
    # Aggregate to target sectors: sum within each (sector, destination, source).
    df = df.groupby(["sector", "destination", "source"], as_index=False)["value"].sum()

    # StatCan omits rows where no trade happened and suppresses confidential
    # cells as NaN. Re-index to the full (sector, destination, source) grid so
    # the parquet is dense, and substitute 0 for both missing-row and NaN
    # cases. _validate will then flag any zero-output (sector, region) cells.
    sectors = tuple(s for s, _ in SECTOR_AGGREGATION)
    idx = pd.MultiIndex.from_product(
        [sectors, PROVINCES, PROVINCES],
        names=["sector", "destination", "source"],
    )
    return (
        df.set_index(["sector", "destination", "source"])
        .reindex(idx)["value"]
        .fillna(0.0)
        .reset_index()
        .sort_values(["sector", "destination", "source"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------- diagnostics


def summarize(df: pd.DataFrame) -> None:
    sectors = df["sector"].drop_duplicates().tolist()
    print(f"sectors:  {len(sectors)}")
    print(f"regions:  {df['source'].nunique()} sources × "
          f"{df['destination'].nunique()} destinations")
    print(f"rows:     {len(df)}   (expected: {len(sectors)} × "
          f"{df['source'].nunique()}² = {len(sectors) * df['source'].nunique()**2})")

    # Gross output per (sector, source) = exports + intra-province trade
    pivot = df.pivot_table(
        index="sector", columns="source", values="value", aggfunc="sum",
    )
    gross_out = pivot  # source side already
    zero_cells = (gross_out == 0).sum().sum()
    near_zero = ((gross_out > 0) & (gross_out < 1)).sum().sum()
    print(f"gross-output zero cells:     {zero_cells} / {gross_out.size}")
    print(f"gross-output near-zero (<1): {near_zero}")
    if zero_cells:
        zeros = gross_out.stack()[gross_out.stack() == 0]
        print("  first 8 zero-output cells (sector, province):")
        for (sec, src), _ in list(zeros.items())[:8]:
            print(f"    {sec!r}  ×  {src!r}")


# ---------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR)
    args = parser.parse_args()

    out_dir = INPUTS_ROOT / f"canada_{args.year}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building bilateral_trade.parquet for {args.year}...")
    trade = build_bilateral_trade(args.year)
    summarize(trade)
    print()

    path = out_dir / "bilateral_trade.parquet"
    trade.to_parquet(path, index=False)
    print(f"  wrote {path}  ({len(trade):>6d} rows, "
          f"{path.stat().st_size/1024:6.1f} KiB)")


if __name__ == "__main__":
    main()
