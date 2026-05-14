"""Build a Canadian calibration of the qge model from Statistics Canada data.

So far this covers two of the eight required inputs:

* ``bilateral_trade.parquet`` — Table 12-10-0088-01, Interprovincial and
  international trade flows (summary level).
* ``employment.parquet`` — Table 14-10-0202-01, SEPH (Survey of Employment,
  Payrolls and Hours), All employees.

Known data-quality caveats:

* **SEPH excludes self-employed.** The Agriculture / Forestry / Fishing
  sector in employment.parquet captures only forestry employees (NAICS 11N);
  agriculture proper (NAICS 111-112) and fishing/hunting (114) are absent.
  This produces zero-employment cells in agriculture-heavy provinces
  (Manitoba, Saskatchewan, NL) which are economically wrong — those
  provinces have substantial agricultural labour, just not on a SEPH-eligible
  payroll. A future refinement should graft in LFS Table 14-10-0023-01
  numbers for [111-112] and [114] to close the gap.
* **A few other zero-employment cells** (Computers/Electronics in NB, NL,
  PEI, SK; Furniture/Other in PEI) likely reflect StatCan confidentiality
  suppression of small-sample cells.

Subsequent passes will add the remaining six inputs (IO matrix, value-added
share, structures share, final-demand share, portfolio share, sectoral
dispersion).

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
TABLE_SEPH = "14100202"  # Employment by industry, annual (Survey of Employment, Payrolls and Hours)
TABLE_LFS = "14100023"   # Labour force characteristics by industry, annual (Labour Force Survey)
TABLE_GDP_INCOME = "36100221"  # GDP, income-based, provincial and territorial, annual


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


# NAICS codes (as the SEPH table reports them) mapped to the same 22 sectors.
# The trade aggregation uses StatCan IO commodity codes (M-codes); SEPH uses
# industry NAICS codes. The two are different classifications, but at the
# 22-sector grain they line up.
#
# Caveat for "Agriculture, Forestry, Fishing": SEPH only covers payroll
# employers and therefore reports only forestry + logging support [11N].
# Agriculture proper (NAICS 111-112) and fishing/hunting (114) are missing
# because most farms and fishery operations are self-employed and outside
# SEPH's frame. The resulting employment series for that sector is an
# undercount of true total labour input. A future refinement would graft in
# LFS Table 14-10-0023-01 numbers for [111-112] and [114].
NAICS_TO_SECTOR: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Agriculture, Forestry, Fishing", ("11N",)),
    ("Mining and Extraction", ("21",)),
    ("Utilities", ("22",)),
    ("Construction", ("23",)),
    ("Food, Beverage, Tobacco", ("311", "312")),
    ("Textile, Apparel, Leather", ("313", "314", "315", "316")),
    ("Wood, Paper, Printing", ("321", "322", "323")),
    ("Petroleum and Chemicals", ("324", "325", "326")),
    ("Non-metallic Mineral Products", ("327",)),
    ("Metals and Machinery", ("331", "332", "333")),
    ("Computers, Electronics, Electrical", ("334", "335")),
    ("Transportation Equipment", ("336",)),
    ("Furniture and Other Manufacturing", ("337", "339")),
    ("Wholesale and Retail Trade", ("41", "44-45")),
    ("Transportation Services", ("48-49",)),
    ("Information and Communication", ("51",)),
    ("Finance and Insurance", ("52",)),
    ("Real Estate, Rental, Leasing", ("53",)),
    ("Professional and Administrative Services", ("54", "55", "56")),
    ("Education", ("61",)),
    ("Health", ("62",)),
    ("Arts, Recreation, Accommodation, Food", ("71", "72")),
    ("Public Administration and Other Services", ("81", "91")),
)


def _naics_aggregation_map() -> dict[str, str]:
    out: dict[str, str] = {}
    for target, codes in NAICS_TO_SECTOR:
        for code in codes:
            if code in out:
                raise ValueError(f"NAICS code {code} assigned twice")
            out[code] = target
    return out


def _naics_first(label: str) -> str | None:
    """Extract the primary (coarsest) NAICS code from a SEPH industry label.

    SEPH encodes labels like ``Utilities [22,221]`` or ``Educational services
    [61,611]``; we keep the first comma-separated code, which is always the
    coarsest level (NAICS-2 for services, NAICS-3 for manufacturing).
    """
    m = re.search(r"\[([^\]]+)\]$", label.strip())
    if m is None:
        return None
    return m.group(1).split(",")[0]


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


# ---------------------------------------------------------------- employment

# LFS NAICS labels (Table 14-10-0023-01) that cover Agriculture/Forestry/Fishing.
# These three rows together capture NAICS 11 (excluding agriculture-support
# services 1151-1152 which LFS folds into the [111-112] aggregate).
_LFS_AG_NAICS = (
    "Agriculture [111-112, 1100, 1151-1152]",
    "Forestry and logging and support activities for forestry [113, 1153]",
    "Fishing, hunting and trapping [114]",
)


def _lfs_agriculture(year: int) -> pd.Series:
    """LFS-sourced Agriculture/Forestry/Fishing employment per province.

    Returns a Series indexed by province name (persons, not thousands). LFS
    captures the self-employed that SEPH misses, which matters most for this
    sector — agriculture in particular is dominated by self-employed farmers.
    """
    csv_path = _fetch_table(TABLE_LFS)
    df = pd.read_csv(csv_path, dtype={"VALUE": float}, low_memory=False)
    df = df[df["REF_DATE"] == year]
    df = df[df["GEO"].isin(PROVINCES)]
    df = df[df["Labour force characteristics"] == "Employment"]
    df = df[df["Age group"] == "15 years and over"]
    df = df[df["Gender"] == "Total - Gender"]
    df = df[
        df["North American Industry Classification System (NAICS)"].isin(_LFS_AG_NAICS)
    ]
    # LFS VALUE is in thousands of persons; convert to persons.
    persons = df.groupby("GEO")["VALUE"].sum() * 1000
    return persons.reindex(PROVINCES).fillna(0.0)


def build_employment(year: int = DEFAULT_YEAR) -> pd.DataFrame:
    """Return long-form (sector, region, value) employment by industry × province.

    Sources:
        SEPH Table 14-10-0202-01 (All employees) for most sectors — provides
        NAICS-3 manufacturing sub-sector detail.
        LFS Table 14-10-0023-01 (Employment, 15+, all genders) for
        Agriculture/Forestry/Fishing — SEPH excludes most self-employed and
        therefore covers only forestry (NAICS 11N), missing the bulk of
        actual labour in that sector.

    Mixing two sources is methodologically imperfect — SEPH counts payroll
    employees only while LFS includes the self-employed — but the
    alternative (SEPH-only) reports zero agricultural employment in
    Manitoba/Saskatchewan/NL, which is economically wrong. Provinces with
    very small SEPH cells in other sectors (Computers/Electronics in NB,
    NL, PEI, SK; Furniture in PEI) are left at zero — the model treats
    them as economically zero, which the trade data confirms is roughly
    accurate for those (sector, region) combinations.
    """
    csv_path = _fetch_table(TABLE_SEPH)
    df = pd.read_csv(csv_path, dtype={"VALUE": float}, low_memory=False)

    df = df[df["REF_DATE"] == year]
    if df.empty:
        raise ValueError(f"no rows for year {year} in {csv_path}")
    df = df[df["GEO"].isin(PROVINCES)]
    df = df[df["Type of employee"] == "All employees"]

    naics_col = "North American Industry Classification System (NAICS)"
    df = df.assign(naics=df[naics_col].apply(_naics_first))
    naics_map = _naics_aggregation_map()
    df["sector"] = df["naics"].map(naics_map)
    df = df[df["sector"].notna()]

    df = df.rename(columns={"VALUE": "value", "GEO": "region"})
    df = df.groupby(["sector", "region"], as_index=False)["value"].sum()
    df["value"] = df["value"].fillna(0.0)

    sectors = tuple(s for s, _ in NAICS_TO_SECTOR)
    idx = pd.MultiIndex.from_product(
        [sectors, PROVINCES], names=["sector", "region"]
    )
    out = (
        df.set_index(["sector", "region"])["value"]
        .reindex(idx)
        .fillna(0.0)
        .reset_index()
    )

    # Overwrite the Agriculture/Forestry/Fishing row with LFS data.
    ag_lfs = _lfs_agriculture(year)
    ag_mask = out["sector"] == "Agriculture, Forestry, Fishing"
    out.loc[ag_mask, "value"] = out.loc[ag_mask, "region"].map(ag_lfs).to_numpy()

    return out.sort_values(["sector", "region"]).reset_index(drop=True)


# ---------------------------------------------------------------- sectoral dispersion

# θ_j (Eaton-Kortum trade elasticity per sector). We store 1/θ in the parquet
# to match the qge model's convention. Source: CPRHS 2017 Table — the same
# values used in their US calibration, mapped onto our 23-sector Canadian
# taxonomy. Where our sectors aggregate two CPRHS sectors, we use the simple
# arithmetic mean of (1/θ). For sectors with no CPRHS analog (Agriculture and
# Mining — CPRHS treats them differently or omits them), we use the CPRHS
# default for non-tradables (1/4.55 ≈ 0.22). A future refinement is to plug
# in published sector-specific Canadian θ estimates.
THETA = 4.55  # CPRHS default non-tradable elasticity (services)


def _cprhs_theta_per_sector() -> dict[str, float]:
    """Return 1/θ per Canadian sector, borrowing CPRHS values where possible."""
    avg = lambda *values: sum(1 / v for v in values) / len(values)
    return {
        "Agriculture, Forestry, Fishing":          1 / THETA,  # no CPRHS analog
        "Mining and Extraction":                   1 / THETA,  # no CPRHS analog
        "Utilities":                               1 / THETA,
        "Construction":                            1 / THETA,
        "Food, Beverage, Tobacco":                 1 / 2.55,
        "Textile, Apparel, Leather":               1 / 5.56,
        "Wood, Paper, Printing":                   avg(9.46, 9.07),   # CPRHS 3, 4
        "Petroleum and Chemicals":                 avg(51.08, 4.75, 1.66),  # CPRHS 5, 6, 7
        "Non-metallic Mineral Products":           1 / 2.76,
        "Metals and Machinery":                    avg(6.78, 1.52),   # CPRHS 9, 10
        "Computers, Electronics, Electrical":      avg(12.79, 10.60), # CPRHS 11, 12
        "Transportation Equipment":                1 / 1.01,
        "Furniture and Other Manufacturing":       1 / 5.00,          # CPRHS 14, 15 both 5
        "Wholesale and Retail Trade":              1 / THETA,
        "Transportation Services":                 1 / THETA,
        "Information and Communication":           1 / THETA,
        "Finance and Insurance":                   1 / THETA,
        "Real Estate, Rental, Leasing":            1 / THETA,
        "Professional and Administrative Services": 1 / THETA,
        "Education":                               1 / THETA,
        "Health":                                  1 / THETA,
        "Arts, Recreation, Accommodation, Food":   1 / THETA,
        "Public Administration and Other Services": 1 / THETA,
    }


def build_sectoral_dispersion() -> pd.DataFrame:
    """Long-form (sector, value) carrying 1/θ_j for the 23 Canadian sectors."""
    thetas = _cprhs_theta_per_sector()
    sectors = tuple(s for s, _ in NAICS_TO_SECTOR)
    missing = [s for s in sectors if s not in thetas]
    if missing:
        raise ValueError(f"theta missing for: {missing}")
    return pd.DataFrame(
        {"sector": sectors, "value": [thetas[s] for s in sectors]}
    )


# ---------------------------------------------------------------- structures share


def build_structures_share(year: int = DEFAULT_YEAR) -> pd.DataFrame:
    """Long-form (region, value) capital income share of factor income.

    B_n = Gross Operating Surplus / (Compensation + GOS + Gross Mixed Income)

    Per-province from StatCan Table 36-10-0221-01 (income-based provincial GDP).
    The model assumes B is constant across sectors within a region; this is a
    simplification consistent with CPRHS. Gross mixed income (self-employed
    earnings) is treated as labour for this calculation — a common convention
    that slightly understates the true capital share in provinces with many
    unincorporated businesses.
    """
    csv_path = _fetch_table(TABLE_GDP_INCOME)
    df = pd.read_csv(csv_path, dtype={"VALUE": float}, low_memory=False)
    df = df[df["REF_DATE"] == year]
    df = df[df["GEO"].isin(PROVINCES)]
    keep = ["Compensation of employees", "Gross operating surplus", "Gross mixed income"]
    df = df[df["Estimates"].isin(keep)]
    pivot = df.pivot_table(
        index="GEO", columns="Estimates", values="VALUE", aggfunc="sum",
    )
    pivot["B"] = pivot["Gross operating surplus"] / (
        pivot["Compensation of employees"]
        + pivot["Gross operating surplus"]
        + pivot["Gross mixed income"]
    )
    out = pivot["B"].reindex(PROVINCES).reset_index()
    out.columns = ["region", "value"]
    return out


# ---------------------------------------------------------------- portfolio share

def build_portfolio_share() -> pd.DataFrame:
    """Long-form (region, value) carrying ι_n = 0 — closed-province assumption.

    CPRHS calibrate ι_n as a residual to match observed US trade balances,
    treating it as the fraction of a region's structures rents flowing into a
    global portfolio. For an initial Canadian calibration we set ι ≡ 0 (every
    province retains all its capital income). This is the simplest defensible
    starting point per the DATA.md guidance and can be tuned later once we
    have interprovincial current-account estimates.
    """
    return pd.DataFrame({"region": list(PROVINCES), "value": [0.0] * len(PROVINCES)})


# ---------------------------------------------------------------- diagnostics


def summarize_trade(df: pd.DataFrame) -> None:
    sectors = df["sector"].drop_duplicates().tolist()
    print(f"sectors:  {len(sectors)}")
    print(f"regions:  {df['source'].nunique()} sources × "
          f"{df['destination'].nunique()} destinations")
    print(f"rows:     {len(df)}   (expected: {len(sectors)} × "
          f"{df['source'].nunique()}² = {len(sectors) * df['source'].nunique()**2})")

    pivot = df.pivot_table(
        index="sector", columns="source", values="value", aggfunc="sum",
    )
    zero_cells = (pivot == 0).sum().sum()
    near_zero = ((pivot > 0) & (pivot < 1)).sum().sum()
    print(f"gross-output zero cells:     {zero_cells} / {pivot.size}")
    print(f"gross-output near-zero (<1): {near_zero}")
    if zero_cells:
        zeros = pivot.stack()[pivot.stack() == 0]
        print("  first 8 zero-output cells (sector, province):")
        for (sec, src), _ in list(zeros.items())[:8]:
            print(f"    {sec!r}  ×  {src!r}")


def summarize_employment(df: pd.DataFrame) -> None:
    sectors = df["sector"].drop_duplicates().tolist()
    print(f"sectors:  {len(sectors)}")
    print(f"regions:  {df['region'].nunique()}")
    print(f"rows:     {len(df)}   (expected: {len(sectors) * df['region'].nunique()})")
    pivot = df.pivot_table(index="sector", columns="region", values="value")
    zero_cells = (pivot == 0).sum().sum()
    print(f"zero-employment cells:       {zero_cells} / {pivot.size}")
    if zero_cells:
        zeros = pivot.stack()[pivot.stack() == 0]
        print("  zero-employment cells (sector, region):")
        for (sec, reg), _ in list(zeros.items())[:8]:
            print(f"    {sec!r}  ×  {reg!r}")
    print(f"total employment: {df['value'].sum():>12,.0f} persons")


# ---------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR)
    args = parser.parse_args()

    out_dir = INPUTS_ROOT / f"canada_{args.year}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building bilateral_trade.parquet for {args.year}...")
    trade = build_bilateral_trade(args.year)
    summarize_trade(trade)
    path = out_dir / "bilateral_trade.parquet"
    trade.to_parquet(path, index=False)
    print(f"  wrote {path}  ({len(trade):>6d} rows, "
          f"{path.stat().st_size/1024:6.1f} KiB)")
    print()

    print(f"Building employment.parquet for {args.year}...")
    employment = build_employment(args.year)
    summarize_employment(employment)
    path = out_dir / "employment.parquet"
    employment.to_parquet(path, index=False)
    print(f"  wrote {path}  ({len(employment):>6d} rows, "
          f"{path.stat().st_size/1024:6.1f} KiB)")
    print()

    print(f"Building structures_share.parquet for {args.year}...")
    B = build_structures_share(args.year)
    print("  B per province:")
    for _, row in B.iterrows():
        print(f"    {row['region']:<30} {row['value']:.4f}")
    path = out_dir / "structures_share.parquet"
    B.to_parquet(path, index=False)
    print(f"  wrote {path}")
    print()

    print("Building sectoral_dispersion.parquet (CPRHS θ values)...")
    theta = build_sectoral_dispersion()
    path = out_dir / "sectoral_dispersion.parquet"
    theta.to_parquet(path, index=False)
    print(f"  wrote {path}  ({len(theta):>6d} rows, "
          f"{path.stat().st_size/1024:6.1f} KiB)")
    print()

    print("Building portfolio_share.parquet (ι ≡ 0 — closed province)...")
    iota = build_portfolio_share()
    path = out_dir / "portfolio_share.parquet"
    iota.to_parquet(path, index=False)
    print(f"  wrote {path}  ({len(iota):>6d} rows, "
          f"{path.stat().st_size/1024:6.1f} KiB)")


if __name__ == "__main__":
    main()
