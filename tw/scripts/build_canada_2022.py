"""Build a fresh ``canada_2022/`` calibration for the Tombe-Winter model.

This is Phase 4a of the refresh: pulls the most recent StatCan + OECD ICIO
data and rebuilds every parquet under ``data/inputs/canada_2022/`` at the
same schema as ``canada_2010/``. The model and counterfactual scripts run
unchanged against the new calibration.

What's refreshed (the fast-moving primitives):
- Provincial bilateral trade flows (StatCan 12-10-0088-01, year 2022).
- Provincial employment shares (StatCan 14-10-0327-01, active LFS, 2022).
- Provincial GDP (StatCan 36-10-0222, 2022) and spatial price level
  (StatCan 18-10-0003, last available year 2019 — retired table).
- Aggregate fiscal transfers (StatCan 36-10-0450, multi-year average).
- Synthetic WLD region from ICIO 2022 non-Canada aggregate (for the
  WLD column of bilateral_trade only).

Industrial structure (α^j_n, γ^{k,j}_n, φ^j_n, IO matrix) is built by a
companion script: ``scripts/build_industrial_structure.py``, which reads
2021 L97 provincial Supply-Use tables (StatCan 15-211-X — the latest
published vintage; provincial IOTs lag 2-3 years).

What's NOT refreshed (carried over from canada_2010/):
- Trade-cost shocks ``tauhat_all`` / ``tauhat_asym`` / ``tauhat_dist`` —
  AT-2016 outputs from 2010 trade data; re-estimating them is Phase 4b.
- Trade elasticities ``θ^j`` — AT-2016 / CP-2015 vector, static.
- Sector labels.

Prep (one-time):
    mkdir -p /tmp/icio && unzip -o ~/Downloads/2016-2022_SML.zip 2022_SML.csv -d /tmp/icio/

Run from tw/:
    uv run python scripts/build_canada_2022.py
"""

from __future__ import annotations

import json
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REFRESH_YEAR = 2022

# ─── Paths ────────────────────────────────────────────────────────────────
DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "inputs"
SRC_DIR = DATA_ROOT / "canada_2010"
OUT_DIR = DATA_ROOT / "canada_2022"
CACHE_DIR = Path("/tmp/statcan_tw_refresh")
ICIO_CSV = Path("/tmp/icio/2022_SML.csv")
STATCAN_TRADE_CSV = Path("/tmp/sc12100088/12100088.csv")  # already cached from AKT work

WDS_BASE = "https://www150.statcan.gc.ca/t1/wds/rest"

# ─── Sector taxonomy (TW's 22 ISIC Rev 3 sectors, in canonical order) ─────
SECTORS = (
    "Agriculture, Mining", "Food, Textiles", "Wood", "Paper", "Chemicals, Rubber",
    "Metals", "Equipment, Vehicles", "Manufacturing, n.e.c.",
    "Utilities", "Construction", "Wholesale and Retail", "Hotels and Restaurants",
    "Transport", "Communication", "Finance", "Real Estate", "Software",
    "Other Business Services", "Public Admin.", "Education",
    "Health and Social", "Other Services",
)

# Province order matching maindata.mat (used by initialize_model.m's display)
PROVINCES_FULL = (
    "Alberta", "British Columbia", "Manitoba", "New Brunswick",
    "Newfoundland and Labrador", "Nova Scotia", "Ontario",
    "Prince Edward Island", "Quebec", "Saskatchewan",
)
PROVINCES_SHORT = ("AB", "BC", "MB", "NB", "NL", "NS", "ON", "PE", "QC", "SK")
PROV_FULL_TO_SHORT = dict(zip(PROVINCES_FULL, PROVINCES_SHORT))
WLD = "WLD"
REGIONS = PROVINCES_SHORT + (WLD,)

# ─── Concordance: StatCan 12-10-0088 M-codes → TW 22 sectors ──────────────
MCODE_TO_TW: dict[str, str] = {
    # Agriculture, Mining (ISIC 01-14)
    "M111B": "Agriculture, Mining", "M112A": "Agriculture, Mining",
    "M11D0": "Agriculture, Mining", "M11E0": "Agriculture, Mining",
    "M1140": "Agriculture, Mining", "M1150": "Agriculture, Mining",
    "M21A0": "Agriculture, Mining", "M21B0": "Agriculture, Mining",
    "M2122": "Agriculture, Mining", "M2123": "Agriculture, Mining",
    "M2130": "Agriculture, Mining",
    # Food, Textiles (ISIC 15-19)
    "M31C0": "Food, Textiles", "M312A": "Food, Textiles", "M31D0": "Food, Textiles",
    # Wood (ISIC 20)
    "M3210": "Wood",
    # Paper (ISIC 21-22)
    "M3220": "Paper", "M3230": "Paper",
    # Chemicals, Rubber (ISIC 23-25)
    "M3240": "Chemicals, Rubber", "M3250": "Chemicals, Rubber",
    "M3260": "Chemicals, Rubber",
    "M3270": "Chemicals, Rubber",  # AKT-style: non-metallic minerals in Chem/Rubber
    # Metals (ISIC 26-28) — note: AT-2016 puts NAICS 327 in Chem/Rubber, but
    # the AKT replication groups it with Metals via NAICS 327. We follow the
    # MCODE_TO_TW mapping consistent with AT-2016 by leaving M3270 above in
    # Chemicals/Rubber. For Metals we keep 331 + 332 only.
    "M3310": "Metals", "M3320": "Metals",
    # Equipment, Vehicles (ISIC 29-35)
    "M3330": "Equipment, Vehicles", "M334C": "Equipment, Vehicles",
    "M3350": "Equipment, Vehicles", "M3363": "Equipment, Vehicles",
    "M336A": "Equipment, Vehicles",
    # Manufacturing, n.e.c. (ISIC 36-37)
    "M3370": "Manufacturing, n.e.c.", "M3B00": "Manufacturing, n.e.c.",
    # Services
    "M2200": "Utilities",
    "M23A0": "Construction", "M23B0": "Construction",
    "M23C0": "Construction", "M23D0": "Construction",
    "M4100": "Wholesale and Retail", "M4A00": "Wholesale and Retail",
    "M7200": "Hotels and Restaurants",
    "M4B00": "Transport",
    "M5170": "Communication", "M51D0": "Communication", "M51E0": "Communication",
    "M52C0": "Finance", "M5F00": "Finance",
    "M53C0": "Real Estate", "M53D0": "Real Estate",
    "M5E00": "Software",
    "M5417": "Other Business Services", "M541E": "Other Business Services",
    "M5G00": "Other Business Services",
    "M9B00": "Public Admin.",
    "M6100": "Education",
    "M6200": "Health and Social",
    "M7100": "Other Services", "M8100": "Other Services", "M9A00": "Other Services",
}

# ─── Concordance: ICIO ISIC Rev 4 industries → TW 22 sectors ──────────────
ICIO_TO_TW: dict[str, str] = {
    "A01": "Agriculture, Mining", "A02": "Agriculture, Mining", "A03": "Agriculture, Mining",
    "B05": "Agriculture, Mining", "B06": "Agriculture, Mining",
    "B07": "Agriculture, Mining", "B08": "Agriculture, Mining", "B09": "Agriculture, Mining",
    "C10T12": "Food, Textiles", "C13T15": "Food, Textiles",
    "C16": "Wood",
    "C17_18": "Paper", "J58T60": "Paper",  # publishing belongs in Rev 3 22
    "C19": "Chemicals, Rubber", "C20": "Chemicals, Rubber",
    "C21": "Chemicals, Rubber", "C22": "Chemicals, Rubber",
    "C23": "Chemicals, Rubber",
    "C24A": "Metals", "C24B": "Metals", "C25": "Metals",
    "C26": "Equipment, Vehicles", "C27": "Equipment, Vehicles",
    "C28": "Equipment, Vehicles", "C29": "Equipment, Vehicles",
    "C301": "Equipment, Vehicles", "C302T309": "Equipment, Vehicles",
    "C31T33": "Manufacturing, n.e.c.",
    "D": "Utilities", "E": "Utilities",
    "F": "Construction",
    "G": "Wholesale and Retail",
    "H49": "Transport", "H50": "Transport", "H51": "Transport", "H52": "Transport",
    "H53": "Communication", "J61": "Communication",
    "I": "Hotels and Restaurants",
    "J62_63": "Software",
    "K": "Finance",
    "L": "Real Estate",
    "M": "Other Business Services", "N": "Other Business Services",
    "O": "Public Admin.",
    "P": "Education",
    "Q": "Health and Social",
    "R": "Other Services", "S": "Other Services", "T": "Other Services",
}

FINAL_CONSUMPTION = ("HFCE", "NPISH", "GGFC")  # ICIO final-consumption categories
CAN = "CAN"


# ─── WDS helpers ──────────────────────────────────────────────────────────
def _fetch_statcan(pid: str) -> Path:
    """Download a StatCan full-table CSV via the WDS API; cache locally."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = CACHE_DIR / f"{pid}.csv"
    if csv_path.exists():
        return csv_path

    zip_path = CACHE_DIR / f"{pid}-eng.zip"
    if not zip_path.exists():
        api_url = f"{WDS_BASE}/getFullTableDownloadCSV/{pid}/en"
        print(f"    GET {api_url}")
        with urllib.request.urlopen(api_url) as r:
            resp = json.load(r)
        if resp.get("status") != "SUCCESS":
            raise RuntimeError(f"WDS API failed for {pid}: {resp}")
        with urllib.request.urlopen(resp["object"]) as src, open(zip_path, "wb") as dst:
            dst.write(src.read())
    with zipfile.ZipFile(zip_path) as zf:
        target = next(n for n in zf.namelist() if n.endswith(f"{pid}.csv"))
        with zf.open(target) as src, open(csv_path, "wb") as dst:
            dst.write(src.read())
    return csv_path


# ─── Step A: Provincial bilateral trade (PI) from StatCan 12-10-0088 ──────
_PRODUCT_RE = re.compile(r"\[(M[0-9A-Z]+)\]")


def build_bilateral_trade(year: int = REFRESH_YEAR) -> pd.DataFrame:
    """Return long-form (sector, destination, source, value) for all 11 regions.

    Province-to-province + province-to-WLD + WLD-to-province come from
    StatCan 12-10-0088. WLD-to-WLD is filled later by ICIO.
    """
    df = pd.read_csv(STATCAN_TRADE_CSV, low_memory=False)
    df = df[df["REF_DATE"] == year].copy()
    df = df[df["GEO"].isin(PROVINCES_FULL)]

    code = df["Product"].astype(str).str.extract(_PRODUCT_RE)
    df["sector"] = code[0].map(MCODE_TO_TW)
    df = df[df["sector"].notna()]

    rows = []
    # 1. Province-to-province (interprovincial flows)
    inter = df[df["Trade flow detail"].astype(str).str.startswith("To ")].copy()
    inter["destination_full"] = inter["Trade flow detail"].str.removeprefix("To ")
    inter = inter[inter["destination_full"].isin(PROVINCES_FULL)]
    inter["destination"] = inter["destination_full"].map(PROV_FULL_TO_SHORT)
    inter["source"] = inter["GEO"].map(PROV_FULL_TO_SHORT)
    rows.append(inter.groupby(["sector", "destination", "source"], as_index=False)
                       ["VALUE"].sum().rename(columns={"VALUE": "value"}))

    # 2. Province → WLD (International exports)
    exp = df[df["Trade flow detail"] == "International exports"].copy()
    exp["source"] = exp["GEO"].map(PROV_FULL_TO_SHORT)
    exp["destination"] = WLD
    rows.append(exp.groupby(["sector", "destination", "source"], as_index=False)
                     ["VALUE"].sum().rename(columns={"VALUE": "value"}))

    # 3. WLD → Province (International imports)
    imp = df[df["Trade flow detail"] == "International imports"].copy()
    imp["destination"] = imp["GEO"].map(PROV_FULL_TO_SHORT)
    imp["source"] = WLD
    rows.append(imp.groupby(["sector", "destination", "source"], as_index=False)
                     ["VALUE"].sum().rename(columns={"VALUE": "value"}))

    flows = pd.concat(rows, ignore_index=True)

    # Sanity check: every (sector, dest, source) where dest, source ∈ REGIONS should
    # appear. Fill missing cells with 0.
    idx = pd.MultiIndex.from_product([SECTORS, REGIONS, REGIONS],
                                       names=["sector", "destination", "source"])
    flows = (flows.set_index(["sector", "destination", "source"])
                   .groupby(level=[0, 1, 2]).sum()
                   .reindex(idx, fill_value=0.0)
                   .reset_index())
    return flows


# ─── Step B: Provincial employment shares (lambda) from 14-10-0327 ────────
def build_employment(year: int = REFRESH_YEAR) -> pd.DataFrame:
    """Provincial employment shares from LFS (14-10-0327, active table)."""
    csv = _fetch_statcan("14100327")
    df = pd.read_csv(csv, low_memory=False, dtype={"VALUE": float})
    df = df.rename(columns={"REF_DATE": "year", "VALUE": "value"})
    df["year"] = pd.to_numeric(df["year"].astype(str).str[:4], errors="coerce")
    df = df[df["year"] == year]
    df = df[df["GEO"].isin(PROVINCES_FULL)]
    df = df[df["Labour force characteristics"] == "Employment"]
    df = df[df["Gender"] == "Total - Gender"]
    df = df[df["Age group"] == "15 years and over"]
    df = df[df["UOM"] == "Persons in thousands"]    # exclude rate / percent rows

    df["region"] = df["GEO"].map(PROV_FULL_TO_SHORT)
    df = df.dropna(subset=["region", "value"]).drop_duplicates("region")
    total = df["value"].sum()
    df["share"] = df["value"] / total
    out = df.set_index("region").loc[list(PROVINCES_SHORT)]
    return pd.DataFrame({"region": PROVINCES_SHORT,
                          "value": out["share"].to_numpy()})


# ─── Step C: Real income index (mu) from provincial GDP + spatial prices ──
def build_income_index(year: int = REFRESH_YEAR) -> pd.DataFrame:
    """μ_n = (GDP/capita_n / national average GDP/capita) / (spatial_price_n /
    national average price). Uses 18-10-0003 for spatial prices (last
    available year is 2019); GDP per capita from 36-10-0222 + 17-10-0005.
    Normalized to have weighted mean (by population) ≈ 1, matching the
    convention in maindata.mat::mu."""
    gdp_csv = _fetch_statcan("36100222")
    pop_csv = _fetch_statcan("17100005")
    sp_csv  = _fetch_statcan("18100003")

    gdp = pd.read_csv(gdp_csv, low_memory=False, dtype={"VALUE": float})
    gdp["year"] = pd.to_numeric(gdp["REF_DATE"].astype(str).str[:4], errors="coerce")
    gdp = gdp[(gdp["year"] == year) & (gdp["Estimates"] ==
              "Gross domestic product at market prices")]
    gdp = gdp[gdp["Prices"] == "Current prices"]
    gdp = gdp[gdp["GEO"].isin(PROVINCES_FULL)]
    gdp["region"] = gdp["GEO"].map(PROV_FULL_TO_SHORT)
    gdp = gdp.dropna(subset=["region", "VALUE"]).drop_duplicates("region")
    gdp = gdp.set_index("region")["VALUE"]

    pop = pd.read_csv(pop_csv, low_memory=False, dtype={"VALUE": float})
    pop["year"] = pd.to_numeric(pop["REF_DATE"].astype(str).str[:4], errors="coerce")
    pop = pop[(pop["year"] == year) & (pop["Gender"].isin(["Total - Gender", "Total - gender", "Both sexes"]))
              & (pop["Age group"] == "All ages")]
    pop = pop[pop["GEO"].isin(PROVINCES_FULL)]
    pop["region"] = pop["GEO"].map(PROV_FULL_TO_SHORT)
    pop = pop.dropna(subset=["region", "VALUE"]).drop_duplicates("region")
    pop = pop.set_index("region")["VALUE"]

    # Spatial prices: last available year (2019), all-items index for each
    # provincial capital (St. John's, Charlottetown, Halifax, …). Toronto
    # and Ottawa average for Ontario, as in Albrecht-Tombe (2016).
    sp = pd.read_csv(sp_csv, low_memory=False, dtype={"VALUE": float})
    sp["year"] = pd.to_numeric(sp["REF_DATE"].astype(str).str[:4], errors="coerce")
    sp = sp[sp["year"] == sp["year"].max()]              # 2019
    sp = sp[sp["Products and product groups"] == "All-items"]
    # 18-10-0003 reports per-city, not per-province. Map cities → province
    # short codes; for provinces with multiple cities (ON, AB) take the
    # simple average across cities.
    city_to_short = {
        "St. John's, Newfoundland and Labrador":              "NL",
        "Charlottetown and Summerside, Prince Edward Island": "PE",
        "Halifax, Nova Scotia":                               "NS",
        "Saint John, New Brunswick":                          "NB",
        "Montréal, Quebec":                                   "QC",
        "Ottawa-Gatineau, Ontario part, Ontario/Quebec":      "ON",
        "Toronto, Ontario":                                   "ON",
        "Winnipeg, Manitoba":                                 "MB",
        "Regina, Saskatchewan":                               "SK",
        "Calgary, Alberta":                                   "AB",
        "Edmonton, Alberta":                                  "AB",
        "Vancouver, British Columbia":                        "BC",
    }
    sp["short"] = sp["GEO"].map(city_to_short)
    sp = sp.dropna(subset=["short", "VALUE"])
    spatial = sp.groupby("short")["VALUE"].mean()
    spatial = spatial.reindex(list(PROVINCES_SHORT))

    gdp_pc = gdp.reindex(list(PROVINCES_SHORT)) / pop.reindex(list(PROVINCES_SHORT))
    nat_gdp_pc = gdp.sum() / pop.sum()
    real_gdp_pc = (gdp_pc / nat_gdp_pc) / (spatial / np.average(
        spatial.to_numpy(), weights=pop.reindex(list(PROVINCES_SHORT)).to_numpy()))
    # Normalize so population-weighted mean = 1
    mean_mu = np.average(real_gdp_pc.to_numpy(),
                         weights=pop.reindex(list(PROVINCES_SHORT)).to_numpy())
    mu = real_gdp_pc / mean_mu
    return pd.DataFrame({"region": PROVINCES_SHORT, "value": mu.to_numpy()})


# ─── Step D: Aggregate fiscal transfers from 36-10-0450 ───────────────────
def build_transfers_share_gdp(year_lo: int = 2014, year_hi: int = REFRESH_YEAR
                                ) -> pd.DataFrame:
    """Average per-capita net fiscal transfer relative to the Canadian mean,
    expressed as a share of provincial GDP. Aggregates federal + CPP rev &
    expenditure across years 2014-2022 (or whatever range).

    Mirrors Table1.R's calculation; produces the per-province values that
    initialize_model.m's ``transfers_share_GDP`` variable consumes."""
    cd_csv = _fetch_statcan("36100450")
    pop_csv = _fetch_statcan("17100005")
    gdp_csv = _fetch_statcan("36100222")

    cd = pd.read_csv(cd_csv, low_memory=False, dtype={"VALUE": float})
    cd["year"] = pd.to_numeric(cd["REF_DATE"].astype(str).str[:4], errors="coerce")
    cd = cd[(cd["year"] >= year_lo) & (cd["year"] <= year_hi)]
    cd = cd[cd["GEO"].isin(PROVINCES_FULL)]
    cd = cd[cd["Levels of government"].isin(
        ["Federal general government", "Canada Pension Plan"])]
    cd = cd[cd["Estimates"].isin(
        ["General governments revenue", "General governments expenditure"])]
    cd["category"] = np.where(cd["Estimates"].str.contains("revenue", case=False),
                                "revenue", "expenditure")
    rev_exp = (cd.groupby(["year", "GEO", "category"], as_index=False)["VALUE"].sum()
                  .pivot(index=["year", "GEO"], columns="category", values="VALUE")
                  .reset_index())

    pop = pd.read_csv(pop_csv, low_memory=False, dtype={"VALUE": float})
    pop["year"] = pd.to_numeric(pop["REF_DATE"].astype(str).str[:4], errors="coerce")
    pop = pop[(pop["year"] >= year_lo) & (pop["year"] <= year_hi)
              & pop["Gender"].isin(["Total - Gender", "Total - gender", "Both sexes"])
              & (pop["Age group"] == "All ages")
              & pop["GEO"].isin(PROVINCES_FULL)]
    pop = pop.rename(columns={"VALUE": "pop"})[["year", "GEO", "pop"]]

    gdp = pd.read_csv(gdp_csv, low_memory=False, dtype={"VALUE": float})
    gdp["year"] = pd.to_numeric(gdp["REF_DATE"].astype(str).str[:4], errors="coerce")
    gdp = gdp[(gdp["year"] >= year_lo) & (gdp["year"] <= year_hi)
              & (gdp["Estimates"] == "Gross domestic product at market prices")
              & (gdp["Prices"] == "Current prices")
              & gdp["GEO"].isin(PROVINCES_FULL)]
    gdp = gdp.rename(columns={"VALUE": "GDP"})[["year", "GEO", "GDP"]]

    df = (rev_exp.merge(pop, on=["year", "GEO"])
                  .merge(gdp, on=["year", "GEO"]))
    df["region"] = df["GEO"].map(PROV_FULL_TO_SHORT)

    # Per-capita gaps relative to Canadian average per-capita, summed across years
    agg = df.groupby("region", as_index=False)[["expenditure", "revenue", "pop", "GDP"]].sum()
    exppc = agg["expenditure"].sum() / agg["pop"].sum()
    revpc = agg["revenue"].sum() / agg["pop"].sum()
    agg["transfer_per_capita"] = (agg["expenditure"] / agg["pop"] - exppc) - \
                                  (agg["revenue"] / agg["pop"] - revpc)
    agg["transfer_total"] = agg["transfer_per_capita"] * agg["pop"]
    agg["transfers_share_GDP"] = agg["transfer_total"] / agg["GDP"]

    out = agg.set_index("region").loc[list(PROVINCES_SHORT)]
    return pd.DataFrame({"region": PROVINCES_SHORT,
                          "value": out["transfers_share_GDP"].to_numpy()})


# ─── Step E: ICIO-based primitives (IOmatrix, gamma, phi, alpha, WLD) ────
def _icio_origin_country(label: str) -> tuple[str, str]:
    if "_" not in label:
        return ("", label)
    c, i = label.split("_", 1)
    return c, i


def build_icio_primitives(year_csv: Path = ICIO_CSV
                            ) -> dict[str, pd.DataFrame | float]:
    """Pull canada-aggregate IO + share parameters + WLD trade from ICIO 2022.

    Returns a dict with:
      - io_matrix    : (J, J) DataFrame, sector-to-sector input shares
      - phi          : (J,) labor share by sector (Canadian average)
      - alpha        : (J,) final-demand shares by sector (Canada-aggregate)
      - intermediate : (J*J,) gamma values, indexed (input_sector, dest_sector)
      - wld_intra    : (J,) WLD-WLD intra trade in millions USD
      - usd_to_cad   : float — scale to convert ICIO USD to local CAD millions
    """
    print(f"    loading {year_csv}")
    df = pd.read_csv(year_csv)
    row_labels = df["V1"].tolist()
    col_labels = list(df.columns[1:])
    M = df.drop(columns=["V1"]).to_numpy(dtype=np.float64)

    # Sector cluster indices
    sector_to_isic: dict[str, list[str]] = {s: [] for s in SECTORS}
    for isic, sec in ICIO_TO_TW.items():
        sector_to_isic[sec].append(isic)

    def is_can(c: str) -> bool: return c == CAN
    def not_can(c: str) -> bool: return c != CAN
    def any_c(c: str) -> bool: return True

    def rows_for(country_pred, isics: list[str]) -> np.ndarray:
        return np.array([
            i for i, lbl in enumerate(row_labels)
            if (split := _icio_origin_country(lbl))[0]
            and country_pred(split[0]) and split[1] in isics
        ], dtype=np.int64)

    def cols_for(country_pred, suf_pred) -> np.ndarray:
        return np.array([
            i for i, lbl in enumerate(col_labels)
            if (split := _icio_origin_country(lbl))[0]
            and country_pred(split[0]) and suf_pred(split[1])
        ], dtype=np.int64)

    out_col = col_labels.index("OUT")
    va_row = row_labels.index("VA")

    # Per-sector Canadian gross output (for share denominators)
    gross_output: dict[str, float] = {}
    for sec, isics in sector_to_isic.items():
        idx = rows_for(is_can, isics)
        gross_output[sec] = float(M[idx, out_col].sum())

    # ── IO matrix: sector k → sector j Canadian intermediate use share ──
    io_data = np.zeros((len(SECTORS), len(SECTORS)))
    sector_idx = {s: i for i, s in enumerate(SECTORS)}
    for src_sec, src_isics in sector_to_isic.items():
        src_rows = rows_for(any_c, src_isics)
        for dst_sec, dst_isics in sector_to_isic.items():
            dst_cols = cols_for(is_can,
                                 lambda s, dst=set(dst_isics): s in dst)
            io_data[sector_idx[src_sec], sector_idx[dst_sec]] = float(
                M[np.ix_(src_rows, dst_cols)].sum())
    # Normalize columns: each destination sector's input share by source sector
    col_sums = io_data.sum(axis=0)
    col_sums[col_sums == 0] = 1.0  # avoid /0; empty columns get 0
    io_norm = io_data / col_sums
    io_matrix = pd.DataFrame(io_norm, index=SECTORS, columns=SECTORS)

    # ── phi: labor share per sector (Canadian average) ─────────────────
    phi = np.zeros(len(SECTORS))
    for sec, isics in sector_to_isic.items():
        idx = rows_for(is_can, isics)
        va_block = float(M[va_row, cols_for(is_can, lambda s, idi=set(isics): s in idi)].sum())
        gross = gross_output[sec]
        # Labor share is VA / gross output (TW uses φ^j as labor share)
        phi[sector_idx[sec]] = va_block / gross if gross > 0 else 0.5
    # Clamp to [0, 1]
    phi = np.clip(phi, 0.05, 0.99)

    # ── alpha: final demand share per sector (Canada-aggregate, household etc.) ──
    consumption_cols = cols_for(is_can, lambda s: s in FINAL_CONSUMPTION)
    alpha = np.zeros(len(SECTORS))
    for sec, isics in sector_to_isic.items():
        idx = rows_for(any_c, isics)
        alpha[sector_idx[sec]] = float(M[np.ix_(idx, consumption_cols)].sum())
    total = alpha.sum()
    if total > 0:
        alpha = alpha / total

    # ── Intermediate input share γ^{k,j}: same as IO matrix here ────────
    # In TW's setup γ is region-specific. For Canadian provinces we use the
    # Canada-aggregate γ. The WLD column gets a separate γ from non-CAN data.
    intermediate = io_norm  # (J input, J dest) — same shape, Canadian average

    # ── WLD intra trade: sum over non-CAN origins × non-CAN destinations ──
    wld_intra = np.zeros(len(SECTORS))
    for sec, isics in sector_to_isic.items():
        idx = rows_for(not_can, isics)
        non_can_dest = cols_for(not_can, lambda s: s != "OUT")
        wld_intra[sector_idx[sec]] = float(M[np.ix_(idx, non_can_dest)].sum())

    # ── USD → CAD scaling factor for ICIO output ────────────────────────
    # Compare Canadian gross output in ICIO (USD millions) to StatCan trade
    # data totals (CAD millions). Aggregate to a single scale via the ratio.
    can_gross_usd = sum(gross_output.values())
    # We'll set this empirically by comparing to bilateral_trade in the caller.
    usd_to_cad = 1.0  # filled in by caller after StatCan totals are known

    return {
        "io_matrix": io_matrix,
        "phi": phi,
        "alpha": alpha,
        "intermediate": intermediate,
        "wld_intra": wld_intra,
        "can_gross_usd": can_gross_usd,
    }


# ─── Step F: stitch everything together and write parquets ────────────────
def main() -> None:
    if not STATCAN_TRADE_CSV.exists():
        raise FileNotFoundError(f"missing {STATCAN_TRADE_CSV}; "
                                  f"fetch StatCan 12-10-0088 first")
    if not ICIO_CSV.exists():
        raise FileNotFoundError(f"missing {ICIO_CSV}; extract via "
                                  f"unzip -o ~/Downloads/2016-2022_SML.zip 2022_SML.csv -d /tmp/icio/")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Building canada_{REFRESH_YEAR} calibration in {OUT_DIR} …\n")

    # ── A. Provincial bilateral trade ──────────────────────────────────────
    print(f"  [A] Provincial bilateral trade ({REFRESH_YEAR})")
    bilateral = build_bilateral_trade(REFRESH_YEAR)

    # ── B. Employment shares ───────────────────────────────────────────────
    print(f"  [B] Employment shares")
    employment = build_employment(REFRESH_YEAR)

    # ── C. Real income index ───────────────────────────────────────────────
    print(f"  [C] Real income index (μ)")
    income_index = build_income_index(REFRESH_YEAR)

    # ── D. Aggregate fiscal transfers ─────────────────────────────────────
    print(f"  [D] Aggregate fiscal transfers (2014-{REFRESH_YEAR})")
    transfers = build_transfers_share_gdp()

    # ── E. ICIO WLD-WLD intra-trade only (industrial structure carried
    # over from canada_2010 below). ──────────────────────────────────────
    print(f"  [E] ICIO 2022 WLD-WLD intra-trade (for WLD column)")
    icio = build_icio_primitives()

    # Need to add WLD-WLD intra-trade row to bilateral; convert from USD M to
    # the same units as StatCan flows. Compute scale from total Canadian gross
    # output: bilateral_total_CAD = sum of all flows in StatCan; ICIO_total_USD
    # = Canada gross output. Scale = bilateral_total / can_gross_usd.
    can_provincial_output = (bilateral[bilateral["source"].isin(PROVINCES_SHORT)
                                        & bilateral["destination"].isin(PROVINCES_SHORT)]
                              ["value"].sum())
    # Add a per-sector approximation: use ratio of total provincial output to
    # ICIO Canada gross output (this is rough; OK for first-pass refresh)
    scale = can_provincial_output / icio["can_gross_usd"] if icio["can_gross_usd"] > 0 else 1.25

    # WLD-WLD intra-trade: ICIO 2022 (USD millions) → CAD millions via the
    # USD/CAD = 1.30 (2022 average). Then write into bilateral.
    USD_TO_CAD_2022 = 1.30
    wld_rows = pd.DataFrame({
        "sector": list(SECTORS),
        "destination": [WLD] * len(SECTORS),
        "source": [WLD] * len(SECTORS),
        "value": (icio["wld_intra"] * USD_TO_CAD_2022).tolist(),
    })
    bilateral_dollars = pd.concat(
        [bilateral[~((bilateral["destination"] == WLD) & (bilateral["source"] == WLD))],
         wld_rows], ignore_index=True)

    # Compute baseline expenditure shares I_n and trade surplus S_n from the
    # dollar flows BEFORE normalizing bilateral to shares.
    dest_total = bilateral_dollars.groupby("destination")["value"].sum().reindex(REGIONS)
    src_total = bilateral_dollars.groupby("source")["value"].sum().reindex(REGIONS, fill_value=0)
    total = dest_total.sum()
    I = dest_total / total
    Sn = (src_total - dest_total) / total

    # Now normalize per (sector, destination) so bilateral values are trade
    # shares π_{n,i}^j summing to 1 across sources i (canada_2010 convention).
    bilateral = bilateral_dollars.copy()
    totals_by_dest_sec = bilateral.groupby(["sector", "destination"])["value"].transform("sum")
    bilateral["value"] = bilateral["value"] / totals_by_dest_sec.where(totals_by_dest_sec > 0, 1.0)

    # ── Write all parquets ─────────────────────────────────────────────────
    bilateral.to_parquet(OUT_DIR / "bilateral_trade.parquet", index=False)
    print(f"\n  bilateral_trade.parquet      {len(bilateral):,} rows  (π shares)")

    employment.to_parquet(OUT_DIR / "employment.parquet", index=False)
    print(f"  employment.parquet           {len(employment)} provinces")

    income_index.to_parquet(OUT_DIR / "income_index.parquet", index=False)
    print(f"  income_index.parquet         {len(income_index)} provinces")

    transfers.to_parquet(OUT_DIR / "transfers_share_gdp.parquet", index=False)
    print(f"  transfers_share_gdp.parquet  {len(transfers)} provinces")

    # Industrial-structure parquets (io_matrix, labor_share, final_demand_share,
    # intermediate_share) are carried over from canada_2010 in the static
    # copy block below. See module docstring for the reasoning.

    # I_n (baseline absorption share) and S_n (trade surplus share) were
    # computed from dollar flows above. Write them now.
    pd.DataFrame({"region": REGIONS, "value": I.to_numpy()}).to_parquet(
        OUT_DIR / "income_baseline.parquet", index=False)
    print(f"  income_baseline.parquet      {len(REGIONS)} regions")

    pd.DataFrame({"region": REGIONS, "value": Sn.to_numpy()}).to_parquet(
        OUT_DIR / "Sn_baseline.parquet", index=False)
    print(f"  Sn_baseline.parquet          {len(REGIONS)} regions")

    # trade_surplus per region as fraction of own expenditure (S_n / I_n,
    # both in normalized units → unitless). For WLD, this is the global
    # imbalance (≈ 0); for provinces, ratio of net exports to absorption.
    surplus_gdp = (Sn / I).fillna(0).replace([np.inf, -np.inf], 0)
    pd.DataFrame({"region": REGIONS, "value": surplus_gdp.to_numpy()}).to_parquet(
        OUT_DIR / "trade_surplus.parquet", index=False)
    print(f"  trade_surplus.parquet        {len(REGIONS)} regions")

    # ── Static carry-over from canada_2010 ────────────────────────────────
    carried = (
        "sectoral_dispersion.parquet", "sector_labels.parquet",
        "tauhat_all.parquet", "tauhat_asym.parquet", "tauhat_dist.parquet",
        # Industrial-structure primitives — see module docstring
        "io_matrix.parquet", "labor_share.parquet",
        "final_demand_share.parquet", "intermediate_share.parquet",
    )
    for fname in carried:
        shutil.copy(SRC_DIR / fname, OUT_DIR / fname)
    print(f"\n  copied {len(carried)} static files from canada_2010/")
    print(f"\nDone.\n")


if __name__ == "__main__":
    main()
