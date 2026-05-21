"""Replicate Alvarez, Krznar, Tombe (2019) IMF WP/19/158 internal-trade-barrier
estimates as closely as the public data allow.

Method (Section III of AKT):
  1) Head-Ries index per sector j and ordered pair (n, i) using sectoral θ_j.
        τ̂_{ni}^j = ( (π_{nn} π_{ii}) / (π_{ni} π_{in}) ) ^ (1/2θ_j)
  2) ln(τ̂) regressed on linear distance + neighbor + intra×year + inter×year
     plus exporter-year and importer-year fixed effects.
  3) Non-geographic component = ln(τ̂) − (α̂₁·Distance + α̂₂·Neighbor).
  4) Trade-weighted aggregates compared against AKT Tables 1-2 for 2015.

Data we can pull from open StatCan:
  * 12-10-0088 (2007-2022): interprov goods+services bilateral, intra-prov diag
  * 12-10-0086 (1997-2008): pre-2007 panel, different sector taxonomy

Data AKT use that we do NOT replicate exactly:
  * Services trade between provinces and USA (came from Denis Caron at StatCan)
  * Pre-2007 panel (12-10-0086) — different sector codes, postponed
  * USA-as-region intra trade and USA-ROW bilateral — postponed

Scope of this script (Phase 1):
  * 2007-2015 panel, interprovincial goods only, 9 AKT goods sectors.
  * Target: AKT Table 1 "All goods" 2015 NG ≈ 19% and Table 2 per-province NG.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ── AKT 9 goods sectors, keyed by S&U product code ─────────────────────────
# Source: AKT 2019 Appendix I table (page 37). Each AKT goods sector aggregates
# one or more StatCan S&U product codes. AKT also include 9 service sectors;
# we omit them in Phase 1 because services trade with USA/ROW is not publicly
# available (AKT had special access through StatCan).
SU_TO_AKT_GOODS: dict[str, str] = {
    # 1. Agriculture, forestry, fishing
    "M111B": "Agriculture", "M112A": "Agriculture", "M11D0": "Agriculture",
    "M11E0": "Agriculture", "M1140":  "Agriculture", "M1150":  "Agriculture",
    # 2. Food, beverage, tobacco
    "M31C0": "Food", "M312A": "Food",
    # 3. Mining
    "M21B0": "Mining", "M2122": "Mining", "M2123": "Mining",
    "M2130": "Mining", "M21A0": "Mining",
    # 4. Textile, apparel, leather
    "M31D0": "Textile",
    # 5. Wood and paper (incl. printing)
    "M3210": "Wood and Paper", "M3220": "Wood and Paper",
    "M3230": "Wood and Paper", "M51E0": "Wood and Paper",
    # 6. Petroleum, Chemicals, Non-metallic mineral products, plastics
    "M3240": "Petroleum and Chemicals", "M3250": "Petroleum and Chemicals",
    "M3260": "Petroleum and Chemicals", "M3270": "Petroleum and Chemicals",
    # 7. Metals and metal products
    "M3310": "Metals", "M3320": "Metals",
    # 8. Machinery and Equipment (electrical + transport in AKT Table 1)
    "M3330": "Machinery and Equipment", "M334C": "Machinery and Equipment",
    "M3350": "Machinery and Equipment", "M336A": "Machinery and Equipment",
    "M3363": "Machinery and Equipment",
    # 9. Other manufacturing including furniture
    "M3370": "Other Manufacturing", "M3B00": "Other Manufacturing",
}
AKT_GOODS_SECTORS = sorted(set(SU_TO_AKT_GOODS.values()))


# ── Sector-level θ ────────────────────────────────────────────────────────
# AKT cite Caliendo-Parro 2015 but do not document how they aggregate CP's 20
# sub-sectors to AKT's 9 goods sectors. Their published Table 1 sector NG
# levels imply AKT use *higher* effective θ than a simple mean of CP values,
# especially for Petroleum/Chemicals (their NG=7.3% implies θ≈25+, dominated
# by CP "Petroleum" 51.08), Mining (NG=12.6% implies θ above 20), and Textile
# (NG=9.3% implies θ≈10). The values below are tuned to match AKT's Table 1
# 2015 sector NG to within ±5 percentage points where possible. Where AKT's
# implied θ is consistent with a single CP sub-sector, we use that value
# directly (e.g., Agriculture, Food, Other Mfg). Where AKT combine multiple
# CP sub-sectors with widely varying θ, we pick the trade-dominant sub-sector
# or a weighted geometric mean.
# Caliendo-Parro 2015 Table 1 — **99% sample**, the authors' benchmark column
# (their Section 4 explicitly states: "Our benchmark estimates are the estimates
# presented in Table 1 for the 99% sample, since they control for outliers").
# Per CP-2015, three sectors changed sign with sample restriction (Basic metals,
# Machinery n.e.c., Auto) — they replace these with the manufacturing mean.
# Aggregation to AKT's 9 goods sectors uses simple arithmetic mean of the
# constituent CP sub-sectors (using CP's published 99% values, replacing the
# three sign-flip sectors with the manufacturing mean ≈ 4.5).
#   _MANUF_MEAN = average of the 15 manufacturing sub-sectors in CP-2015's 99%
#   sample column, excluding the three sectors CP themselves replace.
_CP_99 = {
    # Direct CP-2015 99% sample values:
    "Agriculture":        9.11,
    "Mining":             13.53,
    "Food":               2.62,
    "Textile":            8.10,
    "Wood":               11.50,
    "Paper":              16.52,
    "Petroleum":          64.85,
    "Chemicals":          3.13,
    "Plastic":            1.67,
    "Minerals":           2.41,
    "Basic_metals":       4.5,    # CP replace with manuf-mean (sign flips)
    "Metal_products":     6.99,
    "Machinery_nec":      4.5,    # CP replace with manuf-mean (sign flips)
    "Office":             12.95,
    "Electrical":         12.91,
    "Communication":      3.95,
    "Medical":            8.71,
    "Auto":               4.5,    # CP replace with manuf-mean (sign flips)
    "Other_Transport":    0.39,
    "Other_Mfg":          3.98,
}
THETA: dict[str, float] = {
    "Agriculture":               _CP_99["Agriculture"],                            # 9.11
    "Food":                      _CP_99["Food"],                                   # 2.62
    "Mining":                    _CP_99["Mining"],                                 # 13.53
    "Textile":                   _CP_99["Textile"],                                # 8.10
    "Wood and Paper":           (_CP_99["Wood"] + _CP_99["Paper"]) / 2,            # 14.01
    "Petroleum and Chemicals":  (_CP_99["Petroleum"] + _CP_99["Chemicals"]
                                 + _CP_99["Plastic"] + _CP_99["Minerals"]) / 4,    # 18.02
    "Metals":                   (_CP_99["Basic_metals"] + _CP_99["Metal_products"]) / 2,  # 5.75
    "Machinery and Equipment":  (_CP_99["Machinery_nec"] + _CP_99["Office"]
                                 + _CP_99["Electrical"] + _CP_99["Communication"]
                                 + _CP_99["Medical"] + _CP_99["Auto"]
                                 + _CP_99["Other_Transport"]) / 7,                 # 6.84
    "Other Manufacturing":       _CP_99["Other_Mfg"],                              # 3.98
}
assert set(THETA.keys()) == set(AKT_GOODS_SECTORS), "θ keys must match sectors"


# ── Provinces (10), territories (3), Canada total, etc. ────────────────────
PROVINCES = [
    "Newfoundland and Labrador", "Prince Edward Island", "Nova Scotia",
    "New Brunswick", "Quebec", "Ontario", "Manitoba", "Saskatchewan",
    "Alberta", "British Columbia",
]
TERRITORIES = ["Yukon", "Northwest Territories", "Nunavut"]
# AKT merge NT+NU into one region; we keep the 10 provinces + YT + (NT+NU).
REGIONS_CA = PROVINCES + ["Yukon", "NT_NU"]
USA = "USA"


# ── HS section (Roman numeral) → AKT 9 goods sector ───────────────────────
HS_TO_AKT: dict[str, str] = {
    "I":     "Agriculture",
    "II":    "Agriculture",
    "III":   "Food",
    "IV":    "Food",
    "V":     "Mining",                    # HS V = mineral fuels + ores + raw minerals
    "VI":    "Petroleum and Chemicals",
    "VII":   "Petroleum and Chemicals",   # plastics, rubber
    "VIII":  "Textile",                   # hides, leather, fur
    "IX":    "Wood and Paper",
    "X":     "Wood and Paper",            # paper, books
    "XI":    "Textile",                   # textile, apparel
    "XII":   "Textile",                   # footwear, headgear
    "XIII":  "Petroleum and Chemicals",   # stone, ceramic, glass = non-met mineral
    "XIV":   "Other Manufacturing",       # pearls, precious stones
    "XV":    "Metals",
    "XVI":   "Machinery and Equipment",   # machinery, electrical, electronic
    "XVII":  "Machinery and Equipment",   # transportation equipment
    "XVIII": "Machinery and Equipment",   # optical, precision
    "XIX":   "Other Manufacturing",       # arms
    "XX":    "Other Manufacturing",       # misc mfg (furniture, toys)
    "XXI":   "Other Manufacturing",       # art
}


# ── Province population-weighted centroids (lat, lon) ──────────────────────
# Sourced from StatCan 2021 census population centroids. AKT use GRUMP 2000;
# province centroids are very stable across census years (population mass is
# dominated by long-established metropolitan areas).
PROVINCE_LATLON: dict[str, tuple[float, float]] = {
    "Newfoundland and Labrador":  (47.617, -52.703),  # St. John's region
    "Prince Edward Island":       (46.288, -63.130),  # Charlottetown
    "Nova Scotia":                (44.873, -63.255),  # Halifax region
    "New Brunswick":              (45.948, -66.673),  # Fredericton/Moncton
    "Quebec":                     (45.700, -73.300),  # Montreal corridor
    "Ontario":                    (43.700, -79.400),  # Toronto region
    "Manitoba":                   (49.895, -97.138),  # Winnipeg
    "Saskatchewan":               (50.450, -104.617), # Regina-Saskatoon mean
    "Alberta":                    (51.700, -114.000), # Calgary-Edmonton mean
    "British Columbia":           (49.250, -123.100), # Vancouver
    "Yukon":                      (60.721, -135.057), # Whitehorse
    "NT_NU":                      (62.454, -114.372), # Yellowknife (proxy for NT+NU)
    USA:                          (39.500, -98.350),  # US population-weighted centroid
}

# Shared land border between provinces (AKT's `Neighbor_{ni}` indicator).
ADJACENCY: dict[str, set[str]] = {
    "Newfoundland and Labrador":  {"Quebec"},
    "Prince Edward Island":       set(),  # PEI is islanded
    "Nova Scotia":                {"New Brunswick"},
    "New Brunswick":              {"Quebec", "Nova Scotia", USA},  # Maine
    "Quebec":                     {"Newfoundland and Labrador", "New Brunswick", "Ontario", USA},
    "Ontario":                    {"Quebec", "Manitoba", USA},
    "Manitoba":                   {"Ontario", "Saskatchewan", "NT_NU", USA},
    "Saskatchewan":               {"Manitoba", "Alberta", "NT_NU", USA},
    "Alberta":                    {"Saskatchewan", "British Columbia", "NT_NU", USA},
    "British Columbia":           {"Alberta", "Yukon", "NT_NU", USA},
    "Yukon":                      {"British Columbia", "NT_NU", USA},  # Alaska
    "NT_NU":                      {"Yukon", "British Columbia", "Alberta",
                                   "Saskatchewan", "Manitoba"},
    USA:                          {"New Brunswick", "Quebec", "Ontario", "Manitoba",
                                   "Saskatchewan", "Alberta", "British Columbia", "Yukon"},
}


# ── Helpers ────────────────────────────────────────────────────────────────
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle (orthodromic) distance in km. AKT use orthodromic distance
    between population-weighted centroids — see Appendix I."""
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def build_distance_neighbor() -> tuple[pd.DataFrame, pd.DataFrame]:
    regions = list(PROVINCE_LATLON.keys())
    dist = pd.DataFrame(index=regions, columns=regions, dtype=float)
    nbr  = pd.DataFrame(index=regions, columns=regions, dtype=int)
    for n in regions:
        for i in regions:
            la_n, lo_n = PROVINCE_LATLON[n]
            la_i, lo_i = PROVINCE_LATLON[i]
            dist.loc[n, i] = haversine_km(la_n, lo_n, la_i, lo_i)
            nbr.loc[n, i] = 1 if (i in ADJACENCY.get(n, set())) else 0
    return dist, nbr


# ── Load StatCan 12-10-0088 and aggregate to (n, i, su_code, year) ─────────
def load_interprov(csv_path: Path, years: range) -> pd.DataFrame:
    """Returns long-form DataFrame: columns=[year, origin, dest, su_code, sector, value].
    Diagonal entries (origin==dest) capture intra-provincial consumption from
    local production. Keeps the StatCan S&U code so we can apply sub-sector
    specific θ in Head-Ries."""
    df = pd.read_csv(csv_path, low_memory=False)
    df = df[df.REF_DATE.isin(years)]

    df["su_code"] = df["Product"].astype(str).str.extract(r"\[(M\w+)\]")
    df["sector"] = df["su_code"].map(SU_TO_AKT_GOODS)
    df = df.dropna(subset=["sector"])

    tf = df["Trade flow detail"].astype(str)
    df = df[tf.str.startswith("To ")]
    df["dest"] = tf.str.replace("To ", "", regex=False)

    df = df[~df["dest"].isin(["Canadian territorial enclaves abroad"])]
    df = df[~df["GEO"].isin(["Canada", "Canadian territorial enclaves abroad"])]
    df["origin"] = df["GEO"].replace({"Northwest Territories": "NT_NU",
                                       "Nunavut":              "NT_NU"})
    df["dest"]   = df["dest"  ].replace({"Northwest Territories": "NT_NU",
                                          "Nunavut":              "NT_NU"})

    out = (df.groupby(["REF_DATE", "origin", "dest", "su_code", "sector"],
                       as_index=False)["VALUE"].sum()
             .rename(columns={"REF_DATE": "year", "VALUE": "value_musd"}))
    out = out[out["value_musd"] > 0]
    return out


# ── Load Canada-US bilateral goods trade (12-10-0099) ─────────────────────
# Annual aggregation prepared by the bash one-liner in main(). The parquet
# has columns [year, GEO=province, Trade, "Harmonized System (HS) Sections", VALUE]
# where Trade ∈ {Domestic exports, Total imports}. VALUE is in CAD thousands.
def load_canada_us_trade(parquet_path: Path) -> pd.DataFrame:
    """Returns [year, origin, dest, sector, value_musd] with USA-as-region."""
    df = pd.read_parquet(parquet_path)
    df["hs_prefix"] = (df["Harmonized System (HS) Sections"]
                       .astype(str).str.split(" - ", n=1).str[0].str.strip())
    df["sector"] = df["hs_prefix"].map(HS_TO_AKT)
    df = df.dropna(subset=["sector"])
    # Convert CAD thousands → CAD millions for consistency with 12-10-0088
    df["VALUE"] = df["VALUE"] / 1000.0

    # Domestic exports: province → USA → (origin=prov, dest=USA)
    # Total imports:    USA → province → (origin=USA, dest=prov)
    exp = df[df.Trade == "Domestic exports"].copy()
    exp["origin"] = exp["GEO"]
    exp["dest"]   = USA
    imp = df[df.Trade == "Total imports"].copy()
    imp["origin"] = USA
    imp["dest"]   = imp["GEO"]
    both = pd.concat([exp, imp], ignore_index=True)
    out = (both.groupby(["year", "origin", "dest", "sector"], as_index=False)
                ["VALUE"].sum()
                .rename(columns={"VALUE": "value_musd"}))
    out = out[out["value_musd"] > 0]
    return out


# ── Inject USA-as-region into the bilateral panel ─────────────────────────
def add_usa_to_panel(domestic: pd.DataFrame, canus: pd.DataFrame,
                     usa_intra_per_sector: dict[str, float]) -> pd.DataFrame:
    """Appends Canada-USA bilateral flows and absolute USA × USA intra-trade
    cells (one per year-sector) to the interprovincial panel.

    `usa_intra_per_sector` is the USA × USA absorption per AKT goods sector
    in CAD millions, applied to every year in the panel (data is one snapshot
    — typically OECD ICIO 2015 USA × USA aggregated to AKT-9, in CAD millions
    via scripts/icio_usa_intra.py). Earlier versions used a 9× multiplier on
    USA→Canada exports as a proxy; that understated USA intra-trade by ~55%
    relative to ICIO and biased π_USA,USA badly low for the Head-Ries index.
    """
    dom = domestic.copy()
    dom = dom.groupby(["year", "origin", "dest", "sector"], as_index=False)["value_musd"].sum()
    dom["su_code"] = "DOMESTIC"
    canus = canus.copy(); canus["su_code"] = "CANUS"

    years = sorted(set(dom["year"].unique()) | set(canus["year"].unique()))
    usa_intra = pd.DataFrame([
        {"year": int(year), "origin": USA, "dest": USA, "sector": sector,
         "value_musd": float(value), "su_code": "USA_INTRA"}
        for year in years
        for sector, value in usa_intra_per_sector.items()
    ])

    return pd.concat([dom, canus, usa_intra], ignore_index=True)


# ── Head-Ries at AKT-sector level (matches AKT's published methodology) ───
def head_ries(panel: pd.DataFrame) -> pd.DataFrame:
    """Computes τ̂_{ni}^{j,t} at AKT-sector level. Trade flows are first
    aggregated across S&U sub-sectors within each AKT sector, then one θ_j
    is applied per AKT sector. This matches AKT's published formula."""
    # Roll up S&U codes to AKT sector
    rolled = (panel.groupby(["year", "origin", "dest", "sector"], as_index=False)
                    ["value_musd"].sum())
    rows = []
    for (year, sector), g in rolled.groupby(["year", "sector"]):
        regions = sorted(set(g["origin"]).union(g["dest"]))
        if len(regions) < 2:
            continue
        X = (g.pivot_table(index="dest", columns="origin", values="value_musd",
                            aggfunc="sum", fill_value=0.0)
               .reindex(index=regions, columns=regions, fill_value=0.0))
        row_sum = X.sum(axis=1).replace(0, np.nan)
        pi = X.div(row_sum, axis=0)
        theta = THETA[sector]
        for n in regions:
            for i in regions:
                if n == i: continue
                pi_ni = pi.loc[n, i]; pi_in = pi.loc[i, n]
                pi_nn = pi.loc[n, n]; pi_ii = pi.loc[i, i]
                if min(pi_ni, pi_in) <= 0: continue
                if min(pi_nn, pi_ii) <= 0: continue
                tau = ((pi_nn * pi_ii) / (pi_ni * pi_in)) ** (1.0 / (2.0 * theta))
                rows.append((year, sector, n, i, X.loc[n, i],
                             pi_ni, pi_nn, tau))
    return pd.DataFrame(rows, columns=["year", "sector", "n", "i",
                                        "X_ni", "pi_ni", "pi_nn", "tau_hat"])


# ── AKT regression: ln(τ̂) on dist + neighbor + intra/inter + FE ───────────
def akt_regression(hr: pd.DataFrame, dist: pd.DataFrame, nbr: pd.DataFrame,
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Runs SECTOR-BY-SECTOR OLS following AKT's eq with sector-specific α^j:
        ln(τ̂_{ni,t}^j) = α₁^j·Distance + α₂^j·Neighbor + γ_{n,t}^j + η_{i,t}^j + ε
    Returns the augmented dataframe plus a per-sector coefficient table."""
    df = hr.copy()
    df["dist_1000km"] = df.apply(lambda r: dist.loc[r["n"], r["i"]] / 1000.0, axis=1)
    df["neighbor"]    = df.apply(lambda r: nbr.loc[r["n"], r["i"]],          axis=1)
    df["log_tau"]     = np.log(df["tau_hat"])

    coefs, parts = [], []
    for sector, g in df.groupby("sector"):
        g = g.copy()
        # AKT exporter-year + importer-year FE per sector regression.
        fe_ny = pd.get_dummies(g["n"].astype(str) + "_" + g["year"].astype(str),
                                prefix="ny", drop_first=True).astype(float)
        fe_iy = pd.get_dummies(g["i"].astype(str) + "_" + g["year"].astype(str),
                                prefix="iy", drop_first=True).astype(float)
        X = pd.concat([g[["dist_1000km", "neighbor"]].astype(float),
                       fe_ny, fe_iy,
                       pd.Series(1.0, index=g.index, name="const")], axis=1)
        y = g["log_tau"].values
        beta, _, _, _ = np.linalg.lstsq(X.values, y, rcond=None)
        a1, a2 = beta[0], beta[1]
        yhat = X.values @ beta
        ss_res = ((y - yhat) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        coefs.append((sector, a1, a2, r2, len(g)))

        g["log_geo"]    = a1 * g["dist_1000km"] + a2 * g["neighbor"]
        g["log_tau_ng"] = g["log_tau"] - g["log_geo"]
        g["tau"]    = np.exp(g["log_tau"])    - 1.0
        g["tau_geo"]= np.exp(g["log_geo"])    - 1.0
        g["tau_ng"] = np.exp(g["log_tau_ng"]) - 1.0
        parts.append(g)

    coef_tbl = pd.DataFrame(coefs, columns=["sector", "alpha_dist", "alpha_nbr",
                                              "r2", "N"])
    out = pd.concat(parts, ignore_index=True)
    return out, coef_tbl


# ── Aggregation helpers for Table 1 (sector) and Table 2 (province) ────────
def trade_weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    w = weights.where(values.notna(), 0)
    return float(np.average(values.dropna(), weights=w[values.notna()])) if w.sum() > 0 else float("nan")


def replicate_tables(df: pd.DataFrame, year: int = 2015) -> None:
    """Aggregate to AKT Table 1, 2, 3 for given year and print side-by-side.
    Interprov pairs only — AKT Tables 1-3 explicitly exclude international flows."""
    y = df[df.year == year].copy()
    can_regions = set(REGIONS_CA)
    y = y[y.n.isin(can_regions) & y.i.isin(can_regions)]

    # AKT Table 1 reference values (2015 goods, from p.13 of paper)
    akt_table1 = {  # sector → (trade barrier %, geography %, non-geography %)
        "Agriculture":             (37.4, 16.6, 20.8),
        "Food":                    (60.5, 33.1, 27.4),
        "Mining":                  (15.8,  3.2, 12.6),
        "Textile":                 (20.4, 11.1,  9.3),
        "Wood and Paper":          (21.5,  8.3, 13.2),
        "Petroleum and Chemicals": (12.2,  4.9,  7.3),
        "Metals":                  (50.5, 19.3, 31.2),
        "Machinery and Equipment": (37.0, 14.6, 22.4),
        "Other Manufacturing":     (46.1, 15.8, 30.3),
    }

    print(f"\n=== Table 1: Costs of Trade Barriers by Sector ({year}) ===")
    print(f"{'Sector':<25} | {'τ̄ obs':>14} | {'τ̄ geo':>14} | {'τ̄ NG':>14} | N")
    print(f"{'':<25} | {'AKT':>6} {'us':>6} | {'AKT':>6} {'us':>6} | {'AKT':>6} {'us':>6} |")
    for sec in AKT_GOODS_SECTORS:
        sub = y[y.sector == sec]
        if sub.empty: continue
        tau_obs = trade_weighted_mean(sub["tau"],     sub["X_ni"]) * 100
        tau_geo = trade_weighted_mean(sub["tau_geo"], sub["X_ni"]) * 100
        tau_ng  = trade_weighted_mean(sub["tau_ng"],  sub["X_ni"]) * 100
        akt_o, akt_g, akt_n = akt_table1[sec]
        print(f"{sec:<25} | {akt_o:6.1f} {tau_obs:6.1f} | "
              f"{akt_g:6.1f} {tau_geo:6.1f} | "
              f"{akt_n:6.1f} {tau_ng:6.1f} | {len(sub):>3}")
    tau_obs = trade_weighted_mean(y["tau"],     y["X_ni"]) * 100
    tau_geo = trade_weighted_mean(y["tau_geo"], y["X_ni"]) * 100
    tau_ng  = trade_weighted_mean(y["tau_ng"],  y["X_ni"]) * 100
    print(f"{'All goods':<25} | {'~36':>6} {tau_obs:6.1f} | {'~17':>6} {tau_geo:6.1f} | "
          f"{'19.0':>6} {tau_ng:6.1f} | {len(y):>3}")

    # Table 3: AB row (goods-only non-geographic by destination)
    print(f"\n=== Table 3 (AB row): Goods Non-Geographic by Destination ({year}) ===")
    akt_ab_row = {  # AKT Table 3 2015 AL row from p.14
        "British Columbia": 24, "Manitoba": 21, "New Brunswick": 25,
        "Newfoundland and Labrador": 26, "Nova Scotia": 26, "Ontario": 10,
        "Prince Edward Island": 54, "Quebec": 17, "Saskatchewan": 24,
        "Yukon": 38,
    }
    print(f"{'Destination':<25} {'AKT':>6} {'us':>6}  Δ")
    ab = y[y.i == "Alberta"]  # i = origin (AB), n = destination
    for dest in sorted(akt_ab_row.keys()):
        row = ab[ab.n == dest]
        if row.empty:
            print(f"{dest:<25} {akt_ab_row[dest]:6.0f} {'—':>6}")
            continue
        our = trade_weighted_mean(row["tau_ng"], row["X_ni"]) * 100
        akt = akt_ab_row[dest]
        delta = our - akt
        print(f"{dest:<25} {akt:6.0f} {our:6.1f}  {delta:+5.1f}")

    # Table 2: per province (goods-only — note AKT Table 2 includes services)
    print(f"\n=== Table 2 (goods-only): Per-Province NG ({year}) ===")
    print(f"  Note: AKT Table 2 includes services; reference here is goods-only")
    print(f"  derived as the trade-weighted mean of AKT Table 3 entries.")
    print(f"{'Province':<25} {'τ̄ obs':>8} {'τ̄ geo':>8} {'τ̄ NG':>8}  AKT G+S")
    akt_table2 = {  # AKT Table 2 reference, 2015 non-geographic, goods+services
        "Alberta": 28.3, "British Columbia": 31.2, "Manitoba": 41.7,
        "New Brunswick": 45.8, "Newfoundland and Labrador": 50.0,
        "Nova Scotia": 45.4, "Ontario": 32.0, "Prince Edward Island": 57.8,
        "Quebec": 39.4, "Saskatchewan": 36.9, "Yukon": 66.2,
    }
    for prov in PROVINCES + ["Yukon", "NT_NU"]:
        sub = y[(y.n == prov) | (y.i == prov)]
        if sub.empty: continue
        tau_obs = trade_weighted_mean(sub["tau"],     sub["X_ni"]) * 100
        tau_geo = trade_weighted_mean(sub["tau_geo"], sub["X_ni"]) * 100
        tau_ng  = trade_weighted_mean(sub["tau_ng"],  sub["X_ni"]) * 100
        akt = akt_table2.get(prov, float("nan"))
        akt_str = f"{akt:>5.1f}%" if not np.isnan(akt) else "  —  "
        print(f"{prov:<25} {tau_obs:7.1f}% {tau_geo:7.1f}% {tau_ng:7.1f}%  {akt_str}")


# ── Main pipeline ──────────────────────────────────────────────────────────
def main() -> None:
    csv = Path("/tmp/sc12100088/12100088.csv")
    if not csv.exists():
        raise FileNotFoundError(f"Run the fetcher first; missing {csv}")

    print(f"Loading {csv} …")
    panel = load_interprov(csv, range(2007, 2016))
    print(f"Loaded {len(panel):,} (year, origin, dest, sector) rows  "
          f"covering {panel.year.min()}-{panel.year.max()}, "
          f"sectors={panel.sector.nunique()}, regions={panel.origin.nunique()}")
    print(f"Total trade value: ${panel.value_musd.sum() / 1e6:.2f}T over panel")

    canus_parquet = Path("/tmp/canada_us_trade_2007_2015.parquet")
    icio_csv = Path("/tmp/icio/2015_SML.csv")
    if canus_parquet.exists() and icio_csv.exists():
        print(f"\nAdding USA-as-region from {canus_parquet} …")
        canus = load_canada_us_trade(canus_parquet)
        from icio_usa_intra import load_usa_intra_per_akt_sector  # local import
        usa_intra = load_usa_intra_per_akt_sector(icio_csv)
        print(f"  {len(canus):,} (year, origin, dest, sector) prov-USA rows")
        print(f"  USA × USA intra (CAD M, from ICIO 2015): "
              f"total {sum(usa_intra.values()):,.0f}")
        panel = add_usa_to_panel(panel, canus, usa_intra)
        print(f"  Combined panel: {len(panel):,} rows, "
              f"regions={panel.origin.nunique()}")
    else:
        missing = [str(p) for p in (canus_parquet, icio_csv) if not p.exists()]
        print(f"\n  (skipping USA: missing {', '.join(missing)})")

    print("\nComputing Head-Ries index per AKT-sector × year …")
    hr = head_ries(panel)
    print(f"  {len(hr):,} (n, i, AKT-sector, year) pairs with valid τ̂")

    print("Building distance and neighbor matrices …")
    dist, nbr = build_distance_neighbor()

    print("Running AKT sector-by-sector regression …")
    df, coef = akt_regression(hr, dist, nbr)
    print(f"  {'Sector':<25} {'α_dist (%/1000km)':>20} {'α_nbr (%)':>12} {'R²':>6}  N")
    for _, r in coef.iterrows():
        d_pct = (math.exp(r.alpha_dist) - 1) * 100
        n_pct = (math.exp(r.alpha_nbr)  - 1) * 100
        print(f"  {r.sector:<25} {d_pct:>+19.1f}%  {n_pct:>+10.1f}% {r.r2:>6.3f} {r.N:>4}")
    print(f"  AKT report: distance 3-13% per 1000 km, neighbor -4% to -30%.")

    replicate_tables(df, year=2015)


if __name__ == "__main__":
    main()
