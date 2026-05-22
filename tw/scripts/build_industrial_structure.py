"""Derive TW industrial-structure primitives from StatCan provincial L97 IOTs.

Builds α (final-demand share), γ (regional intermediate share), φ (labor /
value-added share), and the national IO matrix at TW's 22-sector ISIC Rev 3
taxonomy from the provincial Symmetric Input-Output Tables (StatCan 15-211-X,
year 2021 — the latest published vintage).

Writes four parquets into ``data/inputs/canada_2022/``, replacing the
canada_2010 carry-overs that Phase 4a shipped with:

    final_demand_share.parquet   α (J, N)
    intermediate_share.parquet   γ (J, J, N)  -- regional, NEW
    labor_share.parquet          φ (J, N)
    io_matrix.parquet            national IO (J, J)

Source: 14 provincial/territorial L97 xlsx workbooks inside
``15-211-X_2021 Provincial Symmetric IOTs.zip``. Each workbook has a
``BasicPrice`` sheet whose first 186 rows are industry codes (BS / NP / GS
prefix); columns 0..185 are the symmetric intermediate-use block followed
by final-demand columns.

The ROW row (WLD region in TW's 11-region setup) is the Canadian average,
matching the canada_2010 convention. Territories are excluded — only the
10 provinces feed Canadian averages.

Run from tw/:
    uv run python scripts/build_industrial_structure.py \\
        --iot-zip "$HOME/Downloads/15-211-X_2021 Provincial Symmetric IOTs.zip" \\
        --year 2021
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


# 10 provinces + WLD (ROW). Order matches canada_2010 sector_labels convention.
PROVINCE_CODE_TO_NAME: dict[str, str] = {
    "AB": "Alberta",
    "BC": "British Columbia",
    "MB": "Manitoba",
    "NB": "New Brunswick",
    "NL": "Newfoundland and Labrador",
    "NS": "Nova Scotia",
    "ON": "Ontario",
    "PE": "Prince Edward Island",
    "QC": "Quebec",
    "SK": "Saskatchewan",
}
WLD = "WLD"

# 22 TW sectors, ordered as in tw/data/inputs/canada_2010/sector_labels.parquet.
SECTORS: tuple[str, ...] = (
    "Agriculture, Mining",
    "Food, Textiles",
    "Wood",
    "Paper",
    "Chemicals, Rubber",
    "Metals",
    "Equipment, Vehicles",
    "Manufacturing, n.e.c.",
    "Utilities",
    "Construction",
    "Wholesale and Retail",
    "Hotels and Restaurants",
    "Transport",
    "Communication",
    "Finance",
    "Real Estate",
    "Software",
    "Other Business Services",
    "Public Admin.",
    "Education",
    "Health and Social",
    "Other Services",
)

_N_INDUSTRY_COLS = 186


def _l97_to_tw_sector(code: str) -> str | None:
    """Map an L97 industry code to one of TW's 22 ISIC Rev 3 sectors.

    NAICS → ISIC Rev 3 concordance (TW Table 9):
      NAICS 11,21         → 01-14 Agriculture, Mining
      NAICS 22            → 40-41 Utilities
      NAICS 23            → 45 Construction
      NAICS 31* (food/tex)→ 15-19 Food, Textiles
      NAICS 321           → 20 Wood
      NAICS 322,323       → 21-22 Paper (paper + printing/publishing)
      NAICS 324-326       → 23-25 Chemicals, Rubber
      NAICS 327,331,332   → 26-28 Metals (non-metallic min + metals)
      NAICS 333-336       → 29-35 Equipment, Vehicles
      NAICS 337,339       → 36-37 Manufacturing, n.e.c.
      NAICS 41,44,45      → 50-52 Wholesale and Retail
      NAICS 48,49         → 60-63 Transport
      NAICS 511 (ex-5112) → 22 Paper (publishing)
      NAICS 5112          → 72 Software
      NAICS 512,515       → 92 Other Services (motion picture, broadcasting)
      NAICS 517,51A       → 64 Communication
      NAICS 518           → 72 Software (data processing)
      NAICS 52            → 65-67 Finance
      NAICS 53            → 70-71 Real Estate
      NAICS 5415          → 72 Software (computer systems design)
      NAICS 541 (others)  → 73-74 Other Business Services
      NAICS 55,56         → 73-74 Other Business Services
      NAICS 61            → 80 Education
      NAICS 62            → 85 Health and Social
      NAICS 71            → 92 Other Services
      NAICS 72            → 55 Hotels and Restaurants
      NAICS 81            → 90-93 Other Services
      NAICS 91 (GS)       → 75 Public Admin.
    """
    if not isinstance(code, str) or len(code) < 3:
        return None
    p2, body = code[:2], code[2:]

    if p2 == "GS":
        if body.startswith("61"):
            return "Education"
        if body.startswith(("622", "623")):
            return "Health and Social"
        if body.startswith("91"):
            return "Public Admin."
        return None
    if p2 == "NP":
        if body.startswith("61"):
            return "Education"
        if body.startswith(("622", "623", "624")):
            return "Health and Social"
        return "Other Services"
    if p2 != "BS":
        return None

    b2, b3 = body[:2], body[:3]
    if b2 in ("11", "21"):
        return "Agriculture, Mining"
    if b2 == "22":
        return "Utilities"
    if b2 == "23":
        return "Construction"
    if b3 in ("311", "312", "31A", "31B", "313", "314", "315", "316"):
        return "Food, Textiles"
    if b3 == "321":
        return "Wood"
    if b3 in ("322", "323"):
        return "Paper"
    if b3 in ("324", "325", "326"):
        return "Chemicals, Rubber"
    if b3 in ("327", "331", "332"):
        return "Metals"
    if b3 in ("333", "334", "335", "336"):
        return "Equipment, Vehicles"
    if b3 in ("337", "339"):
        return "Manufacturing, n.e.c."
    if b2 in ("41", "44", "45", "4A"):
        return "Wholesale and Retail"
    if b2 in ("48", "49"):
        return "Transport"
    if b3 == "511":
        return "Software" if body.startswith("511200") else "Paper"
    if b3 in ("512", "515"):
        return "Other Services"
    if body.startswith("51A") or b3 == "517":
        return "Communication"
    if b3 == "518":
        return "Software"
    if b2 == "51":
        return "Communication"
    if b2 == "52":
        return "Finance"
    if b2 == "53":
        return "Real Estate"
    if b3 == "541":
        return "Software" if body.startswith("541500") else "Other Business Services"
    if b2 in ("55", "56"):
        return "Other Business Services"
    if b2 == "61":
        return "Education"
    if b2 == "62":
        return "Health and Social"
    if b2 == "71":
        return "Other Services"
    if b2 == "72":
        return "Hotels and Restaurants"
    if b2 == "81":
        return "Other Services"
    if b2 == "91":
        return "Public Admin."
    return None


def _extract_zip(iot_zip: Path, work_dir: Path, year: int) -> dict[str, Path]:
    """Extract the 10 provincial L97 workbooks."""
    work_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    with zipfile.ZipFile(iot_zip, "r") as zf:
        for region in PROVINCE_CODE_TO_NAME:
            entry = f"Link-1997 level/IOTs provincial symmetric {region} L97 {year}.xlsx"
            target = work_dir / Path(entry).name
            if not target.exists():
                with zf.open(zf.getinfo(entry)) as src, target.open("wb") as dst:
                    dst.write(src.read())
            out[region] = target
    return out


def _load_basicprice(xlsx_path: Path) -> pd.DataFrame:
    """Read BasicPrice sheet; rows indexed by L97 code, columns are positional."""
    raw = pd.read_excel(xlsx_path, sheet_name="BasicPrice", header=None)
    body = raw.iloc[8:, 3:].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    body.index = raw.iloc[8:, 1].tolist()
    body.columns = pd.RangeIndex(start=0, stop=body.shape[1])
    return body


def _load_col_codes(xlsx_path: Path) -> list:
    """Row-6 column codes from DomesticUse (final-demand column labels)."""
    df = pd.read_excel(xlsx_path, sheet_name="DomesticUse", header=None, nrows=7)
    return df.iloc[6, 3:].tolist()


def _industry_mapping(sheet: pd.DataFrame) -> list[tuple[str, str]]:
    """[(l97_code, tw_sector)] for industry rows."""
    pairs = []
    for code in sheet.index:
        sector = _l97_to_tw_sector(code) if isinstance(code, str) else None
        if sector is not None:
            pairs.append((code, sector))
    return pairs


def _aggregate_rows(sheet: pd.DataFrame, mapping: list[tuple[str, str]]) -> pd.DataFrame:
    """Sum sheet rows from L97 → TW 22 sectors."""
    out = pd.DataFrame(0.0, index=list(SECTORS), columns=sheet.columns)
    for code, sector in mapping:
        out.loc[sector] = out.loc[sector].add(sheet.loc[code], fill_value=0.0)
    return out


def _aggregate_industry_cols(
    sheet: pd.DataFrame, mapping: list[tuple[str, str]]
) -> pd.DataFrame:
    """Sum the first 186 industry columns from L97 → TW 22 sectors."""
    out = pd.DataFrame(0.0, index=sheet.index, columns=list(SECTORS))
    code_to_sector = dict(mapping)
    col_codes = list(sheet.index[:_N_INDUSTRY_COLS])
    for pos, code in enumerate(col_codes):
        if code in code_to_sector:
            out[code_to_sector[code]] = out[code_to_sector[code]].add(
                sheet.iloc[:, pos], fill_value=0.0
            )
    return out


def _provincial_jj(sheet: pd.DataFrame, mapping: list[tuple[str, str]]) -> pd.DataFrame:
    """Aggregate the J×J intermediate-use block to (TW 22, TW 22).

    Rows = supplying sectors, columns = using sectors.
    """
    industry_codes = [c for c, _ in mapping]
    sub = sheet.loc[industry_codes].iloc[:, :_N_INDUSTRY_COLS]
    row_agg = _aggregate_rows(sub, mapping)
    col_agg = pd.DataFrame(0.0, index=list(SECTORS), columns=list(SECTORS))
    sector_for = dict(mapping)
    for pos, code in enumerate(industry_codes):
        col_agg[sector_for[code]] = col_agg[sector_for[code]].add(
            row_agg.iloc[:, pos], fill_value=0.0
        )
    return col_agg


def build_intermediate_share(
    iots: dict[str, pd.DataFrame], phi_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Regional γ^{k,j}_n + national IO matrix.

    Returns (gamma_long_df, io_matrix_long_df).

    γ (intermediate_share.parquet): column-normalized — γ[:,j,n] sums to 1 per
    (output sector, region). This is the conditional input mix.

    io_matrix.parquet: the **Canadian-aggregate** γ multiplied column-wise by
    (1 - φ^j_Canada) so that IO[:,j] sums to (1 - φ^j). This is the
    canada_2010 convention — the model's equations consume IO directly, not
    γ × (1-φ) at runtime.
    """
    prov_jj: dict[str, pd.DataFrame] = {}
    for code, sheet in iots.items():
        mapping = _industry_mapping(sheet)
        prov_jj[code] = _provincial_jj(sheet, mapping).clip(lower=0.0)

    rows = []
    for code in PROVINCE_CODE_TO_NAME:
        m = prov_jj[code]
        col_sums = m.sum(axis=0)
        norm = m.div(col_sums.where(col_sums > 0, 1.0), axis=1)
        for using in SECTORS:
            for supplying in SECTORS:
                rows.append({
                    "input_sector": supplying,
                    "dest_sector": using,
                    "region": code,
                    "value": float(norm.loc[supplying, using]),
                })

    # WLD γ: Canadian-aggregate, column-normalized.
    agg = sum(prov_jj.values())
    col_sums = agg.sum(axis=0)
    wld_gamma = agg.div(col_sums.where(col_sums > 0, 1.0), axis=1)
    for using in SECTORS:
        for supplying in SECTORS:
            rows.append({
                "input_sector": supplying,
                "dest_sector": using,
                "region": WLD,
                "value": float(wld_gamma.loc[supplying, using]),
            })
    gamma_df = pd.DataFrame(rows)

    # io_matrix = WLD γ × (1 - φ^j_WLD): columns sum to (1 - φ^j_Canada).
    phi_canada = phi_df[phi_df["region"] == WLD].set_index("sector")["value"]
    intermediate_share = 1.0 - phi_canada.reindex(SECTORS).to_numpy()
    io_rows = []
    for j_pos, using in enumerate(SECTORS):
        for supplying in SECTORS:
            io_rows.append({
                "source_sector": supplying,
                "dest_sector": using,
                "value": float(wld_gamma.loc[supplying, using] * intermediate_share[j_pos]),
            })
    io_df = pd.DataFrame(io_rows)
    return gamma_df, io_df


def build_labor_share(iots: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """φ^j_n = primary inputs / total cost per (sector, region).

    Primary inputs = labour comp + employers' SS + mixed income + GOS +
    production taxes (less subsidies). Equals 1 - intermediate share.
    """
    primary_codes = (
        "PRM300000",  # Subsidies on production (negative)
        "PRM400000",  # Taxes on production
        "PRM500000",  # Wages
        "PRM600000",  # Employers' social contributions
        "PRM700000",  # Gross mixed income
        "PRM800000",  # Gross operating surplus
    )
    rows = []
    sector_totals: dict[str, list[float]] = {s: [] for s in SECTORS}
    for code, sheet in iots.items():
        mapping = _industry_mapping(sheet)
        col_agg = _aggregate_industry_cols(sheet, mapping)
        primary = col_agg.loc[list(primary_codes)].sum(axis=0)
        total = col_agg.loc["TOTAL"]
        phi = (primary / total.where(total > 0, 1.0)).clip(lower=0.0, upper=0.99)
        for sector in SECTORS:
            v = float(phi[sector])
            rows.append({"sector": sector, "region": code, "value": v})
            sector_totals[sector].append(v)

    for sector in SECTORS:
        rows.append({
            "sector": sector,
            "region": WLD,
            "value": float(np.mean(sector_totals[sector])),
        })
    return pd.DataFrame(rows).sort_values(["sector", "region"]).reset_index(drop=True)


def build_final_demand_share(
    iots: dict[str, pd.DataFrame], col_codes: dict[str, list]
) -> pd.DataFrame:
    """α^j_n = sector j's share of (household + NPISH + government) consumption.

    Uses BasicPrice rows × final-consumption columns (PEC*, CEN*, CEG*).
    Excludes investment, inventory, trade — Cobb-Douglas final-demand convention.
    """
    rows = []
    canada_alpha = pd.Series(0.0, index=list(SECTORS))
    for code, sheet in iots.items():
        codes = col_codes[code]
        consumption_cols = [
            i for i, c in enumerate(codes)
            if isinstance(c, str)
            and (c.startswith("PEC") or c.startswith("CEN") or c.startswith("CEG"))
        ]
        mapping = _industry_mapping(sheet)
        row_agg = _aggregate_rows(
            sheet.loc[[c for c, _ in mapping]], mapping
        )
        consumption_by_sector = row_agg.iloc[:, consumption_cols].sum(axis=1).clip(lower=0.0)
        total = consumption_by_sector.sum()
        alpha = consumption_by_sector / total if total > 0 else consumption_by_sector
        for sector in SECTORS:
            rows.append({"sector": sector, "region": code, "value": float(alpha[sector])})
            canada_alpha[sector] += alpha[sector]

    canada_alpha = canada_alpha / canada_alpha.sum()
    for sector in SECTORS:
        rows.append({"sector": sector, "region": WLD, "value": float(canada_alpha[sector])})
    return pd.DataFrame(rows).sort_values(["sector", "region"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iot-zip",
        type=Path,
        default=Path.home()
        / "Downloads"
        / "15-211-X_2021 Provincial Symmetric IOTs.zip",
    )
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/statcan_iot"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "inputs" / "canada_2022",
    )
    args = parser.parse_args()

    print(f"Extracting {args.year} L97 workbooks from {args.iot_zip.name}...")
    paths = _extract_zip(args.iot_zip, args.work_dir, args.year)

    basicprice: dict[str, pd.DataFrame] = {}
    col_codes: dict[str, list] = {}
    for code, path in paths.items():
        print(f"  reading {code} from {path.name}")
        basicprice[code] = _load_basicprice(path)
        col_codes[code] = _load_col_codes(path)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Building labor_share.parquet...")
    phi_df = build_labor_share(basicprice)
    phi_df.to_parquet(args.out_dir / "labor_share.parquet", index=False)

    print("Building intermediate_share.parquet + io_matrix.parquet...")
    gamma_df, io_df = build_intermediate_share(basicprice, phi_df)
    gamma_df.to_parquet(args.out_dir / "intermediate_share.parquet", index=False)
    io_df.to_parquet(args.out_dir / "io_matrix.parquet", index=False)

    print("Building final_demand_share.parquet...")
    alpha_df = build_final_demand_share(basicprice, col_codes)
    alpha_df.to_parquet(args.out_dir / "final_demand_share.parquet", index=False)

    print(f"\nWrote 4 parquets to {args.out_dir}")
    print(f"  γ regional: {len(gamma_df)} rows (22 sectors × 22 sectors × 11 regions = {22*22*11})")
    print(f"  IO national: {len(io_df)} rows")
    print(f"  φ:         {len(phi_df)} rows")
    print(f"  α:         {len(alpha_df)} rows")


if __name__ == "__main__":
    main()
