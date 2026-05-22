"""Derive TW industrial-structure primitives from StatCan provincial IOTs.

Builds α (final-demand share), γ (regional intermediate share), φ (labor /
value-added share), and the national IO matrix at TW's 22-sector ISIC Rev 3
taxonomy from the provincial Symmetric Input-Output Tables (StatCan 15-211-X).

Writes four parquets into ``data/inputs/canada_2022/``:

    final_demand_share.parquet   α (J, N)
    intermediate_share.parquet   γ (J, J, N)  -- regional
    labor_share.parquet          φ (J, N)
    io_matrix.parquet            national IO (J, J)

Source: provincial workbooks (one per province) from StatCan 15-211-X at
the chosen aggregation level — Detail (D, ~220 industries) or Link-1997
(L97, 186 industries). Each workbook has a ``BasicPrice`` sheet whose
first rows are industry codes (BS / NP / GS prefix); the leading columns
are the symmetric intermediate-use block followed by final-demand columns.

The WLD region (Rest-of-World in TW's 11-region setup) is the Canadian
average. Territories are excluded — only the 10 provinces feed averages.

Run from tw/ (current 2022 Detail-level vintage):
    uv run python scripts/build_industrial_structure.py \\
        --iot-zip "$HOME/Downloads/2022-20260522T154200Z-3-001.zip" \\
        --year 2022 --level D
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

_INDUSTRY_PREFIXES: tuple[str, ...] = ("BS", "NP", "GS")


def _count_industry_rows(sheet: pd.DataFrame) -> int:
    """Number of leading rows whose label has an industry prefix (BS/NP/GS).
    L97 has 186; Detail has ~220; Summary has ~32."""
    n = 0
    for code in sheet.index:
        if isinstance(code, str) and code[:2] in _INDUSTRY_PREFIXES:
            n += 1
        else:
            break
    return n


def _l97_to_tw_sector(code: str) -> str | None:
    """Map an L97 / Detail-level industry code to TW's 22 ISIC Rev 3 sectors.

    Concordance handles both NAICS 2017 (used in L97 vintages 2014-2021) and
    NAICS 2022 (used in 2022+ Detail). The Information sector (NAICS 51) was
    restructured in NAICS 2022:

      NAICS 2017     →  NAICS 2022    →  TW sector
      511 publishing →  513 publishing → Paper (5131xx) / Software (513200)
      515 broadcast  →  516 broadcast  → Other Services
      519 other info →  519 other info → Other Services

    Other ISIC Rev 3 mapping (TW Table 9):
      NAICS 11,21         → 01-14 Agriculture, Mining
      NAICS 22            → 40-41 Utilities
      NAICS 23            → 45 Construction
      NAICS 31* (food)    → 15-19 Food, Textiles
      NAICS 321           → 20 Wood
      NAICS 322,323       → 21-22 Paper (paper + printing)
      NAICS 324-326       → 23-25 Chemicals, Rubber
      NAICS 327,331,332   → 26-28 Metals
      NAICS 333-336       → 29-35 Equipment, Vehicles
      NAICS 337,339       → 36-37 Manufacturing, n.e.c.
      NAICS 41,44,45      → 50-52 Wholesale and Retail
      NAICS 48,49         → 60-63 Transport
      NAICS 512           → 92 Other Services (motion picture)
      NAICS 517,51A       → 64 Communication
      NAICS 518           → 72 Software (data processing/cloud)
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
        if body.startswith("62"):  # NAICS 62 = health + social assistance (incl. 62A aggregates)
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
    # NAICS 2017 Information sector (used in L97 vintages 2014-2021)
    if b3 == "511":
        return "Software" if body.startswith("511200") else "Paper"
    if b3 in ("512", "515"):
        return "Other Services"
    # NAICS 2022 Information sector reorganization
    if b3 == "513":
        # NAICS 2022 moved publishing 511 → 513. 5131xx = print/digital publishing
        # (newspaper/periodical/book/directory) → Paper; 513200 = software
        # publishers → Software.
        return "Software" if body.startswith("513200") else "Paper"
    if b3 == "516":
        # NAICS 2022 created 516 for broadcasting + content providers
        # (incl. media streaming, was 515 in NAICS 2017).
        return "Other Services"
    if b3 == "519":
        # NAICS 2022 519 = web search, libraries, archives, other info services.
        return "Other Services"
    # Common
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


def _extract_zip(iot_zip: Path, work_dir: Path, year: int, level: str) -> dict[str, Path]:
    """Extract the 10 provincial workbooks for ``level`` ∈ {L97, D}.

    Two StatCan naming conventions are supported:
      - L97 (pre-2022): ``Link-1997 level/IOTs provincial symmetric XX L97 YYYY.xlsx``
      - Detail (2022+): ``{year}/Detail level/IOT provincial XX D YYYY.xlsx``
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    if level == "L97":
        prefix, infix, suffix = "Link-1997 level/", "IOTs provincial symmetric ", " L97 "
    elif level == "D":
        prefix, infix, suffix = f"{year}/Detail level/", "IOT provincial ", " D "
    else:
        raise ValueError(f"unknown level={level!r}; expected 'L97' or 'D'")
    out: dict[str, Path] = {}
    with zipfile.ZipFile(iot_zip, "r") as zf:
        for region in PROVINCE_CODE_TO_NAME:
            entry = f"{prefix}{infix}{region}{suffix}{year}.xlsx"
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
    """Sum the leading industry columns from L97/Detail → TW 22 sectors."""
    n_ind = _count_industry_rows(sheet)
    out = pd.DataFrame(0.0, index=sheet.index, columns=list(SECTORS))
    code_to_sector = dict(mapping)
    col_codes = list(sheet.index[:n_ind])
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
    n_ind = _count_industry_rows(sheet)
    industry_codes = [c for c, _ in mapping]
    sub = sheet.loc[industry_codes].iloc[:, :n_ind]
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


_PRIMARY_INPUT_CODES: tuple[str, ...] = (
    "PRM300000",  # Subsidies on production (negative)
    "PRM400000",  # Taxes on production
    "PRM500000",  # Wages
    "PRM600000",  # Employers' social contributions
    "PRM700000",  # Gross mixed income
    "PRM800000",  # Gross operating surplus
)


def build_labor_share(
    iots: dict[str, pd.DataFrame], national: bool = False
) -> pd.DataFrame:
    """φ^j_n = primary inputs / total cost per (sector, region).

    If ``national=False`` (default), each province carries its own observed
    φ vector and WLD gets the Canadian-aggregate value (production-weighted).

    If ``national=True``, the Canadian-aggregate φ is replicated across every
    province (and WLD) — matches canada_2010 / AT-2016 / OECD-STAN convention
    treating φ as a structural production-function parameter.
    """
    rows = []
    primary_by_region: dict[str, pd.Series] = {}
    total_by_region: dict[str, pd.Series] = {}
    for code, sheet in iots.items():
        mapping = _industry_mapping(sheet)
        col_agg = _aggregate_industry_cols(sheet, mapping)
        primary_by_region[code] = col_agg.loc[list(_PRIMARY_INPUT_CODES)].sum(axis=0)
        total_by_region[code] = col_agg.loc["TOTAL"]

    canada_primary = sum(primary_by_region.values())
    canada_total = sum(total_by_region.values())
    canada_phi = (canada_primary / canada_total.where(canada_total > 0, 1.0)
                  ).clip(lower=0.0, upper=0.99)

    for code in PROVINCE_CODE_TO_NAME:
        if national:
            phi = canada_phi
        else:
            phi = (primary_by_region[code]
                   / total_by_region[code].where(total_by_region[code] > 0, 1.0)
                   ).clip(lower=0.0, upper=0.99)
        for sector in SECTORS:
            rows.append({"sector": sector, "region": code, "value": float(phi[sector])})

    for sector in SECTORS:
        rows.append({"sector": sector, "region": WLD, "value": float(canada_phi[sector])})
    return pd.DataFrame(rows).sort_values(["sector", "region"]).reset_index(drop=True)


_FINAL_DEMAND_PREFIXES: tuple[str, ...] = (
    "PEC",  # household consumption (COICOP detail)
    "CEN",  # NPISH consumption
    "CEG",  # government consumption
    "COB", "COG", "COH",  # construction investment (non-res / govt / residential)
    "MEB", "MEG", "MEN",  # machinery and equipment investment
    "IPB", "IPG", "IPN",  # intellectual-property investment
    "INV",                # change in inventories
)


def build_final_demand_share(
    iots: dict[str, pd.DataFrame], col_codes: dict[str, list]
) -> pd.DataFrame:
    """α^j_n = sector j's share of domestic final demand (ex-exports).

    Uses BasicPrice rows × all domestic final-demand columns: consumption
    (PEC/CEN/CEG) + investment (COB/COG/COH for construction, MEB/MEG/MEN for
    M&E, IPB/IPG/IPN for IP) + inventories (INV). Excludes interprovincial /
    international trade columns. Matches AT-2016 β^j convention (where e.g.
    Construction β ≈ 0.13 reflects residential investment).
    """
    rows = []
    canada_alpha = pd.Series(0.0, index=list(SECTORS))
    for code, sheet in iots.items():
        codes = col_codes[code]
        fd_cols = [
            i for i, c in enumerate(codes)
            if isinstance(c, str) and c.startswith(_FINAL_DEMAND_PREFIXES)
        ]
        mapping = _industry_mapping(sheet)
        row_agg = _aggregate_rows(
            sheet.loc[[c for c, _ in mapping]], mapping
        )
        fd_by_sector = row_agg.iloc[:, fd_cols].sum(axis=1).clip(lower=0.0)
        total = fd_by_sector.sum()
        alpha = fd_by_sector / total if total > 0 else fd_by_sector
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
        default=Path.home() / "Downloads" / "2022-20260522T154200Z-3-001.zip",
    )
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument(
        "--level",
        choices=("D", "L97"),
        default="D",
        help="StatCan aggregation level. D = Detail (~220 industries, 2022+); "
             "L97 = Link-1997 level (186 industries, pre-2022).",
    )
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/statcan_iot"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "inputs" / "canada_2022",
    )
    parser.add_argument(
        "--phi-national",
        action="store_true",
        help="Use Canadian-aggregate φ replicated across provinces (matches "
             "canada_2010 / AT-2016 convention). Default: regional φ.",
    )
    args = parser.parse_args()

    print(f"Extracting {args.year} {args.level} workbooks from {args.iot_zip.name}...")
    paths = _extract_zip(args.iot_zip, args.work_dir, args.year, args.level)

    basicprice: dict[str, pd.DataFrame] = {}
    col_codes: dict[str, list] = {}
    for code, path in paths.items():
        print(f"  reading {code} from {path.name}")
        basicprice[code] = _load_basicprice(path)
        col_codes[code] = _load_col_codes(path)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building labor_share.parquet (national={args.phi_national})...")
    phi_df = build_labor_share(basicprice, national=args.phi_national)
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
