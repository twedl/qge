"""Build a Canadian calibration from StatCan provincial symmetric IOTs.

Source: StatCan catalogue 15-211-X, "Provincial symmetric input-output
tables, Link-1997 level" (L97). One Excel workbook per region — 10
provinces + 3 territories + CE (Canadian extraterritorial enclaves).

The L97 level has 186 industries (BS / NP / GS prefixes); we aggregate
them to the model's 23 sectors. Territories (YT, NT, NU) and CE are
folded into Rest of World. The result is six provincial-IO-derived
parquet files plus two builders carried forward from convert_statcan:

  bilateral_trade, io_matrix, value_added_share, structures_share,
  final_demand_share, portfolio_share       -- from IOT
  employment                                  -- SEPH + LFS (year-stamped)
  sectoral_dispersion                         -- CPRHS theta values

Usage::

    uv run python scripts/build_canada_iot.py \\
        --iot-zip "$HOME/Downloads/15-211-X_2021 Provincial Symmetric IOTs.zip" \\
        --year 2021
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from convert_statcan import (
    INPUTS_ROOT,
    PROVINCES,
    REGIONS,
    ROW,
    ROW_SCALE,
    build_employment,
    build_portfolio_share,
    build_sectoral_dispersion,
    summarize_employment,
    summarize_trade,
)

# ---------------------------------------------------------------- regions

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
TERRITORY_CODES: tuple[str, ...] = ("YT", "NT", "NU", "CE")
ALL_REGION_CODES: tuple[str, ...] = tuple(PROVINCE_CODE_TO_NAME) + TERRITORY_CODES


# ---------------------------------------------------------------- sectors

SECTORS: tuple[str, ...] = (
    "Agriculture, Forestry, Fishing",
    "Mining and Extraction",
    "Utilities",
    "Construction",
    "Food, Beverage, Tobacco",
    "Textile, Apparel, Leather",
    "Wood, Paper, Printing",
    "Petroleum and Chemicals",
    "Non-metallic Mineral Products",
    "Metals and Machinery",
    "Computers, Electronics, Electrical",
    "Transportation Equipment",
    "Furniture and Other Manufacturing",
    "Wholesale and Retail Trade",
    "Transportation Services",
    "Information and Communication",
    "Finance and Insurance",
    "Real Estate, Rental, Leasing",
    "Professional and Administrative Services",
    "Education",
    "Health",
    "Arts, Recreation, Accommodation, Food",
    "Public Administration and Other Services",
)


def _l97_to_sector(code: str) -> str | None:
    """Map an L97 industry code (BS / NP / GS prefix) to one of 23 sectors.

    Uses NAICS prefixes embedded in the L97 code body. Letter aggregations
    like 11A, 31A, 4AA preserve their NAICS-2 family so a 2-char prefix
    check is sufficient outside of manufacturing.
    """
    if not isinstance(code, str) or len(code) < 3:
        return None
    prefix2, body = code[:2], code[2:]
    if prefix2 in ("NP", "GS"):
        if body.startswith("61"):
            return "Education"
        if body.startswith("62"):
            return "Health"
        if body.startswith("71"):
            return "Arts, Recreation, Accommodation, Food"
        if body.startswith(("91", "813", "A")):
            return "Public Administration and Other Services"
        return None
    if prefix2 != "BS":
        return None
    p2, p3 = body[:2], body[:3]
    if p2 == "11":
        return "Agriculture, Forestry, Fishing"
    if p2 == "21":
        return "Mining and Extraction"
    if p2 == "22":
        return "Utilities"
    if p2 == "23":
        return "Construction"
    if p3 in ("311", "312"):
        return "Food, Beverage, Tobacco"
    if p3 in ("31A", "31B", "313", "314", "315", "316"):
        return "Textile, Apparel, Leather"
    if p3 in ("321", "322", "323"):
        return "Wood, Paper, Printing"
    if p3 in ("324", "325", "326"):
        return "Petroleum and Chemicals"
    if p3 == "327":
        return "Non-metallic Mineral Products"
    if p3 in ("331", "332", "333"):
        return "Metals and Machinery"
    if p3 in ("334", "335"):
        return "Computers, Electronics, Electrical"
    if p3 == "336":
        return "Transportation Equipment"
    if p3 in ("337", "339"):
        return "Furniture and Other Manufacturing"
    if p2 in ("41", "44", "45", "4A"):
        return "Wholesale and Retail Trade"
    if p2 in ("48", "49"):
        return "Transportation Services"
    if p2 == "51":
        return "Information and Communication"
    if p2 == "52":
        return "Finance and Insurance"
    if p2 == "53":
        return "Real Estate, Rental, Leasing"
    if p2 in ("54", "55", "56"):
        return "Professional and Administrative Services"
    if p2 == "61":
        return "Education"
    if p2 == "62":
        return "Health"
    if p2 in ("71", "72"):
        return "Arts, Recreation, Accommodation, Food"
    if p2 in ("81", "91"):
        return "Public Administration and Other Services"
    return None


_N_INDUSTRY_COLS = 186  # Cols 0..185 of every IOT sheet are industry intermediate use.


# ---------------------------------------------------------------- loading


def _extract_iot_zip(iot_zip: Path, work_dir: Path, year: int) -> dict[str, Path]:
    """Extract the 14 L97 workbooks. Returns {region_code: path}."""
    work_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    with zipfile.ZipFile(iot_zip, "r") as zf:
        for region in ALL_REGION_CODES:
            entry = f"Link-1997 level/IOTs provincial symmetric {region} L97 {year}.xlsx"
            target = work_dir / Path(entry).name
            if not target.exists():
                try:
                    info = zf.getinfo(entry)
                except KeyError as exc:
                    raise FileNotFoundError(
                        f"entry not in {iot_zip}: {entry}"
                    ) from exc
                with zf.open(info) as src, target.open("wb") as dst:
                    dst.write(src.read())
            out[region] = target
    return out


def _load_iot_sheet(xlsx_path: Path, sheet: str) -> pd.DataFrame:
    raw = pd.read_excel(xlsx_path, sheet_name=sheet, header=None)
    body = raw.iloc[8:, 3:].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    body.index = raw.iloc[8:, 1].tolist()
    body.columns = pd.RangeIndex(start=0, stop=body.shape[1])
    return body


def _load_col_codes(xlsx_path: Path) -> list:
    """Read row 6 col codes from DomesticUse (post the 3-prefix-col layout)."""
    df = pd.read_excel(xlsx_path, sheet_name="DomesticUse", header=None, nrows=7)
    return df.iloc[6, 3:].tolist()


def load_all(iot_zip: Path, work_dir: Path, year: int) -> tuple[dict, dict]:
    """Load IOT sheets and column codes for every region. Returns (iots, col_codes)."""
    paths = _extract_iot_zip(iot_zip, work_dir, year)
    iots: dict[str, dict[str, pd.DataFrame]] = {}
    col_codes: dict[str, list] = {}
    for code, path in paths.items():
        print(f"  reading {code:<3} from {path.name}")
        iots[code] = {
            sheet: _load_iot_sheet(path, sheet)
            for sheet in ("BasicPrice", "DomesticUse", "InternatImportUse", "InterprovImportUse")
        }
        col_codes[code] = _load_col_codes(path)
    return iots, col_codes


# ---------------------------------------------------------------- aggregation


def _industry_codes_with_sectors(sheet: pd.DataFrame) -> list[tuple[str, str]]:
    """[(l97_code, sector_name)] for industry rows in a sheet."""
    pairs = []
    for code in sheet.index:
        sector = _l97_to_sector(code) if isinstance(code, str) else None
        if sector is not None:
            pairs.append((code, sector))
    return pairs


def _aggregate_rows(sheet: pd.DataFrame, mapping: list[tuple[str, str]]) -> pd.DataFrame:
    """Sum sheet rows to (J=23, n_cols)."""
    out = pd.DataFrame(0.0, index=list(SECTORS), columns=sheet.columns)
    for code, sector in mapping:
        out.loc[sector] = out.loc[sector].add(sheet.loc[code], fill_value=0.0)
    return out


def _aggregate_industry_cols(
    sheet: pd.DataFrame, mapping: list[tuple[str, str]]
) -> pd.DataFrame:
    """Sum the first 186 industry columns to (n_rows, J=23). Other cols dropped."""
    out = pd.DataFrame(0.0, index=sheet.index, columns=list(SECTORS))
    code_to_sector = dict(mapping)
    # Col 0..185 codes match the first 186 row codes (symmetric layout).
    col_codes = list(sheet.index[:_N_INDUSTRY_COLS])
    for pos, code in enumerate(col_codes):
        if code in code_to_sector:
            out[code_to_sector[code]] = out[code_to_sector[code]].add(
                sheet.iloc[:, pos], fill_value=0.0
            )
    return out


def _first_region_positions(col_codes: list) -> dict[str, int]:
    """Map region two-letter code → its FIRST column position (= the export col).

    The interprov export block precedes the import block; both use bare
    two-letter region codes. Using first-occurrence picks the export.
    """
    seen: dict[str, int] = {}
    for pos, code in enumerate(col_codes):
        if isinstance(code, str) and code in ALL_REGION_CODES and code not in seen:
            seen[code] = pos
    return seen


# ---------------------------------------------------------------- builders


def build_io_matrix_iot(iots: dict) -> pd.DataFrame:
    """Canadian-average IO matrix: IO[i, j] = share of input i in sector j's
    intermediate bundle. Columns sum to 1 per j.

    Sums the (J×J) provincial intermediate-use matrices, clips small
    negatives from secondary-output adjustments, then normalizes each
    column to a probability vector.
    """
    aggregate = pd.DataFrame(0.0, index=list(SECTORS), columns=list(SECTORS))
    for prov_code in PROVINCE_CODE_TO_NAME:
        bp = iots[prov_code]["BasicPrice"]
        mapping = _industry_codes_with_sectors(bp)
        # Keep only industry rows and industry cols (drop primary inputs, TOTAL).
        industry_codes = [c for c, _ in mapping]
        sub = bp.loc[industry_codes].iloc[:, :_N_INDUSTRY_COLS]
        # Aggregate rows by sector.
        row_agg = _aggregate_rows(sub, mapping)
        # Aggregate cols by sector. Col positions 0..185 correspond to row
        # codes of the same name; reuse mapping.
        col_agg = pd.DataFrame(0.0, index=list(SECTORS), columns=list(SECTORS))
        sector_for = dict(mapping)
        for pos, code in enumerate(industry_codes):
            col_agg[sector_for[code]] = col_agg[sector_for[code]].add(
                row_agg.iloc[:, pos], fill_value=0.0
            )
        aggregate = aggregate.add(col_agg, fill_value=0.0)

    aggregate = aggregate.clip(lower=0.0)
    col_sums = aggregate.sum(axis=0)
    norm = aggregate.div(col_sums.where(col_sums > 0, 1.0), axis=1)

    rows = []
    for using in SECTORS:
        for supplying in SECTORS:
            rows.append({
                "source_sector": supplying,
                "dest_sector": using,
                "value": float(norm.loc[supplying, using]),
            })
    return pd.DataFrame(rows)


def build_value_added_share_iot(iots: dict) -> pd.DataFrame:
    """γ_jn = (basic-price primary inputs) / (basic-price total cost) per sector × province."""
    primary_codes = (
        "PRM300000",  # Subsidies on production (negative)
        "PRM400000",  # Taxes on production
        "PRM500000",  # Wages
        "PRM600000",  # Employers' social contributions
        "PRM700000",  # Gross mixed income
        "PRM800000",  # Gross operating surplus
    )
    rows = []
    for prov_code, prov_name in PROVINCE_CODE_TO_NAME.items():
        bp = iots[prov_code]["BasicPrice"]
        mapping = _industry_codes_with_sectors(bp)
        col_agg = _aggregate_industry_cols(bp, mapping)
        primary = col_agg.loc[list(primary_codes)].sum(axis=0)
        total = col_agg.loc["TOTAL"]
        gamma = (primary / total.where(total > 0, 1.0)).clip(lower=0.0, upper=0.99)
        for sector in SECTORS:
            rows.append({"sector": sector, "region": prov_name, "value": float(gamma[sector])})

    df = pd.DataFrame(rows)
    canada_avg = df.groupby("sector")["value"].mean()
    for sector in SECTORS:
        rows.append({"sector": sector, "region": ROW, "value": float(canada_avg[sector])})
    return pd.DataFrame(rows).sort_values(["sector", "region"]).reset_index(drop=True)


def build_structures_share_iot(iots: dict) -> pd.DataFrame:
    """B_n = GOS / (Compensation + Mixed + GOS), province-wide totals."""
    rows = []
    for prov_code, prov_name in PROVINCE_CODE_TO_NAME.items():
        bp = iots[prov_code]["BasicPrice"]
        mapping = _industry_codes_with_sectors(bp)
        col_agg = _aggregate_industry_cols(bp, mapping)
        wages = col_agg.loc["PRM500000"].sum() + col_agg.loc["PRM600000"].sum()
        mixed = col_agg.loc["PRM700000"].sum()
        gos = col_agg.loc["PRM800000"].sum()
        denom = wages + mixed + gos
        rows.append({
            "region": prov_name,
            "value": float(gos / denom) if denom > 0 else 0.0,
        })
    canada_b = float(np.mean([r["value"] for r in rows]))
    rows.append({"region": ROW, "value": canada_b})
    return pd.DataFrame(rows).sort_values("region").reset_index(drop=True)


def build_final_demand_share_iot(iots: dict, col_codes: dict) -> pd.DataFrame:
    """α_jn = sector j's share of household + NPISH + government consumption.

    Uses BasicPrice values for final-consumption columns (PEC*, CEN*, CEG*).
    Excludes investment, inventory changes, and trade — this matches the
    Cobb-Douglas final-demand interpretation in CPRHS.
    """
    rows = []
    for prov_code, prov_name in PROVINCE_CODE_TO_NAME.items():
        bp = iots[prov_code]["BasicPrice"]
        codes = col_codes[prov_code]
        consumption_cols = [
            i for i, c in enumerate(codes)
            if isinstance(c, str) and (c.startswith("PEC") or c.startswith("CEN") or c.startswith("CEG"))
        ]
        mapping = _industry_codes_with_sectors(bp)
        # Aggregate industry rows to 23 sectors, keep all cols.
        row_agg = _aggregate_rows(
            bp.loc[[c for c, _ in mapping]], mapping
        )
        consumption_by_sector = row_agg.iloc[:, consumption_cols].sum(axis=1).clip(lower=0.0)
        total = consumption_by_sector.sum()
        alpha = consumption_by_sector / total if total > 0 else consumption_by_sector
        for sector in SECTORS:
            rows.append({"sector": sector, "region": prov_name, "value": float(alpha[sector])})

    # ROW α: mean of provincial α renormalized.
    df = pd.DataFrame(rows)
    canada_alpha = df.groupby("sector")["value"].mean()
    canada_alpha = canada_alpha / canada_alpha.sum()
    for sector in SECTORS:
        rows.append({"sector": sector, "region": ROW, "value": float(canada_alpha[sector])})
    return pd.DataFrame(rows).sort_values(["sector", "region"]).reset_index(drop=True)


def build_bilateral_trade_iot(iots: dict, col_codes: dict) -> pd.DataFrame:
    """Bilateral trade [j, origin, destination] from L97 trade blocks.

    * intraprovincial flow = origin's domestic absorption (intermediate +
      final consumption + capital formation + inventory)
    * province → province = origin's DomesticUse interprov-export column
    * province → ROW = origin's INTEX + INTRX + exports to territories
    * ROW → province = destination's InternatImportUse local absorption +
      InterprovImportUse from territories (via territory DomesticUse export
      columns to that destination)
    * ROW → ROW = synthetic, ROW_SCALE × Canadian per-sector gross output -
      ROW exports to Canadian provinces (clamped ≥ 1).
    """
    province_codes = tuple(PROVINCE_CODE_TO_NAME)
    province_names = tuple(PROVINCE_CODE_TO_NAME[c] for c in province_codes)

    # Aggregate each province's DomesticUse rows to 23 sectors.
    dom_by_prov: dict[str, pd.DataFrame] = {}
    for code in province_codes:
        sheet = iots[code]["DomesticUse"]
        dom_by_prov[code] = _aggregate_rows(sheet, _industry_codes_with_sectors(sheet))

    trade: dict[tuple[str, str, str], float] = {}

    # 1. province-origin flows.
    for o_code in province_codes:
        o_name = PROVINCE_CODE_TO_NAME[o_code]
        sheet = dom_by_prov[o_code]
        codes = col_codes[o_code]
        first_seen = _first_region_positions(codes)
        first_export_pos = min(first_seen.values())
        intl_exp_cols = [
            i for i, c in enumerate(codes)
            if isinstance(c, str) and c.startswith(("INTEX", "INTRX"))
        ]

        # n_d = each province
        for d_code in province_codes:
            d_name = PROVINCE_CODE_TO_NAME[d_code]
            if d_code == o_code:
                local_mask = [
                    i for i in range(first_export_pos)
                    if not (isinstance(codes[i], str)
                            and codes[i].startswith(("INTEX", "INTRX", "INTIM")))
                ]
                for j_idx, sector in enumerate(SECTORS):
                    val = float(sheet.iloc[j_idx, local_mask].sum())
                    trade[(sector, o_name, d_name)] = max(val, 0.0)
            else:
                pos = first_seen[d_code]
                for j_idx, sector in enumerate(SECTORS):
                    trade[(sector, o_name, d_name)] = max(float(sheet.iat[j_idx, pos]), 0.0)

        # n_d = ROW (INTEX + INTRX + territory export cols)
        row_cols = list(intl_exp_cols)
        for terr in TERRITORY_CODES:
            if terr in first_seen:
                row_cols.append(first_seen[terr])
        for j_idx, sector in enumerate(SECTORS):
            trade[(sector, o_name, ROW)] = max(float(sheet.iloc[j_idx, row_cols].sum()), 0.0)

    # 2. ROW → each province. Use InternatImportUse + territory DomesticUse exports.
    intl_imp_per_prov: dict[str, pd.Series] = {}
    for code in province_codes:
        sheet = iots[code]["InternatImportUse"]
        mapping = _industry_codes_with_sectors(sheet)
        agg = _aggregate_rows(sheet, mapping)
        codes = col_codes[code]
        # InternatImportUse has fewer cols than DomesticUse (no interprov blocks);
        # we sum everything that's not a re-export (INTRX) or trade-summary col.
        keep = []
        for i in range(agg.shape[1]):
            c = codes[i] if i < len(codes) else None
            if isinstance(c, str):
                if c.startswith(("INTEX", "INTRX", "INTIM")):
                    continue
                if c in ALL_REGION_CODES:
                    continue
                if c == "TOTAL":
                    continue
            keep.append(i)
        intl_imp_per_prov[code] = agg.iloc[:, keep].sum(axis=1).clip(lower=0.0)

    # Territory exports to each province (from territory DomesticUse files).
    terr_to_prov: dict[tuple[str, str], pd.Series] = {}
    for terr in TERRITORY_CODES:
        sheet = iots[terr]["DomesticUse"]
        mapping = _industry_codes_with_sectors(sheet)
        terr_agg = _aggregate_rows(sheet, mapping)
        first_seen_t = _first_region_positions(col_codes[terr])
        for d_code in province_codes:
            if d_code in first_seen_t:
                pos = first_seen_t[d_code]
                terr_to_prov[(terr, d_code)] = terr_agg.iloc[:, pos].clip(lower=0.0)

    for d_code in province_codes:
        d_name = PROVINCE_CODE_TO_NAME[d_code]
        flow = intl_imp_per_prov[d_code].copy()
        for terr in TERRITORY_CODES:
            if (terr, d_code) in terr_to_prov:
                flow = flow.add(terr_to_prov[(terr, d_code)], fill_value=0.0)
        for sector in SECTORS:
            trade[(sector, ROW, d_name)] = float(max(flow[sector], 0.0))

    # 3. ROW → ROW (synthetic).
    canada_go: dict[str, float] = {sector: 0.0 for sector in SECTORS}
    for sector in SECTORS:
        for o_name in province_names:
            for d in (*province_names, ROW):
                canada_go[sector] += trade[(sector, o_name, d)]
    row_exports: dict[str, float] = {sector: 0.0 for sector in SECTORS}
    for sector in SECTORS:
        for d_name in province_names:
            row_exports[sector] += trade[(sector, ROW, d_name)]
    for sector in SECTORS:
        ro_total = ROW_SCALE * canada_go[sector]
        trade[(sector, ROW, ROW)] = max(ro_total - row_exports[sector], 1.0)

    rows = []
    for (sector, source, destination), value in trade.items():
        rows.append({
            "sector": sector, "source": source,
            "destination": destination, "value": value,
        })
    return (
        pd.DataFrame(rows)
        .sort_values(["sector", "source", "destination"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iot-zip", type=Path, required=True,
        help="Path to 15-211-X provincial symmetric IOT zip",
    )
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/statcan_iot"))
    args = parser.parse_args()

    out_dir = INPUTS_ROOT / f"canada_{args.year}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.year} L97 IOTs from {args.iot_zip}...")
    iots, col_codes = load_all(args.iot_zip, args.work_dir, args.year)

    print()
    print("Building io_matrix.parquet (Canadian native, L97 industry-averaged)...")
    io = build_io_matrix_iot(iots)
    print(f"  IO range: min={io['value'].min():.4f}  max={io['value'].max():.4f}  mean={io['value'].mean():.4f}")
    io.to_parquet(out_dir / "io_matrix.parquet", index=False)
    print(f"  wrote {out_dir / 'io_matrix.parquet'}  ({len(io)} rows)")
    print()

    print("Building value_added_share.parquet (γ from L97 BasicPrice)...")
    gamma = build_value_added_share_iot(iots)
    n_outside = ((gamma["value"] < 0) | (gamma["value"] > 1)).sum()
    print(f"  γ mean={gamma['value'].mean():.3f}  min={gamma['value'].min():.3f}  max={gamma['value'].max():.3f}  outside_[0,1]={n_outside}")
    gamma.to_parquet(out_dir / "value_added_share.parquet", index=False)
    print(f"  wrote {out_dir / 'value_added_share.parquet'}  ({len(gamma)} rows)")
    print()

    print("Building structures_share.parquet (B from primary inputs)...")
    B = build_structures_share_iot(iots)
    for _, row in B.iterrows():
        print(f"    {row['region']:<30} B = {row['value']:.4f}")
    B.to_parquet(out_dir / "structures_share.parquet", index=False)
    print(f"  wrote {out_dir / 'structures_share.parquet'}")
    print()

    print("Building final_demand_share.parquet (α from PEC/CEN/CEG cols)...")
    alpha = build_final_demand_share_iot(iots, col_codes)
    col_sums = alpha.groupby("region")["value"].sum()
    print(f"  column sums (should all be 1.0): min={col_sums.min():.4f}, max={col_sums.max():.4f}")
    alpha.to_parquet(out_dir / "final_demand_share.parquet", index=False)
    print(f"  wrote {out_dir / 'final_demand_share.parquet'}  ({len(alpha)} rows)")
    print()

    print("Building bilateral_trade.parquet from L97 trade blocks...")
    trade = build_bilateral_trade_iot(iots, col_codes)
    summarize_trade(trade)
    trade.to_parquet(out_dir / "bilateral_trade.parquet", index=False)
    print(f"  wrote {out_dir / 'bilateral_trade.parquet'}  ({len(trade)} rows)")
    print()

    print(f"Building employment.parquet (SEPH + LFS, {args.year})...")
    emp = build_employment(args.year)
    summarize_employment(emp)
    emp.to_parquet(out_dir / "employment.parquet", index=False)
    print(f"  wrote {out_dir / 'employment.parquet'}  ({len(emp)} rows)")
    print()

    print("Building sectoral_dispersion.parquet (CPRHS θ values)...")
    theta = build_sectoral_dispersion()
    theta.to_parquet(out_dir / "sectoral_dispersion.parquet", index=False)
    print(f"  wrote {out_dir / 'sectoral_dispersion.parquet'}  ({len(theta)} rows)")
    print()

    print("Building portfolio_share.parquet (ι ≡ 0)...")
    iota = build_portfolio_share()
    iota.to_parquet(out_dir / "portfolio_share.parquet", index=False)
    print(f"  wrote {out_dir / 'portfolio_share.parquet'}  ({len(iota)} rows)")


if __name__ == "__main__":
    main()
