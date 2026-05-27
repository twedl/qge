"""Industrial structure from a national L97 IOT — replicates TW's approach.

Inspecting canada_2010's intermediate_share/labor_share/final_demand_share
shows zero cross-region variation: TW used a single Canadian-aggregate value
for γ, φ, and α and replicated it across 11 regions. This script reproduces
that by reading the NATIONAL L97 IOT for a chosen year (StatCan 15-211-X
national variant) and producing the four industrial-structure parquets at
the 22-sector ISIC Rev 3 schema.

Source: National.zip with per-year subfolders. Path to the workbook:
    National/{year}/IOTs national symmetric L97 {year}.xlsx

Run from tw/:
    uv run python scripts/build_industrial_structure_national.py --year 2010 \\
        --national-zip ~/Downloads/National.zip
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_industrial_structure import (  # noqa: E402
    SECTORS, PROVINCE_CODE_TO_NAME, WLD, _l97_to_tw_sector,
    _FINAL_DEMAND_PREFIXES, _PRIMARY_INPUT_CODES,
)


def _extract_national(zip_path: Path, year: int, work_dir: Path) -> Path:
    """Extract a national IOT workbook for one year. Prefers L97; falls back
    to Detail when L97 isn't available (e.g. 2022 onwards). Returns its path.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    candidates = [
        f"National/{year}/IOTs national symmetric L97 {year}.xlsx",
        f"National/{year}/IOTs national symmetric D {year}.xlsx",
        f"National/{year}/IOT national D {year}.xlsx",  # 2022+ naming
        f"National/{year}/IOTs national symmetric L-Public {year}.xls",  # pre-2010
    ]
    with zipfile.ZipFile(zip_path, "r") as zf:
        zip_names = set(zf.namelist())
        entry = next((c for c in candidates if c in zip_names), None)
        if entry is None:
            raise FileNotFoundError(
                f"no national IOT for year {year} in {zip_path}; "
                f"tried {candidates}")
        target = work_dir / Path(entry).name
        if target.exists():
            return target
        with zf.open(zf.getinfo(entry)) as src, target.open("wb") as dst:
            dst.write(src.read())
    return target


def _load_national_basicprice(xlsx_path: Path) -> pd.DataFrame:
    """Load the BasicPrice sheet into a DataFrame indexed by industry code.

    Header layout varies by vintage:
      - 2010-2021 L97: titles row at 11, col codes at 10, data starts row 12
      - 2022+ Detail: titles row at  6, col codes at  5, data starts row  7

    Auto-detect by finding the row where column B is the string "Code"
    (the titles row, which immediately precedes the first data row).
    Col codes are one row above; data starts one row below.
    """
    raw = pd.read_excel(xlsx_path, sheet_name="BasicPrice", header=None)
    titles_row = None
    for r in range(min(20, raw.shape[0])):
        if isinstance(raw.iat[r, 1], str) and raw.iat[r, 1].strip() == "Code":
            titles_row = r
            break
    if titles_row is None:
        raise RuntimeError(f"Could not locate header row in {xlsx_path.name}")
    col_codes_row = titles_row - 1
    data_start_row = titles_row + 1

    body = raw.iloc[data_start_row:, 3:].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    body.index = raw.iloc[data_start_row:, 1].tolist()
    body.columns = raw.iloc[col_codes_row, 3:].tolist()
    body = body.loc[[i for i in body.index if isinstance(i, str)]]
    body = body.loc[:, [c for c in body.columns if isinstance(c, str)]]
    # StatCan workbook has a few duplicate column codes (e.g. when the same
    # NAICS subcode appears as both an industry-block column and a final-
    # demand column). Keep the first occurrence to avoid double-counting.
    body = body.loc[~body.index.duplicated(keep="first"),
                    ~body.columns.duplicated(keep="first")]
    return body


def _industry_codes(df: pd.DataFrame) -> list[str]:
    return [c for c in df.index if isinstance(c, str) and c[:2] in ("BS","NP","GS")]


def build(year: int, zip_path: Path, work_dir: Path) -> dict[str, pd.DataFrame]:
    """Compute Canadian-aggregate α, γ, φ, IO at 22-sector level.

    Returns dict with four DataFrames in the long-form schema expected by
    qge.io.load_inputs (with values replicated across 11 regions).
    """
    xlsx = _extract_national(zip_path, year, work_dir)
    print(f"  reading {xlsx.name} ...")
    df = _load_national_basicprice(xlsx)

    ind_codes = _industry_codes(df)
    # Mapping L97 code → TW 22 sector
    code_to_sector: dict[str, str] = {}
    for code in ind_codes:
        sec = _l97_to_tw_sector(code)
        if sec is not None:
            code_to_sector[code] = sec

    # ── (J × J) intermediate-use matrix at 22-sector ────────────────────
    # First: aggregate rows (supplying industries) into 22 sectors
    used_codes = list(code_to_sector.keys())
    sub = df.loc[used_codes]
    # Sum rows by supplying sector → (22, full_cols)
    sub = sub.copy()
    sub["__sec"] = [code_to_sector[c] for c in used_codes]
    row_agg = sub.groupby("__sec").sum().reindex(SECTORS).fillna(0.0)
    # Sum cols of the industry block by using sector → (22, 22)
    ind_cols = [c for c in row_agg.columns if isinstance(c, str)
                and c[:2] in ("BS", "NP", "GS") and c in code_to_sector]
    iu_full = row_agg[ind_cols]
    col_to_sector = {c: code_to_sector[c] for c in ind_cols}
    iu_full = iu_full.rename(columns=col_to_sector)
    # Multiple cols may share the same using-sector → groupby axis=1
    iu = iu_full.T.groupby(level=0).sum().T  # (22 supplying, 22 using)
    iu = iu.reindex(index=SECTORS, columns=SECTORS).fillna(0.0).clip(lower=0)

    # ── φ_j: labor share = primary inputs / total output per using sector ──
    primary_rows = [c for c in _PRIMARY_INPUT_CODES if c in df.index]
    if not primary_rows:
        # National workbook uses shorter codes: PRM10000..PRM80000 (8 digits)
        primary_rows = [c for c in df.index if isinstance(c, str)
                        and c.startswith("PRM")]
    if "Total" in df.index:
        total_row = "Total"
    elif "TOTAL" in df.index:
        total_row = "TOTAL"
    else:
        raise RuntimeError("no Total row in national workbook")

    # Sum primary inputs per using-industry column, then aggregate to 22 sectors
    primary_per_col = df.loc[primary_rows].sum(axis=0)
    total_per_col = df.loc[total_row]
    primary_per_sec = pd.Series(0.0, index=SECTORS)
    total_per_sec = pd.Series(0.0, index=SECTORS)
    for c in ind_cols:
        s = col_to_sector[c]
        primary_per_sec[s] += float(primary_per_col[c])
        total_per_sec[s]   += float(total_per_col[c])
    phi_can = (primary_per_sec / total_per_sec.where(total_per_sec > 0, 1.0)
              ).clip(lower=0.0, upper=0.99)

    # ── γ^{k,j}: column-normalized intermediate-use ─────────────────────
    col_sums = iu.sum(axis=0)
    gamma_can = iu.div(col_sums.where(col_sums > 0, 1.0), axis=1)

    # ── α_j: final-demand share ─────────────────────────────────────────
    fd_cols = [c for c in df.columns if isinstance(c, str)
               and c.startswith(_FINAL_DEMAND_PREFIXES)]
    fd_supply = df.loc[used_codes, fd_cols].sum(axis=1).clip(lower=0)
    alpha_unnorm = pd.Series(0.0, index=SECTORS)
    for c in used_codes:
        s = code_to_sector[c]
        alpha_unnorm[s] += float(fd_supply[c])
    alpha_can = alpha_unnorm / alpha_unnorm.sum() if alpha_unnorm.sum() > 0 else alpha_unnorm

    # ── IO matrix = γ × (1 - φ), Canada-aggregate (J,J) ─────────────────
    io_can = gamma_can.multiply((1.0 - phi_can).reindex(SECTORS).to_numpy(), axis=1)

    # ── Replicate across 11 regions (10 provs + WLD) ────────────────────
    regions = list(PROVINCE_CODE_TO_NAME) + [WLD]

    gamma_rows = []
    for n in regions:
        for k in SECTORS:
            for j in SECTORS:
                gamma_rows.append({"input_sector": k, "dest_sector": j,
                                    "region": n, "value": float(gamma_can.loc[k, j])})
    gamma_df = pd.DataFrame(gamma_rows)

    phi_rows = []
    for s in SECTORS:
        for n in regions:
            phi_rows.append({"sector": s, "region": n, "value": float(phi_can[s])})
    phi_df = pd.DataFrame(phi_rows).sort_values(["sector","region"]).reset_index(drop=True)

    alpha_rows = []
    for s in SECTORS:
        for n in regions:
            alpha_rows.append({"sector": s, "region": n, "value": float(alpha_can[s])})
    alpha_df = pd.DataFrame(alpha_rows).sort_values(["sector","region"]).reset_index(drop=True)

    io_rows = []
    for k in SECTORS:
        for j in SECTORS:
            io_rows.append({"source_sector": k, "dest_sector": j,
                             "value": float(io_can.loc[k, j])})
    io_df = pd.DataFrame(io_rows)

    return {
        "intermediate_share": gamma_df,
        "labor_share":        phi_df,
        "final_demand_share": alpha_df,
        "io_matrix":          io_df,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2010)
    parser.add_argument("--national-zip", type=Path,
                        default=Path.home() / "Downloads" / "National.zip")
    parser.add_argument("--work-dir", type=Path,
                        default=Path("/tmp/national_iot_extract"))
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output dir (default: data/inputs/canada_{year}_rebuild)")
    args = parser.parse_args()

    out_dir = args.out_dir or (Path(__file__).resolve().parent.parent
                                / "data" / "inputs"
                                / f"canada_{args.year}_rebuild")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building national L97 industrial structure for {args.year} → {out_dir}")
    parts = build(args.year, args.national_zip, args.work_dir)

    for name, df in parts.items():
        path = out_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        print(f"  {name+'.parquet':<30} {len(df):>5} rows")


if __name__ == "__main__":
    main()
