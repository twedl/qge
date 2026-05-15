"""Expand canada_2021/ from 11 regions to 17 by breaking out Canada's
major trading partners (USA, China, UK, Japan, Mexico, Germany) from ROW.

The output dir is ``data/inputs/canada_2021_partners/``. The 11-region
``canada_2021/`` is left untouched so both calibrations remain usable.

Approach:
- Province-to-ROW flows from canada_2021/ are split into province-to-partner
  flows in proportion to Canada's ICIO partner shares per sector (the
  Armington simplification: each province inherits the national export
  mix). Same on the ROW-to-province side. Provincial trade with each
  named partner is therefore the StatCan total × an ICIO-derived ratio.
- Trade between named partners and the residual is taken directly from
  ICIO (converted from USD millions to CAD thousands).
- γ, α for new regions come from ICIO (VA / output and final
  consumption shares). B and employment use Canadian-average and
  Canadian-labor-productivity proxies (documented limitations).

Future improvements (recorded in DATA.md):
- Use StatCan Table 12-10-0099 (provincial merchandise trade by partner)
  for the province-to-partner split instead of the national-aggregate
  Armington share.
- ILO Modelled Estimates for country × sector employment instead of
  Canadian-labor-productivity scaling.
- Country-specific structures share B (e.g., from Penn World Table
  capital share series).

Usage::
    uv run python scripts/expand_icio_partners.py \\
        --icio "$HOME/Downloads/2016-2022_SML/2021_SML.csv" \\
        --src data/inputs/canada_2021/ \\
        --out data/inputs/canada_2021_partners/
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from add_icio_row import (  # noqa: E402
    CAN,
    FINAL_CONSUMPTION,
    ISIC_TO_SECTOR,
    _split,
    load_icio,
)
from build_canada_iot import ROW, SECTORS  # noqa: E402

# ---------------------------------------------------------------- partners

PARTNERS: tuple[str, ...] = ("USA", "CHN", "GBR", "JPN", "MEX", "DEU")
PARTNER_DISPLAY: dict[str, str] = {
    "USA": "United States",
    "CHN": "China",
    "GBR": "United Kingdom",
    "JPN": "Japan",
    "MEX": "Mexico",
    "DEU": "Germany",
}
RESIDUAL_CODE = "RESIDUAL"   # internal key for the non-named non-CAN aggregate
RESIDUAL_DISPLAY = "Rest of World"

# Country-group → display name. Includes Canadian-named provinces
# implicitly — those don't appear here, only the new country regions.
COUNTRY_DISPLAY: dict[str, str] = {**PARTNER_DISPLAY, RESIDUAL_CODE: RESIDUAL_DISPLAY}


# ---------------------------------------------------------------- helpers


def _sector_isics() -> dict[str, list[str]]:
    """{sector_name: [isic codes]}"""
    out: dict[str, list[str]] = {s: [] for s in SECTORS}
    for isic, sec in ISIC_TO_SECTOR.items():
        out[sec].append(isic)
    return out


def _country_for(label: str) -> str:
    if "_" not in label:
        return ""
    return label.split("_", 1)[0]


def _industry_for(label: str) -> str:
    if "_" not in label:
        return ""
    return label.split("_", 1)[1]


def _classify_country(c: str) -> str:
    """Map an ICIO country code to one of our groups: CAN, partner code, or RESIDUAL."""
    if c == CAN:
        return CAN
    if c in PARTNERS:
        return c
    return RESIDUAL_CODE


def _row_indices_by_group_sector(
    row_labels: list[str], sector_isics: dict[str, list[str]]
) -> dict[tuple[str, str], np.ndarray]:
    """Index matrix rows by (country_group, sector_name)."""
    isic_to_sector = {isic: sec for sec, isics in sector_isics.items() for isic in isics}
    out: dict[tuple[str, str], list[int]] = {}
    for i, lbl in enumerate(row_labels):
        c = _country_for(lbl)
        i_code = _industry_for(lbl)
        if not c or i_code not in isic_to_sector:
            continue
        key = (_classify_country(c), isic_to_sector[i_code])
        out.setdefault(key, []).append(i)
    return {k: np.asarray(v, dtype=np.int64) for k, v in out.items()}


def _col_indices_by_group(
    col_labels: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    """Index matrix cols by country_group → {category: positions}.

    Categories: 'industry' (any destination industry col), 'consumption'
    (HFCE/NPISH/GGFC), 'all' (industry + every final-demand col except
    OUT).
    """
    out: dict[str, dict[str, list[int]]] = {}
    for i, lbl in enumerate(col_labels):
        if lbl == "OUT":
            continue
        c = _country_for(lbl)
        if not c:
            continue
        group = _classify_country(c)
        bucket = out.setdefault(group, {"industry": [], "consumption": [], "all": []})
        suf = _industry_for(lbl)
        if suf in ISIC_TO_SECTOR:
            bucket["industry"].append(i)
        if suf in FINAL_CONSUMPTION:
            bucket["consumption"].append(i)
        bucket["all"].append(i)
    return {
        g: {k: np.asarray(v, dtype=np.int64) for k, v in buckets.items()}
        for g, buckets in out.items()
    }


def _col_industry_by_group_sector(
    col_labels: list[str], sector_isics: dict[str, list[str]]
) -> dict[tuple[str, str], np.ndarray]:
    """{(country_group, sector): industry col positions}"""
    isic_to_sector = {isic: sec for sec, isics in sector_isics.items() for isic in isics}
    out: dict[tuple[str, str], list[int]] = {}
    for i, lbl in enumerate(col_labels):
        c = _country_for(lbl)
        suf = _industry_for(lbl)
        if not c or suf not in isic_to_sector:
            continue
        key = (_classify_country(c), isic_to_sector[suf])
        out.setdefault(key, []).append(i)
    return {k: np.asarray(v, dtype=np.int64) for k, v in out.items()}


# ---------------------------------------------------------------- aggregates


def compute_partner_aggregates(M: pd.DataFrame, row_labels, col_labels) -> dict:
    """Per-country-group per-sector aggregates from ICIO."""
    Mn = M.to_numpy()
    sector_isics = _sector_isics()

    row_idx = _row_indices_by_group_sector(row_labels, sector_isics)
    col_idx_by_group = _col_indices_by_group(col_labels)
    col_industry_idx = _col_industry_by_group_sector(col_labels, sector_isics)
    out_col = col_labels.index("OUT")
    va_row = row_labels.index("VA")

    groups = (CAN, *PARTNERS, RESIDUAL_CODE)

    # OUT per (group, sector): sum of OUT col across origin rows.
    gross_output: dict[tuple[str, str], float] = {}
    for (g, s), idx in row_idx.items():
        gross_output[(g, s)] = float(Mn[idx, out_col].sum())

    # Bilateral trade: origin (g_o, sector j) → destination g_d, summed over
    # all destination cols (industry + final demand, not OUT).
    bilateral: dict[tuple[str, str, str], float] = {}
    for (g_o, s), origin_rows in row_idx.items():
        for g_d in groups:
            if g_d not in col_idx_by_group:
                continue
            dest_cols = col_idx_by_group[g_d]["all"]
            bilateral[(s, g_o, g_d)] = float(Mn[np.ix_(origin_rows, dest_cols)].sum())

    # γ: VA / OUT per (group, sector).
    gamma: dict[tuple[str, str], float] = {}
    for (g, s), col_pos in col_industry_idx.items():
        va = float(Mn[va_row, col_pos].sum())
        out_at = float(Mn[col_pos.repeat(1), out_col].sum())  # OUT col is by origin row;
        # the "OUT for industry j in country g" is OUT[(g,j), out_col].
        # Use row_idx for the corresponding (g, s).
        if (g, s) in row_idx:
            out_at = float(Mn[row_idx[(g, s)], out_col].sum())
        gamma[(g, s)] = float(va / out_at) if out_at > 0 else 0.5

    # α: final consumption share by sector for each destination group.
    # α_g_s = (consumption in g on sector s, all origins) / (total consumption in g)
    # The consumption rows are any origin × industry i ∈ s.
    alpha: dict[tuple[str, str], float] = {}
    for g_d in groups:
        if g_d not in col_idx_by_group:
            continue
        consumption_cols = col_idx_by_group[g_d]["consumption"]
        total_by_sector: dict[str, float] = {}
        # For each sector, sum across all origin rows (any group) × these
        # destination cols.
        for s in SECTORS:
            origin_rows_any = np.concatenate(
                [row_idx[(g, s)] for g in groups if (g, s) in row_idx]
            )
            total_by_sector[s] = float(
                Mn[np.ix_(origin_rows_any, consumption_cols)].sum()
            )
        total = sum(total_by_sector.values())
        for s in SECTORS:
            alpha[(g_d, s)] = total_by_sector[s] / total if total > 0 else 0.0

    return {
        "gross_output": gross_output,
        "bilateral": bilateral,
        "gamma": gamma,
        "alpha": alpha,
    }


def compute_canadian_partner_shares(agg: dict) -> dict[tuple[str, str], float]:
    """For each (sector, partner), Canada's export share of total non-CAN exports.

    Used to split province → ROW into province → partner.
    """
    shares: dict[tuple[str, str], float] = {}
    for s in SECTORS:
        partners = (*PARTNERS, RESIDUAL_CODE)
        exports = {p: agg["bilateral"].get((s, CAN, p), 0.0) for p in partners}
        total = sum(exports.values())
        for p in partners:
            shares[(s, p)] = exports[p] / total if total > 0 else 1.0 / len(partners)
    return shares


def compute_canadian_partner_import_shares(agg: dict) -> dict[tuple[str, str], float]:
    """For each (sector, partner), the partner's export share of Canada's total imports.

    Used to split ROW → province into partner → province.
    """
    shares: dict[tuple[str, str], float] = {}
    for s in SECTORS:
        partners = (*PARTNERS, RESIDUAL_CODE)
        imports = {p: agg["bilateral"].get((s, p, CAN), 0.0) for p in partners}
        total = sum(imports.values())
        for p in partners:
            shares[(s, p)] = imports[p] / total if total > 0 else 1.0 / len(partners)
    return shares


# ---------------------------------------------------------------- builder


def expand_calibration(
    src_dir: Path, out_dir: Path, agg: dict, scale: float
) -> None:
    """Build the 17-region calibration in ``out_dir``.

    ``scale`` is the ICIO USD-millions → StatCan CAD-thousands factor
    computed in ``add_icio_row.apply_row_overrides`` (≈ 1247).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    partner_names = tuple(PARTNER_DISPLAY[p] for p in PARTNERS)
    new_country_codes = (*PARTNERS, RESIDUAL_CODE)
    new_country_displays = tuple(COUNTRY_DISPLAY[c] for c in new_country_codes)

    trade = pd.read_parquet(src_dir / "bilateral_trade.parquet")
    gamma = pd.read_parquet(src_dir / "value_added_share.parquet")
    alpha = pd.read_parquet(src_dir / "final_demand_share.parquet")
    B = pd.read_parquet(src_dir / "structures_share.parquet")
    portfolio = pd.read_parquet(src_dir / "portfolio_share.parquet")
    employment = pd.read_parquet(src_dir / "employment.parquet")

    province_names = sorted(set(trade["source"]) - {ROW})

    # ---- bilateral trade ----------
    # Province ↔ Province cells: unchanged.
    province_to_province = trade[
        trade["source"].isin(province_names) & trade["destination"].isin(province_names)
    ].copy()

    # Province → partner / residual: split province → ROW by Canadian export shares.
    exp_shares = compute_canadian_partner_shares(agg)
    p_to_row = trade[
        trade["source"].isin(province_names) & (trade["destination"] == ROW)
    ].set_index(["sector", "source"])["value"]
    province_to_country_rows = []
    for (sector, src), v in p_to_row.items():
        for code in new_country_codes:
            share = exp_shares[(sector, code)]
            province_to_country_rows.append({
                "sector": sector, "source": src,
                "destination": COUNTRY_DISPLAY[code],
                "value": float(v * share),
            })
    province_to_country = pd.DataFrame(province_to_country_rows)

    # Partner / residual → Province: split ROW → province by import shares.
    imp_shares = compute_canadian_partner_import_shares(agg)
    row_to_p = trade[
        (trade["source"] == ROW) & trade["destination"].isin(province_names)
    ].set_index(["sector", "destination"])["value"]
    country_to_province_rows = []
    for (sector, dest), v in row_to_p.items():
        for code in new_country_codes:
            share = imp_shares[(sector, code)]
            country_to_province_rows.append({
                "sector": sector, "source": COUNTRY_DISPLAY[code],
                "destination": dest,
                "value": float(v * share),
            })
    country_to_province = pd.DataFrame(country_to_province_rows)

    # Country × Country (named × named, named ↔ residual, residual × residual):
    # straight from ICIO with the USD-millions → CAD-thousands scale.
    country_to_country_rows = []
    for s in SECTORS:
        for c_o in new_country_codes:
            for c_d in new_country_codes:
                v = agg["bilateral"].get((s, c_o, c_d), 0.0) * scale
                country_to_country_rows.append({
                    "sector": s, "source": COUNTRY_DISPLAY[c_o],
                    "destination": COUNTRY_DISPLAY[c_d],
                    "value": max(float(v), 1.0),  # ensure positive (interior eq)
                })
    country_to_country = pd.DataFrame(country_to_country_rows)

    full_trade = (
        pd.concat([
            province_to_province, province_to_country,
            country_to_province, country_to_country,
        ], ignore_index=True)
        .sort_values(["sector", "source", "destination"])
        .reset_index(drop=True)
    )

    # ---- γ ----------
    province_gamma = gamma[gamma["region"].isin(province_names)].copy()
    country_gamma_rows = []
    for code in new_country_codes:
        for s in SECTORS:
            g = agg["gamma"].get((code, s), 0.5)
            g = float(np.clip(g, 0.01, 0.99))
            country_gamma_rows.append({"sector": s, "region": COUNTRY_DISPLAY[code], "value": g})
    full_gamma = pd.concat([province_gamma, pd.DataFrame(country_gamma_rows)],
                           ignore_index=True).sort_values(["sector", "region"]).reset_index(drop=True)

    # ---- α ----------
    province_alpha = alpha[alpha["region"].isin(province_names)].copy()
    country_alpha_rows = []
    for code in new_country_codes:
        for s in SECTORS:
            a = agg["alpha"].get((code, s), 0.0)
            country_alpha_rows.append({"sector": s, "region": COUNTRY_DISPLAY[code], "value": float(a)})
    # Renormalize each new region's α to sum to 1 (rounding insurance).
    country_alpha = pd.DataFrame(country_alpha_rows)
    for code in new_country_codes:
        rgn = COUNTRY_DISPLAY[code]
        mask = country_alpha["region"] == rgn
        total = country_alpha.loc[mask, "value"].sum()
        if total > 0:
            country_alpha.loc[mask, "value"] /= total
    full_alpha = pd.concat([province_alpha, country_alpha], ignore_index=True) \
        .sort_values(["sector", "region"]).reset_index(drop=True)

    # ---- B ----------
    province_B = B[B["region"].isin(province_names)].copy()
    canada_b = float(province_B["value"].mean())
    country_B_rows = [
        {"region": COUNTRY_DISPLAY[code], "value": canada_b}
        for code in new_country_codes
    ]
    full_B = pd.concat([province_B, pd.DataFrame(country_B_rows)],
                       ignore_index=True).sort_values("region").reset_index(drop=True)

    # ---- portfolio (ι = 0 for all) ----------
    province_iota = portfolio[portfolio["region"].isin(province_names)].copy()
    country_iota_rows = [
        {"region": COUNTRY_DISPLAY[code], "value": 0.0}
        for code in new_country_codes
    ]
    full_iota = pd.concat([province_iota, pd.DataFrame(country_iota_rows)],
                          ignore_index=True).sort_values("region").reset_index(drop=True)

    # ---- employment ----------
    # Scale by gross output ratio assuming Canadian labor productivity per sector.
    province_emp = employment[employment["region"].isin(province_names)].copy()
    # Canadian total employment per sector (sum across provinces).
    can_emp_per_sector = province_emp.groupby("sector")["value"].sum()
    can_go_per_sector = {
        s: agg["gross_output"].get((CAN, s), 0.0) for s in SECTORS
    }
    country_emp_rows = []
    for code in new_country_codes:
        for s in SECTORS:
            country_go = agg["gross_output"].get((code, s), 0.0)
            can_go = can_go_per_sector[s]
            ratio = country_go / can_go if can_go > 0 else 0.0
            emp = float(can_emp_per_sector[s] * ratio)
            country_emp_rows.append({"sector": s, "region": COUNTRY_DISPLAY[code], "value": emp})
    full_emp = pd.concat([province_emp, pd.DataFrame(country_emp_rows)],
                         ignore_index=True).sort_values(["sector", "region"]).reset_index(drop=True)

    # ---- write ----------
    full_trade.to_parquet(out_dir / "bilateral_trade.parquet", index=False)
    full_gamma.to_parquet(out_dir / "value_added_share.parquet", index=False)
    full_alpha.to_parquet(out_dir / "final_demand_share.parquet", index=False)
    full_B.to_parquet(out_dir / "structures_share.parquet", index=False)
    full_iota.to_parquet(out_dir / "portfolio_share.parquet", index=False)
    full_emp.to_parquet(out_dir / "employment.parquet", index=False)

    # Region-invariant files: copy as-is.
    for fn in ("io_matrix.parquet", "sectoral_dispersion.parquet"):
        shutil.copyfile(src_dir / fn, out_dir / fn)

    # Diagnostics
    print(f"  regions in output: {sorted(set(full_trade['source']))}")
    print(f"  bilateral_trade rows: {len(full_trade):5d}  "
          f"(expected: 23 × 17² = {23*17*17})")
    print(f"  value_added_share rows: {len(full_gamma):5d}  "
          f"(expected: 23 × 17 = {23*17})")


# ---------------------------------------------------------------- main


def _compute_scale_factor(src_dir: Path, agg: dict) -> float:
    """Derive ICIO USD-millions → StatCan CAD-thousands by comparing CAN
    gross output in both sources."""
    trade = pd.read_parquet(src_dir / "bilateral_trade.parquet")
    can_go_cad = trade.loc[trade["source"] != ROW, "value"].sum()
    can_go_usd = sum(agg["gross_output"].get((CAN, s), 0.0) for s in SECTORS)
    return float(can_go_cad / can_go_usd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--icio", type=Path, required=True,
                        help="Path to OECD ICIO 2021 SML CSV")
    parser.add_argument("--src", type=Path,
                        default=Path("data/inputs/canada_2021/"))
    parser.add_argument("--out", type=Path,
                        default=Path("data/inputs/canada_2021_partners/"))
    args = parser.parse_args()

    print(f"Loading OECD ICIO from {args.icio}...")
    M, row_labels, col_labels, countries = load_icio(args.icio)
    print(f"  matrix: {M.shape}  countries present: {len(countries)}")

    print("\nComputing per-partner aggregates...")
    agg = compute_partner_aggregates(M, row_labels, col_labels)
    for code in (CAN, *PARTNERS, RESIDUAL_CODE):
        total = sum(agg["gross_output"].get((code, s), 0.0) for s in SECTORS)
        print(f"  {code:9s} total gross output (USD M): {total:>14,.0f}")

    scale = _compute_scale_factor(args.src, agg)
    print(f"\n  ICIO → StatCan units scale: {scale:.2f}")

    print(f"\nWriting expanded 17-region calibration to {args.out}...")
    expand_calibration(args.src, args.out, agg, scale)


if __name__ == "__main__":
    main()
