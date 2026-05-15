"""Replace synthetic ROW data with real OECD ICIO aggregates.

Updates the ROW (Rest of World) column of the canada_YYYY/ parquet files
using OECD ICIO. The current ROW values are synthesized (gross output set
to 50× Canada per sector, γ/α copied from the Canadian average); this
script replaces them with real values aggregated over the ~80
non-Canadian ICIO economies.

What gets replaced (ROW slice only — Canadian data is untouched):

* ``bilateral_trade.parquet`` row where source = destination = ROW. The
  ROW→ROW value per sector becomes the actual non-Canadian world intra-
  trade rather than the synthetic residual.
* ``value_added_share.parquet`` rows where region = ROW. γ per sector
  becomes the labor-and-capital share of ICIO non-Canadian gross output.
* ``final_demand_share.parquet`` rows where region = ROW. α per sector
  becomes the share of non-Canadian household + NPISH + government
  consumption falling on each model sector.

Left as-is (with documented limitations — see DATA.md):

* ``structures_share.parquet`` region = ROW. ICIO's VA row doesn't
  decompose into wages / mixed income / operating surplus; computing B
  requires external income-account data.
* ``employment.parquet`` region = ROW. ICIO carries no employment data;
  ILO Modelled Estimates would be the right replacement.
* ``portfolio_share``, ``sectoral_dispersion``, ``io_matrix`` have no
  ROW-specific entries (single value per region, single value per
  sector, single (J,J) shared across regions).

Usage::

    # File path is to one of the OECD ICIO 2023 release "SML" CSVs.
    uv run python scripts/add_icio_row.py \\
        --icio "$HOME/Downloads/2016-2022_SML/2021_SML.csv" \\
        --calibration data/inputs/canada_2021/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add convert_statcan helpers via the build_canada_iot module's path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_canada_iot import ROW, SECTORS  # noqa: E402

CAN = "CAN"  # OECD ICIO country code for Canada


# ---------------------------------------------------------------- ISIC mapping

# OECD ICIO uses ISIC Rev 4 industry codes. The mapping below collapses the
# 50 ICIO origin industries onto the 23 model sectors. Mirrors the L97 → 23
# mapping in build_canada_iot._l97_to_sector at the NAICS-3 level.
ISIC_TO_SECTOR: dict[str, str] = {
    "A01":      "Agriculture, Forestry, Fishing",
    "A02":      "Agriculture, Forestry, Fishing",
    "A03":      "Agriculture, Forestry, Fishing",
    "B05":      "Mining and Extraction",
    "B06":      "Mining and Extraction",
    "B07":      "Mining and Extraction",
    "B08":      "Mining and Extraction",
    "B09":      "Mining and Extraction",
    "C10T12":   "Food, Beverage, Tobacco",
    "C13T15":   "Textile, Apparel, Leather",
    "C16":      "Wood, Paper, Printing",
    "C17_18":   "Wood, Paper, Printing",
    "C19":      "Petroleum and Chemicals",
    "C20":      "Petroleum and Chemicals",
    "C21":      "Petroleum and Chemicals",
    "C22":      "Petroleum and Chemicals",          # NAICS 326 (plastics/rubber)
    "C23":      "Non-metallic Mineral Products",
    "C24A":     "Metals and Machinery",
    "C24B":     "Metals and Machinery",
    "C25":      "Metals and Machinery",
    "C26":      "Computers, Electronics, Electrical",
    "C27":      "Computers, Electronics, Electrical",
    "C28":      "Metals and Machinery",
    "C29":      "Transportation Equipment",
    "C301":     "Transportation Equipment",
    "C302T309": "Transportation Equipment",
    "C31T33":   "Furniture and Other Manufacturing",
    "D":        "Utilities",
    "E":        "Utilities",
    "F":        "Construction",
    "G":        "Wholesale and Retail Trade",
    "H49":      "Transportation Services",
    "H50":      "Transportation Services",
    "H51":      "Transportation Services",
    "H52":      "Transportation Services",
    "H53":      "Transportation Services",
    "I":        "Arts, Recreation, Accommodation, Food",
    "J58T60":   "Information and Communication",
    "J61":      "Information and Communication",
    "J62_63":   "Information and Communication",
    "K":        "Finance and Insurance",
    "L":        "Real Estate, Rental, Leasing",
    "M":        "Professional and Administrative Services",
    "N":        "Professional and Administrative Services",
    "O":        "Public Administration and Other Services",
    "P":        "Education",
    "Q":        "Health",
    "R":        "Arts, Recreation, Accommodation, Food",
    "S":        "Public Administration and Other Services",
    "T":        "Public Administration and Other Services",
}

# ICIO column categories that are final consumption (Cobb-Douglas α). Excludes
# capital formation (GFCF), inventory (INVNT), and tourism (DPABR — direct
# purchases abroad). HFCE = household, NPISH = non-profits, GGFC = government
# final consumption.
FINAL_CONSUMPTION = ("HFCE", "NPISH", "GGFC")


# ---------------------------------------------------------------- loading


def load_icio(path: Path) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """Read one year of OECD ICIO SML CSV.

    Returns (M, row_labels, col_labels, countries). ``M`` is the full
    numeric matrix (rows × cols) as a NumPy-backed DataFrame with
    string row/col labels. ``row_labels`` carry items like ``CAN_C10T12``,
    ``USA_VA``, ``OUT``. ``col_labels`` add final-demand suffixes
    (``CAN_HFCE``, ``USA_GFCF``) and ``OUT``.
    """
    df = pd.read_csv(path)
    row_labels = df["V1"].tolist()
    col_labels = list(df.columns[1:])
    M = df.drop(columns=["V1"]).to_numpy(dtype=np.float64)
    countries = sorted({c.split("_", 1)[0] for c in col_labels if "_" in c})
    return pd.DataFrame(M, index=row_labels, columns=col_labels), row_labels, col_labels, countries


# ---------------------------------------------------------------- aggregation


def _split(label: str) -> tuple[str, str]:
    if "_" in label:
        c, i = label.split("_", 1)
        return c, i
    return "", label


def _rows_for(labels: list[str], origin_filter, suffix_filter=None) -> np.ndarray:
    """Boolean mask of row positions whose split parts match the filters.

    ``origin_filter`` and ``suffix_filter`` are predicates on the
    country-prefix and industry/category-suffix respectively. If
    ``suffix_filter`` is None we accept anything that has a ``_``
    separator (i.e., country × industry rows, not totals).
    """
    mask = np.zeros(len(labels), dtype=bool)
    for i, lbl in enumerate(labels):
        c, suf = _split(lbl)
        if not c:
            continue
        if origin_filter is not None and not origin_filter(c):
            continue
        if suffix_filter is not None and not suffix_filter(suf):
            continue
        mask[i] = True
    return mask


def _cols_for(labels: list[str], dest_filter, suffix_filter=None) -> np.ndarray:
    return _rows_for(labels, dest_filter, suffix_filter)


# ---------------------------------------------------------------- builders


def compute_row_aggregates(
    M: pd.DataFrame,
    row_labels: list[str],
    col_labels: list[str],
) -> dict:
    """Compute the per-sector ROW aggregates we need.

    All values are in OECD ICIO native units (USD millions). Conversion to
    CAD is applied later in ``apply_row_overrides`` so this function stays
    a pure aggregator.
    """
    Mn = M.to_numpy()

    # Sector buckets at the row side (origin) and col side (destination).
    sectors = list(SECTORS)
    sector_to_isic: dict[str, list[str]] = {s: [] for s in sectors}
    for isic, sec in ISIC_TO_SECTOR.items():
        sector_to_isic[sec].append(isic)

    out: dict = {sector: {} for sector in sectors}

    # Helpers: build row / col index sets per sector
    def row_idx(country_pred, isics: list[str]) -> np.ndarray:
        return np.array([
            i for i, lbl in enumerate(row_labels)
            if (c := _split(lbl))[0]
            and country_pred(c[0]) and c[1] in isics
        ], dtype=np.int64)

    def col_idx(country_pred, suf_pred) -> np.ndarray:
        return np.array([
            i for i, lbl in enumerate(col_labels)
            if (c := _split(lbl))[0]
            and country_pred(c[0]) and suf_pred(c[1])
        ], dtype=np.int64)

    not_can = lambda c: c != CAN  # noqa: E731
    is_can = lambda c: c == CAN  # noqa: E731
    any_country = lambda c: True  # noqa: E731

    out_col_pos = col_labels.index("OUT")
    va_row_pos = row_labels.index("VA")

    # All destination columns that count as "domestic use" (not OUT).
    use_cols = np.array(
        [i for i, lbl in enumerate(col_labels) if lbl != "OUT"],
        dtype=np.int64,
    )

    # All destination columns belonging to non-CAN countries (any suffix).
    non_can_dest = col_idx(not_can, lambda s: True)
    can_dest = col_idx(is_can, lambda s: True)
    non_can_consumption_dest = col_idx(not_can, lambda s: s in FINAL_CONSUMPTION)

    # CAN totals (for the USD→CAD scale).
    can_industry_row_isics = sum(sector_to_isic.values(), [])  # all industries
    can_origin_rows = row_idx(is_can, can_industry_row_isics)
    can_total_gross_output_usd = Mn[can_origin_rows, out_col_pos].sum()

    # Compute per-sector aggregates.
    for sector, isics in sector_to_isic.items():
        # Origin rows = non-CAN countries × ISIC industries in this sector
        non_can_origin = row_idx(not_can, isics)
        if len(non_can_origin) == 0:
            out[sector] = {
                "row_gross_output": 0.0,
                "row_intra_trade": 0.0,
                "row_to_can": 0.0,
                "row_va": 0.0,
                "row_final_consumption": 0.0,
            }
            continue

        # Gross output of non-CAN sector j industries (= sum of OUT col for these rows)
        gross_output = Mn[non_can_origin, out_col_pos].sum()

        # Total absorption by non-CAN destinations (intermediate + final demand cols).
        # This is what "ROW sells to ROW" — the off-diagonal absorption.
        intra_trade = Mn[np.ix_(non_can_origin, non_can_dest)].sum()

        # Exports to Canada from non-CAN sector-j origin.
        to_can = Mn[np.ix_(non_can_origin, can_dest)].sum()

        # Value added in non-CAN sector j (= VA row × industry-col positions
        # for non-CAN countries × ISIC in this sector).
        non_can_industry_cols = col_idx(not_can, lambda s, isics=isics: s in isics)
        va_in_sector = Mn[va_row_pos, non_can_industry_cols].sum()

        # Non-CAN final consumption on this sector (rows of any origin
        # × destinations among non-CAN HFCE/NPISH/GGFC cols, narrowed to
        # this sector by the row side — but α is destination-side
        # consumption-on-sector-j, so we narrow rows to sector j across
        # ALL origins).
        any_origin_in_sector = row_idx(any_country, isics)
        final_consumption = Mn[
            np.ix_(any_origin_in_sector, non_can_consumption_dest)
        ].sum()

        out[sector] = {
            "row_gross_output": float(gross_output),
            "row_intra_trade": float(intra_trade),
            "row_to_can": float(to_can),
            "row_va": float(va_in_sector),
            "row_final_consumption": float(final_consumption),
        }

    out["_meta"] = {
        "can_gross_output_usd": float(can_total_gross_output_usd),
    }
    return out


# ---------------------------------------------------------------- overrides


def apply_row_overrides(calibration_dir: Path, agg: dict) -> None:
    """Update the ROW slice of the three relevant parquet files.

    Uses the calibration's existing Canadian gross output as the yardstick:
    we compute a scale factor by comparing ICIO's Canadian gross output
    (USD millions) to the StatCan-based Canadian gross output already
    encoded in the bilateral_trade matrix (CAD thousands — StatCan IOT
    files report numbers in thousands of CAD despite their header label).
    The scale factor (~1250) bakes in both the unit difference (1000×)
    and the USD→CAD exchange rate (~1.25×) so ICIO ROW values are written
    in the same units as the Canadian data already on disk.
    """
    trade_path = calibration_dir / "bilateral_trade.parquet"
    gamma_path = calibration_dir / "value_added_share.parquet"
    alpha_path = calibration_dir / "final_demand_share.parquet"

    trade = pd.read_parquet(trade_path)
    gamma = pd.read_parquet(gamma_path)
    alpha = pd.read_parquet(alpha_path)

    # USD → CAD scale: ratio of StatCan Canadian gross output (sum of all
    # Canadian-origin trade across all destinations) to ICIO Canadian
    # gross output. Both are total-economy figures.
    can_provinces_mask = trade["source"].apply(
        lambda s: s not in (ROW,)
    )
    canada_go_cad = trade.loc[can_provinces_mask, "value"].sum()
    canada_go_usd = agg["_meta"]["can_gross_output_usd"]
    scale = canada_go_cad / canada_go_usd
    print(f"  ICIO→StatCan units scale: {scale:.2f}  "
          f"(StatCan CAN GO {canada_go_cad/1e9:.2f} B units / "
          f"ICIO CAN GO {canada_go_usd/1e6:.2f} T USD; "
          f"factor bakes in CAD-thousands vs USD-millions × exchange rate)")

    # ROW → ROW per sector in bilateral_trade.
    print("\n  Replacing ROW→ROW values per sector (millions CAD):")
    for sector in SECTORS:
        new_value = max(agg[sector]["row_intra_trade"] * scale, 1.0)
        mask = (
            (trade["sector"] == sector)
            & (trade["source"] == ROW)
            & (trade["destination"] == ROW)
        )
        old = float(trade.loc[mask, "value"].iloc[0])
        trade.loc[mask, "value"] = new_value
        print(f"    {sector:50s} {old:>14,.0f} -> {new_value:>14,.0f}")

    # γ for ROW per sector.
    print("\n  Replacing ROW γ per sector (VA / gross output, non-CAN ICIO):")
    for sector in SECTORS:
        go = agg[sector]["row_gross_output"]
        va = agg[sector]["row_va"]
        new_gamma = max(min(va / go, 0.99), 0.01) if go > 0 else 0.5
        mask = (gamma["sector"] == sector) & (gamma["region"] == ROW)
        old = float(gamma.loc[mask, "value"].iloc[0])
        gamma.loc[mask, "value"] = new_gamma
        print(f"    {sector:50s} {old:.3f} -> {new_gamma:.3f}")

    # α for ROW per sector. Normalize to sum to 1.
    print("\n  Replacing ROW α per sector (final consumption share, non-CAN ICIO):")
    fc_per_sector = {s: agg[s]["row_final_consumption"] for s in SECTORS}
    fc_total = sum(fc_per_sector.values())
    for sector in SECTORS:
        new_alpha = fc_per_sector[sector] / fc_total if fc_total > 0 else 0.0
        mask = (alpha["sector"] == sector) & (alpha["region"] == ROW)
        old = float(alpha.loc[mask, "value"].iloc[0])
        alpha.loc[mask, "value"] = new_alpha
        print(f"    {sector:50s} {old:.3f} -> {new_alpha:.3f}")

    trade.to_parquet(trade_path, index=False)
    gamma.to_parquet(gamma_path, index=False)
    alpha.to_parquet(alpha_path, index=False)
    print(f"\n  wrote {trade_path}, {gamma_path}, {alpha_path}")


# ---------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--icio", type=Path, required=True,
                        help="Path to OECD ICIO 2021 SML CSV")
    parser.add_argument("--calibration", type=Path,
                        default=Path("data/inputs/canada_2021/"),
                        help="Calibration directory whose ROW slice to update")
    args = parser.parse_args()

    print(f"Loading OECD ICIO from {args.icio}...")
    M, row_labels, col_labels, countries = load_icio(args.icio)
    print(f"  matrix shape: {M.shape}")
    print(f"  countries:    {len(countries)}")
    print(f"  CAN present:  {CAN in countries}")

    print("\nComputing non-Canadian aggregates per model sector...")
    agg = compute_row_aggregates(M, row_labels, col_labels)
    print(f"  ROW total gross output (USD M): "
          f"{sum(agg[s]['row_gross_output'] for s in SECTORS):,.0f}")
    print(f"  CAN gross output (USD M):      "
          f"{agg['_meta']['can_gross_output_usd']:,.0f}")

    print(f"\nApplying ROW overrides to {args.calibration}...")
    apply_row_overrides(args.calibration, agg)


if __name__ == "__main__":
    main()
