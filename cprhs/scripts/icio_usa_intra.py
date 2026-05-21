"""Extract USA × USA intra-trade from OECD ICIO at AKT 9-goods grain.

The AKT replication treats USA as a single region and needs π_USA,USA per
sector — i.e., USA's absorption of USA-produced goods. The OECD ICIO 2023
release covers years 1995-2022 in country × industry × country × industry
matrices; we use the 2015 vintage to match AKT's reference year.

Method:
1. Load the 2015 SML CSV (USD millions, ISIC Rev 4 industry).
2. Sum USA-origin × USA-destination cells over all destination categories
   (intermediate use + HFCE/NPISH/GGFC/GFCF/INVNT, excluding OUT). This is
   USA's total absorption of USA-produced output per ICIO industry.
3. Aggregate ICIO industries to AKT's 9 goods sectors using the concordance
   that mirrors AKT's S&U product-code mapping (NAICS 324-327 → Petroleum &
   Chemicals; NAICS 331-332 → Metals; NAICS 333-336 → Machinery & Equipment;
   NAICS 337+339 → Other Manufacturing; etc.).
4. Convert from USD millions (ICIO native) to CAD millions using the 2015
   average exchange rate (1.279 CAD/USD).

Prep (once): extract /tmp/icio/2015_SML.csv from ~/Downloads/2011-2015_SML.zip
    mkdir -p /tmp/icio && unzip -o ~/Downloads/2011-2015_SML.zip 2015_SML.csv -d /tmp/icio/

Standalone run from cprhs/:
    uv run python scripts/icio_usa_intra.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from add_icio_row import _split, load_icio  # noqa: E402

# 2015 USD→CAD average. https://www.bankofcanada.ca exchange-rate annual.
USD_TO_CAD_2015 = 1.279

# OECD ICIO 2023 release uses ISIC Rev 4 industry codes. The mapping below
# concords each ICIO goods industry to one of AKT-2019's 9 goods sectors.
# AKT's mapping (via S&U → NAICS) is documented in scripts/akt_replication.py:
#   AKT Agriculture            ⊃ NAICS 11  (ISIC A)
#   AKT Mining                 ⊃ NAICS 21  (ISIC B)
#   AKT Food                   ⊃ NAICS 311-312 (ISIC C10-12)
#   AKT Textile                ⊃ NAICS 313-316 (ISIC C13-15)
#   AKT Wood and Paper         ⊃ NAICS 321-323 (ISIC C16-18)
#   AKT Petroleum & Chemicals  ⊃ NAICS 324-327 (ISIC C19-23) — includes
#                                non-metallic minerals 327
#   AKT Metals                 ⊃ NAICS 331-332 (ISIC C24-25)
#   AKT Machinery & Equipment  ⊃ NAICS 333-336+3363 (ISIC C26-30)
#   AKT Other Manufacturing    ⊃ NAICS 337+339 (ISIC C31-33)
#
# Services (D, E, F, G, ...) have no AKT-goods equivalent and are dropped.
ICIO_TO_AKT9: dict[str, str] = {
    "A01":      "Agriculture",
    "A02":      "Agriculture",
    "A03":      "Agriculture",
    "B05":      "Mining",
    "B06":      "Mining",
    "B07":      "Mining",
    "B08":      "Mining",
    "B09":      "Mining",
    "C10T12":   "Food",
    "C13T15":   "Textile",
    "C16":      "Wood and Paper",
    "C17_18":   "Wood and Paper",
    "C19":      "Petroleum and Chemicals",
    "C20":      "Petroleum and Chemicals",
    "C21":      "Petroleum and Chemicals",
    "C22":      "Petroleum and Chemicals",
    "C23":      "Petroleum and Chemicals",
    "C24A":     "Metals",
    "C24B":     "Metals",
    "C25":      "Metals",
    "C26":      "Machinery and Equipment",
    "C27":      "Machinery and Equipment",
    "C28":      "Machinery and Equipment",
    "C29":      "Machinery and Equipment",
    "C301":     "Machinery and Equipment",
    "C302T309": "Machinery and Equipment",
    "C31T33":   "Other Manufacturing",
}


def load_usa_intra_per_icio_industry(icio_csv: Path) -> dict[str, float]:
    """USA × USA absorption per ICIO industry (USD millions, native units).

    'Absorption' = sum of USA-origin rows × USA-destination cols across all
    destination categories (intermediate industry use + final-demand cols
    HFCE, NPISH, GGFC, GFCF, INVNT, DPABR), excluding the OUT (gross output)
    column. Returns only goods industries listed in ICIO_TO_AKT9; service
    industries are dropped.
    """
    M, row_labels, col_labels, _ = load_icio(icio_csv)
    Mn = M.to_numpy()
    usa_dest_cols = [i for i, lbl in enumerate(col_labels)
                     if lbl != "OUT" and _split(lbl)[0] == "USA"]
    out: dict[str, float] = {}
    for i, lbl in enumerate(row_labels):
        country, industry = _split(lbl)
        if country != "USA" or industry not in ICIO_TO_AKT9:
            continue
        out[industry] = out.get(industry, 0.0) + float(Mn[i, usa_dest_cols].sum())
    return out


def aggregate_to_akt9(icio_dict: dict[str, float],
                      usd_to_cad: float = USD_TO_CAD_2015) -> dict[str, float]:
    """Aggregate ICIO industries → AKT 9 goods; convert USD → CAD millions."""
    out: dict[str, float] = {}
    for industry, value in icio_dict.items():
        akt = ICIO_TO_AKT9.get(industry)
        if akt is None:
            continue
        out[akt] = out.get(akt, 0.0) + value * usd_to_cad
    return out


def load_usa_intra_per_akt_sector(
    icio_csv: Path = Path("/tmp/icio/2015_SML.csv"),
    usd_to_cad: float = USD_TO_CAD_2015,
) -> dict[str, float]:
    """USA × USA intra-trade per AKT 9-goods sector, in CAD millions."""
    return aggregate_to_akt9(load_usa_intra_per_icio_industry(icio_csv), usd_to_cad)


if __name__ == "__main__":
    csv = Path("/tmp/icio/2015_SML.csv")
    if not csv.exists():
        raise FileNotFoundError(
            f"missing {csv}; first extract via:\n"
            f"  mkdir -p /tmp/icio && unzip -o ~/Downloads/2011-2015_SML.zip "
            f"2015_SML.csv -d /tmp/icio/"
        )

    icio_dict = load_usa_intra_per_icio_industry(csv)
    akt_dict = aggregate_to_akt9(icio_dict)

    print(f"USA × USA intra-trade per ICIO industry (USD millions, 2015):\n")
    for industry in sorted(icio_dict):
        print(f"  {industry:10s}  ${icio_dict[industry]:>14,.0f}  →  {ICIO_TO_AKT9[industry]}")

    print(f"\nAggregated to AKT 9 goods (CAD millions, 2015 @ USD/CAD = {USD_TO_CAD_2015}):\n")
    for akt in sorted(akt_dict):
        print(f"  {akt:30s}  C${akt_dict[akt]:>15,.0f}")
    print(f"\n  TOTAL goods (CAD millions):     C${sum(akt_dict.values()):>15,.0f}")
