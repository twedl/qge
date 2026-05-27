"""M-ish sector schema for the canada_2022_atlm calibration.

30 sectors (20 goods + 10 services) plus an 8-region split that merges the
four Atlantic provinces into a single ATL region. Single source of truth for
all downstream build scripts.

Sources for the goods split: StatCan 12-10-0088-01 M-code product codes
(Supply & Use product taxonomy). Services are intentionally bundled.
"""

from __future__ import annotations


SECTORS: tuple[str, ...] = (
    # ── Goods (20) ───────────────────────────────────────────────────────
    "Crops & animals",
    "Forestry & fishing",
    "Oil and Gas",
    "Mining ex-O&G",
    "Food & beverage",
    "Textiles & apparel",
    "Wood",
    "Paper & printing",
    "Petroleum & coal",
    "Chemicals",
    "Plastics & rubber",
    "Non-metallic minerals",
    "Primary metals",
    "Fabricated metals",
    "Machinery",
    "Electronics & electrical",
    "Motor vehicles & parts",
    "Other transport equipment",
    "Other manufacturing",
    # ── Services (10) ────────────────────────────────────────────────────
    "Utilities",
    "Construction",
    "Wholesale & Retail",
    "Transportation",
    "Information & Communications",
    "Finance",
    "Real Estate",
    "Professional & Business Services",
    "Accommodation & Food",
    "Public & Social",
)


# ── StatCan 12-10-0088 M-codes → 30 sectors ────────────────────────────────
MCODE_TO_SECTOR: dict[str, str] = {
    # Crops & animals
    "M111B": "Crops & animals",
    "M112A": "Crops & animals",
    # Forestry & fishing (incl. support for ag/forestry)
    "M11D0": "Forestry & fishing",
    "M11E0": "Forestry & fishing",
    "M1140": "Forestry & fishing",
    "M1150": "Forestry & fishing",
    # Oil and Gas (M21A0 = oil&gas extraction, M21B0 = oil sands)
    "M21A0": "Oil and Gas",
    "M21B0": "Oil and Gas",
    # Mining ex-O&G (M2122 = metals, M2123 = non-metals, M2130 = support)
    "M2122": "Mining ex-O&G",
    "M2123": "Mining ex-O&G",
    "M2130": "Mining ex-O&G",
    # Food & beverage
    "M31C0": "Food & beverage",
    "M312A": "Food & beverage",
    # Textiles & apparel
    "M31D0": "Textiles & apparel",
    # Wood
    "M3210": "Wood",
    # Paper & printing (ICIO C17_18 bundles them)
    "M3220": "Paper & printing",
    "M3230": "Paper & printing",
    # Petroleum & coal
    "M3240": "Petroleum & coal",
    # Chemicals
    "M3250": "Chemicals",
    # Plastics & rubber
    "M3260": "Plastics & rubber",
    # Non-metallic minerals
    "M3270": "Non-metallic minerals",
    # Primary metals
    "M3310": "Primary metals",
    # Fabricated metals
    "M3320": "Fabricated metals",
    # Machinery
    "M3330": "Machinery",
    # Electronics & electrical (M334C = computer/electronic, M3350 = electrical)
    "M334C": "Electronics & electrical",
    "M3350": "Electronics & electrical",
    # Motor vehicles & parts
    "M3363": "Motor vehicles & parts",
    # Other transport equipment (M336A = aerospace, rail, ship)
    "M336A": "Other transport equipment",
    # Other manufacturing (furniture + misc)
    "M3370": "Other manufacturing",
    "M3B00": "Other manufacturing",
    # ── Services ──
    "M2200": "Utilities",
    "M23A0": "Construction",
    "M23B0": "Construction",
    "M23C0": "Construction",
    "M23D0": "Construction",
    "M4100": "Wholesale & Retail",
    "M4A00": "Wholesale & Retail",
    "M4B00": "Transportation",
    "M5170": "Information & Communications",
    "M51D0": "Information & Communications",
    "M51E0": "Information & Communications",
    "M5E00": "Information & Communications",
    "M52C0": "Finance",
    "M5F00": "Finance",
    "M53C0": "Real Estate",
    "M53D0": "Real Estate",
    "M5417": "Professional & Business Services",
    "M541E": "Professional & Business Services",
    "M5G00": "Professional & Business Services",
    "M7100": "Accommodation & Food",
    "M7200": "Accommodation & Food",
    "M6100": "Public & Social",
    "M6200": "Public & Social",
    "M9B00": "Public & Social",
    "M8100": "Public & Social",
    "M9A00": "Public & Social",
}


# ── Region grouping: 10 provinces + WLD → 7 provinces + WLD ────────────────
# Atlantic = {NB, NS, PE, NL}. All other provinces preserved.
PROV_GROUP: dict[str, str] = {
    "AB": "AB", "BC": "BC", "MB": "MB", "ON": "ON", "QC": "QC", "SK": "SK",
    "NB": "ATL", "NS": "ATL", "PE": "ATL", "NL": "ATL",
}
REGIONS: tuple[str, ...] = ("AB", "BC", "MB", "ON", "QC", "SK", "ATL", "WLD")
WLD = "WLD"

PROVINCES_FULL: dict[str, str] = {
    "Alberta": "AB", "British Columbia": "BC", "Manitoba": "MB",
    "Ontario": "ON", "Quebec": "QC", "Saskatchewan": "SK",
    "New Brunswick": "NB", "Nova Scotia": "NS",
    "Prince Edward Island": "PE", "Newfoundland and Labrador": "NL",
}


# ── L97/Detail NAICS → 30 sectors (used by industrial-structure builder) ──
def naics_to_sector(code: str) -> str | None:
    """Map a StatCan L97/Detail NAICS-prefixed industry code to a sector.

    Codes have prefixes BS (business), NP (non-profit), GS (gov't service).
    Body is the NAICS-2017 (or 2022) 3-6 digit code.
    """
    if not isinstance(code, str) or len(code) < 3:
        return None
    p2, body = code[:2], code[2:]

    if p2 == "GS":
        if body.startswith("61"): return "Public & Social"
        if body.startswith("62"): return "Public & Social"
        if body.startswith("91"): return "Public & Social"
        return None
    if p2 == "NP":
        if body.startswith("61"): return "Public & Social"
        if body.startswith(("622", "623", "624")): return "Public & Social"
        return "Public & Social"
    if p2 != "BS":
        return None

    b2, b3, b4 = body[:2], body[:3], body[:4]

    # Agriculture, Forestry, Fishing, Hunting (NAICS 11)
    if b3 == "111":
        return "Crops & animals"
    if b3 == "112":
        return "Crops & animals"
    if b3 in ("113", "114"):
        return "Forestry & fishing"
    if b3 == "115":
        return "Forestry & fishing"
    if b2 == "11":
        return "Crops & animals"   # generic 11A/11B fallback (combined)

    # Mining (NAICS 21)
    if b3 == "211":
        return "Oil and Gas"
    if b3 == "212":
        return "Mining ex-O&G"
    if b3 == "213":
        return "Mining ex-O&G"
    if b2 == "21":
        return "Mining ex-O&G"     # generic 21 fallback

    # Utilities (NAICS 22) / Construction (NAICS 23)
    if b2 == "22":
        return "Utilities"
    if b2 == "23":
        return "Construction"

    # Manufacturing (NAICS 31-33)
    if b3 in ("311", "312"):
        return "Food & beverage"
    if b3 in ("313", "314", "315", "316", "31A", "31B"):
        return "Textiles & apparel"
    if b3 == "321":
        return "Wood"
    if b3 in ("322", "323"):
        return "Paper & printing"
    if b3 == "324":
        return "Petroleum & coal"
    if b3 == "325":
        return "Chemicals"
    if b3 == "326":
        return "Plastics & rubber"
    if b3 == "327":
        return "Non-metallic minerals"
    if b3 == "331":
        return "Primary metals"
    if b3 == "332":
        return "Fabricated metals"
    if b3 == "333":
        return "Machinery"
    if b3 in ("334", "335"):
        return "Electronics & electrical"
    if b3 == "336":
        if b4 in ("3361", "3362", "3363"):
            return "Motor vehicles & parts"
        return "Other transport equipment"
    if b3 in ("337", "339"):
        return "Other manufacturing"

    # Wholesale & Retail (NAICS 41, 44, 45, 4A combined)
    if b2 in ("41", "44", "45", "4A"):
        return "Wholesale & Retail"

    # Transportation (NAICS 48, 49)
    if b2 in ("48", "49"):
        return "Transportation"

    # Information (NAICS 51)
    if b2 == "51":
        # All of Information sector maps to Information & Communications.
        # In NAICS 2022 includes publishing 513, broadcasting 516, telecom 517,
        # data processing 518, web search 519, etc.
        return "Information & Communications"

    # Finance (NAICS 52) / Real Estate (NAICS 53)
    if b2 == "52":
        return "Finance"
    if b2 == "53":
        return "Real Estate"

    # Professional services (NAICS 54-56)
    if b2 in ("54", "55", "56"):
        return "Professional & Business Services"

    # Education (NAICS 61) / Health (NAICS 62)
    if b2 == "61":
        return "Public & Social"
    if b2 == "62":
        return "Public & Social"

    # Arts, Accommodation, Other (NAICS 71-72-81)
    if b2 == "71":
        return "Accommodation & Food"   # arts often lumped with accommodation
    if b2 == "72":
        return "Accommodation & Food"
    if b2 == "81":
        return "Public & Social"

    # Government (NAICS 91)
    if b2 == "91":
        return "Public & Social"

    return None


# ── ICIO ISIC Rev 4 → 30 sectors (used for WLD aggregate) ─────────────────
ICIO_TO_SECTOR: dict[str, str] = {
    "A01": "Crops & animals",
    "A02": "Forestry & fishing",
    "A03": "Forestry & fishing",
    "B05": "Mining ex-O&G",           # coal
    "B06": "Oil and Gas",
    "B07": "Mining ex-O&G",
    "B08": "Mining ex-O&G",
    "B09": "Mining ex-O&G",
    "C10T12": "Food & beverage",
    "C13T15": "Textiles & apparel",
    "C16": "Wood",
    "C17_18": "Paper & printing",
    "J58T60": "Information & Communications",
    "C19": "Petroleum & coal",
    "C20": "Chemicals",
    "C21": "Chemicals",               # pharma → chemicals
    "C22": "Plastics & rubber",
    "C23": "Non-metallic minerals",
    "C24A": "Primary metals",
    "C24B": "Primary metals",
    "C25": "Fabricated metals",
    "C26": "Electronics & electrical",
    "C27": "Electronics & electrical",
    "C28": "Machinery",
    "C29": "Motor vehicles & parts",
    "C301": "Other transport equipment",
    "C302T309": "Other transport equipment",
    "C31T33": "Other manufacturing",
    "D": "Utilities",
    "E": "Utilities",
    "F": "Construction",
    "G": "Wholesale & Retail",
    "H49": "Transportation",
    "H50": "Transportation",
    "H51": "Transportation",
    "H52": "Transportation",
    "H53": "Information & Communications",
    "I": "Accommodation & Food",
    "J61": "Information & Communications",
    "J62_63": "Information & Communications",
    "K": "Finance",
    "L": "Real Estate",
    "M": "Professional & Business Services",
    "N": "Professional & Business Services",
    "O": "Public & Social",
    "P": "Public & Social",
    "Q": "Public & Social",
    "R": "Accommodation & Food",
    "S": "Public & Social",
    "T": "Public & Social",
}


# ── Trade elasticities (θ) per sector ─────────────────────────────────────
# Goods: Caliendo-Parro 2015 99% sample values (cprhs/scripts/akt_replication.py
# _CP_99). Services: CP-2015 has no estimates; we use 5.0 throughout (TW 2021
# canada_2010 convention). The three CP sectors with sign-flip (Basic_metals,
# Machinery_nec, Auto) are replaced with the manufacturing mean (4.5).
THETA: dict[str, float] = {
    "Crops & animals":              9.11,
    "Forestry & fishing":           9.11,
    "Oil and Gas":                 13.53,
    "Mining ex-O&G":               13.53,
    "Food & beverage":              2.62,
    "Textiles & apparel":           8.10,
    "Wood":                        11.50,
    "Paper & printing":            16.52,
    "Petroleum & coal":            64.85,
    "Chemicals":                    3.13,
    "Plastics & rubber":            1.67,
    "Non-metallic minerals":        2.41,
    "Primary metals":               4.50,
    "Fabricated metals":            6.99,
    "Machinery":                    4.50,
    "Electronics & electrical":   12.93,
    "Motor vehicles & parts":       4.50,
    # Other transport equipment: CP-2015 99% raw value is 0.39 (poorly
    # identified — small N, near-zero CES elasticity → Head-Ries blowup).
    # We apply CP's own substitution rule (Basic_metals/Machinery_nec/Auto
    # all replaced with the manufacturing aggregate 4.49) to this sector too.
    "Other transport equipment":    4.49,
    "Other manufacturing":          3.98,
    "Utilities":                    5.00,
    "Construction":                 5.00,
    "Wholesale & Retail":           5.00,
    "Transportation":               5.00,
    "Information & Communications": 5.00,
    "Finance":                      5.00,
    "Real Estate":                  5.00,
    "Professional & Business Services": 5.00,
    "Accommodation & Food":         5.00,
    "Public & Social":              5.00,
}


def _check_consistency() -> None:
    """Sanity: every sector in MCODE/ICIO maps maps to a SECTORS entry."""
    sector_set = set(SECTORS)
    for code, sec in MCODE_TO_SECTOR.items():
        assert sec in sector_set, f"MCODE {code} -> unknown sector {sec!r}"
    for code, sec in ICIO_TO_SECTOR.items():
        assert sec in sector_set, f"ICIO {code} -> unknown sector {sec!r}"
    assert set(THETA.keys()) == sector_set, (
        f"THETA keys ≠ SECTORS: missing {sector_set - set(THETA.keys())}, "
        f"extra {set(THETA.keys()) - sector_set}"
    )


_check_consistency()
