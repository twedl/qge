"""Diagnose Head-Ries positivity at the 30-sector + ATL-merged schema.

Builds bilateral trade flows directly from the 10 provincial SUT workbooks
(StatCan 15-211-X Detail level), aggregates products → 30 sectors and provinces
→ 7+WLD, and counts:
  - number of (n, j) where π_{n,n,j} = 0 (diagonal failures)
  - number of (n, i, j) where π_{n,i,j} = 0 (off-diagonal failures)
  - resulting Head-Ries (n, i, j) triples with at least one zero in
    {π_{n,n}, π_{i,i}, π_{n,i}, π_{i,n}} — these are τ̄-unrecoverable cells.

Compares against the 22-sector + 11-region canada_2022 calibration as
baseline.

Run from tw/:
    uv run python scripts/diagnose_positivity_atlm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from scripts.sector_schema_m import (  # noqa: E402
    SECTORS, REGIONS, PROV_GROUP, naics_to_sector, THETA, WLD,
)


SUT_DIR = Path("/tmp/statcan_iot")
ICIO_CSV = Path("/tmp/icio/2022_SML.csv")
PROVINCES = ("AB", "BC", "MB", "NB", "NL", "NS", "ON", "PE", "QC", "SK")
TERRITORIES = ("NT", "NU", "YT", "CE")
PROV_AND_TERR = PROVINCES + TERRITORIES

# SUT values are in thousands of CAD; ICIO is in millions of USD.
# Multiplier: USD-millions → CAD-thousands = 1.30 (USD/CAD 2022) × 1000.
ICIO_TO_SUT_UNITS = 1.30 * 1000.0
CAN = "CAN"


def _sut_path(prov: str, year: int = 2022) -> Path:
    return SUT_DIR / f"IOT provincial {prov} D {year}.xlsx"


def _industry_prefixes() -> tuple[str, ...]:
    return ("BS", "NP", "GS")


def _is_industry_code(code) -> bool:
    if not isinstance(code, str):
        return False
    return code[:2] in _industry_prefixes()


def _is_product_row_code(code) -> bool:
    """Rows that represent products (industries + nothing else for diagonal)."""
    if not isinstance(code, str):
        return False
    return code[:2] in _industry_prefixes()


def _fd_prefixes() -> tuple[str, ...]:
    return ("PEC", "CEN", "CEG", "COB", "COG", "COH", "CON",
            "MEB", "MEG", "MEN", "IPB", "IPG", "IPN", "INV")


def _load_sut(prov: str) -> tuple[pd.DataFrame, pd.DataFrame, list, list]:
    """Load DomesticUse + InternatImportUse for one province.

    Returns (domuse_df, intim_df, col_codes, col_labels) where the dfs are
    indexed by the product code (BS/NP/GS) and columns are the same column
    codes from row 6 of the source sheet.
    """
    path = _sut_path(prov)
    if not path.exists():
        raise FileNotFoundError(path)

    # DomesticUse: products (rows) × (industries + FD + INT + province) cols
    raw = pd.read_excel(path, sheet_name="DomesticUse", header=None)
    col_codes = raw.iloc[6, 3:].tolist()
    col_labels = raw.iloc[7, 3:].tolist()
    body = raw.iloc[8:, 3:].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    body.index = raw.iloc[8:, 1].tolist()
    body.columns = pd.Index(col_codes)
    body = body.loc[[c for c in body.index if isinstance(c, str)]]
    domuse = body

    raw2 = pd.read_excel(path, sheet_name="InternatImportUse", header=None)
    col_codes2 = raw2.iloc[6, 3:].tolist()
    body2 = raw2.iloc[8:, 3:].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    body2.index = raw2.iloc[8:, 1].tolist()
    body2.columns = pd.Index(col_codes2)
    body2 = body2.loc[[c for c in body2.index if isinstance(c, str)]]
    intim = body2

    return domuse, intim, col_codes, col_labels


def _aggregate_to_sectors(values_by_product: pd.Series) -> dict[str, float]:
    """Sum a per-product Series into per-30-sector totals."""
    out: dict[str, float] = {s: 0.0 for s in SECTORS}
    for code, v in values_by_product.items():
        sec = naics_to_sector(code) if isinstance(code, str) else None
        if sec is None:
            continue
        out[sec] += float(v)
    return out


def build_flows() -> np.ndarray:
    """flows[n, i, j] in dollar millions, n=dest, i=source.

    n, i index into REGIONS (7 provinces + WLD); j into SECTORS.
    Within-province absorption from DomesticUse BS+NP+GS+FD cols.
    Bilateral exports from DomesticUse province cols (block 1 of two).
    International exports from DomesticUse INTEX/INTRX cols.
    International imports from InternatImportUse BS+NP+GS+FD cols.
    """
    N, J = len(REGIONS), len(SECTORS)
    flows = np.zeros((N, N, J))
    reg_idx = {r: i for i, r in enumerate(REGIONS)}

    fd_prefixes = _fd_prefixes()

    for source_prov in PROVINCES:
        print(f"  loading {source_prov} ...")
        domuse, intim, col_codes, _ = _load_sut(source_prov)

        # Identify column blocks
        ind_cols = [c for c in domuse.columns if isinstance(c, str)
                    and c[:2] in _industry_prefixes()]
        fd_cols = [c for c in domuse.columns if isinstance(c, str)
                   and c.startswith(fd_prefixes)]
        intex_cols = [c for c in domuse.columns if isinstance(c, str)
                      and c in ("INTEX000", "INTRX000")]

        # Province cols appear TWICE in domuse (block 1 = shipments, block 2 = zeros).
        # We need block 1 (first occurrence of each province code).
        province_col_positions: dict[str, int] = {}
        for pos, c in enumerate(domuse.columns):
            if isinstance(c, str) and c in PROV_AND_TERR and c not in province_col_positions:
                province_col_positions[c] = pos

        # Within-source absorption (BS+NP+GS+FD)
        domestic_within = domuse[ind_cols + fd_cols].sum(axis=1)
        dom_within_sec = _aggregate_to_sectors(domestic_within)
        for sec, v in dom_within_sec.items():
            s_idx = reg_idx[source_prov] if source_prov in reg_idx \
                else reg_idx[PROV_GROUP[source_prov]]
            j_idx = SECTORS.index(sec)
            flows[s_idx, s_idx, j_idx] += v

        # International exports (this source ships to WLD)
        intl_exp = domuse[intex_cols].sum(axis=1)
        intl_exp_sec = _aggregate_to_sectors(intl_exp)
        for sec, v in intl_exp_sec.items():
            wld_idx = reg_idx[WLD]
            s_idx = reg_idx.get(source_prov) or reg_idx[PROV_GROUP[source_prov]]
            j_idx = SECTORS.index(sec)
            flows[wld_idx, s_idx, j_idx] += v

        # Province-to-province bilateral exports (source → each dest).
        # After region aggregation, cross-Atlantic flows (e.g. NL→PE) land
        # on the ATL diagonal — that's correct, since both endpoints are in ATL.
        for dest, col_pos in province_col_positions.items():
            if dest not in PROVINCES:
                # Skip territories (NT/NU/YT) and CE — fold into WLD via INTL trade
                continue
            col_series = domuse.iloc[:, col_pos]
            sec_totals = _aggregate_to_sectors(col_series)
            d_idx = reg_idx.get(dest) or reg_idx[PROV_GROUP[dest]]
            s_idx = reg_idx.get(source_prov) or reg_idx[PROV_GROUP[source_prov]]
            for sec, v in sec_totals.items():
                j_idx = SECTORS.index(sec)
                flows[d_idx, s_idx, j_idx] += v

        # International imports (WLD → this source)
        intim_ind_cols = [c for c in intim.columns if isinstance(c, str)
                          and c[:2] in _industry_prefixes()]
        intim_fd_cols = [c for c in intim.columns if isinstance(c, str)
                         and c.startswith(fd_prefixes)]
        intl_imp = intim[intim_ind_cols + intim_fd_cols].sum(axis=1)
        intl_imp_sec = _aggregate_to_sectors(intl_imp)
        for sec, v in intl_imp_sec.items():
            d_idx = reg_idx.get(source_prov) or reg_idx[PROV_GROUP[source_prov]]
            wld_idx = reg_idx[WLD]
            j_idx = SECTORS.index(sec)
            flows[d_idx, wld_idx, j_idx] += v

    return flows


def add_icio_wld_intra(flows: np.ndarray, icio_csv: Path = ICIO_CSV) -> None:
    """Fill WLD→WLD entries from OECD ICIO 2022 (in-place on flows).

    For each sector j, sums non-CAN row × non-CAN column intermediate-use cells
    of the ICIO and converts USD-millions → CAD-thousands.
    """
    from scripts.sector_schema_m import ICIO_TO_SECTOR

    print(f"  loading ICIO {icio_csv} ...")
    df = pd.read_csv(icio_csv)
    row_labels = df["V1"].tolist()
    col_labels = list(df.columns[1:])
    M = df.drop(columns=["V1"]).to_numpy(dtype=np.float64)

    sector_to_isic: dict[str, list[str]] = {s: [] for s in SECTORS}
    for isic, sec in ICIO_TO_SECTOR.items():
        sector_to_isic[sec].append(isic)

    def split(lbl: str) -> tuple[str, str]:
        if "_" not in lbl:
            return "", lbl
        c, i = lbl.split("_", 1)
        return c, i

    # Non-CAN intermediate use: sum across origin rows (non-CAN) × dest cols
    # (non-CAN intermediate-use columns — exclude "OUT", final-demand suffixes).
    fd_suf = {"HFCE", "NPISH", "GGFC", "GFCF", "INVNT", "OUT"}
    wld_idx = REGIONS.index(WLD)
    for sec, isics in sector_to_isic.items():
        row_mask = np.array([
            i for i, lbl in enumerate(row_labels)
            if (split(lbl)[0] not in ("", CAN)) and split(lbl)[1] in isics
        ], dtype=np.int64)
        col_mask = np.array([
            i for i, lbl in enumerate(col_labels)
            if (split(lbl)[0] not in ("", CAN)) and (split(lbl)[1] not in fd_suf)
        ], dtype=np.int64)
        if row_mask.size == 0 or col_mask.size == 0:
            continue
        total_usd_m = float(M[np.ix_(row_mask, col_mask)].sum())
        j_idx = SECTORS.index(sec)
        flows[wld_idx, wld_idx, j_idx] += total_usd_m * ICIO_TO_SUT_UNITS


def compute_shares(flows: np.ndarray) -> np.ndarray:
    """Per-destination share: pi[n, i, j] = flows[n, i, j] / sum_i flows[n, i, j]."""
    N = flows.shape[0]
    J = flows.shape[2]
    totals = flows.sum(axis=1, keepdims=True)         # (N, 1, J)
    safe = np.where(totals > 0, totals, 1.0)
    pi = flows / safe
    return pi


def report_positivity(flows: np.ndarray, pi: np.ndarray) -> None:
    N, _, J = flows.shape
    print("\n" + "=" * 78)
    print("POSITIVITY REPORT — flows in CAD millions (interprov, intra-Canada)")
    print("=" * 78)

    # Total trade volume per (n, j) — gives weights for severity
    totals_nj = flows.sum(axis=1)                      # (N, J)

    zero_diag = []
    for n in range(N - 1):    # exclude WLD diagonal (uses ICIO, not SUT)
        for j in range(J):
            if pi[n, n, j] == 0.0:
                zero_diag.append((REGIONS[n], SECTORS[j], totals_nj[n, j]))

    zero_offdiag = []
    for n in range(N):
        for i in range(N):
            if n == i:
                continue
            for j in range(J):
                if flows[n, i, j] == 0.0 and totals_nj[n, j] > 0:
                    zero_offdiag.append((REGIONS[n], REGIONS[i], SECTORS[j]))

    # Head-Ries unrecoverable: any (n, i, j) where any of pi[n,n,j], pi[i,i,j],
    # pi[n,i,j], pi[i,n,j] is zero (excluding self-pairs).
    hr_fail = []
    for n in range(N):
        for i in range(N):
            if n == i: continue
            for j in range(J):
                vals = (pi[n, n, j], pi[i, i, j], pi[n, i, j], pi[i, n, j])
                if any(v == 0 for v in vals):
                    hr_fail.append((REGIONS[n], REGIONS[i], SECTORS[j], vals))

    # An HR cell is "fillable by ICIO" iff:
    #   - one of {n, i} is WLD (so one diagonal in the HR is pi[WLD,WLD,j])
    #   - all OTHER three pi values are already positive
    # i.e. the only zero blocking HR is pi[WLD,WLD,j].
    wld_idx = REGIONS.index(WLD)
    hr_real = []
    hr_wld_only = []
    for t in hr_fail:
        n_name, i_name, j_name, _vals = t
        n_idx = REGIONS.index(n_name)
        i_idx = REGIONS.index(i_name)
        j_idx = SECTORS.index(j_name)
        involves_wld = (n_idx == wld_idx) or (i_idx == wld_idx)
        # The "other three" pi values, excluding pi[WLD,WLD,j]:
        partner_idx = i_idx if n_idx == wld_idx else n_idx
        others_positive = (
            pi[partner_idx, partner_idx, j_idx] > 0
            and pi[n_idx, i_idx, j_idx] > 0
            and pi[i_idx, n_idx, j_idx] > 0
        )
        if involves_wld and others_positive:
            hr_wld_only.append(t)
        else:
            hr_real.append(t)

    print(f"\nDiagonal failures (pi[n,n,j] = 0): {len(zero_diag)} of {(N-1)*J} "
          f"({100*len(zero_diag)/((N-1)*J):.1f}%)")
    if zero_diag:
        print("  By province × sector (thousand CAD absorption):")
        for n, j, vol in sorted(zero_diag, key=lambda t: -t[2]):
            print(f"    {n:>4}  {j:<35}  abs={vol:,.0f}")

    print(f"\nOff-diagonal failures (pi[n,i,j] = 0): {len(zero_offdiag)} of "
          f"{N*(N-1)*J} ({100*len(zero_offdiag)/(N*(N-1)*J):.1f}%)")

    print(f"\nHead-Ries-unrecoverable (n,i,j): {len(hr_fail)} of "
          f"{N*(N-1)*J} ({100*len(hr_fail)/(N*(N-1)*J):.1f}%)")
    print(f"  of which fillable by ICIO WLD-WLD: {len(hr_wld_only)}")
    print(f"  REAL HR failures (intra-Canada + WLD-prov):    {len(hr_real)} "
          f"({100*len(hr_real)/(N*(N-1)*J):.1f}%)")

    # Per-sector summary
    print("\nPer-sector failure counts:")
    print(f"  {'Sector':<35} {'diag':>5} {'off':>5} {'HR':>5}")
    diag_by_sec = {s: 0 for s in SECTORS}
    off_by_sec = {s: 0 for s in SECTORS}
    hr_by_sec = {s: 0 for s in SECTORS}
    for _, j_name, _ in zero_diag: diag_by_sec[j_name] += 1
    for _, _, j_name in zero_offdiag: off_by_sec[j_name] += 1
    for _, _, j_name, _ in hr_fail: hr_by_sec[j_name] += 1
    for s in SECTORS:
        if diag_by_sec[s] + off_by_sec[s] + hr_by_sec[s] > 0:
            print(f"  {s:<35} {diag_by_sec[s]:>5} {off_by_sec[s]:>5} "
                  f"{hr_by_sec[s]:>5}")


def main() -> None:
    print(f"N = {len(REGIONS)} regions: {REGIONS}")
    print(f"J = {len(SECTORS)} sectors")
    print()
    print("Building dollar flows from SUT workbooks ...\n")
    flows = build_flows()
    print()
    print("Filling WLD-WLD intra-trade from ICIO 2022 ...\n")
    add_icio_wld_intra(flows)

    # Sanity: report total inter-Canada (province × province) trade
    N = len(REGIONS)
    intra_can_mask = np.zeros((N, N), dtype=bool)
    for ni, n in enumerate(REGIONS):
        for ii, i in enumerate(REGIONS):
            if n != WLD and i != WLD:
                intra_can_mask[ni, ii] = True
    intra_diag = sum(flows[i, i, :].sum() for i in range(N - 1))
    intra_off = (flows.sum(axis=2) * intra_can_mask).sum() - intra_diag
    intl_imports = sum(flows[i, N - 1, :].sum() for i in range(N - 1))
    intl_exports = sum(flows[N - 1, i, :].sum() for i in range(N - 1))
    print(f"\n  Intra-province (within-Canada diagonal):  ${intra_diag:>12,.0f}M")
    print(f"  Inter-province (off-diagonal):            ${intra_off:>12,.0f}M")
    print(f"  International imports (to Canada):        ${intl_imports:>12,.0f}M")
    print(f"  International exports (from Canada):      ${intl_exports:>12,.0f}M")

    pi = compute_shares(flows)
    report_positivity(flows, pi)


if __name__ == "__main__":
    main()
