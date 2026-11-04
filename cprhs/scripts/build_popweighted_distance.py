"""Population-weighted (Head-Mayer) internal and bilateral distances for
Canadian provinces and territories, from 2021 Census dissemination areas.

    d_ab = ( Σ_{k∈a} Σ_{l∈b} s_k s_l d_kl^ρ )^(1/ρ),   s_k = pop_k / pop_a

over all DA pairs (k, l), with d_kl the great-circle distance between DA
representative points (population-weighted, from the GAF) and the own-DA
cell d_kk = (2/3)·√(area_k/π). ρ = 1 is the population-weighted arithmetic
mean; ρ = −1 the harmonic mean Head & Mayer (2009) recommend; ρ = 0 the
geometric mean (Σ s s ln d, then exp).

Replaces the area-based internal distance (2/3)·√(area_n/π) for province n,
which overstates internal distance for provinces whose population is
concentrated in a few cities (ON 390 km → 209 km arithmetic) and understates
it for small dispersed ones (NS 88 km → 141 km). In the log-normalized
distance spec ln(d_ni/√(d_nn d_ii)) used by AT-2016 and Díez-Tombe-Yang
(2026), the internal terms are absorbed by the exporter/importer FE, so slopes
are unchanged — but the non-distance residual shifts province by province.

Input: Statistics Canada 2021 Geographic Attribute File (catalogue 92-151-X),
one row per dissemination block. www12.statcan.gc.ca blocks scripted
downloads; fetch it in a browser from
    https://www12.statcan.gc.ca/census-recensement/2021/geo/aip-pia/attribute-attribs/index2021-eng.cfm?year=2021
(CSV, English) and leave the ZIP as ~/Downloads/2021_92-151_X.zip.

Output: data/geography/popweighted_distance_2021.parquet, long form
    (destination, source, measure, regions, value)
value in km; symmetric. regions = 13 (NT and NU separate) or 12 (NT+NU
merged as "NT_NU", the AKT-2019 convention).

Run from cprhs/:
    uv run python scripts/build_popweighted_distance.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

GAF_ZIP = Path.home() / "Downloads" / "2021_92-151_X.zip"
OUT = Path(__file__).resolve().parent.parent / "data" / "geography" / "popweighted_distance_2021.parquet"

R_KM = 6371.0
RHOS = {"arith": 1.0, "rho_0.5": 0.5, "geom": 0.0, "rho_-0.5": -0.5, "harm": -1.0}

PR_NAMES = {
    "10": "Newfoundland and Labrador", "11": "Prince Edward Island", "12": "Nova Scotia",
    "13": "New Brunswick", "24": "Quebec", "35": "Ontario", "46": "Manitoba",
    "47": "Saskatchewan", "48": "Alberta", "59": "British Columbia", "60": "Yukon",
    "61": "Northwest Territories", "62": "Nunavut",
}
NT_NU_MERGE = {"Northwest Territories": "NT_NU", "Nunavut": "NT_NU"}


# ── GAF (dissemination blocks) → populated DAs ─────────────────────────────
def load_das(gaf_path: Path) -> pd.DataFrame:
    """One row per populated DA: region, rep-point lat/lon, population, land area (km²)."""
    cols = {"PRUID_PRIDU": "pruid", "DAUID_ADIDU": "dauid", "DARPLAT_ADLAT": "lat",
            "DARPLONG_ADLONG": "lon", "DBPOP2021_IDPOP2021": "pop",
            "DBAREA2021_IDSUP2021": "area"}
    db = pd.read_csv(gaf_path, usecols=list(cols), encoding="latin-1",
                     dtype={"PRUID_PRIDU": str, "DAUID_ADIDU": str}).rename(columns=cols)
    da = (db.groupby(["pruid", "dauid"], as_index=False)
            .agg(lat=("lat", "first"), lon=("lon", "first"),
                 pop=("pop", "sum"), area=("area", "sum")))
    da = da[da["pop"] > 0].copy()
    da["region"] = da["pruid"].map(PR_NAMES)
    return da


# ── Effective distance ─────────────────────────────────────────────────────
def _pow(d: np.ndarray, rho: float) -> np.ndarray:
    return np.log(d) if rho == 0 else d ** rho


def _inv(m: float, rho: float) -> float:
    return math.exp(m) if rho == 0 else m ** (1.0 / rho)


def _unit_vectors(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    la, lo = np.radians(lat), np.radians(lon)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def _pair_means(a: dict, b: dict, same: bool, chunk: int = 1500) -> dict[str, float]:
    """Σ_k Σ_l s_k s_l d_kl^ρ for every ρ in RHOS (chunked over a's DAs)."""
    total = dict.fromkeys(RHOS, 0.0)
    for start in range(0, len(a["s"]), chunk):
        sl = slice(start, start + chunk)
        d = R_KM * np.arccos(np.clip(a["xyz"][sl] @ b["xyz"].T, -1.0, 1.0))
        if same:  # own-DA cell → within-DA distance
            r = np.arange(d.shape[0])
            d[r, start + r] = a["dself"][sl]
        d = np.maximum(d, 1e-3)  # distinct DAs sharing a rep point
        for name, rho in RHOS.items():
            total[name] += a["s"][sl] @ _pow(d, rho) @ b["s"]
    return total


def effective_distances(da: pd.DataFrame, merge: dict[str, str]) -> pd.DataFrame:
    """Effective distances between DA groups in `da.region`, plus the coarser
    grouping `merge` (region → merged name), built by population-share
    weighting the finer pairwise means of d^ρ."""
    groups, pop = {}, {}
    for reg, g in da.groupby("region"):
        groups[reg] = {"xyz": _unit_vectors(g["lat"].to_numpy(), g["lon"].to_numpy()),
                       "s": (g["pop"] / g["pop"].sum()).to_numpy(),
                       "dself": (2 / 3) * np.sqrt(g["area"].to_numpy() / math.pi)}
        pop[reg] = g["pop"].sum()
    regs = sorted(groups)
    mean = {}
    for ia, ra in enumerate(regs):
        for rb in regs[ia:]:
            for name, v in _pair_means(groups[ra], groups[rb], same=(ra == rb)).items():
                mean[(ra, rb, name)] = mean[(rb, ra, name)] = v

    rows = []
    for level in ({r: r for r in regs}, {r: merge.get(r, r) for r in regs}):
        coarse = sorted(set(level.values()))
        members = {A: [r for r in regs if level[r] == A] for A in coarse}
        for A in coarse:
            for B in coarse:
                PA = sum(pop[r] for r in members[A]); PB = sum(pop[r] for r in members[B])
                for name, rho in RHOS.items():
                    m = sum(pop[x] * pop[y] * mean[(x, y, name)]
                            for x in members[A] for y in members[B]) / (PA * PB)
                    rows.append((A, B, name, len(coarse), _inv(m, rho)))
    return pd.DataFrame(rows, columns=["destination", "source", "measure", "regions", "value"])


def main() -> None:
    da = load_das(GAF_ZIP)
    print(f"{len(da):,} populated DAs, population {da['pop'].sum():,.0f}")
    out = effective_distances(da, NT_NU_MERGE)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"wrote {OUT} ({len(out):,} rows)\n")
    internal = out[(out.destination == out.source) & (out.regions == 13)]
    print(internal.pivot(index="destination", columns="measure", values="value")
                  [list(RHOS)].round(1).to_string())


if __name__ == "__main__":
    main()
