"""Replicate Díez, Tombe & Yang (2026, IMF WP/26/194) non-distance internal
trade costs from StatCan 2021 detail-level interprovincial flows.

Method (DTY §3, eq. 2), per industry j and ordered region pair (n, i):
  1) Head-Ries index ln τ̄_ni = L_ni / θ_j with the θ-free
        L_ni = ½ ln(X_nn X_ii / (X_ni X_in)).
  2) L_ni = α_j ln(d_ni / √(d_nn d_ii)) + γ_n^j + γ_i^j + ε, within industry j
     (pooled to 3-, then 2-digit codes, then goods/services when j has fewer
     than MIN_OBS pairs).
  3) Non-distance cost NG_ni = exp((L_ni − α̂_j ln d̄_ni) / θ_j) − 1, floored at 0
     per region-industry pair (DTY Fig. 3b's spike at 0; AT-2016's
     tauhat_dist = min(1, ·) cap); trade-weighted (X_ni) averages.

What we found matters (scored against DTY Tables 1–3 and the 9.3 / 1.3 /
16.8 % headline):
  * Aggregate product flows to industries BEFORE Head-Ries. Product-level
    indices leave BC←ON goods at −9 %: unwrought gold and single-product
    territory cells dominate.
  * Distances are population-weighted (Head-Mayer) between 2021 Census DAs,
    data/geography/popweighted_distance_2021.parquet. Area-based internal
    distances overstate d_nn for ON/QC and inflate NG.
  * θ_j: goods from Fontagné-Guimbard-Orefice (2022) HS6 (CEPII ProTEE) →
    NAICS (Pierce-Schott, 2007–2011) → SUPC product by longest NAICS prefix,
    averaged over the industry's products; services θ = 5.
  * Industries: "supc3" groups SUPC products by their first 3 digits (96
    groups); "supply" splits each product's flows across the 2021 detail
    IOIC industries (215 with trade) by the origin region's supply shares
    (36-10-0478). Assigning each product to its principal producer instead
    fits worse than either. "supc3" fits the goods bilateral pattern and
    Table 3 better; "supply" the services pattern.
  * The concordance-free robustness levels (uniform θ = 4, 8 → 12.8, 5.5 %)
    point to the ρ = −0.5 power-mean distance for both industry definitions.

Inputs (in --raw, default ~/Downloads):
  12100101-eng.zip  StatCan 12-10-0101, interprovincial trade, detail level
      https://www150.statcan.gc.ca/n1/tbl/csv/12100101-eng.zip
  36100478-eng.zip  StatCan 36-10-0478, supply and use tables, detail level
      https://www150.statcan.gc.ca/n1/tbl/csv/36100478-eng.zip  (424 MB)
  ProTEE_0_1.csv    https://www.cepii.fr/DATA_DOWNLOAD/ProTEE/ProTEE_0_1.csv
  hssicnaics_20181015.zip  Pierce-Schott HS→NAICS concordance
      https://spinup-000d1a-wp-offload-media.s3.amazonaws.com/faculty/wp-content/uploads/sites/47/2019/06/hssicnaics_20181015.zip

Run from dty/:
    uv run python scripts/dty_replication.py [--raw DIR]
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DIST_PQ = Path(__file__).resolve().parent.parent / "data" / "geography" / "popweighted_distance_2021.parquet"
YEAR = 2021
MIN_OBS = 30
THETA_SERVICES = 5.0
DISTANCES = ["rho_0.5", "geom", "rho_-0.5"]
HEADLINE = "rho_-0.5"  # matches DTY's uniform-θ robustness levels (θ = 4, 8 → 12.8, 5.5 %)

REGIONS = {"British Columbia": "BC", "Alberta": "AB", "Saskatchewan": "SK", "Manitoba": "MB",
           "Ontario": "ON", "Quebec": "QC", "New Brunswick": "NB", "Nova Scotia": "NS",
           "Prince Edward Island": "PE", "Newfoundland and Labrador": "NL", "Yukon": "YT",
           "Northwest Territories": "NT", "Nunavut": "NU"}
PROVINCES = list(REGIONS.values())[:10]

# DTY Tables 1 (goods) and 2 (services), %: rows importer n, columns exporter i, order REGIONS.
DTY_GOODS = """- 0.5 1.7 0.5 0.0 0.0 0.8 0.7 4.4 11.2 6.5 5.6 0.9
0.9 - 1.4 0.5 0.1 0.1 1.1 0.4 7.7 2.9 33.8 0.9 26.6
3.8 2.9 - 2.9 1.2 1.6 2.6 3.3 11.3 37.0 62.0 58.7 25.6
1.1 0.5 1.3 - 0.5 0.7 2.4 0.7 8.1 1.9 19.4 10.5 33.7
0.1 0.0 0.1 0.1 - 0.6 1.0 0.4 2.4 1.9 3.2 5.4 3.9
0.5 0.1 1.1 0.2 0.9 - 4.7 4.1 21.8 15.1 1.7 10.8 7.0
2.4 0.2 1.9 0.7 3.0 9.9 - 9.6 11.4 1.0 49.5 29.8 11.1
0.4 0.4 1.7 2.4 1.4 4.6 5.3 - 17.0 10.2 39.0 11.9 2.0
3.4 4.4 13.5 7.3 4.9 14.3 1.0 9.3 - 53.0 586 86.7 81.4
1.3 4.5 25.7 16.9 8.4 6.5 3.6 5.7 9.6 - 341 181 232
3.0 8.2 5.7 6.2 12.1 26.6 26.3 26.8 586 36.4 - 31.8 49.2
28.3 21.2 39.7 23.6 17.1 38.0 56.3 30.3 121 54.4 21.5 - 1.0
19.2 37.2 66.1 48.5 43.6 42.3 11.0 75.0 125 27.4 48.0 28.2 -"""
DTY_SERVICES = """- 27.5 41.3 13.9 4.9 8.6 25.2 41.6 58.4 46.3 47.0 81.4 121
22.6 - 32.9 16.8 4.3 11.8 30.3 35.2 67.7 30.8 59.2 50.1 117
26.4 33.9 - 25.1 16.9 20.4 57.5 58.5 103 56.2 59.0 105 160
20.6 31.1 42.1 - 12.7 19.0 45.0 58.7 78.3 81.9 78.9 71.2 134
7.4 8.9 11.6 12.7 - 18.3 39.0 26.6 26.8 29.0 50.9 37.7 71.3
14.6 19.2 34.8 22.6 16.7 - 52.7 48.5 64.3 44.6 90.2 90.1 96.8
20.1 19.3 47.5 39.7 20.3 34.2 - 57.4 51.2 41.0 135 105 209
18.9 20.2 45.7 43.9 15.7 36.0 54.6 - 77.5 32.5 95.9 83.8 143
23.2 30.2 67.8 70.2 27.4 35.0 42.9 71.8 - 56.5 113 112 310
37.3 18.2 82.9 58.8 27.0 33.9 37.7 37.2 43.4 - 99.0 77.3 137
28.7 35.8 92.4 64.9 58.1 55.8 80.3 116 179 88.3 - 20.0 137
58.5 45.4 157 79.8 58.1 71.6 164 108 152 185 62.2 - 37.3
73.8 84.6 163 180 82.1 56.9 254 90.0 285 127 35.5 36.0 -"""
# DTY Table 3, summary-level sectors (%), with the NAICS-style code prefixes we assign to each.
DTY_SECTORS = {
    "Crop and animal": (0.0, ("111", "112")), "Forestry": (2.4, ("113",)), "Fishing": (15.5, ("114",)),
    "Ag support": (22.9, ("115",)), "Mining": (0.0, ("21",)), "Utilities": (36.2, ("22",)),
    "Construction": (18.4, ("23",)), "Manufacturing": (0.3, ("3",)), "Wholesale": (1.1, ("41",)),
    "Retail": (60.6, ("44", "45", "4A")), "Transport": (12.5, ("48", "49", "4B")),
    "Information": (38.2, ("51",)), "FIRE": (13.1, ("52", "53", "5B")), "Professional": (6.0, ("54",)),
    "Admin": (3.5, ("55", "56")), "Education": (41.8, ("61",)), "Health": (55.5, ("62",)),
    "Arts": (17.0, ("71",)), "Accommodation": (23.6, ("72",)), "Other services": (16.4, ("81",)),
}


def _table(txt: str) -> pd.Series:
    ab = list(REGIONS.values())
    m = [[np.nan if v == "-" else float(v) for v in line.split()] for line in txt.splitlines()]
    return pd.DataFrame(m, index=ab, columns=ab).stack().rename("DTY")


DTY = {"goods": _table(DTY_GOODS), "services": _table(DTY_SERVICES)}
DTY_TABLE3 = pd.Series({s: v for s, (v, _) in DTY_SECTORS.items()}, name="DTY")


def _csv_in_zip(path: Path) -> str:
    return next(n for n in zipfile.ZipFile(path).namelist() if n.endswith(".csv") and "MetaData" not in n)


# ── Inputs ─────────────────────────────────────────────────────────────────
def load_flows(zip0101: Path) -> pd.DataFrame:
    """2021 flows by SUPC product: (code, kind, origin, dest, X), own-region flows included."""
    with zipfile.ZipFile(zip0101) as z, z.open(_csv_in_zip(zip0101)) as f:
        df = pd.read_csv(f, usecols=["REF_DATE", "GEO", "Trade flow detail", "Product", "VALUE"], low_memory=False)
    df = df[df.REF_DATE == YEAR]
    df["code"] = df["Product"].astype(str).str.extract(r"\[([A-Z]{3}\d\w+)\]", expand=False)
    df["kind"] = df.code.str[:3].map({"MPG": "goods", "ENE": "goods", "MPS": "services"})
    tf = df["Trade flow detail"].astype(str)
    df = df[df.kind.notna() & tf.str.startswith("To ")].assign(dest=tf.str.removeprefix("To "))
    df = df[df.GEO.isin(REGIONS) & df.dest.isin(REGIONS) & (df.VALUE > 0)]
    return df.rename(columns={"GEO": "origin", "VALUE": "X"})[["code", "kind", "origin", "dest", "X"]]


def product_theta(codes: pd.Series, protee_csv: Path, ps_zip: Path) -> pd.Series:
    """θ per SUPC goods product: ProTEE HS6 (−sigma; blanks → HS4, HS2 means) → NAICS6
    (Pierce-Schott imports, 2007–2011) → mean over NAICS codes sharing the product's
    longest matching prefix (6 down to 2 digits)."""
    p = pd.read_csv(protee_csv, dtype={"HS6": str})
    p["theta"] = -p["sigma"]
    for k in (4, 2):
        p["theta"] = p["theta"].fillna(p.groupby(p.HS6.str[:k])["theta"].transform("mean"))
    member = "hs_sic_naics_imports_89_117_20180927.csv"
    with zipfile.ZipFile(ps_zip) as z, z.open(member) as f:
        ps = pd.read_csv(f, sep="\t", dtype=str, usecols=["commodity", "year", "naics"])
    ps = ps[ps.year.astype(int).between(2007, 2011)]
    ps["naics"] = ps.naics.str.strip('"')
    ps = ps[ps.naics.str.fullmatch(r"\d{6}")]
    hs_naics = ps.assign(HS6=ps.commodity.str[:6])[["HS6", "naics"]].drop_duplicates()
    naics_theta = hs_naics.merge(p[["HS6", "theta"]], on="HS6").groupby("naics").theta.mean()

    def one(code: str) -> float:
        for k in (6, 5, 4, 3, 2):
            m = naics_theta[naics_theta.index.str.startswith(code[3:3 + k])]
            if len(m):
                return float(m.mean())
        return float("nan")

    return pd.Series({c: one(c) for c in codes}, dtype=float)


def supply_shares(zip0478: Path) -> pd.DataFrame:
    """(region, code, ind, s): share of IOIC industry `ind` in region's 2021 supply of
    SUPC product `code` at basic prices (business, government and NPISH industries)."""
    keep = []
    with zipfile.ZipFile(zip0478) as z, z.open(_csv_in_zip(zip0478)) as f:
        for ch in pd.read_csv(f, chunksize=2_000_000, usecols=["REF_DATE", "GEO", "Supply and use", "Valuation",
                                                               "Industry", "Product", "VALUE"]):
            keep.append(ch[(ch.REF_DATE == YEAR) & (ch["Supply and use"] == "Supply")
                           & (ch.Valuation == "Basic price") & (ch.VALUE > 0)])
    df = pd.concat(keep)
    df["ind"] = df.Industry.str.extract(r"\[(\w+)\]$", expand=False)
    df["code"] = df.Product.str.extract(r"\[(\w+)\]$", expand=False)
    df = df[df.ind.str.match(r"^(BS|GS|NP)", na=False) & df.code.notna()]
    df["s"] = df.VALUE / df.groupby(["GEO", "code"]).VALUE.transform("sum")
    return df.rename(columns={"GEO": "region"})[["region", "code", "ind", "s"]]


# ── Industries ─────────────────────────────────────────────────────────────
def industries_supc3(flows: pd.DataFrame, theta_p: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Group products by SUPC prefix + first 3 digits; θ_j = mean θ over the group's products."""
    f = flows.assign(ind=flows.code.str[:6])
    th = f.drop_duplicates("code").assign(t=lambda d: d.code.map(theta_p)).groupby("ind").t.mean()
    return f.groupby(["ind", "kind", "origin", "dest"], as_index=False).X.sum(), th


def industries_supply(flows: pd.DataFrame, theta_p: pd.Series, shares: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split each product's flow across IOIC industries by the origin's supply shares (national
    shares where the origin reports no supply). θ_j = flow-weighted mean of product θ; an
    industry is goods or services by the majority of its flow value."""
    have = set(zip(shares.region, shares.code))
    regional = np.array([(o, c) in have for o, c in zip(flows.origin, flows.code)])
    f = pd.concat([flows[regional].merge(shares, left_on=["origin", "code"], right_on=["region", "code"]),
                   flows[~regional].merge(shares[shares.region == "Canada"].drop(columns="region"), on="code")])
    f["X"] = f.X * f.s
    f["ind"] = f.ind.str[:2] + "_" + f.ind.str[2:]  # BS_211110: code[3:] is NAICS-like, as for SUPC
    t = f.code.map(theta_p)
    w = f.assign(tX=t * f.X, Xt=f.X.where(t.notna(), 0.0)).groupby("ind")[["tX", "Xt"]].sum()
    kind = f.groupby(["ind", "kind"]).X.sum().unstack(fill_value=0).idxmax(axis=1)
    th = (w.tX / w.Xt).where(kind == "goods", THETA_SERVICES)
    out = f.groupby(["ind", "origin", "dest"], as_index=False).X.sum()
    return out.assign(kind=out.ind.map(kind)), th


# ── Head-Ries, distance regression, non-distance cost ──────────────────────
def head_ries(flows: pd.DataFrame) -> pd.DataFrame:
    """Ordered pairs (n importer, i exporter) with all four flows positive: X_ni, L."""
    own = flows[flows.origin == flows.dest].set_index(["ind", "origin"]).X
    f = flows[flows.origin != flows.dest].rename(columns={"dest": "n", "origin": "i", "X": "X_ni"})
    rev = f.rename(columns={"n": "i", "i": "n", "X_ni": "X_in"})[["ind", "n", "i", "X_in"]]
    hr = f.merge(rev, on=["ind", "n", "i"])
    hr["X_nn"] = own.reindex(pd.MultiIndex.from_arrays([hr.ind, hr.n])).to_numpy()
    hr["X_ii"] = own.reindex(pd.MultiIndex.from_arrays([hr.ind, hr.i])).to_numpy()
    hr = hr[(hr.X_nn > 0) & (hr.X_ii > 0)].reset_index(drop=True)
    hr["L"] = 0.5 * np.log(hr.X_nn * hr.X_ii / (hr.X_ni * hr.X_in))
    return hr


def regression_groups(hr: pd.DataFrame) -> pd.Series:
    """The industry itself if it has ≥ MIN_OBS pairs, else kind + 3-digit, 2-digit, kind."""
    keys = [hr.ind, hr.kind + "_" + hr.ind.str[3:6], hr.kind + "_" + hr.ind.str[3:5], hr.kind]
    grp = pd.Series(index=hr.index, dtype=object)
    for lv, k in enumerate(keys):
        take = grp.isna() & ((k.map(k.value_counts()) >= MIN_OBS) | (lv == len(keys) - 1))
        grp[take] = f"{lv}:" + k[take]
    return grp


def log_rel_distance(hr: pd.DataFrame, measure: str) -> pd.Series:
    d = pd.read_parquet(DIST_PQ)
    d = d[(d.regions == 13) & (d.measure == measure)].set_index(["destination", "source"]).value
    get = lambda a, b: d.reindex(pd.MultiIndex.from_arrays([a, b])).to_numpy()  # noqa: E731
    return pd.Series(np.log(get(hr.n, hr.i) / np.sqrt(get(hr.n, hr.n) * get(hr.i, hr.i))), index=hr.index)


def non_distance(hr: pd.DataFrame, lnd: pd.Series, grp: pd.Series) -> pd.Series:
    """θ-free non-distance index L − α̂_g ln d̄, α̂_g from L on ln d̄ + importer and exporter FE."""
    out = pd.Series(index=hr.index, dtype=float)
    for _, idx in hr.groupby(grp).groups.items():
        g, x = hr.loc[idx], lnd.loc[idx].to_numpy()
        X = np.hstack([x[:, None], pd.get_dummies(g.n, drop_first=True).to_numpy(float),
                       pd.get_dummies(g.i, drop_first=True).to_numpy(float), np.ones((len(g), 1))])
        alpha = np.linalg.lstsq(X, g.L.to_numpy(), rcond=None)[0][0]
        out.loc[idx] = g.L.to_numpy() - alpha * x
    return out


# ── Scoring against DTY ────────────────────────────────────────────────────
def wavg(v: pd.Series, w: pd.Series) -> float:
    return 100 * float((v * w).sum() / w.sum())


def bilateral(hr: pd.DataFrame, ng: pd.Series, kind: str) -> pd.Series:
    k = hr.kind == kind
    d = pd.DataFrame({"imp": hr.n[k].map(REGIONS), "exp": hr.i[k].map(REGIONS), "v": ng[k] * hr.X_ni[k], "w": hr.X_ni[k]})
    s = d.groupby(["imp", "exp"])[["v", "w"]].sum()
    return 100 * s.v / s.w


def sector(code: str) -> str | None:
    """DTY Table 3 sector of an industry code; None for government and NPISH (non-market)."""
    if code[:2] in ("GS", "NP"):
        return None
    best = max(((len(p), name) for name, (_, ps) in DTY_SECTORS.items() for p in ps if code[3:].startswith(p)),
               default=(0, None))
    return best[1]


def sector_means(hr: pd.DataFrame, ng: pd.Series) -> pd.Series:
    """Trade-weighted NG (%) per DTY Table 3 sector (market industries only)."""
    t = pd.DataFrame({"sec": hr.ind.map(sector), "v": ng * hr.X_ni, "w": hr.X_ni}).dropna().groupby("sec").sum()
    return 100 * t.v / t.w


def score(hr: pd.DataFrame, ngL: pd.Series, theta: pd.Series) -> dict:
    ng = (np.exp(ngL / theta) - 1).clip(lower=0)
    r = {"NG": wavg(ng, hr.X_ni),
         "NG_θ4": wavg((np.exp(ngL / 4) - 1).clip(lower=0), hr.X_ni),
         "NG_θ8": wavg((np.exp(ngL / 8) - 1).clip(lower=0), hr.X_ni),
         "total": wavg(np.exp(hr.L / theta) - 1, hr.X_ni)}
    for kind in ("goods", "services"):
        k = hr.kind == kind
        r[f"NG_{kind[0]}"] = wavg(ng[k], hr.X_ni[k])
        m = pd.concat([bilateral(hr, ng, kind).rename("ours"), DTY[kind]], axis=1, join="inner")
        prov = np.array([a in PROVINCES and b in PROVINCES for a, b in m.index])
        r[f"ρprov_{kind[0]}"] = m[prov].ours.corr(m[prov].DTY, method="spearman")
        r[f"MAEprov_{kind[0]}"] = (m[prov].ours - m[prov].DTY).abs().mean()
        r[f"ρterr_{kind[0]}"] = m[~prov].ours.corr(m[~prov].DTY, method="spearman")
    t = pd.concat([sector_means(hr, ng), DTY_TABLE3], axis=1, join="inner")
    r["ρTable3"] = t.iloc[:, 0].corr(t.iloc[:, 1], method="spearman")
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, default=Path.home() / "Downloads")
    raw = ap.parse_args().raw

    flows = load_flows(raw / "12100101-eng.zip")
    theta_p = product_theta(flows.code[flows.kind == "goods"].unique(), raw / "ProTEE_0_1.csv",
                            raw / "hssicnaics_20181015.zip")
    theta_p = theta_p.reindex(flows.code.unique()).fillna(
        pd.Series(THETA_SERVICES, index=flows.code[flows.kind == "services"].unique()))
    shares = supply_shares(raw / "36100478-eng.zip")
    builds = {"supc3": industries_supc3(flows, theta_p), "supply": industries_supply(flows, theta_p, shares)}

    rows, goods, sectors = [], {}, {}
    for name, (f, th) in builds.items():
        hr = head_ries(f)
        theta, grp = hr.ind.map(th), regression_groups(hr)
        ok = theta.notna()
        print(f"{name}: {hr.ind.nunique()} industries, {len(hr):,} pairs, {grp.nunique()} regression groups")
        for measure in DISTANCES:
            ngL = non_distance(hr, log_rel_distance(hr, measure), grp)
            rows.append({"industries": name, "distance": measure, **score(hr[ok], ngL[ok], theta[ok])})
            if measure == HEADLINE:
                ng = (np.exp(ngL[ok] / theta[ok]) - 1).clip(lower=0)
                goods[name], sectors[name] = bilateral(hr[ok], ng, "goods"), sector_means(hr[ok], ng)

    pd.set_option("display.width", 250)
    pd.set_option("display.float_format", lambda v: f"{v:.2f}")
    print("\nDTY: NG 9.3 (θ=4 12.8, θ=8 5.5), goods 1.3, services 16.8, total τ̄ 79")
    print(pd.DataFrame(rows).to_string(index=False))
    pd.set_option("display.float_format", lambda v: f"{v:5.1f}")
    for name, s in [*goods.items(), ("DTY Table 1", DTY["goods"])]:
        print(f"\nGoods NG (%), provinces, rows = importer — {name}" + (f" ({HEADLINE})" if name in goods else ""))
        print(s.unstack().reindex(index=PROVINCES, columns=PROVINCES).to_string())
    print(f"\nNG (%) by summary sector ({HEADLINE}) vs DTY Table 3")
    print(pd.concat([DTY_TABLE3, pd.DataFrame(sectors)], axis=1).loc[DTY_TABLE3.index].to_string())


if __name__ == "__main__":
    main()
