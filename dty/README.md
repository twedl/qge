# dty — Díez, Tombe & Yang (2026) replication

Replicates the trade-cost measurement in:

> Díez, F. J., Tombe, T., and Yang, Y. (2026). "Barriers Within Borders:
> Internal Trade and Domestic Market Integration." IMF Working Paper
> WP/26/194, September 2026.

DTY estimate Canadian internal trade costs from StatCan 2021 detail-level
interprovincial flows. They then split those costs into a geographic part and
a non-distance part. The non-distance costs go into a 230-sector
Caliendo–Parro model with interprovincial migration, which is not ported
here. This directory reproduces only the estimation of those costs: Tables
1–3 and the headline 9.3 % (goods 1.3 %, services 16.8 %). Results are in
[`REPLICATION.md`](REPLICATION.md).

## Method

For each industry j and ordered region pair (n importer, i exporter) over
the 13 provinces and territories:

1. Head-Ries index: `ln τ̄_ni = L_ni / θ_j`, with the θ-free
   `L_ni = ½ ln(X_nn X_ii / (X_ni X_in))`.
2. Distance regression (DTY eq. 2):
   `L_ni = α_j ln(d_ni / √(d_nn d_ii)) + γ_n^j + γ_i^j + ε`, estimated within
   industry j. Industries with fewer than 30 pairs are pooled at the 3-digit
   level, then 2-digit, then goods vs services.
3. Non-distance cost: `NG_ni = exp((L_ni − α̂_j ln d̄_ni) / θ_j) − 1`,
   floored at 0 per region-industry pair. Reported averages are weighted
   by trade (X_ni).

Choices not stated in the paper, and what we found about each:

- **Aggregate before Head-Ries.** Computing the index per product, then
  averaging, leaves BC←ON goods at −9 %. The main causes are unwrought
  gold and territory cells that rest on a single product. Aggregating
  flows to industries first gives 0.2 % (DTY 0.0 %).
- **Internal distance.** We use Head–Mayer population-weighted distances
  between and within regions, over all pairs of 2021 Census dissemination
  areas: `d_ab = (Σ_k Σ_l s_k s_l d_kl^ρ)^(1/ρ)`. Area-based internal
  distance `(2/3)√(area/π)` overstates ON and QC (390 and 467 km, against
  118 and 100 km geometric). DTY's uniform-θ robustness levels pin down
  ρ = −0.5: θ = 4 and 8 give 12.8 and 5.5 %; we get 13.1 and 5.6.
- **Industries.** `supc3` groups SUPC products by their first 3 digits (96
  groups). `supply` splits each product's flows across the 2021 IOIC detail
  industries (215 with trade) using the origin region's supply shares.
  Assigning each product to its principal producer instead fits worse than
  either.
- **θ.** Goods: Fontagné, Guimbard & Orefice (2022) HS6 elasticities
  (CEPII ProTEE) mapped HS6 → NAICS (Pierce–Schott, 2007–2011) → SUPC by
  longest NAICS prefix, then averaged within each industry. Services:
  θ = 5.
- **Symmetric NG.** In our product-level tests, adding the AT-2016 Waugh
  exporter-FE asymmetry worsened the fit to Tables 1–2. DTY's PE↔YT goods
  cell is 586 % in both directions.

## Inputs

Download these into one folder (default `~/Downloads`):

| file | source |
|---|---|
| `12100101-eng.zip` | StatCan 12-10-0101, interprovincial trade flows, detail level — https://www150.statcan.gc.ca/n1/tbl/csv/12100101-eng.zip |
| `36100478-eng.zip` | StatCan 36-10-0478, supply and use tables, detail level (424 MB) — https://www150.statcan.gc.ca/n1/tbl/csv/36100478-eng.zip |
| `ProTEE_0_1.csv` | CEPII ProTEE — https://www.cepii.fr/DATA_DOWNLOAD/ProTEE/ProTEE_0_1.csv |
| `hssicnaics_20181015.zip` | Pierce–Schott HS→NAICS — https://spinup-000d1a-wp-offload-media.s3.amazonaws.com/faculty/wp-content/uploads/sites/47/2019/06/hssicnaics_20181015.zip |
| `2021_92-151_X.zip` | StatCan 2021 Geographic Attribute File, CSV — https://www12.statcan.gc.ca/census-recensement/2021/geo/aip-pia/attribute-attribs/index2021-eng.cfm?year=2021 (only needed to rebuild the distances; www12 blocks scripted downloads, so use a browser) |

`data/geography/popweighted_distance_2021.parquet` is committed. Its columns
are `(destination, source, measure, regions, value)`, with value in km:

- `measure`: one of `arith`, `rho_0.5`, `geom`, `rho_-0.5`, `harm`
  (ρ = 1, 0.5, 0, −0.5, −1).
- `regions`: 13 (NT and NU separate) or 12 (`NT_NU` merged, the AKT-2019
  convention).

## Running

```sh
cd dty
uv sync
uv run pytest
uv run python scripts/dty_replication.py [--raw DIR]   # ~20 s
uv run python scripts/build_popweighted_distance.py    # rebuilds the distances, ~45 s
```

`dty_replication.py` prints a scorecard for each industry definition and
distance measure. It also prints the province goods matrix next to DTY
Table 1, and NG by summary sector next to DTY Table 3.
