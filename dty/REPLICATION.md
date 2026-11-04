# Replication

Replicates the non-distance trade-cost estimates in Díez, F. J., Tombe, T., & Yang, Y. (2026). Barriers Within Borders: Internal Trade and Domestic Market Integration. IMF Working Paper WP/26/194.

All numbers come from `uv run python scripts/dty_replication.py`. Method and inputs are in [`README.md`](README.md).

## Versions

- **Paper**: values printed in DTY (text, Tables 1–3).
- **supc3**: SUPC products grouped by their first 3 digits (96 industries, 7,564 pairs with a defined index).
- **supply**: product flows split across 2021 IOIC detail industries by the origin's supply shares from 36-10-0478 (215 industries, 17,780 pairs).

Each version uses one of three population-weighted distance measures: ρ = 0.5, geometric (ρ = 0), or ρ = −0.5. **Headline: ρ = −0.5**, the one that matches DTY's uniform-θ robustness levels.

## Data sources

| Primitive | Paper | Ours |
|---|---|---|
| Bilateral flows, own-region purchases | StatCan detail-level flows, 2021, ~500 products | StatCan 12-10-0101 (2021), all MPG, ENE and MPS products |
| Industries | 230 detail-level sectors | `supc3` or `supply` (above) |
| Distance d_ni, internal d_nn | not stated | Head–Mayer power means over 2021 Census DA pairs (GAF 92-151-X) |
| Goods θ | Fontagné, Guimbard & Orefice (2022), product level | ProTEE HS6 → Pierce–Schott NAICS → SUPC, averaged within industry |
| Services θ | 5 | 5 |
| Zero floor | not stated (cells at 0.0; Fig. 3b spike at 0) | NG ≥ 0 per region-industry pair |

## Headline levels (trade-weighted %)

| | Paper | supc3 ρ=0.5 | supc3 geom | **supc3 ρ=−0.5** | supply ρ=0.5 | supply geom | **supply ρ=−0.5** |
|---|---:|---:|---:|---:|---:|---:|---:|
| Non-distance, all | 9.3 | 15.0 | 12.0 | **9.3** | 13.9 | 11.0 | **8.3** |
| Non-distance, uniform θ = 4 | 12.8 | 21.7 | 17.1 | **13.1** | 21.3 | 17.0 | **13.0** |
| Non-distance, uniform θ = 8 | 5.5 | 9.3 | 7.3 | **5.6** | 8.9 | 7.2 | **5.5** |
| Non-distance, goods | 1.3 | 2.3 | 1.3 | **0.7** | 2.5 | 1.8 | **1.2** |
| Non-distance, services | 16.8 | 24.5 | 20.0 | **15.7** | 22.3 | 17.8 | **13.5** |
| Total τ̄ (Head-Ries) | 79 | 62.0 | 62.0 | **62.0** | 65.7 | 65.7 | **65.7** |

The uniform-θ rows do not depend on the θ concordance, so they are the cleanest check on the distance measure. Total τ̄ does not depend on distance at all. Its gap to 79 % comes from goods θ: the ProTEE HS6 averages are high, about 13 on average. FGO's pooled sector-level elasticities average 5.2 (GTAP) and 6.2 (ICIO). In an earlier scratch run of `supply` with ICIO θ (one that still left out the SUPC codes containing letters), total τ̄ rose to 74 %, and goods non-distance cost rose to 2.0 % (ρ = −0.5) and 3.2 % (geometric).

## Fit to Tables 1–2 (bilateral averages)

Spearman rank correlation and mean absolute error (pp) against DTY Tables 1 (goods) and 2 (services). "Province" covers the 90 province–province cells; "territory" covers cells involving YT, NT or NU.

| | supc3 ρ=0.5 | supc3 geom | supc3 ρ=−0.5 | supply ρ=0.5 | supply geom | supply ρ=−0.5 |
|---|---:|---:|---:|---:|---:|---:|
| Goods, province: Spearman | 0.83 | 0.82 | 0.79 | 0.79 | 0.75 | 0.72 |
| Goods, province: MAE | 4.9 | 3.7 | 3.1 | 6.4 | 4.8 | 3.9 |
| Goods, territory: Spearman | 0.33 | 0.33 | 0.18 | 0.36 | 0.27 | 0.13 |
| Services, province: Spearman | 0.90 | 0.88 | 0.81 | 0.93 | 0.90 | 0.83 |
| Services, province: MAE | 9.0 | 7.8 | 10.7 | 5.8 | 7.0 | 12.6 |
| Services, territory: Spearman | 0.85 | 0.84 | 0.82 | 0.90 | 0.89 | 0.85 |
| Table 3 sectors: Spearman | 0.82 | 0.82 | 0.79 | 0.78 | 0.74 | 0.71 |

## Table 1 — Goods, province pairs (%, rows = importer, columns = exporter)

### Paper

| | BC | AB | SK | MB | ON | QC | NB | NS | PE | NL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BC | – | 0.5 | 1.7 | 0.5 | 0.0 | 0.0 | 0.8 | 0.7 | 4.4 | 11.2 |
| AB | 0.9 | – | 1.4 | 0.5 | 0.1 | 0.1 | 1.1 | 0.4 | 7.7 | 2.9 |
| SK | 3.8 | 2.9 | – | 2.9 | 1.2 | 1.6 | 2.6 | 3.3 | 11.3 | 37.0 |
| MB | 1.1 | 0.5 | 1.3 | – | 0.5 | 0.7 | 2.4 | 0.7 | 8.1 | 1.9 |
| ON | 0.1 | 0.0 | 0.1 | 0.1 | – | 0.6 | 1.0 | 0.4 | 2.4 | 1.9 |
| QC | 0.5 | 0.1 | 1.1 | 0.2 | 0.9 | – | 4.7 | 4.1 | 21.8 | 15.1 |
| NB | 2.4 | 0.2 | 1.9 | 0.7 | 3.0 | 9.9 | – | 9.6 | 11.4 | 1.0 |
| NS | 0.4 | 0.4 | 1.7 | 2.4 | 1.4 | 4.6 | 5.3 | – | 17.0 | 10.2 |
| PE | 3.4 | 4.4 | 13.5 | 7.3 | 4.9 | 14.3 | 1.0 | 9.3 | – | 53.0 |
| NL | 1.3 | 4.5 | 25.7 | 16.9 | 8.4 | 6.5 | 3.6 | 5.7 | 9.6 | – |

### supc3, ρ = −0.5

| | BC | AB | SK | MB | ON | QC | NB | NS | PE | NL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BC | – | 1.0 | 0.2 | 0.2 | 0.2 | 0.0 | 0.2 | 0.3 | 7.3 | 4.7 |
| AB | 0.1 | – | 0.3 | 0.1 | 0.0 | 0.1 | 0.1 | 0.2 | 7.1 | 15.5 |
| SK | 1.1 | 0.4 | – | 0.1 | 10.5 | 1.2 | 6.9 | 1.1 | 12.4 | 30.1 |
| MB | 0.1 | 0.0 | 0.0 | – | 0.0 | 0.1 | 0.2 | 0.1 | 0.5 | 8.8 |
| ON | 0.0 | 0.8 | 0.0 | 0.0 | – | 0.3 | 1.7 | 0.3 | 1.2 | 5.3 |
| QC | 0.0 | 0.1 | 0.4 | 0.2 | 0.2 | – | 3.2 | 1.1 | 9.3 | 4.2 |
| NB | 0.6 | 0.1 | 2.1 | 0.6 | 1.5 | 3.7 | – | 1.6 | 4.4 | 1.0 |
| NS | 0.1 | 0.2 | 0.6 | 0.2 | 1.2 | 3.0 | 1.4 | – | 3.0 | 0.9 |
| PE | 20.8 | 4.3 | 12.7 | 1.0 | 2.5 | 13.4 | 2.6 | 5.0 | – | 16.7 |
| NL | 10.3 | 9.6 | 30.1 | 10.0 | 8.8 | 6.8 | 9.5 | 4.7 | 11.1 | – |

### supply, ρ = −0.5

| | BC | AB | SK | MB | ON | QC | NB | NS | PE | NL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BC | – | 0.1 | 2.3 | 0.6 | 0.2 | 0.6 | 1.4 | 0.2 | 20.0 | 2.1 |
| AB | 0.2 | – | 0.4 | 0.5 | 0.9 | 0.8 | 0.9 | 0.6 | 30.6 | 12.4 |
| SK | 1.7 | 0.5 | – | 0.9 | 1.9 | 4.2 | 6.8 | 2.7 | 17.3 | 15.8 |
| MB | 0.5 | 0.4 | 1.1 | – | 1.0 | 1.9 | 0.8 | 1.3 | 57.2 | 7.3 |
| ON | 0.2 | 0.5 | 0.1 | 1.1 | – | 1.1 | 0.6 | 0.7 | 7.4 | 3.2 |
| QC | 0.3 | 0.5 | 5.9 | 2.9 | 1.3 | – | 1.5 | 1.0 | 11.3 | 3.3 |
| NB | 1.6 | 2.9 | 14.5 | 2.8 | 4.6 | 2.7 | – | 2.0 | 9.0 | 1.0 |
| NS | 0.5 | 3.3 | 0.7 | 1.4 | 2.2 | 3.3 | 4.9 | – | 9.0 | 0.3 |
| PE | 9.8 | 15.0 | 13.2 | 4.2 | 9.1 | 12.7 | 1.2 | 4.1 | – | 48.3 |
| NL | 2.8 | 13.2 | 28.8 | 11.3 | 10.6 | 8.5 | 4.2 | 1.9 | 24.0 | – |

## Table 3 — Summary-level sectors (%, ρ = −0.5)

Sectors are assigned by NAICS-style code prefix. Government and NPISH industries are excluded, as in DTY. "–" means no pairs with a defined index and θ.

| Sector | Paper | supc3 | supply |
|---|---:|---:|---:|
| Crop and animal production | 0.0 | 0.0 | 0.2 |
| Forestry and logging | 2.4 | 2.7 | 0.0 |
| Fishing, hunting and trapping | 15.5 | 4.9 | 5.1 |
| Support activities for agriculture and forestry | 22.9 | 59.1 | 26.2 |
| Mining, quarrying, and oil and gas extraction | 0.0 | 9.2 | 16.8 |
| Utilities | 36.2 | – | 9.4 |
| Other activities of the construction industry | 18.4 | – | 5.2 |
| Manufacturing | 0.3 | 0.9 | 1.1 |
| Wholesale trade | 1.1 | 0.4 | 0.2 |
| Retail trade | 60.6 | 61.9 | 55.0 |
| Transportation and warehousing | 12.5 | 3.6 | 3.2 |
| Information and cultural industries | 38.2 | 28.8 | 24.8 |
| Finance, insurance, real estate, rental and leasing | 13.1 | 38.3 | 26.8 |
| Professional, scientific and technical services | 6.0 | 2.0 | 4.1 |
| Administrative and support, waste management | 3.5 | 2.6 | 6.6 |
| Educational services | 41.8 | 20.3 | – |
| Health care and social assistance | 55.5 | 18.7 | 20.5 |
| Arts, entertainment and recreation | 17.0 | 4.1 | 6.4 |
| Accommodation and food services | 23.6 | 21.4 | 18.5 |
| Other services (except public administration) | 16.4 | 15.9 | 19.8 |

## Open issues

- **Goods total τ̄ vs goods NG.** Lower, sector-level θ brings total τ̄ toward 79 %, but it also pushes goods NG above 1.3 %. No θ source we tried hits both.
- **Goods rows and columns for small provinces.** The NL import row and the PE export column run high (e.g. MB←PE 57 % in `supply`, DTY 8 %), as does SK←ON in `supc3` (10.5 %, DTY 1.2 %).
- **Territories.** Territory goods cells barely track Table 1 (Spearman 0.1–0.4). Many rest on one or two industries.
- **Sectors.** FIRE and mining come out too high. Education, health, utilities and transport come out too low. In `supc3`, utilities and construction have no defined cells, and so does education in `supply`.
- **Industry definition.** The coarser `supc3` grouping fits the goods pattern better than the 215 IOIC industries, even though DTY use 230 detail sectors. How DTY move from products to sectors is not stated.
