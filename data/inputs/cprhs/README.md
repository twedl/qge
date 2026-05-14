# CPRHS 2017 reference calibration

This directory holds the seven raw inputs of the Caliendo, Parro, Rossi-Hansberg
and Sarte (2017) "Impact of Regional and Sectoral Productivity Changes in the
U.S. Economy" calibration, materialized from the paper's MATLAB replication
kit by `scripts/convert_cprhs.py`.

## Provenance

Source: `CPRHS replication files/Data_and_Baseline_economies/` (the .mat and
.txt files distributed with the paper). Labels come from `Readme.pdf` of the
replication kit. Sectoral dispersion (1/θ_j) is the vector hardcoded in
`CPRHS_Benchmark.m` lines 19-21.

## Schema

All files are long-form parquet with a `value` column. Categorical columns
hold human-readable sector and region names.

| file                          | columns                                  | rows  |
|-------------------------------|------------------------------------------|------:|
| `bilateral_trade.parquet`     | `sector, destination, source, value`     | 65000 |
| `employment.parquet`          | `sector, region, value`                  |  1300 |
| `io_matrix.parquet`           | `source_sector, dest_sector, value`      |   676 |
| `value_added_share.parquet`   | `sector, region, value` (γ)              |  1300 |
| `structures_share.parquet`    | `region, value` (B; per-state)           |    50 |
| `final_demand_share.parquet`  | `sector, region, value` (α)              |  1300 |
| `portfolio_share.parquet`     | `region, value` (ι)                      |    50 |
| `sectoral_dispersion.parquet` | `sector, value` (1/θ)                    |    26 |

Sectors and regions are taken from `employment.parquet` in first-appearance
order; all other files must use the same label set. Loaded via
`qge.io.load_inputs(directory)`.
