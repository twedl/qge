# Replication

Replicates Tombe, T., & Winter, J. (2021). Fiscal integration with internal trade: Quantifying the effects of federal transfers in Canada. *Canadian Journal of Economics*, 54(2), 521–556.

## Versions

- **Paper** — values printed in Tables 4–8 of Tombe & Winter (2021).
- **MAT'10** — our Python port of the model run on TW's published `maindata.mat` (2010 calibration).
- **Rep'10** — our Python port run on a 2010 calibration rebuilt independently from primary StatCan and OECD ICIO sources.
- **Full'22** — our Python port run on a 2022 calibration built from primary sources using the same pipeline as Rep'10.

## Data sources

| Primitive | MAT'10 | Rep'10 | Full'22 |
|---|---|---|---|
| Bilateral trade π | `maindata.mat` | StatCan 12-10-0088 (2010), M+G+N product codes | StatCan 12-10-0088 (2022), M+G+N product codes |
| Employment λ | `maindata.mat` | StatCan 14-10-0327 (2010) | StatCan 14-10-0327 (2022) |
| GDP, population | `maindata.mat` | StatCan 36-10-0222 + 17-10-0005 (2010) | StatCan 36-10-0222 + 17-10-0005 (2022) |
| Spatial price index | `maindata.mat` | StatCan 18-10-0003 (2010) | StatCan 18-10-0003 (2019; table retired) |
| Fiscal transfers T | `maindata.mat` | StatCan 36-10-0450 (2010, single year, matching TW §3.3) | StatCan 36-10-0450 (2022, single year) |
| WLD intra-trade (ICIO row) | `maindata.mat` | OECD ICIO 2010 SML | OECD ICIO 2022 SML |
| Industrial structure (α, γ, φ, IO) | `maindata.mat` | National L97 IOT 2010 (StatCan 15-211-X) | National Detail IOT 2022 (StatCan 15-211-X) |
| Trade-cost shocks (`tauhat_*`) | `maindata.mat` (from AT-2016) | rebuilt from 2010 PI via AT-2016 method | rebuilt from 2022 PI via AT-2016 method |
| Trade elasticity θ | `maindata.mat` (AT-2016 Table 9) | `maindata.mat` (carried over) | `maindata.mat` (carried over) |

## Table 4 — Effect of fiscal transfers (real income, %)

### Col 1 — All transfers

| Prov | Paper | MAT'10 | Rep'10 | Full'22 |
|---|---:|---:|---:|---:|
| BC | -1.7 | -1.67 | -1.18 | -2.63 |
| AB | -8.3 | -8.31 | -7.41 | -4.26 |
| SK | -0.9 | -0.85 | -0.42 | +1.34 |
| MB | +7.2 | +7.20 | +7.97 | +11.82 |
| ON | -0.8 | -0.77 | -0.30 | -1.88 |
| QC | +3.4 | +3.39 | +3.96 | +4.45 |
| NB | +15.6 | +15.55 | +17.08 | +21.04 |
| NS | +17.1 | +17.12 | +19.54 | +23.01 |
| PE | +29.8 | +29.75 | +35.66 | +60.46 |
| NL | +10.4 | +10.40 | +9.93 | +9.38 |
| **CAN** | **-0.7** | **-0.74** | **-0.76** | **-0.19** |

### Col 2 — Equalization only

| Prov | Paper | MAT'10 | Rep'10 | Full'22 |
|---|---:|---:|---:|---:|
| BC | -0.9 | -0.87 | -0.78 | -0.76 |
| AB | -1.1 | -1.11 | -0.92 | -0.97 |
| SK | -1.0 | -1.04 | -0.88 | -0.99 |
| MB | +2.3 | +2.29 | +2.44 | +2.58 |
| ON | -0.8 | -0.81 | -0.70 | -0.64 |
| QC | +1.6 | +1.64 | +1.74 | +1.82 |
| NB | +3.9 | +3.91 | +4.17 | +4.18 |
| NS | +1.7 | +1.65 | +1.85 | +1.87 |
| PE | +4.6 | +4.58 | +4.97 | +5.25 |
| NL | -1.0 | -1.03 | -0.89 | -0.85 |
| **CAN** | **-0.1** | **-0.13** | **-0.08** | **-0.03** |

### Col 3 — Income-sensitive only

| Prov | Paper | MAT'10 | Rep'10 | Full'22 |
|---|---:|---:|---:|---:|
| BC | +5.0 | +5.02 | +6.01 | +5.64 |
| AB | -10.1 | -10.08 | -10.53 | -12.86 |
| SK | -7.2 | -7.18 | -8.74 | -11.30 |
| MB | +5.6 | +5.64 | +4.91 | +7.08 |
| ON | -0.3 | -0.34 | +0.61 | +0.95 |
| QC | +5.7 | +5.66 | +8.45 | +9.00 |
| NB | +0.8 | +0.81 | +3.88 | +9.00 |
| NS | +5.3 | +5.25 | +12.94 | +19.06 |
| PE | +16.3 | +16.30 | +20.39 | +21.58 |
| NL | -2.1 | -2.09 | -5.47 | -0.85 |
| **CAN** | **-1.1** | **-1.07** | **-1.33** | **-0.96** |

## Table 5 — Gains from observed trade vs autarky (real income, %)

| Prov | Paper | MAT'10 | Rep'10 | Full'22 |
|---|---:|---:|---:|---:|
| BC | +29.9 | +29.86 | +29.31 | +24.98 |
| AB | +6.0 | +6.02 | +11.79 | +6.96 |
| SK | +19.0 | +19.02 | +20.14 | +8.56 |
| MB | +36.4 | +36.38 | +33.42 | +28.60 |
| ON | +11.3 | +11.33 | +13.47 | +15.61 |
| QC | +21.1 | +21.11 | +20.28 | +19.96 |
| NB | +63.4 | +63.37 | +43.52 | +47.99 |
| NS | +56.9 | +56.94 | +47.68 | +55.58 |
| PE | +66.5 | +66.46 | +56.13 | +46.94 |
| NL | +35.7 | +35.70 | +22.28 | +26.21 |
| **CAN** | **+16.4** | **+16.41** | **+17.58** | **+16.34** |

## Table 6 — Lower internal trade costs (real income, %)

### Asymmetric

| Prov | Paper | MAT'10 | Rep'10 | Full'22 |
|---|---:|---:|---:|---:|
| BC | +2.8 | +2.76 | +2.23 | +2.51 |
| AB | +3.1 | +3.10 | +2.76 | +2.71 |
| SK | +3.9 | +3.94 | +2.17 | +3.11 |
| MB | +3.6 | +3.58 | +3.44 | +4.20 |
| ON | +4.1 | +4.09 | +4.95 | +5.17 |
| QC | +1.9 | +1.92 | +2.28 | +2.35 |
| NB | +2.8 | +2.81 | +2.47 | +3.62 |
| NS | +2.1 | +2.09 | +2.66 | +3.61 |
| PE | +6.5 | +6.48 | +9.05 | +9.61 |
| NL | +3.2 | +3.22 | +2.66 | +3.72 |
| **CAN** | **+3.2** | **+3.18** | **+3.32** | **+3.57** |

### Non-distance

| Prov | Paper | MAT'10 | Rep'10 | Full'22 |
|---|---:|---:|---:|---:|
| BC | +4.9 | +4.86 | +11.17 | +11.92 |
| AB | +6.3 | +6.34 | +11.28 | +12.66 |
| SK | +15.9 | +15.90 | +23.58 | +27.84 |
| MB | +7.9 | +7.90 | +22.30 | +28.96 |
| ON | +3.9 | +3.87 | +9.93 | +10.42 |
| QC | +7.0 | +6.98 | +19.59 | +21.42 |
| NB | +19.2 | +19.21 | +25.13 | +30.92 |
| NS | +14.4 | +14.35 | +18.43 | +22.03 |
| PE | +23.9 | +23.92 | +29.77 | +34.70 |
| NL | +24.3 | +24.27 | +25.27 | +29.03 |
| **CAN** | **+7.3** | **+7.30** | **+14.49** | **+16.09** |

### All internal

| Prov | Paper | MAT'10 | Rep'10 | Full'22 |
|---|---:|---:|---:|---:|
| BC | +59.3 | +59.34 | +54.14 | +56.91 |
| AB | +56.8 | +56.80 | +47.83 | +50.10 |
| SK | +74.2 | +74.17 | +61.23 | +71.09 |
| MB | +81.5 | +81.54 | +70.64 | +80.92 |
| ON | +38.4 | +38.38 | +38.43 | +39.77 |
| QC | +44.9 | +44.86 | +45.33 | +48.95 |
| NB | +79.8 | +79.83 | +82.91 | +96.35 |
| NS | +77.2 | +77.21 | +84.72 | +94.43 |
| PE | +144.9 | +144.93 | +134.61 | +148.11 |
| NL | +102.4 | +102.38 | +87.73 | +98.03 |
| **CAN** | **+55.9** | **+55.91** | **+51.15** | **+54.44** |

## Table 7 — Unilateral 10% trade-cost reduction (real income at target province, %)

### Import

| Prov | Paper | MAT'10 | Rep'10 | Full'22 |
|---|---:|---:|---:|---:|
| BC | +4.3 | +4.31 | +4.08 | +4.39 |
| AB | +4.6 | +4.56 | +3.80 | +4.58 |
| SK | +5.4 | +5.40 | +4.45 | +5.17 |
| MB | +4.8 | +4.77 | +4.51 | +4.86 |
| ON | +4.4 | +4.38 | +3.99 | +4.00 |
| QC | +4.4 | +4.38 | +4.14 | +4.33 |
| NB | +5.0 | +4.95 | +4.94 | +5.71 |
| NS | +4.1 | +4.06 | +4.48 | +4.79 |
| PE | +4.9 | +4.94 | +5.16 | +5.40 |
| NL | +6.4 | +6.39 | +5.26 | +5.66 |

### Export

| Prov | Paper | MAT'10 | Rep'10 | Full'22 |
|---|---:|---:|---:|---:|
| BC | +1.0 | +1.00 | +0.80 | +1.18 |
| AB | +1.6 | +1.57 | +0.70 | +1.59 |
| SK | +1.8 | +1.84 | +0.85 | +1.66 |
| MB | +1.1 | +1.12 | +0.88 | +1.26 |
| ON | +2.1 | +2.09 | +1.63 | +1.62 |
| QC | +1.4 | +1.41 | +1.21 | +1.38 |
| NB | +1.2 | +1.15 | +1.20 | +1.94 |
| NS | +0.3 | +0.30 | +0.79 | +1.08 |
| PE | +1.1 | +1.12 | +1.39 | +1.66 |
| NL | +2.8 | +2.79 | +1.60 | +2.06 |

## Table 8 — Bilateral 10% liberalization (real income at target provinces, %)

### Panel A — with transfers

| Scenario | Prov | Paper | MAT'10 | Rep'10 | Full'22 |
|---|---|---:|---:|---:|---:|
| New West | BC | +1.26 | +1.26 | +1.27 | +1.30 |
| | AB | +1.07 | +1.07 | +0.93 | +0.99 |
| | SK | +2.24 | +2.24 | +1.86 | +1.86 |
| ON–QC | ON | +1.11 | +1.11 | +0.96 | +0.83 |
| | QC | +1.82 | +1.82 | +1.67 | +1.63 |
| Maritime | NB | +0.47 | +0.47 | +0.48 | +0.55 |
| | NS | +0.44 | +0.44 | +0.58 | +0.63 |
| | PE | +1.80 | +1.80 | +1.90 | +2.28 |

### Panel B — no transfers

| Scenario | Prov | Paper | MAT'10 | Rep'10 | Full'22 |
|---|---|---:|---:|---:|---:|
| New West | BC | +1.53 | +1.53 | +1.54 | +1.52 |
| | AB | +1.37 | +1.37 | +1.26 | +1.40 |
| | SK | +3.17 | +3.17 | +3.04 | +3.21 |
| ON–QC | ON | +1.22 | +1.22 | +1.07 | +0.90 |
| | QC | +2.39 | +2.39 | +2.24 | +2.23 |
| Maritime | NB | +1.07 | +1.07 | +1.09 | +1.25 |
| | NS | +1.01 | +1.01 | +1.19 | +1.14 |
| | PE | +3.19 | +3.19 | +3.88 | +8.97 |
