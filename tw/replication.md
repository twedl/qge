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
| Bilateral trade π | `maindata.mat` | StatCan 12-10-0088 (2010) | StatCan 12-10-0088 (2022) |
| Employment λ | `maindata.mat` | StatCan 14-10-0327 (2010) | StatCan 14-10-0327 (2022) |
| GDP, population | `maindata.mat` | StatCan 36-10-0222 + 17-10-0005 (2010) | StatCan 36-10-0222 + 17-10-0005 (2022) |
| Spatial price index | `maindata.mat` | StatCan 18-10-0003 (2010) | StatCan 18-10-0003 (2019; table retired) |
| Fiscal transfers T | `maindata.mat` | StatCan 36-10-0450 (2008–2012 avg) | StatCan 36-10-0450 (2020–2024 avg) |
| WLD intra-trade (ICIO row) | `maindata.mat` | OECD ICIO 2010 SML | OECD ICIO 2022 SML |
| Industrial structure (α, γ, φ, IO) | `maindata.mat` | National L97 IOT 2010 (StatCan 15-211-X) | National Detail IOT 2022 (StatCan 15-211-X) |
| Trade-cost shocks (`tauhat_*`) | `maindata.mat` (from AT-2016) | rebuilt from 2010 PI via AT-2016 method | rebuilt from 2022 PI via AT-2016 method |
| Trade elasticity θ | `maindata.mat` (AT-2016 Table 9) | `maindata.mat` (carried over) | `maindata.mat` (carried over) |

## Table 4 — Effect of fiscal transfers (real income, %)

### Col 1 — All transfers

| Prov | Paper | MAT'10 | Rep'10 | Full'22 |
|---|---:|---:|---:|---:|
| BC | -1.7 | -1.67 | -1.57 | -2.38 |
| AB | -8.3 | -8.31 | -7.13 | -5.34 |
| SK | -0.9 | -0.85 | -0.06 | +0.78 |
| MB | +7.2 | +7.20 | +8.34 | +8.45 |
| ON | -0.8 | -0.77 | -0.74 | -1.48 |
| QC | +3.4 | +3.39 | +4.62 | +5.15 |
| NB | +15.6 | +15.55 | +17.45 | +19.91 |
| NS | +17.1 | +17.12 | +21.92 | +21.98 |
| PE | +29.8 | +29.75 | +33.21 | +33.28 |
| NL | +10.4 | +10.40 | +8.38 | +7.78 |
| **CAN** | **-0.7** | **-0.74** | **-0.72** | **-0.26** |

### Col 2 — Equalization only

| Prov | Paper | MAT'10 | Rep'10 | Full'22 |
|---|---:|---:|---:|---:|
| BC | -0.9 | -0.87 | -0.77 | -0.76 |
| AB | -1.1 | -1.11 | -0.92 | -0.97 |
| SK | -1.0 | -1.04 | -0.88 | -0.99 |
| MB | +2.3 | +2.29 | +2.43 | +2.56 |
| ON | -0.8 | -0.81 | -0.69 | -0.63 |
| QC | +1.6 | +1.64 | +1.74 | +1.81 |
| NB | +3.9 | +3.91 | +4.14 | +4.16 |
| NS | +1.7 | +1.65 | +1.85 | +1.86 |
| PE | +4.6 | +4.58 | +4.92 | +5.19 |
| NL | -1.0 | -1.03 | -0.88 | -0.85 |
| **CAN** | **-0.1** | **-0.13** | **-0.08** | **-0.02** |

### Col 3 — Income-sensitive only

| Prov | Paper | MAT'10 | Rep'10 | Full'22 |
|---|---:|---:|---:|---:|
| BC | +5.0 | +5.02 | +5.63 | +5.26 |
| AB | -10.1 | -10.08 | -10.44 | -12.91 |
| SK | -7.2 | -7.18 | -8.25 | -11.25 |
| MB | +5.6 | +5.64 | +5.17 | +7.83 |
| ON | -0.3 | -0.34 | +0.33 | +0.89 |
| QC | +5.7 | +5.66 | +8.67 | +9.28 |
| NB | +0.8 | +0.81 | +4.20 | +8.91 |
| NS | +5.3 | +5.25 | +12.12 | +18.00 |
| PE | +16.3 | +16.30 | +20.85 | +21.93 |
| NL | -2.1 | -2.09 | -5.13 | -0.73 |
| **CAN** | **-1.1** | **-1.07** | **-1.32** | **-0.96** |

## Table 5 — Gains from observed trade vs autarky (real income, %)

| Prov | Paper | MAT'10 | Rep'10 | Full'22 |
|---|---:|---:|---:|---:|
| BC | +29.9 | +29.86 | +29.66 | +25.27 |
| AB | +6.0 | +6.02 | +11.96 | +6.92 |
| SK | +19.0 | +19.02 | +20.67 | +8.51 |
| MB | +36.4 | +36.38 | +34.16 | +29.43 |
| ON | +11.3 | +11.33 | +13.71 | +15.88 |
| QC | +21.1 | +21.11 | +20.73 | +20.38 |
| NB | +63.4 | +63.37 | +44.79 | +48.94 |
| NS | +56.9 | +56.94 | +48.47 | +56.34 |
| PE | +66.5 | +66.46 | +58.14 | +48.56 |
| NL | +35.7 | +35.70 | +22.59 | +26.46 |
| **CAN** | **+16.4** | **+16.41** | **+17.90** | **+16.58** |

## Table 6 — Lower internal trade costs (real income, %)

### Asymmetric

| Prov | Paper | MAT'10 | Rep'10 | Full'22 |
|---|---:|---:|---:|---:|
| BC | +2.8 | +2.76 | +2.28 | +2.55 |
| AB | +3.1 | +3.10 | +2.88 | +2.78 |
| SK | +3.9 | +3.94 | +2.57 | +3.40 |
| MB | +3.6 | +3.58 | +3.75 | +4.62 |
| ON | +4.1 | +4.09 | +5.15 | +5.28 |
| QC | +1.9 | +1.92 | +2.37 | +2.42 |
| NB | +2.8 | +2.81 | +2.69 | +3.77 |
| NS | +2.1 | +2.09 | +2.93 | +3.81 |
| PE | +6.5 | +6.48 | +9.88 | +10.38 |
| NL | +3.2 | +3.22 | +2.94 | +3.94 |
| **CAN** | **+3.2** | **+3.18** | **+3.48** | **+3.68** |

### Non-distance

| Prov | Paper | MAT'10 | Rep'10 | Full'22 |
|---|---:|---:|---:|---:|
| BC | +4.9 | +4.86 | +10.87 | +11.68 |
| AB | +6.3 | +6.34 | +11.06 | +12.44 |
| SK | +15.9 | +15.90 | +23.59 | +27.86 |
| MB | +7.9 | +7.90 | +22.13 | +29.08 |
| ON | +3.9 | +3.87 | +9.72 | +10.26 |
| QC | +7.0 | +6.98 | +19.59 | +21.49 |
| NB | +19.2 | +19.21 | +24.84 | +30.35 |
| NS | +14.4 | +14.35 | +17.86 | +21.45 |
| PE | +23.9 | +23.92 | +29.37 | +34.41 |
| NL | +24.3 | +24.27 | +25.41 | +28.91 |
| **CAN** | **+7.3** | **+7.30** | **+14.33** | **+15.97** |

### All internal

| Prov | Paper | MAT'10 | Rep'10 | Full'22 |
|---|---:|---:|---:|---:|
| BC | +59.3 | +59.34 | +53.31 | +56.20 |
| AB | +56.8 | +56.80 | +47.52 | +49.76 |
| SK | +74.2 | +74.17 | +61.67 | +71.28 |
| MB | +81.5 | +81.54 | +70.52 | +81.33 |
| ON | +38.4 | +38.38 | +37.94 | +39.42 |
| QC | +44.9 | +44.86 | +45.19 | +48.88 |
| NB | +79.8 | +79.83 | +82.49 | +95.25 |
| NS | +77.2 | +77.21 | +83.19 | +92.95 |
| PE | +144.9 | +144.93 | +133.95 | +147.86 |
| NL | +102.4 | +102.38 | +88.19 | +98.02 |
| **CAN** | **+55.9** | **+55.91** | **+50.80** | **+54.12** |

## Table 7 — Unilateral 10% trade-cost reduction (real income at target province, %)

### Import

| Prov | Paper | MAT'10 | Rep'10 | Full'22 |
|---|---:|---:|---:|---:|
| BC | +4.3 | +4.31 | +4.13 | +4.45 |
| AB | +4.6 | +4.56 | +3.95 | +4.71 |
| SK | +5.4 | +5.40 | +4.76 | +5.39 |
| MB | +4.8 | +4.77 | +4.63 | +5.06 |
| ON | +4.4 | +4.38 | +4.07 | +4.09 |
| QC | +4.4 | +4.38 | +4.26 | +4.44 |
| NB | +5.0 | +4.95 | +5.19 | +5.87 |
| NS | +4.1 | +4.06 | +4.61 | +4.88 |
| PE | +4.9 | +4.94 | +5.35 | +5.62 |
| NL | +6.4 | +6.39 | +5.54 | +5.87 |

### Export

| Prov | Paper | MAT'10 | Rep'10 | Full'22 |
|---|---:|---:|---:|---:|
| BC | +1.0 | +1.00 | +0.84 | +1.23 |
| AB | +1.6 | +1.57 | +0.83 | +1.72 |
| SK | +1.8 | +1.84 | +1.14 | +1.88 |
| MB | +1.1 | +1.12 | +0.99 | +1.45 |
| ON | +2.1 | +2.09 | +1.70 | +1.69 |
| QC | +1.4 | +1.41 | +1.30 | +1.47 |
| NB | +1.2 | +1.15 | +1.42 | +2.08 |
| NS | +0.3 | +0.30 | +0.89 | +1.15 |
| PE | +1.1 | +1.12 | +1.55 | +1.85 |
| NL | +2.8 | +2.79 | +1.87 | +2.25 |

## Table 8 — Bilateral 10% liberalization (real income at target provinces, %)

### Panel A — with transfers

| Scenario | Prov | Paper | MAT'10 | Rep'10 | Full'22 |
|---|---|---:|---:|---:|---:|
| New West | BC | +1.26 | +1.26 | +1.29 | +1.32 |
| | AB | +1.07 | +1.07 | +0.97 | +1.03 |
| | SK | +2.24 | +2.24 | +1.98 | +1.96 |
| ON–QC | ON | +1.11 | +1.11 | +0.97 | +0.84 |
| | QC | +1.82 | +1.82 | +1.74 | +1.69 |
| Maritime | NB | +0.47 | +0.47 | +0.55 | +0.60 |
| | NS | +0.44 | +0.44 | +0.63 | +0.67 |
| | PE | +1.80 | +1.80 | +2.07 | +2.47 |

### Panel B — no transfers

| Scenario | Prov | Paper | MAT'10 | Rep'10 | Full'22 |
|---|---|---:|---:|---:|---:|
| New West | BC | +1.53 | +1.53 | +1.54 | +1.57 |
| | AB | +1.37 | +1.37 | +1.30 | +1.40 |
| | SK | +3.17 | +3.17 | +3.16 | +3.27 |
| ON–QC | ON | +1.22 | +1.22 | +1.08 | +0.92 |
| | QC | +2.39 | +2.39 | +2.34 | +2.31 |
| Maritime | NB | +1.07 | +1.07 | +1.15 | +1.27 |
| | NS | +1.01 | +1.01 | +1.32 | +1.15 |
| | PE | +3.19 | +3.19 | +3.74 | +5.68 |
