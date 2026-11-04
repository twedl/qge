# Counterfactuals

## Alberta separation grid

Driver script: `scripts/counterfactual_ab_grid.py`.

Every scenario sets **fiscal = off**: T_AB → 0, redistributed per worker among the other 9 provinces (`scripts/counterfactual_ab_scenarios.py::transfer_exit_sn_shift`). Two remaining levers; 3 × 3 = 9 scenarios.

### Lever definitions

- **Labour** ∈ {normal, medium, frozen}
  - `normal` — κ = 1.5 for all 10 provinces (model default).
  - `medium` — κ_AB = 0.75 (half elasticity), other 9 provinces stay at 1.5. Implemented via `kappa_override` argument added to `qge.equations.model_equations` and `qge.solver.solve`.
  - `frozen` — `frozen_idx = (AB,)`: AB labour pinned at baseline λ_AB; remaining 1 − λ_AB mass distributed across the other 9 by Fréchet.
- **Trade** ∈ {normal, +10%, WLD-walls}
  - `+10%` — uniform `tauhat[AB↔p, j] = 1.10` for all 9 RoC provinces p and all 22 sectors j; WLD pairs untouched (`scripts/counterfactual_ab_grid.py::build_tauhat_uniform_10pct`).
  - `WLD-walls` — AB↔RoC non-distance friction raised to the AB↔WLD level (`scripts/counterfactual_ab_walls.py::build_tauhat_ab_walls`).

### Aggregate row definitions

- **CAN − AB**: real income / real wage are income-share-weighted (`ω₂`) and GDP-share-weighted (`ω`) averages renormalized to sum to 1 across the 9 non-AB provinces, then × λ₂/λ. Imports/exports are simple sums of the 9 non-AB provincial flows.
- **CAN**: full model aggregates. By trade-balance construction (Σ S_n = 0), CAN import % change equals CAN export % change.

### Results — Full'22 calibration

All metrics are % change vs baseline. Real imports / exports are deflated by the province's aggregate price index `aggPhat`.

#### Scenario 1 — Labour: normal, Trade: normal

| Prov | Δ real income | Δ real wage | Δ real imports | Δ real exports |
|---|---:|---:|---:|---:|
| BC | -0.68 | +0.02 | -0.87 | +0.62 |
| AB | +3.68 | -0.04 | +6.74 | +0.67 |
| SK | -0.42 | +0.02 | -0.31 | +0.23 |
| MB | -0.71 | +0.01 | -0.87 | +0.40 |
| ON | -0.58 | -0.00 | -0.73 | +0.46 |
| QC | -0.78 | -0.02 | -1.08 | +0.44 |
| NB | -0.69 | -0.02 | -0.86 | +0.16 |
| NS | -0.89 | -0.05 | -1.49 | +0.46 |
| PE | -0.99 | -0.03 | -1.63 | +0.14 |
| NL | -0.64 | +0.01 | -0.91 | -0.00 |
| **CAN − AB** | **-1.44** | **-0.78** | **-0.86** | **+0.42** |
| **CAN** | **+0.22** | **+0.32** | **+0.46** | **+0.46** |

#### Scenario 2 — Labour: normal, Trade: +10%

| Prov | Δ real income | Δ real wage | Δ real imports | Δ real exports |
|---|---:|---:|---:|---:|
| BC | -1.32 | -0.59 | -6.15 | -4.99 |
| AB | +2.23 | -1.78 | -9.33 | -13.38 |
| SK | -1.20 | -0.59 | -7.51 | -5.65 |
| MB | -1.30 | -0.41 | -4.37 | -3.09 |
| ON | -1.02 | -0.33 | -2.99 | -1.65 |
| QC | -1.21 | -0.26 | -3.22 | -1.47 |
| NB | -1.05 | -0.16 | -0.95 | +0.37 |
| NS | -1.27 | -0.23 | -2.12 | -0.03 |
| PE | -1.33 | -0.11 | -1.80 | +0.35 |
| NL | -0.99 | -0.08 | -1.26 | +0.09 |
| **CAN − AB** | **-1.76** | **-0.98** | **-3.91** | **-2.51** |
| **CAN** | **-0.47** | **-0.36** | **-4.83** | **-4.81** |

#### Scenario 3 — Labour: normal, Trade: WLD-walls

| Prov | Δ real income | Δ real wage | Δ real imports | Δ real exports |
|---|---:|---:|---:|---:|
| BC | -1.89 | -1.11 | -9.73 | -8.73 |
| AB | -0.00 | -4.02 | -28.76 | -29.93 |
| SK | -2.00 | -1.19 | -12.44 | -9.58 |
| MB | -1.90 | -0.87 | -7.58 | -6.32 |
| ON | -1.49 | -0.80 | -5.62 | -4.35 |
| QC | -1.63 | -0.55 | -4.94 | -3.04 |
| NB | -1.44 | -0.34 | -0.80 | +0.81 |
| NS | -1.67 | -0.46 | -2.65 | -0.46 |
| PE | -1.66 | -0.21 | -1.74 | +0.78 |
| NL | -1.36 | -0.17 | -1.45 | +0.35 |
| **CAN − AB** | **-1.93** | **-1.09** | **-6.78** | **-5.36** |
| **CAN** | **-1.32** | **-1.23** | **-10.41** | **-10.36** |

#### Scenario 4 — Labour: medium, Trade: normal

| Prov | Δ real income | Δ real wage | Δ real imports | Δ real exports |
|---|---:|---:|---:|---:|
| BC | -0.75 | -0.01 | -0.77 | +0.79 |
| AB | +3.78 | +0.05 | +5.08 | -0.98 |
| SK | -0.49 | +0.00 | -0.24 | +0.41 |
| MB | -0.78 | -0.02 | -0.73 | +0.61 |
| ON | -0.65 | -0.03 | -0.60 | +0.68 |
| QC | -0.84 | -0.04 | -0.91 | +0.68 |
| NB | -0.76 | -0.03 | -0.57 | +0.53 |
| NS | -0.96 | -0.07 | -1.25 | +0.81 |
| PE | -1.05 | -0.05 | -1.35 | +0.49 |
| NL | -0.68 | -0.00 | -0.61 | +0.36 |
| **CAN − AB** | **-1.19** | **-0.49** | **-0.76** | **+0.60** |
| **CAN** | **+0.09** | **+0.19** | **+0.27** | **+0.27** |

#### Scenario 5 — Labour: medium, Trade: +10%

| Prov | Δ real income | Δ real wage | Δ real imports | Δ real exports |
|---|---:|---:|---:|---:|
| BC | -1.35 | -0.60 | -6.06 | -4.84 |
| AB | +2.29 | -1.73 | -10.22 | -14.29 |
| SK | -1.24 | -0.60 | -7.41 | -5.50 |
| MB | -1.34 | -0.42 | -4.26 | -2.93 |
| ON | -1.06 | -0.35 | -2.89 | -1.50 |
| QC | -1.24 | -0.27 | -3.10 | -1.31 |
| NB | -1.09 | -0.16 | -0.77 | +0.59 |
| NS | -1.31 | -0.24 | -1.97 | +0.19 |
| PE | -1.36 | -0.12 | -1.63 | +0.57 |
| NL | -1.02 | -0.08 | -1.08 | +0.31 |
| **CAN − AB** | **-1.60** | **-0.80** | **-3.83** | **-2.38** |
| **CAN** | **-0.55** | **-0.44** | **-4.91** | **-4.88** |

#### Scenario 6 — Labour: medium, Trade: WLD-walls

| Prov | Δ real income | Δ real wage | Δ real imports | Δ real exports |
|---|---:|---:|---:|---:|
| BC | -1.89 | -1.11 | -9.73 | -8.73 |
| AB | -0.00 | -4.02 | -28.76 | -29.93 |
| SK | -2.00 | -1.19 | -12.44 | -9.58 |
| MB | -1.90 | -0.87 | -7.58 | -6.32 |
| ON | -1.49 | -0.80 | -5.62 | -4.35 |
| QC | -1.63 | -0.55 | -4.94 | -3.04 |
| NB | -1.44 | -0.34 | -0.80 | +0.81 |
| NS | -1.67 | -0.46 | -2.65 | -0.46 |
| PE | -1.66 | -0.21 | -1.74 | +0.78 |
| NL | -1.36 | -0.17 | -1.45 | +0.35 |
| **CAN − AB** | **-1.93** | **-1.09** | **-6.78** | **-5.36** |
| **CAN** | **-1.32** | **-1.23** | **-10.41** | **-10.36** |

#### Scenario 7 — Labour: frozen, Trade: normal

| Prov | Δ real income | Δ real wage | Δ real imports | Δ real exports |
|---|---:|---:|---:|---:|
| BC | -0.85 | -0.05 | -0.63 | +1.05 |
| AB | +3.92 | +0.18 | +2.59 | -3.45 |
| SK | -0.59 | -0.03 | -0.13 | +0.67 |
| MB | -0.88 | -0.05 | -0.51 | +0.92 |
| ON | -0.76 | -0.07 | -0.41 | +1.00 |
| QC | -0.92 | -0.07 | -0.66 | +1.04 |
| NB | -0.86 | -0.06 | -0.15 | +1.08 |
| NS | -1.06 | -0.09 | -0.89 | +1.32 |
| PE | -1.12 | -0.07 | -0.94 | +1.01 |
| NL | -0.76 | -0.02 | -0.17 | +0.90 |
| **CAN − AB** | **-0.82** | **-0.06** | **-0.62** | **+0.87** |
| **CAN** | **-0.11** | **-0.02** | **-0.01** | **-0.01** |

#### Scenario 8 — Labour: frozen, Trade: +10%

| Prov | Δ real income | Δ real wage | Δ real imports | Δ real exports |
|---|---:|---:|---:|---:|
| BC | -1.43 | -0.63 | -5.85 | -4.53 |
| AB | +2.41 | -1.63 | -12.19 | -16.29 |
| SK | -1.32 | -0.62 | -7.22 | -5.18 |
| MB | -1.42 | -0.45 | -4.01 | -2.58 |
| ON | -1.15 | -0.38 | -2.69 | -1.17 |
| QC | -1.31 | -0.30 | -2.84 | -0.95 |
| NB | -1.18 | -0.18 | -0.40 | +1.09 |
| NS | -1.39 | -0.26 | -1.64 | +0.67 |
| PE | -1.43 | -0.14 | -1.25 | +1.04 |
| NL | -1.09 | -0.09 | -0.68 | +0.79 |
| **CAN − AB** | **-1.27** | **-0.40** | **-3.65** | **-2.09** |
| **CAN** | **-0.72** | **-0.61** | **-5.08** | **-5.06** |

#### Scenario 9 — Labour: frozen, Trade: WLD-walls

| Prov | Δ real income | Δ real wage | Δ real imports | Δ real exports |
|---|---:|---:|---:|---:|
| BC | -1.94 | -1.12 | -9.55 | -8.47 |
| AB | +0.07 | -3.97 | -29.92 | -31.16 |
| SK | -2.05 | -1.20 | -12.25 | -9.31 |
| MB | -1.95 | -0.88 | -7.37 | -6.04 |
| ON | -1.55 | -0.82 | -5.44 | -4.09 |
| QC | -1.68 | -0.56 | -4.74 | -2.76 |
| NB | -1.50 | -0.35 | -0.53 | +1.16 |
| NS | -1.73 | -0.48 | -2.41 | -0.12 |
| PE | -1.71 | -0.22 | -1.47 | +1.12 |
| NL | -1.40 | -0.18 | -1.17 | +0.69 |
| **CAN − AB** | **-1.70** | **-0.81** | **-6.62** | **-5.12** |
| **CAN** | **-1.43** | **-1.34** | **-10.47** | **-10.42** |
