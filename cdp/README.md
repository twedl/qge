# cdp

Python implementation of the Caliendo-Dvorkin-Parro (2019) dynamic labor-market trade model: *Trade and Labor Market Dynamics: General Equilibrium Analysis of the China Trade Shock* (Econometrica).

Reference: [Lorenzo Caliendo · CDP research](https://sites.google.com/site/lorenzocaliendo/research/cdp).

## Status

- **Phase 1 — Base_Year (static initial 2000 equilibrium) is done.** Verified against `Base_year.mat` to solver tolerance (~1e-7).
- **Phase 2a — Step 1 data construction is done.** Quarterly interpolation of yearly bilateral trade, μ-driven labor evolution, value-added and wage time series. Verified against `Baseline_2000_2007_economy_actual_data.mat` to machine epsilon.

24 tests pass.

Remaining phases:
- **Phase 2b** — Step 2 dynamic baseline solver (28 quarter-by-quarter temporary equilibria; uses `solve_tvf.m`)
- **Phase 2c** — Step 3 forward simulation from 2007 with constant fundamentals
- **Phase 2d** — Step 4 stitch (combine 2a-2c into the full dynamic baseline)
- **Phase 3** — counterfactual with China-shock removed
- **Phase 4** — employment / welfare effect figures
- **Phase 5** — extensions (SSDI, persistence, CES, real home production)

## Quick start

```sh
cd cdp
uv sync
uv run python scripts/convert_cdp_txt.py            # one-time: .txt → parquet
uv run pytest                                        # 18 tests, ~15s
```

```python
from qge.io import load_inputs
from qge.models.base_year import compute_baseline

raw = load_inputs()                       # data/inputs/cdp_2000/
result = compute_baseline(raw=raw)        # 165 iterations, ~10s
print(result.om.shape, result.Xp.shape)   # (22, 87), (22, 87)
```

## Model

- **22 productive sectors** (the non-employment sector enters only in the dynamic phase)
- **87 regions** = 50 US states + 37 countries (incl. Rest of World as a residual)
- Eaton-Kortum trade structure with **region-specific input-output coefficients**: US block (shared across all 50 states) + 37 foreign country blocks
- US states have **sector-specific wages**; foreign countries have **one wage per country** (no sectoral wage dispersion abroad)
- Final-demand shares **α are region-invariant** in this calibration (CDP's data.m aggregates across regions and broadcasts)

## Inputs (`data/inputs/cdp_2000/`)

Six parquet files derived from the CDP MATLAB replication kit (catalog 13758):

| file | columns | meaning |
|---|---|---|
| `bilateral_trade.parquet` | `sector, destination, source, value` | xbilat with non-tradable US-state diagonals filled from GO |
| `value_added_share.parquet` | `sector, region, value` | γ |
| `gross_output.parquet` | `sector, region, value` | GO (also recoverable from xbilat row sums) |
| `structures_share.parquet` | `region, value` | B (US states) ⊔ B (foreign countries) |
| `io_coefficients.parquet` | `country, source_sector, dest_sector, value` | 38 blocks: United States + 37 foreign countries |
| `sectoral_dispersion.parquet` | `sector, value` | 1/θ_j |

## Project layout

```
cdp/
├── pyproject.toml, uv.lock, README.md
├── qge/
│   ├── io.py                          # RawInputs, load_inputs, data.m transformations
│   ├── helpers.py                     # P_h_om, Dinprime, expenditurenew, GMCnew
│   ├── dynamic.py                     # Step 1 quarterly series + LMC
│   └── models/base_year.py            # solvewnew, compute_baseline
├── data/inputs/cdp_2000/              # 6 parquet files
├── scripts/convert_cdp_txt.py         # .txt → parquet converter
├── tests/                             # 18 tests
└── CDP replication files/             # MATLAB source (gitignored)
```

This sub-project is intentionally independent of [`../cprhs/`](../cprhs/); no code is shared.
