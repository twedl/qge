# cdp

Python implementation of the Caliendo-Dvorkin-Parro (2019) dynamic labor-market trade model: *Trade and Labor Market Dynamics: General Equilibrium Analysis of the China Trade Shock* (Econometrica).

Reference: [Lorenzo Caliendo · CDP research](https://sites.google.com/site/lorenzocaliendo/research/cdp).

## Status

- **Phase 1 — Base_Year (static initial 2000 equilibrium) is done.** Verified against `Base_year.mat` to solver tolerance (~1e-7).
- **Phase 2a — Step 1 data construction is done.** Quarterly interpolation of yearly bilateral trade, μ-driven labor evolution, value-added and wage time series. Verified against `Baseline_2000_2007_economy_actual_data.mat` to machine epsilon.
- **Phase 2b — Step 2 dynamic baseline 2000-2007 is done.** 28 quarter-by-quarter temporary-equilibrium solves with data-target factor prices and trade shares. Verified against `Baseline_2000_2007_economy_actual.mat` (rtol=1e-4 absorbing accumulated solver-tolerance round-off).
- **Phase 2c — Step 3 forward simulation from 2007 is done.** 200-period dynamic forward simulation with constant fundamentals, forward-looking value functions Yt, and migration flows μ derived from the Bellman recursion. Outer fixed-point on Yt converges in one iteration when seeded with the saved `Hvectnoshock`. Verified against `Baseline_2007.mat` and `Baseline_economy_2007_forward.mat` (rtol=5e-3 to 2e-2 absorbing the half-step inconsistency in the saved fixture and 200-quarter accumulation).
- **Phase 2d — Step 4 stitch is done.** Combines Phase 2a/2b/2c outputs into a single 200-quarter baseline economy (220 transitions for μ). Pure array splicing — no new computation. Verified against `Baseline_economy.mat` (HDF5/v7.3, loaded via h5py).
- **Phase 3 — China-shock counterfactual is done.** Inverts the estimated 2000-2007 China TFP gains in 12 tradable sectors (region 56, sectors 0..11). Outer Bellman fixed-point on the value-function path V; mu and labor evolution use baseline references; 200 inner temporary equilibria with `A_hat = 1/china_TFP`. Verified against `Counterfactual_economy.mat`.

- **Phase 4 — Employment and welfare effects.** Pure analysis layer (`qge/effects.py`) over Phase 2d + Phase 3 outputs. Computes the long-run sectoral / regional employment-share differences induced by the China shock and the consumption-equivalent welfare change per labor market (paper eq. 28). Slow tests check signs and aggregate magnitudes against the paper.

**Phases 1–4 are complete.** 31 fast tests pass; the full integration across Phase 2b/2c/2d/3/4 is `@pytest.mark.slow` (~15 min, dominated by the 200 inner solves).

## Future work

- **Phase 5 — extensions.** SSDI (Section 5.3.1), persistent migration (Section 5.3.2), CES utility (Appendix 3.2), real home production (Footnote 56), constant-fundamentals variant (Section 4.3). Each replaces parts of the temporary-equilibrium solver and re-runs Phase 2 + 3.
- **DataFrame reporting layer.** Like `cprhs/`'s `.regional_summary()` / `.sectoral_summary()` / `.as_dataframe()` — labels every solver output with sector and region names so the results are interactive-friendly.

Remaining work (less significant):
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
│   ├── dynamic_helpers.py             # P_h_om_tvf, Dinprime_tvf
│   ├── forward_dynamics.py            # mu path, labor evolution, Bellman update
│   ├── counterfactual_dynamics.py     # China shock, mu_cf, Bellman update for V
│   └── models/
│       ├── base_year.py               # solvewnew, compute_baseline
│       ├── dynamic_baseline.py        # solve_tvf, compute_dynamic_baseline_2000_2007
│       ├── forward_simulation.py      # compute_baseline_forward_2007 (Step 3)
│       ├── baseline_economy.py        # stitch_baseline_economy (Step 4)
│       └── counterfactual.py          # compute_counterfactual_economy (Phase 3)
├── data/inputs/cdp_2000/              # 6 parquet files
├── scripts/convert_cdp_txt.py         # .txt → parquet converter
├── tests/                             # 18 tests
└── CDP replication files/             # MATLAB source (gitignored)
```

This sub-project is intentionally independent of [`../cprhs/`](../cprhs/); no code is shared.
