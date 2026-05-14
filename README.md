# qge

A Python port of the Caliendo, Parro, Rossi-Hansberg, and Sarte (CPRHS) quantitative general-equilibrium model of the US economy — *The Impact of Regional and Sectoral Productivity Changes in the U.S. Economy*. Translates the paper's MATLAB replication kit into idiomatic NumPy/SciPy and ships the inputs as labeled long-form parquet so the model can be driven by data outside the original calibration.

## Status

The Benchmark variant of the model is fully ported:

- Baseline equilibrium from raw 2007 data.
- Regional counterfactuals: 10% TFP shock in each of 50 states.
- Sectoral counterfactuals: 10% TFP shock in each of 26 sectors.
- Aggregate TFP / GDP / welfare elasticities.

All outputs match the published MATLAB workspaces to machine epsilon (relative error around 1e-13 to 1e-16) on the values the paper reports. **22 tests pass in ~40 seconds.**

Not yet implemented: the NS / NR / NRNS / Efficient model variants, the four real-economy applications (California computers, North Dakota oil, NYC FIRE, Hurricane Katrina), and the geographic-barriers counterfactuals. The translation machinery is in place for all of them.

## Install

The project uses [uv](https://github.com/astral-sh/uv) for environment management.

```sh
git clone <repo>
cd qge
uv sync
```

That creates a `.venv` and installs `numpy`, `scipy`, `pandas`, `pyarrow`, plus `pytest` for tests.

## Quick start

```python
from qge.io import load_inputs
from qge.models.benchmark import (
    compute_baseline,
    compute_regional_shock,
    compute_sectoral_shock,
    regional_sweep,
    sectoral_sweep,
)
from qge.elasticities import regional_elasticities, sectoral_elasticities

# Baseline equilibrium from the shipped CPRHS calibration (~10 seconds).
raw = load_inputs()                      # data/inputs/cprhs/
baseline = compute_baseline(raw=raw)

# One regional shock: 10% TFP boost in California (state index 4, 0-indexed).
ca = compute_regional_shock(region=4)
elast = regional_elasticities(ca, region=4, Ln=baseline.Ln)
print(elast.TFP, elast.GDP, elast.welfare)

# Full sweep of all 50 regional shocks + elasticities (~10 minutes).
sweep = regional_sweep(verbose=True)
for region_name, row in zip(raw.regions, sweep.elasticities):
    print(f"{region_name:<20} {row.TFP:+.4f}  {row.GDP:+.4f}  {row.welfare:+.4f}")
```

## Project layout

```
qge/
├── io.py                  Long-form parquet loader, RawInputs dataclass, validation
├── helpers.py             Per-iteration math (P_h_om, Dinprime, Lchange,
│                          expenditure, GMC, neweq, GOTFP, GDP)
├── elasticities.py        Aggregate TFP / GDP / welfare elasticity formulas
└── models/
    └── benchmark.py       compute_baseline + shock + sweep entry points

data/inputs/cprhs/         CPRHS reference calibration (committed parquet)

scripts/
└── convert_cprhs.py       MATLAB .mat → canonical parquet converter

tests/                     22 tests — baseline, shocks, elasticities, parquet
                           round-trip against the published .mat workspaces
```

## Input data

Inputs are seven long-form parquet files plus a sectoral-dispersion vector, all carrying human-readable sector and region labels:

| file                          | columns                                  | meaning                       |
|-------------------------------|------------------------------------------|-------------------------------|
| `bilateral_trade.parquet`     | `sector, destination, source, value`     | trade flow                    |
| `employment.parquet`          | `sector, region, value`                  | sector × region employment    |
| `io_matrix.parquet`           | `source_sector, dest_sector, value`      | input-output coefficient      |
| `value_added_share.parquet`   | `sector, region, value`                  | γ — value-added share         |
| `structures_share.parquet`    | `sector, region, value`                  | B — structures share          |
| `final_demand_share.parquet`  | `sector, region, value`                  | α — final-demand share        |
| `portfolio_share.parquet`     | `region, value`                          | ι — global portfolio share    |
| `sectoral_dispersion.parquet` | `sector, value`                          | 1/θ — Eaton-Kortum elasticity |

The schema is **dimension- and label-agnostic**: nothing in the model assumes 26 sectors or 50 US states. Labels come from the data and propagate to results.

`qge.io.load_inputs(directory)` reads the eight files, validates ranges and finiteness, and returns a `RawInputs` dataclass.

## Bringing your own calibration

Drop your parquet files into a sibling directory and point the loader at it:

```python
raw_ca = load_inputs("data/inputs/canada_2020/")
result = compute_baseline(raw=raw_ca)
```

There are no code changes required. The number of sectors and regions is whatever your data provides; the sectoral dispersion vector ships in the same directory.

The MATLAB-to-parquet converter (`scripts/convert_cprhs.py`) is the first worked example of how to land in this schema; any other ingestion path (BEA, BLS, Statistics Canada, …) is a peer of it.

## Testing

```sh
uv run pytest                  # all 22 tests, ~40s
uv run pytest tests/test_parquet_io.py -v
```

The verification suite requires the CPRHS MATLAB replication kit (available from [the author's site](https://sites.google.com/site/lorenzocaliendo/research/CPRHS)) placed at `CPRHS replication files/` — that folder is gitignored. Without it, the tests skip cleanly and the model still runs against the shipped parquet calibration.

## Reference

Caliendo, L., Parro, F., Rossi-Hansberg, E., and Sarte, P.-D. *The Impact of Regional and Sectoral Productivity Changes in the U.S. Economy.* Replication materials: <https://sites.google.com/site/lorenzocaliendo/research/CPRHS>.
