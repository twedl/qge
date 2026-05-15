# qge

A Python port of the Caliendo, Parro, Rossi-Hansberg, and Sarte (CPRHS) quantitative general-equilibrium model — *The Impact of Regional and Sectoral Productivity Changes in the U.S. Economy*. Translates the paper's MATLAB replication kit into idiomatic NumPy/SciPy, ships the inputs as labeled long-form parquet, and exposes results as pandas DataFrames indexed by sector and region names. The core solver is calibration-agnostic — designed for swap-in of Canadian (or any other) data.

## Status

Ported and verified against the MATLAB workspaces to machine epsilon (relative error 1e-13 to 1e-16):

- **Benchmark baseline** — 2007 US equilibrium from raw data.
- **Counterfactuals** — regional (50-state) and sectoral (26-sector) 10% TFP shocks.
- **Aggregate elasticities** — TFP, GDP, welfare per shock, plus full sweeps.
- **Applications** — California computers boom, North Dakota oil boom, NYC FIRE contraction, Hurricane Katrina (structures shock).
- **Model variants** — NS (no sectoral linkages), NR (no regional trade), NRNS (both) baselines.
- **Geographic-barriers** counterfactuals — distance and other-barriers scenarios.
- **Reporting layer** — `.regional_summary()`, `.sectoral_summary()`, `.as_dataframe()` on every result type; outputs are pandas DataFrames indexed by sector / region names.
- **Canadian calibration** — `data/inputs/canada_2021/` built from StatCan provincial symmetric IOTs (catalogue 15-211-X, Link-1997 level). 23 sectors × 11 regions (10 provinces + Rest of World; territories folded into ROW). Baseline solves in ~0.2s; full regional sweep in ~3s.

**44 tests pass in ~100 seconds.**

Not yet ported: the variant shock scripts (Regional_shocks_NS, Sectoral_shocks_NR, Regional_shocks_NRNS — they use `P_h_omNI`) and the Efficient (planner's allocation) model. The translation machinery is in place for both.

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
    regional_sweep,
)
from qge.applications import shock_california_computers
from qge.elasticities import regional_elasticities

# Baseline equilibrium from the shipped CPRHS calibration (~10 seconds).
raw = load_inputs()                      # data/inputs/cprhs/
baseline = compute_baseline(raw=raw)

# One regional shock: 10% TFP boost in California.
ca = compute_regional_shock(region=4, raw=raw)
elast = regional_elasticities(ca, region=4, Ln=baseline.Ln)
print(f"California — TFP {elast.TFP:+.3f}  GDP {elast.GDP:+.3f}  welfare {elast.welfare:+.3f}")

# Labeled DataFrame output (top 5 states by labor inflow).
print(ca.regional_summary().nlargest(5, "L_hat")[["L_hat", "TFPn_hat", "GDPn_hat"]])

# A real-world counterfactual.
boom = shock_california_computers(raw=raw)
print(boom.sectoral_summary().nlargest(3, "TFPj_hat"))

# Full regional sweep + elasticity table (~10 minutes).
sweep = regional_sweep(verbose=True)
print(sweep.as_dataframe().round(4))
```

## Project layout

```
qge/
├── io.py                  Long-form parquet loader, RawInputs, validation
├── helpers.py             Per-iteration math (P_h_om, Dinprime, Lchange,
│                          expenditure, GMC, neweq, GOTFP, GDP)
├── elasticities.py        Aggregate TFP / GDP / welfare formulas
├── applications.py        Four CPRHS Section 6 counterfactuals
├── geographic.py          Trade-cost reduction (distance / other_barriers)
└── models/
    ├── benchmark.py       compute_baseline + shock + sweep entry points
    └── variants.py        NS / NR / NRNS structural restrictions

data/inputs/cprhs/         CPRHS reference calibration (committed parquet)
    ├── *.parquet          eight core calibration files
    ├── applications/      shock data for the four applications
    └── geographic_barriers/   kappa_hat shocks for trade-cost analysis
data/inputs/canada_2021/   Canadian calibration (StatCan IOTs L97, 2021)

scripts/
├── convert_cprhs.py       MATLAB .mat → CPRHS parquet
├── convert_statcan.py     StatCan WDS API helpers (employment, θ, portfolio)
└── build_canada_iot.py    StatCan 15-211-X IOTs → canada_YYYY/ (top-level Canadian entry point)

tests/                     44 tests — baseline, shocks, applications,
                           variants, geographic, elasticities, reporting,
                           parquet round-trip, validation
```

The `qge.applications` and `qge.geographic` modules are CPRHS-specific by design (hardcode US state names and paper-derived constants). Everything else — `io`, `helpers`, `elasticities`, `models/benchmark`, `models/variants` — is calibration-agnostic and operates on whatever's in `RawInputs`.

## Input data

The model reads eight parquet files plus two optional shock-data subfolders. **See [DATA.md](DATA.md)** for the full reference: per-file shape, semantic role, CPRHS source, and Canadian analogue.

| file                          | columns                                  | meaning                       |
|-------------------------------|------------------------------------------|-------------------------------|
| `bilateral_trade.parquet`     | `sector, destination, source, value`     | trade flow                    |
| `employment.parquet`          | `sector, region, value`                  | sector × region employment    |
| `io_matrix.parquet`           | `source_sector, dest_sector, value`      | input-output coefficient      |
| `value_added_share.parquet`   | `sector, region, value`                  | γ — value-added share         |
| `structures_share.parquet`    | `region, value`                          | B — structures share per state |
| `final_demand_share.parquet`  | `sector, region, value`                  | α — final-demand share        |
| `portfolio_share.parquet`     | `region, value`                          | ι — global portfolio share    |
| `sectoral_dispersion.parquet` | `sector, value`                          | 1/θ — Eaton-Kortum elasticity |

`qge.io.load_inputs(directory)` reads them, validates shapes / ranges / consistency (e.g. final-demand shares sum to 1 per region; no zero `xbilat` rows; no zero `IO` columns), and returns a `RawInputs` dataclass carrying the arrays plus the sector and region labels.

## Bringing your own calibration

Drop your parquet files into a sibling directory and point the loader at it:

```python
raw_ca = load_inputs("data/inputs/canada_2021/")
baseline_ca = compute_baseline(raw=raw_ca)
```

The number of sectors and regions follows your data; labels propagate all the way through to the result DataFrames. **[DATA.md](DATA.md)** documents what each input is, suggests Canadian sources, and gives a priority order for assembly. `scripts/build_canada_iot.py` is the worked example for Canada — it builds all eight inputs from the StatCan provincial symmetric IOTs (one Excel workbook per province + territory) using a 186-industry L97 taxonomy aggregated to 23 model sectors.

A few calibration choices the model can't make for you:

- **Sector taxonomy and tradable list** — `compute_baseline_nr` requires `tradable=[...]` (sector names) explicitly. There's no default.
- **Structures-share aggregation** — the model assumes `B` is constant across sectors per region. If your raw source varies by sector, aggregate before feeding in; the loader rejects sector-varying data with a clear error rather than silently picking row 0.
- **Global portfolio** — `ι ≡ 0` is the simplest defensible first pass (closed-region capital ownership).

The MATLAB-to-parquet converter (`scripts/convert_cprhs.py`) is the first worked example of how to land in this schema; `scripts/build_canada_iot.py` is a second (StatCan IOTs → 23 sectors × 11 regions). Any other ingestion path (BEA, BLS, Eurostat, …) is a peer of these.

## Testing

```sh
uv run pytest                                # all 44 tests, ~100s
uv run pytest tests/test_benchmark_baseline.py -v
```

The verification suite requires the CPRHS MATLAB replication kit (available from [the author's site](https://sites.google.com/site/lorenzocaliendo/research/CPRHS)) placed at `CPRHS replication files/` — that folder is gitignored. Without it, the tests skip cleanly and the model still runs against the shipped parquet calibration.

## Reference

Caliendo, L., Parro, F., Rossi-Hansberg, E., and Sarte, P.-D. *The Impact of Regional and Sectoral Productivity Changes in the U.S. Economy.* Replication materials: <https://sites.google.com/site/lorenzocaliendo/research/CPRHS>.
