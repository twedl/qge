# tw — Tombe & Winter (2021) replication

Python port of:

> Tombe, T. and Winter, J. (2021). "Fiscal integration with internal trade:
> Quantifying the effects of federal transfers in Canada." *Canadian Journal
> of Economics* 54(2), 522–556.

The original replication package is a MATLAB + R combination shipped via
Dataverse (https://doi.org/10.5683/SP3/SHDOHY). This package converts the
MATLAB inputs to long-form parquet, ports the hat-algebra solver to Python /
NumPy / SciPy, and reproduces the paper's main counterfactual tables.

## Model summary

Caliendo-Parro (2015) multi-sector hat-algebra trade model **plus**:

- Frechet provincial migration with shape parameter κ = 1.5.
- Endogenous trade imbalances driven by fiscal transfers.
- Income-sensitive transfers `t_n ∝ w_n^(−ζ)` with calibrated ζ = 0.3268.
- Exogenous transfers `T̄_n` from StatCan 36-10-0450 (federal revenue and
  spending by province).

11 regions (10 provinces + ROW), 22 sectors, 2010 vintage. Trade costs
imported from Albrecht-Tombe (2016).

## Status

- **Phase 0 (DONE):** ingest `maindata.mat` + `FiscalData.RData` →
  long-form parquets under `data/inputs/canada_2010/`.
- **Phase 1 (DONE):** port `initialize_model.m`, `model_equations.m`,
  `solve_model.m`, `eliminate_transfers.m`.
- **Phase 2 (IN PROGRESS):** port the counterfactual tables (Table 4 done;
  Tables 5-8 + Figures 2-3 still to port).
- **Phase 3 (TODO):** fetch `FiscalData` directly from StatCan WDS API to
  avoid the manual RData dependency.

## Table 4 — acceptance test ✓

`scripts/table4.py` reproduces every cell of Table 4 to one decimal place
against both the published CJE 2021 version (vol 54(2), pp.522-556) and the
August 2019 working paper at
https://www.trevortombe.com/files/tombe_winter_2019.pdf .

## Layout

```
tw/
  qge/
    io.py             — load_inputs() + RawInputs dataclass
    initialize.py     — solve initial equilibrium (port of initialize_model.m)
    equations.py      — hat-algebra inner loop (port of model_equations.m)
    solver.py         — outer fsolve wrapper (port of solve_model.m)
    eliminate.py      — alternative no-transfer baseline (port of eliminate_transfers.m)
  scripts/
    convert_maindata.py    — maindata.mat → parquet (run once)
    convert_fiscaldata.py  — FiscalData.RData → parquet (run once)
    table4.py              — replicate paper's Table 4
    (other table*.py / figure*.py to follow)
  data/inputs/canada_2010/   — long-form parquets
  tests/
```

## Running

```bash
cd tw/
uv sync
uv run python scripts/convert_maindata.py     # one-time, requires the .mat file
uv run python scripts/convert_fiscaldata.py   # one-time, requires the .RData file
uv run python scripts/table4.py
```

## Data provenance caveats

- `transfers_share_GDP` and `surplus_gdp` in `maindata.mat` are aggregates over
  2007-2016 from the Tombe & Winter 2017 vintage of StatCan 36-10-0450. The
  current StatCan revision may differ. Phase 3 will re-derive these via the
  WDS API.
- `tauhat_all`, `tauhat_asym`, `tauhat_dist` come from Albrecht-Tombe (2016)
  and represent counterfactual trade-cost reductions (all-internal-cost,
  asymmetry-only, distance-only).
- Trade elasticities θ^j are sector-specific, also from Albrecht-Tombe (2016).
  See `cprhs/THETA.md` (Part 2a) for the AT-2016 / CP-2015 chain of provenance.
