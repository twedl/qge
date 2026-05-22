# tw — Tombe & Winter (2021) replication

Python port of the model and counterfactuals from:

> Tombe, T. and Winter, J. (2021). "Fiscal integration with internal trade:
> Quantifying the effects of federal transfers in Canada." *Canadian Journal
> of Economics / Revue canadienne d'Économique* 54(2), 522–556.
> https://doi.org/10.1111/caje.12491

Data and code in this directory derive from the authors' published
replication archive:

> Tombe, Trevor; Winter, Jennifer, 2022, "Replication Data and Code for:
> Fiscal integration with internal trade: Quantifying the effects of federal
> transfers in Canada", https://doi.org/10.5683/SP3/SHDOHY, Borealis, V1.

The archive ships a MATLAB + R toolchain (`maindata.mat`, `FiscalData.RData`,
ten `.m` scripts, two `.R` scripts). This Python package re-implements the
solver and counterfactuals in NumPy / SciPy, converts the inputs to
long-form parquet, and reproduces every table and figure in the published
paper.

## Model summary

Caliendo & Parro (2015) multi-sector hat-algebra trade model **plus**:

- Frechet provincial migration with shape parameter κ = 1.5.
- Endogenous trade imbalances driven by fiscal transfers.
- Income-sensitive transfers `t_n ∝ w_n^(−ζ)` with calibrated ζ = 0.3268.
- Exogenous transfers `T̄_n` from StatCan 36-10-0450 (federal revenue and
  spending by province).

11 regions (10 provinces + Rest of World), 22 sectors. Two calibrations
ship: `canada_2010/` (the paper's published vintage) and `canada_2022/`
(refreshed; see [Calibrations](#calibrations) below). Trade costs come
from Albrecht & Tombe (2016).

## Calibrations

| dir | year | source | notes |
|---|---|---|---|
| `data/inputs/canada_2010/` | 2010 | `maindata.mat` + `FiscalData.RData` (Borealis archive) | paper baseline |
| `data/inputs/canada_2022/` | 2022 | StatCan WDS + OECD ICIO 2023 release | latest data; built by `scripts/build_canada_2022.py` |

Run any script against either calibration via the `TW_CALIBRATION` env
var (defaults to `canada_2010`):

```sh
TW_CALIBRATION=canada_2022 uv run python scripts/table4.py
```

The 2022 calibration **refreshes**:
- Bilateral trade flows (StatCan 12-10-0088, 2022).
- Employment shares (StatCan 14-10-0327 active LFS, 2022).
- Real income index (StatCan 36-10-0222 GDP + 18-10-0003 spatial prices).
- Fiscal transfers (StatCan 36-10-0450, multi-year average).
- WLD-WLD intra-trade (OECD ICIO 2023 release, year 2022).
- **Industrial structure α, γ, φ, IO matrix** — built by
  `scripts/build_industrial_structure.py` from 2021 L97 provincial
  Supply-Use tables (StatCan 15-211-X). γ varies by region (canada_2010
  replicated a single national matrix across all 11 regions). The IO
  matrix is constructed as γ × (1 − φ) at the Canadian aggregate so
  that columns sum to (1 − φ^j), matching the canada_2010 convention.

The provincial IOTs lag 2-3 years, so 2022-data primitives sit on top
of 2021 industrial structure — a slow-moving structure with ~1y mismatch.

What's still carried over from canada_2010/:
- AT-2016 trade-cost shocks (`tauhat_all/asym/dist`) — re-estimating
  these on 2022 data is Phase 4b.
- Sectoral dispersion `θ^j` (AT-2016 / CP-2015 vector, static).
- Sector labels.

A small baseline accounting residual (∑ S_n ≈ 3·10⁻⁴ I) persists
because trade flows and IO are now built from independent StatCan
sources without joint calibration. canada_2010's maindata.mat jointly
calibrated π so that ∑ S_n = machine epsilon; we accept the residual
in exchange for direct data lineage.

Selected 2010 → 2022 changes (Table 4, All fiscal transfers, % effect
on real income):

| Prov | 2010 | 2022 |
|---|---|---|
| BC | -1.7 | -2.2 |
| AB | -8.3 | -6.7 |
| SK | -0.9 | +0.8 |
| MB | +7.2 | +12.3 |
| ON | -0.8 | +2.6 |
| QC | +3.4 | +6.7 |
| NB | +15.6 | +41.6 |
| NS | +17.1 | +26.2 |
| PE | +29.8 | +26.2 |
| NL | +10.4 | -5.0 |
| **CAN aggregate** | **-0.74** | **+1.13** |

Alberta's transfer-induced income loss narrows; Maritime gains widen
sharply (NB more than doubles); Newfoundland flips from net beneficiary
to net loser as its oil-driven income premium fades. The CAN aggregate
flips sign: transfers now generate a small net welfare gain at the
national level (vs. a small net deadweight cost in 2010).

## Acceptance tests ✓

Every paper table and figure reproduces the published values. Reference
values were cross-checked against both the August 2019 working paper
(https://www.trevortombe.com/files/tombe_winter_2019.pdf) and the published
CJE 2021 version.

| Script | Replicates | Status |
|---|---|---|
| `scripts/table1.py`  | Interprovincial fiscal transfers in Canada, 2007–2016 | ✓ all cells match |
| `scripts/table2.py`  | Shapley decomposition of fiscal-transfer sources | ✓ GDP-% and transfer-% columns match exactly; correlation column within ±0.2 |
| `scripts/table4.py`  | Effect of interprovincial transfers (all / equalization / income-sensitive) | ✓ all cells match |
| `scripts/table5.py`  | Gains from observed trade relative to autarky | ✓ all cells match |
| `scripts/table6.py`  | Gains from lower internal trade costs (asym / non-distance / all) | ✓ all cells match |
| `scripts/table7.py`  | Unilateral import vs export cost reductions per province | ✓ all cells within 0.1pp |
| `scripts/table8.py`  | Bilateral trade liberalization (New West / ON–QC / Maritime), both panels | ✓ all cells match |
| `scripts/figure2.py` | Welfare-maximizing ζ across fiscal-integration sweep | ✓ ζ\* = −0.24 (panel a), −0.35 (panel b) |
| `scripts/figure3.py` | Per-sector output change vs upstreamness for AB and PE | ✓ data matches; plot externally |

Figure scripts write CSV data to `data/outputs/`; plot rendering is left
to downstream code. Table 3 is reproduced directly from Albrecht & Tombe
(2016) in the original paper and is not ported here.

## Layout

```
tw/
  qge/
    io.py              — load_inputs() + RawInputs dataclass
    initialize.py      — solve initial equilibrium (port of initialize_model.m
                          and eliminate_transfers.m)
    equations.py       — hat-algebra inner loop (port of model_equations.m)
    solver.py          — outer fsolve wrapper (port of solve_model.m)
  scripts/
    convert_maindata.py     — maindata.mat → parquets (one-time)
    convert_fiscaldata.py   — FiscalData.RData → parquets (one-time)
    fetch_fiscal_data.py    — refresh fiscal_*.parquet from StatCan WDS API
    table1.py, table2.py    — fiscal-transfer descriptives
    table4.py … table8.py   — model-driven counterfactuals
    figure2.py, figure3.py  — sweep / per-sector data extracts
  data/inputs/canada_2010/  — long-form parquets (inputs to all scripts)
  data/outputs/             — table / figure outputs
  tests/
```

## Running

```bash
cd tw/
uv sync

# One-time data preparation (requires the Dataverse archive extracted to
# /tmp/tombe_winter_2021/DATA/):
uv run python scripts/convert_maindata.py
uv run python scripts/convert_fiscaldata.py

# (Optional) refresh the fiscal data from the latest StatCan vintage:
uv run python scripts/fetch_fiscal_data.py

# Any table or figure:
uv run python scripts/table4.py
uv run python scripts/figure2.py
```

## Data provenance

- `bilateral_trade`, `io_matrix`, `intermediate_share`, `labor_share`,
  `final_demand_share`, `sectoral_dispersion`, `employment`, `income_index`,
  `trade_surplus`, `income_baseline`, `Sn_baseline` — extracted from
  `maindata.mat` (the Dataverse archive). Trade flows trace to StatCan
  12-10-0088-01 (2010); productivity / value-added shares to OECD STAN;
  trade-cost shocks `tauhat_all`, `tauhat_asym`, `tauhat_dist` and sectoral
  trade elasticities `θ^j` to Albrecht & Tombe (2016).
- `transfers_share_GDP` (a 2017 vintage aggregate over 2007–2016 of StatCan
  36-10-0450) is taken from `maindata.mat`. Refresh via
  `scripts/fetch_fiscal_data.py` to pull the current StatCan revision.
- `fiscal_currentdata`, `fiscal_pop`, `fiscal_GDP_prov`, `fiscal_provnames`
  — extracted from `FiscalData.RData` (the Dataverse archive's 2017
  vintage). The WDS fetcher writes the same schema from live StatCan tables
  36-10-0450, 17-10-0005, and 36-10-0222.

## Provenance for sectoral trade elasticities θ^j

Sectoral trade elasticities are inherited from Albrecht & Tombe (2016)
Table 9, which in turn aggregates Caliendo & Parro (2015) estimates. See
`cprhs/THETA.md` Part 2a for the full chain of provenance and the
sector-by-sector concordance.
