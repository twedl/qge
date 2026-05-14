# Data inputs

Reference for everything the model reads. The intended audience is anyone preparing a new calibration (Canadian provinces, a different sector taxonomy, a different year) — the goal here is to make every shape, semantic role, and underlying source explicit.

For shape, `J` = number of sectors and `N` = number of regions in your calibration. CPRHS uses `J = 26`, `N = 50` (US states with Virginia and DC merged). The model is dimension-generic; nothing in `qge/` requires those specific values.

## Hard prerequisite: interior equilibria

The model is a *change between two equilibria*. Both the baseline and any counterfactual must be valid equilibria, which means **every `(sector, region)` cell must have strictly positive gross output**. Eaton-Kortum's Fréchet productivity assigns every region positive expected output in every sector — zero cells fall outside the model's domain, not just its numerical tolerance.

`qge.io.load_inputs` enforces this at load time: a calibration with a `(sector, region)` cell whose gross output (sum of bilateral exports from that source in that sector) is zero will be rejected with the offending cells named.

**Practical implication for taxonomy choice.** Your sector × region grid must be coarse enough that every cell has positive production. If your candidate taxonomy has empty cells (e.g. *Petroleum and Coal* × *PEI*), you have to aggregate:

- *sector consolidation* — fold the empty sector into a broader category until production is positive everywhere; or
- *region consolidation* — merge the empty region with neighbours (e.g. Maritime provinces); or
- *both* — likely necessary for Canadian data at province × ~25-sector granularity.

Aggregation loses analytical detail (you can't decompose effects within the aggregate), but it's the price of staying inside the model's domain.

## Eight core calibration files

Each file lives at `data/inputs/<calibration_name>/<filename>.parquet`. All categorical columns hold human-readable labels (string sector and region names); the numerical column is always called `value`.

---

### `bilateral_trade.parquet` — *xbilat*

| | |
|---|---|
| **Long form** | `(sector, destination, source, value)` |
| **Rows** | `J · N · N` (CPRHS: 65 000) |
| **Implied dense shape** | `(J·N, N)` stacked by sector |
| **Constraint** | `value ≥ 0` |

**Source for CPRHS:** US Commodity Flow Survey 2007 for goods sectors, BEA IO accounts for services (treated as intra-state for non-tradables).

**Role in the model:** the foundational trade matrix. Derives initial expenditure `X0[j,n] = Σ_source xbilat[j,n,source]`, trade shares `Din = xbilat / row_sums`, net exports per state `Bn = Σ_j (exports − imports)`, and per-sector source totals `E`.

**Canadian analogue:** StatCan **Provincial Input-Output tables** (CANSIM 36-10-0479-01 family) plus interprovincial trade flows. The single hardest input to assemble — bilateral interprovincial trade for ~20 sectors. Likely a multi-table reconstruction.

---

### `employment.parquet` — *L_j_n*

| | |
|---|---|
| **Long form** | `(sector, region, value)` |
| **Rows** | `J · N` (CPRHS: 1 300) |
| **Implied dense shape** | `(J, N)` |
| **Constraint** | `value ≥ 0` |

**Source for CPRHS:** BLS Quarterly Census of Employment and Wages (QCEW) 2007.

**Role in the model:** derives `Ln = Σ_j L_j_n / total` — population/employment shares per region. Used as the labor-supply weights in welfare and labor-market clearing. Also used in `neweq` for share-normalization of post-equilibrium employment.

**Canadian analogue:** StatCan **Survey of Employment, Payrolls and Hours** (CANSIM 14-10-0202-01) or **Labour Force Survey**, aggregated to province × NAICS.

---

### `io_matrix.parquet` — *IO*

| | |
|---|---|
| **Long form** | `(source_sector, dest_sector, value)` |
| **Rows** | `J · J` (CPRHS: 676) |
| **Implied dense shape** | `(J, J)` — rows = source, columns = destination |
| **Constraint** | `value ≥ 0`; column sums must be strictly positive |

**Source for CPRHS:** BEA Make/Use tables aggregated to 26 sectors, 2007.

**Role in the model:** intermediate-input technology. The model normalizes internally — each column (destination sector) is rescaled to sum to 1, giving "share of source-sector inputs in destination's intermediate bundle." Then multiplied by `(1 − γ)` per region to form `G_3d`, the input-output network used everywhere prices propagate.

**Canadian analogue:** StatCan **Symmetric Input-Output Tables** (CANSIM 36-10-0594-01, industry × industry). Need to aggregate to your sector taxonomy and verify orientation. **Watch:** if you flip rows/columns, the model is silently wrong (no validation will catch it).

---

### `value_added_share.parquet` — *gamma* (γ)

| | |
|---|---|
| **Long form** | `(sector, region, value)` |
| **Rows** | `J · N` |
| **Implied dense shape** | `(J, N)` |
| **Constraint** | `0 ≤ value ≤ 1` |

**Source for CPRHS:** BEA GDP-by-industry-by-state — computed as `value_added / gross_output` per (sector, state).

**Role in the model:** `γ` is the share of gross output paid to factors (labor + structures); **`(1 − γ)` is the intermediate-input share**. Enters input-bundle cost `c = exp(γ·ln w + (1−γ)·ln p)` and shows up in `VALjn`, `G_3d`, the price equation, and the labor-demand condition.

**Canadian analogue:** StatCan **GDP at basic prices by industry by province**, divided by gross output (Use tables column totals).

---

### `structures_share.parquet` — *B*

| | |
|---|---|
| **Long form** | `(region, value)` |
| **Rows** | `N` (CPRHS: 50) |
| **Implied dense shape** | `(N,)` |
| **Constraint** | `0 ≤ value < 1` |

**Important assumption:** the model assumes the structures share is **constant across sectors within a region**. If your source data varies by sector, you must aggregate to a single per-region number (an output-weighted average is the natural choice) before feeding in.

**Source for CPRHS:** capital income share within value added; CPRHS draws from BEA fixed-asset accounts.

**Role in the model:** `B` is the share of value added paid to structures (capital); `(1 − B)` is the labor share. Enters labor-supply and goods-market-clearing via `L_hat^(1−B)` and `wf0 = om · L_hat^(−B)`. The Hurricane Katrina application also enters as `H_hat^B · L_hat^(1−B)`.

**Canadian analogue:** StatCan **Capital, Labour and Multifactor Productivity** tables (CANSIM 36-10-0208-01) for capital income shares by industry × province. Aggregate to per-province values before parquet.

---

### `final_demand_share.parquet` — *alphas* (α)

| | |
|---|---|
| **Long form** | `(sector, region, value)` |
| **Rows** | `J · N` |
| **Implied dense shape** | `(J, N)` |
| **Constraint** | `0 ≤ value ≤ 1`; **each region's column sums to 1** |

**Source for CPRHS:** BEA Personal Consumption Expenditures by state, allocated across sectors.

**Role in the model:** Cobb-Douglas final-demand weights. Enters the aggregate price index `P_index_n = Π_j phat[j,n]^α[j,n]` and the expenditure equation. Sets consumption preferences per region.

**Canadian analogue:** StatCan **Final Domestic Demand by Province** (CANSIM 36-10-0222-01) crossed with the national split of consumption by product. Reconciling provincial PCE with national category shares is a common pain point.

---

### `portfolio_share.parquet` — *io* (ι)

| | |
|---|---|
| **Long form** | `(region, value)` |
| **Rows** | `N` (CPRHS: 50) |
| **Implied dense shape** | `(N,)` |
| **Constraint** | `0 ≤ value ≤ 1` |

**Source for CPRHS:** **calibrated**, not directly observed. Chosen as a residual so the model reproduces observed trade balances given the rest of the data.

**Role in the model:** region n surrenders `ι_n · VAR_n` of its structures rents to a "global portfolio" that gets redistributed by population share. The model's mechanism for capital flows across regions without modeling them explicitly. If `ι ≡ 0`, each region keeps all of its own capital income (closed-region assumption).

**Canadian analogue:** the trickiest input. Options:
1. Set `ι ≡ 0` and assume closed-province capital ownership. Simplest defensible first pass.
2. Calibrate as residual to match interprovincial current-account balances (if assemblable).
3. Use a uniform value (e.g., 0.5) and treat as a sensitivity parameter.

---

### `sectoral_dispersion.parquet` — *T* = 1/θ

| | |
|---|---|
| **Long form** | `(sector, value)` |
| **Rows** | `J` (CPRHS: 26) |
| **Implied dense shape** | `(J,)` |
| **Constraint** | `value > 0` |

**Source for CPRHS:** estimated. Sectoral trade-cost elasticities. The 26 values come from `CPRHS_Benchmark.m` lines 19–21 — manufacturing sectors use estimates from Caliendo-Parro 2015, non-tradables get a default `1/4.55`.

**Role in the model:** Eaton-Kortum dispersion per sector. Higher `1/θ` = more heterogeneous productivity = less price-sensitive trade. Enters the price equation `phat^(−T) = D_in_k · [ λ^(γ/T) · c^(−1/T) ]` and the trade-share update.

**Canadian analogue:** **reuse CPRHS values** if you keep the same (or matched) sector taxonomy — θ is a sector property, not country-specific. For different sectors, use Caliendo-Parro 2015 or related global estimates.

---

## Optional application data

Only required if you re-run the four CPRHS Section 6 counterfactuals against Canadian data. Skip otherwise.

### `applications/measured_tfp_2002_2007.parquet`

`(sector, region, value)`, `J · N` rows. Measured TFP changes 2002–2007. The California Computers application picks one entry, converts to fundamental TFP, and annualizes. The rest of the matrix is loaded but unused.

### `applications/north_dakota_lambda.parquet`

`(sector, value)`, `J` rows. Per-sector productivity shock used to study the Bakken oil boom in North Dakota.

---

## Optional geographic-barriers data

Only required if you re-run the trade-cost reduction counterfactuals.

### `geographic_barriers/kappa_distance.parquet`
### `geographic_barriers/kappa_other_barriers.parquet`

Both `(sector, destination, source, value)`, `J · N · N` rows. `kappa_hat` matrices that remove geographic-distance trade costs or non-distance ("other") barriers, respectively. Tradable sectors carry the calibrated reductions; non-tradable sectors are left at 1.

**Source for CPRHS:** estimates from a gravity model on US data.

**Canadian analogue:** would require estimating Canadian provincial gravity coefficients. Separate research project.

---

## Suggested order for assembling a Canadian Benchmark calibration

| Priority | Input | Difficulty | Notes |
|---|---|---|---|
| 1 | `sectoral_dispersion` | trivial | Reuse CPRHS θ_j with matched sectors |
| 2 | `employment` | easy | LFS or SEPH |
| 3 | `gamma` | medium | GDP-by-industry / Gross output |
| 4 | `final_demand_share` | medium | PCE by province × sector |
| 5 | `io_matrix` | medium | StatCan symmetric IO |
| 6 | `structures_share` | medium | Per-province capital share |
| 7 | `portfolio_share` | judgment | Start with `ι = 0` (closed-province) |
| 8 | `bilateral_trade` | **hard** | Provincial Input-Output trade flows. The big one. |

**Sectoral taxonomy** — pick early. CPRHS's 26 sectors map roughly to NAICS 2-digit; StatCan's IO Tables come in S-level (35 industries), L-level (~50), and W-level (200+). Picking S- or L-level keeps the parquet sizes manageable.

**Regions** — CPRHS merges Virginia and DC for `N = 50`. Canada has 10 provinces + 3 territories = 13 candidate regions. Smaller `N` means everything is faster and cheaper to assemble. Consider dropping territories if data is too sparse.

**Tradable / non-tradable split** — required for the No-Regional-Trade variant (`compute_baseline_nr`). CPRHS marks the first 15 of 26 sectors as tradable; you'll need to make the equivalent designation for your sectors and pass them by name.

---

## How the loader uses these files

```python
from qge.io import load_inputs
from qge.models.benchmark import compute_baseline

raw = load_inputs("data/inputs/canada_2020/")   # validates ranges, shapes, completeness
result = compute_baseline(raw=raw)               # everything below derives from `raw`
```

`load_inputs` does the long-form → ndarray conversion, fills out `RawInputs` (including labels), and runs `_validate`. Once `raw` is built, the model itself doesn't know it's looking at Canadian data — labels propagate through to every `BenchmarkResult` / `BenchmarkShockResult` / `*SweepResult` and the DataFrame helpers (`.regional_summary()`, `.as_dataframe()`, etc.) display them.
