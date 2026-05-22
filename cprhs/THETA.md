# Trade elasticity θ — sources and concordances

Reference document for the trade-elasticity parameter `θ_j` used in the QGE
calibration (`sectoral_dispersion.parquet` stores `T_j = 1/θ_j`). Existence of
this file is a one-stop catalogue so that the choice of θ_j vector for any
given calibration can be traced back to a primary academic estimate.

This document grows in parts:
- **Part 1 (done):** original sources of θ and the industry classifications
  each paper estimates against — see "Original sources" below.
- **Part 2a (done):** AT-2016 Table 9 derived from CP-2015 by `derive_at_theta.py`.
  AT use Full sample + arith / ISIC-count weighted mean — *different from
  our benchmark column*.
- **Part 2b (TODO):** infer AKT-2019's effective θ via Head-Ries matching
  (inverse problem — AKT do not publish θ; we back it out from their Tables
  1–2 NG numbers).
- **Part 3 (TODO):** derive `sectoral_dispersion.parquet` for our 23-sector
  NAICS calibration from the benchmark.

## Model role of θ

In the Eaton-Kortum framework that the QGE model uses, `θ_j` is the dispersion
parameter of the Fréchet productivity distribution in sector `j`. Equivalently,
it is the (negative of the) cost-elasticity of bilateral trade flows:
`∂ ln(π_{ni}^j) / ∂ ln(τ_{ni}^j) = −θ_j`. Larger θ → trade flows respond more
sharply to trade costs → gains from trade liberalization are smaller. The same
number plays the role of `σ − 1` in Armington-CES models where σ is the
elasticity of substitution; mapping between frameworks requires identifying
which moment of the demand-cum-supply system is being estimated.

Sectoral heterogeneity in θ matters because welfare gains from liberalization
in high-θ sectors are smaller per percentage-point reduction in trade costs
than in low-θ sectors. Choosing a θ source is therefore a first-order
calibration decision, not a tuning knob.

---

## Benchmark θ vector

For our replication work we adopt **CP-2015's 99%-sample column with CP's
published corrections** as the starting point for every derived θ vector. We
call this the *benchmark θ*. It is the same vector CP-2015 plug into their
NAFTA simulation (CP-2015 p.20: "use the trade elasticities, θ^j sourced from
the estimates presented in Table 1, column 5") with the substitution rule
they spell out on p.18–19: Basic metals, Machinery n.e.c., and Auto are
replaced by the manufacturing-aggregate elasticity because their per-sector
estimates flip sign under the 97.5% restriction.

Why anchor every downstream vector to this benchmark: it is the *only* θ
vector that CP-2015 themselves operationalize. The Full and 97.5% columns are
robustness reporting. Threading every derivation (AT-2016, AKT-2019, CPRHS-
2017, our 23-sector NAICS) through CP's published benchmark keeps the
provenance chain auditable.

| # | ISIC Rev3 | Sector | Benchmark θ | Source within CP-2015 Table 1 |
|---|---|---|---|---|
| 1 | 01–05 | Agriculture | 9.11 | 99% sample |
| 2 | 10–14 | Mining | 13.53 | 99% sample |
| 3 | 15–16 | Food, beverages, tobacco | 2.62 | 99% sample |
| 4 | 17–19 | Textile, apparel, leather | 8.10 | 99% sample |
| 5 | 20 | Wood | 11.50 | 99% sample |
| 6 | 21–22 | Paper, printing, publishing | 16.52 | 99% sample |
| 7 | 23 | Petroleum, coal | 64.85 | 99% sample |
| 8 | 24 | Chemicals (excl. China) | 3.13 | 99% sample |
| 9 | 25 | Plastics, rubber | 1.67 | 99% sample |
| 10 | 26 | Non-metallic minerals | 2.41 | 99% sample |
| 11 | 27 | **Basic metals** | **4.49** | substitution (mfg-aggregate, 99% sample) |
| 12 | 28 | Fabricated metal products | 6.99 | 99% sample |
| 13 | 29 | **Machinery n.e.c.** | **4.49** | substitution (mfg-aggregate, 99% sample) |
| 14 | 30 | Office, computing | 12.95 | 99% sample |
| 15 | 31 | Electrical machinery | 12.91 | 99% sample |
| 16 | 32 | Radio, TV, communications | 3.95 | 99% sample |
| 17 | 33 | Medical, precision | 8.71 | 99% sample |
| 18 | 34 | **Auto** | **4.49** | substitution (mfg-aggregate, 99% sample) |
| 19 | 35 | Other transport equipment | 0.39 | 99% sample |
| 20 | 36–37 | Other manufacturing | 3.98 | 99% sample |

**Interpretation of the substitution value.** CP-2015 p.19: "we replace them
by the mean estimate for the manufacturing sector." The phrase is mildly
ambiguous; the most defensible numerical reading is CP-2015 Table 1's
bottom-row Aggregate elasticity for the 99% sample = **4.49** (a pooled
regression, N = 5102, s.e. 0.39). A simple per-sector average of the 17
non-substituted manufacturing θ would be ~10.7, which is too far from
anything CP cite to be a plausible reading. Our `akt_replication.py` uses
4.5 (rounded), which is within ~0.2% of 4.49.

**Note on the Chemicals row.** CP-2015 publish two versions of the Chemicals
estimate: with-China and without-China. The benchmark above uses the
without-China value (3.13 for the 99% sample) because that is what CP put in
Table 1. The with-China value (−0.64 under 99% sample) is reported only in
their p.19 footnote.

This vector has not yet been materialized as a parquet/CSV. When Parts 2 and
3 land we'll probably want to store it under
`cprhs/data/inputs/calibrations/cp2015_benchmark.parquet` (20 sectors × 1
value), so derived θ vectors can load + transform it programmatically rather
than copy-pasting numbers.

---

## Original sources

### Caliendo & Parro 2015 — *Restud*

**The primary source used by Albrecht-Tombe 2016, Alvarez-Krznar-Tombe 2019,
and CPRHS 2017.**

- Citation: Lorenzo Caliendo and Fernando Parro, "Estimates of the Trade and
  Welfare Effects of NAFTA," *Review of Economic Studies* 82(1), 1–44 (2015).
  Working paper: NBER WP 18508 (2012).
- Method: GMM tetrad estimator that uses bilateral tariff variation as an
  instrument for trade costs. Exporter and importer fixed effects net out
  multilateral resistance. Estimates `θ_j` separately for each manufacturing
  sub-sector.
- Data: bilateral trade flows and applied tariffs across OECD/non-OECD country
  pairs, year 1993.
- Industry classification: **20 sectors**, ISIC Rev3 mostly at 2-digit grain
  with a few 3-digit splits within machinery and transport. 15 manufacturing
  sub-sectors + Agriculture + Mining + 3 service buckets where θ is not
  estimated (services are non-traded in CP's NAFTA application).
- Reported θ values — **CP-2015 Table 1, full picture** (p.18, year 1993):

  Three sample-restriction columns. The 99% and 97.5% columns drop the bottom
  1% / 2.5% of countries by trade share within each sector and re-estimate.
  CP's preferred / benchmark column is the **99% sample** (their words).
  Each cell reports (θ, s.e., N).

  | # | ISIC Rev3 | Sector | Full sample θ | 99% sample θ | 97.5% sample θ |
  |---|---|---|---|---|---|
  | 1 | 01–05 | Agriculture | 8.11 | 9.11 | 16.88 |
  | 2 | 10–14 | Mining | 15.72 | 13.53 | 17.39 |
  | 3 | 15–16 | Food, beverages, tobacco | 2.55 | 2.62 | 2.46 |
  | 4 | 17–19 | Textile, apparel, leather | 5.56 | 8.10 | 1.74 |
  | 5 | 20 | Wood | 10.83 | 11.50 | 11.22 |
  | 6 | 21–22 | Paper, printing, publishing | 9.07 | 16.52 | 2.57 |
  | 7 | 23 | Petroleum, coal | 51.08 | 64.85 | 61.25 |
  | 8 | 24 | Chemicals | 4.75 | 3.13 | 2.94 |
  | 9 | 25 | Plastics, rubber | 1.66 | 1.67 | 0.60 |
  | 10 | 26 | Non-metallic minerals | 2.76 | 2.41 | 2.99 |
  | 11 | 27 | Basic metals | 7.99 | 3.28 † | **−0.05** |
  | 12 | 28 | Fabricated metal products | 4.30 | 6.99 | 0.52 |
  | 13 | 29 | Machinery n.e.c. | 1.52 | 1.45 † | **−2.82** |
  | 14 | 30 | Office, computing | 12.79 | 12.95 | 11.47 |
  | 15 | 31 | Electrical machinery | 10.60 | 12.91 | 3.37 |
  | 16 | 32 | Radio, TV, communications | 7.07 | 3.95 | 4.82 |
  | 17 | 33 | Medical, precision | 9.98 | 8.71 | 1.97 |
  | 18 | 34 | Auto | 1.01 | 1.84 † | **−3.06** |
  | 19 | 35 | Other transport equipment | 0.37 | 0.39 | 0.53 |
  | 20 | 36–37 | Other manufacturing | 5.00 | 3.98 | 3.06 |
  | **—** | **all mfg pooled** | **Aggregate elasticity** | **4.45** | **4.49** | **3.29** |

  F-test of common θ across sectors: F(17, 7294) = 7.52, Prob > F = 0.00 →
  sectoral heterogeneity is statistically significant.

  † **What CP actually plug into the NAFTA model** is *not* the 99%-sample θ
  for the three sectors marked † (basic metals, machinery n.e.c., auto). Per
  p.18–19 of the WP:

  > "As we compare across estimates, we note that three sectors are not
  > robust since they changed sign with the sample restriction. These sectors
  > are Basic metals, Machinery n.e.c., and Auto. … Our benchmark estimates
  > are the estimates presented in Table 1 for the 99% sample, since they
  > control for outliers. **For the sectors Basic metals, Machinery n.e.c.,
  > and Auto we replace them by the mean estimate for the manufacturing
  > sector.**"

  And p.20:

  > "…use the trade elasticities, θ^j sourced from the estimates presented in
  > Table 1, **column 5**, to quantify the effects of tariff changes."

  Table 1 column 5 = the 99%-sample θ column. So CP's published vector is:
  17 of 18 manufacturing sectors plus Ag + Mining at their 99%-sample Table 1
  estimate, and 3 of 18 manufacturing sectors (basic metals, machinery
  n.e.c., auto) overridden with the "mean estimate for the manufacturing
  sector." The most natural reading of that phrase is the bottom-row
  **Aggregate elasticity for the 99% sample = 4.49** (a pooled regression
  with N = 5102, distinct from a simple per-sector mean). Our
  `akt_replication.py` uses 4.5 for these three, consistent with this
  reading.

- **Petroleum (sector 7) is an extreme outlier** at θ ≈ 51–65 across columns.
  Aggregations that lump petroleum with other chemicals are highly sensitive
  to weights.
- **Chemicals (sector 8) treats China as an outlier.** Footnote on p.19: with
  China included, Chemicals θ = 1.39 (full), −0.64 (99%), −0.93 (97.5%).
  Table 1 excludes China for Chemicals only; China's share is 5% of trade in
  that sector.
- **Robustness with FE.** CP re-estimate Table 1 with importer and exporter
  fixed effects and present those values in Table A.1 of the Appendix
  "Additional Results." We have not yet transcribed Table A.1 into this doc.

### Costinot, Donaldson & Komunjer 2012 — *Restud*

- Citation: Arnaud Costinot, Dave Donaldson, Ivana Komunjer, "What Goods Do
  Countries Trade? A Quantitative Exploration of Ricardo's Ideas," *Review of
  Economic Studies* 79(2), 581–608 (2012). Working paper: NBER WP 16262 (2010).
- Method: cross-sectional estimator identifying θ from the joint variation of
  productivity (proxied by inverse producer prices) and bilateral export
  shares. Different identification strategy than CP-2015 — no tariff data.
- Data: OECD STAN bilateral trade database + GGDC Productivity Level Database,
  year 1997, 21 countries (18 European + Japan, Korea, USA).
- Industry classification: **13 manufacturing sectors**, ISIC Rev3.1 at roughly
  2-digit grain (their Table 1):

  | Code | Sector | ISIC Rev3.1 |
  |---|---|---|
  | Food | Food products, beverages, tobacco | 15–16 |
  | Textiles | Textiles, leather, footwear | 17–19 |
  | Wood | Wood and products of wood and cork | 20 |
  | Paper | Pulp, paper, printing, publishing | 21–22 |
  | Fuel | Coke, refined petroleum, nuclear fuel | 23 |
  | Chemicals | Chemicals | 24 |
  | Plastic | Rubber and plastics | 25 |
  | Minerals | Other non-metallic mineral products | 26 |
  | Metals | Basic + fabricated metal products | 27–28 |
  | Machinery | Machinery n.e.c. | 29 |
  | Electrical | Electrical + optical equipment | 30–33 |
  | Transport | Transport equipment | 34–35 |
  | Misc. Manuf. | Manufacturing n.e.c. | 36–37 |

- Reported θ: single sector-pooled value **θ ≈ 6.53**. CDK identify θ from
  cross-sectoral productivity differences, so do not produce a per-sector
  vector — their estimate is the elasticity of bilateral exports with respect
  to observed productivity, common across sectors.
- Less commonly used as a direct calibration source because of the single-θ
  output, but the sector list is useful as an alternative concordance against
  CP-2015 (and overlaps closely with AT-2016 Table 9).

### Imbs & Méjean 2017 — *Review of International Economics*

- Citation: Jean Imbs and Isabelle Méjean, "Trade Elasticities," *Review of
  International Economics* 25(2), 383–402 (2017). Earlier CEPII / NBER versions
  circulated as "Elasticity Optimism" (2010, 2013).
- Method: estimate sector-specific Armington elasticity `σ_s` per HS-3 product
  using cross-country bilateral trade plus tariff data, then aggregate to
  country-specific trade elasticities. Effectively translates to EK θ under
  the σ–1 mapping for sector-pooled exercises.
- Data: cross-section of 28 developed and developing countries.
- Industry classification: HS-3 / 3-digit ISIC product level (more than 60
  product-sector cells per country). Country-specific aggregates are then
  computed using each country's import-mix weights.
- Reported θ values: median sector-level σ ≈ 5 (so EK θ ≈ 5 under translation),
  but cross-sector dispersion is wide and *cross-country* dispersion of the
  aggregate is also large. They argue earlier work mis-estimated aggregate
  elasticities by failing to account for country-specific import mixes.
- Provenance: this is the "country-specific θ" benchmark in the literature;
  not used by AT/AKT but worth knowing about when the calibration target is a
  non-OECD country.

### Fontagné, Guimbard & Orefice 2022 — *JIE* (i-MIP resource)

- Citation: Lionel Fontagné, Houssein Guimbard, Gianluca Orefice, "Tariff-based
  product-level Trade Elasticity," *Journal of International Economics* 137,
  103593 (2022). Public dataset hosted by i-MIP (Institut de Macroéconomie
  Internationale Paris): https://i-mip.eu/en/resources/tariff-based-product-level-trade-elasticity/
- Method: tariff-based identification at HS-6, using bilateral tariff variation
  across product-country pairs as the cost-shifter, with importer-time and
  exporter-time fixed effects. Validated out-of-sample against post-EPA
  US–Chile import flows.
- Data: universe of country pairs, 2001–2016 (tariff data extended through
  2019). Trade flows from BACI; tariffs from MAcMap-HS6.
- Industry classification: the package ships **multiple concordance tables**:
  - HS-6 (raw estimation grain)
  - HS-4 (aggregated)
  - GTAP 10/11
  - WIOD (2013 and 2016 vintages)
  - OECD TiVA
  - OECD ICIO

  Each concordance gives a product/sector-specific θ. The HS-6 vector has
  ~5000 entries; aggregated vectors are much smaller.
- Reported variable: trade elasticity `θ_p` (i.e., the elasticity of import
  demand with respect to tariff-induced price changes — the EK-style trade
  elasticity, not the Armington substitution elasticity σ).
- Provenance / access: replication materials on Mendeley, source code in STATA
  and SAS, datasets via Google Drive / Dropbox.
- This is the most recent and finest-grained tariff-based θ vector available
  publicly. Their ICIO concordance is the natural input for our
  `canada_2021_partners` calibration if we switch from CP-2015-based θ to a
  more contemporary vector.

### Eaton & Kortum 2002 — *Econometrica*

- Citation: Jonathan Eaton and Samuel Kortum, "Technology, Geography, and
  Trade," *Econometrica* 70(5), 1741–1779 (2002). Working paper: NBER WP 6253
  (1997, under the title "Technology and Bilateral Trade").
- Method: gravity equation for bilateral trade in manufactures derived from
  the Ricardian-Fréchet model. Identifies θ jointly with state-of-technology
  and geographic-barrier parameters from a system of equations.
- Data: 19 OECD countries, 1990 manufacturing trade + producer prices + wages.
- Industry classification: **aggregate** — all manufacturing pooled into one
  sector.
- Reported θ values (their Table VIII, "Summary of Parameters"):
  - **θ = 8.28** (baseline, method-of-moments using geographic / source-fixed
    effects from Section 3 of the paper)
  - **θ = 3.60** (alternative, 2SLS using wage data correcting for the
    endogeneity of wages — Section 5.2)
  - **θ = 12.86** (alternative, 2SLS using price data correcting for
    measurement error in prices — Section 5.3)
- The canonical single-θ value cited in subsequent work is θ ≈ 8.28. Lower
  estimates (e.g. 3.60) are sometimes cited as the "preferred wage-data" value.

### Simonovska & Waugh 2014 — *JIE*

- Citation: Ina Simonovska and Michael E. Waugh, "The Elasticity of Trade:
  Estimates and Evidence," *Journal of International Economics* 92(1), 34–50
  (2014). Working paper: NBER WP 16796 (2011).
- Method: simulated method-of-moments correction of the EK-2002 estimator.
  They argue EK's price-based estimator suffers from measurement-error bias
  that pushes θ upward; their bias-corrected estimator yields a lower value.
- Data: 123 countries, year 2004 trade flows + 2003–2005 prices from the World
  Bank ICP.
- Industry classification: **aggregate** — manufactures pooled.
- Reported θ: **θ ≈ 4.0** (preferred), with the implication that the gains
  from international trade are roughly *double* those implied by EK's higher
  estimates.
- This is the actual source of the "θ = 4 benchmark" that often appears in
  later quantitative work (and that we use as `UNIFORM_THETA = 4.0` in
  `scripts/counterfactual_alberta.py`). The label "Caliendo-Parro convention"
  for θ = 4 is a misattribution — CP-2015 themselves do not endorse a single
  scalar.

---

## Surveys and conventions

These do not estimate θ; they synthesize the literature and provide
widely-cited reference values.

### Costinot & Rodríguez-Clare 2014 — Handbook of Int'l Econ Vol 4

- Citation: Arnaud Costinot and Andrés Rodríguez-Clare, "Trade Theory with
  Numbers: Quantifying the Consequences of Globalization," Chapter 4, in
  *Handbook of International Economics* Vol 4, eds. Gopinath, Helpman, Rogoff,
  Elsevier (2014). Working paper: NBER WP 18896 (2013).
- Role: methodological survey of quantitative gravity models. They do not
  estimate θ themselves. For their multi-sector quantitative exercises (Tables
  1–3) they use *Caliendo & Parro 2010* (the working paper version of
  CP-2015) for sectoral θ_j.
- The "θ = 5" reference value that AT-2016 cites: CRC discuss θ = 5 in their
  concluding remarks as a representative single-θ value implying optimal
  tariffs of 20% in the one-sector Armington benchmark. The attribution of "θ
  = 5 for services" to CRC is *interpretive* — CRC do not specifically argue
  for θ = 5 in services. The convention reflects literature consensus that
  services trade elasticities are not credibly identified by available
  methods, with θ in the 4–6 range serving as a default.

### Head & Mayer 2014 — Handbook of Int'l Econ Vol 4

- Citation: Keith Head and Thierry Mayer, "Gravity Equations: Workhorse,
  Toolkit, and Cookbook," Chapter 3, in *Handbook of International Economics*
  Vol 4, eds. Gopinath, Helpman, Rogoff, Elsevier (2014). Working paper: CEPII
  WP 2013-27.
- Role: methodological cookbook for gravity-equation estimation, with a
  meta-analysis of trade-elasticity estimates across the published literature.
  Supplementary materials online at https://sites.google.com/site/hiegravity/
- Reported summary: the literature's modal trade elasticity is around 5, with
  most estimates falling in the 4–6 range. They advocate not relying on a
  single estimator; instead reporting results under several θ values to
  document robustness.

---

## Cross-paper notes for the QGE calibration

1. **CP-2015 is upstream of AT, AKT, and CPRHS.** Any of these "uses
   Caliendo-Parro 2015" claims means a re-aggregation of the 20-sector CP
   vector above. Verifying a derived calibration always reduces to two
   questions: (a) what concordance from the 20 CP sectors to the target
   taxonomy? (b) what weighting scheme (simple mean, trade-weighted, harmonic)
   for the aggregation?
2. **Sample column matters in CP-2015.** CP report three sample columns —
   Full, 99% sample, 97.5% sample — and they differ by factors of 2–5 in
   several sectors. CP's benchmark is the 99% sample.
3. **CP's "Table 1, column 5" is NOT a clean vector.** CP override three
   sectors (basic metals, machinery n.e.c., auto) with the manufacturing
   pooled aggregate (≈ 4.49 under the 99% sample) because the per-sector
   estimates flip sign under the 97.5% restriction. Any code that claims to
   "use the 99% sample" needs to be explicit about whether it applies this
   substitution. Our `akt_replication.py` does (with 4.5 for the three
   sectors); `cprhs/qge/io.py` `_CPRHS_T` does **not** (it uses CP's Full
   sample column directly, including 1.52 for Machinery and 1.01 for Auto).
4. **Petroleum is an extreme outlier.** θ_Petroleum ≈ 51 (full or 99%
   sample) or 65 (rounded in some sources). Aggregations that combine
   petroleum with chemicals and plastics are highly weight-sensitive. AT-2016
   reports θ ≈ 19.16 for ISIC 23–25; this is an unweighted mean and may not
   be the right aggregation for our purposes.
5. **CDK-2012 industry list is the cleanest match for AT-2016 Table 9.** CDK's
   13 manufacturing sectors line up nearly 1-to-1 with AT's manufacturing
   rows. CDK's single θ ≈ 6.53 is a useful sanity check on aggregations.
6. **Imbs-Méjean 2017 and Fontagné-Guimbard-Orefice 2022 are the relevant
   contemporary alternatives** to CP-2015 if we want a vector estimated on
   data from the 2000s rather than 1993. Both are tariff/HS-based.
7. **Single-θ shortcuts.** EK-2002 (θ = 8.28), SW-2014 (θ = 4.0), and Head-Mayer
   modal (θ ≈ 5) are the three canonical scalar choices. Our current
   `counterfactual_alberta.py` uses SW-2014's θ = 4.0. CP-2015 themselves
   report a pooled estimate (Table 1 last row) of 4.45 / 4.49 / 3.29 across
   their three sample columns — close to SW-2014.
8. **Translating frameworks.** Armington σ from Broda-Weinstein or
   Imbs-Méjean is *not* directly comparable to EK θ; the equivalence θ ≡ σ − 1
   holds only under specific assumptions (single-sector CES, no firm-level
   heterogeneity). Use translations with care.

## Part 2a — Derive AT-2016 Table 9 from CP-2015 (forward — DONE)

AT-2016 publish a θ vector for 22 ISIC Rev3 sectors (Table 9, paper p.26).
AT's footnote: "trade elasticity is from the Caliendo and Parro (2015)
estimates, averaged up to a slightly higher level of aggregation. Sectors 40
and above have elasticities of 5, consistent with Costinot and Rodriguez-
Clare (2014)."

`scripts/derive_at_theta.py` tests 24 combinations of (CP-2015 column,
aggregation scheme) against AT Table 9. Schemes: simple arithmetic mean of
θ, harmonic mean of θ, N-weighted (CP's 99%-sample N), ISIC-2-digit-code-
count weighted. Columns: Full sample, 99% sample, 99% sample + corrections
(our benchmark), 97.5% sample.

**Convention identified: AT-2016 uses CP-2015's *Full sample* column with
simple arithmetic mean (essentially identical to ISIC-2-digit-count weighted
mean — they differ only for the Food+Textiles row, where ISIC-count
weighting fits AT slightly better).**

Score on the 8 AT manufacturing rows (RMSE / MAE / exact-to-3-decimals):

| (column, scheme) | RMSE | MAE | exact/8 |
|---|---|---|---|
| `(full, ISIC_theta)` | **0.072** | **0.027** | **7** |
| `(full, arith_theta)` | 0.179 | 0.065 | 7 |
| `(99sub, arith_theta)` ← benchmark + simple mean | 3.062 | 1.955 | 0 |
| `(99, arith_theta)` ← 99% raw + simple mean | 3.064 | 1.944 | 0 |
| `(99sub, ISIC_theta)` ← benchmark + ISIC-weighted | 3.086 | 2.023 | 0 |

Per-sector AT vs derived for the winning scheme `(full, ISIC_theta)`:

| AT sector | ISIC | AT θ | derived θ | residual |
|---|---|---|---|---|
| Agriculture, Mining | 01–14 | 11.92 | 11.915 | −0.005 |
| **Food, Textiles** | 15–19 | **4.56** | **4.356** | **−0.204** |
| Wood | 20 | 10.83 | 10.830 | 0.000 |
| Paper | 21–22 | 9.07 | 9.070 | 0.000 |
| Chemicals, Rubber | 23–25 | 19.16 | 19.163 | +0.003 |
| Metals | 26–28 | 5.02 | 5.017 | −0.003 |
| Equipment, Vehicles | 29–35 | 6.19 | 6.191 | +0.001 |
| Manufacturing, n.e.c. | 36–37 | 5.00 | 5.000 | 0.000 |

### Implication for the benchmark

**AT-2016 does not aggregate the benchmark we defined.** Our benchmark is CP's
99%-sample column with corrections (basic metals, machinery n.e.c., auto
overridden by 4.49). AT-2016 instead uses **the Full sample column with no
corrections** — pulling in 1.52 for Machinery, 1.01 for Auto, 7.99 for Basic
metals, etc. AT's footnote ("from the Caliendo and Parro (2015) estimates")
elides this choice; CP-2015 themselves do *not* operationalize the Full
sample column (they use the 99% sample for the NAFTA simulation, p.20).

**We keep our benchmark unchanged** (it remains the column CP themselves
adopt, and the natural input for any new calibration). Part 2a records that
AT use a *different* column from the same paper.

### Unexplained: Food, Textiles

AT publish θ = 4.56 for ISIC 15–19, but no (column, scheme) combination
tested reproduces it. Closest:

| candidate | derived | residual to 4.56 |
|---|---|---|
| `(99, ISIC_invtheta)` | 4.41 | −0.15 |
| `(full, ISIC_theta)` ← global winner | 4.36 | −0.20 |
| `(full, arith_theta)` | 4.06 | −0.50 |
| CP-2015 99%-sample manufacturing aggregate | 4.49 | −0.07 |

Plausible explanations:
1. **Typo in AT Table 9** — 4.36 → 4.56 is a single-digit transposition.
2. **External trade-share weights** — AT's own Canadian IO data gives
   Textile a higher trade share than Food. A 1:2 Food:Textile weighting on
   the Full sample reproduces 4.56 *exactly* — (1·2.55 + 2·5.56)/3 = 4.557.
   But the ISIC-2-digit code count weighting (which would be the natural
   weight without trade data) gives 2:3 (= 4.356), not 1:2.
3. **Aggregate substitution** — AT replaces Food+Textiles with the CP
   manufacturing aggregate (4.49), with rounding to 4.56. Unusual choice
   but explains the value.

We have not chased this further — the global aggregation convention is
clear, and Food/Textiles is a single outlier worth ≤ 0.2 in any one row.

### Downstream: Tombe & Winter (2021) reuse AT-2016 Table 9 verbatim

Tombe & Winter (2021, *Canadian Journal of Economics* 54(2), 522–556) adopt
the AT-2016 Table 9 θ vector **as-is** for their 22-sector Canadian
calibration. The values appear in:

- the data appendix (Borealis [doi:10.5683/SP3/SHDOHY](https://doi.org/10.5683/SP3/SHDOHY)
  archive) as the `theta` variable in `maindata.mat`,
- Table 10 (p.33) of the August 2019 working paper
  (https://www.trevortombe.com/files/tombe_winter_2019.pdf), reprinted in
  human-readable form,
- the published CJE 2021 article only in prose (p.540), citing CP-2015 +
  CRC-2014 without listing values.

The published article's `tw/data/inputs/canada_2010/sectoral_dispersion.parquet`
in our port stores `T_j = 1/θ^j` from the same numbers. So the provenance
chain CP-2015 → AT-2016 → TW-2021 is a literal copy-paste at each step —
no re-aggregation between AT and TW.

## Part 2b — Infer AKT-2019 effective θ via Head-Ries matching (inverse)

AKT-2019 do **not** publish their θ vector. Their Tables 1–2 report
non-geographic trade barriers per sector and per province; these are computed
from a Head-Ries index that depends on θ. This is an **inverse problem**:
find the 9-sector θ vector that, when plugged into AKT's Head-Ries pipeline
applied to public StatCan data, best reproduces AKT's published numbers.

Method (`scripts/akt_replication.py` is already most of the way there):
1. Aggregate the 20-sector benchmark to AKT's 9 goods sectors via the S&U
   product-code concordance already in the script.
2. Try multiple candidate vectors:
   - Direct CP-99-with-substitution benchmark aggregated to AKT-9 (this is
     what `_CP_99` currently does, with the dominant CP sub-sector per AKT
     bucket).
   - Trade-weighted using StatCan interprovincial flow shares as weights.
   - Range of single-θ pooled values (4, 5, 6, 8.28) as ablation baselines.
3. Score each candidate by:
   - **Primary:** mean absolute deviation from AKT Table 1 (per-sector NG)
     and Table 2 (per-province NG) for 2015.
   - **Secondary:** AKT's headline "interprovincial all-goods NG ≈ 19%"
     summary statistic.
4. The θ vector with the lowest score is our inferred AKT-effective θ.
   Document it next to the AT-2016 derived vector for comparison.

Caveat: AKT also use proprietary StatCan data (services trade between
provinces and the USA, pre-2007 interprovincial detail) that we cannot
replicate exactly. The inverse problem is therefore only well-posed up to
the residual model misspecification from those data gaps. Plan to report the
fit quality, not just the inferred θ.

## Part 3 — Rebuild sectoral_dispersion.parquet from the benchmark

`cprhs/data/inputs/canada_2010/sectoral_dispersion.parquet` is currently
built by `convert_statcan._cprhs_theta_per_sector()`, which uses CP-2015's
Full sample column (not the benchmark 99%-sample-with-substitutions) and a
different aggregation rule from what AT-2016 or AKT-2019 use. The mismatch
shows up most starkly for Petroleum and Chemicals (3.60 vs benchmark
implies ~19) and Transportation Equipment (1.01 vs benchmark implies ~4–6
after substitution).

Once Parts 2a/2b clarify our preferred aggregation conventions, rebuild
`_cprhs_theta_per_sector()` to:
1. Start from the benchmark θ vector (20 CP-2015 sectors).
2. Apply the concordance CP-2015 → 23 NAICS sectors that mirrors AT-2016's
   convention where possible. The NAICS-to-CP-ISIC mapping is straightforward
   except for two sectors that split CP rows differently from AT (our
   "Wood, Paper, Printing" lumps CP Wood + Paper; our "Metals and Machinery"
   straddles CP ISIC 27–29).
3. Regenerate `sectoral_dispersion.parquet` for both `canada_2010` and
   `canada_2021_partners`. Document the diff against the current vector and
   the welfare-gain sensitivity in a separate validation note.
