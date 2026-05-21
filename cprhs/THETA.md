# Trade elasticity θ — sources and concordances

Reference document for the trade-elasticity parameter `θ_j` used in the QGE
calibration (`sectoral_dispersion.parquet` stores `T_j = 1/θ_j`). Existence of
this file is a one-stop catalogue so that the choice of θ_j vector for any
given calibration can be traced back to a primary academic estimate.

This is part 1: **original sources of θ** and the industry classification each
paper estimates against. Subsequent sections (to be filled in) will document
how derived works (Albrecht-Tombe 2016, Alvarez-Krznar-Tombe 2019, CPRHS 2017,
our own builds) re-aggregate the originals.

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
- Reported θ values (per CP-2015 Table 1, "99% sample" column — the authors'
  preferred benchmark, winsorizing the top 1% of trade-growth observations):

  | # | ISIC Rev3 | Sector | θ |
  |---|---|---|---|
  | 1 | 01–05 | Agriculture | 9.11 |
  | 2 | 10–14 | Mining | 13.53 |
  | 3 | 15–16 | Food, beverages, tobacco | 2.62 |
  | 4 | 17–19 | Textile, apparel, leather | 8.10 |
  | 5 | 20 | Wood | 11.50 |
  | 6 | 21–22 | Paper, printing, publishing | 16.52 |
  | 7 | 23 | Petroleum, coal | 64.85 |
  | 8 | 24 | Chemicals | 3.13 |
  | 9 | 25 | Plastics, rubber | 1.67 |
  | 10 | 26 | Non-metallic minerals | 2.41 |
  | 11 | 27 | Basic metals | 3.28 † |
  | 12 | 28 | Fabricated metal products | 6.99 |
  | 13 | 29 | Machinery n.e.c. | 1.45 † |
  | 14 | 30 | Office, computing | 12.95 |
  | 15 | 31 | Electrical machinery | 12.91 |
  | 16 | 32 | Radio, TV, communications | 3.95 |
  | 17 | 33 | Medical, precision | 8.71 |
  | 18 | 34 | Auto | 1.84 † |
  | 19 | 35 | Other transport equipment | 0.39 |
  | 20 | 36–37 | Other manufacturing | 3.98 |

  † CP themselves flag these three sectors (basic metals, machinery n.e.c.,
  auto) as having an unreliable / sign-flipped estimate under winsorization
  and recommend replacing with the manufacturing mean (~4.5).

- Petroleum is an extreme outlier (θ ≈ 51–65 depending on column). Any
  aggregation that lumps petroleum with other chemicals is highly sensitive to
  weights.
- CP do not report a pooled all-manufacturing θ; downstream papers compute
  one if they need one.

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
2. **Sample column matters in CP-2015.** CP report both "All sample" and "99%
   sample" columns; values can differ by factors of 2–5 in some sectors. The
   user-saved `_CP_99` dict in `scripts/akt_replication.py` uses the
   "99% sample" benchmark column, which is the one CP themselves recommend.
3. **Petroleum is an extreme outlier.** θ_Petroleum ≈ 51 (99% sample) or 65
   (all sample). Aggregations that combine petroleum with chemicals and
   plastics are highly weight-sensitive. AT-2016 reports θ ≈ 19.16 for ISIC
   23–25; this is an unweighted mean and may not be the right aggregation for
   our purposes.
4. **CDK-2012 industry list is the cleanest match for AT-2016 Table 9.** CDK's
   13 manufacturing sectors line up nearly 1-to-1 with AT's manufacturing
   rows. CDK's single θ ≈ 6.53 is a useful sanity check on aggregations.
5. **Imbs-Méjean 2017 and Fontagné-Guimbard-Orefice 2022 are the relevant
   contemporary alternatives** to CP-2015 if we want a vector estimated on
   data from the 2000s rather than 1993. Both are tariff/HS-based.
6. **Single-θ shortcuts.** EK-2002 (θ = 8.28), SW-2014 (θ = 4.0), and Head-Mayer
   modal (θ ≈ 5) are the three canonical scalar choices. Our current
   `counterfactual_alberta.py` uses SW-2014's θ = 4.0.
7. **Translating frameworks.** Armington σ from Broda-Weinstein or
   Imbs-Méjean is *not* directly comparable to EK θ; the equivalence θ ≡ σ − 1
   holds only under specific assumptions (single-sector CES, no firm-level
   heterogeneity). Use translations with care.

## To-be-filled sections

- **Derived calibrations.** Concordance of CP-2015 (20 sectors) →
  AT-2016 (22 ISIC Rev3 sectors), AKT-2019 (9 S&U-coded goods sectors),
  CPRHS-2017 (15 tradable + 11 non-tradable NAICS sectors), and our
  `canada_2010` / `canada_2021_partners` 23-sector taxonomy. With aggregation
  weights for each step.
- **Validation.** Side-by-side comparison of θ vectors implied by each
  derivation path; sensitivity of welfare gains to the choice.
- **Recommended default.** Which θ vector should `sectoral_dispersion.parquet`
  carry by default, with what aggregation method, and where the alternatives
  live.
