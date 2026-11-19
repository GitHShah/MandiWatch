# Mandi Watch — Decisions Log

This log tracks every data cleaning filter, record drop, modeling choice, and architectural assumption with timestamps, reasoning, and impacted row counts.

---

### Decision 001: Archive Legacy Flawed Work to `_legacy_archive/`
- **Date:** 2026-11-03
- **Decision:** Moved 38 legacy files and directories (scratches, un-scoped cache, ad-hoc onion scripts, monthly forecasting models) into `_legacy_archive/` preserving original relative folder hierarchies. Retained raw CSVs in `data/raw/` immutable.
- **Reason:** Legacy code suffered from critical accuracy killers: modeled monthly raw prices instead of weekly log prices, pooled varieties and grades together, hard-coded constants in Python instead of `config.yaml`, and had leftover scratch builds.
- **Impacted Files:** 38 items moved to `_legacy_archive/`. 0 raw records modified (48,229,507 raw rows preserved untouched in `data/raw/`).
- **Status:** Complete.

---

### Decision 002: Raw Dataset Profiling and Section 2 CONFIRM Audit
- **Date:** 2026-11-04
- **Decision:** Profiled all 12 raw CSVs (`2015.csv` to `2026.csv`) with DuckDB in read-only mode. Computed cardinalities, annual coverage, missing rates, price distributions, price triplet bounds, and name variant candidates without dropping or cleaning any data.
- **Reason:** Satisfy Phase 0 CONFIRM items in Section 2 of `MANDI_WATCH_PROJECT_SPEC.md` and establish verified ground truth before implementing Phase 1 cleaning rules.
- **Row Counts / Findings:** Total 48,229,507 rows. 0 missing values in categorical and date fields. Non-positive min/max prices present in ~1.01% / 0.99% of rows. Inverted price triplets (`Modal < Min` or `Modal > Max`) identified in 498,989 rows (1.03%). Generated `reports/tables/data_profile.md` and `data/interim/name_fix_candidates.csv` (29 fuzzy pairs).
- **Status:** Complete. Awaiting user clarification on price semantics, units, and dataset confidence.

---

### Decision 003: Full Phase 1 Data Cleaning, Scope Selection, and Weekly Aggregation
- **Date:** 2026-11-04
- **Decision:** Executed complete Phase 1 pipeline (Steps 1 to 13 of Section 5):
  1. Standardized text with casing/whitespace normalization and applied 6 auto-approved mappings from `name_fixes.csv`.
  2. Verified dates and positive modal prices (0 rows dropped).
  3. Deduplicated exact keys on (market, commodity, variety, grade, date) using median conflict resolution: 9,587 duplicate groups resolved, 48,229,507 -> 48,219,920 rows (removed 9,587 rows, 0.02%).
  4. Flagged and dropped rolling MAD outliers (>5.0 MAD from centered 15-day rolling median) and unit errors (~10x/100x series median): 4,936,279 MAD outliers and 95,133 unit errors dropped (5,012,780 rows, 10.40%). 43,207,140 rows retained.
  5. Detected flat runs (>= 14 consecutive identical prices): flagged 7,624,616 rows across 38,723 series (`is_flat_run=True`), all retained.
  6. Saved `data/interim/clean_daily_all.parquet` (43,207,140 rows, 173.9 MB).
  7. Automated scope selection: ranked commodities and states by distinct reporting weeks and markets with >= 104 weeks history. Selected top 8 commodities (`Potato`, `Onion`, `Wheat`, `Tomato`, `Paddy (Dhan)(Common)`, `Brinjal`, `Green Chilli`, `Cauliflower`) and top 3 states (`Maharashtra`, `Madhya Pradesh`, `Uttar Pradesh`). Saved `reports/tables/scope_ranking.csv`.
  8. Retained single mode variety and mode grade per commodity: 4,601,519 in-scope rows -> 2,022,145 rows (dropped 2,579,374 rows, 56.05% due to non-primary variety/grade elimination). Saved `data/interim/clean_daily.parquet` (7.88 MB).
  9. Aggregated to ISO weeks per series (`market + commodity`): 436,045 weekly records. Handled year-boundary week 53 / Jan 1 edge cases.
  10. Series quality filter (>= 104 weeks, <= 40% missing weeks): kept 955 series (354,712 weekly records), dropped 1,608 series. Saved `reports/tables/series_kept.csv` and `series_dropped.csv`.
  11. Linear short gap interpolation (gaps <= 2 weeks) for lag features only: 8,395 weeks interpolated, marked by `is_interpolated=True`. Gaps > 2 weeks left un-interpolated. Saved `data/processed/weekly.parquet` (410,230 grid rows, 3.46 MB).
  12. Generated coverage visuals: `reports/figures/market_year_reporting_heatmap.png` and `reports/figures/series_count_by_year.png`.
- **Reason:** Complete Phase 1 spec compliance, eliminating variety/grade noise, enforcing series stationarity and quality standards, and providing clean weekly data for modeling and EDA.
- **Status:** Complete. All 9 unit tests passing.


---

### Decision 004: Validation Error Diagnostics & Concentration Analysis
- **Date:** 2026-11-04
- **Decision:** Conducted comprehensive validation set error diagnostics without touching the test set. Segmented 39,433 validation rows across commodity, season, series length, weekly reporting frequency, gap status, and flat run status. Saved tables to `reports/tables/val_breakdown_*.csv`.
- **Key Findings:**
  1. **Commodity Dispersion:** Forecast error is heavily concentrated in volatile perishables: Tomato (MAE ₹416.79/q, MAPE 18.80%), Green Chilli (MAE ₹333.66/q, MAPE 12.11%), Cauliflower (MAE ₹276.48/q, MAPE 16.59%), and Onion (MAE ₹272.86/q, MAPE 11.22%). Foodgrains exhibit minimal error (Wheat MAE ₹51.97/q, MAPE 2.10%; Paddy MAE ₹47.27/q, MAPE 2.21%).
  2. **Seasonal Vulnerability:** Monsoon (MAE ₹263.37/q) and Post-Monsoon (MAE ₹245.24/q) have significantly higher errors than Summer (MAE ₹159.89/q) and Winter (MAE ₹181.02/q) due to monsoon logistics disruption, flash floods, and variable harvest timing.
  3. **Reporting Density Penalty:** Unreported gap weeks have an MAE of ₹286.41/q (13.61% MAPE) compared to ₹197.67/q (8.89% MAPE) for mandis with >5 reports/week.
  4. **Flat Run Stale Prices:** Flat run rows exhibit artificially low error (MAE ₹99.03/q) due to copy-pasting of static prices by local enumerators.
- **Status:** Complete.

---

### Decision 005: Validation-Based Model Ablation & Selection of Delta-Target Formulation
- **Date:** 2026-11-04
- **Decision:** Tested 6 model variations strictly on the validation set against the baseline global model (Val MAE: ₹213.34/q, Val MAPE: 9.71%):
  - *(a) Dropping sparsest series (<100 obs in train):* Worsened Val MAE to ₹215.30/q (+1.96 ₹/q). Rejected.
  - *(b1) Shorter training window (2020+ only):* Worsened Val MAE to ₹215.21/q (+1.87 ₹/q). Rejected.
  - *(b2) Recency sample weighting (3-year half-life):* Worsened Val MAE to ₹214.25/q (+0.91 ₹/q). Rejected.
  - *(c) Era indicator feature:* Val MAE ₹213.05/q (-0.29 ₹/q). Statistically indistinguishable from noise (~0.1%). Rejected.
  - *(d) Simpler core feature set (12 features):* Severely worsened Val MAE to ₹224.33/q (+10.99 ₹/q). Rejected.
  - *(e) Per-commodity separate models:* Severely worsened Val MAE to ₹222.47/q (+9.13 ₹/q) due to loss of cross-mandi shared feature representations. Rejected.
  - *(f) Predicting price change vs last observed price ($\Delta y = \log(y_{t+h}) - \log(y_t^{\text{obs}})$):* **Decisively improved Val MAE to ₹201.17/q (-12.17 ₹/q, -5.7% error reduction; Val MAPE 9.20%).** Accepted.
- **Final Test Set Evaluation (Re-run ONCE):**
  - Held-out test set evaluation of the Delta formulation confirmed the improvement: **Test MAE dropped from ₹169.62/q to ₹157.79/q (Test MAPE: 7.88%)**, beating the persistence baseline (₹190.08/q) by ₹32.29/quintal (a 17.0% overall error reduction).
- **Status:** Complete.

---

### Decision 006: Spatial Recommender Engine Architecture & Transport Formulation
- **Date:** 2026-11-04
- **Decision:** Implemented `src/recommender.py` utilizing the precomputed 2-week-ahead LightGBM point forecasts and calibrated 10-90 conformal prediction intervals:
  1. Farmer coordinates mapped to district centroids via `data/interim/district_centroids.parquet`.
  2. Candidate markets filtered within configurable radius (default: 150 km) that passed the historical quality filter (>=104 weeks history).
  3. Distance calculated via haversine formula with a road tortuosity factor of 1.35 (accounting for real-world road network detour vs straight-line distance).
  4. Transport cost per quintal computed as `distance_km * transport_rate` (default: 0.25 INR/q/km, explicitly flagged as an assumed user parameter).
  5. F1 Farmer objective: `net_price = forecast_price - transport_cost`, ranked descending (top 5 options returned).
  6. B1 Bulk Buyer objective: `landed_cost = forecast_price + transport_cost`, `total_cost = landed_cost * quantity_quintals`, ranked ascending.
  7. Structured safety warnings surfaced for: insufficient candidates (<3 markets), high uncertainty (p90 - p10 > 30% of forecast), stale/weak history, and overlapping confidence intervals where price differences are statistically within forecast error bounds.
  8. Interactive Folium mapping with distinct farmer/buyer origin icons, candidate color coding, and road connection polylines.
- **Reason:** Provide practical, transparent decision support for agricultural transport economics while guarding against blind overconfidence.
- **Status:** Complete.

---

### Decision 007: Held-Out Test Evaluation of Recommender Engine (26,352 Scenarios)
- **Date:** 2026-11-04
- **Decision:** Evaluated recommender decisions across 26,352 pseudo-farmer test cases (every test week, commodity, and district in scope across 2024–2026) against realized market prices:
  1. Benchmark comparison against: Oracle Best Market (hindsight upper bound), Nearest Market (default farmer baseline), and Historical Best Market (expanding past mean).
  2. Metrics calculated: Realized regret (₹/q and %), top-1 hit rate, top-3 hit rate, and Spearman rank correlation between forecast and realized net orderings.
  3. Hypothesis testing: Paired Wilcoxon signed-rank and paired t-test on net price differences against nearest market.
  4. Sensitivity analysis: Transport rate varied across [0.00, 0.15, 0.25, 0.50] INR/q/km.
  5. Spatial Spread vs MAE Ratio: Validated ranking viability by confirming that geographic price spread strictly exceeds forecast MAE across all 8 commodities ($1.83\times$ to $5.53\times$).
- **Key Findings:**
  - Mean regret of model: ₹108.16/q (4.54%) [Median: ₹10.00/q] vs Nearest Market regret of ₹338.65/q (13.50%).
  - Net economic gain over nearest market: **+₹230.49/quintal** ($p < 10^{-15}$, Cohen's $d = 0.459$).
  - Top-1 hit rate: 44.60% (vs 18.25% nearest); Top-3 hit rate: 71.34% (vs 42.10% nearest).
  - High sensitivity robustness: >91% of recommended markets are invariant across transport rates from 0.00 to 0.50 INR/q/km.
- **Status:** Complete. Saved all summary tables and figures.

---

### Decision 008: Experimental Timing Decision Engine (F2 Sell/Hold and B2 Buy/Wait)
- **Date:** 2026-11-04
- **Decision:** Implemented Section 12 experimental features (`src/timing.py`):
  1. Configured holding costs in `config.yaml` with explicit differentiation between perishable commodities (Tomato: ₹50, Green Chilli: ₹40, Cauliflower: ₹35, Brinjal: ₹25 /q/wk) and durable commodities (Onion: ₹15, Potato: ₹12, Wheat: ₹8, Paddy: ₹8 /q/wk).
  2. Formulated target: probability of price increase exceeding holding hurdle, trained LightGBM binary classifier (`models/lgb_timing_clf_h2.joblib`) on leak-free feature table (`features_h2.parquet`).
  3. Evaluated on held-out test split (2024–2026; 34,969 cases) comparing Model Timing vs Always Sell Now vs Always Sell at Seasonal Peak.
  4. Integrated an expandable experimental timing section in both Farmer and Buyer tabs of `app/streamlit_app.py`.
- **Key Empirical Findings:**
  - Model Timing achieves 83.9% accuracy and selectively recommends holding in only 12.5% of test weeks, delivering an average gain of +₹31.35/quintal (+1.50%).
  - Naive seasonal holding ("Always Sell at Seasonal Peak") severely loses money (-₹131.94/q vs selling now) because physical storage, spoilage, and finance costs rapidly outpace nominal price appreciation.
  - Crucial Limitation Logged: Unforecastable exogenous shocks (weather anomalies, flash floods, export bans, stock limits) cannot be predicted from price history.
- **Status:** Complete. Saved `reports/tables/timing_evaluation.csv`.
