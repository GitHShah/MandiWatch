# Mandi Watch: Project Specification (for the coding agent)

You are building a data science course project called **Mandi Watch**. Read this whole file before writing any code. Follow the phases in order. Where something is marked **CONFIRM**, inspect the real data or ask the user before assuming.

---

## 1. Goal

An assistant for farmers and bulk buyers built on about 10 years of mandi (agricultural market) price data.

- **Farmer interface**
  - App F1 (**core**): given commodity, time period and the farmer's district, recommend the best markets to sell in (after transport cost).
  - App F2 (**stretch, experimental**): should the farmer sell now or hold for a short time.
- **Bulk buyer interface**
  - App B1 (**core**): given commodity, quantity, source region and timing, recommend the cheapest markets to buy from (landed cost).
  - App B2 (**stretch, experimental**): should the buyer buy now or wait.

**Priority: F1 and B1 must be complete, tested and evaluated before any work on F2 and B2.**

---

## 2. Data

Columns: `state, district, market, commodity, variety, grade, price, date`.

- Time span about 10 years. Data is **daily but irregular**: markets report only on trading days and many series have gaps.
- There is **no supply / arrival quantity** column. Do not invent one. Treat this as a documented limitation.
- **CONFIRM** with the user or the data: the exact column names and file path, what `price` means (modal, min or max), and the unit (likely ₹/quintal).
- A missing day does **not** mean price zero or unchanged. Never forward-fill across long gaps.

---

## 3. Fixed design decisions (do not change)

1. **Scope freeze:** select a small set of commodities and states with the best coverage, by script, not by hand.
   - Default: top 5 to 10 commodities and 2 to 4 states ranked by number of reporting weeks and number of markets with sufficient history. Make these configurable in `config.yaml`.
2. **Forecast target:** weekly price, not raw daily price.
   - Aggregate daily to ISO weeks per series using the **median** price.
   - Model **log price**.
   - Default horizon: **2 weeks ahead**. Support 1, 2 and 4 via config.
3. **Series key:** `market + commodity`.
   - For variety and grade, keep the single most frequent variety and grade per commodity (configurable) so series are comparable. Log what was dropped.
4. **Two-stage architecture:**
   - **Stage 1, forecasting:** predicts a market's expected price. It has **no farmer-specific inputs and no distance feature**.
   - **Stage 2, recommendation:** combines forecasts with the user's location and transport cost at query time.
5. **Time-based evaluation only.** Never use random train/test splits.
6. The app is **decision support**, not guaranteed advice. State this in the UI.

---

## 4. Project structure

```
mandi_watch/
  config.yaml
  requirements.txt
  README.md
  data/
    raw/                 # original dataset, never modified
    interim/             # cleaned daily data
    processed/           # weekly table, feature table
    geo/                 # market_coordinates.csv (cached), district_centroids.csv
  notebooks/
    01_eda.ipynb
    02_statistical_tests.ipynb
    03_modeling.ipynb
    04_recommender_evaluation.ipynb
  src/
    config.py
    data_cleaning.py
    geo.py
    features.py
    stats_tests.py
    baselines.py
    models.py
    recommender.py
    evaluation.py
  app/
    streamlit_app.py
  models/                # saved models
  reports/
    figures/
    tables/
    final_report.md
  tests/
```

Rules for the code: fixed random seeds, functions with docstrings, no hard-coded paths (use config), save all intermediate outputs, log every filtering step with row counts.

---

## 5. Phase 1: Data cleaning (`data_cleaning.py`)

1. Load raw data. Print shape, dtypes, date range, number of unique states, districts, markets, commodities.
2. Standardize text: strip whitespace, consistent casing, fix obvious spelling variants of market, district and commodity names (use a mapping file `data/interim/name_fixes.csv` that is saved and reviewable).
3. Parse `date`. Drop rows with unparseable dates or non-positive prices. Report counts.
4. Remove duplicates (same market, commodity, variety, grade, date).
5. Outlier handling: per series, flag prices far from the rolling median (for example more than 5 times the median absolute deviation). Flag first, then remove or winsorize, and report how many.
6. Detect long flat runs of identical prices (for example 14 or more consecutive reports) and flag them. Keep them but add a feature or flag.
7. Apply the scope selection (section 3, item 1) and save `data/interim/clean_daily.parquet`.
8. Aggregate to weekly: median price, plus `n_reports` (reports in the week), min, max, and std where more than one report exists. Save `data/processed/weekly.parquet`.
9. Series quality filter: keep series with at least a minimum number of weeks of history (default 104 weeks) and a maximum share of missing weeks (default 40%). Save the list of kept and dropped series and report the counts.
10. Do **not** impute long gaps. Short gaps (up to 2 weeks) may be linearly interpolated in log price only for lag features, and mark them.

---

## 6. Phase 2: Geography (`geo.py`)

1. Build `market_coordinates.csv` with columns `state, district, market, lat, lon, source`.
   - Geocode with `geopy` (Nominatim) using a descriptive query like `"{market}, {district}, {state}, India"`. Respect rate limits (about 1 request per second). Cache every result.
   - Fallback to the district centroid when the market cannot be found, and set `source = "district_centroid"`.
   - Validate that coordinates fall inside India's bounding box and roughly inside the right state. Flag suspicious ones for manual review.
   - Allow a manual override file `market_coordinates_manual.csv`.
2. Build `district_centroids.csv` the same way (one row per district in scope). This is used as the farmer's location.
3. Precompute a market-to-market distance matrix (haversine, km) for the neighbor features.

---

## 7. Phase 3: EDA and statistical testing (`stats_tests.py`, notebooks 01 and 02)

Purpose: justify the design and show that market choice matters. These tests **do not produce the recommendation**.

Important: weekly and daily prices are autocorrelated, so p-values are optimistic. Run the group-comparison tests on **monthly aggregates** (monthly mean of weekly prices per market) and say so in the report. Always report **effect sizes** and differences in ₹/quintal, not only p-values. Apply multiple-comparison correction (Benjamini-Hochberg or Bonferroni) wherever many tests are run.

| Test / tool | Question to answer | Notes |
|---|---|---|
| Summary stats, histograms, boxplots | Distribution of price per commodity and state | Use log price too |
| Shapiro-Wilk + QQ plot | Is log price roughly normal? | Use a random sample for large n; rely on QQ plots too |
| Levene test | Equal variances across markets? | Decides ANOVA type |
| One-way ANOVA (Welch) or Kruskal-Wallis | Do mean prices differ across markets for one commodity in one month or season? | Report eta-squared; follow with Tukey HSD or Dunn |
| Two-way ANOVA (market x month) | Do market and season effects interact, i.e. does the best market change by season? | `statsmodels` |
| Paired t-test or Wilcoxon signed-rank | Is market A different from market B over the same periods? | Report Cohen's d and mean difference in ₹ |
| Chi-square test of independence | Is the top-ranked market independent of season? Does grade mix depend on state? | Use Cramer's V |
| Pearson / Spearman correlation | Do neighboring markets' prices move together? | Supports neighbor features |
| ADF and KPSS | Stationarity of each series (and of log differences) | Informs forecasting |
| ACF / PACF, seasonal decomposition (STL) | Seasonality and trend | Include plots |
| Ljung-Box | Do model residuals still contain autocorrelation? | Run on final model residuals |
| Diebold-Mariano | Is the model's forecast error significantly lower than a baseline's? | Phase 5 |

Deliver: a table of all tests with hypothesis, statistic, p-value, effect size, and a plain-English conclusion, saved in `reports/tables/`.

---

## 8. Phase 4: Features and forecasting (`features.py`, `baselines.py`, `models.py`)

### 8.1 Feature table
One row per `market + commodity + week t`. Target: `log_price(t + h)`. **All features must use only information available at the end of week t.**

- Lags of log weekly price: 1, 2, 3, 4, 8, 52 weeks.
- Rolling mean of log price over 4 and 12 weeks; rolling standard deviation over 4 weeks.
- Momentum: change in log price over the last 1 and 4 weeks.
- Seasonality: week of year as sine and cosine; month.
- `n_reports` in the current week, weeks since the last report, flat-run flag.
- **Market premium:** expanding mean of (market log price minus state-commodity average log price), computed with past data only.
- **Neighbor feature:** mean lagged log price of the k nearest markets (default k = 5, within a configurable radius) for the same commodity, using the precomputed market-to-market distances.
- Static: market latitude and longitude, state and district as categorical, commodity as categorical.
- Optional (only if time and data permit, otherwise skip): MSP, festival calendar, diesel price, rainfall. The project is valid without them.

### 8.2 Splits
- Test: the last 12 months of data. Validation: the 12 months before that. Train: everything earlier.
- Optionally add rolling-origin cross-validation for the final model.
- Never let rolling or expanding features see the future.

### 8.3 Baselines (must be implemented and reported)
1. Last observed weekly price.
2. Seasonal naive (same week last year).
3. Market historical average (past data only).

### 8.4 Models
1. **Statistical:** SARIMA or ETS on a handful of well-covered series (`statsmodels`), as a comparison.
2. **Main:** global **LightGBM** regressor on the feature table, predicting log price.
3. **Uncertainty:** LightGBM quantile models (for example 10th and 90th percentiles) to give a prediction interval. Check empirical coverage on the test set and report it.

### 8.5 Metrics
MAE and MAPE (back-transformed to ₹/quintal), per commodity and overall. Compare against every baseline. Use the Diebold-Mariano test for the main model versus the best baseline. If the main model does **not** beat the baselines, report that honestly; do not tune against the test set.

---

## 9. Phase 5: Recommender (`recommender.py`)

Inputs from the user: state, district, commodity, target horizon (default 2 weeks), search radius (default 150 km), transport rate.

Steps:
1. Farmer's coordinates = district centroid.
2. Candidate markets = markets for that commodity within the radius that pass the history filter.
3. Distance = haversine x road factor (default **1.35**, configurable) to approximate road distance.
4. Transport cost per quintal = distance_km x rate. Default rate is a **placeholder of 0.25 ₹/quintal/km**, adjustable in the UI. Label it clearly as an assumption.
5. Get the forecasted price (and 10th to 90th percentile interval) for each candidate.
6. **Farmer (F1):** net price = forecast price minus transport cost. Rank descending. Show the top 5 with forecast, interval, distance, transport cost and net price.
7. **Buyer (B1):** landed cost = forecast price plus transport cost. Rank ascending. Also take quantity as input and show total cost = quantity x landed cost.
8. Warnings to show in the app:
   - Fewer than a minimum number of candidates: say so.
   - A market with weak history or wide interval: flag it as low confidence.
   - If the top markets' net prices are within the typical forecast error of each other, say "these options are not clearly different".
   - Buyer: "No arrival/supply data is available, so availability of your quantity is not checked."
9. Show a map (`folium` or `plotly`) of the farmer's district and the recommended markets.

---

## 10. Phase 6: Evaluating the recommender (`evaluation.py`, notebook 04)

Use the held-out test period only.

1. For each test week, commodity and each district in scope (as a pseudo-farmer location), produce the recommendation.
2. Compare the **recommended market's realized net price** (actual price in that week minus transport) with:
   - the **nearest market**,
   - the **market with the highest historical average**,
   - the **oracle best** market (hindsight).
3. Metrics:
   - **Regret** = oracle net price minus the net price at the recommended market (in ₹/quintal and as a percent).
   - Top-1 and top-3 hit rate.
   - Spearman correlation between predicted and actual rankings.
4. **Hypothesis test:** H0 = the recommended market's realized net price equals the nearest market's. Use a paired Wilcoxon or paired t-test on per-case differences and report the effect size. State the result honestly whichever way it goes.
5. **Sensitivity analysis:** repeat with transport rates of 0, 0.15, 0.25 and 0.5 ₹/quintal/km. Show how often the recommendation changes. If the ranking depends mostly on the assumed rate, say so.
6. Compare how much prices differ across nearby markets versus the forecast error. If the spread is smaller than the error, state that the ranking is weak for that commodity.

---

## 11. Phase 7: Streamlit app (`app/streamlit_app.py`)

- Two pages or tabs: **Farmer** and **Bulk Buyer**.
- Farmer page: dropdowns for state, district, commodity, horizon; sliders for radius and transport rate; results table, map and warnings.
- Buyer page: same plus quantity.
- Footer on every page: "Based on historical data up to <last date in dataset>. Forecasts are estimates, not guarantees. Transport cost is an assumed rate."
- Load precomputed forecasts and saved models; do not retrain inside the app. Cache heavy loads.
- Handle missing or insufficient data gracefully with a clear message, never a crash or a made-up number.

---

## 12. Stretch phase: sell/hold and buy now/later (F2 and B2)

Only after everything above works. Label these **experimental** in the app.

- Predict whether the price change over the horizon exceeds a threshold, as classification or via the forecast interval.
- Show probability and expected change, not a hard yes/no.
- Include a holding-cost input (storage, spoilage, interest per week) and compare expected gain with that cost. Perishable commodities need a different default than durable ones.
- Evaluate with precision, recall and a simple simulated profit versus "always sell now" and "always sell at the seasonal peak".
- Report that shocks (weather, policy, export bans) cannot be predicted.

---

## 13. Known limitations (must appear in the final report)

1. No arrival/supply quantity data: cannot explain gluts or shortages or judge market capacity.
2. Price type (modal/min/max) and unit assumptions.
3. Irregular reporting, gaps, copied or stale prices.
4. Transport cost is an assumed rate; road distance is approximated; district centroid is not the real farm location.
5. Commission, fees, quality discounts and local credit arrangements are not in the data.
6. Shocks (weather, policy, export bans) are not predictable from price history.
7. Persistent market premium may dominate the ranking.
8. Sparse markets give noisy forecasts; some series were excluded.
9. Autocorrelation makes classical test p-values optimistic.
10. The app uses data only up to the dataset's last date.
11. Results cover only the selected commodities and states.

---

## 14. Deliverables

1. Cleaned and weekly datasets, geo files, saved models.
2. Notebooks 01 to 04 running top to bottom.
3. `src/` modules and `tests/` (at minimum: no-leakage test for features, haversine test, recommender ranking test).
4. Working Streamlit app.
5. `reports/final_report.md` with sections: problem, data description, cleaning decisions with counts, EDA, statistical tests table, features, models and baselines, forecast results, recommender results (regret, hypothesis test, sensitivity), limitations, conclusion.
6. `README.md` with setup steps (`pip install -r requirements.txt`), how to run the pipeline in order, and how to launch the app.

---

## 15. Tech stack

Python 3.10+, `pandas`, `numpy`, `pyarrow`, `scipy`, `statsmodels`, `scikit-learn`, `lightgbm`, `geopy`, `plotly`, `folium`, `streamlit`, `matplotlib`, `seaborn`, `pytest`, `pyyaml`. Add `prophet` only if it installs cleanly and is clearly useful.

---

## 16. Suggested order of work

1. Confirm columns, price type and unit; run Phase 1 and report the coverage statistics.
2. Choose and freeze the scope (commodities, states).
3. Geography files.
4. EDA and statistical tests.
5. Features, baselines, LightGBM, quantile models.
6. Recommender and its evaluation.
7. Streamlit app for F1 and B1.
8. Report and README.
9. Stretch: F2 and B2, only if time remains.

## 17. Rules for the agent

- Never fabricate data or results. If something fails or is missing, say so and propose options.
- Never use future information in features, and never use a random split.
- Do not tune on the test set.
- Log every decision (filters, drops, assumptions) in `reports/decisions_log.md`.
- If the real data differs from the description above (column names, frequency, price type), stop and ask the user before continuing.
