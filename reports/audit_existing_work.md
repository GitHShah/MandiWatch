# Mandi Watch: Audit of Existing Work

**Date:** 2026-11-03  
**Status:** Read-Only Audit Completed — Pending User Approval  
**Auditor:** Mandi Watch Coding Agent  

---

## 1. Raw Dataset Identification

- **File Path(s):** `data/raw/` containing 12 annual CSV files:
  - `data/raw/2015.csv` (328,883,026 bytes)
  - `data/raw/2016.csv` (327,044,067 bytes)
  - `data/raw/2017.csv` (370,118,243 bytes)
  - `data/raw/2018.csv` (388,655,161 bytes)
  - `data/raw/2019.csv` (393,263,366 bytes)
  - `data/raw/2020.csv` (323,209,377 bytes)
  - `data/raw/2021.csv` (348,738,783 bytes)
  - `data/raw/2022.csv` (409,171,893 bytes)
  - `data/raw/2023.csv` (403,215,327 bytes)
  - `data/raw/2024.csv` (532,414,234 bytes)
  - `data/raw/2025.csv` (573,676,886 bytes)
  - `data/raw/2026.csv` (62,701,768 bytes)
- **Format:** CSV (comma-delimited text, UTF-8/ASCII)
- **Total Shape:** `(48,229,507 rows, 11 columns)` across the 2015–2026 span
- **Column Names & Dtypes:**
  - `State` (*string / object*): State name
  - `District` (*string / object*): District name
  - `Market` (*string / object*): Mandi market name
  - `Commodity` (*string / object*): Crop / commodity name
  - `Variety` (*string / object*): Variety name
  - `Grade` (*string / object*): Grade designation (e.g. Medium, FAQ)
  - `Arrival_Date` (*string / object*): Trading date string (`YYYY-MM-DD` or `DD/MM/YYYY`)
  - `Min_Price` (*float64*): Daily reported minimum price
  - `Max_Price` (*float64*): Daily reported maximum price
  - `Modal_Price` (*float64*): Daily reported modal (most frequent clearing) price
  - `Commodity_Code` (*int64*): Integer code for commodity
- **Date Range:** `2015-01-01` to `2026-04-21` (~11.3 years; 2026 is partial through late April)
- **Unit & Supply:** Price unit is standard Agmarknet ₹/quintal (100 kg). No arrival/volume column exists.
- **Sample Rows (First 5 records from 2015.csv):**

| State | District | Market | Commodity | Variety | Grade | Arrival_Date | Min_Price | Max_Price | Modal_Price | Commodity_Code |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Andhra Pradesh | Anantapur | Anantapur | Ground Nut Seed | Ground Nut Seed | Medium | 2015-01-01 | 5500.0 | 5700.0 | 5600.0 | 268 |
| Andhra Pradesh | Anantapur | Anantapur | Groundnut | Local | Medium | 2015-01-01 | 3800.0 | 4000.0 | 3900.0 | 10 |
| Andhra Pradesh | Anantapur | Dharmavaram | Groundnut | Local | Medium | 2015-01-01 | 3900.0 | 4200.0 | 4100.0 | 10 |
| Andhra Pradesh | Anantapur | Dharmavaram | Paddy (Dhan)(Common) | B P T | Medium | 2015-01-01 | 1100.0 | 1200.0 | 1150.0 | 2 |
| Andhra Pradesh | Anantapur | Gooti | Groundnut | Local | Medium | 2015-01-01 | 3700.0 | 3900.0 | 3800.0 | 10 |

---

## 2. Inventory of Workspace Files & Folders with Classification

### Root Files & Folders
| Path | Size / Type | Classification | Reason |
| :--- | :--- | :--- | :--- |
| `MANDI_WATCH_PROJECT_SPEC.md` | 17,351 B | **KEEP** | Authoritative project specification and source of truth. |
| `pytest.ini` | 42 B | **KEEP** | Standard pytest configuration. |
| `requirements.txt` | 106 B | **KEEP** | Package dependencies (will be augmented with `lightgbm`, `geopy`, etc.). |
| `.gitignore` | 96 B | **KEEP** | Git ignore rules for virtualenvs, caches, and large data files. |
| `README.md` | 6,292 B | **REWRITE** | Documents legacy multi-commodity cache pipeline instead of the new two-stage recommender architecture. |
| `app.py` | 14,432 B | **REWRITE** | Monolithic multi-commodity explorer; must be moved and restructured to `app/streamlit_app.py` with Farmer (F1) and Bulk Buyer (B1) tabs. |
| `tmpbuild/` | DIR | **DELETE** | Leftover temporary build directory with test parquet files. |
| `tmpbuild3/` | DIR | **DELETE** | Leftover temporary build directory with test parquet files. |
| `tmpcheck/` | DIR | **DELETE** | Leftover scratch verification directory. |
| `tmpcheck2/` | DIR | **DELETE** | Leftover scratch verification directory. |
| `tmpfailure/` | DIR | **DELETE** | Leftover failed build experiment directory. |
| `.pytest_tmp/` | DIR | **DELETE** | Temporary test run artifacts. |
| `models/` | DIR | **KEEP** | Target directory for saved LightGBM and baseline models. |

---

### `src/` Directory
| Path | Size / Type | Classification | Reason |
| :--- | :--- | :--- | :--- |
| `src/__init__.py` | 0 B | **KEEP** | Standard Python package initializer. |
| `src/config.py` | 2,174 B | **REWRITE** | Hard-codes paths and constants in Python instead of loading from `config.yaml` as required. |
| `src/audit_data.py` | 8,711 B | **KEEP** | Comprehensive chunked raw data auditor; validates columns, boundaries, and nulls. |
| `src/data_io.py` | 3,353 B | **REWRITE** | Good header alias normalization, but needs refactoring to support `config.yaml` and save to `data/interim/clean_daily.parquet`. |
| `src/clean_data.py` | 0 B | **DELETE** | Empty placeholder file; superseded by `src/data_cleaning.py`. |
| `src/commodity_cache.py` | 33,976 B | **REWRITE** | Massive 34KB engine built to cache all 300+ commodities un-scoped; spec requires script-based scope freezing to top commodities/states. |
| `src/aggregation.py` | 4,728 B | **REWRITE** | Aggregates to monthly median without variety/grade filtering; must be rewritten for weekly aggregation with `n_reports`, min, max, std. |
| `src/anomalies.py` | 1,580 B | **REWRITE** | Sound past-only rolling MAD math, but operates on monthly series rather than daily per-series outlier cleaning and winsorization. |
| `src/seasonality.py` | 2,707 B | **REWRITE** | Limited to Friedman test on monthly commodity aggregates; missing Phase 3 ANOVA, Kruskal-Wallis, Levene, and ADF/KPSS stationarity tests. |
| `src/test_seasonality.py` | 4,312 B | **REWRITE** | Utility functions for monthly complete-year indexing and Holm adjustments; should be merged into `src/stats_tests.py`. |
| `src/forecasting.py` | 4,933 B | **REWRITE** | **Major accuracy killer**: forecasts raw price on monthly data instead of weekly log price; lacks LightGBM and quantile prediction intervals. |
| `src/forecast_analysis.py` | 1,327 B | **DELETE** | Ad-hoc Onion-only script; superseded by `src/models.py` and `notebooks/03_modeling.ipynb`. |
| `src/dashboard_data.py` | 1,232 B | **REWRITE** | Built for legacy Parquet cache querying; needs restructuring for the two-stage recommender app. |
| `src/report.py` | 4,643 B | **REWRITE** | Generates legacy Onion markdown report; needs updating for `reports/final_report.md` per Section 14. |
| `src/profile_data.py` | 1,637 B | **DELETE** | One-off profiling script; superseded by Phase 1 data cleaning logs. |
| `src/profile_onion.py` | 4,960 B | **DELETE** | Hard-coded Onion-only profiling script; superseded by automated scope selection. |
| `src/eda_prices.py` | 3,510 B | **DELETE** | Ad-hoc Onion plot generator; superseded by `notebooks/01_eda.ipynb`. |
| `src/eda_seasonality.py` | 3,012 B | **DELETE** | Ad-hoc Onion seasonality script; superseded by `notebooks/02_statistical_tests.ipynb`. |
| `src/evaluate_2026.py` | 2,882 B | **DELETE** | Ad-hoc Onion 2026 holdout evaluation; superseded by Phase 6 recommender evaluation. |
| `src/alerts.py` | 0 B | **DELETE** | Empty placeholder file (0 bytes). |
| `src/train_model.py` | 0 B | **DELETE** | Empty placeholder file (0 bytes); superseded by `src/models.py`. |

---

### `tests/` Directory
| Path | Size / Type | Classification | Reason |
| :--- | :--- | :--- | :--- |
| `tests/test_data_io.py` | 832 B | **KEEP** | Valid unit tests for header normalization and date parsing. |
| `tests/test_aggregation.py` | 548 B | **REWRITE** | Tests monthly aggregation; needs rewrite for weekly aggregation rules and stats. |
| `tests/test_anomalies.py` | 579 B | **REWRITE** | Tests monthly rolling MAD; needs rewrite for per-series daily outlier detection. |
| `tests/test_forecasting.py` | 1,439 B | **REWRITE** | Tests monthly naive baselines; needs tests for weekly log price, LightGBM, and feature no-leakage verification. |
| `tests/test_multi_commodity.py` | 13,196 B | **DELETE** | Tests legacy 300-commodity Parquet cache engine that is superseded by Phase 1 scope freezing. |
| `tests/test_statistics.py` | 498 B | **REWRITE** | Tests Kendall's W / Friedman; needs expansion for Phase 3 statistical tests (ANOVA, Wilcoxon, stationarity). |

---

### `docs/`, `plots/`, and `reports/` Directories
| Path | Size / Type | Classification | Reason |
| :--- | :--- | :--- | :--- |
| `docs/data_dictionary.md` | 935 B | **KEEP** | Accurate reference for Agmarknet schema and field definitions. |
| `docs/methodology.md` | 3,311 B | **KEEP** | Valuable documentation on Agmarknet data characteristics and caveats. |
| `docs/limitations.md` | 650 B | **KEEP** | Directly aligns with Section 13 limitations required in final report. |
| `docs/implementation_log.md` | 661 B | **DELETE** | Implementation notes from the superseded cache architecture. |
| `docs/project_overview.md` | 504 B | **DELETE** | Superseded overview of the un-scoped multi-commodity browser. |
| `plots/` (4 PNG files) | ~375 KB | **DELETE** | Ad-hoc Onion plots; will be regenerated cleanly into `reports/figures/`. |
| `reports/decisions_log.md` | 194 B | **KEEP** | Active log tracking all cleaning filters, drops, and modeling decisions with row counts. |
| `reports/mandiwatch_report.md` | 2,826 B | **DELETE** | Old Onion-only markdown report; will be superseded by `reports/final_report.md`. |

---

### `data/` Subdirectories (Excluding Raw Data)
| Path | Type | Classification | Reason |
| :--- | :--- | :--- | :--- |
| `data/raw/` | DIR | **KEEP** | **Source of truth.** 12 annual raw CSVs (2015–2026), never modified. |
| `data/cache/` | DIR | **DELETE** | Legacy un-scoped multi-commodity Parquet cache and build logs; superseded by Phase 1 `data/interim/clean_daily.parquet` and `data/processed/weekly.parquet`. |
| `data/processed/` (existing CSVs) | DIR | **DELETE** | Old ad-hoc Onion summaries; will be cleanly replaced with `weekly.parquet` and feature tables. |

---

## 3. Evaluation Against the 8 Common Accuracy Killers

1. **Random train/test splits or shuffling:**
   - *Status in legacy code:* Legacy baseline code used chronological split, but only tested a single 12-month holdout without validation splits.
   - *Fix needed:* Enforce strict 3-way time split: **Test** = last 12 months (2025-05 to 2026-04), **Validation** = preceding 12 months (2024-05 to 2025-04), **Train** = prior history (2015 to 2024-04).
2. **Features that leak the future:**
   - *Status in legacy code:* Anomaly detection properly used `shift(1)`, but no ML feature table was constructed.
   - *Fix needed:* Ensure all lag, rolling mean, rolling std, expanding market premium, and neighbor features use strictly $\le t$ information. Add explicit unit test asserting no future leakage.
3. **Forward-filling across long gaps:**
   - *Status in legacy code:* Old code threw an error on any missing calendar month rather than handling gaps.
   - *Fix needed:* Implement spec rule: drop sparse series (<104 weeks history or >40% missing); never forward-fill long gaps; only allow linear interpolation of short gaps ($\le 2$ weeks) in log price for lag computation.
4. **Modelling raw daily price instead of weekly log price:**
   - *Status in legacy code:* **Critical flaw.** Legacy code modelled raw monthly prices.
   - *Fix needed:* Aggregate daily transactions to ISO weeks using median price, compute `log_price = ln(price)`, and model 2-week-ahead log price.
5. **Mixing different varieties/grades in one series:**
   - *Status in legacy code:* **Critical flaw.** Legacy code aggregated across all varieties and grades within a market, introducing severe price discontinuities.
   - *Fix needed:* Lock series key to `market + commodity`, filtering to the single most frequent variety and grade per commodity.
6. **Unhandled outliers or unit errors:**
   - *Status in legacy code:* Only filtered `modal_price > 0`. Outlier detection was exploratory in Streamlit, not applied in data cleaning.
   - *Fix needed:* Phase 1 per-series rolling MAD outlier filtering (>5x MAD) and flat-run detection ($\ge 14$ consecutive days) with row counts logged.
7. **Evaluating on training data:**
   - *Status in legacy code:* Baselines used holdouts, but no ML models existed.
   - *Fix needed:* Evaluate LightGBM, quantile models, and recommender strictly on Validation and Test sets.
8. **Hard-coded paths:**
   - *Status in legacy code:* `src/config.py` hard-coded paths and parameters in Python module constants.
   - *Fix needed:* Implement `config.yaml` as the sole configuration root for all paths, random seeds, horizons, and thresholds.

---

## 4. Required Next Steps (Upon User Approval)

1. Remove deleted scratch/experimental directories (`tmpbuild*`, `tmpcheck*`, `tmpfailure`, `data/cache`, legacy onion scripts).
2. Create `config.yaml` with random seed (42), paths, top commodity/state selection parameters, and horizons.
3. Build `src/data_cleaning.py` to produce `data/interim/clean_daily.parquet` and `data/processed/weekly.parquet`.
