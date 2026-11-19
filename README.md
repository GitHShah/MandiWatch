# Mandi Watch: Agricultural Price Forecasting and Spatial Arbitrage Recommender System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests: Pytest](https://img.shields.io/badge/tests-pytest-green.svg)](https://docs.pytest.org/)

**Mandi Watch** is a data science decision-support system designed to empower Indian smallholder farmers and bulk buyers by mitigating spatial price dispersion across regional agricultural markets (APMC mandis). Built on 11.3 years of daily Agmarknet records (2015–2026; 48.2M raw rows), the system pairs multi-horizon machine learning price forecasts with spatial transport economics.

---

## 1. System Architecture

The project employs a decoupled two-stage architecture:
1. **Stage 1 (Forecasting Engine):** Predicts market-level expected weekly prices and 10–90% conformal uncertainty intervals 2 weeks ahead using an autoregressive gradient-boosted decision tree ensemble (LightGBM). Forecasting incorporates past price lags, trailing momentum, seasonality, and spatial neighborhood co-movement.
2. **Stage 2 (Spatial Recommender Engine):** Integrates price forecasts at query time with a spatial road distance model (haversine $\times$ 1.35 tortuosity factor) and assumed freight transport rates:
   - **Farmer Interface ($F_1$):** Maximizes net take-home price ($\text{Forecast} - \text{Transport Cost}$).
   - **Bulk Buyer Interface ($B_1$):** Minimizes landed procurement cost ($\text{Forecast} + \text{Transport Cost}$) across user-specified lot sizes.
   - **Decision Support Safeguards:** Automatically flags low historical reporting density, high interval uncertainty, and overlapping candidate error bounds.

---

## 2. Directory Structure

```
MandiWatch/
├── config.yaml                    # Single source of truth for paths, seeds & hyperparameters
├── requirements.txt               # Pinned Python package dependencies
├── run_pipeline.py                # End-to-end master pipeline script
├── app/
│   └── streamlit_app.py           # Production Streamlit UI (Farmer F1 & Buyer B1)
├── data/
│   ├── raw/                       # Immutable source CSV archives (2015.csv - 2026.csv)
│   ├── interim/                   # Cleaned daily parquet files & name fix mappings
│   ├── processed/                 # weekly.parquet, features_h2.parquet, forecasts_h2.parquet
│   └── geo/                       # District centroids, market coordinates, distance matrix
├── models/
│   ├── lightgbm_model_h2.joblib   # Trained LightGBM point forecasting model
│   └── lightgbm_quantiles_h2.joblib # Trained pinball quantile regressors (p10, p50, p90)
├── notebooks/
│   ├── 01_eda.ipynb               # Exploratory Data Analysis & market coverage
│   ├── 02_statistical_tests.ipynb # Formal hypothesis testing suite (Normality, ANOVA, ADF)
│   ├── 03_modeling.ipynb          # Baselines, LightGBM training, calibration & ablations
│   └── 04_recommender_evaluation.ipynb # Held-out test period simulation & regret analysis
├── reports/
│   ├── final_report.md            # Comprehensive formal project report
│   ├── decisions_log.md           # Chronological audit log of all filtering & modeling choices
│   ├── figures/                   # Publication-quality charts (PNG, 300 DPI)
│   └── tables/                    # CSV & MD tables backing all reported statistics
├── src/
│   ├── config.py                  # Pydantic-based configuration loader
│   ├── data_cleaning.py           # Text cleaning, deduplication, MAD outlier rejection
│   ├── geo.py                     # Geocoding, road distance, and spatial nearest neighbors
│   ├── features.py                # Autoregressive lags, rolling statistics, leak-free splits
│   ├── baselines.py               # Persistence, seasonal naive, and historical mean baselines
│   ├── models.py                  # LightGBM training, conformal intervals & Diebold-Mariano
│   ├── recommender.py             # F1 / B1 optimization logic, warnings & Folium map
│   ├── evaluation.py              # Forecast & recommender evaluation suite
│   └── stats_tests.py             # Statistical hypothesis testing framework
└── tests/
    ├── test_baselines.py          # Synthetic series verification for baselines
    ├── test_cleaning.py           # Outlier, deduplication & flat-run unit tests
    ├── test_no_leakage.py         # Temporal split & feature leak-prevention unit tests
    ├── test_recommender.py        # Haversine, road distance & ranking logic unit tests
    ├── test_scenarios.py          # Integration test verifying 6 real-world query scenarios
    ├── test_stats.py              # Statistical testing mathematical correctness tests
    └── test_weekly.py             # Aggregation and gap interpolation unit tests
```

---

## 3. Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### Virtual Environment Setup
```bash
# Clone repository
git clone https://github.com/your-repo/MandiWatch.git
cd MandiWatch

# Create and activate virtual environment
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate
# On Windows (cmd / PowerShell):
.venv\Scripts\activate

# Install required dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Pipeline Execution & Reproduction

You can reproduce the entire pipeline end-to-end using a single command or run individual stages sequentially:

### Option A: Master Pipeline Script (`run_pipeline.py`)

```bash
# Run the complete pipeline end-to-end
python run_pipeline.py

# Run pipeline skipping daily cleaning (uses existing weekly.parquet):
python run_pipeline.py --skip-cleaning

# Run individual pipeline stages:
python run_pipeline.py --step geo
python run_pipeline.py --step features
python run_pipeline.py --step baselines
python run_pipeline.py --step models
python run_pipeline.py --step recommender
```

### Option B: Step-by-Step Module Execution

If you prefer to run each module directly:

```bash
# 1. Clean daily data, select scope, and build weekly grid
python src/data_cleaning.py

# 2. Geocode markets and calculate distance matrix
python src/geo.py

# 3. Engineer leak-free multi-horizon feature tables (h=2)
python src/features.py

# 4. Evaluate past-only benchmark baselines
python src/baselines.py

# 5. Train LightGBM models and calibrate conformal prediction intervals
python src/models.py

# 6. Run held-out test recommender evaluation simulation
python src/evaluation.py
```

---

## 5. Launching the Interactive Web Application

The interactive web dashboard is built using Streamlit and Folium:

```bash
streamlit run app/streamlit_app.py
```

Once started, open your browser at `http://localhost:8501`.

### App Features:
- **Tab 1: Farmer ($F_1$):** Select State, District, Crop, and Horizon. Adjust maximum search radius (default: 150 km) and assumed freight rate (default: ₹0.25/q/km). Inspect the ranked top-5 candidate mandis, net prices, conformal confidence intervals, and interactive Folium road route map.
- **Tab 2: Bulk Buyer ($B_1$):** Input purchase quantity (quintals) to discover lowest landed acquisition cost markets with total procurement outlays.
- **Dynamic Cascading Menus:** Dropdowns strictly update to show only valid district-commodity combinations.
- **Transparent Disclaimers & Risk Alerts:** Clear warnings highlight when options have overlapping uncertainty intervals, weak historical reporting, or small candidate pools.

---

## 6. Running the QA & Test Suite

The test suite covers data cleaning, temporal leak-prevention, baseline convergence, statistical tests, spatial geometry, and recommender ranking logic.

```bash
# Run all unit and integration tests
pytest -v

# Run specific test modules
pytest tests/test_no_leakage.py
pytest tests/test_recommender.py
pytest tests/test_baselines.py
```

To re-execute all 4 Jupyter notebooks top-to-bottom:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_statistical_tests.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_modeling.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_recommender_evaluation.ipynb
```

---

## 7. Key Findings Summary

| Metric Dimension | Key Empirical Result | Reference Table |
| :--- | :--- | :--- |
| **Forecasting Accuracy** | LightGBM achieves **MAE of ₹158.58/q (7.91% MAPE)** on held-out test data vs Last Observed baseline of ₹190.08/q (16.6% error reduction, DM stat = -21.86, $p < 10^{-15}$). | `reports/tables/forecast_model_comparison.csv` |
| **Foodgrain Boundary** | For MSP-stabilized crops (**Wheat** and **Paddy**), the persistence baseline beats LightGBM (Wheat MAE ₹42.49 vs ₹68.89). Machine learning dominates on perishable horticulture. | `reports/tables/diebold_mariano_test.csv` |
| **Uncertainty Calibration** | Conformal calibration yields an empirical **85.93% coverage** on held-out test data for the nominal 80% interval. | `reports/tables/interval_coverage.csv` |
| **Recommender Regret** | Across 26,352 test scenarios, model mean regret is **₹108.16/q (4.54%)** with a **median regret of ₹10.00/q**, compared to ₹338.65/q (13.50%) for the nearest mandi. | `reports/tables/recommender_regret_summary.csv` |
| **Net Economic Value** | Recommending optimal markets delivers an average net gain of **+₹230.49/quintal** over the nearest market ($p < 10^{-15}$, Cohen's $d = 0.459$). | `reports/tables/recommender_hypothesis_test.csv` |
| **Freight Robustness** | **>91.7% of recommendations are invariant** to transport rates between 0.00 and 0.50 ₹/q/km because regional price dispersion dwarfs freight costs. | `reports/tables/recommender_sensitivity_analysis.csv` |
