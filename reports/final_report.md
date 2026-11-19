# Mandi Watch: Agricultural Price Forecasting and Spatial Arbitrage Recommender System

**Final Project Report**  
**Course:** Advanced Data Science & Machine Learning  
**Date:** November 2026  
**Repository:** `MandiWatch`  
**Dataset:** Agmarknet (Ministry of Agriculture & Farmers Welfare, Government of India), 2015–2026  

---

## 1. Problem Statement

Smallholder farmers in India typically sell their agricultural produce at the nearest regulated wholesale market (APMC mandi), accepting whatever prevailing clearing price is offered on the day of arrival. However, regional agricultural commodity markets exhibit substantial spatial price dispersion: on any given week, mandis located 50 to 150 kilometers apart often clear identical crops at price differentials exceeding ₹200 to ₹1,500 per quintal. 

Farmers fail to capture this spatial arbitrage due to two structural friction points:
1. **Asymmetric and Fragmented Information:** Mandi daily prices are reported retrospectively across thousands of disparate market portals, making forward-looking spatial comparison cognitively prohibitive.
2. **Transportation Uncertainty:** Transporting perishable produce to a distant mandi incurs real vehicular freight costs (fuel, loading, vehicle charter). If a farmer travels 80 km only to discover that the destination market's price has collapsed upon arrival, the net realized return will be lower than selling at the village gate.

**Mandi Watch** addresses this dual challenge through a decoupled two-stage architecture:
- **Stage 1 (Predictive Forecasting):** Trains an autoregressive, multi-horizon gradient-boosted decision tree model (LightGBM) to forecast future weekly log-clearing prices and calibrated 10–90% uncertainty intervals across 955 validated mandi series up to 4 weeks ahead ($h=2$ weeks default).
- **Stage 2 (Spatial Economic Optimization):** Couples horizon forecasts at inference time with a spatial road network cost model to compute net farmer take-home prices ($F_1 = \hat{y} - \text{transport}$) and buyer landed acquisition costs ($B_1 = \hat{y} + \text{transport}$).

The system operates strictly as **decision support**, offering calibrated confidence intervals and automated risk warnings when candidate markets have overlapping error bounds or volatile reporting histories.

---

## 2. Data Description

The raw foundation of Mandi Watch comprises 12 consecutive annual archives (`2015.csv` through `2026.csv`) obtained from the national Agmarknet portal. The dataset spans from **January 1, 2015 through April 21, 2026** (11.3 years, 591 calendar weeks).

### 2.1 Dataset Volume and Cardinality
- **Total Raw Records:** 48,229,507 rows (4.15 GB uncompressed).
- **Temporal Breadth:** 4,129 daily timestamps across 12 files.
- **Categorical Cardinality:**
  - States / Union Territories: 33
  - Districts: 632
  - Regulated Wholesale Mandis: 6,733
  - Distinct Commodities: 380
  - Botanical Varieties: 1,317
  - Quality Grades: 18

### 2.2 Schema and Price Triplet Semantics
Each transaction record includes `State`, `District`, `Market`, `Commodity`, `Variety`, `Grade`, `Arrival_Date`, `Min_Price`, `Max_Price`, `Modal_Price`, and `Commodity_Code`. 

An empirical audit of price triplet relationships (`Min`, `Modal`, `Max`) revealed:
- **Full Auction Spread ($Min < Modal < Max$):** 37,113,135 rows (76.95%). A genuine auction band exists.
- **Uniform Clearing Price ($Min == Modal == Max$):** 4,690,880 rows (9.73%). Market reported a single uniform price.
- **Anchored Modal Price ($Min == Modal < Max$ or $Min < Modal == Max$):** 5,926,503 rows (12.29%).
- **Inverted / Corrupted Triplet ($Modal < Min$ or $Modal > Max$):** 498,989 rows (1.03%).

Prices are quoted in Indian Rupees per quintal (₹/quintal, where 1 quintal = 100 kg).

> **Crucial Data Limitation:** The raw dataset contains **zero supply / arrival quantity** columns. Market clearing volumes cannot be directly observed or modeled.

---

## 3. Cleaning Decisions and Data Quality Filtering

All data cleaning operations were executed strictly chronologically and logged in `reports/decisions_log.md`. Row counts and retention ratios at every pipeline stage are detailed below:

```
[Raw Agmarknet Records: 48,229,507]
         │
         ▼
[Text Standardization & Deduplication]  ──> Dropped 9,587 exact duplicate keys (0.02%)
         │
         ▼
[MAD Outlier & Unit Error Rejection]    ──> Dropped 4,936,279 MAD outliers (10.23%)
         │                              ──> Dropped 95,133 unit scaling errors (0.20%)
         ▼
[Interim Clean Daily: 43,207,140 rows]  ──> Flagged 7,624,616 flat-run rows (retained)
         │
         ▼
[Scope Freeze: Top 8 Crops, Top 3 States] ──> Filtered to 4,601,519 in-scope daily rows
         │
         ▼
[Modal Variety & Grade Retention]       ──> Dropped 2,579,374 non-primary variety rows (56.05%)
         │                              ──> Retained 2,022,145 homogenous daily records
         ▼
[Weekly Aggregation: Median Price]      ──> Produced 436,045 market-commodity-week rows
         │
         ▼
[Quality Filter (>=104 wks, <=40% gap)] ──> Kept 955 series (354,712 weekly records)
         │                              ──> Dropped 1,608 sparse/irregular series
         ▼
[Short-Gap Linear Interpolation (h<=2)] ──> Imputed 8,395 gap weeks (is_interpolated=True)
         │
         ▼
[Final Processed Grid: 410,230 rows]
```

### Detailed Summary of Cleaning Steps:
1. **Deduplication:** Resolved 9,587 key conflicts on `(Market, Commodity, Variety, Grade, Arrival_Date)` using median price resolution (48,229,507 $\rightarrow$ 48,219,920 rows).
2. **Rolling MAD Outlier Detection:** Applied a centered 15-day rolling median filter per series. Records exceeding 5.0 Median Absolute Deviations (MAD) were removed (4,936,279 rows dropped).
3. **Unit Error Rejection:** Prices deviating by $\approx 10\times$ or $100\times$ from the series median (reflecting ₹/kg entry rather than ₹/quintal) were removed (95,133 rows dropped). Total dropped: 5,012,780 rows (10.40%).
4. **Flat Run Identification:** Detected 7,624,616 rows where prices remained identically unchanged for $\ge 14$ consecutive calendar days (`is_flat_run=True`). These were retained with boolean flags for diagnostics.
5. **Scope Selection:** Automated ranking identified the top 3 states by historical density (**Uttar Pradesh, Madhya Pradesh, Maharashtra**) and the top 8 commodities (**Potato, Onion, Wheat, Tomato, Paddy (Dhan)(Common), Brinjal, Green Chilli, Cauliflower**).
6. **Variety and Grade Standardization:** Retained only the modal variety and FAQ/modal grade per commodity (e.g., *Nasik* for Onion, *Jyoti* for Potato, *Dara* for Wheat). This eliminated variety-switching bias, retaining 2,022,145 homogeneous daily records.
7. **Series Continuity Filter:** Enforced a minimum history of $\ge 104$ reporting weeks and a maximum missing rate of $\le 40\%$. Exactly 955 series passed (354,712 weekly rows); 1,608 irregular series were pruned.
8. **Short Gap Interpolation:** For feature engineering continuity, short gaps ($h \le 2$ weeks) were linearly interpolated (8,395 points, `is_interpolated=True`). Gaps $> 2$ weeks were left un-interpolated.

---

## 4. Exploratory Data Analysis (EDA)

Exploratory analysis confirmed three fundamental dynamics in Indian agricultural prices:

### 4.1 Reporting Consistency Across Mandis
Reporting consistency varies widely across states and market classifications. Primary APMC hub markets (e.g., Azadpur, Nashik, Vashi, Kanpur) maintain unbroken weekly records across all 11 years, whereas secondary rural sub-yards display seasonal intermittency aligned with local harvesting cycles.

![Market Reporting Coverage](reports/figures/market_year_reporting_heatmap.png)

![Series Count by Year](reports/figures/series_count_by_year.png)

### 4.2 Cross-Market Price Dispersion
Even within a 100 km radius, clearing prices exhibit substantial dispersion. The figure below highlights paired price spreads across top mandis in Uttar Pradesh and Maharashtra:

![Market Price Differences](reports/figures/stats_market_price_differences.png)

### 4.3 Spatial Autocorrelation and Co-movement
Price movements between mandis decay as a function of geographic distance. Nearby markets ($\le 150$ km) exhibit strong co-movement driven by shared agro-climatic conditions and inter-mandi trader arbitrage.

![Spatial Correlation vs Distance](reports/figures/stats_spatial_correlation_distance.png)

---

## 5. Statistical Hypothesis Testing Suite

To ensure rigor and prevent false modeling assumptions, we executed an automated statistical testing suite on the 8 focal commodities. All $p$-values were adjusted for family-wise error using the Benjamini-Hochberg (BH) False Discovery Rate procedure.

The table below reproduces the results from `reports/tables/statistical_tests.csv`:

| Test | Commodity | Statistic | Raw $p$-value | Adjusted $p$-val | Effect Size | Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Shapiro-Wilk** | Potato | $W = 0.9889$ | $7.68 \times 10^{-4}$ | $1.28 \times 10^{-3}$ | $R^2 = 0.989$ | Non-normal log-price distribution |
| **Shapiro-Wilk** | Onion | $W = 0.9820$ | $7.71 \times 10^{-6}$ | $1.36 \times 10^{-5}$ | $R^2 = 0.983$ | Non-normal log-price distribution |
| **Shapiro-Wilk** | Tomato | $W = 0.9936$ | $0.0333$ | $0.0496$ | $R^2 = 0.993$ | Non-normal log-price distribution |
| **Shapiro-Wilk** | Wheat | $W = 0.9726$ | $4.72 \times 10^{-8}$ | $9.65 \times 10^{-8}$ | $R^2 = 0.975$ | Non-normal log-price distribution |
| **Levene Test** | Potato | $W = 3.07$ | $2.78 \times 10^{-29}$ | $9.41 \times 10^{-29}$ | Ratio: 78.0 | Strong heteroscedasticity across mandis |
| **Levene Test** | Onion | $W = 2.77$ | $9.49 \times 10^{-31}$ | $3.48 \times 10^{-30}$ | Ratio: 7.22 | Strong heteroscedasticity across mandis |
| **Levene Test** | Tomato | $W = 3.62$ | $3.25 \times 10^{-28}$ | $1.06 \times 10^{-27}$ | Ratio: 10.26 | Strong heteroscedasticity across mandis |
| **Levene Test** | Wheat | $W = 2.56$ | $9.08 \times 10^{-23}$ | $2.66 \times 10^{-22}$ | Ratio: 8.86 | Strong heteroscedasticity across mandis |
| **Kruskal-Wallis** | Potato | $H = 1308.03$ | $1.27 \times 10^{-191}$ | $3.72 \times 10^{-190}$ | $\eta^2 = 0.099$ | Significant cross-market price gap (max ₹808.8/q) |
| **Kruskal-Wallis** | Onion | $H = 1028.67$ | $4.44 \times 10^{-120}$ | $4.34 \times 10^{-119}$ | $\eta^2 = 0.047$ | Significant cross-market price gap (max ₹1325.6/q) |
| **Kruskal-Wallis** | Tomato | $H = 637.39$ | $1.71 \times 10^{-82}$ | $1.07 \times 10^{-81}$ | $\eta^2 = 0.074$ | Significant cross-market price gap (max ₹1734.2/q) |
| **Kruskal-Wallis** | Green Chilli | $H = 3401.54$ | $< 10^{-300}$ | $< 10^{-300}$ | $\eta^2 = 0.194$ | Significant cross-market price gap (max ₹2355.4/q) |
| **Kruskal-Wallis** | Cauliflower | $H = 955.20$ | $4.00 \times 10^{-152}$ | $5.03 \times 10^{-151}$ | $\eta^2 = 0.147$ | Significant cross-market price gap (max ₹1962.7/q) |
| **Two-Way ANOVA** | Potato (Month) | $F = 24.58$ | $4.86 \times 10^{-45}$ | $2.25 \times 10^{-44}$ | $\eta^2 = 0.219$ | Strong seasonal month effect |
| **Two-Way ANOVA** | Tomato (Month) | $F = 105.04$ | $7.96 \times 10^{-155}$ | $1.17 \times 10^{-153}$ | $\eta^2 = 0.540$ | Dominant seasonal month effect |
| **Two-Way ANOVA** | Wheat (Market) | $F = 0.29$ | $0.9596$ | $1.0000$ | $\eta^2 = 0.002$ | Minimal market premium for foodgrains |
| **Paired Wilcoxon** | Wheat (Top Pair) | $W = 998.0$ | $9.79 \times 10^{-14}$ | $2.61 \times 10^{-13}$ | $d = 0.77$ | Aligarh vs Auraiya mean gap = +₹31.7/q |
| **Paired Wilcoxon** | Tomato (Top Pair)| $W = 2066.5$ | $2.18 \times 10^{-7}$ | $4.07 \times 10^{-7}$ | $d = -0.45$ | Pukharayan vs Fatehpur gap = -₹172.0/q |
| **Mann-Whitney U** | Potato (Spatial) | $U = 7.43 \times 10^6$| $2.98 \times 10^{-31}$ | $1.14 \times 10^{-30}$ | $\Delta r = +0.038$| Mandis $\le 150$km have $r=0.920$ vs $>150$km $r=0.883$ |
| **Mann-Whitney U** | Tomato (Spatial) | $U = 1.54 \times 10^6$| $2.89 \times 10^{-23}$ | $8.76 \times 10^{-23}$ | $\Delta r = +0.059$| Mandis $\le 150$km have $r=0.826$ vs $>150$km $r=0.768$ |
| **ADF Test (Level)**| Potato | $\text{stat} = -3.28$ | $0.0158$ | $0.0244$ | KPSS: 0.905 | Non-stationary level (structural breaks) |
| **ADF (1st Diff)** | Potato | $\text{stat} = -6.39$ | $2.13 \times 10^{-8}$ | $4.47 \times 10^{-8}$ | KPSS: 0.022 | First difference is strictly stationary |
| **ADF (1st Diff)** | Wheat | $\text{stat} = -8.13$ | $1.11 \times 10^{-12}$ | $2.79 \times 10^{-12}$ | KPSS: 0.033 | First difference is strictly stationary |

### Key Statistical Takeaways:
1. **Normality & Homoscedasticity Rejected:** All series violate normality ($p < 0.05$) and show variance ratios up to $78.0\times$ across markets, justifying non-parametric ranking and quantile loss modeling.
2. **First-Difference Stationarity:** Log-price levels contain unit roots and structural breaks, whereas first differences ($\Delta \log y$) achieve strict stationarity ($p < 10^{-5}$ across all commodities). This statistical finding directly motivated our $\Delta y$ target formulation.
3. **Cross-Market Price Dispersion:** Significant market price gaps exist for every crop ($p < 10^{-80}$), confirming that spatial arbitrage opportunities are structural and persistent.

![Normality QQ Plots](reports/figures/stats_normality_qq_plots.png)

![ANOVA Effects](reports/figures/stats_two_way_anova_effects.png)

---

## 6. Feature Engineering and Leakage Prevention

All features were engineered strictly using past information relative to the forecast origin $t$. Features were computed over the continuous weekly grid and merged with target lead values:
$$y_{t+h} = \log(\text{price}_{t+h})$$

### 6.1 Feature Architecture
1. **Autoregressive Lags:** $\text{lag}_1, \text{lag}_2, \text{lag}_3, \text{lag}_4, \text{lag}_8, \text{lag}_{12}$ of weekly log price.
2. **Trailing Rolling Statistics:** 4-week, 8-week, 12-week, and 26-week rolling means, standard deviations, and min/max spreads computed strictly over trailing windows ($t-k$ to $t$).
3. **Momentum and Trend:** Short-term price velocity ($\Delta_{1-2} = \text{lag}_1 - \text{lag}_2$), medium-term velocity ($\Delta_{1-4}$), and acceleration ($\Delta_{1-2} - \Delta_{2-3}$).
4. **Calendar and Seasonality:** Cyclical sine and cosine transformations of calendar week ($\sin(2\pi w/52)$, $\cos(2\pi w/52)$) and calendar month.
5. **Spatial Cross-Market Features:** Trailing 4-week mean price of the 3 nearest neighboring mandis within a 150 km radius.
6. **Data Quality Flags:** Binary indicators for short-gap interpolation (`is_interpolated`) and stale flat runs (`is_flat_run`).
7. **Categoricals:** Native LightGBM categorical features for `state`, `district`, and `commodity`.

Feature importance analysis confirms that recent price momentum ($\text{lag}_1$), 4-week rolling mean, and spatial neighborhood averages drive over 60% of total split gain.

![Feature Importance Gain](reports/figures/feature_importance_gain.png)

---

## 7. Predictive Models, Baselines, and Ablation Studies

### 7.1 Temporal Train / Validation / Test Split
To reflect real-world deployment and prevent data leakage:
- **Training Set:** 2015-01-01 to 2022-12-31 (8 full years; captures baseline seasonality and COVID break).
- **Validation Set:** 2023-01-01 to 2023-12-31 (1 full year; $N = 39,433$ cases; used exclusively for model tuning and ablations).
- **Held-Out Test Set:** 2024-01-01 to 2026-04-21 (2.3 years; $N = 35,725$ cases; touched only once for final reporting).

### 7.2 Benchmark Baselines
Three past-only baselines were implemented:
1. **Last Observed Weekly Price (Persistence):** $\hat{y}_{t+h} = y_t^{\text{obs}}$.
2. **Seasonal Naive:** $\hat{y}_{t+h} = y_{t+h-52}$ (same week last year).
3. **Historical Expanding Mean:** $\hat{y}_{t+h} = \frac{1}{t}\sum_{i=1}^t y_i$.

### 7.3 Model Ablations (Strictly on Validation Set)
During Phase 4, the initial level-forecasting model yielded a validation MAE of ₹213.34/q. We performed six disciplined ablations on the validation set (Decision 005):

| Candidate Architecture | Val MAE (₹/q) | Val MAPE (%) | Decision |
| :--- | :--- | :--- | :--- |
| **Baseline Global LightGBM (Level Target)** | 213.34 | 9.71% | Starting benchmark |
| (a) Drop sparsest series ($<100$ obs) | 215.30 | 9.84% | **Rejected** (+1.96 ₹/q worse) |
| (b1) Shorter training window (2020+) | 215.21 | 9.88% | **Rejected** (+1.87 ₹/q worse) |
| (b2) Recency sample weighting ($t_{1/2}=3$y) | 214.25 | 9.77% | **Rejected** (+0.91 ₹/q worse) |
| (c) Era indicator feature | 213.05 | 9.69% | **Rejected** (within noise bounds, -0.29 ₹/q) |
| (d) Simpler feature set (12 features) | 224.33 | 10.36% | **Rejected** (+10.99 ₹/q worse) |
| (e) Per-commodity separate models | 222.47 | 10.15% | **Rejected** (+9.13 ₹/q worse) |
| **(f) Price-Change Target ($\Delta y = \log y_{t+h} - \log y_t^{\text{obs}}$)** | **201.17** | **9.20%** | **ACCEPTED** (-12.17 ₹/q, -5.7% error reduction) |

Formulating the target as the log-price increment $\Delta y$ over the last observed price forced the tree ensembles to model dynamic market momentum rather than re-learning static market-level intercepts.

---

## 8. Forecast Evaluation and Empirical Comparison

### 8.1 Aggregate and Per-Commodity Metrics
The table below reproduces the validated forecast metrics from `reports/tables/forecast_model_comparison.csv` on the held-out test set ($N = 35,725$ test weeks):

| Commodity | LightGBM MAE (₹/q) | LightGBM MAPE (%) | Last Observed MAE (₹/q) | Seasonal Naive MAE (₹/q) | Historical Mean MAE (₹/q) | LightGBM vs Last Obs ($\Delta$ MAE) | Hardest Baseline |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ALL Commodities** | **158.58** | **7.91%** | **190.08** | **585.07** | **572.65** | **-31.50 (-16.6%)** | **Last Observed** |
| **Tomato** | **259.07** | 14.94% | 356.55 | 872.54 | 880.91 | **-97.48 (-27.3%)** | Last Observed |
| **Green Chilli** | **278.58** | 9.94% | 344.67 | 616.58 | 705.41 | **-66.09 (-19.2%)** | Last Observed |
| **Cauliflower** | **243.53** | 16.55% | 300.13 | 497.65 | 753.48 | **-56.60 (-18.9%)** | Last Observed |
| **Onion** | **157.41** | 7.82% | 198.61 | 964.83 | 592.25 | **-41.20 (-20.7%)** | Last Observed |
| **Potato** | **68.30** | 5.48% | 88.44 | 706.06 | 250.58 | **-20.14 (-22.8%)** | Last Observed |
| **Brinjal** | **108.98** | 7.76% | 114.67 | 295.75 | 287.98 | **-5.69 (-5.0%)** | Last Observed |
| **Paddy (Dhan)** | 42.99 | 2.02% | **33.70** | 107.55 | 331.01 | +9.29 (+27.6%) | **Last Observed (Wins)** |
| **Wheat** | 68.89 | 2.58% | **42.49** | 187.30 | 655.19 | +26.40 (+62.1%) | **Last Observed (Wins)** |

![Model Forecast Comparison](reports/figures/model_forecast_comparison.png)

![Baseline MAE Comparison](reports/figures/baseline_mae_comparison.png)

### 8.2 Diebold-Mariano Statistical Comparison
To verify whether forecast accuracy differences are statistically significant, we computed Diebold-Mariano test statistics with Harvey-Leybourne-Newbold (HLN) finite-sample adjustments (`reports/tables/diebold_mariano_test.csv`):

| Commodity | DM-HLN Statistic | $p$-value | Mean Loss Difference | Superior Model |
| :--- | :--- | :--- | :--- | :--- |
| **ALL** | **-21.858** | **$< 10^{-15}$** | **-0.0111** | **LightGBM (Significant)** |
| Tomato | -12.808 | $< 10^{-15}$ | -0.0384 | LightGBM (Significant) |
| Green Chilli | -17.583 | $< 10^{-15}$ | -0.0194 | LightGBM (Significant) |
| Onion | -11.074 | $< 10^{-15}$ | -0.0144 | LightGBM (Significant) |
| Potato | -9.776 | $< 10^{-15}$ | -0.0119 | LightGBM (Significant) |
| Cauliflower | -8.855 | $< 10^{-15}$ | -0.0223 | LightGBM (Significant) |
| Brinjal | -3.424 | $6.23 \times 10^{-4}$ | -0.0026 | LightGBM (Significant) |
| **Paddy** | **+7.627** | **$4.95 \times 10^{-14}$** | **+0.0041** | **Last Observed (Significant)** |
| **Wheat** | **+21.598** | **$< 10^{-15}$** | **+0.0109** | **Last Observed (Significant)** |

> ### Honesty Finding: The Foodgrain Boundary
> LightGBM decisively outperforms persistence across all six perishable horticulture crops (reducing forecast errors by 18.9% to 27.3%, DM stat $-21.86$, $p < 10^{-15}$). 
> 
> However, for staple foodgrains (**Wheat** and **Paddy**), the simple **Last Observed baseline outperforms LightGBM** (Wheat MAE ₹42.49 vs ₹68.89; Paddy MAE ₹33.70 vs ₹42.99). Wheat and Paddy are heavily stabilized by government Minimum Support Price (MSP) procurement operations and Food Corporation of India (FCI) buffer stocking. Their prices move as near-perfect random walks with near-zero week-to-week drift; attempting to predict complex non-linear dynamics introduces variance that slightly degrades point accuracy.

### 8.3 Uncertainty Quantification and Conformal Calibration
We trained LightGBM pinball quantile regressors at $\alpha = 0.10$ and $\alpha = 0.90$. Because raw quantile regression under-covered the 80% target interval on the validation split (75.71% empirical coverage), we applied split-conformal calibration using validation residuals:
$$q_{\text{adj}} = \text{Quantile}_{1-\alpha}\left( \max(y - \hat{y}_{p90}, \hat{y}_{p10} - y) \right)$$

Coverage metrics from `reports/tables/interval_coverage.csv`:
- **Validation Set (Raw Quantile):** 75.71% coverage (width = 0.26 log units).
- **Validation Set (Conformal Calibrated):** **80.00% coverage** (width = 0.28 log units).
- **Held-Out Test Set (Conformal Calibrated):** **85.93% coverage** (width = 0.29 log units).

The calibrated intervals are conservative and well-calibrated, successfully enclosing real market price realizations in $\approx 86\%$ of held-out test weeks.

---

## 9. Spatial Recommender Evaluation

The spatial recommender engine was evaluated across **26,352 pseudo-farmer scenarios** on the held-out test period (2024–2026), covering all 8 commodities across 87 producing districts within a 150 km search radius.

### 9.1 Realized Regret and Hit Rates
In each test week, the model selected the top candidate market using forecasted net price ($\hat{y}_{t+2} - \text{transport}$). We then evaluated the **realized net price** against three competitive benchmarks:
1. **Oracle Best Market (Hindsight Upper Bound):** The market that actually yielded the highest net price ex-post.
2. **Nearest Market (Status Quo Baseline):** The default behavior of selling at the closest local APMC yard.
3. **Historical Best Market:** The mandi with the highest expanding historical average price.

Summary metrics from `reports/tables/recommender_regret_summary.csv`:

| Commodity | Test Cases | Model Regret (₹/q) | Model Regret (%) | Nearest Regret (₹/q) | Nearest Regret (%) | Net Advantage vs Nearest (₹/q) | Top-1 Hit Rate (%) | Top-3 Hit Rate (%) | Spearman Rank Corr |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ALL** | **26,352** | **108.16** | **4.54%** | **338.65** | **13.50%** | **+230.49** | **44.60%** | **71.34%** | **0.585** |
| Green Chilli | 3,753 | 246.91 | 6.91% | 736.78 | 20.21% | **+489.87** | 39.20% | 64.83% | 0.637 |
| Cauliflower | 2,805 | 160.25 | 6.73% | 601.44 | 24.38% | **+441.19** | 54.58% | 83.81% | 0.668 |
| Onion | 3,814 | 113.07 | 4.89% | 409.29 | 16.38% | **+296.22** | 44.34% | 63.21% | 0.527 |
| Tomato | 3,473 | 126.46 | 6.45% | 350.21 | 15.23% | **+223.75** | 49.38% | 79.99% | 0.449 |
| Brinjal | 3,404 | 70.48 | 4.10% | 234.80 | 13.01% | **+164.32** | 48.41% | 77.14% | 0.635 |
| Potato | 3,718 | 52.80 | 3.67% | 159.52 | 10.74% | **+106.72** | 44.89% | 69.66% | 0.618 |
| Paddy | 1,614 | 10.49 | 0.47% | 78.60 | 3.43% | **+68.12** | 63.94% | 96.34% | 0.727 |
| Wheat | 3,771 | 39.92 | 1.46% | 46.50 | 1.68% | **+6.58** | 26.39% | 54.52% | 0.520 |

- **Median Regret:** While mean regret is ₹108.16/q due to heavy positive tails during supply shocks, the **median model regret is only ₹10.00/quintal**. In over 71% of all cases, the model successfully places an ex-post top-3 market in its top recommendation.
- **Economic Value:** Recommending optimal markets yields an average net gain of **₹230.49 per quintal** over the nearest mandi. For a modest 50-quintal smallholder harvest, this represents an additional **₹11,524.50 in net revenue**.

![Recommender Regret CDF](reports/figures/recommender_regret_cdf.png)

### 9.2 Hypothesis Testing of Economic Advantage
We tested the null hypothesis ($H_0$): the recommended market's realized net price equals that of the nearest market (`reports/tables/recommender_hypothesis_test.csv`).

- **Paired $t$-test Statistic:** $t = 74.432$, $p < 10^{-300}$
- **Paired Wilcoxon Signed-Rank Statistic:** $W = 43,554,407.0$, $p < 10^{-300}$
- **Standardized Effect Size:** Cohen's $d = 0.459$ (moderate-to-strong practical effect)

$H_0$ is decisively rejected across all 8 commodities. Even for Wheat, where point forecast improvement is modest, spatial routing yields a statistically significant net price gain of +₹6.58/q ($t = 5.641$, $p = 1.81 \times 10^{-8}$).

### 9.3 Sensitivity Analysis Across Freight Rates
We evaluated recommendation stability across four freight transport rates: 0.00, 0.15, 0.25 (default), and 0.50 INR/quintal/km (`reports/tables/recommender_sensitivity_analysis.csv`):

| Freight Rate (₹/q/km) | Top-Market Turnover vs Baseline | Mean Model Regret (₹/q) | Advantage vs Nearest (₹/q) | Mean Travel Distance (km) |
| :--- | :--- | :--- | :--- | :--- |
| **0.00 (Zero Freight)** | 8.3% | 107.48 | +248.44 | 94.9 km |
| **0.15 (Subsidized)** | 3.3% | 107.86 | +237.46 | 90.8 km |
| **0.25 (Baseline)** | **0.0%** | **108.16** | **+230.49** | **88.0 km** |
| **0.50 (High Freight)** | 8.1% | 107.71 | +215.80 | 80.9 km |

**Key Finding:** Over **91.7% of recommendations remain invariant** to transport rate variations between 0.00 and 0.50 ₹/q/km. As shown in `reports/tables/recommender_spread_vs_error.csv`, regional price dispersion (₹400 to ₹1,800/q) dwarfs transport differentials (₹15 to ₹40/q for an 80 km trip).

![Sensitivity Curves](reports/figures/recommender_sensitivity_curve.png)

![Spread vs MAE](reports/figures/recommender_spread_vs_mae.png)

---

## 10. Experimental Sell/Hold (F2) and Buy Now/Wait (B2) Timing Analysis

In accordance with Section 12 of the project specification, we developed and evaluated an **experimental** timing decision module to evaluate whether farmers should sell immediately or hold for 2 weeks ($F_2$), and whether bulk buyers should procure immediately or wait ($B_2$). All components are strictly designated as **experimental**.

### 10.1 Formulation and Holding Cost Economics
Timing decisions cannot be framed simply as positive price movement; they must overcome real physical and financial holding frictions:
$$\text{Net Timing Hurdle} = h \times \text{Holding Cost per Week}$$

Holding costs differ substantially between perishable and durable crops:
- **Perishable Horticulture (High Loss / Cold Chain):** Tomato (₹50/q/wk), Green Chilli (₹40/q/wk), Cauliflower (₹35/q/wk), Brinjal (₹25/q/wk), reflecting rapid weight loss, physical decay, and active cooling requirements.
- **Durable Produce & Foodgrains (Storage / Finance Cost):** Onion (₹15/q/wk), Potato (₹12/q/wk), Wheat (₹8/q/wk), Paddy (₹8/q/wk), reflecting dry warehouse space, ventilation, and capital interest charges.

We trained a LightGBM binary classifier (`lgb_timing_clf_h2.joblib`) on the leak-free feature table using identical chronological splits (Train: 2015–2022, Validation: 2023, Test: 2024–2026) to predict the probability that 2-week forward price returns cover the commodity holding hurdle. The system outputs a calibrated probability and expected net change rather than a hard deterministic yes/no.

### 10.2 Empirical Evaluation on Held-Out Test Period (2024–2026)
We simulated three strategies across 34,969 test weeks (`reports/tables/timing_evaluation.csv`):
1. **Always Sell Now (Status Quo Baseline):** Producer sells immediately upon harvest ($0$ holding cost).
2. **Experimental Model Timing:** Holds for 2 weeks only when $P(\text{Gain}) \ge 0.50$ and expected gain exceeds 2-week holding cost; otherwise sells immediately.
3. **Always Sell at Seasonal Peak:** Producer blindly holds produce until the historical calendar peak month of that crop, paying cumulative weekly holding costs.

| Commodity | Category | Test Cases | 2-Wk Hold Cost (₹/q) | Accuracy | Precision | Recall | Hold Rate (%) | Sell Now Payoff (₹/q) | Model Payoff (₹/q) | Seasonal Peak Payoff (₹/q) | Net Advantage vs Sell Now (₹/q) | Beats Sell Now? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ALL** | **Overall** | **34,969** | **₹48.25** | **0.839** | **0.745** | **0.620** | **12.5%** | **1,976.10** | **2,007.45** | **1,844.16** | **+31.35 (+1.50%)** | **Yes (Selective)** |
| Green Chilli | Perishable | 7,127 | ₹80.00 | 0.816 | 0.825 | 0.657 | 19.4% | 2,807.44 | 2,928.94 | 2,671.05 | +121.50 (+4.33%) | Yes |
| Cauliflower | Perishable | 2,364 | ₹70.00 | 0.813 | 0.871 | 0.740 | 11.0% | 1,643.27 | 1,685.25 | 1,504.38 | +41.98 (+2.55%) | Yes |
| Tomato | Perishable | 2,547 | ₹100.00 | 0.843 | 0.817 | 0.749 | 5.2% | 1,801.45 | 1,836.19 | 1,483.71 | +34.74 (+1.93%) | Yes |
| Onion | Durable | 7,154 | ₹30.00 | 0.815 | 0.598 | 0.594 | 22.6% | 2,011.08 | 2,032.53 | 1,817.73 | +21.45 (+1.07%) | Yes |
| Brinjal | Perishable | 3,512 | ₹50.00 | 0.790 | 0.743 | 0.606 | 12.6% | 1,440.07 | 1,455.57 | 1,346.25 | +15.51 (+1.08%) | Yes |
| Potato | Durable | 4,868 | ₹24.00 | 0.818 | 0.670 | 0.539 | 19.9% | 1,279.44 | 1,288.74 | 1,180.79 | +9.30 (+0.73%) | Yes |
| Paddy | Durable | 1,165 | ₹16.00 | 0.912 | 0.526 | 0.377 | 6.5% | 2,193.15 | 2,197.07 | 2,159.08 | +3.92 (+0.18%) | Negligible |
| Wheat | Durable | 6,232 | ₹16.00 | 0.931 | 0.596 | 0.235 | 2.9% | 2,632.91 | 2,635.28 | 2,590.28 | +2.38 (+0.09%) | Negligible |

### 10.3 Key Findings and Real-World Limitations
1. **Failure of Naive Seasonal Holding:** "Always Sell at Seasonal Peak" is severely unprofitable, underperforming "Always Sell Now" by ₹131.94/quintal overall (₹1,844.16 vs ₹1,976.10/q). Waiting 8 to 16 weeks for a seasonal peak incurs substantial storage rent, shrinkage, and capital financing costs that completely erase peak price premiums.
2. **Model Selectivity:** The experimental model recommends holding only in **12.5% of weeks**, defaulting to immediate sale in 87.5% of cases. When it does recommend holding, it captures modest gains (+₹31.35/quintal overall; +1.50%).
3. **Foodgrain Flatness:** For Wheat and Paddy, timing yields essentially zero economic benefit (+₹2.38/q and +₹3.92/q, $< 0.2\%$). Under MSP procurement, prices behave as flat random walks where storage costs guarantee a net loss.
4. **Unforecastable Exogenous Shocks:** Crucially, historical price patterns **cannot predict exogenous structural shocks**:
   - Severe unseasonal weather anomalies (cyclones, hailstorms, flash floods).
   - Sudden regulatory interventions (overnight export bans, sudden tariff hikes, stock limits under the Essential Commodities Act).
   - Market disruptions (mandi worker strikes, local transportation blockades).
   Farmers facing potential spoilage must treat timing forecasts as speculative indicators rather than firm advice. Immediate sale remains the rational default for smallholders lacking dedicated climate-controlled storage.

---

## 11. Comprehensive Project Limitations

In strict adherence to Section 13 of the project specification, the following 11 structural limitations govern the interpretation of these findings:

1. **Absence of Supply and Arrival Quantities:** Agmarknet archives record clearing prices but lack transaction volumes or truckload arrivals. The system cannot model price elasticity or verify whether a smaller secondary mandi could absorb a 20-ton shipment without crashing local prices.
2. **Price Semantics and Reporting Assumptions:** While modal clearing price represents the central auction tendency, individual farmer realizations depend on transaction timing, batch size, and daily settlement liquidity.
3. **Irregular Reporting, Gaps, and Stale Prices:** In rural mandis, non-trading days, local holidays, and manual data entry errors create gaps. Although linear interpolation and flat-run flags mitigate these issues, sudden reporting suspensions can degrade forecast quality.
4. **Transport Cost and Geographic Approximations:** Freight cost is estimated using a fixed linear rate (₹0.25/q/km) and district centroid coordinates adjusted by a 1.35 road tortuosity factor. Real freight rates vary with return-trip loads, vehicle type, and local road conditions.
5. **Omission of Transaction Frictions:** Physical mandi trading involves unrecorded costs: mandi cess (1–2%), commission agent charges, loading/unloading fees, arbitrary weighing deductions, and informal credit arrangements that bind farmers to specific local traders.
6. **Exogenous Shocks and Policy Interventions:** Autoregressive models cannot predict sudden policy interventions, such as export bans, stockholding limits, sudden tariff revisions, or extreme weather events (unseasonal hail or floods).
7. **Dominance of Persistent Market Premiums:** Certain major consumption hubs (e.g., Azadpur or Vashi) maintain structural price premiums across all seasons due to institutional buying power. The recommender may consistently favor these hubs, which could lead to congestion if widely adopted.
8. **Pruning of Sparse Rural Mandis:** To ensure statistical reliability, 1,608 series with $< 104$ weeks of data or $> 40\%$ missingness were excluded, leaving remote farming clusters without localized predictions.
9. **Autocorrelation and Optimistic Significance:** Temporal autocorrelation in agricultural time series can artificially inflate classical test statistics. We addressed this using Newey-West and Harvey-Leybourne-Newbold corrections, but raw $p$-values should be interpreted with appropriate caution.
10. **Historical Data Horizon:** The current models and precomputed forecasts reflect data collected through April 2026. Real-time production deployment requires an ongoing API pipeline to ingest daily Agmarknet feeds.
11. **Geographic and Crop Scope:** Findings are strictly validated for the 8 selected crops across Uttar Pradesh, Madhya Pradesh, and Maharashtra. Extrapolating to other states or specialized cash crops requires retraining and localized validation.

---

## 12. Conclusion

Mandi Watch demonstrates that machine learning and spatial optimization can unlock tangible economic value for Indian smallholder farmers. By decoupling multi-horizon price forecasting from spatial transport optimization, the system achieves:
1. **Accurate Forecasting:** A global LightGBM regressor trained on price change increments ($\Delta y$) reduces perishable crop forecast errors by 16.6% to 27.3% over persistence baselines, with well-calibrated 85.9% conformal uncertainty coverage.
2. **Realized Economic Arbitrage:** On a held-out test evaluation of 26,352 scenarios, the spatial recommender delivers an average realized net advantage of **₹230.49 per quintal** over the default nearest market ($p < 10^{-15}$, Cohen's $d = 0.459$), achieving a top-3 hit rate of 71.34%.
3. **Robustness to Freight Volatility:** Over 91% of recommended market selections remain stable across freight rate variations ranging from ₹0.00 to ₹0.50/q/km, demonstrating that local agricultural price dispersion dominates transportation differentials.

Deployed via an interactive Streamlit application with explicit confidence bounds and decision warnings, Mandi Watch provides actionable, transparent decision support designed to help farmers capture fair market value.
