"""
Unit tests for statistical hypothesis testing functions in src/stats_tests.py.
Tested exclusively on synthetic data with known ground-truth statistical properties.
"""

import numpy as np
import pandas as pd
import pytest

import src.stats_tests as st


@pytest.fixture
def rng():
    return np.random.RandomState(42)


def test_shapiro_wilk_normal_vs_skewed(rng):
    """Verifies that Shapiro-Wilk identifies Gaussian vs skewed synthetic distributions."""
    normal_data = rng.normal(loc=10.0, scale=1.0, size=400)
    res_normal = st.test_shapiro_wilk(normal_data, max_n=500, random_state=42)
    assert res_normal["p_value"] > 0.05
    assert res_normal["is_normal"] is True
    assert res_normal["qq_r2"] > 0.95

    # Strongly skewed exponential distribution
    skewed_data = rng.exponential(scale=5.0, size=400)
    res_skewed = st.test_shapiro_wilk(skewed_data, max_n=500, random_state=42)
    assert res_skewed["p_value"] < 0.01
    assert res_skewed["is_normal"] is False
    assert res_skewed["skewness"] > 1.0


def test_levene_equal_vs_unequal_variance(rng):
    """Verifies Levene test distinguishes homoscedastic from heteroscedastic market data."""
    # Equal variances
    m1_p = rng.normal(8.0, 0.2, 12)
    m2_p = rng.normal(8.0, 0.2, 12)
    df_homo = pd.DataFrame({
        "commodity": "TestCrop",
        "market": ["MarketA"] * 12 + ["MarketB"] * 12,
        "log_price": np.concatenate([m1_p, m2_p]),
    })
    res_homo = st.test_levene_homoscedasticity(df_homo, "TestCrop", group_col="market", value_col="log_price")
    assert res_homo["equal_variance"] is True
    assert res_homo["p_value"] > 0.05

    # Heteroscedastic variances (5x scale difference)
    m1_p_hetero = rng.normal(8.0, 0.1, 12)
    m2_p_hetero = rng.normal(8.0, 1.0, 12)
    df_hetero = pd.DataFrame({
        "commodity": "TestCrop",
        "market": ["MarketA"] * 12 + ["MarketB"] * 12,
        "log_price": np.concatenate([m1_p_hetero, m2_p_hetero]),
    })
    res_hetero = st.test_levene_homoscedasticity(df_hetero, "TestCrop", group_col="market", value_col="log_price")
    assert res_hetero["equal_variance"] is False
    assert res_hetero["p_value"] < 0.05
    assert res_hetero["variance_ratio"] > 10.0


def test_one_way_price_differences_synthetic(rng):
    """Verifies Kruskal-Wallis detects large synthetic price shift across markets."""
    n_obs = 20
    # Market A around ₹1500, Market B around ₹2500
    p_a = rng.normal(1500, 50, n_obs)
    p_b = rng.normal(2500, 50, n_obs)
    df = pd.DataFrame({
        "commodity": "Wheat",
        "market": ["Mandi_A"] * n_obs + ["Mandi_B"] * n_obs,
        "price_inr": np.concatenate([p_a, p_b]),
    })
    res = st.test_one_way_price_differences(df, "Wheat", group_col="market", value_col="price_inr")
    assert res["significant"] is True
    assert res["p_value"] < 0.001
    assert res["eta_squared"] > 0.5
    assert 900.0 <= res["max_spread_inr"] <= 1100.0


def test_paired_market_differences_synthetic(rng):
    """Verifies paired Wilcoxon test detects constant price premium with Cohen's d."""
    months = [f"2023-{m:02d}" for m in range(1, 13)]
    # Mandi A is consistently ₹200 higher than Mandi B
    base = rng.normal(2000, 100, 12)
    p_a = base + 200.0 + rng.normal(0, 10, 12)
    p_b = base

    df = pd.DataFrame({
        "commodity": "Tomato",
        "year_month": months + months,
        "market": ["Mandi_A"] * 12 + ["Mandi_B"] * 12,
        "price_inr": np.concatenate([p_a, p_b]),
    })
    res = st.test_paired_market_differences(df, "Tomato", "Mandi_A", "Mandi_B", value_col="price_inr")
    assert res["significant"] is True
    assert res["p_value"] < 0.01
    assert 180.0 <= res["mean_diff_inr"] <= 220.0
    assert res["cohens_d"] > 2.0  # Huge effect size


def test_benjamini_hochberg():
    """Verifies Benjamini-Hochberg FDR p-value adjustment."""
    raw_p = [0.001, 0.01, 0.04, 0.08, 0.50]
    reject, adj_p = st.apply_benjamini_hochberg(raw_p, alpha=0.05)
    assert len(adj_p) == len(raw_p)
    assert all(adj >= raw for adj, raw in zip(adj_p, raw_p))
    assert reject[0] is True or reject[0] == np.bool_(True)
    assert reject[-1] is False or reject[-1] == np.bool_(False)


def test_stationarity_adf_kpss(rng):
    """Verifies ADF/KPSS detect stationary white noise vs random walk unit root."""
    # Stationary AR(1)
    stationary_ts = np.zeros(200)
    for t in range(1, 200):
        stationary_ts[t] = 0.5 * stationary_ts[t - 1] + rng.normal(0, 1)

    res_stat = st.test_stationarity(stationary_ts)
    assert res_stat["adf_p"] < 0.05
    assert res_stat["is_stationary"] is True

    # Unit root random walk
    rw = np.cumsum(rng.normal(0.1, 1.0, 200))
    res_rw = st.test_stationarity(rw)
    assert res_rw["adf_p"] > 0.05
    assert res_rw["is_stationary"] is False


def test_diebold_mariano_superior_model(rng):
    """Verifies Diebold-Mariano test flags significantly better forecast errors."""
    n = 100
    # Model 1 has small errors, Model 2 has huge errors
    e1 = rng.normal(0, 1.0, n)
    e2 = rng.normal(0, 4.0, n)

    res = st.test_diebold_mariano(e1, e2, h=2, criterion="MSE")
    assert res["significant"] is True
    assert res["p_value"] < 0.01
    assert res["mean_differential"] < 0  # e1^2 < e2^2


def test_ljung_box_white_noise_vs_autocorrelated(rng):
    """Verifies Ljung-Box test flags serial autocorrelation."""
    white_noise = rng.normal(0, 1, 150)
    res_wn = st.test_ljung_box(white_noise, lags=[6, 12])
    assert res_wn["has_autocorrelation"] is False

    # Highly autocorrelated AR(1)
    ar1 = np.zeros(150)
    for t in range(1, 150):
        ar1[t] = 0.8 * ar1[t - 1] + rng.normal(0, 1)
    res_ar = st.test_ljung_box(ar1, lags=[6, 12])
    assert res_ar["has_autocorrelation"] is True
