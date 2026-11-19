"""
Tests for baseline models (Section 8.3 of spec).
Verifies:
1. Last observed gives zero error on a constant series.
2. Seasonal naive gives zero error on a perfectly periodic 52-week synthetic series.
3. Market historical average gives zero error on a constant series.
4. Baseline forecasts strictly use past data (no lookahead).
"""

import numpy as np
import pandas as pd
import pytest

from src.baselines import add_baseline_predictions
from src.evaluation import compute_metrics, evaluate_forecasts


def test_last_observed_constant_series():
    """Last observed price on a constant series must yield exactly zero error."""
    dates = pd.date_range("2020-01-01", periods=100, freq="W-MON")
    df = pd.DataFrame({
        "market": "M1",
        "commodity": "Potato",
        "week_start": dates,
        "log_price": np.log(1500.0),  # constant 1500 INR
    })
    h = 2
    df[f"target_h{h}"] = df["log_price"].shift(-h)

    res = add_baseline_predictions(df, h=h)
    valid = res.dropna(subset=[f"target_h{h}", "pred_last_observed"])

    metrics = compute_metrics(np.exp(valid[f"target_h{h}"]), np.exp(valid["pred_last_observed"]))
    assert np.isclose(metrics["mae"], 0.0, atol=1e-5), f"Expected 0 MAE, got {metrics['mae']}"
    assert np.isclose(metrics["mape"], 0.0, atol=1e-5), f"Expected 0 MAPE, got {metrics['mape']}"


def test_historical_mean_constant_series():
    """Historical average on a constant series must yield exactly zero error."""
    dates = pd.date_range("2020-01-01", periods=100, freq="W-MON")
    df = pd.DataFrame({
        "market": "M1",
        "commodity": "Onion",
        "week_start": dates,
        "log_price": np.log(2000.0),
    })
    h = 2
    df[f"target_h{h}"] = df["log_price"].shift(-h)

    res = add_baseline_predictions(df, h=h)
    valid = res.dropna(subset=[f"target_h{h}", "pred_historical_mean"])

    metrics = compute_metrics(np.exp(valid[f"target_h{h}"]), np.exp(valid["pred_historical_mean"]))
    assert np.isclose(metrics["mae"], 0.0, atol=1e-5)
    assert np.isclose(metrics["mape"], 0.0, atol=1e-5)


def test_seasonal_naive_perfect_seasonality():
    """
    Seasonal naive on a perfectly periodic 52-week series must yield zero error
    once 52 weeks of history are available.
    """
    n_weeks = 52 * 4  # 4 years
    dates = pd.date_range("2018-01-01", periods=n_weeks, freq="W-MON")
    # Synthetic seasonal pattern: 1000 + 500 * sin(2*pi*w/52)
    w = np.arange(n_weeks) % 52
    prices = 1000.0 + 500.0 * np.sin(2.0 * np.pi * w / 52.0)
    prices = np.maximum(prices, 100.0)

    df = pd.DataFrame({
        "market": "M1",
        "commodity": "Tomato",
        "week_start": dates,
        "log_price": np.log(prices),
    })
    h = 2
    df[f"target_h{h}"] = df["log_price"].shift(-h)

    res = add_baseline_predictions(df, h=h)
    # Check after year 1 (index >= 52) where 52-week lag exists without fallback
    eval_slice = res.iloc[52:].dropna(subset=[f"target_h{h}", "pred_seasonal_naive"])

    metrics = compute_metrics(np.exp(eval_slice[f"target_h{h}"]), np.exp(eval_slice["pred_seasonal_naive"]))
    assert np.isclose(metrics["mae"], 0.0, atol=1e-3), f"Expected near-zero MAE on pure sine, got {metrics['mae']}"


def test_baselines_no_future_leakage():
    """Verifies that altering prices in the future after week t does not change baseline predictions at week t."""
    dates = pd.date_range("2020-01-01", periods=100, freq="W-MON")
    prices = np.linspace(1000, 2000, 100)
    df = pd.DataFrame({
        "market": "M1",
        "commodity": "Wheat",
        "week_start": dates,
        "log_price": np.log(prices),
    })
    h = 2
    res_clean = add_baseline_predictions(df, h=h)

    # Corrupt future after index 60
    df_corrupted = df.copy()
    df_corrupted.loc[61:, "log_price"] = np.log(99999.0)
    res_corrupted = add_baseline_predictions(df_corrupted, h=h)

    # Check predictions at index 60 (end of week t)
    for col in ["pred_last_observed", "pred_historical_mean", "pred_seasonal_naive"]:
        assert np.isclose(res_clean.loc[60, col], res_corrupted.loc[60, col]), (
            f"Leakage detected in {col}: future changes affected current prediction!"
        )
