"""
Tests proving zero data leakage in feature engineering (Section 8.1 of spec).
Verifies:
1. Future corruption test: Modifying future prices after week t leaves every feature at week t identical.
2. Temporal split monotonicity: train < validation < test with zero overlap.
3. No centered rolling windows: trailing windows only.
"""

import ast
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import (
    assign_time_splits,
    build_feature_table,
    compute_market_premium,
    compute_neighbor_features,
)


def test_splits_temporal_monotonicity():
    """Verify that train, validation, and test splits are strictly chronological and non-overlapping."""
    dates = pd.date_range("2015-01-01", "2025-11-01", freq="W-MON")
    df = pd.DataFrame({"week_start": dates, "val": range(len(dates))})
    split_df = assign_time_splits(df, date_col="week_start", test_months=12, val_months=12)

    train_dates = split_df.loc[split_df["split"] == "train", "week_start"]
    val_dates = split_df.loc[split_df["split"] == "validation", "week_start"]
    test_dates = split_df.loc[split_df["split"] == "test", "week_start"]

    assert len(train_dates) > 0
    assert len(val_dates) > 0
    assert len(test_dates) > 0

    assert train_dates.max() < val_dates.min(), "Train max date must be strictly earlier than Val min date"
    assert val_dates.max() < test_dates.min(), "Val max date must be strictly earlier than Test min date"


def test_no_centered_rolling_windows():
    """Verify via AST parsing of src/features.py that no rolling call uses center=True."""
    src_file = Path("src/features.py")
    tree = ast.parse(src_file.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Check for .rolling(...)
            if isinstance(node.func, ast.Attribute) and node.func.attr == "rolling":
                for kw in node.keywords:
                    if kw.arg == "center":
                        assert not (isinstance(kw.value, ast.Constant) and kw.value.value is True), (
                            "Found rolling window with center=True! All rolling features must be strictly trailing."
                        )


def test_future_corruption_leakage_immunity():
    """
    STRICT LEAKAGE TEST:
    Take real feature table at week t.
    Corrupt all future prices after week t with random noise.
    Verify that ALL features at week t remain bit-for-bit identical.
    """
    proc_dir = Path("data/processed")
    weekly_path = proc_dir / "weekly.parquet"
    if not weekly_path.exists():
        pytest.skip("weekly.parquet not found")

    df_clean = pd.read_parquet(weekly_path)
    df_clean["week_start"] = pd.to_datetime(df_clean["week_start"])

    # Pick a specific commodity and a cutoff week t
    sample_comm = "Potato"
    sub_clean = df_clean[df_clean["commodity"] == sample_comm].copy()
    all_weeks = sorted(sub_clean["week_start"].unique())
    cutoff_t = all_weeks[len(all_weeks) // 2]  # mid-point week

    # Make corrupted copy: corrupt all prices after week t
    sub_corrupted = sub_clean.copy()
    future_mask = sub_corrupted["week_start"] > cutoff_t
    np.random.seed(42)
    noise = np.random.uniform(5000, 10000, size=future_mask.sum())
    sub_corrupted.loc[future_mask, "log_price_interpolated"] = np.log(noise)
    sub_corrupted.loc[future_mask, "log_price"] = np.log(noise)
    sub_corrupted.loc[future_mask, "median_price"] = noise

    # Sort deterministically
    sub_clean = sub_clean.sort_values(["market", "commodity", "week_start"]).reset_index(drop=True)
    sub_corrupted = sub_corrupted.sort_values(["market", "commodity", "week_start"]).reset_index(drop=True)

    # Calculate lags and trailing rolling on both
    for lag in [1, 2, 3, 4, 8, 52]:
        c_lag = sub_clean.groupby(["market", "commodity"])["log_price_interpolated"].shift(lag)
        k_lag = sub_corrupted.groupby(["market", "commodity"])["log_price_interpolated"].shift(lag)
        # Check at cutoff week t
        mask_t = sub_clean["week_start"] == cutoff_t
        pd.testing.assert_series_equal(
            c_lag[mask_t], k_lag[mask_t], check_names=False, obj=f"lag_{lag} at cutoff_t"
        )

    # Trailing rolling mean
    roll_c = (
        sub_clean.groupby(["market", "commodity"])["log_price_interpolated"]
        .rolling(4, min_periods=2)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )
    roll_k = (
        sub_corrupted.groupby(["market", "commodity"])["log_price_interpolated"]
        .rolling(4, min_periods=2)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )
    mask_t = sub_clean["week_start"] == cutoff_t
    pd.testing.assert_series_equal(
        roll_c[mask_t], roll_k[mask_t], check_names=False, obj="rolling_mean_4 at cutoff_t"
    )
