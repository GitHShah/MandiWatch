"""Unit tests for ISO weekly aggregation, series quality filters, and short gap interpolation."""
import numpy as np
import pandas as pd
import pytest

from src.data_cleaning import (
    aggregate_daily_to_iso_weeks,
    filter_series_quality,
    interpolate_short_gaps,
)


def test_iso_week_edge_cases():
    """Verify ISO week mapping correctly handles year-boundary edge cases.

    For example:
    - 2015-12-31 (Thursday), 2016-01-01 (Friday), 2016-01-02 (Saturday), 2016-01-03 (Sunday)
      all belong to ISO year 2015, week 53.
    - 2016-01-04 (Monday) belongs to ISO year 2016, week 01.
    """
    df = pd.DataFrame({
        "state": ["UP"] * 5,
        "district": ["Agra"] * 5,
        "market": ["Agra"] * 5,
        "commodity": ["Potato"] * 5,
        "variety": ["Desi"] * 5,
        "grade": ["FAQ"] * 5,
        "date": pd.to_datetime(["2015-12-31", "2016-01-01", "2016-01-02", "2016-01-03", "2016-01-04"]),
        "price": [1000.0, 1020.0, 1040.0, 1060.0, 1100.0],
        "is_flat_run": [False, False, False, False, False],
    })

    weekly = aggregate_daily_to_iso_weeks(df)
    assert len(weekly) == 2

    # Week 1: ISO 2015-W53 with 4 daily reports
    w53 = weekly[(weekly["iso_year"] == 2015) & (weekly["iso_week"] == 53)]
    assert len(w53) == 1
    assert w53["n_reports"].iloc[0] == 4
    # Median of [1000, 1020, 1040, 1060] is 1030
    assert w53["median_price"].iloc[0] == 1030.0
    assert w53["min_price"].iloc[0] == 1000.0
    assert w53["max_price"].iloc[0] == 1060.0
    assert not pd.isna(w53["std_price"].iloc[0])
    assert w53["week_start"].iloc[0] == pd.Timestamp("2015-12-28").date()

    # Week 2: ISO 2016-W01 with 1 daily report
    w1 = weekly[(weekly["iso_year"] == 2016) & (weekly["iso_week"] == 1)]
    assert len(w1) == 1
    assert w1["n_reports"].iloc[0] == 1
    assert w1["median_price"].iloc[0] == 1100.0
    # With n_reports == 1, std_price must be NaN/None
    assert pd.isna(w1["std_price"].iloc[0])
    assert w1["week_start"].iloc[0] == pd.Timestamp("2016-01-04").date()


def test_weekly_flat_run_share():
    """Verify flat-run share is calculated correctly in weekly aggregation."""
    df = pd.DataFrame({
        "state": ["UP"] * 4,
        "district": ["Agra"] * 4,
        "market": ["Agra"] * 4,
        "commodity": ["Potato"] * 4,
        "variety": ["Desi"] * 4,
        "grade": ["FAQ"] * 4,
        "date": pd.to_datetime(["2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]),
        "price": [1000.0, 1000.0, 1000.0, 1000.0],
        "is_flat_run": [True, True, False, False],
    })

    weekly = aggregate_daily_to_iso_weeks(df)
    assert len(weekly) == 1
    assert weekly["flat_run_share"].iloc[0] == 0.5


def test_gap_interpolation_rule():
    """Verify short gap rule: gaps <= 2 weeks are linearly interpolated; gaps >= 3 are NOT.

    - 1-week gap (missing 1 week between 2 reports): interpolated.
    - 2-week gap (missing 2 weeks between 2 reports): interpolated.
    - 3-week gap (missing 3 weeks between 2 reports): NOT interpolated (stays NaN).
    """
    # Create weekly series with:
    # W1: 10.0
    # W2: missing (1-week gap between W1 and W3) -> should interpolate
    # W3: 12.0
    # W4, W5: missing (2-week gap between W3 and W6) -> should interpolate
    # W6: 15.0
    # W7, W8, W9: missing (3-week gap between W6 and W10) -> must NOT interpolate
    # W10: 19.0
    mondays = pd.date_range("2023-01-02", periods=10, freq="W-MON")
    df = pd.DataFrame({
        "market": ["M1"] * 4,
        "commodity": ["C1"] * 4,
        "week_start": [mondays[0].date(), mondays[2].date(), mondays[5].date(), mondays[9].date()],
        "median_price": [np.exp(1.0), np.exp(3.0), np.exp(6.0), np.exp(10.0)],
        "log_price": [1.0, 3.0, 6.0, 10.0],
    })

    interp_df = interpolate_short_gaps(df, max_gap=2)

    # Output dataframe has all calendar weeks between min and max for this series
    assert len(interp_df) == 10

    # Week index 1 (W2): 1-week gap interpolated: (1.0 + 3.0)/2 = 2.0
    w2 = interp_df[interp_df["week_start"] == mondays[1].date()].iloc[0]
    assert w2["is_interpolated"] is True or w2["is_interpolated"] == 1
    assert pytest.approx(w2["log_price_interpolated"], 0.001) == 2.0

    # Week index 3 and 4 (W4, W5): 2-week gap interpolated linearly from 3.0 to 6.0: 4.0, 5.0
    w4 = interp_df[interp_df["week_start"] == mondays[3].date()].iloc[0]
    w5 = interp_df[interp_df["week_start"] == mondays[4].date()].iloc[0]
    assert w4["is_interpolated"] is True or w4["is_interpolated"] == 1
    assert w5["is_interpolated"] is True or w5["is_interpolated"] == 1
    assert pytest.approx(w4["log_price_interpolated"], 0.001) == 4.0
    assert pytest.approx(w5["log_price_interpolated"], 0.001) == 5.0

    # Week index 6, 7, 8 (W7, W8, W9): 3-week gap must NOT be interpolated (must stay NaN)
    for idx in [6, 7, 8]:
        w = interp_df[interp_df["week_start"] == mondays[idx].date()].iloc[0]
        assert pd.isna(w["log_price_interpolated"])
        assert not bool(w["is_interpolated"])


def test_series_quality_filter():
    """Verify series quality filtering: min_weeks >= 104 and max_missing_share <= 0.40."""
    # Series 1: 105 weeks out of 110 (missing share ~4.5%) -> KEEP
    # Series 2: 80 weeks out of 85 (missing share ~5.9%) -> DROP (too_few_weeks)
    # Series 3: 110 weeks out of 250 (missing share 56% > 40%) -> DROP (high_missing_share)
    records = []
    # Series 1
    w_s1 = pd.date_range("2020-01-06", periods=105, freq="W-MON")
    for w in w_s1:
        records.append({"market": "M1", "commodity": "C1", "week_start": w.date(), "median_price": 1000.0})

    # Series 2
    w_s2 = pd.date_range("2020-01-06", periods=80, freq="W-MON")
    for w in w_s2:
        records.append({"market": "M2", "commodity": "C1", "week_start": w.date(), "median_price": 1000.0})

    # Series 3 (spans 250 weeks with 110 active)
    w_s3 = pd.date_range("2018-01-01", periods=250, freq="W-MON")
    # Take only every 2nd or 3rd week to get 110 weeks
    active_w_s3 = w_s3[:220:2]  # exactly 110 weeks
    for w in active_w_s3:
        records.append({"market": "M3", "commodity": "C1", "week_start": w.date(), "median_price": 1000.0})

    df = pd.DataFrame(records)
    kept_df, dropped_df, filtered_weekly = filter_series_quality(df, min_weeks=104, max_missing_share=0.40)

    assert len(kept_df) == 1
    assert kept_df["series_id"].iloc[0] == "M1__C1"

    assert len(dropped_df) == 2
    drop_m2 = dropped_df[dropped_df["market"] == "M2"].iloc[0]
    assert "too_few_weeks" in drop_m2["reason"]

    drop_m3 = dropped_df[dropped_df["market"] == "M3"].iloc[0]
    assert "high_missing_share" in drop_m3["reason"]
