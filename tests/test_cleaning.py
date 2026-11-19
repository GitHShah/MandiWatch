"""Unit tests for Mandi Watch data cleaning pipeline (src/data_cleaning.py)."""
from __future__ import annotations

import tempfile
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
import pytest

from src.config import AppConfig, CleaningConfig, DataRangeConfig, ForecastConfig, GeoConfig, ModelsConfig, PathsConfig, ScopeConfig, SplitConfig
from src.data_cleaning import (
    build_or_load_name_fixes,
    step1_load_raw_data,
    step2_standardize_text,
    step3_parse_dates_and_prices,
    step4_deduplicate_records,
    step5_handle_outliers,
    step6_detect_flat_runs,
    step7_save_clean_daily,
)


@pytest.fixture
def test_config(tmp_path: Path) -> AppConfig:
    """Fixture providing a self-contained AppConfig using temporary directories."""
    raw_dir = tmp_path / "raw"
    interim_dir = tmp_path / "interim"
    processed_dir = tmp_path / "processed"
    tables_dir = tmp_path / "tables"

    for d in [raw_dir, interim_dir, processed_dir, tables_dir]:
        d.mkdir(parents=True, exist_ok=True)

    paths = PathsConfig(
        raw_dir=raw_dir,
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        geo_dir=tmp_path / "geo",
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        figures_dir=tmp_path / "figures",
        tables_dir=tables_dir,
        notebooks_dir=tmp_path / "notebooks",
    )

    cleaning = CleaningConfig(
        mad_multiplier=5.0,
        flat_run_length=14,
        min_weeks=104,
        max_missing_share=0.40,
        max_interpolation_gap=2,
        outlier_action="drop",
        rolling_window=15,
    )

    return AppConfig(
        project_name="Mandi Watch Test",
        random_seed=42,
        paths=paths,
        data_range=DataRangeConfig("2023-01-01", "2023-12-31"),
        scope=ScopeConfig(n_commodities=2, n_states=1),
        forecast=ForecastConfig(2, [1, 2, 4], "log"),
        split=SplitConfig(12, 12),
        cleaning=cleaning,
        geo=GeoConfig(1.35, 150.0, 0.25, "ASSUMPTION"),
        models=ModelsConfig({"n_estimators": 10}, [0.1, 0.5, 0.9]),
    )


def test_name_fixes_auto_approval(tmp_path: Path):
    """Test that build_or_load_name_fixes auto-approves casing/whitespace and leaves spelling pending."""
    candidates_csv = tmp_path / "name_fix_candidates.csv"
    fixes_csv = tmp_path / "name_fixes.csv"

    # Synthetic candidate data
    candidates_data = [
        {"entity_type": "commodity", "variant_a": "Kutki", "variant_b": "kutki", "similarity": 1.0, "issue_type": "casing_or_whitespace"},
        {"entity_type": "state", "variant_a": "Uttarakhand", "variant_b": "Uttrakhand", "similarity": 0.952, "issue_type": "spelling_variant"},
    ]
    pd.DataFrame(candidates_data).to_csv(candidates_csv, index=False)

    fixes_df = build_or_load_name_fixes(candidates_csv, fixes_csv)
    assert len(fixes_df) == 2

    # Check casing variant is approved='yes'
    casing_row = fixes_df[fixes_df["issue_type"] == "casing_or_whitespace"].iloc[0]
    assert casing_row["approved"] == "yes"
    assert casing_row["clean_name"] == "Kutki"

    # Check spelling variant is approved='pending'
    spelling_row = fixes_df[fixes_df["issue_type"] == "spelling_variant"].iloc[0]
    assert spelling_row["approved"] == "pending"


def test_parse_dates_and_drop_non_positive_prices():
    """Test that unparseable dates and non-positive prices are dropped."""
    con = duckdb.connect(":memory:")
    con.execute("""
        CREATE TABLE stage_text_clean AS
        SELECT * FROM (VALUES
            ('UP', 'Agra', 'Agra APMC', 'Potato', 'Desi', 'FAQ', '2023-01-01', 1000.0, 900.0, 1100.0, 1),
            ('UP', 'Agra', 'Agra APMC', 'Potato', 'Desi', 'FAQ', '15/01/2023', 1050.0, 950.0, 1150.0, 1),
            ('UP', 'Agra', 'Agra APMC', 'Potato', 'Desi', 'FAQ', 'invalid-date', 1000.0, 900.0, 1100.0, 1), -- bad date
            ('UP', 'Agra', 'Agra APMC', 'Potato', 'Desi', 'FAQ', '2023-01-02', 0.0, 0.0, 0.0, 1),           -- zero price
            ('UP', 'Agra', 'Agra APMC', 'Potato', 'Desi', 'FAQ', '2023-01-03', -50.0, -60.0, -40.0, 1)       -- negative price
        ) as t(state, district, market, commodity, variety, grade, date_raw, price, min_price, max_price, commodity_code);
    """)

    metrics = step3_parse_dates_and_prices(con)
    assert metrics["rows_before"] == 5
    assert metrics["rows_after"] == 2
    assert metrics["rows_removed"] == 3

    surviving = con.execute("SELECT date, price FROM stage_valid_dates_prices ORDER BY date;").df()
    assert len(surviving) == 2
    assert (surviving["price"] > 0).all()


def test_deduplication_and_median_conflict_resolution():
    """Test deduplication keeps median price when duplicates conflict."""
    con = duckdb.connect(":memory:")
    con.execute("""
        CREATE TABLE stage_valid_dates_prices AS
        SELECT * FROM (VALUES
            ('UP', 'Agra', 'Agra APMC', 'Potato', 'Desi', 'FAQ', '2023-01-01'::DATE, 1000.0, 900.0, 1100.0, 1),
            ('UP', 'Agra', 'Agra APMC', 'Potato', 'Desi', 'FAQ', '2023-01-01'::DATE, 1200.0, 900.0, 1300.0, 1), -- duplicate conflict
            ('UP', 'Agra', 'Agra APMC', 'Potato', 'Desi', 'FAQ', '2023-01-02'::DATE, 1050.0, 950.0, 1150.0, 1),
            ('UP', 'Agra', 'Agra APMC', 'Potato', 'Desi', 'FAQ', '2023-01-02'::DATE, 1050.0, 950.0, 1150.0, 1)  -- exact duplicate
        ) as t(state, district, market, commodity, variety, grade, date, price, min_price, max_price, commodity_code);
    """)

    metrics = step4_deduplicate_records(con)
    assert metrics["rows_before"] == 4
    assert metrics["rows_after"] == 2
    assert metrics["conflicting_groups"] == 1

    df = con.execute("SELECT date::VARCHAR as date, price FROM stage_deduped ORDER BY date;").df()
    # On 2023-01-01, median of 1000 and 1200 is 1100
    assert df.loc[df["date"] == "2023-01-01", "price"].iloc[0] == 1100.0
    # On 2023-01-02, price is 1050
    assert df.loc[df["date"] == "2023-01-02", "price"].iloc[0] == 1050.0


def test_outlier_mad_and_100x_unit_error_flagging(test_config: AppConfig):
    """Test outlier MAD flagging (>5 MAD) and ~100x unit error flagging."""
    con = duckdb.connect(":memory:")
    dates = pd.date_range("2023-01-01", periods=20, freq="D")
    rows = []

    # 18 normal prices around 1000, 1 extreme 100x unit error (100,000), 1 moderate MAD outlier (3,000)
    for i, d in enumerate(dates):
        d_str = str(d.date())
        if i == 10:
            price = 100_000.0  # 100x unit error
        elif i == 15:
            price = 4_000.0    # MAD outlier
        else:
            price = 1000.0 + (i % 3) * 10.0
        rows.append(f"('UP', 'Agra', 'Agra APMC', 'Potato', 'Desi', 'FAQ', '{d_str}'::DATE, {price}, {price*0.9}, {price*1.1}, 1, 1)")

    values_str = ",\n".join(rows)
    con.execute(f"""
        CREATE TABLE stage_deduped AS
        SELECT * FROM (VALUES
            {values_str}
        ) as t(state, district, market, commodity, variety, grade, date, price, min_price, max_price, commodity_code, raw_report_count);
    """)

    metrics, comm_outliers = step5_handle_outliers(con, test_config)
    assert metrics["total_unit_errors"] >= 1
    assert metrics["total_mad_outliers"] >= 1

    # In drop mode, the outliers must be removed
    assert metrics["rows_after"] < metrics["rows_before"]


def test_flat_run_detection():
    """Test that runs of 14+ identical consecutive prices are flagged as is_flat_run=True."""
    con = duckdb.connect(":memory:")
    dates = pd.date_range("2023-01-01", periods=25, freq="D")
    rows = []

    # First 16 days constant price (1000.0) -> flat run
    # Next 9 days fluctuating -> not flat run
    for i, d in enumerate(dates):
        d_str = str(d.date())
        price = 1000.0 if i < 16 else 1000.0 + i * 20.0
        rows.append(f"('UP', 'Agra', 'Agra APMC', 'Potato', 'Desi', 'FAQ', '{d_str}'::DATE, {price}, 900.0, 1100.0, 1, 1, false, false)")

    values_str = ",\n".join(rows)
    con.execute(f"""
        CREATE TABLE stage_outlier_resolved AS
        SELECT * FROM (VALUES
            {values_str}
        ) as t(state, district, market, commodity, variety, grade, date, price, min_price, max_price, commodity_code, raw_report_count, is_outlier_mad, is_unit_error);
    """)

    metrics = step6_detect_flat_runs(con, run_length=14)
    assert metrics["flat_run_rows"] == 16
    assert metrics["affected_series"] == 1
    assert metrics["rows_after"] == 25  # All rows retained!

    df = con.execute("SELECT date, is_flat_run, run_len FROM stage_with_flat_runs ORDER BY date;").df()
    assert (df.iloc[:16]["is_flat_run"] == True).all()
    assert (df.iloc[16:]["is_flat_run"] == False).all()
