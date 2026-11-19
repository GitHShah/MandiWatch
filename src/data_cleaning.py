"""Data cleaning pipeline for Mandi Watch (Steps 1 to 6 of Section 5 of Spec).

Loads raw Agmarknet daily price records, standardizes text with a reviewable mapping,
validates dates and positive prices, deduplicates records with median conflict resolution,
detects outliers via rolling MAD and unit errors, identifies flat runs, and exports
data/interim/clean_daily_all.parquet before scope selection.
"""
from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import AppConfig, load_config

# Configure logger
logger = logging.getLogger("mandi_watch.cleaning")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def build_or_load_name_fixes(candidates_path: Path, fixes_path: Path) -> pd.DataFrame:
    """Build or load the reviewable name mapping file data/interim/name_fixes.csv.

    Auto-approves exact casing and whitespace variants (approved='yes').
    Marks all spelling/fuzzy variants as 'pending' for manual review.

    Args:
        candidates_path: Path to name_fix_candidates.csv.
        fixes_path: Target path for name_fixes.csv.

    Returns:
        DataFrame containing candidate mappings with approval status.
    """
    fixes_path.parent.mkdir(parents=True, exist_ok=True)
    if fixes_path.exists():
        logger.info(f"Loading existing name fixes mapping from {fixes_path}")
        return pd.read_csv(fixes_path)

    if not candidates_path.exists():
        logger.warning(f"Candidates file {candidates_path} not found. Creating empty name_fixes.csv")
        empty_df = pd.DataFrame(columns=["entity_type", "raw_name", "clean_name", "similarity", "issue_type", "approved"])
        empty_df.to_csv(fixes_path, index=False)
        return empty_df

    candidates_df = pd.read_csv(candidates_path)
    fixes_rows = []

    for _, row in candidates_df.iterrows():
        entity = str(row["entity_type"]).strip().lower()
        var_a = str(row["variant_a"]).strip()
        var_b = str(row["variant_b"]).strip()
        sim = float(row.get("similarity", 1.0))
        issue = str(row.get("issue_type", "unknown"))

        # Check for case/whitespace variants
        is_case_ws = (issue == "casing_or_whitespace") or (var_a.lower() == var_b.lower())

        if is_case_ws:
            # Pick canonical name: title case or standard casing
            canonical = var_a if (var_a.istitle() or len(var_a) > len(var_b)) else var_b
            non_canonical = var_b if canonical == var_a else var_a
            fixes_rows.append({
                "entity_type": entity,
                "raw_name": non_canonical,
                "clean_name": canonical,
                "similarity": sim,
                "issue_type": "casing_or_whitespace",
                "approved": "yes",  # Auto-approve exact casing/whitespace
            })
        else:
            fixes_rows.append({
                "entity_type": entity,
                "raw_name": var_b,
                "clean_name": var_a,
                "similarity": sim,
                "issue_type": issue,
                "approved": "pending",  # Require explicit human review for spelling
            })

    fixes_df = pd.DataFrame(fixes_rows).drop_duplicates(subset=["entity_type", "raw_name"])
    fixes_df.to_csv(fixes_path, index=False)
    logger.info(f"Generated {len(fixes_df)} name fix rules ({len(fixes_df[fixes_df['approved']=='yes'])} auto-approved) at {fixes_path}")
    return fixes_df


def step1_load_raw_data(con: duckdb.DuckDBPyConnection, raw_dir: Path) -> Dict[str, Any]:
    """Step 1: Load raw data into DuckDB and report schema, dtypes, date range, and unique counts.

    Args:
        con: DuckDB connection.
        raw_dir: Directory containing raw CSV files.

    Returns:
        Dict with raw data summary metrics.
    """
    logger.info("=== STEP 1: Loading raw data ===")
    csv_pattern = str(raw_dir / "*.csv").replace("\\", "/")

    # Read all CSV files into a persistent raw view
    con.execute(f"""
        CREATE OR REPLACE VIEW raw_source AS
        SELECT 
            trim(State) as state,
            trim(District) as district,
            trim(Market) as market,
            trim(Commodity) as commodity,
            trim(Variety) as variety,
            trim(Grade) as grade,
            Arrival_Date as date_raw,
            CAST(Min_Price AS DOUBLE) as min_price,
            CAST(Max_Price AS DOUBLE) as max_price,
            CAST(Modal_Price AS DOUBLE) as price,
            CAST(Commodity_Code AS BIGINT) as commodity_code
        FROM read_csv_auto('{csv_pattern}', all_varchar=false, ignore_errors=true);
    """)

    # Materialize into table for high performance
    logger.info("Materializing raw data into DuckDB table...")
    con.execute("CREATE OR REPLACE TABLE raw_stage AS SELECT * FROM raw_source;")

    stats = con.execute("""
        SELECT 
            count(*) as row_count,
            count(distinct state) as n_states,
            count(distinct district) as n_districts,
            count(distinct market) as n_markets,
            count(distinct commodity) as n_commodities,
            count(distinct variety) as n_varieties,
            count(distinct grade) as n_grades,
            min(date_raw) as min_date_str,
            max(date_raw) as max_date_str
        FROM raw_stage;
    """).df().iloc[0]

    n_rows = int(stats["row_count"])
    logger.info(f"Loaded raw dataset: {n_rows:,} rows")
    logger.info(f"Cardinality: {int(stats['n_states']):,} states, {int(stats['n_districts']):,} districts, "
                f"{int(stats['n_markets']):,} markets, {int(stats['n_commodities']):,} commodities, "
                f"{int(stats['n_varieties']):,} varieties, {int(stats['n_grades']):,} grades")

    return {
        "step": "1_load_raw",
        "rows_before": n_rows,
        "rows_after": n_rows,
        "rows_removed": 0,
        "pct_removed": 0.0,
        "n_states": int(stats["n_states"]),
        "n_districts": int(stats["n_districts"]),
        "n_markets": int(stats["n_markets"]),
        "n_commodities": int(stats["n_commodities"]),
    }


def step2_standardize_text(con: duckdb.DuckDBPyConnection, fixes_path: Path) -> Dict[str, Any]:
    """Step 2: Standardize text (whitespace, casing) and apply approved name fixes.

    Args:
        con: DuckDB connection.
        fixes_path: Path to name_fixes.csv.

    Returns:
        Dict with step metrics.
    """
    logger.info("=== STEP 2: Standardizing text and applying approved name mappings ===")
    rows_before = con.execute("SELECT count(*) FROM raw_stage;").fetchone()[0]

    # Load approved fixes into DuckDB
    con.execute("DROP TABLE IF EXISTS approved_fixes;")
    con.execute("""
        CREATE TABLE approved_fixes (
            entity_type VARCHAR,
            raw_name VARCHAR,
            clean_name VARCHAR
        );
    """)

    if fixes_path.exists():
        fixes_df = pd.read_csv(fixes_path)
        approved_only = fixes_df[fixes_df["approved"].str.lower() == "yes"][["entity_type", "raw_name", "clean_name"]].copy()
        con.execute("INSERT INTO approved_fixes SELECT * FROM approved_only;")
        logger.info(f"Loaded {len(approved_only)} approved name fixes into mapping table")

    # Standardize whitespace and apply approved mapping
    con.execute("""
        CREATE OR REPLACE TABLE stage_text_clean AS
        SELECT 
            COALESCE(sf.clean_name, regexp_replace(trim(r.state), '\\s+', ' ', 'g')) as state,
            COALESCE(df.clean_name, regexp_replace(trim(r.district), '\\s+', ' ', 'g')) as district,
            COALESCE(mf.clean_name, regexp_replace(trim(r.market), '\\s+', ' ', 'g')) as market,
            COALESCE(cf.clean_name, regexp_replace(trim(r.commodity), '\\s+', ' ', 'g')) as commodity,
            regexp_replace(trim(r.variety), '\\s+', ' ', 'g') as variety,
            regexp_replace(trim(r.grade), '\\s+', ' ', 'g') as grade,
            r.date_raw,
            r.price,
            r.min_price,
            r.max_price,
            r.commodity_code
        FROM raw_stage r
        LEFT JOIN approved_fixes sf ON sf.entity_type = 'state' AND r.state = sf.raw_name
        LEFT JOIN approved_fixes df ON df.entity_type = 'district' AND r.district = df.raw_name
        LEFT JOIN approved_fixes mf ON mf.entity_type = 'market' AND r.market = mf.raw_name
        LEFT JOIN approved_fixes cf ON cf.entity_type = 'commodity' AND r.commodity = cf.raw_name;
    """)

    rows_after = con.execute("SELECT count(*) FROM stage_text_clean;").fetchone()[0]
    logger.info(f"Standardize text: {rows_before:,} -> {rows_after:,} rows (0 removed)")

    return {
        "step": "2_standardize_text",
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_removed": rows_before - rows_after,
        "pct_removed": 0.0,
    }


def step3_parse_dates_and_prices(con: duckdb.DuckDBPyConnection) -> Dict[str, Any]:
    """Step 3: Parse dates, drop unparseable dates and non-positive prices.

    Args:
        con: DuckDB connection.

    Returns:
        Dict with step metrics.
    """
    logger.info("=== STEP 3: Parsing dates and filtering non-positive prices ===")
    rows_before = con.execute("SELECT count(*) FROM stage_text_clean;").fetchone()[0]

    # Filter out unparseable dates and non-positive prices
    con.execute("""
        CREATE OR REPLACE TABLE stage_valid_dates_prices AS
        SELECT 
            state, district, market, commodity, variety, grade,
            try_strptime(CAST(date_raw AS VARCHAR), ['%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y'])::DATE as date,
            price,
            min_price,
            max_price,
            commodity_code
        FROM stage_text_clean
        WHERE try_strptime(CAST(date_raw AS VARCHAR), ['%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y']) IS NOT NULL
          AND price IS NOT NULL
          AND price > 0;
    """)

    rows_after = con.execute("SELECT count(*) FROM stage_valid_dates_prices;").fetchone()[0]
    rows_removed = rows_before - rows_after
    pct_removed = (rows_removed / rows_before) * 100 if rows_before > 0 else 0.0

    logger.info(f"Date & price validation: {rows_before:,} -> {rows_after:,} rows (removed {rows_removed:,}, {pct_removed:.4f}%)")

    return {
        "step": "3_parse_dates_and_prices",
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_removed": rows_removed,
        "pct_removed": round(pct_removed, 4),
    }


def step4_deduplicate_records(con: duckdb.DuckDBPyConnection) -> Dict[str, Any]:
    """Step 4: Remove exact duplicates on (market, commodity, variety, grade, date).

    When duplicate rows have conflicting prices, keeps the median price and logs conflict count.

    Args:
        con: DuckDB connection.

    Returns:
        Dict with step metrics.
    """
    logger.info("=== STEP 4: Deduplicating on (market, commodity, variety, grade, date) ===")
    rows_before = con.execute("SELECT count(*) FROM stage_valid_dates_prices;").fetchone()[0]

    # Check for price conflicts before grouping
    conflict_stats = con.execute("""
        WITH dup_groups AS (
            SELECT 
                market, commodity, variety, grade, date,
                count(*) as group_size,
                min(price) as min_p,
                max(price) as max_p
            FROM stage_valid_dates_prices
            GROUP BY market, commodity, variety, grade, date
            HAVING count(*) > 1
        )
        SELECT 
            count(*) as total_duplicate_groups,
            count(CASE WHEN min_p < max_p THEN 1 END) as conflicting_price_groups,
            sum(group_size) as total_duplicate_rows
        FROM dup_groups;
    """).df().iloc[0]

    n_dup_groups = int(conflict_stats["total_duplicate_groups"])
    n_conflicts = int(conflict_stats["conflicting_price_groups"])
    n_dup_rows = int(conflict_stats["total_duplicate_rows"] or 0)
    logger.info(f"Duplicate check: {n_dup_groups:,} duplicate groups ({n_dup_rows:,} rows total), "
                f"{n_conflicts:,} had differing prices resolved via median.")

    # Fast deduplication with median price resolution for conflicting duplicates:
    # 99.98% of rows have no duplicate key. Using an anti-join for unique rows and
    # running median aggregation ONLY on the duplicate groups avoids building 48M quantile states.
    con.execute("""
        CREATE OR REPLACE TABLE dup_keys AS
        SELECT market, commodity, variety, grade, date
        FROM stage_valid_dates_prices
        GROUP BY market, commodity, variety, grade, date
        HAVING count(*) > 1;

        CREATE OR REPLACE TABLE stage_deduped AS
        -- 1. All rows that have NO duplicate
        SELECT 
            v.state, v.district, v.market, v.commodity, v.variety, v.grade, v.date,
            v.price, v.min_price, v.max_price, v.commodity_code,
            1::BIGINT as raw_report_count
        FROM stage_valid_dates_prices v
        ANTI JOIN dup_keys d
          ON v.market = d.market
         AND v.commodity = d.commodity
         AND v.variety = d.variety
         AND v.grade = d.grade
         AND v.date = d.date

        UNION ALL

        -- 2. Only the duplicate rows resolved via median
        SELECT 
            arbitrary(v.state) as state,
            arbitrary(v.district) as district,
            v.market, v.commodity, v.variety, v.grade, v.date,
            median(v.price) as price,
            min(v.min_price) as min_price,
            max(v.max_price) as max_price,
            arbitrary(v.commodity_code) as commodity_code,
            count(*)::BIGINT as raw_report_count
        FROM stage_valid_dates_prices v
        SEMI JOIN dup_keys d
          ON v.market = d.market
         AND v.commodity = d.commodity
         AND v.variety = d.variety
         AND v.grade = d.grade
         AND v.date = d.date
        GROUP BY v.market, v.commodity, v.variety, v.grade, v.date;

        DROP TABLE IF EXISTS dup_keys;
    """)

    rows_after = con.execute("SELECT count(*) FROM stage_deduped;").fetchone()[0]
    rows_removed = rows_before - rows_after
    pct_removed = (rows_removed / rows_before) * 100 if rows_before > 0 else 0.0

    logger.info(f"Deduplication: {rows_before:,} -> {rows_after:,} rows (removed {rows_removed:,}, {pct_removed:.4f}%)")

    return {
        "step": "4_deduplicate_records",
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_removed": rows_removed,
        "pct_removed": round(pct_removed, 4),
        "conflicting_groups": n_conflicts,
    }


def step5_handle_outliers(con: duckdb.DuckDBPyConnection, config: AppConfig) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Step 5: Flag prices > 5 MAD from centered rolling median and likely unit errors (~10x/100x).

    Winsorizes or drops according to config.cleaning.outlier_action.

    Args:
        con: DuckDB connection.
        config: Application configuration.

    Returns:
        Tuple of (step metrics dict, per-commodity outlier breakdown DataFrame).
    """
    logger.info("=== STEP 5: Outlier handling (Rolling MAD & Unit Errors) ===")
    rows_before = con.execute("SELECT count(*) FROM stage_deduped;").fetchone()[0]
    mad_mult = config.cleaning.mad_multiplier
    action = config.cleaning.outlier_action.lower()
    half_window = config.cleaning.rolling_window // 2

    # Compute series median, rolling median, and rolling MAD
    logger.info("Computing rolling medians and MAD per series...")
    con.execute(f"""
        CREATE OR REPLACE TABLE stage_with_rolling_stats AS
        SELECT 
            state, district, market, commodity, variety, grade, date, price, min_price, max_price, commodity_code, raw_report_count,
            MEDIAN(price) OVER (PARTITION BY market, commodity) as series_median,
            MEDIAN(price) OVER (PARTITION BY market, commodity ORDER BY date ROWS BETWEEN {half_window} PRECEDING AND {half_window} FOLLOWING) as rolling_median
        FROM stage_deduped;
    """)

    # Compute rolling MAD = median(abs(price - rolling_median))
    con.execute(f"""
        CREATE OR REPLACE TABLE stage_flagged_outliers AS
        WITH diffs AS (
            SELECT 
                *,
                abs(price - rolling_median) as abs_diff
            FROM stage_with_rolling_stats
        ),
        with_mad AS (
            SELECT 
                *,
                greatest(MEDIAN(abs_diff) OVER (PARTITION BY market, commodity ORDER BY date ROWS BETWEEN {half_window} PRECEDING AND {half_window} FOLLOWING), 1.0) as rolling_mad
            FROM diffs
        )
        SELECT 
            state, district, market, commodity, variety, grade, date, price, min_price, max_price, commodity_code, raw_report_count,
            series_median, rolling_median, rolling_mad,
            CASE 
                WHEN abs(price - rolling_median) > ({mad_mult} * rolling_mad) THEN true 
                ELSE false 
            END as is_outlier_mad,
            CASE 
                WHEN (price >= 8.0 * series_median) OR (price <= 0.125 * series_median) THEN true 
                ELSE false 
            END as is_unit_error
        FROM with_mad;
    """)

    # Per-commodity breakdown of flagged outliers
    comm_outlier_df = con.execute("""
        SELECT 
            commodity,
            count(*) as total_reports,
            count(CASE WHEN is_outlier_mad THEN 1 END) as mad_outliers,
            count(CASE WHEN is_unit_error THEN 1 END) as unit_errors,
            count(CASE WHEN is_outlier_mad OR is_unit_error THEN 1 END) as any_outlier
        FROM stage_flagged_outliers
        GROUP BY commodity
        ORDER BY any_outlier DESC;
    """).df()

    total_mad_outliers = int(comm_outlier_df["mad_outliers"].sum())
    total_unit_errors = int(comm_outlier_df["unit_errors"].sum())
    total_any_outliers = int(comm_outlier_df["any_outlier"].sum())

    logger.info(f"Outliers flagged: {total_mad_outliers:,} (> {mad_mult} MAD), "
                f"{total_unit_errors:,} unit errors, {total_any_outliers:,} total outlier flags.")

    # Apply configured action: drop, winsorize, or flag_only
    if action == "drop":
        logger.info("Applying outlier action: DROP")
        con.execute("""
            CREATE OR REPLACE TABLE stage_outlier_resolved AS
            SELECT * FROM stage_flagged_outliers
            WHERE NOT (is_outlier_mad OR is_unit_error);
        """)
    elif action == "winsorize":
        logger.info("Applying outlier action: WINSORIZE")
        con.execute(f"""
            CREATE OR REPLACE TABLE stage_outlier_resolved AS
            SELECT 
                state, district, market, commodity, variety, grade, date,
                CASE 
                    WHEN is_outlier_mad OR is_unit_error THEN 
                        least(greatest(price, rolling_median - ({mad_mult} * rolling_mad)), rolling_median + ({mad_mult} * rolling_mad))
                    ELSE price 
                END as price,
                min_price, max_price, commodity_code, raw_report_count,
                series_median, rolling_median, rolling_mad,
                is_outlier_mad, is_unit_error
            FROM stage_flagged_outliers;
        """)
    else:  # flag_only
        logger.info("Applying outlier action: FLAG_ONLY (keeping all rows)")
        con.execute("CREATE OR REPLACE TABLE stage_outlier_resolved AS SELECT * FROM stage_flagged_outliers;")

    rows_after = con.execute("SELECT count(*) FROM stage_outlier_resolved;").fetchone()[0]
    rows_removed = rows_before - rows_after
    pct_removed = (rows_removed / rows_before) * 100 if rows_before > 0 else 0.0

    logger.info(f"Outlier action ({action}): {rows_before:,} -> {rows_after:,} rows (removed {rows_removed:,}, {pct_removed:.4f}%)")

    metrics = {
        "step": f"5_outliers_{action}",
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_removed": rows_removed,
        "pct_removed": round(pct_removed, 4),
        "total_mad_outliers": total_mad_outliers,
        "total_unit_errors": total_unit_errors,
        "total_any_outliers": total_any_outliers,
    }
    return metrics, comm_outlier_df


def step6_detect_flat_runs(con: duckdb.DuckDBPyConnection, run_length: int = 14) -> Dict[str, Any]:
    """Step 6: Detect flat runs of 14+ identical consecutive price reports per series.

    Marks is_flat_run = True and retains all rows.

    Args:
        con: DuckDB connection.
        run_length: Minimum consecutive reports to constitute a flat run (default 14).

    Returns:
        Dict with step metrics.
    """
    logger.info(f"=== STEP 6: Detecting flat runs (>={run_length} consecutive identical prices) ===")
    rows_before = con.execute("SELECT count(*) FROM stage_outlier_resolved;").fetchone()[0]

    con.execute(f"""
        CREATE OR REPLACE TABLE stage_with_flat_runs AS
        WITH diffs AS (
            SELECT 
                *,
                CASE 
                    WHEN price = lag(price, 1) OVER (PARTITION BY market, commodity ORDER BY date) THEN 0 
                    ELSE 1 
                END as is_new_run
            FROM stage_outlier_resolved
        ),
        run_groups AS (
            SELECT 
                *,
                sum(is_new_run) OVER (PARTITION BY market, commodity ORDER BY date) as run_id
            FROM diffs
        ),
        run_counts AS (
            SELECT 
                *,
                count(*) OVER (PARTITION BY market, commodity, run_id) as run_len
            FROM run_groups
        )
        SELECT 
            state, district, market, commodity, variety, grade, date, price, min_price, max_price, commodity_code, raw_report_count,
            is_outlier_mad, is_unit_error,
            CASE WHEN run_len >= {run_length} THEN true ELSE false END as is_flat_run,
            run_len
        FROM run_counts;
    """)

    flat_stats = con.execute("""
        SELECT 
            count(CASE WHEN is_flat_run THEN 1 END) as flat_run_rows,
            count(distinct CASE WHEN is_flat_run THEN market || '_' || commodity END) as affected_series
        FROM stage_with_flat_runs;
    """).df().iloc[0]

    flat_rows = int(flat_stats["flat_run_rows"])
    n_affected = int(flat_stats["affected_series"])
    rows_after = con.execute("SELECT count(*) FROM stage_with_flat_runs;").fetchone()[0]

    logger.info(f"Flat runs detected: {flat_rows:,} rows across {n_affected:,} series flagged as is_flat_run=True (all retained).")

    return {
        "step": "6_flat_run_detection",
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_removed": 0,
        "pct_removed": 0.0,
        "flat_run_rows": flat_rows,
        "affected_series": n_affected,
    }


def step7_save_clean_daily(con: duckdb.DuckDBPyConnection, output_path: Path) -> Dict[str, Any]:
    """Step 7: Save clean_daily_all.parquet before scope selection.

    Args:
        con: DuckDB connection.
        output_path: Path to write clean_daily_all.parquet.

    Returns:
        Dict with export metrics.
    """
    logger.info(f"=== STEP 7: Exporting clean daily dataset to {output_path} ===")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_str = str(output_path).replace("\\", "/")

    t0 = time.time()
    con.execute(f"""
        COPY (
            SELECT 
                state, district, market, commodity, variety, grade,
                date,
                price,
                min_price,
                max_price,
                commodity_code,
                raw_report_count,
                is_outlier_mad,
                is_unit_error,
                is_flat_run,
                run_len
            FROM stage_with_flat_runs
            ORDER BY market, commodity, date
        ) TO '{out_str}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """)

    elapsed = time.time() - t0
    final_rows = con.execute("SELECT count(*) FROM stage_with_flat_runs;").fetchone()[0]
    file_bytes = output_path.stat().st_size
    logger.info(f"Export complete in {elapsed:.2f}s: {final_rows:,} rows, {file_bytes / (1024**2):.1f} MB ({file_bytes:,} bytes)")

    return {
        "step": "7_save_clean_daily_all",
        "rows_before": final_rows,
        "rows_after": final_rows,
        "rows_removed": 0,
        "pct_removed": 0.0,
        "file_size_mb": round(file_bytes / (1024**2), 2),
    }


def step8_select_scope(
    con: duckdb.DuckDBPyConnection,
    config: AppConfig,
    source_table: str = "stage_with_flat_runs"
) -> Tuple[List[str], List[str], pd.DataFrame]:
    """Step 8: Automated scope selection for commodities and states.

    Ranks commodities and states by:
    (a) number of distinct reporting weeks, and
    (b) number of markets with at least 104 weeks of history.
    Picks top N commodities (default 8) and top M states (default 3) unless overrides exist.

    Args:
        con: DuckDB connection.
        config: Application configuration.
        source_table: Source table name in DuckDB.

    Returns:
        Tuple of (selected_commodities, selected_states, ranking_df).
    """
    logger.info("=== STEP 8: Scope Selection by script ===")

    # 1. Commodity ranking
    comm_ranking = con.execute(f"""
        WITH comm_market_weeks AS (
            SELECT 
                commodity,
                market,
                count(distinct (date_part('isoyear', date), date_part('week', date))) as n_weeks
            FROM {source_table}
            GROUP BY commodity, market
        ),
        comm_agg AS (
            SELECT 
                f.commodity as name,
                'commodity' as entity_type,
                count(distinct (date_part('isoyear', f.date), date_part('week', f.date))) as distinct_weeks,
                count(distinct CASE WHEN cmw.n_weeks >= {config.cleaning.min_weeks} THEN cmw.market END) as markets_with_104_weeks,
                count(*) as total_reports
            FROM {source_table} f
            LEFT JOIN comm_market_weeks cmw ON f.commodity = cmw.commodity AND f.market = cmw.market
            GROUP BY f.commodity
        )
        SELECT 
            name, entity_type, distinct_weeks, markets_with_104_weeks, total_reports,
            row_number() OVER (ORDER BY markets_with_104_weeks DESC, distinct_weeks DESC, total_reports DESC) as rank
        FROM comm_agg
        ORDER BY rank;
    """).df()

    # 2. State ranking
    state_ranking = con.execute(f"""
        WITH state_market_weeks AS (
            SELECT 
                state,
                market,
                count(distinct (date_part('isoyear', date), date_part('week', date))) as n_weeks
            FROM {source_table}
            GROUP BY state, market
        ),
        state_agg AS (
            SELECT 
                f.state as name,
                'state' as entity_type,
                count(distinct (date_part('isoyear', f.date), date_part('week', f.date))) as distinct_weeks,
                count(distinct CASE WHEN smw.n_weeks >= {config.cleaning.min_weeks} THEN smw.market END) as markets_with_104_weeks,
                count(*) as total_reports
            FROM {source_table} f
            LEFT JOIN state_market_weeks smw ON f.state = smw.state AND f.market = smw.market
            GROUP BY f.state
        )
        SELECT 
            name, entity_type, distinct_weeks, markets_with_104_weeks, total_reports,
            row_number() OVER (ORDER BY markets_with_104_weeks DESC, distinct_weeks DESC, total_reports DESC) as rank
        FROM state_agg
        ORDER BY rank;
    """).df()

    # Apply selection
    if config.scope.manual_commodities:
        selected_commodities = [c.strip() for c in config.scope.manual_commodities]
        logger.info(f"Using manual commodities override: {selected_commodities}")
    else:
        selected_commodities = comm_ranking.head(config.scope.n_commodities)["name"].tolist()
        logger.info(f"Auto-selected top {config.scope.n_commodities} commodities: {selected_commodities}")

    if config.scope.manual_states:
        selected_states = [s.strip() for s in config.scope.manual_states]
        logger.info(f"Using manual states override: {selected_states}")
    else:
        selected_states = state_ranking.head(config.scope.n_states)["name"].tolist()
        logger.info(f"Auto-selected top {config.scope.n_states} states: {selected_states}")

    comm_ranking["in_scope"] = comm_ranking["name"].isin(selected_commodities)
    state_ranking["in_scope"] = state_ranking["name"].isin(selected_states)

    combined_ranking = pd.concat([comm_ranking, state_ranking], ignore_index=True)
    ranking_path = config.paths.tables_dir / "scope_ranking.csv"
    ranking_path.parent.mkdir(parents=True, exist_ok=True)
    combined_ranking.to_csv(ranking_path, index=False)
    logger.info(f"Saved scope rankings table to {ranking_path}")

    return selected_commodities, selected_states, combined_ranking


def step9_filter_single_variety_grade(
    con: duckdb.DuckDBPyConnection,
    in_scope_commodities: List[str],
    in_scope_states: List[str],
    config: AppConfig,
    output_path: Path,
    source_table: str = "stage_with_flat_runs"
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Step 9: Keep single most frequent variety and grade per in-scope commodity.

    Filters to in-scope commodities and states, retains mode variety and mode grade,
    and exports data/interim/clean_daily.parquet.

    Args:
        con: DuckDB connection.
        in_scope_commodities: List of commodities in scope.
        in_scope_states: List of states in scope.
        config: Application configuration.
        output_path: Target path for clean_daily.parquet.
        source_table: Source DuckDB table name.

    Returns:
        Tuple of (step metrics dict, per-commodity breakdown DataFrame).
    """
    logger.info("=== STEP 9: Filtering to in-scope entities and single variety/grade mode ===")
    comm_list_str = ", ".join(f"'{c}'" for c in in_scope_commodities)
    state_list_str = ", ".join(f"'{s}'" for s in in_scope_states)

    # 1. Filter to in-scope commodities and states
    con.execute(f"""
        CREATE OR REPLACE TABLE stage_scoped_raw AS
        SELECT * FROM {source_table}
        WHERE commodity IN ({comm_list_str})
          AND state IN ({state_list_str});
    """)
    rows_scoped = con.execute("SELECT count(*) FROM stage_scoped_raw;").fetchone()[0]
    logger.info(f"In-scope state & commodity filter: {rows_scoped:,} rows")

    # 2. Determine mode variety and mode grade per commodity
    con.execute("""
        CREATE OR REPLACE TABLE commodity_modes AS
        WITH var_counts AS (
            SELECT commodity, variety, count(*) as cnt,
                   row_number() OVER (PARTITION BY commodity ORDER BY count(*) DESC, variety ASC) as rn
            FROM stage_scoped_raw
            GROUP BY commodity, variety
        ),
        grade_counts AS (
            SELECT commodity, grade, count(*) as cnt,
                   row_number() OVER (PARTITION BY commodity ORDER BY count(*) DESC, grade ASC) as rn
            FROM stage_scoped_raw
            GROUP BY commodity, grade
        )
        SELECT 
            v.commodity,
            v.variety as mode_variety,
            g.grade as mode_grade
        FROM (SELECT commodity, variety FROM var_counts WHERE rn = 1) v
        JOIN (SELECT commodity, grade FROM grade_counts WHERE rn = 1) g
          ON v.commodity = g.commodity;
    """)

    modes_df = con.execute("SELECT * FROM commodity_modes;").df()
    logger.info(f"Identified single mode variety and grade for {len(modes_df)} commodities:\n{modes_df.to_string(index=False)}")

    # 3. Filter to mode variety and grade
    con.execute("""
        CREATE OR REPLACE TABLE stage_clean_daily_scoped AS
        SELECT s.*
        FROM stage_scoped_raw s
        JOIN commodity_modes m
          ON s.commodity = m.commodity
         AND s.variety = m.mode_variety
         AND s.grade = m.mode_grade;
    """)

    rows_kept = con.execute("SELECT count(*) FROM stage_clean_daily_scoped;").fetchone()[0]
    rows_dropped = rows_scoped - rows_kept
    drop_pct = (rows_dropped / rows_scoped * 100) if rows_scoped > 0 else 0.0
    logger.info(f"Variety & grade selection: {rows_scoped:,} -> {rows_kept:,} rows (dropped {rows_dropped:,}, {drop_pct:.2f}%)")

    # Per-commodity breakdown
    breakdown_df = con.execute("""
        WITH scoped_tot AS (
            SELECT commodity, count(*) as total_rows
            FROM stage_scoped_raw
            GROUP BY commodity
        ),
        kept_tot AS (
            SELECT commodity, variety, grade, count(*) as kept_rows
            FROM stage_clean_daily_scoped
            GROUP BY commodity, variety, grade
        )
        SELECT 
            s.commodity,
            k.variety as mode_variety,
            k.grade as mode_grade,
            s.total_rows as raw_scoped_rows,
            k.kept_rows,
            (s.total_rows - k.kept_rows) as dropped_rows,
            round(((s.total_rows - k.kept_rows) * 100.0 / s.total_rows), 2) as dropped_share_pct
        FROM scoped_tot s
        LEFT JOIN kept_tot k ON s.commodity = k.commodity
        ORDER BY s.total_rows DESC;
    """).df()

    # Export clean_daily.parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_str = str(output_path).replace("\\", "/")
    con.execute(f"""
        COPY (
            SELECT 
                state, district, market, commodity, variety, grade,
                date, price, min_price, max_price, commodity_code,
                raw_report_count, is_outlier_mad, is_unit_error, is_flat_run, run_len
            FROM stage_clean_daily_scoped
            ORDER BY market, commodity, date
        ) TO '{out_str}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """)
    file_bytes = output_path.stat().st_size
    logger.info(f"Saved {rows_kept:,} rows to {output_path} ({file_bytes / (1024**2):.2f} MB)")

    con.execute("DROP TABLE IF EXISTS stage_scoped_raw;")

    metrics = {
        "step": "9_filter_single_variety_grade",
        "rows_before": rows_scoped,
        "rows_after": rows_kept,
        "rows_removed": rows_dropped,
        "pct_removed": round(drop_pct, 4),
        "file_size_mb": round(file_bytes / (1024**2), 2),
    }
    return metrics, breakdown_df


def aggregate_daily_to_iso_weeks(
    data: Union[duckdb.DuckDBPyConnection, pd.DataFrame]
) -> pd.DataFrame:
    """Step 10: Aggregate daily price records to ISO weeks per series (market + commodity).

    Handles ISO week year boundaries (week 53 and Jan 1) correctly.
    Calculates median price, n_reports, min, max, std (sample std if n_reports > 1, else NaN),
    and flat-run share.

    Args:
        data: DuckDB connection with table stage_clean_daily_scoped, or a pandas DataFrame.

    Returns:
        DataFrame aggregated to ISO weeks per series.
    """
    logger.info("=== STEP 10: Aggregating daily records to ISO weeks ===")
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        if not np.issubdtype(df["date"].dtype, np.datetime64):
            df["date"] = pd.to_datetime(df["date"])

        cal = df["date"].dt.isocalendar()
        df["iso_year"] = cal.year.astype(int)
        df["iso_week"] = cal.week.astype(int)
        df["week_start"] = (df["date"] - pd.to_timedelta(df["date"].dt.weekday, unit="D")).dt.date

        if "is_flat_run" not in df.columns:
            df["is_flat_run"] = False

        group_cols = ["state", "district", "market", "commodity", "variety", "grade", "iso_year", "iso_week", "week_start"]
        extra_cols = [c for c in ["commodity_code"] if c in df.columns]

        grouped = df.groupby(group_cols, as_index=False).agg(
            median_price=("price", "median"),
            n_reports=("price", "count"),
            min_price=("price", "min"),
            max_price=("price", "max"),
            std_price=("price", lambda x: x.std(ddof=1) if len(x) > 1 else np.nan),
            flat_run_share=("is_flat_run", lambda x: float(x.mean()))
        )
        if extra_cols:
            first_vals = df.groupby(["market", "commodity"])[extra_cols].first().reset_index()
            grouped = grouped.merge(first_vals, on=["market", "commodity"], how="left")

        return grouped.sort_values(["market", "commodity", "week_start"]).reset_index(drop=True)

    con = data
    con.execute("""
        CREATE OR REPLACE TABLE stage_weekly AS
        SELECT 
            state,
            district,
            market,
            commodity,
            variety,
            grade,
            commodity_code,
            date_part('isoyear', date)::INT as iso_year,
            date_part('week', date)::INT as iso_week,
            date_trunc('week', date)::DATE as week_start,
            MEDIAN(price) as median_price,
            COUNT(*)::INT as n_reports,
            MIN(price) as min_price,
            MAX(price) as max_price,
            STDDEV_SAMP(price) as std_price,
            AVG(CASE WHEN is_flat_run THEN 1.0 ELSE 0.0 END) as flat_run_share
        FROM stage_clean_daily_scoped
        GROUP BY state, district, market, commodity, variety, grade, commodity_code,
                 date_part('isoyear', date), date_part('week', date), date_trunc('week', date);
    """)

    weekly_df = con.execute("SELECT * FROM stage_weekly ORDER BY market, commodity, week_start;").df()
    logger.info(f"Aggregated daily to ISO weeks: {len(weekly_df):,} weekly records")
    return weekly_df


def filter_series_quality(
    weekly_df: pd.DataFrame,
    min_weeks: int = 104,
    max_missing_share: float = 0.40
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Step 11: Series quality filter.

    Retains series (market + commodity) with:
    - at least min_weeks (default 104) of history, and
    - missing share <= max_missing_share (default 40%), measured between first and last report.

    Args:
        weekly_df: Aggregated weekly DataFrame.
        min_weeks: Minimum reporting weeks required.
        max_missing_share: Maximum allowed share of missing weeks.

    Returns:
        Tuple of (kept_series_df, dropped_series_df, filtered_weekly_df).
    """
    logger.info(f"=== STEP 11: Filtering series quality (min_weeks>={min_weeks}, max_missing<={max_missing_share:.0%}) ===")
    series_records = []

    for (mkt, comm), grp in weekly_df.groupby(["market", "commodity"]):
        w_starts = pd.to_datetime(grp["week_start"]).dt.date
        min_w = w_starts.min()
        max_w = w_starts.max()
        actual_w = int(len(grp["week_start"].unique()))
        total_possible_w = ((max_w - min_w).days // 7) + 1
        missing_w = max(0, total_possible_w - actual_w)
        missing_share = missing_w / total_possible_w if total_possible_w > 0 else 0.0

        is_kept = (actual_w >= min_weeks) and (missing_share <= max_missing_share)
        if is_kept:
            reason = "kept"
        elif actual_w < min_weeks:
            reason = f"too_few_weeks ({actual_w} < {min_weeks})"
        else:
            reason = f"high_missing_share ({missing_share:.1%} > {max_missing_share:.0%})"

        series_records.append({
            "market": mkt,
            "commodity": comm,
            "series_id": f"{mkt}__{comm}",
            "min_week": min_w,
            "max_week": max_w,
            "actual_weeks": actual_w,
            "total_possible_weeks": total_possible_w,
            "missing_weeks": missing_w,
            "missing_share": round(missing_share, 4),
            "reason": reason,
            "is_kept": is_kept,
        })

    eval_df = pd.DataFrame(series_records)
    kept_df = eval_df[eval_df["is_kept"]].drop(columns=["is_kept"]).reset_index(drop=True)
    dropped_df = eval_df[~eval_df["is_kept"]].drop(columns=["is_kept"]).reset_index(drop=True)

    kept_keys = set(zip(kept_df["market"], kept_df["commodity"]))
    filtered_weekly = weekly_df[
        weekly_df.apply(lambda r: (r["market"], r["commodity"]) in kept_keys, axis=1)
    ].copy().reset_index(drop=True)

    logger.info(f"Series filter results: {len(kept_df):,} kept, {len(dropped_df):,} dropped out of {len(eval_df):,} series.")
    logger.info(f"Weekly rows in kept series: {len(filtered_weekly):,} (from {len(weekly_df):,})")

    return kept_df, dropped_df, filtered_weekly


def interpolate_short_gaps(weekly_df: pd.DataFrame, max_gap: int = 2) -> pd.DataFrame:
    """Linearly interpolate log price for short gaps (<= max_gap weeks) for lag features only.

    Do NOT impute long gaps (> max_gap weeks). Mark interpolated rows with is_interpolated=True.

    Args:
        weekly_df: Weekly DataFrame.
        max_gap: Maximum contiguous missing weeks to interpolate (default 2).

    Returns:
        DataFrame with full weekly calendar grid per series and log_price_interpolated column.
    """
    df = weekly_df.copy()
    if "log_price" not in df.columns:
        df["log_price"] = np.log(df["median_price"].astype(float))

    series_dfs = []

    for (mkt, comm), grp in df.groupby(["market", "commodity"]):
        grp = grp.sort_values("week_start").copy()
        grp["week_dt"] = pd.to_datetime(grp["week_start"])
        min_date = grp["week_dt"].min()
        max_date = grp["week_dt"].max()

        full_mondays = pd.date_range(min_date, max_date, freq="W-MON")
        grid = pd.DataFrame({"week_dt": full_mondays})
        grid["week_start"] = grid["week_dt"].dt.date
        grid["iso_year"] = grid["week_dt"].dt.isocalendar().year.astype(int)
        grid["iso_week"] = grid["week_dt"].dt.isocalendar().week.astype(int)

        drop_cols = [c for c in ["week_start", "iso_year", "iso_week"] if c in grp.columns]
        merged = grid.merge(grp.drop(columns=drop_cols, errors="ignore"), on="week_dt", how="left")
        merged["market"] = mkt
        merged["commodity"] = comm

        for col in ["state", "district", "variety", "grade", "commodity_code"]:
            if col in grp.columns:
                merged[col] = grp[col].iloc[0]

        # Interpolate short gaps
        s = merged["log_price"]
        is_na = s.isna()
        s_interp = s.interpolate(method="linear")

        merged["log_price_interpolated"] = s.copy()
        merged["is_interpolated"] = False

        if is_na.any():
            block_id = (~is_na).cumsum()
            for _, block_indices in s[is_na].groupby(block_id[is_na]):
                if len(block_indices) <= max_gap:
                    merged.loc[block_indices.index, "log_price_interpolated"] = s_interp.loc[block_indices.index]
                    merged.loc[block_indices.index, "is_interpolated"] = True

        merged = merged.drop(columns=["week_dt"])
        series_dfs.append(merged)

    return pd.concat(series_dfs, ignore_index=True) if series_dfs else pd.DataFrame()


def step12_create_weekly_dataset(
    filtered_weekly_df: pd.DataFrame,
    kept_series_df: pd.DataFrame,
    config: AppConfig,
    output_path: Path
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Step 12: Build and export data/processed/weekly.parquet with short gap interpolation for lag features.

    Args:
        filtered_weekly_df: Weekly DataFrame filtered to kept series.
        kept_series_df: DataFrame of kept series metadata.
        config: Application configuration.
        output_path: Target path for weekly.parquet.

    Returns:
        Tuple of (processed_weekly_df, export_metrics).
    """
    logger.info("=== STEP 12: Creating data/processed/weekly.parquet ===")
    max_gap = config.cleaning.max_interpolation_gap
    processed_df = interpolate_short_gaps(filtered_weekly_df, max_gap=max_gap)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_parquet(output_path, compression="zstd", index=False)
    file_bytes = output_path.stat().st_size

    n_reported = len(processed_df[~processed_df["log_price"].isna()])
    n_interpolated = int(processed_df["is_interpolated"].sum())
    n_total = len(processed_df)

    logger.info(f"Saved {n_total:,} weekly rows ({n_reported:,} reported, {n_interpolated:,} interpolated <= {max_gap}w) "
                f"to {output_path} ({file_bytes / (1024**2):.2f} MB)")

    metrics = {
        "step": "12_save_weekly_processed",
        "total_weekly_grid_rows": n_total,
        "reported_rows": n_reported,
        "interpolated_rows": n_interpolated,
        "file_size_mb": round(file_bytes / (1024**2), 2),
    }
    return processed_df, metrics


def step13_generate_coverage_visuals(
    weekly_df: pd.DataFrame,
    figures_dir: Path
) -> List[Path]:
    """Step 13: Generate coverage visuals and heatmaps.

    Saves:
    1. reports/figures/market_year_reporting_heatmap.png
    2. reports/figures/series_count_by_year.png

    Args:
        weekly_df: Processed weekly DataFrame (or reported weekly DataFrame).
        figures_dir: Directory to save figure plots.

    Returns:
        List of generated figure file paths.
    """
    logger.info(f"=== STEP 13: Generating coverage visuals in {figures_dir} ===")
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    # Filter to reported rows for coverage analysis
    active_df = weekly_df[~weekly_df["median_price"].isna()].copy()
    if active_df.empty:
        active_df = weekly_df.copy()

    # 1. Heatmap: Reporting weeks by market and year
    pivot = active_df.groupby(["market", "iso_year"])["iso_week"].nunique().unstack(fill_value=0)
    plt.figure(figsize=(14, max(8, len(pivot) * 0.22)))
    sns.heatmap(pivot, cmap="YlGnBu", cbar_kws={"label": "Reporting Weeks / Year"}, annot=False)
    plt.title("Reporting Weeks by Market and Year (2015 - 2026)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("ISO Year", fontsize=11)
    plt.ylabel("Market", fontsize=11)
    plt.tight_layout()
    heatmap_path = figures_dir / "market_year_reporting_heatmap.png"
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    generated.append(heatmap_path)
    logger.info(f"Saved market-year reporting heatmap to {heatmap_path}")

    # 2. Per-commodity count of series kept over time
    plt.figure(figsize=(12, 6))
    series_yr = active_df.groupby(["commodity", "iso_year"])["market"].nunique().reset_index()
    sns.lineplot(data=series_yr, x="iso_year", y="market", hue="commodity", marker="o", linewidth=2)
    plt.title("Active Series Count per Commodity by Year (2015 - 2026)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("ISO Year", fontsize=11)
    plt.ylabel("Number of Active Markets Reporting", fontsize=11)
    plt.xticks(sorted(active_df["iso_year"].unique()))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    chart_path = figures_dir / "series_count_by_year.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    generated.append(chart_path)
    logger.info(f"Saved active series by year chart to {chart_path}")

    return generated


def run_cleaning(config: Optional[AppConfig] = None) -> Tuple[Path, pd.DataFrame]:
    """Execute the complete data cleaning and scope selection pipeline (Steps 1-13).

    Args:
        config: Optional AppConfig. Loaded from config.yaml if None.

    Returns:
        Tuple of (clean_daily_all.parquet Path, cleaning summary DataFrame).
    """
    t_start = time.time()
    if config is None:
        config = load_config()

    logger.info("Initializing Mandi Watch data cleaning & scope selection pipeline...")
    interim_dir = config.paths.interim_dir
    processed_dir = config.paths.processed_dir
    tables_dir = config.paths.tables_dir
    figures_dir = config.paths.figures_dir

    candidates_path = interim_dir / "name_fix_candidates.csv"
    fixes_path = interim_dir / "name_fixes.csv"
    output_parquet_all = interim_dir / "clean_daily_all.parquet"
    output_parquet_scoped = interim_dir / "clean_daily.parquet"
    output_weekly = processed_dir / "weekly.parquet"

    summary_path = tables_dir / "cleaning_summary.csv"
    outliers_path = tables_dir / "commodity_outliers_breakdown.csv"
    variety_path = tables_dir / "variety_grade_breakdown.csv"
    kept_path = tables_dir / "series_kept.csv"
    dropped_path = tables_dir / "series_dropped.csv"

    # 1. Build or verify name fixes review table
    build_or_load_name_fixes(candidates_path, fixes_path)

    # 2. Connect to DuckDB with disk-backed cache database
    db_file = interim_dir / "_cleaning_temp.duckdb"
    if db_file.exists():
        try:
            db_file.unlink()
        except Exception:
            pass

    con = duckdb.connect(str(db_file))
    con.execute("PRAGMA threads=4;")
    con.execute("PRAGMA memory_limit='4GB';")
    con.execute("PRAGMA preserve_insertion_order=false;")

    summary_rows = []

    try:
        # Step 1: Load raw data
        s1 = step1_load_raw_data(con, config.paths.raw_dir)
        summary_rows.append(s1)

        # Step 2: Standardize text
        s2 = step2_standardize_text(con, fixes_path)
        summary_rows.append(s2)
        con.execute("DROP TABLE IF EXISTS raw_stage;")

        # Step 3: Parse dates and positive prices
        s3 = step3_parse_dates_and_prices(con)
        summary_rows.append(s3)
        con.execute("DROP TABLE IF EXISTS stage_text_clean;")

        # Step 4: Deduplicate
        s4 = step4_deduplicate_records(con)
        summary_rows.append(s4)
        con.execute("DROP TABLE IF EXISTS stage_valid_dates_prices;")

        # Step 5: Outliers
        s5, comm_outliers = step5_handle_outliers(con, config)
        summary_rows.append(s5)
        comm_outliers.to_csv(outliers_path, index=False)
        con.execute("DROP TABLE IF EXISTS stage_deduped;")
        con.execute("DROP TABLE IF EXISTS stage_with_rolling_stats;")
        con.execute("DROP TABLE IF EXISTS stage_flagged_outliers;")

        # Step 6: Flat runs
        s6 = step6_detect_flat_runs(con, config.cleaning.flat_run_length)
        summary_rows.append(s6)
        con.execute("DROP TABLE IF EXISTS stage_outlier_resolved;")

        # Step 7: Export clean_daily_all.parquet (unscoped)
        s7 = step7_save_clean_daily(con, output_parquet_all)
        summary_rows.append(s7)

        # Step 8: Automated Scope Selection
        selected_comms, selected_states, ranking_df = step8_select_scope(con, config)

        # Step 9: Single variety/grade mode selection -> clean_daily.parquet
        s9, variety_breakdown = step9_filter_single_variety_grade(
            con, selected_comms, selected_states, config, output_parquet_scoped
        )
        summary_rows.append(s9)
        variety_breakdown.to_csv(variety_path, index=False)
        con.execute("DROP TABLE IF EXISTS stage_with_flat_runs;")

        # Step 10: Weekly aggregation
        weekly_df = aggregate_daily_to_iso_weeks(con)
        con.execute("DROP TABLE IF EXISTS stage_clean_daily_scoped;")
        con.execute("DROP TABLE IF EXISTS stage_weekly;")

        # Step 11: Series quality filter
        kept_series_df, dropped_series_df, filtered_weekly = filter_series_quality(
            weekly_df,
            min_weeks=config.cleaning.min_weeks,
            max_missing_share=config.cleaning.max_missing_share
        )
        kept_series_df.to_csv(kept_path, index=False)
        dropped_series_df.to_csv(dropped_path, index=False)

        # Step 12: Save weekly.parquet with short gap interpolation for lags
        processed_weekly, s12 = step12_create_weekly_dataset(
            filtered_weekly, kept_series_df, config, output_weekly
        )
        summary_rows.append(s12)

        # Step 13: Coverage visuals
        step13_generate_coverage_visuals(processed_weekly, figures_dir)

        # Save cleaning summary table
        summary_df = pd.DataFrame(summary_rows)
        tables_dir.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Comprehensive cleaning summary saved to {summary_path}")

        # Highlight high-impact steps (>5% removed)
        for _, r in summary_df.iterrows():
            if "pct_removed" in r and r["pct_removed"] > 5.0:
                logger.warning(f"HIGH IMPACT STEP: {r['step']} removed {r['rows_removed']:,} rows ({r['pct_removed']:.2f}% > 5%)")

        logger.info(f"Complete data cleaning & scope pipeline finished in {time.time() - t_start:.2f}s")
        return output_parquet_all, summary_df

    finally:
        con.close()
        if db_file.exists():
            try:
                db_file.unlink()
            except Exception:
                pass


def run_weekly_pipeline(config: Optional[AppConfig] = None) -> Tuple[Path, pd.DataFrame]:
    """Execute Steps 10-13 starting from data/interim/clean_daily.parquet.

    Args:
        config: Optional AppConfig.

    Returns:
        Tuple of (output_weekly_path, processed_weekly_df).
    """
    if config is None:
        config = load_config()

    output_parquet_scoped = config.paths.interim_dir / "clean_daily.parquet"
    output_weekly = config.paths.processed_dir / "weekly.parquet"
    figures_dir = config.paths.figures_dir
    tables_dir = config.paths.tables_dir

    if not output_parquet_scoped.exists():
        raise FileNotFoundError(f"{output_parquet_scoped} does not exist. Run full cleaning first.")

    logger.info(f"Loading scoped daily data from {output_parquet_scoped}...")
    scoped_df = pd.read_parquet(output_parquet_scoped)

    # Step 10: Weekly aggregation
    weekly_df = aggregate_daily_to_iso_weeks(scoped_df)

    # Step 11: Series quality filter
    kept_series_df, dropped_series_df, filtered_weekly = filter_series_quality(
        weekly_df,
        min_weeks=config.cleaning.min_weeks,
        max_missing_share=config.cleaning.max_missing_share,
    )
    tables_dir.mkdir(parents=True, exist_ok=True)
    kept_series_df.to_csv(tables_dir / "series_kept.csv", index=False)
    dropped_series_df.to_csv(tables_dir / "series_dropped.csv", index=False)

    # Step 12: Save weekly.parquet with short gap interpolation for lags
    processed_weekly, s12 = step12_create_weekly_dataset(
        filtered_weekly, kept_series_df, config, output_weekly
    )

    # Step 13: Coverage visuals
    step13_generate_coverage_visuals(processed_weekly, figures_dir)

    logger.info("Weekly pipeline completed successfully.")
    return output_weekly, processed_weekly


if __name__ == "__main__":
    run_cleaning()
