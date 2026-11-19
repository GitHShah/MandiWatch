"""
Feature engineering pipeline for Mandi Watch.
Implements Section 8.1 of the project specification:
- One row per (market + commodity + week t).
- Target: log_price(t + h), where h is configurable (default 2; supports 1, 2, 4).
- STRICT LEAKAGE PREVENTION: All features computed using ONLY information available at the END of week t.
- Features:
  - Lags of log weekly price: 1, 2, 3, 4, 8, 52 (from gap-filled log price).
  - Trailing rolling mean (4 and 12 weeks), trailing rolling std (4 weeks) - never centered!
  - Momentum: change in log price over 1 and 4 weeks.
  - Seasonality: sin/cos of ISO week-of-year, month, sin/cos of month.
  - Reporting indicators: n_reports this week, weeks since last report, flat-run flag.
  - Market premium: expanding mean of (market log price minus state-commodity mean log price), PAST data only.
  - Neighbor feature: mean lagged log price of k nearest markets (default k=5, radius <= 150km) for the same commodity.
  - Static features: lat, lon, state, district, commodity (as categoricals).
  - is_interpolated as a feature.
- Time-based splits:
  - test: last 12 months.
  - validation: preceding 12 months.
  - train: earlier.
- Output: data/processed/features_h{h}.parquet.
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.config import load_config

logger = logging.getLogger(__name__)


def assign_time_splits(
    df: pd.DataFrame,
    date_col: str = "week_start",
    test_months: int = 12,
    val_months: int = 12,
) -> pd.DataFrame:
    """
    Assigns strictly non-overlapping, chronological evaluation splits:
    - test: last `test_months` of the dataset.
    - validation: `val_months` preceding the test set.
    - train: all observations strictly earlier than validation.
    """
    df = df.copy()
    dates = pd.to_datetime(df[date_col])
    max_date = dates.max()

    test_start = max_date - pd.DateOffset(months=test_months)
    val_start = test_start - pd.DateOffset(months=val_months)

    conditions = [
        dates >= test_start,
        (dates >= val_start) & (dates < test_start),
        dates < val_start,
    ]
    choices = ["test", "validation", "train"]
    df["split"] = np.select(conditions, choices, default="train")

    logger.info(
        f"Split boundaries: Train < {val_start.strftime('%Y-%m-%d')}, "
        f"Val [{val_start.strftime('%Y-%m-%d')} to {test_start.strftime('%Y-%m-%d')}), "
        f"Test >= {test_start.strftime('%Y-%m-%d')}"
    )
    return df


def compute_neighbor_features(
    df: pd.DataFrame,
    dist_matrix: pd.DataFrame,
    k: int = 5,
    max_radius_km: float = 150.0,
    price_col: str = "log_price_interpolated",
) -> pd.Series:
    """
    Computes the neighbor feature: mean LAGGED log price of the k nearest markets
    for the SAME commodity within max_radius_km.
    
    STRICT LEAKAGE PREVENTION:
    Uses strictly lagged prices (shift=1) available at week t.
    """
    neighbor_series = pd.Series(index=df.index, dtype=float, name="neighbor_lag1")

    # Map each market to its k nearest neighbors within radius
    neighbor_map: Dict[str, List[str]] = {}
    valid_mkts = [m for m in dist_matrix.index if m in dist_matrix.columns]
    for mkt in valid_mkts:
        dists = dist_matrix.loc[mkt]
        # Candidates within radius excluding self
        cands = dists[(dists > 0) & (dists <= max_radius_km)].sort_values()
        neighbor_map[mkt] = cands.index.tolist()

    # Process per commodity
    for comm, group in df.groupby("commodity"):
        sub = group[["market", "week_start", price_col]].copy()
        sub["week_start"] = pd.to_datetime(sub["week_start"])
        # Pivot to week_start x market grid and SHIFT BY 1 (past information only!)
        piv = sub.pivot(index="week_start", columns="market", values=price_col).shift(1)

        # For each market, average the up-to-k available nearest neighbors
        comm_neighbor_df = pd.DataFrame(index=piv.index, columns=piv.columns, dtype=float)
        for mkt in piv.columns:
            if mkt in neighbor_map:
                nearby = [nb for nb in neighbor_map[mkt] if nb in piv.columns][:k]
                if nearby:
                    comm_neighbor_df[mkt] = piv[nearby].mean(axis=1)

        # Stack back to (week_start, market) Series
        stacked = comm_neighbor_df.stack(future_stack=True)
        # Map values to the group rows
        key_tuples = list(zip(sub["week_start"], sub["market"]))
        mapped_vals = stacked.reindex(key_tuples).values
        neighbor_series.loc[group.index] = mapped_vals

    return neighbor_series


def compute_market_premium(
    df: pd.DataFrame,
    price_col: str = "log_price_interpolated",
) -> pd.Series:
    """
    Computes expanding market premium: expanding mean of
    (market log price minus state-commodity average log price), computed using PAST DATA ONLY.
    
    STRICT LEAKAGE PREVENTION:
    The market price and state-commodity mean are both shifted by 1 (lagged to week t-1),
    and the expanding mean uses min_periods=1 on strictly past data.
    """
    df_sorted = df[["state", "commodity", "market", "week_start", price_col]].copy()
    df_sorted["week_start"] = pd.to_datetime(df_sorted["week_start"])

    # 1. State-commodity average log price at each week (lagged by 1 week)
    state_comm_df = (
        df_sorted.groupby(["state", "commodity", "week_start"])[price_col]
        .mean()
        .reset_index()
        .sort_values(["state", "commodity", "week_start"])
    )
    state_comm_df["state_comm_lag1"] = (
        state_comm_df.groupby(["state", "commodity"])[price_col].shift(1)
    )

    df_sorted = df_sorted.merge(
        state_comm_df[["state", "commodity", "week_start", "state_comm_lag1"]],
        on=["state", "commodity", "week_start"],
        how="left",
    )
    # Lagged market price
    df_sorted["market_lag1"] = df_sorted.groupby(["market", "commodity"])[price_col].shift(1)
    # Past spread
    df_sorted["past_spread"] = df_sorted["market_lag1"] - df_sorted["state_comm_lag1"]

    # Expanding mean over past spread per series
    expanding_premium = (
        df_sorted.groupby(["market", "commodity"])["past_spread"]
        .expanding(min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )
    expanding_premium.index = df.index
    expanding_premium.name = "market_premium"
    return expanding_premium


def build_feature_table(
    weekly_path: Path,
    dist_matrix_path: Path,
    coords_path: Path,
    h: int = 2,
    k_neighbors: int = 5,
    max_radius_km: float = 150.0,
) -> pd.DataFrame:
    """
    Constructs the end-to-end feature table from weekly.parquet.
    
    Target: log_price(t + h) per series.
    All features use strictly information available at the end of week t.
    """
    logger.info(f"Building feature table for horizon h={h} weeks...")
    df = pd.read_parquet(weekly_path)
    dist_matrix = pd.read_parquet(dist_matrix_path)
    coords_df = pd.read_csv(coords_path).drop_duplicates(subset=["market", "district", "state"])

    df["week_start"] = pd.to_datetime(df["week_start"])

    # Deduplicate in case of duplicate entries for same (market, commodity, week_start)
    # Keep the row with the largest n_reports
    if df.duplicated(subset=["market", "commodity", "week_start"]).any():
        logger.info("Deduplicating multiple records per (market, commodity, week_start)...")
        df = (
            df.sort_values("n_reports", ascending=False)
            .drop_duplicates(subset=["market", "commodity", "week_start"], keep="first")
        )

    # Sort deterministically
    df = df.sort_values(["market", "commodity", "week_start"]).reset_index(drop=True)

    series_grp = df.groupby(["market", "commodity"])

    # 1. Target: log_price at week (t + h)
    # Using true observed log_price for the target
    df[f"target_h{h}"] = series_grp["log_price"].shift(-h)

    # 2. Lags of gap-filled log price: 1, 2, 3, 4, 8, 52
    lag_cols = [1, 2, 3, 4, 8, 52]
    for lag in lag_cols:
        df[f"lag_{lag}"] = series_grp["log_price_interpolated"].shift(lag)

    # 3. Trailing Rolling Statistics (4 and 12 weeks mean, 4 weeks std)
    # STRICT LEAKAGE PREVENTION: trailing windows only, never centered!
    # Available at end of week t: rolling over lag_0 (current week t)
    df["rolling_mean_4"] = (
        series_grp["log_price_interpolated"]
        .rolling(window=4, min_periods=2)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )
    df["rolling_mean_12"] = (
        series_grp["log_price_interpolated"]
        .rolling(window=12, min_periods=4)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )
    df["rolling_std_4"] = (
        series_grp["log_price_interpolated"]
        .rolling(window=4, min_periods=2)
        .std(ddof=1)
        .reset_index(level=[0, 1], drop=True)
    )

    # 4. Momentum: Change in log price over 1 and 4 weeks
    df["momentum_1"] = df["log_price_interpolated"] - df["lag_1"]
    df["momentum_4"] = df["log_price_interpolated"] - df["lag_4"]

    # 5. Seasonality: sin/cos of week of year and month
    iso_week = df["week_start"].dt.isocalendar().week.astype(float)
    df["sin_week"] = np.sin(2.0 * np.pi * iso_week / 52.1775)
    df["cos_week"] = np.cos(2.0 * np.pi * iso_week / 52.1775)
    df["month"] = df["week_start"].dt.month
    df["sin_month"] = np.sin(2.0 * np.pi * df["month"] / 12.0)
    df["cos_month"] = np.cos(2.0 * np.pi * df["month"] / 12.0)

    # 6. Reporting indicators:
    # - n_reports: reports this week (0 if missing/unreported)
    df["n_reports"] = df["n_reports"].fillna(0.0)

    # - weeks_since_last_report: consecutive weeks without raw arrival
    is_reported = (~df["is_interpolated"]) & (df["n_reports"] > 0)
    last_reported_date = pd.Series(
        np.where(is_reported, df["week_start"], pd.NaT), index=df.index
    ).groupby([df["market"], df["commodity"]]).ffill()
    df["weeks_since_last_report"] = (
        (df["week_start"] - pd.to_datetime(last_reported_date)).dt.days / 7.0
    ).fillna(52.0)

    # - flat_run_flag: binary flag indicating if consecutive flat price run occurred
    df["flat_run_flag"] = (df["flat_run_share"].fillna(0.0) > 0.0).astype(int)

    # - is_interpolated: boolean feature
    df["is_interpolated"] = df["is_interpolated"].astype(int)

    # 7. Market Premium: Expanding mean of (market - state mean) using PAST DATA ONLY
    logger.info("Computing expanding past-only market premium...")
    df["market_premium"] = compute_market_premium(df, price_col="log_price_interpolated")

    # 8. Neighbor Feature: mean lagged log price of k nearest markets for same commodity
    logger.info("Computing k-nearest neighbor lag features...")
    df["neighbor_lag1"] = compute_neighbor_features(
        df, dist_matrix, k=k_neighbors, max_radius_km=max_radius_km, price_col="log_price_interpolated"
    )

    # 9. Static Features: Coordinates and Categoricals
    coords_sub = coords_df[["state", "district", "market", "lat", "lon"]]
    df = df.merge(coords_sub, on=["state", "district", "market"], how="left")

    # 10. Time-based Evaluation Splits
    config = load_config()
    test_m = config.split.test_months if hasattr(config, "split") else 12
    val_m = config.split.val_months if hasattr(config, "split") else 12
    df = assign_time_splits(df, date_col="week_start", test_months=test_m, val_months=val_m)

    # Convert categoricals to category dtype
    cat_cols = ["state", "district", "market", "commodity"]
    for c in cat_cols:
        df[c] = df[c].astype("category")

    logger.info(f"Feature table built: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


def generate_and_save_features(
    horizons: List[int] = [1, 2, 4],
    output_dir: Optional[Path] = None,
) -> Dict[int, pd.DataFrame]:
    """
    Builds and saves feature tables for all requested forecast horizons (e.g. h=1, 2, 4).
    Saves to data/processed/features_h{h}.parquet.
    """
    config = load_config()
    proc_dir = output_dir if output_dir is not None else Path(config.paths.processed_dir)
    proc_dir.mkdir(parents=True, exist_ok=True)

    weekly_path = proc_dir / "weekly.parquet"
    dist_matrix_path = Path(config.paths.geo_dir) / "market_distance_matrix.parquet"
    coords_path = Path(config.paths.geo_dir) / "market_coordinates.csv"

    feature_tables = {}
    for h in horizons:
        feat_df = build_feature_table(
            weekly_path=weekly_path,
            dist_matrix_path=dist_matrix_path,
            coords_path=coords_path,
            h=h,
            k_neighbors=5,
            max_radius_km=config.geo.default_radius_km,
        )
        out_path = proc_dir / f"features_h{h}.parquet"
        feat_df.to_parquet(out_path, index=False)
        logger.info(f"Saved feature table for h={h} to {out_path}")
        feature_tables[h] = feat_df

    return feature_tables


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    generate_and_save_features(horizons=[2])
