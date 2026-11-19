"""
Baseline forecasting models for Mandi Watch.
Implements Section 8.3 of the project specification:
1. Last observed weekly price (persistence).
2. Seasonal naive (same week last year: week t + h - 52, with fallback to last observed).
3. Market historical average (expanding mean of past observed prices up to week t).

STRICT LEAKAGE PREVENTION:
All baselines strictly use past observations available at or before week t.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation import evaluate_forecasts

logger = logging.getLogger(__name__)


def add_baseline_predictions(df: pd.DataFrame, h: int = 2) -> pd.DataFrame:
    """
    Computes baseline predictions for forecast horizon h using strictly past data:
    1. pred_last_observed: most recent observed price at or before week t.
    2. pred_seasonal_naive: price at week (t + h - 52), falling back to last observed.
    3. pred_historical_mean: expanding mean of all past observed prices up to week t.
    
    All returned in log_price scale matching target_h{h}.
    """
    df = df.copy()
    df["week_start"] = pd.to_datetime(df["week_start"])
    df = df.sort_values(["market", "commodity", "week_start"]).reset_index(drop=True)

    grp = df.groupby(["market", "commodity"])

    # 1. Last Observed Weekly Price (Persistence)
    # Forward-fill the observed log_price per series up to week t
    df["pred_last_observed"] = grp["log_price"].ffill()

    # 2. Market Historical Average (Expanding Mean of Past Observations)
    # Expanding mean of past observed log_prices up to week t
    df["pred_historical_mean"] = (
        grp["log_price"]
        .expanding(min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    # 3. Seasonal Naive: same week last year (week t + h - 52)
    # Target is at t + h. Same week last year was at (t + h) - 52 weeks = t - (52 - h) weeks.
    # Because h <= 4 < 52, lag_offset = 52 - h > 0 (always strictly in the past!).
    lag_offset = 52 - h
    logger.info(f"Computing seasonal naive baseline for h={h} (lag={lag_offset} weeks)...")
    df["pred_seasonal_naive"] = grp["log_price"].shift(lag_offset)

    # If the exact 52-week-ago price is missing, fall back to last observed
    df["pred_seasonal_naive"] = df["pred_seasonal_naive"].fillna(df["pred_last_observed"])

    return df


def run_baselines_pipeline(
    features_path: Path,
    h: int = 2,
    output_table_path: Optional[Path] = None,
    output_fig_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs baselines evaluation pipeline:
    - Adds baseline forecasts.
    - Computes MAE, MAPE, RMSE overall and per commodity on validation and test sets.
    - Saves reports/tables/baseline_metrics.csv.
    - Generates and saves comparison figure to reports/figures/baseline_mae_comparison.png.
    """
    logger.info(f"Loading feature table from {features_path}...")
    df = pd.read_parquet(features_path)

    target_col = f"target_h{h}"
    df = add_baseline_predictions(df, h=h)

    # Baseline models to evaluate
    baseline_cols = {
        "Last Observed": "pred_last_observed",
        "Seasonal Naive": "pred_seasonal_naive",
        "Historical Mean": "pred_historical_mean",
    }

    metrics_list = []
    for model_name, col in baseline_cols.items():
        m_df = evaluate_forecasts(
            df,
            pred_col=col,
            target_col=target_col,
            model_name=model_name,
            splits=["validation", "test"],
            is_log_scale=True,
        )
        metrics_list.append(m_df)

    all_metrics = pd.concat(metrics_list, ignore_index=True)

    # Save metrics table
    if output_table_path is not None:
        output_table_path.parent.mkdir(parents=True, exist_ok=True)
        all_metrics.to_csv(output_table_path, index=False)
        logger.info(f"Saved baseline metrics to {output_table_path}")

    # Plot comparison bar chart
    if output_fig_path is not None:
        output_fig_path.parent.mkdir(parents=True, exist_ok=True)
        plot_baseline_comparison(all_metrics, output_fig_path)
        logger.info(f"Saved baseline comparison figure to {output_fig_path}")

    return df, all_metrics


def plot_baseline_comparison(metrics_df: pd.DataFrame, output_path: Path) -> None:
    """Plots comparison of MAE and MAPE across baselines for validation and test sets."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    val_df = metrics_df[(metrics_df["split"] == "validation") & (metrics_df["commodity"] == "ALL")]
    test_df = metrics_df[(metrics_df["split"] == "test") & (metrics_df["commodity"] == "ALL")]

    models = val_df["model"].tolist()
    x = np.arange(len(models))
    width = 0.35

    # Panel 1: MAE (INR/quintal)
    axes[0].bar(x - width/2, val_df["mae_inr"], width, label="Validation", color="#2b5c8f", alpha=0.9)
    axes[0].bar(x + width/2, test_df["mae_inr"], width, label="Test", color="#d95f02", alpha=0.9)
    axes[0].set_ylabel("MAE (INR / quintal)", fontsize=11, fontweight="bold")
    axes[0].set_title("Overall Forecast Error: MAE across Baselines", fontsize=12, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, fontsize=10)
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # Panel 2: MAPE (%)
    axes[1].bar(x - width/2, val_df["mape_pct"], width, label="Validation", color="#2b5c8f", alpha=0.9)
    axes[1].bar(x + width/2, test_df["mape_pct"], width, label="Test", color="#d95f02", alpha=0.9)
    axes[1].set_ylabel("MAPE (%)", fontsize=11, fontweight="bold")
    axes[1].set_title("Overall Forecast Error: MAPE across Baselines", fontsize=12, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, fontsize=10)
    axes[1].legend()
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    features_path = Path("data/processed/features_h2.parquet")
    table_path = Path("reports/tables/baseline_metrics.csv")
    fig_path = Path("reports/figures/baseline_mae_comparison.png")
    run_baselines_pipeline(features_path, h=2, output_table_path=table_path, output_fig_path=fig_path)
