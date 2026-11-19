"""
Machine learning modeling and uncertainty quantification pipeline for Mandi Watch.
Implements Section 8.4 of the project specification with the Delta-Target formulation:
1. Target: Price change relative to last observed price: Delta_log = log_price(t + h) - log_price(t)
   Prediction: log_price_hat = log_price(t) + Delta_log_hat
2. Main model: Global LightGBM regressor on delta target with early stopping on validation.
3. Uncertainty: LightGBM quantile regression models (alpha=0.1, 0.5, 0.9) with conformal calibration.
4. Feature importance (gain) extraction and interpretation.
5. Statistical comparison: SARIMA / ETS benchmark on representative series.
6. Diebold-Mariano test (with HLN correction) against the best baseline.
7. Ljung-Box autocorrelation test on residuals.
8. Artifact saving: models to models/ and precomputed forecasts to data/processed/forecasts_h{h}.parquet.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.evaluation import compute_coverage, compute_metrics, evaluate_forecasts
from src.stats_tests import test_diebold_mariano

logger = logging.getLogger(__name__)

# Canonical feature columns for modeling
FEATURE_COLS = [
    "lag_1", "lag_2", "lag_3", "lag_4", "lag_8", "lag_52",
    "rolling_mean_4", "rolling_mean_12", "rolling_std_4",
    "momentum_1", "momentum_4",
    "sin_week", "cos_week", "month", "sin_month", "cos_month",
    "n_reports", "weeks_since_last_report", "flat_run_flag", "is_interpolated",
    "market_premium", "neighbor_lag1",
    "lat", "lon",
    "state", "district", "commodity"
]

CATEGORICAL_COLS = ["state", "district", "commodity"]


def prepare_delta_datasets(
    df: pd.DataFrame,
    target_col: str = "target_h2",
    base_col: str = "lag_1",
    feature_cols: Optional[List[str]] = None,
    cat_cols: Optional[List[str]] = None,
) -> Dict[str, Tuple[pd.DataFrame, pd.Series, pd.Series]]:
    """Splits dataframe into X, y_delta, and base_price per split."""
    if feature_cols is None:
        feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    if cat_cols is None:
        cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]

    data = {}
    for split_name in ["train", "validation", "test"]:
        sub = df[df["split"] == split_name].dropna(subset=[target_col, base_col]).copy()
        X = sub[feature_cols].copy()
        for c in cat_cols:
            if c in X.columns:
                X[c] = X[c].astype("category")
        y_delta = sub[target_col].astype(float) - sub[base_col].astype(float)
        base_price = sub[base_col].astype(float)
        data[split_name] = (X, y_delta, base_price)

    return data


def tune_lightgbm_delta(
    X_train: pd.DataFrame,
    y_train_delta: pd.Series,
    X_val: pd.DataFrame,
    y_val_delta: pd.Series,
    cat_cols: Optional[List[str]] = None,
    seed: int = 42,
) -> Tuple[lgb.LGBMRegressor, Dict[str, Any]]:
    """
    Tuning LightGBM delta regressor on the VALIDATION set only.
    """
    logger.info("Tuning LightGBM delta regressor on validation set...")
    if cat_cols is None:
        cat_cols = [c for c in CATEGORICAL_COLS if c in X_train.columns]

    candidates = [
        {"learning_rate": 0.08, "num_leaves": 45, "min_child_samples": 40, "subsample": 0.85},
        {"learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 30, "subsample": 0.80},
        {"learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 50, "subsample": 0.80},
    ]

    best_val_mae = float("inf")
    best_params = None
    best_model = None

    for i, params in enumerate(candidates):
        model = lgb.LGBMRegressor(
            objective="regression_l1",
            n_estimators=1000,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
            **params,
        )
        model.fit(
            X_train,
            y_train_delta,
            eval_set=[(X_val, y_val_delta)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
            categorical_feature=cat_cols,
        )
        val_pred = model.predict(X_val)
        val_mae = float(np.mean(np.abs(y_val_delta.values - val_pred)))
        logger.info(f"Config {i+1}: {params} -> Val MAE (delta scale): {val_mae:.4f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_params = params
            best_model = model

    logger.info(f"Selected best delta hyperparameters: {best_params} with Val delta MAE: {best_val_mae:.4f}")
    return best_model, best_params


def train_quantile_delta_models(
    X_train: pd.DataFrame,
    y_train_delta: pd.Series,
    X_val: pd.DataFrame,
    y_val_delta: pd.Series,
    base_params: Dict[str, Any],
    quantiles: List[float] = [0.1, 0.5, 0.9],
    cat_cols: Optional[List[str]] = None,
    seed: int = 42,
) -> Dict[float, lgb.LGBMRegressor]:
    """Trains LightGBM quantile regression models for delta uncertainty intervals."""
    logger.info(f"Training delta quantile models for alphas {quantiles}...")
    if cat_cols is None:
        cat_cols = [c for c in CATEGORICAL_COLS if c in X_train.columns]

    q_models = {}
    for q in quantiles:
        q_model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=q,
            n_estimators=800,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
            **base_params,
        )
        q_model.fit(
            X_train,
            y_train_delta,
            eval_set=[(X_val, y_val_delta)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
            categorical_feature=cat_cols,
        )
        q_models[q] = q_model
        logger.info(f"Trained delta quantile model alpha={q} (best iteration: {q_model.best_iteration_})")

    return q_models


def calibrate_conformal_interval(
    y_val: np.ndarray,
    p10_val: np.ndarray,
    p90_val: np.ndarray,
    target_coverage: float = 0.80,
) -> float:
    """Computes symmetric conformal residual adjustment on the validation set."""
    valid_mask = (~np.isnan(y_val)) & (~np.isnan(p10_val)) & (~np.isnan(p90_val))
    if not np.any(valid_mask):
        return 0.0
    scores = np.maximum(p10_val[valid_mask] - y_val[valid_mask], y_val[valid_mask] - p90_val[valid_mask])
    q_adj = float(np.quantile(scores, target_coverage))
    logger.info(f"Conformal calibration: residual quantile adjustment = {q_adj:.4f}")
    return q_adj


def run_modeling_pipeline(
    features_path: Path,
    h: int = 2,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Runs the complete modeling, tuning, uncertainty quantification, diagnostics,
    and artifact export workflow using the Delta formulation.
    """
    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = ROOT / "reports" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = ROOT / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(features_path)
    target_col = f"target_h{h}"

    # 1. Prepare Delta Datasets
    data = prepare_delta_datasets(df, target_col=target_col, base_col="lag_1")
    X_train, y_train_delta, base_train = data["train"]
    X_val, y_val_delta, base_val = data["validation"]
    X_test, y_test_delta, base_test = data["test"]

    # 2. Main Model Tuning (Validation set only!)
    best_point_model, best_params = tune_lightgbm_delta(
        X_train, y_train_delta, X_val, y_val_delta, seed=42
    )

    # 3. Uncertainty Quantile Models
    q_models = train_quantile_delta_models(
        X_train, y_train_delta, X_val, y_val_delta, base_params=best_params, quantiles=[0.1, 0.5, 0.9], seed=42
    )

    # 4. Predictions on Entire Dataset
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    for c in CATEGORICAL_COLS:
        if c in df.columns:
            df[c] = df[c].astype("category")

    base_series = df["lag_1"].fillna(df["log_price_interpolated"])

    delta_point = best_point_model.predict(df[feature_cols])
    delta_p10 = q_models[0.1].predict(df[feature_cols])
    delta_p50 = q_models[0.5].predict(df[feature_cols])
    delta_p90 = q_models[0.9].predict(df[feature_cols])

    df["pred_lgb_point_log"] = base_series + delta_point
    df["pred_lgb_p10_log"] = base_series + delta_p10
    df["pred_lgb_p50_log"] = base_series + delta_p50
    df["pred_lgb_p90_log"] = base_series + delta_p90

    # 5. Uncertainty Coverage & Conformal Calibration
    val_sub = df[df["split"] == "validation"].dropna(subset=[target_col, "pred_lgb_p10_log", "pred_lgb_p90_log"])
    test_sub = df[df["split"] == "test"].dropna(subset=[target_col, "pred_lgb_p10_log", "pred_lgb_p90_log"])

    cov_val_raw = compute_coverage(
        val_sub[target_col], val_sub["pred_lgb_p10_log"], val_sub["pred_lgb_p90_log"]
    )
    cov_test_raw = compute_coverage(
        test_sub[target_col], test_sub["pred_lgb_p10_log"], test_sub["pred_lgb_p90_log"]
    )

    q_adj = calibrate_conformal_interval(
        val_sub[target_col].values,
        val_sub["pred_lgb_p10_log"].values,
        val_sub["pred_lgb_p90_log"].values,
        target_coverage=0.80,
    )
    df["pred_lgb_p10_calib_log"] = df["pred_lgb_p10_log"] - q_adj
    df["pred_lgb_p90_calib_log"] = df["pred_lgb_p90_log"] + q_adj
    df["pred_lgb_point_inr"] = np.exp(df["pred_lgb_point_log"])
    df["pred_lgb_p10_calib_inr"] = np.exp(df["pred_lgb_p10_calib_log"])
    df["pred_lgb_p90_calib_inr"] = np.exp(df["pred_lgb_p90_calib_log"])

    val_sub = df[df["split"] == "validation"].dropna(subset=[target_col, "pred_lgb_p10_calib_log", "pred_lgb_p90_calib_log"])
    test_sub = df[df["split"] == "test"].dropna(subset=[target_col, "pred_lgb_p10_calib_log", "pred_lgb_p90_calib_log"])
    cov_val_calib = compute_coverage(
        val_sub[target_col], val_sub["pred_lgb_p10_calib_log"], val_sub["pred_lgb_p90_calib_log"]
    )
    cov_test_calib = compute_coverage(
        test_sub[target_col], test_sub["pred_lgb_p10_calib_log"], test_sub["pred_lgb_p90_calib_log"]
    )

    coverage_summary = pd.DataFrame([
        {"split": "validation", "type": "Raw Quantile", "coverage_pct": cov_val_raw["coverage_80"], "width_log": cov_val_raw["interval_width"]},
        {"split": "validation", "type": "Conformal Calibrated", "coverage_pct": cov_val_calib["coverage_80"], "width_log": cov_val_calib["interval_width"]},
        {"split": "test", "type": "Raw Quantile", "coverage_pct": cov_test_raw["coverage_80"], "width_log": cov_test_raw["interval_width"]},
        {"split": "test", "type": "Conformal Calibrated", "coverage_pct": cov_test_calib["coverage_80"], "width_log": cov_test_calib["interval_width"]},
    ])
    coverage_summary.to_csv(tables_dir / "interval_coverage.csv", index=False)

    # 6. Forecast Metrics Evaluation against Baselines
    from src.baselines import add_baseline_predictions
    df = add_baseline_predictions(df, h=h)

    models_to_eval = {
        "LightGBM": "pred_lgb_point_log",
        "Last Observed": "pred_last_observed",
        "Seasonal Naive": "pred_seasonal_naive",
        "Historical Mean": "pred_historical_mean",
    }

    eval_tables = []
    for m_name, m_col in models_to_eval.items():
        m_eval = evaluate_forecasts(
            df, pred_col=m_col, target_col=target_col, model_name=m_name, splits=["validation", "test"]
        )
        eval_tables.append(m_eval)

    all_comparison = pd.concat(eval_tables, ignore_index=True)
    all_comparison.to_csv(tables_dir / "forecast_model_comparison.csv", index=False)

    # 7. Diebold-Mariano Test (LightGBM vs Last Observed on Test Set)
    test_eval_df = df[df["split"] == "test"].dropna(subset=[target_col, "pred_lgb_point_log", "pred_last_observed"])
    e_lgb = test_eval_df[target_col].values - test_eval_df["pred_lgb_point_log"].values
    e_base = test_eval_df[target_col].values - test_eval_df["pred_last_observed"].values

    dm_overall = test_diebold_mariano(e_lgb, e_base, h=h, criterion="MAE")
    dm_records = [{
        "commodity": "ALL",
        "dm_stat": dm_overall["statistic"],
        "p_value": dm_overall["p_value"],
        "mean_loss_diff": dm_overall["mean_differential"],
        "significant": dm_overall["significant"],
        "interpretation": dm_overall["interpretation"],
    }]

    for comm, group in test_eval_df.groupby("commodity"):
        e_lgb_c = group[target_col].values - group["pred_lgb_point_log"].values
        e_base_c = group[target_col].values - group["pred_last_observed"].values
        dm_comm = test_diebold_mariano(e_lgb_c, e_base_c, h=h, criterion="MAE")
        dm_records.append({
            "commodity": comm,
            "dm_stat": dm_comm["statistic"],
            "p_value": dm_comm["p_value"],
            "mean_loss_diff": dm_comm["mean_differential"],
            "significant": dm_comm["significant"],
            "interpretation": dm_comm["interpretation"],
        })

    dm_table = pd.DataFrame(dm_records)
    dm_table.to_csv(tables_dir / "diebold_mariano_test.csv", index=False)

    # 8. Residual Ljung-Box Test
    residuals = e_lgb
    lb_res = acorr_ljungbox(residuals, lags=[4, 8, 12], return_df=True)
    lb_res.to_csv(tables_dir / "residuals_ljung_box.csv")

    # 9. Feature Importance
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance_gain": best_point_model.booster_.feature_importance(importance_type="gain"),
        "importance_split": best_point_model.booster_.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False)
    importance_df.to_csv(tables_dir / "feature_importance.csv", index=False)

    plot_feature_importance(importance_df, figures_dir / "feature_importance_gain.png")
    plot_model_comparison(all_comparison, figures_dir / "model_forecast_comparison.png")

    # 10. Save Models and Precomputed Forecasts
    joblib.dump(best_point_model, models_dir / f"lgb_point_h{h}.joblib")
    best_point_model.booster_.save_model(str(models_dir / f"lgb_point_h{h}.txt"))
    for q, qm in q_models.items():
        joblib.dump(qm, models_dir / f"lgb_q{int(q*100)}_h{h}.joblib")
        qm.booster_.save_model(str(models_dir / f"lgb_q{int(q*100)}_h{h}.txt"))

    forecast_cols = [
        "market", "commodity", "state", "district", "week_start", "lat", "lon", "split",
        f"target_h{h}", "pred_lgb_point_inr", "pred_lgb_p10_calib_inr", "pred_lgb_p90_calib_inr",
        "pred_lgb_point_log", "pred_lgb_p10_calib_log", "pred_lgb_p90_calib_log",
        "pred_last_observed", "pred_seasonal_naive", "pred_historical_mean"
    ]
    forecast_df = df[forecast_cols].copy()
    forecast_df = forecast_df.rename(columns={
        f"target_h{h}": "target_log_price",
        "pred_lgb_point_inr": "pred_point_inr",
        "pred_lgb_p10_calib_inr": "pred_p10_inr",
        "pred_lgb_p90_calib_inr": "pred_p90_inr",
        "pred_lgb_point_log": "pred_point_log",
        "pred_lgb_p10_calib_log": "pred_p10_log",
        "pred_lgb_p90_calib_log": "pred_p90_log",
    })
    forecast_df["target_week_start"] = pd.to_datetime(forecast_df["week_start"]) + pd.DateOffset(weeks=h)
    forecast_df["realized_price_inr"] = np.exp(forecast_df["target_log_price"])

    out_forecast_path = ROOT / "data" / "processed" / f"forecasts_h{h}.parquet"
    forecast_df.to_parquet(out_forecast_path, index=False)
    logger.info(f"Saved precomputed forecasts table to {out_forecast_path}")

    return {
        "all_comparison": all_comparison,
        "coverage_summary": coverage_summary,
        "dm_table": dm_table,
        "importance_df": importance_df,
    }


def plot_feature_importance(importance_df: pd.DataFrame, output_path: Path) -> None:
    top15 = importance_df.head(15).iloc[::-1]
    plt.figure(figsize=(10, 6))
    plt.barh(top15["feature"], top15["importance_gain"], color="#1f77b4", alpha=0.9)
    plt.xlabel("Total Gain", fontsize=11, fontweight="bold")
    plt.title("Top 15 Features by LightGBM Importance (Gain)", fontsize=12, fontweight="bold")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_model_comparison(comparison_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    val_df = comparison_df[(comparison_df["split"] == "validation") & (comparison_df["commodity"] == "ALL")]
    test_df = comparison_df[(comparison_df["split"] == "test") & (comparison_df["commodity"] == "ALL")]

    models = val_df["model"].tolist()
    x = np.arange(len(models))
    width = 0.35

    axes[0].bar(x - width/2, val_df["mae_inr"], width, label="Validation", color="#2b5c8f", alpha=0.9)
    axes[0].bar(x + width/2, test_df["mae_inr"], width, label="Test", color="#d95f02", alpha=0.9)
    axes[0].set_ylabel("MAE (INR / quintal)", fontsize=11, fontweight="bold")
    axes[0].set_title("Overall Model Comparison: MAE", fontsize=12, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, fontsize=10, rotation=15)
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    axes[1].bar(x - width/2, val_df["mape_pct"], width, label="Validation", color="#2b5c8f", alpha=0.9)
    axes[1].bar(x + width/2, test_df["mape_pct"], width, label="Test", color="#d95f02", alpha=0.9)
    axes[1].set_ylabel("MAPE (%)", fontsize=11, fontweight="bold")
    axes[1].set_title("Overall Model Comparison: MAPE", fontsize=12, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, fontsize=10, rotation=15)
    axes[1].legend()
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    features_path = ROOT / "data" / "processed" / "features_h2.parquet"
    run_modeling_pipeline(features_path, h=2)
