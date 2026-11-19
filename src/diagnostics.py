"""
Diagnostic and ablation experiment pipeline for Mandi Watch.
Diagnoses forecast error concentrations and tests 6 architectural/data variations
STRICTLY on the VALIDATION set (without touching the test set).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.evaluation import compute_metrics
from src.models import CATEGORICAL_COLS, FEATURE_COLS, prepare_datasets

logger = logging.getLogger(__name__)


def compute_error_breakdowns(df: pd.DataFrame, pred_col: str, target_col: str = "target_h2") -> Dict[str, pd.DataFrame]:
    """
    Computes validation error breakdowns by:
    commodity, state, market, series length, n_reports, year, season, and gap/flat flags.
    """
    val_df = df[df["split"] == "validation"].dropna(subset=[target_col, pred_col]).copy()
    val_df["realized_inr"] = np.exp(val_df[target_col])
    val_df["pred_inr"] = np.exp(val_df[pred_col])
    val_df["abs_err"] = np.abs(val_df["realized_inr"] - val_df["pred_inr"])
    val_df["pct_err"] = val_df["abs_err"] / val_df["realized_inr"] * 100.0

    # 1. Season definition (Indian Agricultural Meteorological Seasons)
    val_df["month"] = pd.to_datetime(val_df["week_start"]).dt.month
    season_map = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Summer", 4: "Summer", 5: "Summer",
        6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
        10: "Post-Monsoon", 11: "Post-Monsoon",
    }
    val_df["season"] = val_df["month"].map(season_map)

    # 2. Series length bucket
    series_lens = df.groupby(["market", "commodity"])["target_h2"].count().rename("total_series_len")
    val_df = val_df.join(series_lens, on=["market", "commodity"])
    val_df["series_len_bucket"] = pd.qcut(val_df["total_series_len"], q=3, labels=["Short (<180w)", "Medium (180-360w)", "Long (>360w)"])

    # 3. n_reports bucket
    val_df["reports_bucket"] = pd.cut(
        val_df["n_reports"].fillna(0),
        bins=[-1, 0.5, 2.5, 5.5, 100],
        labels=["0 (Unreported/Gap)", "1-2 (Sparse)", "3-5 (Normal)", ">5 (Dense)"],
    )

    # Helper aggregator
    def agg_metrics(group_col: str) -> pd.DataFrame:
        agg = val_df.groupby(group_col, observed=True).agg(
            n_obs=("abs_err", "count"),
            mae_inr=("abs_err", "mean"),
            mape_pct=("pct_err", "mean"),
            median_abs_err=("abs_err", "median"),
            rmse_inr=("abs_err", lambda x: np.sqrt(np.mean(x**2))),
        ).reset_index()
        agg["mae_inr"] = agg["mae_inr"].round(2)
        agg["mape_pct"] = agg["mape_pct"].round(2)
        agg["median_abs_err"] = agg["median_abs_err"].round(2)
        agg["rmse_inr"] = agg["rmse_inr"].round(2)
        return agg

    breakdowns = {
        "by_commodity": agg_metrics("commodity").sort_values("mae_inr", ascending=False),
        "by_state": agg_metrics("state").sort_values("mae_inr", ascending=False),
        "by_season": agg_metrics("season").sort_values("mae_inr", ascending=False),
        "by_series_len": agg_metrics("series_len_bucket"),
        "by_reports": agg_metrics("reports_bucket"),
        "by_is_interpolated": agg_metrics("is_interpolated"),
        "by_flat_run": agg_metrics("flat_run_flag"),
        "top_markets_worst": agg_metrics("market").sort_values("mae_inr", ascending=False).head(10),
        "top_markets_best": agg_metrics("market").sort_values("mae_inr", ascending=True).head(10),
    }

    return breakdowns


def run_ablation_experiments(df: pd.DataFrame, h: int = 2) -> pd.DataFrame:
    """
    Runs systematic ablation experiments strictly on the VALIDATION set:
    Baseline: Current Global LightGBM model.
    (a) Dropping sparsest series
    (b1) Shorter training window (2020+)
    (b2) Exponential recency sample weights
    (c) Era indicator (pre-COVID, COVID, inflation shock)
    (d) Simpler feature set (core top 12 features)
    (e) Per-commodity models
    (f) Predicting price change vs last price (delta target)
    """
    logger.info("Running systematic ablation experiments on validation set...")
    target_col = f"target_h{h}"

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]

    # Baseline Model Evaluation on Validation
    X_train_df = df[df["split"] == "train"].dropna(subset=[target_col]).copy()
    y_train = X_train_df[target_col].astype(float)
    X_val_df = df[df["split"] == "validation"].dropna(subset=[target_col]).copy()
    y_val = X_val_df[target_col].astype(float)

    base_params = {
        "objective": "regression_l1",
        "n_estimators": 600,
        "learning_rate": 0.08,
        "num_leaves": 45,
        "min_child_samples": 40,
        "subsample": 0.85,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }

    def train_and_eval(X_tr, y_tr, X_v, y_v, c_cols=cat_cols, sample_weight=None, target_is_delta=False, last_price_val=None):
        model = lgb.LGBMRegressor(**base_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_v, y_v)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
            categorical_feature=[c for c in c_cols if c in X_tr.columns],
            sample_weight=sample_weight,
        )
        pred_val_log = model.predict(X_v)
        if target_is_delta:
            pred_val_log = last_price_val + pred_val_log
            y_v_true = last_price_val + y_v
        else:
            y_v_true = y_v

        # Back-transform to INR
        val_realized = np.exp(y_v_true)
        val_pred = np.exp(pred_val_log)
        m = compute_metrics(val_realized, val_pred)
        return m, model

    # Benchmark: Baseline Global Model
    m_base, _ = train_and_eval(X_train_df[feature_cols], y_train, X_val_df[feature_cols], y_val)
    results = [{
        "experiment": "0. Baseline Global LightGBM",
        "val_mae_inr": m_base["mae"],
        "val_mape_pct": m_base["mape"],
        "val_rmse_inr": m_base["rmse"],
        "delta_mae_vs_base": 0.0,
        "improved": False,
    }]
    logger.info(f"Baseline Global Val MAE: {m_base['mae']} ₹/q, MAPE: {m_base['mape']}%")

    # (a) Dropping sparsest series (< 100 observations in train)
    series_train_counts = X_train_df.groupby(["market", "commodity"])["target_h2"].transform("count")
    dense_mask = series_train_counts >= 100
    X_tr_dense = X_train_df[dense_mask]
    y_tr_dense = y_train[dense_mask]
    m_dense, _ = train_and_eval(X_tr_dense[feature_cols], y_tr_dense, X_val_df[feature_cols], y_val)
    results.append({
        "experiment": "a. Drop Sparsest Series (<100 obs in train)",
        "val_mae_inr": m_dense["mae"],
        "val_mape_pct": m_dense["mape"],
        "val_rmse_inr": m_dense["rmse"],
        "delta_mae_vs_base": round(m_dense["mae"] - m_base["mae"], 2),
        "improved": m_dense["mae"] < m_base["mae"],
    })
    logger.info(f"Exp (a) Val MAE: {m_dense['mae']} ₹/q (delta: {m_dense['mae'] - m_base['mae']:+.2f})")

    # (b1) Shorter training window (2020+)
    recent_mask = pd.to_datetime(X_train_df["week_start"]).dt.year >= 2020
    X_tr_recent = X_train_df[recent_mask]
    y_tr_recent = y_train[recent_mask]
    m_recent, _ = train_and_eval(X_tr_recent[feature_cols], y_tr_recent, X_val_df[feature_cols], y_val)
    results.append({
        "experiment": "b1. Shorter Training Window (2020+ only)",
        "val_mae_inr": m_recent["mae"],
        "val_mape_pct": m_recent["mape"],
        "val_rmse_inr": m_recent["rmse"],
        "delta_mae_vs_base": round(m_recent["mae"] - m_base["mae"], 2),
        "improved": m_recent["mae"] < m_base["mae"],
    })
    logger.info(f"Exp (b1) Val MAE: {m_recent['mae']} ₹/q (delta: {m_recent['mae'] - m_base['mae']:+.2f})")

    # (b2) Recency sample weights: linear/exp ramp up to 1.0 for recent years
    train_dates = pd.to_datetime(X_train_df["week_start"])
    max_train_date = train_dates.max()
    days_diff = (max_train_date - train_dates).dt.days
    recency_weights = np.exp(-days_diff / (365.25 * 3.0)).values  # 3-year half-life
    m_rec_wt, _ = train_and_eval(X_train_df[feature_cols], y_train, X_val_df[feature_cols], y_val, sample_weight=recency_weights)
    results.append({
        "experiment": "b2. Recency Sample Weighting (3-yr half-life)",
        "val_mae_inr": m_rec_wt["mae"],
        "val_mape_pct": m_rec_wt["mape"],
        "val_rmse_inr": m_rec_wt["rmse"],
        "delta_mae_vs_base": round(m_rec_wt["mae"] - m_base["mae"], 2),
        "improved": m_rec_wt["mae"] < m_base["mae"],
    })
    logger.info(f"Exp (b2) Val MAE: {m_rec_wt['mae']} ₹/q (delta: {m_rec_wt['mae'] - m_base['mae']:+.2f})")

    # (c) Era indicator (pre-COVID, COVID, inflation shock, recent)
    def assign_era(date_series):
        yr = pd.to_datetime(date_series).dt.year
        return pd.Categorical(np.where(yr < 2020, "pre_covid", np.where(yr <= 2021, "covid", np.where(yr <= 2023, "inflation_shock", "recent"))))

    X_tr_era = X_train_df[feature_cols].copy()
    X_val_era = X_val_df[feature_cols].copy()
    X_tr_era["era"] = assign_era(X_train_df["week_start"])
    X_val_era["era"] = assign_era(X_val_df["week_start"])
    m_era, _ = train_and_eval(X_tr_era, y_train, X_val_era, y_val, c_cols=cat_cols + ["era"])
    results.append({
        "experiment": "c. Era Indicator Feature",
        "val_mae_inr": m_era["mae"],
        "val_mape_pct": m_era["mape"],
        "val_rmse_inr": m_era["rmse"],
        "delta_mae_vs_base": round(m_era["mae"] - m_base["mae"], 2),
        "improved": m_era["mae"] < m_base["mae"],
    })
    logger.info(f"Exp (c) Val MAE: {m_era['mae']} ₹/q (delta: {m_era['mae'] - m_base['mae']:+.2f})")

    # (d) Fewer or simpler features: core 12 features only
    core_features = [
        "lag_1", "lag_2", "lag_4", "rolling_mean_4", "momentum_1",
        "neighbor_lag1", "market_premium", "lat", "lon", "state", "district", "commodity"
    ]
    core_cols = [c for c in core_features if c in X_train_df.columns]
    m_simple, _ = train_and_eval(X_train_df[core_cols], y_train, X_val_df[core_cols], y_val)
    results.append({
        "experiment": "d. Simpler Core Features Only (12 features)",
        "val_mae_inr": m_simple["mae"],
        "val_mape_pct": m_simple["mape"],
        "val_rmse_inr": m_simple["rmse"],
        "delta_mae_vs_base": round(m_simple["mae"] - m_base["mae"], 2),
        "improved": m_simple["mae"] < m_base["mae"],
    })
    logger.info(f"Exp (d) Val MAE: {m_simple['mae']} ₹/q (delta: {m_simple['mae'] - m_base['mae']:+.2f})")

    # (e) Per-commodity models vs global model
    logger.info("Training separate models per commodity...")
    comm_val_preds = pd.Series(index=X_val_df.index, dtype=float)
    for comm in X_train_df["commodity"].unique():
        tr_c = X_train_df[X_train_df["commodity"] == comm]
        val_c = X_val_df[X_val_df["commodity"] == comm]
        if len(val_c) == 0:
            continue
        c_cols = [c for c in ["state", "district"] if c in tr_c.columns]
        feat_sub = [c for c in feature_cols if c != "commodity"]
        mod_c = lgb.LGBMRegressor(**base_params)
        mod_c.fit(
            tr_c[feat_sub], tr_c[target_col],
            eval_set=[(val_c[feat_sub], val_c[target_col])],
            callbacks=[lgb.early_stopping(50, verbose=False)],
            categorical_feature=c_cols,
        )
        comm_val_preds.loc[val_c.index] = mod_c.predict(val_c[feat_sub])

    val_realized_all = np.exp(y_val)
    val_pred_comm = np.exp(comm_val_preds)
    m_per_comm = compute_metrics(val_realized_all, val_pred_comm)
    results.append({
        "experiment": "e. Per-Commodity Models (8 separate models)",
        "val_mae_inr": m_per_comm["mae"],
        "val_mape_pct": m_per_comm["mape"],
        "val_rmse_inr": m_per_comm["rmse"],
        "delta_mae_vs_base": round(m_per_comm["mae"] - m_base["mae"], 2),
        "improved": m_per_comm["mae"] < m_base["mae"],
    })
    logger.info(f"Exp (e) Val MAE: {m_per_comm['mae']} ₹/q (delta: {m_per_comm['mae'] - m_base['mae']:+.2f})")

    # (f) Predicting change vs last observed price (delta target)
    # delta = target_log - lag_1_log
    X_tr_f = X_train_df.dropna(subset=["lag_1", target_col]).copy()
    X_val_f = X_val_df.dropna(subset=["lag_1", target_col]).copy()
    y_tr_delta = X_tr_f[target_col] - X_tr_f["lag_1"]
    y_val_delta = X_val_f[target_col] - X_val_f["lag_1"]

    m_delta, _ = train_and_eval(
        X_tr_f[feature_cols], y_tr_delta,
        X_val_f[feature_cols], y_val_delta,
        target_is_delta=True,
        last_price_val=X_val_f["lag_1"].values
    )
    results.append({
        "experiment": "f. Predicting Price Change (Delta target)",
        "val_mae_inr": m_delta["mae"],
        "val_mape_pct": m_delta["mape"],
        "val_rmse_inr": m_delta["rmse"],
        "delta_mae_vs_base": round(m_delta["mae"] - m_base["mae"], 2),
        "improved": m_delta["mae"] < m_base["mae"],
    })
    logger.info(f"Exp (f) Val MAE: {m_delta['mae']} ₹/q (delta: {m_delta['mae'] - m_base['mae']:+.2f})")

    results_df = pd.DataFrame(results)
    return results_df


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    features_path = ROOT / "data" / "processed" / "features_h2.parquet"
    logger.info(f"Loading feature table from {features_path}...")
    df = pd.read_parquet(features_path)

    # 1. Error breakdowns on validation
    logger.info("Computing error breakdowns on validation set...")
    # Use baseline predictions in df to get base error
    from src.baselines import add_baseline_predictions
    df = add_baseline_predictions(df, h=2)

    # Load existing LGBM point predictions or compute them
    import joblib
    model_path = ROOT / "models" / "lgb_point_h2.joblib"
    if model_path.exists():
        lgb_model = joblib.load(model_path)
        feature_cols = [c for c in FEATURE_COLS if c in df.columns]
        for c in CATEGORICAL_COLS:
            if c in df.columns:
                df[c] = df[c].astype("category")
        df["pred_lgb_point_log"] = lgb_model.predict(df[feature_cols])

    breakdowns = compute_error_breakdowns(df, pred_col="pred_lgb_point_log", target_col="target_h2")

    tables_dir = ROOT / "reports" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    for k, v in breakdowns.items():
        v.to_csv(tables_dir / f"val_breakdown_{k}.csv", index=False)

    # 2. Run ablation experiments strictly on validation
    ablation_results = run_ablation_experiments(df, h=2)
    ablation_results.to_csv(tables_dir / "ablation_validation_results.csv", index=False)
    logger.info("Ablation results saved to reports/tables/ablation_validation_results.csv")


if __name__ == "__main__":
    main()
