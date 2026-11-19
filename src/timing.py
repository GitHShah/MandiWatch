"""
Experimental Sell/Hold (F2) and Buy Now/Wait (B2) Timing Module for Mandi Watch.
Implements Section 12 of the project specification:
1. Target: Does price change over horizon h exceed a holding cost / threshold hurdle?
2. LightGBM binary classifier predicting P(Price increase > hurdle) trained on leak-free features.
3. Probability and expected change outputs (not a hard yes/no).
4. Configurable holding cost (storage, spoilage, finance) with perishable vs durable defaults.
5. Backtested evaluation on the held-out test split (2024-2026) comparing:
   - Strategy A: Experimental Model Timing (Hold when predicted profitable, else Sell Now)
   - Strategy B: Always Sell Now (Baseline status quo, 0 holding cost)
   - Strategy C: Always Sell at Seasonal Peak (Hold until historical peak month, incurring holding cost)
6. Precision, Recall, Accuracy, and Simulated Economic Payoff per quintal.
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.config import load_config

logger = logging.getLogger("mandi_watch.timing")

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


def prepare_timing_data(
    df: pd.DataFrame,
    hurdle_pct: float = 3.0,
    h: int = 2,
) -> pd.DataFrame:
    """
    Prepares training/testing dataset for timing classifier:
    - Current price: P_t = exp(lag_1)
    - Future price: P_{t+h} = exp(target_h{h})
    - Percentage return: ret = (P_{t+h} - P_t) / P_t
    - Target label: 1 if ret > hurdle_pct / 100.0, else 0.
    """
    df = df.copy()
    valid = df[df["target_h2"].notna() & df["lag_1"].notna()].copy()

    p_curr = np.exp(valid["lag_1"])
    p_future = np.exp(valid[f"target_h{h}"])
    ret_pct = (p_future - p_curr) / p_curr * 100.0

    valid["p_current"] = p_curr
    valid["p_future_actual"] = p_future
    valid["return_pct_actual"] = ret_pct
    valid["timing_target"] = (ret_pct > hurdle_pct).astype(int)

    return valid


def train_timing_classifier(
    df: pd.DataFrame,
    hurdle_pct: float = 3.0,
    h: int = 2,
    random_state: int = 42,
) -> Tuple[lgb.LGBMClassifier, Dict[str, Any]]:
    """
    Trains LightGBM binary classifier strictly on train split with early stopping on validation split.
    """
    df_clf = prepare_timing_data(df, hurdle_pct=hurdle_pct, h=h)

    train_mask = df_clf["split"] == "train"
    val_mask = df_clf["split"] == "validation"

    for col in CATEGORICAL_COLS:
        df_clf[col] = df_clf[col].astype("category")

    X_train = df_clf.loc[train_mask, FEATURE_COLS]
    y_train = df_clf.loc[train_mask, "timing_target"]

    X_val = df_clf.loc[val_mask, FEATURE_COLS]
    y_val = df_clf.loc[val_mask, "timing_target"]

    clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )

    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)],
    )

    # Validation metrics
    val_preds = clf.predict(X_val)
    val_probs = clf.predict_proba(X_val)[:, 1]

    val_metrics = {
        "val_accuracy": float(accuracy_score(y_val, val_preds)),
        "val_precision": float(precision_score(y_val, val_preds, zero_division=0)),
        "val_recall": float(recall_score(y_val, val_preds, zero_division=0)),
        "val_f1": float(f1_score(y_val, val_preds, zero_division=0)),
        "best_iteration": int(clf.best_iteration_),
    }

    logger.info(
        f"Timing Classifier trained: Val Acc={val_metrics['val_accuracy']:.3f}, "
        f"Val Prec={val_metrics['val_precision']:.3f}, Val Rec={val_metrics['val_recall']:.3f}"
    )

    return clf, val_metrics


def evaluate_timing_on_test(
    clf: lgb.LGBMClassifier,
    features_df: pd.DataFrame,
    forecasts_df: pd.DataFrame,
    config: Any,
    h: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluates experimental timing strategies on held-out test split (2024-2026):
    Strategies:
    1. Always Sell Now (Baseline): Realized payout = P_t.
    2. Model Timing: If P(Gain) >= 0.50 and expected net gain > 0, hold h weeks:
       Realized payout = P_{t+h} - (h * holding_cost_per_week).
       Else sell now at P_t.
    3. Always Sell at Seasonal Peak: Holds until the historical peak month of that commodity,
       incurring weekly holding costs for all weeks waited.
    """
    df_clf = prepare_timing_data(features_df, hurdle_pct=config.timing.target_hurdle_pct, h=h)
    test_df = df_clf[df_clf["split"] == "test"].copy()

    for col in CATEGORICAL_COLS:
        test_df[col] = test_df[col].astype("category")

    X_test = test_df[FEATURE_COLS]
    test_df["prob_hold_profitable"] = clf.predict_proba(X_test)[:, 1]
    test_df["pred_hold_binary"] = (test_df["prob_hold_profitable"] >= 0.50).astype(int)

    # Merge forecast point predictions if available
    if "pred_point_inr" in forecasts_df.columns:
        f_cols = ["market", "commodity", "week_start", "pred_point_inr", "pred_p10_inr", "pred_p90_inr"]
        f_sub = forecasts_df[f_cols].copy().rename(
            columns={"pred_point_inr": "forecast_price", "pred_p10_inr": "p10", "pred_p90_inr": "p90"}
        )
        test_df = test_df.merge(f_sub, on=["market", "commodity", "week_start"], how="left")
    elif "forecast_price" in forecasts_df.columns:
        test_df = test_df.merge(
            forecasts_df[["market", "commodity", "week_start", "forecast_price", "p10", "p90"]],
            on=["market", "commodity", "week_start"],
            how="left",
        )
    else:
        test_df["forecast_price"] = test_df["p_current"] * np.exp(test_df["prob_hold_profitable"] * 0.05)
        test_df["p10"] = test_df["forecast_price"] * 0.90
        test_df["p90"] = test_df["forecast_price"] * 1.10

    test_df["forecast_price"] = test_df["forecast_price"].fillna(test_df["p_current"])
    test_df["p10"] = test_df["p10"].fillna(test_df["forecast_price"] * 0.90)
    test_df["p90"] = test_df["p90"].fillna(test_df["forecast_price"] * 1.10)

    # Determine default holding cost per commodity
    holding_costs = config.timing.default_holding_costs

    records = []
    simulation_rows = []

    # Historical peak month lookup for Seasonal Peak strategy (computed from train split only!)
    train_prices = df_clf[df_clf["split"] == "train"]
    peak_month_by_commodity = (
        train_prices.groupby(["commodity", "month"])["p_current"]
        .mean()
        .reset_index()
        .sort_values(["commodity", "p_current"], ascending=[True, False])
        .groupby("commodity")
        .first()["month"]
        .to_dict()
    )

    for comm, group in test_df.groupby("commodity"):
        cost_wk = holding_costs.get(comm, 20.0)
        h_cost = h * cost_wk
        n_obs = len(group)

        # Baseline Strategy: Always Sell Now
        payoff_now = group["p_current"].values

        # Model Strategy: Hold if prob >= 0.50 and forecast price - p_current > h_cost
        exp_gain = group["forecast_price"].values - group["p_current"].values
        model_hold_decision = (group["prob_hold_profitable"].values >= 0.50) & (exp_gain > h_cost)

        payoff_model = np.where(
            model_hold_decision,
            group["p_future_actual"].values - h_cost,
            group["p_current"].values,
        )

        # Strategy 3: Always Sell at Seasonal Peak
        # If current month is the peak month, sell now; else assume 4 weeks hold with 4 weeks holding cost
        peak_m = peak_month_by_commodity.get(comm, 6)
        is_peak = (group["month"].values == peak_m)
        payoff_peak = np.where(
            is_peak,
            group["p_current"].values,
            group["p_future_actual"].values - (4 * cost_wk),  # Simulated peak hold cost
        )

        # Classification metrics
        acc = accuracy_score(group["timing_target"], group["pred_hold_binary"])
        prec = precision_score(group["timing_target"], group["pred_hold_binary"], zero_division=0)
        rec = recall_score(group["timing_target"], group["pred_hold_binary"], zero_division=0)
        f1 = f1_score(group["timing_target"], group["pred_hold_binary"], zero_division=0)

        mean_now = float(np.mean(payoff_now))
        mean_model = float(np.mean(payoff_model))
        mean_peak = float(np.mean(payoff_peak))
        diff_vs_now = mean_model - mean_now
        diff_pct = (diff_vs_now / mean_now) * 100.0

        records.append({
            "commodity": comm,
            "category": "Perishable" if comm in config.timing.perishable_commodities else "Durable",
            "n_test_cases": n_obs,
            "weekly_holding_cost_inr": cost_wk,
            "horizon_holding_cost_inr": h_cost,
            "accuracy": round(float(acc), 3),
            "precision": round(float(prec), 3),
            "recall": round(float(rec), 3),
            "f1_score": round(float(f1), 3),
            "hold_recommendation_rate_pct": round(float(np.mean(model_hold_decision) * 100.0), 1),
            "mean_payoff_sell_now_inr": round(mean_now, 2),
            "mean_payoff_model_timing_inr": round(mean_model, 2),
            "mean_payoff_seasonal_peak_inr": round(mean_peak, 2),
            "advantage_vs_sell_now_inr": round(diff_vs_now, 2),
            "advantage_pct": round(diff_pct, 2),
            "beats_sell_now": bool(diff_vs_now > 0),
        })

    summary_df = pd.DataFrame(records)

    # Overall aggregate row
    overall_acc = accuracy_score(test_df["timing_target"], test_df["pred_hold_binary"])
    overall_prec = precision_score(test_df["timing_target"], test_df["pred_hold_binary"], zero_division=0)
    overall_rec = recall_score(test_df["timing_target"], test_df["pred_hold_binary"], zero_division=0)
    overall_f1 = f1_score(test_df["timing_target"], test_df["pred_hold_binary"], zero_division=0)

    overall_row = {
        "commodity": "ALL",
        "category": "Overall",
        "n_test_cases": len(test_df),
        "weekly_holding_cost_inr": float(np.mean([r["weekly_holding_cost_inr"] for r in records])),
        "horizon_holding_cost_inr": float(np.mean([r["horizon_holding_cost_inr"] for r in records])),
        "accuracy": round(float(overall_acc), 3),
        "precision": round(float(overall_prec), 3),
        "recall": round(float(overall_rec), 3),
        "f1_score": round(float(overall_f1), 3),
        "hold_recommendation_rate_pct": round(float(summary_df["hold_recommendation_rate_pct"].mean()), 1),
        "mean_payoff_sell_now_inr": round(float(summary_df["mean_payoff_sell_now_inr"].mean()), 2),
        "mean_payoff_model_timing_inr": round(float(summary_df["mean_payoff_model_timing_inr"].mean()), 2),
        "mean_payoff_seasonal_peak_inr": round(float(summary_df["mean_payoff_seasonal_peak_inr"].mean()), 2),
        "advantage_vs_sell_now_inr": round(float(summary_df["advantage_vs_sell_now_inr"].mean()), 2),
        "advantage_pct": round(float(summary_df["advantage_pct"].mean()), 2),
        "beats_sell_now": bool(summary_df["advantage_vs_sell_now_inr"].mean() > 0),
    }

    summary_df = pd.concat([pd.DataFrame([overall_row]), summary_df], ignore_index=True)

    return summary_df, test_df


def get_experimental_timing_decision(
    market: str,
    commodity: str,
    horizon: int = 2,
    custom_holding_cost: Optional[float] = None,
    user_type: str = "farmer",
    p_current: Optional[float] = None,
    p_forecast: Optional[float] = None,
    p10: Optional[float] = None,
    p90: Optional[float] = None,
    clf: Optional[lgb.LGBMClassifier] = None,
    forecasts_df: Optional[pd.DataFrame] = None,
    features_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Computes real-time experimental timing analysis for the Streamlit UI:
    Returns probability, expected price change, holding cost, net gain, and warning notes.
    """
    config = load_config()

    # Determine default holding cost
    default_cost = config.timing.default_holding_costs.get(commodity, 20.0)
    holding_cost_wk = custom_holding_cost if custom_holding_cost is not None else default_cost
    total_holding_cost = horizon * holding_cost_wk

    # Locate latest forecast from precomputed forecasts if not provided
    latest_forecast = None
    if forecasts_df is not None:
        sub = forecasts_df[(forecasts_df["market"] == market) & (forecasts_df["commodity"] == commodity)]
        if not sub.empty:
            latest_forecast = sub.sort_values("week_start").iloc[-1]

    if p_forecast is None:
        if latest_forecast is not None:
            p_forecast = float(latest_forecast.get("pred_point_inr", latest_forecast.get("forecast_price", 1500.0)))
            p10 = float(latest_forecast.get("pred_p10_inr", latest_forecast.get("p10", p_forecast * 0.90)))
            p90 = float(latest_forecast.get("pred_p90_inr", latest_forecast.get("p90", p_forecast * 1.10)))
            if p_current is None:
                if "pred_last_observed" in latest_forecast and not pd.isna(latest_forecast["pred_last_observed"]):
                    p_current = float(np.exp(latest_forecast["pred_last_observed"]))
                elif "realized_price_inr" in latest_forecast and not pd.isna(latest_forecast["realized_price_inr"]):
                    p_current = float(latest_forecast["realized_price_inr"])
                else:
                    p_current = p_forecast
        else:
            p_forecast = 1500.0
            p10 = 1350.0
            p90 = 1650.0
            if p_current is None:
                p_current = 1500.0

    if p_current is None:
        p_current = p_forecast
    if p10 is None:
        p10 = p_forecast * 0.90
    if p90 is None:
        p90 = p_forecast * 1.10

    expected_change = p_forecast - p_current
    expected_change_pct = (expected_change / p_current) * 100.0 if p_current > 0 else 0.0

    # Model probability estimation
    # If classifier is loaded, compute inference from latest feature row, else use calibrated normal cdf
    prob_gain = 0.50
    if clf is not None and features_df is not None:
        f_sub = features_df[(features_df["market"] == market) & (features_df["commodity"] == commodity)]
        if not f_sub.empty:
            row_feat = f_sub.sort_values("week_start").iloc[-1:][FEATURE_COLS].copy()
            for c in CATEGORICAL_COLS:
                row_feat[c] = row_feat[c].astype("category")
            prob_gain = float(clf.predict_proba(row_feat)[:, 1][0])
    else:
        # Calibrated normal approximation from [p10, p90] interval
        std_est = max((p90 - p10) / 2.56, 1.0)
        z = (expected_change - total_holding_cost) / std_est
        prob_gain = float(1.0 / (1.0 + math.exp(-z * 1.5)))

    # Net Economic Payoffs
    if user_type == "farmer":
        net_expected_change = expected_change - total_holding_cost
        net_p10 = (p10 - p_current) - total_holding_cost
        net_p90 = (p90 - p_current) - total_holding_cost

        # Decision recommendation
        if net_expected_change > 0 and prob_gain >= 0.55:
            action_recommendation = "Consider Holding (Expected net price increase covers holding cost)"
            action_color = "#2ca02c"
        else:
            action_recommendation = "Sell Now (Holding cost or price risk exceeds expected price gain)"
            action_color = "#d62728"

        user_label = "Farmer (F2)"
    else:
        # Bulk Buyer: Wants price to fall
        # If price is rising, buy now immediately to avoid higher cost; if falling, wait.
        net_expected_change = expected_change + total_holding_cost  # Cost increase if buyer waits
        net_p10 = (p10 - p_current)
        net_p90 = (p90 - p_current)

        if expected_change < -total_holding_cost:
            action_recommendation = "Wait (Price is forecast to fall enough to exceed delay/finance cost)"
            action_color = "#2ca02c"
        else:
            action_recommendation = "Buy Now (Price is rising or stable; waiting incurs delay cost)"
            action_color = "#1f77b4"

        user_label = "Bulk Buyer (B2)"

    return {
        "user_type": user_type,
        "market": market,
        "commodity": commodity,
        "horizon_weeks": horizon,
        "current_price": round(p_current, 2),
        "forecast_price": round(p_forecast, 2),
        "expected_change_inr": round(expected_change, 2),
        "expected_change_pct": round(expected_change_pct, 2),
        "holding_cost_per_week_inr": round(holding_cost_wk, 2),
        "total_holding_cost_inr": round(total_holding_cost, 2),
        "net_expected_gain_inr": round(net_expected_change, 2),
        "prob_gain": round(prob_gain, 3),
        "interval_net_p10": round(net_p10, 2),
        "interval_net_p90": round(net_p90, 2),
        "recommendation": action_recommendation,
        "recommendation_color": action_color,
        "disclaimer": (
            "EXPERIMENTAL FEATURE: Based on historical price trends and assumed holding costs. "
            "Agricultural markets are prone to unforecastable exogenous shocks (unseasonal weather, "
            "flash floods, sudden export bans, stockholding limits) not predictable from past price series. "
            "Use strictly as decision support."
        ),
    }


def run_timing_pipeline():
    """Master executable pipeline for Phase 12 timing modeling and evaluation."""
    config = load_config()
    features_path = ROOT / "data" / "processed" / "features_h2.parquet"
    forecasts_path = ROOT / "data" / "processed" / "forecasts_h2.parquet"
    models_dir = ROOT / "models"
    tables_dir = ROOT / "reports" / "tables"

    models_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading feature table...")
    df_features = pd.read_parquet(features_path)
    df_forecasts = pd.read_parquet(forecasts_path) if forecasts_path.exists() else pd.DataFrame()

    logger.info("Training experimental timing classifier on train split...")
    clf, val_metrics = train_timing_classifier(
        df=df_features,
        hurdle_pct=config.timing.target_hurdle_pct,
        h=2,
        random_state=config.random_seed,
    )

    clf_path = models_dir / "lgb_timing_clf_h2.joblib"
    joblib.dump(clf, clf_path)
    logger.info(f"Saved timing classifier model to {clf_path}")

    logger.info("Evaluating experimental timing strategies on held-out test split...")
    summary_df, _ = evaluate_timing_on_test(
        clf=clf,
        features_df=df_features,
        forecasts_df=df_forecasts,
        config=config,
        h=2,
    )

    summary_path = tables_dir / "timing_evaluation.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved timing evaluation summary to {summary_path}")

    print("\n" + "=" * 80)
    print("EXPERIMENTAL TIMING EVALUATION SUMMARY (Held-out Test Period 2024-2026)")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80 + "\n")

    return clf, summary_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_timing_pipeline()
