"""
Forecast and Recommender evaluation metrics module for Mandi Watch.
Implements:
- Section 8.5: Point forecast and interval coverage metrics (MAE, MAPE, RMSE, Coverage).
- Section 10: Recommender evaluation on held-out test period:
  1. Regret vs Oracle, Nearest market, and Historical best.
  2. Top-1 and Top-3 hit rates, Spearman ranking correlation.
  3. Paired hypothesis tests (Wilcoxon and Paired t-test with Cohen's d).
  4. Sensitivity analysis across transport rates (0.0, 0.15, 0.25, 0.50 INR/q/km).
  5. Cross-market price dispersion (Spread) vs Forecast Error (MAE).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import sys

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_rel, wilcoxon

from src.config import load_config
from src.geo import haversine_distance, road_distance

logger = logging.getLogger(__name__)
config = load_config()


def compute_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
) -> Dict[str, float]:
    """
    Computes standard point forecast metrics in INR per quintal:
    - MAE: Mean Absolute Error (INR/quintal)
    - MAPE: Mean Absolute Percentage Error (%)
    - RMSE: Root Mean Squared Error (INR/quintal)
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    # Valid non-NaN pairs
    valid_mask = (~np.isnan(y_true_arr)) & (~np.isnan(y_pred_arr)) & (y_true_arr > 0)
    if not np.any(valid_mask):
        return {"mae": np.nan, "mape": np.nan, "rmse": np.nan, "n_obs": 0}

    y_t = y_true_arr[valid_mask]
    y_p = y_pred_arr[valid_mask]

    err = y_t - y_p
    abs_err = np.abs(err)
    mae = float(np.mean(abs_err))
    mape = float(np.mean(abs_err / y_t) * 100.0)
    rmse = float(np.sqrt(np.mean(err ** 2)))

    return {
        "mae": round(mae, 2),
        "mape": round(mape, 2),
        "rmse": round(rmse, 2),
        "n_obs": int(len(y_t)),
    }


def compute_coverage(
    y_true: Union[np.ndarray, pd.Series],
    p10: Union[np.ndarray, pd.Series],
    p90: Union[np.ndarray, pd.Series],
) -> Dict[str, float]:
    """
    Computes empirical coverage of the 10-90 uncertainty interval:
    Target nominal coverage: 80%.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    p10_arr = np.asarray(p10, dtype=float)
    p90_arr = np.asarray(p90, dtype=float)

    valid_mask = (~np.isnan(y_true_arr)) & (~np.isnan(p10_arr)) & (~np.isnan(p90_arr))
    if not np.any(valid_mask):
        return {"coverage_80": np.nan, "interval_width": np.nan, "n_obs": 0}

    y_t = y_true_arr[valid_mask]
    lo = p10_arr[valid_mask]
    hi = p90_arr[valid_mask]

    covered = (y_t >= lo) & (y_t <= hi)
    coverage = float(np.mean(covered) * 100.0)
    mean_width = float(np.mean(hi - lo))

    return {
        "coverage_80": round(coverage, 2),
        "interval_width": round(mean_width, 2),
        "n_obs": int(len(y_t)),
    }


def evaluate_forecasts(
    df: pd.DataFrame,
    pred_col: str,
    target_col: str = "target_h2",
    model_name: str = "model",
    splits: Optional[List[str]] = None,
    is_log_scale: bool = True,
) -> pd.DataFrame:
    """
    Evaluates point predictions against realized future price.
    Returns standard metrics table across splits and commodities.
    """
    if splits is None:
        splits = ["validation", "test"]

    records = []
    eval_df = df.copy()
    if is_log_scale:
        eval_df["_realized_inr"] = np.exp(eval_df[target_col])
        eval_df["_pred_inr"] = np.exp(eval_df[pred_col])
    else:
        eval_df["_realized_inr"] = eval_df[target_col]
        eval_df["_pred_inr"] = eval_df[pred_col]

    eval_df = eval_df.dropna(subset=["_realized_inr", "_pred_inr"])
    commodities = ["ALL"] + sorted(eval_df["commodity"].dropna().unique().tolist())

    for split in splits:
        split_df = eval_df[eval_df["split"] == split]
        if len(split_df) == 0:
            continue

        for comm in commodities:
            sub = split_df if comm == "ALL" else split_df[split_df["commodity"] == comm]
            if len(sub) == 0:
                continue

            metrics = compute_metrics(sub["_realized_inr"], sub["_pred_inr"])
            records.append({
                "model": model_name,
                "split": split,
                "commodity": comm,
                "n_obs": metrics["n_obs"],
                "mae_inr": metrics["mae"],
                "mape_pct": metrics["mape"],
                "rmse_inr": metrics["rmse"],
            })

    return pd.DataFrame(records)


# =========================================================================
# SECTION 10: RECOMMENDER EVALUATION ON HELD-OUT TEST PERIOD
# =========================================================================


def simulate_recommender_test_period(
    forecasts_df: pd.DataFrame,
    centroids_df: pd.DataFrame,
    transport_rate: float = 0.25,
    radius_km: float = 150.0,
    road_factor: float = 1.35,
    min_candidates: int = 2,
) -> pd.DataFrame:
    """
    Simulates F1 farmer recommendations across all (test_week, commodity, district)
    combinations in the held-out test split.

    For each case:
    - Finds all candidate markets within radius with valid forecasts & realized prices.
    - Evaluates:
      1. Model Recommendation (highest predicted net price).
      2. Nearest Market (minimum distance).
      3. Historical Best Market (highest expanding historical mean).
      4. Oracle Market (highest realized net price in hindsight).
    - Computes regret, hit rates, rank correlations, and cross-market price spread.
    """
    # Filter strictly to test split with observed target prices and predictions
    test_data = forecasts_df[
        (forecasts_df["split"] == "test")
        & (forecasts_df["realized_price_inr"].notnull())
        & (forecasts_df["pred_point_inr"].notnull())
    ].copy()

    # Pre-extract district centroid coordinates
    centroids_clean = centroids_df.drop_duplicates(subset=["state", "district"]).reset_index(drop=True)
    dist_states = centroids_clean["state"].values
    dist_names = centroids_clean["district"].values
    dist_lats = centroids_clean["lat"].values
    dist_lons = centroids_clean["lon"].values
    n_districts = len(centroids_clean)

    sim_records = []

    # Group by test week and commodity
    grouped = test_data.groupby(["week_start", "commodity"])

    for (week_start, comm), group in grouped:
        mkt_names = group["market"].values
        mkt_districts = group["district"].values
        mkt_states = group["state"].values
        mkt_lats = group["lat"].values
        mkt_lons = group["lon"].values
        pred_prices = group["pred_point_inr"].values
        realized_prices = group["realized_price_inr"].values
        hist_prices = group["pred_historical_mean"].values if "pred_historical_mean" in group.columns else pred_prices
        n_mkts = len(group)

        if n_mkts < min_candidates:
            continue

        # Vectorized pairwise haversine distance between all districts and active markets
        lat1 = dist_lats[:, None]
        lon1 = dist_lons[:, None]
        lat2 = mkt_lats[None, :]
        lon2 = mkt_lons[None, :]

        h_dists = haversine_distance(lat1, lon1, lat2, lon2)
        r_dists = road_distance(h_dists, road_factor=road_factor)

        # Loop through districts
        for i in range(n_districts):
            d_state = dist_states[i]
            d_dist = dist_names[i]
            dists = r_dists[i, :]

            # Filter candidates within radius
            cand_mask = dists <= radius_km
            n_cands = int(np.sum(cand_mask))

            if n_cands < min_candidates:
                continue

            c_names = mkt_names[cand_mask]
            c_dists = dists[cand_mask]
            c_pred = pred_prices[cand_mask]
            c_real = realized_prices[cand_mask]
            c_hist = hist_prices[cand_mask]

            # Transport cost per candidate
            c_trans = c_dists * transport_rate

            # Net prices
            pred_net = c_pred - c_trans
            real_net = c_real - c_trans
            hist_net = c_hist - c_trans

            # 1. Oracle Best (Hindsight maximum realized net price)
            oracle_idx = int(np.argmax(real_net))
            oracle_mkt = c_names[oracle_idx]
            oracle_real_net = float(real_net[oracle_idx])
            oracle_real_price = float(c_real[oracle_idx])

            # 2. Model Recommended (argmax predicted net price)
            rec_idx = int(np.argmax(pred_net))
            rec_mkt = c_names[rec_idx]
            rec_real_net = float(real_net[rec_idx])
            rec_dist = float(c_dists[rec_idx])
            rec_trans = float(c_trans[rec_idx])

            # 3. Nearest Market (argmin distance)
            near_idx = int(np.argmin(c_dists))
            near_mkt = c_names[near_idx]
            near_real_net = float(real_net[near_idx])
            near_dist = float(c_dists[near_idx])

            # 4. Historical Best (argmax historical net price)
            hist_idx = int(np.argmax(hist_net))
            hist_mkt = c_names[hist_idx]
            hist_real_net = float(real_net[hist_idx])

            # Metrics
            regret_rec = max(0.0, oracle_real_net - rec_real_net)
            regret_near = max(0.0, oracle_real_net - near_real_net)
            regret_hist = max(0.0, oracle_real_net - hist_real_net)

            denom = oracle_real_net if oracle_real_net > 0 else 1.0
            regret_rec_pct = (regret_rec / denom) * 100.0
            regret_near_pct = (regret_near / denom) * 100.0
            regret_hist_pct = (regret_hist / denom) * 100.0

            delta_vs_nearest = rec_real_net - near_real_net
            delta_vs_hist = rec_real_net - hist_real_net

            # Top-1 and Top-3 hit
            is_top1 = 1 if rec_idx == oracle_idx else 0
            # Top-3 threshold
            sorted_real_net = np.sort(real_net)[::-1]
            top3_cutoff = sorted_real_net[min(2, len(sorted_real_net) - 1)]
            is_top3 = 1 if rec_real_net >= top3_cutoff else 0

            # Spearman correlation between predicted net rank and actual realized net rank
            if n_cands >= 3:
                sc, _ = spearmanr(pred_net, real_net)
                spearman_val = float(sc) if not np.isnan(sc) else 0.0
            else:
                spearman_val = np.nan

            # Local cross-market price spread
            price_spread = float(np.max(c_real) - np.min(c_real))

            sim_records.append({
                "week_start": week_start,
                "commodity": comm,
                "district": d_dist,
                "state": d_state,
                "n_candidates": n_cands,
                "oracle_market": oracle_mkt,
                "oracle_net_price": oracle_real_net,
                "rec_market": rec_mkt,
                "rec_net_price": rec_real_net,
                "rec_dist_km": rec_dist,
                "rec_trans_cost": rec_trans,
                "near_market": near_mkt,
                "near_net_price": near_real_net,
                "near_dist_km": near_dist,
                "hist_market": hist_mkt,
                "hist_net_price": hist_real_net,
                "regret_rec_inr": regret_rec,
                "regret_rec_pct": regret_rec_pct,
                "regret_near_inr": regret_near,
                "regret_near_pct": regret_near_pct,
                "regret_hist_inr": regret_hist,
                "regret_hist_pct": regret_hist_pct,
                "delta_vs_nearest_inr": delta_vs_nearest,
                "delta_vs_hist_inr": delta_vs_hist,
                "is_top1_hit": is_top1,
                "is_top3_hit": is_top3,
                "spearman_corr": spearman_val,
                "price_spread_inr": price_spread,
            })

    sim_df = pd.DataFrame(sim_records)
    logger.info(f"Simulated {len(sim_df)} recommender test decisions across {grouped.ngroups} (week, commodity) cohorts.")
    return sim_df


def compute_recommender_metrics(sim_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes overall and per-commodity recommender benchmark performance:
    - Regret of Recommended Market vs Nearest Market vs Historical Best.
    - Realized Net Price Advantage vs Nearest Market.
    - Top-1 and Top-3 Hit Rates.
    - Mean Spearman Rank Correlation.
    """
    def _agg(sub: pd.DataFrame) -> Dict[str, Any]:
        return {
            "n_cases": len(sub),
            "mean_rec_regret_inr": round(float(sub["regret_rec_inr"].mean()), 2),
            "median_rec_regret_inr": round(float(sub["regret_rec_inr"].median()), 2),
            "mean_rec_regret_pct": round(float(sub["regret_rec_pct"].mean()), 2),
            "mean_near_regret_inr": round(float(sub["regret_near_inr"].mean()), 2),
            "mean_near_regret_pct": round(float(sub["regret_near_pct"].mean()), 2),
            "mean_hist_regret_inr": round(float(sub["regret_hist_inr"].mean()), 2),
            "mean_advantage_vs_near_inr": round(float(sub["delta_vs_nearest_inr"].mean()), 2),
            "top1_hit_rate_pct": round(float(sub["is_top1_hit"].mean() * 100.0), 2),
            "top3_hit_rate_pct": round(float(sub["is_top3_hit"].mean() * 100.0), 2),
            "mean_spearman_corr": round(float(sub["spearman_corr"].dropna().mean()), 3),
        }

    # Overall
    overall_dict = _agg(sim_df)
    overall_dict["commodity"] = "ALL"
    overall_df = pd.DataFrame([overall_dict])

    # Per commodity
    by_comm_list = []
    for comm, sub in sim_df.groupby("commodity"):
        c_dict = _agg(sub)
        c_dict["commodity"] = comm
        by_comm_list.append(c_dict)

    by_comm_df = pd.DataFrame(by_comm_list).sort_values("mean_advantage_vs_near_inr", ascending=False).reset_index(drop=True)

    summary_df = pd.concat([overall_df, by_comm_df], ignore_index=True)
    return summary_df, by_comm_df


def run_recommender_hypothesis_test(sim_df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs paired hypothesis testing between the Recommended market's realized net price
    and the Nearest market's realized net price.
    H0: Recommended net price == Nearest net price (Delta == 0).
    Runs:
    - Paired Wilcoxon Signed-Rank Test
    - Paired t-test
    - Cohen's d effect size: mean(Delta) / std(Delta)
    """
    records = []
    commodities = ["ALL"] + sorted(sim_df["commodity"].unique().tolist())

    for comm in commodities:
        sub = sim_df if comm == "ALL" else sim_df[sim_df["commodity"] == comm]
        diffs = sub["delta_vs_nearest_inr"].values
        n = len(diffs)

        if n < 10 or np.all(diffs == 0):
            continue

        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
        cohens_d = mean_diff / std_diff if std_diff > 1e-6 else 0.0

        # Paired t-test
        t_stat, t_p = ttest_rel(sub["rec_net_price"], sub["near_net_price"])

        # Paired Wilcoxon test (filtering zero differences as standard)
        non_zero_diffs = diffs[diffs != 0]
        if len(non_zero_diffs) >= 10:
            w_res = wilcoxon(diffs, alternative="two-sided")
            w_stat, w_p = float(w_res.statistic), float(w_res.pvalue)
        else:
            w_stat, w_p = np.nan, np.nan

        sig = bool(w_p < 0.05 if not np.isnan(w_p) else t_p < 0.05)
        if mean_diff > 0 and sig:
            interpretation = f"Recommended market significantly beats nearest market (+₹{mean_diff:.2f}/q, d={cohens_d:.3f})"
        elif mean_diff < 0 and sig:
            interpretation = f"Nearest market significantly beats recommended market (-₹{abs(mean_diff):.2f}/q, d={cohens_d:.3f})"
        else:
            interpretation = f"No significant difference between recommended and nearest market (mean diff: ₹{mean_diff:.2f}/q, p={w_p:.4f})"

        records.append({
            "commodity": comm,
            "n_cases": n,
            "mean_diff_inr": round(mean_diff, 2),
            "std_diff_inr": round(std_diff, 2),
            "cohens_d": round(cohens_d, 3),
            "t_statistic": round(float(t_stat), 3),
            "t_p_value": float(t_p),
            "wilcoxon_stat": round(w_stat, 2) if not np.isnan(w_stat) else np.nan,
            "wilcoxon_p_value": float(w_p) if not np.isnan(w_p) else np.nan,
            "significant_05": sig,
            "interpretation": interpretation,
        })

    return pd.DataFrame(records)


def run_sensitivity_analysis(
    forecasts_df: pd.DataFrame,
    centroids_df: pd.DataFrame,
    rates: Optional[List[float]] = None,
    base_rate: float = 0.25,
    radius_km: float = 150.0,
) -> pd.DataFrame:
    """
    Sensitivity analysis across transport rates [0.0, 0.15, 0.25, 0.50] INR/q/km.
    Computes:
    - How frequently the recommended market changes compared to the baseline 0.25 rate.
    - Mean regret and mean net price at each transport rate.
    """
    if rates is None:
        rates = [0.0, 0.15, 0.25, 0.50]

    rate_results = {}
    for r in rates:
        logger.info(f"Running simulation for transport rate: ₹{r:.2f}/q/km...")
        sim_r = simulate_recommender_test_period(
            forecasts_df=forecasts_df,
            centroids_df=centroids_df,
            transport_rate=r,
            radius_km=radius_km,
        )
        rate_results[r] = sim_r

    base_sim = rate_results[base_rate]
    base_mkts = base_sim["rec_market"].values

    records = []
    for r in rates:
        sim_r = rate_results[r]
        r_mkts = sim_r["rec_market"].values

        # Recommendation turnover vs base_rate
        if len(r_mkts) == len(base_mkts):
            turnover_pct = float(np.mean(r_mkts != base_mkts) * 100.0)
        else:
            turnover_pct = np.nan

        mean_regret = float(sim_r["regret_rec_inr"].mean())
        mean_regret_pct = float(sim_r["regret_rec_pct"].mean())
        mean_near_regret = float(sim_r["regret_near_inr"].mean())
        mean_adv = float(sim_r["delta_vs_nearest_inr"].mean())
        mean_dist = float(sim_r["rec_dist_km"].mean())

        records.append({
            "transport_rate": r,
            "n_cases": len(sim_r),
            "turnover_vs_base_pct": round(turnover_pct, 1),
            "mean_rec_regret_inr": round(mean_regret, 2),
            "mean_rec_regret_pct": round(mean_regret_pct, 2),
            "mean_near_regret_inr": round(mean_near_regret, 2),
            "advantage_vs_nearest_inr": round(mean_adv, 2),
            "mean_recommended_distance_km": round(mean_dist, 1),
        })

    return pd.DataFrame(records)


def compute_spread_vs_forecast_error(
    sim_df: pd.DataFrame,
    val_mae_dict: Dict[str, float],
) -> pd.DataFrame:
    """
    Compares the local cross-market price dispersion (Spread = max - min)
    within the 150km radius with the model's forecast error (MAE).
    If Spread < Error, the local price ranking is weak for that commodity.
    """
    records = []
    for comm, sub in sim_df.groupby("commodity"):
        mean_spread = float(sub["price_spread_inr"].mean())
        median_spread = float(sub["price_spread_inr"].median())
        val_mae = float(val_mae_dict.get(comm, 100.0))

        spread_mae_ratio = mean_spread / val_mae if val_mae > 0 else 0.0
        ranking_power = "Strong" if spread_mae_ratio >= 1.5 else ("Moderate" if spread_mae_ratio >= 1.0 else "Weak")

        records.append({
            "commodity": comm,
            "n_cases": len(sub),
            "mean_price_spread_inr": round(mean_spread, 2),
            "median_price_spread_inr": round(median_spread, 2),
            "val_mae_inr": round(val_mae, 2),
            "spread_to_mae_ratio": round(spread_mae_ratio, 2),
            "ranking_power": ranking_power,
            "flag_spread_below_mae": bool(mean_spread < val_mae),
        })

    return pd.DataFrame(records).sort_values("spread_to_mae_ratio", ascending=False).reset_index(drop=True)


# =========================================================================
# VISUALIZATION FUNCTIONS (SAVED TO reports/figures/)
# =========================================================================


def plot_recommender_regret_cdf(sim_df: pd.DataFrame, output_path: Path) -> None:
    """Plots empirical CDF of regret: Model vs Nearest Market vs Historical Best."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    rec_regret = np.sort(sim_df["regret_rec_inr"].values)
    near_regret = np.sort(sim_df["regret_near_inr"].values)
    hist_regret = np.sort(sim_df["regret_hist_inr"].values)

    cdf_rec = np.arange(1, len(rec_regret) + 1) / len(rec_regret)
    cdf_near = np.arange(1, len(near_regret) + 1) / len(near_regret)
    cdf_hist = np.arange(1, len(hist_regret) + 1) / len(hist_regret)

    ax.plot(rec_regret, cdf_rec, label=f"Model Recommendation (Mean Regret: ₹{np.mean(rec_regret):.1f}/q)", color="#2ca02c", lw=2.5)
    ax.plot(near_regret, cdf_near, label=f"Nearest Market Baseline (Mean Regret: ₹{np.mean(near_regret):.1f}/q)", color="#1f77b4", lw=2, linestyle="--")
    ax.plot(hist_regret, cdf_hist, label=f"Historical Best Baseline (Mean Regret: ₹{np.mean(hist_regret):.1f}/q)", color="#ff7f0e", lw=2, linestyle=":")

    ax.set_xlim(0, float(np.percentile(near_regret, 95)))
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Realized Regret vs Oracle Best (₹ / quintal)", fontsize=11)
    ax.set_ylabel("Empirical Cumulative Probability", fontsize=11)
    ax.set_title("Recommender Evaluation: Cumulative Distribution of Realized Regret (Held-out Test Period)", fontsize=12, pad=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved regret CDF plot to {output_path}")


def plot_sensitivity_curve(sens_df: pd.DataFrame, output_path: Path) -> None:
    """Plots sensitivity of recommendation turnover and net price advantage across transport rates."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(8, 5))

    rates = sens_df["transport_rate"].values
    turnovers = sens_df["turnover_vs_base_pct"].values
    advantages = sens_df["advantage_vs_nearest_inr"].values

    color1 = "#1f77b4"
    ax1.set_xlabel("Assumed Transport Rate (₹ / quintal / km)", fontsize=11)
    ax1.set_ylabel("Turnover vs Base Rate 0.25 (%)", color=color1, fontsize=11)
    line1 = ax1.plot(rates, turnovers, marker="o", lw=2.5, color=color1, label="Recommendation Turnover (%)")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color2 = "#2ca02c"
    ax2.set_ylabel("Net Advantage vs Nearest Market (₹ / quintal)", color=color2, fontsize=11)
    line2 = ax2.plot(rates, advantages, marker="s", lw=2.5, color=color2, linestyle="--", label="Net Advantage vs Nearest (₹/q)")
    ax2.tick_params(axis="y", labelcolor=color2)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", frameon=True)

    plt.title("Transport Rate Sensitivity: Recommendation Stability vs Assumed Cost", fontsize=12, pad=12)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved sensitivity plot to {output_path}")


def plot_spread_vs_error(spread_df: pd.DataFrame, output_path: Path) -> None:
    """Plots comparison of local market price spread vs forecast error (MAE) by commodity."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))

    comms = spread_df["commodity"].values
    spreads = spread_df["mean_price_spread_inr"].values
    maes = spread_df["val_mae_inr"].values

    x = np.arange(len(comms))
    width = 0.38

    rects1 = ax.bar(x - width / 2, spreads, width, label="Mean Local Price Spread (150km)", color="#2ca02c", alpha=0.85)
    rects2 = ax.bar(x + width / 2, maes, width, label="Forecast Error (Validation MAE)", color="#d62728", alpha=0.85)

    ax.set_ylabel("INR / quintal", fontsize=11)
    ax.set_title("Market Dispersion vs Forecast Error: Where Spatial Arbitrage is Feasible", fontsize=12, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(comms, rotation=25, ha="right", fontsize=10)
    ax.legend(frameon=True)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved spread vs error plot to {output_path}")
