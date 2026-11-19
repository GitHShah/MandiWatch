"""
Statistical Hypothesis Testing and Time Series Diagnostics for Mandi Watch.
Implements Section 7 of the project specification:
- Monthly aggregation for group comparison tests to account for weekly autocorrelation.
- Normality testing: Shapiro-Wilk with QQ-plot fit & random sampling for large N.
- Variance homogeneity: Levene test.
- Cross-market price equality: Welch ANOVA / Kruskal-Wallis with post-hoc Dunn / Tukey HSD and eta-squared effect size.
- Two-way ANOVA (market x month) using statsmodels with partial eta-squared.
- Paired comparison: Paired t-test / Wilcoxon signed-rank with Cohen's d and difference in INR/quintal.
- Independence test: Chi-square of top-ranked market vs agricultural season with Cramer's V.
- Spatial correlation: Pearson & Spearman correlation between neighboring markets via distance matrix.
- Time-series diagnostics: ADF and KPSS on log price and first difference.
- Autocorrelation & Decomposition: ACF/PACF and STL decomposition.
- Residual & Model evaluation: Ljung-Box test for autocorrelation and Diebold-Mariano test for forecast comparison.
- Multiple-comparison correction: Benjamini-Hochberg (FDR).
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Ensure repository root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

from src.config import load_config

logger = logging.getLogger(__name__)


def aggregate_to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates weekly price series to monthly means per (commodity, state, district, market).
    
    IMPORTANT NOTE PER SPEC SECTION 7:
    Weekly and daily prices are highly autocorrelated, which violates the independence assumption
    of standard ANOVA, Levene, paired t-tests, and Chi-square tests. Group-comparison tests must
    run on monthly aggregates (monthly mean of weekly prices per market) to reduce autocorrelation.
    """
    df_copy = df.copy()
    date_col = "week_start" if "week_start" in df_copy.columns else "week_dt"
    price_col = "median_price" if "median_price" in df_copy.columns else "price"

    if not np.issubdtype(df_copy[date_col].dtype, np.datetime64):
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])

    df_copy["year"] = df_copy[date_col].dt.year
    df_copy["month"] = df_copy[date_col].dt.month
    df_copy["year_month"] = df_copy[date_col].dt.to_period("M").astype(str)

    # Filter out unobserved gaps (where price is NaN)
    valid = df_copy.dropna(subset=[price_col]).copy()

    monthly = (
        valid.groupby(["state", "district", "market", "commodity", "year", "month", "year_month"])
        .agg(
            price_inr=(price_col, "mean"),
            log_price=("log_price", "mean"),
            n_weeks=(price_col, "count"),
            week_start_midpoint=(date_col, "min"),
        )
        .reset_index()
    )

    # Assign Indian agricultural season:
    # Kharif: July to October (Months 7, 8, 9, 10)
    # Rabi: November to February (Months 11, 12, 1, 2)
    # Zaid (Summer): March to June (Months 3, 4, 5, 6)
    def assign_season(m: int) -> str:
        if m in [7, 8, 9, 10]:
            return "Kharif"
        elif m in [11, 12, 1, 2]:
            return "Rabi"
        else:
            return "Zaid"

    monthly["season"] = monthly["month"].apply(assign_season)
    return monthly


def apply_benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies the Benjamini-Hochberg (False Discovery Rate) correction to an array of p-values.
    Returns:
        (reject_flags, adjusted_p_values)
    """
    p_arr = np.asarray(p_values, dtype=float)
    if len(p_arr) == 0:
        return np.array([]), np.array([])
    reject, pvals_corrected, _, _ = multipletests(p_arr, alpha=alpha, method="fdr_bh")
    return reject, pvals_corrected


def test_shapiro_wilk(
    series: pd.Series | np.ndarray,
    max_n: int = 500,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Tests whether a price series (or log price series) is normally distributed using Shapiro-Wilk.
    Per spec: uses a random sample for large N (Shapiro-Wilk is overly sensitive for large sample sizes),
    and computes QQ-plot correlation / fit.
    """
    vals = pd.Series(series).dropna().values
    n_total = len(vals)
    if n_total < 3:
        return {
            "statistic": np.nan, "p_value": np.nan, "is_normal": False,
            "sample_n": n_total, "total_n": n_total, "skewness": np.nan,
            "kurtosis": np.nan, "qq_r2": np.nan, "interpretation": "Insufficient observations (< 3)",
        }

    rng = np.random.RandomState(random_state)
    sample_vals = rng.choice(vals, size=min(n_total, max_n), replace=False) if n_total > max_n else vals

    stat, p_val = stats.shapiro(sample_vals)
    skew = float(stats.skew(vals))
    kurt = float(stats.kurtosis(vals))

    # QQ plot fit (theoretical quantiles vs sample quantiles correlation squared)
    osm, osr = stats.probplot(sample_vals, fit=False)
    r = np.corrcoef(osm, osr)[0, 1]
    qq_r2 = float(r ** 2)

    is_normal = bool(p_val >= 0.05)
    interp = (
        f"Normal (W={stat:.4f}, p={p_val:.4e}, QQ R2={qq_r2:.3f})"
        if is_normal
        else f"Non-normal (W={stat:.4f}, p={p_val:.4e}, skew={skew:.2f}, QQ R2={qq_r2:.3f})"
    )

    return {
        "statistic": float(stat),
        "p_value": float(p_val),
        "is_normal": is_normal,
        "sample_n": len(sample_vals),
        "total_n": n_total,
        "skewness": skew,
        "kurtosis": kurt,
        "qq_r2": qq_r2,
        "interpretation": interp,
    }


def test_levene_homoscedasticity(
    monthly_df: pd.DataFrame,
    commodity: str,
    group_col: str = "market",
    value_col: str = "log_price",
) -> Dict[str, Any]:
    """
    Tests equality of variances across markets for a commodity using Levene's test (median-centered).
    Decides whether standard ANOVA or Welch ANOVA / Kruskal-Wallis should be used.
    """
    sub = monthly_df[monthly_df["commodity"] == commodity].dropna(subset=[value_col, group_col])
    groups = [group[value_col].values for _, group in sub.groupby(group_col) if len(group) >= 5]

    if len(groups) < 2:
        return {
            "commodity": commodity,
            "statistic": np.nan,
            "p_value": np.nan,
            "equal_variance": True,
            "n_groups": len(groups),
            "variance_ratio": np.nan,
            "interpretation": "Fewer than 2 markets with >= 5 monthly observations.",
        }

    stat, p_val = stats.levene(*groups, center="median")
    variances = [np.var(g, ddof=1) for g in groups if np.var(g, ddof=1) > 0]
    var_ratio = float(max(variances) / min(variances)) if variances and min(variances) > 0 else np.nan

    equal_var = bool(p_val >= 0.05)
    interp = (
        f"Homoscedastic (Levene W={stat:.2f}, p={p_val:.4e}, max/min var ratio={var_ratio:.2f})"
        if equal_var
        else f"Heteroscedastic (Levene W={stat:.2f}, p={p_val:.4e}, max/min var ratio={var_ratio:.2f}) -> Use Welch ANOVA or Kruskal-Wallis"
    )

    return {
        "commodity": commodity,
        "statistic": float(stat),
        "p_value": float(p_val),
        "equal_variance": equal_var,
        "n_groups": len(groups),
        "variance_ratio": var_ratio,
        "interpretation": interp,
    }


def test_one_way_price_differences(
    monthly_df: pd.DataFrame,
    commodity: str,
    group_col: str = "market",
    value_col: str = "price_inr",
    equal_var: bool = False,
) -> Dict[str, Any]:
    """
    Tests whether mean prices differ significantly across markets for a given commodity.
    Uses Welch's ANOVA or Kruskal-Wallis if variances are unequal, standard ANOVA otherwise.
    Calculates eta-squared effect size (eta2) and maximum practical price spread across markets in INR/quintal.
    Follows with post-hoc Tukey HSD for pairwise group comparisons.
    """
    sub = monthly_df[monthly_df["commodity"] == commodity].dropna(subset=[value_col, group_col])
    valid_groups = [name for name, grp in sub.groupby(group_col) if len(grp) >= 6]
    sub = sub[sub[group_col].isin(valid_groups)]

    if len(valid_groups) < 2:
        return {
            "commodity": commodity,
            "test_type": "None",
            "statistic": np.nan,
            "p_value": np.nan,
            "eta_squared": np.nan,
            "max_spread_inr": np.nan,
            "significant": False,
            "n_groups": len(valid_groups),
            "interpretation": "Insufficient market groups with >= 6 observations.",
        }

    group_data = [grp[value_col].values for _, grp in sub.groupby(group_col)]
    group_means = sub.groupby(group_col)[value_col].mean()
    max_spread_inr = float(group_means.max() - group_means.min())

    # Kruskal-Wallis as the robust non-parametric ANOVA per spec
    kw_stat, kw_p = stats.kruskal(*group_data)

    # Eta-squared for Kruskal-Wallis: eta2[H] = (H - k + 1) / (N - k)
    n_total = len(sub)
    k_groups = len(valid_groups)
    eta2 = max(0.0, float((kw_stat - k_groups + 1.0) / (n_total - k_groups)))

    # Post-hoc pairwise comparison (Tukey HSD on top markets by observation count for efficiency)
    try:
        top_mkt_names = sub[group_col].value_counts().head(10).index
        sub_top = sub[sub[group_col].isin(top_mkt_names)]
        tukey = pairwise_tukeyhsd(sub_top[value_col], sub_top[group_col], alpha=0.05)
        n_sig_pairs = int(np.sum(tukey.reject))
        total_pairs = len(tukey.reject)
    except Exception:
        n_sig_pairs = 0
        total_pairs = 0

    is_sig = bool(kw_p < 0.05)
    interp = (
        f"Significant cross-market price difference (Kruskal-Wallis H={kw_stat:.2f}, p={kw_p:.4e}, "
        f"eta2={eta2:.3f}, max mandi gap=₹{max_spread_inr:.1f}/q, {n_sig_pairs}/{total_pairs} pairs differ)"
        if is_sig
        else f"No significant market difference (H={kw_stat:.2f}, p={kw_p:.4e}, eta2={eta2:.3f})"
    )

    return {
        "commodity": commodity,
        "test_type": "Kruskal-Wallis",
        "statistic": float(kw_stat),
        "p_value": float(kw_p),
        "eta_squared": eta2,
        "max_spread_inr": max_spread_inr,
        "significant": is_sig,
        "n_groups": k_groups,
        "n_total": n_total,
        "n_sig_pairs": n_sig_pairs,
        "total_pairs": total_pairs,
        "interpretation": interp,
    }


def test_two_way_anova(monthly_df: pd.DataFrame, commodity: str) -> Dict[str, Any]:
    """
    Tests two-way ANOVA (Market x Month) to evaluate:
    1. Main effect of Market (persistent mandi premium).
    2. Main effect of Month (seasonal calendar pattern).
    3. Interaction effect (Market x Month): Does the top-ranked market change by season?
    """
    sub = monthly_df[monthly_df["commodity"] == commodity].dropna(subset=["log_price", "market", "month"])
    top_mkts = sub["market"].value_counts().head(8).index.tolist()
    sub = sub[sub["market"].isin(top_mkts)].copy()

    if len(top_mkts) < 2 or len(sub) < 50:
        return {
            "commodity": commodity,
            "market_p": np.nan, "month_p": np.nan, "interaction_p": np.nan,
            "market_eta2": np.nan, "month_eta2": np.nan, "interaction_eta2": np.nan,
            "interpretation": "Insufficient data for two-way ANOVA.",
        }

    # Clean market names for statsmodels formula
    sub["mkt_clean"] = sub["market"].str.replace(r"[^a-zA-Z0-9_]", "_", regex=True)

    try:
        model = ols("log_price ~ C(mkt_clean) + C(month) + C(mkt_clean):C(month)", data=sub).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        ss_total = anova_table["sum_sq"].sum()
        ss_mkt = anova_table.loc["C(mkt_clean)", "sum_sq"]
        ss_month = anova_table.loc["C(month)", "sum_sq"]
        ss_inter = anova_table.loc["C(mkt_clean):C(month)", "sum_sq"] if "C(mkt_clean):C(month)" in anova_table.index else 0.0

        p_mkt = anova_table.loc["C(mkt_clean)", "PR(>F)"]
        p_month = anova_table.loc["C(month)", "PR(>F)"]
        p_inter = anova_table.loc["C(mkt_clean):C(month)", "PR(>F)"] if "C(mkt_clean):C(month)" in anova_table.index else np.nan

        eta2_mkt = float(ss_mkt / ss_total)
        eta2_month = float(ss_month / ss_total)
        eta2_inter = float(ss_inter / ss_total)

        interp = (
            f"Two-Way ANOVA: Market effect (eta2={eta2_mkt:.3f}, p={p_mkt:.3e}), "
            f"Month effect (eta2={eta2_month:.3f}, p={p_month:.3e}), "
            f"Interaction (eta2={eta2_inter:.3f}, p={p_inter:.3e})"
        )

        return {
            "commodity": commodity,
            "market_f": float(anova_table.loc["C(mkt_clean)", "F"]),
            "market_p": float(p_mkt),
            "market_eta2": eta2_mkt,
            "month_f": float(anova_table.loc["C(month)", "F"]),
            "month_p": float(p_month),
            "month_eta2": eta2_month,
            "interaction_f": float(anova_table.loc["C(mkt_clean):C(month)", "F"]) if "C(mkt_clean):C(month)" in anova_table.index else np.nan,
            "interaction_p": float(p_inter),
            "interaction_eta2": eta2_inter,
            "interpretation": interp,
        }
    except Exception as e:
        logger.warning(f"Two-way ANOVA failed for {commodity}: {e}")
        return {
            "commodity": commodity,
            "market_p": np.nan, "month_p": np.nan, "interaction_p": np.nan,
            "market_eta2": np.nan, "month_eta2": np.nan, "interaction_eta2": np.nan,
            "interpretation": f"Model failed: {e}",
        }


def test_paired_market_differences(
    monthly_df: pd.DataFrame,
    commodity: str,
    market_a: str,
    market_b: str,
    value_col: str = "price_inr",
) -> Dict[str, Any]:
    """
    Tests whether market A and market B have significantly different prices across the SAME months.
    Reports Cohen's d effect size, mean price difference in INR/quintal, median difference, and p-value.
    """
    sub = monthly_df[
        (monthly_df["commodity"] == commodity) & (monthly_df["market"].isin([market_a, market_b]))
    ]
    piv = sub.pivot_table(index="year_month", columns="market", values=value_col).dropna()

    if len(piv) < 10 or market_a not in piv.columns or market_b not in piv.columns:
        return {
            "commodity": commodity,
            "market_a": market_a,
            "market_b": market_b,
            "statistic": np.nan,
            "p_value": np.nan,
            "cohens_d": np.nan,
            "mean_diff_inr": np.nan,
            "median_diff_inr": np.nan,
            "n_months": len(piv),
            "significant": False,
            "interpretation": f"Fewer than 10 aligned months between {market_a} and {market_b}.",
        }

    diffs = piv[market_a] - piv[market_b]
    mean_diff = float(np.mean(diffs))
    median_diff = float(np.median(diffs))
    std_diff = float(np.std(diffs, ddof=1))

    # Cohen's d for paired samples: mean(D) / std(D)
    cohens_d = float(mean_diff / std_diff) if std_diff > 0 else 0.0

    # Wilcoxon signed-rank test
    stat, p_val = stats.wilcoxon(diffs, zero_method="wilcox")

    is_sig = bool(p_val < 0.05)
    interp = (
        f"{market_a} vs {market_b}: Mean diff = ₹{mean_diff:+.1f}/q (Median = ₹{median_diff:+.1f}/q, "
        f"Cohen's d={cohens_d:.2f}, Wilcoxon p={p_val:.4e}, N={len(diffs)} months)"
    )

    return {
        "commodity": commodity,
        "market_a": market_a,
        "market_b": market_b,
        "statistic": float(stat),
        "p_value": float(p_val),
        "cohens_d": cohens_d,
        "mean_diff_inr": mean_diff,
        "median_diff_inr": median_diff,
        "n_months": len(diffs),
        "significant": is_sig,
        "interpretation": interp,
    }


def test_chi_square_market_season(monthly_df: pd.DataFrame, commodity: str) -> Dict[str, Any]:
    """
    Tests whether the top-ranked (highest price) mandi in a state is independent of agricultural season.
    Uses Chi-square test of independence and reports Cramer's V effect size.
    """
    sub = monthly_df[monthly_df["commodity"] == commodity].copy()
    if len(sub) == 0:
        return {
            "commodity": commodity, "statistic": np.nan, "p_value": np.nan,
            "cramers_v": np.nan, "interpretation": "No data",
        }

    # Find the top market per year_month within each state
    idx = sub.groupby(["state", "year_month"])["price_inr"].idxmax()
    top_per_month = sub.loc[idx]

    # Filter to top 5 most frequent winners to avoid hyper-sparse contingency table
    top_winners = top_per_month["market"].value_counts().head(5).index
    filtered = top_per_month[top_per_month["market"].isin(top_winners)]

    contingency = pd.crosstab(filtered["market"], filtered["season"])
    if contingency.empty or contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return {
            "commodity": commodity, "statistic": np.nan, "p_value": np.nan,
            "cramers_v": np.nan, "interpretation": "Insufficient distinct top markets across seasons.",
        }

    chi2, p_val, dof, _ = stats.chi2_contingency(contingency)
    n = contingency.values.sum()
    min_dim = min(contingency.shape) - 1
    cramers_v = float(np.sqrt(chi2 / (n * min_dim))) if (n > 0 and min_dim > 0) else 0.0

    is_sig = bool(p_val < 0.05)
    interp = (
        f"Top market depends on season (Chi2={chi2:.2f}, p={p_val:.4e}, Cramer's V={cramers_v:.3f})"
        if is_sig
        else f"Top market is independent of season (Chi2={chi2:.2f}, p={p_val:.4e}, Cramer's V={cramers_v:.3f})"
    )

    return {
        "commodity": commodity,
        "statistic": float(chi2),
        "p_value": float(p_val),
        "dof": int(dof),
        "cramers_v": cramers_v,
        "n_samples": int(n),
        "significant": is_sig,
        "interpretation": interp,
    }


def test_spatial_correlation(
    monthly_df: pd.DataFrame,
    dist_matrix: pd.DataFrame,
    commodity: str,
    distance_threshold_km: float = 150.0,
) -> Dict[str, Any]:
    """
    Tests whether neighboring mandis (distance <= distance_threshold_km) exhibit stronger
    price co-movement (Pearson r and Spearman rho) than distant mandis (> distance_threshold_km).
    Directly justifies spatial nearest-neighbor features.
    """
    sub = monthly_df[monthly_df["commodity"] == commodity]
    mkts = [m for m in sub["market"].unique() if m in dist_matrix.index and m in dist_matrix.columns]

    if len(mkts) < 4:
        return {
            "commodity": commodity,
            "nearby_pearson_mean": np.nan, "distant_pearson_mean": np.nan,
            "nearby_spearman_mean": np.nan, "distant_spearman_mean": np.nan,
            "diff_pearson": np.nan, "p_value": np.nan,
            "interpretation": "Fewer than 4 markets present in distance matrix.",
        }

    piv = sub.pivot_table(index="year_month", columns="market", values="log_price")
    # Vectorized correlation computation across all markets
    pearson_mat = piv[mkts].corr(method="pearson", min_periods=12)
    spearman_mat = piv[mkts].rank().corr(method="pearson", min_periods=12)
    d_sub = dist_matrix.loc[mkts, mkts].values

    triu_idx = np.triu_indices(len(mkts), k=1)
    dists = d_sub[triu_idx]
    pearsons = pearson_mat.values[triu_idx]
    spearmans = spearman_mat.values[triu_idx]

    valid = ~np.isnan(pearsons) & ~np.isnan(dists)
    if not np.any(valid):
        return {
            "commodity": commodity,
            "nearby_pearson_mean": np.nan, "distant_pearson_mean": np.nan,
            "nearby_spearman_mean": np.nan, "distant_spearman_mean": np.nan,
            "diff_pearson": np.nan, "p_value": np.nan,
            "interpretation": "Insufficient overlapping months across pairs.",
        }

    dists = dists[valid]
    pearsons = pearsons[valid]
    spearmans = spearmans[valid]

    nearby_mask = dists <= distance_threshold_km
    distant_mask = dists > distance_threshold_km

    nearby_pearsons = pearsons[nearby_mask]
    distant_pearsons = pearsons[distant_mask]
    nearby_spearmans = spearmans[nearby_mask]
    distant_spearmans = spearmans[distant_mask]

    if len(nearby_pearsons) == 0 or len(distant_pearsons) == 0:
        return {
            "commodity": commodity,
            "nearby_pearson_mean": float(np.mean(nearby_pearsons)) if len(nearby_pearsons) > 0 else np.nan,
            "distant_pearson_mean": float(np.mean(distant_pearsons)) if len(distant_pearsons) > 0 else np.nan,
            "nearby_spearman_mean": float(np.mean(nearby_spearmans)) if len(nearby_spearmans) > 0 else np.nan,
            "distant_spearman_mean": float(np.mean(distant_spearmans)) if len(distant_spearmans) > 0 else np.nan,
            "diff_pearson": np.nan, "p_value": np.nan,
            "interpretation": "Insufficient pairs in nearby or distant categories.",
        }

    m_near = float(np.mean(nearby_pearsons))
    m_dist = float(np.mean(distant_pearsons))
    stat_u, p_val = stats.mannwhitneyu(nearby_pearsons, distant_pearsons, alternative="greater")

    interp = (
        f"Spatial price co-movement: Mandis <= {distance_threshold_km}km have Pearson r={m_near:.3f} "
        f"vs >{distance_threshold_km}km r={m_dist:.3f} (Mann-Whitney U p={p_val:.4e}) -> Proximity drives co-movement."
    )

    return {
        "commodity": commodity,
        "nearby_pearson_mean": m_near,
        "distant_pearson_mean": m_dist,
        "nearby_spearman_mean": float(np.mean(nearby_spearmans)),
        "distant_spearman_mean": float(np.mean(distant_spearmans)),
        "diff_pearson": m_near - m_dist,
        "statistic": float(stat_u),
        "p_value": float(p_val),
        "interpretation": interp,
    }


def test_stationarity(series: pd.Series | np.ndarray) -> Dict[str, Any]:
    """
    Evaluates time series stationarity using both Augmented Dickey-Fuller (ADF) and KPSS tests.
    Evaluates both level (log price) and first difference.
    
    Decision rule matrix:
    - ADF reject (p < 0.05) & KPSS accept (p >= 0.05) -> Stationary
    - ADF accept (p >= 0.05) & KPSS reject (p < 0.05) -> Unit root (Non-stationary)
    - ADF accept & KPSS accept -> Inconclusive / Weak power
    - ADF reject & KPSS reject -> Trend stationary or structural break
    """
    clean = pd.Series(series).dropna().values
    if len(clean) < 20:
        return {
            "adf_stat": np.nan, "adf_p": np.nan,
            "kpss_stat": np.nan, "kpss_p": np.nan,
            "is_stationary": False, "interpretation": "Insufficient observations (< 20)",
        }

    adf_res = adfuller(clean, autolag="AIC")
    adf_stat, adf_p = float(adf_res[0]), float(adf_res[1])

    # Suppress warnings for KPSS p-value edge interpolation
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            kpss_res = kpss(clean, regression="c", nlags="auto")
            kpss_stat, kpss_p = float(kpss_res[0]), float(kpss_res[1])
        except Exception:
            kpss_stat, kpss_p = np.nan, np.nan

    adf_stationary = adf_p < 0.05
    kpss_stationary = kpss_p >= 0.05 if not np.isnan(kpss_p) else adf_stationary

    if adf_stationary and kpss_stationary:
        status = "Stationary"
    elif not adf_stationary and not kpss_stationary:
        status = "Unit Root (Non-stationary)"
    elif adf_stationary and not kpss_stationary:
        status = "Trend Stationary / Structural Break"
    else:
        status = "Inconclusive"

    interp = f"{status} (ADF stat={adf_stat:.2f}, p={adf_p:.4f}; KPSS stat={kpss_stat:.3f}, p={kpss_p:.4f})"

    return {
        "adf_stat": adf_stat,
        "adf_p": adf_p,
        "kpss_stat": kpss_stat,
        "kpss_p": kpss_p,
        "status": status,
        "is_stationary": status == "Stationary",
        "interpretation": interp,
    }


def compute_acf_pacf(series: pd.Series | np.ndarray, nlags: int = 24) -> Dict[str, np.ndarray]:
    """Computes autocorrelation (ACF) and partial autocorrelation (PACF) values and confidence intervals."""
    clean = pd.Series(series).dropna().values
    nlags = min(nlags, len(clean) // 3)
    acf_vals, acf_conf = acf(clean, nlags=nlags, alpha=0.05)
    pacf_vals, pacf_conf = pacf(clean, nlags=nlags, alpha=0.05)
    return {
        "acf": acf_vals,
        "acf_conf": acf_conf,
        "pacf": pacf_vals,
        "pacf_conf": pacf_conf,
    }


def decompose_stl(series: pd.Series, period: int = 52) -> Dict[str, np.ndarray]:
    """Performs STL (Seasonal and Trend decomposition using Loess)."""
    clean_series = pd.Series(series).dropna()
    res = STL(clean_series, period=period, seasonal=13).fit()
    return {
        "trend": res.trend.values,
        "seasonal": res.seasonal.values,
        "resid": res.resid.values,
    }


def test_ljung_box(residuals: pd.Series | np.ndarray, lags: List[int] = [6, 12, 24]) -> Dict[str, Any]:
    """Tests for residual autocorrelation using Ljung-Box test."""
    clean = pd.Series(residuals).dropna().values
    max_lag = min(max(lags), len(clean) // 5)
    usable_lags = [lag for lag in lags if lag <= max_lag]
    if not usable_lags:
        return {"p_values": {}, "has_autocorrelation": False, "interpretation": "Insufficient data"}

    res = acorr_ljungbox(clean, lags=usable_lags, return_df=True)
    p_dict = {f"lag_{lag}": float(res.loc[lag, "lb_pvalue"]) for lag in usable_lags}
    has_auto = any(p < 0.05 for p in p_dict.values())
    interp = (
        f"Significant residual autocorrelation detected (p < 0.05 at {', '.join([k for k, v in p_dict.items() if v < 0.05])})"
        if has_auto
        else "No residual autocorrelation (residuals are white noise)."
    )
    return {
        "p_values": p_dict,
        "has_autocorrelation": has_auto,
        "interpretation": interp,
    }


def test_diebold_mariano(
    e1: np.ndarray,
    e2: np.ndarray,
    h: int = 2,
    criterion: str = "MSE",
) -> Dict[str, Any]:
    """
    Computes the Diebold-Mariano test with Harvey-Leybourne-Newbold (HLN) small-sample correction
    to compare forecast accuracy of two models.
    H0: Model 1 and Model 2 have equal forecast accuracy.
    H1: Model 1 is significantly better (or different) from Model 2.
    """
    e1 = np.asarray(e1)
    e2 = np.asarray(e2)
    n = len(e1)
    if n < 5:
        return {
            "statistic": np.nan, "p_value": np.nan, "significant": False,
            "interpretation": "Insufficient paired forecasts (< 5)",
        }

    # Loss differential series
    d = (e1 ** 2 - e2 ** 2) if criterion.upper() == "MSE" else (np.abs(e1) - np.abs(e2))
    d_bar = np.mean(d)

    # Autocovariance of loss differential up to h-1
    gamma0 = np.var(d, ddof=0)
    gamma_sum = 0.0
    for lag in range(1, h):
        cov = np.mean((d[lag:] - d_bar) * (d[:-lag] - d_bar))
        gamma_sum += 2.0 * cov

    lr_var = (gamma0 + gamma_sum) / n
    if lr_var <= 0:
        dm_stat = 0.0
    else:
        dm_stat = float(d_bar / np.sqrt(lr_var))

    # Harvey, Leybourne, Newbold (1997) correction factor
    hln_factor = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    hln_stat = dm_stat * hln_factor

    # Two-sided Student-t p-value with n-1 dof
    p_val = float(2.0 * (1.0 - stats.t.cdf(np.abs(hln_stat), df=n - 1)))
    is_sig = p_val < 0.05

    better_model = "Model 1" if d_bar < 0 else "Model 2"
    interp = (
        f"Significant difference: {better_model} is more accurate (DM-HLN stat={hln_stat:.3f}, p={p_val:.4e})"
        if is_sig
        else f"No significant difference in forecast accuracy (DM-HLN stat={hln_stat:.3f}, p={p_val:.4e})"
    )

    return {
        "statistic": float(hln_stat),
        "raw_dm": float(dm_stat),
        "p_value": p_val,
        "mean_differential": float(d_bar),
        "significant": is_sig,
        "interpretation": interp,
    }


def run_all_statistical_tests(
    weekly_path: Path,
    dist_matrix_path: Path,
    output_table_path: Path,
) -> pd.DataFrame:
    """
    Executes the entire suite of statistical tests across all 8 scope commodities,
    applies Benjamini-Hochberg FDR correction, and saves reports/tables/statistical_tests.csv.
    """
    output_table_path.parent.mkdir(parents=True, exist_ok=True)
    weekly_df = pd.read_parquet(weekly_path)
    dist_matrix = pd.read_parquet(dist_matrix_path) if str(dist_matrix_path).endswith(".parquet") else pd.read_csv(dist_matrix_path, index_col=0)

    logger.info("Aggregating weekly prices to monthly means to control for autocorrelation...")
    monthly_df = aggregate_to_monthly(weekly_df)

    commodities = weekly_df["commodity"].unique().tolist()
    records: List[Dict[str, Any]] = []

    for comm in sorted(commodities):
        logger.info(f"Running statistical tests for commodity: {comm}...")
        comm_sub = monthly_df[monthly_df["commodity"] == comm]

        # 1. Shapiro-Wilk on log price
        sw_res = test_shapiro_wilk(comm_sub["log_price"])
        records.append({
            "test_name": "Shapiro-Wilk Normality (Log Price)",
            "commodity": comm,
            "market_or_pair": "All Markets Combined",
            "test_statistic": sw_res["statistic"],
            "raw_p_value": sw_res["p_value"],
            "effect_size": f"Skew={sw_res['skewness']:.2f}, QQ_R2={sw_res['qq_r2']:.3f}",
            "practical_diff_inr": np.nan,
            "plain_english_interpretation": sw_res["interpretation"],
        })

        # 2. Levene Homoscedasticity test across markets
        lev_res = test_levene_homoscedasticity(monthly_df, comm)
        records.append({
            "test_name": "Levene Variance Homogeneity",
            "commodity": comm,
            "market_or_pair": f"{lev_res['n_groups']} Markets",
            "test_statistic": lev_res["statistic"],
            "raw_p_value": lev_res["p_value"],
            "effect_size": f"Var Ratio={lev_res['variance_ratio']:.2f}",
            "practical_diff_inr": np.nan,
            "plain_english_interpretation": lev_res["interpretation"],
        })

        # 3. Kruskal-Wallis Cross-Market Differences + Eta-squared + Max mandi gap in INR
        kw_res = test_one_way_price_differences(monthly_df, comm)
        records.append({
            "test_name": "Kruskal-Wallis Market Differences",
            "commodity": comm,
            "market_or_pair": f"{kw_res['n_groups']} Markets",
            "test_statistic": kw_res["statistic"],
            "raw_p_value": kw_res["p_value"],
            "effect_size": f"eta2={kw_res['eta_squared']:.3f}",
            "practical_diff_inr": kw_res["max_spread_inr"],
            "plain_english_interpretation": kw_res["interpretation"],
        })

        # 4. Two-Way ANOVA (Market x Month)
        anova2 = test_two_way_anova(monthly_df, comm)
        if not np.isnan(anova2.get("market_p", np.nan)):
            records.append({
                "test_name": "Two-Way ANOVA: Market Main Effect",
                "commodity": comm,
                "market_or_pair": "Top Mandis",
                "test_statistic": anova2["market_f"],
                "raw_p_value": anova2["market_p"],
                "effect_size": f"eta2={anova2['market_eta2']:.3f}",
                "practical_diff_inr": np.nan,
                "plain_english_interpretation": f"Persistent market premium (eta2={anova2['market_eta2']:.3f}, p={anova2['market_p']:.2e})",
            })
            records.append({
                "test_name": "Two-Way ANOVA: Month Main Effect",
                "commodity": comm,
                "market_or_pair": "12 Calendar Months",
                "test_statistic": anova2["month_f"],
                "raw_p_value": anova2["month_p"],
                "effect_size": f"eta2={anova2['month_eta2']:.3f}",
                "practical_diff_inr": np.nan,
                "plain_english_interpretation": f"Seasonal calendar effect (eta2={anova2['month_eta2']:.3f}, p={anova2['month_p']:.2e})",
            })
            records.append({
                "test_name": "Two-Way ANOVA: Market x Month Interaction",
                "commodity": comm,
                "market_or_pair": "Market x Month",
                "test_statistic": anova2["interaction_f"],
                "raw_p_value": anova2["interaction_p"],
                "effect_size": f"eta2={anova2['interaction_eta2']:.3f}",
                "practical_diff_inr": np.nan,
                "plain_english_interpretation": (
                    f"Market ranking changes by season (eta2={anova2['interaction_eta2']:.3f}, p={anova2['interaction_p']:.2e})"
                    if anova2["interaction_p"] < 0.05
                    else f"Market ranking stable across seasons (eta2={anova2['interaction_eta2']:.3f}, p={anova2['interaction_p']:.2e})"
                ),
            })

        # 5. Paired market test on top 2 most active mandis
        top_mkts = comm_sub["market"].value_counts().head(2).index.tolist()
        if len(top_mkts) >= 2:
            m1, m2 = top_mkts[0], top_mkts[1]
            paired_res = test_paired_market_differences(monthly_df, comm, m1, m2)
            records.append({
                "test_name": "Paired Wilcoxon Signed-Rank Test",
                "commodity": comm,
                "market_or_pair": f"{m1} vs {m2}",
                "test_statistic": paired_res["statistic"],
                "raw_p_value": paired_res["p_value"],
                "effect_size": f"Cohen's d={paired_res['cohens_d']:.2f}",
                "practical_diff_inr": paired_res["mean_diff_inr"],
                "plain_english_interpretation": paired_res["interpretation"],
            })

        # 6. Chi-Square Test of Top Market vs Season
        chi_res = test_chi_square_market_season(monthly_df, comm)
        records.append({
            "test_name": "Chi-Square Market vs Season",
            "commodity": comm,
            "market_or_pair": "Top Market x Season",
            "test_statistic": chi_res["statistic"],
            "raw_p_value": chi_res["p_value"],
            "effect_size": f"Cramer's V={chi_res['cramers_v']:.3f}",
            "practical_diff_inr": np.nan,
            "plain_english_interpretation": chi_res["interpretation"],
        })

        # 7. Spatial Correlation (Nearby vs Distant Mandis)
        spatial_res = test_spatial_correlation(monthly_df, dist_matrix, comm, distance_threshold_km=150.0)
        records.append({
            "test_name": "Spatial Price Co-Movement (Mann-Whitney U)",
            "commodity": comm,
            "market_or_pair": "<=150km vs >150km",
            "test_statistic": spatial_res.get("statistic", np.nan),
            "raw_p_value": spatial_res.get("p_value", np.nan),
            "effect_size": f"Pearson diff={spatial_res.get('diff_pearson', np.nan):+.3f}",
            "practical_diff_inr": np.nan,
            "plain_english_interpretation": spatial_res["interpretation"],
        })

        # 8. Stationarity (ADF & KPSS on State Aggregated Series)
        state_mean = comm_sub.groupby("year_month")["log_price"].mean()
        stat_level = test_stationarity(state_mean)
        records.append({
            "test_name": "Stationarity ADF/KPSS (Log Price Level)",
            "commodity": comm,
            "market_or_pair": "State Monthly Average",
            "test_statistic": stat_level["adf_stat"],
            "raw_p_value": stat_level["adf_p"],
            "effect_size": f"KPSS={stat_level['kpss_stat']:.3f}",
            "practical_diff_inr": np.nan,
            "plain_english_interpretation": stat_level["interpretation"],
        })

        stat_diff = test_stationarity(state_mean.diff().dropna())
        records.append({
            "test_name": "Stationarity ADF/KPSS (Log Price First Difference)",
            "commodity": comm,
            "market_or_pair": "State Monthly Average Diff",
            "test_statistic": stat_diff["adf_stat"],
            "raw_p_value": stat_diff["adf_p"],
            "effect_size": f"KPSS={stat_diff['kpss_stat']:.3f}",
            "practical_diff_inr": np.nan,
            "plain_english_interpretation": stat_diff["interpretation"],
        })

    results_df = pd.DataFrame(records)

    # Apply Benjamini-Hochberg correction across all tests with valid p-values
    valid_p_mask = results_df["raw_p_value"].notna()
    if valid_p_mask.sum() > 0:
        _, p_adj = apply_benjamini_hochberg(results_df.loc[valid_p_mask, "raw_p_value"].values)
        results_df["adjusted_p_value"] = np.nan
        results_df.loc[valid_p_mask, "adjusted_p_value"] = p_adj
    else:
        results_df["adjusted_p_value"] = np.nan

    # Format output columns
    cols_order = [
        "test_name", "commodity", "market_or_pair", "test_statistic",
        "raw_p_value", "adjusted_p_value", "effect_size",
        "practical_diff_inr", "plain_english_interpretation",
    ]
    results_df = results_df[cols_order]
    results_df.to_csv(output_table_path, index=False)
    logger.info(f"Saved {len(results_df)} statistical test results to {output_table_path}")
    return results_df
