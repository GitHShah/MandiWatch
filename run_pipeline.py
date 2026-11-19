"""
Mandi Watch — End-to-End Pipeline Runner
========================================

Executes the full Mandi Watch data science and machine learning pipeline in order:
1. Cleaning & Quality Filtering (Agmarknet daily -> clean daily -> weekly.parquet)
2. Geocoding & Spatial Matrices (District centroids, market coordinates, distance matrix)
3. Feature Engineering (Lags, rolling stats, momentum, seasonality, spatial features)
4. Baselines Evaluation (Last Observed, Seasonal Naive, Historical Mean)
5. Machine Learning Modeling (LightGBM on delta-target, conformal quantile intervals)
6. Spatial Recommender Evaluation (Held-out test period simulation, regret, hypothesis tests)

Usage:
------
    # Run full pipeline from scratch
    python run_pipeline.py

    # Run specific step(s)
    python run_pipeline.py --step features
    python run_pipeline.py --step modeling
    python run_pipeline.py --step recommender

    # Fast run skipping heavy daily cleaning (uses existing weekly.parquet)
    python run_pipeline.py --skip-cleaning
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Configure ROOT and logging
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("run_pipeline")


def run_stage_cleaning(config):
    """Stage 1: Clean daily data, apply scope filters, and build weekly.parquet."""
    logger.info("=" * 70)
    logger.info("STAGE 1: DATA CLEANING & WEEKLY AGGREGATION")
    logger.info("=" * 70)
    from src.data_cleaning import run_cleaning, run_weekly_pipeline

    clean_daily_path = Path(config.paths.interim_dir) / "clean_daily.parquet"
    if not clean_daily_path.exists():
        logger.info("Running raw daily data cleaning and scope selection...")
        run_cleaning()
    else:
        logger.info(f"Using existing scoped daily parquet at {clean_daily_path}")

    logger.info("Running weekly aggregation and quality filter pipeline...")
    output_weekly, _ = run_weekly_pipeline(config)
    logger.info(f"Weekly dataset generated successfully at {output_weekly}")


def run_stage_geo(config):
    """Stage 2: Geocoding and spatial matrices."""
    logger.info("=" * 70)
    logger.info("STAGE 2: GEOCODING & SPATIAL DISTANCE MATRIX")
    logger.info("=" * 70)
    from src.geo import initialize_geo_data

    dist_centroids, market_coords, dist_matrix = initialize_geo_data()
    logger.info(
        f"Geocoding initialized: {len(dist_centroids)} districts, "
        f"{len(market_coords)} markets, matrix shape: {dist_matrix.shape}"
    )


def run_stage_features(config, horizon: int = 2):
    """Stage 3: Feature engineering."""
    logger.info("=" * 70)
    logger.info(f"STAGE 3: FEATURE ENGINEERING (Horizon h={horizon})")
    logger.info("=" * 70)
    from src.features import generate_and_save_features

    feature_tables = generate_and_save_features(horizons=[horizon])
    df_h = feature_tables[horizon]
    logger.info(f"Features created for h={horizon}: shape={df_h.shape}")


def run_stage_baselines(config, horizon: int = 2):
    """Stage 4: Past-only benchmark baselines."""
    logger.info("=" * 70)
    logger.info("STAGE 4: FORECAST BASELINES EVALUATION")
    logger.info("=" * 70)
    from src.baselines import run_baselines_pipeline

    features_path = Path(config.paths.processed_dir) / f"features_h{horizon}.parquet"
    table_path = Path(config.paths.tables_dir) / "baseline_metrics.csv"
    fig_path = Path(config.paths.figures_dir) / "baseline_mae_comparison.png"

    metrics_df = run_baselines_pipeline(
        features_path=features_path,
        h=horizon,
        output_table_path=table_path,
        output_fig_path=fig_path,
    )
    logger.info(f"Baselines evaluation complete. Metrics saved to {table_path}")


def run_stage_models(config, horizon: int = 2):
    """Stage 5: Main LightGBM modeling & conformal uncertainty calibration."""
    logger.info("=" * 70)
    logger.info("STAGE 5: LIGHTGBM MODELING & UNCERTAINTY QUANTIFICATION")
    logger.info("=" * 70)
    from src.models import run_modeling_pipeline

    features_path = Path(config.paths.processed_dir) / f"features_h{horizon}.parquet"
    results = run_modeling_pipeline(features_path=features_path, h=horizon)
    logger.info("Machine learning modeling pipeline completed successfully.")


def run_stage_recommender(config, horizon: int = 2):
    """Stage 6: Held-out test period recommender evaluation."""
    logger.info("=" * 70)
    logger.info("STAGE 6: SPATIAL RECOMMENDER HELD-OUT EVALUATION")
    logger.info("=" * 70)
    from src.evaluation import (
        evaluate_recommender_simulation,
        run_hypothesis_test_recommended_vs_nearest,
        run_sensitivity_analysis,
        simulate_recommender_test_period,
        compute_market_dispersion_vs_mae,
        plot_recommender_regret_cdf,
        plot_sensitivity_analysis,
        plot_spread_vs_error,
    )

    forecasts_path = Path(config.paths.processed_dir) / f"forecasts_h{horizon}.parquet"
    centroids_path = Path(config.paths.geo_dir) / "district_centroids.csv"

    if not forecasts_path.exists():
        raise FileNotFoundError(f"Precomputed forecasts not found at {forecasts_path}. Run modeling first.")

    forecasts_df = pd.read_parquet(forecasts_path)
    centroids_df = pd.read_csv(centroids_path)

    # 1. Base simulation (rate = 0.25 INR/q/km)
    logger.info("Running test period simulation across all districts and test weeks...")
    sim_df = simulate_recommender_test_period(
        forecasts_df=forecasts_df,
        centroids_df=centroids_df,
        transport_rate=config.geo.transport_rate_per_km,
        radius_km=config.geo.default_radius_km,
    )

    # 2. Regret metrics
    regret_summary_df = evaluate_recommender_simulation(sim_df)
    regret_summary_path = Path(config.paths.tables_dir) / "recommender_regret_summary.csv"
    regret_summary_df.to_csv(regret_summary_path, index=False)
    logger.info(f"Saved recommender regret summary to {regret_summary_path}")

    # 3. Paired Hypothesis testing
    hypo_df = run_hypothesis_test_recommended_vs_nearest(sim_df)
    hypo_path = Path(config.paths.tables_dir) / "recommender_hypothesis_test.csv"
    hypo_df.to_csv(hypo_path, index=False)
    logger.info(f"Saved hypothesis test results to {hypo_path}")

    # 4. Sensitivity analysis across transport rates
    sens_df = run_sensitivity_analysis(
        forecasts_df=forecasts_df,
        centroids_df=centroids_df,
        rates=[0.0, 0.15, 0.25, 0.50],
        base_rate=0.25,
        radius_km=config.geo.default_radius_km,
    )
    sens_path = Path(config.paths.tables_dir) / "recommender_sensitivity_analysis.csv"
    sens_df.to_csv(sens_path, index=False)
    logger.info(f"Saved sensitivity analysis to {sens_path}")

    # 5. Spread vs MAE analysis
    val_mae_dict = {
        "Tomato": 401.15,
        "Green Chilli": 328.41,
        "Cauliflower": 256.77,
        "Onion": 246.35,
        "Brinjal": 161.28,
        "Potato": 90.90,
        "Wheat": 43.86,
        "Paddy (Dhan)(Common)": 40.61,
    }
    spread_df = compute_market_dispersion_vs_mae(sim_df, val_mae_dict=val_mae_dict)
    spread_path = Path(config.paths.tables_dir) / "recommender_spread_vs_error.csv"
    spread_df.to_csv(spread_path, index=False)
    logger.info(f"Saved spread vs MAE table to {spread_path}")

    # 6. Figures
    fig_dir = Path(config.paths.figures_dir)
    plot_recommender_regret_cdf(sim_df, fig_dir / "recommender_regret_cdf.png")
    plot_sensitivity_analysis(sens_df, fig_dir / "recommender_sensitivity_curve.png")
    plot_spread_vs_error(spread_df, fig_dir / "recommender_spread_vs_mae.png")
    logger.info("Saved all recommender evaluation figures.")


def run_stage_timing(config):
    """Stage 7: Experimental sell/hold (F2) and buy now/wait (B2) timing."""
    logger.info("=" * 70)
    logger.info("STAGE 7: EXPERIMENTAL TIMING ANALYSIS (F2 & B2)")
    logger.info("=" * 70)
    from src.timing import run_timing_pipeline

    run_timing_pipeline()
    logger.info("Experimental timing modeling & evaluation complete.")


def main():
    import pandas as pd
    from src.config import load_config

    parser = argparse.ArgumentParser(description="Mandi Watch Pipeline Runner")
    parser.add_argument(
        "--step",
        choices=["cleaning", "geo", "features", "baselines", "models", "recommender", "timing", "all"],
        default="all",
        help="Specify which pipeline step to run (default: all)",
    )
    parser.add_argument(
        "--skip-cleaning",
        action="store_true",
        help="Skip raw data cleaning and use existing weekly.parquet",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=2,
        help="Forecast horizon in weeks (default: 2)",
    )

    args = parser.parse_args()
    config = load_config()

    start_time = time.time()
    logger.info(f"Starting Mandi Watch Pipeline at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Selected Step: {args.step}, Horizon: {args.horizon} weeks")

    try:
        if args.step == "all":
            if not args.skip_cleaning:
                run_stage_cleaning(config)
            else:
                logger.info("Skipping data cleaning as requested (--skip-cleaning).")

            run_stage_geo(config)
            run_stage_features(config, horizon=args.horizon)
            run_stage_baselines(config, horizon=args.horizon)
            run_stage_models(config, horizon=args.horizon)
            run_stage_recommender(config, horizon=args.horizon)
            run_stage_timing(config)

        elif args.step == "cleaning":
            run_stage_cleaning(config)
        elif args.step == "geo":
            run_stage_geo(config)
        elif args.step == "features":
            run_stage_features(config, horizon=args.horizon)
        elif args.step == "baselines":
            run_stage_baselines(config, horizon=args.horizon)
        elif args.step == "models":
            run_stage_models(config, horizon=args.horizon)
        elif args.step == "recommender":
            run_stage_recommender(config, horizon=args.horizon)
        elif args.step == "timing":
            run_stage_timing(config)

        total_elapsed = time.time() - start_time
        logger.info("=" * 70)
        logger.info(f"PIPELINE COMPLETED SUCCESSFULLY IN {total_elapsed:.1f} SECONDS")
        logger.info("=" * 70)

    except Exception as e:
        logger.exception(f"Pipeline failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
