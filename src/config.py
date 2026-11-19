"""Configuration loader and typed configuration models for Mandi Watch."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class PathsConfig:
    """Filesystem paths for all project data, outputs, and artifacts."""
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    geo_dir: Path
    models_dir: Path
    reports_dir: Path
    figures_dir: Path
    tables_dir: Path
    notebooks_dir: Path


@dataclass
class DataRangeConfig:
    """Date span configuration for dataset filtering."""
    start_date: str
    end_date: str


@dataclass
class ScopeConfig:
    """Scope filtering configuration for automated commodity and state selection."""
    n_commodities: int
    n_states: int
    manual_commodities: List[str] = field(default_factory=list)
    manual_states: List[str] = field(default_factory=list)


@dataclass
class ForecastConfig:
    """Forecasting target and horizon parameters."""
    horizon_weeks: int
    allowed_horizons: List[int]
    target_transform: str


@dataclass
class SplitConfig:
    """Chronological evaluation split configuration."""
    test_months: int
    val_months: int


@dataclass
class CleaningConfig:
    """Data cleaning thresholds and quality filters."""
    mad_multiplier: float
    flat_run_length: int
    min_weeks: int
    max_missing_share: float
    max_interpolation_gap: int
    outlier_action: str = "drop"
    rolling_window: int = 15


@dataclass
class GeoConfig:
    """Geographic, distance approximation, and transport cost parameters."""
    road_factor: float
    default_radius_km: float
    transport_rate_per_km: float
    transport_rate_comment: str


@dataclass
class ModelsConfig:
    """Hyperparameters for machine learning and quantile forecasting models."""
    lightgbm: dict[str, Any]
    quantiles: List[float]


@dataclass
class TimingConfig:
    """Configuration for experimental sell/hold and buy now/wait timing decisions."""
    target_hurdle_pct: float
    default_holding_costs: dict[str, float]
    perishable_commodities: List[str]
    durable_commodities: List[str]


@dataclass
class AppConfig:
    """Root configuration object aggregating all typed sub-configurations."""
    project_name: str
    random_seed: int
    paths: PathsConfig
    data_range: DataRangeConfig
    scope: ScopeConfig
    forecast: ForecastConfig
    split: SplitConfig
    cleaning: CleaningConfig
    geo: GeoConfig
    models: ModelsConfig
    timing: TimingConfig = field(default_factory=lambda: TimingConfig(3.0, {}, [], []))
    raw_dict: dict[str, Any] = field(default_factory=dict)


def load_config(config_path: Optional[Path | str] = None) -> AppConfig:
    """Load configuration from config.yaml and return a typed AppConfig instance.

    Args:
        config_path: Optional path to config.yaml. Defaults to PROJECT_ROOT / 'config.yaml'.

    Returns:
        AppConfig with typed, validated configuration parameters.
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Resolve paths relative to PROJECT_ROOT
    p = data["paths"]
    paths = PathsConfig(
        raw_dir=PROJECT_ROOT / p["raw_dir"],
        interim_dir=PROJECT_ROOT / p["interim_dir"],
        processed_dir=PROJECT_ROOT / p["processed_dir"],
        geo_dir=PROJECT_ROOT / p["geo_dir"],
        models_dir=PROJECT_ROOT / p["models_dir"],
        reports_dir=PROJECT_ROOT / p["reports_dir"],
        figures_dir=PROJECT_ROOT / p["figures_dir"],
        tables_dir=PROJECT_ROOT / p["tables_dir"],
        notebooks_dir=PROJECT_ROOT / p["notebooks_dir"],
    )

    data_range = DataRangeConfig(
        start_date=str(data["data_range"]["start_date"]),
        end_date=str(data["data_range"]["end_date"]),
    )

    scope = ScopeConfig(
        n_commodities=int(data["scope"]["n_commodities"]),
        n_states=int(data["scope"]["n_states"]),
        manual_commodities=list(data["scope"].get("manual_commodities", [])),
        manual_states=list(data["scope"].get("manual_states", [])),
    )

    forecast = ForecastConfig(
        horizon_weeks=int(data["forecast"]["horizon_weeks"]),
        allowed_horizons=list(data["forecast"]["allowed_horizons"]),
        target_transform=str(data["forecast"]["target_transform"]),
    )

    split = SplitConfig(
        test_months=int(data["split"]["test_months"]),
        val_months=int(data["split"]["val_months"]),
    )

    cleaning = CleaningConfig(
        mad_multiplier=float(data["cleaning"]["mad_multiplier"]),
        flat_run_length=int(data["cleaning"]["flat_run_length"]),
        min_weeks=int(data["cleaning"]["min_weeks"]),
        max_missing_share=float(data["cleaning"]["max_missing_share"]),
        max_interpolation_gap=int(data["cleaning"]["max_interpolation_gap"]),
        outlier_action=str(data["cleaning"].get("outlier_action", "drop")),
        rolling_window=int(data["cleaning"].get("rolling_window", 15)),
    )

    geo = GeoConfig(
        road_factor=float(data["geo"]["road_factor"]),
        default_radius_km=float(data["geo"]["default_radius_km"]),
        transport_rate_per_km=float(data["geo"]["transport_rate_per_km"]),
        transport_rate_comment=str(data["geo"].get("transport_rate_comment", "")),
    )

    models = ModelsConfig(
        lightgbm=dict(data["models"]["lightgbm"]),
        quantiles=[float(q) for q in data["models"]["quantiles"]],
    )

    timing_data = data.get("timing", {})
    timing = TimingConfig(
        target_hurdle_pct=float(timing_data.get("target_hurdle_pct", 3.0)),
        default_holding_costs={k: float(v) for k, v in timing_data.get("default_holding_costs", {}).items()},
        perishable_commodities=list(timing_data.get("perishable_commodities", ["Tomato", "Green Chilli", "Cauliflower", "Brinjal"])),
        durable_commodities=list(timing_data.get("durable_commodities", ["Potato", "Onion", "Wheat", "Paddy (Dhan)(Common)"])),
    )

    return AppConfig(
        project_name=str(data["project"]["name"]),
        random_seed=int(data["project"]["random_seed"]),
        paths=paths,
        data_range=data_range,
        scope=scope,
        forecast=forecast,
        split=split,
        cleaning=cleaning,
        geo=geo,
        models=models,
        timing=timing,
        raw_dict=data,
    )
