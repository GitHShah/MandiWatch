"""
Mandi Watch — Recommender Engine (Stage 2)
Implements Section 9 of the Project Specification.

Combines Stage 1 precomputed price forecasts with query-time user inputs:
- Location (Farmer / Buyer district centroid)
- Search radius (km)
- Transport rate (INR/quintal/km — Assumed rate)
- Target horizon h (default: 2 weeks)
- For buyers: purchase quantity (quintals)

Supports:
- Farmer Mode (F1): Net price = Forecast price - Transport cost (Ranked descending)
- Bulk Buyer Mode (B1): Landed cost = Forecast price + Transport cost (Ranked ascending)
- Structured decision-support warnings (candidate scarcity, forecast uncertainty, insignificance)
- Interactive Folium map generation
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import sys

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import folium
import numpy as np
import pandas as pd

from src.config import load_config
from src.geo import haversine_distance, road_distance

logger = logging.getLogger(__name__)
config = load_config()

# Standard empirical validation MAEs per commodity for significance check (fallback)
DEFAULT_VAL_MAE = {
    "Tomato": 401.15,
    "Green Chilli": 328.41,
    "Cauliflower": 256.77,
    "Onion": 246.35,
    "Brinjal": 161.28,
    "Potato": 90.90,
    "Wheat": 43.86,
    "Paddy (Dhan)(Common)": 40.61,
}


@dataclass
class RecommendationItem:
    """Represents a single ranked market recommendation."""
    rank: int
    market: str
    district: str
    state: str
    forecast_price: float
    p10: float
    p90: float
    haversine_km: float
    distance_km: float
    transport_cost: float
    net_price: float  # For farmer: forecast - transport
    landed_cost: float  # For buyer: forecast + transport
    total_cost: Optional[float] = None  # For buyer: quantity * landed_cost
    lat: float = 0.0
    lon: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RecommendationResult:
    """Structured response object containing recommendations and decision warnings."""
    user_type: str  # 'farmer' or 'buyer'
    state: str
    district: str
    commodity: str
    horizon: int
    radius_km: float
    transport_rate: float
    origin_lat: float
    origin_lon: float
    forecast_week: str
    target_week: str
    recommendations: List[RecommendationItem]
    warnings: List[str]
    all_candidates_count: int
    success: bool
    message: str
    quantity: Optional[float] = None
    commodity_mae: float = 0.0

    def to_dataframe(self) -> pd.DataFrame:
        """Converts top recommendations to a clean presentation DataFrame."""
        if not self.recommendations:
            return pd.DataFrame()

        rows = []
        for item in self.recommendations:
            row = {
                "Rank": item.rank,
                "Market": item.market,
                "District": item.district,
                "State": item.state,
                "Forecast Price (₹/q)": round(item.forecast_price, 2),
                "10-90% Interval (₹/q)": f"[{item.p10:.2f}, {item.p90:.2f}]",
                "Road Distance (km)": round(item.distance_km, 1),
                "Transport Cost (₹/q)": round(item.transport_cost, 2),
            }
            if self.user_type == "farmer":
                row["Net Price (₹/q)"] = round(item.net_price, 2)
            else:
                row["Landed Cost (₹/q)"] = round(item.landed_cost, 2)
                if item.total_cost is not None:
                    row["Total Cost (₹)"] = round(item.total_cost, 2)
            rows.append(row)

        return pd.DataFrame(rows)


def load_cached_forecasts(horizon: int = 2) -> pd.DataFrame:
    """
    Loads precomputed forecasts for the requested horizon from data/processed/.
    Does not retrain models.
    """
    processed_dir = Path(config.paths.processed_dir)
    file_path = processed_dir / f"forecasts_h{horizon}.parquet"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Precomputed forecasts table not found at {file_path}. "
            "Please ensure Stage 1 model training/forecasting has been executed."
        )

    df = pd.read_parquet(file_path)
    return df


def load_district_centroids() -> pd.DataFrame:
    """Loads district centroids from data/geo/district_centroids.csv."""
    geo_dir = Path(config.paths.geo_dir)
    file_path = geo_dir / "district_centroids.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"District centroids file not found at {file_path}.")

    return pd.read_csv(file_path)


def load_validation_mae() -> Dict[str, float]:
    """Loads validation MAE per commodity from reports/tables/forecast_model_comparison.csv."""
    tables_dir = Path(config.paths.tables_dir)
    comparison_file = tables_dir / "forecast_model_comparison.csv"

    val_mae = dict(DEFAULT_VAL_MAE)
    if comparison_file.exists():
        try:
            df = pd.read_csv(comparison_file)
            lgb_val = df[(df["model"] == "LightGBM") & (df["split"] == "validation")]
            for _, r in lgb_val.iterrows():
                val_mae[r["commodity"]] = float(r["mae_inr"])
        except Exception as e:
            logger.warning(f"Could not load validation MAE table: {e}. Using fallback values.")

    return val_mae


def get_recommendations(
    state: str,
    district: str,
    commodity: str,
    user_type: str = "farmer",
    horizon: int = 2,
    radius_km: float = 150.0,
    transport_rate: float = 0.25,
    quantity: Optional[float] = None,
    forecast_week: Optional[Union[str, pd.Timestamp]] = None,
    road_factor: Optional[float] = None,
    top_k: int = 5,
    forecasts_df: Optional[pd.DataFrame] = None,
    centroids_df: Optional[pd.DataFrame] = None,
    val_mae_dict: Optional[Dict[str, float]] = None,
) -> RecommendationResult:
    """
    Generates market recommendations for a farmer or bulk buyer.

    Parameters:
    -----------
    state : str
        User's state (e.g. 'Uttar Pradesh', 'Maharashtra', 'Madhya Pradesh')
    district : str
        User's district (e.g. 'Agra', 'Nashik')
    commodity : str
        Produce commodity (e.g. 'Potato', 'Onion', 'Tomato')
    user_type : str, default 'farmer'
        'farmer' for F1 (net price = forecast - transport, descending)
        'buyer' for B1 (landed cost = forecast + transport, ascending)
    horizon : int, default 2
        Forecast horizon in weeks
    radius_km : float, default 150.0
        Maximum search radius in kilometers
    transport_rate : float, default 0.25
        Transport cost rate in INR per quintal per kilometer (ASSUMPTION)
    quantity : Optional[float], default None
        Purchase quantity in quintals (used for bulk buyers)
    forecast_week : Optional[str or Timestamp], default None
        Forecast origin week. Defaults to latest available week in dataset.
    road_factor : Optional[float], default None
        Multiplier to convert haversine distance to approximate road distance (default: 1.35)
    top_k : int, default 5
        Number of top recommendations to return
    forecasts_df : Optional[DataFrame], default None
        Pre-loaded forecasts DataFrame (avoids disk read if cached)
    centroids_df : Optional[DataFrame], default None
        Pre-loaded district centroids
    val_mae_dict : Optional[Dict], default None
        Pre-loaded commodity validation MAEs

    Returns:
    --------
    RecommendationResult
    """
    rf = road_factor if road_factor is not None else float(config.geo.road_factor)
    user_type_clean = user_type.strip().lower()
    if user_type_clean not in ("farmer", "buyer"):
        raise ValueError(f"user_type must be either 'farmer' or 'buyer', got '{user_type}'")

    # 1. District Centroid Coordinates
    if centroids_df is None:
        centroids_df = load_district_centroids()

    # Match state and district case-insensitively
    d_match = centroids_df[
        (centroids_df["state"].str.lower() == state.strip().lower())
        & (centroids_df["district"].str.lower() == district.strip().lower())
    ]

    if d_match.empty:
        return RecommendationResult(
            user_type=user_type_clean,
            state=state,
            district=district,
            commodity=commodity,
            horizon=horizon,
            radius_km=radius_km,
            transport_rate=transport_rate,
            origin_lat=0.0,
            origin_lon=0.0,
            forecast_week="",
            target_week="",
            recommendations=[],
            warnings=[],
            all_candidates_count=0,
            success=False,
            message=f"Location '{district}, {state}' not found in geographic database.",
            quantity=quantity,
        )

    origin_lat = float(d_match.iloc[0]["lat"])
    origin_lon = float(d_match.iloc[0]["lon"])

    # 2. Load Precomputed Forecasts
    if forecasts_df is None:
        try:
            forecasts_df = load_cached_forecasts(horizon=horizon)
        except Exception as e:
            return RecommendationResult(
                user_type=user_type_clean,
                state=state,
                district=district,
                commodity=commodity,
                horizon=horizon,
                radius_km=radius_km,
                transport_rate=transport_rate,
                origin_lat=origin_lat,
                origin_lon=origin_lon,
                forecast_week="",
                target_week="",
                recommendations=[],
                warnings=[],
                all_candidates_count=0,
                success=False,
                message=f"Error loading forecasts table: {e}",
                quantity=quantity,
            )

    # 3. Filter by Commodity
    comm_df = forecasts_df[forecasts_df["commodity"].str.lower() == commodity.strip().lower()].copy()
    if comm_df.empty:
        return RecommendationResult(
            user_type=user_type_clean,
            state=state,
            district=district,
            commodity=commodity,
            horizon=horizon,
            radius_km=radius_km,
            transport_rate=transport_rate,
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            forecast_week="",
            target_week="",
            recommendations=[],
            warnings=[],
            all_candidates_count=0,
            success=False,
            message=f"No forecast data available for commodity '{commodity}'.",
            quantity=quantity,
        )

    # 4. Filter by Forecast Week (Latest origin if not specified)
    if forecast_week is None:
        latest_week = comm_df["week_start"].max()
        comm_df = comm_df[comm_df["week_start"] == latest_week].copy()
    else:
        comm_week_ts = pd.to_datetime(forecast_week)
        comm_df = comm_df[comm_df["week_start"] == comm_week_ts].copy()
        if comm_df.empty:
            return RecommendationResult(
                user_type=user_type_clean,
                state=state,
                district=district,
                commodity=commodity,
                horizon=horizon,
                radius_km=radius_km,
                transport_rate=transport_rate,
                origin_lat=origin_lat,
                origin_lon=origin_lon,
                forecast_week=str(forecast_week),
                target_week="",
                recommendations=[],
                warnings=[],
                all_candidates_count=0,
                success=False,
                message=f"No forecast data available for origin week '{forecast_week}'.",
                quantity=quantity,
            )

    origin_week_str = str(comm_df["week_start"].iloc[0])[:10]
    target_week_str = str(comm_df["target_week_start"].iloc[0])[:10] if "target_week_start" in comm_df.columns else ""

    # Drop candidate markets with missing forecasts
    valid_candidates = comm_df.dropna(subset=["pred_point_inr", "lat", "lon"]).copy()

    if valid_candidates.empty:
        return RecommendationResult(
            user_type=user_type_clean,
            state=state,
            district=district,
            commodity=commodity,
            horizon=horizon,
            radius_km=radius_km,
            transport_rate=transport_rate,
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            forecast_week=origin_week_str,
            target_week=target_week_str,
            recommendations=[],
            warnings=[],
            all_candidates_count=0,
            success=False,
            message=f"No valid price forecasts found for '{commodity}' in origin week {origin_week_str}.",
            quantity=quantity,
        )

    # 5. Calculate Haversine & Road Distances
    mkt_lats = valid_candidates["lat"].values
    mkt_lons = valid_candidates["lon"].values

    h_dists = haversine_distance(origin_lat, origin_lon, mkt_lats, mkt_lons)
    r_dists = road_distance(h_dists, road_factor=rf)

    valid_candidates["haversine_km"] = h_dists
    valid_candidates["distance_km"] = r_dists

    # 6. Radius Filter
    in_radius = valid_candidates[valid_candidates["distance_km"] <= radius_km].copy()
    all_candidates_count = len(in_radius)

    if in_radius.empty:
        return RecommendationResult(
            user_type=user_type_clean,
            state=state,
            district=district,
            commodity=commodity,
            horizon=horizon,
            radius_km=radius_km,
            transport_rate=transport_rate,
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            forecast_week=origin_week_str,
            target_week=target_week_str,
            recommendations=[],
            warnings=[
                f"No markets found trading '{commodity}' within {radius_km:.0f} km of {district}, {state}. "
                "Try expanding the search radius."
            ],
            all_candidates_count=0,
            success=False,
            message="No markets found within the specified search radius.",
            quantity=quantity,
        )

    # 7. Compute Transport Cost, Net Price, Landed Cost
    in_radius["transport_cost"] = in_radius["distance_km"] * transport_rate
    in_radius["net_price"] = in_radius["pred_point_inr"] - in_radius["transport_cost"]
    in_radius["landed_cost"] = in_radius["pred_point_inr"] + in_radius["transport_cost"]

    # 8. Ranking
    if user_type_clean == "farmer":
        ranked = in_radius.sort_values(by="net_price", ascending=False).reset_index(drop=True)
    else:
        ranked = in_radius.sort_values(by="landed_cost", ascending=True).reset_index(drop=True)

    # Top-K
    top_df = ranked.head(top_k).reset_index(drop=True)

    # 9. Build Recommendation Items
    items: List[RecommendationItem] = []
    for idx, row in top_df.iterrows():
        p10_val = float(row["pred_p10_inr"]) if pd.notna(row.get("pred_p10_inr")) else float(row["pred_point_inr"]) * 0.90
        p90_val = float(row["pred_p90_inr"]) if pd.notna(row.get("pred_p90_inr")) else float(row["pred_point_inr"]) * 1.10
        total_cost_val = float(row["landed_cost"]) * quantity if (quantity is not None and user_type_clean == "buyer") else None

        item = RecommendationItem(
            rank=idx + 1,
            market=str(row["market"]),
            district=str(row["district"]),
            state=str(row["state"]),
            forecast_price=float(row["pred_point_inr"]),
            p10=p10_val,
            p90=p90_val,
            haversine_km=float(row["haversine_km"]),
            distance_km=float(row["distance_km"]),
            transport_cost=float(row["transport_cost"]),
            net_price=float(row["net_price"]),
            landed_cost=float(row["landed_cost"]),
            total_cost=total_cost_val,
            lat=float(row["lat"]),
            lon=float(row["lon"]),
        )
        items.append(item)

    # 10. Generate Structured Decision Warnings
    warnings: List[str] = []

    # Warning A: Buyer arrival limitation (mandatory per spec section 9)
    if user_type_clean == "buyer":
        warnings.append(
            "Decision Notice: No arrival/supply data is available in the public record, "
            "so availability of your requested quantity cannot be guaranteed."
        )

    # Warning B: Candidate scarcity
    if all_candidates_count < 3:
        warnings.append(
            f"Low Market Density: Only {all_candidates_count} candidate market(s) found within {radius_km:.0f} km. "
            "Consider expanding the search radius to explore more competitive options."
        )

    # Warning C: Statistical indistinguishability ("these options are not clearly different")
    if val_mae_dict is None:
        val_mae_dict = load_validation_mae()
    comm_mae = val_mae_dict.get(commodity, DEFAULT_VAL_MAE.get(commodity, 150.0))

    if len(items) >= 2:
        if user_type_clean == "farmer":
            score_diff = abs(items[0].net_price - items[1].net_price)
        else:
            score_diff = abs(items[0].landed_cost - items[1].landed_cost)

        if score_diff < comm_mae:
            warnings.append(
                f"Statistically Indistinguishable: The top options are not clearly different. "
                f"The difference between Rank 1 and Rank 2 (₹{score_diff:.2f}/q) is smaller than the typical "
                f"forecast error (validation MAE: ₹{comm_mae:.2f}/q) for {commodity}."
            )

    # Warning D: Low confidence / wide prediction interval
    wide_interval_mkts = []
    for it in items:
        spread = it.p90 - it.p10
        if it.forecast_price > 0 and (spread / it.forecast_price) > 0.40:
            wide_interval_mkts.append(f"{it.market} (interval spread: ₹{spread:.0f}/q)")
    if wide_interval_mkts:
        warnings.append(
            f"High Forecast Uncertainty: {len(wide_interval_mkts)} recommended market(s) have a wide 10-90% interval "
            f"(>40% of price): {', '.join(wide_interval_mkts[:3])}."
        )

    # Warning E: Transport cost dominates price spread
    if len(items) >= 2:
        price_spread = max(it.forecast_price for it in items) - min(it.forecast_price for it in items)
        max_transport = max(it.transport_cost for it in items)
        if user_type_clean == "farmer" and max_transport > price_spread:
            warnings.append(
                f"Transport Cost Dominance: Transport cost (up to ₹{max_transport:.2f}/q) exceeds the gross "
                f"price spread (₹{price_spread:.2f}/q) among candidates. Selling at the nearest mandi may yield "
                "a higher net return."
            )

    return RecommendationResult(
        user_type=user_type_clean,
        state=state,
        district=district,
        commodity=commodity,
        horizon=horizon,
        radius_km=radius_km,
        transport_rate=transport_rate,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        forecast_week=origin_week_str,
        target_week=target_week_str,
        recommendations=items,
        warnings=warnings,
        all_candidates_count=all_candidates_count,
        success=True,
        message=f"Successfully identified {len(items)} recommended market(s).",
        quantity=quantity,
        commodity_mae=comm_mae,
    )


def build_recommender_map(result: RecommendationResult) -> folium.Map:
    """
    Builds an interactive Folium map visualizing the recommendation.
    Shows:
    - User district centroid marker
    - Search radius circle boundary
    - Recommended market markers colored by rank
    - Road connecting lines with tooltips showing distance & transport cost
    """
    if not result.success or not result.recommendations:
        # Return fallback centered on central India
        m = folium.Map(location=[23.0, 78.0], zoom_start=6, tiles="OpenStreetMap")
        return m

    # Center map on user location
    m = folium.Map(
        location=[result.origin_lat, result.origin_lon],
        zoom_start=8,
        tiles="OpenStreetMap",
    )

    # 1. User Origin Marker
    user_label = "Farmer Origin" if result.user_type == "farmer" else "Buyer Origin"
    origin_popup = folium.Popup(
        f"<div style='font-family: Arial; font-size: 13px;'>"
        f"<b>{user_label}</b><br>"
        f"<b>District:</b> {result.district}<br>"
        f"<b>State:</b> {result.state}<br>"
        f"<b>Commodity:</b> {result.commodity}<br>"
        f"<b>Radius:</b> {result.radius_km:.0f} km<br>"
        f"<b>Transport Rate:</b> ₹{result.transport_rate:.2f}/q/km (Assumed)</div>",
        max_width=250,
    )
    folium.Marker(
        location=[result.origin_lat, result.origin_lon],
        popup=origin_popup,
        tooltip=f"{user_label}: {result.district}",
        icon=folium.Icon(color="blue", icon="home", prefix="fa"),
    ).add_to(m)

    # 2. Search Radius Circle
    folium.Circle(
        location=[result.origin_lat, result.origin_lon],
        radius=result.radius_km * 1000,
        color="#3186cc",
        weight=1.5,
        fill=True,
        fill_color="#3186cc",
        fill_opacity=0.08,
        dash_array="5, 10",
        tooltip=f"Search Radius: {result.radius_km:.0f} km",
    ).add_to(m)

    # 3. Market Markers & Connecting Polylines
    rank_colors = {1: "green", 2: "darkgreen", 3: "orange", 4: "cadetblue", 5: "gray"}

    lats = [result.origin_lat]
    lons = [result.origin_lon]

    for item in result.recommendations:
        lats.append(item.lat)
        lons.append(item.lon)

        color = rank_colors.get(item.rank, "blue")
        icon_name = "star" if item.rank == 1 else "shopping-cart"

        # Popup content
        if result.user_type == "farmer":
            score_label = "Net Price"
            score_val = f"₹{item.net_price:.2f}/q"
        else:
            score_label = "Landed Cost"
            score_val = f"₹{item.landed_cost:.2f}/q"

        total_line = f"<br><b>Total Cost:</b> ₹{item.total_cost:,.2f}" if item.total_cost is not None else ""

        popup_html = (
            f"<div style='font-family: Arial; font-size: 13px; line-height: 1.4;'>"
            f"<b style='font-size: 14px;'>#{item.rank} {item.market}</b><br>"
            f"<i>{item.district}, {item.state}</i><hr style='margin: 4px 0;'>"
            f"<b>{score_label}:</b> <span style='color: green; font-size: 14px; font-weight: bold;'>{score_val}</span><br>"
            f"<b>Forecast Price:</b> ₹{item.forecast_price:.2f}/q<br>"
            f"<b>10-90% Interval:</b> [{item.p10:.2f}, {item.p90:.2f}]<br>"
            f"<b>Road Distance:</b> {item.distance_km:.1f} km<br>"
            f"<b>Transport Cost:</b> ₹{item.transport_cost:.2f}/q"
            f"{total_line}</div>"
        )

        folium.Marker(
            location=[item.lat, item.lon],
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=f"#{item.rank} {item.market} ({score_label}: {score_val})",
            icon=folium.Icon(color=color, icon=icon_name, prefix="fa"),
        ).add_to(m)

        # Polyline connecting origin to market
        line_color = "#2ca02c" if item.rank == 1 else "#7f7f7f"
        folium.PolyLine(
            locations=[[result.origin_lat, result.origin_lon], [item.lat, item.lon]],
            color=line_color,
            weight=2.5 if item.rank == 1 else 1.5,
            opacity=0.8 if item.rank == 1 else 0.5,
            dash_array="4, 6" if item.rank > 1 else None,
            tooltip=f"{item.market}: {item.distance_km:.1f} km (Transport: ₹{item.transport_cost:.2f}/q)",
        ).add_to(m)

    # Fit bounds
    if len(lats) > 1:
        m.fit_bounds([[min(lats) - 0.1, min(lons) - 0.1], [max(lats) + 0.1, max(lons) + 0.1]])

    return m
