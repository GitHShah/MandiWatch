"""
Tests for Recommender Engine (Stage 2).
Covers:
1. Ranking order for F1 (Farmer: descending by net price)
2. Ranking order for B1 (Buyer: ascending by landed cost, total cost calculation)
3. Transport cost arithmetic (road distance * rate)
4. Radius filter enforcement
5. Graceful empty-candidate handling (no crashes or invented numbers)
6. "Not clearly different" warning trigger based on validation MAE threshold
"""

import numpy as np
import pandas as pd
import pytest

from src.recommender import (
    DEFAULT_VAL_MAE,
    RecommendationResult,
    build_recommender_map,
    get_recommendations,
)


@pytest.fixture
def synthetic_geo_and_forecasts():
    """Builds controlled synthetic district centroids and precomputed forecasts."""
    # Farmer located at Agra: (27.18, 78.00)
    centroids_df = pd.DataFrame([
        {"state": "Uttar Pradesh", "district": "Agra", "lat": 27.18, "lon": 78.00, "source": "seed"},
        {"state": "Uttar Pradesh", "district": "Mathura", "lat": 27.49, "lon": 77.67, "source": "seed"},
    ])

    # Three synthetic markets with known coords & forecast prices
    # Market A: Close (approx 10 km), moderate price
    # Market B: Far (approx 120 km), high price
    # Market C: Very Far (approx 250 km), very high price (outside 150km radius)
    forecasts_df = pd.DataFrame([
        {
            "market": "Mandi_Near",
            "district": "Agra",
            "state": "Uttar Pradesh",
            "commodity": "Potato",
            "week_start": pd.Timestamp("2025-11-03"),
            "target_week_start": pd.Timestamp("2025-11-17"),
            "lat": 27.20,
            "lon": 78.02,
            "pred_point_inr": 1000.0,
            "pred_p10_inr": 920.0,
            "pred_p90_inr": 1080.0,
        },
        {
            "market": "Mandi_Medium",
            "district": "Mathura",
            "state": "Uttar Pradesh",
            "commodity": "Potato",
            "week_start": pd.Timestamp("2025-11-03"),
            "target_week_start": pd.Timestamp("2025-11-17"),
            "lat": 27.50,
            "lon": 77.68,
            "pred_point_inr": 1100.0,
            "pred_p10_inr": 1010.0,
            "pred_p90_inr": 1190.0,
        },
        {
            "market": "Mandi_Far",
            "district": "Lucknow",
            "state": "Uttar Pradesh",
            "commodity": "Potato",
            "week_start": pd.Timestamp("2025-11-03"),
            "target_week_start": pd.Timestamp("2025-11-17"),
            "lat": 26.85,
            "lon": 80.95,
            "pred_point_inr": 1500.0,
            "pred_p10_inr": 1350.0,
            "pred_p90_inr": 1650.0,
        },
    ])

    return centroids_df, forecasts_df


def test_f1_farmer_ranking_descending(synthetic_geo_and_forecasts):
    """F1: Net price = forecast - transport, sorted descending."""
    centroids_df, forecasts_df = synthetic_geo_and_forecasts

    res = get_recommendations(
        state="Uttar Pradesh",
        district="Agra",
        commodity="Potato",
        user_type="farmer",
        radius_km=150.0,
        transport_rate=0.25,
        centroids_df=centroids_df,
        forecasts_df=forecasts_df,
        val_mae_dict={"Potato": 50.0},
    )

    assert res.success is True
    assert len(res.recommendations) == 2  # Mandi_Far is outside 150km

    # Verify descending net price order
    net_prices = [r.net_price for r in res.recommendations]
    assert net_prices == sorted(net_prices, reverse=True)

    # Verify arithmetic
    for r in res.recommendations:
        expected_transport = r.distance_km * 0.25
        assert pytest.approx(r.transport_cost, 0.01) == expected_transport
        assert pytest.approx(r.net_price, 0.01) == (r.forecast_price - r.transport_cost)


def test_b1_buyer_ranking_ascending(synthetic_geo_and_forecasts):
    """B1: Landed cost = forecast + transport, sorted ascending, total cost = quantity * landed cost."""
    centroids_df, forecasts_df = synthetic_geo_and_forecasts

    res = get_recommendations(
        state="Uttar Pradesh",
        district="Agra",
        commodity="Potato",
        user_type="buyer",
        radius_km=150.0,
        transport_rate=0.25,
        quantity=50.0,  # 50 quintals
        centroids_df=centroids_df,
        forecasts_df=forecasts_df,
        val_mae_dict={"Potato": 50.0},
    )

    assert res.success is True
    assert len(res.recommendations) == 2

    # Verify ascending landed cost order
    landed_costs = [r.landed_cost for r in res.recommendations]
    assert landed_costs == sorted(landed_costs, reverse=False)

    # Verify buyer warning about arrival quantity
    assert any("No arrival/supply data is available" in w for w in res.warnings)

    # Verify total cost calculation
    for r in res.recommendations:
        assert r.total_cost is not None
        assert pytest.approx(r.total_cost, 0.01) == (r.landed_cost * 50.0)


def test_transport_cost_arithmetic(synthetic_geo_and_forecasts):
    """Verifies haversine * road_factor * rate arithmetic."""
    centroids_df, forecasts_df = synthetic_geo_and_forecasts

    rate = 0.50
    rf = 1.35
    res = get_recommendations(
        state="Uttar Pradesh",
        district="Agra",
        commodity="Potato",
        user_type="farmer",
        radius_km=150.0,
        transport_rate=rate,
        road_factor=rf,
        centroids_df=centroids_df,
        forecasts_df=forecasts_df,
    )

    item = res.recommendations[0]
    expected_road_dist = item.haversine_km * rf
    assert pytest.approx(item.distance_km, 0.01) == expected_road_dist
    assert pytest.approx(item.transport_cost, 0.01) == (expected_road_dist * rate)


def test_radius_filter(synthetic_geo_and_forecasts):
    """Only candidates within the specified radius are retained."""
    centroids_df, forecasts_df = synthetic_geo_and_forecasts

    # Radius of 30 km includes only Mandi_Near (distance ~ 3-4 km)
    res_tight = get_recommendations(
        state="Uttar Pradesh",
        district="Agra",
        commodity="Potato",
        radius_km=30.0,
        centroids_df=centroids_df,
        forecasts_df=forecasts_df,
    )
    assert len(res_tight.recommendations) == 1
    assert res_tight.recommendations[0].market == "Mandi_Near"

    # Radius of 400 km includes all 3 markets
    res_wide = get_recommendations(
        state="Uttar Pradesh",
        district="Agra",
        commodity="Potato",
        radius_km=400.0,
        centroids_df=centroids_df,
        forecasts_df=forecasts_df,
    )
    assert len(res_wide.recommendations) == 3


def test_empty_candidate_handling(synthetic_geo_and_forecasts):
    """Graceful empty-candidate handling without crashing or inventing numbers."""
    centroids_df, forecasts_df = synthetic_geo_and_forecasts

    # 1. Unrecognized district
    res_bad_dist = get_recommendations(
        state="Uttar Pradesh",
        district="NonExistentDistrict",
        commodity="Potato",
        centroids_df=centroids_df,
        forecasts_df=forecasts_df,
    )
    assert res_bad_dist.success is False
    assert len(res_bad_dist.recommendations) == 0
    assert "not found" in res_bad_dist.message

    # 2. Radius too small (0.1 km)
    res_zero_rad = get_recommendations(
        state="Uttar Pradesh",
        district="Agra",
        commodity="Potato",
        radius_km=0.1,
        centroids_df=centroids_df,
        forecasts_df=forecasts_df,
    )
    assert res_zero_rad.success is False
    assert len(res_zero_rad.recommendations) == 0
    assert "search radius" in res_zero_rad.message

    # 3. Commodity not present
    res_bad_comm = get_recommendations(
        state="Uttar Pradesh",
        district="Agra",
        commodity="Dragonfruit",
        centroids_df=centroids_df,
        forecasts_df=forecasts_df,
    )
    assert res_bad_comm.success is False
    assert len(res_bad_comm.recommendations) == 0


def test_not_clearly_different_warning_triggers(synthetic_geo_and_forecasts):
    """Warning triggers when price difference is within validation MAE."""
    centroids_df, forecasts_df = synthetic_geo_and_forecasts

    # Difference between Near and Medium is around ₹85/q.
    # Case 1: MAE = 200.0 (> 85) -> Warning MUST trigger
    res_warn = get_recommendations(
        state="Uttar Pradesh",
        district="Agra",
        commodity="Potato",
        radius_km=150.0,
        centroids_df=centroids_df,
        forecasts_df=forecasts_df,
        val_mae_dict={"Potato": 200.0},
    )
    assert any("Statistically Indistinguishable" in w for w in res_warn.warnings)

    # Case 2: MAE = 10.0 (< 85) -> Warning MUST NOT trigger
    res_no_warn = get_recommendations(
        state="Uttar Pradesh",
        district="Agra",
        commodity="Potato",
        radius_km=150.0,
        centroids_df=centroids_df,
        forecasts_df=forecasts_df,
        val_mae_dict={"Potato": 10.0},
    )
    assert not any("Statistically Indistinguishable" in w for w in res_no_warn.warnings)


def test_folium_map_generation(synthetic_geo_and_forecasts):
    """Verifies that Folium map builds properly for valid recommendations."""
    centroids_df, forecasts_df = synthetic_geo_and_forecasts

    res = get_recommendations(
        state="Uttar Pradesh",
        district="Agra",
        commodity="Potato",
        user_type="farmer",
        radius_km=150.0,
        centroids_df=centroids_df,
        forecasts_df=forecasts_df,
    )
    f_map = build_recommender_map(res)
    assert f_map is not None
    html = f_map.get_root().render()
    assert "leaflet" in html.lower()
    assert "Agra" in html
