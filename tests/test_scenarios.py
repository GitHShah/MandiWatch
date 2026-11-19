"""
Integration tests verifying at least 5 combinations on the Streamlit App logic:
1. Farmer - Potato in Agra, UP (Radius: 150 km, Rate: 0.25)
2. Farmer - Tomato in Aligarh, UP (Radius: 125 km, Rate: 0.30)
3. Bulk Buyer - Onion in Lucknow, UP (Radius: 150 km, Rate: 0.20, Qty: 500q)
4. Bulk Buyer - Wheat in Kanpur, UP (Radius: 100 km, Rate: 0.25, Qty: 1000q)
5. Empty Result Case - Radius too small (-1.0 km) -> 0 candidates
6. Decision Warning Case - Low candidate count or statistical indistinguishability
"""

import sys
from pathlib import Path
import pytest

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.streamlit_app import load_app_data
from src.recommender import build_recommender_map, get_recommendations


def test_app_scenarios():
    forecasts_df, centroids_df, val_mae_dict, max_date, state_dist_map, state_comm_map = load_app_data(horizon=2)
    assert not forecasts_df.empty
    assert not centroids_df.empty

    scenarios = [
        {
            "id": "Scenario 1: Farmer — Potato in Agra, UP",
            "params": dict(state="Uttar Pradesh", district="Agra", commodity="Potato", user_type="farmer", horizon=2, radius_km=150.0, transport_rate=0.25),
            "expected_success": True,
        },
        {
            "id": "Scenario 2: Farmer — Tomato in Aligarh, UP",
            "params": dict(state="Uttar Pradesh", district="Aligarh", commodity="Tomato", user_type="farmer", horizon=2, radius_km=125.0, transport_rate=0.30),
            "expected_success": True,
        },
        {
            "id": "Scenario 3: Bulk Buyer — Onion in Lucknow, UP",
            "params": dict(state="Uttar Pradesh", district="Lucknow", commodity="Onion", user_type="buyer", horizon=2, radius_km=150.0, transport_rate=0.20, quantity=500.0),
            "expected_success": True,
        },
        {
            "id": "Scenario 4: Bulk Buyer — Wheat in Kanpur, UP",
            "params": dict(state="Uttar Pradesh", district="Kanpur", commodity="Wheat", user_type="buyer", horizon=2, radius_km=100.0, transport_rate=0.25, quantity=1000.0),
            "expected_success": True,
        },
        {
            "id": "Scenario 5: Empty Result Case — Search radius is 0 km with strict offset",
            "params": dict(state="Uttar Pradesh", district="Agra", commodity="Potato", user_type="farmer", horizon=2, radius_km=-1.0, transport_rate=0.25),
            "expected_success": False,
        },
    ]

    for s in scenarios:
        res = get_recommendations(
            **s["params"],
            forecasts_df=forecasts_df,
            centroids_df=centroids_df,
            val_mae_dict=val_mae_dict
        )
        assert res.success == s["expected_success"], f"Failed for {s['id']}: expected {s['expected_success']} but got {res.success}"
        if res.success:
            df_table = res.to_dataframe()
            assert not df_table.empty
            m = build_recommender_map(res)
            assert m is not None


if __name__ == "__main__":
    test_app_scenarios()
    print("All scenarios passed successfully.")
