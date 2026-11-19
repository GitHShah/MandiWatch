"""
Unit tests for the experimental timing module (src/timing.py).
Tests:
1. Target calculation and return percentage logic.
2. Holding cost configuration and perishable vs durable differentiation.
3. Probability and net gain arithmetic.
4. Robustness against extreme holding costs.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.timing import get_experimental_timing_decision, prepare_timing_data


def test_holding_cost_perishable_vs_durable():
    config = load_config()
    costs = config.timing.default_holding_costs

    # Perishables should have higher holding cost than durables
    assert costs["Tomato"] > costs["Potato"]
    assert costs["Green Chilli"] > costs["Wheat"]
    assert costs["Cauliflower"] > costs["Paddy (Dhan)(Common)"]
    assert costs["Brinjal"] > costs["Wheat"]


def test_timing_target_preparation():
    # Synthetic dataframe
    df = pd.DataFrame({
        "lag_1": [np.log(100.0), np.log(100.0), np.log(100.0)],
        "target_h2": [np.log(110.0), np.log(101.0), np.log(90.0)],
        "split": ["train", "val", "test"]
    })

    prep = prepare_timing_data(df, hurdle_pct=3.0, h=2)
    # Row 0: +10% return > 3% hurdle -> 1
    # Row 1: +1% return <= 3% hurdle -> 0
    # Row 2: -10% return <= 3% hurdle -> 0
    assert prep["timing_target"].tolist() == [1, 0, 0]
    assert np.isclose(prep["return_pct_actual"].iloc[0], 10.0, atol=0.1)


def test_experimental_timing_decision_farmer_math():
    res = get_experimental_timing_decision(
        market="Test_Mandi",
        commodity="Potato",
        horizon=2,
        custom_holding_cost=10.0,
        user_type="farmer",
    )

    # Total holding cost = 2 * 10 = 20
    assert res["total_holding_cost_inr"] == 20.0
    assert res["net_expected_gain_inr"] == round(res["expected_change_inr"] - 20.0, 2)
    assert 0.0 <= res["prob_gain"] <= 1.0
    assert "EXPERIMENTAL" in res["disclaimer"]


def test_experimental_timing_decision_buyer_math():
    res = get_experimental_timing_decision(
        market="Test_Mandi",
        commodity="Onion",
        horizon=2,
        custom_holding_cost=15.0,
        user_type="buyer",
    )

    assert res["total_holding_cost_inr"] == 30.0
    assert 0.0 <= res["prob_gain"] <= 1.0
    assert res["user_type"] == "buyer"
