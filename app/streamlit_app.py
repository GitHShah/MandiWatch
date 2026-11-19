"""
Mandi Watch — Streamlit Web Application
Implements Section 11 of the Project Specification.

Decision support tool for Indian agricultural markets (mandis):
- Tab 1: Farmer (F1) — Best selling markets (net price = forecast - transport cost)
- Tab 2: Bulk Buyer (B1) — Lowest procurement cost (landed cost = forecast + transport cost, total cost)
- Interactive Folium spatial maps
- Dynamic dropdown filtering (valid states, districts, commodities)
- Structured econometric & risk warnings
- Caching with st.cache_data (no retraining)
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from src.config import load_config
from src.recommender import (
    DEFAULT_VAL_MAE,
    RecommendationResult,
    build_recommender_map,
    get_recommendations,
    load_cached_forecasts,
    load_district_centroids,
    load_validation_mae,
)

logger = logging.getLogger(__name__)
config = load_config()

# Page configuration
st.set_page_config(
    page_title="Mandi Watch — Decision Support",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_app_data(horizon: int = 2):
    """Loads and caches forecasts, centroids, and validation MAE dictionary."""
    forecasts_df = load_cached_forecasts(horizon=horizon)
    centroids_df = load_district_centroids()
    val_mae_dict = load_validation_mae()

    # Determine max date in dataset for footer
    max_date = str(forecasts_df["week_start"].max())[:10]

    # Pre-extract valid state -> districts mapping
    state_district_map = {}
    for st_name, group in centroids_df.groupby("state"):
        state_district_map[st_name] = sorted(group["district"].unique().tolist())

    # Pre-extract valid state -> commodities mapping from forecasts
    state_comm_map = {}
    for st_name, group in forecasts_df.groupby("state"):
        state_comm_map[st_name] = sorted(group["commodity"].unique().tolist())

    return forecasts_df, centroids_df, val_mae_dict, max_date, state_district_map, state_comm_map


def render_header(max_date: str):
    """Renders clean title and the mandatory Decision Support notice."""
    st.title("🌾 Mandi Watch")
    st.caption("Empirical Agricultural Market Forecasting & Spatial Discovery Engine")

    st.info(
        "⚖️ **Decision support, not guaranteed advice**: Mandi Watch provides econometric and machine learning forecasts "
        "to assist farmers and bulk buyers in market discovery. Historical patterns and assumed transport costs do not "
        "guarantee future spot prices, transaction execution, or product availability."
    )


def render_footer(max_date: str):
    """Renders standard mandatory disclaimer footer."""
    st.markdown("---")
    st.markdown(
        f"<div style='text-align: center; color: #7f8c8d; font-size: 13px; padding: 10px 0;'>"
        f"Based on historical data up to <b>{max_date}</b>. Forecasts are estimates, not guarantees. "
        f"Transport cost is an assumed rate.</div>",
        unsafe_allow_html=True,
    )


def render_results_section(result: RecommendationResult, user_type: str):
    """Renders KPI metric cards, decision warnings, results table, and map."""
    if not result.success:
        st.error(f"❌ **{result.message}**")
        for w in result.warnings:
            st.warning(f"⚠️ {w}")
        return

    # Top KPI Metrics
    top_mandi = result.recommendations[0]
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Top Recommended Mandi", f"#{top_mandi.rank} {top_mandi.market}", f"{top_mandi.district}, {top_mandi.state}")

    with col2:
        if user_type == "farmer":
            st.metric("Expected Net Price", f"₹{top_mandi.net_price:,.2f} / q", f"Forecast: ₹{top_mandi.forecast_price:,.2f}")
        else:
            st.metric("Expected Landed Cost", f"₹{top_mandi.landed_cost:,.2f} / q", f"Forecast: ₹{top_mandi.forecast_price:,.2f}")

    with col3:
        st.metric("Road Distance", f"{top_mandi.distance_km:.1f} km", f"Transport: ₹{top_mandi.transport_cost:.2f}/q")

    with col4:
        if user_type == "buyer" and top_mandi.total_cost is not None:
            st.metric("Total Procurement Cost", f"₹{top_mandi.total_cost:,.2f}", f"Qty: {result.quantity:.0f} quintals")
        else:
            st.metric("10-90% Prediction Interval", f"[{top_mandi.p10:.0f}, {top_mandi.p90:.0f}]", f"MAE: ±₹{result.commodity_mae:.1f}/q")

    # Warnings section
    if result.warnings:
        st.markdown("#### ⚠️ Decision Notices & Model Warnings")
        for w in result.warnings:
            if "Decision Notice" in w:
                st.info(f"ℹ️ {w}")
            else:
                st.warning(f"⚠️ {w}")

    # Layout: Table (Left/Top) & Map (Right/Bottom)
    tab_view_1, tab_view_2 = st.tabs(["📊 Ranked Market Recommendations", "🗺️ Geographic Interactive Map"])

    with tab_view_1:
        table_df = result.to_dataframe()
        st.dataframe(table_df, use_container_width=True, hide_index=True)

    with tab_view_2:
        folium_map = build_recommender_map(result)
        st_folium(folium_map, height=480, use_container_width=True)


def render_experimental_timing_section(
    result: RecommendationResult,
    user_type: str,
    horizon: int,
    forecasts_df: pd.DataFrame,
):
    """Renders the experimental timing decision (Sell/Hold F2 or Buy Now/Wait B2)."""
    if not result.success or not result.recommendations:
        return

    top_mandi = result.recommendations[0]
    comm = result.commodity
    mkt = top_mandi.market

    from src.timing import get_experimental_timing_decision

    expander_title = (
        "🧪 Experimental: Sell Now vs. Hold Timing Analysis (App F2)"
        if user_type == "farmer"
        else "🧪 Experimental: Buy Now vs. Wait Timing Analysis (App B2)"
    )

    with st.expander(expander_title, expanded=False):
        st.markdown(
            f"**Candidate Market:** `{mkt}` ({top_mandi.district}, {top_mandi.state}) | "
            f"**Commodity:** `{comm}` | **Horizon:** `{horizon} weeks ahead`"
        )

        default_cost = config.timing.default_holding_costs.get(comm, 20.0)
        is_perishable = comm in config.timing.perishable_commodities

        t_col1, t_col2 = st.columns([2, 3])
        with t_col1:
            cost_label = (
                f"Holding Cost — Storage, Spoilage & Interest (₹/q/week)"
                if user_type == "farmer"
                else f"Delay / Financing Cost (₹/q/week)"
            )
            custom_cost = st.slider(
                cost_label,
                min_value=0.0,
                max_value=150.0,
                value=float(default_cost),
                step=5.0,
                key=f"{user_type}_holding_cost_slider",
            )
            if is_perishable:
                st.caption(f"ℹ️ *Perishable commodity (Default: ₹{default_cost:.1f}/q/wk): Reflects rapid physical degradation & weight loss.*")
            else:
                st.caption(f"ℹ️ *Durable commodity (Default: ₹{default_cost:.1f}/q/wk): Reflects warehouse/cold-store rent & finance interest.*")

        timing_res = get_experimental_timing_decision(
            market=mkt,
            commodity=comm,
            horizon=horizon,
            custom_holding_cost=custom_cost,
            user_type=user_type,
            p_forecast=top_mandi.forecast_price,
            p10=top_mandi.p10,
            p90=top_mandi.p90,
            forecasts_df=forecasts_df,
        )

        with t_col2:
            st.markdown("#### Expected Payoffs & Probabilities")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric(
                    "Expected Price Change",
                    f"{timing_res['expected_change_inr']:+.1f} ₹/q",
                    f"{timing_res['expected_change_pct']:+.1f}%",
                )
            with m2:
                st.metric(
                    f"Holding Cost ({horizon} wks)",
                    f"₹{timing_res['total_holding_cost_inr']:.1f} / q",
                    f"₹{timing_res['holding_cost_per_week_inr']:.1f}/wk",
                )
            with m3:
                st.metric(
                    "Expected Net Advantage",
                    f"{timing_res['net_expected_gain_inr']:+.1f} ₹/q",
                    f"Prob: {timing_res['prob_gain']*100:.1f}%",
                )

        # Recommendation banner
        st.markdown(
            f"<div style='padding: 12px; margin-top: 10px; margin-bottom: 10px; border-radius: 6px; background-color: rgba(30, 144, 255, 0.08); border-left: 5px solid {timing_res['recommendation_color']}; font-size: 1.0rem; font-weight: 600;'>"
            f"🎯 <b>Directional Analysis:</b> {timing_res['recommendation']} "
            f"(Estimated probability price movement covers holding cost: <b>{timing_res['prob_gain']*100:.1f}%</b>)"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.caption(
            f"📊 **Net Uncertainty Range ({horizon}-week 10-90% interval):** "
            f"[{timing_res['interval_net_p10']:+.1f} ₹/q, {timing_res['interval_net_p90']:+.1f} ₹/q]."
        )

        st.warning(
            f"⚠️ **{timing_res['disclaimer']}**"
        )


def main():
    # Load cached data
    try:
        forecasts_df, centroids_df, val_mae_dict, max_date, state_district_map, state_comm_map = load_app_data(horizon=2)
    except Exception as e:
        st.error(f"Failed to load application dataset: {e}")
        st.stop()

    render_header(max_date)

    # Main Tabs
    farmer_tab, buyer_tab = st.tabs(["🧑‍🌾 Farmer Interface (Sell Produce)", "🏢 Bulk Buyer Interface (Procure Lots)"])

    # -------------------------------------------------------------------------
    # TAB 1: FARMER
    # -------------------------------------------------------------------------
    with farmer_tab:
        st.markdown("### 🌾 Recommend Best Markets to Sell In (App F1)")
        st.caption("Ranks candidate mandis by **Net Price = Forecast Price − Transport Cost**.")

        # Input filters
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)

        with f_col1:
            states_available = sorted(list(state_district_map.keys()))
            f_state = st.selectbox("Select State", states_available, index=states_available.index("Uttar Pradesh") if "Uttar Pradesh" in states_available else 0, key="f_state")

        with f_col2:
            districts_available = state_district_map.get(f_state, [])
            default_dist_idx = districts_available.index("Agra") if "Agra" in districts_available else 0
            f_dist = st.selectbox("Select Your District", districts_available, index=default_dist_idx, key="f_dist")

        with f_col3:
            comms_available = state_comm_map.get(f_state, ["Potato", "Onion", "Tomato", "Wheat"])
            default_comm_idx = comms_available.index("Potato") if "Potato" in comms_available else 0
            f_comm = st.selectbox("Select Commodity", comms_available, index=default_comm_idx, key="f_comm")

        with f_col4:
            f_horizon = st.selectbox("Forecast Horizon (Weeks Ahead)", [2, 1, 4], index=0, key="f_horizon")

        # Sliders
        s_col1, s_col2 = st.columns(2)
        with s_col1:
            f_radius = st.slider("Search Radius (km)", min_value=25, max_value=300, value=150, step=25, key="f_rad")
        with s_col2:
            f_rate = st.slider("Transport Rate — Assumed Rate (₹ / quintal / km)", min_value=0.05, max_value=1.00, value=0.25, step=0.05, key="f_rate")

        # Run query
        if f_horizon != 2:
            try:
                f_forecasts = load_cached_forecasts(horizon=f_horizon)
            except Exception:
                f_forecasts = None
        else:
            f_forecasts = forecasts_df

        if f_forecasts is None:
            st.warning(f"⚠️ Precomputed forecasts for horizon h={f_horizon} are not currently generated. Please select horizon 2.")
        else:
            farmer_result = get_recommendations(
                state=f_state,
                district=f_dist,
                commodity=f_comm,
                user_type="farmer",
                horizon=f_horizon,
                radius_km=float(f_radius),
                transport_rate=float(f_rate),
                forecasts_df=f_forecasts,
                centroids_df=centroids_df,
                val_mae_dict=val_mae_dict,
            )
            render_results_section(farmer_result, user_type="farmer")
            render_experimental_timing_section(
                result=farmer_result,
                user_type="farmer",
                horizon=f_horizon,
                forecasts_df=f_forecasts,
            )

    # -------------------------------------------------------------------------
    # TAB 2: BULK BUYER
    # -------------------------------------------------------------------------
    with buyer_tab:
        st.markdown("### 🏢 Recommend Cheapest Markets to Buy From (App B1)")
        st.caption("Ranks candidate mandis by **Landed Cost = Forecast Price + Transport Cost**, computing total procurement expense.")

        b_col1, b_col2, b_col3, b_col4 = st.columns(4)

        with b_col1:
            b_state = st.selectbox("Select State", states_available, index=states_available.index("Uttar Pradesh") if "Uttar Pradesh" in states_available else 0, key="b_state")

        with b_col2:
            b_districts = state_district_map.get(b_state, [])
            default_b_dist_idx = b_districts.index("Lucknow") if "Lucknow" in b_districts else 0
            b_dist = st.selectbox("Select Sourcing/Destination District", b_districts, index=default_b_dist_idx, key="b_dist")

        with b_col3:
            b_comms = state_comm_map.get(b_state, ["Onion", "Potato", "Tomato", "Wheat"])
            default_b_comm_idx = b_comms.index("Onion") if "Onion" in b_comms else 0
            b_comm = st.selectbox("Select Commodity", b_comms, index=default_b_comm_idx, key="b_comm")

        with b_col4:
            b_qty = st.number_input("Purchase Quantity (Quintals)", min_value=1.0, max_value=10000.0, value=200.0, step=10.0, key="b_qty")

        # Sliders
        bs_col1, bs_col2, bs_col3 = st.columns(3)
        with bs_col1:
            b_horizon = st.selectbox("Horizon (Weeks Ahead)", [2, 1, 4], index=0, key="b_horizon")
        with bs_col2:
            b_radius = st.slider("Search Radius (km)", min_value=25, max_value=300, value=150, step=25, key="b_rad")
        with bs_col3:
            b_rate = st.slider("Transport Rate — Assumed Rate (₹ / quintal / km)", min_value=0.05, max_value=1.00, value=0.25, step=0.05, key="b_rate")

        if b_horizon != 2:
            try:
                b_forecasts = load_cached_forecasts(horizon=b_horizon)
            except Exception:
                b_forecasts = None
        else:
            b_forecasts = forecasts_df

        if b_forecasts is None:
            st.warning(f"⚠️ Precomputed forecasts for horizon h={b_horizon} are not currently generated. Please select horizon 2.")
        else:
            buyer_result = get_recommendations(
                state=b_state,
                district=b_dist,
                commodity=b_comm,
                user_type="buyer",
                horizon=b_horizon,
                radius_km=float(b_radius),
                transport_rate=float(b_rate),
                quantity=float(b_qty),
                forecasts_df=b_forecasts,
                centroids_df=centroids_df,
                val_mae_dict=val_mae_dict,
            )
            render_results_section(buyer_result, user_type="buyer")
            render_experimental_timing_section(
                result=buyer_result,
                user_type="buyer",
                horizon=b_horizon,
                forecasts_df=b_forecasts,
            )

    render_footer(max_date)


if __name__ == "__main__":
    main()
