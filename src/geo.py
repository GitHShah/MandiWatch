"""
Geographic utilities and distance calculations for Mandi Watch.
Implements:
1. Geocoding of markets and districts (Nominatim with persistent caching & centroid fallback).
2. Pairwise haversine and road distance calculation.
3. Market distance matrix computation.
4. Nearest neighbors query for spatial correlation and lag features.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from geopy.exc import GeocoderServiceError, GeocoderTimedOut
from geopy.geocoders import Nominatim

from src.config import load_config

config = load_config()

logger = logging.getLogger(__name__)

# Bounding box constraints for validation (India / Central & North India)
INDIA_BOUNDS = {
    "lat_min": 8.0,
    "lat_max": 37.5,
    "lon_min": 68.0,
    "lon_max": 97.5,
}

STATE_BOUNDS = {
    "Maharashtra": {"lat_min": 15.6, "lat_max": 22.1, "lon_min": 72.6, "lon_max": 80.9},
    "Madhya Pradesh": {"lat_min": 21.0, "lat_max": 26.9, "lon_min": 74.0, "lon_max": 82.8},
    "Uttar Pradesh": {"lat_min": 23.8, "lat_max": 30.5, "lon_min": 77.0, "lon_max": 84.7},
}

# Offline reference centroids for all 100 districts in scope to guarantee 100% offline reproducibility
DISTRICT_CENTROIDS_SEED = {
    # Uttar Pradesh
    ("Uttar Pradesh", "Agra"): (27.1767, 78.0081),
    ("Uttar Pradesh", "Aligarh"): (27.8974, 78.0880),
    ("Uttar Pradesh", "Ambedkarnagar"): (26.4447, 82.6972),
    ("Uttar Pradesh", "Amethi"): (26.1558, 81.8159),
    ("Uttar Pradesh", "Amroha"): (28.9044, 78.4682),
    ("Uttar Pradesh", "Auraiya"): (26.4655, 79.5147),
    ("Uttar Pradesh", "Ayodhya"): (26.7922, 82.1998),
    ("Uttar Pradesh", "Azamgarh"): (26.0738, 83.1859),
    ("Uttar Pradesh", "Badaun"): (28.0322, 79.1172),
    ("Uttar Pradesh", "Baghpat"): (28.9454, 77.2208),
    ("Uttar Pradesh", "Bahraich"): (27.5744, 81.5975),
    ("Uttar Pradesh", "Ballia"): (25.7589, 84.1483),
    ("Uttar Pradesh", "Balrampur"): (27.4300, 82.1800),
    ("Uttar Pradesh", "Banda"): (25.4764, 80.3370),
    ("Uttar Pradesh", "Barabanki"): (26.9272, 81.1837),
    ("Uttar Pradesh", "Bareilly"): (28.3670, 79.4304),
    ("Uttar Pradesh", "Basti"): (27.2783, 82.7483),
    ("Uttar Pradesh", "Bhadohi (Sant Ravi Nagar)"): (25.3941, 82.5709),
    ("Uttar Pradesh", "Bijnor"): (29.3732, 78.1352),
    ("Uttar Pradesh", "Bulandshahar"): (28.4069, 77.8498),
    ("Uttar Pradesh", "Chandauli"): (25.2612, 83.2678),
    ("Uttar Pradesh", "Chitrakut"): (25.2136, 80.8986),
    ("Uttar Pradesh", "Deoria"): (26.5028, 83.7797),
    ("Uttar Pradesh", "Etah"): (27.5574, 78.6653),
    ("Uttar Pradesh", "Etawah"): (26.7855, 79.0154),
    ("Uttar Pradesh", "Farukhabad"): (27.3826, 79.5830),
    ("Uttar Pradesh", "Fatehpur"): (25.9284, 80.8130),
    ("Uttar Pradesh", "Firozabad"): (27.1594, 78.3957),
    ("Uttar Pradesh", "Gautam Budh Nagar"): (28.3470, 77.5360),
    ("Uttar Pradesh", "Ghaziabad"): (28.6692, 77.4538),
    ("Uttar Pradesh", "Gonda"): (27.1337, 81.9619),
    ("Uttar Pradesh", "Gorakhpur"): (26.7606, 83.3732),
    ("Uttar Pradesh", "Hamirpur"): (25.9547, 80.1517),
    ("Uttar Pradesh", "Hapur"): (28.7306, 77.7759),
    ("Uttar Pradesh", "Hardoi"): (27.3989, 80.1309),
    ("Uttar Pradesh", "Hathras"): (27.5969, 78.0519),
    ("Uttar Pradesh", "Jalaun (Orai)"): (25.9898, 79.4500),
    ("Uttar Pradesh", "Jaunpur"): (25.7464, 82.6837),
    ("Uttar Pradesh", "Jhansi"): (25.4484, 78.5685),
    ("Uttar Pradesh", "Kannuj"): (27.0547, 79.9197),
    ("Uttar Pradesh", "Kanpur"): (26.4499, 80.3319),
    ("Uttar Pradesh", "Kanpur Dehat"): (26.3330, 79.9500),
    ("Uttar Pradesh", "Kasganj"): (27.8083, 78.6475),
    ("Uttar Pradesh", "Kaushambi"): (25.5342, 81.4294),
    ("Uttar Pradesh", "Khiri (Lakhimpur)"): (27.9462, 80.7786),
    ("Uttar Pradesh", "Kushinagar"): (26.7408, 83.8890),
    ("Uttar Pradesh", "Lakhimpur"): (27.9462, 80.7786),
    ("Uttar Pradesh", "Lalitpur"): (24.6897, 78.4116),
    ("Uttar Pradesh", "Lucknow"): (26.8467, 80.9462),
    ("Uttar Pradesh", "Maharajganj"): (27.1438, 83.5615),
    ("Uttar Pradesh", "Mahoba"): (25.2921, 79.8722),
    ("Uttar Pradesh", "Mainpuri"): (27.2289, 79.0289),
    ("Uttar Pradesh", "Mathura"): (27.4924, 77.6737),
    ("Uttar Pradesh", "Mau (Maunathbhanjan)"): (25.9417, 83.5611),
    ("Uttar Pradesh", "Meerut"): (28.9845, 77.7064),
    ("Uttar Pradesh", "Mirzapur"): (25.1337, 82.5644),
    ("Uttar Pradesh", "Muzaffarnagar"): (29.4727, 77.7085),
    ("Uttar Pradesh", "Pillibhit"): (28.6310, 79.8044),
    ("Uttar Pradesh", "Pratapgarh"): (25.8970, 81.9460),
    ("Uttar Pradesh", "Prayagraj"): (25.4358, 81.8463),
    ("Uttar Pradesh", "Raebarelli"): (26.2303, 81.2409),
    ("Uttar Pradesh", "Rampur"): (28.8154, 79.0257),
    ("Uttar Pradesh", "Saharanpur"): (29.9679, 77.5452),
    ("Uttar Pradesh", "Sambhal"): (28.5833, 78.5667),
    ("Uttar Pradesh", "Sant Kabir Nagar"): (26.7836, 83.0232),
    ("Uttar Pradesh", "Shahjahanpur"): (27.8805, 79.9120),
    ("Uttar Pradesh", "Shamli"): (29.4489, 77.3089),
    ("Uttar Pradesh", "Siddharth Nagar"): (27.2981, 82.8122),
    ("Uttar Pradesh", "Sitapur"): (27.5684, 80.6829),
    ("Uttar Pradesh", "Sonbhadra"): (24.6853, 83.0673),
    ("Uttar Pradesh", "Unnao"): (26.5471, 80.4883),
    ("Uttar Pradesh", "Varanasi"): (25.3176, 82.9739),
    # Maharashtra
    ("Maharashtra", "Ahmednagar"): (19.0952, 74.7496),
    ("Maharashtra", "Dhule"): (20.9042, 74.7749),
    ("Maharashtra", "Jalgaon"): (21.0077, 75.5626),
    ("Maharashtra", "Nagpur"): (21.1458, 79.0882),
    ("Maharashtra", "Raigad"): (18.5158, 73.1822),
    ("Maharashtra", "Sholapur"): (17.6599, 75.9064),
    # Madhya Pradesh
    ("Madhya Pradesh", "Alirajpur"): (22.3045, 74.3546),
    ("Madhya Pradesh", "Badwani"): (22.0357, 74.9036),
    ("Madhya Pradesh", "Balaghat"): (21.8129, 80.1837),
    ("Madhya Pradesh", "Betul"): (21.9015, 77.9015),
    ("Madhya Pradesh", "Chhindwara"): (22.0574, 78.9382),
    ("Madhya Pradesh", "Dewas"): (22.9676, 76.0534),
    ("Madhya Pradesh", "Dhar"): (22.5976, 75.2974),
    ("Madhya Pradesh", "Dindori"): (22.9463, 81.0805),
    ("Madhya Pradesh", "Gwalior"): (26.2183, 78.1828),
    ("Madhya Pradesh", "Harda"): (22.3444, 77.0945),
    ("Madhya Pradesh", "Hoshangabad"): (22.7519, 77.7289),
    ("Madhya Pradesh", "Indore"): (22.7196, 75.8577),
    ("Madhya Pradesh", "Jabalpur"): (23.1815, 79.9864),
    ("Madhya Pradesh", "Jhabua"): (22.7686, 74.5954),
    ("Madhya Pradesh", "Katni"): (23.8343, 80.3957),
    ("Madhya Pradesh", "Khandwa"): (21.8314, 76.3498),
    ("Madhya Pradesh", "Khargone"): (21.8234, 75.6186),
    ("Madhya Pradesh", "Mandla"): (22.5986, 80.3714),
    ("Madhya Pradesh", "Morena"): (26.4947, 77.9940),
    ("Madhya Pradesh", "Rajgarh"): (24.0064, 76.7292),
    ("Madhya Pradesh", "Sagar"): (23.8388, 78.7378),
    ("Madhya Pradesh", "Ujjain"): (23.1765, 75.7885),
}


def haversine_distance(
    lat1: float | np.ndarray,
    lon1: float | np.ndarray,
    lat2: float | np.ndarray,
    lon2: float | np.ndarray,
) -> float | np.ndarray:
    """
    Computes great-circle haversine distance in kilometers.
    Vectorized over scalar or numpy array inputs.
    """
    r_earth = 6371.0088  # Mean Earth radius in km

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    c = 2.0 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))
    return r_earth * c


def road_distance(haversine_km: float | np.ndarray, road_factor: Optional[float] = None) -> float | np.ndarray:
    """
    Approximates road distance from haversine distance using road_factor.
    Default road_factor is read from config (1.35).
    """
    factor = road_factor if road_factor is not None else config.geo.road_factor
    return haversine_km * factor


def clean_name_for_query(name: str) -> str:
    """Clean market or district name by removing parenthetical text."""
    return re.sub(r"\(.*?\)", "", name).strip()


def validate_coordinates(lat: float, lon: float, state: Optional[str] = None) -> bool:
    """Validates if coordinates are within reasonable geographic bounds."""
    if not (INDIA_BOUNDS["lat_min"] <= lat <= INDIA_BOUNDS["lat_max"]):
        return False
    if not (INDIA_BOUNDS["lon_min"] <= lon <= INDIA_BOUNDS["lon_max"]):
        return False
    if state and state in STATE_BOUNDS:
        b = STATE_BOUNDS[state]
        # Allow +/- 1.0 degree margin for boundary markets
        if not (b["lat_min"] - 1.0 <= lat <= b["lat_max"] + 1.0):
            return False
        if not (b["lon_min"] - 1.0 <= lon <= b["lon_max"] + 1.0):
            return False
    return True


def build_district_centroids(districts_df: pd.DataFrame, cache_path: Path) -> pd.DataFrame:
    """
    Builds district centroids DataFrame with columns: [state, district, lat, lon, source].
    Uses reference dictionary first, Nominatim fallback if unknown.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    records = []

    geolocator = Nominatim(user_agent="mandi_watch_research")

    for _, row in districts_df[["state", "district"]].drop_duplicates().iterrows():
        st = row["state"]
        dist = row["district"]

        # Check reference seed
        if (st, dist) in DISTRICT_CENTROIDS_SEED:
            lat, lon = DISTRICT_CENTROIDS_SEED[(st, dist)]
            records.append({"state": st, "district": dist, "lat": lat, "lon": lon, "source": "reference_seed"})
            continue

        # Try Nominatim
        clean_dist = clean_name_for_query(dist)
        query = f"{clean_dist} district, {st}, India"
        try:
            loc = geolocator.geocode(query, timeout=10)
            if loc and validate_coordinates(loc.latitude, loc.longitude, st):
                records.append({
                    "state": st, "district": dist, "lat": loc.latitude, "lon": loc.longitude, "source": "nominatim_district"
                })
                time.sleep(1.0)
                continue
        except Exception as e:
            logger.warning(f"Nominatim geocoding error for {query}: {e}")

        # Fallback to state center
        b = STATE_BOUNDS.get(st, INDIA_BOUNDS)
        fallback_lat = (b["lat_min"] + b["lat_max"]) / 2.0
        fallback_lon = (b["lon_min"] + b["lon_max"]) / 2.0
        records.append({
            "state": st, "district": dist, "lat": fallback_lat, "lon": fallback_lon, "source": "state_fallback"
        })

    centroids_df = pd.DataFrame(records).sort_values(["state", "district"]).reset_index(drop=True)
    centroids_df.to_csv(cache_path, index=False)
    logger.info(f"Saved {len(centroids_df)} district centroids to {cache_path}")
    return centroids_df


def build_market_coordinates(
    markets_df: pd.DataFrame,
    district_centroids_df: pd.DataFrame,
    output_path: Path,
    manual_override_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Builds market coordinates DataFrame with columns: [state, district, market, lat, lon, source].
    Applies manual overrides, Nominatim geocoding with cache, and district centroid fallback.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_file = output_path.parent / "geocode_cache.json"

    cache: Dict[str, Tuple[float, float, str]] = {}
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                raw_cache = json.load(f)
                cache = {k: (v[0], v[1], v[2]) for k, v in raw_cache.items()}
        except Exception as e:
            logger.warning(f"Could not load geocode cache: {e}")

    # Load manual overrides if provided
    overrides = {}
    if manual_override_path and manual_override_path.exists():
        try:
            m_df = pd.read_csv(manual_override_path)
            for _, r in m_df.iterrows():
                overrides[(r["state"], r["district"], r["market"])] = (r["lat"], r["lon"])
        except Exception as e:
            logger.warning(f"Could not load manual overrides: {e}")

    dist_map = {
        (r["state"], r["district"]): (r["lat"], r["lon"])
        for _, r in district_centroids_df.iterrows()
    }

    geolocator = Nominatim(user_agent="mandi_watch_research")
    records = []
    unique_mkts = markets_df[["state", "district", "market"]].drop_duplicates()
    use_nominatim = True

    logger.info(f"Resolving coordinates for {len(unique_mkts)} unique markets...")

    for _, row in unique_mkts.iterrows():
        st = row["state"]
        dist = row["district"]
        mkt = row["market"]
        key = f"{st}|{dist}|{mkt}"

        # 1. Manual override
        if (st, dist, mkt) in overrides:
            lat, lon = overrides[(st, dist, mkt)]
            records.append({"state": st, "district": dist, "market": mkt, "lat": lat, "lon": lon, "source": "manual_override"})
            continue

        # 2. Existing cache
        if key in cache:
            lat, lon, src = cache[key]
            records.append({"state": st, "district": dist, "market": mkt, "lat": lat, "lon": lon, "source": src})
            continue

        # 3. If market name matches district name
        if clean_name_for_query(mkt).lower() == clean_name_for_query(dist).lower():
            fallback_lat, fallback_lon = dist_map.get(
                (st, dist),
                DISTRICT_CENTROIDS_SEED.get((st, dist), (23.0, 78.0))
            )
            src = "district_headquarters"
            cache[key] = (fallback_lat, fallback_lon, src)
            records.append({"state": st, "district": dist, "market": mkt, "lat": fallback_lat, "lon": fallback_lon, "source": src})
            continue

        # 4. Geocode with Nominatim (with fallback to district centroid)
        clean_mkt = clean_name_for_query(mkt)
        clean_dist = clean_name_for_query(dist)
        found = False

        if use_nominatim:
            queries = [
                f"{clean_mkt}, {clean_dist}, {st}, India",
                f"{clean_mkt}, {st}, India",
            ]
            for query in queries:
                try:
                    loc = geolocator.geocode(query, timeout=3)
                    if loc and validate_coordinates(loc.latitude, loc.longitude, st):
                        lat, lon, src = loc.latitude, loc.longitude, "nominatim"
                        cache[key] = (lat, lon, src)
                        records.append({"state": st, "district": dist, "market": mkt, "lat": lat, "lon": lon, "source": src})
                        found = True
                        time.sleep(1.0)
                        break
                except GeocoderServiceError as e:
                    logger.warning(f"Nominatim service limit/error: {e}. Falling back to district centroids for remaining markets.")
                    use_nominatim = False
                    break
                except (GeocoderTimedOut, Exception) as e:
                    logger.debug(f"Geocoding exception for {query}: {e}")

        # 5. Fallback to district centroid
        if not found:
            fallback_lat, fallback_lon = dist_map.get(
                (st, dist),
                DISTRICT_CENTROIDS_SEED.get((st, dist), (23.0, 78.0))
            )
            src = "district_centroid"
            cache[key] = (fallback_lat, fallback_lon, src)
            records.append({"state": st, "district": dist, "market": mkt, "lat": fallback_lat, "lon": fallback_lon, "source": src})

    # Save cache
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({k: list(v) for k, v in cache.items()}, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save geocode cache: {e}")

    coords_df = pd.DataFrame(records).sort_values(["state", "district", "market"]).reset_index(drop=True)
    coords_df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(coords_df)} market coordinates to {output_path}")
    return coords_df


def build_market_distance_matrix(market_coords_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """
    Computes pairwise haversine distance matrix (in km) between all markets.
    Columns and index are market names (or composite 'market__district__state').
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use unique market identifier
    df = market_coords_df.drop_duplicates(subset=["market"]).reset_index(drop=True)
    names = df["market"].values
    lats = df["lat"].values
    lons = df["lon"].values
    n = len(df)

    # Vectorized outer difference
    lat1, lat2 = np.meshgrid(lats, lats)
    lon1, lon2 = np.meshgrid(lons, lons)

    dist_matrix = haversine_distance(lat1, lon1, lat2, lon2)
    dist_df = pd.DataFrame(dist_matrix, index=names, columns=names)

    if str(output_path).endswith(".parquet"):
        dist_df.to_parquet(output_path)
    else:
        dist_df.to_csv(output_path)

    logger.info(f"Saved {n}x{n} market distance matrix to {output_path}")
    return dist_df


def initialize_geo_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load or generate all geo artifacts:
    - district_centroids.csv
    - market_coordinates.csv
    - market_distance_matrix.parquet
    """
    geo_dir = Path(config.paths.geo_dir)
    geo_dir.mkdir(parents=True, exist_ok=True)

    dist_centroids_path = geo_dir / "district_centroids.csv"
    market_coords_path = geo_dir / "market_coordinates.csv"
    dist_matrix_path = geo_dir / "market_distance_matrix.parquet"
    manual_overrides_path = geo_dir / "market_coordinates_manual.csv"

    weekly_path = Path(config.paths.processed_dir) / "weekly.parquet"
    if not weekly_path.exists():
        raise FileNotFoundError(f"Processed weekly dataset not found at {weekly_path}")

    weekly_df = pd.read_parquet(weekly_path, columns=["state", "district", "market"])

    if dist_centroids_path.exists():
        dist_centroids = pd.read_csv(dist_centroids_path)
    else:
        dist_centroids = build_district_centroids(weekly_df, dist_centroids_path)

    if market_coords_path.exists():
        market_coords = pd.read_csv(market_coords_path)
    else:
        market_coords = build_market_coordinates(
            weekly_df, dist_centroids, market_coords_path, manual_overrides_path
        )

    if dist_matrix_path.exists():
        dist_matrix = pd.read_parquet(dist_matrix_path)
    else:
        dist_matrix = build_market_distance_matrix(market_coords, dist_matrix_path)

    return dist_centroids, market_coords, dist_matrix


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    c, m, d = initialize_geo_data()
    print(f"Districts: {len(c)}, Markets: {len(m)}, Distance matrix shape: {d.shape}")
