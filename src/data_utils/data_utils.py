"""
Data utilities for SIH26071 — sample districts/events and synthetic fallbacks.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from ui.up_locations_data import UP_DISTRICT_LOCATIONS

UP_DISTRICTS = [
    {"name": loc["name"], "lat": loc["lat"], "lon": loc["lon"]}
    for loc in UP_DISTRICT_LOCATIONS
]

BELAGAVI_DISTRICTS = [
    {"name": "Belagavi city", "lat": 15.2993, "lon": 74.5040},
]

SAMPLE_DISTRICTS = UP_DISTRICTS

UP_HISTORICAL_EVENTS = {
    "Central UP heavy rainfall — 10 Jul 2019": {
        "lat": 26.8467, "lon": 80.9462, "peak_mm": 145, "peak_date": "2019-07-10",
    },
    "Eastern UP — 14 Aug 2019": {
        "lat": 26.7606, "lon": 83.3732, "peak_mm": 120, "peak_date": "2019-08-14",
    },
    "Late monsoon eastern UP — 28 Sep 2019": {
        "lat": 25.3176, "lon": 82.9739, "peak_mm": 130, "peak_date": "2019-09-28",
    },
}

SAMPLE_HISTORICAL_EVENTS = UP_HISTORICAL_EVENTS


def get_districts_for_region(region_id: str = "uttar_pradesh") -> list:
    if region_id == "belagavi":
        return BELAGAVI_DISTRICTS
    return UP_DISTRICTS


def get_sample_districts(region_id: str | None = None) -> list:
    from config.regions import get_active_region

    region = get_active_region(region_id)
    return get_districts_for_region(region.id)


def get_sample_historical_events(region_id: str | None = None) -> dict:
    from config.regions import get_active_region

    region = get_active_region(region_id)
    if region.id == "belagavi":
        return {
            "Belagavi monsoon — Aug 2019": {"lat": 15.2993, "lon": 74.5040, "peak_mm": 120},
        }
    return UP_HISTORICAL_EVENTS


def generate_synthetic_rainfall_series(hours: int = 24, base_mm: float = 5.0, seed: int = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    now = datetime.utcnow()
    timestamps = [now - timedelta(hours=h) for h in range(hours, 0, -1)]
    trend = np.linspace(0, 1, hours) ** 2
    noise = rng.normal(0, 1.5, hours)
    rainfall = np.clip(base_mm + trend * 40 + noise, 0, None)
    return pd.DataFrame({"timestamp": timestamps, "rainfall_mm": rainfall})


def generate_synthetic_grid(
    lat_center: float,
    lon_center: float,
    size_deg: float = 0.5,
    n: int = 12,
    seed: int = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lats = np.linspace(lat_center - size_deg, lat_center + size_deg, n)
    lons = np.linspace(lon_center - size_deg, lon_center + size_deg, n)
    rows = []
    for lat in lats:
        for lon in lons:
            rainfall = max(0.0, rng.normal(35, 15))
            elevation_proxy = rng.uniform(0, 1)
            rows.append(
                {"lat": lat, "lon": lon, "rainfall_mm": rainfall, "elevation_proxy": elevation_proxy}
            )
    return pd.DataFrame(rows)
