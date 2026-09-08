"""Location search and public advisory text for the flood warning dashboard."""

from __future__ import annotations

import numpy as np

from alerts.alert_thresholds import IMD_THRESHOLDS_24H, combined_alert_level
from ui.up_locations_data import UP_DISTRICT_LOCATIONS, UP_LOCATIONS

SYSTEM_MODULES = [
    {"name": "Satellite rainfall", "source": "CHIRPS — harmonised to 0.25° grid"},
    {"name": "Radar rainfall", "source": "MERRA-2 hourly (nowcast) + daily aggregate"},
    {"name": "Radar format (IMD Doppler)", "source": "Jaipur sample volumes for 0–6 h nowcast"},
    {"name": "Observational weather", "source": "IMD 0.25° daily gridded rainfall (reference)"},
    {"name": "NWP model", "source": "ERA5 precipitation — bias-corrected"},
    {"name": "AI/ML integration", "source": "Multi-source ensemble + inverse-variance fusion"},
    {"name": "0–6 h nowcast", "source": "Radar optical-flow extrapolation"},
    {"name": "1–7 day forecast", "source": "NWP + satellite + radar + obs → ML fused daily"},
    {"name": "Heavy rainfall early warning", "source": "IMD colour-coded thresholds (Green→Red)"},
    {"name": "Inundation prediction", "source": "SCS-CN runoff + terrain TWI pluvial model"},
]

LEVEL_COLORS = {
    "Green": "#2e7d32",
    "Yellow": "#f9a825",
    "Orange": "#ef6c00",
    "Red": "#c62828",
}

DISTRICT_COUNT = len(UP_DISTRICT_LOCATIONS)


def location_label(loc: dict) -> str:
    """Unique label for dropdowns (city · district)."""
    if loc["name"].lower() == loc["district"].lower():
        return loc["district"]
    return f"{loc['name']} · {loc['district']}"


def _searchable_text(loc: dict) -> str:
    aliases = " ".join(loc.get("aliases", []))
    return f"{loc['name']} {loc['district']} {aliases}".lower()


def search_locations(query: str, limit: int = 75) -> list[dict]:
    q = query.strip().lower()
    if not q:
        return UP_LOCATIONS
    scored: list[tuple[int, dict]] = []
    for loc in UP_LOCATIONS:
        text = _searchable_text(loc)
        if q in text:
            score = 0 if text.startswith(q) or loc["district"].lower().startswith(q) else 1
            scored.append((score, loc))
    scored.sort(key=lambda item: (item[0], item[1]["district"]))
    return [loc for _, loc in scored[:limit]]


def resolve_location(label: str) -> dict | None:
    for loc in UP_LOCATIONS:
        if location_label(loc) == label:
            return loc
    return None


def _nearest_index(lat_arr: np.ndarray, lon_arr: np.ndarray, lat: float, lon: float) -> tuple[int, int]:
    lat_idx = int(np.argmin(np.abs(lat_arr - lat)))
    lon_idx = int(np.argmin(np.abs(lon_arr - lon)))
    return lat_idx, lon_idx


def sample_grid_value(
    lat_arr: np.ndarray,
    lon_arr: np.ndarray,
    field: np.ndarray,
    lat: float,
    lon: float,
) -> float:
    lat_idx, lon_idx = _nearest_index(lat_arr, lon_arr, lat, lon)
    return float(field[lat_idx, lon_idx])


def rainfall_alert_level(rainfall_mm: float) -> str:
    if rainfall_mm <= IMD_THRESHOLDS_24H["green_upper"]:
        return "Green"
    if rainfall_mm <= IMD_THRESHOLDS_24H["yellow_upper"]:
        return "Yellow"
    if rainfall_mm <= IMD_THRESHOLDS_24H["orange_upper"]:
        return "Orange"
    return "Red"


def location_forecast(results: dict, location: dict) -> dict:
    lat_arr, lon_arr = results["lat"], results["lon"]
    lat, lon = location["lat"], location["lon"]
    rainfall = sample_grid_value(lat_arr, lon_arr, results["fused_rainfall"], lat, lon)
    depth = sample_grid_value(lat_arr, lon_arr, results["flood_depth"], lat, lon)
    risk = int(round(sample_grid_value(lat_arr, lon_arr, results["flood_risk"], lat, lon)))
    confidence = sample_grid_value(lat_arr, lon_arr, results["confidence"], lat, lon)
    runoff = sample_grid_value(lat_arr, lon_arr, results["runoff"], lat, lon)
    return {
        "name": location["name"],
        "district": location["district"],
        "lat": lat,
        "lon": lon,
        "rainfall_mm": rainfall,
        "runoff_mm": runoff,
        "flood_depth_m": depth,
        "flood_risk": risk,
        "confidence": confidence,
        "alert_level": combined_alert_level(rainfall, risk, depth),
    }


def get_advisory_steps(forecast: dict) -> list[dict]:
    level = forecast["alert_level"]
    depth = forecast["flood_depth_m"]
    risk = forecast["flood_risk"]

    steps: list[dict] = []
    if level == "Green":
        steps.extend(
            [
                {"heading": "Routine monitoring", "detail": "Follow IMD district bulletins during active monsoon days."},
                {"heading": "Household readiness", "detail": "Keep torch, medicines, and documents in a waterproof kit."},
                {"heading": "Evacuation plan", "detail": "Identify safe shelter and higher ground near your locality."},
            ]
        )
    elif level == "Yellow":
        steps.extend(
            [
                {"heading": "Travel caution", "detail": "Avoid low-lying routes; postpone non-essential movement."},
                {"heading": "Property protection", "detail": "Clear drains; move livestock and valuables to upper floors."},
                {"heading": "Community alert", "detail": "Inform family and local ward / panchayat representatives."},
            ]
        )
    elif level == "Orange":
        steps.extend(
            [
                {"heading": "Prepare evacuation", "detail": "Pack essentials and charge communication devices now."},
                {"heading": "Road safety", "detail": "Do not drive or walk through flooded roads or nullahs."},
                {"heading": "Official orders", "detail": "Follow district disaster control room and NDMA advisories."},
            ]
        )
    else:
        steps.extend(
            [
                {"heading": "Immediate evacuation", "detail": "Move to designated shelter or highest safe ground nearby."},
                {"heading": "Emergency contact", "detail": "Call 112 or the district disaster helpline."},
                {"heading": "Avoid hazard zones", "detail": "Stay away from riverbanks, basements, and submerged areas."},
            ]
        )

    if depth >= 0.3 or risk >= 2:
        steps.insert(
            0,
            {
                "heading": "Inundation risk",
                "detail": f"Modelled water depth ~{depth:.2f} m — treat the area as flood-prone.",
            },
        )
    return steps


def model_reliability(metrics: dict | None) -> float:
    """Holdout-based model trust score in [0, 1] from ensemble metrics."""
    if not metrics:
        return 0.55
    ens = metrics.get("ensemble", {})
    r2 = float(ens.get("r2", 0.25))
    mae = float(ens.get("mae_mm", 10.0))
    # R² drives primary trust; MAE modulates slightly (lower error -> higher trust)
    score = 0.42 + 0.45 * max(0.0, r2) + 0.08 * max(0.0, 1.0 - mae / 20.0)
    return float(np.clip(score, 0.35, 0.92))


def forecast_reliability(spatial_confidence: float, metrics: dict | None) -> float:
    """Blend grid fusion confidence with trained-model holdout reliability."""
    base = model_reliability(metrics)
    spatial = float(np.nan_to_num(spatial_confidence, nan=0.0))
    if spatial <= 0.05:
        return base
    return float(np.clip(0.55 * base + 0.45 * spatial, 0.35, 0.95))


def reliability_label(score: float) -> str:
    if score >= 0.75:
        return "High"
    if score >= 0.55:
        return "Moderate"
    return "Fair"
