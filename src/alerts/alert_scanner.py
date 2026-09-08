"""Scan pipeline results for district-level heavy-rain and flood alerts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from alerts.alert_thresholds import combined_alert_level, is_actionable_alert
from ui.up_locations_data import UP_LOCATIONS


def _nearest_index(lat_arr: np.ndarray, lon_arr: np.ndarray, lat: float, lon: float) -> tuple[int, int]:
    return int(np.argmin(np.abs(lat_arr - lat))), int(np.argmin(np.abs(lon_arr - lon)))


def _sample(results: dict, field: str, lat: float, lon: float) -> float:
    li, lj = _nearest_index(results["lat"], results["lon"], lat, lon)
    return float(results[field][li, lj])


def location_label(loc: dict) -> str:
    if loc["name"].lower() == loc["district"].lower():
        return loc["district"]
    return f"{loc['name']} · {loc['district']}"


def forecast_at_location(results: dict, location: dict) -> dict:
    lat, lon = location["lat"], location["lon"]
    rainfall = _sample(results, "fused_rainfall", lat, lon)
    depth = _sample(results, "flood_depth", lat, lon)
    risk = int(round(_sample(results, "flood_risk", lat, lon)))
    confidence = _sample(results, "confidence", lat, lon)
    runoff = _sample(results, "runoff", lat, lon)
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


def is_actionable(level: str, flood_risk: int) -> bool:
    return is_actionable_alert(level, flood_risk)


def scan_district_alerts(results: dict) -> list[dict[str, Any]]:
    """Return per-district forecast rows with combined alert levels."""
    rows: list[dict[str, Any]] = []
    for loc in UP_LOCATIONS:
        fc = forecast_at_location(results, loc)
        rows.append(
            {
                "location": location_label(loc),
                "name": fc["name"],
                "district": fc["district"],
                "lat": fc["lat"],
                "lon": fc["lon"],
                "rainfall_mm": round(fc["rainfall_mm"], 1),
                "runoff_mm": round(fc["runoff_mm"], 1),
                "flood_depth_m": round(fc["flood_depth_m"], 2),
                "flood_risk": fc["flood_risk"],
                "confidence": round(fc["confidence"], 3),
                "alert_level": fc["alert_level"],
                "actionable": is_actionable(fc["alert_level"], fc["flood_risk"]),
            }
        )
    return rows


def active_alerts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r["actionable"]]


def build_alert_summary(results: dict, region_name: str = "Uttar Pradesh") -> dict[str, Any]:
    """Build a serialisable alert summary for notifications and persistence."""
    rows = scan_district_alerts(results)
    active = active_alerts(rows)
    by_level: dict[str, int] = {"Green": 0, "Yellow": 0, "Orange": 0, "Red": 0}
    for row in rows:
        by_level[row["alert_level"]] = by_level.get(row["alert_level"], 0) + 1

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "region": region_name,
        "total_districts": len(rows),
        "active_count": len(active),
        "by_level": by_level,
        "districts": rows,
        "active_districts": active,
    }


def save_alert_summary(summary: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return path


def load_alert_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
