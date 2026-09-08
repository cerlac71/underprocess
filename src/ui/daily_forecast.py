"""SIH26071 integrated forecast — thin UI wrapper (see forecasting.integrated_forecast)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr

from config.bootstrap import bootstrap_project

PROJECT_ROOT = bootstrap_project()
INTEGRATED_PATH = PROJECT_ROOT / "integrated_forecast.nc"

from forecasting.integrated_forecast import (
    IntegratedForecastMeta,
    build_integrated_forecast,
    forecast_series_at_point,
    outlook_summary,
)

__all__ = [
    "IntegratedForecastMeta",
    "INTEGRATED_PATH",
    "build_integrated_forecast",
    "build_daily_forecast_dataset",
    "daily_series_at_point",
    "outlook_summary",
    "batch_district_forecasts",
    "high_rainfall_probability",
]


def build_daily_forecast_dataset() -> tuple[xr.Dataset, str]:
    """Load cached integrated forecast or build and persist it."""
    if INTEGRATED_PATH.exists():
        with xr.open_dataset(INTEGRATED_PATH) as ds:
            loaded = ds.load()
        label = (
            f"SIH26071 integrated forecast — Satellite + Radar + IMD observations + ERA5 NWP "
            f"with AI/ML fusion & inundation ({ds.attrs.get('title', 'Uttar Pradesh')}, "
            f"{int(loaded.sizes.get('time', 0))} days)"
        )
        return loaded, label

    ds, meta = build_integrated_forecast()
    try:
        ds.to_netcdf(INTEGRATED_PATH)
    except Exception:
        pass
    label = (
        f"SIH26071 integrated forecast — Satellite + Radar + IMD observations + ERA5 NWP "
        f"with AI/ML fusion & inundation ({meta.region}, {meta.n_days} days)"
    )
    return ds, label


def daily_series_at_point(ds: xr.Dataset, lat: float, lon: float) -> pd.DataFrame:
    return forecast_series_at_point(ds, lat, lon)


def batch_district_forecasts(ds: xr.Dataset, locations: list[dict]) -> pd.DataFrame:
    """Summarise 7-day forecast for many district HQs in one pass."""
    import numpy as np

    from alerts.alert_thresholds import combined_alert_level

    issue_idx = int(ds.attrs.get("issue_index", len(ds.time) - 1))
    forecast_mask = ds["forecast_phase"].values == "forecast"
    if not forecast_mask.any():
        forecast_mask = np.ones(len(ds.time), dtype=bool)
        forecast_mask[: issue_idx + 1] = False
        if not forecast_mask.any():
            forecast_mask = np.array([True] * len(ds.time))

    rows = []
    for loc in locations:
        li = int(np.argmin(np.abs(ds.lat.values - loc["lat"])))
        lj = int(np.argmin(np.abs(ds.lon.values - loc["lon"])))
        rain = ds["ml_fused_rainfall"].isel(lat=li, lon=lj).values[forecast_mask]
        depth = ds["flood_depth_m"].isel(lat=li, lon=lj).values[forecast_mask]
        risk = ds["flood_risk"].isel(lat=li, lon=lj).values[forecast_mask]
        total_mm = float(np.nansum(rain))
        peak_mm = float(np.nanmax(rain)) if rain.size else 0.0
        peak_risk = int(np.nanmax(risk)) if risk.size else 0
        peak_depth = float(np.nanmax(depth)) if depth.size else 0.0
        peak_alert = combined_alert_level(peak_mm, peak_risk, peak_depth)
        rows.append(
            {
                "District": loc["district"],
                "Location": loc["name"],
                "Predicted (mm)": round(total_mm, 1),
                "Peak (mm)": round(peak_mm, 1),
                "Flood Risk": peak_alert,
                "Risk Class": peak_risk,
                "Trend": "↑" if total_mm > 80 else ("→" if total_mm > 30 else "↓"),
            }
        )
    return pd.DataFrame(rows).sort_values("Predicted (mm)", ascending=False)


def high_rainfall_probability(df: pd.DataFrame, threshold_mm: float = 64.0) -> float:
    """Share of forecast days exceeding IMD Yellow threshold."""
    forecast = df[df["phase"] == "forecast"]
    if forecast.empty:
        forecast = df.tail(7)
    if forecast.empty:
        return 0.0
    return float((forecast["ml_fused_mm"] >= threshold_mm).mean() * 100)
