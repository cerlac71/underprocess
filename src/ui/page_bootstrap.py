"""Shared bootstrap and cached loaders for Streamlit multi-page app."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.bootstrap import bootstrap_project

PROJECT_ROOT = bootstrap_project()

import pandas as pd
import streamlit as st
import xarray as xr

from config.regions import get_active_region
from harmonisation.harmonisation import define_target_grid, harmonize_all
from nowcasting.nowcasting import OpticalFlowNowcaster
from pipeline.pipeline import CONFIG, load_real_data, resolve_nowcast_source, run_pipeline
from ui.daily_forecast import (
    batch_district_forecasts,
    build_daily_forecast_dataset,
    daily_series_at_point,
    high_rainfall_probability,
    outlook_summary,
)
from ui.location_guide import _nearest_index

_region = get_active_region()
RESULTS_PATH = PROJECT_ROOT / "pipeline_results.nc"
METRICS_PATH = Path(CONFIG["data"]["metrics_path"])
VALIDATION_PATH = _region.model_dir(PROJECT_ROOT) / "flood_validation_metrics.json"


@st.cache_data(show_spinner=False)
def load_results(path: str) -> dict:
    with xr.open_dataset(path) as ds:
        out = {var: ds[var].values for var in ("fused_rainfall", "runoff", "flood_depth", "flood_risk", "confidence")}
        out["lat"] = ds["lat"].values
        out["lon"] = ds["lon"].values
        return out


@st.cache_data(show_spinner=False)
def load_metrics(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def run_full_pipeline() -> str:
    run_pipeline()
    return str(RESULTS_PATH)


@st.cache_data(show_spinner=False)
def nowcast_context() -> dict:
    """Run optical-flow nowcast on hourly radar frames for the active region."""
    cfg = CONFIG["target_grid"]
    target_grid = define_target_grid(
        cfg["lat_min"], cfg["lat_max"], cfg["lon_min"], cfg["lon_max"],
        cfg["resolution_km"], grid_step_deg=cfg.get("grid_step_deg"),
    )
    data = load_real_data(target_grid)
    harmonised = harmonize_all(
        satellite_ds=data["satellite"],
        radar_ds=data["radar"],
        nwp_ds=data["nwp"],
        station_df=data["station_df"],
        target_grid=target_grid,
        target_freq="1D",
    )
    nowcast_ds, source_label, nowcast_cfg = resolve_nowcast_source(
        target_grid, data, harmonised, True,
    )
    nowcaster = OpticalFlowNowcaster(use_decay=True, decay_factor=0.95)
    leads = nowcast_cfg["lead_times_min"]
    nowcasts = nowcaster.nowcast(
        nowcast_ds,
        lead_times_min=leads,
        timestep_min=nowcast_cfg["timestep_min"],
    )
    lead_key = max(leads)
    return {
        "nowcast_ds": nowcast_ds,
        "nowcasts": nowcasts,
        "lat": nowcast_ds["lat"].values,
        "lon": nowcast_ds["lon"].values,
        "source": source_label,
        "lead_hours": lead_key / 60.0,
        "lead_key": lead_key,
        "frames": int(nowcast_ds.sizes.get("time", 0)),
        "latest": nowcast_ds["rainfall"].isel(time=-1).values,
        "forecast": nowcasts[lead_key],
    }


def nowcast_series_at_point(lat: float, lon: float) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Observed hourly series and lead-time forecast at one location."""
    ctx = nowcast_context()
    lat_arr, lon_arr = ctx["lat"], ctx["lon"]
    li, lj = _nearest_index(lat_arr, lon_arr, lat, lon)
    ds = ctx["nowcast_ds"]

    observed = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(ds["time"].values),
            "rainfall_mm": ds["rainfall"].isel(lat=li, lon=lj).values.astype(float),
        }
    )
    forecast = pd.DataFrame(
        [
            {"lead_min": lead, "rainfall_mm": float(field[li, lj])}
            for lead, field in sorted(ctx["nowcasts"].items())
        ]
    )
    return observed, forecast, ctx


@st.cache_data(show_spinner=False)
def daily_forecast_context() -> dict:
    """Build SIH26071 integrated daily forecast (all 4 sources + ML + inundation)."""
    ds, label = build_daily_forecast_dataset()
    return {
        "dataset": ds,
        "label": label,
        "lat": ds.lat.values,
        "lon": ds.lon.values,
        "n_days": int(ds.sizes.get("time", 0)),
        "period": ds.attrs.get("period", ""),
        "issue_time": ds.attrs.get("issue_time", ""),
    }


def daily_forecast_series_at_point(lat: float, lon: float) -> tuple[pd.DataFrame, dict]:
    """Daily observed vs NWP vs fused forecast at one location."""
    ctx = daily_forecast_context()
    df = daily_series_at_point(ctx["dataset"], lat, lon)
    return df, ctx


def sidebar_pipeline_controls() -> dict | None:
    """Standard sidebar: region info + pipeline refresh."""
    from ui.styles import inject_theme

    inject_theme()
    st.sidebar.caption(f"{_region.name} · MoES / IMD · 0.25°")
    if METRICS_PATH.exists():
        m = load_metrics(str(METRICS_PATH))
        st.sidebar.metric("Ensemble MAE (holdout)", f"{m['ensemble']['mae_mm']:.1f} mm/day")
    if st.sidebar.button("Run / refresh pipeline", type="primary", use_container_width=True):
        with st.spinner("Running UP pipeline…"):
            run_full_pipeline()
        st.sidebar.success("Pipeline finished.")
        st.cache_data.clear()
        st.rerun()
    if RESULTS_PATH.exists():
        return load_results(str(RESULTS_PATH))
    st.sidebar.warning("No `pipeline_results.nc` yet — run the pipeline.")
    return None
