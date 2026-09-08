#!/usr/bin/env python3
"""SIH26071 multi-tab Streamlit dashboard — real Uttar Pradesh data path."""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from config.bootstrap import bootstrap_project

PROJECT_ROOT = bootstrap_project()

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import xarray as xr

matplotlib.use("Agg")

from config.regions import get_active_region
from src.alerts.alert_thresholds import AlertThresholdModule
from src.alerts.lead_time_tagging import AlertWithLeadTime
from src.data_utils.data_utils import get_districts_for_region, get_sample_historical_events
from src.harmonisation.harmonisation import define_target_grid, harmonize_all
from src.nowcasting.nowcasting import OpticalFlowNowcaster
from src.pipeline.pipeline import CONFIG, load_real_data, resolve_nowcast_source, run_pipeline
from streamlit_folium import st_folium
from ui.location_guide import (
    LEVEL_COLORS,
    SYSTEM_MODULES,
    UP_LOCATIONS,
    forecast_reliability,
    get_advisory_steps,
    location_forecast,
    location_label,
    resolve_location,
    search_locations,
)
from ui.map_viz import build_warning_map

_region = get_active_region()
DISTRICTS = get_districts_for_region(_region.id)
RESULTS_PATH = PROJECT_ROOT / "pipeline_results.nc"
METRICS_PATH = Path(CONFIG["data"]["metrics_path"])

st.set_page_config(page_title=f"Rainfall EWS — {_region.name}", layout="wide")


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
def nowcast_demo():
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
        target_grid, data, harmonised, True
    )
    nowcaster = OpticalFlowNowcaster(use_decay=True, decay_factor=0.95)
    leads = nowcast_cfg["lead_times_min"]
    nowcasts = nowcaster.nowcast(
        nowcast_ds,
        lead_times_min=leads,
        timestep_min=nowcast_cfg["timestep_min"],
    )
    latest = nowcast_ds["rainfall"].isel(time=-1).values
    lead_key = max(leads)
    return {
        "latest": latest,
        "forecast": nowcasts[lead_key],
        "lat": nowcast_ds["lat"].values,
        "lon": nowcast_ds["lon"].values,
        "source": source_label,
        "lead_hours": lead_key / 60.0,
        "frames": int(nowcast_ds.sizes.get("time", 0)),
    }


def summarize_results(res: dict) -> dict:
    am = AlertThresholdModule(
        terrain_adjustment=True,
        hysteresis_window=CONFIG["alerts"]["hysteresis_window"],
        cooldown_period=CONFIG["alerts"]["cooldown_period"],
    )
    terrain_features = {
        "slope": 10.0,
        "flow_accum": 100.0,
        "twi": 6.0,
        "cn": CONFIG["scs"]["default_cn"],
    }
    alert = am.compute_alert(
        region_id=0,
        timestamp=pd.Timestamp.now(),
        rainfall_forecast_p90=float(np.nanpercentile(res["fused_rainfall"], 90)),
        antecedent_rainfall=20.0,
        terrain_features=terrain_features,
        flood_depth=float(np.nanmean(res["flood_depth"])),
        flood_risk=int(round(float(np.nanmean(res["flood_risk"])))),
    )
    confidence = float(np.nanmean(res["confidence"]))
    tagged = AlertWithLeadTime(
        issue_time=datetime.utcnow(),
        valid_time=datetime.utcnow() + timedelta(hours=24),
        source_module="nowcasting",
        alert_level=alert.alert_level,
        confidence=confidence,
        risk_score=float(np.nanmean(res["flood_risk"])) * 100 / 3,
    )
    return {
        "level": alert.alert_level,
        "message": tagged.message,
        "lead_time_hours": tagged.lead_time_hours,
        "lead_time_bucket": tagged.lead_time_bucket,
        "confidence": confidence,
    }


st.sidebar.title("Integrated Forecast System")
st.sidebar.caption(f"{_region.name} · MoES / IMD")
st.sidebar.markdown(
    "Heavy rainfall early warning and inundation prediction using satellite, "
    "radar, observational, and NWP data with AI/ML correction."
)
if METRICS_PATH.exists():
    m = load_metrics(str(METRICS_PATH))
    st.sidebar.metric("Ensemble MAE (2019 holdout)", f"{m['ensemble']['mae_mm']:.1f} mm/day")

if st.sidebar.button("Run Full Pipeline", type="primary", use_container_width=True):
    with st.spinner("Running UP pipeline..."):
        run_full_pipeline()
    st.sidebar.success("Pipeline finished.")
    st.rerun()

results = load_results(str(RESULTS_PATH)) if RESULTS_PATH.exists() else None
if results is None:
    st.sidebar.warning("No results yet. Run the pipeline first.")

data_cfg = CONFIG["data"]
st.sidebar.markdown("---")
st.sidebar.markdown("**Data feeds**")
for mod in SYSTEM_MODULES:
    st.sidebar.caption(f"• {mod['name']}")
st.sidebar.caption(
    f"Domain: {_region.name} ({data_cfg['start_date']} → {data_cfg['end_date']}) · "
    f"{_region.approx_grid_cells:,} cells @ 0.25°"
)

tab_search, tab_overview, tab_nowcast, tab_map, tab_alerts, tab_history = st.tabs(
    ["Location lookup", "Overview", "Nowcast", "Inundation map", "Warnings", "Historical events"]
)

with tab_search:
    st.header("Location lookup")
    if results is None:
        st.info("Run the pipeline from the sidebar first.")
    else:
        query = st.text_input("District or city", placeholder="Start typing — Varanasi, Allahabad, Kanpur Dehat…")
        filtered = search_locations(query) if query.strip() else UP_LOCATIONS
        labels = [location_label(loc) for loc in filtered]
        if not labels:
            st.error("Location not found.")
        else:
            chosen_label = st.selectbox(
                f"{len(labels)} suggestion(s)" if query.strip() else f"All {len(labels)} districts",
                labels,
            )
            selected = resolve_location(chosen_label)
            forecast = location_forecast(results, selected)
            rel = forecast_reliability(forecast["confidence"], load_metrics(str(METRICS_PATH)) if METRICS_PATH.exists() else None)
            advisory = get_advisory_steps(forecast)
            alert_color = LEVEL_COLORS.get(forecast["alert_level"], "#546e7a")

            c1, c2 = st.columns([1.6, 1])
            with c1:
                st_folium(
                    build_warning_map(results, selected=selected, show_rainfall=True, show_risk=True),
                    width="100%",
                    height=580,
                    returned_objects=[],
                )
            with c2:
                st.markdown(f"### {forecast['name']} ({forecast['district']})")
                st.markdown(
                    f"<span style='color:{alert_color};font-size:1.2rem;font-weight:700;'>"
                    f"IMD {forecast['alert_level']}</span>",
                    unsafe_allow_html=True,
                )
                st.metric("24 h rainfall", f"{forecast['rainfall_mm']:.1f} mm")
                st.metric("Inundation depth", f"{forecast['flood_depth_m']:.2f} m")
                st.metric("Reliability", f"{rel:.0%}")
                st.metric("Risk class", f"{forecast['flood_risk']}/3")
                st.subheader("Public advisory")
                for i, step in enumerate(advisory, 1):
                    st.markdown(f"{i}. **{step['heading']}** — {step['detail']}")

with tab_overview:
    st.title(f"Integrated Rainfall EWS — {_region.name}")
    st.markdown(
        f"Multi-source harmonisation, ML bias correction, and inundation modelling "
        f"for `{_region.id}` using `{data_cfg['model_path']}`."
    )
    if results is not None:
        summary = summarize_results(results)
        n_cells = results["flood_risk"].size
        cells_at_risk = int((results["flood_risk"] >= 1).sum())
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Alert Level", summary["level"])
        c2.metric("Mean Fused Rainfall", f"{np.nanmean(results['fused_rainfall']):.1f} mm")
        c3.metric("Max Flood Depth", f"{np.nanmax(results['flood_depth']):.2f} m")
        c4.metric("Cells at Risk", f"{cells_at_risk / n_cells * 100:.1f}%")
        c5.metric("Confidence", f"{summary['confidence']:.2f}")
    else:
        st.info("Run the pipeline from the sidebar.")

with tab_nowcast:
    st.header("0–6 Hour Rainfall Nowcast")
    st.caption("Optical-flow extrapolation on hourly radar frames across the active region.")
    if results is None:
        st.info("Run the pipeline first.")
    else:
        try:
            demo = nowcast_demo()
            st.caption(
                f"Source: **{demo.get('source', 'radar')}** · "
                f"{demo.get('frames', '?')} input frames · "
                f"+{demo.get('lead_hours', 6):.0f} h forecast"
            )
            fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))
            extent = [demo["lon"][0], demo["lon"][-1], demo["lat"][0], demo["lat"][-1]]
            ax0.imshow(demo["latest"], origin="lower", extent=extent, cmap="Blues")
            ax0.set_title("Latest hourly rainfall (mm/hr)")
            ax1.imshow(demo["forecast"], origin="lower", extent=extent, cmap="Blues")
            ax1.set_title(f"Nowcast +{demo.get('lead_hours', 6):.0f} h")
            st.pyplot(fig)
            plt.close(fig)
        except Exception as exc:
            st.warning(f"Nowcast demo unavailable: {exc}")

with tab_map:
    st.header("Rainfall & Inundation Map")
    if results is None:
        st.info("Run the pipeline first.")
    else:
        show_rain = st.checkbox("Rainfall overlay", value=True)
        show_risk = st.checkbox("Flood risk overlay", value=False)
        st_folium(
            build_warning_map(results, show_rainfall=show_rain, show_risk=show_risk),
            width="100%",
            height=620,
            returned_objects=[],
        )

with tab_alerts:
    st.header("Alert Summary")
    if results is None:
        st.info("Run the pipeline first.")
    else:
        summary = summarize_results(results)
        color = LEVEL_COLORS.get(summary["level"], "#9e9e9e")
        st.markdown(
            f"### Current alert level: <span style='color:{color};font-weight:bold'>{summary['level']}</span>",
            unsafe_allow_html=True,
        )
        st.markdown(f"**Message:** {summary['message']}")
        a1, a2, a3 = st.columns(3)
        a1.metric("Lead Time", f"{summary['lead_time_hours']:.1f} h")
        a2.metric("Confidence", f"{summary['confidence']:.2f}")
        a3.metric("Reference city", DISTRICTS[0]["name"])
        st.dataframe(pd.DataFrame(DISTRICTS), use_container_width=True)

with tab_history:
    st.header("Historical Flood Events — Uttar Pradesh")
    metrics_path = PROJECT_ROOT / "models" / "regions" / _region.id / "flood_validation_metrics.json"
    if metrics_path.exists():
        report = json.loads(metrics_path.read_text(encoding="utf-8"))
        agg = report.get("aggregate", {})
        if agg:
            h1, h2, h3, h4, h5 = st.columns(5)
            h1.metric("Holdout events", agg.get("n_events", 0))
            h2.metric("Mean CSI (satellite)", f"{agg.get('mean_csi_satellite', 0) or 0:.3f}")
            h3.metric("Mean F1 (satellite)", f"{agg.get('mean_f1_satellite', 0) or 0:.3f}")
            h4.metric("Mean CSI (pluvial)", f"{agg.get('mean_csi_hydrology', 0):.3f}")
            h5.metric("Mean depth RMSE", f"{agg.get('mean_depth_rmse_m', 0):.4f} m")
        rows = []
        for e in report.get("events", []):
            if "error" in e:
                continue
            sat = e.get("metrics_vs_satellite_observed") or {}
            rows.append(
                {
                    "Event": e["event_name"],
                    "Peak": e["peak_date"],
                    "CSI (satellite)": round(sat.get("csi", 0) or 0, 3),
                    "CSI (pluvial)": round(e["metrics_vs_imd_hydrology"]["csi"], 3),
                    "CSI (runoff)": round(e.get("metrics_runoff_vs_imd", {}).get("csi", 0), 3),
                    "F1": round(sat.get("f1", 0) or e["metrics_vs_imd_hydrology"]["f1"], 3),
                }
            )
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("Full report: `python scripts/run_flood_validation.py` · see Historical Backtest page")
    else:
        st.info("Run `python scripts/run_flood_validation.py` to generate CSI/F1 holdout metrics.")
    events = get_sample_historical_events(_region.id)
    cols = st.columns(len(events))
    for col, (name, info) in zip(cols, events.items()):
        with col:
            st.metric(label=name, value=f"{info['peak_mm']} mm peak")
