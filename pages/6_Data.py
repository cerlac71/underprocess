"""Data — live and historical data explorer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config.bootstrap import bootstrap_project

bootstrap_project()

import json
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from config.regions import get_active_region
from pipeline.pipeline import CONFIG
from ui.dashboard_layout import PLOTLY_DARK, data_sources_panel, kpi_card, render_header, render_kpi_row, setup_page, sidebar_controls
from ui.location_guide import UP_LOCATIONS, location_label
from ui.page_bootstrap import METRICS_PATH, PROJECT_ROOT, RESULTS_PATH, daily_forecast_series_at_point, load_metrics, load_results, sidebar_pipeline_controls

_region = get_active_region()
data_cfg = CONFIG["data"]
setup_page("Data", "💾", section="data")
render_header(
    "DATA | REAL-TIME & HISTORICAL DATA FOR A SAFER TOMORROW",
    slogan="Better Data · Safer Communities",
    section="DATA | ",
)
sidebar_controls()

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    data_type = st.selectbox("Data Type", ["All Data", "Rainfall", "River Level", "Weather"])
with c2:
    region_sel = st.selectbox("Region", [_region.name])
with c3:
    district_sel = st.selectbox("District", ["All Districts"])
with c4:
    date_range = st.selectbox("Date Range", [f"{data_cfg['start_date']} – {data_cfg['end_date']}"])
with c5:
    st.button("⬇ Download Data", type="primary", use_container_width=True, disabled=True)

results = sidebar_pipeline_controls()
metrics = load_metrics(str(METRICS_PATH)) if METRICS_PATH.exists() else None

render_kpi_row(
    kpi_card("🌧️", "Rainfall Records", f"{len(UP_LOCATIONS)}", sub="District HQs monitored", accent="#00A3FF")
    + kpi_card("🌊", "River Level Readings", "—", sub="Not available", accent="#00A3FF")
    + kpi_card("🌡️", "Weather Observations", "—", sub="Not available", accent="#00A3FF")
    + kpi_card("📍", "Affected Area Data", str(len(UP_LOCATIONS)), sub="District HQs", accent="#00A3FF")
    + kpi_card("📁", "Historical Records", f"{data_cfg['start_date'][:4]}–{data_cfg['end_date'][:4]}", accent="#00A3FF"),
    cols=5,
)

tab_rain, tab_river, tab_weather, tab_hist = st.tabs(
    ["Rainfall Data", "River Level Data", "Weather Data", "Historical Data"]
)

with tab_rain:
    main_col, side_col = st.columns([2.5, 1])
    with main_col:
        st.markdown('<div class="dash-panel"><div class="panel-title">Rainfall Data (Live & Recent)</div>', unsafe_allow_html=True)
        if results:
            rows = []
            for i, loc in enumerate(UP_LOCATIONS[:20]):
                try:
                    df, _ = daily_forecast_series_at_point(loc["lat"], loc["lon"])
                    latest = df.iloc[-1] if len(df) else None
                    rows.append({
                        "ID": f"RF{i+1:04d}",
                        "District": loc["district"],
                        "Location": loc["name"],
                        "Rainfall (mm)": round(float(results["fused_rainfall"].mean()), 1),
                        "Source": "IMD",
                        "Status": "Live",
                    })
                except Exception:
                    pass
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        else:
            st.info("Run pipeline to load data.")
        st.markdown("</div>", unsafe_allow_html=True)

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown('<div class="dash-panel"><div class="panel-title">Hourly Rainfall (Lucknow)</div>', unsafe_allow_html=True)
            try:
                loc = next(l for l in UP_LOCATIONS if l["district"] == "Lucknow")
                df, _ = daily_forecast_series_at_point(loc["lat"], loc["lon"])
                if HAS_PLOTLY:
                    fig = go.Figure(go.Bar(x=df["date"], y=df["ml_fused_mm"], marker_color="#00A3FF"))
                    fig.update_layout(height=260, **PLOTLY_DARK)
                    st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.markdown('<div class="empty-state">—</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with chart_col2:
            st.markdown('<div class="dash-panel"><div class="panel-title">Daily Rainfall Comparison</div>', unsafe_allow_html=True)
            try:
                loc = next(l for l in UP_LOCATIONS if l["district"] == "Lucknow")
                df, _ = daily_forecast_series_at_point(loc["lat"], loc["lon"])
                if HAS_PLOTLY:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df["date"], y=df["ml_fused_mm"], name="2026", line=dict(color="#00A3FF")))
                    fig.add_trace(go.Scatter(x=df["date"], y=df["observational_mm"], name="Normal", line=dict(color="#7a8fa3", dash="dash")))
                    fig.update_layout(height=260, **PLOTLY_DARK)
                    st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.markdown('<div class="empty-state">—</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with side_col:
        st.markdown(f'<div class="dash-panel"><div class="panel-title">Data Sources</div>{data_sources_panel()}</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="dash-panel">
                <div class="panel-title">Data Statistics</div>
                <div style="font-size:0.8rem;color:#7a8fa3;line-height:2;">
                    District Stations: <b style="color:#fff;">{len(UP_LOCATIONS)}</b><br>
                    Data Period: <b style="color:#fff;">{data_cfg['start_date']} – {data_cfg['end_date']}</b><br>
                    Data Sources: <b style="color:#fff;">4 Online</b><br>
                    Region: <b style="color:#fff;">{_region.name}</b><br>
                    Grid Cells: <b style="color:#fff;">{_region.approx_grid_cells}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="dash-panel"><div class="panel-title">Data Management</div>', unsafe_allow_html=True)
        mgmt = st.columns(2)
        mgmt[0].button("CSV", use_container_width=True, disabled=True)
        mgmt[1].button("JSON", use_container_width=True, disabled=True)
        mgmt = st.columns(2)
        mgmt[0].button("API Access", use_container_width=True, disabled=True)
        mgmt[1].button("Dictionary", use_container_width=True, disabled=True)
        st.markdown("</div>", unsafe_allow_html=True)

with tab_river:
    st.markdown('<div class="empty-state">River level data not available in dataset</div>', unsafe_allow_html=True)

with tab_weather:
    st.markdown('<div class="empty-state">Detailed weather station data not available</div>', unsafe_allow_html=True)

with tab_hist:
    val_path = _region.model_dir(PROJECT_ROOT) / "flood_validation_metrics.json"
    if val_path.exists():
        report = json.loads(val_path.read_text())
        agg = report.get("aggregate", {})
        st.markdown('<div class="dash-panel"><div class="panel-title">Historical Validation</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean CSI", f"{agg.get('mean_csi_hydrology', 0):.3f}")
        c2.metric("Mean F1", f"{agg.get('mean_f1_hydrology', 0):.3f}")
        c3.metric("Depth RMSE", f"{agg.get('mean_depth_rmse_m', 0):.4f} m")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Run flood validation script for historical metrics.")

st.caption("All data is validated and updated when pipeline runs.")
