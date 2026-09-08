"""Live Map — real-time rainfall and flood monitoring."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config.bootstrap import bootstrap_project

bootstrap_project()

import streamlit as st
import streamlit.components.v1 as components
from streamlit_folium import st_folium

from ui.dashboard_layout import (
    compute_region_summary,
    district_forecast_table,
    filter_bar,
    kpi_card,
    render_header,
    render_kpi_row,
    setup_page,
    sidebar_controls,
)
from ui.location_guide import UP_LOCATIONS, location_label, resolve_location, search_locations
from ui.map_viz import build_warning_map
from ui.page_bootstrap import METRICS_PATH, RESULTS_PATH, load_metrics, load_results

setup_page("Live Map", "🗺️", section="live_map")
render_header(
    "LIVE MAP | REAL-TIME DATA · SAFER COMMUNITIES",
    slogan="Prepare Today · Stay Aware · Stay Safe",
    section="LIVE MAP | ",
)
layer_opts = sidebar_controls()
filters = filter_bar()

if not RESULTS_PATH.exists():
    st.info("Run the pipeline from the sidebar to load map data.")
    st.stop()

results = load_results(str(RESULTS_PATH))
metrics = load_metrics(str(METRICS_PATH)) if METRICS_PATH.exists() else None
summary = compute_region_summary(results, metrics)

query = filters["search"] or ""
filtered = search_locations(query) if query.strip() else UP_LOCATIONS
labels = [location_label(loc) for loc in filtered[:30]]
selected_label = st.selectbox("Focus location", labels, label_visibility="collapsed")
selected = resolve_location(selected_label)

show_rain = layer_opts["show_rainfall"] and (
    "Rainfall" in filters["data_layer"] or filters["data_layer"] == "All Layers"
)
show_risk = layer_opts["show_risk"] and (
    "Flood" in filters["data_layer"] or filters["data_layer"] == "All Layers"
)
if show_rain and show_risk:
    map_mode = "combined"
elif show_risk:
    map_mode = "risk"
else:
    map_mode = "rainfall"

map_col, side_col = st.columns([2.2, 1], gap="medium")

with map_col:
    st.markdown('<div class="map-frame">', unsafe_allow_html=True)
    st_folium(
        build_warning_map(
            results,
            selected=selected,
            show_rainfall=show_rain,
            show_risk=show_risk,
            show_stations=layer_opts["show_stations"],
            show_alerts=True,
            map_mode=map_mode,
        ),
        width="stretch",
        height=580,
        returned_objects=[],
    )
    st.markdown("</div>", unsafe_allow_html=True)

    timeline_hour = st.slider("Timeline", 0, 6, 6, format="%d h")
    components.html(
        f"""
        <div class="timeline-bar">
            <span style="color:#00A3FF;">⏸</span>
            <span class="timeline-live">● Live</span>
            <div style="flex:1;height:4px;background:rgba(0,163,255,0.2);border-radius:2px;position:relative;">
                <div style="width:75%;height:100%;background:#00A3FF;border-radius:2px;"></div>
                <div style="position:absolute;right:25%;top:-5px;width:14px;height:14px;
                            background:#00A3FF;border-radius:50%;border:2px solid #fff;"></div>
            </div>
            <span style="color:#7a8fa3;font-size:0.75rem;">Now · {timeline_hour}h</span>
        </div>
        """,
        height=50,
    )

with side_col:
    render_kpi_row(
        kpi_card("🌧️", "Current Rainfall", f"{summary['total_rainfall_mm']} mm/day", accent="#00A3FF")
        + kpi_card("🌊", "Peak Inundation", f"{summary['peak_depth_m']:.2f} m", accent="#FF3B30")
        + kpi_card("⚠️", "Active Alerts", f"{summary['active_alerts']} Districts", accent="#FF3B30")
        + kpi_card("📊", "Cells at Risk", f"{summary['cells_at_risk_pct']}%", accent="#FF9500"),
        cols=2,
    )

    st.markdown('<div class="dash-panel"><div class="panel-title">Live Stations</div>', unsafe_allow_html=True)
    stations = district_forecast_table(results, metrics, limit=8)
    st.dataframe(stations[["Station", "Rainfall (mm)", "Depth (m)", "Alert"]], hide_index=True, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="dash-panel"><div class="panel-title">Map Layers</div>', unsafe_allow_html=True)
    layers = [
        ("Rainfall (Live)", layer_opts["show_rainfall"]),
        ("Flood Risk Zones", layer_opts["show_risk"]),
        ("District Boundary", False),
        ("Weather Stations", layer_opts["show_stations"]),
        ("River Network", False),
        ("Affected Areas", layer_opts["show_risk"]),
    ]
    for name, on in layers:
        icon = "☑" if on else "☐"
        st.markdown(f"<span style='color:#7a8fa3;font-size:0.8rem;'>{icon} {name}</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="dash-panel"><div class="panel-title">Satellite (Live)</div>'
        '<div class="empty-state">Himawari-9 preview not available</div></div>',
        unsafe_allow_html=True,
    )

st.caption("Real pipeline output from pipeline_results.nc — fused rainfall and terrain-based inundation on UP 0.25° grid.")
