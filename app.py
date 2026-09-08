"""Home — Heavy Rainfall and Flood Prediction Dashboard."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from config.bootstrap import bootstrap_project

PROJECT_ROOT = bootstrap_project()

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit_folium import st_folium

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from config.regions import get_active_region
from ui.dashboard_layout import (
    PLOTLY_DARK,
    compute_region_summary,
    district_forecast_table,
    kpi_card,
    render_alert_cards,
    render_header,
    render_kpi_row,
    setup_page,
    sidebar_controls,
)
from ui.location_guide import LEVEL_COLORS, UP_LOCATIONS, location_forecast, location_label, resolve_location, search_locations
from ui.map_viz import build_warning_map
from ui.page_bootstrap import (
    METRICS_PATH,
    RESULTS_PATH,
    daily_forecast_context,
    daily_forecast_series_at_point,
    load_metrics,
    load_results,
)
from ui.daily_forecast import outlook_summary

_region = get_active_region()
setup_page("Home", "🏠", section="home")
render_header(
    "DATA DRIVEN · SAFER TOMORROW",
    slogan="MONITOR · PREDICT · PREPARE · PROTECT",
    section="HOME | ",
)
layer_opts = sidebar_controls(RESULTS_PATH.exists())

if not RESULTS_PATH.exists():
    st.warning("Forecast not loaded. Click **Refresh Forecast** in the sidebar.")
    st.stop()

results = load_results(str(RESULTS_PATH))
metrics = load_metrics(str(METRICS_PATH)) if METRICS_PATH.exists() else None
summary = compute_region_summary(results, metrics)

conf_pct = f"{summary['confidence']:.0%}" if summary["confidence"] is not None else "—"
risk_color = "#FF3B30" if summary["max_risk"] >= 2 else ("#FF9500" if summary["max_risk"] >= 1 else "#34C759")

render_kpi_row(
    kpi_card("🌧️", "Total Rainfall (24h)", f"{summary['total_rainfall_mm']} mm", accent="#00A3FF")
    + kpi_card("⚠️", "Flood Risk", summary["risk_label"], accent=risk_color)
    + kpi_card("📍", "Affected Areas", f"{summary['affected_districts']} Districts", accent="#FF9500")
    + kpi_card(
        "🔔",
        "Active Alerts",
        str(summary["active_alerts"]),
        sub=f"{summary['high_alerts']} High | {summary['med_alerts']} Medium" if summary["active_alerts"] else "",
        accent="#FF3B30",
    )
    + kpi_card("🌊", "Peak Inundation", f"{summary['peak_depth_m']:.2f} m", accent="#FF3B30")
    + kpi_card("🛡️", "Model Confidence", conf_pct, sub="Ensemble ML", accent="#00A3FF"),
    cols=6,
)

# Location picker for forecast + weather
search_col, map_mode_col = st.columns([2, 1])
with search_col:
    query = st.text_input("location", placeholder="Search location (e.g. Lucknow, Varanasi)…", label_visibility="collapsed")
    filtered = search_locations(query) if query.strip() else search_locations("Lucknow")[:1] or UP_LOCATIONS[:1]
    labels = [location_label(loc) for loc in filtered[:20]]
    chosen = st.selectbox("loc", labels, label_visibility="collapsed")
    selected_loc = resolve_location(chosen) or UP_LOCATIONS[0]

with map_mode_col:
    map_view = st.radio("Map view", ["Rainfall", "Flood Risk", "Combined"], horizontal=True, label_visibility="collapsed")

map_mode = {"Rainfall": "rainfall", "Flood Risk": "risk", "Combined": "combined"}[map_view]
show_rain = map_mode in ("rainfall", "combined")
show_risk = map_mode in ("risk", "combined")

col_chart, col_map, col_alerts = st.columns([1.2, 1.3, 0.8], gap="small")

with col_chart:
    st.markdown('<div class="dash-panel"><div class="panel-title">Rainfall Forecast (Next 7 Days)</div>', unsafe_allow_html=True)
    try:
        with st.spinner("Loading forecast…"):
            daily_df, _ = daily_forecast_series_at_point(selected_loc["lat"], selected_loc["lon"])
        forecast_df = daily_df[daily_df["phase"] == "forecast"]
        if forecast_df.empty:
            forecast_df = daily_df.tail(7)
        outlook = outlook_summary(daily_df, days=7, phase="forecast")

        if HAS_PLOTLY and not forecast_df.empty:
            fig = go.Figure()
            dates = forecast_df["date"].dt.strftime("%b %d")
            fig.add_trace(go.Bar(x=dates, y=forecast_df["ml_fused_mm"], name="Forecast", marker_color="#00A3FF"))
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=forecast_df["observational_mm"],
                    name="Historical",
                    line=dict(dash="dash", color="#7a8fa3"),
                )
            )
            fig.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=40), **PLOTLY_DARK)
            fig.update_yaxes(title_text="Rainfall (mm)")
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"**{selected_loc['name']}** · Total forecast: {outlook['total_ml_mm']:.0f} mm · Peak alert: {outlook['peak_alert']}")
        else:
            st.dataframe(forecast_df[["date", "ml_fused_mm", "alert_level"]], hide_index=True)
    except Exception as exc:
        st.markdown(f'<div class="empty-state">Forecast unavailable: {exc}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_map:
    st.markdown('<div class="map-frame">', unsafe_allow_html=True)
    st_folium(
        build_warning_map(
            results,
            selected=selected_loc,
            show_rainfall=show_rain,
            show_risk=show_risk,
            show_stations=layer_opts["show_stations"],
            show_alerts=True,
            map_mode=map_mode,
        ),
        width="stretch",
        height=400,
        returned_objects=[],
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col_alerts:
    render_alert_cards(summary["district_rows"], title="Recent Alerts", limit=6)

st.markdown("---")
b1, b2, b3, b4 = st.columns(4)

with b1:
    st.markdown('<div class="dash-panel"><div class="panel-title">River Water Levels</div>', unsafe_allow_html=True)
    st.markdown('<div class="empty-state">River gauge data not available</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with b2:
    st.markdown('<div class="dash-panel"><div class="panel-title">Affected Districts</div>', unsafe_allow_html=True)
    aff_df = district_forecast_table(results, metrics, limit=75)
    aff_df = aff_df[aff_df["Risk"] >= 1] if (aff_df["Risk"] >= 1).any() else aff_df.nlargest(6, "Rainfall (mm)")
    st.dataframe(
        aff_df[["District", "Rainfall (mm)", "Risk", "Alert"]].rename(columns={"Alert": "Status"}),
        hide_index=True,
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with b3:
    st.markdown('<div class="dash-panel"><div class="panel-title">Weather Summary</div>', unsafe_allow_html=True)
    fc = location_forecast(results, selected_loc)
    peak_color = LEVEL_COLORS.get(fc["alert_level"], "#00A3FF")
    st.markdown(
        f"""
        <div style="text-align:center;padding:12px 0;">
            <div style="font-size:2.5rem;font-weight:700;color:#fff;">{fc['rainfall_mm']:.0f} mm</div>
            <div style="color:{peak_color};font-weight:600;">{fc['alert_level']} — 24h Rainfall</div>
            <div style="font-size:0.85rem;color:#c8d6e5;margin-top:4px;">{selected_loc['name']}, {selected_loc['district']}</div>
            <div style="color:#7a8fa3;font-size:0.8rem;margin-top:8px;">
                Depth: {fc['flood_depth_m']:.2f} m · Risk: {fc['flood_risk']}/3
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with b4:
    st.markdown(
        """
        <div class="dash-panel">
            <div class="panel-title">System Architecture</div>
            <div style="font-size:0.72rem;color:#7a8fa3;line-height:1.8;">
                <b style="color:#00A3FF;">Data Sources</b> → CHIRPS · IMD · ERA5 · MERRA-2<br>
                <b style="color:#00A3FF;">Pipeline</b> → Ingestion · Preprocessing<br>
                <b style="color:#00A3FF;">Prediction</b> → ML/DL Ensemble Models<br>
                <b style="color:#00A3FF;">Application</b> → Dashboard · Alerts<br>
                <b style="color:#00A3FF;">Users</b> → Government · Public
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

components.html(
    """
    <div class="mission-box" style="margin-top:12px;">
        <h4>⚠️ EARLY WARNING</h4>
        <p style="margin:0;font-size:0.85rem;">
            Monitor heavy rainfall and possible flooding across Uttar Pradesh.
            Take necessary precautions in high-risk districts.
        </p>
    </div>
    """,
    height=90,
)

st.caption(f"SIH26071 · {_region.name} · {len(UP_LOCATIONS)} district monitoring points · MoES / IMD prototype")
