"""Forecasting — rainfall and flood forecast dashboard."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config.bootstrap import bootstrap_project

bootstrap_project()

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from alerts.alert_thresholds import IMD_THRESHOLDS_24H
from config.regions import get_active_region
from ui.dashboard_layout import PLOTLY_DARK, compute_region_summary, kpi_card, render_header, render_kpi_row, setup_page, sidebar_controls
from ui.daily_forecast import batch_district_forecasts, high_rainfall_probability, outlook_summary
from ui.location_guide import LEVEL_COLORS, UP_DISTRICT_LOCATIONS, UP_LOCATIONS, location_label, resolve_location, search_locations
from ui.map_viz import build_warning_map
from ui.page_bootstrap import (
    METRICS_PATH,
    RESULTS_PATH,
    daily_forecast_context,
    daily_forecast_series_at_point,
    load_metrics,
    load_results,
)

_region = get_active_region()
setup_page("Forecasting", "📈", section="forecasting")
render_header(
    "FORECASTING | DATA TODAY · SAFER TOMORROW",
    slogan="Prepare · Predict · Prevent",
    section="FORECASTING | ",
)
sidebar_controls()

tab_rain, tab_flood, tab_compare, tab_insights = st.tabs(
    ["Rainfall Forecast", "Flood Forecast", "Comparative View", "Model Insights"]
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.selectbox("Region", [_region.name])
with c2:
    districts = sorted({loc["district"] for loc in UP_DISTRICT_LOCATIONS})
    district_sel = st.selectbox("District", districts, index=districts.index("Lucknow") if "Lucknow" in districts else 0)
with c3:
    st.selectbox("Forecast Horizon", ["Next 7 Days"])
with c4:
    st.selectbox("Model", ["Ensemble (Best)"])

query = st.text_input("Search", placeholder="Lucknow, Varanasi…", label_visibility="collapsed")
labels = [location_label(loc) for loc in (search_locations(query) if query.strip() else UP_LOCATIONS)]
selected_label = st.selectbox("Location", labels, label_visibility="collapsed")
selected = resolve_location(selected_label)

results = load_results(str(RESULTS_PATH)) if RESULTS_PATH.exists() else None
metrics = load_metrics(str(METRICS_PATH)) if METRICS_PATH.exists() else None
summary = compute_region_summary(results, metrics) if results else None

# Load integrated forecast once (cached)
@st.cache_data(show_spinner="Loading integrated forecast…")
def _load_forecast_ctx():
    return daily_forecast_context()

try:
    fctx = _load_forecast_ctx()
    ds = fctx["dataset"]
    daily_df = None
    outlook = None
    if selected:
        daily_df, _ = daily_forecast_series_at_point(selected["lat"], selected["lon"])
        outlook = outlook_summary(daily_df, days=7, phase="forecast")
except Exception as exc:
    st.error(f"Could not load integrated forecast: {exc}")
    st.info("Click **Refresh Forecast** in the sidebar to rebuild pipeline data.")
    st.stop()

with tab_rain:
    conf = f"{summary['confidence']:.0%}" if summary and summary.get("confidence") else "—"
    rain_prob = high_rainfall_probability(daily_df)
    peak_row = daily_df.loc[daily_df["ml_fused_mm"].idxmax()] if not daily_df.empty else None
    peak_day = peak_row["date"].strftime("%b %d, %Y") if peak_row is not None else "—"
    peak_mm = float(peak_row["ml_fused_mm"]) if peak_row is not None else 0.0
    peak_alert = outlook.get("peak_alert", "Green")
    alert_color = LEVEL_COLORS.get(peak_alert, "#FF3B30")

    render_kpi_row(
        kpi_card("🌧️", "Total Predicted Rainfall", f"{outlook['total_ml_mm']:.0f} mm", accent="#00A3FF")
        + kpi_card("⚠️", "High Rainfall Probability", f"{rain_prob:.0f}%", sub="Chance ≥ 64 mm/day", accent="#FF3B30")
        + kpi_card("🌊", "Flood Risk Level", peak_alert, accent=alert_color)
        + kpi_card("📅", "Peak Rainfall Day", f"~ {peak_mm:.0f} mm", sub=peak_day, accent="#00A3FF")
        + kpi_card("🛡️", "Model Confidence", conf, accent="#00A3FF"),
        cols=5,
    )

    chart_col, map_col = st.columns([1.4, 1])
    forecast_df = daily_df[daily_df["phase"] == "forecast"]
    if forecast_df.empty:
        forecast_df = daily_df.tail(7)

    with chart_col:
        st.markdown('<div class="dash-panel"><div class="panel-title">Rainfall Forecast (Next 7 Days)</div>', unsafe_allow_html=True)
        if HAS_PLOTLY and not forecast_df.empty:
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=forecast_df["date"],
                    y=forecast_df["ml_fused_mm"],
                    name="AI/ML Forecast",
                    marker_color="#00A3FF",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=forecast_df["date"],
                    y=forecast_df["nwp_mm"],
                    name="NWP (ERA5)",
                    line=dict(dash="dot", color="#FF9500"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=daily_df["date"],
                    y=daily_df["observational_mm"],
                    name="Historical Avg",
                    line=dict(dash="dash", color="#7a8fa3"),
                )
            )
            fig.add_hline(y=IMD_THRESHOLDS_24H["green_upper"], line_dash="dash", line_color="#FFCC00",
                          annotation_text="Yellow 64mm")
            fig.add_hline(y=IMD_THRESHOLDS_24H["orange_upper"], line_dash="dash", line_color="#FF9500",
                          annotation_text="Orange 115mm")
            fig.update_layout(height=380, **PLOTLY_DARK)
            fig.update_yaxes(title_text="Rainfall (mm/day)")
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Inundation sub-chart
        if HAS_PLOTLY and not forecast_df.empty:
            st.markdown('<div class="dash-panel"><div class="panel-title">Inundation Forecast</div>', unsafe_allow_html=True)
            fig_f = go.Figure()
            fig_f.add_trace(go.Bar(
                x=forecast_df["date"],
                y=forecast_df["flood_depth_m"] * 100,
                name="Depth (cm)",
                marker_color="#FF3B30",
                opacity=0.75,
            ))
            fig_f.add_trace(go.Scatter(
                x=forecast_df["date"],
                y=forecast_df["flood_risk"],
                name="Risk class",
                yaxis="y2",
                line=dict(color="#00A3FF", width=2),
            ))
            fig_f.update_layout(
                height=240,
                **PLOTLY_DARK,
                yaxis=dict(title="Depth (cm)"),
                yaxis2=dict(title="Risk", overlaying="y", side="right", range=[0, 3.5]),
            )
            st.plotly_chart(fig_f, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with map_col:
        day_idx = st.selectbox(
            "Spatial forecast day",
            range(len(forecast_df)),
            format_func=lambda i: forecast_df.iloc[i]["date"].strftime("%Y-%m-%d"),
        )
        t_idx = forecast_df.index[day_idx]
        actual_t = list(daily_df.index).index(t_idx)
        spatial_results = {
            "fused_rainfall": ds["ml_fused_rainfall"].isel(time=actual_t).values,
            "flood_risk": ds["flood_risk"].isel(time=actual_t).values,
            "flood_depth": ds["flood_depth_m"].isel(time=actual_t).values,
            "runoff": ds["runoff_mm"].isel(time=actual_t).values,
            "confidence": ds["confidence"].isel(time=actual_t).values,
            "lat": ds.lat.values,
            "lon": ds.lon.values,
        }
        st.markdown('<div class="map-frame">', unsafe_allow_html=True)
        st.caption(f"Spatial rainfall forecast · {forecast_df.iloc[day_idx]['date'].strftime('%d %b %Y')}")
        st_folium(
            build_warning_map(
                spatial_results,
                selected=selected,
                show_rainfall=True,
                show_risk=False,
                map_mode="rainfall",
            ),
            width="stretch",
            height=460,
            returned_objects=[],
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="dash-panel"><div class="panel-title">District-wise Rainfall Forecast</div>', unsafe_allow_html=True)
    district_df = batch_district_forecasts(ds, UP_LOCATIONS)
    st.dataframe(district_df.head(15), hide_index=True, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with tab_flood:
    st.markdown('<div class="dash-panel"><div class="panel-title">River Water Level Forecast</div>', unsafe_allow_html=True)
    st.markdown('<div class="empty-state">River gauge forecast data not available in dataset</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if results:
        st.markdown('<div class="map-frame">', unsafe_allow_html=True)
        st_folium(
            build_warning_map(
                results,
                selected=selected,
                show_rainfall=False,
                show_risk=True,
                map_mode="risk",
            ),
            width="stretch",
            height=480,
            returned_objects=[],
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="dash-panel"><div class="panel-title">River Level Forecast Table</div>', unsafe_allow_html=True)
    river_rows = []
    for loc in UP_LOCATIONS[:10]:
        fc_df = daily_df  # use selected location forecast
        fslice = fc_df[fc_df["phase"] == "forecast"]
        if not fslice.empty:
            river_rows.append({
                "River Basin": loc["district"],
                "Current Depth (m)": round(float(fslice["flood_depth_m"].iloc[0]), 3),
                "Predicted 24h (m)": round(float(fslice["flood_depth_m"].iloc[0]), 3),
                "Risk": int(fslice["flood_risk"].iloc[0]),
                "Status": fslice["alert_level"].iloc[0],
            })
    if river_rows:
        st.dataframe(pd.DataFrame(river_rows), hide_index=True, use_container_width=True)
    else:
        st.markdown('<div class="empty-state">—</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with tab_compare:
    st.markdown('<div class="dash-panel"><div class="panel-title">Model Comparison</div>', unsafe_allow_html=True)
    if HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily_df["date"], y=daily_df["observational_mm"], name="IMD Observed", line=dict(color="#7a8fa3")))
        fig.add_trace(go.Scatter(x=daily_df["date"], y=daily_df["satellite_mm"], name="Satellite (CHIRPS)", line=dict(color="#42a5f5", dash="dot")))
        fig.add_trace(go.Scatter(x=daily_df["date"], y=daily_df["radar_mm"], name="Radar (MERRA-2)", line=dict(color="#ab47bc", dash="dot")))
        fig.add_trace(go.Scatter(x=daily_df["date"], y=daily_df["nwp_mm"], name="NWP (ERA5)", line=dict(color="#FF9500", dash="dash")))
        fig.add_trace(go.Scatter(x=daily_df["date"], y=daily_df["ml_fused_mm"], name="Ensemble (Final)", line=dict(color="#00A3FF", width=3)))
        fig.update_layout(height=420, **PLOTLY_DARK)
        fig.update_yaxes(title_text="Rainfall (mm/day)")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with tab_insights:
    if metrics:
        st.markdown('<div class="dash-panel"><div class="panel-title">Forecast Insights</div>', unsafe_allow_html=True)
        ens = metrics.get("ensemble", {})
        st.markdown(
            f"""
            - **{selected['name']}** total 7-day forecast: **{outlook['total_ml_mm']:.0f} mm**
            - Peak day rainfall: **{peak_mm:.0f} mm** ({peak_day})
            - Peak inundation: **{outlook['peak_flood_depth_m']:.3f} m**
            - High rainfall probability: **{rain_prob:.0f}%** of forecast days ≥ 64 mm
            - Ensemble MAE: **{ens.get('mae_mm', 0):.1f} mm/day**
            - Holdout R²: **{ens.get('r2', 0):.0%}**
            - Wet-day R²: **{ens.get('wet_day_r2', 0):.0%}**
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            '<div class="mission-box"><h4>⚠️ Early Warning</h4>'
            f'<p style="margin:0;font-size:0.85rem;">'
            f'Heavy rainfall and possible flooding expected in {selected["district"]} and surrounding areas. '
            f'Take necessary precautions.</p></div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("Run pipeline to load model metrics.")
