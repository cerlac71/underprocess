"""Reports — rainfall and flood analysis reports."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config.bootstrap import bootstrap_project

bootstrap_project()

import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from config.regions import get_active_region
from ui.dashboard_layout import PLOTLY_DARK, compute_region_summary, district_forecast_table, kpi_card, render_header, render_kpi_row, setup_page, sidebar_controls
from ui.location_guide import UP_LOCATIONS
from ui.page_bootstrap import METRICS_PATH, RESULTS_PATH, daily_forecast_series_at_point, load_metrics, load_results
from ui.daily_forecast import outlook_summary

_region = get_active_region()
setup_page("Reports", "📋", section="reports")
render_header(
    "REPORTS | DATA INSIGHTS FOR A SAFER TOMORROW",
    slogan="From Data to Action",
    section="REPORTS | ",
)
sidebar_controls()

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    report_type = st.selectbox("Report Type", ["Rainfall & Flood Report"])
with c2:
    region_sel = st.selectbox("Region", [_region.name])
with c3:
    district_sel = st.selectbox("District", ["All Districts"])
with c4:
    period = st.selectbox("Time Period", ["Last 7 Days"])
with c5:
    st.button("📊 Generate Report", type="primary", use_container_width=True)

if not RESULTS_PATH.exists():
    st.warning("Run pipeline to generate reports.")
    st.stop()

results = load_results(str(RESULTS_PATH))
metrics = load_metrics(str(METRICS_PATH)) if METRICS_PATH.exists() else None
summary = compute_region_summary(results, metrics)

render_kpi_row(
    kpi_card("🌧️", "Total Rainfall", f"{summary['total_rainfall_mm']} mm", accent="#00A3FF")
    + kpi_card("🌊", "Peak Inundation", f"{summary['peak_depth_m']:.2f} m", accent="#FF3B30")
    + kpi_card("📍", "Affected Areas", f"{summary['affected_districts']} Districts", accent="#FF9500")
    + kpi_card("👥", "Affected Population", "—", sub="Not available", accent="#FF3B30")
    + kpi_card("🔔", "Alerts Issued", str(summary["active_alerts"]), sub=f"{summary['high_alerts']} High | {summary['med_alerts']} Med", accent="#FF3B30"),
    cols=5,
)

left, center, right = st.columns([1, 1.2, 1])

with left:
    st.markdown('<div class="dash-panel"><div class="panel-title">Rainfall Report</div>', unsafe_allow_html=True)
    try:
        loc = UP_LOCATIONS[0]
        daily_df, _ = daily_forecast_series_at_point(loc["lat"], loc["lon"])
        if HAS_PLOTLY:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=daily_df["date"], y=daily_df["ml_fused_mm"], name="Actual", marker_color="#00A3FF"))
            fig.add_trace(go.Scatter(x=daily_df["date"], y=daily_df["observational_mm"], name="Normal", line=dict(dash="dash", color="#fff")))
            fig.update_layout(height=280, **PLOTLY_DARK)
            st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.markdown('<div class="empty-state">—</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="dash-panel"><div class="panel-title">Alert Summary</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="text-align:center;padding:20px;">
            <div style="font-size:2rem;font-weight:700;color:#fff;">{summary['active_alerts']}</div>
            <div style="color:#7a8fa3;">Total Active Alerts</div>
            <div style="margin-top:12px;font-size:0.8rem;">
                <span style="color:#FF3B30;">● High {summary['high_alerts']}</span> &nbsp;
                <span style="color:#FF9500;">● Medium {summary['med_alerts']}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="dash-panel"><div class="panel-title">Key Insights</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        - State-wide mean rainfall: **{summary['total_rainfall_mm']} mm/day**
        - Peak inundation depth: **{summary['peak_depth_m']:.2f} m**
        - **{summary['affected_districts']}** districts with elevated risk
        - **{summary['cells_at_risk_pct']}%** of grid cells at risk ≥ 1
        - Model confidence: **{f"{summary['confidence']:.0%}" if summary.get('confidence') else "—"}**
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with center:
    st.markdown('<div class="dash-panel"><div class="panel-title">Flood Risk Report</div>', unsafe_allow_html=True)
    risk_df = district_forecast_table(results, metrics, limit=10)
    risk_df["Status"] = risk_df["Alert"]
    st.dataframe(risk_df[["District", "Rainfall (mm)", "Risk", "Status"]], hide_index=True, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="dash-panel"><div class="panel-title">River Water Level Report</div>', unsafe_allow_html=True)
    st.markdown('<div class="empty-state">River gauge data not available</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="dash-panel"><div class="panel-title">Recommendations</div>', unsafe_allow_html=True)
    st.markdown(
        """
        ✅ Continue monitoring high-risk districts<br>
        ✅ Issue alerts for Orange/Red thresholds<br>
        ✅ Coordinate relief teams in flood-prone areas<br>
        ✅ Update risk models with latest observations<br>
        ✅ Share advisories with local authorities
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="dash-panel"><div class="panel-title">Affected Areas Report</div>', unsafe_allow_html=True)
    aff = district_forecast_table(results, metrics, limit=10)
    aff = aff[aff["Risk"] >= 1] if (aff["Risk"] >= 1).any() else aff.head(6)
    st.dataframe(aff[["District", "Rainfall (mm)", "Depth (m)", "Alert"]], hide_index=True, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="dash-panel"><div class="panel-title">Download Reports</div>', unsafe_allow_html=True)
    btn_cols = st.columns(2)
    btn_cols[0].button("📄 PDF Report", use_container_width=True, disabled=True)
    btn_cols[1].button("📊 Excel Report", use_container_width=True, disabled=True)
    btn_cols = st.columns(2)
    btn_cols[0].button("🗺️ Map Report", use_container_width=True, disabled=True)
    btn_cols[1].button("📝 Summary", use_container_width=True, disabled=True)
    st.caption("Export not yet implemented")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="dash-panel" style="text-align:center;font-style:italic;color:#7a8fa3;">'
        '"Data Today · Resilient Communities Tomorrow."</div>',
        unsafe_allow_html=True,
    )
