"""Alerts — real-time monitoring and notifications."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config.bootstrap import bootstrap_project

bootstrap_project()

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit_folium import st_folium

from alerts.subscriptions import SubscriptionStore
from ui.dashboard_layout import (
    compute_region_summary,
    filter_bar,
    kpi_card,
    render_alert_cards,
    render_header,
    render_kpi_row,
    setup_page,
    sidebar_controls,
    status_badge,
)
from ui.location_guide import LEVEL_COLORS, UP_DISTRICT_LOCATIONS, UP_LOCATIONS, location_forecast, location_label
from ui.map_viz import build_warning_map
from ui.page_bootstrap import METRICS_PATH, PROJECT_ROOT, load_metrics, sidebar_pipeline_controls

setup_page("Alerts", "🚨", section="alerts")
render_header(
    "REAL-TIME MONITORING · EARLY WARNINGS · SAFER COMMUNITIES",
    slogan="Prepare Today · Stay Aware · Stay Safe",
    section="ALERTS | ",
)
layer_opts = sidebar_controls()
filters = filter_bar()

results = sidebar_pipeline_controls()
if results is None:
    st.stop()

metrics = load_metrics(str(METRICS_PATH)) if METRICS_PATH.exists() else None
summary = compute_region_summary(results, metrics)
sub_store = SubscriptionStore(PROJECT_ROOT / "alerts" / "subscriptions.json")

rows = []
for loc in UP_LOCATIONS:
    fc = location_forecast(results, loc)
    rows.append({
        "Location": location_label(loc),
        "District": loc["district"],
        "Rainfall (mm)": round(fc["rainfall_mm"], 1),
        "Alert": fc["alert_level"],
        "Risk": fc["flood_risk"],
        "Depth (m)": round(fc["flood_depth_m"], 2),
    })
df = pd.DataFrame(rows)
active = df[df["Alert"] != "Green"]
high = df[df["Alert"].isin(["Orange", "Red"])]
med = df[df["Alert"] == "Yellow"]
low = df[(df["Alert"] == "Green") & (df["Risk"] >= 1)]

map_col, side_col = st.columns([2, 1], gap="medium")

with map_col:
    st.markdown('<div class="map-frame">', unsafe_allow_html=True)
    st_folium(
        build_warning_map(
            results,
            show_rainfall=layer_opts["show_rainfall"],
            show_risk=True,
            show_alerts=True,
            map_mode="combined",
        ),
        width="stretch",
        height=520,
        returned_objects=[],
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if not active.empty:
        top = active.iloc[0]
        st.markdown(
            f"""
            <div class="dash-panel" style="border-left:4px solid {LEVEL_COLORS.get(top['Alert'], '#FF3B30')};">
                <div class="panel-title">Alert Details</div>
                <div style="display:flex;gap:16px;flex-wrap:wrap;">
                    <div><b>Heavy Rainfall Warning</b><br>
                    <span style="color:{LEVEL_COLORS.get(top['Alert'])};">Severity: {top['Alert']}</span></div>
                    <div><b>District</b><br>{top['District']}</div>
                    <div><b>Rainfall</b><br>{top['Rainfall (mm)']} mm (24h)</div>
                    <div><b>Depth</b><br>{top['Depth (m)']} m</div>
                    <div><b>Risk</b><br>{top['Risk']} / 3</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with side_col:
    render_kpi_row(
        kpi_card("⚠️", "Active Alerts", f"{len(active)} Districts", accent="#FF3B30")
        + kpi_card("🔴", "High", str(len(high)), accent="#FF3B30")
        + kpi_card("🟠", "Medium", str(len(med)), accent="#FF9500")
        + kpi_card("🔵", "Low", str(len(low)), accent="#00A3FF"),
        cols=2,
    )

    tab_latest, tab_all = st.tabs(["Latest Alerts", "All Alerts"])
    with tab_latest:
        render_alert_cards(summary["district_rows"], title="Latest Alerts", limit=8)
    with tab_all:
        st.dataframe(df, hide_index=True, use_container_width=True)

    st.markdown('<div class="dash-panel"><div class="panel-title">Alert Notifications</div>', unsafe_allow_html=True)
    st.toggle("Push Notifications", value=False)
    st.toggle("Email Alerts", value=True)
    st.toggle("SMS Alerts", value=False)
    st.toggle("Critical Alerts Only", value=True)
    st.markdown("</div>", unsafe_allow_html=True)

with st.expander("🔔 Subscribe for alerts"):
    districts = sorted({loc["district"] for loc in UP_DISTRICT_LOCATIONS})
    with st.form("subscribe"):
        c1, c2 = st.columns(2)
        with c1:
            district = st.selectbox("District", districts)
            min_level = st.selectbox("Minimum level", ["Yellow", "Orange", "Red"])
        with c2:
            email = st.text_input("Email")
            phone = st.text_input("Phone (+91)")
        if st.form_submit_button("Subscribe", type="primary"):
            if email.strip() or phone.strip():
                sub_store.add(district=district, email=email.strip() or None, phone=phone.strip() or None, min_level=min_level)
                st.success("Subscribed!")
            else:
                st.error("Provide email or phone.")

st.caption("Data sources: IMD · CWC · ISRO · State Disaster Management")
