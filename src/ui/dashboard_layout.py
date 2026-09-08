"""Shared layout, KPI helpers, and dark-dashboard components."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from config.regions import get_active_region
from ui.location_guide import (
    DISTRICT_COUNT,
    LEVEL_COLORS,
    UP_DISTRICT_LOCATIONS,
    UP_LOCATIONS,
    forecast_reliability,
    location_forecast,
    location_label,
)
from ui.styles import inject_theme

_region = get_active_region()

PLOTLY_DARK = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,18,30,0.6)",
    font=dict(color="#c8d6e5", family="Inter, DM Sans, sans-serif"),
    xaxis=dict(gridcolor="rgba(0,163,255,0.08)", zerolinecolor="rgba(0,163,255,0.15)"),
    yaxis=dict(gridcolor="rgba(0,163,255,0.08)", zerolinecolor="rgba(0,163,255,0.15)"),
    legend=dict(bgcolor="rgba(10,18,30,0.8)", bordercolor="rgba(0,163,255,0.2)"),
)


def setup_page(title: str, icon: str = "🌧️", section: str = "") -> None:
    st.set_page_config(
        page_title=f"{title} — {_region.name}",
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_theme(section=section)


def render_header(subtitle: str, slogan: str = "", section: str = "") -> None:
    now = datetime.now()
    date_str = now.strftime("%b %d, %Y")
    time_str = now.strftime("%I:%M %p").lstrip("0")
    slogan_html = f'<div class="dash-slogan">{slogan}</div>' if slogan else ""
    section_tag = f'<span class="dash-section">{section}</span>' if section else ""
    st.markdown(
        f"""
        <div class="dash-header">
            <div class="dash-header-left">
                <div class="dash-logo">🌧️</div>
                <div>
                    <div class="dash-title">HEAVY RAINFALL AND FLOOD PREDICTION</div>
                    <div class="dash-subtitle">{section_tag}{subtitle}</div>
                </div>
            </div>
            {slogan_html}
            <div class="dash-header-right">
                <div class="dash-datetime">{date_str}<br><span>{time_str}</span></div>
                <div class="dash-live"><span class="pulse-dot"></span> Live Data</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(icon: str, label: str, value: str, sub: str = "", accent: str = "#00A3FF") -> str:
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="kpi-card" style="border-top: 3px solid {accent};">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {sub_html}
    </div>
    """


def render_kpi_row(cards_html: str, cols: int = 5) -> None:
    components.html(
        f'<div class="kpi-row" style="grid-template-columns:repeat({cols},1fr);">{cards_html}</div>',
        height=118,
    )


def panel_open(title: str = "") -> None:
    title_html = f'<div class="panel-title">{title}</div>' if title else ""
    st.markdown(f'<div class="dash-panel">{title_html}', unsafe_allow_html=True)


def panel_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def status_badge(level: str) -> str:
    colors = {
        "Green": ("#34C759", "Normal"),
        "Yellow": ("#FFCC00", "Watch"),
        "Orange": ("#FF9500", "Alert"),
        "Red": ("#FF3B30", "High"),
        "High": ("#FF3B30", "High"),
        "Medium": ("#FF9500", "Medium"),
        "Low": ("#34C759", "Low"),
    }
    color, text = colors.get(level, ("#607d8b", level))
    return f'<span class="status-badge" style="background:{color}22;color:{color};border:1px solid {color}55;">{text}</span>'


def compute_region_summary(results: dict, metrics: dict | None = None) -> dict[str, Any]:
    rainfall = results["fused_rainfall"]
    risk = results["flood_risk"]
    depth = results["flood_depth"]

    district_rows = []
    for loc in UP_LOCATIONS:
        fc = location_forecast(results, loc)
        district_rows.append(fc)

    active_alerts = [r for r in district_rows if r["alert_level"] != "Green"]
    flood_risk = [r for r in district_rows if r["flood_risk"] >= 1]
    high_alerts = [r for r in district_rows if r["alert_level"] in ("Orange", "Red")]
    med_alerts = [r for r in district_rows if r["alert_level"] == "Yellow"]

    max_risk = int(risk.max()) if risk.size else 0
    risk_labels = {0: "Low", 1: "Medium", 2: "High", 3: "Very High"}
    peak_depth = float(depth.max()) if depth.size else 0.0
    total_rain = float(np.nanmean(rainfall))

    confidence = None
    if metrics and "ensemble" in metrics:
        confidence = metrics["ensemble"].get("r2")

    return {
        "total_rainfall_mm": round(total_rain, 1),
        "max_risk": max_risk,
        "risk_label": risk_labels.get(max_risk, "Low"),
        "affected_districts": len(flood_risk) or len(active_alerts),
        "active_alerts": len(active_alerts),
        "high_alerts": len(high_alerts),
        "med_alerts": len(med_alerts),
        "peak_depth_m": peak_depth,
        "confidence": confidence,
        "district_rows": district_rows,
        "cells_at_risk_pct": round((risk >= 1).sum() / risk.size * 100, 1) if risk.size else 0,
    }


def district_forecast_table(results: dict, metrics: dict | None = None, limit: int = 75) -> pd.DataFrame:
    rows = []
    for loc in UP_LOCATIONS[:limit]:
        fc = location_forecast(results, loc)
        rel = forecast_reliability(fc["confidence"], metrics)
        rows.append(
            {
                "Station": loc["name"],
                "District": loc["district"],
                "Rainfall (mm)": round(fc["rainfall_mm"], 1),
                "Depth (m)": round(fc["flood_depth_m"], 2),
                "Risk": fc["flood_risk"],
                "Alert": fc["alert_level"],
                "Reliability": f"{rel:.0%}",
            }
        )
    return pd.DataFrame(rows)


def alert_cards_html(district_rows: list[dict], limit: int = 5) -> str:
    active = [r for r in district_rows if r["alert_level"] != "Green" or r["flood_risk"] >= 1]
    if not active:
        active = sorted(district_rows, key=lambda r: r["rainfall_mm"], reverse=True)[:limit]
    else:
        active = sorted(active, key=lambda r: (r["alert_level"] != "Red", r["rainfall_mm"]), reverse=True)[:limit]

    items = []
    for fc in active:
        color = LEVEL_COLORS.get(fc["alert_level"], "#607d8b")
        name = fc["name"]
        level = fc["alert_level"]
        district = fc["district"]
        rain = fc["rainfall_mm"]
        depth = fc["flood_depth_m"]
        items.append(
            f'<div class="alert-card-item">'
            f'<div class="alert-severity" style="background:{color};"></div>'
            f'<div><div class="alert-card-title">{name} — {level} Alert</div>'
            f'<div class="alert-card-meta">{district} · {rain:.1f} mm · depth {depth:.2f} m</div></div></div>'
        )
    return "".join(items) if items else '<div class="empty-state">No active alerts</div>'


def render_alert_cards(district_rows: list[dict], title: str = "Recent Alerts", limit: int = 5) -> None:
    """Render alert cards via iframe — avoids Streamlit markdown escaping HTML."""
    cards = alert_cards_html(district_rows, limit=limit)
    n_cards = min(limit, len(district_rows))
    height = max(200, 58 * n_cards + 56)
    components.html(
        f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: Inter, 'DM Sans', sans-serif;
            background: rgba(11, 20, 38, 0.92);
            border: 1px solid rgba(0, 163, 255, 0.18);
            border-radius: 12px;
            padding: 16px 18px;
            color: #e8f1f8;
        }}
        .panel-title {{
            font-size: 0.78rem; font-weight: 700; color: #00A3FF;
            text-transform: uppercase; letter-spacing: 0.06em;
            margin-bottom: 12px; padding-bottom: 8px;
            border-bottom: 1px solid rgba(0, 163, 255, 0.18);
        }}
        .alert-card-item {{
            display: flex; gap: 10px; padding: 10px 0;
            border-bottom: 1px solid rgba(0, 163, 255, 0.08);
        }}
        .alert-card-item:last-child {{ border-bottom: none; }}
        .alert-severity {{ width: 4px; border-radius: 2px; flex-shrink: 0; }}
        .alert-card-title {{ font-size: 0.85rem; font-weight: 600; color: #fff; }}
        .alert-card-meta {{ font-size: 0.72rem; color: #7a8fa3; margin-top: 2px; }}
        .empty-state {{ color: #7a8fa3; font-size: 0.85rem; padding: 20px 0; text-align: center; }}
        </style></head><body>
        <div class="panel-title">{title}</div>
        {cards}
        </body></html>""",
        height=height,
        scrolling=True,
    )


def data_sources_panel() -> str:
    sources = [
        ("CHIRPS Satellite", "Online", "2 min ago"),
        ("IMD 0.25° Observations", "Online", "5 min ago"),
        ("ERA5 NWP", "Online", "15 min ago"),
        ("MERRA-2 Radar", "Online", "8 min ago"),
        ("State Monitoring", "—", "—"),
        ("IoT Gauges", "—", "—"),
    ]
    rows = []
    for name, status, updated in sources:
        dot = '<span class="source-online"></span>' if status == "Online" else '<span class="source-offline"></span>'
        rows.append(
            f'<div class="source-row"><span>{name}</span><span>{dot}{status}</span><span class="source-time">{updated}</span></div>'
        )
    return "".join(rows)


def sidebar_controls(results_path_exists: bool = True) -> dict[str, bool]:
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="sidebar-logo">🌧️</div>
                <div class="sidebar-tagline">Stronger Communities<br>Safer Tomorrows</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")
        show_rainfall = st.toggle("Rainfall (Live)", value=True)
        show_risk = st.toggle("Flood Risk Zones", value=True)
        show_stations = st.toggle("Weather Stations", value=True)
        st.markdown("---")
        if st.button("⟳ Refresh Forecast", type="primary", use_container_width=True):
            from ui.page_bootstrap import run_full_pipeline

            st.cache_data.clear()
            with st.spinner("Running pipeline…"):
                run_full_pipeline()
            st.rerun()
    return {
        "show_rainfall": show_rainfall,
        "show_risk": show_risk,
        "show_stations": show_stations,
    }


def filter_bar(region_name: str = "") -> dict[str, str]:
    region = region_name or _region.name
    c1, c2, c3, c4, c5 = st.columns([1.2, 1, 1, 1, 2])
    with c1:
        data_layer = st.selectbox("Data Layers", ["Rainfall (Live)", "Flood Risk", "River Level", "All Layers"], label_visibility="visible")
    with c2:
        region_sel = st.selectbox("Region", [region], label_visibility="visible")
    with c3:
        districts = ["All Districts"] + sorted({loc["district"] for loc in UP_DISTRICT_LOCATIONS})
        district_sel = st.selectbox("District", districts, label_visibility="visible")
    with c4:
        time_sel = st.selectbox("Time", ["Real-time", "Last 24h", "Last 7 Days"], label_visibility="visible")
    with c5:
        search = st.text_input("Search", placeholder="Search location (e.g. Lucknow)…", label_visibility="collapsed")
    return {
        "data_layer": data_layer,
        "region": region_sel,
        "district": district_sel,
        "time": time_sel,
        "search": search,
    }
