"""Dark-theme Streamlit styling matching the reference dashboard designs."""

from __future__ import annotations

import streamlit as st

from ui.location_guide import LEVEL_COLORS


def inject_theme(section: str = "") -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {{
            --bg-primary: #050a14;
            --bg-card: #0b1426;
            --bg-panel: rgba(11, 20, 38, 0.92);
            --border: rgba(0, 163, 255, 0.18);
            --accent: #00A3FF;
            --accent-dim: #007bff;
            --text: #e8f1f8;
            --text-muted: #7a8fa3;
            --danger: #FF3B30;
            --warning: #FF9500;
            --success: #34C759;
        }}

        html, body, [class*="css"] {{
            font-family: 'Inter', 'DM Sans', sans-serif;
            color: var(--text);
        }}

        .stApp {{
            background: linear-gradient(180deg, #040d1a 0%, #050a14 40%, #0a121e 100%);
        }}

        header[data-testid="stHeader"] {{
            background: transparent !important;
        }}

        .block-container {{
            padding: 0.5rem 1.5rem 2rem;
            max-width: 100%;
        }}

        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #040d1a 0%, #0a1528 100%) !important;
            border-right: 1px solid var(--border);
        }}
        [data-testid="stSidebar"] * {{ color: var(--text) !important; }}
        [data-testid="stSidebar"] .stMarkdown h3 {{ color: var(--accent) !important; }}

        [data-testid="stSidebarNav"] {{
            padding-top: 0.5rem;
        }}
        [data-testid="stSidebarNav"] a {{
            color: var(--text-muted) !important;
            border-radius: 8px;
            margin: 2px 8px;
            padding: 8px 12px !important;
        }}
        [data-testid="stSidebarNav"] a:hover {{
            background: rgba(0, 163, 255, 0.12) !important;
            color: var(--accent) !important;
        }}
        [data-testid="stSidebarNav"] a[aria-current="page"] {{
            background: rgba(0, 163, 255, 0.2) !important;
            color: var(--accent) !important;
            border-left: 3px solid var(--accent);
        }}

        .sidebar-brand {{
            text-align: center;
            padding: 12px 0 8px;
        }}
        .sidebar-logo {{ font-size: 2rem; }}
        .sidebar-tagline {{
            font-size: 0.65rem;
            color: var(--text-muted) !important;
            line-height: 1.4;
            margin-top: 8px;
            letter-spacing: 0.03em;
        }}

        /* Header */
        .dash-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: linear-gradient(90deg, rgba(0,163,255,0.08) 0%, transparent 60%);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 14px 24px;
            margin-bottom: 1rem;
        }}
        .dash-header-left {{
            display: flex;
            align-items: center;
            gap: 14px;
        }}
        .dash-logo {{ font-size: 2rem; }}
        .dash-title {{
            font-size: 1.15rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            color: #fff;
        }}
        .dash-subtitle {{
            font-size: 0.72rem;
            color: var(--accent);
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-top: 2px;
        }}
        .dash-section {{
            color: var(--accent);
            font-weight: 700;
        }}
        .dash-slogan {{
            font-style: italic;
            color: var(--text-muted);
            font-size: 0.9rem;
        }}
        .dash-header-right {{ text-align: right; }}
        .dash-datetime {{
            font-size: 0.8rem;
            color: var(--text-muted);
        }}
        .dash-datetime span {{ color: var(--text); font-weight: 600; }}
        .dash-live {{
            display: flex;
            align-items: center;
            justify-content: flex-end;
            gap: 6px;
            font-size: 0.75rem;
            color: var(--success);
            margin-top: 4px;
        }}
        .pulse-dot {{
            width: 8px; height: 8px;
            background: var(--success);
            border-radius: 50%;
            animation: pulse 2s infinite;
        }}
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; box-shadow: 0 0 0 0 rgba(52,199,89,0.6); }}
            50% {{ opacity: 0.7; box-shadow: 0 0 0 6px rgba(52,199,89,0); }}
        }}

        /* KPI cards */
        .kpi-row {{
            display: grid;
            gap: 12px;
            margin: 0.5rem 0 1rem;
        }}
        .kpi-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 14px 16px;
            min-height: 96px;
        }}
        .kpi-icon {{ font-size: 1.2rem; margin-bottom: 4px; }}
        .kpi-label {{
            font-size: 0.68rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .kpi-value {{
            font-size: 1.35rem;
            font-weight: 700;
            color: #fff;
            margin-top: 2px;
        }}
        .kpi-sub {{
            font-size: 0.72rem;
            color: var(--text-muted);
            margin-top: 2px;
        }}

        /* Panels */
        .dash-panel {{
            background: var(--bg-panel);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px 18px;
            margin-bottom: 12px;
        }}
        .panel-title {{
            font-size: 0.78rem;
            font-weight: 700;
            color: var(--accent);
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 1px solid var(--border);
        }}

        /* Alert cards */
        .alert-card-item {{
            display: flex;
            gap: 10px;
            padding: 10px 0;
            border-bottom: 1px solid rgba(0,163,255,0.08);
        }}
        .alert-severity {{
            width: 4px;
            border-radius: 2px;
            flex-shrink: 0;
        }}
        .alert-card-title {{
            font-size: 0.85rem;
            font-weight: 600;
            color: #fff;
        }}
        .alert-card-meta {{
            font-size: 0.72rem;
            color: var(--text-muted);
            margin-top: 2px;
        }}
        .empty-state {{
            color: var(--text-muted);
            font-size: 0.85rem;
            padding: 20px 0;
            text-align: center;
        }}

        /* Status badges */
        .status-badge {{
            display: inline-block;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 0.72rem;
            font-weight: 600;
        }}

        /* Data sources */
        .source-row {{
            display: grid;
            grid-template-columns: 1.5fr 1fr 1fr;
            font-size: 0.78rem;
            padding: 6px 0;
            border-bottom: 1px solid rgba(0,163,255,0.06);
            color: var(--text-muted);
        }}
        .source-online {{
            display: inline-block;
            width: 6px; height: 6px;
            background: var(--success);
            border-radius: 50%;
            margin-right: 4px;
        }}
        .source-offline {{
            display: inline-block;
            width: 6px; height: 6px;
            background: #555;
            border-radius: 50%;
            margin-right: 4px;
        }}
        .source-time {{ text-align: right; }}

        /* Map frame */
        .map-frame {{
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--border);
            box-shadow: 0 4px 24px rgba(0,0,0,0.4);
        }}

        /* Streamlit widgets dark */
        .stSelectbox > div > div,
        .stTextInput > div > div > input,
        .stMultiSelect > div > div {{
            background: var(--bg-card) !important;
            border-color: var(--border) !important;
            color: var(--text) !important;
        }}
        .stTabs [data-baseweb="tab-list"] {{
            background: transparent;
            gap: 4px;
        }}
        .stTabs [data-baseweb="tab"] {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px 8px 0 0;
            color: var(--text-muted);
        }}
        .stTabs [aria-selected="true"] {{
            background: rgba(0,163,255,0.15) !important;
            color: var(--accent) !important;
            border-color: var(--accent) !important;
        }}

        div[data-testid="stMetric"] {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 12px;
        }}
        div[data-testid="stMetric"] label {{
            color: var(--text-muted) !important;
        }}
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
            color: #fff !important;
        }}

        .stDataFrame {{
            border: 1px solid var(--border);
            border-radius: 10px;
        }}

        /* Timeline bar */
        .timeline-bar {{
            display: flex;
            align-items: center;
            gap: 12px;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 10px 16px;
            margin-top: 8px;
        }}
        .timeline-live {{
            color: var(--success);
            font-size: 0.75rem;
            font-weight: 600;
        }}

        /* Mission box */
        .mission-box {{
            background: linear-gradient(135deg, rgba(255,59,48,0.15) 0%, rgba(11,20,38,0.9) 100%);
            border: 1px solid rgba(255,59,48,0.3);
            border-radius: 12px;
            padding: 16px 20px;
            color: #fff;
        }}
        .mission-box h4 {{
            color: var(--danger);
            margin: 0 0 8px 0;
            font-size: 0.85rem;
        }}

        /* Hide streamlit footer */
        footer {{ visibility: hidden; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="dash-header">
            <div class="dash-header-left">
                <div class="dash-logo">🌧️</div>
                <div>
                    <div class="dash-title">{title}</div>
                    <div class="dash-subtitle">{subtitle}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str) -> str:
    return (
        f'<div class="kpi-card" style="border-top:3px solid #00A3FF;">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div></div>'
    )


def alert_banner(level: str, text: str) -> None:
    colour = LEVEL_COLORS.get(level, "#546e7a")
    st.markdown(
        f'<div class="dash-panel" style="border-left:4px solid {colour};color:{colour};">{text}</div>',
        unsafe_allow_html=True,
    )


def advisory_card(heading: str, detail: str) -> str:
    return (
        f'<div class="alert-card-item"><div class="alert-severity" style="background:#00A3FF;"></div>'
        f'<div><div class="alert-card-title">{heading}</div>'
        f'<div class="alert-card-meta">{detail}</div></div></div>'
    )
