"""Settings — dashboard configuration."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config.bootstrap import bootstrap_project

bootstrap_project()

import os
import streamlit as st

from config.regions import get_active_region, list_regions
from ui.dashboard_layout import render_header, setup_page
from ui.location_guide import DISTRICT_COUNT, SYSTEM_MODULES

_region = get_active_region()
setup_page("Settings", "⚙️", section="settings")
render_header("SETTINGS | SYSTEM CONFIGURATION", section="SETTINGS | ")

st.markdown('<div class="dash-panel"><div class="panel-title">Region Configuration</div>', unsafe_allow_html=True)
regions = list_regions()
for r in regions:
    active = "✓" if r["id"] == _region.id else ""
    st.markdown(f"**{r['name']}** {active} — {r['grid_cells']} grid cells")
st.caption(f"Active: `SIH_REGION={os.environ.get('SIH_REGION', 'uttar_pradesh')}`")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="dash-panel"><div class="panel-title">Data Sources</div>', unsafe_allow_html=True)
for mod in SYSTEM_MODULES:
    st.markdown(f"**{mod['name']}** — {mod['source']}")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="dash-panel"><div class="panel-title">Coverage</div>', unsafe_allow_html=True)
st.markdown(f"- **{DISTRICT_COUNT}** district headquarters monitored")
st.markdown(f"- Region: **{_region.name}**")
st.markdown(f"- Grid: **{_region.grid_step}°** (~{_region.approx_grid_cells} cells)")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="dash-panel"><div class="panel-title">Alert Channels</div>', unsafe_allow_html=True)
st.text_input("SMTP Server", placeholder="—", disabled=True)
st.text_input("SMS Gateway", placeholder="—", disabled=True)
st.text_input("Webhook URL", placeholder="Configure in .env")
st.caption("Set credentials in `.env` file for email/SMS notifications.")
st.markdown("</div>", unsafe_allow_html=True)
