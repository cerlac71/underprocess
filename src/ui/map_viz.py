"""Readable rainfall / flood maps for the flood warning dashboard."""

from __future__ import annotations

import base64
import io

import folium
import matplotlib.colors as mcolors
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom

from ui.location_guide import LEVEL_COLORS, location_forecast
from ui.up_boundary_mask import feathered_up_mask, mask_field_to_up, masked_bounds
from ui.up_locations_data import UP_LOCATIONS

DARK_BASE_TILES = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
DARK_BASE_ATTR = "Tiles © Esri"
OSM_TILES = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
OSM_ATTR = "© OpenStreetMap contributors"

UPSCALE = 6

RAIN_ZONES = [
    (200, "#7B2D8E", "Extreme"),
    (115, "#FF3B30", "Very High"),
    (64, "#FF9500", "High"),
    (20, "#FFCC00", "Moderate"),
    (5, "#34C759", "Light"),
    (0, "#00A3FF", "Very Light"),
]

RISK_RGB = {
    1: mcolors.to_rgb("#34C759"),
    2: mcolors.to_rgb("#FF9500"),
    3: mcolors.to_rgb("#FF3B30"),
}

RAIN_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "rain_ref", [c for _, c, _ in reversed(RAIN_ZONES)]
)


def _rgba_to_data_uri(rgba: np.ndarray) -> str:
    rgb = (np.clip(rgba[:, :, :3], 0, 1) * 255).astype(np.uint8)
    alpha = (np.clip(rgba[:, :, 3], 0, 1) * 255).astype(np.uint8)
    image = Image.fromarray(rgb)
    image.putalpha(Image.fromarray(alpha))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _smooth_field(field: np.ndarray, sigma: float = 2.8, upscale: int = UPSCALE) -> np.ndarray:
    clean = np.nan_to_num(field, nan=0.0).astype(np.float32)
    if clean.size == 0:
        return clean
    smoothed = gaussian_filter(clean, sigma=sigma)
    if upscale > 1:
        smoothed = zoom(smoothed, upscale, order=3)
    return smoothed


def _apply_state_mask_to_alpha(
    alpha: np.ndarray,
    lat_arr: np.ndarray,
    lon_arr: np.ndarray,
) -> np.ndarray:
    """Clip overlay to Uttar Pradesh outline — removes neighbours + square corners."""
    state_soft = feathered_up_mask(lat_arr, lon_arr, sigma=1.4, upscale=UPSCALE)
    if state_soft.shape != alpha.shape:
        state_soft = zoom(
            feathered_up_mask(lat_arr, lon_arr, sigma=1.4, upscale=1),
            alpha.shape[0] / len(lat_arr),
            order=1,
        )
    return alpha * state_soft


def _prediction_hotspots(field: np.ndarray, lat_arr: np.ndarray, lon_arr: np.ndarray) -> np.ndarray:
    """Hotspots only inside UP — no spill into Uttarakhand / Haryana / etc."""
    field = mask_field_to_up(field, lat_arr, lon_arr)
    smooth = _smooth_field(field)
    if smooth.size == 0 or not np.any(smooth > 0):
        return smooth

    local_bg = gaussian_filter(smooth, sigma=5.0)
    hotspot = np.maximum(smooth - local_bg, 0.0)
    finite = smooth[smooth > 0]
    p75 = float(np.percentile(finite, 75))
    absolute = np.where(smooth >= p75, smooth - p75 * 0.6, 0.0)
    combined = np.maximum(hotspot, absolute)
    peak = float(np.percentile(combined[combined > 0], 95)) if (combined > 0).any() else 1.0
    return combined / max(peak, 1e-6)


def _alpha_from_hotspots(hotspot_field: np.ndarray) -> np.ndarray:
    norm = np.clip(hotspot_field, 0.0, 1.0)
    alpha = np.where(norm > 0.06, norm**1.15 * 0.85, 0.0)
    alpha *= gaussian_filter((norm > 0.06).astype(np.float32), sigma=2.8)
    return alpha


def rainfall_prediction_overlay(
    rainfall: np.ndarray,
    lat_arr: np.ndarray,
    lon_arr: np.ndarray,
) -> str:
    """Rainfall coloured by prediction, clipped to Uttar Pradesh only."""
    rainfall = mask_field_to_up(rainfall, lat_arr, lon_arr)
    hotspots = _prediction_hotspots(rainfall, lat_arr, lon_arr)
    alpha = _alpha_from_hotspots(hotspots)
    alpha = _apply_state_mask_to_alpha(alpha, lat_arr, lon_arr)

    if not np.any(alpha > 0.01):
        return _rgba_to_data_uri(np.zeros((*hotspots.shape, 4), dtype=np.float32))

    smooth_rain = _smooth_field(rainfall)
    finite = smooth_rain[smooth_rain > 0]
    vmax = max(float(np.percentile(finite, 98)), 10.0) if finite.size else 10.0
    rgba = RAIN_CMAP(np.clip(smooth_rain / vmax, 0, 1))
    rgba[..., 3] = alpha
    return _rgba_to_data_uri(rgba)


def risk_prediction_overlay(
    risk: np.ndarray,
    lat_arr: np.ndarray,
    lon_arr: np.ndarray,
) -> str:
    """Flood risk only inside UP where risk ≥ 1 is predicted."""
    risk = mask_field_to_up(risk.astype(np.float32), lat_arr, lon_arr)
    smooth = _smooth_field(risk, sigma=2.0)
    rgba = np.zeros((*smooth.shape, 4), dtype=np.float32)

    for level in (3, 2, 1):
        rgb = RISK_RGB[level]
        mask = smooth >= (level - 0.45)
        strength = np.clip((smooth - (level - 0.45)) / 0.9, 0, 1)
        layer_alpha = strength * {1: 0.48, 2: 0.62, 3: 0.78}[level]
        for c in range(3):
            rgba[..., c] = np.where(mask, np.maximum(rgba[..., c], rgb[c] * strength), rgba[..., c])
        rgba[..., 3] = np.maximum(rgba[..., 3], layer_alpha)

    rgba[..., 3] = _apply_state_mask_to_alpha(rgba[..., 3], lat_arr, lon_arr)
    return _rgba_to_data_uri(rgba)


def combined_prediction_overlay(
    rainfall: np.ndarray,
    risk: np.ndarray,
    lat_arr: np.ndarray,
    lon_arr: np.ndarray,
) -> str:
    rainfall = mask_field_to_up(rainfall, lat_arr, lon_arr)
    risk = mask_field_to_up(risk.astype(np.float32), lat_arr, lon_arr)

    hotspots = _prediction_hotspots(rainfall, lat_arr, lon_arr)
    alpha = _alpha_from_hotspots(hotspots)
    alpha = _apply_state_mask_to_alpha(alpha, lat_arr, lon_arr) * 0.65

    smooth_rain = _smooth_field(rainfall)
    finite = smooth_rain[smooth_rain > 0]
    vmax = max(float(np.percentile(finite, 98)), 10.0) if finite.size else 10.0
    rgba = RAIN_CMAP(np.clip(smooth_rain / vmax, 0, 1))
    rgba[..., 3] = alpha

    smooth_risk = _smooth_field(risk, sigma=1.8)
    for level in (3, 2, 1):
        rgb = RISK_RGB[level]
        mask = smooth_risk >= (level - 0.4)
        strength = np.clip((smooth_risk - (level - 0.4)) / 0.9, 0, 1)
        blend = {1: 0.4, 2: 0.55, 3: 0.7}[level]
        for c in range(3):
            rgba[..., c] = np.where(
                mask,
                rgba[..., c] * (1 - blend * strength) + rgb[c] * blend * strength,
                rgba[..., c],
            )
        risk_a = _apply_state_mask_to_alpha(strength * blend * 0.8, lat_arr, lon_arr)
        rgba[..., 3] = np.maximum(rgba[..., 3], risk_a)

    return _rgba_to_data_uri(rgba)


def _legend_html(mode: str = "rainfall") -> str:
    if mode == "risk":
        items = "".join(
            f'<div style="display:flex;align-items:center;gap:6px;margin:3px 0;">'
            f'<span style="width:16px;height:16px;background:{c};border-radius:50%;"></span>'
            f'<span style="color:#c8d6e5;font-size:11px;">{l}</span></div>'
            for c, l in [("#34C759", "Low Risk"), ("#FF9500", "Medium"), ("#FF3B30", "High")]
        )
        title = "Flood Risk — Uttar Pradesh"
    else:
        items = "".join(
            f'<div style="display:flex;align-items:center;gap:6px;margin:3px 0;">'
            f'<span style="width:16px;height:16px;background:{c};border-radius:3px;"></span>'
            f'<span style="color:#c8d6e5;font-size:11px;">{l}</span></div>'
            for _t, c, l in RAIN_ZONES
        )
        title = "Rainfall — Uttar Pradesh"

    return f"""
    <div style="position:fixed;bottom:32px;left:32px;z-index:9999;
        background:rgba(5,10,20,0.94);padding:14px 18px;border-radius:12px;
        border:1px solid rgba(0,163,255,0.3);font-family:Inter,sans-serif;min-width:155px;">
      <div style="color:#00A3FF;font-weight:700;font-size:11px;margin-bottom:8px;
          text-transform:uppercase;">{title}</div>
      {items}
      <div style="color:#7a8fa3;font-size:9px;margin-top:8px;">Clipped to UP state boundary</div>
    </div>
    """


def _alert_icon_html(color: str) -> str:
    return (
        f'<div style="filter:drop-shadow(0 2px 4px rgba(0,0,0,0.8));">'
        f'<div style="width:0;height:0;margin:0 auto;'
        f'border-left:11px solid transparent;border-right:11px solid transparent;'
        f'border-bottom:20px solid {color};position:relative;">'
        f'<span style="position:absolute;top:5px;left:-4px;color:white;'
        f'font-weight:bold;font-size:11px;">!</span></div></div>'
    )


def _risk_halo_color(fc: dict) -> str | None:
    if fc["alert_level"] == "Red" or fc["flood_risk"] >= 3:
        return "#FF3B30"
    if fc["alert_level"] == "Orange" or fc["flood_risk"] >= 2:
        return "#FF9500"
    if fc["alert_level"] == "Yellow" or fc["flood_risk"] >= 1:
        return "#FFCC00"
    if fc["rainfall_mm"] >= 20:
        return "#34C759"
    return None


def build_warning_map(
    results: dict,
    *,
    selected: dict | None = None,
    show_rainfall: bool = True,
    show_risk: bool = False,
    show_stations: bool = True,
    show_alerts: bool = True,
    map_mode: str = "rainfall",
    zoom: int = 7,
) -> folium.Map:
    """Map overlay clipped to Uttar Pradesh — no predictions shown in other states."""
    lat, lon = np.asarray(results["lat"]), np.asarray(results["lon"])
    center_lat = selected["lat"] if selected else float(np.mean(lat))
    center_lon = selected["lon"] if selected else float(np.mean(lon))
    map_zoom = 9 if selected else zoom

    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=map_zoom,
        tiles=DARK_BASE_TILES,
        attr=DARK_BASE_ATTR,
        control_scale=True,
    )
    folium.TileLayer(tiles=OSM_TILES, attr=OSM_ATTR, name="OpenStreetMap", overlay=False).add_to(fmap)

    rainfall = mask_field_to_up(results["fused_rainfall"], lat, lon)
    risk = mask_field_to_up(results["flood_risk"], lat, lon)

    mode = map_mode
    if show_risk and show_rainfall:
        mode = "combined"
    elif show_risk:
        mode = "risk"
    elif show_rainfall:
        mode = "rainfall"

    # Tight bounds inside UP — not the full rectangular training grid
    bounds = masked_bounds(rainfall, lat, lon)
    legend_mode = "rainfall"

    if mode == "rainfall" and show_rainfall:
        folium.raster_layers.ImageOverlay(
            rainfall_prediction_overlay(rainfall, lat, lon),
            bounds=bounds,
            opacity=1.0,
            name="Rainfall (UP only)",
        ).add_to(fmap)
    elif mode == "risk" and show_risk:
        folium.raster_layers.ImageOverlay(
            risk_prediction_overlay(risk, lat, lon),
            bounds=bounds,
            opacity=1.0,
            name="Flood risk (UP only)",
        ).add_to(fmap)
        legend_mode = "risk"
    elif mode == "combined":
        folium.raster_layers.ImageOverlay(
            combined_prediction_overlay(rainfall, risk, lat, lon),
            bounds=bounds,
            opacity=1.0,
            name="Rainfall + risk (UP only)",
        ).add_to(fmap)

    selected_key = (selected["district"], selected["name"]) if selected else None

    for city in UP_LOCATIONS:
        is_selected = selected_key == (city["district"], city["name"])
        is_major = city.get("tier", 0) == 1
        fc = location_forecast(results, city)
        halo = _risk_halo_color(fc)

        if halo and show_alerts:
            folium.CircleMarker(
                location=[city["lat"], city["lon"]],
                radius=20 if fc["flood_risk"] >= 2 else 14,
                color=halo, weight=2, fill=True, fill_color=halo, fill_opacity=0.15,
            ).add_to(fmap)

        if show_alerts and (fc["alert_level"] in ("Orange", "Red") or fc["flood_risk"] >= 2):
            alert_color = LEVEL_COLORS.get(fc["alert_level"], "#FF3B30")
            folium.Marker(
                location=[city["lat"], city["lon"]],
                icon=folium.DivIcon(
                    html=_alert_icon_html(alert_color),
                    icon_size=(22, 22), icon_anchor=(11, 20),
                ),
                popup=(
                    f"<b style='color:{alert_color}'>{fc['alert_level']} Alert</b><br>"
                    f"<b>{city['name']}</b> · {city['district']}<br>"
                    f"Rainfall: {fc['rainfall_mm']:.1f} mm<br>"
                    f"Depth: {fc['flood_depth_m']:.2f} m · Risk: {fc['flood_risk']}/3"
                ),
            ).add_to(fmap)
        elif show_stations:
            folium.CircleMarker(
                location=[city["lat"], city["lon"]],
                radius=10 if is_selected else (5 if is_major else 3),
                color="#ffffff",
                weight=2 if is_selected or is_major else 1,
                fill=True,
                fill_color="#00A3FF" if is_selected else ("#5a8fc7" if is_major else "#3a5a7a"),
                fill_opacity=0.9,
                popup=(
                    f"<b>{city['name']}</b><br>{city['district']} · {fc['rainfall_mm']:.1f} mm<br>"
                    f"Alert: {fc['alert_level']} · Risk: {fc['flood_risk']}/3"
                ),
            ).add_to(fmap)
            if is_selected or is_major:
                folium.Marker(
                    location=[city["lat"], city["lon"]],
                    icon=folium.DivIcon(
                        html=(
                            f"<div style='font-size:{'12' if is_selected else '10'}px;font-weight:700;"
                            f"color:{'#00A3FF' if is_selected else '#c8dff5'};"
                            f"text-shadow:0 0 6px #000;white-space:nowrap;'>{city['name']}</div>"
                        ),
                        icon_size=(120, 18),
                        icon_anchor=(0, -12 if is_selected else -8),
                    ),
                ).add_to(fmap)

    if show_rainfall or show_risk:
        fmap.get_root().html.add_child(folium.Element(_legend_html(legend_mode)))

    folium.LayerControl(collapsed=True).add_to(fmap)
    return fmap


# Backward-compatible aliases (tests / legacy callers)
def rainfall_zone_overlay_png(rainfall: np.ndarray, **kwargs) -> str:
    lat = kwargs.get("lat")
    lon = kwargs.get("lon")
    if lat is None or lon is None:
        return ""
    return rainfall_prediction_overlay(rainfall, np.asarray(lat), np.asarray(lon))


rainfall_overlay_png = rainfall_zone_overlay_png
risk_zone_overlay_png = risk_prediction_overlay
risk_overlay_png = risk_prediction_overlay
