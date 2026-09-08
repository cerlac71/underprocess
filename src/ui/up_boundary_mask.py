"""Uttar Pradesh state boundary mask for map overlays.

Training grid is a rectangular bbox (23.5–30.5°N, 77–84.5°E) which includes
parts of Uttarakhand, Haryana, MP, Rajasthan, and Bihar. This mask restricts
visualisation to the UP state outline only.
"""

from __future__ import annotations

import numpy as np
from matplotlib.path import Path
from scipy.ndimage import gaussian_filter

# Simplified UP state outline (lat, lon) — clockwise, ~40 vertices
# Traces approximate administrative boundary; excludes neighbouring states.
UP_OUTLINE_LATLON: list[tuple[float, float]] = [
    (28.97, 77.10), (29.38, 77.05), (29.72, 77.35), (29.95, 77.85),
    (30.05, 78.45), (29.88, 79.10), (29.55, 79.65), (29.25, 80.15),
    (28.95, 80.55), (28.55, 80.95), (28.15, 81.35), (27.75, 81.65),
    (27.35, 81.95), (26.95, 82.25), (26.55, 82.55), (26.15, 82.85),
    (25.75, 83.15), (25.35, 83.45), (24.95, 83.75), (24.55, 84.05),
    (24.15, 84.30), (23.95, 84.15), (23.85, 83.75), (23.82, 83.25),
    (23.85, 82.65), (23.90, 82.05), (23.95, 81.45), (24.05, 80.85),
    (24.25, 80.25), (24.55, 79.65), (24.95, 79.05), (25.45, 78.55),
    (26.05, 78.15), (26.65, 77.85), (27.25, 77.55), (27.85, 77.35),
    (28.45, 77.20), (28.97, 77.10),
]

_path = Path([(lon, lat) for lat, lon in UP_OUTLINE_LATLON])


def up_state_mask(lat_arr: np.ndarray, lon_arr: np.ndarray) -> np.ndarray:
    """Boolean mask (lat × lon) — True only for grid cells inside Uttar Pradesh."""
    lat_grid, lon_grid = np.meshgrid(lat_arr, lon_arr, indexing="ij")
    points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
    inside = _path.contains_points(points).reshape(lat_grid.shape)
    return inside


def feathered_up_mask(
    lat_arr: np.ndarray,
    lon_arr: np.ndarray,
    *,
    sigma: float = 1.2,
    upscale: int = 6,
) -> np.ndarray:
    """
    Soft 0–1 mask feathered at the state border (for smooth overlay edges).
    Upscaled to match overlay resolution.
    """
    from scipy.ndimage import zoom

    hard = up_state_mask(lat_arr, lon_arr).astype(np.float32)
    soft = gaussian_filter(hard, sigma=sigma)
    soft = np.clip(soft, 0.0, 1.0)
    if upscale > 1:
        soft = zoom(soft, upscale, order=1)
    return soft


def mask_field_to_up(
    field: np.ndarray,
    lat_arr: np.ndarray,
    lon_arr: np.ndarray,
) -> np.ndarray:
    """Zero out predictions outside Uttar Pradesh."""
    masked = field.copy()
    masked[~up_state_mask(lat_arr, lon_arr)] = 0.0
    return masked


def masked_bounds(
    field: np.ndarray,
    lat_arr: np.ndarray,
    lon_arr: np.ndarray,
    *,
    min_value: float = 0.5,
) -> list[list[float]]:
    """Tight lat/lon bounds of visible UP data (not full rectangular grid)."""
    state = up_state_mask(lat_arr, lon_arr)
    active = state & np.isfinite(field) & (field > min_value)
    if not active.any():
        active = state
    rows = np.any(active, axis=1)
    cols = np.any(active, axis=0)
    if not rows.any() or not cols.any():
        return [[float(lat_arr.min()), float(lon_arr.min())],
                [float(lat_arr.max()), float(lon_arr.max())]]
    r_idx = np.where(rows)[0]
    c_idx = np.where(cols)[0]
    return [
        [float(lat_arr[r_idx[0]]), float(lon_arr[c_idx[0]])],
        [float(lat_arr[r_idx[-1]]), float(lon_arr[c_idx[-1]])],
    ]
