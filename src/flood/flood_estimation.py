#!/usr/bin/env python3
"""
Flood Extent and Depth Estimation Module for SIH26071.

This module converts runoff depth (with uncertainty) into flood extent
(binary mask) and flood depth (continuous grid) using two rule‑based
approaches:

1. **Fill‑and‑spill / planar water surface** – accumulates runoff volume
   per sub‑basin and fills terrain from the outlet upward, conserving
   volume. Best for urban pluvial flooding because it handles local
   depressions and ponding, not just river overflow.

2. **HAND (Height Above Nearest Drainage)** – estimates flood depth as
   the difference between a stream stage and the HAND value. Fast but
   assumes flooding originates from the stream network, which is less
   appropriate for purely pluvial urban flooding far from channels.

Stage 2 (ML segmentation) is provided as an architecture outline.

All outputs follow the uncertainty schema: separate maps for p10, p50,
p90 runoff quantiles.

Author: SIH26071 Team
"""

import logging
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple, List, Union
import heapq

# Optional imports
try:
    import richdem as rd
    RICHDEM_AVAILABLE = True
except ImportError:
    RICHDEM_AVAILABLE = False

try:
    import pyflwdir
    PYFLWDIR_AVAILABLE = True
except ImportError:
    PYFLWDIR_AVAILABLE = False

try:
    import folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# 1. HAND computation
# ----------------------------------------------------------------------
def compute_hand(dem: np.ndarray,
                 flow_dir: np.ndarray,
                 stream_mask: np.ndarray) -> np.ndarray:
    """
    Compute Height Above Nearest Drainage (HAND).

    HAND is the vertical distance from each cell to the nearest stream
    cell along the flow path. It is a flood susceptibility index:
    cells with small HAND are more likely to be inundated.

    Parameters
    ----------
    dem : np.ndarray
        Filled DEM (2D).
    flow_dir : np.ndarray
        D8 flow direction codes (richdem convention).
    stream_mask : np.ndarray
        Binary mask of stream cells.

    Returns
    -------
    np.ndarray
        HAND values (same shape as dem).
    """
    if PYFLWDIR_AVAILABLE:
        # Use pyflwdir for efficient HAND
        flw = pyflwdir.from_dem(dem, fdir=flow_dir)
        hand = flw.hand(stream_mask)
        return hand
    else:
        # Fallback: recursive upstream accumulation of elevation difference
        # Not as efficient but works for small grids.
        logger.warning("pyflwdir not available; using simplified HAND fallback.")
        hand = np.full(dem.shape, np.inf, dtype=np.float32)
        # Initialize with stream cells (HAND=0)
        hand[stream_mask] = 0.0

        # Map from D8 codes to neighbor offsets (richdem D8 codes:
        # 1=E,2=NE,3=N,4=NW,5=W,6=SW,7=S,8=SE)
        # We'll use a simplified approach: iterate multiple times until convergence.
        # Not production-grade but illustrative.
        for _ in range(100):
            changed = False
            for i in range(dem.shape[0]):
                for j in range(dem.shape[1]):
                    if not stream_mask[i, j]:
                        # Check downstream neighbor and compute HAND
                        # This is crude; actual flow direction needed.
                        # We'll use a simple downhill neighbor.
                        min_neighbor = np.inf
                        for di, dj in [(-1,0),(1,0),(0,-1),(0,1),
                                       (-1,-1),(-1,1),(1,-1),(1,1)]:
                            ni, nj = i+di, j+dj
                            if 0 <= ni < dem.shape[0] and 0 <= nj < dem.shape[1]:
                                if hand[ni, nj] < min_neighbor:
                                    min_neighbor = hand[ni, nj] + (dem[i,j] - dem[ni,nj])
                        if min_neighbor < hand[i, j]:
                            hand[i, j] = min_neighbor
                            changed = True
            if not changed:
                break
        # Clip negative HAND to 0
        hand = np.maximum(hand, 0)
        return hand.astype(np.float32)


# ----------------------------------------------------------------------
# 2. Fill‑and‑spill flood depth (sub‑basin bathtub)
# ----------------------------------------------------------------------
def compute_flood_depth_fill_spill(
    dem: np.ndarray,
    subbasin_id: np.ndarray,
    runoff_volume: Dict[int, float],
    cell_area: float = 900.0  # m², adjust based on grid resolution
) -> np.ndarray:
    """
    Compute flood depth using a bathtub fill model per sub‑basin.

    For each sub‑basin, the total runoff volume (m³) is distributed from
    the lowest cells upward until the volume is exhausted. The water
    surface is assumed horizontal within the sub‑basin.

    Parameters
    ----------
    dem : np.ndarray
        Elevation (m).
    subbasin_id : np.ndarray
        Integer array with unique ID for each sub‑basin (0 = outside).
    runoff_volume : dict
        Maps sub‑basin ID to total runoff volume (m³) accumulated over
        the simulation period.
    cell_area : float
        Area of a single grid cell in m².

    Returns
    -------
    np.ndarray
        Flood depth (m) for each cell.
    """
    depth = np.zeros_like(dem, dtype=np.float32)
    for basin_id, vol in runoff_volume.items():
        if vol <= 0:
            continue
        mask = (subbasin_id == basin_id)
        if not mask.any():
            continue

        basin_elev = dem[mask]
        sorted_idx = np.argsort(basin_elev)
        sorted_elev = basin_elev[sorted_idx]

        # Cumulative volume as we fill upward. Raising the surface from
        # sorted_elev[i] to sorted_elev[i+1] inundates i+1 cells.
        n_inundated = np.arange(1, sorted_elev.size)
        cum_volume = np.cumsum(np.diff(sorted_elev) * n_inundated * cell_area)
        # Insert 0 at start
        cum_volume = np.concatenate([[0], cum_volume])

        # Find water surface level where cumulative volume equals vol
        # Interpolate if needed
        if vol >= cum_volume[-1]:
            water_level = sorted_elev[-1]  # entire basin flooded
        else:
            idx = np.searchsorted(cum_volume, vol) - 1
            if idx < 0:
                idx = 0
            # Linear interpolation between idx and idx+1
            frac = (vol - cum_volume[idx]) / (cum_volume[idx+1] - cum_volume[idx])
            water_level = sorted_elev[idx] + frac * (sorted_elev[idx+1] - sorted_elev[idx])

        # Assign depth
        depth[mask] = np.maximum(water_level - dem[mask], 0)

    return depth


# ----------------------------------------------------------------------
# 3. HAND‑based flood depth
# ----------------------------------------------------------------------
def compute_flood_depth_hand(
    hand: np.ndarray,
    stream_stage: Dict[int, float],
    stream_reach_id: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute flood depth from HAND and stream stage.

    If stream_reach_id is provided, each stream segment has its own stage
    (from accumulated runoff in that reach). Otherwise, a single stage is
    used for all streams (approximation).

    Parameters
    ----------
    hand : np.ndarray
        HAND grid.
    stream_stage : dict
        Maps reach ID (or a single key) to water surface elevation
        relative to stream bed (stage in meters).
    stream_reach_id : np.ndarray, optional
        Array with reach IDs for stream cells (same shape as hand).
        If None, all streams assumed to have the same stage.

    Returns
    -------
    np.ndarray
        Flood depth (m).
    """
    depth = np.zeros_like(hand, dtype=np.float32)
    if stream_reach_id is None:
        # Single stage
        stage = stream_stage.get(0, 0.0)
        depth = np.maximum(stage - hand, 0)
    else:
        # Per‑reach stage
        for rid, stage in stream_stage.items():
            mask = (stream_reach_id == rid)
            # For cells whose HAND is less than stage, depth = stage - HAND
            # Only cells downstream of that reach? Simplified: apply globally.
            # More accurate: need distance/downstream propagation, but we ignore.
            # This is a limitation; for hackathon, we use simple threshold.
            depth = np.maximum(depth, np.where(hand <= stage, stage - hand, 0))
    return depth


# ----------------------------------------------------------------------
# 4. Risk classification
# ----------------------------------------------------------------------
def classify_risk(depth: np.ndarray) -> np.ndarray:
    """
    Convert flood depth to risk categories.

    Thresholds based on common impact levels:
    - No flood: depth == 0
    - Minor (< 0.3 m): ankle‑deep, low risk
    - Moderate (0.3–1.0 m): vehicle‑disabling, medium risk
    - Severe (> 1.0 m): building‑damaging, high risk
    These roughly follow NDMA urban flood guidelines (adjust as needed).
    """
    risk = np.zeros_like(depth, dtype=np.int8)
    risk[(depth > 0) & (depth < 0.3)] = 1
    risk[(depth >= 0.3) & (depth < 1.0)] = 2
    risk[depth >= 1.0] = 3
    return risk


# ----------------------------------------------------------------------
# 5. Sanity check
# ----------------------------------------------------------------------
def sanity_check_flood(depth: np.ndarray, slope: np.ndarray) -> bool:
    """
    Flag obviously wrong flood predictions (e.g., deep water on steep slopes
    or hilltops). Returns True if no obvious errors.
    """
    # Flood depth should be near zero on steep slopes (slope > 20°)
    steep_mask = slope > 20.0
    if np.any(depth[steep_mask] > 0.5):
        logger.error("Flood depth > 0.5 m on steep slopes – possible error.")
        return False
    return True


# ----------------------------------------------------------------------
# 6. ML architecture (Stage 2, not trained)
# ----------------------------------------------------------------------
def build_unet_flood_model(input_channels=5, output_channels=1):
    """
    U‑Net architecture for flood segmentation/depth regression.
    Input: [runoff, slope, HAND, TWI, flow_accumulation]
    Output: depth map (continuous) or risk map (segmentation).
    """
    import torch
    import torch.nn as nn

    class UNet(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            # Encoder
            self.enc1 = self.conv_block(in_channels, 64)
            self.enc2 = self.conv_block(64, 128)
            self.enc3 = self.conv_block(128, 256)
            self.enc4 = self.conv_block(256, 512)
            # Decoder
            self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
            self.dec1 = self.conv_block(512, 256)
            self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
            self.dec2 = self.conv_block(256, 128)
            self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
            self.dec3 = self.conv_block(128, 64)
            self.final = nn.Conv2d(64, out_channels, kernel_size=1)

        def conv_block(self, in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(nn.MaxPool2d(2)(e1))
            e3 = self.enc3(nn.MaxPool2d(2)(e2))
            e4 = self.enc4(nn.MaxPool2d(2)(e3))
            d1 = self.dec1(torch.cat([self.up1(e4), e3], 1))
            d2 = self.dec2(torch.cat([self.up2(d1), e2], 1))
            d3 = self.dec3(torch.cat([self.up3(d2), e1], 1))
            return self.final(d3)

    return UNet(input_channels, output_channels)


# ----------------------------------------------------------------------
# 7. Visualization (folium and static)
# ----------------------------------------------------------------------
def plot_flood_depth_static(depth: np.ndarray, risk: np.ndarray,
                            title: str = 'Flood Depth'):
    """Static plot using matplotlib."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(depth, cmap='viridis', origin='lower')
    axes[0].set_title('Depth (m)')
    fig.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(risk, cmap='RdYlGn_r', origin='lower', vmin=0, vmax=3)
    axes[1].set_title('Risk (0=none,1=minor,2=moderate,3=severe)')
    fig.colorbar(im1, ax=axes[1])
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def create_folium_map(depth: np.ndarray, risk: np.ndarray,
                      lat: np.ndarray, lon: np.ndarray,
                      zoom_start: int = 13) -> 'folium.Map':
    """
    Create an interactive folium map with flood depth overlay.
    """
    if not FOLIUM_AVAILABLE:
        raise ImportError("folium not installed.")
    # Create colormap for risk
    from folium import plugins
    import branca.colormap as cm
    colormap = cm.LinearColormap(['green', 'yellow', 'orange', 'red'],
                                 vmin=0, vmax=3)
    # We'll add a raster overlay using ImageOverlay
    # Convert risk to RGBA image
    risk_colors = np.zeros((risk.shape[0], risk.shape[1], 4), dtype=np.uint8)
    risk_colors[risk == 0] = [0, 0, 0, 0]        # transparent
    risk_colors[risk == 1] = [0, 255, 0, 128]    # green, semi-transparent
    risk_colors[risk == 2] = [255, 165, 0, 180]  # orange
    risk_colors[risk == 3] = [255, 0, 0, 220]    # red

    # Coordinates for overlay
    lon_min, lon_max = lon[0], lon[-1]
    lat_min, lat_max = lat[0], lat[-1]
    # Create map centered on area
    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)
    # Add image overlay
    folium.raster_layers.ImageOverlay(
        image=risk_colors,
        bounds=[[lat_min, lon_min], [lat_max, lon_max]],
        opacity=0.7,
        name='Flood Risk'
    ).add_to(m)
    # Add colormap legend
    colormap.caption = 'Flood Risk (0=none,1=minor,2=moderate,3=severe)'
    m.add_child(colormap)
    return m


# ----------------------------------------------------------------------
# Main demonstration
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # Create dummy terrain and runoff
    lat = np.linspace(10, 11, 20)
    lon = np.linspace(75, 76, 20)
    lon2d, lat2d = np.meshgrid(lon, lat)
    dem = 100 + 10 * np.sin(lon2d * 5) + 5 * np.cos(lat2d * 8)
    # Simple subbasin: label all as one basin
    subbasin_id = np.ones_like(dem, dtype=int)
    subbasin_id[0, :] = 0  # outside basin
    # Runoff volume for basin 1
    runoff_volume = {1: 5000.0}  # m³

    # Compute flood depth
    depth = compute_flood_depth_fill_spill(dem, subbasin_id, runoff_volume)
    risk = classify_risk(depth)

    # Plot
    plot_flood_depth_static(depth, risk, 'Demo Flood Depth')