#!/usr/bin/env python3
"""
Terrain Analysis Module for SIH26071.

This module fetches a Digital Elevation Model (DEM) for a given bounding box,
processes it to derive hydrological features, and outputs them as bands in an
xarray Dataset aligned to the harmonized rainfall grid.

Hydrological processing is performed with `richdem` (fast, C++ backend) and
`pysheds` (Python-friendly, good for watershed delineation). The module
includes pit filling (or breaching), flow direction (D8 and D-infinity),
flow accumulation, stream network extraction, watershed delineation, and
terrain‑derived features (slope, TWI, distance‑to‑stream).

Author: SIH26071 Team
"""

import os
import logging
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict, Union

# Optional imports
try:
    import richdem as rd
    RICHDEM_AVAILABLE = True
except ImportError:
    RICHDEM_AVAILABLE = False
    logging.warning("richdem not installed. Install with `pip install richdem`.")

try:
    from pysheds.grid import Grid
    PYSHEDS_AVAILABLE = True
except ImportError:
    PYSHEDS_AVAILABLE = False
    logging.warning("pysheds not installed. Install with `pip install pysheds`.")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import rioxarray
    RIOXARRAY_AVAILABLE = True
except ImportError:
    RIOXARRAY_AVAILABLE = False

from scipy import ndimage
from scipy.spatial import cKDTree

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# 1. DEM Acquisition
# ----------------------------------------------------------------------
def fetch_srtm_dem(bbox: Tuple[float, float, float, float],
                   output_file: str = 'dem.tif',
                   source: str = 'opentopography',
                   resolution: float = 30.0) -> None:
    """
    Fetch a DEM for the given bounding box.

    Parameters
    ----------
    bbox : tuple
        (lon_min, lat_min, lon_max, lat_max) in EPSG:4326.
    output_file : str
        Path where the GeoTIFF will be saved.
    source : str
        'opentopography' (default) or 'py3dep' (fallback).
    resolution : float
        Desired horizontal resolution in meters (30 for SRTM, 10 for NED).

    Notes
    -----
    OpenTopography API requires an API key (free registration). For hackathon
    use, py3dep may be simpler as it wraps multiple public sources.
    """
    if source == 'opentopography':
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library is required for OpenTopography.")
        # OpenTopography global DEM API endpoint
        url = "https://portal.opentopography.org/API/globaldem"
        # You need to provide your API key. Set environment variable OPENTOPO_API_KEY.
        api_key = os.environ.get('OPENTOPO_API_KEY', '')
        if not api_key:
            raise ValueError("OpenTopography API key not found. Set OPENTOPO_API_KEY environment variable.")
        params = {
            'demtype': 'SRTMGL1',  # 30m SRTM
            'south': bbox[1],
            'north': bbox[3],
            'west': bbox[0],
            'east': bbox[2],
            'outputFormat': 'GTiff',
            'API_Key': api_key
        }
        response = requests.get(url, params=params, stream=True)
        if response.status_code == 200:
            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"DEM saved to {output_file}")
        else:
            raise RuntimeError(f"OpenTopography request failed: {response.status_code} {response.text}")
    elif source == 'py3dep':
        try:
            import py3dep
            dem = py3dep.get_map(bbox, resolution=resolution)
            dem.rio.to_raster(output_file)
            logger.info(f"DEM saved to {output_file}")
        except ImportError:
            raise ImportError("py3dep not installed. Install with `pip install py3dep`.")


def fetch_bhuvan_dem(bbox: Tuple[float, float, float, float],
                     output_file: str = 'dem_bhuvan.tif') -> None:
    """
    Download CartoDEM from Bhuvan (ISRO). Requires a Bhuvan account and
    manual interaction; this function is a placeholder demonstrating the
    endpoint. In practice, use the Bhuvan web portal or their API if available.
    """
    logger.warning("Bhuvan DEM download requires manual authentication. "
                   "Please download CartoDEM from https://bhuvan.nrsc.gov.in/ "
                   "and provide the file path to the processing functions.")
    # Actual implementation would require session handling and form submission.
    raise NotImplementedError("Bhuvan download not automated. Use manual download.")


def reproject_dem_to_target(dem_file: str,
                            target_grid: xr.Dataset,
                            method: str = 'bilinear') -> np.ndarray:
    """
    Reproject/resample the DEM to match the harmonized grid.

    Parameters
    ----------
    dem_file : str
        Path to the DEM GeoTIFF.
    target_grid : xr.Dataset
        Harmonized grid with 'lat' and 'lon' coordinates.
    method : str
        Resampling method. 'bilinear' is appropriate for elevation because
        elevation is a continuous field; conservative regridding is only for
        flux‑conservative variables (like rainfall).

    Returns
    -------
    np.ndarray
        2D array of elevation on the target grid.
    """
    if not RIOXARRAY_AVAILABLE:
        # Fallback using xarray and scipy
        dem_ds = xr.open_dataset(dem_file, engine='rasterio')
        dem = dem_ds['band_data'][0]
        # Assuming dem has spatial coords; use xarray interp
        dem_reproj = dem.interp(lat=target_grid['lat'], lon=target_grid['lon'],
                                method=method)
        return dem_reproj.values
    else:
        dem = rioxarray.open_rasterio(dem_file)
        dem = dem.rio.reproject_match(target_grid)
        return dem.values[0]


# ----------------------------------------------------------------------
# 2. Pit Filling / Depression Handling
# ----------------------------------------------------------------------
def fill_depressions(dem: np.ndarray, method: str = 'fill') -> np.ndarray:
    """
    Remove spurious depressions from the DEM.

    Parameters
    ----------
    dem : np.ndarray
        2D elevation array (must be float32).
    method : str
        'fill' (raise pits to spill level) or 'breach' (carve outlet channels).
        For urban flood modeling, 'breach' is often preferable because
        real depressions (ponds, tanks) are retained as storage areas
        rather than being filled.

    Returns
    -------
    np.ndarray
        Depression‑processed DEM.
    """
    if not RICHDEM_AVAILABLE:
        raise ImportError("richdem is required for depression handling.")
    # Convert to richdem array (float32)
    rdem = rd.rdarray(dem, no_data=-9999.0)
    if method == 'fill':
        rd.FillDepressions(rdem, epsilon=False, in_place=True)
    elif method == 'breach':
        rd.BreachDepressions(rdem, in_place=True)
    else:
        raise ValueError("method must be 'fill' or 'breach'")
    return np.array(rdem)


# ----------------------------------------------------------------------
# 3. Flow Direction
# ----------------------------------------------------------------------
def compute_flow_direction_d8(dem_filled: np.ndarray) -> np.ndarray:
    """
    Compute D8 flow direction (each cell flows to steepest downslope neighbor).

    Returns array of codes (1=E,2=NE,3=N,...,128=SE) or 0 for sinks.
    """
    if not RICHDEM_AVAILABLE:
        raise ImportError("richdem is required for D8 flow direction.")
    rdem = rd.rdarray(dem_filled, no_data=-9999.0)
    flowdir = rd.FlowDirectionD8(rdem)
    return np.array(flowdir)


def compute_flow_direction_dinf(dem_filled: np.ndarray) -> np.ndarray:
    """
    Compute D‑infinity flow direction (multiple flow directions).
    This returns an array of angles (radians) representing the flow direction.
    """
    if not RICHDEM_AVAILABLE:
        raise ImportError("richdem is required for D‑infinity flow direction.")
    rdem = rd.rdarray(dem_filled, no_data=-9999.0)
    flowdir = rd.FlowDirectionDInf(rdem)
    return np.array(flowdir)


# ----------------------------------------------------------------------
# 4. Flow Accumulation
# ----------------------------------------------------------------------
def compute_flow_accumulation(flow_direction: np.ndarray) -> np.ndarray:
    """
    Compute flow accumulation (number of upstream cells) from D8 flow direction.
    """
    if not RICHDEM_AVAILABLE:
        raise ImportError("richdem is required for flow accumulation.")
    # richdem expects flowdir as a rdarray
    fdir = rd.rdarray(flow_direction, no_data=0)
    accum = rd.FlowAccumFromFlowdir(fdir)
    return np.array(accum)


# ----------------------------------------------------------------------
# 5. Stream Network Extraction
# ----------------------------------------------------------------------
def extract_streams(flow_accum: np.ndarray,
                    threshold: Optional[int] = None) -> np.ndarray:
    """
    Extract stream network by thresholding flow accumulation.

    Parameters
    ----------
    flow_accum : np.ndarray
        Flow accumulation grid.
    threshold : int, optional
        Minimum accumulation to classify as stream. If None, a heuristic
        threshold of 100 cells is used (adjust based on DEM resolution and
        drainage density). Sanity check against OSM waterways.

    Returns
    -------
    np.ndarray
        Binary mask (1 = stream, 0 = not stream).
    """
    if threshold is None:
        # Common default: 100 cells (with 30m resolution ~ 9 ha).
        threshold = 100
    return (flow_accum >= threshold).astype(np.uint8)


# ----------------------------------------------------------------------
# 6. Watershed / Sub-basin Delineation
# ----------------------------------------------------------------------
def delineate_watershed_pysheds(dem_filled: np.ndarray,
                                pour_point: Tuple[int, int],
                                flow_dir: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Delineate the watershed for a given pour point using pysheds.

    Parameters
    ----------
    dem_filled : np.ndarray
        Filled DEM.
    pour_point : tuple
        (row, col) of the outlet cell.
    flow_dir : np.ndarray, optional
        D8 flow direction (if not provided, computed internally).

    Returns
    -------
    np.ndarray
        Boolean mask of the watershed.
    """
    if not PYSHEDS_AVAILABLE:
        raise ImportError("pysheds is required for watershed delineation.")
    if flow_dir is None:
        flow_dir = compute_flow_direction_d8(dem_filled)

    # Create pysheds Grid object
    grid = Grid()
    grid.add_gridded_data(dem_filled, data_name='dem')
    grid.add_gridded_data(flow_dir, data_name='dir')

    # Compute flow accumulation if not already done (needed for catchment)
    accum = grid.accumulation(data='dir', dirmap=(1,2,4,8,16,32,64,128))
    grid.add_gridded_data(accum, data_name='acc')

    # Delineate catchment
    catchment = grid.catchment(x=pour_point[1], y=pour_point[0], data='dir',
                               dirmap=(1,2,4,8,16,32,64,128),
                               recursionlimit=15000)
    return catchment.astype(bool)


def delineate_all_subbasins(dem_filled: np.ndarray,
                            flow_dir: Optional[np.ndarray] = None,
                            stream_threshold: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Automatically delineate all sub‑basins by splitting at stream junctions.

    This is a simplified approach: identify major streams, then use each
    stream segment as a pour point for its upstream catchment.

    Returns
    -------
    subbasin_id : np.ndarray
        Array with unique integer ID for each subbasin (0 = none).
    stream_mask : np.ndarray
        Binary stream mask used for splitting.
    """
    if flow_dir is None:
        flow_dir = compute_flow_direction_d8(dem_filled)
    accum = compute_flow_accumulation(flow_dir)
    streams = extract_streams(accum, threshold=stream_threshold)

    # Label connected stream segments
    structure = np.ones((3,3), dtype=int)
    labeled_streams, n_segments = ndimage.label(streams, structure=structure)

    # For each stream segment, find its outlet (the cell with maximum accumulation
    # within that segment) and delineate its catchment.
    subbasin_id = np.zeros_like(dem_filled, dtype=np.int32)
    for seg_id in range(1, n_segments+1):
        seg_mask = (labeled_streams == seg_id)
        # Outlet = cell with max accumulation within segment
        seg_accum = np.where(seg_mask, accum, -1)
        outlet_idx = np.unravel_index(np.argmax(seg_accum), accum.shape)
        try:
            catchment = delineate_watershed_pysheds(dem_filled, outlet_idx, flow_dir)
            subbasin_id[catchment] = seg_id
        except:
            logger.warning(f"Failed to delineate subbasin for segment {seg_id}")
    return subbasin_id, streams


# ----------------------------------------------------------------------
# 7. Terrain‑Derived Features
# ----------------------------------------------------------------------
def compute_slope(dem: np.ndarray, resolution: float = 30.0) -> np.ndarray:
    """
    Compute slope in degrees using the Horn method.
    """
    dzdx = ndimage.sobel(dem, axis=1) / (8 * resolution)
    dzdy = ndimage.sobel(dem, axis=0) / (8 * resolution)
    slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    return np.degrees(slope_rad)


def compute_twi(flow_accum: np.ndarray, slope: np.ndarray) -> np.ndarray:
    """
    Compute Topographic Wetness Index (TWI).

    TWI = ln( a / tan(beta) ), where a is specific catchment area
    (flow accumulation * cell size) and beta is slope in radians.
    """
    # Avoid division by zero
    slope_rad = np.radians(slope)
    slope_rad = np.where(slope_rad < 1e-6, 1e-6, slope_rad)
    # Specific catchment area = flow accumulation * pixel area
    # We assume pixel area = resolution^2, but in log space the constant cancels.
    twi = np.log(flow_accum + 1) - np.log(np.tan(slope_rad) + 1e-6)
    return twi


def compute_distance_to_stream(stream_mask: np.ndarray,
                               resolution: float = 30.0) -> np.ndarray:
    """
    Compute Euclidean distance (in meters) to nearest stream cell.
    """
    if not stream_mask.any():
        return np.full(stream_mask.shape, np.nan)
    # Coordinates of stream cells
    rows, cols = np.where(stream_mask)
    stream_coords = np.column_stack([rows, cols])
    # All cells
    all_rows, all_cols = np.indices(stream_mask.shape)
    all_coords = np.column_stack([all_rows.ravel(), all_cols.ravel()])
    # Use cKDTree for fast nearest neighbor
    tree = cKDTree(stream_coords)
    distances, _ = tree.query(all_coords, k=1)
    return (distances * resolution).reshape(stream_mask.shape)


# ----------------------------------------------------------------------
# 8. Output Dataset Creation
# ----------------------------------------------------------------------
def build_terrain_dataset(dem: np.ndarray,
                          flow_accum: np.ndarray,
                          slope: np.ndarray,
                          twi: np.ndarray,
                          stream_mask: np.ndarray,
                          distance_to_stream: np.ndarray,
                          target_grid: xr.Dataset) -> xr.Dataset:
    """
    Combine all terrain features into one xarray Dataset aligned to the
    harmonized grid.

    Parameters
    ----------
    dem, flow_accum, slope, twi, stream_mask, distance_to_stream : np.ndarray
        Arrays computed on the target grid.
    target_grid : xr.Dataset
        Grid definition with 'lat' and 'lon' coordinates.

    Returns
    -------
    xr.Dataset
        Dataset with variables: elevation, flow_accumulation, slope, twi,
        stream_mask, distance_to_stream.
    """
    ds = xr.Dataset(
        {
            'elevation': (('lat', 'lon'), dem.astype(np.float32)),
            'flow_accumulation': (('lat', 'lon'), flow_accum.astype(np.float32)),
            'slope': (('lat', 'lon'), slope.astype(np.float32)),
            'twi': (('lat', 'lon'), twi.astype(np.float32)),
            'stream_mask': (('lat', 'lon'), stream_mask.astype(np.uint8)),
            'distance_to_stream': (('lat', 'lon'), distance_to_stream.astype(np.float32))
        },
        coords={'lat': target_grid['lat'], 'lon': target_grid['lon']}
    )
    ds.attrs['description'] = 'Terrain‑derived hydrological features for inundation modeling'
    return ds


# ----------------------------------------------------------------------
# 9. Visualization
# ----------------------------------------------------------------------
def visualize_terrain(ds: xr.Dataset, watersheds: Optional[np.ndarray] = None) -> None:
    """
    Plot DEM, flow accumulation (log‑scaled), and watersheds (if provided)
    as a 3‑panel figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # DEM
    im0 = axes[0].imshow(ds['elevation'], origin='lower', cmap='terrain')
    axes[0].set_title('Elevation (m)')
    fig.colorbar(im0, ax=axes[0], label='m')

    # Flow accumulation (log scale)
    flow_accum = ds['flow_accumulation'].values
    # Avoid log(0)
    log_accum = np.log10(flow_accum + 1)
    im1 = axes[1].imshow(log_accum, origin='lower', cmap='viridis')
    axes[1].set_title('Flow Accumulation (log10)')
    fig.colorbar(im1, ax=axes[1], label='log10(accum)')

    # Watersheds or TWI as third panel
    if watersheds is not None:
        im2 = axes[2].imshow(watersheds, origin='lower', cmap='tab20')
        axes[2].set_title('Sub‑basins')
        fig.colorbar(im2, ax=axes[2], label='Basin ID')
    else:
        twi = ds['twi'].values
        im2 = axes[2].imshow(twi, origin='lower', cmap='Blues')
        axes[2].set_title('Topographic Wetness Index')
        fig.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# Example pipeline (if run as script)
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # This example assumes you have a target_grid (harmonized) defined.
    # For demonstration, create a dummy grid.
    lat = np.linspace(10, 11, 100)  # roughly 30m resolution
    lon = np.linspace(75, 76, 100)
    target_grid = xr.Dataset(coords={'lat': lat, 'lon': lon})

    # Simulate a DEM (e.g., a simple valley)
    lon2d, lat2d = np.meshgrid(lon, lat)
    dem = 100 + 50 * np.sin(lon2d * 5) + 30 * np.cos(lat2d * 8)
    dem = dem.astype(np.float32)

    # Process (requires richdem installed)
    if RICHDEM_AVAILABLE:
        dem_filled = fill_depressions(dem, method='fill')
        flowdir = compute_flow_direction_d8(dem_filled)
        accum = compute_flow_accumulation(flowdir)
        streams = extract_streams(accum, threshold=50)
        slope = compute_slope(dem_filled, resolution=30)
        twi = compute_twi(accum, slope)
        dist_stream = compute_distance_to_stream(streams, resolution=30)
        ds = build_terrain_dataset(dem_filled, accum, slope, twi, streams,
                                   dist_stream, target_grid)
        visualize_terrain(ds)
    else:
        logger.error("Install richdem to run the demo pipeline.")