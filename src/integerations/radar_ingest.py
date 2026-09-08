#!/usr/bin/env python3
"""
Doppler Weather Radar Data Ingestion Module for SIH26071.

This module provides functions to load, decode, and convert radar reflectivity
data into rainfall intensity estimates, regrid onto a common grid, and
integrate with the existing pipeline. It supports NetCDF and GRIB2 formats
commonly used for radar data (e.g., NEXRAD Level III, IMD Doppler products).

Key features:
- Converts dBZ to rainfall rate using the Marshall-Palmer Z-R relationship.
- Masks non‑precipitating echoes (noise floor) and missing data.
- Regrids to any user‑specified common grid for fusion with other datasets.
- Provides a drop‑in replacement for `generate_synthetic_rainfall_series`
  via `get_rainfall_series`.
- Includes a function to generate a synthetic NetCDF file for testing.

Radar quirks (not fully handled in this version but important to know):
- Range folding: echoes from beyond the unambiguous range appear at wrong ranges.
- Ground clutter: non‑meteorological echoes from hills, buildings, etc.
- Beam blockage: partial or total loss of signal behind obstacles.
- Attenuation: signal loss in heavy rain, especially at shorter wavelengths.
- Bright band: enhanced reflectivity at the melting layer.
"""

import os
import re
import glob
import logging
import datetime as dt
from typing import Optional, Tuple, List, Union, Dict, Any

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata

# Attempt to import GRIB readers
try:
    import cfgrib
    HAS_CFGRIB = True
except ImportError:
    HAS_CFGRIB = False

try:
    import pygrib
    HAS_PYGRIB = True
except ImportError:
    HAS_PYGRIB = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Constants and defaults
# ----------------------------------------------------------------------
# Default common grid over India (adjust as needed)
DEFAULT_LAT = np.arange(6.0, 38.0, 0.1)   # 6N to 38N
DEFAULT_LON = np.arange(66.0, 100.0, 0.1) # 66E to 100E

# Default Z-R relationship parameters (Marshall-Palmer)
DEFAULT_A = 200.0
DEFAULT_B = 1.6

# Default noise floor for reflectivity (dBZ)
DEFAULT_NOISE_FLOOR_DBZ = -30.0

# Variable name patterns for reflectivity
REFLECTIVITY_PATTERNS = ['reflectivity', 'dbz', 'ref', 'radar', 'dz', 'corr_ref',
                         'refl', 'zh', 'zdr']

# Timestamp patterns in filenames (common radar file naming)
TIMESTAMP_PATTERNS = [
    r'(?P<ts>\d{8}_\d{4})',                    # YYYYMMDD_HHMM
    r'(?P<ts>\d{4}-\d{2}-\d{2}[_T]\d{2}-\d{2})', # YYYY-MM-DD_HH-MM or YYYY-MM-DDTHH-MM
    r'(?P<ts>\d{12})',                         # YYYYMMDDHHMM
    r'(?P<ts>\d{8})'                           # YYYYMMDD
]

# Global cache to hold the latest consolidated DataFrame
_RADAR_DATA_CACHE: Optional[pd.DataFrame] = None
DEFAULT_CACHE_FILE = 'radar_rainfall_cache.parquet'

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def _parse_timestamp_from_filename(filename: str) -> Optional[dt.datetime]:
    """
    Extract observation timestamp from a radar filename.

    Parameters
    ----------
    filename : str
        File name (or full path) containing a timestamp.

    Returns
    -------
    dt.datetime or None
        Parsed datetime (timezone naive) or None if no pattern matches.
    """
    base = os.path.basename(filename)
    for pattern in TIMESTAMP_PATTERNS:
        match = re.search(pattern, base)
        if match:
            ts_str = match.group('ts')
            for fmt in ['%Y%m%d_%H%M', '%Y-%m-%d_%H-%M', '%Y-%m-%dT%H-%M',
                        '%Y%m%d%H%M', '%Y%m%d']:
                try:
                    return dt.datetime.strptime(ts_str, fmt)
                except ValueError:
                    continue
    return None


def _find_reflectivity_variable(ds: xr.Dataset) -> str:
    """
    Identify the reflectivity variable in a radar Dataset.

    Searches variable names against common patterns. If exactly one data
    variable exists, returns it. IMD Doppler files use ``Z`` (dBZ).

    Parameters
    ----------
    ds : xr.Dataset
        Loaded radar dataset.

    Returns
    -------
    str
        Name of the reflectivity variable.
    """
    if "Z" in ds.data_vars:
        return "Z"
    var_names = list(ds.data_vars)
    for pattern in REFLECTIVITY_PATTERNS:
        for name in var_names:
            if pattern in name.lower():
                return name
    if len(var_names) == 1:
        return var_names[0]
    raise ValueError("No reflectivity variable found in radar dataset.")


def _extract_lat_lon(ds: xr.Dataset, var_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract 2D latitude and longitude arrays from a radar Dataset.

    Handles 1D coordinates (meshgridded) or 2D variables.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing lat/lon.
    var_name : str
        Name of the data variable (may have coordinate attributes).

    Returns
    -------
    lat2d, lon2d : np.ndarray
        2D arrays of the same shape as the data.
    """
    var = ds[var_name]
    lat = None
    lon = None

    # Look for coordinate variables
    for coord_name in ['lat', 'latitude', 'Lat', 'Latitude', 'LAT']:
        if coord_name in ds.coords or coord_name in ds.variables:
            lat = ds[coord_name].values
            break
    for coord_name in ['lon', 'longitude', 'Lon', 'Longitude', 'LON']:
        if coord_name in ds.coords or coord_name in ds.variables:
            lon = ds[coord_name].values
            break

    # Check attributes
    if lat is None and 'latitude' in var.attrs:
        lat = var.attrs['latitude']
    if lon is None and 'longitude' in var.attrs:
        lon = var.attrs['longitude']

    if lat is None or lon is None:
        raise ValueError("Latitude/longitude not found in radar dataset.")

    # Convert to 2D if necessary
    if lat.ndim == 1 and lon.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon, lat)
    elif lat.ndim == 2 and lon.ndim == 2:
        lat2d, lon2d = lat, lon
    else:
        raise ValueError("Lat/lon arrays must be either 1D or 2D.")

    # Ensure shape matches data
    data_shape = var.shape
    if lat2d.shape != data_shape or lon2d.shape != data_shape:
        # Try to squeeze or broadcast if possible (e.g., extra time dimension)
        if var.ndim == 3 and var.shape[0] == 1:
            lat2d = np.broadcast_to(lat2d, data_shape[1:])
            lon2d = np.broadcast_to(lon2d, data_shape[1:])
        else:
            raise ValueError("Lat/lon shape does not match data shape.")

    return lat2d, lon2d


# ----------------------------------------------------------------------
# Core functions
# ----------------------------------------------------------------------
def load_radar_file(filepath: str) -> xr.Dataset:
    """
    Load a radar file (NetCDF or GRIB2) into an xarray Dataset.

    Supports standard lat/lon grids and IMD Doppler sweep files (variable ``Z``).

    Parameters
    ----------
    filepath : str
        Path to the radar file.

    Returns
    -------
    xr.Dataset
        Dataset containing reflectivity and coordinates.

    Raises
    ------
    ValueError
        If file format is unsupported or cannot be read.
    """
    from integerations.imd_radar_ingest import is_imd_radar_file, load_imd_sweep

    if is_imd_radar_file(filepath):
        return load_imd_sweep(filepath)

    ext = os.path.splitext(filepath)[1].lower()
    if ext in ['.nc', '.nc4', '.netcdf']:
        ds = xr.open_dataset(filepath)
    elif ext in ['.grib2', '.grb2', '.grb']:
        if HAS_CFGRIB:
            try:
                ds = cfgrib.open_dataset(filepath)
            except Exception as e:
                logger.warning(f"cfgrib failed: {e}; trying pygrib")
                if HAS_PYGRIB:
                    ds = _read_grib_with_pygrib(filepath)
                else:
                    raise ValueError("GRIB2 file but no working reader (install cfgrib or pygrib).")
        elif HAS_PYGRIB:
            ds = _read_grib_with_pygrib(filepath)
        else:
            raise ValueError("GRIB2 file detected but neither cfgrib nor pygrib is installed.")
    else:
        raise ValueError(f"Unsupported radar file extension: {ext}")

    # Ensure reflectivity variable exists
    _find_reflectivity_variable(ds)  # raises if not found
    return ds


def _read_grib_with_pygrib(filepath: str) -> xr.Dataset:
    """
    Read a GRIB file using pygrib and convert to xarray Dataset.

    This is a fallback if cfgrib is not available.

    Parameters
    ----------
    filepath : str
        Path to GRIB file.

    Returns
    -------
    xr.Dataset
        Dataset with reflectivity as a 2D variable and lat/lon coordinates.
    """
    grbs = pygrib.open(filepath)
    # Assume first message contains reflectivity (usually the case)
    grb = grbs.message(1)
    data, lats, lons = grb.data()
    # Create DataArray
    da = xr.DataArray(data, dims=['y', 'x'], coords={'lat': (['y', 'x'], lats),
                                                    'lon': (['y', 'x'], lons)},
                      name='reflectivity')
    ds = da.to_dataset()
    ds.attrs['time'] = grb.validDate
    grbs.close()
    return ds


def reflectivity_to_rainfall(dbz_array: np.ndarray,
                             a: float = DEFAULT_A,
                             b: float = DEFAULT_B,
                             noise_floor_dbz: float = DEFAULT_NOISE_FLOOR_DBZ) -> np.ndarray:
    """
    Convert radar reflectivity (dBZ) to rainfall rate (mm/hr) using the
    Marshall‑Palmer Z‑R relationship.

    The relationship is Z = a * R^b, where Z is in mm^6 m^-3 and R is in
    mm/hr. Solving for R gives R = (Z/a)^(1/b). Z in linear units is
    10^(dBZ/10).

    Values below the noise floor (or NaN) are masked to NaN.

    Parameters
    ----------
    dbz_array : np.ndarray
        Reflectivity in dBZ.
    a : float
        Multiplicative coefficient of the Z-R relationship (default 200).
    b : float
        Exponent of the Z-R relationship (default 1.6).
    noise_floor_dbz : float
        Reflectivity threshold below which echoes are considered noise
        and masked (default -30 dBZ).

    Returns
    -------
    np.ndarray
        Rainfall rate in mm/hr, same shape as input, with NaN for masked
        values.
    """
    dbz = np.asarray(dbz_array, dtype=np.float32)
    # Mask invalid values
    mask = np.isnan(dbz) | (dbz < noise_floor_dbz)
    # Convert to linear Z
    z_linear = np.power(10.0, dbz / 10.0)
    # Compute rainfall rate
    with np.errstate(divide='ignore', invalid='ignore'):
        rainfall = np.power(z_linear / a, 1.0 / b)
    # Apply mask
    rainfall = np.where(mask, np.nan, rainfall)
    return rainfall.astype(np.float32)


def regrid_to_common_grid(lat: np.ndarray,
                          lon: np.ndarray,
                          values: np.ndarray,
                          target_lat: np.ndarray,
                          target_lon: np.ndarray,
                          method: str = 'nearest') -> np.ndarray:
    """
    Resample a radar field onto a common lat/lon grid using nearest‑neighbor
    interpolation.

    Parameters
    ----------
    lat, lon : np.ndarray
        2D arrays of source coordinates.
    values : np.ndarray
        2D array of values to regrid (e.g., rainfall rate).
    target_lat, target_lon : np.ndarray
        1D arrays defining the target grid.
    method : str
        Interpolation method passed to scipy.interpolate.griddata
        ('nearest' or 'linear').

    Returns
    -------
    np.ndarray
        2D array of regridded values on the target grid.
    """
    # Flatten valid points
    valid_mask = ~np.isnan(values)
    points = np.column_stack([lon[valid_mask], lat[valid_mask]])
    vals = values[valid_mask]

    if len(vals) == 0:
        logger.warning("No valid data points to regrid; returning all NaN.")
        target_lon2d, target_lat2d = np.meshgrid(target_lon, target_lat)
        return np.full(target_lat2d.shape, np.nan, dtype=np.float32)

    # Target grid points
    target_lon2d, target_lat2d = np.meshgrid(target_lon, target_lat)
    target_points = np.column_stack([target_lon2d.ravel(), target_lat2d.ravel()])

    # Perform interpolation
    regridded_flat = griddata(points, vals, target_points, method=method)
    regridded = regridded_flat.reshape(target_lat2d.shape).astype(np.float32)
    return regridded


def batch_ingest(folder_path: str,
                 output_format: str = "parquet",
                 target_lat: np.ndarray = DEFAULT_LAT,
                 target_lon: np.ndarray = DEFAULT_LON,
                 a: float = DEFAULT_A,
                 b: float = DEFAULT_B,
                 noise_floor_dbz: float = DEFAULT_NOISE_FLOOR_DBZ) -> pd.DataFrame:
    """
    Process all radar files in a folder and return a consolidated DataFrame.

    For each file:
    1. Load radar data.
    2. Extract reflectivity.
    3. Convert to rainfall rate using Z-R.
    4. Regrid to a common grid.
    5. Append to a DataFrame with columns: timestamp, lat, lon, value,
       variable='rainfall', source='radar'.

    The resulting DataFrame is also stored in the global cache
    (``_RADAR_DATA_CACHE``) for use by ``get_rainfall_series``.

    Parameters
    ----------
    folder_path : str
        Path to folder containing radar files.
    output_format : str, optional
        If 'parquet', save the DataFrame to a Parquet file in the same folder.
    target_lat, target_lon : np.ndarray
        1D arrays defining the common grid.
    a, b : float
        Z-R relationship parameters.
    noise_floor_dbz : float
        Reflectivity noise floor.

    Returns
    -------
    pd.DataFrame
        Consolidated DataFrame with columns:
        timestamp, lat, lon, value, variable, source.
    """
    global _RADAR_DATA_CACHE

    all_files = []
    for ext in ['*.nc', '*.nc4', '*.netcdf', '*.grib2', '*.grb2', '*.grb']:
        all_files.extend(glob.glob(os.path.join(folder_path, ext)))
    all_files = sorted(list(set(all_files)))

    logger.info(f"Found {len(all_files)} radar files in {folder_path}")

    master_df = pd.DataFrame(columns=['timestamp', 'lat', 'lon', 'value', 'variable', 'source'])
    processed_count = 0
    skipped_files = []

    for filepath in all_files:
        try:
            # Parse timestamp
            timestamp = _parse_timestamp_from_filename(filepath)
            if timestamp is None:
                logger.warning(f"Could not parse timestamp from {filepath}, skipping.")
                skipped_files.append(filepath)
                continue

            # Load and process
            ds = load_radar_file(filepath)
            var_name = _find_reflectivity_variable(ds)
            reflectivity = ds[var_name].values
            lat2d, lon2d = _extract_lat_lon(ds, var_name)

            # If 3D, take first time step
            if reflectivity.ndim == 3:
                reflectivity = reflectivity[0]
                # Adjust lat/lon if needed (they should already be 2D)
                if lat2d.ndim == 3:
                    lat2d = lat2d[0]
                if lon2d.ndim == 3:
                    lon2d = lon2d[0]

            # Convert to rainfall
            rainfall = reflectivity_to_rainfall(reflectivity, a, b, noise_floor_dbz)

            # Regrid
            regridded = regrid_to_common_grid(lat2d, lon2d, rainfall, target_lat, target_lon)

            # Create DataFrame for this file
            target_lon2d, target_lat2d = np.meshgrid(target_lon, target_lat)
            df_file = pd.DataFrame({
                'timestamp': timestamp,
                'lat': target_lat2d.ravel(),
                'lon': target_lon2d.ravel(),
                'value': regridded.ravel(),
                'variable': 'rainfall',
                'source': 'radar'
            })

            # Drop rows with NaN rainfall
            df_file = df_file.dropna(subset=['value'])

            # Append
            master_df = pd.concat([master_df, df_file], ignore_index=True)
            processed_count += 1
            logger.debug(f"Processed {filepath}")

        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")
            skipped_files.append(filepath)
            continue

    logger.info(f"Successfully processed {processed_count} files.")
    if skipped_files:
        logger.warning(f"Skipped {len(skipped_files)} files due to errors.")

    # Update global cache
    _RADAR_DATA_CACHE = master_df

    # Optionally save to Parquet
    if output_format.lower() == 'parquet' and len(master_df) > 0:
        out_path = os.path.join(folder_path, 'radar_rainfall_consolidated.parquet')
        master_df.to_parquet(out_path, index=False)
        logger.info(f"Saved consolidated data to {out_path}")

    return master_df


def get_rainfall_series(lat: float,
                        lon: float,
                        radius_km: float = 10.0,
                        hours: int = 24) -> pd.DataFrame:
    """
    Retrieve a time series of rainfall for a given location from the latest
    radar data.

    This function is a drop‑in replacement for
    ``generate_synthetic_rainfall_series`` in ``src/data_utils.py``.
    It returns a DataFrame with columns ``timestamp`` and ``rainfall_mm``
    (note: rainfall rate in mm/hr; for hourly accumulation, multiply by 1 h).

    The function uses the global cache ``_RADAR_DATA_CACHE`` populated by
    ``batch_ingest``. If that cache is empty, it attempts to load a
    consolidated Parquet file from the current working directory
    (``radar_rainfall_cache.parquet``). If neither is available, it returns
    an empty DataFrame with the correct columns.

    Parameters
    ----------
    lat : float
        Latitude of the point of interest.
    lon : float
        Longitude of the point of interest.
    radius_km : float
        Radius (in km) around the point to average radar pixels.
    hours : int
        Number of past hours to include.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'timestamp' and 'rainfall_mm'.
    """
    global _RADAR_DATA_CACHE

    df = _RADAR_DATA_CACHE
    if df is None or df.empty:
        # Try to load from default cache file
        if os.path.exists(DEFAULT_CACHE_FILE):
            df = pd.read_parquet(DEFAULT_CACHE_FILE)
        else:
            logger.warning("No radar data cache available. Call batch_ingest first.")
            return pd.DataFrame(columns=['timestamp', 'rainfall_mm'])

    # Filter by time window
    now = dt.datetime.now()
    start_time = now - dt.timedelta(hours=hours)
    df_time = df[df['timestamp'] >= start_time].copy()

    if df_time.empty:
        return pd.DataFrame(columns=['timestamp', 'rainfall_mm'])

    # Approximate radius conversion: 1 degree latitude ~ 111 km, longitude
    # varies with latitude; use a simple approximation for speed.
    lat_radius = radius_km / 111.0
    lon_radius = radius_km / (111.0 * np.cos(np.radians(lat)))

    # Filter by location
    df_loc = df_time[
        (df_time['lat'] >= lat - lat_radius) &
        (df_time['lat'] <= lat + lat_radius) &
        (df_time['lon'] >= lon - lon_radius) &
        (df_time['lon'] <= lon + lon_radius)
    ]

    if df_loc.empty:
        return pd.DataFrame(columns=['timestamp', 'rainfall_mm'])

    # Group by timestamp and compute mean rainfall over the selected pixels
    series = df_loc.groupby('timestamp')['value'].mean().reset_index()
    series.rename(columns={'value': 'rainfall_mm'}, inplace=True)

    return series


# ----------------------------------------------------------------------
# Sample data generation
# ----------------------------------------------------------------------
def generate_sample_radar_file(filepath: str) -> None:
    """
    Write a small synthetic NetCDF radar file for testing.

    The file contains a 2D reflectivity field on a simple lat/lon grid over
    a portion of India, with a timestamp embedded in the filename.
    The reflectivity is a synthetic storm cell with a Gaussian shape.

    Parameters
    ----------
    filepath : str
        Output path (should end with .nc).
    """
    # Define grid (smaller for speed)
    lat = np.arange(15.0, 25.0, 0.1)   # 15N to 25N
    lon = np.arange(70.0, 80.0, 0.1)   # 70E to 80E
    lon2d, lat2d = np.meshgrid(lon, lat)

    # Create synthetic reflectivity: a storm centered at (20N, 75E)
    center_lat, center_lon = 20.0, 75.0
    sigma_lat, sigma_lon = 0.5, 0.5
    dbz = 45.0 * np.exp(-((lat2d - center_lat)**2 / (2*sigma_lat**2) +
                          (lon2d - center_lon)**2 / (2*sigma_lon**2)))
    # Add some noise
    dbz = dbz + np.random.normal(0, 2.0, dbz.shape)
    # Mask values below noise floor
    dbz[dbz < -30] = np.nan

    # Create dataset
    ds = xr.Dataset(
        {
            'reflectivity': (['lat', 'lon'], dbz.astype(np.float32))
        },
        coords={
            'lat': lat,
            'lon': lon
        }
    )
    ds['reflectivity'].attrs['units'] = 'dBZ'
    ds['reflectivity'].attrs['long_name'] = 'Radar Reflectivity'
    ds.attrs['description'] = 'Synthetic radar file for testing SIH26071'
    ds.attrs['created'] = str(dt.datetime.now())

    # Write to NetCDF
    ds.to_netcdf(filepath)
    logger.info(f"Sample radar file written to {filepath}")


# ----------------------------------------------------------------------
# Main demonstration
# ----------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    import tempfile

    # Demonstrate with a temporary folder containing synthetic radar files
    tmp_dir = tempfile.mkdtemp(prefix='radar_demo_')
    print(f"Creating sample radar files in {tmp_dir}...")

    # Generate three files with different timestamps
    for i, hour_offset in enumerate([0, 1, 2]):
        ts = dt.datetime.now() - dt.timedelta(hours=hour_offset)
        filename = f"radar_{ts.strftime('%Y%m%d_%H%M')}.nc"
        filepath = os.path.join(tmp_dir, filename)
        generate_sample_radar_file(filepath)

    # Run batch ingestion
    print("\nRunning batch ingestion...")
    df = batch_ingest(tmp_dir, output_format='parquet')

    # Print summary
    if not df.empty:
        print("\n=== Batch Ingestion Summary ===")
        print(f"Total files processed: {df['timestamp'].nunique()}")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Min rainfall rate (mm/hr): {df['value'].min():.2f}")
        print(f"Max rainfall rate (mm/hr): {df['value'].max():.2f}")
        print(f"Mean rainfall rate (mm/hr): {df['value'].mean():.2f}")
        print(f"Total grid points: {len(df)}")
    else:
        print("No data processed.")

    # Demonstrate get_rainfall_series
    print("\nTesting get_rainfall_series for a location...")
    series = get_rainfall_series(lat=20.0, lon=75.0, radius_km=20, hours=24)
    print(f"Retrieved {len(series)} time steps.")
    if not series.empty:
        print(series.head())

    # Clean up
    import shutil
    shutil.rmtree(tmp_dir)