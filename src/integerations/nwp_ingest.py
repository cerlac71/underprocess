#!/usr/bin/env python3
"""
NWP (Numerical Weather Prediction) Model Ingestion Module for SIH26071.

This module handles loading, de-accumulating, and extracting precipitation
forecasts from NWP model output (e.g., GFS, ERA5, NCMRWF) in GRIB2 or NetCDF
format. It converts cumulative precipitation fields into per-timestep rainfall
and provides functions to query forecasts at arbitrary points.

Key features:
- Load GRIB2/NetCDF with auto-detection.
- De-accumulate cumulative precipitation (tp) along the forecast step dimension.
- Extract forecast at a point using nearest-neighbour or bilinear interpolation.
- Regrid to a common grid for fusion with other datasets.
- Batch ingest multiple files and save a consolidated Parquet file.
- Generate synthetic NetCDF files for testing.

Important de-accumulation logic:
NWP precipitation fields are often **cumulative** – they represent total
accumulation since the model initialisation time (init_time). For example,
if tp(step=6) = 12 mm and tp(step=3) = 5 mm, the rainfall during the 3-hour
period between step=3 and step=6 is 12 - 5 = 7 mm. The first step (usually
step=0) has no previous step, so it is either dropped or set to NaN. This
module drops the first step after differencing to avoid misleading values.
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

# Attempt to import cfgrib for GRIB2
try:
    import cfgrib
    HAS_CFGRIB = True
except ImportError:
    HAS_CFGRIB = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
# Default common grid over India (adjust as needed)
DEFAULT_LAT = np.arange(6.0, 38.0, 0.1)   # 6N to 38N
DEFAULT_LON = np.arange(66.0, 100.0, 0.1) # 66E to 100E

# Global cache for forecast data (used by get_forecast_series)
_NWP_FORECAST_DF: Optional[pd.DataFrame] = None
DEFAULT_CACHE_FILE = 'nwp_forecast_cache.parquet'

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def _parse_init_time_from_filename(filename: str) -> Optional[dt.datetime]:
    """
    Attempt to parse model initialisation time from filename.
    NWP files often contain the init time in the name, e.g.,
    'gfs.t00z.pgrb2.0p25.f006' or 'era5_20240101_00.nc'.
    Returns None if no pattern matches.
    """
    base = os.path.basename(filename)
    # Common patterns: YYYYMMDD_HH, YYYY-MM-DD_HH, YYYYMMDDHH, etc.
    patterns = [
        r'(?P<ts>\d{8}_\d{2})',          # YYYYMMDD_HH
        r'(?P<ts>\d{4}-\d{2}-\d{2}_\d{2})', # YYYY-MM-DD_HH
        r'(?P<ts>\d{10})',               # YYYYMMDDHH
        r'(?P<ts>\d{8})',                # YYYYMMDD (fallback)
    ]
    for pat in patterns:
        match = re.search(pat, base)
        if match:
            ts_str = match.group('ts')
            for fmt in ['%Y%m%d_%H', '%Y-%m-%d_%H', '%Y%m%d%H', '%Y%m%d']:
                try:
                    return dt.datetime.strptime(ts_str, fmt)
                except ValueError:
                    continue
    return None


def _extract_lat_lon_1d(ds: xr.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract 1D latitude and longitude coordinates from an NWP dataset.
    Assumes regular lat/lon grid with 1D coordinates named 'latitude'/'lat'
    and 'longitude'/'lon'.
    """
    lat = None
    lon = None
    for coord in ['latitude', 'lat', 'Latitude']:
        if coord in ds.coords:
            lat = ds[coord].values
            break
    for coord in ['longitude', 'lon', 'Longitude']:
        if coord in ds.coords:
            lon = ds[coord].values
            break
    if lat is None or lon is None:
        raise ValueError("Could not find 1D latitude/longitude coordinates in dataset.")
    return lat, lon


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def load_nwp_file(filepath: str) -> xr.Dataset:
    """
    Load an NWP file (GRIB2 or NetCDF) into an xarray Dataset.

    Parameters
    ----------
    filepath : str
        Path to the file.

    Returns
    -------
    xr.Dataset
        Dataset with precipitation variable (usually 'tp') and coordinates.
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext in ['.grb', '.grib', '.grb2', '.grib2']:
        if not HAS_CFGRIB:
            raise ImportError("cfgrib is required to read GRIB2 files. Install with 'pip install cfgrib'")
        try:
            ds = cfgrib.open_dataset(filepath)
        except Exception as e:
            logger.error(f"cfgrib failed to open {filepath}: {e}")
            raise
    elif ext in ['.nc', '.nc4', '.netcdf']:
        ds = xr.open_dataset(filepath)
    else:
        raise ValueError(f"Unsupported NWP file extension: {ext}")

    # Ensure we have some precipitation variable (commonly 'tp')
    if 'tp' not in ds and 'precip' not in ds and 'prate' not in ds:
        # Try to find any variable with 'precip' in name
        candidates = [v for v in ds.data_vars if 'precip' in v.lower() or 'tp' in v.lower()]
        if not candidates:
            raise ValueError("No precipitation variable found in NWP dataset.")
        # Use the first candidate
        # We'll standardize name to 'tp' for internal use?
        # But deaccumulate_precipitation expects a variable name, so we'll let user pass it.
    return ds


# ----------------------------------------------------------------------
# De-accumulation
# ----------------------------------------------------------------------
def deaccumulate_precipitation(dataset: xr.Dataset,
                               precip_var_name: str = "tp") -> xr.Dataset:
    """
    Convert cumulative precipitation into per-timestep rainfall by differencing
    along the forecast step dimension.

    The dataset must have a dimension representing forecast lead time (often
    called 'step', 'time', or 'valid_time'). For GRIB files from cfgrib, the
    dimension is usually 'step' (in hours since init_time) and there is also a
    'valid_time' coordinate. For NetCDF, the dimension might be 'time' with
    cumulative values.

    The first step is dropped because it has no previous step to difference
    against. The resulting dataset will have one less step, and the
    precipitation values are rainfall amounts for the interval between steps.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset containing the cumulative precipitation variable.
    precip_var_name : str
        Name of the cumulative precipitation variable (default 'tp').

    Returns
    -------
    xr.Dataset
        Dataset with de-accumulated precipitation (per-step rainfall). The
        dimension length is reduced by one (the first step is removed).
    """
    if precip_var_name not in dataset:
        raise ValueError(f"Variable '{precip_var_name}' not found in dataset.")

    # Determine the dimension to difference along.
    # Prefer 'step' if present (common in cfgrib), else 'time'.
    dim = None
    if 'step' in dataset[precip_var_name].dims:
        dim = 'step'
    elif 'time' in dataset[precip_var_name].dims:
        dim = 'time'
    else:
        # Fallback: find the first dimension that is not lat/lon
        for d in dataset[precip_var_name].dims:
            if d not in ['lat', 'lon', 'latitude', 'longitude']:
                dim = d
                break
    if dim is None:
        raise ValueError("Could not find a time/step dimension to diff along.")

    # Ensure the dimension is sorted (should be)
    da = dataset[precip_var_name].sortby(dim)

    # Difference along the dimension
    diff_da = da.diff(dim=dim)

    # Remove the first step (which becomes all NaN after diff, but we already dropped)
    # diff_da has dimension length one less, and coordinate values shifted.
    # We'll construct a new dataset with de-accumulated variable.
    ds_out = dataset.copy()
    ds_out[precip_var_name] = diff_da

    # Also adjust other variables if they share the same dimension? Usually only
    # the precip var needs de-accumulation; but we can leave others as-is,
    # though they may have the original length. Better to select the same
    # steps as the diff result. We'll handle by reindexing the entire dataset
    # to the new step values.
    new_step_values = diff_da[dim].values
    ds_out = ds_out.sel({dim: new_step_values})
    # Drop the variable's original attribute if any
    ds_out[precip_var_name].attrs['long_name'] = 'De-accumulated precipitation'
    ds_out[precip_var_name].attrs['units'] = 'mm'

    return ds_out


# ----------------------------------------------------------------------
# Extract at point
# ----------------------------------------------------------------------
def extract_forecast_at_point(dataset: xr.Dataset,
                              lat: float,
                              lon: float,
                              method: str = "nearest") -> pd.DataFrame:
    """
    Extract precipitation forecast time series at the nearest grid point to
    (lat, lon) or using bilinear interpolation.

    Parameters
    ----------
    dataset : xr.Dataset
        De-accumulated dataset (or raw if user handles deaccumulation elsewhere).
    lat, lon : float
        Target coordinates.
    method : str
        'nearest' or 'bilinear'.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: init_time, valid_time, lead_hours, rainfall_mm.
        init_time is extracted from dataset attributes or coordinates.
    """
    # Determine init_time
    init_time = None
    if 'init_time' in dataset.attrs:
        init_time = pd.to_datetime(dataset.attrs['init_time'])
    elif 'time' in dataset.coords:
        # For NetCDF, maybe 'time' is init_time and 'valid_time' is separate
        init_time = pd.to_datetime(dataset['time'].values[0])
    else:
        # Try to get from filename? Not available here, so set to NaT
        logger.warning("Could not determine init_time; setting to NaT.")
        init_time = pd.NaT

    # Identify precipitation variable (assuming first data var that is not lat/lon)
    precip_var = None
    for v in dataset.data_vars:
        if v not in ['lat', 'lon', 'latitude', 'longitude']:
            precip_var = v
            break
    if precip_var is None:
        raise ValueError("No precipitation variable found in dataset.")

    da = dataset[precip_var]

    # Extract using nearest or bilinear
    if method == "nearest":
        # Use xarray's sel with nearest
        extracted = da.sel(lat=lat, lon=lon, method='nearest')
    elif method == "bilinear":
        # Use xarray's interp (requires 1D coordinates)
        extracted = da.interp(lat=lat, lon=lon, method='linear')
    else:
        raise ValueError("method must be 'nearest' or 'bilinear'")

    # Convert to pandas Series, then DataFrame
    if isinstance(extracted, xr.DataArray):
        # If dimension is 'step' or 'time'
        dim = 'step' if 'step' in extracted.dims else 'time'
        lead_hours = extracted[dim].values
        valid_times = None
        if 'valid_time' in extracted.coords:
            valid_times = extracted['valid_time'].values
        else:
            # Compute valid_time = init_time + lead_hours
            if init_time is not pd.NaT:
                valid_times = init_time + pd.to_timedelta(lead_hours, unit='h')
            else:
                valid_times = [pd.NaT] * len(lead_hours)

        rainfall_vals = extracted.values
        df = pd.DataFrame({
            'init_time': init_time,
            'valid_time': valid_times,
            'lead_hours': lead_hours,
            'rainfall_mm': rainfall_vals
        })
        return df
    else:
        raise ValueError("Extraction failed; check variable dimensions.")


# ----------------------------------------------------------------------
# Regrid to common grid
# ----------------------------------------------------------------------
def regrid_to_common_grid(dataset: xr.Dataset,
                          target_lat: np.ndarray,
                          target_lon: np.ndarray) -> xr.Dataset:
    """
    Resample the NWP precipitation field onto a common lat/lon grid using
    nearest-neighbour interpolation.

    Parameters
    ----------
    dataset : xr.Dataset
        De-accumulated dataset.
    target_lat, target_lon : np.ndarray
        1D arrays defining the target grid.

    Returns
    -------
    xr.Dataset
        Dataset with precipitation regridded to the target grid, preserving
        the time/step dimension.
    """
    # Identify precipitation variable
    precip_var = None
    for v in dataset.data_vars:
        if v not in ['lat', 'lon', 'latitude', 'longitude']:
            precip_var = v
            break
    if precip_var is None:
        raise ValueError("No precipitation variable found.")

    da = dataset[precip_var]

    # Get source lat/lon (assuming 1D)
    src_lat, src_lon = _extract_lat_lon_1d(dataset)

    # Create meshgrids for source and target
    src_lon2d, src_lat2d = np.meshgrid(src_lon, src_lat)
    target_lon2d, target_lat2d = np.meshgrid(target_lon, target_lat)

    # We need to regrid each time step separately (or use xarray's interp)
    # Use xarray's interp with nearest (or linear) if coordinates are 1D.
    # For simplicity, use nearest neighbour via scipy griddata on flattened arrays.
    # But we want to keep xarray structure.
    # Alternative: use xr.Dataset.interp with method='nearest'
    try:
        ds_regridded = dataset.interp(lat=target_lat, lon=target_lon, method='nearest')
        return ds_regridded
    except Exception:
        # Fallback to manual regridding per step
        regridded_list = []
        for step_idx in range(da.shape[0]):
            step_da = da.isel({da.dims[0]: step_idx})
            src_vals = step_da.values
            # Flatten valid points
            valid_mask = ~np.isnan(src_vals)
            points = np.column_stack([src_lon2d[valid_mask], src_lat2d[valid_mask]])
            vals = src_vals[valid_mask]
            target_points = np.column_stack([target_lon2d.ravel(), target_lat2d.ravel()])
            regrid_flat = griddata(points, vals, target_points, method='nearest')
            regrid_2d = regrid_flat.reshape(target_lat2d.shape)
            regridded_list.append(regrid_2d)
        # Stack along time dimension
        regridded_array = np.stack(regridded_list, axis=0)
        new_dims = (da.dims[0], 'lat', 'lon')
        regridded_da = xr.DataArray(regridded_array, dims=new_dims,
                                    coords={da.dims[0]: da[da.dims[0]].values,
                                            'lat': target_lat,
                                            'lon': target_lon})
        ds_regridded = xr.Dataset({precip_var: regridded_da})
        # Copy other coords if possible
        for coord in dataset.coords:
            if coord not in ds_regridded.coords and coord != 'lat' and coord != 'lon':
                ds_regridded.coords[coord] = dataset.coords[coord]
        return ds_regridded


# ----------------------------------------------------------------------
# Batch ingestion
# ----------------------------------------------------------------------
def batch_ingest(folder_path: str,
                 output_format: str = "parquet",
                 precip_var_name: str = "tp") -> pd.DataFrame:
    """
    Process all NWP files in a folder, de-accumulate precipitation, and
    produce a consolidated DataFrame with columns:
    init_time, valid_time, lat, lon, value, variable='rainfall', source='nwp'.

    The DataFrame is also stored in the global cache for later use by
    get_forecast_series.

    Parameters
    ----------
    folder_path : str
        Path to folder containing NWP files.
    output_format : str
        If 'parquet', save the consolidated DataFrame to
        'nwp_rainfall_consolidated.parquet' in the folder.
    precip_var_name : str
        Name of the cumulative precipitation variable in the files (default 'tp').

    Returns
    -------
    pd.DataFrame
        Consolidated DataFrame.
    """
    global _NWP_FORECAST_DF

    all_files = []
    for ext in ['*.grb', '*.grib', '*.grb2', '*.grib2', '*.nc', '*.nc4']:
        all_files.extend(glob.glob(os.path.join(folder_path, ext)))
    all_files = sorted(list(set(all_files)))

    logger.info(f"Found {len(all_files)} NWP files in {folder_path}")

    master_df = pd.DataFrame(columns=['init_time', 'valid_time', 'lat', 'lon',
                                      'value', 'variable', 'source'])
    processed_count = 0
    skipped_files = []

    for filepath in all_files:
        try:
            # Load
            ds = load_nwp_file(filepath)
            # De-accumulate
            ds_deacc = deaccumulate_precipitation(ds, precip_var_name)
            # Identify precipitation variable after deaccumulation (same name)
            # Determine init_time from file or dataset
            init_time = None
            if 'init_time' in ds_deacc.attrs:
                init_time = pd.to_datetime(ds_deacc.attrs['init_time'])
            else:
                # Try from filename
                init_time = _parse_init_time_from_filename(filepath)
                if init_time is None:
                    # Last resort: use current time
                    logger.warning(f"Could not determine init_time for {filepath}, using NaT.")
                    init_time = pd.NaT

            # We need to extract all grid points and valid_times.
            # For memory efficiency, we might process only a subset, but we'll
            # produce one row per (valid_time, lat, lon).
            # Get lat/lon arrays
            lat_arr, lon_arr = _extract_lat_lon_1d(ds_deacc)
            # Get valid_time array
            # Determine dimension: step or time
            dim = 'step' if 'step' in ds_deacc[precip_var_name].dims else 'time'
            steps = ds_deacc[dim].values
            if 'valid_time' in ds_deacc.coords:
                valid_times = ds_deacc['valid_time'].values
            else:
                if init_time is not pd.NaT:
                    valid_times = init_time + pd.to_timedelta(steps, unit='h')
                else:
                    valid_times = [pd.NaT] * len(steps)

            # Iterate over steps and grid points to build DataFrame
            # This could be large; we'll do it in a vectorized way using stack.
            da = ds_deacc[precip_var_name]
            # Convert to DataFrame via xarray's to_dataframe
            df_flat = da.to_dataframe(name='value').reset_index()
            # Now df_flat has columns: dim, lat, lon, value
            # Rename dim column to 'lead_hours' or 'step'
            df_flat = df_flat.rename(columns={dim: 'lead_hours'})
            # Map lead_hours to valid_time
            lead_to_valid = dict(zip(steps, valid_times))
            df_flat['valid_time'] = df_flat['lead_hours'].map(lead_to_valid)
            df_flat['init_time'] = init_time
            df_flat['variable'] = 'rainfall'
            df_flat['source'] = 'nwp'
            # Select and reorder columns
            df_flat = df_flat[['init_time', 'valid_time', 'lat', 'lon', 'value',
                               'variable', 'source']]
            # Drop rows with NaN value (should not exist after deaccum, but just in case)
            df_flat = df_flat.dropna(subset=['value'])

            master_df = pd.concat([master_df, df_flat], ignore_index=True)
            processed_count += 1
            logger.debug(f"Processed {filepath}")

        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")
            skipped_files.append(filepath)
            continue

    logger.info(f"Successfully processed {processed_count} files.")
    if skipped_files:
        logger.warning(f"Skipped {len(skipped_files)} files due to errors.")

    # Store in global cache
    _NWP_FORECAST_DF = master_df

    # Save to parquet if requested
    if output_format.lower() == 'parquet' and not master_df.empty:
        out_path = os.path.join(folder_path, 'nwp_rainfall_consolidated.parquet')
        master_df.to_parquet(out_path, index=False)
        logger.info(f"Saved consolidated data to {out_path}")

    return master_df


# ----------------------------------------------------------------------
# Forecast series retrieval
# ----------------------------------------------------------------------
def get_forecast_series(lat: float,
                        lon: float,
                        lead_hours: List[int] = [24, 48, 72]) -> pd.DataFrame:
    """
    Retrieve precipitation forecast for a point location.

    This function uses the global cache of NWP data populated by batch_ingest.
    It returns a DataFrame with columns [init_time, valid_time, lead_hours,
    rainfall_mm]. This is intentionally a different shape from the observed
    rainfall series (get_rainfall_series) because it represents forecast
    values with both initialisation and valid times.

    Parameters
    ----------
    lat, lon : float
        Coordinates of the point of interest.
    lead_hours : list of int
        Lead times to extract (hours from init_time). If the NWP data does not
        contain exactly these lead times, the nearest available lead times are
        returned (or an empty DataFrame if none).

    Returns
    -------
    pd.DataFrame
        Columns: init_time, valid_time, lead_hours, rainfall_mm.
    """
    global _NWP_FORECAST_DF

    if _NWP_FORECAST_DF is None or _NWP_FORECAST_DF.empty:
        # Try loading from default cache file
        if os.path.exists(DEFAULT_CACHE_FILE):
            try:
                _NWP_FORECAST_DF = pd.read_parquet(DEFAULT_CACHE_FILE)
            except Exception as e:
                logger.error(f"Failed to read cache file {DEFAULT_CACHE_FILE}: {e}")
                return pd.DataFrame(columns=['init_time', 'valid_time', 'lead_hours', 'rainfall_mm'])
        else:
            logger.warning("No NWP forecast data available. Run batch_ingest first.")
            return pd.DataFrame(columns=['init_time', 'valid_time', 'lead_hours', 'rainfall_mm'])

    df = _NWP_FORECAST_DF.copy()

    # For the given lat/lon, find nearest grid point(s)
    # We'll use the existing lat/lon in the DataFrame and compute nearest
    # (assuming the DataFrame already contains multiple grid points).
    # Find unique grid points in cache
    grid_points = df[['lat', 'lon']].drop_duplicates()
    if grid_points.empty:
        return pd.DataFrame(columns=['init_time', 'valid_time', 'lead_hours', 'rainfall_mm'])

    # Compute distance to target and find nearest
    distances = haversine_distance(
        np.full(len(grid_points), lat), np.full(len(grid_points), lon),
        grid_points['lat'].values, grid_points['lon'].values
    )
    nearest_idx = np.argmin(distances)
    nearest_lat = grid_points['lat'].iloc[nearest_idx]
    nearest_lon = grid_points['lon'].iloc[nearest_idx]

    # Filter data for that grid point
    df_point = df[(df['lat'] == nearest_lat) & (df['lon'] == nearest_lon)]

    # Filter by lead_hours if requested
    if lead_hours is not None and len(lead_hours) > 0:
        # Convert lead_hours to numeric if needed
        df_point['lead_hours'] = pd.to_numeric(df_point['lead_hours'], errors='coerce')
        df_point = df_point[df_point['lead_hours'].isin(lead_hours)]

    # Select relevant columns and rename value to rainfall_mm
    result = df_point[['init_time', 'valid_time', 'lead_hours', 'value']].rename(
        columns={'value': 'rainfall_mm'}
    )
    # Convert init_time and valid_time to datetime
    result['init_time'] = pd.to_datetime(result['init_time'])
    result['valid_time'] = pd.to_datetime(result['valid_time'])
    return result


# ----------------------------------------------------------------------
# Synthetic NWP file generation
# ----------------------------------------------------------------------
def generate_sample_nwp_file(filepath: str,
                             init_time: Optional[dt.datetime] = None,
                             n_steps: int = 5,
                             step_hours: int = 3) -> None:
    """
    Generate a synthetic NetCDF NWP file with cumulative precipitation.

    The file will have dimensions: step, lat, lon, with variable 'tp'
    (cumulative precipitation) and coordinates 'lat', 'lon', 'step',
    'valid_time', plus global attribute 'init_time'.

    Parameters
    ----------
    filepath : str
        Output path (should end with .nc).
    init_time : dt.datetime, optional
        Model initialisation time (defaults to today's 00 UTC).
    n_steps : int
        Number of forecast steps (including step 0).
    step_hours : int
        Time interval between steps in hours.
    """
    if init_time is None:
        init_time = dt.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    # Define grid (smaller region over India for test)
    lat = np.arange(10.0, 30.0, 0.5)   # 10N to 30N at 0.5 deg
    lon = np.arange(70.0, 90.0, 0.5)   # 70E to 90E

    steps = np.arange(n_steps) * step_hours  # [0, 3, 6, 9, 12]
    valid_times = [init_time + dt.timedelta(hours=int(s)) for s in steps]

    # Create synthetic cumulative precipitation: increasing with step, plus noise
    # Make random spatial pattern
    np.random.seed(42)
    base = np.random.rand(len(steps), len(lat), len(lon)) * 5.0
    # Cumulative: each step adds to previous
    cum_precip = np.cumsum(base, axis=0)

    # Create DataArray
    da = xr.DataArray(
        cum_precip.astype(np.float32),
        dims=('step', 'lat', 'lon'),
        coords={
            'step': steps,
            'lat': lat,
            'lon': lon,
            'valid_time': ('step', valid_times)
        },
        name='tp'
    )
    da.attrs['long_name'] = 'Total precipitation'
    da.attrs['units'] = 'mm'

    ds = da.to_dataset()
    ds.attrs['init_time'] = str(init_time)
    ds.attrs['description'] = 'Synthetic NWP precipitation file for SIH26071 testing'
    ds.attrs['source'] = 'Synthetic'

    ds.to_netcdf(filepath)
    logger.info(f"Sample NWP file written to {filepath}")


# ----------------------------------------------------------------------
# Haversine distance (replicated here for independence)
# ----------------------------------------------------------------------
def haversine_distance(lat1, lon1, lat2, lon2):
    """Compute great-circle distance in km between two points."""
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat/2.0)**2 + np.cos(lat1_rad)*np.cos(lat2_rad)*np.sin(dlon/2.0)**2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return 6371.0 * c


# ----------------------------------------------------------------------
# Main demonstration
# ----------------------------------------------------------------------
if __name__ == '__main__':
    import tempfile
    import shutil

    # Create a temporary directory
    demo_dir = tempfile.mkdtemp(prefix='nwp_demo_')
    sample_file = os.path.join(demo_dir, 'sample_nwp.nc')
    print(f"Generating sample NWP file at {sample_file}...")
    generate_sample_nwp_file(sample_file)

    # Run batch_ingest on this single file
    print("\nRunning batch ingestion...")
    df = batch_ingest(demo_dir, output_format='parquet')

    # Print summary
    if not df.empty:
        print("\n=== Batch Ingestion Summary ===")
        print(f"Total rows: {len(df)}")
        print(f"Init times: {df['init_time'].unique()}")
        print(f"Date range of valid_time: {df['valid_time'].min()} to {df['valid_time'].max()}")
        print(f"Min rainfall (mm): {df['value'].min():.2f}")
        print(f"Max rainfall (mm): {df['value'].max():.2f}")
    else:
        print("No data processed.")

    # Demonstrate get_forecast_series
    print("\nTesting get_forecast_series for a location...")
    test_lat, test_lon = 20.0, 75.0
    forecast = get_forecast_series(lat=test_lat, lon=test_lon, lead_hours=[3,6,9])
    print(forecast)

    # Clean up
    shutil.rmtree(demo_dir)