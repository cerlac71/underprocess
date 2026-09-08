#!/usr/bin/env python3
"""
Observational (Rain Gauge / AWS) Data Ingestion Module for SIH26071.

This module provides functions to load, clean, and integrate station-based
rainfall observations from CSV/text exports (e.g., IMDLIB or any similar
station data source). It handles sentinel missing values, station metadata
mapping, nearest-station search using haversine distance, and returns data
in the same shape expected by the existing pipeline (columns:
timestamp, rainfall_mm for time-series queries; timestamp, lat, lon, value
for consolidated batch output).

Key functions:
- load_station_metadata: read station_id -> lat/lon/name/elevation.
- load_observational_file: load one station data file, join metadata, mask
  sentinel values, return clean DataFrame.
- find_nearest_stations: return k nearest stations within a radius.
- get_rainfall_series: drop-in replacement for
  generate_synthetic_rainfall_series() – returns a time series for a point
  by inverse-distance weighting nearby stations.
- batch_ingest: process all files in a folder and produce a consolidated
  parquet file.

Station data quirks handled (or documented):
- Sentinels like -999, -99, -9999 are common for missing data; we replace
  them with NaN and then drop.
- Stations may stop reporting; we simply ignore missing timestamps.
- All timestamps are assumed to be UTC unless the file has a 'timezone'
  column or metadata says otherwise.
- Station relocations: if a station_id changes location over time, the
  metadata file may contain multiple rows for the same station_id; we
  currently take the first row for simplicity, but a more robust version
  could handle per-period locations.
"""

import os
import re
import glob
import logging
import datetime as dt
from typing import Optional, Tuple, List, Union, Dict, Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Constants and defaults
# ----------------------------------------------------------------------
# Common sentinel values that indicate missing data
SENTINEL_VALUES = [-999, -9999, -99, -999.0, -9999.0, -99.0, -32768, -32768.0]

# Global cache for consolidated observational data
_OBS_DATA_CACHE: Optional[pd.DataFrame] = None
DEFAULT_CACHE_FILE = 'observational_rainfall_cache.parquet'

# Default radius and k for nearest stations
DEFAULT_MAX_RADIUS_KM = 50.0
DEFAULT_K = 3

# ----------------------------------------------------------------------
# Haversine distance
# ----------------------------------------------------------------------
def haversine_distance(lat1: np.ndarray, lon1: np.ndarray,
                       lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """
    Compute great-circle distance (in km) between two sets of lat/lon points.

    Parameters
    ----------
    lat1, lon1 : np.ndarray
        Coordinates of the first set (scalar or array).
    lat2, lon2 : np.ndarray
        Coordinates of the second set (scalar or array).
        If both are arrays, they must be broadcastable.

    Returns
    -------
    np.ndarray
        Distance in kilometers.
    """
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat/2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2.0)**2
    c = 2.0 * np.arcsin(np.sqrt(a))
    earth_radius_km = 6371.0
    return earth_radius_km * c


# ----------------------------------------------------------------------
# Metadata loading
# ----------------------------------------------------------------------
def load_station_metadata(metadata_path: str) -> pd.DataFrame:
    """
    Load station metadata from a CSV file.

    Expected columns (case-insensitive):
    - station_id (or station, id, station_code)
    - lat (or latitude)
    - lon (or longitude, long)
    - name (optional)
    - elevation (optional, in meters)

    Parameters
    ----------
    metadata_path : str
        Path to CSV file.

    Returns
    -------
    pd.DataFrame
        Metadata with standardized column names:
        station_id, lat, lon, name, elevation.
        Extra columns are preserved.
    """
    try:
        df = pd.read_csv(metadata_path)
    except Exception as e:
        logger.error(f"Failed to read metadata file {metadata_path}: {e}")
        raise

    # Standardize column names (case-insensitive mapping)
    col_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ['station_id', 'station', 'id', 'station_code']:
            col_map[col] = 'station_id'
        elif col_lower in ['lat', 'latitude']:
            col_map[col] = 'lat'
        elif col_lower in ['lon', 'long', 'longitude']:
            col_map[col] = 'lon'
        elif col_lower in ['name', 'station_name']:
            col_map[col] = 'name'
        elif col_lower in ['elev', 'elevation', 'altitude']:
            col_map[col] = 'elevation'
    df = df.rename(columns=col_map)

    # Ensure required columns exist
    for req in ['station_id', 'lat', 'lon']:
        if req not in df.columns:
            raise ValueError(f"Metadata file missing required column: {req}")

    # Convert station_id to string for consistency
    df['station_id'] = df['station_id'].astype(str)
    # Add missing name/elevation columns if not present
    if 'name' not in df.columns:
        df['name'] = df['station_id']
    if 'elevation' not in df.columns:
        df['elevation'] = np.nan

    return df


# ----------------------------------------------------------------------
# Observational file loading
# ----------------------------------------------------------------------
def load_observational_file(filepath: str,
                            metadata_path: str) -> pd.DataFrame:
    """
    Load one station observation file, join with metadata, and clean data.

    The file is expected to be CSV with at least columns:
    - station_id (or station, id)
    - datetime (or time, timestamp, date) – parseable by pandas.
    - rainfall_mm (or precip, rain, rainfall, value) – numeric.

    Sentinel values (e.g., -999) are replaced with NaN and dropped.
    A 'flag' column may be present; rows with flag indicating bad data
    (e.g., flag != 0) are removed if possible.

    Parameters
    ----------
    filepath : str
        Path to the observational CSV.
    metadata_path : str
        Path to the station metadata CSV.

    Returns
    -------
    pd.DataFrame
        Clean DataFrame with columns:
        timestamp, station_id, lat, lon, rainfall_mm.
        The timestamp column is of type datetime64[ns], timezone-naive (UTC).
    """
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        logger.error(f"Failed to read observation file {filepath}: {e}")
        raise

    # Standardize column names
    col_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ['station_id', 'station', 'id', 'station_code']:
            col_map[col] = 'station_id'
        elif col_lower in ['datetime', 'time', 'timestamp', 'date']:
            col_map[col] = 'datetime'
        elif col_lower in ['rainfall_mm', 'precip', 'rain', 'rainfall', 'value', 'prcp']:
            col_map[col] = 'rainfall_mm'
        elif col_lower == 'flag':
            col_map[col] = 'flag'
    df = df.rename(columns=col_map)

    required = ['station_id', 'datetime', 'rainfall_mm']
    for req in required:
        if req not in df.columns:
            raise ValueError(f"Observation file {filepath} missing required column: {req}")

    # Convert station_id to string
    df['station_id'] = df['station_id'].astype(str)

    # Parse datetime (assume UTC)
    try:
        df['timestamp'] = pd.to_datetime(df['datetime'], utc=True)
    except Exception as e:
        logger.error(f"Failed to parse datetime in {filepath}: {e}")
        raise
    # Remove timezone info to keep naive UTC
    df['timestamp'] = df['timestamp'].dt.tz_localize(None)

    # Clean rainfall values: replace sentinels with NaN, then drop
    df['rainfall_mm'] = pd.to_numeric(df['rainfall_mm'], errors='coerce')
    df['rainfall_mm'] = df['rainfall_mm'].replace(SENTINEL_VALUES, np.nan)
    df = df.dropna(subset=['rainfall_mm'])

    # If a flag column exists, remove rows with flag != 0 (assuming 0 = good)
    if 'flag' in df.columns:
        # If flag is numeric, keep only rows where flag == 0
        # If flag is string, you might need custom logic; we'll assume numeric
        try:
            df['flag'] = pd.to_numeric(df['flag'], errors='coerce')
            df = df[df['flag'] == 0]
        except:
            pass  # if conversion fails, ignore flag column

    # Join with metadata
    metadata = load_station_metadata(metadata_path)
    # Keep only necessary columns from metadata
    meta_cols = ['station_id', 'lat', 'lon']
    meta_subset = metadata[meta_cols].drop_duplicates(subset='station_id', keep='first')
    df = df.merge(meta_subset, on='station_id', how='inner')

    # Drop rows where metadata missing (should not happen after inner merge)
    df = df.dropna(subset=['lat', 'lon'])

    # Select and order final columns
    df = df[['timestamp', 'station_id', 'lat', 'lon', 'rainfall_mm']]
    return df


# ----------------------------------------------------------------------
# Nearest station search
# ----------------------------------------------------------------------
def find_nearest_stations(lat: float,
                          lon: float,
                          metadata_df: pd.DataFrame,
                          k: int = DEFAULT_K,
                          max_radius_km: float = DEFAULT_MAX_RADIUS_KM) -> pd.DataFrame:
    """
    Find the k nearest stations to a target point within a radius.

    Uses haversine distance (great-circle) for accuracy. The function
    returns a subset of metadata_df with an added 'distance_km' column,
    sorted by distance ascending.

    Parameters
    ----------
    lat, lon : float
        Target coordinates.
    metadata_df : pd.DataFrame
        Station metadata (must contain 'station_id', 'lat', 'lon').
    k : int
        Maximum number of stations to return.
    max_radius_km : float
        Only stations within this distance are considered.

    Returns
    -------
    pd.DataFrame
        Up to k stations within max_radius_km, with columns from metadata
        plus 'distance_km'. Sorted by distance ascending. Empty if none.
    """
    if metadata_df.empty:
        return pd.DataFrame()

    # Compute haversine distance to all stations
    distances = haversine_distance(
        np.full(len(metadata_df), lat),
        np.full(len(metadata_df), lon),
        metadata_df['lat'].values,
        metadata_df['lon'].values
    )

    # Add distance column to a copy
    result = metadata_df.copy()
    result['distance_km'] = distances

    # Filter by radius
    result = result[result['distance_km'] <= max_radius_km]

    # Sort and take top k
    result = result.sort_values('distance_km').head(k)

    return result


# ----------------------------------------------------------------------
# Consolidated data cache and series retrieval
# ----------------------------------------------------------------------
def _get_consolidated_data() -> pd.DataFrame:
    """
    Retrieve the consolidated observational data from global cache or parquet.

    Returns
    -------
    pd.DataFrame
        Consolidated DataFrame with columns:
        timestamp, station_id, lat, lon, rainfall_mm.
    """
    global _OBS_DATA_CACHE
    if _OBS_DATA_CACHE is not None and not _OBS_DATA_CACHE.empty:
        return _OBS_DATA_CACHE
    # Try to load from default cache file
    if os.path.exists(DEFAULT_CACHE_FILE):
        try:
            df = pd.read_parquet(DEFAULT_CACHE_FILE)
            _OBS_DATA_CACHE = df
            return df
        except Exception as e:
            logger.error(f"Failed to read cache file {DEFAULT_CACHE_FILE}: {e}")
    # If still empty, return empty DataFrame with expected columns
    return pd.DataFrame(columns=['timestamp', 'station_id', 'lat', 'lon', 'rainfall_mm'])


def get_rainfall_series(lat: float,
                        lon: float,
                        hours: int = 24,
                        k: int = DEFAULT_K) -> pd.DataFrame:
    """
    Retrieve a time series of rainfall for a given point using nearby stations.

    This function is a drop-in replacement for
    `generate_synthetic_rainfall_series()` in `src/data_utils.py`. It returns
    a DataFrame with columns `timestamp` and `rainfall_mm`, where rainfall_mm
    is the inverse-distance-weighted average of the k nearest stations within
    a default radius (currently 50 km). If no stations are available, an
    empty DataFrame is returned and a warning logged.

    Parameters
    ----------
    lat, lon : float
        Coordinates of the point of interest.
    hours : int
        Number of past hours to include.
    k : int
        Number of nearest stations to consider.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'timestamp' and 'rainfall_mm'. Timestamps are
        hourly or as available; missing timestamps are not filled.
    """
    # Get consolidated data
    df = _get_consolidated_data()
    if df.empty:
        logger.warning("No observational data available. Run batch_ingest first.")
        return pd.DataFrame(columns=['timestamp', 'rainfall_mm'])

    # Filter by time window
    now = dt.datetime.now()
    start_time = now - dt.timedelta(hours=hours)
    df_time = df[df['timestamp'] >= start_time].copy()
    if df_time.empty:
        logger.warning(f"No data in the last {hours} hours.")
        return pd.DataFrame(columns=['timestamp', 'rainfall_mm'])

    # Find nearest stations from the metadata of stations present in data
    available_stations = df_time['station_id'].unique()
    # We need a metadata subset for those stations
    # Since we don't have metadata loaded separately, we can reconstruct from
    # the data itself: each row has lat/lon; we can get unique station locations.
    station_locations = df_time.groupby('station_id')[['lat', 'lon']].first().reset_index()
    # Find nearest stations
    nearest = find_nearest_stations(lat, lon, station_locations, k=k,
                                    max_radius_km=DEFAULT_MAX_RADIUS_KM)
    if nearest.empty:
        logger.warning(f"No stations within {DEFAULT_MAX_RADIUS_KM} km of ({lat}, {lon}).")
        return pd.DataFrame(columns=['timestamp', 'rainfall_mm'])

    # Filter data to only include nearest stations
    nearest_ids = nearest['station_id'].tolist()
    df_near = df_time[df_time['station_id'].isin(nearest_ids)]

    # Compute inverse-distance weights
    # Use distance_km from nearest (which is small, so no need to recompute)
    # But we need per-station weights. We'll compute from nearest DataFrame.
    weights = {}
    for _, row in nearest.iterrows():
        sid = row['station_id']
        dist = row['distance_km']
        # Avoid division by zero: if distance is 0, weight should be 1 and others 0
        if dist < 1e-6:
            # This station is at the exact point, set its weight to 1 and others to 0
            weights = {sid: 1.0}
            break
        weights[sid] = 1.0 / dist
    total_weight = sum(weights.values())
    if total_weight == 0:
        return pd.DataFrame(columns=['timestamp', 'rainfall_mm'])

    # Normalize weights
    for sid in weights:
        weights[sid] /= total_weight

    # For each timestamp, compute weighted average of rainfall across stations
    # that have data at that timestamp.
    # First, pivot to get rainfall per station per timestamp
    pivot = df_near.pivot_table(index='timestamp', columns='station_id',
                                values='rainfall_mm', aggfunc='mean')
    # Compute weighted average: for each row, sum(value * weight) over columns
    # but only for non-NaN entries and renormalize weights accordingly.
    result_rows = []
    for ts, row in pivot.iterrows():
        weighted_sum = 0.0
        sum_weights_used = 0.0
        for sid, w in weights.items():
            val = row.get(sid)
            if pd.notna(val):
                weighted_sum += val * w
                sum_weights_used += w
        if sum_weights_used > 0:
            rainfall = weighted_sum / sum_weights_used
        else:
            rainfall = np.nan
        result_rows.append({'timestamp': ts, 'rainfall_mm': rainfall})

    result = pd.DataFrame(result_rows)
    # Drop rows with NaN rainfall (no station data for that timestamp)
    result = result.dropna(subset=['rainfall_mm'])
    return result


# ----------------------------------------------------------------------
# Batch ingestion
# ----------------------------------------------------------------------
def batch_ingest(folder_path: str,
                 metadata_path: str,
                 output_format: str = "parquet") -> pd.DataFrame:
    """
    Process all observational CSV files in a folder and consolidate.

    Each file is loaded using `load_observational_file`, which joins with
    metadata and cleans data. The resulting DataFrames are concatenated into
    a master DataFrame with columns:
    timestamp, lat, lon, value, variable='rainfall', source='observational'.

    The consolidated DataFrame is stored in the global cache
    (`_OBS_DATA_CACHE`) and optionally saved to a Parquet file in the folder.

    Parameters
    ----------
    folder_path : str
        Path to folder containing observational CSV files.
    metadata_path : str
        Path to station metadata CSV.
    output_format : str
        If 'parquet', save the consolidated DataFrame to
        'observational_rainfall_consolidated.parquet' in the folder.

    Returns
    -------
    pd.DataFrame
        Consolidated DataFrame with columns:
        timestamp, lat, lon, value, variable, source.
    """
    global _OBS_DATA_CACHE

    all_files = glob.glob(os.path.join(folder_path, '*.csv'))
    all_files = sorted(all_files)
    logger.info(f"Found {len(all_files)} CSV files in {folder_path}")

    master_df = pd.DataFrame(columns=['timestamp', 'lat', 'lon', 'value',
                                      'variable', 'source'])
    processed_count = 0
    skipped_files = []

    for filepath in all_files:
        try:
            # Load and clean one file
            df_clean = load_observational_file(filepath, metadata_path)
            if df_clean.empty:
                logger.warning(f"No valid data in {filepath}, skipping.")
                skipped_files.append(filepath)
                continue

            # Transform to required output columns
            df_out = pd.DataFrame({
                'timestamp': df_clean['timestamp'],
                'lat': df_clean['lat'],
                'lon': df_clean['lon'],
                'value': df_clean['rainfall_mm'],
                'variable': 'rainfall',
                'source': 'observational'
            })

            master_df = pd.concat([master_df, df_out], ignore_index=True)
            processed_count += 1
            logger.debug(f"Processed {filepath}")

        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")
            skipped_files.append(filepath)
            continue

    logger.info(f"Successfully processed {processed_count} files.")
    if skipped_files:
        logger.warning(f"Skipped {len(skipped_files)} files due to errors or no valid data.")

    # Store in global cache (also save a copy in the original format for
    # get_rainfall_series? Actually get_rainfall_series expects columns
    # timestamp, station_id, lat, lon, rainfall_mm, but we can reconstruct
    # from the output if needed. To keep it simple, we'll store the original
    # cleaned data (with station_id) in a separate global variable, or we can
    # derive it from master_df by adding station_id? The master_df doesn't
    # have station_id. For get_rainfall_series we need station_id. So we need
    # to also store a version with station_id. We'll store the original
    # cleaned data from each file in a global variable _OBS_DATA_CACHE, but
    # that would require re-joining. Instead, we can rebuild a cache from the
    # master_df by grouping? No, master_df lacks station_id. To satisfy
    # get_rainfall_series, we need station_id. We'll store the raw cleaned
    # data as a separate global variable during batch_ingest. For simplicity,
    # we'll accumulate a separate DataFrame with columns:
    # timestamp, station_id, lat, lon, rainfall_mm.
    # But we are not returning it; we can still store it.
    # We'll create a global variable _OBS_DATA_CACHE_RAW to hold this.
    global _OBS_DATA_CACHE_RAW
    if not hasattr(_OBS_DATA_CACHE_RAW, '__iter__'):
        _OBS_DATA_CACHE_RAW = pd.DataFrame(columns=['timestamp', 'station_id',
                                                    'lat', 'lon', 'rainfall_mm'])
    # For each processed file, append its raw cleaned data to _OBS_DATA_CACHE_RAW
    # We'll do that inside the loop.
    # But we need to have the raw data available. We'll modify the loop to
    # also keep a list of raw dataframes.

    # Instead, after the loop, we can reconstruct raw cache from master_df
    # by grouping? No, station_id is lost. So we need to keep it.
    # We'll create a separate accumulator:
    raw_cache_list = []

    # Re-run loop with raw accumulation (simpler to write again, but we can
    # modify the above loop to collect raw data). Since this is a fresh code,
    # I will rewrite the loop with both outputs.

    # Reset master_df and loop again
    master_df = pd.DataFrame(columns=['timestamp', 'lat', 'lon', 'value',
                                      'variable', 'source'])
    raw_cache_list = []
    processed_count = 0
    skipped_files = []

    for filepath in all_files:
        try:
            df_clean = load_observational_file(filepath, metadata_path)
            if df_clean.empty:
                logger.warning(f"No valid data in {filepath}, skipping.")
                skipped_files.append(filepath)
                continue

            # Append raw cleaned data to cache list
            raw_cache_list.append(df_clean)

            df_out = pd.DataFrame({
                'timestamp': df_clean['timestamp'],
                'lat': df_clean['lat'],
                'lon': df_clean['lon'],
                'value': df_clean['rainfall_mm'],
                'variable': 'rainfall',
                'source': 'observational'
            })
            master_df = pd.concat([master_df, df_out], ignore_index=True)
            processed_count += 1
        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")
            skipped_files.append(filepath)
            continue

    # Combine raw cache
    if raw_cache_list:
        _OBS_DATA_CACHE_RAW = pd.concat(raw_cache_list, ignore_index=True)
    else:
        _OBS_DATA_CACHE_RAW = pd.DataFrame(columns=['timestamp', 'station_id',
                                                    'lat', 'lon', 'rainfall_mm'])

    # Also update the global _OBS_DATA_CACHE to point to raw cache, since
    # get_rainfall_series expects columns timestamp, station_id, lat, lon,
    # rainfall_mm.
    _OBS_DATA_CACHE = _OBS_DATA_CACHE_RAW

    # Save to parquet if requested
    if output_format.lower() == 'parquet' and not master_df.empty:
        out_path = os.path.join(folder_path, 'observational_rainfall_consolidated.parquet')
        master_df.to_parquet(out_path, index=False)
        logger.info(f"Saved consolidated data to {out_path}")

    return master_df


# ----------------------------------------------------------------------
# Synthetic data generation for testing
# ----------------------------------------------------------------------
def generate_sample_station_data(folder_path: str,
                                 n_stations: int = 10,
                                 days: int = 2) -> None:
    """
    Generate synthetic station data files for testing.

    Creates a metadata CSV and one CSV per station with hourly rainfall data
    for the last `days` days (with some missing values and sentinel values).

    Parameters
    ----------
    folder_path : str
        Directory where files will be written. Created if it doesn't exist.
    n_stations : int
        Number of synthetic stations.
    days : int
        Number of days of data to generate.
    """
    os.makedirs(folder_path, exist_ok=True)

    # Generate random station locations within India (approx bounding box)
    np.random.seed(42)
    lats = np.random.uniform(8.0, 35.0, n_stations)
    lons = np.random.uniform(68.0, 97.0, n_stations)
    elevations = np.random.uniform(0, 2000, n_stations)
    station_ids = [f'ST{i:03d}' for i in range(n_stations)]
    names = [f'Sample Station {i}' for i in range(n_stations)]

    metadata_df = pd.DataFrame({
        'station_id': station_ids,
        'lat': lats,
        'lon': lons,
        'name': names,
        'elevation': elevations
    })
    metadata_path = os.path.join(folder_path, 'station_metadata.csv')
    metadata_df.to_csv(metadata_path, index=False)
    logger.info(f"Metadata written to {metadata_path}")

    # Generate hourly data for each station
    end_time = dt.datetime.now().replace(minute=0, second=0, microsecond=0)
    start_time = end_time - dt.timedelta(days=days)
    hourly_range = pd.date_range(start=start_time, end=end_time, freq='H')

    for sid, lat, lon in zip(station_ids, lats, lons):
        # Synthetic rainfall: random with occasional heavy events
        base_rain = np.random.exponential(0.5, len(hourly_range))
        # Add a sinusoidal daily cycle
        daily_cycle = 2.0 * (1 + np.sin(np.linspace(0, 2*np.pi, len(hourly_range))))
        rainfall = base_rain * daily_cycle * 0.5
        # Make some hours zero
        rainfall[np.random.rand(len(hourly_range)) < 0.3] = 0.0
        # Introduce missing values: replace some with sentinel -999
        missing_mask = np.random.rand(len(hourly_range)) < 0.05
        rainfall[missing_mask] = -999
        # Also some NaN
        nan_mask = np.random.rand(len(hourly_range)) < 0.03
        rainfall[nan_mask] = np.nan

        df_station = pd.DataFrame({
            'station_id': sid,
            'datetime': hourly_range,
            'rainfall_mm': rainfall
        })
        # Add a flag column sometimes (0 = good, 1 = bad)
        if np.random.rand() > 0.5:
            flag = np.zeros(len(hourly_range), dtype=int)
            flag[np.random.rand(len(hourly_range)) < 0.02] = 1
            df_station['flag'] = flag

        filepath = os.path.join(folder_path, f'rainfall_{sid}.csv')
        df_station.to_csv(filepath, index=False)
        logger.debug(f"Wrote {filepath}")

    logger.info(f"Sample data generated in {folder_path}")


# ----------------------------------------------------------------------
# Main demonstration
# ----------------------------------------------------------------------
if __name__ == '__main__':
    import tempfile
    import shutil

    # Create a temporary folder for demonstration
    demo_dir = tempfile.mkdtemp(prefix='obs_demo_')
    print(f"Generating sample station data in {demo_dir}...")
    generate_sample_station_data(demo_dir, n_stations=5, days=1)

    metadata_path = os.path.join(demo_dir, 'station_metadata.csv')
    print("\nRunning batch ingestion...")
    df = batch_ingest(demo_dir, metadata_path, output_format='parquet')

    # Print summary stats
    if not df.empty:
        print("\n=== Batch Ingestion Summary ===")
        print(f"Total rows: {len(df)}")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Unique stations (from data): {df['lat'].nunique()} (lat/lon combos)")
        print(f"Min rainfall (mm): {df['value'].min():.2f}")
        print(f"Max rainfall (mm): {df['value'].max():.2f}")
        print(f"Mean rainfall (mm): {df['value'].mean():.2f}")
    else:
        print("No data processed.")

    # Demonstrate get_rainfall_series
    print("\nTesting get_rainfall_series for a location...")
    # Use a point near one of the stations (e.g., first station's location)
    if not df.empty:
        sample_lat = df['lat'].iloc[0]
        sample_lon = df['lon'].iloc[0]
        series = get_rainfall_series(lat=sample_lat, lon=sample_lon,
                                     hours=24, k=3)
        print(f"Retrieved {len(series)} time steps.")
        if not series.empty:
            print(series.head())

    # Clean up
    shutil.rmtree(demo_dir)