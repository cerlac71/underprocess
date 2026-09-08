#!/usr/bin/env python3
"""
Data Harmonization Module for SIH26071.

This module combines observations and forecasts from multiple sources
(rain gauges, radar, satellite, NWP) onto a common spatial grid and
temporal resolution, ready for AI/ML model input.

Key features:
- Spatial regridding using conservative remapping (for precipitation)
  or bilinear/nearest interpolation (for non-conservative fields).
- Temporal aggregation to a common time step (default: hourly).
- Unit normalization to mm per time step.
- Explicit handling of missing data, spatial extents, and UTC alignment.
- Validation plots for visual sanity checks.

Design decisions explained in code comments and docstrings.
"""


import logging
import numpy as np
import pandas as pd
import xarray as xr
from typing import Optional, List, Dict, Tuple, Union
from datetime import datetime, timedelta

# Configure logging FIRST
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional imports
try:
    import xesmf as xe
    HAS_XESMF = True
except ImportError:
    HAS_XESMF = False
    logger.info("xesmf not installed; falling back to scipy.interpolate for regridding.")   # <-- now logger exists


from scipy.interpolate import griddata

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# 1. Target grid definition
# ----------------------------------------------------------------------
def define_target_grid(lat_min: float, lat_max: float,
                       lon_min: float, lon_max: float,
                       resolution_km: float = 1.0,
                       grid_step_deg: float | None = None) -> xr.Dataset:
    """
    Create a regular lat/lon grid with given resolution (in km).

    The grid is used as the common output for all sources.
    Cell boundaries are computed for conservative regridding.

    Parameters
    ----------
    lat_min, lat_max : float
        Southern and northern bounds.
    lon_min, lon_max : float
        Western and eastern bounds.
    resolution_km : float
        Approximate grid spacing in km (converted to degrees). Ignored if
        ``grid_step_deg`` is set.
    grid_step_deg : float, optional
        Fixed lat/lon spacing in degrees (e.g. 0.25 for IMD 0.25° grid).

    Returns
    -------
    xr.Dataset
        Dataset with lat/lon coordinates (1D) and optional cell boundaries.
    """
    if grid_step_deg is not None:
        dlat = dlon = float(grid_step_deg)
    else:
        deg_per_km = 1.0 / 111.0
        dlat = resolution_km * deg_per_km
        dlon = resolution_km * deg_per_km

    lat = np.arange(lat_min, lat_max + dlat * 0.5, dlat)
    lon = np.arange(lon_min, lon_max + dlon * 0.5, dlon)

    # For conservative regridding, we need cell corners.
    # For a regular grid with constant spacing, corners are halfway between centres.
    lat_b = np.concatenate([[lat[0] - dlat/2], lat + dlat/2])
    lon_b = np.concatenate([[lon[0] - dlon/2], lon + dlon/2])

    ds = xr.Dataset(
        coords={
            'lat': lat,
            'lon': lon,
            'lat_b': lat_b,
            'lon_b': lon_b
        }
    )
    ds.attrs['description'] = 'Common target grid for harmonization'
    return ds


# ----------------------------------------------------------------------
# 2. Spatial regridding
# ----------------------------------------------------------------------
def regrid_source(source_ds: xr.Dataset,
                  target_grid: xr.Dataset,
                  var_name: str,
                  method: str = 'conservative') -> xr.DataArray:
    """
    Regrid a source variable onto the target grid.

    Parameters
    ----------
    source_ds : xr.Dataset
        Source dataset containing the variable and lat/lon coordinates.
    target_grid : xr.Dataset
        Target grid dataset (from define_target_grid).
    var_name : str
        Name of the variable to regrid.
    method : str
        'conservative' for accumulative variables (rainfall),
        'bilinear' or 'nearest' for other fields.

    Returns
    -------
    xr.DataArray
        Regridded variable on the target grid.
    """
    # Check coordinate names
    src_lat_name = None
    src_lon_name = None
    for coord in source_ds.coords:
        if coord.lower() in ['lat', 'latitude']:
            src_lat_name = coord
        elif coord.lower() in ['lon', 'longitude']:
            src_lon_name = coord
    if src_lat_name is None or src_lon_name is None:
        raise ValueError("Source dataset missing lat/lon coordinates.")

    # Ensure target grid has lat/lon
    if 'lat' not in target_grid.coords or 'lon' not in target_grid.coords:
        raise ValueError("Target grid must have 'lat' and 'lon' coordinates.")

    if method == 'conservative':
        if HAS_XESMF:
            # Build xESMF regridder using cell corners
            # Need to provide bounds for source and target.
            # We'll derive source bounds if not present; else assume half grid spacing.
            if 'lat_b' in source_ds and 'lon_b' in source_ds:
                src_lat_b = source_ds['lat_b']
                src_lon_b = source_ds['lon_b']
            else:
                # Approximate bounds from centred coordinates
                src_lat = source_ds[src_lat_name]
                src_lon = source_ds[src_lon_name]
                dlat = np.gradient(src_lat)
                dlon = np.gradient(src_lon)
                # Ensure arrays are 1D for simplicity
                if src_lat.ndim == 1:
                    src_lat_b = np.concatenate([src_lat - dlat/2, [src_lat[-1] + dlat[-1]/2]])
                    src_lon_b = np.concatenate([src_lon - dlon/2, [src_lon[-1] + dlon[-1]/2]])
                else:
                    raise NotImplementedError("2D bounds not implemented for fallback")

            regridder = xe.Regridder(
                source_ds, target_grid, method='conservative',
                periodic=False, ignore_degenerate=True
            )
            regridded = regridder(source_ds[var_name])
            return regridded

        else:
            logger.warning("xesmf not available for conservative regridding; "
                           "falling back to bilinear (mass may not be conserved).")
            return _regrid_fallback(source_ds, target_grid, var_name, 'bilinear', src_lat_name, src_lon_name)
    else:
        return _regrid_fallback(source_ds, target_grid, var_name, method, src_lat_name, src_lon_name)


def _regrid_fallback(source_ds, target_grid, var_name, method, src_lat_name, src_lon_name):
    """
    Fallback regridding using scipy.interpolate.griddata.

    Each spatial slice is interpolated independently; any leading dimensions
    (time, init_time, lead_time, ensemble member, ...) are preserved.
    """
    src = source_ds[var_name]

    # Callers pass xesmf method names; griddata only knows linear/nearest/cubic.
    method = {'bilinear': 'linear', 'conservative': 'linear'}.get(method, method)

    lat_dims = tuple(source_ds[src_lat_name].dims)
    lon_dims = tuple(source_ds[src_lon_name].dims)
    spatial_dims = lat_dims + tuple(d for d in lon_dims if d not in lat_dims)
    extra_dims = tuple(d for d in src.dims if d not in spatial_dims)

    # lat/lon are either 1D coordinate vectors or full 2D curvilinear grids.
    lat_vals = source_ds[src_lat_name].values
    lon_vals = source_ds[src_lon_name].values
    if lat_vals.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)
    else:
        lat2d, lon2d = lat_vals, lon_vals
    src_points = np.column_stack([lon2d.ravel(), lat2d.ravel()])

    target_lon2d, target_lat2d = np.meshgrid(target_grid['lon'].values,
                                             target_grid['lat'].values)
    target_points = np.column_stack([target_lon2d.ravel(), target_lat2d.ravel()])

    extra_shape = tuple(src.sizes[d] for d in extra_dims)
    values_nd = src.transpose(*extra_dims, *spatial_dims).values
    slices = values_nd.reshape(-1, *values_nd.shape[len(extra_dims):])

    regridded = []
    for data in slices:
        valid = ~np.isnan(data)
        regrid_flat = griddata(src_points[valid.ravel()], data[valid], target_points, method=method)
        regridded.append(regrid_flat.reshape(target_lat2d.shape))

    regridded = np.stack(regridded, axis=0).reshape(extra_shape + target_lat2d.shape)

    coords = {'lat': target_grid['lat'].values, 'lon': target_grid['lon'].values}
    for dim in extra_dims:
        if dim in src.coords:
            coords[dim] = src[dim]

    return xr.DataArray(regridded, dims=extra_dims + ('lat', 'lon'),
                        coords=coords, name=var_name)


# ----------------------------------------------------------------------
# 3. Temporal aggregation
# ----------------------------------------------------------------------
def temporal_aggregate(da: xr.DataArray,
                       target_freq: str = '1H',
                       agg_method: str = 'sum') -> xr.DataArray:
    """
    Aggregate a time series to a target frequency.

    Parameters
    ----------
    da : xr.DataArray
        Input with a time dimension named 'time' (or 'step' for NWP).
    target_freq : str
        Pandas-style frequency string (e.g., '1H', '3H').
    agg_method : str
        'sum' for accumulative variables (rainfall), 'mean' for state variables.

    Returns
    -------
    xr.DataArray
        Aggregated data at the new frequency. Timestamps are left-bound.
    """
    # Ensure time is datetime
    if not np.issubdtype(da['time'].dtype, np.datetime64):
        da['time'] = pd.to_datetime(da['time'])

    # If input is already at lower frequency, we may not need aggregation
    # But we'll still resample to make uniform bins.
    resampler = da.resample(time=target_freq)
    if agg_method == 'sum':
        return resampler.sum(min_count=1)  # min_count=1 keeps NaN if no data
    elif agg_method == 'mean':
        return resampler.mean()
    else:
        raise ValueError("agg_method must be 'sum' or 'mean'")


# ----------------------------------------------------------------------
# 4. Unit normalization
# ----------------------------------------------------------------------
def normalize_rainfall_units(da: xr.DataArray,
                             source_name: str) -> xr.DataArray:
    """
    Convert rainfall variables to mm per time step.

    Recognized units: mm, mm/hr, kg/m2/s, m/s (if water equivalent).
    For rate units, we multiply by the time step duration.
    This function assumes the DataArray has a 'units' attribute; if not,
    it warns and leaves as-is.

    Parameters
    ----------
    da : xr.DataArray
        Rainfall variable.
    source_name : str
        Descriptive name for logging.

    Returns
    -------
    xr.DataArray
        Converted data in mm per timestep.
    """
    units = da.attrs.get('units', '').lower()
    if not units:
        logger.warning(f"Source '{source_name}' has no units attribute; assuming mm.")
        units = 'mm'

    # Determine conversion factor
    if units in ['mm', 'mm/timestep', 'kg/m2', 'mm/hr_mean_proxy', 'mm/day']:
        # Already accumulation per timestep or kg/m2 ≈ mm water equivalent
        factor = 1.0
    elif units in ['mm/hr', 'mm/h']:
        # Rate: multiply by hours per timestep
        # We need the timestep; assume hourly if not given
        # Use frequency from time coordinate if possible
        try:
            dt_hours = (da['time'].diff('time') / np.timedelta64(1, 'h')).median()
            factor = dt_hours
        except:
            factor = 1.0
            logger.warning(f"Cannot determine time step for {source_name}; using factor=1")
    elif units in ['kg/m2/s', 'kg m-2 s-1']:
        # Rate per second: convert to mm per timestep
        # 1 kg/m2/s = 1 mm/s (for water). Multiply by seconds in timestep.
        try:
            dt_seconds = (da['time'].diff('time') / np.timedelta64(1, 's')).median()
            factor = dt_seconds
        except:
            factor = 3600.0  # assume hourly
            logger.warning(f"Cannot determine time step for {source_name}; assuming hourly")
    else:
        # Unknown unit; raise or warn?
        logger.warning(f"Unknown rainfall units '{units}' for {source_name}. No conversion applied.")
        factor = 1.0

    if factor != 1.0:
        da = da * factor
        da.attrs['units'] = 'mm'
    return da


# ----------------------------------------------------------------------
# 5. Harmonization orchestrator
# ----------------------------------------------------------------------
def harmonize_all(satellite_ds: Optional[xr.Dataset],
                  radar_ds: Optional[xr.Dataset],
                  nwp_ds: Optional[xr.Dataset],
                  station_df: Optional[pd.DataFrame],
                  target_grid: xr.Dataset,
                  target_freq: str = '1H',
                  fill_value: Optional[float] = None) -> xr.Dataset:
    """
    Harmonize all sources onto the target grid and time step.

    Parameters
    ----------
    satellite_ds, radar_ds, nwp_ds : xr.Dataset or None
        Each dataset should contain a rainfall variable named 'rainfall'
        or the appropriate name for that source. If None, skipped.
    station_df : pd.DataFrame or None
        Station observations with columns: 'timestamp', 'lat', 'lon', 'rainfall_mm'.
        Stations are NOT gridded here; kept for validation/bias correction.
    target_grid : xr.Dataset
        Output grid from define_target_grid.
    target_freq : str
        Temporal resolution (default '1H').
    fill_value : float or None
        If specified, fill NaN cells with this value; else leave NaN.

    Returns
    -------
    xr.Dataset
        Dataset containing variables 'satellite_rainfall', 'radar_rainfall',
        'nwp_rainfall' on the identical grid and time step. Station data is
        attached as a separate DataFrame attribute if provided.
    """
    combined = xr.Dataset()
    combined.attrs['target_grid'] = target_grid.attrs.get('description', '')
    combined.attrs['target_freq'] = target_freq

    # Process satellite
    if satellite_ds is not None:
        # Assume variable name 'rainfall' or find first data var
        sat_var = 'rainfall' if 'rainfall' in satellite_ds else next(iter(satellite_ds.data_vars))
        sat_da = regrid_source(satellite_ds, target_grid, sat_var, method='conservative')
        sat_da = temporal_aggregate(sat_da, target_freq, 'sum')
        sat_da = normalize_rainfall_units(sat_da, 'satellite')
        combined['satellite_rainfall'] = sat_da

    # Process radar (MERRA-2 hourly proxy: daily mean intensity matches training pipeline)
    if radar_ds is not None:
        radar_var = 'rainfall' if 'rainfall' in radar_ds else next(iter(radar_ds.data_vars))
        radar_da = regrid_source(radar_ds, target_grid, radar_var, method='conservative')
        radar_units = str(radar_ds[radar_var].attrs.get('units', radar_ds.attrs.get('units', ''))).lower()
        radar_agg = 'mean' if target_freq.upper().endswith('D') and radar_units in ('mm/hr', 'mm/h') else 'sum'
        radar_da = temporal_aggregate(radar_da, target_freq, radar_agg)
        if radar_agg == 'mean' and radar_units in ('mm/hr', 'mm/h'):
            radar_da.attrs['units'] = 'mm/hr_mean_proxy'
        radar_da = normalize_rainfall_units(radar_da, 'radar')
        combined['radar_rainfall'] = radar_da

    # Process NWP
    if nwp_ds is not None:
        nwp_var = 'tp' if 'tp' in nwp_ds else ('rainfall' if 'rainfall' in nwp_ds else next(iter(nwp_ds.data_vars)))
        # NWP may need de-accumulation if cumulative; but deaccumulation is assumed done prior.
        # We'll assume the variable is already per timestep.
        nwp_da = regrid_source(nwp_ds, target_grid, nwp_var, method='conservative')
        # If NWP frequency is lower than target, we may need to upsample or leave as is.
        # For now, we aggregate to target if higher frequency; if lower, keep native.
        # We'll resample to target but may create NaN for missing timesteps.
        nwp_da = nwp_da.resample(time=target_freq).asfreq()  # asfreq, do not interpolate
        nwp_da = normalize_rainfall_units(nwp_da, 'nwp')
        combined['nwp_rainfall'] = nwp_da

    # Station data: we'll not grid it; instead attach as DataFrame in attrs.
    if station_df is not None and not station_df.empty:
        # Store in combined.attrs as a dict? Can't store DataFrame directly in attrs.
        # We'll save as a global attribute (pickle or string). For simplicity, we
        # just return it separately? Or attach as a coordinate? Better: store as a
        # separate variable? Since we are returning Dataset, we can store station
        # data as an auxiliary DataFrame in the attrs via pickle (not recommended).
        # Instead, we'll add a note in the docstring that station data is not gridded.
        logger.info("Station data not gridded; keep as separate DataFrame for validation.")
        combined.attrs['station_data_available'] = True

    # Clip to intersection of all variables (spatial extent)
    combined = _clip_to_intersection(combined)

    # Fill NaNs if requested
    if fill_value is not None:
        combined = combined.fillna(fill_value)
        logger.info(f"Filled NaN cells with {fill_value}.")

    return combined


def _clip_to_intersection(ds: xr.Dataset) -> xr.Dataset:
    """
    Clip all variables to the common spatial intersection.
    """
    # If any variable is missing, use others' extent
    # We'll just use min/max of lat/lon across variables that exist.
    lat_mins, lat_maxs = [], []
    lon_mins, lon_maxs = [], []
    for var in ds.data_vars:
        if 'lat' in ds[var].dims and 'lon' in ds[var].dims:
            lat_mins.append(ds[var]['lat'].min().item())
            lat_maxs.append(ds[var]['lat'].max().item())
            lon_mins.append(ds[var]['lon'].min().item())
            lon_maxs.append(ds[var]['lon'].max().item())
    if lat_mins:
        # Intersection: latest southern edge to earliest northern edge.
        lat_min = max(lat_mins)
        lat_max = min(lat_maxs)
        lon_min = max(lon_mins)
        lon_max = min(lon_maxs)
        # Select intersection
        ds = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
        logger.info(f"Clipped to intersection: lat [{lat_min:.2f}, {lat_max:.2f}], "
                    f"lon [{lon_min:.2f}, {lon_max:.2f}]")
    return ds


# ----------------------------------------------------------------------
# 6. Validation plotting
# ----------------------------------------------------------------------
def validate_alignment(ds: xr.Dataset,
                       timestep: Optional[pd.Timestamp] = None) -> None:
    """
    Plot one timestep of each variable side by side for visual check.

    Parameters
    ----------
    ds : xr.Dataset
        Harmonized dataset from harmonize_all.
    timestep : pd.Timestamp, optional
        Which time step to plot. If None, selects the first time step.
    """
    import matplotlib.pyplot as plt

    if timestep is None:
        timestep = ds['time'].values[0]

    vars_to_plot = [v for v in ds.data_vars if 'rainfall' in v]
    if not vars_to_plot:
        print("No rainfall variables found in dataset.")
        return

    fig, axes = plt.subplots(1, len(vars_to_plot), figsize=(5*len(vars_to_plot), 4))
    if len(vars_to_plot) == 1:
        axes = [axes]

    for ax, var in zip(axes, vars_to_plot):
        data = ds[var].sel(time=timestep, method='nearest')
        im = ax.pcolormesh(data['lon'], data['lat'], data, cmap='viridis')
        ax.set_title(var)
        fig.colorbar(im, ax=ax, label='mm')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
    plt.suptitle(f'Rainfall at {timestep}')
    plt.tight_layout()
    import matplotlib
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(fig)


# ----------------------------------------------------------------------
# 7. Optional: IDW interpolation for station data (if you choose to grid)
# ----------------------------------------------------------------------
def idw_interpolate(station_df: pd.DataFrame,
                    target_grid: xr.Dataset,
                    power: float = 2.0,
                    max_distance_km: float = 50.0) -> xr.DataArray:
    """
    Interpolate station data onto grid using Inverse Distance Weighting.
    This is provided for completeness but NOT used in the default pipeline
    because station data is better used for validation/bias correction.
    """
    # Convert target lat/lon to 2D
    lon2d, lat2d = np.meshgrid(target_grid['lon'].values, target_grid['lat'].values)
    target_points = np.column_stack([lon2d.ravel(), lat2d.ravel()])

    # For each timestamp, perform IDW
    timestamps = station_df['timestamp'].unique()
    results = []
    for ts in timestamps:
        df_t = station_df[station_df['timestamp'] == ts]
        if df_t.empty:
            results.append(np.full(lat2d.shape, np.nan))
            continue
        src_lats = df_t['lat'].values
        src_lons = df_t['lon'].values
        src_vals = df_t['rainfall_mm'].values

        # Compute distances in km using approximate flat earth (fine for small domain)
        # For better: use haversine (not shown for brevity)
        dist = np.sqrt((target_points[:,0][:,None] - src_lons)**2 +
                       (target_points[:,1][:,None] - src_lats)**2) * 111.0  # approx km
        # Mask out stations beyond max_distance
        weights = np.where(dist <= max_distance_km, 1.0 / (dist ** power), 0)
        # Handle zero distance
        weights = np.where(dist == 0, 1e6, weights)
        sum_weights = weights.sum(axis=1)
        # Avoid division by zero
        sum_weights[sum_weights == 0] = np.nan
        interp_vals = np.sum(weights * src_vals, axis=1) / sum_weights
        results.append(interp_vals.reshape(lat2d.shape))

    da = xr.DataArray(np.stack(results),
                      dims=('time', 'lat', 'lon'),
                      coords={'time': timestamps,
                              'lat': target_grid['lat'].values,
                              'lon': target_grid['lon'].values},
                      name='station_rainfall')
    return da