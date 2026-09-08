#!/usr/bin/env python3
"""
IMD gridded rainfall ingestion — SIH26071.

Reads IMD 0.25° daily NetCDF files (RF25_indYYYY_rfp25.nc, 1901–present)
and returns xarray Datasets with dims (time, lat, lon) and a 'rainfall'
variable in mm/day, ready for harmonisation/pipeline consumption.

Raw files store missing cells as -999.0; converted to NaN here.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

IMD_MISSING = -999.0


def find_imd_file(year: int, data_dir) -> Path:
    """Locate the yearly IMD file for a given year."""
    path = Path(data_dir) / f"RF25_ind{year}_rfp25.nc"
    if not path.exists():
        raise FileNotFoundError(f"IMD file not found: {path.resolve()}")
    return path


def discover_training_years(
    data_dir=None,
    year_start: int = 2010,
    year_end: int = 2019,
    region_id: str | None = None,
) -> list[int]:
    """Return years where IMD + regional CHIRPS/ERA5/MERRA-2 yearly files all exist."""
    from config.regions import get_active_region

    region = get_active_region(region_id)
    imd_dir = region.imd_dir()
    sat_dir = region.satellite_dir()
    nwp_dir = region.nwp_dir()
    radar_dir = region.radar_dir()

    years = []
    for year in range(year_start, year_end + 1):
        if not (imd_dir / f"RF25_ind{year}_rfp25.nc").exists():
            continue
        if not (sat_dir / f"chirps_{region.id}_{year}.nc").exists():
            # Legacy Belagavi naming
            legacy = Path(data_dir or imd_dir.parent) / "satellite" / f"chirps_belagavi_{year}.nc"
            if region.id == "belagavi" and legacy.exists():
                pass
            else:
                continue
        era5 = nwp_dir / f"era5_{region.id}_{year}.nc"
        legacy_era5 = (imd_dir.parent / "nwp" / f"era5_belagavi_{year}.nc")
        if not era5.exists() and not (region.id == "belagavi" and legacy_era5.exists()):
            continue
        merra = radar_dir / f"merra2_{region.id}_{year}.nc"
        legacy_merra = imd_dir.parent / "radar" / f"merra2_belagavi_{year}.nc"
        if not merra.exists() and not (region.id == "belagavi" and legacy_merra.exists()):
            continue
        years.append(year)
    return years


def load_imd_rainfall_multi_year(
    years: list[int],
    data_dir,
    bbox: dict,
    monsoon_only: bool = True,
) -> xr.Dataset:
    """Load and concatenate IMD daily rainfall for multiple years over a bbox."""
    if not years:
        raise ValueError("No years provided for IMD multi-year load.")
    parts = []
    for year in sorted(years):
        start = f"{year}-06-01" if monsoon_only else f"{year}-01-01"
        end = f"{year}-09-30" if monsoon_only else f"{year}-12-31"
        parts.append(load_imd_rainfall(year, data_dir, bbox, start, end)["rainfall"])
    merged = xr.concat(parts, dim="time").sortby("time")
    return xr.Dataset({"rainfall": merged})


def load_imd_rainfall(
    year: int,
    data_dir,
    bbox: dict,
    start_date: str,
    end_date: str,
) -> xr.Dataset:
    """
    Load IMD daily rainfall for a bounding box and date range.

    Parameters
    ----------
    year : int
        Year of the file to open.
    data_dir : str or Path
        Directory containing RF25_indYYYY_rfp25.nc files.
    bbox : dict
        Keys lat_min, lat_max, lon_min, lon_max (degrees, ascending).
    start_date, end_date : str
        ISO date strings (inclusive).

    Returns
    -------
    xr.Dataset
        Dataset with 'rainfall' (time, lat, lon) in mm/day; missing = NaN.
    """
    with xr.open_dataset(find_imd_file(year, data_dir)) as raw:
        ds = raw.rename({"TIME": "time", "LATITUDE": "lat", "LONGITUDE": "lon"})
        sub = ds.sel(
            time=slice(start_date, end_date),
            lat=slice(bbox["lat_min"], bbox["lat_max"]),
            lon=slice(bbox["lon_min"], bbox["lon_max"]),
        ).load()

    da = sub["RAINFALL"].rename("rainfall").where(sub["RAINFALL"] > -100.0)
    da.attrs["units"] = "mm"
    return xr.Dataset({"rainfall": da})


def imd_to_station_df(ds: xr.Dataset, n_points: int = 4, seed: int = 42) -> pd.DataFrame:
    """
    Sample the gridded analysis at fixed points to emulate station series.

    Returns a DataFrame with columns timestamp, lat, lon, rainfall_mm —
    one row per (day, point).
    """
    rng = np.random.default_rng(seed)
    lats = ds["lat"].values
    lons = ds["lon"].values
    picks = list(
        zip(
            rng.choice(lats, min(n_points, lats.size), replace=False),
            rng.choice(lons, min(n_points, lons.size), replace=False),
        )
    )

    rows = []
    for t in ds["time"].values:
        for la, lo in picks:
            val = float(ds["rainfall"].sel(time=t, lat=la, lon=lo, method="nearest"))
            rows.append(
                {"timestamp": pd.Timestamp(t), "lat": la, "lon": lo, "rainfall_mm": val}
            )
    return pd.DataFrame(rows)
