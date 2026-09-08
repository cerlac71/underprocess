#!/usr/bin/env python3
"""Download open-source training data for a configured region (2010-2019).

Sources:
  - CHIRPS v2.0 daily p25 (UCSB)           -> data/regions/{region}/satellite/
  - ERA5 hourly (Open-Meteo)               -> data/regions/{region}/nwp/
  - MERRA-2 hourly (NASA POWER)            -> data/regions/{region}/radar/
  - IMD gridded (missing years only)       -> data/gridded_rainfall/  (all-India, shared)

Run:
    python scripts/download_training_years.py --region uttar_pradesh
    python scripts/download_training_years.py --region india --years 2018 2019
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import shutil
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from config.regions import get_active_region

YEAR_START = 2010
YEAR_END = 2019
MONSOON_ONLY = True  # Jun-Sep — flood-relevant, saves ~70% space

CHIRPS_BASE = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/tifs/p25"
IMD_CANDIDATES = [
    "https://www.imdpune.gov.in/cmpg/GridDd/RF25/RF25_IND{year}_DAILY_GRIDDED_DATA/Rainfall%20Data/RF25_ind{year}_rfp25.nc",
    "https://imdpune.gov.in/cmpg/GridDd/RF25/RF25_IND{year}_DAILY_GRIDDED_DATA/Rainfall%20Data/RF25_ind{year}_rfp25.nc",
]


def _daterange(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _season_dates(year: int) -> list[date]:
    if MONSOON_ONLY:
        return list(_daterange(date(year, 6, 1), date(year, 9, 30)))
    return list(_daterange(date(year, 1, 1), date(year, 12, 31)))


def _grid_points(region) -> tuple[np.ndarray, np.ndarray]:
    lats = np.arange(region.lat_min, region.lat_max + region.grid_step * 0.1, region.grid_step)
    lons = np.arange(region.lon_min, region.lon_max + region.grid_step * 0.1, region.grid_step)
    return lats, lons


def download_chirps_year(year: int, session: requests.Session, region) -> Path:
    """Download daily CHIRPS p25 GeoTIFFs and build a clipped yearly NetCDF."""
    raw_dir = region.satellite_dir(PROJECT_ROOT) / "chirps_raw" / str(year)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_nc = region.satellite_dir(PROJECT_ROOT) / f"chirps_{region.id}_{year}.nc"

    dates = _season_dates(year)
    downloaded = 0
    for d in dates:
        fname = f"chirps-v2.0.{d.year}.{d.month:02d}.{d.day:02d}.tif.gz"
        tif_path = raw_dir / fname.replace(".gz", "")
        if tif_path.exists():
            downloaded += 1
            continue
        gz_path = raw_dir / fname
        if not gz_path.exists():
            url = f"{CHIRPS_BASE}/{year}/{fname}"
            try:
                resp = session.get(url, timeout=120)
                if resp.status_code == 404:
                    logger.warning("CHIRPS missing: %s", url)
                    continue
                resp.raise_for_status()
                gz_path.write_bytes(resp.content)
            except Exception as exc:
                logger.warning("CHIRPS download failed %s: %s", url, exc)
                continue
        try:
            with gzip.open(gz_path, "rb") as src, open(tif_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
        except Exception as exc:
            logger.warning("CHIRPS decompress failed %s: %s", gz_path, exc)
            continue
        downloaded += 1
        time.sleep(0.05)

    if downloaded == 0:
        raise RuntimeError(f"No CHIRPS files for {year}")

    logger.info("Building CHIRPS NetCDF for %s (%d days)...", year, downloaded)
    import rasterio
    from rasterio.windows import from_bounds

    bounds = (region.lon_min, region.lat_min, region.lon_max, region.lat_max)
    times, stacks = [], []
    for d in dates:
        tif_path = raw_dir / f"chirps-v2.0.{d.year}.{d.month:02d}.{d.day:02d}.tif"
        if not tif_path.exists():
            continue
        with rasterio.open(tif_path) as src:
            window = from_bounds(*bounds, transform=src.transform)
            data = src.read(1, window=window).astype(np.float32)
            nodata = src.nodata if src.nodata is not None else -9999
            data[data == nodata] = np.nan
            transform = src.window_transform(window)
            lons = np.linspace(transform.c, transform.c + transform.a * data.shape[1], data.shape[1])
            lats = np.linspace(transform.f, transform.f + transform.e * data.shape[0], data.shape[0])
        stacks.append(data)
        times.append(pd.Timestamp(d))

    if not stacks:
        raise RuntimeError(f"Could not stack CHIRPS for {year}")

    da = xr.DataArray(
        np.stack(stacks),
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": lats, "lon": lons},
        name="rainfall",
    )
    da.attrs["units"] = "mm/day"
    da.attrs["source"] = "CHIRPS v2.0 p25"
    da.to_dataset().to_netcdf(out_nc)
    logger.info("Saved %s (%.1f MB)", out_nc, out_nc.stat().st_size / 1e6)
    return out_nc


def download_era5_year(year: int, session: requests.Session, region) -> Path:
    out_nc = region.nwp_dir(PROJECT_ROOT) / f"era5_{region.id}_{year}.nc"
    if out_nc.exists() and out_nc.stat().st_size > 1000:
        logger.info("Skipping ERA5 %s (exists)", year)
        return out_nc

    lats, lons = _grid_points(region)
    start = f"{year}-06-01" if MONSOON_ONLY else f"{year}-01-01"
    end = f"{year}-09-30" if MONSOON_ONLY else f"{year}-12-31"
    frames = []
    for lat in lats:
        for lon in lons:
            resp = session.get(
                "https://archive-api.open-meteo.com/v1/archive",
                params={
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "start_date": start,
                    "end_date": end,
                    "hourly": "precipitation",
                    "timezone": "UTC",
                },
                timeout=120,
            )
            resp.raise_for_status()
            hourly = resp.json()["hourly"]
            frames.append(pd.DataFrame({
                "time": pd.to_datetime(hourly["time"]).tz_localize(None),
                "lat": lat, "lon": lon,
                "tp": hourly["precipitation"],
            }))
            time.sleep(0.12)

    table = pd.concat(frames, ignore_index=True)
    ds = table.set_index(["time", "lat", "lon"])["tp"].to_xarray().rename("tp").sortby(["time", "lat", "lon"])
    ds.attrs["source"] = "ECMWF ERA5 via Open-Meteo"
    ds.attrs["units"] = "mm"
    out_nc.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_nc)
    logger.info("Saved %s", out_nc)
    return out_nc


def download_merra2_year(year: int, session: requests.Session, region) -> Path:
    out_nc = region.radar_dir(PROJECT_ROOT) / f"merra2_{region.id}_{year}.nc"
    if out_nc.exists() and out_nc.stat().st_size > 1000:
        logger.info("Skipping MERRA-2 %s (exists)", year)
        return out_nc

    lats, lons = _grid_points(region)
    start = date(year, 6, 1) if MONSOON_ONLY else date(year, 1, 1)
    end = date(year, 9, 30) if MONSOON_ONLY else date(year, 12, 31)
    frames = []
    for lat in lats:
        for lon in lons:
            resp = session.get(
                "https://power.larc.nasa.gov/api/temporal/hourly/point",
                params={
                    "parameters": "PRECTOTCORR",
                    "community": "AG",
                    "longitude": float(lon),
                    "latitude": float(lat),
                    "start": start.strftime("%Y%m%d"),
                    "end": end.strftime("%Y%m%d"),
                    "format": "JSON",
                },
                timeout=120,
            )
            resp.raise_for_status()
            values = resp.json()["properties"]["parameter"]["PRECTOTCORR"]
            for key, value in values.items():
                ts = datetime.strptime(key, "%Y%m%d%H")
                frames.append({"time": pd.Timestamp(ts), "lat": lat, "lon": lon, "rainfall": value})
            time.sleep(0.35)

    table = pd.DataFrame(frames)
    ds = table.set_index(["time", "lat", "lon"])["rainfall"].to_xarray().sortby(["time", "lat", "lon"])
    ds.attrs["source"] = "NASA POWER MERRA-2"
    ds.attrs["units"] = "mm/hr"
    out_nc.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_nc)
    logger.info("Saved %s", out_nc)
    return out_nc


def download_missing_imd(years: list[int], session: requests.Session) -> list[int]:
    imd_dir = DATA / "gridded_rainfall"
    imd_dir.mkdir(parents=True, exist_ok=True)
    fetched = []
    for year in years:
        dest = imd_dir / f"RF25_ind{year}_rfp25.nc"
        if dest.exists():
            continue
        for template in IMD_CANDIDATES:
            url = template.format(year=year)
            try:
                resp = session.get(url, timeout=120, stream=True)
                if resp.status_code != 200:
                    continue
                with dest.open("wb") as handle:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        handle.write(chunk)
                logger.info("Downloaded IMD %s", dest.name)
                fetched.append(year)
                break
            except Exception as exc:
                logger.warning("IMD %s failed: %s", year, exc)
        else:
            logger.warning("IMD %s not available from open mirrors — skip", year)
    return fetched


def try_imerg_year(year: int, session: requests.Session) -> Path | None:
    """Try NASA GPM IMERG daily; returns None if auth/block."""
    out_nc = DATA / "satellite" / "imerg" / f"imerg_daily_{year}.nc"
    if out_nc.exists():
        return out_nc
    # IMERG Final daily requires Earthdata — probe one file
    sample = (
        f"https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/IMERGDF06/{year}/06/"
        f"3B-DAY.MS.MRG.3IMERG.{year}0601-S000000-E235959.V07B.HDF5"
    )
    try:
        resp = session.head(sample, timeout=30)
        if resp.status_code in (401, 403, 404):
            logger.warning("GPM IMERG requires NASA Earthdata login — skipping")
            return None
    except Exception:
        logger.warning("GPM IMERG unreachable — skipping")
        return None
    return None


def merge_chirps(years: list[int], region) -> Path:
    paths = [region.satellite_dir(PROJECT_ROOT) / f"chirps_{region.id}_{y}.nc" for y in years]
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise FileNotFoundError("No yearly CHIRPS files to merge")
    datasets = [xr.open_dataset(p)["rainfall"] for p in paths]
    merged = xr.concat(datasets, dim="time").sortby("time").to_dataset(name="rainfall")
    out = region.satellite_dir(PROJECT_ROOT) / f"chirps_{region.id}_{years[0]}0101_{years[-1]}1231.nc"
    merged.to_netcdf(out)
    for ds in datasets:
        ds.close()
    logger.info("Merged CHIRPS -> %s (%.1f MB)", out, out.stat().st_size / 1e6)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default=None, help="Region id: belagavi, uttar_pradesh, india")
    parser.add_argument("--years", nargs="*", type=int, default=list(range(YEAR_START, YEAR_END + 1)))
    args = parser.parse_args()
    region = get_active_region(args.region)
    years = sorted(args.years)
    session = requests.Session()
    manifest = {
        "region": region.id,
        "region_name": region.name,
        "bbox": region.bbox,
        "grid_cells": region.approx_grid_cells,
        "years": years,
        "monsoon_only": MONSOON_ONLY,
        "downloaded": {},
        "skipped": [],
    }

    logger.info(
        "Downloading training data for %s (%d grid cells, bbox %s)",
        region.name, region.approx_grid_cells, region.bbox,
    )

    for sub in (region.satellite_dir(PROJECT_ROOT), region.nwp_dir(PROJECT_ROOT), region.radar_dir(PROJECT_ROOT)):
        sub.mkdir(parents=True, exist_ok=True)

    # Missing IMD years in 2010-2019 window
    existing_imd = {int(p.stem.replace("RF25_ind", "").replace("_rfp25", ""))
                    for p in (DATA / "gridded_rainfall").glob("RF25_ind*_rfp25.nc")}
    imd_needed = [y for y in years if y not in existing_imd]
    if imd_needed:
        manifest["imd_fetched"] = download_missing_imd(imd_needed, session)

    for year in years:
        entry = {}
        try:
            entry["chirps"] = str(download_chirps_year(year, session, region).relative_to(PROJECT_ROOT))
        except Exception as exc:
            logger.error("CHIRPS %s: %s", year, exc)
            entry["chirps_error"] = str(exc)
        try:
            entry["era5"] = str(download_era5_year(year, session, region).relative_to(PROJECT_ROOT))
        except Exception as exc:
            logger.error("ERA5 %s: %s", year, exc)
            entry["era5_error"] = str(exc)
        try:
            entry["merra2"] = str(download_merra2_year(year, session, region).relative_to(PROJECT_ROOT))
        except Exception as exc:
            logger.error("MERRA-2 %s: %s", year, exc)
            entry["merra2_error"] = str(exc)
        manifest["downloaded"][str(year)] = entry

    try_imerg_year(years[0], session)  # probe once

    try:
        manifest["chirps_merged"] = str(merge_chirps(years, region).relative_to(PROJECT_ROOT))
    except Exception as exc:
        manifest["chirps_merge_error"] = str(exc)

    manifest["finished_at"] = datetime.utcnow().isoformat() + "Z"
    out = region.data_root(PROJECT_ROOT) / "training_manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.info("Done. Manifest -> %s", out)


if __name__ == "__main__":
    main()
