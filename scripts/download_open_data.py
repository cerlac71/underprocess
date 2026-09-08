#!/usr/bin/env python3
"""Download open-source datasets for the Belagavi Aug 2019 validation window.

Sources (all documented in data/README.md):
  - ERA5 hourly precipitation (Open-Meteo archive) -> data/nwp/
  - NASA POWER MERRA-2 hourly precipitation (grid) -> data/radar/
  - ESA WorldCover 10 m land cover (clipped tile) -> data/landcover/
  - ISRIC SoilGrids texture (sampled grid) -> data/soil/

Run from the project root:
    python scripts/download_open_data.py
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data"

START_DATE = "2019-07-20"
END_DATE = "2019-08-08"
LAT_MIN, LAT_MAX = 14.5, 15.5
LON_MIN, LON_MAX = 74.5, 75.5
GRID_STEP = 0.25  # degrees; matches IMD/CHIRPS resolution

WORLDCOVER_TILE_URL = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
    "ESA_WorldCover_10m_2021_v200_N12E075_Map.tif"
)

# SCS curve numbers by ESA WorldCover class (agriculture/forest/urban defaults).
WORLDCOVER_CN = {
    10: 77,  # Tree cover
    20: 61,  # Shrubland
    30: 72,  # Grassland
    40: 82,  # Cropland
    50: 98,  # Built-up
    60: 98,  # Bare / sparse vegetation
    70: 100,  # Snow and ice
    80: 100,  # Permanent water bodies
    90: 100,  # Herbaceous wetland
    95: 100,  # Mangroves
    100: 100,  # Moss and lichen
}


def _grid_points() -> tuple[np.ndarray, np.ndarray]:
    lats = np.arange(LAT_MIN, LAT_MAX + GRID_STEP * 0.1, GRID_STEP)
    lons = np.arange(LON_MIN, LON_MAX + GRID_STEP * 0.1, GRID_STEP)
    return lats, lons


def download_era5_nwp(out_path: Path) -> None:
    """ERA5 hourly precipitation from the Open-Meteo historical archive."""
    lats, lons = _grid_points()
    frames = []
    session = requests.Session()
    for lat in lats:
        for lon in lons:
            logger.info("ERA5 Open-Meteo %.2f, %.2f", lat, lon)
            response = session.get(
                "https://archive-api.open-meteo.com/v1/archive",
                params={
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "start_date": START_DATE,
                    "end_date": END_DATE,
                    "hourly": "precipitation",
                    "timezone": "UTC",
                },
                timeout=120,
            )
            response.raise_for_status()
            payload = response.json()
            hourly = payload["hourly"]
            df = pd.DataFrame(
                {
                    "time": pd.to_datetime(hourly["time"]).tz_localize(None),
                    "lat": lat,
                    "lon": lon,
                    "tp": hourly["precipitation"],
                }
            )
            frames.append(df)
            time.sleep(0.15)

    table = pd.concat(frames, ignore_index=True)
    table["tp"] = table["tp"].astype(np.float32)
    ds = (
        table.set_index(["time", "lat", "lon"])["tp"]
        .to_xarray()
        .rename("tp")
        .sortby(["time", "lat", "lon"])
    )
    ds.attrs.update(
        {
            "source": "ECMWF ERA5 via Open-Meteo Historical Archive",
            "url": "https://open-meteo.com/en/docs/historical-weather-api",
            "units": "mm",
            "init_time": f"{START_DATE}T00:00:00Z",
            "description": "Hourly accumulated precipitation; used as NWP/reanalysis input.",
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path)
    logger.info("Saved ERA5 NWP -> %s", out_path)


def _power_hourly(lat: float, lon: float, session: requests.Session) -> pd.DataFrame:
    response = session.get(
        "https://power.larc.nasa.gov/api/temporal/hourly/point",
        params={
            "parameters": "PRECTOTCORR",
            "community": "AG",
            "longitude": float(lon),
            "latitude": float(lat),
            "start": START_DATE.replace("-", ""),
            "end": END_DATE.replace("-", ""),
            "format": "JSON",
        },
        timeout=120,
    )
    response.raise_for_status()
    values = response.json()["properties"]["parameter"]["PRECTOTCORR"]
    rows = []
    for key, value in values.items():
        ts = datetime.strptime(key, "%Y%m%d%H")
        rows.append({"time": pd.Timestamp(ts), "lat": lat, "lon": lon, "rainfall": value})
    return pd.DataFrame(rows)


def download_merra2_radar_proxy(out_path: Path) -> None:
    """NASA POWER MERRA-2 hourly precipitation on a lat/lon grid.

    This is an open, gauge-corrected reanalysis product. It is stored under
    data/radar/ as a high-temporal-resolution rainfall estimate until IMD
    Doppler radar archives are available.
    """
    lats, lons = _grid_points()
    frames = []
    session = requests.Session()
    for lat in lats:
        for lon in lons:
            logger.info("NASA POWER MERRA-2 %.2f, %.2f", lat, lon)
            frames.append(_power_hourly(lat, lon, session))
            time.sleep(0.35)

    table = pd.concat(frames, ignore_index=True)
    table["rainfall"] = table["rainfall"].astype(np.float32)
    ds = (
        table.set_index(["time", "lat", "lon"])["rainfall"]
        .to_xarray()
        .rename("rainfall")
        .sortby(["time", "lat", "lon"])
    )
    ds.attrs.update(
        {
            "source": "NASA POWER / MERRA-2 PRECTOTCORR",
            "url": "https://power.larc.nasa.gov/",
            "units": "mm/hr",
            "description": (
                "Hourly precipitation from NASA POWER (MERRA-2). "
                "Open proxy for radar QPE until IMD Doppler data is integrated."
            ),
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path)
    logger.info("Saved MERRA-2 radar proxy -> %s", out_path)


def download_worldcover_clip(out_path: Path) -> None:
    """Download the public ESA WorldCover tile and clip to the study bbox."""
    import rasterio
    from rasterio.windows import from_bounds

    raw_path = DATA / "landcover" / "worldcover_N12E075_raw.tif"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if not raw_path.exists():
        logger.info("Downloading ESA WorldCover tile (this may take a few minutes)...")
        with requests.get(WORLDCOVER_TILE_URL, stream=True, timeout=600) as response:
            response.raise_for_status()
            with raw_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)

    bounds = (LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)
    with rasterio.open(raw_path) as source:
        window = from_bounds(*bounds, transform=source.transform)
        data = source.read(1, window=window)
        transform = source.window_transform(window)
        profile = source.profile.copy()
        profile.update(
            {
                "height": data.shape[0],
                "width": data.shape[1],
                "transform": transform,
                "compress": "deflate",
            }
        )
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data, 1)
            dst.update_tags(
                source="ESA WorldCover 10 m v200 (2021)",
                url="https://esa-worldcover.org/",
                note="Clipped to Belagavi study bbox for SCS curve-number mapping.",
            )
    logger.info("Saved clipped WorldCover -> %s", out_path)


def _soil_texture(lat: float, lon: float, session: requests.Session) -> dict:
    out = {}
    for prop in ("sand", "clay", "silt"):
        for attempt in range(3):
            response = session.get(
                "https://rest.isric.org/soilgrids/v2.0/properties/query",
                params={
                    "lon": float(lon),
                    "lat": float(lat),
                    "property": prop,
                    "depth": "0-5cm",
                    "value": "mean",
                },
                timeout=120,
            )
            if response.status_code in (429, 500):
                time.sleep(2.0 * (attempt + 1))
                continue
            response.raise_for_status()
            out[prop] = response.json()["properties"]["layers"][0]["depths"][0]["values"]["mean"]
            break
        else:
            out[prop] = {"sand": 450, "clay": 250, "silt": 300}[prop]
    return out


def _cn_from_texture(sand: float, clay: float, silt: float, land_cover: int) -> float:
    base = WORLDCOVER_CN.get(int(land_cover), 79)
    # SoilGrids reports sand/clay/silt in g/kg (0-1000 scale).
    sand_pct = sand / 10.0
    clay_pct = clay / 10.0
    if sand_pct >= 50:
        return max(55, base - 8)
    if clay_pct >= 35:
        return min(95, base + 6)
    return float(base)


def build_cn_from_worldcover(worldcover_path: Path, out_path: Path, step_deg: float = 0.05) -> None:
    """Derive SCS curve number from ESA WorldCover (no external API calls)."""
    import rasterio
    from rasterio.transform import from_origin

    lats = np.arange(LAT_MIN, LAT_MAX, step_deg)
    lons = np.arange(LON_MIN, LON_MAX, step_deg)
    height, width = len(lats), len(lons)
    cn = np.full((height, width), np.nan, dtype=np.float32)
    transform = from_origin(LON_MIN, LAT_MAX, step_deg, step_deg)

    with rasterio.open(worldcover_path) as lc:
        base_profile = lc.profile.copy()
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                try:
                    row, col = lc.index(lon, lat)
                    land_cover = int(lc.read(1, window=((row, row + 1), (col, col + 1)))[0, 0])
                except Exception:
                    land_cover = 40
                cn[i, j] = WORLDCOVER_CN.get(int(land_cover), 79)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = base_profile.copy()
    profile.update(
        {
            "height": height,
            "width": width,
            "count": 1,
            "dtype": "float32",
            "transform": transform,
            "compress": "deflate",
        }
    )
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(cn, 1)
        dst.update_tags(
            source="ESA WorldCover 10 m v200 (2021)",
            url="https://esa-worldcover.org/",
            units="curve_number",
        )
    logger.info("Saved WorldCover CN raster -> %s", out_path)


def build_soil_cn_raster(worldcover_path: Path, out_path: Path, step_deg: float = 0.25) -> None:
    """Sample SoilGrids on a coarse grid and combine with WorldCover for CN."""
    import rasterio
    from rasterio.transform import from_origin

    lats = np.arange(LAT_MIN, LAT_MAX, step_deg)
    lons = np.arange(LON_MIN, LON_MAX, step_deg)
    height, width = len(lats), len(lons)
    cn = np.full((height, width), np.nan, dtype=np.float32)
    transform = from_origin(LON_MIN, LAT_MAX, step_deg, step_deg)

    with rasterio.open(worldcover_path) as lc:
        session = requests.Session()
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                try:
                    row, col = lc.index(lon, lat)
                    land_cover = int(lc.read(1, window=((row, row + 1), (col, col + 1)))[0, 0])
                except Exception:
                    land_cover = 40
                texture = _soil_texture(lat, lon, session)
                cn[i, j] = _cn_from_texture(
                    texture["sand"], texture["clay"], texture["silt"], land_cover
                )
                time.sleep(0.2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        compress="deflate",
    ) as dst:
        dst.write(cn, 1)
        dst.update_tags(
            source="ISRIC SoilGrids v2 + ESA WorldCover",
            url="https://soilgrids.org/",
            units="curve_number",
        )
    logger.info("Saved soil-derived CN raster -> %s", out_path)


def write_manifest(paths: dict[str, Path]) -> None:
    manifest = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "period": {"start": START_DATE, "end": END_DATE},
        "bbox": {"lat_min": LAT_MIN, "lat_max": LAT_MAX, "lon_min": LON_MIN, "lon_max": LON_MAX},
        "files": {key: str(Path(path).resolve().relative_to(PROJECT_ROOT)) for key, path in paths.items()},
    }
    manifest_path = DATA / "open_data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote manifest -> %s", manifest_path)


def main() -> None:
    paths = {
        "nwp_era5": DATA / "nwp" / "era5_belagavi_20190720_20190808.nc",
        "radar_merra2": DATA / "radar" / "merra2_power_belagavi_20190720_20190808.nc",
        "landcover": DATA / "landcover" / "worldcover_belagavi.tif",
        "soil_cn": DATA / "soil" / "cn_belagavi.tif",
    }

    if not paths["nwp_era5"].exists():
        download_era5_nwp(paths["nwp_era5"])
    else:
        logger.info("Skipping ERA5; file exists.")

    if not paths["radar_merra2"].exists():
        download_merra2_radar_proxy(paths["radar_merra2"])
    else:
        logger.info("Skipping MERRA-2; file exists.")

    if not paths["landcover"].exists():
        download_worldcover_clip(paths["landcover"])
    else:
        logger.info("Skipping WorldCover; file exists.")

    if not paths["soil_cn"].exists():
        try:
            build_soil_cn_raster(paths["landcover"], paths["soil_cn"])
        except Exception as exc:
            logger.warning("SoilGrids unavailable (%s); using WorldCover CN only.", exc)
            build_cn_from_worldcover(paths["landcover"], paths["soil_cn"])
    else:
        logger.info("Skipping soil CN; file exists.")

    write_manifest(paths)
    logger.info("Open-data download complete.")


if __name__ == "__main__":
    main()
