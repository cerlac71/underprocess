#!/usr/bin/env python3
"""Download SRTM DEM and derive slope/TWI/flow accumulation for a region.

Uses NASA SRTM 30 m tiles (AWS Open Data, no API key). Outputs GeoTIFFs under
data/regions/{region_id}/terrain/.

Example:
    export SIH_REGION=uttar_pradesh
    python scripts/download_region_terrain.py --region uttar_pradesh
"""

from __future__ import annotations

import argparse
import gzip
import logging
import os
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.transform import from_origin
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config.regions import get_active_region

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

WGS84_CRS = '+proj=longlat +datum=WGS84 +no_defs'
SRTM_BASE = "https://elevation-tiles-prod.s3.amazonaws.com/skadi"
OPEN_METEO_ELEVATION = "https://api.open-meteo.com/v1/elevation"
OUTPUT_DEG = 0.25  # matches IMD/model grid; keeps API calls manageable
BATCH_SIZE = 20


def _tile_path(lat_index: int, lon_index: int) -> str:
    ns = "N" if lat_index >= 0 else "S"
    ew = "E" if lon_index >= 0 else "W"
    return f"{ns}{abs(lat_index):02d}/{ew}{abs(lon_index):03d}.hgt.gz"


def _download_tile(lat_index: int, lon_index: int, cache_dir: Path) -> Path | None:
    import requests

    rel = _tile_path(lat_index, lon_index)
    out = cache_dir / rel.replace("/", "_")
    if out.exists() and out.stat().st_size > 1000:
        return out
    url = f"{SRTM_BASE}/{rel}"
    logger.info("Downloading %s", url)
    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
    except Exception as exc:
        logger.warning("Tile unavailable %s: %s", rel, exc)
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(response.content)
    return out


def _read_hgt_gz(path: Path, lat_index: int, lon_index: int) -> tuple[np.ndarray, rasterio.Affine]:
    with gzip.open(path, "rb") as handle:
        raw = handle.read()
    side = int(np.sqrt(len(raw) / 2))
    dem = np.frombuffer(raw, dtype=">i2").reshape(side, side).astype(np.float32)
    dem[dem <= -32768] = np.nan
    west = float(lon_index)
    north = float(lat_index + 1)
    transform = from_origin(west, north, 1.0 / (side - 1), 1.0 / (side - 1))
    return dem, transform


def _fetch_elevations_open_meteo(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Fetch elevations for a lat/lon grid using the free Open-Meteo API."""
    import requests
    import time

    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    flat_lats = lat_grid.ravel()
    flat_lons = lon_grid.ravel()
    elevations = np.full(flat_lats.shape, np.nan, dtype=np.float32)

    session = requests.Session()
    for start in range(0, flat_lats.size, BATCH_SIZE):
        end = min(start + BATCH_SIZE, flat_lats.size)
        params = {
            "latitude": ",".join(f"{v:.4f}" for v in flat_lats[start:end]),
            "longitude": ",".join(f"{v:.4f}" for v in flat_lons[start:end]),
        }
        for attempt in range(6):
            response = session.get(OPEN_METEO_ELEVATION, params=params, timeout=120)
            if response.status_code == 429:
                wait = 2 ** attempt
                logger.warning("Rate limited; retrying in %ss", wait)
                time.sleep(wait)
                continue
            response.raise_for_status()
            break
        else:
            response.raise_for_status()
        elevations[start:end] = np.asarray(response.json()["elevation"], dtype=np.float32)
        logger.info("Fetched elevations %d/%d", end, flat_lats.size)
        time.sleep(1.0)

    return elevations.reshape(lat_grid.shape)


def _build_dem_from_open_meteo(bbox: dict, step_deg: float) -> tuple[np.ndarray, rasterio.Affine]:
    lats = np.arange(bbox["lat_min"], bbox["lat_max"] + step_deg * 0.5, step_deg)
    lons = np.arange(bbox["lon_min"], bbox["lon_max"] + step_deg * 0.5, step_deg)
    dem = _fetch_elevations_open_meteo(lats, lons)
    transform = from_origin(bbox["lon_min"], bbox["lat_max"], step_deg, step_deg)
    return dem.astype(np.float32), transform


def _download_and_merge(bbox: dict, cache_dir: Path) -> tuple[np.ndarray, rasterio.Affine]:
    import requests

    lat_min = int(np.floor(bbox["lat_min"]))
    lat_max = int(np.floor(bbox["lat_max"]))
    lon_min = int(np.floor(bbox["lon_min"]))
    lon_max = int(np.floor(bbox["lon_max"]))

    datasets = []
    session = requests.Session()
    for lat in range(lat_min, lat_max + 1):
        for lon in range(lon_min, lon_max + 1):
            rel = _tile_path(lat, lon)
            out = cache_dir / rel.replace("/", "_")
            if not out.exists():
                url = f"{SRTM_BASE}/{rel}"
                logger.info("Downloading %s", url)
                try:
                    response = session.get(url, timeout=120)
                    response.raise_for_status()
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(response.content)
                except Exception as exc:
                    logger.warning("Skipping tile %s: %s", rel, exc)
                    continue
            dem, transform = _read_hgt_gz(out, lat, lon)
            profile = {
                "driver": "GTiff",
                "height": dem.shape[0],
                "width": dem.shape[1],
                "count": 1,
                "dtype": "float32",
                "crs": WGS84_CRS,
                "transform": transform,
                "nodata": np.nan,
            }
            mem = rasterio.io.MemoryFile()
            with mem.open(**profile) as dst:
                dst.write(dem, 1)
                datasets.append(mem.open())

    if not datasets:
        raise FileNotFoundError("No SRTM tiles downloaded for this bbox.")

    mosaic, out_transform = merge(datasets)
    dem = mosaic[0].astype(np.float32)
    dem[dem <= -32768] = np.nan
    for ds in datasets:
        ds.close()
    return dem, out_transform


def _resample_to_grid(
    dem: np.ndarray, src_transform: rasterio.Affine, bbox: dict, step_deg: float
) -> tuple[np.ndarray, rasterio.Affine]:
    lats = np.arange(bbox["lat_min"], bbox["lat_max"] + step_deg * 0.5, step_deg)
    lons = np.arange(bbox["lon_min"], bbox["lon_max"] + step_deg * 0.5, step_deg)
    height, width = len(lats), len(lons)
    out = np.full((height, width), np.nan, dtype=np.float32)
    out_transform = from_origin(bbox["lon_min"], bbox["lat_max"], step_deg, step_deg)

    with rasterio.io.MemoryFile() as mem:
        profile = {
            "driver": "GTiff",
            "height": dem.shape[0],
            "width": dem.shape[1],
            "count": 1,
            "dtype": "float32",
            "crs": WGS84_CRS,
            "transform": src_transform,
            "nodata": np.nan,
        }
        with mem.open(**profile) as src:
            src.write(dem, 1)
            window = rasterio.windows.from_bounds(
                bbox["lon_min"], bbox["lat_min"], bbox["lon_max"], bbox["lat_max"], out_transform
            )
            data = src.read(
                1,
                window=window,
                out_shape=(height, width),
                resampling=Resampling.bilinear,
                masked=True,
            )
            out = np.asarray(data.filled(np.nan), dtype=np.float32)

    return out, out_transform


def _fill_nans(dem: np.ndarray) -> np.ndarray:
    filled = dem.copy()
    mask = ~np.isfinite(filled)
    if not mask.any():
        return filled
    coords = ndimage.distance_transform_edt(mask, return_distances=False, return_indices=True)
    filled[mask] = filled[tuple(coords[:, mask])]
    return filled


def _pixel_size_meters(lat: float, step_deg: float) -> tuple[float, float]:
    lat_m = step_deg * 111_320.0
    lon_m = step_deg * 111_320.0 * np.cos(np.radians(lat))
    return lon_m, lat_m


def _compute_slope(dem: np.ndarray, step_deg: float, lat_center: float) -> np.ndarray:
    lon_m, lat_m = _pixel_size_meters(lat_center, step_deg)
    dzdx = ndimage.sobel(dem, axis=1) / (8.0 * lon_m)
    dzdy = ndimage.sobel(dem, axis=0) / (8.0 * lat_m)
    return np.degrees(np.arctan(np.hypot(dzdx, dzdy))).astype(np.float32)


def _compute_flow_accum(dem: np.ndarray) -> np.ndarray:
    """Simple D8 flow accumulation on a coarsened DEM."""
    filled = _fill_nans(dem)
    rows, cols = filled.shape
    order = np.argsort(filled.ravel())[::-1]
    accum = np.ones_like(filled, dtype=np.float32)
    offsets = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

    for flat_idx in order:
        r, c = divmod(int(flat_idx), cols)
        center = filled[r, c]
        best = None
        best_drop = 0.0
        for dr, dc in offsets:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                drop = center - filled[nr, nc]
                if drop > best_drop:
                    best_drop = drop
                    best = (nr, nc)
        if best is not None:
            accum[best] += accum[r, c]
    return accum


def _compute_twi(flow_accum: np.ndarray, slope_deg: np.ndarray) -> np.ndarray:
    slope_rad = np.radians(np.clip(slope_deg, 0.1, 89.9))
    return np.log(flow_accum + 1.0) - np.log(np.tan(slope_rad) + 1e-6)


def _write_geotiff(path: Path, array: np.ndarray, transform: rasterio.Affine) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": WGS84_CRS,
        "transform": transform,
        "compress": "deflate",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(np.float32), 1)
        dst.update_tags(source="Open-Meteo Copernicus DEM 90 m / SRTM", units="m")


def download_region_terrain(region_id: str | None = None, step_deg: float = OUTPUT_DEG) -> dict:
    region = get_active_region(region_id)
    bbox = region.bbox
    terrain_dir = region.terrain_dir(PROJECT_ROOT)
    cache_dir = terrain_dir / "srtm_cache"

    paths = {
        "dem": terrain_dir / f"dem_{region.id}.tif",
        "slope": terrain_dir / f"slope_{region.id}.tif",
        "twi": terrain_dir / f"twi_{region.id}.tif",
        "flow_accum": terrain_dir / f"flow_accum_{region.id}.tif",
    }
    if all(p.exists() for p in paths.values()):
        logger.info("Terrain already exists for %s; skipping.", region.id)
        return {k: str(v) for k, v in paths.items()}

    logger.info("Fetching elevation grid for %s (%s)", region.name, bbox)
    dem, transform = _build_dem_from_open_meteo(bbox, step_deg)
    dem = _fill_nans(dem)

    lat_center = (bbox["lat_min"] + bbox["lat_max"]) / 2.0
    slope = _compute_slope(dem, step_deg, lat_center)
    flow_accum = _compute_flow_accum(dem)
    twi = _compute_twi(flow_accum, slope).astype(np.float32)

    _write_geotiff(paths["dem"], dem, transform)
    _write_geotiff(paths["slope"], slope, transform)
    _write_geotiff(paths["twi"], twi, transform)
    _write_geotiff(paths["flow_accum"], flow_accum, transform)

    logger.info("Saved terrain -> %s", terrain_dir)
    return {k: str(v) for k, v in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SRTM terrain for a SIH region.")
    parser.add_argument("--region", default=None, help="Region id (default: SIH_REGION env)")
    parser.add_argument("--step-deg", type=float, default=OUTPUT_DEG, help="Output grid step in degrees")
    args = parser.parse_args()
    if args.region:
        os.environ["SIH_REGION"] = args.region
    result = download_region_terrain(args.region, step_deg=args.step_deg)
    for key, path in result.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
