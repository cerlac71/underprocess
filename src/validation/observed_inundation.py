"""Satellite-observed inundation from JRC Global Surface Water (Landsat, 30 m).

Downloads monthly water-detection rasters and occurrence (permanent-water baseline)
from the JRC open FTP archive, mosaics tiles over Uttar Pradesh, and resamples to
the project 0.25° grid for CSI/F1 validation against model inundation.

Data source: Pekel et al. (2016) Nature / JRC GSW v1.4
https://global-surface-water.appspot.com/download
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
from rasterio.windows import from_bounds

from config.regions import Region, get_active_region

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
JRC_BASE = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GSWE"

# UP bbox (23.5–30.5°N, 77–84.5°E) spans JRC 10° tiles at row 5 (20–30°N), cols 25–26 (70–90°E).
UP_TILE_ROW = 5
UP_TILE_COLS = (25, 26)

WATER_CLASS = 2
NODATA_CLASS = 0
PERMANENT_WATER_OCCURRENCE = 90  # exclude rivers/lakes (0–100 scale)


def validation_dir(region: Region, project_root: Path | None = None) -> Path:
    return region.data_root(project_root) / "validation" / "jrc_gsw"


def _tile_offset(index: int) -> str:
    return f"{index * 40000:010d}"


def _monthly_tile_url(year: int, month: int, row: int, col: int) -> str:
    yyyymm = f"{year}_{month:02d}"
    ro = _tile_offset(row)
    co = _tile_offset(col)
    return f"{JRC_BASE}/MonthlyHistory/VER4-0/tiles/{year}/{yyyymm}/{yyyymm}-{ro}-{co}.tif"


def _occurrence_tile_url(row: int, col: int) -> str:
    ro = _tile_offset(row)
    co = _tile_offset(col)
    return f"{JRC_BASE}/Aggregated/VER4-0/occurrence/tiles/occurrence-{ro}-{co}.tif"


def _download_file(url: str, dest: Path, timeout: int = 600) -> Path:
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest
    logger.info("Downloading %s", url)
    response = requests.get(url, timeout=timeout, stream=True)
    response.raise_for_status()
    with dest.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1 << 20):
            if chunk:
                handle.write(chunk)
    return dest


def download_jrc_tiles(
    region_id: str = "uttar_pradesh",
    years_months: list[tuple[int, int]] | None = None,
    project_root: Path | None = None,
) -> Path:
    """Download JRC monthly-history and occurrence tiles covering the region."""
    region = get_active_region(region_id)
    root = validation_dir(region, project_root)
    raw = root / "raw"
    if years_months is None:
        years_months = [(2019, 7), (2019, 8), (2019, 9)]

    for row in (UP_TILE_ROW,):
        for col in UP_TILE_COLS:
            occ_name = f"occurrence_{row}_{col}.tif"
            _download_file(_occurrence_tile_url(row, col), raw / "occurrence" / occ_name)

    for year, month in years_months:
        for col in UP_TILE_COLS:
            name = f"{year}_{month:02d}_row{UP_TILE_ROW}_col{col}.tif"
            _download_file(
                _monthly_tile_url(year, month, UP_TILE_ROW, col),
                raw / "monthly" / name,
            )

    manifest = {
        "source": "JRC Global Surface Water v1.4 MonthlyHistory + Occurrence",
        "citation": "Pekel et al. (2016) Nature; JRC GSW open data FTP",
        "tile_row": UP_TILE_ROW,
        "tile_cols": list(UP_TILE_COLS),
        "months": [f"{y}-{m:02d}" for y, m in years_months],
    }
    (root / "manifest.json").write_text(
        __import__("json").dumps(manifest, indent=2),
        encoding="utf-8",
    )
    logger.info("JRC GSW tiles cached under %s", root)
    return root


def _read_window(path: Path, bounds: tuple[float, float, float, float]) -> tuple[np.ndarray, rasterio.Affine, rasterio.crs.CRS]:
    with rasterio.open(path) as ds:
        window = from_bounds(*bounds, transform=ds.transform)
        data = ds.read(1, window=window, boundless=True, fill_value=0)
        return data, ds.window_transform(window), ds.crs


def _mosaic_tiles(paths: list[Path], bounds: tuple[float, float, float, float]) -> tuple[np.ndarray, rasterio.Affine, rasterio.crs.CRS]:
    """Mosaic JRC 10° tiles over the region bbox (west-to-east column order)."""
    if len(paths) == 1:
        return _read_window(paths[0], bounds)

    lon_min, lat_min, lon_max, lat_max = bounds
    pieces = []
    for path in paths:
        data, transform, crs = _read_window(path, bounds)
        west = transform.c
        east = west + transform.a * data.shape[1]
        pieces.append({"data": data, "transform": transform, "west": west, "east": east, "crs": crs})

    pieces.sort(key=lambda p: p["west"])
    if len(pieces) == 1:
        return pieces[0]["data"], pieces[0]["transform"], pieces[0]["crs"]

    # Horizontal stitch at tile boundary (UP spans cols 25 and 26).
    left, right = pieces[0], pieces[1]
    split_lon = right["west"]
    left_cols = max(1, int(round((split_lon - left["west"]) / left["transform"].a)))
    left_cols = min(left_cols, left["data"].shape[1])
    mosaic = np.concatenate([left["data"][:, :left_cols], right["data"]], axis=1)
    transform = left["transform"]
    return mosaic, transform, left["crs"]


def _region_bounds(region: Region) -> tuple[float, float, float, float]:
    return region.lon_min, region.lat_min, region.lon_max, region.lat_max


def _resample_to_model_grid(
    source: np.ndarray,
    src_transform: rasterio.Affine,
    src_crs: rasterio.crs.CRS,
    lat: np.ndarray,
    lon: np.ndarray,
    resampling: Resampling = Resampling.average,
) -> np.ndarray:
    """Resample a single-band JRC raster onto the model lat/lon vectors."""
    grid_step = float(abs(lon[1] - lon[0])) if len(lon) > 1 else 0.25
    lat_min, lat_max = float(lat.min()), float(lat.max())
    lon_min, lon_max = float(lon.min()), float(lon.max())
    dst_height = lat.size
    dst_width = lon.size
    dst_transform = from_origin(
        lon_min - grid_step / 2.0,
        lat_max + grid_step / 2.0,
        grid_step,
        grid_step,
    )
    dest = np.zeros((dst_height, dst_width), dtype=np.float32)
    reproject(
        source=source.astype(np.float32),
        destination=dest,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=src_crs,
        resampling=resampling,
    )
    return dest


def _load_monthly_water_fraction(
    region: Region,
    year: int,
    month: int,
    lat: np.ndarray,
    lon: np.ndarray,
    project_root: Path | None = None,
) -> np.ndarray:
    """Fraction of 30 m pixels classified as water within each 0.25° cell."""
    root = validation_dir(region, project_root)
    paths = [
        root / "raw" / "monthly" / f"{year}_{month:02d}_row{UP_TILE_ROW}_col{col}.tif"
        for col in UP_TILE_COLS
    ]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing JRC monthly tiles: {missing[0].name}. "
            "Run: python scripts/download_flood_observations.py"
        )

    bounds = _region_bounds(region)
    mosaic, transform, crs = _mosaic_tiles(paths, bounds)
    water = (mosaic == WATER_CLASS).astype(np.float32)
    return _resample_to_model_grid(water, transform, crs, lat, lon, Resampling.average)


def _load_permanent_water_mask(
    region: Region,
    lat: np.ndarray,
    lon: np.ndarray,
    project_root: Path | None = None,
) -> np.ndarray:
    root = validation_dir(region, project_root)
    paths = [
        root / "raw" / "occurrence" / f"occurrence_{UP_TILE_ROW}_{col}.tif"
        for col in UP_TILE_COLS
    ]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing JRC occurrence tile: {missing[0].name}. "
            "Run: python scripts/download_flood_observations.py"
        )

    bounds = _region_bounds(region)
    mosaic, transform, crs = _mosaic_tiles(paths, bounds)
    occ = _resample_to_model_grid(mosaic.astype(np.float32), transform, crs, lat, lon, Resampling.average)
    return occ >= PERMANENT_WATER_OCCURRENCE


def load_satellite_observed_mask(
    peak_date: str,
    window_start: str,
    window_end: str,
    lat: np.ndarray,
    lon: np.ndarray,
    region_id: str = "uttar_pradesh",
    water_fraction_threshold: float = 0.05,
    project_root: Path | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Build a binary observed inundation mask from JRC Landsat monthly water history.

    For each month in the event window, water fraction per 0.25° cell is computed.
    The maximum fraction across months is used, then permanent water (occurrence ≥ 90%)
    is excluded. A cell is inundated if max water fraction ≥ threshold.
    """
    import pandas as pd

    region = get_active_region(region_id)
    root = project_root or PROJECT_ROOT
    start = pd.Timestamp(window_start)
    end = pd.Timestamp(window_end)
    months = pd.period_range(start.to_period("M"), end.to_period("M"), freq="M")

    frac_stack = []
    for period in months:
        frac = _load_monthly_water_fraction(
            region, period.year, period.month, lat, lon, root,
        )
        frac_stack.append(frac)

    if not frac_stack:
        peak = pd.Timestamp(peak_date)
        frac_stack = [_load_monthly_water_fraction(region, peak.year, peak.month, lat, lon, root)]

    max_frac = np.max(np.stack(frac_stack, axis=0), axis=0)
    permanent = _load_permanent_water_mask(region, lat, lon, root)
    observed = (max_frac >= water_fraction_threshold) & ~permanent

    meta = {
        "source": "JRC GSW v1.4 MonthlyHistory (Landsat 5/7/8)",
        "type": "satellite_observed_surface_water",
        "resolution_native_m": 30,
        "resolution_validation_deg": region.grid_step,
        "water_fraction_threshold": water_fraction_threshold,
        "permanent_water_excluded": True,
        "permanent_occurrence_cutoff": PERMANENT_WATER_OCCURRENCE,
        "months_used": [str(p) for p in months] if len(months) else [peak_date[:7]],
        "citation": "Pekel, J.-F. et al. High-resolution mapping of global surface water. Nature 540, 418–422 (2016).",
    }
    return observed, meta


def observations_available(region_id: str = "uttar_pradesh", project_root: Path | None = None) -> bool:
    region = get_active_region(region_id)
    root = validation_dir(region, project_root)
    monthly = root / "raw" / "monthly"
    occurrence = root / "raw" / "occurrence"
    if not monthly.exists() or not occurrence.exists():
        return False
    return bool(list(monthly.glob("*.tif"))) and bool(list(occurrence.glob("*.tif")))
