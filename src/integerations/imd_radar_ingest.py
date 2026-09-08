"""IMD Doppler weather radar ingestion (NetCDF sweep / volume format).

Processes official IMD IRIS-style NetCDF files (one sweep per file) into
lat/lon rainfall fields for the SIH26071 pipeline. Sample volumes are
distributed via open-radar-data (Jaipur) and pyscancf_examples (Goa).
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

from integerations.radar_ingest import reflectivity_to_rainfall, regrid_to_common_grid

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMD_SAMPLE_ROOT = PROJECT_ROOT / "data" / "radar" / "imd_doppler_sample"
IMD_SAMPLE_RAW = IMD_SAMPLE_ROOT / "raw"
IMD_SAMPLE_PROCESSED = IMD_SAMPLE_ROOT / "imd_doppler_rainfall.nc"
IMD_SAMPLE_MANIFEST = IMD_SAMPLE_ROOT / "manifest.json"

IMD_FILENAME_RE = re.compile(
    r"^(?P<site>[A-Z]{3})(?P<yy>\d{2})(?P<mm>\d{2})(?P<dd>\d{2})(?P<hms>\d{6})-IMD-B\.nc"
)

# Open sample volumes (2 Jaipur scans + 1 Goa scan for format coverage).
SAMPLE_VOLUMES = [
    {
        "id": "jpr_vol1",
        "site": "Jaipur",
        "site_code": "JPR",
        "stem": "JPR220822135253-IMD-B.nc",
        "url_base": "https://raw.githubusercontent.com/openradar/open-radar-data/main/data/IMD/",
        "subdir": "jpr",
    },
    {
        "id": "jpr_vol2",
        "site": "Jaipur",
        "site_code": "JPR",
        "stem": "JPR220822140253-IMD-B.nc",
        "url_base": "https://raw.githubusercontent.com/openradar/open-radar-data/main/data/IMD/",
        "subdir": "jpr",
    },
    {
        "id": "goa_vol1",
        "site": "Goa",
        "site_code": "GOA",
        "stem": "GOA210516024101-IMD-B.nc",
        "url_base": "https://raw.githubusercontent.com/syedhamidali/pyscancf_examples/main/data/goa16/",
        "subdir": "goa",
    },
]

BELOW_THRESHOLD_DBZ = -128.0


def is_imd_radar_file(filepath: str | Path) -> bool:
    name = Path(filepath).name
    return bool(IMD_FILENAME_RE.match(name.split(".nc")[0] + ".nc" if ".nc" in name else name)) or "-IMD-B.nc" in name


def parse_imd_timestamp(filepath: str | Path) -> datetime | None:
    """Parse observation time from an IMD filename (e.g. JPR220822135253-IMD-B.nc)."""
    base = Path(filepath).name.split(".nc")[0]
    if not base.endswith("-IMD-B"):
        return None
    token = base.replace("-IMD-B", "")
    match = IMD_FILENAME_RE.match(token + "-IMD-B.nc")
    if not match:
        return None
    parts = match.groupdict()
    return datetime.strptime(
        f"{parts['yy']}{parts['mm']}{parts['dd']}{parts['hms']}",
        "%y%m%d%H%M%S",
    )


def group_imd_volume_files(directory: str | Path) -> dict[str, list[Path]]:
    """Group sweep files into volumes by filename stem (before .nc.N suffix)."""
    directory = Path(directory)
    groups: dict[str, list[Path]] = {}
    for path in sorted(directory.glob("*-IMD-B.nc*")):
        stem = path.name.split(".nc")[0] + ".nc"
        groups.setdefault(stem, []).append(path)
    for stem in groups:
        groups[stem] = sorted(groups[stem], key=lambda p: (len(p.name.split(".nc")), p.name))
    return groups


def _polar_to_latlon(
    site_lat: float,
    site_lon: float,
    azimuth_deg: np.ndarray,
    range_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Great-circle destination from radar site for each azimuth/range pair."""
    earth_radius = 6_371_000.0
    lat1 = np.radians(site_lat)
    lon1 = np.radians(site_lon)
    az = np.radians(azimuth_deg)
    ang_dist = range_m / earth_radius

    sin_lat1 = np.sin(lat1)
    cos_lat1 = np.cos(lat1)
    sin_ad = np.sin(ang_dist)
    cos_ad = np.cos(ang_dist)

    lat2 = np.arcsin(sin_lat1 * cos_ad + cos_lat1 * sin_ad * np.cos(az))
    lon2 = lon1 + np.arctan2(
        np.sin(az) * sin_ad * cos_lat1,
        cos_ad - sin_lat1 * np.sin(lat2),
    )
    return np.degrees(lat2), np.degrees(lon2)


def load_imd_sweep(filepath: str | Path) -> xr.Dataset:
    """Load a single IMD sweep NetCDF file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(path)
    return xr.open_dataset(path)


def sweep_reflectivity_to_points(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract lat, lon, dBZ arrays from one IMD sweep."""
    if "Z" not in ds:
        raise ValueError(f"No reflectivity variable 'Z' in {getattr(ds, 'encoding', {}).get('source', 'IMD sweep')}")

    site_lat = float(ds.siteLat.values)
    site_lon = float(ds.siteLon.values)
    first_gate = float(ds.firstGateRange.values)
    gate_size = float(ds.gateSize.values)
    n_bins = ds.sizes["bin"]
    range_m = first_gate + gate_size * (np.arange(n_bins, dtype=np.float32) + 0.5)

    azimuth = ds.radialAzim.values.astype(np.float32)
    dbz = ds.Z.values.astype(np.float32)

    # Broadcast azimuth along range bins.
    az2d = np.broadcast_to(azimuth[:, None], dbz.shape)
    range2d = np.broadcast_to(range_m[None, :], dbz.shape)

    lat2d, lon2d = _polar_to_latlon(site_lat, site_lon, az2d, range2d)

    mask = (dbz <= BELOW_THRESHOLD_DBZ) | ~np.isfinite(dbz)
    lat_flat = lat2d[~mask]
    lon_flat = lon2d[~mask]
    dbz_flat = dbz[~mask]
    return lat_flat, lon_flat, dbz_flat


def process_imd_volume(
    sweep_files: Iterable[str | Path],
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    noise_floor_dbz: float = -10.0,
) -> tuple[np.ndarray, dict]:
    """Composite sweeps in one volume to a lat/lon rainfall rate grid (mm/hr)."""
    sweep_paths = [Path(p) for p in sweep_files]
    if not sweep_paths:
        raise ValueError("No sweep files provided.")

    rainfall_layers: list[np.ndarray] = []
    meta: dict = {}

    for path in sweep_paths:
        with xr.open_dataset(path) as ds:
            if not meta:
                meta = {
                    "site_lat": float(ds.siteLat.values),
                    "site_lon": float(ds.siteLon.values),
                    "site_code": path.name[:3],
                    "timestamp": str(ds.esStartTime.values),
                }
            lat_pts, lon_pts, dbz_pts = sweep_reflectivity_to_points(ds)
            rain_pts = reflectivity_to_rainfall(dbz_pts, noise_floor_dbz=noise_floor_dbz)
            layer = regrid_to_common_grid(
                lat_pts,
                lon_pts,
                rain_pts,
                target_lat,
                target_lon,
                method="nearest",
            )
            rainfall_layers.append(layer)

    composite = np.nanmax(np.stack(rainfall_layers, axis=0), axis=0)
    meta["n_sweeps"] = len(sweep_paths)
    meta["source_files"] = [p.name for p in sweep_paths]
    return composite.astype(np.float32), meta


def _grid_around_site(site_lat: float, site_lon: float, radius_deg: float = 2.5, step: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    lats = np.arange(site_lat - radius_deg, site_lat + radius_deg + step * 0.1, step)
    lons = np.arange(site_lon - radius_deg, site_lon + radius_deg + step * 0.1, step)
    return lats.astype(np.float64), lons.astype(np.float64)


def build_imd_rainfall_dataset(
    volume_dirs: Iterable[str | Path],
    grid_step: float = 0.25,
    radius_deg: float = 2.5,
) -> xr.Dataset:
    """Build a (time, lat, lon) rainfall dataset from one or more IMD volume folders."""
    records: list[dict] = []

    for vol_dir in volume_dirs:
        vol_dir = Path(vol_dir)
        groups = group_imd_volume_files(vol_dir)
        for stem, sweeps in sorted(groups.items()):
            ts = parse_imd_timestamp(stem)
            if ts is None:
                logger.warning("Skipping volume with unparseable name: %s", stem)
                continue
            with xr.open_dataset(sweeps[0]) as ds0:
                site_lat = float(ds0.siteLat.values)
                site_lon = float(ds0.siteLon.values)
            target_lat, target_lon = _grid_around_site(site_lat, site_lon, radius_deg, grid_step)
            rainfall, meta = process_imd_volume(sweeps, target_lat, target_lon)
            records.append(
                {
                    "time": pd.Timestamp(ts),
                    "rainfall": rainfall,
                    "lat": target_lat,
                    "lon": target_lon,
                    "meta": meta,
                }
            )

    if not records:
        raise ValueError("No IMD volumes processed.")

    # Volumes from the same radar site share one target grid — stack directly.
    lat = records[0]["lat"]
    lon = records[0]["lon"]
    times = [r["time"] for r in records]
    stack = np.stack([r["rainfall"] for r in records], axis=0).astype(np.float32)

    ds_out = xr.Dataset(
        {"rainfall": (("time", "lat", "lon"), stack)},
        coords={"time": times, "lat": lat, "lon": lon},
    )
    ds_out.attrs.update(
        {
            "source": "IMD Doppler Weather Radar (official NetCDF sweep format)",
            "units": "mm/hr",
            "description": (
                "Rainfall rate derived from IMD reflectivity (Z) using Marshall-Palmer Z-R. "
                "Built from open sample volumes until institutional Lucknow/UP radar archives arrive."
            ),
            "processor": "SIH26071 imd_radar_ingest.py",
            "volumes": json.dumps([r["meta"] for r in records]),
        }
    )
    return ds_out


def process_imd_sample_to_netcdf(
    raw_root: Path | None = None,
    out_path: Path | None = None,
    use_jaipur_nowcast_pair: bool = True,
) -> Path:
    """Process downloaded sample volumes into a pipeline-ready NetCDF file."""
    raw_root = raw_root or IMD_SAMPLE_RAW
    out_path = out_path or IMD_SAMPLE_PROCESSED

    if use_jaipur_nowcast_pair:
        volume_dirs = [raw_root / "jpr"]
    else:
        volume_dirs = [raw_root / "jpr", raw_root / "goa"]

    ds = build_imd_rainfall_dataset(volume_dirs)
    if use_jaipur_nowcast_pair and ds.sizes["time"] > 2:
        # Keep the two Jaipur volumes that are 10 minutes apart for nowcasting.
        ds = ds.isel(time=slice(0, 2))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path)
    logger.info("Wrote IMD sample rainfall -> %s", out_path)
    return out_path


def imd_sample_available() -> bool:
    return IMD_SAMPLE_PROCESSED.exists() and IMD_SAMPLE_PROCESSED.stat().st_size > 1000


def load_imd_sample_rainfall(path: Path | None = None) -> xr.Dataset | None:
    """Load processed IMD sample rainfall for nowcast / demo."""
    path = path or IMD_SAMPLE_PROCESSED
    if not path.exists():
        return None
    with xr.open_dataset(path) as ds:
        return ds.load()


def write_sample_manifest(raw_root: Path | None = None) -> Path:
    raw_root = raw_root or IMD_SAMPLE_RAW
    manifest = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "description": "IMD Doppler sample volumes for SIH26071 integration (until real UP radar data).",
        "processed_file": str(IMD_SAMPLE_PROCESSED.relative_to(PROJECT_ROOT)),
        "volumes": SAMPLE_VOLUMES,
        "raw_files": {
            vol["id"]: [p.name for p in sorted((raw_root / vol["subdir"]).glob(f"{vol['stem']}*"))]
            for vol in SAMPLE_VOLUMES
        },
        "notes": [
            "Jaipur volumes (JPR) are 10 minutes apart — used for optical-flow nowcast demo.",
            "Goa volume (GOA) demonstrates second radar site / dual-pol moments.",
            "Replace with Lucknow Doppler archives from radarapi.imd.gov.in for UP production.",
        ],
    }
    IMD_SAMPLE_ROOT.mkdir(parents=True, exist_ok=True)
    IMD_SAMPLE_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return IMD_SAMPLE_MANIFEST
